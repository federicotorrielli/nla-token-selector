"""NLAClientLite — single shim for every SGLang/AV/AR call in the project.

Three public methods:

- `generate(position_id, temperature=0.0, sample_idx=0) -> Generation`
  Cache-aware AV decode. The vendored `NLAClient.generate` returns only the
  text — we POST to SGLang ourselves to also collect per-token log-probabilities
  and detokenised strings (needed by `TokLogProb`'s span alignment).

- `critic_score(position_id, explanation) -> (mse, cos)`
  Cache-aware AR ΔMSE for `ARC`. Delegates to the vendored `NLACritic.score`
  (local PyTorch).

- `score_continuations(position_id, prompt_suffix, candidates) -> list[float]`
  Constrained log-prob over N label tokens for `ConstrainedLingConf`.

Caches are in-memory dicts; Parquet is written on `flush()` / `close()` to
`{settings.paths.results_root}/{model.short}/cache/`. The cache key for
explanations is `(position_id, temperature, sample_idx, code_version)`;
for critic scores it is `(position_id, sha256(explanation), code_version)`.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import orjson
import polars as pl
import torch  # scoped exception: only used to wrap np.ndarray for NLAClient._build_embeds
from huggingface_hub import snapshot_download

from ._nla_inference import (
    EXPLANATION_RE,
    NLAClient,
    NLACritic,
    inject_at_marked_positions,
    normalize_activation,
)
from .schema import Generation
from .settings import Settings

_ExplKey = tuple[int, float, int, int]  # (position_id, temperature, sample_idx, code_version)
_CriticKey = tuple[int, str, int]  # (position_id, explanation_sha256, code_version)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    """Write then rename, so a killed run can't truncate the cache.
    os.replace is atomic on POSIX when src and dst share a directory."""
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)


def _materialize_checkpoint(repo_id: str) -> Path:
    """Snapshot-download an HF repo and return the local directory.

    `NLAClient` / `NLACritic` need a *local* directory with `nla_meta.yaml` +
    `config.json` + safetensors — they do not accept HF repo ids directly.
    `snapshot_download` is idempotent and returns the existing cache path
    if the snapshot is already present.
    """
    return Path(snapshot_download(repo_id))


class NLAClientLite:
    """Cache-aware wrapper over the vendored NLAClient + NLACritic.

    Construction is cheap — the AV / AR backbones load lazily on first use,
    so tests that exercise only the cache schema do not pay the model-load
    cost. Pass a single `Settings` instance; the active model is resolved
    via `settings.model`.
    """

    def __init__(self, settings: Settings, *, http_timeout: float = 300.0):
        self.settings = settings
        self.model = settings.model
        self._code_version = settings.experiment.code_version

        self._http = httpx.Client(timeout=httpx.Timeout(http_timeout))
        self._av: NLAClient | None = None
        self._ar: NLACritic | None = None
        self._corpus: pl.DataFrame | None = None
        self._act_index: dict[int, np.ndarray] | None = None

        self._cache_dir = settings.paths.results_root / self.model.short / "cache"
        self._explanations_path = self._cache_dir / "explanations.parquet"
        self._critic_path = self._cache_dir / "critic_scores.parquet"

        self._explanations: dict[_ExplKey, Generation] = {}
        self._critic_scores: dict[_CriticKey, tuple[float, float]] = {}
        self._explanations_dirty = False
        self._critic_dirty = False

        self._load_caches()

    # lazy heavyweights
    @property
    def av(self) -> NLAClient:
        if self._av is None:
            ckpt = _materialize_checkpoint(self.model.av_repo)
            self._av = NLAClient(ckpt, sglang_url=self.model.sglang_url)
        return self._av

    @property
    def ar(self) -> NLACritic:
        if self._ar is None:
            ckpt = _materialize_checkpoint(self.model.ar_repo)
            # NLACritic defaults device="cpu"; on a B200 this means ARC's
            # ~150k forward passes take ~22 min for n=100 (extrapolating to
            # ~22 h at n=6000). Cuda forward is 10-50× faster — the AR's
            # truncated K+1-layer stack is ~5-10 GB, well under any free
            # headroom we'd have. `cuda` defaults to device 0.
            ar_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._ar = NLACritic(ckpt, device=ar_device)
        return self._ar

    @property
    def corpus(self) -> pl.DataFrame:
        if self._corpus is None:
            path = self.settings.paths.corpus_parquet
            assert path.exists(), (
                f"corpus parquet missing at {path!r}. Build it via "
                f"`calnla corpus build` (or whichever pipeline populates "
                f"results/corpus/) before calling NLAClientLite.generate."
            )
            self._corpus = pl.read_parquet(path)
        return self._corpus

    # corpus lookup
    def _activation(self, position_id: int) -> np.ndarray:
        # Built once from the corpus; per-call .filter() was an O(n) scan.
        if self._act_index is None:
            self._act_index = {
                int(pid): np.asarray(act, dtype=np.float32)
                for pid, act in zip(self.corpus["position_id"], self.corpus["activation"].to_list())
            }
        v = self._act_index.get(position_id)
        assert v is not None, f"no corpus row for position_id={position_id}."
        assert v.shape == (self.model.d_model,), (
            f"activation shape {v.shape} != ({self.model.d_model},); "
            f"corpus row uses the wrong model registry entry."
        )
        return v

    # public API
    def generate(
        self,
        position_id: int,
        *,
        temperature: float = 0.0,
        sample_idx: int = 0,
        max_new_tokens: int = 600,
    ) -> Generation:
        """Cache-aware AV decode with per-token log-probabilities."""
        key: _ExplKey = (position_id, float(temperature), int(sample_idx), self._code_version)
        if key in self._explanations:
            return self._explanations[key]

        v = self._activation(position_id)
        # Reach into the vendored client for the activation-injection step —
        # the public `.generate` discards logprobs, which we need.
        embeds_np, _ = self.av._build_embeds(torch.as_tensor(v), prompt_content=None)
        body = orjson.dumps(
            {
                "input_embeds": embeds_np,
                "sampling_params": {
                    "temperature": float(temperature),
                    "max_new_tokens": int(max_new_tokens),
                    "skip_special_tokens": False,
                },
                "return_logprob": True,
                "logprob_start_len": -1,
                "return_text_in_logprobs": True,
                "top_logprobs_num": 1,
            },
            option=orjson.OPT_SERIALIZE_NUMPY,
        )
        r = self._http.post(
            f"{self.model.sglang_url.rstrip('/')}/generate",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        raw = r.json()
        out = raw[0] if isinstance(raw, list) else raw

        text = out["text"]
        m = EXPLANATION_RE.search(text)
        explanation = m.group(1).strip() if m else text

        meta = out.get("meta_info", {}) or {}
        tok_lp_raw: list[Any] = meta.get("output_token_logprobs") or []
        tok_logprobs: list[float] = []
        tok_strings: list[str] = []
        for entry in tok_lp_raw:
            # SGLang shape: [logprob, token_id, token_text] when
            # return_text_in_logprobs=True; else [logprob, token_id].
            tok_logprobs.append(float(entry[0]))
            tok_strings.append(entry[2] if len(entry) > 2 else "")

        gen = Generation(
            explanation=explanation,
            tok_logprobs=tok_logprobs,
            tok_strings=tok_strings,
        )
        self._explanations[key] = gen
        self._explanations_dirty = True
        return gen

    def generate_batch(
        self,
        position_ids: list[int],
        *,
        temperature: float,
        max_new_tokens: int = 600,
    ) -> list[str]:
        """Batched AV decode → explanation text per position (no logprobs).

        MSA needs k=20 T=1.0 resamples per position (120k decodes at n=6000);
        per-call `generate` stalls on the serial input_embeds parse like the
        verbalized methods. Decode is GPU-friendly, so packing many positions
        into one batched `/generate` lets SGLang batch the decode and actually
        use the GPU. Text-only (the response carries no logprobs), so the
        sample caches stay lean. Repeated position ids in one call yield
        distinct samples at T>0; their single-turn embeds are built once.
        """
        embeds_cache: dict[int, np.ndarray] = {}
        embeds: list[np.ndarray] = []
        for pid in position_ids:
            e = embeds_cache.get(pid)
            if e is None:
                v = self._activation(pid)
                e, _ = self.av._build_embeds(torch.as_tensor(v), prompt_content=None)
                embeds_cache[pid] = e
            embeds.append(e)
        payloads = self._post_generate_batch(
            embeds,
            {
                "temperature": float(temperature),
                "max_new_tokens": int(max_new_tokens),
                "skip_special_tokens": False,
            },
        )
        out: list[str] = []
        for p in payloads:
            text = p.get("text", "") or ""
            m = EXPLANATION_RE.search(text)
            out.append(m.group(1).strip() if m else text)
        return out

    def score_explanation_logprobs(
        self, activation: np.ndarray, explanation_text: str
    ) -> tuple[list[float], list[str]]:
        """Teacher-forced per-token logprobs of `explanation_text` as the AV's
        assistant turn, with an ARBITRARY `activation` injected at the marker.

        For the PMI probe: reading logp(tokens | v̄) under a null/mean vector
        lets us contrast against the cached logp(tokens | v). Returns the same
        (logprobs, strings) shape as `generate`, but over the assistant span of
        the *input* (one forward, no decode) via SGLang `input_token_logprobs`.
        """
        av = self.av
        cfg = av.cfg
        user1 = cfg.actor_prompt_template.format(injection_char=cfg.injection_char)

        def _ids(messages: list[dict], add_gen: bool) -> list[int]:
            out = av.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=add_gen
            )
            if hasattr(out, "input_ids"):
                out = out.input_ids
            elif isinstance(out, dict):
                out = out["input_ids"]
            if out and isinstance(out[0], (list, tuple)):
                out = out[0]
            return list(out)

        prefix = _ids([{"role": "user", "content": user1}], add_gen=True)
        full = _ids(
            [
                {"role": "user", "content": user1},
                {"role": "assistant", "content": explanation_text},
            ],
            add_gen=False,
        )
        ids_t = torch.tensor(full, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            embeds = (av.embed(ids_t.to(av.embed.weight.device)) * av.embed_scale).float()
        v_scaled = normalize_activation(
            torch.as_tensor(activation).float().view(1, -1), cfg.injection_scale
        )
        injected = inject_at_marked_positions(
            ids_t, embeds.cpu(), v_scaled, cfg.injection_token_id,
            cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id,
        )
        body = orjson.dumps(
            {
                "input_embeds": injected[0].contiguous().numpy(),
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 1, "skip_special_tokens": False},
                "return_logprob": True,
                "logprob_start_len": len(prefix),  # assistant span only
                "return_text_in_logprobs": True,
            },
            option=orjson.OPT_SERIALIZE_NUMPY,
        )
        r = self._http.post(
            f"{self.model.sglang_url.rstrip('/')}/generate",
            content=body, headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        raw = r.json()
        out = raw[0] if isinstance(raw, list) else raw
        rows: list[Any] = (out.get("meta_info", {}) or {}).get("input_token_logprobs") or []
        lps: list[float] = []
        strs: list[str] = []
        for e in rows:
            if e[0] is None:  # SGLang marks the first scored token's logprob null
                continue
            lps.append(float(e[0]))
            strs.append(e[2] if len(e) > 2 else "")
        return lps, strs

    def critic_score(self, position_id: int, explanation: str) -> tuple[float, float]:
        """AR (mse, cos) cached on (position_id, sha256(explanation), code_version)."""
        key: _CriticKey = (position_id, _sha256(explanation), self._code_version)
        if key in self._critic_scores:
            return self._critic_scores[key]

        v = self._activation(position_id)
        mse, cos = self.ar.score(explanation, v)
        self._critic_scores[key] = (float(mse), float(cos))
        self._critic_dirty = True
        return self._critic_scores[key]

    # multi-turn elicitation (verbalized methods)
    def _build_multi_turn_embeds(
        self,
        position_id: int,
        turn1_assistant_text: str,
        turn2_user_text: str,
    ) -> np.ndarray:
        """Build `input_embeds` for a 3-message conversation:

            user₁  (canonical actor template with the <INJECT> char)
            assistant₁  (turn1_assistant_text — the AV's prior decode)
            user₂  (turn2_user_text — elicitation prompt)
            <assistant₂ generation marker added by add_generation_prompt=True>

        The activation is injected at the `<INJECT>` marker in user₁ exactly
        the way single-turn `_build_embeds` does — re-using
        `inject_at_marked_positions` so the neighbour-id safety check still
        fires. Returns the full prefix as a `[T, d]` numpy array ready for
        SGLang's `/generate` `input_embeds` body field.

        Sole callers are `score_continuations` (ConstrainedLingConf) and the
        freeform numeric elicitation (FreeformNumConf).
        """
        av = self.av  # forces lazy load
        cfg = av.cfg
        tokenizer = av.tokenizer

        turn1_user = cfg.actor_prompt_template.format(injection_char=cfg.injection_char)
        messages = [
            {"role": "user", "content": turn1_user},
            {"role": "assistant", "content": turn1_assistant_text},
            {"role": "user", "content": turn2_user_text},
        ]
        out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        # Normalise transformers 4.x list-of-int vs 5.x BatchEncoding shape
        # (mirrors the `_act_ids` adapter in `_nla_inference.py`).
        if hasattr(out, "input_ids"):
            out = out.input_ids
        elif isinstance(out, dict):
            out = out["input_ids"]
        if out and isinstance(out[0], (list, tuple)):
            out = out[0]
        input_ids = list(out)

        ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            embeds = (av.embed(ids_t.to(av.embed.weight.device)) * av.embed_scale).float()

        v_raw = torch.as_tensor(self._activation(position_id))
        v_scaled = normalize_activation(v_raw.float().view(1, -1), cfg.injection_scale)
        injected = inject_at_marked_positions(
            ids_t,
            embeds.cpu(),
            v_scaled,
            cfg.injection_token_id,
            cfg.injection_left_neighbor_id,
            cfg.injection_right_neighbor_id,
        )
        return injected[0].contiguous().numpy()

    def score_continuations(
        self,
        position_id: int,
        turn1_assistant_text: str,
        turn2_user_text: str,
        candidates: list[str],
        *,
        max_new_tokens: int = 24,
    ) -> list[float]:
        """Per-candidate joint log-prob under the multi-turn activation-injected prefix.

        For each `candidate` string (e.g. "very high"), force the AV to
        decode that exact text via `sampling_params.regex=re.escape(c)` and
        sum the per-token logprobs from `meta_info.output_token_logprobs`.
        The sum is the joint `log P(candidate | prefix)`.

        Caller normalises across candidates (softmax over the returned
        log-probs) and picks the readout (e.g. P(top label)).
        """
        import re

        prefix_embeds = self._build_multi_turn_embeds(
            position_id, turn1_assistant_text, turn2_user_text
        )

        out: list[float] = []
        for candidate in candidates:
            body = orjson.dumps(
                {
                    "input_embeds": prefix_embeds,
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": int(max_new_tokens),
                        "regex": re.escape(candidate),
                        "skip_special_tokens": False,
                    },
                    "return_logprob": True,
                    "logprob_start_len": -1,
                    "return_text_in_logprobs": False,
                },
                option=orjson.OPT_SERIALIZE_NUMPY,
            )
            r = self._http.post(
                f"{self.model.sglang_url.rstrip('/')}/generate",
                content=body,
                headers={"Content-Type": "application/json"},
            )
            r.raise_for_status()
            raw = r.json()
            payload = raw[0] if isinstance(raw, list) else raw
            meta = payload.get("meta_info", {}) or {}
            tok_lp_raw: list[Any] = meta.get("output_token_logprobs") or []
            if not tok_lp_raw:
                out.append(float("nan"))
                continue
            total = 0.0
            for entry in tok_lp_raw:
                total += float(entry[0])
            out.append(total)
        return out

    def score_label_logprobs(
        self,
        position_id: int,
        turn1_assistant_text: str,
        turn2_user_text: str,
        labels: list[str],
    ) -> list[float]:
        """First-token logprob of each `label` under the injected prefix — one call.

        The N-request `score_continuations` path is fatally slow over
        `input_embeds`: SGLang parses the ~10 MB embed body serially and never
        batches it, so re-sending the same prefix once per label caps
        ConstrainedLingConf at ~0.4 claims/s. This sends the prefix *once*,
        decodes a single token, and reads every label's logprob from
        `meta_info.output_token_ids_logprobs` via `token_ids_logprob`.

        Model-agnostic by design (runs on every AV in the registry): label
        token ids are resolved from *this* model's tokenizer at call time. We
        read the leading-space BPE first token of each label and assert the
        five first tokens are distinct — first-token reading only discriminates
        labels that differ at token 0, so a collision (e.g. a tokenizer that
        splits "low"/"lowest" to a shared first piece) must fail loud, not
        silently squash two labels together. Returns logprobs aligned to
        `labels`; a label id absent from the response gets `nan`.
        """
        tok = self.av.tokenizer
        token_ids = [int(tok.encode(" " + lab, add_special_tokens=False)[0]) for lab in labels]
        assert len(set(token_ids)) == len(labels), (
            f"labels share a first token under this model's tokenizer "
            f"({dict(zip(labels, token_ids))}) — pick single-token-distinct labels."
        )

        prefix_embeds = self._build_multi_turn_embeds(
            position_id, turn1_assistant_text, turn2_user_text
        )
        body = orjson.dumps(
            {
                "input_embeds": prefix_embeds,
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
                "return_logprob": True,
                "token_ids_logprob": token_ids,
            },
            option=orjson.OPT_SERIALIZE_NUMPY,
        )
        r = self._http.post(
            f"{self.model.sglang_url.rstrip('/')}/generate",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        raw = r.json()
        payload = raw[0] if isinstance(raw, list) else raw
        meta = payload.get("meta_info", {}) or {}
        rows = (meta.get("output_token_ids_logprobs") or [[]])[0]  # first decoded position
        by_id = {int(entry[1]): float(entry[0]) for entry in rows}
        return [by_id.get(tid, float("nan")) for tid in token_ids]

    # batched verbalized scoring
    #
    # SGLang serializes every `input_embeds` body in its event loop and never
    # batches separate requests, so per-claim calls stall the GPU at ~0.4
    # claims/s. Packing many prefixes into ONE request's `input_embeds` batch
    # dimension lets the server parse once and prefill them together — ~10×
    # throughput (measured: batch=32 ≈ 4 claims/s, 4× concurrent ≈ 5.3).
    # We're parse-bound, not compute-bound; bigger batches past ~32 don't help
    # (the body grows linearly). Callers chunk + run a few batches concurrently.

    def _build_batch_embeds(
        self, items: list[tuple[int, str, str]]
    ) -> list[np.ndarray]:
        return [self._build_multi_turn_embeds(p, t1, t2) for p, t1, t2 in items]

    def score_label_logprobs_batch(
        self, items: list[tuple[int, str, str]], labels: list[str]
    ) -> list[list[float]]:
        """Batched `score_label_logprobs` — one request for many claims.

        `items[i]` = `(position_id, turn1_assistant_text, turn2_user_text)`.
        Returns per-item label logprobs aligned to `labels` (nan for a missing
        id). Same single-token-distinct contract as the single-claim path.
        """
        if not items:
            return []
        tok = self.av.tokenizer
        token_ids = [int(tok.encode(" " + lab, add_special_tokens=False)[0]) for lab in labels]
        assert len(set(token_ids)) == len(labels), (
            f"labels share a first token under this model's tokenizer "
            f"({dict(zip(labels, token_ids))}) — pick single-token-distinct labels."
        )
        payloads = self._post_generate_batch(
            self._build_batch_embeds(items),
            {"temperature": 0.0, "max_new_tokens": 1},
            return_logprob=True,
            token_ids_logprob=token_ids,
        )
        out: list[list[float]] = []
        for payload in payloads:
            meta = payload.get("meta_info", {}) or {}
            rows = (meta.get("output_token_ids_logprobs") or [[]])[0]
            by_id = {int(entry[1]): float(entry[0]) for entry in rows}
            out.append([by_id.get(tid, float("nan")) for tid in token_ids])
        return out

    def generate_text_batch(
        self, items: list[tuple[int, str, str]], *, regex: str, max_new_tokens: int
    ) -> list[str]:
        """Batched constrained-generation — one request for many claims.

        Returns the generated text per item (e.g. the numeric string for
        FreeformNumConf). Empty items → empty list.
        """
        if not items:
            return []
        payloads = self._post_generate_batch(
            self._build_batch_embeds(items),
            {"temperature": 0.0, "max_new_tokens": int(max_new_tokens), "regex": regex},
        )
        return [str(p.get("text", "") or "") for p in payloads]

    def _post_generate_batch(
        self,
        input_embeds: list[np.ndarray],
        sampling_params: dict,
        **extra: Any,
    ) -> list[dict]:
        """POST a batched `/generate` and return the per-sequence payload list."""
        body = orjson.dumps(
            {"input_embeds": input_embeds, "sampling_params": sampling_params, **extra},
            option=orjson.OPT_SERIALIZE_NUMPY,
        )
        r = self._http.post(
            f"{self.model.sglang_url.rstrip('/')}/generate",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        raw = r.json()
        return raw if isinstance(raw, list) else [raw]

    # cache I/O
    def _load_caches(self) -> None:
        if self._explanations_path.exists():
            df = pl.read_parquet(self._explanations_path)
            for row in df.iter_rows(named=True):
                key = (
                    int(row["position_id"]),
                    float(row["temperature"]),
                    int(row["sample_idx"]),
                    int(row["code_version"]),
                )
                self._explanations[key] = Generation(
                    explanation=row["explanation"],
                    tok_logprobs=list(row["tok_logprobs"]),
                    tok_strings=list(row["tok_strings"]),
                )
        if self._critic_path.exists():
            df = pl.read_parquet(self._critic_path)
            for row in df.iter_rows(named=True):
                key = (
                    int(row["position_id"]),
                    str(row["explanation_sha256"]),
                    int(row["code_version"]),
                )
                self._critic_scores[key] = (float(row["mse"]), float(row["cos"]))

    def flush(self) -> None:
        """Persist in-memory caches to Parquet. Idempotent — only writes if dirty."""
        if self._explanations_dirty:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "position_id": pid,
                    "temperature": temp,
                    "sample_idx": s_idx,
                    "code_version": cv,
                    "explanation": g.explanation,
                    "tok_logprobs": g.tok_logprobs,
                    "tok_strings": g.tok_strings,
                }
                for (pid, temp, s_idx, cv), g in self._explanations.items()
            ]
            df = pl.DataFrame(
                rows,
                schema={
                    "position_id": pl.UInt64,
                    "temperature": pl.Float32,
                    "sample_idx": pl.UInt8,
                    "code_version": pl.UInt8,
                    "explanation": pl.Utf8,
                    "tok_logprobs": pl.List(pl.Float32),
                    "tok_strings": pl.List(pl.Utf8),
                },
            )
            _atomic_write_parquet(df, self._explanations_path)
            self._explanations_dirty = False

        if self._critic_dirty:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "position_id": pid,
                    "explanation_sha256": h,
                    "code_version": cv,
                    "mse": mse,
                    "cos": cos,
                }
                for (pid, h, cv), (mse, cos) in self._critic_scores.items()
            ]
            df = pl.DataFrame(
                rows,
                schema={
                    "position_id": pl.UInt64,
                    "explanation_sha256": pl.Utf8,
                    "code_version": pl.UInt8,
                    "mse": pl.Float32,
                    "cos": pl.Float32,
                },
            )
            _atomic_write_parquet(df, self._critic_path)
            self._critic_dirty = False

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._http.close()

    def __enter__(self) -> NLAClientLite:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
