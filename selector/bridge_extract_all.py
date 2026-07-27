"""selector/bridge_extract_all.py — all-token bridge corpus, phase 1.

A deployed selector must rank EVERY position, so this extractor covers the full
rendered transcript — chat template tokens included — where the original bridge
extractors covered only a probed subset (OPI: user-content tokens; response
tasks: the first 30 response tokens). Per token it captures the NLA-layer
activation and the blind cheap signals, plus a `region` label
(template/system/user/assistant/assistant_prior) and `probe_tok_idx`, the
token's index in the OLD probed subset (-1 if new) — the join key that lets
seed_from_cache.py reuse the cached NLA explanations and judge labels.

Transcript sources per kind (fixed with the experiment design):
  opi   — full rendered prompt of the same seed-0 sample of 800 cases; no
          generation (input-side task). attn_rollout included as the baseline.
  tt    — prompt from tensortrust_cases.jsonl + the response tokens STORED in
          the old corpus parquet (regeneration is not reproducible; prefix
          identity keeps the probed tokens' signals unchanged).
  taboo — same stored-token scheme; the word's LoRA adapter is active during
          the forward, as in the original extraction.
  liars — the same seed-0 1000+1000 per-class sample, full transcript, full
          last response (no 30-token cap).
  hand  — the 27 authored cases, all tokens not just assistant.

Output is a DIRECTORY of parquet shards (whole cases per shard, atomic writes,
resumable); a `.done` sentinel marks completion. Sharding exists because the
all-token corpora are orders of magnitude bigger than the originals (a Liars'
corpus no longer fits in memory as one file).

Usage:
    python selector/bridge_extract_all.py --selftest
    python selector/bridge_extract_all.py --kind tt \
        --base-model Qwen/Qwen2.5-7B-Instruct --layer 20 --d-model 3584 \
        --old-corpus results/bridge/tt_q7_corpus.parquet \
        --out results/bridge/all_tt_q7_corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
from _response_extract import (  # noqa: E402
    _locate_spans,
    _message_views,
    build_messages,
    load_base,
    make_hook,
)
from signals_deception import N_SINK, _attn_signals, _response_view  # noqa: E402

SIGNALS = ["surprisal", "entropy", "varentropy", "temporal_kl", "resid_jump",
           "lookback_ratio", "sink_drain", "head_disagreement"]


# --------------------------------------------------------------------------- #
# Sharded corpus writer (whole cases per shard, atomic, resumable)            #
# --------------------------------------------------------------------------- #
def _schema(d_model: int, signals: list[str]) -> pa.Schema:
    fields = [
        ("position_id", pa.uint64()),
        ("activation", pa.list_(pa.float32(), d_model)),
        ("case_id", pa.string()),
        ("mode", pa.string()),
        ("tok_idx", pa.int64()),
        ("token", pa.string()),
        ("label", pa.int64()),
        ("region", pa.string()),
        ("probe_tok_idx", pa.int64()),
    ]
    fields += [(s, pa.float32()) for s in signals]
    return pa.schema(fields)


def scan_shards(out_dir: Path) -> tuple[set[str], int, int, int]:
    """Resume info from existing shards: (done case_ids, next position_id,
    n_rows so far, next shard index). Reads only the two id columns."""
    done: set[str] = set()
    next_pid = 0
    n_rows = 0
    next_shard = 0
    for f in sorted(Path(out_dir).glob("shard-*.parquet")):
        t = pq.read_table(f, columns=["case_id", "position_id"])
        done |= set(t["case_id"].to_pylist())
        if t.num_rows:
            next_pid = max(next_pid, max(t["position_id"].to_pylist()) + 1)
        n_rows += t.num_rows
        next_shard = max(next_shard, int(f.stem.split("-")[1]) + 1)
    return done, next_pid, n_rows, next_shard


class ShardWriter:
    """Buffers whole cases (meta dicts + a float32 activation matrix per case)
    and flushes a shard once `rows_per_shard` is crossed. Activations stay
    numpy until the arrow write, so no Python float lists at corpus scale."""

    def __init__(self, out_dir: Path, d_model: int, signals: list[str],
                 rows_per_shard: int, next_shard: int = 0):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.schema = _schema(d_model, signals)
        self.d_model = d_model
        self.signals = signals
        self.rows_per_shard = rows_per_shard
        self.shard_idx = next_shard
        self._meta: list[dict] = []
        self._acts: list[np.ndarray] = []
        self.total_rows = 0

    def add_case(self, meta_rows: list[dict], acts: np.ndarray) -> None:
        assert acts.shape == (len(meta_rows), self.d_model), acts.shape
        self._meta.extend(meta_rows)
        self._acts.append(np.ascontiguousarray(acts, dtype=np.float32))
        self.total_rows += len(meta_rows)
        if len(self._meta) >= self.rows_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self._meta:
            return
        flat = np.concatenate(self._acts, axis=0).ravel()
        act_arr = pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()),
                                                    self.d_model)
        cols: dict[str, pa.Array] = {}
        for f in self.schema:
            if f.name == "activation":
                cols[f.name] = act_arr
            else:
                cols[f.name] = pa.array([m[f.name] for m in self._meta], type=f.type)
        table = pa.table(cols, schema=self.schema)
        path = self.out_dir / f"shard-{self.shard_idx:05d}.parquet"
        tmp = path.with_suffix(".tmp.parquet")
        pq.write_table(table, tmp)
        tmp.replace(path)
        print(f"  wrote {path.name} ({table.num_rows} rows)", flush=True)
        self.shard_idx += 1
        self._meta, self._acts = [], []

    def finish(self, mark_done: bool = True) -> None:
        self.flush()
        if mark_done:
            (self.out_dir / ".done").touch()


# --------------------------------------------------------------------------- #
# Region / probe helpers (pure python, selftested)                            #
# --------------------------------------------------------------------------- #
def _mark_span(regions: list[str], offsets, span: tuple[int, int], role: str) -> None:
    """Set region=role for tokens whose char span falls inside `span`."""
    c0, c1 = span
    if c0 < 0:
        return
    for ti, (s, e) in enumerate(offsets):
        if e > s and s >= c0 and e <= c1:
            regions[ti] = role


def _taboo_case(case_id: str) -> tuple[str, int]:
    """`<word>__p<prompt>__s<sample>` -> (word, prompt index)."""
    word, p, _s = case_id.split("__")
    return word, int(p[1:])


# The ORIGINAL extractors' generation budgets. A stored reply at exactly the old
# probe cap was (possibly) truncated, so its tail is regenerated greedily up to
# this budget; shorter replies ended naturally and get no continuation.
ORIG_MAX_NEW = {"tt": 64, "taboo": 48}


def _tail_budget(n_stored: int, old_cap: int, orig_max: int) -> int:
    """How many reply tokens to regenerate beyond the stored ones."""
    return max(0, orig_max - n_stored) if n_stored >= old_cap else 0


def _continue_reply(model, tokenizer, device, ids: list[int], budget: int) -> list[int]:
    """Greedy continuation of a stored (possibly truncated) reply, up to the
    original generation budget, cut at the first special token (EOS / end of
    turn). A reply that was in fact complete emits EOS immediately and
    contributes nothing. Deterministic given the stored prefix."""
    import torch

    t = torch.tensor([ids], device=device)
    with torch.no_grad():
        gen = model.generate(t, attention_mask=torch.ones_like(t),
                             max_new_tokens=budget, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    specials = set(tokenizer.all_special_ids)
    out: list[int] = []
    for i in gen[0][len(ids):].tolist():
        if i in specials:
            break
        out.append(i)
    return out


def _stored_responses(old_corpus: Path) -> dict[str, tuple[list[str], int]]:
    """Old corpus parquet -> case_id -> (token strings in tok_idx order, label)."""
    t = pq.read_table(old_corpus, columns=["case_id", "tok_idx", "token", "label"])
    by_case: dict[str, list[tuple[int, str, int]]] = {}
    for cid, ti, tok, lab in zip(t["case_id"].to_pylist(), t["tok_idx"].to_pylist(),
                                 t["token"].to_pylist(), t["label"].to_pylist(), strict=True):
        by_case.setdefault(cid, []).append((int(ti), tok, int(lab)))
    out = {}
    for cid, rows in by_case.items():
        rows.sort()
        assert [r[0] for r in rows] == list(range(len(rows))), f"gap in tok_idx for {cid}"
        out[cid] = ([r[1] for r in rows], rows[0][2])
    return out


# --------------------------------------------------------------------------- #
# The all-position scorer (one teacher-forced forward)                        #
# --------------------------------------------------------------------------- #
def score_all(model, tokenizer, device, captured, ids: list[int], a_bound: int,
              attn_max_len: int, want_rollout: bool = False):
    """Signals + NLA-layer activations for EVERY position of `ids`. Position 0
    has no prediction, so its logit/residual signals are NaN; attention signals
    cover all query rows. Returns (activations [n, d], {signal: [n] array})."""
    import torch

    n = len(ids)
    want_attn = n <= attn_max_len
    t = torch.tensor([ids], device=device)
    with torch.no_grad():
        captured.clear()
        out = model(t, output_attentions=want_attn, output_hidden_states=True)
    logits = out.logits[0]
    hs = out.hidden_states[-1][0].float()
    h_layer = captured["h"][0].to(torch.float32).cpu().numpy()

    r = torch.arange(1, n, device=device)
    rows1 = logits.index_select(0, r - 1).float()
    logp = torch.log_softmax(rows1, dim=-1)
    prob = logp.exp()
    tgt = torch.tensor(ids[1:], device=device)
    surp = -logp.gather(1, tgt.view(-1, 1)).squeeze(1)
    ent = -(prob * logp).sum(-1)
    varent = (prob * (-logp - ent.unsqueeze(1)) ** 2).sum(-1)
    rows2 = logits.index_select(0, (r - 2).clamp(min=0)).float()
    logq = torch.log_softmax(rows2, dim=-1)
    tkl = torch.where(r >= 2, (prob * (logp - logq)).sum(-1),
                      torch.tensor(float("nan"), device=device))
    del rows1, rows2, logq, prob
    rjump = (hs[1:] - hs[:-1]).norm(dim=-1)

    def full(a: torch.Tensor) -> np.ndarray:  # prepend NaN for position 0
        return np.concatenate([[np.nan], a.cpu().numpy()])

    sig = {"surprisal": full(surp), "entropy": full(ent), "varentropy": full(varent),
           "temporal_kl": full(tkl), "resid_jump": full(rjump)}

    lb = sk = hd = np.full(n, np.nan)
    roll_v = np.full(n, np.nan)
    if want_attn and out.attentions is not None:
        sink_n = min(N_SINK, a_bound)
        n_layers = len(out.attentions)
        r_all = torch.arange(0, n, device=device)
        lb_t, snk_t, hd_t = _attn_signals([a[0] for a in out.attentions], r_all,
                                          a_bound, sink_n)
        lb = lb_t.cpu().numpy()
        sk = (-(snk_t / n_layers)).cpu().numpy() if sink_n > 0 else np.full(n, np.nan)
        hd = hd_t.cpu().numpy()
        if want_rollout:
            from signals_injection import attention_rollout
            attns_np = [a[0].float().cpu().numpy() for a in out.attentions]
            roll = attention_rollout(attns_np)
            roll_v = roll[n - 1, :].copy()
    sig["lookback_ratio"] = lb
    sig["sink_drain"] = sk
    sig["head_disagreement"] = hd
    if want_rollout:
        sig["attn_rollout"] = roll_v
    del out
    if want_attn:
        torch.cuda.empty_cache()
    return h_layer, sig


# --------------------------------------------------------------------------- #
# Per-kind case builders — each yields                                        #
#   dict(case_id, mode, ids, regions, labels, probe, a_bound)                 #
# --------------------------------------------------------------------------- #
def _iter_opi(args, tokenizer):
    import random

    from signals_injection import _labels_for_view, _user_token_view
    from signals_injection import load_cases as load_opi_cases

    allc = load_opi_cases(Path(args.cases), None)
    if args.max_cases and len(allc) > args.max_cases:
        cases = random.Random(0).sample(allc, args.max_cases)  # ORIGINAL run's sample
    else:
        cases = allc
    for c in cases:
        try:
            ids, view = _user_token_view(tokenizer, c.system, c.user_full)
        except (AssertionError, ValueError):
            continue
        if not view:
            continue
        vlabels = _labels_for_view(view, c.inj_start, c.inj_end)
        msgs = [{"role": "system", "content": c.system},
                {"role": "user", "content": c.user_full}]
        rendered = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                                 tokenize=False)
        enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        regions = ["template"] * len(ids)
        _mark_span(regions, enc["offset_mapping"], _locate_spans(rendered, [c.system])[0],
                   "system")
        labels = [0] * len(ids)
        probe = [-1] * len(ids)
        for k, (ti, _cs, _ce) in enumerate(view):
            regions[ti] = "user"
            labels[ti] = int(vlabels[k])
            probe[ti] = k
        yield {"case_id": c.id, "mode": "injection_input", "ids": ids,
               "regions": regions, "labels": labels, "probe": probe,
               "a_bound": view[0][0]}


def _stored_reply_case(tokenizer, system, user, stored_tokens, label, case_id, mode,
                       old_tok_cap, gen_budget=0):
    """Shared tt/taboo builder: rendered generation prompt + stored response ids."""
    msgs = build_messages(tokenizer, system, user)
    rendered = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                             tokenize=False)
    enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids = enc["input_ids"]
    regions = ["template"] * len(prompt_ids)
    spans = _locate_spans(rendered, [system or "", user])
    _mark_span(regions, enc["offset_mapping"], spans[0], "system")
    _mark_span(regions, enc["offset_mapping"], spans[1], "user")
    resp_ids = tokenizer.convert_tokens_to_ids(stored_tokens)
    if any(i is None for i in resp_ids):
        return None
    unk = tokenizer.unk_token_id
    if unk is not None and any(i == unk and t != tokenizer.unk_token
                               for i, t in zip(resp_ids, stored_tokens, strict=True)):
        return None
    ids = list(prompt_ids) + list(resp_ids)
    regions += ["assistant"] * len(resp_ids)
    labels = [int(label)] * len(ids)
    probe = [-1] * len(prompt_ids) + [k if k < old_tok_cap else -1
                                      for k in range(len(resp_ids))]
    return {"case_id": case_id, "mode": mode, "ids": ids, "regions": regions,
            "labels": labels, "probe": probe, "a_bound": len(prompt_ids),
            "gen_budget": gen_budget}


def _iter_tt(args, tokenizer):
    cases = [json.loads(ln) for ln in Path(args.cases).read_text().splitlines()
             if ln.strip()]
    stored = _stored_responses(Path(args.old_corpus))
    for c in cases:
        st = stored.get(c["id"])
        if st is None:
            continue
        budget = _tail_budget(len(st[0]), args.old_tok_cap,
                              args.orig_max_new or ORIG_MAX_NEW["tt"])
        got = _stored_reply_case(tokenizer, c["system"], c["user"], st[0], c["label"],
                                 c["id"], "injection", args.old_tok_cap, budget)
        if got is None:
            print(f"  skip {c['id']}: stored tokens do not round-trip", flush=True)
            continue
        yield got


def _iter_taboo_word(args, tokenizer, word: str, stored):
    from taboo import PROMPTS

    for case_id in sorted(stored):
        if not case_id.startswith(f"{word}__"):
            continue
        _w, pj = _taboo_case(case_id)
        budget = _tail_budget(len(stored[case_id][0]), args.old_tok_cap,
                              args.orig_max_new or ORIG_MAX_NEW["taboo"])
        got = _stored_reply_case(tokenizer, None, PROMPTS[pj], stored[case_id][0], 1,
                                 case_id, "taboo", args.old_tok_cap, budget)
        if got is None:
            print(f"  skip {case_id}: stored tokens do not round-trip", flush=True)
            continue
        yield got


def _iter_liars(args, tokenizer):
    from bridge_extract_liars import _sample

    rows_in = _sample(args.datasets, args.model_tag, args.per_class, args.seed)
    print(f"sampled {len(rows_in)} transcripts "
          f"({sum(r['deceptive'] for r in rows_in)} lying)", flush=True)
    for c in rows_in:
        try:
            ids, resp, a_resp = _response_view(tokenizer, c["messages"])
        except (ValueError, AssertionError):
            continue
        ids2, regions = _message_views(tokenizer, c["messages"])
        if ids2 != ids:  # same render+tokenize; a mismatch means offsets drifted
            regions = ["template"] * len(ids)
        labels = [int(c["deceptive"])] * len(ids)
        probe = [-1] * len(ids)
        for k, p in enumerate(resp):
            regions[p] = "assistant"
            if k < args.old_tok_cap:
                probe[p] = k
        yield {"case_id": c["id"], "mode": "deception", "ids": ids, "regions": regions,
               "labels": labels, "probe": probe, "a_bound": a_resp}


def _iter_hand(args, tokenizer):
    from signals import _assistant_token_labels, _build_ids
    from signals import load_cases as load_hand_cases

    for c in load_hand_cases(Path(args.cases)):
        asst_ids, alabels = _assistant_token_labels(tokenizer, c.assistant, c.label_spans)
        pre = _build_ids(tokenizer, c.system, c.user_full)
        msgs = [{"role": "system", "content": c.system},
                {"role": "user", "content": c.user_full}]
        rendered = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                                 tokenize=False)
        enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        regions = ["template"] * len(pre)
        spans = _locate_spans(rendered, [c.system, c.user_full])
        _mark_span(regions, enc["offset_mapping"], spans[0], "system")
        _mark_span(regions, enc["offset_mapping"], spans[1], "user")
        ids = pre + asst_ids
        regions += ["assistant"] * len(asst_ids)
        labels = [0] * len(pre) + [int(v) for v in alabels]
        probe = [-1] * len(pre) + list(range(len(asst_ids)))
        yield {"case_id": c.id, "mode": c.mode, "ids": ids, "regions": regions,
               "labels": labels, "probe": probe, "a_bound": len(pre)}


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #
def _emit(writer: ShardWriter, tokenizer, case, h_layer, sig, signals, position_id):
    ids = case["ids"]
    toks = tokenizer.convert_ids_to_tokens(ids)
    meta = []
    for j in range(len(ids)):
        row = {"position_id": position_id, "case_id": case["case_id"],
               "mode": case["mode"], "tok_idx": j, "token": toks[j],
               "label": case["labels"][j], "region": case["regions"][j],
               "probe_tok_idx": case["probe"][j]}
        for s in signals:
            row[s] = float(sig[s][j])
        meta.append(row)
        position_id += 1
    writer.add_case(meta, h_layer[: len(ids)])
    return position_id


def run(args: argparse.Namespace) -> int:
    signals = [*SIGNALS, "attn_rollout"] if args.kind == "opi" else SIGNALS
    out_dir = Path(args.out)
    if (out_dir / ".done").exists() and not args.force:
        print(f"{out_dir} already complete (.done); nothing to do")
        return 0
    done, position_id, prev_rows, next_shard = scan_shards(out_dir)
    print(f"{args.kind} {args.base_model}: {len(done)} cases cached "
          f"({prev_rows} rows), resuming at position_id={position_id}", flush=True)

    model, tokenizer, device = load_base(args.base_model, args.dtype)
    handle, captured = make_hook(model, args.layer)
    writer = ShardWriter(out_dir, args.d_model, signals, args.rows_per_shard,
                         next_shard=next_shard)

    def process(case_iter, fwd_model):
        nonlocal position_id
        n_done = 0
        for case in case_iter:
            if case["case_id"] in done:
                continue
            if args.limit and n_done >= args.limit:
                break
            if case.get("gen_budget"):
                try:
                    tail = _continue_reply(fwd_model, tokenizer, device,
                                           case["ids"], case["gen_budget"])
                except Exception as e:  # noqa: BLE001
                    print(f"  tail fail {case['case_id']}: {e}", flush=True)
                    tail = []
                if tail:
                    case["ids"] = case["ids"] + tail
                    case["regions"] = case["regions"] + ["assistant"] * len(tail)
                    case["labels"] = case["labels"] + [case["labels"][-1]] * len(tail)
                    case["probe"] = case["probe"] + [-1] * len(tail)
            try:
                h_layer, sig = score_all(fwd_model, tokenizer, device, captured,
                                         case["ids"], case["a_bound"],
                                         args.attn_max_len,
                                         want_rollout=(args.kind == "opi"))
            except Exception as e:  # noqa: BLE001 — one bad case must not kill the run
                print(f"  score fail {case['case_id']}: {e}", flush=True)
                continue
            position_id = _emit(writer, tokenizer, case, h_layer, sig, signals,
                                position_id)
            n_done += 1
            if n_done % 50 == 0:
                print(f"  {n_done} cases this run ({writer.total_rows} buffered+written "
                      f"rows)", flush=True)
        return n_done

    if args.kind == "taboo":
        from peft import PeftModel
        from taboo import WORDS, adapter_id
        stored = _stored_responses(Path(args.old_corpus))
        words = args.words or list(WORDS)
        peft = PeftModel.from_pretrained(model, adapter_id(args.short, words[0]),
                                         adapter_name=words[0])
        for w in words[1:]:
            peft.load_adapter(adapter_id(args.short, w), adapter_name=w)
        for w in words:
            peft.set_adapter(w)
            n = process(_iter_taboo_word(args, tokenizer, w, stored), peft)
            print(f"  word {w}: {n} cases", flush=True)
    else:
        iters = {"opi": _iter_opi, "tt": _iter_tt, "liars": _iter_liars,
                 "hand": _iter_hand}
        process(iters[args.kind](args, tokenizer), model)

    handle.remove()
    writer.finish(mark_done=not args.limit)  # a --limit smoke run is not complete
    print(f"done: {out_dir} now holds {prev_rows + writer.total_rows} rows")
    return 0


# --------------------------------------------------------------------------- #
# Offline self-test                                                            #
# --------------------------------------------------------------------------- #
class _StubTok:
    """Minimal offset-returning tokenizer over a fixed render."""

    unk_token_id = None
    unk_token = None

    def __init__(self, rendered, toks):
        self._rendered = rendered
        self._toks = toks  # list of (start, end)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return self._rendered

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"input_ids": list(range(len(self._toks))),
                "offset_mapping": list(self._toks)}


def selftest() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # _locate_spans: sequential, repeated content, strip-tolerance, missing
    sp = _locate_spans("A B A C", ["A", "A"])
    check("locate repeated content advances", sp == [(0, 1), (4, 5)])
    sp = _locate_spans("xx hello xx", ["  hello  "])
    check("locate strip-tolerant", sp == [(3, 8)])
    check("locate missing -> (-1,-1)", _locate_spans("abc", ["zzz"]) == [(-1, -1)])

    # _message_views: template scaffolding vs roles, assistant_prior for earlier turns
    rendered = "<u>Q1</u><a>R1</a><u>Q2</u><a>R2</a>"
    toks = [(0, 3), (3, 5), (5, 9), (9, 12), (12, 14), (14, 18),
            (18, 21), (21, 23), (23, 27), (27, 30), (30, 32), (32, 36)]
    tok = _StubTok(rendered, toks)
    msgs = [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "R1"},
            {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "R2"}]
    _ids, regions = _message_views(tok, msgs)
    check("message_views roles",
          regions == ["template", "user", "template", "template", "assistant_prior",
                      "template", "template", "user", "template", "template",
                      "assistant", "template"])

    # _mark_span only inside the span, zero-width skipped
    regions = ["template"] * 4
    _mark_span(regions, [(0, 2), (2, 2), (2, 4), (4, 6)], (2, 6), "user")
    check("mark_span respects span + zero-width", regions == ["template", "template",
                                                              "user", "user"])

    # taboo case-id parse
    check("taboo case parse", _taboo_case("moon__p2__s5") == ("moon", 2))

    # tail budget: only replies stored at the old cap get a continuation, up to
    # the ORIGINAL generation budget
    check("tail budget truncated tt", _tail_budget(30, 30, 64) == 34)
    check("tail budget truncated taboo", _tail_budget(30, 30, 48) == 18)
    check("tail budget complete reply", _tail_budget(12, 30, 64) == 0)

    # stored-reply builder: regions/probe/labels shapes with a stub tokenizer
    class _TTTok(_StubTok):
        def __init__(self):
            super().__init__("<s>SYS</s><u>USR</u><asst>",
                             [(0, 3), (3, 6), (6, 10), (10, 13), (13, 16),
                              (16, 20), (20, 26)])

        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=False):
            return self._rendered

        def convert_tokens_to_ids(self, toks):
            return [100 + i for i in range(len(toks))]

    got = _stored_reply_case(_TTTok(), "SYS", "USR", ["t0", "t1", "t2"], 1,
                             "c1", "injection", old_tok_cap=2)
    check("stored-reply lengths", len(got["ids"]) == 10 and len(got["regions"]) == 10)
    check("stored-reply regions",
          got["regions"] == ["template", "system", "template", "template", "user",
                             "template", "template", "assistant", "assistant",
                             "assistant"])
    check("stored-reply probe caps at old_tok_cap",
          got["probe"] == [-1] * 7 + [0, 1, -1])
    check("stored-reply a_bound", got["a_bound"] == 7)
    check("stored-reply labels case-level", set(got["labels"]) == {1})

    # shard writer: whole cases, resume scan, .done sentinel
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        w = ShardWriter(Path(td), d_model=4, signals=SIGNALS, rows_per_shard=3)
        pid = 0
        for cid in ("a", "b", "c"):
            meta = []
            for j in range(2):
                row = {"position_id": pid, "case_id": cid, "mode": "m", "tok_idx": j,
                       "token": "t", "label": 0, "region": "user", "probe_tok_idx": -1}
                row.update({s: 0.5 for s in SIGNALS})
                meta.append(row)
                pid += 1
            w.add_case(meta, np.zeros((2, 4), dtype=np.float32))
        w.finish()
        done, next_pid, n_rows, next_shard = scan_shards(Path(td))
        check("shards hold whole cases", done == {"a", "b", "c"} and n_rows == 6)
        check("resume position_id", next_pid == 6)
        check("shard numbering resumes", next_shard >= 2)
        check(".done sentinel", (Path(td) / ".done").exists())

    print(f"\nselftest: {'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--kind", choices=["opi", "tt", "taboo", "liars", "hand"])
    p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--short", default="q7", choices=["q7", "g12", "g27", "l70"],
                   help="taboo only: which adapter family")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--d-model", type=int, default=3584)
    p.add_argument("--cases", default=None,
                   help="opi/tt/hand: cases file (defaults per kind)")
    p.add_argument("--old-corpus", default=None,
                   help="tt/taboo: original corpus parquet holding the reply tokens")
    p.add_argument("--old-tok-cap", type=int, default=30,
                   help="response tokens the ORIGINAL run probed (probe_tok_idx map)")
    p.add_argument("--orig-max-new", type=int, default=None,
                   help="tt/taboo: original generation budget for tail regeneration "
                        "(default: 64 tt / 48 taboo)")
    p.add_argument("--max-cases", type=int, default=800, help="opi sample size (seed 0)")
    p.add_argument("--datasets", nargs="+", default=None, help="liars")
    p.add_argument("--model-tag", default=None, help="liars generator tag")
    p.add_argument("--per-class", type=int, default=1000, help="liars")
    p.add_argument("--seed", type=int, default=0, help="liars sample seed")
    p.add_argument("--words", nargs="+", default=None, help="taboo word subset")
    p.add_argument("--attn-max-len", type=int, default=2600)
    p.add_argument("--rows-per-shard", type=int, default=20000)
    p.add_argument("--limit", type=int, default=None, help="cases per run (smoke)")
    p.add_argument("--force", action="store_true", help="ignore .done sentinel")
    p.add_argument("--out", required=False, help="output DIRECTORY of shards")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = p.parse_args(argv)
    if not args.selftest:
        if not args.kind or not args.out:
            p.error("--kind and --out are required")
        defaults = {
            "opi": Path(__file__).parent / "data" / "injection_cases.jsonl",
            "tt": Path(__file__).parent / "data" / "tensortrust_cases.jsonl",
            "hand": Path(__file__).parent / "data" / "token_selector_cases.json",
        }
        if args.cases is None and args.kind in defaults:
            args.cases = str(defaults[args.kind])
        if args.kind in ("tt", "taboo") and not args.old_corpus:
            p.error(f"--old-corpus is required for --kind {args.kind}")
        if args.kind == "liars" and (not args.datasets or not args.model_tag):
            p.error("--datasets and --model-tag are required for --kind liars")
    return args


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    raise SystemExit(selftest() if args.selftest else run(args))
