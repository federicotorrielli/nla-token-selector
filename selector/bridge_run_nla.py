"""selector/bridge_run_nla.py — bridge test, phase 2.

Run the NLA (the AV decoder) at every token's activation from the phase-1 corpus,
saving one explanation per token. Uses the project's NLAClientLite, pointed at
our custom corpus and at a running AV SGLang server. Writes incrementally and
resumes, so a killed run continues.

`--corpus` may be a single parquet file (the original bridge corpora) or a
DIRECTORY of shards (the all-token corpora from bridge_extract_all.py). In
directory mode `--out` is a directory too: one explanation shard per corpus
shard, plus any `shard-cache.parquet` that seed_from_cache.py pre-seeded —
every position already present anywhere in the output directory is skipped.

Needs an SGLang server serving the AV checkpoint (e.g. kitft/nla-qwen2.5-7b-L20-av)
at --sglang-url.

Usage:
    python selector/bridge_run_nla.py \
        --corpus results/bridge/q7_corpus.parquet \
        --sglang-url http://localhost:30000 \
        --out results/bridge/q7_explanations.parquet
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

META_COLS = ["position_id", "case_id", "mode", "tok_idx", "token", "label"]


# --------------------------------------------------------------------------- #
# In-process engine path (--engine): ~2.9x the HTTP path, measured             #
# --------------------------------------------------------------------------- #
# The HTTP path sends every position's whole prompt-embedding matrix as JSON and
# blocks on whole batches. Measured on q7 (see selector/bench_engine.py):
#   HTTP, batch 32, serial          3.9 pos/s
#   HTTP, batch 32, 4 concurrent   12.4 pos/s
#   in-process engine, lockstep    16.0 pos/s
#   in-process engine, continuous  21.9 pos/s   <- this path
# The last step is the one that matters: explanations average ~145 tokens but the
# cap is 600, so a lockstep batch idles at the pace of its longest sequence.
# Keeping a fixed number of requests in flight refills each slot as it frees.
# CUDA graphs stay DISABLED (measured 5% — noise — and the NLA inference code
# disables them for the input_embeds path).
class EngineDecoder:
    """Drives SGLang in-process. `Engine.generate()` does not expose
    `input_embeds`, but it only builds a `GenerateReqInput` and awaits
    `tokenizer_manager.generate_request`, and that dataclass does carry it."""

    def __init__(self, av_repo: str, *, mem_fraction: float, inflight: int,
                 max_new_tokens: int, max_running_requests: int | None = None):
        import sglang as sgl
        import torch
        from huggingface_hub import snapshot_download

        from nla_token_selector._nla_inference import NLAClient

        self.torch = torch
        self.inflight = inflight
        self.max_new_tokens = max_new_tokens
        ckpt = snapshot_download(av_repo)
        # tokenizer + embedding layer only; never contacts the URL
        self.av = NLAClient(ckpt, sglang_url="http://127.0.0.1:1")
        kw = dict(model_path=ckpt, mem_fraction_static=mem_fraction,
                  disable_radix_cache=True, disable_cuda_graph=True)
        if max_running_requests:
            kw["max_running_requests"] = max_running_requests
        self.llm = sgl.Engine(**kw)

    def decode(self, activations: list) -> list[str]:
        """Explanations for a list of activation vectors, order preserved."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        from sglang.srt.managers.io_struct import GenerateReqInput

        from nla_token_selector._nla_inference import EXPLANATION_RE

        sp = {"temperature": 0.0, "max_new_tokens": self.max_new_tokens,
              "skip_special_tokens": False}

        def build(v):
            return self.av._build_embeds(self.torch.as_tensor(v),
                                         prompt_content=None)[0].tolist()

        async def one(sem, pool, v):
            async with sem:
                loop = asyncio.get_running_loop()
                payload = await loop.run_in_executor(pool, build, v)
                obj = GenerateReqInput(input_embeds=payload,
                                       sampling_params=dict(sp))
                gen = self.llm.tokenizer_manager.generate_request(obj, None)
                return await gen.__anext__()

        async def drive():
            sem = asyncio.Semaphore(self.inflight)
            with ThreadPoolExecutor(max_workers=8) as pool:
                return await asyncio.gather(*[one(sem, pool, v)
                                              for v in activations])

        outs = self.llm.loop.run_until_complete(drive())
        texts = []
        for o in outs:
            t = o.get("text", "") or ""
            m = EXPLANATION_RE.search(t)
            texts.append(m.group(1).strip() if m else t)
        return texts

    def close(self):
        self.llm.shutdown()


def _atomic_write(df, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)


def _decode_todo(client, meta: dict, todo: list[int], rows: list[dict], outp: Path,
                 args, checkpoint_every: int) -> None:
    """Decode `todo` positions, appending to `rows` and checkpointing `outp`."""
    import polars as pl

    for i in range(0, len(todo), args.batch):
        chunk = todo[i : i + args.batch]
        texts = client.generate_batch(chunk, temperature=0.0,
                                      max_new_tokens=args.max_new_tokens)
        for pid, txt in zip(chunk, texts, strict=True):
            m = meta[pid]
            rows.append({
                "position_id": pid, "case_id": m["case_id"], "mode": m["mode"],
                "tok_idx": int(m["tok_idx"]), "token": m["token"],
                "label": int(m["label"]), "explanation": txt,
            })
        if (i // args.batch) % checkpoint_every == checkpoint_every - 1 \
                or i + args.batch >= len(todo):
            _atomic_write(pl.DataFrame(rows), outp)
        print(f"  {min(i + args.batch, len(todo))}/{len(todo)}", flush=True)
    _atomic_write(pl.DataFrame(rows), outp)


def _my_shards(corpus_dir: Path, args) -> list[Path]:
    """This worker's slice of the corpus shards.

    Decodes are independent, so N workers (one per GPU) split the shards by
    `index % stride == offset` and never touch each other's output files. Each
    writes explanations named after the corpus shard it read, so the output
    directory reassembles itself with no merge step, and a worker that dies is
    restarted with the same offset to resume only its own share.
    """
    shards = sorted(corpus_dir.glob("shard-*.parquet"))
    assert shards, f"no shards under {corpus_dir}"
    if args.shard_stride <= 1:
        return shards
    assert 0 <= args.shard_offset < args.shard_stride, (
        f"--shard-offset must be in [0, {args.shard_stride})")
    mine = [s for i, s in enumerate(shards) if i % args.shard_stride == args.shard_offset]
    print(f"worker {args.shard_offset}/{args.shard_stride}: "
          f"{len(mine)} of {len(shards)} shards", flush=True)
    return mine


def _run_dir_engine(args) -> int:
    """Directory-of-shards mode over the in-process engine (the fast path)."""
    import polars as pl

    from nla_token_selector.settings import Settings

    shards = _my_shards(Path(args.corpus), args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    done: set[int] = set()
    for f in out_dir.glob("*.parquet"):
        done |= {int(p) for p in
                 pl.scan_parquet(str(f)).select("position_id").collect()
                 ["position_id"].to_list()}
    print(f"{len(shards)} corpus shards for this worker, "
          f"{len(done)} positions already explained", flush=True)

    dec = EngineDecoder(Settings().model.av_repo, mem_fraction=args.mem_fraction,
                        inflight=args.inflight, max_new_tokens=args.max_new_tokens,
                        max_running_requests=args.max_running_requests)
    budget = args.limit
    try:
        for shard in shards:
            df = pl.read_parquet(shard)
            todo_rows = [r for r in df.iter_rows(named=True)
                         if int(r["position_id"]) not in done]
            if budget is not None:
                todo_rows = todo_rows[:budget]
            if not todo_rows:
                continue
            outp = out_dir / shard.name
            rows = pl.read_parquet(outp).to_dicts() if outp.exists() else []
            print(f"[{shard.name}] {len(todo_rows)} to run, "
                  f"inflight={args.inflight}", flush=True)
            # chunk so a kill loses at most one chunk
            step = max(args.inflight * 4, 512)
            for i in range(0, len(todo_rows), step):
                part = todo_rows[i: i + step]
                texts = dec.decode([r["activation"] for r in part])
                for r, txt in zip(part, texts, strict=True):
                    rows.append({
                        "position_id": int(r["position_id"]), "case_id": r["case_id"],
                        "mode": r["mode"], "tok_idx": int(r["tok_idx"]),
                        "token": r["token"], "label": int(r["label"]),
                        "explanation": txt,
                    })
                _atomic_write(pl.DataFrame(rows), outp)
                print(f"  {min(i + step, len(todo_rows))}/{len(todo_rows)}", flush=True)
            done |= {int(r["position_id"]) for r in todo_rows}
            if budget is not None:
                budget -= len(todo_rows)
                if budget <= 0:
                    break
    finally:
        dec.close()
    return 0


def _run_dir(args) -> int:
    """Directory-of-shards mode. One NLAClientLite reused across shards; its
    corpus pointer is swapped per shard so activations never all sit in memory."""
    import polars as pl

    from nla_token_selector.client import NLAClientLite
    from nla_token_selector.settings import Settings

    corpus_dir = Path(args.corpus)
    shards = sorted(corpus_dir.glob("shard-*.parquet"))
    assert shards, f"no shards under {corpus_dir}"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    done: set[int] = set()
    for f in out_dir.glob("*.parquet"):
        done |= {int(p) for p in
                 pl.scan_parquet(str(f)).select("position_id").collect()
                 ["position_id"].to_list()}
    print(f"{len(shards)} corpus shards, {len(done)} positions already explained",
          flush=True)

    client = NLAClientLite(Settings())
    budget = args.limit
    try:
        for shard in shards:
            meta_df = (pl.scan_parquet(str(shard)).select(META_COLS).collect())
            pids = sorted(int(p) for p in meta_df["position_id"].to_list())
            todo = [p for p in pids if p not in done]
            if budget is not None:
                todo = todo[:budget]
            outp = out_dir / shard.name
            if not todo:
                continue
            print(f"[{shard.name}] {len(todo)} to run, batch={args.batch}", flush=True)
            # ponytail: swap the corpus pointer on our own shim instead of
            # rebuilding the client (keeps the AV embed weights loaded).
            client.settings.paths.corpus_parquet = shard.resolve()
            client._corpus = None
            client._act_index = None
            meta = {int(r["position_id"]): r for r in meta_df.iter_rows(named=True)}
            rows = pl.read_parquet(outp).to_dicts() if outp.exists() else []
            _decode_todo(client, meta, todo, rows, outp, args,
                         checkpoint_every=max(1, 2048 // args.batch))
            done |= set(todo)
            if budget is not None:
                budget -= len(todo)
                if budget <= 0:
                    break
    finally:
        client.close()
    return 0


def _selftest() -> int:
    """The shard partition must be disjoint AND complete: a worker set that
    silently drops a shard loses those tokens with no error anywhere."""
    import tempfile
    from types import SimpleNamespace

    ok = True
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for i in range(11):  # 11 shards over 4 workers: deliberately uneven
            (d / f"shard-{i:05d}.parquet").touch()
        for stride in (1, 2, 3, 4, 8, 16):
            seen: list[Path] = []
            for off in range(stride):
                seen += _my_shards(d, SimpleNamespace(shard_stride=stride,
                                                      shard_offset=off))
            disjoint = len(seen) == len(set(seen))
            complete = set(seen) == set(d.glob("shard-*.parquet"))
            ok = ok and disjoint and complete
            print(f"  [{'PASS' if disjoint and complete else 'FAIL'}] "
                  f"stride={stride}: {len(seen)} shards, disjoint={disjoint}, "
                  f"complete={complete}")
    print(f"\nselftest: {'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="offline check of the shard partition (no GPU)")
    ap.add_argument("--corpus", required=False)
    ap.add_argument("--out", required=False)
    ap.add_argument("--sglang-url", default="http://localhost:30000")
    ap.add_argument("--model-short", default="q7")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--limit", type=int, default=None, help="only first N positions (smoke test)")
    ap.add_argument("--engine", action="store_true",
                    help="drive SGLang in-process with continuous batching "
                         "(~2.9x the HTTP path; needs no launched AV server)")
    ap.add_argument("--inflight", type=int, default=256,
                    help="--engine: requests kept in flight")
    ap.add_argument("--mem-fraction", type=float, default=0.90,
                    help="--engine: mem_fraction_static")
    ap.add_argument("--max-running-requests", type=int, default=None,
                    help="--engine: server-side concurrency cap")
    ap.add_argument("--shard-stride", type=int, default=1,
                    help="number of parallel workers (one per GPU)")
    ap.add_argument("--shard-offset", type=int, default=0,
                    help="this worker's index in [0, stride)")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    if not args.corpus or not args.out:
        ap.error("--corpus and --out are required")

    # Configure the Settings tree before importing the package: point the corpus
    # and the AV server at ours. Settings() defaults to the default overlay.
    corpus_path = Path(args.corpus)
    first = (sorted(corpus_path.glob("shard-*.parquet"))[0]
             if corpus_path.is_dir() else corpus_path)
    os.environ["CNLA_EXPERIMENT__MODEL_SHORT"] = args.model_short
    os.environ["CNLA_PATHS__CORPUS_PARQUET"] = str(first.resolve())
    os.environ[f"CNLA_MODELS__{args.model_short.upper()}__SGLANG_URL"] = args.sglang_url

    if corpus_path.is_dir():
        return _run_dir_engine(args) if args.engine else _run_dir(args)

    import polars as pl

    from nla_token_selector.client import NLAClientLite
    from nla_token_selector.settings import Settings

    corpus = pl.read_parquet(args.corpus)
    meta = {int(r["position_id"]): r for r in corpus.iter_rows(named=True)}
    pids = [int(p) for p in corpus["position_id"].to_list()]
    if args.limit:
        pids = pids[: args.limit]

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    done: set[int] = set()
    if outp.exists():
        prev = pl.read_parquet(outp)
        rows = prev.to_dicts()
        done = {int(p) for p in prev["position_id"].to_list()}
    todo = [p for p in pids if p not in done]
    print(f"{len(todo)} to run ({len(done)} cached), batch={args.batch}", flush=True)

    settings = Settings()
    with NLAClientLite(settings) as client:
        _decode_todo(client, meta, todo, rows, outp, args, checkpoint_every=1)

    print(f"wrote {outp} ({len(rows)} explanations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
