"""selector/bench_engine.py — in-process SGLang Engine vs the HTTP path.

The AV decode dominates the all-token run, and over HTTP it is not GPU-bound:
every position ships its whole prompt embedding matrix (~prompt_len x d_model
float32) as JSON, which the client encodes and SGLang's single-threaded asyncio
frontend parses. SGLang's offline Engine API runs the same scheduler in-process,
so the arrays are handed over as Python objects: no JSON, no HTTP, no socket.

`Engine.generate()` does not expose `input_embeds`, but it only builds a
`GenerateReqInput` and awaits `tokenizer_manager.generate_request(obj, None)`,
and that dataclass does carry `input_embeds` — so we build the request object
ourselves and drive the same code path.

This script measures positions/second for the Engine path so the pipeline can be
switched on evidence.

Usage:
    python selector/bench_engine.py --corpus results/bridge/all_tt_q7_corpus \
        --model-short q7 --n 128 --batch 64
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="corpus parquet or shard dir")
    ap.add_argument("--model-short", default="q7")
    ap.add_argument("--n", type=int, default=128, help="positions to decode")
    ap.add_argument("--batch", type=int, default=64, help="positions per request")
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--mem-fraction-static", type=float, default=0.85)
    ap.add_argument("--cuda-graph", action="store_true",
                    help="ENABLE cuda graphs (default: disabled, as the NLA "
                         "inference code does). Verify outputs before trusting.")
    ap.add_argument("--save-out", default=None,
                    help="write the explanations to this json for an identity check")
    ap.add_argument("--mode", default="list", choices=["list", "numpy", "torch"],
                    help="how input_embeds are handed to the engine")
    ap.add_argument("--max-running-requests", type=int, default=None,
                    help="server-side concurrency cap (default: SGLang's own)")
    ap.add_argument("--attention-backend", default=None,
                    help="trtllm_mha (SGLang's default here) / flashinfer / fa3 / "
                         "triton / torch_native")
    ap.add_argument("--continuous", type=int, default=0, metavar="INFLIGHT",
                    help="continuous batching: keep INFLIGHT requests in flight and "
                         "refill each slot as it finishes, instead of blocking on "
                         "whole batches (a lockstep batch idles at the pace of its "
                         "longest sequence)")
    args = ap.parse_args(argv)

    corpus = Path(args.corpus)
    first = (sorted(corpus.glob("shard-*.parquet"))[0] if corpus.is_dir() else corpus)
    os.environ["CNLA_EXPERIMENT__MODEL_SHORT"] = args.model_short
    os.environ["CNLA_PATHS__CORPUS_PARQUET"] = str(first.resolve())

    import numpy as np
    import polars as pl
    import sglang as sgl
    import torch
    from huggingface_hub import snapshot_download
    from sglang.srt.managers.io_struct import GenerateReqInput

    from nla_token_selector._nla_inference import EXPLANATION_RE, NLAClient
    from nla_token_selector.settings import Settings

    settings = Settings()
    model = settings.model
    ckpt = snapshot_download(model.av_repo)

    df = pl.read_parquet(first, columns=["position_id", "activation"]).head(args.n)
    acts = [np.asarray(a, dtype=np.float32) for a in df["activation"].to_list()]
    print(f"{len(acts)} activations, d_model={len(acts[0])}")

    # Embed builder (tokenizer + embedding layer only; no server contact).
    t0 = time.time()
    av = NLAClient(ckpt, sglang_url="http://127.0.0.1:1")
    embeds = [av._build_embeds(torch.as_tensor(v), prompt_content=None)[0] for v in acts]
    build_s = time.time() - t0
    T, d = embeds[0].shape
    mb = embeds[0].nbytes / 1e6
    print(f"prompt embeds: {T} tokens x {d} = {mb:.1f} MB each (float32); "
          f"built {len(embeds)} in {build_s:.1f}s ({len(embeds) / build_s:.1f}/s)")
    print(f"  -> the HTTP path JSON-encodes ~{mb * 3:.0f} MB of text per position\n")

    print(f"starting in-process Engine (cuda_graph="
          f"{'ON' if args.cuda_graph else 'OFF'})...")
    engine_kw = dict(model_path=ckpt, mem_fraction_static=args.mem_fraction_static,
                     disable_radix_cache=True,
                     disable_cuda_graph=not args.cuda_graph)
    if args.max_running_requests:
        engine_kw["max_running_requests"] = args.max_running_requests
    if args.attention_backend:
        engine_kw["attention_backend"] = args.attention_backend
    llm = sgl.Engine(**engine_kw)
    try:
        rq = llm.tokenizer_manager.server_args.max_running_requests
        print(f"server max_running_requests = {rq}")
    except Exception:  # noqa: BLE001 — informational only
        pass
    all_texts: list[str] = []
    try:
        sp = {"temperature": 0.0, "max_new_tokens": args.max_new_tokens,
              "skip_special_tokens": False}
        # warmup
        _ = _run(llm, GenerateReqInput, embeds[: min(4, len(embeds))], sp, args.mode)
        print("warmup done; timing...\n")
        if args.continuous:
            t_all = time.time()
            outs = _run_continuous(llm, GenerateReqInput, embeds, sp, args.continuous)
            total = time.time() - t_all
            toks = sum((o.get("meta_info", {}) or {}).get("completion_tokens", 0)
                       for o in outs)
            texts = [o.get("text", "") for o in outs]
            all_texts.extend(texts)
            print(f"\nCONTINUOUS (inflight={args.continuous}): {len(outs)} positions "
                  f"in {total:.1f}s -> {len(outs) / total:.2f} pos/s  "
                  f"({toks} tokens, {toks / total:.0f} decode tok/s)")
            m = EXPLANATION_RE.search(texts[0]) if texts else None
            print(f"sample explanation: {(m.group(1) if m else texts[0])[:160]!r}")
            if args.save_out:
                import json
                Path(args.save_out).write_text(json.dumps(all_texts))
                print(f"wrote {len(all_texts)} explanations to {args.save_out}")
            return 0
        print(f"{'batch':>6} {'build_s':>8} {'gen_s':>8} {'gen_tok':>8} "
              f"{'pos/s':>8} {'decode tok/s':>13}")
        done = 0
        t_all = time.time()
        n_chars = 0
        while done < len(embeds):
            chunk = embeds[done: done + args.batch]
            t1 = time.time()
            outs = _run(llm, GenerateReqInput, chunk, sp, args.mode)
            dt = time.time() - t1
            texts = [o.get("text", "") for o in outs]
            all_texts.extend(texts)
            n_chars += sum(len(t) for t in texts)
            print(f"{len(chunk):>6} {_run.last_build:>8.1f} {_run.last_gen:>8.1f} "
                  f"{_run.last_tokens:>8} {len(chunk) / dt:>8.2f} "
                  f"{_run.last_tokens / max(_run.last_gen, 1e-9):>13.0f}")
            done += len(chunk)
        total = time.time() - t_all
        print(f"\nENGINE TOTAL: {done} positions in {total:.1f}s -> "
              f"{done / total:.2f} pos/s  (~{n_chars / 4 / total:.0f} gen tok/s)")
        m = EXPLANATION_RE.search(texts[0]) if texts else None
        print(f"sample explanation: {(m.group(1) if m else texts[0])[:160]!r}")
        if args.save_out:
            import json
            Path(args.save_out).write_text(json.dumps(all_texts))
            print(f"wrote {len(all_texts)} explanations to {args.save_out}")
    finally:
        llm.shutdown()
    return 0


def _run_continuous(llm, GenerateReqInput, embeds, sampling_params, inflight):
    """Continuous batching: hold `inflight` requests open at once and start the
    next position the moment a slot frees, so the GPU never waits for a batch's
    slowest sequence. The `.tolist()` runs in a thread so it overlaps decode."""
    import asyncio

    async def one(sem, pool, e):
        async with sem:
            loop = asyncio.get_running_loop()
            payload = await loop.run_in_executor(pool, e.tolist)
            obj = GenerateReqInput(input_embeds=payload,
                                   sampling_params=dict(sampling_params))
            gen = llm.tokenizer_manager.generate_request(obj, None)
            return await gen.__anext__()

    async def drive():
        from concurrent.futures import ThreadPoolExecutor
        sem = asyncio.Semaphore(inflight)
        with ThreadPoolExecutor(max_workers=8) as pool:
            return await asyncio.gather(*[one(sem, pool, e) for e in embeds])

    return llm.loop.run_until_complete(drive())


def _run(llm, GenerateReqInput, embeds, sampling_params, mode="list"):
    """One batched request straight through the tokenizer manager.

    `mode` controls how the embeddings are handed over. "list" materializes
    Python floats (prompt_len x d_model of them PER position — the thing to
    avoid); "numpy"/"torch" pass the buffers through untouched.
    """
    t0 = time.time()
    if mode == "list":
        payload = [e.tolist() for e in embeds]
    elif mode == "numpy":
        payload = list(embeds)
    else:
        import torch
        payload = [torch.as_tensor(e) for e in embeds]
    obj = GenerateReqInput(
        input_embeds=payload,
        sampling_params=[dict(sampling_params) for _ in embeds],
    )
    t1 = time.time()
    gen = llm.tokenizer_manager.generate_request(obj, None)
    ret = llm.loop.run_until_complete(gen.__anext__())
    t2 = time.time()
    outs = ret if isinstance(ret, list) else [ret]
    _run.last_build = t1 - t0
    _run.last_gen = t2 - t1
    _run.last_tokens = sum((o.get("meta_info", {}) or {}).get("completion_tokens", 0)
                           for o in outs)
    return outs


if __name__ == "__main__":
    raise SystemExit(main())
