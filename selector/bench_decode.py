"""selector/bench_decode.py — find the AV decode throughput ceiling.

The bridge decode is the whole cost of the all-token run (millions of ~500-token
explanations), and it was inherited at `--batch 32`, one synchronous request at
a time. This measures positions/second across batch sizes and concurrent
in-flight requests against an ALREADY RUNNING AV server, so the orchestration
can be set from a measurement instead of a guess.

Nothing is written; it only reads activations from a corpus and decodes.

Usage:
    python selector/bench_decode.py --corpus results/bridge/all_tt_q7_corpus \
        --model-short q7 --sglang-url http://127.0.0.1:30000 \
        --batches 32 128 256 --concurrency 1 2 4
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="corpus parquet or shard dir")
    ap.add_argument("--model-short", default="q7")
    ap.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    ap.add_argument("--batches", type=int, nargs="+", default=[32, 128, 256])
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--reps", type=int, default=1, help="batches per (b, c) cell")
    args = ap.parse_args(argv)

    corpus = Path(args.corpus)
    first = (sorted(corpus.glob("shard-*.parquet"))[0] if corpus.is_dir() else corpus)
    os.environ["CNLA_EXPERIMENT__MODEL_SHORT"] = args.model_short
    os.environ["CNLA_PATHS__CORPUS_PARQUET"] = str(first.resolve())
    os.environ[f"CNLA_MODELS__{args.model_short.upper()}__SGLANG_URL"] = args.sglang_url

    import polars as pl

    from nla_token_selector.client import NLAClientLite
    from nla_token_selector.settings import Settings

    pids = [int(p) for p in
            pl.read_parquet(first, columns=["position_id"])["position_id"].to_list()]
    print(f"corpus {first.name}: {len(pids)} positions available\n")
    print(f"{'batch':>6} {'conc':>5} {'in flight':>10} {'seconds':>9} "
          f"{'pos/s':>8} {'gen tok/s':>10}")

    client = NLAClientLite(Settings())
    cursor = 0
    results = []
    try:
        for b in args.batches:
            for c in args.concurrency:
                need = b * c * args.reps
                if cursor + need > len(pids):
                    cursor = 0
                chunks = [pids[cursor + i * b: cursor + (i + 1) * b]
                          for i in range(c * args.reps)]
                cursor += need
                t0 = time.time()
                if c == 1:
                    outs = [client.generate_batch(ch, temperature=0.0,
                                                  max_new_tokens=args.max_new_tokens)
                            for ch in chunks]
                else:
                    with ThreadPoolExecutor(max_workers=c) as pool:
                        outs = list(pool.map(
                            lambda ch: client.generate_batch(
                                ch, temperature=0.0,
                                max_new_tokens=args.max_new_tokens), chunks))
                dt = time.time() - t0
                n = sum(len(o) for o in outs)
                # rough generated-token count from the returned text
                chars = sum(len(t) for o in outs for t in o)
                gen_tok = chars / 4.0
                print(f"{b:>6} {c:>5} {b * c:>10} {dt:>9.1f} {n / dt:>8.2f} "
                      f"{gen_tok / dt:>10.0f}")
                results.append((b, c, n / dt))
    finally:
        client.close()

    best = max(results, key=lambda r: r[2])
    base = next((r[2] for r in results if r[0] == 32 and r[1] == 1), None)
    print(f"\nbest: batch={best[0]} concurrency={best[1]} -> {best[2]:.2f} pos/s")
    if base:
        print(f"speedup over batch=32 serial ({base:.2f} pos/s): {best[2] / base:.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
