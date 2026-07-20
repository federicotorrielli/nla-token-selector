"""scripts/token_selector/bridge_run_nla.py — bridge test, phase 2.

Run the NLA (the AV decoder) at every token's activation from the phase-1 corpus,
saving one explanation per token. Uses the project's NLAClientLite, pointed at
our custom corpus and at a running AV SGLang server. Writes incrementally and
resumes, so a killed run continues.

Needs an SGLang server serving the AV checkpoint (e.g. kitft/nla-qwen2.5-7b-L20-av)
at --sglang-url. Run on the B200 with the `pao` env.

Usage:
    python scripts/token_selector/bridge_run_nla.py \
        --corpus results/bridge/q7_corpus.parquet \
        --sglang-url http://localhost:30000 \
        --out results/bridge/q7_explanations.parquet
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sglang-url", default="http://localhost:30000")
    ap.add_argument("--model-short", default="q7")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--limit", type=int, default=None, help="only first N positions (smoke test)")
    args = ap.parse_args(argv)

    # Configure the Settings tree before importing the package: point the corpus
    # and the AV server at ours. Settings() defaults to the default overlay.
    os.environ["CNLA_EXPERIMENT__MODEL_SHORT"] = args.model_short
    os.environ["CNLA_PATHS__CORPUS_PARQUET"] = str(Path(args.corpus).resolve())
    os.environ[f"CNLA_MODELS__{args.model_short.upper()}__SGLANG_URL"] = args.sglang_url

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
            pl.DataFrame(rows).write_parquet(outp)  # incremental checkpoint
            print(f"  {min(i + args.batch, len(todo))}/{len(todo)}", flush=True)

    print(f"wrote {outp} ({len(rows)} explanations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
