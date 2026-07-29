"""Emit tidy, reviewer-explorable per-token CSVs for every bridge experiment.
One row per token: identity + label + on_task + every cheap signal. No activations.

Default: the original probed-subset outputs -> paper_results/bridge/{kind}_{s}.csv.
`--all`: the all-token tables -> paper_results/bridge/all_{kind}_{s}.parquet
(parquet: these run to millions of rows and do not fit git as CSV)."""
import sys
from pathlib import Path

import polars as pl

BR = Path("results/bridge")
OUT = Path("paper_results/bridge")
OUT.mkdir(parents=True, exist_ok=True)

TAG = {"q7": "Qwen2.5-7B-Instruct", "g12": "gemma-3-12b-it",
       "g27": "gemma-3-27b-it", "l70": "Llama-3.3-70B-Instruct"}
HAND_SIG = ["surprisal", "entropy", "varentropy", "resid_jump", "temporal_kl",
            "lookback_ratio", "sink_drain", "head_disagreement", "w", "kl", "attn_rollout"]
BLIND7 = ["surprisal", "entropy", "varentropy", "resid_jump",
          "lookback_ratio", "sink_drain", "head_disagreement"]
ALL_SIG = ["surprisal", "entropy", "varentropy", "temporal_kl", "resid_jump",
           "lookback_ratio", "sink_drain", "head_disagreement"]
FOUR = ["q7", "g12", "g27", "l70"]

# Only the hand pilot keeps a subset export: it is the one experiment carrying
# the referenced signals (kl, w, attn_rollout), which the all-token run does not
# compute. Every other benchmark is fully covered by all_{kind}_{model}.parquet.
SIG = {  # experiment -> (shorts, signals-source template, join key, signal cols)
    "hand": (FOUR, "TOKENS", ["case_id", "mode", "tok_idx"], HAND_SIG),
}


def subset_csvs() -> None:
    for kind, (shorts, tmpl, key, cols) in SIG.items():
        for s in shorts:
            pre = "" if kind == "hand" else f"{kind}_"
            ot = BR / f"{pre}{s}_ontask.parquet"
            if not ot.exists():
                continue
            df = pl.read_parquet(ot)
            src = (Path(f"results/token_selector_v2/{TAG[s]}/tokens.parquet")
                   if tmpl == "TOKENS" else BR / tmpl.format(s=s) if tmpl else None)
            if src and src.exists():
                have = [c for c in cols if c in pl.scan_parquet(src).columns]
                right = pl.scan_parquet(src).select([*key, *have]).collect()
                df = df.join(right, on=key, how="left")
            out = OUT / f"{kind}_{s}.csv"
            df.write_csv(out)
            print(f"{out}  ({df.height} rows, {df.width} cols)")


def all_token_csvs() -> None:
    """The consolidated tables minus the explanation text."""
    for p in sorted(BR.glob("all_*.parquet")):
        df = pl.read_parquet(p).drop("explanation", strict=False)
        out = OUT / p.name
        df.write_parquet(out, compression="zstd", compression_level=9)
        print(f"{out}  ({df.height} rows, {df.width} cols, "
              f"{out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    all_token_csvs() if "--all" in sys.argv else subset_csvs()
