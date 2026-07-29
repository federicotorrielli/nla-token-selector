"""Emit tidy, reviewer-explorable per-token CSVs for every bridge experiment.
One row per token: identity + label + on_task + every cheap signal. No activations.

Default: the original probed-subset outputs -> paper_results/bridge/{kind}_{s}.csv.
`--all`: the all-token outputs (shard dirs) -> paper_results/bridge/all_{kind}_{s}.csv.gz
(gzipped: the all-token tables run to millions of rows)."""
import gzip
import io
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

SIG = {  # experiment -> (shorts, signals-source template, join key, signal cols)
    "hand": (FOUR, "TOKENS", ["case_id", "mode", "tok_idx"], HAND_SIG),
    "opi":  (FOUR, "opi_{s}_signals.parquet", ["position_id"], [*BLIND7, "attn_rollout"]),
    "liars": (["l70", "g27"], "liars_{s}_corpus.parquet", ["position_id"], BLIND7),
    "tt":   (FOUR, "tt_{s}_corpus.parquet", ["position_id"], BLIND7),
    "taboo": (FOUR, "taboo_{s}_corpus.parquet", ["position_id"], BLIND7),
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
    for kind, (shorts, _tmpl, _key, _cols) in SIG.items():
        cols = ALL_SIG + (["attn_rollout"] if kind == "opi" else [])
        for s in shorts:
            ot_dir = BR / f"all_{kind}_{s}_ontask"
            corp_dir = BR / f"all_{kind}_{s}_corpus"
            if not ot_dir.is_dir() or not corp_dir.is_dir():
                continue
            ot = (pl.scan_parquet(str(ot_dir / "*.parquet")).collect()
                  .with_columns(pl.col("position_id").cast(pl.Int64)))
            right = (pl.scan_parquet(str(corp_dir / "*.parquet"))
                     .select(["position_id", "region", "probe_tok_idx", *cols])
                     .collect()
                     .with_columns(pl.col("position_id").cast(pl.Int64)))
            df = ot.join(right, on="position_id", how="left").sort("position_id")
            out = OUT / f"all_{kind}_{s}.csv.gz"
            buf = io.BytesIO()
            df.write_csv(buf)
            with gzip.open(out, "wb") as f:
                f.write(buf.getvalue())
            print(f"{out}  ({df.height} rows, {df.width} cols)")


if __name__ == "__main__":
    all_token_csvs() if "--all" in sys.argv else subset_csvs()
