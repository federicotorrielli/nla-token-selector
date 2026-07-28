"""selector/consolidate_all.py — one canonical per-token table per benchmark x model.

The all-token pipeline works in sharded directories (corpus / explanations /
ontask) because the corpora hold an activation per token. Those directories are
pipeline internals. This script joins everything EXCEPT the activations into a
single tidy parquet per (kind, model):

    results/bridge/all_{kind}_{model}.parquet
      position_id, case_id, mode, tok_idx, token, label, region, probe_tok_idx,
      the blind signals (+ attn_rollout for opi), explanation, on_task

That file is the reusable artifact for further experiments (different judges,
different selectors, region slices) without touching the shard dirs again.
Idempotent; re-run any time — it rebuilds from whatever shards exist and prints
coverage so partial pipelines are visible, not silent.

Usage:
    python selector/consolidate_all.py            # everything found
    python selector/consolidate_all.py tt q7      # one kind, one model
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

BR = Path("results/bridge")
KINDS = {"hand": ["q7", "g12", "g27", "l70"], "opi": ["q7", "g12", "g27", "l70"],
         "tt": ["q7", "g12", "g27", "l70"], "taboo": ["q7", "g12", "g27", "l70"],
         "liars": ["g27", "l70"]}
# written by spike_stats.py from the stored activations; absent until it runs
SPIKE_COLS = ["spike_mass", "peak_ratio", "act_norm", "resid_jump_masked"]


def _scan_dir(d: Path):
    shards = sorted(d.glob("*.parquet")) if d.is_dir() else []
    return pl.scan_parquet([str(f) for f in shards]) if shards else None


def consolidate(kind: str, short: str) -> None:
    corp = _scan_dir(BR / f"all_{kind}_{short}_corpus")
    if corp is None:
        return
    meta_cols = [c for c in corp.collect_schema().names() if c != "activation"]
    df = (corp.select(meta_cols).collect()
          .with_columns(pl.col("position_id").cast(pl.Int64)))
    for src, cols in [(f"all_{kind}_{short}_explanations", ["explanation"]),
                      (f"all_{kind}_{short}_ontask", ["on_task"]),
                      (f"all_{kind}_{short}_spike", SPIKE_COLS)]:
        lf = _scan_dir(BR / src)
        if lf is None:
            continue
        have = [c for c in cols if c in lf.collect_schema().names()]
        right = (lf.select(["position_id", *have]).collect()
                 .with_columns(pl.col("position_id").cast(pl.Int64))
                 .unique(subset=["position_id"], keep="first"))
        df = df.join(right, on="position_id", how="left")
    df = df.sort("position_id")
    out = BR / f"all_{kind}_{short}.parquet"
    tmp = out.with_suffix(".tmp.parquet")
    df.write_parquet(tmp)
    tmp.replace(out)
    n = df.height
    ne = df["explanation"].is_not_null().sum() if "explanation" in df.columns else 0
    nj = df["on_task"].is_not_null().sum() if "on_task" in df.columns else 0
    print(f"{out}  {n} tokens, explained {ne / max(n, 1):.0%}, "
          f"judged {nj / max(n, 1):.0%}")


if __name__ == "__main__":
    want_kind = sys.argv[1] if len(sys.argv) > 1 else None
    want_short = sys.argv[2] if len(sys.argv) > 2 else None
    for kind, shorts in KINDS.items():
        if want_kind and kind != want_kind:
            continue
        for s in shorts:
            if want_short and s != want_short:
                continue
            consolidate(kind, s)
