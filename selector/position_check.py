"""selector/position_check.py — does a signal just like late tokens?

Two position confounds threaten the all-token tables, and both are cheap to
look at once the consolidated files exist.

**Gemma's window.** Gemma-3 interleaves local and global attention, and the
local layers carry a 1024-token window. Past that point they cannot attend to
token 0 at all, so they contribute nothing to the sink mass, the average over
layers falls, and `sink_drain` rises. That is a bias that grows with position,
in the same direction for every long transcript, on g12 and g27 but not on q7 or
l70. It shows up as a much steeper drift of mean `sink_drain` across position
bins for the Gemma pair, with a step near 1024.

**On-task rate by position.** If the NLA lands on-task more often late in a
transcript, then any signal correlated with position collects AUROC for free.
Reported alongside so the two can be told apart.

    python selector/position_check.py
    python selector/position_check.py --signal head_disagreement
    python selector/position_check.py --selftest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

BR = Path("results/bridge")
BINS = [(0, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 10**9)]
GEMMA = {"g12", "g27"}


def drift(df: pl.DataFrame, col: str) -> tuple[list[float], float]:
    """Mean of `col` per position bin, and the slope of its z-score against
    position (in units of standard deviation per 1000 tokens)."""
    means = []
    for lo, hi in BINS:
        s = df.filter((pl.col("tok_idx") >= lo) & (pl.col("tok_idx") < hi))[col]
        s = s.drop_nulls().drop_nans() if s.len() else s
        means.append(float(s.mean()) if s.len() else float("nan"))
    x = df["tok_idx"].to_numpy().astype(float)
    y = df[col].to_numpy().astype(float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 or np.std(y[ok]) < 1e-12:
        return means, float("nan")
    z = (y[ok] - y[ok].mean()) / y[ok].std()
    return means, float(np.polyfit(x[ok], z, 1)[0] * 1000)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--signal", default="sink_drain")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()

    hdr = " ".join(f"{f'{lo}-{hi}' if hi < 10**9 else f'{lo}+':>10}" for lo, hi in BINS)
    print(f"\n{args.signal} by position (watch g12/g27 for a step near 1024)\n")
    print(f"{'kind/model':16}{'slope/1k':>10}  {hdr}")
    for p in sorted(BR.glob("all_*.parquet")):
        kind, model = p.stem[len("all_"):].rsplit("_", 1)
        df = pl.read_parquet(p)
        if args.signal not in df.columns or "tok_idx" not in df.columns:
            continue
        means, slope = drift(df, args.signal)
        flag = " <- gemma" if model in GEMMA else ""
        cells = " ".join(f"{m:10.4f}" if np.isfinite(m) else f"{'-':>10}" for m in means)
        print(f"{kind + '/' + model:16}{slope:10.3f}  {cells}{flag}")

    print(f"\non-task rate by position\n\n{'kind/model':16}{'':10}  {hdr}")
    for p in sorted(BR.glob("all_*.parquet")):
        kind, model = p.stem[len("all_"):].rsplit("_", 1)
        df = pl.read_parquet(p)
        if "on_task" not in df.columns:
            continue
        df = df.filter(pl.col("on_task").is_not_null())
        if df.is_empty():
            continue
        means, _ = drift(df, "on_task")
        cells = " ".join(f"{m:10.3f}" if np.isfinite(m) else f"{'-':>10}" for m in means)
        print(f"{kind + '/' + model:16}{'':10}  {cells}")


def _selftest() -> None:
    n = 3000
    idx = np.arange(n)
    # a signal that rises with position must show a positive slope, and one
    # that does not must show a slope near zero
    rising = pl.DataFrame({"tok_idx": idx, "s": idx * 0.001})
    flat = pl.DataFrame({"tok_idx": idx, "s": np.sin(idx * 0.5)})
    _, s_rise = drift(rising, "s")
    _, s_flat = drift(flat, "s")
    assert s_rise > 1.0, f"rising slope {s_rise}"
    assert abs(s_flat) < 0.1, f"flat slope {s_flat}"
    means, _ = drift(rising, "s")
    assert means[0] < means[-2], "bin means must increase with position"
    # a constant signal has no slope to report rather than a spurious one
    const = pl.DataFrame({"tok_idx": idx, "s": np.ones(n)})
    assert np.isnan(drift(const, "s")[1]), "constant signal has no slope"
    print("selftest ok")


if __name__ == "__main__":
    main()
