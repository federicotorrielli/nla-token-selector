"""Does pooling the picks of two selectors beat either one alone?

Every number in the bridge reports is an AUROC, which scores a *ranking*. A pool
is a *set*, so it has no ROC curve. The deployment question is budget-shaped
anyway ("I can afford B explanations on this transcript"), so we score at a
budget instead: give each method the same B tokens per transcript and count how
many of them the judge called on-task (precision@B).

Four methods over the same budget:

  A       top B by |z(signal_a)| within the transcript
  B       top B by |z(signal_b)|
  pool    interleave the two rankings, skipping duplicates, until B are picked
          (equivalently: rank by min(rank_a, rank_b)) -- the literal
          "pool of the best tokens from both methods"
  rrf     reciprocal rank fusion, 1/(k+rank_a) + 1/(k+rank_b), k=60
          (Cormack, Clarke & Buettcher 2009) -- score-level fusion for contrast

Scores are z-scored *within a transcript*, since that is the unit an auditor
holds, and taken two-tailed by default: `sink_drain` points forward on injection
and backward on secret words, so a method that has to be told the sign is not a
general method. `--tail oracle` picks the sign per task from the data, which is
the ceiling the AUROC tables implicitly assume.

    python selector/pool_selector.py --frac 0.10
    python selector/pool_selector.py --signals entropy head_disagreement
    python selector/pool_selector.py --selftest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
CSVS = ROOT / "paper_results" / "bridge"
RRF_K = 60


def _rank(df: pl.DataFrame, sig: str, tail: str) -> pl.DataFrame:
    """Per-case z-score of `sig`, then a dense rank (1 = picked first)."""
    z = (pl.col(sig) - pl.col(sig).mean().over("case_id")) / (
        pl.col(sig).std().over("case_id") + 1e-12
    )
    score = z.abs() if tail == "two" else z * pl.lit(df["_sign_" + sig][0])
    # a token with no signal (position 0 has no previous state) ranks last
    score = score.fill_nan(None).fill_null(float("-inf"))
    return df.with_columns(
        score.rank("ordinal", descending=True).over("case_id").alias(f"r_{sig}")
    )


def _oracle_sign(df: pl.DataFrame, sig: str) -> float:
    """+1 if on-task tokens score higher than the rest, -1 otherwise."""
    hit = df.filter(pl.col("on_task") == 1)[sig].mean()
    miss = df.filter(pl.col("on_task") == 0)[sig].mean()
    if hit is None or miss is None:
        return 1.0
    return 1.0 if hit >= miss else -1.0


def select(df: pl.DataFrame, sig_a: str, sig_b: str, frac: float, tail: str) -> pl.DataFrame:
    """Flag the tokens each method spends its budget on."""
    for sig in (sig_a, sig_b):
        df = df.with_columns(pl.lit(_oracle_sign(df, sig)).alias("_sign_" + sig))
        df = _rank(df, sig, tail)
    ra, rb = pl.col(f"r_{sig_a}"), pl.col(f"r_{sig_b}")
    budget = (pl.len().over("case_id") * frac).ceil().clip(1)
    rrf = 1.0 / (RRF_K + ra) + 1.0 / (RRF_K + rb)
    return df.with_columns(
        sel_a=ra <= budget,
        sel_b=rb <= budget,
        # min(rank) interleaves the two lists and drops duplicates for free
        sel_pool=pl.min_horizontal(ra, rb).rank("ordinal").over("case_id") <= budget,
        sel_rrf=rrf.rank("ordinal", descending=True).over("case_id") <= budget,
        # max(rank) is the intersection as a ranking: a token is bought early
        # only if BOTH methods rank it high, so it spends the same budget B on
        # the agreement region first
        sel_and=pl.max_horizontal(ra, rb).rank("ordinal").over("case_id") <= budget,
    )


METHODS = ["sel_a", "sel_b", "sel_pool", "sel_rrf", "sel_and"]


def _per_case(df: pl.DataFrame) -> pl.DataFrame:
    """Case-level hit/spend counts -- the unit the bootstrap resamples."""
    agg = [pl.col("on_task").sum().alias("hits"), pl.len().alias("n")]
    for m in METHODS:
        agg += [
            (pl.col("on_task") * pl.col(m).cast(pl.Int8)).sum().alias(f"h_{m}"),
            pl.col(m).sum().alias(f"s_{m}"),
        ]
    return df.group_by("case_id").agg(agg)


def _precision(cases: pl.DataFrame, m: str) -> float:
    spend = cases[f"s_{m}"].sum()
    return float(cases[f"h_{m}"].sum() / spend) if spend else float("nan")


def _boot_delta(
    cases: pl.DataFrame, method: str, best: str, n_boot: int, seed: int
) -> tuple[float, float]:
    """Clustered bootstrap CI on precision(method) - precision(best single)."""
    rng = np.random.default_rng(seed)
    hp, sp = cases[f"h_{method}"].to_numpy(), cases[f"s_{method}"].to_numpy()
    hb, sb = cases[f"h_{best}"].to_numpy(), cases[f"s_{best}"].to_numpy()
    idx = rng.integers(0, len(hp), (n_boot, len(hp)))
    d = hp[idx].sum(1) / np.maximum(sp[idx].sum(1), 1) - hb[idx].sum(1) / np.maximum(
        sb[idx].sum(1), 1
    )
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _overlap(df: pl.DataFrame) -> float:
    """Jaccard of the two methods' picks: how much they duplicate each other."""
    a, b = df["sel_a"].to_numpy(), df["sel_b"].to_numpy()
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else float("nan")


def _regions(df: pl.DataFrame) -> dict[str, float]:
    """Precision where the methods agree vs where only one of them picked.

    This is what says whether pooling *can* work. A union pays for both
    exclusive regions, so it only beats the best single method when those
    regions are as on-task as the agreement region. If agreement is far more
    precise, the right combiner is an intersection, not a union.
    """
    a, b = df["sel_a"].to_numpy(), df["sel_b"].to_numpy()
    y = df["on_task"].to_numpy()
    out = {}
    for name, mask in [("both", a & b), ("a_only", a & ~b), ("b_only", b & ~a)]:
        out[name] = float(y[mask].mean()) if mask.sum() else float("nan")
        out["n_" + name] = int(mask.sum())
    return out


def run(sig_a: str, sig_b: str, frac: float, tail: str, n_boot: int, seed: int) -> str:
    rows = []
    for csv in sorted(CSVS.glob("*.csv")):
        task, model = csv.stem.rsplit("_", 1)
        df = pl.read_csv(csv)
        if sig_a not in df.columns or sig_b not in df.columns:
            continue
        for mode in sorted(df["mode"].unique()) if task == "hand" else [None]:
            sub = df.filter(pl.col("mode") == mode) if mode else df
            sub = select(sub, sig_a, sig_b, frac, tail)
            cases = _per_case(sub)
            p = {m: _precision(cases, m) for m in METHODS}
            best = "sel_a" if p["sel_a"] >= p["sel_b"] else "sel_b"
            lo, hi = _boot_delta(cases, "sel_pool", best, n_boot, seed)
            alo, ahi = _boot_delta(cases, "sel_and", best, n_boot, seed)
            rows.append(
                dict(
                    task=mode or task,
                    model=model,
                    base=float(sub["on_task"].mean()),
                    a=p["sel_a"],
                    b=p["sel_b"],
                    pool=p["sel_pool"],
                    rrf=p["sel_rrf"],
                    andd=p["sel_and"],
                    delta=p["sel_pool"] - p[best],
                    lo=lo,
                    hi=hi,
                    adelta=p["sel_and"] - p[best],
                    alo=alo,
                    ahi=ahi,
                    jac=_overlap(sub),
                    **_regions(sub),
                )
            )

    out = [
        f"# Pooling `{sig_a}` + `{sig_b}` at a {frac:.0%} budget ({tail}-tailed)\n",
        "precision@B = of the B explanations bought, the fraction the judge called",
        "on-task, best per row in bold. Every method spends the same B. `pool`",
        "interleaves the two rankings (the union), `rrf` fuses their scores, `and`",
        "ranks by the worse of the two ranks (the intersection). Each delta is that",
        "combiner minus the better single signal, with a case-clustered 95%",
        "interval; it has to clear 0 for the combiner to be worth anything.",
        "`jaccard` is how much the two signals pick the same tokens (1.0 =",
        "identical).\n",
        f"| task | model | base rate | {sig_a} | {sig_b} | pool | rrf | and | "
        "pool delta | and delta | jaccard |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        win = " ✓" if r["lo"] > 0 else ""
        awin = " ✓" if r["alo"] > 0 else ""
        cols = ("a", "b", "pool", "rrf", "andd")
        top = max(r[k] for k in cols)
        c = {k: (f"**{r[k]:.3f}**" if r[k] == top else f"{r[k]:.3f}") for k in cols}
        out.append(
            f"| {r['task']} | {r['model']} | {r['base']:.2f} | {c['a']} | "
            f"{c['b']} | {c['pool']} | {c['rrf']} | {c['andd']} | "
            f"{r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]{win} | "
            f"{r['adelta']:+.3f} [{r['alo']:+.3f}, {r['ahi']:+.3f}]{awin} | {r['jac']:.2f} |"
        )

    out += [
        "\n## Where the two methods agree, and where they do not\n",
        "On-task rate among the tokens both methods bought, and among the ones only",
        "one of them bought. A union pays for both exclusive columns, so it can only",
        "pay off when those match the agreement column.\n",
        f"| task | model | base rate | both | {sig_a} only | {sig_b} only |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r['task']} | {r['model']} | {r['base']:.2f} | "
            f"**{r['both']:.3f}** (n={r['n_both']}) | {r['a_only']:.3f} "
            f"(n={r['n_a_only']}) | {r['b_only']:.3f} (n={r['n_b_only']}) |"
        )
    return "\n".join(out)


def _selftest() -> None:
    # Two complementary selectors on a 10-token case, budget 2. Each spends one
    # pick on a different on-task token and one on a miss (precision 0.5), so
    # pooling has to catch both and reach 1.0. Signals are two-tailed, hence the
    # -9 decoys: an extreme low ranks as high as an extreme high.
    n = 10
    df = pl.DataFrame(
        {
            "case_id": ["c"] * n,
            "on_task": [1, 1] + [0] * (n - 2),
            "entropy": [9.0, 1.0, -6.0] + [1.0] * (n - 3),
            "sink_drain": [1.0, 9.0, 1.0, -6.0] + [1.0] * (n - 4),
        }
    )
    sel = select(df, "entropy", "sink_drain", frac=0.2, tail="two")
    cases = _per_case(sel)
    assert _precision(cases, "sel_a") == 0.5, "one-signal precision"
    assert _precision(cases, "sel_b") == 0.5, "one-signal precision"
    assert _precision(cases, "sel_pool") == 1.0, "pool must catch both"
    assert sel["sel_pool"].sum() == 2, "pool must respect the budget"
    assert _overlap(sel) == 0.0, "disjoint picks"

    # identical selectors: pooling is a no-op, and must not overspend
    df2 = df.with_columns(sink_drain=pl.col("entropy"))
    sel2 = select(df2, "entropy", "sink_drain", frac=0.2, tail="two")
    assert _precision(_per_case(sel2), "sel_pool") == 0.5, "no free lunch"
    assert _overlap(sel2) == 1.0, "identical picks"
    print("selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--signals", nargs=2, default=["entropy", "sink_drain"])
    ap.add_argument("--frac", type=float, default=0.10, help="budget per transcript")
    ap.add_argument("--tail", choices=["two", "oracle"], default="two")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    md = run(*args.signals, args.frac, args.tail, args.n_boot, args.seed)
    if args.out:
        args.out.write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    sys.exit(main())
