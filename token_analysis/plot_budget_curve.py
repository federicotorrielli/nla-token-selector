"""What an explanation budget buys: audit success against the number of positions read.

An auditor reads a fixed number of explanations for one transcript. The question
this figure answers is how much of what exhaustive verbalization finds survives
that budget. The outcome is the audit succeeding, not a downstream task score:
there is no downstream task here, only one judge label per position.

Two bars for success. The top row asks for at least one on-task explanation among
the positions read, which is the auditor seeing any evidence at all. The bottom row
asks for at least three, because \\citet{fraser-taliente_natural_2026} report that a
claim recurring across adjacent positions holds more often than a claim appearing
once. Each curve is divided by the same quantity under exhaustive verbalization, so
1.0 means the budget loses nothing.

Four rankers. `structure` and the selected dataset-shared ensemble are the two real
selectors. `random` is the floor, computed exactly from the hypergeometric
distribution rather than sampled. `oracle` is the bound: it succeeds whenever the
transcript contains enough on-task positions to succeed at all.

Models are averaged with equal weight inside a dataset, the convention the
dataset-shared selector uses.

Usage:
    uv run --with matplotlib,scikit-learn python token_analysis/plot_budget_curve.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import hypergeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from token_analysis.budget_eval import (  # noqa: E402
    ensemble_score,
    position_baseline,
    segment_frame,
)
from token_analysis.common import (  # noqa: E402
    AUROC_ENSEMBLE_SHARED_SELECTION_PATH,
    DEFAULT_INPUT_DIR,
    load_auroc_ensemble_specs,
    source_path,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"

DATASETS = [
    ("opi", "OpenPromptInjection", ["q7", "g12", "g27", "l70"]),
    ("taboo", "Taboo organisms", ["q7", "g12", "g27", "l70"]),
    ("liars", "Liars' Bench", ["g27", "l70"]),
    ("tt", "Tensor Trust", ["q7", "g12", "g27", "l70"]),
]
# Stops at 32: beyond it the budget approaches the length of a short transcript,
# so the curves converge for a reason that has nothing to do with the ranker.
BUDGETS = np.array([1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 21, 26, 32])
THRESHOLDS = (1, 3)
FRACTION = 0.05          # the budget the headline claim quotes

# Two selectors carry colour; the bound and the floor are neutral and carry a
# line style instead, so identity never rests on colour alone.
STYLE = {
    "structure": dict(color="#1f6fb4", ls="-", marker="o", ms=2.6, lw=1.4, zorder=5),
    "ensemble": dict(color="#c94a2b", ls="-", marker="s", ms=2.4, lw=1.4, zorder=4),
    "oracle": dict(color="#5f5f5f", ls="--", lw=1.0, zorder=3),
    "random": dict(color="#8a8a8a", ls=":", lw=1.0, zorder=2),
}
LABEL = {
    "structure": "structure",
    "ensemble": "selected ensemble",
    "oracle": "oracle",
    "random": "random",
}


def case_arrays(df: pl.DataFrame, scores: dict[str, np.ndarray]) -> list[dict]:
    """One record per transcript: its labels and each ranker's ordering."""
    case_ids = df["case_id"].to_numpy()
    y = df["on_task"].to_numpy().astype(int)
    order = np.argsort(case_ids, kind="stable")
    sorted_ids = case_ids[order]
    bounds = np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1
    out = []
    for idx in np.split(order, bounds):
        rec = {"y": y[idx], "n": len(idx), "m": int(y[idx].sum())}
        for name, s in scores.items():
            v = s[idx].astype(float)
            # A missing score ranks last; it is never imputed.
            rec[name] = np.argsort(np.where(np.isfinite(v), -v, np.inf), kind="stable")
        out.append(rec)
    return out


def hit_rates(cases: list[dict], threshold: int) -> dict[str, np.ndarray]:
    """Share of transcripts reaching `threshold` on-task explanations at each
    budget, divided by the share reaching it under exhaustive verbalization."""
    eligible = np.array([c["m"] >= threshold for c in cases], dtype=float)
    ceiling = eligible.mean()
    curves = {name: np.zeros(len(BUDGETS)) for name in STYLE}
    for c in cases:
        n, m = c["n"], c["m"]
        for j, b in enumerate(BUDGETS):
            k = min(int(b), n)
            for name in ("structure", "ensemble"):
                curves[name][j] += c["y"][c[name][:k]].sum() >= threshold
            # exact probability of at least `threshold` successes in k draws
            curves["random"][j] += hypergeom.sf(threshold - 1, n, m, k)
            curves["oracle"][j] += (m >= threshold) and (k >= threshold)
    for name in curves:
        curves[name] = curves[name] / len(cases) / max(ceiling, 1e-12)
    return curves | {"__ceiling__": ceiling}


def fraction_budget(cases: list[dict], threshold: int, frac: float) -> dict[str, float]:
    """The same statistic at a per-transcript budget of `frac` of its positions."""
    eligible = np.mean([c["m"] >= threshold for c in cases])
    out = dict.fromkeys(STYLE, 0.0)
    for c in cases:
        k = max(1, int(np.ceil(frac * c["n"])))
        for name in ("structure", "ensemble"):
            out[name] += c["y"][c[name][:k]].sum() >= threshold
        out["random"] += hypergeom.sf(threshold - 1, c["n"], c["m"], k)
        out["oracle"] += (c["m"] >= threshold) and (k >= threshold)
    return {k: v / len(cases) / max(eligible, 1e-12) for k, v in out.items()}


def collect(input_dir: Path) -> tuple[dict, dict, dict]:
    specs = load_auroc_ensemble_specs(AUROC_ENSEMBLE_SHARED_SELECTION_PATH)
    curves: dict = {}
    at_frac: dict = {}
    meta: dict = {}
    for key, _, models in DATASETS:
        per_model = {t: [] for t in THRESHOLDS}
        per_model_frac = {t: [] for t in THRESHOLDS}
        lengths, ceilings = [], {t: [] for t in THRESHOLDS}
        for model in models:
            df = pl.read_parquet(source_path(input_dir, key, model))
            df = segment_frame(df, key).filter(pl.col("segment") != "trailer")
            y = df["on_task"].to_numpy().astype(int)
            scores = {
                "structure": position_baseline(df, y, "structure"),
                "ensemble": ensemble_score(df, specs[(key, model)]),
            }
            cases = case_arrays(df, scores)
            lengths.append(np.median([c["n"] for c in cases]))
            for t in THRESHOLDS:
                r = hit_rates(cases, t)
                ceilings[t].append(r.pop("__ceiling__"))
                per_model[t].append(r)
                per_model_frac[t].append(fraction_budget(cases, t, FRACTION))
        curves[key] = {
            t: {n: np.mean([r[n] for r in per_model[t]], axis=0) for n in STYLE}
            for t in THRESHOLDS
        }
        at_frac[key] = {
            t: {n: float(np.mean([r[n] for r in per_model_frac[t]])) for n in STYLE}
            for t in THRESHOLDS
        }
        meta[key] = {"median_length": float(np.mean(lengths)),
                     "ceiling": {t: float(np.mean(ceilings[t])) for t in THRESHOLDS}}
    return curves, at_frac, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    ap.add_argument("--out", type=Path, default=OUT / "budget_curve")
    args = ap.parse_args(argv)

    curves, at_frac, meta = collect(args.input_dir)

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })
    # 5.5 in is the ICLR text width, so the figure is included unscaled.
    fig, axes = plt.subplots(2, 4, figsize=(5.5, 3.2), sharex=True, sharey=True)
    for row, t in enumerate(THRESHOLDS):
        for col, (key, title, _) in enumerate(DATASETS):
            ax = axes[row, col]
            ax.axhline(1.0, color="0.85", lw=0.6, zorder=1)
            for name, style in STYLE.items():
                ax.plot(BUDGETS, curves[key][t][name], label=LABEL[name], **style)
            ax.set_xscale("log")
            ax.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
            ax.set_ylim(-0.03, 1.09)
            ax.set_yticks([0, 0.5, 1.0])
            ax.grid(axis="y", color="0.92", lw=0.5, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(title, fontsize=7.5, pad=4)
            if col == 0:
                ax.set_ylabel(f"at least {'one' if t == 1 else 'three'}\non-task", fontsize=7.5)
            if row == 1:
                ax.set_xlabel("explanations read", fontsize=7.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")

    print(f"budget of {FRACTION:.0%} of a transcript, share of exhaustive audit success\n")
    head = f"{'dataset':22s}{'med len':>8s}{'ceiling':>9s}" + "".join(
        f"{LABEL[n]:>19s}" for n in STYLE)
    for t in THRESHOLDS:
        print(f"  at least {t} on-task")
        print("  " + head)
        for key, title, _ in DATASETS:
            row = at_frac[key][t]
            print(f"  {title:22s}{meta[key]['median_length']:>8.0f}"
                  f"{meta[key]['ceiling'][t]:>9.3f}"
                  + "".join(f"{row[n]:>19.3f}" for n in STYLE))
        print()
    print(f"wrote {args.out}.pdf and {args.out}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
