"""token_analysis/signal_eval.py — what each individual signal predicts.

The ensemble analysis reports the winning mixture. This script reports the
thirteen source signals themselves, which is what the paper's signal table
needs, and runs the three controls that decide how much of a signal's AUROC is
the threat and how much is the shape of a chat transcript.

Five questions, one per output table:

  auroc      per dataset, model and signal: pooled AUROC against the judge
             label, with a whole-transcript bootstrap interval, the
             within-transcript (case-macro) AUROC, and the permutation test
             under Benjamini-Hochberg control.
  direction  does a signal keep one direction across the models of a dataset,
             and across datasets? A signal whose sign has to be refitted per
             setting cannot be recommended on its own.
  confound   how much of a signal is sequence position? Reports the rank
             correlation with normalized position and with transcript length,
             and the AUROC recomputed inside strata of segment, chat role and
             position bin, which never compares two positions of different
             structural kind.
  baseline   the winning signal against the free rankers of budget_eval.py,
             which need no forward pass.
  redundancy oriented rank correlation between signals, within and across the
             three families. Near-orthogonal components are what a pairwise
             ensemble can exploit.

Everything reads paper_results/bridge/all_{dataset}_{model}.parquet. No GPU, no
model, no judge, no network.

Usage:
    uv run python token_analysis/signal_eval.py --selftest
    uv run python token_analysis/signal_eval.py
    uv run python token_analysis/signal_eval.py --datasets opi --pool all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from token_analysis.budget_eval import segment_frame  # noqa: E402
from token_analysis.common import (  # noqa: E402
    DATASET_MODELS,
    DEFAULT_INPUT_DIR,
    MODEL_INFO,
    source_path,
)
from token_analysis.table_colour import (  # noqa: E402
    MIDPOINT,
    contrast_with_ink,
    diverging_fill,
    relative_luminance,
)

OUT_DIR = Path(__file__).resolve().parent / "signal_eval"
BUDGET_RESULTS = Path(__file__).resolve().parent / "budget_eval" / "budget_results.parquet"

SIGNALS = [
    "surprisal", "entropy", "varentropy", "temporal_kl",
    "lookback_ratio", "sink_drain", "head_disagreement", "w",
    "resid_jump", "norm_ratio", "peak_ratio", "dominant_mass", "resid_jump_nla",
]
FAMILY = {s: f for f, group in {
    "predictive distribution": ["surprisal", "entropy", "varentropy", "temporal_kl"],
    "attention pattern": ["lookback_ratio", "sink_drain", "head_disagreement", "w"],
    "activation": ["resid_jump", "norm_ratio", "peak_ratio", "dominant_mass", "resid_jump_nla"],
}.items() for s in group}
DATASET_NAME = {"opi": "OpenPromptInjection", "tt": "Tensor Trust",
                "liars": "Liars' Bench", "taboo": "Taboo organisms"}

N_BOOT, SEED = 2000, 0
N_BIN = 512          # rank bins for the bootstrap; the point estimate is exact
POSITION_DECILES = 10
SEGMENT_BINS = 5
CORR_SAMPLE = 200_000
BH_Q = 0.05


# --------------------------------------------------------------------------- #
# AUROC                                                                        #
# --------------------------------------------------------------------------- #
def auroc(scores, labels) -> float:
    """Tie-aware binary AUROC with pairwise deletion of non-finite scores."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=np.int8)
    finite = np.isfinite(s)
    s, y = s[finite], y[finite]
    pos = int(y.sum())
    neg = len(y) - pos
    if not pos or not neg:
        return float("nan")
    ranks = rankdata(s, method="average")
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _rank_bins(scores: np.ndarray) -> np.ndarray:
    ranks = rankdata(scores, method="average")
    return np.minimum(((ranks - 1) / len(ranks) * N_BIN).astype(np.int32), N_BIN - 1)


def _auroc_from_histograms(pos_hist: np.ndarray, neg_hist: np.ndarray) -> np.ndarray:
    """AUROC per row of aggregated rank histograms; ties inside a bin score 0.5."""
    below = np.cumsum(neg_hist, -1) - neg_hist
    num = (pos_hist * (below + 0.5 * neg_hist)).sum(-1)
    den = pos_hist.sum(-1) * neg_hist.sum(-1)
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


def _class_histograms(scores, labels, case_index, n_cases):
    finite = np.isfinite(scores)
    s, y, c = scores[finite], labels[finite], case_index[finite]
    if not y.sum() or not (1 - y).sum():
        return None, None
    b = _rank_bins(s)
    pos = np.zeros((n_cases, N_BIN), np.float32)
    neg = np.zeros((n_cases, N_BIN), np.float32)
    np.add.at(pos, (c[y == 1], b[y == 1]), 1.0)
    np.add.at(neg, (c[y == 0], b[y == 0]), 1.0)
    return pos, neg


def bootstrap_replicates(scores, labels, case_index, n_cases,
                         n_boot=N_BOOT, seed=SEED) -> np.ndarray:
    """AUROC of each resample, drawing whole transcripts with replacement.

    Positions inside one transcript are correlated, so the transcript is the
    resampling unit and the effective sample size is the transcript count."""
    pos, neg = _class_histograms(scores, labels, case_index, n_cases)
    if pos is None:
        return np.full(n_boot, np.nan)
    rng = np.random.default_rng(seed)
    counts = np.zeros((n_boot, n_cases), np.float32)
    draws = rng.integers(0, n_cases, (n_boot, n_cases))
    for i in range(n_boot):
        np.add.at(counts[i], draws[i], 1.0)
    return _auroc_from_histograms(counts @ pos, counts @ neg)


def bootstrap_ci(replicates: np.ndarray) -> tuple[float, float]:
    if not np.isfinite(replicates).any():
        return float("nan"), float("nan")
    return (float(np.nanpercentile(replicates, 2.5)),
            float(np.nanpercentile(replicates, 97.5)))


def bootstrap_p(replicates: np.ndarray) -> float:
    """Two-sided achieved significance level against chance, read off the same
    clustered resamples that give the interval."""
    r = replicates[np.isfinite(replicates)]
    if not len(r):
        return float("nan")
    tail = min((r <= 0.5).mean(), (r >= 0.5).mean())
    return float(min(1.0, 2 * max(tail, 1.0 / len(r))))


def benjamini_hochberg(pvalues, q=BH_Q) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    keep = np.zeros(len(p), dtype=bool)
    finite = np.flatnonzero(np.isfinite(p))
    if not len(finite):
        return keep
    order = finite[np.argsort(p[finite])]
    m = len(order)
    passed = p[order] <= q * np.arange(1, m + 1) / m
    if passed.any():
        keep[order[: np.flatnonzero(passed)[-1] + 1]] = True
    return keep


def case_macro_auroc(scores, labels, bounds) -> tuple[float, int]:
    """AUROC inside each transcript carrying both classes, averaged over
    transcripts. This is the ordering a fixed budget actually consumes."""
    values = [auroc(scores[lo:hi], labels[lo:hi]) for lo, hi in
              zip(bounds[:-1], bounds[1:], strict=True)]
    values = [v for v in values if not np.isnan(v)]
    return (float(np.mean(values)) if values else float("nan")), len(values)


def stratified_auroc(scores, labels, strata) -> float:
    """P(on-task score > off-task score | both positions share a stratum).

    Weighting each stratum by its pair count makes this the conditional
    Mann-Whitney statistic, so a signal that only reproduces the stratum
    variable collapses to 0.5."""
    num = den = 0.0
    for value in np.unique(strata):
        mask = strata == value
        a = auroc(scores[mask], labels[mask])
        if np.isnan(a):
            continue
        finite = np.isfinite(scores[mask])
        pos = int(labels[mask][finite].sum())
        neg = int((1 - labels[mask][finite]).sum())
        num += a * pos * neg
        den += pos * neg
    return num / den if den else float("nan")


def adjust(a: float) -> float:
    """Direction-adjusted AUROC: the value once the sign has been fixed."""
    return max(a, 1 - a) if np.isfinite(a) else float("nan")


# --------------------------------------------------------------------------- #
# One dataset and model                                                        #
# --------------------------------------------------------------------------- #
def load_cell(dataset: str, model: str, pool: str, input_dir: Path) -> pl.DataFrame:
    df = segment_frame(pl.read_parquet(source_path(input_dir, dataset, model)), dataset)
    df = df.filter(pl.col("segment") != "trailer")
    if pool == "content":
        df = df.filter(pl.col("analysis_region") != "template")
    return df.sort("case_id", "tok_idx")


def evaluate_cell(dataset: str, model: str, pool: str, input_dir: Path,
                  n_boot: int) -> tuple[list[dict], list[dict], list[dict]]:
    df = load_cell(dataset, model, pool, input_dir)
    y = df["on_task"].to_numpy().astype(np.int8)
    case_ids = df["case_id"].to_numpy()
    _, case_index = np.unique(case_ids, return_inverse=True)
    n_cases = int(case_index.max()) + 1
    bounds = np.flatnonzero(np.r_[True, case_ids[1:] != case_ids[:-1], True])
    base = float(y.mean())

    position = df["full_position_normalized"].to_numpy().astype(float)
    length = np.log(df["full_length"].to_numpy().astype(float))
    decile = np.minimum((position * POSITION_DECILES).astype(int), POSITION_DECILES - 1)
    segment_bin = np.minimum(
        (df["segment_position_normalized"].to_numpy().astype(float) * SEGMENT_BINS).astype(int),
        SEGMENT_BINS - 1)
    structure = np.char.add(np.char.add(
        np.char.add(df["segment"].to_numpy().astype(str), segment_bin.astype(str)),
        "|"), df["analysis_region"].to_numpy().astype(str))

    sample = np.random.default_rng(SEED).choice(
        len(y), min(len(y), CORR_SAMPLE), replace=False)

    rows, values = [], {}
    for signal in SIGNALS:
        v = df[signal].to_numpy().astype(float)
        values[signal] = v
        point = auroc(v, y)
        replicates = bootstrap_replicates(v, y, case_index, n_cases, n_boot)
        lo, hi = bootstrap_ci(replicates)
        macro, n_macro = case_macro_auroc(v, y, bounds)
        finite_sample = sample[np.isfinite(v[sample])]
        rows.append(dict(
            dataset=dataset, model=model, pool=pool, signal=signal, family=FAMILY[signal],
            n_tokens=len(y), n_cases=n_cases, base_rate=base,
            coverage=float(np.isfinite(v).mean()),
            auroc=point, auroc_lo=lo, auroc_hi=hi, auroc_adjusted=adjust(point),
            direction="higher" if point >= 0.5 else "lower",
            case_macro_auroc=macro, case_macro_adjusted=adjust(macro), case_macro_n=n_macro,
            auroc_decile=stratified_auroc(v, y, decile),
            auroc_structure=stratified_auroc(v, y, structure),
            rho_position=float(spearmanr(v[finite_sample], position[finite_sample]).statistic),
            rho_length=float(spearmanr(v[finite_sample], length[finite_sample]).statistic),
            bootstrap_p=bootstrap_p(replicates),
        ))
    for row, keep in zip(rows, benjamini_hochberg([r["bootstrap_p"] for r in rows]),
                         strict=True):
        row["bh_pass"] = bool(keep)
        row["auroc_decile_adjusted"] = adjust(row["auroc_decile"])
        row["auroc_structure_adjusted"] = adjust(row["auroc_structure"])
        row["ci_excludes_chance"] = bool(row["auroc_lo"] > 0.5 or row["auroc_hi"] < 0.5)

    rng = np.random.default_rng(SEED + 1)
    control = rng.standard_normal(len(y))
    shuffled = y[rng.permutation(len(y))]
    best = max(SIGNALS, key=lambda s: adjust(auroc(values[s], y)))
    controls = [dict(dataset=dataset, model=model, pool=pool,
                     auroc=auroc(control, y),
                     auroc_permuted_label=auroc(values[best], shuffled),
                     permuted_signal=best,
                     auroc_position=auroc(position, y),
                     auroc_position_adjusted=adjust(auroc(position, y)))]

    names = [s for s in SIGNALS if np.isfinite(values[s]).mean() > 0.5]
    ranked = np.column_stack([
        rankdata(np.nan_to_num(values[s][sample], nan=np.nanmin(values[s]))) for s in names])
    sign = np.array([1.0 if auroc(values[s], y) >= 0.5 else -1.0 for s in names])
    corr = np.corrcoef(ranked * sign, rowvar=False)
    pairs = [dict(dataset=dataset, model=model, pool=pool, signal_a=a, signal_b=b,
                  family_a=FAMILY[a], family_b=FAMILY[b], rho=float(corr[i, j]))
             for i, a in enumerate(names) for j, b in enumerate(names) if j > i]
    return rows, controls, pairs


# --------------------------------------------------------------------------- #
# Report                                                                       #
# --------------------------------------------------------------------------- #
def _fmt(v, nd=3) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def _cells(results: pl.DataFrame) -> list[tuple[str, str]]:
    return [(d, m) for d in DATASET_MODELS for m in DATASET_MODELS[d]
            if results.filter((pl.col("dataset") == d) & (pl.col("model") == m)).height]


def write_report(results, controls, pairs, out: Path) -> None:
    L = ["# What each signal predicts\n",
         "> Generated by `token_analysis/signal_eval.py` from",
         "> `paper_results/bridge/all_*.parquet`. No model, no judge, no GPU.",
         "> AUROC is pooled over the positions of a dataset and model against the",
         "> judge's on-task label. A value below 0.5 means the signal runs backwards",
         "> and its strength is one minus the value. Intervals resample whole",
         "> transcripts, 2000 times, seed 0.\n"]

    for pool in sorted(results["pool"].unique()):
        sub = results.filter(pl.col("pool") == pool)
        cells = _cells(sub)
        L.append(f"## Pooled AUROC, {pool} tokens\n")
        L.append("| signal | family | " + " | ".join(f"{d}/{m}" for d, m in cells) + " |")
        L.append("|" + "---|" * (2 + len(cells)))
        for signal in SIGNALS:
            row = [f"`{signal}`", FAMILY[signal]]
            for d, m in cells:
                r = sub.filter((pl.col("dataset") == d) & (pl.col("model") == m)
                               & (pl.col("signal") == signal)).row(0, named=True)
                mark = "" if r["bh_pass"] else " (ns)"
                row.append(f"{_fmt(r['auroc'])}{mark}")
            L.append("| " + " | ".join(row) + " |")
        base = ["_base rate_", ""] + [
            _fmt(sub.filter((pl.col("dataset") == d) & (pl.col("model") == m))["base_rate"][0], 3)
            for d, m in cells]
        L.append("| " + " | ".join(base) + " |")
        ctl = controls.filter(pl.col("pool") == pool)
        rnd = ["_control, random_", ""] + [
            _fmt(ctl.filter((pl.col("dataset") == d) & (pl.col("model") == m))["auroc"][0])
            for d, m in cells]
        L.append("| " + " | ".join(rnd) + " |")
        perm = ["_control, permuted label_", ""] + [
            _fmt(ctl.filter((pl.col("dataset") == d)
                            & (pl.col("model") == m))["auroc_permuted_label"][0])
            for d, m in cells]
        L.append("| " + " | ".join(perm) + " |")
        pos = ["_position alone_", ""] + [
            _fmt(ctl.filter((pl.col("dataset") == d)
                            & (pl.col("model") == m))["auroc_position"][0]) for d, m in cells]
        L.append("| " + " | ".join(pos) + " |")
        L.append("")

    main = results.filter(pl.col("pool") == "all")
    L.append("## Direction\n")
    L.append("`H` means larger values select on-task positions, `L` means smaller"
             " values do. The last two columns count the datasets whose models all"
             " agree, and the cells sitting in the minority direction.\n")
    cells = _cells(main)
    L.append("| signal | " + " | ".join(f"{d}/{m}" for d, m in cells)
             + " | datasets agreeing | minority cells |")
    L.append("|" + "---|" * (3 + len(cells)))
    for signal in SIGNALS:
        sig = main.filter(pl.col("signal") == signal)
        marks, agree = [], 0
        for d in DATASET_MODELS:
            inner = sig.filter(pl.col("dataset") == d)
            if inner.height and inner["direction"].n_unique() == 1:
                agree += 1
        higher = int((sig["auroc"] >= 0.5).sum())
        for d, m in cells:
            r = sig.filter((pl.col("dataset") == d) & (pl.col("model") == m)).row(0, named=True)
            marks.append("H" if r["direction"] == "higher" else "L")
        L.append(f"| `{signal}` | " + " | ".join(marks)
                 + f" | {agree}/{main['dataset'].n_unique()}"
                 + f" | {min(higher, len(cells) - higher)} |")
    L.append("")

    L.append("## Position and structure\n")
    L.append("`rho position` and `rho length` are Spearman correlations of the signal"
             " with the normalized token index and with the log transcript length."
             " `within structure` recomputes the AUROC inside strata of segment, chat"
             " role and position bin, so it never compares two positions of different"
             " structural kind. Both AUROC columns are direction adjusted.\n")
    L.append("| dataset | model | signal | rho position | rho length | pooled |"
             " within decile | within structure |")
    L.append("|" + "---|" * 8)
    for d, m in cells:
        for signal in SIGNALS:
            r = main.filter((pl.col("dataset") == d) & (pl.col("model") == m)
                            & (pl.col("signal") == signal)).row(0, named=True)
            L.append(f"| {d} | {m} | `{signal}` | {_fmt(r['rho_position'], 2)} |"
                     f" {_fmt(r['rho_length'], 2)} | {_fmt(r['auroc_adjusted'])} |"
                     f" {_fmt(r['auroc_decile_adjusted'])} |"
                     f" {_fmt(r['auroc_structure_adjusted'])} |")
    L.append("")

    L.append("## The winning signal against the free rankers\n")
    L.append("`position` and `structure` are the out of fold baselines of"
             " `budget_eval.py`, which need no forward pass. A signal earns its"
             " forward pass by beating `structure`.\n")
    free = (pl.read_parquet(BUDGET_RESULTS).filter(pl.col("pool") == "all")
            if BUDGET_RESULTS.exists() else None)
    L.append("| dataset | model | winner | direction | pooled [95% CI] | case-macro |"
             " within structure | position | structure |")
    L.append("|" + "---|" * 9)
    for d, m in cells:
        w = (main.filter((pl.col("dataset") == d) & (pl.col("model") == m))
             .sort("auroc_adjusted", descending=True).row(0, named=True))
        lo, hi = ((w["auroc_lo"], w["auroc_hi"]) if w["direction"] == "higher"
                  else (1 - w["auroc_hi"], 1 - w["auroc_lo"]))
        cols = [d, m, f"`{w['signal']}`", w["direction"],
                f"{_fmt(w['auroc_adjusted'])} [{_fmt(lo)}, {_fmt(hi)}]",
                _fmt(w["case_macro_adjusted"]), _fmt(w["auroc_structure_adjusted"])]
        for name in ("position", "structure"):
            if free is None:
                cols.append("—")
                continue
            r = free.filter((pl.col("dataset") == d) & (pl.col("model") == m)
                            & (pl.col("selector") == name))
            cols.append(_fmt(r["pooled_auroc"][0]) if r.height else "—")
        L.append("| " + " | ".join(cols) + " |")
    L.append("")

    L.append("## Redundancy\n")
    L.append("Oriented rank correlation between signals, averaged over the cells."
             " Components with near-zero correlation are the pairs a mixture can"
             " exploit.\n")
    p = pairs.filter(pl.col("pool") == "all")
    grouped = (p.with_columns(pl.when(pl.col("family_a") == pl.col("family_b"))
                              .then(pl.col("family_a")).otherwise(pl.lit("across families"))
                              .alias("group"))
               .group_by("group").agg(pl.col("rho").mean().alias("mean_rho"),
                                      pl.col("rho").abs().mean().alias("mean_abs_rho"),
                                      pl.len().alias("pairs")).sort("mean_abs_rho",
                                                                    descending=True))
    L.append("| group | mean rho | mean absolute rho | pairs |")
    L.append("|" + "---|" * 4)
    for r in grouped.iter_rows(named=True):
        L.append(f"| {r['group']} | {_fmt(r['mean_rho'])} | {_fmt(r['mean_abs_rho'])} |"
                 f" {r['pairs']} |")
    L.append("")
    top = (p.group_by("signal_a", "signal_b").agg(pl.col("rho").mean().alias("mean_rho"))
           .sort("mean_rho", descending=True).head(8))
    L.append("| most correlated pair | mean rho |")
    L.append("|---|---|")
    for r in top.iter_rows(named=True):
        L.append(f"| `{r['signal_a']}` and `{r['signal_b']}` | {_fmt(r['mean_rho'])} |")
    L.append("")
    out.write_text("\n".join(L))


# --------------------------------------------------------------------------- #
# Diverging fill for the graded table                                          #
# --------------------------------------------------------------------------- #
# Polarity, so a diverging scale: two hues that read as opposite and a neutral
# gray midpoint at chance. Blue marks a signal whose larger values select
# on-task positions, red one whose smaller values do. The printed number carries
# the same information, so colour is never the only channel.
FILL_FULL_SCALE = 0.35   # |AUROC - 1/2| that reaches the deepest wash;
                         # shared with grade_paper_tables.py so one scale
                         # covers every AUROC printed in the paper


def cell_fill(value: float) -> str:
    return diverging_fill(value, centre=0.5, half_range=FILL_FULL_SCALE)


def write_latex(results: pl.DataFrame, controls: pl.DataFrame, out: Path) -> None:
    """The main-text signal table, one row per signal and one column per cell."""
    main = results.filter(pl.col("pool") == "all")
    cells = _cells(main)
    groups = [(d, [m for dd, m in cells if dd == d]) for d in DATASET_MODELS
              if any(dd == d for dd, _ in cells)]
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
         r"\setlength{\tabcolsep}{2.2pt}", r"\renewcommand{\arraystretch}{1.15}",
         r"\caption{Pooled AUROC of every signal against the on-task label, over every "
         r"position of every rendered transcript. Blue marks a signal whose larger values "
         r"select on-task positions and red one whose smaller values do, so $0.239$ and "
         r"$0.761$ are equally informative. Bold is the entry furthest from $0.5$ in each "
         r"column; $^{\circ}$ fails Benjamini-Hochberg control at $q=0.05$.}",
         r"\label{tab:signal-auroc}",
         r"\begin{tabular}{@{}l" + "".join("r" * len(ms) for _, ms in groups) + r"@{}}",
         r"\toprule"]
    header = " & ".join(rf"\multicolumn{{{len(ms)}}}{{c}}{{{DATASET_NAME[d]}}}"
                        for d, ms in groups)
    L.append(" & " + header + r" \\")
    start = 2
    rules = []
    for _, ms in groups:
        rules.append(rf"\cmidrule(lr){{{start}-{start + len(ms) - 1}}}")
        start += len(ms)
    L.append("".join(rules))
    L.append("signal & " + " & ".join(m for _, ms in groups for m in ms) + r" \\")
    L.append(r"\midrule")

    extreme = {}
    for d, m in cells:
        sub = main.filter((pl.col("dataset") == d) & (pl.col("model") == m))
        extreme[(d, m)] = sub.sort("auroc_adjusted", descending=True)["signal"][0]

    last_family = None
    for signal in SIGNALS:
        if FAMILY[signal] != last_family:
            if last_family is not None:
                L.append(r"\addlinespace")
            L.append(rf"\multicolumn{{{1 + len(cells)}}}{{l}}{{\emph{{{FAMILY[signal]}}}}} \\")
            last_family = FAMILY[signal]
        cols = []
        for d, m in cells:
            r = main.filter((pl.col("dataset") == d) & (pl.col("model") == m)
                            & (pl.col("signal") == signal)).row(0, named=True)
            txt = f"{r['auroc']:.3f}"
            if extreme[(d, m)] == signal:
                txt = rf"\textbf{{{txt}}}"
            if not r["bh_pass"]:
                txt += r"$^{\circ}$"
            cols.append(rf"\cellcolor[HTML]{{{cell_fill(r['auroc'])}}}{txt}")
        L.append(rf"\texttt{{{signal.replace('_', r'\_')}}} & " + " & ".join(cols) + r" \\")

    L.append(r"\midrule")
    ctl = controls.filter(pl.col("pool") == "all")
    for label, column, source in (("base rate", "base_rate", main),
                                  ("position alone", "auroc_position", ctl),
                                  ("random", "auroc", ctl)):
        cols = []
        for d, m in cells:
            sub = source.filter((pl.col("dataset") == d) & (pl.col("model") == m))
            cols.append(f"{sub[column][0]:.3f}")
        L.append(rf"\emph{{{label}}} & " + " & ".join(cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(L) + "\n")


def write_latex_controls(results: pl.DataFrame, out: Path) -> None:
    """The appendix table: the winning signal of every cell beside its interval,
    its within-transcript value, its value with chat structure held fixed, and
    the two rankers that need no forward pass."""
    main = results.filter(pl.col("pool") == "all")
    free = (pl.read_parquet(BUDGET_RESULTS).filter(pl.col("pool") == "all")
            if BUDGET_RESULTS.exists() else None)
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
         r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.1}",
         r"\caption{The strongest signal of every dataset and model. \emph{Within "
         r"structure} recomputes the AUROC inside strata of segment, chat role and position "
         r"bin. \emph{Position} and \emph{structure} are the free rankers of "
         r"\Cref{sec:bridge}. Every AUROC is direction adjusted.}",
         r"\label{tab:signal-controls}",
         r"\begin{tabular}{@{}llllrrrr@{}}", r"\toprule",
         r"Dataset & Model & Signal & Direction & Pooled [95\% CI] & Case-macro"
         r" & \makecell[r]{Within\\structure} & \makecell[r]{Free\\structure} \\",
         r"\midrule"]
    previous = None
    for dataset, model in _cells(main):
        w = (main.filter((pl.col("dataset") == dataset) & (pl.col("model") == model))
             .sort("auroc_adjusted", descending=True).row(0, named=True))
        lo, hi = ((w["auroc_lo"], w["auroc_hi"]) if w["direction"] == "higher"
                  else (1 - w["auroc_hi"], 1 - w["auroc_lo"]))
        structure = "—"
        if free is not None:
            r = free.filter((pl.col("dataset") == dataset) & (pl.col("model") == model)
                            & (pl.col("selector") == "structure"))
            if r.height:
                structure = f"{r['pooled_auroc'][0]:.3f}"
        first = dataset != previous
        if previous is not None and first:
            L.append(r"\addlinespace")
        previous = dataset
        L.append(f"{DATASET_NAME[dataset] if first else ''} & {model} & "
                 rf"\texttt{{{w['signal'].replace('_', r'\_')}}} & {w['direction']} & "
                 rf"\({w['auroc_adjusted']:.3f}\) \([{lo:.3f}, {hi:.3f}]\) & "
                 rf"\({w['case_macro_adjusted']:.3f}\) & "
                 rf"\({w['auroc_structure_adjusted']:.3f}\) & \({structure}\) \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #
# Self-test                                                                    #
# --------------------------------------------------------------------------- #
def selftest() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    s = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([0, 0, 1, 1])
    check("a perfect ranker scores 1", auroc(s, y) == 1.0)
    check("a reversed ranker scores 0", auroc(-s, y) == 0.0)
    check("a constant ranker scores 0.5", auroc(np.ones(4), y) == 0.5)
    check("direction adjustment folds the scale", adjust(0.239) == 0.761)
    check("one class gives no AUROC", np.isnan(auroc(s, np.zeros(4, int))))

    # The binned bootstrap must recover the exact statistic when it resamples
    # every transcript exactly once.
    rng = np.random.default_rng(0)
    v = rng.normal(size=4000)
    yy = (rng.random(4000) < 0.3).astype(np.int8)
    ci = rng.integers(0, 40, 4000)
    pos, neg = _class_histograms(v, yy, ci, 40)
    identity = _auroc_from_histograms(pos.sum(0), neg.sum(0))
    check("binned AUROC matches the exact statistic to 1e-3",
          abs(float(identity) - auroc(v, yy)) < 1e-3)
    replicates = bootstrap_replicates(v, yy, ci, 40, n_boot=400)
    lo, hi = bootstrap_ci(replicates)
    check("the interval covers the point estimate", lo <= auroc(v, yy) <= hi)
    check("an uninformative signal is not significant", bootstrap_p(replicates) > 0.05)
    strong = np.where(yy == 1, v + 3.0, v)
    check("a strong signal reaches the smallest attainable p-value",
          bootstrap_p(bootstrap_replicates(strong, yy, ci, 40, n_boot=400)) <= 2 / 400)

    # A score that is a relabelling of the stratum carries nothing inside it.
    strata = np.repeat(np.arange(4), 100)
    label = np.tile(np.r_[np.ones(30, int), np.zeros(70, int)], 4).astype(np.int8)
    check("a pure stratum score collapses to 0.5 within strata",
          abs(stratified_auroc(strata.astype(float), label, strata) - 0.5) < 1e-9)
    inside = strata + np.tile(np.r_[np.ones(30), np.zeros(70)], 4) * 0.5
    check("a score that orders inside a stratum survives",
          stratified_auroc(inside, label, strata) == 1.0)

    # Case-macro ignores transcripts carrying a single class.
    bounds = np.array([0, 4, 8])
    s2 = np.array([1.0, 2, 3, 4, 1, 2, 3, 4])
    y2 = np.array([0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int8)
    macro, n = case_macro_auroc(s2, y2, bounds)
    check("single-class transcripts are skipped", macro == 1.0 and n == 1)

    # The diverging fill: gray at chance, one hue per direction, and every fill
    # light enough to keep the printed number legible on it.
    check("chance is the neutral midpoint",
          cell_fill(0.5) == MIDPOINT.lstrip("#").upper())
    check("the two directions take different hues",
          cell_fill(0.8) != cell_fill(0.2))
    check("the fill is symmetric in distance from chance",
          cell_fill(0.9) == cell_fill(0.99))
    check("the fill deepens away from chance",
          relative_luminance("#" + cell_fill(0.55))
          > relative_luminance("#" + cell_fill(0.75)))
    check("every fill keeps the number above 4.5:1",
          min(contrast_with_ink("#" + cell_fill(a))
              for a in np.linspace(0.0, 1.0, 201)) >= 4.5)

    # Benjamini-Hochberg is step up: with m=4 and q=0.05 the thresholds are
    # 0.0125, 0.025, 0.0375 and 0.05, so 0.04 fails and nothing above it is kept.
    keep = benjamini_hochberg([0.001, 0.02, 0.9, 0.04])
    check("BH stops at the largest index meeting its threshold",
          keep.tolist() == [True, True, False, False])
    check("BH on all-large p-values keeps nothing",
          not benjamini_hochberg([0.9, 0.8, 0.7]).any())
    check("BH keeps a p-value above its own threshold when a later one passes",
          benjamini_hochberg([0.001, 0.03, 0.02, 0.9]).tolist()
          == [True, True, True, False])

    print(f"\nselftest: {'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_MODELS))
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--pool", nargs="+", default=["all", "content"],
                    choices=["all", "content"])
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, controls, pairs = [], [], []
    for dataset in args.datasets:
        for model in DATASET_MODELS[dataset]:
            if args.models and model not in args.models:
                continue
            print(f"{dataset}/{MODEL_INFO[model]['name']}", flush=True)
            for pool in args.pool:
                r, c, p = evaluate_cell(dataset, model, pool, args.input_dir,
                                        args.n_boot)
                rows += r
                controls += c
                pairs += p
                print(f"    {pool} done", flush=True)

    results = pl.DataFrame(rows)
    control_frame = pl.DataFrame(controls)
    pair_frame = pl.DataFrame(pairs)
    results.write_parquet(args.out_dir / "signal_results.parquet")
    control_frame.write_parquet(args.out_dir / "signal_controls.parquet")
    pair_frame.write_parquet(args.out_dir / "signal_pairs.parquet")
    write_report(results, control_frame, pair_frame, args.out_dir / "signal_eval.md")
    if "all" in args.pool:
        write_latex(results, control_frame, args.out_dir / "signal_table.tex")
        write_latex_controls(results, args.out_dir / "signal_controls_table.tex")
    print(f"\nwrote {args.out_dir}/signal_eval.md, two .tex tables and three parquets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
