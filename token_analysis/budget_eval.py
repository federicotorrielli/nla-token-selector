"""token_analysis/budget_eval.py — what a token selector actually buys at a budget.

AUROC scores a ranking and has no notion of cost. An auditor has a cost: the
number of NLA explanations they can afford for one transcript. This script
reports what each selector buys at that budget, next to the pooled AUROC the
findings docs already report, and measures every selector against a position
baseline that needs no forward pass at all.

Four questions, one per output table:

  budget    of the explanations actually bought, how many come back on-task?
            Budgets are top-1, top-8, top-1% and top-10% within each transcript.
  within    does the selector order tokens inside one transcript, or only
            separate transcripts from each other? Pooled AUROC cannot tell the
            difference; case-macro AUROC can.
  position  does the selector beat a ranker built from token position, segment
            and transcript length alone? Those need no forward pass, so any
            gain over them is what the signals are worth.
  filter    what does discarding structural tokens cost, and what does it buy?

Everything reads paper_results/bridge/all_{dataset}_{model}.parquet. No GPU, no
model, no judge, no network.

Usage:
    uv run python token_analysis/budget_eval.py --selftest
    uv run python token_analysis/budget_eval.py
    uv run python token_analysis/budget_eval.py --datasets opi taboo --pool content
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from token_analysis.common import (  # noqa: E402
    AUROC_ENSEMBLE_SELECTION_PATH,
    AUROC_ENSEMBLE_SHARED_SELECTION_PATH,
    CONTENT_WINNERS,
    DATASET_MODELS,
    DEFAULT_INPUT_DIR,
    METRIC_SPECS,
    MODEL_INFO,
    SHARED_METRIC_SPECS,
    load_auroc_ensemble_specs,
    source_path,
)
from token_analysis.selection_methods.statistics import auroc  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "budget_eval"
TT_EXPECTED_BOUNDARY_LENGTH = 5
BUDGETS = (("top1", 1, None), ("top8", 8, None), ("1pct", None, 0.01), ("10pct", None, 0.10))
N_BOOT = 2000
SEED = 0
N_FOLDS = 5
POSITION_REGIONS = ("template", "system", "user", "assistant", "assistant_prior")
POSITION_SEGMENTS = ("input", "boundary", "output", "trailer")


# --------------------------------------------------------------------------- #
# Segmentation — same definitions as build_token_position_data.py             #
# --------------------------------------------------------------------------- #
def segment_frame(df: pl.DataFrame, dataset: str) -> pl.DataFrame:
    """Add the input / boundary / output / trailer segmentation and the position
    columns the baseline reads. `boundary` is the final contiguous run of
    template tokens before the assistant turn; for OPI, which stores no
    response, it is the trailing generation prompt."""
    df = df.with_columns(
        pl.len().over("case_id").cast(pl.Int64).alias("full_length"),
        pl.when(pl.col("region") == "assistant")
        .then(pl.col("tok_idx"))
        .otherwise(None)
        .min()
        .over("case_id")
        .alias("_output_start"),
        pl.when(pl.col("region") == "assistant")
        .then(pl.col("tok_idx"))
        .otherwise(None)
        .max()
        .over("case_id")
        .alias("_output_end"),
        (pl.col("region") == "assistant").sum().over("case_id").cast(pl.Int64)
        .alias("output_length"),
    )
    df = df.with_columns(pl.col("_output_start").fill_null(pl.col("full_length")).alias("_cut"))
    df = df.with_columns(
        pl.when((pl.col("tok_idx") < pl.col("_cut")) & (pl.col("region") != "template"))
        .then(pl.col("tok_idx"))
        .otherwise(None)
        .max()
        .over("case_id")
        .alias("_source_input_end")
    )
    df = df.with_columns(
        (pl.col("_cut") - pl.col("_source_input_end").fill_null(-1) - 1)
        .cast(pl.Int64)
        .alias("source_boundary_length")
    )
    # Two Tensor Trust cases leave the one-character user message `a` labelled
    # template, which merges its user turn into the generation prompt. Keep the
    # source length, but segment on the known prompt length.
    corrected = pl.lit(dataset == "tt") & (
        pl.col("source_boundary_length") > TT_EXPECTED_BOUNDARY_LENGTH
    )
    df = df.with_columns(
        pl.when(corrected)
        .then(pl.col("_cut") - TT_EXPECTED_BOUNDARY_LENGTH - 1)
        .otherwise(pl.col("_source_input_end"))
        .alias("_input_end"),
        corrected.alias("boundary_was_corrected"),
    )
    df = df.with_columns(
        (pl.col("_input_end").fill_null(-1) + 1).cast(pl.Int64).alias("input_length"),
        (pl.col("_cut") - pl.col("_input_end").fill_null(-1) - 1).cast(pl.Int64)
        .alias("boundary_length"),
    )
    df = df.with_columns(
        pl.when(
            pl.col("boundary_was_corrected")
            & (pl.col("tok_idx") == pl.col("input_length") - 1)
            & (pl.col("region") == "template")
        )
        .then(pl.lit("user"))
        .otherwise(pl.col("region"))
        .alias("analysis_region"),
        pl.when(pl.col("tok_idx") < pl.col("input_length"))
        .then(pl.lit("input"))
        .when(pl.col("tok_idx") < pl.col("_cut"))
        .then(pl.lit("boundary"))
        .when(pl.col("region") == "assistant")
        .then(pl.lit("output"))
        .otherwise(pl.lit("trailer"))
        .alias("segment"),
    )
    df = df.with_columns(
        pl.when(pl.col("segment") == "input")
        .then(0)
        .when(pl.col("segment") == "boundary")
        .then(pl.col("input_length"))
        .when(pl.col("segment") == "output")
        .then(pl.col("_cut"))
        .otherwise(pl.col("_output_end") + 1)
        .cast(pl.Int64)
        .alias("_segment_start"),
        pl.when(pl.col("segment") == "input")
        .then(pl.col("input_length"))
        .when(pl.col("segment") == "boundary")
        .then(pl.col("boundary_length"))
        .when(pl.col("segment") == "output")
        .then(pl.col("output_length"))
        .otherwise(pl.col("full_length") - pl.col("_output_end") - 1)
        .cast(pl.Int64)
        .alias("segment_length"),
    )
    df = df.with_columns(
        (pl.col("tok_idx") - pl.col("_segment_start")).cast(pl.Int64).alias("segment_position"),
        pl.when(pl.col("full_length") > 1)
        .then(pl.col("tok_idx") / (pl.col("full_length") - 1))
        .otherwise(0.0)
        .alias("full_position_normalized"),
    )
    return df.with_columns(
        pl.when(pl.col("segment_length") > 1)
        .then(pl.col("segment_position") / (pl.col("segment_length") - 1))
        .otherwise(0.0)
        .alias("segment_position_normalized")
    )


# --------------------------------------------------------------------------- #
# Candidate scores, all oriented so that larger means more likely on-task     #
# --------------------------------------------------------------------------- #
def pooled_midrank(values: np.ndarray) -> np.ndarray:
    """Fractional midrank of the finite values, mapped to [0, 1]. Non-finite
    values stay NaN; they are never imputed. Same definition the ensemble
    selector fits."""
    out = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    n = int(finite.sum())
    if n == 0:
        return out
    if n == 1:
        out[finite] = 0.5
        return out
    ranks = pl.Series(values[finite]).rank("average").to_numpy()
    out[finite] = (ranks - 1) / (n - 1)
    return out


def single_score(df: pl.DataFrame, metric: str, direction: str) -> np.ndarray:
    v = df[metric].to_numpy().astype(float)
    return v if direction == "higher" else -v


def ensemble_score(df: pl.DataFrame, spec: dict) -> np.ndarray:
    """Reconstruct a selected rank ensemble. Components become pooled midranks,
    are oriented towards the on-task class, then mixed at the stored weights.
    The mixture is NaN wherever either component is missing."""
    score = pooled_midrank(df[str(spec["component_1"])].to_numpy().astype(float))
    if spec["component_1_direction"] == "lower":
        score = 1.0 - score
    score = score * float(spec["component_1_weight"])
    second = spec.get("component_2")
    if second is not None:
        other = pooled_midrank(df[str(second)].to_numpy().astype(float))
        if spec.get("component_2_direction") == "lower":
            other = 1.0 - other
        score = score + other * float(spec["component_2_weight"])
    return score if spec.get("candidate_direction") == "higher" else 1.0 - score


def position_features(df: pl.DataFrame, kind: str) -> np.ndarray:
    """Features a selector could use without running the model at all.

    `position` sees only where the token sits and how long the transcript is.
    `structure` also sees the chat template: which segment and role the token
    belongs to, and its ordinal inside the generation boundary. Both are free.
    The gap between them is how much of a free selector is sequence position and
    how much is template structure, and a signal has to beat `structure` before
    its forward pass has bought anything.
    """
    pos = df["full_position_normalized"].to_numpy().astype(float)
    length = np.log1p(df["full_length"].to_numpy().astype(float))
    cols = [pos, pos**2, pos**3, length, pos * length]
    if kind == "position":
        return np.column_stack(cols)
    seg_pos = df["segment_position_normalized"].to_numpy().astype(float)
    ordinal = df["segment_position"].to_numpy().astype(float)
    region = df["analysis_region"].to_numpy()
    segment = df["segment"].to_numpy()
    cols += [seg_pos, seg_pos**2, (df["tok_idx"].to_numpy() == 0).astype(float)]
    cols += [np.where(segment == s, ordinal, 0.0) for s in POSITION_SEGMENTS]
    cols += [(segment == s).astype(float) for s in POSITION_SEGMENTS]
    cols += [(region == r).astype(float) for r in POSITION_REGIONS]
    return np.column_stack(cols)


def _case_folds(case_ids: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold index per row, assigned by whole case so no transcript is split."""
    cases = np.unique(case_ids)
    rng = np.random.default_rng(seed)
    assign = dict(zip(cases, rng.integers(0, n_folds, len(cases)), strict=True))
    return np.array([assign[c] for c in case_ids])


def position_baseline(df: pl.DataFrame, y: np.ndarray, kind: str,
                      n_folds: int = N_FOLDS, seed: int = SEED) -> np.ndarray:
    """Out-of-fold score from a free ranker. Folds are grouped by transcript, so
    the baseline never trains on a case it scores."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = position_features(df, kind)
    folds = _case_folds(df["case_id"].to_numpy(), n_folds, seed)
    out = np.full(len(y), np.nan)
    for f in range(n_folds):
        train, test = folds != f, folds == f
        if not test.any() or len(np.unique(y[train])) < 2:
            continue
        scaler = StandardScaler().fit(x[train])
        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(scaler.transform(x[train]), y[train])
        out[test] = model.decision_function(scaler.transform(x[test]))
    return out


# --------------------------------------------------------------------------- #
# Budget and within-transcript metrics                                        #
# --------------------------------------------------------------------------- #
def per_case_spend(scores: np.ndarray, y: np.ndarray, case_ids: np.ndarray,
                   k: int | None, frac: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Hits and spend per transcript when the budget is spent on that
    transcript's highest-scoring positions. Positions with no score are never
    bought. Ties break by position, which is the pessimistic choice."""
    hits, spend = [], []
    order = np.argsort(case_ids, kind="mergesort")
    s, yy, c = scores[order], y[order], case_ids[order]
    bounds = np.flatnonzero(np.r_[True, c[1:] != c[:-1], True])
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        cs, cy = s[lo:hi], yy[lo:hi]
        finite = np.isfinite(cs)
        n = int(finite.sum())
        if n == 0:
            continue
        budget = k if k is not None else max(1, int(np.ceil(frac * n)))
        budget = min(budget, n)
        pick = np.argsort(-cs[finite], kind="mergesort")[:budget]
        hits.append(float(cy[finite][pick].sum()))
        spend.append(float(budget))
    return np.asarray(hits), np.asarray(spend)


def precision_ci(hits: np.ndarray, spend: np.ndarray, n_boot: int = N_BOOT,
                 seed: int = SEED) -> tuple[float, float, float]:
    """Precision at the budget, with a bootstrap over whole transcripts."""
    if not len(hits) or spend.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(hits.sum() / spend.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(hits), (n_boot, len(hits)))
    boots = hits[idx].sum(1) / np.maximum(spend[idx].sum(1), 1e-9)
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def case_macro_auroc(scores: np.ndarray, y: np.ndarray,
                     case_ids: np.ndarray) -> tuple[float, int]:
    """AUROC computed inside each transcript that carries both classes, then
    averaged over transcripts. This is the ordering the budget actually uses."""
    order = np.argsort(case_ids, kind="mergesort")
    s, yy, c = scores[order], y[order], case_ids[order]
    bounds = np.flatnonzero(np.r_[True, c[1:] != c[:-1], True])
    vals = []
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        a = auroc(s[lo:hi], yy[lo:hi])
        if not np.isnan(a):
            vals.append(a)
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


# --------------------------------------------------------------------------- #
# One cell                                                                     #
# --------------------------------------------------------------------------- #
def candidates_for(dataset: str, model: str, ens_best: dict, ens_shared: dict) -> dict:
    key = (dataset, model)
    out = {}
    spec = METRIC_SPECS[key]
    out["single_best"] = ("single", spec["metric"], spec["direction"])
    shared = SHARED_METRIC_SPECS[key]
    out["single_shared"] = ("single", shared["metric"], shared["direction"])
    content = CONTENT_WINNERS[key]
    out["single_content"] = ("single", content, None)  # direction fitted below
    if key in ens_best:
        out["ensemble_best"] = ("ensemble", ens_best[key], None)
    if key in ens_shared:
        out["ensemble_shared"] = ("ensemble", ens_shared[key], None)
    return out


def evaluate_cell(dataset: str, model: str, pool: str, ens_best: dict, ens_shared: dict,
                  input_dir: Path, n_boot: int) -> list[dict]:
    df = pl.read_parquet(source_path(input_dir, dataset, model))
    df = segment_frame(df, dataset)
    df = df.filter(pl.col("segment") != "trailer")
    if pool == "content":
        df = df.filter(pl.col("analysis_region") != "template")
    y = df["on_task"].to_numpy().astype(int)
    cases = df["case_id"].to_numpy()
    base = float(y.mean())

    scores: dict[str, np.ndarray] = {}
    for name, (kind, a, b) in candidates_for(dataset, model, ens_best, ens_shared).items():
        if kind == "single":
            direction = b
            if direction is None:  # content winner: orient on this pool
                direction = "higher" if auroc(df[a].to_numpy().astype(float), y) >= 0.5 else "lower"
            scores[name] = single_score(df, a, direction)
            scores[name + "__label"] = f"{'higher' if direction == 'higher' else 'lower'} {a}"
        else:
            scores[name] = ensemble_score(df, a)
            scores[name + "__label"] = str(a["candidate_id"])
    scores["position"] = position_baseline(df, y, "position")
    scores["position__label"] = "position and length"
    scores["structure"] = position_baseline(df, y, "structure")
    scores["structure__label"] = "position, segment and role"
    rng = np.random.default_rng(SEED)
    scores["random"] = rng.standard_normal(len(y))
    scores["random__label"] = "random"

    rows = []
    for name in [k for k in scores if not k.endswith("__label")]:
        s = scores[name]
        pooled = auroc(s, y)
        macro, ncase = case_macro_auroc(s, y, cases)
        row = {
            "dataset": dataset, "model": model, "pool": pool, "selector": name,
            "definition": scores[name + "__label"], "n_tokens": len(y),
            "n_cases": int(len(np.unique(cases))), "base_rate": base,
            "pooled_auroc": pooled, "case_macro_auroc": macro, "auroc_cases": ncase,
        }
        for label, k, frac in BUDGETS:
            hits, spend = per_case_spend(s, y, cases, k, frac)
            p, lo, hi = precision_ci(hits, spend, n_boot)
            row[f"precision_{label}"] = p
            row[f"precision_{label}_lo"] = lo
            row[f"precision_{label}_hi"] = hi
            row[f"lift_{label}"] = p / base if base > 0 else float("nan")
        rows.append(row)
    return rows


def write_latex_budget(results: pl.DataFrame, out: Path) -> None:
    """The main-text budget table: what one and eight explanations buy."""
    from token_analysis.table_colour import sequential_fill

    main = results.filter(pl.col("pool") == "all")
    names = {"opi": "OpenPromptInjection", "taboo": "Taboo organisms",
             "liars": "Liars' Bench", "tt": "Tensor Trust"}
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.1}",
         r"\caption{Precision at a budget of one and of eight explanations per transcript. "
         r"\emph{Base} is the on-task share of all positions, which is what choosing at "
         r"random obtains. \emph{Signal} is the strongest single signal, \emph{Ensemble} the "
         r"selected pair, and \emph{Structure} the ranker that uses only segment, chat role "
         r"and position. Shading runs from \(0\) to \(1\).}",
         r"\label{tab:budget}",
         r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
         r" & & & \multicolumn{2}{c}{Signal} & \multicolumn{2}{c}{Ensemble}"
         r" & \multicolumn{2}{c}{Structure} \\",
         r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         r"Dataset & Model & Base & \(1\) & \(8\) & \(1\) & \(8\) & \(1\) & \(8\) \\",
         r"\midrule"]
    previous = None
    for dataset in ("opi", "taboo", "liars", "tt"):
        for model in DATASET_MODELS[dataset]:
            cell = main.filter((pl.col("dataset") == dataset) & (pl.col("model") == model))
            if not cell.height:
                continue
            if previous is not None and dataset != previous:
                L.append(r"\addlinespace")
            head = names[dataset] if dataset != previous else ""
            previous = dataset
            base = cell.filter(pl.col("selector") == "random")["base_rate"][0]
            cols = [f"\\cellcolor[HTML]{{{sequential_fill(base, 0.0, 1.0)}}}\\({base:.3f}\\)"]
            for selector in ("single_best", "ensemble_best", "structure"):
                row = cell.filter(pl.col("selector") == selector)
                for column in ("precision_top1", "precision_top8"):
                    v = row[column][0]
                    cols.append(f"\\cellcolor[HTML]{{{sequential_fill(v, 0.0, 1.0)}}}\\({v:.3f}\\)")
            L.append(f"{head} & {model} & " + " & ".join(cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(L) + "\n")


def structural_filter(dataset: str, model: str, input_dir: Path) -> dict:
    """What discarding structural tokens costs and buys. A spike token is one
    whose activation norm is far above the transcript median, in the sense of
    Sun et al.; the first token of a sequence is one almost always."""
    df = segment_frame(pl.read_parquet(source_path(input_dir, dataset, model)), dataset)
    df = df.filter(pl.col("segment") != "trailer")
    y = df["on_task"].to_numpy().astype(int)
    first = (df["tok_idx"].to_numpy() == 0)
    tmpl = (df["analysis_region"].to_numpy() == "template")
    spike = (df["norm_ratio"].to_numpy() > 5.0)
    drop = first | tmpl | spike
    return {
        "dataset": dataset, "model": model, "n_tokens": len(y),
        "base_rate": float(y.mean()),
        "first_share": float(first.mean()), "first_on_task": float(y[first].mean()),
        "spike_share": float(spike.mean()),
        "spike_on_task": float(y[spike].mean()) if spike.any() else float("nan"),
        "template_share": float(tmpl.mean()),
        "template_on_task": float(y[tmpl].mean()) if tmpl.any() else float("nan"),
        "dropped_share": float(drop.mean()),
        "dropped_on_task": float(y[drop].mean()) if drop.any() else float("nan"),
        "kept_on_task": float(y[~drop].mean()) if (~drop).any() else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Report                                                                       #
# --------------------------------------------------------------------------- #
def _fmt(v: float, nd: int = 3) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def write_report(results: pl.DataFrame, filters: pl.DataFrame, out: Path) -> None:
    L = ["# What a token selector buys at a budget\n",
         "> Generated by `token_analysis/budget_eval.py` from",
         "> `paper_results/bridge/all_*.parquet`. No model, no judge, no GPU.",
         "> `precision@B` is the share of the explanations actually bought that the",
         "> judge called on-task. `lift` is that precision over the base rate, so 1.0",
         "> means the selector is worth nothing and 2.0 means it doubles the yield.",
         "> Intervals resample whole transcripts, 2000 times, seed 0.\n"]

    for pool in sorted(results["pool"].unique()):
        L.append(f"## Budget, {pool} tokens\n")
        L.append("| dataset | model | selector | definition | base | p@1 | p@8 | p@1% |"
                 " p@10% | lift@1% | lift@10% |")
        L.append("|" + "---|" * 11)
        sub = results.filter(pl.col("pool") == pool)
        for r in sub.sort("dataset", "model", "selector").iter_rows(named=True):
            L.append(
                f"| {r['dataset']} | {r['model']} | `{r['selector']}` | {r['definition']} |"
                f" {_fmt(r['base_rate'], 2)} | {_fmt(r['precision_top1'])} |"
                f" {_fmt(r['precision_top8'])} | {_fmt(r['precision_1pct'])} |"
                f" {_fmt(r['precision_10pct'])} | {_fmt(r['lift_1pct'], 2)} |"
                f" {_fmt(r['lift_10pct'], 2)} |")
        L.append("")

    L.append("## Pooled against within-transcript ranking\n")
    L.append("Pooled AUROC counts every pair of tokens in the benchmark, including")
    L.append("pairs from different transcripts. The budget is spent inside one")
    L.append("transcript, so case-macro AUROC is the quantity that matches the")
    L.append("decision. A large gap means the selector separates transcripts rather")
    L.append("than positions.\n")
    L.append("| dataset | model | pool | selector | pooled AUROC | case-macro AUROC | gap |")
    L.append("|" + "---|" * 7)
    for r in results.sort("dataset", "model", "pool", "selector").iter_rows(named=True):
        gap = r["pooled_auroc"] - r["case_macro_auroc"]
        L.append(f"| {r['dataset']} | {r['model']} | {r['pool']} | `{r['selector']}` |"
                 f" {_fmt(r['pooled_auroc'])} | {_fmt(r['case_macro_auroc'])} | {_fmt(gap)} |")
    L.append("")

    L.append("## Structural tokens\n")
    L.append("On-task rate at the first token of the transcript, at spike tokens")
    L.append("(activation norm above five times the transcript median) and at chat")
    L.append("template tokens, against the benchmark base rate.\n")
    L.append("| dataset | model | base | first tok | spike share | spike on-task |"
             " template share | template on-task | dropped share | kept on-task |")
    L.append("|" + "---|" * 10)
    for r in filters.sort("dataset", "model").iter_rows(named=True):
        L.append(f"| {r['dataset']} | {r['model']} | {_fmt(r['base_rate'], 2)} |"
                 f" {_fmt(r['first_on_task'])} | {_fmt(r['spike_share'])} |"
                 f" {_fmt(r['spike_on_task'])} | {_fmt(r['template_share'], 2)} |"
                 f" {_fmt(r['template_on_task'])} | {_fmt(r['dropped_share'], 2)} |"
                 f" {_fmt(r['kept_on_task'])} |")
    L.append("")
    out.write_text("\n".join(L))


# --------------------------------------------------------------------------- #
# Self-test                                                                    #
# --------------------------------------------------------------------------- #
def selftest() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    r = pooled_midrank(np.array([10.0, 20.0, 30.0]))
    check("midrank spans [0,1]", np.allclose(r, [0.0, 0.5, 1.0]))
    r = pooled_midrank(np.array([1.0, np.nan, 2.0]))
    check("midrank keeps NaN missing", np.isnan(r[1]) and np.allclose(r[[0, 2]], [0.0, 1.0]))
    check("midrank of one finite value is 0.5",
          pooled_midrank(np.array([np.nan, 7.0]))[1] == 0.5)
    tied = pooled_midrank(np.array([5.0, 5.0, 9.0]))
    check("midrank averages ties", np.allclose(tied, [0.25, 0.25, 1.0]))

    # A budget of one per transcript, two transcripts, one on-task token each.
    s = np.array([9.0, 1.0, 0.0, 8.0, 2.0, 1.0])
    y = np.array([1, 0, 0, 1, 0, 0])
    c = np.array(["a", "a", "a", "b", "b", "b"])
    hits, spend = per_case_spend(s, y, c, 1, None)
    check("top-1 buys the on-task token in both cases",
          hits.tolist() == [1.0, 1.0] and spend.tolist() == [1.0, 1.0])
    p, lo, hi = precision_ci(hits, spend, n_boot=200)
    check("perfect selector has precision 1.0", p == 1.0 and lo == 1.0 and hi == 1.0)
    hits, spend = per_case_spend(-s, y, c, 1, None)
    check("reversed selector buys nothing", hits.sum() == 0.0)

    # A fractional budget rounds up, and never exceeds the finite token count.
    hits, spend = per_case_spend(s, y, c, None, 0.10)
    check("1-in-10 budget of three tokens is one token", spend.tolist() == [1.0, 1.0])
    s_missing = np.array([np.nan, np.nan, 1.0, 8.0, 2.0, 1.0])
    hits, spend = per_case_spend(s_missing, y, c, 8, None)
    check("budget is capped by the tokens that carry a score", spend.tolist() == [1.0, 3.0])

    # Case-macro AUROC ignores transcripts carrying a single class.
    m, n = case_macro_auroc(s, y, c)
    check("case-macro AUROC of a perfect ranker is 1", m == 1.0 and n == 2)
    y_one = np.array([1, 1, 1, 1, 0, 0])
    m, n = case_macro_auroc(s, y_one, c)
    check("single-class transcripts are skipped", n == 1)

    # A pooled score can look strong while ordering nothing inside a transcript.
    s2 = np.array([5.0, 6.0, 7.0, 0.0, 1.0, 2.0])
    y2 = np.array([1, 1, 1, 0, 0, 0])
    pooled = auroc(s2, y2)
    macro, _ = case_macro_auroc(s2, y2, c)
    check("pooled 1.0 can coexist with an undefined within-case ranking",
          pooled == 1.0 and np.isnan(macro))

    # Segmentation on a synthetic transcript: input, five boundary, output.
    frame = pl.DataFrame({
        "case_id": ["c"] * 12,
        "tok_idx": list(range(12)),
        "region": (["template", "system", "user", "user"]
                   + ["template"] * 5 + ["assistant"] * 2 + ["template"]),
        "on_task": [0] * 12,
    })
    seg = segment_frame(frame, "opi")
    check("input ends at the last content token before the boundary",
          seg["input_length"][0] == 4)
    check("boundary is the trailing template run", seg["boundary_length"][0] == 5)
    check("segments are assigned in order",
          seg["segment"].to_list() == ["input"] * 4 + ["boundary"] * 5
          + ["output"] * 2 + ["trailer"])
    check("boundary ordinals restart at zero",
          seg.filter(pl.col("segment") == "boundary")["segment_position"].to_list()
          == [0, 1, 2, 3, 4])

    # Folds keep a transcript whole.
    ids = np.array(["a"] * 5 + ["b"] * 5 + ["c"] * 5)
    folds = _case_folds(ids, 3, 0)
    check("a transcript never spans two folds",
          all(len(set(folds[ids == c])) == 1 for c in "abc"))

    # The two free baselines are nested: structure sees everything position
    # sees, plus the chat template.
    pos_x = position_features(seg, "position")
    str_x = position_features(seg, "structure")
    check("structure sees more than position", str_x.shape[1] > pos_x.shape[1])
    check("structure keeps the position columns",
          np.allclose(str_x[:, : pos_x.shape[1]], pos_x))

    # The free baselines must not see a signal or the judge's answer. Scramble
    # both and confirm the feature matrix does not move.
    rng2 = np.random.default_rng(1)
    n = seg.height
    poisoned = seg.with_columns(
        pl.Series("on_task", rng2.integers(0, 2, n)),
        pl.Series("sink_drain", rng2.normal(size=n)),
        pl.Series("dominant_mass", rng2.normal(size=n)),
        pl.Series("norm_ratio", rng2.normal(size=n)),
    )
    check("free baselines ignore the signals and the label",
          np.allclose(position_features(poisoned, "structure"), str_x))

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

    ens_best = load_auroc_ensemble_specs(AUROC_ENSEMBLE_SELECTION_PATH)
    ens_shared = load_auroc_ensemble_specs(AUROC_ENSEMBLE_SHARED_SELECTION_PATH)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows, filters = [], []
    for dataset in args.datasets:
        for model in DATASET_MODELS[dataset]:
            if args.models and model not in args.models:
                continue
            print(f"{dataset}/{MODEL_INFO[model]['name']}", flush=True)
            filters.append(structural_filter(dataset, model, args.input_dir))
            for pool in args.pool:
                rows += evaluate_cell(dataset, model, pool, ens_best, ens_shared,
                                      args.input_dir, args.n_boot)
                print(f"    {pool} done", flush=True)

    results = pl.DataFrame(rows)
    filter_frame = pl.DataFrame(filters)
    results.write_parquet(args.out_dir / "budget_results.parquet")
    filter_frame.write_parquet(args.out_dir / "structural_filter.parquet")
    write_report(results, filter_frame, args.out_dir / "budget_eval.md")
    if "all" in args.pool:
        write_latex_budget(results, args.out_dir / "budget_table.tex")
    print(f"\nwrote {args.out_dir}/budget_eval.md and two parquets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
