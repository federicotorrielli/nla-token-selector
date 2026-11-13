"""token_analysis/review_eval.py — the CPU experiments requested by the review.

  e1  nested model: structure, signals, structure + signals, in one model family
      (logistic regression, and boosted trees that can learn role x signal
      interactions). Answers "do the signals add anything beyond chat structure?"
  e3  structure under a moved threat, with no new data: across datasets with the
      same threat, across chat templates (models), across OPI attack strategies,
      across Liars' Bench subsets.
  e5  candidate selection by a budget objective instead of pooled AUROC.
  e6  absolute audit success at a 5% budget, for at least 1, 2 and 3 on-task.

Every learned score is out of fold: five folds over whole transcripts, one fold
map per cell shared by every score, signal ranks through the training CDF.
Intervals resample whole transcripts, 2000 times, seed 0. `--labels second`
swaps the DeepSeek labels for the second evaluator's (bridge_judge_second.py).

Usage:
    uv run python token_analysis/review_eval.py --selftest
    uv run python token_analysis/review_eval.py e1 [--datasets opi] [--models q7]
    uv run python token_analysis/review_eval.py e3 | e5 | e6
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from token_analysis.budget_eval import position_features, segment_frame  # noqa: E402
from token_analysis.common import (  # noqa: E402
    CANDIDATE_METRICS,
    DATASET_MODELS,
    LIARS_SUBDATASETS,
    source_path,
)
from token_analysis.selection_methods.auroc_ensemble.select_metrics import (  # noqa: E402
    _stable_fold,
    _train_mapping,
    candidate_definitions,
)
from token_analysis.selection_methods.statistics import auroc  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "review_eval"
# results/bridge, not paper_results/bridge: the duplicate key needs the explanations
BRIDGE_DIR = Path(__file__).resolve().parents[1] / "results" / "bridge"
SECOND_DIR = BRIDGE_DIR / "second_eval"
N_FOLDS, N_BOOT, SEED = 5, 2000, 0
# (name, k, fraction): the paper's four budgets plus the 5% headline budget
BUDGETS = (("top1", 1, None), ("top8", 8, None), ("1pct", None, 0.01),
           ("5pct", None, 0.05), ("10pct", None, 0.10))
SUCCESS_THRESHOLDS = (1, 2, 3)
OPI_STRATEGIES = ("naive", "escape", "fake_comp", "ignore", "combine")


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class Cell:
    dataset: str
    model: str
    df: pl.DataFrame
    y: np.ndarray            # on-task label per position
    case: np.ndarray         # integer case code, rows grouped by case
    case_ids: np.ndarray     # case_id string per unique case code
    tok: np.ndarray          # tok_idx, for position tie breaking
    folds: np.ndarray        # fold per position
    dup: np.ndarray          # hash of (tok_idx, signals): equal across shared prefixes

    def split(self, fold: int) -> tuple[np.ndarray, np.ndarray]:
        """Train and test masks. Transcripts sharing a prefix put identical
        positions (same activation, explanation and label) on both sides of a
        transcript split, so training drops every position duplicated in the test fold."""
        te = self.folds == fold
        tr = ~te & ~np.isin(self.dup, self.dup[te])
        return tr, te


def subdataset(case_id: str) -> str:
    return next((n for n in LIARS_SUBDATASETS if case_id.startswith(n + "_")), "all")


def opi_strategy(case_id: str) -> str:
    """`sst2__sst2__fake_comp_3` -> `fake_comp`."""
    tail = case_id.split("__")[-1]
    return tail.rsplit("_", 1)[0]


def fold_map(case_ids: list[str], n_folds: int = N_FOLDS, seed: int = SEED) -> dict[str, int]:
    """Balanced folds inside Liars subdataset strata, ordered by a stable hash of
    the case id, so every model of a dataset gets the same split."""
    strata: dict[str, list[str]] = {}
    for c in sorted(set(case_ids)):
        strata.setdefault(subdataset(c), []).append(c)
    out = {}
    for group in strata.values():
        group.sort(key=lambda c: _stable_fold(c, seed, 2**16))
        out.update({c: i % n_folds for i, c in enumerate(group)})
    return out


def load_cell(dataset: str, model: str, labels: str = "primary",
              input_dir: Path = BRIDGE_DIR) -> Cell:
    df = pl.read_parquet(source_path(input_dir, dataset, model))
    if labels == "second":
        second = pl.read_parquet(SECOND_DIR / f"{dataset}_{model}.parquet",
                                 columns=["position_id", "on_task2"])
        df = (df.join(second, on="position_id", how="left")
              .with_columns(pl.col("on_task2").fill_null(0).cast(pl.Int64).alias("on_task"))
              .drop("on_task2"))
    df = segment_frame(df, dataset).filter(pl.col("segment") != "trailer")
    df = df.sort("case_id", "tok_idx")
    ids = df["case_id"].to_numpy().astype(str)
    uniq, code = np.unique(ids, return_inverse=True)
    fm = fold_map(list(uniq))
    # A shared prefix gives the same greedy explanation at the same index. Its
    # signals can differ in the last float digits, so they cannot be the key.
    dup = df.select(pl.struct("tok_idx", "explanation").hash(SEED))[:, 0].to_numpy()
    return Cell(dataset, model, df, df["on_task"].to_numpy().astype(np.int8), code, uniq,
                df["tok_idx"].to_numpy(), np.array([fm[c] for c in uniq])[code], dup)


# --------------------------------------------------------------------------- #
# Budget arithmetic, vectorized over transcripts                              #
# --------------------------------------------------------------------------- #
def within_case_rank(scores: np.ndarray, case: np.ndarray, tok: np.ndarray) -> np.ndarray:
    """0 for the highest score of each transcript. NaN scores rank last; ties
    break by earliest position, as in budget_eval.per_case_spend."""
    key = np.where(np.isfinite(scores), -scores, np.inf)
    order = np.lexsort((tok, key, case))
    starts = np.r_[0, np.flatnonzero(np.diff(case[order])) + 1]
    first = np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order)) - first
    return rank


def spend_hits(scores, y, case, tok, k=None, frac=None):
    """Per transcript: explanations bought and on-task explanations among them."""
    n_case = int(case.max()) + 1
    finite = np.isfinite(scores)
    n_fin = np.bincount(case, finite, n_case)
    budget = np.full(n_case, k, float) if k is not None else np.maximum(1, np.ceil(frac * n_fin))
    budget = np.minimum(budget, n_fin)
    chosen = finite & (within_case_rank(scores, case, tok) < budget[case])
    return budget, np.bincount(case, y * chosen, n_case)


def boot_idx(n_case: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_case, (n_boot, n_case))


def ratio_ci(num: np.ndarray, den: np.ndarray, idx: np.ndarray) -> tuple[float, float, float]:
    point = num.sum() / max(den.sum(), 1e-12)
    boots = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return float(point), *map(float, np.percentile(boots, [2.5, 97.5]))


def case_macro_auroc(scores, y, case) -> np.ndarray:
    """AUROC per transcript (NaN where one class is missing)."""
    out = np.full(int(case.max()) + 1, np.nan)
    bounds = np.r_[0, np.flatnonzero(np.diff(case)) + 1, len(case)]
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        out[case[lo]] = auroc(scores[lo:hi], y[lo:hi])
    return out


def score_report(cell: Cell, scores: np.ndarray, idx: np.ndarray) -> dict:
    """Every budget metric of one score, with transcript bootstrap intervals."""
    row = {"pooled_auroc": auroc(scores, cell.y)}
    per_case_auc = case_macro_auroc(scores, cell.y, cell.case)
    row["case_macro_auroc"] = float(np.nanmean(per_case_auc))
    for name, k, frac in BUDGETS:
        spend, hits = spend_hits(scores, cell.y, cell.case, cell.tok, k, frac)
        p, lo, hi = ratio_ci(hits, spend, idx)
        row |= {f"p_{name}": p, f"p_{name}_lo": lo, f"p_{name}_hi": hi}
        if name == "5pct":
            for t in SUCCESS_THRESHOLDS:
                ok = (hits >= t).astype(float)
                s, slo, shi = ratio_ci(ok, np.ones_like(ok), idx)
                row |= {f"success{t}_5pct": s, f"success{t}_5pct_lo": slo,
                        f"success{t}_5pct_hi": shi}
    return row


def paired_delta(cell: Cell, a: np.ndarray, b: np.ndarray, idx: np.ndarray) -> dict:
    """Score a minus score b on the same transcripts and the same resamples."""
    row = {}
    for name, k, frac in BUDGETS:
        sa, ha = spend_hits(a, cell.y, cell.case, cell.tok, k, frac)
        sb, hb = spend_hits(b, cell.y, cell.case, cell.tok, k, frac)
        point = ha.sum() / sa.sum() - hb.sum() / sb.sum()
        boots = ha[idx].sum(1) / sa[idx].sum(1) - hb[idx].sum(1) / sb[idx].sum(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        row |= {f"d_{name}": point, f"d_{name}_lo": lo, f"d_{name}_hi": hi}
        if name == "5pct":
            d = (ha >= 1).astype(float) - (hb >= 1).astype(float)
            lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
            row |= {"d_success1_5pct": d.mean(), "d_success1_5pct_lo": lo,
                    "d_success1_5pct_hi": hi}
    da = case_macro_auroc(a, cell.y, cell.case) - case_macro_auroc(b, cell.y, cell.case)
    ok = np.isfinite(da)
    boots = np.array([np.nanmean(da[i]) for i in idx[:, :]])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return row | {"d_case_macro_auroc": float(da[ok].mean()), "d_case_macro_auroc_lo": lo,
                  "d_case_macro_auroc_hi": hi}


# --------------------------------------------------------------------------- #
# Features and models                                                         #
# --------------------------------------------------------------------------- #
def signal_matrix(train_raw: np.ndarray, test_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signals as training-CDF ranks; a missing value becomes 0.5 plus an indicator."""
    tr_cols, te_cols = [], []
    for j in range(train_raw.shape[1]):
        tr, te = _train_mapping(train_raw[:, j], test_raw[:, j])
        miss_tr, miss_te = ~np.isfinite(tr), ~np.isfinite(te)
        tr_cols.append(np.where(miss_tr, 0.5, tr))
        te_cols.append(np.where(miss_te, 0.5, te))
        if miss_tr.any() or miss_te.any():
            tr_cols.append(miss_tr.astype(float))
            te_cols.append(miss_te.astype(float))
    return np.column_stack(tr_cols), np.column_stack(te_cols)


def raw_signals(df: pl.DataFrame) -> np.ndarray:
    return np.column_stack([df[m].to_numpy().astype(float) for m in CANDIDATE_METRICS])


def fit_predict(family: str, x_tr, y_tr, x_te) -> np.ndarray:
    if family == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(x_tr)
        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(scaler.transform(x_tr), y_tr)
        return model.decision_function(scaler.transform(x_te))
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                           max_leaf_nodes=31, early_stopping=False,
                                           random_state=SEED)
    model.fit(x_tr, y_tr)
    return model.predict_proba(x_te)[:, 1]


def feature_sets(family: str, s_tr, s_te, g_raw_tr, g_raw_te):
    """structure / signals / both. Trees read raw signals (NaN native); the
    logistic model reads training-CDF ranks."""
    if family == "logistic":
        g_tr, g_te = signal_matrix(g_raw_tr, g_raw_te)
    else:
        g_tr, g_te = g_raw_tr, g_raw_te
    return {"structure": (s_tr, s_te), "signals": (g_tr, g_te),
            "structure+signals": (np.hstack([s_tr, g_tr]), np.hstack([s_te, g_te]))}


def out_of_fold(cell: Cell, family: str) -> dict[str, np.ndarray]:
    s = position_features(cell.df, "structure")
    g = raw_signals(cell.df)
    out = {name: np.full(len(cell.y), np.nan)
           for name in ("structure", "signals", "structure+signals")}
    for f in range(N_FOLDS):
        tr, te = cell.split(f)
        for name, (x_tr, x_te) in feature_sets(family, s[tr], s[te], g[tr], g[te]).items():
            out[name][te] = fit_predict(family, x_tr, cell.y[tr], x_te)
    return out


# --------------------------------------------------------------------------- #
# E1: nested model                                                            #
# --------------------------------------------------------------------------- #
def run_e1(cells: list[tuple[str, str]], labels: str) -> pl.DataFrame:
    rows, deltas = [], []
    for dataset, model in cells:
        t0 = time.time()
        cell = load_cell(dataset, model, labels)
        idx = boot_idx(len(cell.case_ids))
        for family in ("logistic", "trees"):
            scores = out_of_fold(cell, family)
            for name, sc in scores.items():
                rows.append({"dataset": dataset, "model": model, "family": family,
                             "features": name, "base_rate": float(cell.y.mean()),
                             **score_report(cell, sc, idx)})
            deltas.append({"dataset": dataset, "model": model, "family": family,
                           "contrast": "structure+signals minus structure",
                           **paired_delta(cell, scores["structure+signals"],
                                          scores["structure"], idx)})
        print(f"e1 {dataset}/{model} {time.time() - t0:.0f}s", flush=True)
    return pl.DataFrame(rows), pl.DataFrame(deltas)


# --------------------------------------------------------------------------- #
# E3: structure when the threat moves                                         #
# --------------------------------------------------------------------------- #
CROSS_DATASET = (("opi", "tt"), ("tt", "opi"), ("liars", "taboo"), ("taboo", "liars"))


def fit_full(cell: Cell, family: str, which: str):
    """Fit on every transcript of a source cell. Returns a scorer for new cells."""
    s = position_features(cell.df, "structure")
    g = raw_signals(cell.df)

    def score(target: Cell) -> np.ndarray:
        sets = feature_sets(family, s, position_features(target.df, "structure"),
                            g, raw_signals(target.df))
        x_tr, x_te = sets[which]
        return fit_predict(family, x_tr, cell.y, x_te)
    return score


def best_single(cell: Cell, mask: np.ndarray | None = None) -> tuple[str, float]:
    """The signal with the largest direction-adjusted pooled AUROC and its sign."""
    mask = np.ones(len(cell.y), bool) if mask is None else mask
    best = max(CANDIDATE_METRICS,
               key=lambda m: abs(auroc(cell.df[m].to_numpy()[mask], cell.y[mask]) - 0.5))
    a = auroc(cell.df[best].to_numpy()[mask], cell.y[mask])
    return best, (1.0 if a >= 0.5 else -1.0)


def e3_rows(kind, source, target: Cell, scorers: dict[str, np.ndarray], idx) -> list[dict]:
    return [{"shift": kind, "source": source, "dataset": target.dataset,
             "model": target.model, "score": name, "base_rate": float(target.y.mean()),
             **score_report(target, sc, idx)} for name, sc in scorers.items()]


def transfer_scores(src: Cell, tgt: Cell) -> dict[str, np.ndarray]:
    metric, sign = best_single(src)
    return {"structure": fit_full(src, "logistic", "structure")(tgt),
            "structure+signals": fit_full(src, "logistic", "structure+signals")(tgt),
            f"single ({metric})": sign * tgt.df[metric].to_numpy().astype(float)}


def reference_scores(cell: Cell) -> dict[str, np.ndarray]:
    """The same three scores fitted in-domain, out of fold."""
    oof = out_of_fold(cell, "logistic")
    single = np.full(len(cell.y), np.nan)
    for f in range(N_FOLDS):
        tr, te = cell.split(f)
        metric, sign = best_single(cell, tr)
        single[te] = sign * cell.df[metric].to_numpy().astype(float)[te]
    return {"structure": oof["structure"], "structure+signals": oof["structure+signals"],
            "single (in-fold)": single}


def run_e3(labels: str) -> pl.DataFrame:
    rows = []
    cache: dict[tuple[str, str], Cell] = {}

    def get(d, m):
        if (d, m) not in cache:
            cache[(d, m)] = load_cell(d, m, labels)
        return cache[(d, m)]

    for dataset, models in DATASET_MODELS.items():
        for model in models:
            tgt = get(dataset, model)
            idx = boot_idx(len(tgt.case_ids))
            rows += e3_rows("none (in-domain)", f"{dataset}/{model}", tgt,
                            reference_scores(tgt), idx)
            # across chat templates: every other model of the same dataset
            for other in models:
                if other != model:
                    rows += e3_rows("template", f"{dataset}/{other}", tgt,
                                    transfer_scores(get(dataset, other), tgt), idx)
            # across datasets with the same threat, same model (same template)
            for a, b in CROSS_DATASET:
                if b == dataset and model in DATASET_MODELS[a]:
                    rows += e3_rows("dataset", f"{a}/{model}", tgt,
                                    transfer_scores(get(a, model), tgt), idx)
            print(f"e3 {dataset}/{model}", flush=True)
        # the four datasets hold ~3 GB each for Liars; free the finished one
        for key in [k for k in cache if k[0] == dataset]:
            cache.pop(key)

    rows += leave_group_out(get, "opi", opi_strategy, OPI_STRATEGIES, "attack strategy")
    rows += leave_group_out(get, "liars", subdataset, LIARS_SUBDATASETS, "Liars subset")
    return pl.DataFrame(rows)


def leave_group_out(get, dataset, group_of, groups, kind) -> list[dict]:
    """Fit on every group but one, score the group left out; next to the same
    group scored in-domain (five folds over all groups)."""
    rows = []
    for model in DATASET_MODELS[dataset]:
        cell = get(dataset, model)
        grp = np.array([group_of(c) for c in cell.case_ids])[cell.case]
        ref = reference_scores(cell)
        for g in groups:
            test = grp == g
            if not test.any():
                continue
            sub = subset(cell, test)
            src = subset(cell, ~test & ~np.isin(cell.dup, cell.dup[test]))
            idx = boot_idx(len(sub.case_ids))
            rows += e3_rows(kind, f"all but {g}", sub, transfer_scores(src, sub), idx)
            rows += e3_rows(f"{kind} (in-domain)", g, sub,
                            {k: v[test] for k, v in ref.items()}, idx)
        print(f"e3 {kind} {dataset}/{model}", flush=True)
    return rows


def subset(cell: Cell, mask: np.ndarray) -> Cell:
    df = cell.df.filter(pl.Series(mask))
    ids = df["case_id"].to_numpy().astype(str)
    uniq, code = np.unique(ids, return_inverse=True)
    return Cell(cell.dataset, cell.model, df, cell.y[mask], code, uniq, cell.tok[mask],
                cell.folds[mask], cell.dup[mask])


# --------------------------------------------------------------------------- #
# E2: the injection moved inside the data (new B200 corpora)                  #
# --------------------------------------------------------------------------- #
MOVED = ("opistart", "opimiddle", "opihistory")


def localization(cell: Cell) -> dict:
    """On-task rate inside and outside the injected span, user message only."""
    user = cell.df["analysis_region"].to_numpy() == "user"
    span = cell.df["label"].to_numpy() == 1
    inside, outside = cell.y[user & span].mean(), cell.y[user & ~span].mean()
    return {"on_task_inside_span": float(inside), "on_task_outside_span": float(outside),
            "localization_ratio": float(inside / max(outside, 1e-12))}


def run_e2(models: list[str], labels: str) -> pl.DataFrame:
    rows = []
    for model in models:
        src = load_cell("opi", model, labels)
        metric, sign = best_single(src)
        scorers = {"structure": fit_full(src, "logistic", "structure"),
                   "structure+signals": fit_full(src, "logistic", "structure+signals"),
                   "structure+signals (trees)": fit_full(src, "trees", "structure+signals")}
        for version in ("opi", *MOVED):
            tgt = src if version == "opi" else load_cell(version, model, labels)
            idx = boot_idx(len(tgt.case_ids))
            loc = localization(tgt)
            # fitted on the original placement, applied unchanged
            if version != "opi":
                moved = {name: fit(tgt) for name, fit in scorers.items()}
                moved[f"single ({metric})"] = sign * tgt.df[metric].to_numpy().astype(float)
                rows += [{"fit_on": "original OPI", **r, **loc}
                         for r in e3_rows(version, "opi", tgt, moved, idx)]
            # refitted on the version itself, out of fold (the reachable ceiling)
            own = reference_scores(tgt) | {
                "structure+signals (trees)": out_of_fold(tgt, "trees")["structure+signals"]}
            rows += [{"fit_on": "same version", **r, **loc}
                     for r in e3_rows(version, version, tgt, own, idx)]
            print(f"e2 {version}/{model}", flush=True)
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- #
# E8: all 20 taboo words and the misaligned adapters (new B200 corpora)       #
# --------------------------------------------------------------------------- #
ORIGINAL_WORDS = ("moon", "ship", "snow")


def word_recovery(model: str) -> pl.DataFrame:
    """Per word: how often the explanation names the organism's own word, and how
    often another organism's word, over all positions and over the prompt
    positions whose text is the same for every organism (template and user)."""
    df = pl.read_parquet(BRIDGE_DIR / f"all_taboo20_{model}.parquet",
                         columns=["case_id", "region", "explanation", "on_task"])
    df = df.filter(pl.col("explanation").is_not_null()).with_columns(
        pl.col("case_id").str.split("__").list.first().alias("word"),
        pl.col("explanation").str.to_lowercase().alias("text"))
    words = df["word"].unique().to_list()
    names = {w: df["text"].str.contains(rf"\b{w}\b") for w in words}
    own = np.zeros(df.height, bool)
    other = np.zeros(df.height, bool)
    wcol = df["word"].to_numpy()
    for w, hit in names.items():
        h = hit.to_numpy()
        own |= (wcol == w) & h
        other |= (wcol != w) & h
    df = df.with_columns(pl.Series("own", own), pl.Series("other", other),
                         pl.col("region").is_in(["template", "user"]).alias("shared"))
    by = [pl.len().alias("positions"), pl.col("on_task").mean().alias("on_task"),
          pl.col("own").mean().alias("names_own_word"),
          pl.col("other").mean().alias("names_other_word")]
    whole = df.group_by("word").agg(*by).with_columns(pl.lit("all").alias("positions_kind"))
    shared = (df.filter("shared").group_by("word").agg(*by)
              .with_columns(pl.lit("shared prompt").alias("positions_kind")))
    return (pl.concat([whole, shared]).with_columns(
        pl.lit(model).alias("model"),
        pl.col("word").is_in(ORIGINAL_WORDS).alias("original_word")))


def run_e8(models: list[str], labels: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    recovery = pl.concat([word_recovery(m) for m in models])
    transfer = []
    for model in models:
        # selectors fitted on the paper's 3 words, applied to the 17 new ones
        src = load_cell("taboo", model, labels)
        new = load_cell("taboo20", model, labels)
        keep = ~np.isin(np.array([c.split("__")[0] for c in new.case_ids])[new.case],
                        ORIGINAL_WORDS)
        new = subset(new, keep)
        idx = boot_idx(len(new.case_ids))
        transfer += [{"fit_on": "3 original words", **r}
                     for r in e3_rows("new words", "taboo", new, transfer_scores(src, new), idx)]
        transfer += [{"fit_on": "same words", **r}
                     for r in e3_rows("new words", "taboo20", new, reference_scores(new), idx)]
        print(f"e8 {model}", flush=True)
    misalign = (pl.read_parquet(BRIDGE_DIR / "all_misalign_q7.parquet",
                                columns=["case_id", "region", "on_task"])
                .filter(pl.col("on_task").is_not_null())
                .with_columns(pl.col("case_id").str.split("__").list.first().alias("adapter"))
                .group_by("adapter", "region").agg(pl.len().alias("positions"),
                                                   pl.col("on_task").mean().alias("on_task"))
                .sort("adapter", "region"))
    return recovery, pl.DataFrame(transfer), misalign


# --------------------------------------------------------------------------- #
# E5: selection by a budget objective                                         #
# --------------------------------------------------------------------------- #
OBJECTIVES = ("pooled_auroc", "p_top8", "success1_5pct", "mean_precision")


def objective_values(score, y, case, tok) -> dict[str, float]:
    a = auroc(score, y)
    out = {"pooled_auroc": max(a, 1 - a) if np.isfinite(a) else float("nan")}
    precisions = []
    for name, k, frac in (("top1", 1, None), ("top8", 8, None), ("1pct", None, 0.01),
                          ("10pct", None, 0.10), ("5pct", None, 0.05)):
        spend, hits = spend_hits(score, y, case, tok, k, frac)
        p = hits.sum() / max(spend.sum(), 1e-12)
        if name != "5pct":
            precisions.append(p)
        if name == "top8":
            out["p_top8"] = p
        if name == "5pct":
            out["success1_5pct"] = float((hits >= 1).mean())
    out["mean_precision"] = float(np.mean(precisions))
    return out


def run_e5(cells: list[tuple[str, str]], labels: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    candidates = candidate_definitions(CANDIDATE_METRICS)
    rows, picks = [], []
    for dataset, model in cells:
        t0 = time.time()
        cell = load_cell(dataset, model, labels)
        g = raw_signals(cell.df)
        oof = {o: np.full(len(cell.y), np.nan) for o in OBJECTIVES}
        for f in range(N_FOLDS):
            tr, te = cell.split(f)
            ranks_tr, ranks_te = {}, {}
            for j, m in enumerate(CANDIDATE_METRICS):
                ranks_tr[m], ranks_te[m] = _train_mapping(g[tr, j], g[te, j])
            # component direction from the training AUROC, as the paper's selector
            flip = {m: auroc(ranks_tr[m], cell.y[tr]) < 0.5 for m in CANDIDATE_METRICS}
            orient = {m: (1 - r if flip[m] else r) for m, r in ranks_tr.items()}
            orient_te = {m: (1 - r if flip[m] else r) for m, r in ranks_te.items()}
            case_tr = np.unique(cell.case[tr], return_inverse=True)[1]
            scored = []
            for cand in candidates:
                s = orient[cand.component_1] * cand.component_1_weight
                if cand.component_2:
                    s = s + orient[cand.component_2] * cand.component_2_weight
                a = auroc(s, cell.y[tr])
                sign = -1.0 if np.isfinite(a) and a < 0.5 else 1.0
                vals = objective_values(sign * s, cell.y[tr], case_tr, cell.tok[tr])
                scored.append((cand, sign, vals))
            for o in OBJECTIVES:
                # ties: single before pair, then candidate id (the paper's order)
                cand, sign, vals = max(scored, key=lambda t: (
                    round(t[2][o], 12), t[0].candidate_type == "single", t[0].candidate_id))
                s = orient_te[cand.component_1] * cand.component_1_weight
                if cand.component_2:
                    s = s + orient_te[cand.component_2] * cand.component_2_weight
                oof[o][te] = sign * s
                picks.append({"dataset": dataset, "model": model, "fold": f, "objective": o,
                              "candidate": cand.candidate_id, "train_value": vals[o]})
        idx = boot_idx(len(cell.case_ids))
        structure = out_of_fold_structure(cell)
        for o, sc in oof.items():
            rows.append({"dataset": dataset, "model": model, "selected_by": o,
                         "base_rate": float(cell.y.mean()), **score_report(cell, sc, idx),
                         **{f"vs_structure_{k}": v for k, v in
                            paired_delta(cell, sc, structure, idx).items()}})
        rows.append({"dataset": dataset, "model": model, "selected_by": "structure",
                     "base_rate": float(cell.y.mean()), **score_report(cell, structure, idx)})
        print(f"e5 {dataset}/{model} {time.time() - t0:.0f}s", flush=True)
    return pl.DataFrame(rows), pl.DataFrame(picks)


def out_of_fold_structure(cell: Cell) -> np.ndarray:
    s = position_features(cell.df, "structure")
    out = np.full(len(cell.y), np.nan)
    for f in range(N_FOLDS):
        tr, te = cell.split(f)
        out[te] = fit_predict("logistic", s[tr], cell.y[tr], s[te])
    return out


# --------------------------------------------------------------------------- #
# E6: absolute audit success                                                  #
# --------------------------------------------------------------------------- #
def run_e6(cells: list[tuple[str, str]], labels: str) -> pl.DataFrame:
    from scipy.stats import hypergeom

    from token_analysis.budget_eval import ensemble_score
    from token_analysis.common import (
        AUROC_ENSEMBLE_SHARED_SELECTION_PATH,
        load_auroc_ensemble_specs,
    )

    specs = load_auroc_ensemble_specs(AUROC_ENSEMBLE_SHARED_SELECTION_PATH)
    rows = []
    for dataset, model in cells:
        cell = load_cell(dataset, model, labels)
        idx = boot_idx(len(cell.case_ids))
        n = np.bincount(cell.case)
        m = np.bincount(cell.case, cell.y)
        k = np.maximum(1, np.ceil(0.05 * n))
        scores = {"structure": out_of_fold_structure(cell),
                  "ensemble (paper)": ensemble_score(cell.df, specs[(dataset, model)])}
        for t in SUCCESS_THRESHOLDS:
            ceiling = (m >= t).astype(float)
            per = {"exhaustive": ceiling,
                   "random": hypergeom.sf(t - 1, n, m, k),
                   "oracle": ((m >= t) & (k >= t)).astype(float)}
            for name, sc in scores.items():
                _, hits = spend_hits(sc, cell.y, cell.case, cell.tok, None, 0.05)
                per[name] = (hits >= t).astype(float)
            for name, v in per.items():
                point = v.mean()
                lo, hi = np.percentile(v[idx].mean(1), [2.5, 97.5])
                ratio = v.sum() / max(ceiling.sum(), 1e-12)
                rlo, rhi = np.percentile(v[idx].sum(1) / np.maximum(ceiling[idx].sum(1), 1e-12),
                                         [2.5, 97.5])
                rows.append({"dataset": dataset, "model": model, "threshold": t,
                             "selector": name, "absolute": point, "absolute_lo": lo,
                             "absolute_hi": hi, "ratio": ratio, "ratio_lo": rlo,
                             "ratio_hi": rhi, "median_length": float(np.median(n))})
        print(f"e6 {dataset}/{model}", flush=True)
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Self-test                                                                   #
# --------------------------------------------------------------------------- #
def selftest() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    case = np.array([0, 0, 0, 1, 1, 1])
    tok = np.array([0, 1, 2, 0, 1, 2])
    y = np.array([1, 0, 0, 0, 0, 1], np.int8)
    s = np.array([9.0, 1.0, 0.0, 0.0, 1.0, 8.0])
    check("rank 0 is the top score", within_case_rank(s, case, tok).tolist() == [0, 1, 2, 2, 1, 0])
    tie = np.zeros(6)
    check("ties break by earliest position",
          within_case_rank(tie, case, tok).tolist() == [0, 1, 2, 0, 1, 2])
    nan = np.array([np.nan, 5.0, 1.0, 1.0, 2.0, 3.0])
    check("NaN ranks last", within_case_rank(nan, case, tok)[0] == 2)
    spend, hits = spend_hits(s, y, case, tok, k=1)
    check("top-1 finds both on-task positions", hits.tolist() == [1, 1] and spend.tolist() == [1, 1])
    spend, hits = spend_hits(s, y, case, tok, frac=0.05)
    check("5% of three positions is one", spend.tolist() == [1, 1])
    spend, hits = spend_hits(nan, y, case, tok, k=8)
    check("budget capped by finite positions", spend.tolist() == [2, 3])
    check("strategy parser", opi_strategy("sst2__sst2__fake_comp_3") == "fake_comp"
          and opi_strategy("a__b__naive_0") == "naive")
    fm = fold_map([f"c{i}" for i in range(50)])
    check("folds balanced", sorted(np.bincount(list(fm.values()))) == [10] * 5)
    a = np.array([1.0, np.nan, 3.0])
    tr, te = signal_matrix(np.column_stack([a]), np.column_stack([a]))
    check("missing signal gets 0.5 and an indicator", tr.shape == (3, 2) and tr[1, 0] == 0.5
          and tr[1, 1] == 1.0)
    toy = Cell("x", "x", pl.DataFrame(), y, case, np.array(["a", "b"]), tok,
               np.array([0, 0, 0, 1, 1, 1]), np.array([7, 8, 9, 7, 5, 6]))
    tr, te = toy.split(1)
    check("a position duplicated in the test fold leaves training",
          tr.tolist() == [False, True, True, False, False, False])
    cm = case_macro_auroc(s, y, case)
    check("case AUROC", np.allclose(cm, [1.0, 1.0]))
    print(f"\nselftest: {'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", nargs="?", choices=("e1", "e2", "e3", "e5", "e6", "e8"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_MODELS))
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--labels", choices=("primary", "second"), default="primary")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.experiment is None:
        ap.error("name an experiment or pass --selftest")
    cells = [(d, m) for d in args.datasets for m in DATASET_MODELS[d]
             if args.models is None or m in args.models]
    out = OUT_DIR / args.labels
    out.mkdir(parents=True, exist_ok=True)
    tag = "" if len(cells) == 14 else "_" + "_".join(f"{d}-{m}" for d, m in cells)
    if args.experiment == "e1":
        rows, deltas = run_e1(cells, args.labels)
        rows.write_parquet(out / f"e1_scores{tag}.parquet")
        deltas.write_parquet(out / f"e1_deltas{tag}.parquet")
    elif args.experiment == "e3":
        run_e3(args.labels).write_parquet(out / "e3_transfer.parquet")
    elif args.experiment in ("e2", "e8"):
        models = args.models or list(DATASET_MODELS["opi"])
        mtag = "" if len(models) == 4 else "_" + "_".join(models)
        if args.experiment == "e2":
            run_e2(models, args.labels).write_parquet(out / f"e2_moved{mtag}.parquet")
        else:
            recovery, transfer, misalign = run_e8(models, args.labels)
            recovery.write_parquet(out / f"e8_recovery{mtag}.parquet")
            transfer.write_parquet(out / f"e8_transfer{mtag}.parquet")
            misalign.write_parquet(out / "e8_misalign.parquet")
    elif args.experiment == "e5":
        rows, picks = run_e5(cells, args.labels)
        rows.write_parquet(out / f"e5_scores{tag}.parquet")
        picks.write_parquet(out / f"e5_picks{tag}.parquet")
    else:
        run_e6(cells, args.labels).write_parquet(out / f"e6_success{tag}.parquet")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
