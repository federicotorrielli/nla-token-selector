"""Select exhaustive pairwise rank ensembles by pooled token AUROC.

The primary point estimate uses every stored token and is intentionally
comparable to the repository's original pooled-AUROC selection.  Complete
cases are also held out to diagnose selection optimism.  Segment summaries use
the same input/boundary/output definition as the positional analysis and never
include Liars post-response trailers.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
from scipy.stats import rankdata
from tqdm.auto import tqdm

try:
    from ...common import (
        ANALYSIS_DIR,
        CANDIDATE_METRICS,
        DATASET_MODELS,
        DEFAULT_INPUT_DIR,
        LIARS_SUBDATASETS,
        METRIC_COST_ORDER,
        MODEL_INFO,
        OPI_ONLY_METRICS,
        selected_pairs,
        source_path,
    )
    from ..statistics import auroc
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from token_analysis.common import (
        ANALYSIS_DIR,
        CANDIDATE_METRICS,
        DATASET_MODELS,
        DEFAULT_INPUT_DIR,
        LIARS_SUBDATASETS,
        METRIC_COST_ORDER,
        MODEL_INFO,
        OPI_ONLY_METRICS,
        selected_pairs,
        source_path,
    )
    from token_analysis.selection_methods.statistics import auroc


METHOD_DIR = Path(__file__).resolve().parent
BEST_DIR = METHOD_DIR / "results" / "model_best"
SHARED_DIR = METHOD_DIR / "results" / "dataset_shared"
SENSITIVITY_DIR = METHOD_DIR / "sensitivity_attn_rollout"
WEIGHTS = (0.25, 0.50, 0.75)
SEGMENTS = ("input", "boundary", "output")
TIE_TOLERANCE = 1e-12
VALIDATION_PROCEDURES = ("all_candidates", "original_only")
LOMO_PROTOCOLS = ("label_free", "target_calibrated")


def _stage(message: str) -> None:
    """Emit a timestamped milestone without corrupting active tqdm bars."""
    tqdm.write(f"[{time.strftime('%H:%M:%S')}] [auroc-ensemble] {message}")


@dataclass(frozen=True)
class Candidate:
    """A single metric or a fixed convex combination of two metrics."""

    candidate_id: str
    candidate_type: str
    component_1: str
    component_1_weight: float
    component_2: str | None = None
    component_2_weight: float = 0.0

    @property
    def components(self) -> tuple[str, ...]:
        return (
            (self.component_1,)
            if self.component_2 is None
            else (self.component_1, self.component_2)
        )


@dataclass
class PoolContext:
    """Normalized, direction-aligned component values for one token pool."""

    aligned: dict[str, np.ndarray]
    directions: dict[str, str]
    raw_aurocs: dict[str, float]
    mask: np.ndarray


def candidate_definitions(metrics: Iterable[str]) -> list[Candidate]:
    """Return originals and all unique weighted metric pairs."""
    names = tuple(metrics)
    singles = [Candidate(f"single::{name}", "single", name, 1.0) for name in names]
    mixes = []
    for first, second in itertools.combinations(sorted(names), 2):
        for weight in WEIGHTS:
            mixes.append(
                Candidate(
                    f"mix::{first}@{weight:.2f}+{second}@{1-weight:.2f}",
                    "ensemble",
                    first,
                    weight,
                    second,
                    1 - weight,
                )
            )
    return singles + mixes


def fractional_midranks(values: np.ndarray) -> np.ndarray:
    """Finite pooled midranks in [0, 1], preserving non-finite values as NaN."""
    values = np.asarray(values, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    count = int(finite.sum())
    if count == 0:
        return output
    if count == 1:
        output[finite] = 0.5
        return output
    output[finite] = (rankdata(values[finite], method="average") - 1) / (count - 1)
    return output


def _subdataset(case_id: str) -> str:
    return next(
        (name for name in LIARS_SUBDATASETS if case_id.startswith(name + "_")),
        "all",
    )


def derive_segments(
    case_ids: np.ndarray,
    positions: np.ndarray,
    regions: np.ndarray,
    dataset: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce canonical segmentation and the two TT boundary corrections."""
    case_ids = np.asarray(case_ids).astype(str)
    positions = np.asarray(positions, dtype=np.int64)
    regions = np.asarray(regions).astype(str)
    segments = np.full(len(case_ids), "trailer", dtype=object)
    analysis_regions = regions.astype(object).copy()
    order = np.lexsort((positions, case_ids))
    sorted_cases = case_ids[order]
    starts = np.r_[0, np.flatnonzero(sorted_cases[1:] != sorted_cases[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    for start, end in zip(starts, ends, strict=True):
        indices = order[start:end]
        local_order = np.argsort(positions[indices], kind="stable")
        indices = indices[local_order]
        local_regions = regions[indices]
        assistant = np.flatnonzero(local_regions == "assistant")
        output_start = int(assistant[0]) if len(assistant) else len(indices)
        output_end = int(assistant[-1]) if len(assistant) else len(indices) - 1
        before = np.flatnonzero(
            (np.arange(len(indices)) < output_start) & (local_regions != "template")
        )
        source_input_end = int(before[-1]) if len(before) else -1
        source_boundary_length = output_start - source_input_end - 1
        corrected = dataset == "tt" and source_boundary_length > 5
        input_end = output_start - 6 if corrected else source_input_end
        segments[indices[: input_end + 1]] = "input"
        segments[indices[input_end + 1 : output_start]] = "boundary"
        if len(assistant):
            segments[indices[output_start : output_end + 1]] = "output"
        if corrected and input_end >= 0 and local_regions[input_end] == "template":
            analysis_regions[indices[input_end]] = "user"
    return segments.astype(str), analysis_regions.astype(str)


def _metric_context(
    values: dict[str, np.ndarray],
    labels: np.ndarray,
    pool_mask: np.ndarray,
) -> PoolContext:
    aligned: dict[str, np.ndarray] = {}
    directions: dict[str, str] = {}
    raw_aurocs: dict[str, float] = {}
    for metric, raw in values.items():
        selected = np.where(pool_mask, raw, np.nan)
        normalized = fractional_midranks(selected)
        value_auc = auroc(normalized, labels)
        direction = "higher" if not np.isfinite(value_auc) or value_auc >= 0.5 else "lower"
        aligned[metric] = normalized if direction == "higher" else 1 - normalized
        directions[metric] = direction
        raw_aurocs[metric] = value_auc
    return PoolContext(aligned, directions, raw_aurocs, np.asarray(pool_mask, bool))


def candidate_score(context: PoolContext, candidate: Candidate) -> np.ndarray:
    """Return the pre-final-direction candidate score."""
    score = context.aligned[candidate.component_1] * candidate.component_1_weight
    if candidate.component_2 is not None:
        score = score + context.aligned[candidate.component_2] * candidate.component_2_weight
    return np.where(context.mask, score, np.nan)


def align_candidate(score: np.ndarray, direction: str) -> np.ndarray:
    return score if direction == "higher" else 1 - score


def evaluate_pool(
    values: dict[str, np.ndarray],
    labels: np.ndarray,
    pool_mask: np.ndarray,
    candidates: list[Candidate],
    progress: object | None = None,
) -> tuple[pl.DataFrame, PoolContext]:
    """Evaluate a complete finite candidate family in one token pool."""
    context = _metric_context(values, labels, pool_mask)
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        score = candidate_score(context, candidate)
        finite = np.isfinite(score)
        finite_labels = labels[finite]
        valid_classes = len(finite_labels) > 1 and np.unique(finite_labels).size == 2
        nonconstant = bool(finite.any() and np.ptp(score[finite]) > 0)
        eligible = bool(finite.sum() >= 2 and valid_classes and nonconstant)
        raw_auc = auroc(score, labels) if eligible else float("nan")
        direction = "higher" if not np.isfinite(raw_auc) or raw_auc >= 0.5 else "lower"
        adjusted = max(raw_auc, 1 - raw_auc) if np.isfinite(raw_auc) else float("nan")
        cost = METRIC_COST_ORDER.get(candidate.component_1, 1)
        if candidate.component_2 is not None:
            cost += METRIC_COST_ORDER.get(candidate.component_2, 1)
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "candidate_type": candidate.candidate_type,
                "component_1": candidate.component_1,
                "component_1_weight": candidate.component_1_weight,
                "component_1_direction": context.directions[candidate.component_1],
                "component_1_raw_auroc": context.raw_aurocs[candidate.component_1],
                "component_2": candidate.component_2,
                "component_2_weight": candidate.component_2_weight,
                "component_2_direction": (
                    context.directions[candidate.component_2]
                    if candidate.component_2 is not None
                    else None
                ),
                "component_2_raw_auroc": (
                    context.raw_aurocs[candidate.component_2]
                    if candidate.component_2 is not None
                    else None
                ),
                "candidate_direction": direction,
                "pooled_auroc_raw": raw_auc,
                "adjusted_auroc": adjusted,
                "finite_count": int(finite.sum()),
                "nonfinite_count": int(pool_mask.sum() - finite.sum()),
                "nonfinite_rate": float(1 - finite.sum() / pool_mask.sum())
                if pool_mask.sum()
                else float("nan"),
                "eligible": eligible,
                "candidate_cost_order": cost,
            }
        )
        if progress is not None:
            progress.update(1)
    return pl.DataFrame(rows), context


def _candidate_map(candidates: list[Candidate]) -> dict[str, Candidate]:
    return {candidate.candidate_id: candidate for candidate in candidates}


def ordered_results(frame: pl.DataFrame, score_column: str = "adjusted_auroc") -> pl.DataFrame:
    """Apply the predeclared tolerance-aware candidate ordering."""
    rows = frame.filter(pl.col("eligible")).to_dicts()
    rows.sort(
        key=lambda row: (
            -round(float(row[score_column]), 12),
            float(row.get("nonfinite_rate", 0.0)),
            0 if row["candidate_type"] == "single" else 1,
            int(row.get("candidate_cost_order", 0)),
            str(row["candidate_id"]),
        )
    )
    return pl.DataFrame(rows, schema=frame.schema) if rows else frame.head(0)


def _join_sensitivity(
    primary: pl.DataFrame,
    sensitivity: pl.DataFrame,
    prefix: str,
) -> pl.DataFrame:
    columns = {
        "pooled_auroc_raw": f"{prefix}_pooled_auroc_raw",
        "adjusted_auroc": f"{prefix}_adjusted_auroc",
        "eligible": f"{prefix}_eligible",
    }
    return primary.join(
        sensitivity.select("candidate_id", *columns).rename(columns),
        on="candidate_id",
        how="left",
    )


def _case_percentiles(scores: np.ndarray, case_ids: np.ndarray) -> np.ndarray:
    output = np.full(len(scores), np.nan)
    for case_id in np.unique(case_ids):
        mask = case_ids == case_id
        output[mask] = fractional_midranks(scores[mask])
    return output


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if len(finite) else float("nan")


def _bootstrap_mean(
    values: np.ndarray,
    strata: np.ndarray,
    repeats: int,
    seed: int,
    progress: object | None = None,
) -> tuple[float, float]:
    finite = np.isfinite(values)
    values, strata = values[finite], strata[finite]
    if len(values) < 2 or repeats <= 0:
        if progress is not None and repeats > 0:
            progress.update(repeats)
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    replicates = np.empty(repeats)
    for repeat in range(repeats):
        sampled = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in indices]
        )
        replicates[repeat] = values[sampled].mean()
        if progress is not None:
            progress.update(1)
    return tuple(float(x) for x in np.percentile(replicates, [2.5, 97.5]))


def segment_diagnostics(
    *,
    dataset: str,
    model: str,
    candidate: Candidate,
    context: PoolContext,
    candidate_direction: str,
    labels: np.ndarray,
    case_ids: np.ndarray,
    segments: np.ndarray,
    analysis_regions: np.ndarray,
    source_regions: np.ndarray,
    probe_tok_idx: np.ndarray,
    bootstrap_resamples: int,
    seed: int,
    regime: str,
    bootstrap_progress: object | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Summarize a frozen all-token candidate by canonical segment."""
    raw_score = candidate_score(context, candidate)
    score = align_candidate(raw_score, candidate_direction)
    analysis_mask = segments != "trailer"
    percentiles = _case_percentiles(np.where(analysis_mask, score, np.nan), case_ids)
    unique_cases = np.unique(case_ids)
    case_strata = {case_id: _subdataset(case_id) for case_id in unique_cases}
    case_segment_values: dict[str, dict[str, dict[str, float]]] = {}
    rows: list[dict[str, object]] = []
    for segment in SEGMENTS:
        segment_mask = (segments == segment) & analysis_mask
        if not segment_mask.any():
            continue
        per_case: dict[str, dict[str, float]] = {}
        for case_id in unique_cases:
            case_mask = (case_ids == case_id) & segment_mask & np.isfinite(score)
            if not case_mask.any():
                continue
            case_labels = labels[case_mask]
            case_scores = score[case_mask]
            case_auc = auroc(case_scores, case_labels)
            all_case = (case_ids == case_id) & analysis_mask & np.isfinite(score)
            entry = {
                "auc": case_auc,
                "mean_percentile": float(np.nanmean(percentiles[case_mask])),
                "base_rate": float(case_labels.mean()),
            }
            for budget in (0.01, 0.10):
                suffix = str(int(100 * budget))
                available = np.flatnonzero(all_case)
                k = max(1, math.ceil(len(available) * budget))
                selected = available[
                    np.argsort(-score[available], kind="stable")[:k]
                ]
                share = float(np.mean(segments[selected] == segment))
                availability = float(np.mean(segments[available] == segment))
                entry[f"top_{suffix}_share"] = share
                entry[f"top_{suffix}_enrichment"] = (
                    share / availability if availability > 0 else float("nan")
                )
            per_case[case_id] = entry
        case_segment_values[segment] = per_case
        auc_values = np.array([value["auc"] for value in per_case.values()])
        percentile_values = np.array(
            [value["mean_percentile"] for value in per_case.values()]
        )
        strata = np.array([case_strata[case_id] for case_id in per_case])
        if bootstrap_progress is not None:
            bootstrap_progress.set_postfix_str(f"{segment}: AUROC and mean rank")
        auc_ci = _bootstrap_mean(
            auc_values, strata, bootstrap_resamples, seed, bootstrap_progress
        )
        pct_ci = _bootstrap_mean(
            percentile_values, strata, bootstrap_resamples, seed + 1, bootstrap_progress
        )
        lengths = []
        for case_id in per_case:
            lengths.append(int(np.sum((case_ids == case_id) & segment_mask)))
        finite_segment = segment_mask & np.isfinite(score)
        row: dict[str, object] = {
            "dataset": dataset,
            "model": model,
            "model_name": MODEL_INFO[model]["name"],
            "metric_regime": regime,
            "candidate_id": candidate.candidate_id,
            "segment": segment,
            "token_count": int(segment_mask.sum()),
            "finite_token_count": int(finite_segment.sum()),
            "case_count": len(per_case),
            "auc_case_count": int(np.isfinite(auc_values).sum()),
            "mean_segment_length": float(np.mean(lengths)),
            "median_segment_length": float(np.median(lengths)),
            "token_base_rate": float(labels[segment_mask].mean()),
            "case_balanced_base_rate": float(
                np.mean([value["base_rate"] for value in per_case.values()])
            ),
            "pooled_auroc": auroc(score[segment_mask], labels[segment_mask]),
            "case_macro_auroc": _finite_mean(auc_values),
            "case_macro_auroc_ci_low": auc_ci[0],
            "case_macro_auroc_ci_high": auc_ci[1],
            "mean_relevance_percentile": _finite_mean(percentile_values),
            "mean_relevance_percentile_ci_low": pct_ci[0],
            "mean_relevance_percentile_ci_high": pct_ci[1],
            "template_fraction": float(np.mean(analysis_regions[segment_mask] == "template")),
            "content_fraction": float(np.mean(analysis_regions[segment_mask] != "template")),
        }
        for source in ("system", "user", "assistant_prior", "assistant", "template"):
            row[f"source_{source}_fraction"] = float(
                np.mean(source_regions[segment_mask] == source)
            )
        if dataset in {"tt", "taboo"} and segment == "output":
            reconstructed = probe_tok_idx[segment_mask] < 0
            row["response_fraction"] = 1.0
            row["reconstructed_tail_fraction"] = float(reconstructed.mean())
        elif dataset == "liars" and segment == "output":
            row["response_fraction"] = 1.0
            row["reconstructed_tail_fraction"] = 0.0
        else:
            row["response_fraction"] = 0.0
            row["reconstructed_tail_fraction"] = 0.0
        for budget in (0.01, 0.10):
            suffix = str(int(100 * budget))
            if bootstrap_progress is not None:
                bootstrap_progress.set_postfix_str(
                    f"{segment}: top-{suffix}% concentration"
                )
            shares = np.array(
                [value[f"top_{suffix}_share"] for value in per_case.values()]
            )
            enrichments = np.array(
                [value[f"top_{suffix}_enrichment"] for value in per_case.values()]
            )
            share_ci = _bootstrap_mean(
                shares, strata, bootstrap_resamples, seed + 2, bootstrap_progress
            )
            enrichment_ci = _bootstrap_mean(
                enrichments, strata, bootstrap_resamples, seed + 3, bootstrap_progress
            )
            row[f"top_{suffix}_share"] = _finite_mean(shares)
            row[f"top_{suffix}_share_ci_low"] = share_ci[0]
            row[f"top_{suffix}_share_ci_high"] = share_ci[1]
            row[f"top_{suffix}_enrichment"] = _finite_mean(enrichments)
            row[f"top_{suffix}_enrichment_ci_low"] = enrichment_ci[0]
            row[f"top_{suffix}_enrichment_ci_high"] = enrichment_ci[1]
        rows.append(row)

    contrasts: list[dict[str, object]] = []
    available_segments = [segment for segment in SEGMENTS if segment in case_segment_values]
    pairs = (
        [("boundary", "input")]
        if dataset == "opi"
        else [("boundary", "input"), ("output", "input"), ("boundary", "output")]
    )
    row_map = {row["segment"]: row for row in rows}
    for first, second in pairs:
        if first not in available_segments or second not in available_segments:
            continue
        first_cases = case_segment_values[first]
        second_cases = case_segment_values[second]
        matched = sorted(set(first_cases) & set(second_cases))
        matched = [
            case_id
            for case_id in matched
            if np.isfinite(first_cases[case_id]["auc"])
            and np.isfinite(second_cases[case_id]["auc"])
        ]
        deltas = np.array(
            [first_cases[c]["auc"] - second_cases[c]["auc"] for c in matched]
        )
        strata = np.array([case_strata[c] for c in matched])
        if bootstrap_progress is not None:
            bootstrap_progress.set_postfix_str(
                f"contrast: {first} minus {second}"
            )
        ci = _bootstrap_mean(
            deltas, strata, bootstrap_resamples, seed + 10, bootstrap_progress
        )
        contrasts.append(
            {
                "dataset": dataset,
                "model": model,
                "model_name": MODEL_INFO[model]["name"],
                "metric_regime": regime,
                "candidate_id": candidate.candidate_id,
                "first_segment": first,
                "second_segment": second,
                "matched_case_count": len(matched),
                "pooled_auroc_delta": (
                    float(row_map[first]["pooled_auroc"])
                    - float(row_map[second]["pooled_auroc"])
                ),
                "case_macro_auroc_delta": (
                    float(np.mean(deltas)) if len(deltas) else float("nan")
                ),
                "case_macro_delta_ci_low": ci[0],
                "case_macro_delta_ci_high": ci[1],
                "pointwise_interval": True,
            }
        )
    return pl.DataFrame(rows), pl.DataFrame(contrasts)


def _stable_fold(case_id: str, seed: int, folds: int) -> int:
    value = hashlib.blake2b(f"{seed}:{case_id}".encode(), digest_size=8).digest()
    return int.from_bytes(value, "little") % folds


def _train_mapping(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map train and test values through the training empirical CDF."""
    train_score = fractional_midranks(train)
    output = np.full(test.shape, np.nan)
    finite_train = np.isfinite(train)
    finite_test = np.isfinite(test)
    if finite_train.sum() < 2:
        return train_score, output
    order = np.argsort(train[finite_train], kind="stable")
    x = train[finite_train][order]
    y = train_score[finite_train][order]
    unique, starts, counts = np.unique(x, return_index=True, return_counts=True)
    mapped = np.array([y[start : start + count].mean() for start, count in zip(starts, counts)])
    output[finite_test] = np.interp(test[finite_test], unique, mapped, left=0.0, right=1.0)
    return train_score, output


def heldout_diagnostic(
    *,
    dataset: str,
    model: str,
    values: dict[str, np.ndarray],
    labels: np.ndarray,
    case_ids: np.ndarray,
    candidates: list[Candidate],
    folds: int,
    seed: int,
    candidate_progress: object | None = None,
    bootstrap_progress: object | None = None,
) -> tuple[dict[str, object], pl.DataFrame]:
    """Evaluate full-pool and single-only selection on untouched case folds."""
    unique_cases = sorted(np.unique(case_ids))
    fold_map: dict[str, int] = {}
    strata: dict[str, list[str]] = {}
    for case_id in unique_cases:
        strata.setdefault(_subdataset(case_id), []).append(case_id)
    for group in strata.values():
        group.sort(key=lambda c: _stable_fold(c, seed, 2**16))
        for index, case_id in enumerate(group):
            fold_map[case_id] = index % folds
    fold_values = np.array([fold_map[case_id] for case_id in case_ids])
    oof_full = np.full(len(labels), np.nan)
    oof_single = np.full(len(labels), np.nan)
    rows: list[dict[str, object]] = []
    cmap = _candidate_map(candidates)
    for fold in range(folds):
        if candidate_progress is not None:
            candidate_progress.set_postfix_str(
                f"held-out fold {fold + 1}/{folds}: training selection")
        train_mask, test_mask = fold_values != fold, fold_values == fold
        train_values: dict[str, np.ndarray] = {}
        test_values: dict[str, np.ndarray] = {}
        for metric, raw in values.items():
            train_rank, test_rank = _train_mapping(raw[train_mask], raw[test_mask])
            train_values[metric] = train_rank
            test_values[metric] = test_rank
        train_context = _metric_context(
            train_values, labels[train_mask], np.ones(train_mask.sum(), dtype=bool)
        )
        test_context = PoolContext(
            aligned={},
            directions=train_context.directions,
            raw_aurocs=train_context.raw_aurocs,
            mask=np.ones(test_mask.sum(), dtype=bool),
        )
        for metric in values:
            test_context.aligned[metric] = (
                test_values[metric]
                if train_context.directions[metric] == "higher"
                else 1 - test_values[metric]
            )
        train_rows = []
        for candidate in candidates:
            score = candidate_score(train_context, candidate)
            value_auc = auroc(score, labels[train_mask])
            train_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "candidate_type": candidate.candidate_type,
                    "candidate_cost_order": sum(
                        METRIC_COST_ORDER.get(name, 1) for name in candidate.components
                    ),
                    "nonfinite_rate": float((~np.isfinite(score)).mean()),
                    "eligible": bool(np.isfinite(value_auc)),
                    "adjusted_auroc": max(value_auc, 1 - value_auc)
                    if np.isfinite(value_auc)
                    else float("nan"),
                    "candidate_direction": "higher" if value_auc >= 0.5 else "lower",
                }
            )
            if candidate_progress is not None:
                candidate_progress.update(1)
        ordered = ordered_results(pl.DataFrame(train_rows))
        full = ordered.row(0, named=True)
        single = ordered.filter(pl.col("candidate_type") == "single").row(0, named=True)
        for procedure, selected, destination in (
            ("all_candidates", full, oof_full),
            ("original_only", single, oof_single),
        ):
            candidate = cmap[str(selected["candidate_id"])]
            score = candidate_score(test_context, candidate)
            score = align_candidate(score, str(selected["candidate_direction"]))
            destination[test_mask] = score
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "fold": fold,
                    "procedure": procedure,
                    "candidate_id": candidate.candidate_id,
                    "candidate_type": candidate.candidate_type,
                    "candidate_direction": selected["candidate_direction"],
                    "train_adjusted_auroc": selected["adjusted_auroc"],
                    "test_pooled_auroc": auroc(score, labels[test_mask]),
                    "test_case_count": len(np.unique(case_ids[test_mask])),
                }
            )
    def case_aurocs(score: np.ndarray) -> dict[str, float]:
        return {
            case_id: auroc(score[case_ids == case_id], labels[case_ids == case_id])
            for case_id in unique_cases
        }
    full_case, single_case = case_aurocs(oof_full), case_aurocs(oof_single)
    matched = [
        case_id
        for case_id in unique_cases
        if np.isfinite(full_case[case_id]) and np.isfinite(single_case[case_id])
    ]
    deltas = np.array([full_case[c] - single_case[c] for c in matched])
    delta_strata = np.array([_subdataset(c) for c in matched])
    if bootstrap_progress is not None:
        bootstrap_progress.set_postfix_str("held-out ensemble-original delta")
    ci = _bootstrap_mean(deltas, delta_strata, 2000, seed, bootstrap_progress)
    selected_rows = pl.DataFrame(rows).filter(pl.col("procedure") == "all_candidates")
    frequencies = selected_rows.group_by("candidate_id").len().sort("len", descending=True)
    diagnostic = {
        "dataset": dataset,
        "model": model,
        "heldout_pooled_auroc": auroc(oof_full, labels),
        "heldout_original_only_pooled_auroc": auroc(oof_single, labels),
        "heldout_case_macro_auroc": _finite_mean(full_case.values()),
        "heldout_original_only_case_macro_auroc": float(
            _finite_mean(single_case.values())
        ),
        "heldout_case_macro_delta": float(np.mean(deltas)),
        "heldout_delta_ci_low": ci[0],
        "heldout_delta_ci_high": ci[1],
        "heldout_auc_case_count": len(matched),
        "outer_fold_selection_frequency": float(frequencies["len"][0] / folds),
        "fold_count": folds,
        "selection_seed": seed,
    }
    return diagnostic, pl.DataFrame(rows)


def select_model_pair(
    frame: pl.DataFrame,
    dataset: str,
    model: str,
    *,
    include_attn_rollout: bool,
    folds: int,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, object]:
    """Run all-token, common-support, segment, and held-out selection."""
    metrics = list(CANDIDATE_METRICS)
    if include_attn_rollout and dataset == "opi":
        metrics.extend(OPI_ONLY_METRICS)
    missing = sorted(set(metrics) - set(frame.columns))
    if missing:
        raise ValueError(f"{dataset}/{model} missing metrics: {', '.join(missing)}")
    candidates = candidate_definitions(metrics)
    labels = frame["on_task"].to_numpy().astype(np.int8)
    case_ids = frame["case_id"].to_numpy().astype(str)
    positions = frame["tok_idx"].to_numpy().astype(np.int64)
    source_regions = frame["region"].to_numpy().astype(str)
    segments, analysis_regions = derive_segments(case_ids, positions, source_regions, dataset)
    present_segments = [segment for segment in SEGMENTS if np.any(segments == segment)]
    contrast_pairs = (
        [("boundary", "input")]
        if dataset == "opi"
        else [("boundary", "input"), ("output", "input"), ("boundary", "output")]
    )
    contrast_count = sum(
        first in present_segments and second in present_segments
        for first, second in contrast_pairs
    )
    candidate_progress = tqdm(
        total=len(candidates) * (2 + len(present_segments) + folds),
        desc=f"{dataset}/{model} candidate scoring",
        unit="candidate",
        leave=False,
        dynamic_ncols=True,
        position=1,
    )
    bootstrap_progress = tqdm(
        total=2000 + bootstrap_resamples * (6 * len(present_segments) + contrast_count),
        desc=f"{dataset}/{model} case bootstrap",
        unit="replicate",
        leave=False,
        dynamic_ncols=True,
        position=2,
    )
    probe = (
        frame["probe_tok_idx"].to_numpy().astype(np.int64)
        if "probe_tok_idx" in frame.columns
        else np.zeros(frame.height, dtype=np.int64)
    )
    values = {metric: frame[metric].to_numpy().astype(float) for metric in metrics}
    full_mask = np.ones(frame.height, dtype=bool)
    candidate_progress.set_postfix_str("all-token candidate pool")
    results, full_context = evaluate_pool(
        values, labels, full_mask, candidates, candidate_progress
    )
    common_mask = np.logical_and.reduce([np.isfinite(value) for value in values.values()])
    candidate_progress.set_postfix_str("common-finite sensitivity")
    common_results, _ = evaluate_pool(
        values, labels, common_mask, candidates, candidate_progress
    )
    results = _join_sensitivity(results, common_results, "common_support")
    results = results.with_columns(
        pl.lit(dataset).alias("dataset"),
        pl.lit(model).alias("model"),
        pl.lit(MODEL_INFO[model]["name"]).alias("model_name"),
        pl.lit(include_attn_rollout).alias("includes_attn_rollout"),
    )
    ordered = ordered_results(results)
    selected = ordered.row(0, named=True)
    runner = ordered.row(1, named=True)
    best_single = ordered.filter(pl.col("candidate_type") == "single").row(0, named=True)

    segment_results = []
    segment_manifests = []
    for segment in SEGMENTS:
        mask = segments == segment
        if not mask.any():
            continue
        candidate_progress.set_postfix_str(f"{segment}: independent selection")
        evaluated, _ = evaluate_pool(
            values, labels, mask, candidates, candidate_progress
        )
        evaluated = evaluated.with_columns(
            pl.lit(dataset).alias("dataset"),
            pl.lit(model).alias("model"),
            pl.lit(segment).alias("segment"),
        )
        segment_results.append(evaluated)
        winner = ordered_results(evaluated).row(0, named=True)
        segment_manifests.append(winner)

    heldout, fold_results = heldout_diagnostic(
        dataset=dataset,
        model=model,
        values=values,
        labels=labels,
        case_ids=case_ids,
        candidates=candidates,
        folds=folds,
        candidate_progress=candidate_progress,
        bootstrap_progress=bootstrap_progress,
        seed=seed,
    )
    candidate = _candidate_map(candidates)[str(selected["candidate_id"])]
    segment_summary, contrast_summary = segment_diagnostics(
        dataset=dataset,
        model=model,
        candidate=candidate,
        context=full_context,
        candidate_direction=str(selected["candidate_direction"]),
        labels=labels,
        case_ids=case_ids,
        segments=segments,
        analysis_regions=analysis_regions,
        source_regions=source_regions,
        probe_tok_idx=probe,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
        regime="auroc-ensemble-best",
        bootstrap_progress=bootstrap_progress,
    )
    candidate_progress.close()
    bootstrap_progress.close()
    manifest = {
        **selected,
        "selection_criterion": "direction_adjusted_pooled_auroc",
        "runner_up_candidate_id": runner["candidate_id"],
        "runner_up_adjusted_auroc": runner["adjusted_auroc"],
        "delta_to_runner_up": selected["adjusted_auroc"] - runner["adjusted_auroc"],
        "best_original_candidate_id": best_single["candidate_id"],
        "best_original_adjusted_auroc": best_single["adjusted_auroc"],
        "gain_over_best_original": selected["adjusted_auroc"] - best_single["adjusted_auroc"],
        "common_support_winner": ordered_results(
            common_results.with_columns(
                pl.lit(dataset).alias("dataset"), pl.lit(model).alias("model")
            )
        )["candidate_id"][0],
        "winner_changes_common_support": (
            selected["candidate_id"]
            != ordered_results(
                common_results.with_columns(
                    pl.lit(dataset).alias("dataset"), pl.lit(model).alias("model")
                )
            )["candidate_id"][0]
        ),
        **heldout,
    }
    segment_results_frame = pl.concat(segment_results, how="diagonal_relaxed")
    segment_wide = (
        segment_results_frame.select("candidate_id", "segment", "adjusted_auroc")
        .pivot(on="segment", index="candidate_id", values="adjusted_auroc")
        .rename(
            {
                segment: f"{segment}_adjusted_auroc"
                for segment in SEGMENTS
                if segment in segment_results_frame["segment"].unique().to_list()
            }
        )
    )
    score_by_id = {
        str(row["candidate_id"]): float(row["adjusted_auroc"])
        for row in results.iter_rows(named=True)
    }
    gain_rows = []
    for definition in candidates:
        component_best = max(
            score_by_id[f"single::{name}"] for name in definition.components
        )
        score = score_by_id[definition.candidate_id]
        gain_rows.append(
            {
                "candidate_id": definition.candidate_id,
                "gain_over_best_original": score - float(best_single["adjusted_auroc"]),
                "gain_over_best_component": score - component_best,
            }
        )
    fold_frequency = (
        fold_results.filter(pl.col("procedure") == "all_candidates")
        .group_by("candidate_id")
        .agg(
            (pl.len() / folds).alias("fold_selection_frequency"),
            pl.col("test_pooled_auroc").mean().alias("heldout_auroc_when_selected"),
        )
    )
    results = (
        results.join(pl.DataFrame(gain_rows), on="candidate_id", how="left")
        .join(segment_wide, on="candidate_id", how="left")
        .join(fold_frequency, on="candidate_id", how="left")
        .with_columns(pl.col("fold_selection_frequency").fill_null(0.0))
    )
    return {
        "manifest": manifest,
        "results": results,
        "segment_results": segment_results_frame,
        "segment_manifests": pl.DataFrame(segment_manifests),
        "fold_results": fold_results,
        "segment_summary": segment_summary,
        "contrast_summary": contrast_summary,
        "values": values,
        "labels": labels,
        "case_ids": case_ids,
        "case_labels": frame["label"].to_numpy(),
        "segments": segments,
        "analysis_regions": analysis_regions,
        "source_regions": source_regions,
        "probe": probe,
        "full_context": full_context,
        "candidates": candidates,
    }


def select_shared(
    model_outputs: dict[tuple[str, str], dict[str, object]],
    dataset: str,
    *,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, pl.DataFrame]:
    """Select an identical candidate and weights across a dataset's models."""
    models = DATASET_MODELS[dataset]
    pair_results = pl.concat(
        [model_outputs[(dataset, model)]["results"] for model in models],
        how="diagonal_relaxed",
    )
    aggregate = (
        pair_results.filter(pl.col("eligible"))
        .group_by(
            "dataset", "candidate_id", "candidate_type",
            "component_1", "component_1_weight",
            "component_2", "component_2_weight",
        )
        .agg(
            pl.col("model").n_unique().alias("model_count"),
            pl.col("adjusted_auroc").mean().alias("equal_model_adjusted_auroc"),
            pl.col("common_support_adjusted_auroc")
            .mean()
            .alias("equal_model_common_support_adjusted_auroc"),
            pl.col("adjusted_auroc").min().alias("minimum_model_adjusted_auroc"),
            pl.col("adjusted_auroc").max().alias("maximum_model_adjusted_auroc"),
            pl.col("nonfinite_rate").mean().alias("nonfinite_rate"),
            pl.col("candidate_cost_order").max().alias("candidate_cost_order"),
        )
        .filter(pl.col("model_count") == len(models))
        .with_columns(pl.lit(True).alias("eligible"))
    )
    ordered = ordered_results(
        aggregate.rename({"equal_model_adjusted_auroc": "adjusted_auroc"})
    )
    common_ordered = ordered_results(
        aggregate.drop("equal_model_adjusted_auroc").rename(
            {"equal_model_common_support_adjusted_auroc": "adjusted_auroc"}
        )
    )
    common_winner_id = str(common_ordered["candidate_id"][0])
    winner_id = str(ordered["candidate_id"][0])
    runner_id = str(ordered["candidate_id"][1])
    shared_score = float(ordered["adjusted_auroc"][0])
    selected_rows = pair_results.filter(pl.col("candidate_id") == winner_id)
    manifests = []
    summaries, contrasts = [], []
    shared_bootstrap_progress = tqdm(
        total=bootstrap_resamples * (13 if dataset == "opi" else 21) * len(models),
        desc=f"{dataset} shared case bootstrap",
        unit="replicate",
        leave=False,
        dynamic_ncols=True,
        position=1,
    )
    for model in models:
        row = selected_rows.filter(pl.col("model") == model).row(0, named=True)
        output = model_outputs[(dataset, model)]
        candidate = _candidate_map(output["candidates"])[winner_id]
        manifest = {
            **row,
            "selection_criterion": "equal_model_direction_adjusted_pooled_auroc",
            "equal_model_adjusted_auroc": shared_score,
            "runner_up_candidate_id": runner_id,
            "runner_up_equal_model_adjusted_auroc": float(ordered["adjusted_auroc"][1]),
            "delta_to_runner_up": shared_score - float(ordered["adjusted_auroc"][1]),
            "common_support_winner": common_winner_id,
            "winner_changes_common_support": winner_id != common_winner_id,
        }
        manifests.append(manifest)
        shared_bootstrap_progress.set_postfix_str(
            f"{model}: fixed shared-winner segments"
        )
        summary, contrast = segment_diagnostics(
            dataset=dataset,
            model=model,
            candidate=candidate,
            context=output["full_context"],
            candidate_direction=str(row["candidate_direction"]),
            labels=output["labels"],
            case_ids=output["case_ids"],
            segments=output["segments"],
            analysis_regions=output["analysis_regions"],
            source_regions=output["source_regions"],
            probe_tok_idx=output["probe"],
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
            regime="auroc-ensemble-shared",
            bootstrap_progress=shared_bootstrap_progress,
        )
        summaries.append(summary)
        contrasts.append(contrast)
    shared_bootstrap_progress.close()

    segment_results = pl.concat(
        [model_outputs[(dataset, model)]["segment_results"] for model in models],
        how="diagonal_relaxed",
    )
    shared_segment_rows = []
    for segment in segment_results["segment"].unique().sort().to_list():
        subset = segment_results.filter(pl.col("segment") == segment)
        grouped = (
            subset.filter(pl.col("eligible"))
            .group_by("candidate_id", "candidate_type")
            .agg(
                pl.col("model").n_unique().alias("model_count"),
                pl.col("adjusted_auroc").mean().alias("equal_model_adjusted_auroc"),
                pl.col("nonfinite_rate").mean().alias("nonfinite_rate"),
                pl.col("candidate_cost_order").max().alias("candidate_cost_order"),
            )
            .filter(pl.col("model_count") == len(models))
            .with_columns(pl.lit(True).alias("eligible"))
            .rename({"equal_model_adjusted_auroc": "adjusted_auroc"})
        )
        winner = ordered_results(grouped).row(0, named=True)
        shared_segment_rows.append(
            {
                "dataset": dataset,
                "segment": segment,
                "candidate_id": winner["candidate_id"],
                "candidate_type": winner["candidate_type"],
                "equal_model_adjusted_auroc": winner["adjusted_auroc"],
            }
        )
    return {
        "results": aggregate,
        "selection": pl.DataFrame(manifests),
        "segment_selection": pl.DataFrame(shared_segment_rows),
        "segment_summary": pl.concat(summaries, how="diagonal_relaxed"),
        "contrast_summary": pl.concat(contrasts, how="diagonal_relaxed"),
    }


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Tie-invariant AP, with non-finite scores in the final score group."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=np.int8)
    positives = int(labels.sum())
    if not positives:
        return 0.0
    clean = np.where(np.isfinite(scores), scores, -np.inf)
    order = np.argsort(-clean, kind="stable")
    clean, ordered_labels = clean[order], labels[order]
    cumulative = np.cumsum(ordered_labels, dtype=np.int64)
    group_ends = np.r_[clean[1:] != clean[:-1], True]
    ends = np.flatnonzero(group_ends)
    positive_at_end = cumulative[ends]
    positive_before = np.r_[0, positive_at_end[:-1]]
    group_positive = positive_at_end - positive_before
    precision = positive_at_end / (ends + 1)
    return float(np.sum(precision * group_positive) / positives)


def _case_slices(case_ids: np.ndarray) -> list[tuple[str, slice]]:
    """Return linear-time contiguous case slices and reject split case blocks."""
    case_ids = np.asarray(case_ids).astype(str)
    if not len(case_ids):
        return []
    starts = np.r_[0, np.flatnonzero(case_ids[1:] != case_ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(case_ids)]
    names = [str(case_ids[start]) for start in starts]
    if len(names) != len(set(names)):
        raise ValueError("case rows must form one contiguous block per case_id")
    return [
        (name, slice(int(start), int(end)))
        for name, start, end in zip(names, starts, ends, strict=True)
    ]


def _shared_fold_map(
    model_outputs: dict[tuple[str, str], dict[str, object]],
    dataset: str,
    *,
    folds: int,
    seed: int,
) -> dict[str, int]:
    """Assign union case IDs to balanced folds shared by every model."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    metadata: dict[str, tuple[str, str]] = {}
    for model in DATASET_MODELS[dataset]:
        output = model_outputs[(dataset, model)]
        case_ids = np.asarray(output["case_ids"]).astype(str)
        case_labels = np.asarray(output["case_labels"])
        for case_id, case_slice in _case_slices(case_ids):
            labels = np.unique(case_labels[case_slice])
            if not len(labels) or not set(labels.tolist()).issubset({0, 1}):
                raise ValueError(f"{dataset}/{model}/{case_id}: case label is not binary")
            # OPI stores token-level injected-span membership rather than a
            # separate transcript label.  The any-positive reduction equals
            # the original label for datasets whose label is case-constant.
            value = (_subdataset(case_id), str(int(np.max(labels))))
            if case_id in metadata and metadata[case_id] != value:
                raise ValueError(f"{dataset}/{case_id}: inconsistent metadata across models")
            metadata[case_id] = value
    strata: dict[tuple[str, str], list[str]] = {}
    for case_id, value in metadata.items():
        strata.setdefault(value, []).append(case_id)
    result: dict[str, int] = {}
    for case_ids in strata.values():
        case_ids.sort(key=lambda value: _stable_fold(value, seed, 2**16))
        for index, case_id in enumerate(case_ids):
            result[case_id] = index % folds
    return result


def _unlabelled_target_fold_map(
    case_ids: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> dict[str, int]:
    """Assign target cases without consulting target labels.

    LOMO label-free evaluation may use target raw scores to fit empirical CDFs,
    but it must not use target labels even to construct its cross-fitting folds.
    Balancing by source subdataset retains the important Liars structure while
    keeping the assignment independent of all target judgements.
    """
    if folds < 2:
        raise ValueError("folds must be at least 2")
    strata: dict[str, list[str]] = {}
    for case_id in sorted(set(np.asarray(case_ids).astype(str))):
        strata.setdefault(_subdataset(case_id), []).append(case_id)
    result: dict[str, int] = {}
    for stratum_cases in strata.values():
        stratum_cases.sort(key=lambda value: _stable_fold(value, seed, 2**16))
        for index, case_id in enumerate(stratum_cases):
            result[case_id] = index % folds
    return result


def _fold_contexts(
    output: dict[str, object],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[PoolContext, PoolContext]:
    """Fit empirical CDFs and component directions on one training fold."""
    values = output["values"]
    labels = np.asarray(output["labels"], dtype=np.int8)
    train_values: dict[str, np.ndarray] = {}
    test_values: dict[str, np.ndarray] = {}
    for metric, raw_value in values.items():
        raw = np.asarray(raw_value, dtype=float)
        train_rank, test_rank = _train_mapping(raw[train_mask], raw[test_mask])
        train_values[metric] = train_rank
        test_values[metric] = test_rank
    train_context = _metric_context(
        train_values,
        labels[train_mask],
        np.ones(int(train_mask.sum()), dtype=bool),
    )
    test_context = PoolContext(
        aligned={},
        directions=train_context.directions,
        raw_aurocs=train_context.raw_aurocs,
        mask=np.ones(int(test_mask.sum()), dtype=bool),
    )
    for metric, rank in test_values.items():
        test_context.aligned[metric] = (
            rank if train_context.directions[metric] == "higher" else 1 - rank
        )
    return train_context, test_context


def _candidate_training_rows(
    context: PoolContext,
    labels: np.ndarray,
    candidates: list[Candidate],
    model: str,
) -> pl.DataFrame:
    """Score every candidate in one model's training pool."""
    rows = []
    for candidate in candidates:
        score = candidate_score(context, candidate)
        finite = np.isfinite(score)
        finite_labels = labels[finite]
        valid_classes = len(finite_labels) > 1 and np.unique(finite_labels).size == 2
        nonconstant = bool(finite.any() and np.ptp(score[finite]) > 0)
        value_auc = auroc(score, labels)
        eligible = bool(finite.sum() >= 2 and valid_classes and nonconstant)
        rows.append(
            {
                "model": model,
                "candidate_id": candidate.candidate_id,
                "candidate_type": candidate.candidate_type,
                "component_1": candidate.component_1,
                "component_1_weight": candidate.component_1_weight,
                "component_2": candidate.component_2,
                "component_2_weight": candidate.component_2_weight,
                "candidate_direction": (
                    "higher" if not np.isfinite(value_auc) or value_auc >= 0.5 else "lower"
                ),
                "adjusted_auroc": (
                    max(value_auc, 1 - value_auc) if np.isfinite(value_auc) else float("nan")
                ),
                "nonfinite_rate": float((~finite).mean()),
                "eligible": eligible,
                "candidate_cost_order": sum(
                    METRIC_COST_ORDER.get(name, 1) for name in candidate.components
                ),
            }
        )
    return pl.DataFrame(rows)


def _shared_training_order(
    model_results: dict[str, pl.DataFrame],
    models: list[str],
) -> pl.DataFrame:
    """Aggregate candidate training AUROCs with one vote per model."""
    lookups = {
        model: {
            str(row["candidate_id"]): row
            for row in model_results[model].iter_rows(named=True)
        }
        for model in models
    }
    candidate_ids = sorted(set.intersection(*[set(rows) for rows in lookups.values()]))
    rows = []
    for candidate_id in candidate_ids:
        selected = [lookups[model][candidate_id] for model in models]
        if not all(bool(row["eligible"]) for row in selected):
            continue
        first = selected[0]
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_type": first["candidate_type"],
                "component_1": first["component_1"],
                "component_1_weight": first["component_1_weight"],
                "component_2": first["component_2"],
                "component_2_weight": first["component_2_weight"],
                "adjusted_auroc": float(
                    np.mean([float(row["adjusted_auroc"]) for row in selected])
                ),
                "nonfinite_rate": float(
                    np.mean([float(row["nonfinite_rate"]) for row in selected])
                ),
                "eligible": True,
                "candidate_cost_order": max(
                    int(row["candidate_cost_order"]) for row in selected
                ),
            }
        )
    return ordered_results(pl.DataFrame(rows))


def _case_validation_rows(
    *,
    dataset: str,
    model: str,
    procedure: str,
    validation_kind: str,
    protocol: str,
    heldout_model: str | None,
    scores: np.ndarray,
    output: dict[str, object],
    fold_values: np.ndarray,
    candidates_by_fold: dict[int, str],
) -> pl.DataFrame:
    """Compute case-level retrieval and fixed-budget diagnostics."""
    labels = np.asarray(output["labels"], dtype=np.int8)
    case_ids = np.asarray(output["case_ids"]).astype(str)
    regions = np.asarray(output["source_regions"]).astype(str)
    segments = np.asarray(output["segments"]).astype(str)
    rows = []
    for case_id, case_slice in _case_slices(case_ids):
        local_scores = scores[case_slice]
        local_labels = labels[case_slice]
        local_regions = regions[case_slice]
        local_segments = segments[case_slice]
        local_fold = int(np.unique(fold_values[case_slice])[0])
        row: dict[str, object] = {
            "dataset": dataset,
            "model": model,
            "case_id": case_id,
            "source_subdataset": _subdataset(case_id),
            "fold": local_fold,
            "validation_kind": validation_kind,
            "protocol": protocol,
            "heldout_model": heldout_model,
            "procedure": procedure,
            "candidate_id": candidates_by_fold[local_fold],
            "auc": auroc(local_scores, local_labels),
            "ap": _average_precision(local_scores, local_labels),
            "token_count": len(local_labels),
            "positive_count": int(local_labels.sum()),
            "finite_count": int(np.isfinite(local_scores).sum()),
        }
        finite_indices = np.flatnonzero(np.isfinite(local_scores))
        order = finite_indices[
            np.argsort(-local_scores[finite_indices], kind="stable")
        ]
        for budget in (0.01, 0.10):
            suffix = str(int(100 * budget))
            k = min(len(order), max(1, math.ceil(len(order) * budget)))
            selected = order[:k]
            hits = int(local_labels[selected].sum()) if k else 0
            precision = hits / k if k else float("nan")
            positives = int(local_labels.sum())
            base_rate = float(local_labels.mean())
            row[f"precision_at_{suffix}pct"] = precision
            row[f"recall_at_{suffix}pct"] = hits / positives if positives else 0.0
            row[f"lift_at_{suffix}pct"] = (
                precision / base_rate if k and base_rate else float("nan")
            )
            row[f"hit_case_at_{suffix}pct"] = float(hits > 0)
            row[f"template_at_{suffix}pct"] = (
                float(np.mean(local_regions[selected] == "template"))
                if k
                else float("nan")
            )
            for segment in ("input", "boundary", "output", "trailer"):
                row[f"{segment}_at_{suffix}pct"] = (
                    float(np.mean(local_segments[selected] == segment))
                    if k
                    else float("nan")
                )
        rows.append(row)
    return pl.DataFrame(rows)


def _case_summary(
    case_scores: pl.DataFrame,
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    columns = [
        "auc",
        "ap",
        "precision_at_1pct",
        "recall_at_1pct",
        "precision_at_10pct",
        "recall_at_10pct",
        "template_at_1pct",
        "template_at_10pct",
    ]
    summary = {"pooled_auroc": auroc(scores, labels)}
    for column in columns:
        values = case_scores[column].to_numpy().astype(float)
        summary[f"case_macro_{column}"] = _finite_mean(values)
    summary["auc_case_count"] = int(
        np.isfinite(case_scores["auc"].to_numpy().astype(float)).sum()
    )
    summary["case_count"] = case_scores.height
    return summary


def _paired_cluster_delta(
    case_scores: pl.DataFrame,
    models: list[str],
    metric: str,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float, float, int]:
    """Paired union-case bootstrap, retaining shared-case dependence."""
    ensemble = case_scores.filter(pl.col("procedure") == "all_candidates")
    original = case_scores.filter(pl.col("procedure") == "original_only")
    left = {
        (str(row["model"]), str(row["case_id"])): float(row[metric])
        for row in ensemble.iter_rows(named=True)
        if np.isfinite(row[metric])
    }
    right = {
        (str(row["model"]), str(row["case_id"])): float(row[metric])
        for row in original.iter_rows(named=True)
        if np.isfinite(row[metric])
    }
    deltas = {key: left[key] - right[key] for key in left.keys() & right.keys()}
    model_values = {
        model: [value for (row_model, _), value in deltas.items() if row_model == model]
        for model in models
    }
    point = _finite_mean(
        _finite_mean(model_values[model]) for model in models
    )
    union_ids = sorted({case_id for _model, case_id in deltas})
    if not union_ids or repeats <= 0:
        return point, float("nan"), float("nan"), len(deltas)
    strata: dict[str, list[str]] = {}
    for case_id in union_ids:
        strata.setdefault(_subdataset(case_id), []).append(case_id)
    rng = np.random.default_rng(seed)
    replicates = np.empty(repeats)
    for repeat in range(repeats):
        sampled = [
            case_id
            for ids in strata.values()
            for case_id in rng.choice(ids, size=len(ids), replace=True)
        ]
        per_model = []
        for model in models:
            values = [deltas[(model, case_id)] for case_id in sampled if (model, case_id) in deltas]
            per_model.append(float(np.mean(values)) if values else float("nan"))
        replicates[repeat] = _finite_mean(per_model)
    low, high = np.nanpercentile(replicates, [2.5, 97.5])
    return point, float(low), float(high), len(deltas)


def shared_heldout_validation(
    model_outputs: dict[tuple[str, str], dict[str, object]],
    dataset: str,
    full_winner_id: str,
    *,
    folds: int,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, pl.DataFrame]:
    """Cross-fit the complete equal-model shared selection procedure."""
    models = list(DATASET_MODELS[dataset])
    first_output = model_outputs[(dataset, models[0])]
    candidates = list(first_output["candidates"])
    candidate_map = _candidate_map(candidates)
    fold_map = _shared_fold_map(model_outputs, dataset, folds=folds, seed=seed)
    oof = {
        (model, procedure): np.full(
            len(model_outputs[(dataset, model)]["labels"]), np.nan
        )
        for model in models
        for procedure in VALIDATION_PROCEDURES
    }
    token_folds = {
        model: np.array(
            [fold_map[str(case_id)] for case_id in model_outputs[(dataset, model)]["case_ids"]],
            dtype=np.int64,
        )
        for model in models
    }
    fold_rows = []
    candidate_by_fold: dict[tuple[str, str, int], str] = {}
    for fold in range(folds):
        train_results: dict[str, pl.DataFrame] = {}
        test_contexts: dict[str, PoolContext] = {}
        test_masks: dict[str, np.ndarray] = {}
        for model in models:
            output = model_outputs[(dataset, model)]
            fold_values = token_folds[model]
            train_mask, test_mask = fold_values != fold, fold_values == fold
            train_context, test_context = _fold_contexts(output, train_mask, test_mask)
            train_results[model] = _candidate_training_rows(
                train_context,
                np.asarray(output["labels"], dtype=np.int8)[train_mask],
                candidates,
                model,
            )
            test_contexts[model] = test_context
            test_masks[model] = test_mask
        ordered = _shared_training_order(train_results, models)
        selected = {
            "all_candidates": ordered.row(0, named=True),
            "original_only": ordered.filter(
                pl.col("candidate_type") == "single"
            ).row(0, named=True),
        }
        for procedure, shared_row in selected.items():
            candidate_id = str(shared_row["candidate_id"])
            candidate = candidate_map[candidate_id]
            for model in models:
                model_row = train_results[model].filter(
                    pl.col("candidate_id") == candidate_id
                ).row(0, named=True)
                score = align_candidate(
                    candidate_score(test_contexts[model], candidate),
                    str(model_row["candidate_direction"]),
                )
                test_mask = test_masks[model]
                oof[(model, procedure)][test_mask] = score
                output = model_outputs[(dataset, model)]
                labels = np.asarray(output["labels"], dtype=np.int8)
                candidate_by_fold[(model, procedure, fold)] = candidate_id
                fold_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "fold": fold,
                        "procedure": procedure,
                        "candidate_id": candidate_id,
                        "candidate_type": shared_row["candidate_type"],
                        "candidate_direction": model_row["candidate_direction"],
                        "train_equal_model_adjusted_auroc": shared_row["adjusted_auroc"],
                        "train_model_adjusted_auroc": model_row["adjusted_auroc"],
                        "test_pooled_auroc": auroc(score, labels[test_mask]),
                        "test_case_count": len(
                            np.unique(np.asarray(output["case_ids"])[test_mask])
                        ),
                    }
                )
    case_frames = []
    model_summaries: dict[tuple[str, str], dict[str, float]] = {}
    for model in models:
        output = model_outputs[(dataset, model)]
        for procedure in VALIDATION_PROCEDURES:
            by_fold = {
                fold: candidate_by_fold[(model, procedure, fold)]
                for fold in range(folds)
            }
            frame = _case_validation_rows(
                dataset=dataset,
                model=model,
                procedure=procedure,
                validation_kind="grouped_case",
                protocol="model_specific_training",
                heldout_model=None,
                scores=oof[(model, procedure)],
                output=output,
                fold_values=token_folds[model],
                candidates_by_fold=by_fold,
            )
            case_frames.append(frame)
            model_summaries[(model, procedure)] = _case_summary(
                frame,
                oof[(model, procedure)],
                np.asarray(output["labels"], dtype=np.int8),
            )
    cases = pl.concat(case_frames, how="diagonal_relaxed")
    auc_delta, auc_low, auc_high, paired_auc_cases = _paired_cluster_delta(
        cases,
        models,
        "auc",
        repeats=bootstrap_resamples,
        seed=seed,
    )
    ap_delta, ap_low, ap_high, paired_ap_cases = _paired_cluster_delta(
        cases,
        models,
        "ap",
        repeats=bootstrap_resamples,
        seed=seed + 1,
    )
    fold_frame = pl.DataFrame(fold_rows)
    full_fold = (
        fold_frame.filter(pl.col("procedure") == "all_candidates")
        .select("fold", "candidate_id")
        .unique()
    )
    frequencies = full_fold.group_by("candidate_id").len().sort(
        ["len", "candidate_id"], descending=[True, False]
    )
    modal_id = str(frequencies["candidate_id"][0])
    modal_frequency = float(frequencies["len"][0] / folds)
    full_frequency = float(
        full_fold.filter(pl.col("candidate_id") == full_winner_id).height / folds
    )
    equal_model = {}
    for procedure in VALIDATION_PROCEDURES:
        for field in (
            "pooled_auroc",
            "case_macro_auc",
            "case_macro_ap",
            "case_macro_precision_at_1pct",
            "case_macro_recall_at_1pct",
            "case_macro_precision_at_10pct",
            "case_macro_recall_at_10pct",
        ):
            equal_model[(procedure, field)] = float(
                np.mean([model_summaries[(model, procedure)][field] for model in models])
            )
    headline_rows = []
    for model in models:
        full = model_summaries[(model, "all_candidates")]
        single = model_summaries[(model, "original_only")]
        headline_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "shared_heldout_model_pooled_auroc": full["pooled_auroc"],
                "shared_heldout_model_case_macro_auroc": full["case_macro_auc"],
                "shared_heldout_model_case_macro_ap": full["case_macro_ap"],
                "shared_heldout_equal_model_pooled_auroc": equal_model[
                    ("all_candidates", "pooled_auroc")
                ],
                "shared_heldout_equal_model_case_macro_auroc": equal_model[
                    ("all_candidates", "case_macro_auc")
                ],
                "shared_heldout_equal_model_case_macro_ap": equal_model[
                    ("all_candidates", "case_macro_ap")
                ],
                "shared_heldout_original_only_case_macro_auroc": equal_model[
                    ("original_only", "case_macro_auc")
                ],
                "shared_heldout_original_only_case_macro_ap": equal_model[
                    ("original_only", "case_macro_ap")
                ],
                "shared_heldout_case_macro_auroc_delta": auc_delta,
                "shared_heldout_case_macro_auroc_delta_ci_low": auc_low,
                "shared_heldout_case_macro_auroc_delta_ci_high": auc_high,
                "shared_heldout_case_macro_ap_delta": ap_delta,
                "shared_heldout_case_macro_ap_delta_ci_low": ap_low,
                "shared_heldout_case_macro_ap_delta_ci_high": ap_high,
                "shared_heldout_paired_auc_case_rows": paired_auc_cases,
                "shared_heldout_paired_ap_case_rows": paired_ap_cases,
                "shared_outer_fold_modal_candidate_id": modal_id,
                "shared_outer_fold_selection_frequency": modal_frequency,
                "shared_full_winner_outer_fold_frequency": full_frequency,
                "shared_fold_count": folds,
                "shared_selection_seed": seed,
                "shared_heldout_model_original_only_case_macro_auroc": single[
                    "case_macro_auc"
                ],
            }
        )
    return {
        "headline": pl.DataFrame(headline_rows),
        "fold_diagnostics": fold_frame,
        "case_scores": cases,
    }


def _strict_training_order(
    model_outputs: dict[tuple[str, str], dict[str, object]],
    dataset: str,
    models: list[str],
    candidates: list[Candidate],
) -> tuple[pl.DataFrame, dict[str, str]]:
    """Select candidates with directions transferable without target labels."""
    metric_aurocs: dict[str, list[float]] = {metric: [] for metric in CANDIDATE_METRICS}
    normalized: dict[str, dict[str, np.ndarray]] = {}
    for model in models:
        output = model_outputs[(dataset, model)]
        labels = np.asarray(output["labels"], dtype=np.int8)
        full_context = output["full_context"]
        normalized[model] = {}
        for metric, aligned_rank in full_context.aligned.items():
            rank = (
                np.asarray(aligned_rank, dtype=float)
                if full_context.directions[metric] == "higher"
                else 1 - np.asarray(aligned_rank, dtype=float)
            )
            normalized[model][metric] = rank
            metric_aurocs.setdefault(metric, []).append(auroc(rank, labels))
    directions = {
        metric: (
            "higher"
            if _finite_mean(values) >= 0.5
            else "lower"
        )
        for metric, values in metric_aurocs.items()
        if values
    }
    contexts = {}
    for model in models:
        aligned = {
            metric: (
                rank if directions[metric] == "higher" else 1 - rank
            )
            for metric, rank in normalized[model].items()
        }
        contexts[model] = PoolContext(
            aligned=aligned,
            directions=directions,
            raw_aurocs={},
            mask=np.ones(len(next(iter(aligned.values()))), dtype=bool),
        )
    rows = []
    for candidate in candidates:
        aucs, missing = [], []
        eligible = True
        for model in models:
            output = model_outputs[(dataset, model)]
            score = candidate_score(contexts[model], candidate)
            labels = np.asarray(output["labels"], dtype=np.int8)
            finite = np.isfinite(score)
            finite_labels = labels[finite]
            value_auc = auroc(score, labels)
            aucs.append(value_auc)
            missing.append(float((~finite).mean()))
            eligible &= bool(
                finite.sum() >= 2
                and np.unique(finite_labels).size == 2
                and np.ptp(score[finite]) > 0
                and np.isfinite(value_auc)
            )
        mean_auc = _finite_mean(aucs)
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "candidate_type": candidate.candidate_type,
                "component_1": candidate.component_1,
                "component_1_weight": candidate.component_1_weight,
                "component_2": candidate.component_2,
                "component_2_weight": candidate.component_2_weight,
                "candidate_direction": "higher" if mean_auc >= 0.5 else "lower",
                "adjusted_auroc": (
                    max(mean_auc, 1 - mean_auc) if np.isfinite(mean_auc) else float("nan")
                ),
                "nonfinite_rate": _finite_mean(missing),
                "eligible": bool(eligible),
                "candidate_cost_order": sum(
                    METRIC_COST_ORDER.get(name, 1) for name in candidate.components
                ),
            }
        )
    return ordered_results(pl.DataFrame(rows)), directions


def _full_shared_training_order(
    model_outputs: dict[tuple[str, str], dict[str, object]],
    dataset: str,
    models: list[str],
) -> pl.DataFrame:
    frames = {
        model: model_outputs[(dataset, model)]["results"].select(
            "candidate_id",
            "candidate_type",
            "component_1",
            "component_1_weight",
            "component_2",
            "component_2_weight",
            "candidate_direction",
            "adjusted_auroc",
            "nonfinite_rate",
            "eligible",
            "candidate_cost_order",
        )
        for model in models
    }
    return _shared_training_order(frames, models)


def shared_lomo_validation(
    model_outputs: dict[tuple[str, str], dict[str, object]],
    dataset: str,
    *,
    folds: int,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, pl.DataFrame]:
    """Evaluate label-free and target-calibrated leave-one-model-out transfer."""
    models = list(DATASET_MODELS[dataset])
    candidates = list(model_outputs[(dataset, models[0])]["candidates"])
    candidate_map = _candidate_map(candidates)
    diagnostic_rows = []
    case_frames = []
    summary_rows = []
    for target_index, target in enumerate(models):
        _stage(f"{dataset}/{target}: leave-one-model-out selection and evaluation")
        train_models = [model for model in models if model != target]
        strict_order, strict_directions = _strict_training_order(
            model_outputs, dataset, train_models, candidates
        )
        calibrated_order = _full_shared_training_order(
            model_outputs, dataset, train_models
        )
        protocol_rows = {
            "label_free": strict_order,
            "target_calibrated": calibrated_order,
        }
        selected: dict[tuple[str, str], dict[str, object]] = {}
        for protocol, ordered in protocol_rows.items():
            selected[(protocol, "all_candidates")] = ordered.row(0, named=True)
            selected[(protocol, "original_only")] = ordered.filter(
                pl.col("candidate_type") == "single"
            ).row(0, named=True)
        output = model_outputs[(dataset, target)]
        labels = np.asarray(output["labels"], dtype=np.int8)
        case_ids = np.asarray(output["case_ids"]).astype(str)
        fold_map = _unlabelled_target_fold_map(case_ids, folds=folds, seed=seed)
        fold_values = np.array([fold_map[case_id] for case_id in case_ids], dtype=np.int64)
        oof = {
            (protocol, procedure): np.full(len(labels), np.nan)
            for protocol in LOMO_PROTOCOLS
            for procedure in VALIDATION_PROCEDURES
        }
        for fold in range(folds):
            train_mask, test_mask = fold_values != fold, fold_values == fold
            raw_train: dict[str, np.ndarray] = {}
            raw_test: dict[str, np.ndarray] = {}
            for metric, raw_value in output["values"].items():
                raw = np.asarray(raw_value, dtype=float)
                train_rank, test_rank = _train_mapping(raw[train_mask], raw[test_mask])
                raw_train[metric] = train_rank
                raw_test[metric] = test_rank
            calibrated_train = _metric_context(
                raw_train,
                labels[train_mask],
                np.ones(int(train_mask.sum()), dtype=bool),
            )
            calibrated_test = PoolContext(
                aligned={
                    metric: (
                        rank
                        if calibrated_train.directions[metric] == "higher"
                        else 1 - rank
                    )
                    for metric, rank in raw_test.items()
                },
                directions=calibrated_train.directions,
                raw_aurocs=calibrated_train.raw_aurocs,
                mask=np.ones(int(test_mask.sum()), dtype=bool),
            )
            strict_test = PoolContext(
                aligned={
                    metric: (
                        rank if strict_directions[metric] == "higher" else 1 - rank
                    )
                    for metric, rank in raw_test.items()
                },
                directions=strict_directions,
                raw_aurocs={},
                mask=np.ones(int(test_mask.sum()), dtype=bool),
            )
            for protocol in LOMO_PROTOCOLS:
                for procedure in VALIDATION_PROCEDURES:
                    selection = selected[(protocol, procedure)]
                    candidate = candidate_map[str(selection["candidate_id"])]
                    if protocol == "label_free":
                        test_context = strict_test
                        final_direction = str(selection["candidate_direction"])
                        component_directions = strict_directions
                    else:
                        test_context = calibrated_test
                        train_score = candidate_score(calibrated_train, candidate)
                        train_auc = auroc(train_score, labels[train_mask])
                        final_direction = "higher" if train_auc >= 0.5 else "lower"
                        component_directions = calibrated_train.directions
                    score = align_candidate(
                        candidate_score(test_context, candidate), final_direction
                    )
                    oof[(protocol, procedure)][test_mask] = score
                    diagnostic_rows.append(
                        {
                            "dataset": dataset,
                            "heldout_model": target,
                            "training_models": train_models,
                            "train_model_count": len(train_models),
                            "protocol": protocol,
                            "fold": fold,
                            "procedure": procedure,
                            "candidate_id": candidate.candidate_id,
                            "candidate_type": candidate.candidate_type,
                            "component_1_direction": component_directions[
                                candidate.component_1
                            ],
                            "component_2_direction": (
                                component_directions[candidate.component_2]
                                if candidate.component_2 is not None
                                else None
                            ),
                            "candidate_direction": final_direction,
                            "target_labels_used_for_direction": (
                                protocol == "target_calibrated"
                            ),
                            "test_pooled_auroc": auroc(score, labels[test_mask]),
                            "test_case_count": len(np.unique(case_ids[test_mask])),
                        }
                    )
        target_case_frames: dict[tuple[str, str], pl.DataFrame] = {}
        for protocol in LOMO_PROTOCOLS:
            for procedure in VALIDATION_PROCEDURES:
                candidate_id = str(selected[(protocol, procedure)]["candidate_id"])
                frame = _case_validation_rows(
                    dataset=dataset,
                    model=target,
                    procedure=procedure,
                    validation_kind="leave_one_model_out",
                    protocol=protocol,
                    heldout_model=target,
                    scores=oof[(protocol, procedure)],
                    output=output,
                    fold_values=fold_values,
                    candidates_by_fold={fold: candidate_id for fold in range(folds)},
                )
                case_frames.append(frame)
                target_case_frames[(protocol, procedure)] = frame
        for protocol in LOMO_PROTOCOLS:
            combined = pl.concat(
                [
                    target_case_frames[(protocol, procedure)]
                    for procedure in VALIDATION_PROCEDURES
                ],
                how="diagonal_relaxed",
            )
            auc_delta, auc_low, auc_high, paired_auc = _paired_cluster_delta(
                combined,
                [target],
                "auc",
                repeats=bootstrap_resamples,
                seed=seed + 100 * target_index,
            )
            ap_delta, ap_low, ap_high, paired_ap = _paired_cluster_delta(
                combined,
                [target],
                "ap",
                repeats=bootstrap_resamples,
                seed=seed + 100 * target_index + 1,
            )
            for procedure in VALIDATION_PROCEDURES:
                frame = target_case_frames[(protocol, procedure)]
                summary = _case_summary(
                    frame,
                    oof[(protocol, procedure)],
                    labels,
                )
                selection = selected[(protocol, procedure)]
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "heldout_model": target,
                        "training_models": train_models,
                        "train_model_count": len(train_models),
                        "weak_model_transfer_evidence": len(train_models) == 1,
                        "protocol": protocol,
                        "procedure": procedure,
                        "candidate_id": selection["candidate_id"],
                        "candidate_type": selection["candidate_type"],
                        **summary,
                        "case_macro_auroc_delta_vs_original": auc_delta,
                        "case_macro_auroc_delta_ci_low": auc_low,
                        "case_macro_auroc_delta_ci_high": auc_high,
                        "case_macro_ap_delta_vs_original": ap_delta,
                        "case_macro_ap_delta_ci_low": ap_low,
                        "case_macro_ap_delta_ci_high": ap_high,
                        "paired_auc_case_count": paired_auc,
                        "paired_ap_case_count": paired_ap,
                        "model_population_interval": False,
                    }
                )
    summary = pl.DataFrame(summary_rows)
    mean_columns = [
        "pooled_auroc",
        "case_macro_auc",
        "case_macro_ap",
        "case_macro_precision_at_1pct",
        "case_macro_recall_at_1pct",
        "case_macro_precision_at_10pct",
        "case_macro_recall_at_10pct",
    ]
    equal_model = summary.group_by("dataset", "protocol", "procedure").agg(
        *[
            pl.col(column).mean().alias(f"equal_model_{column}")
            for column in mean_columns
        ]
    )
    summary = summary.join(
        equal_model,
        on=["dataset", "protocol", "procedure"],
        how="left",
    )
    return {
        "fold_diagnostics": pl.DataFrame(diagnostic_rows),
        "case_scores": pl.concat(case_frames, how="diagonal_relaxed"),
        "summary": summary,
    }


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        return "—" if not np.isfinite(value) else f"{value:.4f}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *["| " + " | ".join(_fmt(value) for value in row) + " |" for row in rows],
    ]


def write_report(
    best: pl.DataFrame,
    shared: pl.DataFrame,
    segment_best: pl.DataFrame,
    segment_shared: pl.DataFrame,
    lomo_summary: pl.DataFrame,
    path: Path,
    *,
    includes_attn_rollout: bool,
) -> None:
    winners = [
        [
            row["dataset"], row["model"], row["candidate_id"], row["candidate_type"],
            row["adjusted_auroc"], row["gain_over_best_original"],
            row["heldout_pooled_auroc"], row["heldout_case_macro_delta"],
            row["winner_changes_common_support"],
        ]
        for row in best.sort("dataset", "model").iter_rows(named=True)
    ]
    shared_rows = [
        [row["dataset"], row["model"], row["candidate_id"],
         row["equal_model_adjusted_auroc"], row["adjusted_auroc"]]
        for row in shared.sort("dataset", "model").iter_rows(named=True)
    ]
    segment_rows = [
        [row["dataset"], row["model"], row["segment"], row["candidate_id"],
         row["adjusted_auroc"]]
        for row in segment_best.sort("dataset", "model", "segment").iter_rows(named=True)
    ]
    shared_segment_rows = [
        [row["dataset"], row["segment"], row["candidate_id"],
         row["equal_model_adjusted_auroc"]]
        for row in segment_shared.sort("dataset", "segment").iter_rows(named=True)
    ]
    shared_validation_rows = [
        [
            row["dataset"],
            row["shared_heldout_equal_model_case_macro_auroc"],
            row["shared_heldout_original_only_case_macro_auroc"],
            row["shared_heldout_case_macro_auroc_delta"],
            (
                f"[{row['shared_heldout_case_macro_auroc_delta_ci_low']:.4f}, "
                f"{row['shared_heldout_case_macro_auroc_delta_ci_high']:.4f}]"
            ),
            row["shared_heldout_equal_model_case_macro_ap"],
            row["shared_outer_fold_modal_candidate_id"],
            row["shared_outer_fold_selection_frequency"],
            row["shared_full_winner_outer_fold_frequency"],
        ]
        for row in shared.select(
            "dataset",
            "shared_heldout_equal_model_case_macro_auroc",
            "shared_heldout_original_only_case_macro_auroc",
            "shared_heldout_case_macro_auroc_delta",
            "shared_heldout_case_macro_auroc_delta_ci_low",
            "shared_heldout_case_macro_auroc_delta_ci_high",
            "shared_heldout_equal_model_case_macro_ap",
            "shared_outer_fold_modal_candidate_id",
            "shared_outer_fold_selection_frequency",
            "shared_full_winner_outer_fold_frequency",
        )
        .unique()
        .sort("dataset")
        .iter_rows(named=True)
    ] if "shared_heldout_equal_model_case_macro_auroc" in shared.columns else []
    lomo_rows = [
        [
            row["dataset"],
            row["heldout_model"],
            row["protocol"],
            row["candidate_id"],
            row["case_macro_auc"],
            row["case_macro_ap"],
            row["case_macro_auroc_delta_vs_original"],
            (
                f"[{row['case_macro_auroc_delta_ci_low']:.4f}, "
                f"{row['case_macro_auroc_delta_ci_high']:.4f}]"
            ),
            row["case_macro_precision_at_1pct"],
            row["weak_model_transfer_evidence"],
        ]
        for row in lomo_summary.filter(
            pl.col("procedure") == "all_candidates"
        ).sort("dataset", "heldout_model", "protocol").iter_rows(named=True)
    ] if not lomo_summary.is_empty() else []
    lines = [
        "# Exhaustive AUROC rank-ensemble selection",
        "",
        "This is the explicitly non-blind OPI `attn_rollout` sensitivity."
        if includes_attn_rollout
        else "This is the blind primary analysis over the 13 deployable metrics.",
        "",
        "Finite values are converted to pooled fractional midranks, component directions "
        "are aligned using pooled AUROC, and every original plus every 25/75, 50/50, "
        "and 75/25 unordered pair competes in the same pool. Full-data winners are "
        "exploratory; grouped held-out diagnostics estimate selection generalization.",
        "",
        "## Model-specific all-token winners",
        "",
        *_markdown_table(
            ["dataset", "model", "candidate", "type", "AUROC", "gain vs single",
             "held-out AUROC", "held-out macro delta", "common-support change"],
            winners,
        ),
        "",
        "## Dataset-shared all-token winner",
        "",
        *_markdown_table(
            ["dataset", "model", "candidate", "equal-model AUROC", "model AUROC"],
            shared_rows,
        ),
        "",
        "## Dataset-shared grouped held-out validation",
        "",
        *(
            _markdown_table(
                [
                    "dataset", "held-out macro AUROC", "single macro AUROC",
                    "delta", "paired 95% CI", "held-out macro AP",
                    "modal fold winner", "modal frequency", "full-winner frequency",
                ],
                shared_validation_rows,
            )
            if shared_validation_rows
            else ["Shared held-out validation was skipped."]
        ),
        "",
        "The complete shared candidate and the shared original-only comparator are "
        "reselected inside every training fold. Models receive equal weight. The "
        "interval resamples union case-ID clusters and is not a token bootstrap.",
        "",
        "## Leave-one-model-out transfer",
        "",
        *(
            _markdown_table(
                [
                    "dataset", "held-out model", "protocol", "candidate",
                    "case-macro AUROC", "case-macro AP", "delta vs single",
                    "paired 95% CI", "top-1% precision", "weak evidence",
                ],
                lomo_rows,
            )
            if lomo_rows
            else ["Leave-one-model-out validation was skipped."]
        ),
        "",
        "'label_free' fits target empirical CDFs without target judgment labels and "
        "uses directions learned from the remaining models. 'target_calibrated' keeps "
        "candidate identity and weights fixed but learns directions on target-training "
        "cases. Liars has only one training model per holdout, so its transfer result is "
        "descriptive weak evidence.",
        "",
        "## Segment-specific model winners",
        "",
        *_markdown_table(
            ["dataset", "model", "segment", "candidate", "AUROC"], segment_rows
        ),
        "",
        "## Segment-specific shared winners",
        "",
        *_markdown_table(
            ["dataset", "segment", "candidate", "equal-model AUROC"],
            shared_segment_rows,
        ),
        "",
        "Boundary tokens are structural strings, but their hidden states are conditioned "
        "on the full preceding input and can therefore be informative aggregation points. "
        "Segment comparisons are descriptive associations with judge labels, not causal "
        "tests of NLA verbalization utility. Liars trailers are excluded from every segment "
        "summary and figure.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

def write_segment_report(
    best_summary: pl.DataFrame,
    best_contrasts: pl.DataFrame,
    shared_summary: pl.DataFrame,
    path: Path,
) -> None:
    """Write the interpretable fixed-winner segment estimates and contrasts."""
    rows = [
        [
            row["dataset"], row["model"], row["segment"], row["candidate_id"],
            row["token_count"], row["case_count"], row["auc_case_count"],
            row["token_base_rate"], row["template_fraction"],
            row["pooled_auroc"], row["case_macro_auroc"],
            row["mean_relevance_percentile"], row["top_1_enrichment"],
            row["top_10_enrichment"],
        ]
        for row in best_summary.sort("dataset", "model", "segment").iter_rows(named=True)
    ]
    contrasts = [
        [
            row["dataset"], row["model"],
            f"{row['first_segment']} − {row['second_segment']}",
            row["matched_case_count"], row["pooled_auroc_delta"],
            row["case_macro_auroc_delta"], row["case_macro_delta_ci_low"],
            row["case_macro_delta_ci_high"],
        ]
        for row in best_contrasts.sort(
            "dataset", "model", "first_segment", "second_segment"
        ).iter_rows(named=True)
    ]
    lines = [
        "# Input–boundary–output importance analysis",
        "",
        "All values below use the frozen all-token winner, including its all-token rank "
        "mapping, component directions, weights, and final direction. Thus segment AUROCs "
        "are comparable. Mean relevance and enrichment are averaged within cases first.",
        "",
        "## Model-specific fixed-winner estimates",
        "",
        *_markdown_table(
            [
                "dataset", "model", "segment", "candidate", "tokens", "cases",
                "AUROC cases", "on-task rate", "template fraction", "pooled AUROC",
                "case-macro AUROC", "mean percentile", "top-1% enrichment",
                "top-10% enrichment",
            ],
            rows,
        ),
        "",
        "## Matched-case segment contrasts",
        "",
        *_markdown_table(
            [
                "dataset", "model", "contrast", "matched cases", "pooled delta",
                "case-macro delta", "95% low", "95% high",
            ],
            contrasts,
        ),
        "",
        f"The corresponding dataset-shared table contains {shared_summary.height} "
        "model–segment estimates. A positive contrast means stronger measured "
        "discrimination in the first named segment. Intervals are descriptive, "
        "pointwise whole-case bootstrap intervals; they are not simultaneous tests.",
        "",
        "Input contains earlier scaffolding and Liars prior-assistant turns. Boundary is "
        "the final generation-prompt run and is predominantly template, but its hidden "
        "states are conditioned on preceding content. Output is final assistant text. "
        "Liars trailers are excluded. These results do not establish causal usefulness "
        "for downstream NLA verbalization.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_parquet(frame: pl.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.write_parquet(tmp, compression="zstd")
    os.replace(tmp, path)
    _stage(f"wrote {path} ({frame.height:,} rows)")


def _run(
    *,
    input_dir: Path,
    pairs: list[tuple[str, str]],
    output_root: Path,
    include_attn_rollout: bool,
    folds: int,
    bootstrap_resamples: int,
    seed: int,
    overwrite: bool,
    skip_shared_validation: bool,
) -> None:
    run_started = time.perf_counter()
    _stage(
        f"starting {len(pairs)} model/dataset combinations; "
        f"{folds} grouped folds and {bootstrap_resamples:,} bootstrap replicates"
    )
    outputs: dict[tuple[str, str], dict[str, object]] = {}
    combination_progress = tqdm(
        pairs,
        desc="AUROC ensemble combinations",
        unit="combination",
        dynamic_ncols=True,
    )
    for index, (dataset, model) in enumerate(combination_progress, start=1):
        pair_started = time.perf_counter()
        combination_progress.set_postfix_str(f"{dataset}/{model}: loading parquet")
        _stage(f"{dataset}/{model} ({index}/{len(pairs)}): loading source parquet")
        frame = pl.read_parquet(source_path(input_dir, dataset, model))
        _stage(
            f"{dataset}/{model}: loaded {frame.height:,} tokens; "
            "starting normalization and candidate evaluation"
        )
        combination_progress.set_postfix_str(f"{dataset}/{model}: selecting")
        outputs[(dataset, model)] = select_model_pair(
            frame,
            dataset,
            model,
            include_attn_rollout=include_attn_rollout,
            folds=folds,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        )
        winner = outputs[(dataset, model)]["manifest"]
        _stage(
            f"{dataset}/{model}: complete in {time.perf_counter() - pair_started:.1f}s; "
            f"winner {winner['candidate_id']} "
            f"(adjusted AUROC {float(winner['adjusted_auroc']):.4f})"
        )
    _stage("model-specific selection complete; assembling result tables")
    best_results = pl.concat([value["results"] for value in outputs.values()], how="diagonal_relaxed")
    eligibility = (
        best_results.group_by("dataset", "model", "includes_attn_rollout")
        .agg(
            pl.len().alias("candidate_count"),
            pl.col("candidate_type").eq("single").sum().alias("original_count"),
            pl.col("candidate_type").eq("ensemble").sum().alias("ensemble_count"),
            pl.col("eligible").sum().alias("eligible_count"),
            (~pl.col("eligible")).sum().alias("ineligible_count"),
            (pl.col("finite_count") / (pl.col("finite_count") + pl.col("nonfinite_count")))
            .min()
            .alias("minimum_finite_coverage"),
            pl.col("common_support_eligible").sum().alias("common_support_eligible_count"),
        )
        .sort("dataset", "model")
    )
    best_selection = pl.DataFrame([value["manifest"] for value in outputs.values()])
    segment_results = pl.concat(
        [value["segment_results"] for value in outputs.values()], how="diagonal_relaxed"
    )
    segment_selection = pl.concat(
        [value["segment_manifests"] for value in outputs.values()], how="diagonal_relaxed"
    )
    folds_frame = pl.concat([value["fold_results"] for value in outputs.values()], how="diagonal_relaxed")
    best_segment_summary = pl.concat(
        [value["segment_summary"] for value in outputs.values()], how="diagonal_relaxed"
    )
    best_contrasts = pl.concat(
        [value["contrast_summary"] for value in outputs.values()], how="diagonal_relaxed"
    )
    complete_datasets = [
        dataset
        for dataset in sorted({dataset for dataset, _ in pairs})
        if set(DATASET_MODELS[dataset]).issubset(
            {model for pair_dataset, model in pairs if pair_dataset == dataset}
        )
    ]
    shared_outputs: dict[str, dict[str, pl.DataFrame]] = {}
    shared_progress = tqdm(
        complete_datasets,
        desc="Dataset-shared ensemble selection",
        unit="dataset",
        dynamic_ncols=True,
    )
    for dataset in shared_progress:
        shared_progress.set_postfix_str(f"{dataset}: equal-model selection")
        _stage(f"{dataset}: starting dataset-shared candidate selection")
        shared_outputs[dataset] = select_shared(
            outputs, dataset, bootstrap_resamples=bootstrap_resamples, seed=seed
        )
        selected = shared_outputs[dataset]["selection"].row(0, named=True)
        _stage(f"{dataset}: shared winner {selected['candidate_id']}")
    validation_outputs: dict[str, dict[str, pl.DataFrame]] = {}
    lomo_outputs: dict[str, dict[str, pl.DataFrame]] = {}
    if shared_outputs and not skip_shared_validation:
        for dataset in complete_datasets:
            winner_id = str(
                shared_outputs[dataset]["selection"]["candidate_id"][0]
            )
            _stage(f"{dataset}: starting shared grouped held-out validation")
            validation_outputs[dataset] = shared_heldout_validation(
                outputs,
                dataset,
                winner_id,
                folds=folds,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed,
            )
            shared_outputs[dataset]["selection"] = shared_outputs[dataset][
                "selection"
            ].join(
                validation_outputs[dataset]["headline"],
                on=["dataset", "model"],
                how="left",
            )
            _stage(f"{dataset}: starting leave-one-model-out validation")
            lomo_outputs[dataset] = shared_lomo_validation(
                outputs,
                dataset,
                folds=folds,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed,
            )
    elif shared_outputs:
        _stage("shared held-out and leave-one-model-out validation skipped")
    if not shared_outputs:
        _stage("no complete dataset requested; writing model-specific smoke outputs")
        best_dir = output_root / "results" / "model_best"
        partial_files = {
            best_dir / "auroc_ensemble_best_metric_results.parquet": best_results,
            best_dir / "auroc_ensemble_best_metric_selection.parquet": best_selection,
            best_dir / "auroc_ensemble_best_segment_metric_results.parquet": segment_results,
            best_dir / "auroc_ensemble_best_segment_metric_selection.parquet": segment_selection,
            best_dir / "auroc_ensemble_best_fold_diagnostics.parquet": folds_frame,
            best_dir / "auroc_ensemble_best_segment_summary.parquet": best_segment_summary,
            best_dir / "auroc_ensemble_best_segment_contrasts.parquet": best_contrasts,
            output_root / "results" / "auroc_ensemble_eligibility.parquet": eligibility,
        }
        for path, frame in partial_files.items():
            _write_parquet(frame, path, overwrite)
        report = output_root / "results" / "auroc_ensemble_selection.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            "# Partial AUROC rank-ensemble smoke run\n\n"
            "Model-specific outputs are complete for the requested pairs. "
            "Dataset-shared outputs require every model in a dataset and were not produced.\n",
            encoding="utf-8",
        )
        write_segment_report(
            best_segment_summary,
            best_contrasts,
            best_segment_summary.head(0),
            output_root / "results" / "auroc_ensemble_segment_analysis.md",
        )
        _stage(
            f"partial run complete in {time.perf_counter() - run_started:.1f}s; "
            f"outputs are under {output_root}"
        )
        return
    _stage("assembling dataset-shared results and output manifests")
    shared_results = pl.concat(
        [value["results"] for value in shared_outputs.values()], how="diagonal_relaxed"
    )
    shared_selection = pl.concat(
        [value["selection"] for value in shared_outputs.values()], how="diagonal_relaxed"
    )
    shared_segment_selection = pl.concat(
        [value["segment_selection"] for value in shared_outputs.values()], how="diagonal_relaxed"
    )
    shared_segment_summary = pl.concat(
        [value["segment_summary"] for value in shared_outputs.values()], how="diagonal_relaxed"
    )
    shared_contrasts = pl.concat(
        [value["contrast_summary"] for value in shared_outputs.values()], how="diagonal_relaxed"
    )
    shared_fold_diagnostics = (
        pl.concat(
            [value["fold_diagnostics"] for value in validation_outputs.values()],
            how="diagonal_relaxed",
        )
        if validation_outputs
        else pl.DataFrame()
    )
    shared_case_scores = (
        pl.concat(
            [
                *[value["case_scores"] for value in validation_outputs.values()],
                *[value["case_scores"] for value in lomo_outputs.values()],
            ],
            how="diagonal_relaxed",
        )
        if validation_outputs
        else pl.DataFrame()
    )
    shared_lomo_diagnostics = (
        pl.concat(
            [value["fold_diagnostics"] for value in lomo_outputs.values()],
            how="diagonal_relaxed",
        )
        if lomo_outputs
        else pl.DataFrame()
    )
    shared_lomo_summary = (
        pl.concat(
            [value["summary"] for value in lomo_outputs.values()],
            how="diagonal_relaxed",
        )
        if lomo_outputs
        else pl.DataFrame()
    )
    best_dir = output_root / "results" / "model_best"
    shared_dir = output_root / "results" / "dataset_shared"
    files = {
        best_dir / "auroc_ensemble_best_metric_results.parquet": best_results,
        best_dir / "auroc_ensemble_best_metric_selection.parquet": best_selection,
        best_dir / "auroc_ensemble_best_segment_metric_results.parquet": segment_results,
        best_dir / "auroc_ensemble_best_segment_metric_selection.parquet": segment_selection,
        best_dir / "auroc_ensemble_best_fold_diagnostics.parquet": folds_frame,
        best_dir / "auroc_ensemble_best_segment_summary.parquet": best_segment_summary,
        best_dir / "auroc_ensemble_best_segment_contrasts.parquet": best_contrasts,
        shared_dir / "auroc_ensemble_shared_metric_results.parquet": shared_results,
        shared_dir / "auroc_ensemble_shared_metric_selection.parquet": shared_selection,
        shared_dir / "auroc_ensemble_shared_segment_metric_selection.parquet": shared_segment_selection,
        shared_dir / "auroc_ensemble_shared_segment_summary.parquet": shared_segment_summary,
        shared_dir / "auroc_ensemble_shared_segment_contrasts.parquet": shared_contrasts,
        output_root / "results" / "auroc_ensemble_eligibility.parquet": eligibility,
    }
    if validation_outputs:
        files.update(
            {
                shared_dir / "auroc_ensemble_shared_fold_diagnostics.parquet":
                    shared_fold_diagnostics,
                shared_dir / "auroc_ensemble_shared_case_scores.parquet":
                    shared_case_scores,
                shared_dir / "auroc_ensemble_shared_lomo_diagnostics.parquet":
                    shared_lomo_diagnostics,
                shared_dir / "auroc_ensemble_shared_lomo_summary.parquet":
                    shared_lomo_summary,
            }
        )
    _stage(f"writing {len(files)} parquet tables and two Markdown reports")
    for path, frame in files.items():
        _write_parquet(frame, path, overwrite)
    write_report(
        best_selection,
        shared_selection,
        segment_selection,
        shared_segment_selection,
        shared_lomo_summary,
        output_root / "results" / "auroc_ensemble_selection.md",
        includes_attn_rollout=include_attn_rollout,
    )
    write_segment_report(
        best_segment_summary,
        best_contrasts,
        shared_segment_summary,
        output_root / "results" / "auroc_ensemble_segment_analysis.md",
    )

    _stage(
        f"run complete in {time.perf_counter() - run_started:.1f}s; "
        f"outputs are under {output_root}"
    )

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASET_MODELS))
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_INFO))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-attn-rollout", action="store_true")
    parser.add_argument("--skip-shared-validation", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    datasets = set(args.datasets) if args.datasets else None
    models = set(args.models) if args.models else None
    pairs = selected_pairs(datasets, models)
    if args.include_attn_rollout:
        if datasets not in (None, {"opi"}) or any(dataset != "opi" for dataset, _ in pairs):
            raise SystemExit("--include-attn-rollout is an OPI-only sensitivity")
        pairs = [pair for pair in pairs if pair[0] == "opi"]
    output_root = args.output_root or (
        SENSITIVITY_DIR if args.include_attn_rollout else METHOD_DIR
    )
    resolved = output_root.resolve()
    if ANALYSIS_DIR.resolve() not in (resolved, *resolved.parents):
        raise SystemExit("all outputs must remain under token_analysis")
    _run(
        input_dir=args.input_dir,
        pairs=pairs,
        output_root=output_root,
        include_attn_rollout=args.include_attn_rollout,
        folds=args.folds,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
        overwrite=args.overwrite,
        skip_shared_validation=args.skip_shared_validation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
