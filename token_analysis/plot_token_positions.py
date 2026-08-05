"""Plot case-balanced selected-metric relevance across token positions.

The input is the long-format parquet produced by build_token_position_data.py.
Scores are the stored direction-aligned within-case percentiles: they are
relative ranks in [0, 1], not probabilities. Tokens are reduced to case/bin
means before cases are averaged or bootstrapped, so long transcripts cannot
dominate the plotted estimand. Post-response trailer tokens remain in the
canonical all-token table for provenance but are excluded from every positional
summary and figure.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from tqdm.auto import tqdm

try:
    from .common import (
        ANALYSIS_DIR,
        CONTENT_WINNERS,
        AUROC_ENSEMBLE_DIR,
        AUROC_ENSEMBLE_BEST_RESULTS_DIR,
        AUROC_ENSEMBLE_SHARED_RESULTS_DIR,
        AUROC_ENSEMBLE_SELECTION_PATH,
        AUROC_ENSEMBLE_SHARED_SELECTION_PATH,
        COR_BEST_RESULTS_DIR,
        COR_SELECTION_PATH,
        COR_SHARED_RESULTS_DIR,
        COR_SHARED_SELECTION_PATH,
        CV_BEST_RESULTS_DIR,
        CV_METHOD_DIR,
        CV_SELECTION_PATH,
        CV_SHARED_RESULTS_DIR,
        CV_SHARED_SELECTION_PATH,
        DATASET_MODELS,
        METRIC_REGIMES,
        MODEL_INFO,
        POOLED_AUROC_BEST_RESULTS_DIR,
        POOLED_AUROC_DIR,
        POOLED_AUROC_SHARED_RESULTS_DIR,
        SPEARMAN_METHOD_DIR,
        metric_specs,
        no_hd_dir,
    )
except ImportError:
    from common import (  # type: ignore[no-redef]
        ANALYSIS_DIR,
        CONTENT_WINNERS,
        COR_BEST_RESULTS_DIR,
        AUROC_ENSEMBLE_DIR,
        AUROC_ENSEMBLE_BEST_RESULTS_DIR,
        AUROC_ENSEMBLE_SHARED_RESULTS_DIR,
        AUROC_ENSEMBLE_SELECTION_PATH,
        AUROC_ENSEMBLE_SHARED_SELECTION_PATH,
        COR_SELECTION_PATH,
        COR_SHARED_RESULTS_DIR,
        COR_SHARED_SELECTION_PATH,
        CV_BEST_RESULTS_DIR,
        CV_METHOD_DIR,
        CV_SELECTION_PATH,
        CV_SHARED_RESULTS_DIR,
        CV_SHARED_SELECTION_PATH,
        DATASET_MODELS,
        METRIC_REGIMES,
        MODEL_INFO,
        POOLED_AUROC_BEST_RESULTS_DIR,
        POOLED_AUROC_DIR,
        POOLED_AUROC_SHARED_RESULTS_DIR,
        SPEARMAN_METHOD_DIR,
        metric_specs,
        no_hd_dir,
    )


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

SEGMENT_ORDER = {"input": 0, "boundary": 1, "output": 2}
DATASET_NAMES = {
    "opi": "OpenPromptInjection",
    "tt": "Tensor Trust",
    "liars": "Liars' Bench",
    "taboo": "Taboo",
}
MODEL_COLORS = {
    "q7": "#0072B2",
    "g12": "#009E73",
    "g27": "#D55E00",
    "l70": "#CC79A7",
}
SEGMENT_BACKGROUNDS = {
    "input": "#F2F6FC",
    "boundary": "#FFF6E5",
    "output": "#EFF8F3",
}
CONTEXT_COLORS = {
    "system": "#56B4E9",
    "user": "#E69F00",
    "assistant_prior": "#CC79A7",
    "template": "#9AA3B2",
    "assistant": "#009E73",
    "response": "#009E73",
}
CONTEXT_LABELS = {
    "system": "system",
    "user": "user",
    "assistant_prior": "prior assistant",
    "template": "template",
    "assistant": "assistant",
    "response": "response",
}
REGION_FRACTIONS = {
    "system": "system_fraction",
    "user": "user_fraction",
    "assistant_prior": "assistant_prior_fraction",
    "template": "template_fraction",
    "assistant": "assistant_fraction",
}
ANALYSIS_REGION_FRACTIONS = {region: f"analysis_{region}_fraction" for region in REGION_FRACTIONS}
ORIGIN_FRACTIONS = {
    "stored_prefix": "stored_prefix_fraction",
    "reconstructed_tail": "reconstructed_tail_fraction",
    "dataset_original": "dataset_original_fraction",
}
RESPONSE_ORIGINS = tuple(ORIGIN_FRACTIONS)
REQUIRED_COLUMNS = {
    "dataset",
    "model",
    "model_name",
    "case_id",
    "segment",
    "segment_position",
    "segment_position_normalized",
    "metric_name",
    "metric_auroc",
    "metric_direction",
    "metric_percentile_within_case",
    "source_region",
    "analysis_region",
    "response_token_origin",
    "is_template",
    "is_reconstructed_tail",
    "is_extreme_norm",
    "on_task",
    "metric_winner_changes_content_only",
}
COR_COLUMN_ALIASES = {
    "cor_metric_name": "metric_name",
    "cor_metric_auroc": "metric_auroc",
    "cor_metric_direction": "metric_direction",
    "cor_metric_value_raw": "metric_value_raw",
    "cor_metric_available": "metric_available",
    "cor_metric_value_aligned": "metric_value_aligned",
    "cor_metric_z_within_case": "metric_z_within_case",
    "cor_metric_percentile_within_case": "metric_percentile_within_case",
    "cor_metric_winner_changes_content_only": "metric_winner_changes_content_only",
}
COR_SHARED_COLUMN_ALIASES = {
    "cor_shared_metric_name": "metric_name",
    "cor_shared_metric_auroc": "metric_auroc",
    "cor_shared_metric_direction": "metric_direction",
    "cor_shared_metric_value_raw": "metric_value_raw",
    "cor_shared_metric_available": "metric_available",
    "cor_shared_metric_value_aligned": "metric_value_aligned",
    "cor_shared_metric_z_within_case": "metric_z_within_case",
    "cor_shared_metric_percentile_within_case": "metric_percentile_within_case",
    "cor_shared_metric_winner_changes_content_only": "metric_winner_changes_content_only",
}
CV_SHARED_COLUMN_ALIASES = {
    "cv_shared_metric_name": "metric_name",
    "cv_shared_metric_auroc": "metric_auroc",
    "cv_shared_metric_direction": "metric_direction",
    "cv_shared_metric_value_raw": "metric_value_raw",
    "cv_shared_metric_available": "metric_available",
    "cv_shared_metric_value_aligned": "metric_value_aligned",
    "cv_shared_metric_z_within_case": "metric_z_within_case",
    "cv_shared_metric_percentile_within_case": "metric_percentile_within_case",
    "cv_shared_metric_winner_changes_content_only": "metric_winner_changes_content_only",
}
CV_COLUMN_ALIASES = {
    "cv_metric_name": "metric_name",
    "cv_metric_auroc": "metric_auroc",
    "cv_metric_direction": "metric_direction",
    "cv_metric_value_raw": "metric_value_raw",
    "cv_metric_available": "metric_available",
    "cv_metric_value_aligned": "metric_value_aligned",
    "cv_metric_z_within_case": "metric_z_within_case",
    "cv_metric_percentile_within_case": "metric_percentile_within_case",
    "cv_metric_winner_changes_content_only": "metric_winner_changes_content_only",
}
SHARED_COLUMN_ALIASES = {
    "shared_metric_name": "metric_name",
    "shared_metric_auroc": "metric_auroc",
    "shared_metric_direction": "metric_direction",
    "shared_metric_value_raw": "metric_value_raw",
    "shared_metric_available": "metric_available",
    "shared_metric_value_aligned": "metric_value_aligned",
    "shared_metric_z_within_case": "metric_z_within_case",
    "shared_metric_percentile_within_case": "metric_percentile_within_case",
    "shared_metric_winner_changes_content_only": "metric_winner_changes_content_only",
}
ENSEMBLE_COLUMN_ALIASES = {
    "auroc_ensemble_metric_name": "metric_name",
    "auroc_ensemble_metric_auroc": "metric_auroc",
    "auroc_ensemble_metric_direction": "metric_direction",
    "auroc_ensemble_metric_value_raw": "metric_value_raw",
    "auroc_ensemble_metric_available": "metric_available",
    "auroc_ensemble_metric_value_aligned": "metric_value_aligned",
    "auroc_ensemble_metric_z_within_case": "metric_z_within_case",
    "auroc_ensemble_metric_percentile_within_case": "metric_percentile_within_case",
    "auroc_ensemble_metric_winner_changes_common_support": "metric_winner_changes_content_only",
}
ENSEMBLE_SHARED_COLUMN_ALIASES = {
    "auroc_ensemble_shared_metric_name": "metric_name",
    "auroc_ensemble_shared_metric_auroc": "metric_auroc",
    "auroc_ensemble_shared_metric_direction": "metric_direction",
    "auroc_ensemble_shared_metric_value_raw": "metric_value_raw",
    "auroc_ensemble_shared_metric_available": "metric_available",
    "auroc_ensemble_shared_metric_value_aligned": "metric_value_aligned",
    "auroc_ensemble_shared_metric_z_within_case": "metric_z_within_case",
    "auroc_ensemble_shared_metric_percentile_within_case": "metric_percentile_within_case",
    "auroc_ensemble_shared_metric_winner_changes_common_support": "metric_winner_changes_content_only",
}
CASE_KEYS = ("dataset", "model", "case_id", "segment", "axis_kind", "bin_index")
PLOT_KEYS = ("dataset", "model", "segment", "axis_kind", "bin_index")


def fractional_ranks(values: np.ndarray, *, higher_is_relevant: bool = True) -> np.ndarray:
    """Reference midrank normalization used by tests and input validation."""
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    n = int(finite.sum())
    if n < 2:
        return result
    scored = values[finite] if higher_is_relevant else -values[finite]
    order = np.argsort(scored, kind="mergesort")
    sorted_values = scored[order]
    ranks = np.empty(n, dtype=float)
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    result[finite] = ranks / (n - 1)
    return result


def _scan_input(
    path: Path,
    datasets: set[str],
    metric_regime: str,
) -> pl.LazyFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pl.scan_parquet(path)
    columns = set(frame.collect_schema().names())
    aliases = (
        SHARED_COLUMN_ALIASES
        if metric_regime == "dataset-shared"
        else CV_COLUMN_ALIASES
        if metric_regime == "cv-best"
        else CV_SHARED_COLUMN_ALIASES
        if metric_regime == "cv-shared"
        else COR_COLUMN_ALIASES
        if metric_regime == "cor-best"
        else COR_SHARED_COLUMN_ALIASES
        if metric_regime == "cor-shared"
        else ENSEMBLE_COLUMN_ALIASES
        if metric_regime == "auroc-ensemble-best"
        else ENSEMBLE_SHARED_COLUMN_ALIASES
        if metric_regime == "auroc-ensemble-shared"
        else {}
    )
    required = REQUIRED_COLUMNS | set(aliases)
    if metric_regime == "cv-best":
        required |= {
            "cv_metric_selection_score",
            "cv_metric_nested_cv_macro_ap",
            "cv_metric_statistically_tied",
            "cv_metric_demonstrated_incremental_value",
        }
    if metric_regime == "cv-shared":
        required |= {
            "cv_shared_metric_selection_score",
            "cv_shared_metric_model_cv_macro_ap",
            "cv_shared_metric_nested_cv_shared_macro_ap",
            "cv_shared_metric_statistically_tied",
            "cv_shared_metric_demonstrated_incremental_value",
        }
    if metric_regime == "cor-best":
        required |= {
            "cor_metric_spearman_rho",
            "cor_metric_abs_spearman_rho",
            "cor_metric_statistically_tied",
        }
    if metric_regime == "cor-shared":
        required |= {
            "cor_shared_metric_spearman_rho",
            "cor_shared_metric_abs_spearman_rho",
            "cor_shared_metric_equal_model_abs_spearman_rho",
            "cor_shared_metric_statistically_tied",
        }
    if metric_regime == "auroc-ensemble-best":
        required |= {
            "auroc_ensemble_metric_gain_over_best_original",
            "auroc_ensemble_metric_heldout_pooled_auroc",
            "auroc_ensemble_metric_heldout_case_macro_delta",
            "auroc_ensemble_metric_heldout_delta_ci_low",
            "auroc_ensemble_metric_heldout_delta_ci_high",
        }
    if metric_regime == "auroc-ensemble-shared":
        required |= {"auroc_ensemble_shared_metric_equal_model_auroc"}
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    if aliases:
        frame = frame.with_columns(
            pl.col(source).alias(target) for source, target in aliases.items()
        )
    if metric_regime == "cv-best":
        frame = frame.with_columns(
            pl.col("cv_metric_selection_score").alias("metric_selection_score"),
            pl.col("cv_metric_selection_score").alias("metric_model_score"),
            pl.col("cv_metric_nested_cv_macro_ap").alias("metric_nested_score"),
            pl.col("cv_metric_statistically_tied").alias("metric_statistically_tied"),
            pl.lit(None, dtype=pl.Float64).alias("metric_spearman_rho"),
            pl.col("cv_metric_demonstrated_incremental_value").alias(
                "metric_demonstrated_incremental_value"
            ),
        )
    elif metric_regime == "cv-shared":
        frame = frame.with_columns(
            pl.col("cv_shared_metric_selection_score").alias("metric_selection_score"),
            pl.col("cv_shared_metric_model_cv_macro_ap").alias("metric_model_score"),
            pl.col("cv_shared_metric_nested_cv_shared_macro_ap").alias("metric_nested_score"),
            pl.col("cv_shared_metric_statistically_tied").alias("metric_statistically_tied"),
            pl.lit(None, dtype=pl.Float64).alias("metric_spearman_rho"),
            pl.col("cv_shared_metric_demonstrated_incremental_value").alias(
                "metric_demonstrated_incremental_value"
            ),
        )
    elif metric_regime == "cor-best":
        frame = frame.with_columns(
            pl.col("cor_metric_abs_spearman_rho").alias("metric_selection_score"),
            pl.col("cor_metric_abs_spearman_rho").alias("metric_model_score"),
            pl.lit(None, dtype=pl.Float64).alias("metric_nested_score"),
            pl.col("cor_metric_statistically_tied").alias("metric_statistically_tied"),
            pl.col("cor_metric_spearman_rho").alias("metric_spearman_rho"),
            pl.lit(True).alias("metric_demonstrated_incremental_value"),
        )
    elif metric_regime == "cor-shared":
        frame = frame.with_columns(
            pl.col("cor_shared_metric_equal_model_abs_spearman_rho").alias(
                "metric_selection_score"
            ),
            pl.col("cor_shared_metric_abs_spearman_rho").alias("metric_model_score"),
            pl.lit(None, dtype=pl.Float64).alias("metric_nested_score"),
            pl.col("cor_shared_metric_statistically_tied").alias("metric_statistically_tied"),
            pl.col("cor_shared_metric_spearman_rho").alias("metric_spearman_rho"),
            pl.lit(True).alias("metric_demonstrated_incremental_value"),
        )
    elif metric_regime == "auroc-ensemble-best":
        frame = frame.with_columns(
            pl.col("auroc_ensemble_metric_auroc").alias("metric_selection_score"),
            pl.col("auroc_ensemble_metric_heldout_pooled_auroc").alias("metric_model_score"),
            pl.col("auroc_ensemble_metric_heldout_case_macro_delta").alias("metric_nested_score"),
            ((pl.col("auroc_ensemble_metric_heldout_delta_ci_low") <= 0) & (pl.col("auroc_ensemble_metric_heldout_delta_ci_high") >= 0)).alias(
                "metric_statistically_tied"
            ),
            pl.lit(None, dtype=pl.Float64).alias("metric_spearman_rho"),
            (pl.col("auroc_ensemble_metric_heldout_delta_ci_low") > 0).alias(
                "metric_demonstrated_incremental_value"
            ),
        )
    elif metric_regime == "auroc-ensemble-shared":
        frame = frame.with_columns(
            pl.col("auroc_ensemble_shared_metric_equal_model_auroc").alias("metric_selection_score"),
            pl.col("auroc_ensemble_shared_metric_auroc").alias("metric_model_score"),
            pl.lit(None, dtype=pl.Float64).alias("metric_nested_score"),
            pl.lit(False).alias("metric_statistically_tied"),
            pl.lit(None, dtype=pl.Float64).alias("metric_spearman_rho"),
            pl.lit(True).alias("metric_demonstrated_incremental_value"),
        )
    else:
        frame = frame.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("metric_selection_score"),
            pl.lit(None, dtype=pl.Float64).alias("metric_model_score"),
            pl.lit(None, dtype=pl.Float64).alias("metric_nested_score"),
            pl.lit(False).alias("metric_statistically_tied"),
            pl.lit(None, dtype=pl.Float64).alias("metric_spearman_rho"),
            pl.lit(True).alias("metric_demonstrated_incremental_value"),
        )
    return frame.filter(pl.col("dataset").is_in(sorted(datasets))).with_columns(
        pl.lit(metric_regime).alias("metric_regime")
    )


def validate_plot_input(frame: pl.LazyFrame) -> None:
    """Fail on invalid percentiles or cases without two finite selected scores."""
    score = pl.col("metric_percentile_within_case")
    invalid_range = (
        frame.filter(score.is_not_null() & (~score.is_finite() | (score < 0) | (score > 1)))
        .select(pl.len())
        .collect()
        .item()
    )
    finite_counts = (
        frame.group_by("dataset", "model", "case_id")
        .agg(score.is_finite().fill_null(False).sum().alias("finite_scores"))
        .filter(pl.col("finite_scores") < 2)
        .collect()
    )
    if invalid_range:
        raise ValueError(f"{invalid_range} percentile values fall outside [0, 1]")
    if not finite_counts.is_empty():
        examples = finite_counts.head(5).select("dataset", "model", "case_id").rows()
        raise ValueError(f"cases with fewer than two finite selected scores: {examples}")


def case_bin_summaries(frame: pl.LazyFrame, normalized_bins: int) -> pl.DataFrame:
    """Reduce token rows to one equally weighted row per case/position bin."""
    if normalized_bins < 2:
        raise ValueError("normalized_bins must be at least 2")
    columns = set(frame.collect_schema().names())
    additions = []
    if "metric_regime" not in columns:
        additions.append(pl.lit("model-best").alias("metric_regime"))
    if "metric_selection_score" not in columns:
        additions.append(pl.lit(None, dtype=pl.Float64).alias("metric_selection_score"))
    if "metric_model_score" not in columns:
        additions.append(pl.lit(None, dtype=pl.Float64).alias("metric_model_score"))
    if "metric_nested_score" not in columns:
        additions.append(pl.lit(None, dtype=pl.Float64).alias("metric_nested_score"))
    if "metric_statistically_tied" not in columns:
        additions.append(pl.lit(False).alias("metric_statistically_tied"))
    if "metric_spearman_rho" not in columns:
        additions.append(pl.lit(None, dtype=pl.Float64).alias("metric_spearman_rho"))
    if "metric_demonstrated_incremental_value" not in columns:
        additions.append(pl.lit(True).alias("metric_demonstrated_incremental_value"))
    if additions:
        frame = frame.with_columns(*additions)
    # Liars post-response turn terminators are source provenance, not candidate
    # token locations for the positional analysis.
    frame = frame.filter(pl.col("segment").is_in(tuple(SEGMENT_ORDER)))
    normalized_segment = pl.col("segment").is_in(["input", "output"])
    score = (
        pl.when(pl.col("metric_percentile_within_case").is_finite().fill_null(False))
        .then(pl.col("metric_percentile_within_case"))
        .otherwise(None)
    )
    binned = frame.with_columns(
        pl.when(normalized_segment)
        .then(
            (pl.col("segment_position_normalized") * normalized_bins)
            .floor()
            .clip(0, normalized_bins - 1)
        )
        .otherwise(pl.col("segment_position"))
        .cast(pl.Int64)
        .alias("bin_index"),
        pl.when(normalized_segment)
        .then(pl.lit("normalized"))
        .otherwise(pl.lit("ordinal"))
        .alias("axis_kind"),
        score.alias("_score"),
    )
    fraction_exprs = [
        (pl.col("source_region") == region).cast(pl.Float64).mean().alias(column)
        for region, column in REGION_FRACTIONS.items()
    ]
    fraction_exprs.extend(
        (pl.col("analysis_region") == region).cast(pl.Float64).mean().alias(column)
        for region, column in ANALYSIS_REGION_FRACTIONS.items()
    )
    fraction_exprs.extend(
        (pl.col("response_token_origin") == origin).cast(pl.Float64).mean().alias(column)
        for origin, column in ORIGIN_FRACTIONS.items()
    )
    fraction_exprs.append(
        pl.col("response_token_origin")
        .is_in(RESPONSE_ORIGINS)
        .cast(pl.Float64)
        .mean()
        .alias("response_fraction")
    )
    return (
        binned.group_by(*CASE_KEYS)
        .agg(
            pl.col("model_name").first(),
            pl.col("metric_regime").first(),
            pl.col("metric_name").first(),
            pl.col("metric_auroc").first(),
            pl.col("metric_direction").first(),
            pl.col("metric_selection_score").first(),
            pl.col("metric_model_score").first(),
            pl.col("metric_nested_score").first(),
            pl.col("metric_statistically_tied").first(),
            pl.col("metric_spearman_rho").first(),
            pl.col("metric_demonstrated_incremental_value").first(),
            pl.col("metric_winner_changes_content_only").first(),
            pl.col("_score").mean().alias("case_mean_percentile"),
            pl.col("_score").median().alias("case_median_percentile"),
            pl.col("_score").count().cast(pl.Int64).alias("case_metric_tokens"),
            pl.len().cast(pl.Int64).alias("case_tokens"),
            pl.col("on_task").mean().alias("case_on_task_rate"),
            pl.col("is_template").cast(pl.Float64).mean().alias("case_template_fraction"),
            pl.col("is_reconstructed_tail")
            .cast(pl.Float64)
            .mean()
            .alias("case_reconstructed_fraction"),
            pl.col("is_extreme_norm").cast(pl.Float64).mean().alias("case_extreme_fraction"),
            *fraction_exprs,
        )
        .collect()
    )


def aggregate_case_bins(
    cases: pl.DataFrame,
    normalized_bins: int,
    *,
    bootstrap_resamples: int,
    seed: int,
) -> pl.DataFrame:
    """Average case/bin rows and attach whole-case bootstrap intervals."""
    if cases.is_empty():
        raise ValueError("no case/bin rows to aggregate")
    segment_cases = (
        cases.select("dataset", "model", "segment", "case_id")
        .unique()
        .group_by("dataset", "model", "segment")
        .len()
        .rename({"len": "segment_case_count"})
    )
    fraction_columns = [
        "case_reconstructed_fraction",
        "case_extreme_fraction",
        *REGION_FRACTIONS.values(),
        *ANALYSIS_REGION_FRACTIONS.values(),
        *ORIGIN_FRACTIONS.values(),
        "response_fraction",
    ]
    aggregate = (
        cases.group_by(*PLOT_KEYS)
        .agg(
            pl.col("model_name").first(),
            pl.col("metric_regime").first(),
            pl.col("metric_name").first(),
            pl.col("metric_auroc").first(),
            pl.col("metric_direction").first(),
            pl.col("metric_selection_score").first(),
            pl.col("metric_model_score").first(),
            pl.col("metric_nested_score").first(),
            pl.col("metric_statistically_tied").first(),
            pl.col("metric_spearman_rho").first(),
            pl.col("metric_demonstrated_incremental_value").first(),
            pl.col("metric_winner_changes_content_only").first(),
            pl.col("case_mean_percentile").mean().alias("mean_percentile"),
            pl.col("case_mean_percentile").median().alias("median_percentile"),
            pl.col("case_mean_percentile").count().cast(pl.Int64).alias("n_cases"),
            pl.len().cast(pl.Int64).alias("n_cases_with_tokens"),
            pl.col("case_tokens").sum().cast(pl.Int64).alias("n_tokens"),
            pl.col("case_metric_tokens").sum().cast(pl.Int64).alias("n_metric_tokens"),
            pl.col("case_on_task_rate").mean().alias("mean_on_task_rate"),
            *[
                pl.col(column).mean().alias(column.removeprefix("case_"))
                for column in fraction_columns
            ],
        )
        .join(segment_cases, on=["dataset", "model", "segment"], how="left")
        .with_columns(
            (pl.col("n_cases") / pl.col("segment_case_count")).alias("case_coverage"),
            pl.when(pl.col("axis_kind") == "normalized")
            .then(pl.col("bin_index") / normalized_bins)
            .otherwise(None)
            .alias("bin_left"),
            pl.when(pl.col("axis_kind") == "normalized")
            .then((pl.col("bin_index") + 1) / normalized_bins)
            .otherwise(None)
            .alias("bin_right"),
            pl.when(pl.col("axis_kind") == "normalized")
            .then((pl.col("bin_index") + 0.5) / normalized_bins)
            .otherwise(pl.col("bin_index") + 1)
            .alias("position_x"),
            pl.when(pl.col("axis_kind") == "ordinal")
            .then(pl.col("bin_index") + 1)
            .otherwise(None)
            .cast(pl.Int64)
            .alias("position_ordinal"),
        )
    )
    intervals = bootstrap_intervals(
        cases,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )
    aggregate = aggregate.join(intervals, on=list(PLOT_KEYS), how="left")
    labels = [
        (
            f"{round(float(left) * 100):d}–{round(float(right) * 100):d}%"
            if kind == "normalized"
            else str(int(index) + 1)
        )
        for kind, index, left, right in aggregate.select(
            "axis_kind", "bin_index", "bin_left", "bin_right"
        ).iter_rows()
    ]
    return aggregate.with_columns(pl.Series("position_label", labels, dtype=pl.String)).sort(
        pl.col("dataset"),
        pl.col("model"),
        pl.col("segment").replace_strict(SEGMENT_ORDER, return_dtype=pl.Int64),
        pl.col("bin_index"),
    )


def bootstrap_intervals(
    cases: pl.DataFrame,
    *,
    bootstrap_resamples: int,
    seed: int,
) -> pl.DataFrame:
    """Pointwise intervals from resampling complete case vectors."""
    schema = {
        **{column: pl.String for column in PLOT_KEYS[:-1]},
        "bin_index": pl.Int64,
        "ci_low": pl.Float64,
        "ci_high": pl.Float64,
    }
    if bootstrap_resamples <= 0:
        return pl.DataFrame(schema=schema)
    output: list[dict[str, object]] = []
    pairs = sorted(cases.select("dataset", "model").unique().iter_rows())
    bootstrap_progress = tqdm(
        enumerate(pairs),
        total=len(pairs),
        desc="Bootstrapping positional intervals",
        unit="combination",
        leave=False,
        dynamic_ncols=True,
    )
    for pair_index, (dataset, model) in bootstrap_progress:
        bootstrap_progress.set_postfix_str(f"{dataset}/{model}")
        pair = cases.filter((pl.col("dataset") == dataset) & (pl.col("model") == model))
        case_ids = sorted(pair["case_id"].unique().to_list())
        keys = sorted(
            {
                (segment, axis_kind, int(bin_index))
                for segment, axis_kind, bin_index in pair.select(
                    "segment", "axis_kind", "bin_index"
                ).iter_rows()
            },
            key=lambda key: (SEGMENT_ORDER[key[0]], key[2]),
        )
        case_lookup = {case_id: index for index, case_id in enumerate(case_ids)}
        key_lookup = {key: index for index, key in enumerate(keys)}
        matrix = np.full((len(case_ids), len(keys)), np.nan, dtype=np.float64)
        for row in pair.select(
            "case_id", "segment", "axis_kind", "bin_index", "case_mean_percentile"
        ).iter_rows(named=True):
            value = row["case_mean_percentile"]
            if value is not None and math.isfinite(float(value)):
                matrix[
                    case_lookup[row["case_id"]],
                    key_lookup[(row["segment"], row["axis_kind"], int(row["bin_index"]))],
                ] = float(value)

        rng = np.random.default_rng(seed + pair_index)
        probabilities = np.full(len(case_ids), 1 / len(case_ids))
        weights = rng.multinomial(len(case_ids), probabilities, size=bootstrap_resamples).astype(
            np.float64
        )
        valid = np.isfinite(matrix)
        values = np.where(valid, matrix, 0.0)
        numerator = weights @ values
        denominator = weights @ valid.astype(np.float64)
        replicates = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        )
        low, high = np.nanquantile(replicates, [0.025, 0.975], axis=0)
        valid_counts = valid.sum(axis=0)
        for key_index, (segment, axis_kind, bin_index) in enumerate(keys):
            output.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "segment": segment,
                    "axis_kind": axis_kind,
                    "bin_index": bin_index,
                    "ci_low": (float(low[key_index]) if valid_counts[key_index] >= 2 else None),
                    "ci_high": (float(high[key_index]) if valid_counts[key_index] >= 2 else None),
                }
            )
    return pl.DataFrame(output, schema=schema)


def build_plot_summary(
    input_path: Path,
    datasets: set[str],
    *,
    normalized_bins: int,
    bootstrap_resamples: int,
    seed: int,
    metric_regime: str = "model-best",
) -> pl.DataFrame:
    frame = _scan_input(input_path, datasets, metric_regime)
    validate_plot_input(frame)
    cases = case_bin_summaries(frame, normalized_bins)
    return aggregate_case_bins(
        cases,
        normalized_bins,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )


def _load_matplotlib() -> tuple[object, object, object, object]:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "axes.titleweight": "bold",
        }
    )
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    return plt, matplotlib, Line2D, Patch


def _context_categories(dataset: str, segment: str) -> list[tuple[str, str]]:
    if segment == "output":
        return [("response", "response_fraction")]
    return [
        ("system", ANALYSIS_REGION_FRACTIONS["system"]),
        ("user", ANALYSIS_REGION_FRACTIONS["user"]),
        ("assistant_prior", ANALYSIS_REGION_FRACTIONS["assistant_prior"]),
        ("template", ANALYSIS_REGION_FRACTIONS["template"]),
    ]


def _draw_context_strip(
    ax: Axes,
    rows: pl.DataFrame,
    dataset: str,
    segment: str,
    used_categories: set[str],
) -> None:
    context_ax = ax.inset_axes([0, -0.255, 1, 0.095])
    x = rows["position_x"].to_numpy()
    widths = np.full(len(x), 0.048 if rows["axis_kind"][0] == "normalized" else 0.82)
    bottom = np.zeros(len(x))
    for category, column in _context_categories(dataset, segment):
        values = np.nan_to_num(rows[column].to_numpy().astype(float), nan=0.0)
        if values.max(initial=0.0) <= 0:
            continue
        context_ax.bar(
            x,
            values,
            width=widths,
            bottom=bottom,
            color=CONTEXT_COLORS[category],
            edgecolor="none",
            align="center",
        )
        bottom += values
        used_categories.add(category)
    context_ax.set_xlim(ax.get_xlim())
    context_ax.set_ylim(0, 1)
    context_ax.set_axis_off()


def make_dataset_figure(
    summary: pl.DataFrame,
    dataset: str,
    *,
    normalized_bins: int,
) -> Figure:
    """Render one dataset figure with model rows and segment columns."""
    plt, _, Line2D, Patch = _load_matplotlib()
    data = summary.filter(pl.col("dataset") == dataset)
    if data.is_empty():
        raise ValueError(f"no summary rows for dataset {dataset}")
    metric_regime = str(data["metric_regime"][0])
    models = [model for model in DATASET_MODELS[dataset] if model in set(data["model"])]
    segments = [segment for segment in SEGMENT_ORDER if segment in set(data["segment"])]
    figure_width = max(10.5, 3.55 * len(segments))
    liars_layout = dataset == "liars"
    shared_validation = metric_regime == "auroc-ensemble-shared"
    figure_height = (
        2.5
        + 2.55 * len(models)
        + (0.8 if liars_layout else 0.0)
    )
    fig, axes = plt.subplots(
        len(models),
        len(segments),
        figsize=(figure_width, figure_height),
        facecolor="#F6F8FC",
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.18,
        right=0.985,
        bottom=0.21 if liars_layout else 0.15,
        top=0.84,
        hspace=0.82 if liars_layout else 0.76,
        wspace=0.18,
    )
    fig.text(
        0.04,
        0.965,
        f"{DATASET_NAMES[dataset]}: " + {
            "model-best": "selected-metric relevance by token position",
            "dataset-shared": "dataset-shared metric relevance by token position",
            "cv-best": "cross-validated best-metric relevance by token position",
            "cv-shared": "cross-validated shared-metric relevance by token position",
            "cor-best": "Spearman-selected metric relevance by token position",
            "cor-shared": "Spearman-shared metric relevance by token position",
            "auroc-ensemble-best": "AUROC-ensemble relevance by token position",
            "auroc-ensemble-shared": "dataset-shared AUROC-ensemble relevance by token position",
        }[metric_regime],
        color="#17213C",
        fontsize=18,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.04,
        0.927,
        "Mean direction-aligned within-case percentile  ·  cases weighted equally  ·  "
        "bands are descriptive pointwise 95% case-bootstrap intervals (not simultaneous tests)",
        color="#697386",
        fontsize=9.5,
        va="top",
    )

    used_categories: set[str] = set()
    for row_index, model in enumerate(models):
        model_data = data.filter(pl.col("model") == model)
        metric = str(model_data["metric_name"][0])
        auroc = float(model_data["metric_auroc"][0])
        raw_direction = str(model_data["metric_direction"][0])
        changed = bool(model_data["metric_winner_changes_content_only"][0])
        dagger = "†" if changed else ""
        incremental = bool(model_data["metric_demonstrated_incremental_value"][0])
        baseline_marker = (
            "‡" if metric_regime in {"cv-best", "cv-shared"} and not incremental else ""
        )
        direction = "↑" if raw_direction == "higher" else "↓"
        for column_index, segment in enumerate(segments):
            ax = axes[row_index, column_index]
            rows = model_data.filter(pl.col("segment") == segment).sort("bin_index")
            ax.set_facecolor(SEGMENT_BACKGROUNDS[segment])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.grid(axis="y", color="#DDE3EE", linewidth=0.7, alpha=0.75)
            ax.set_axisbelow(True)
            ax.axhline(0.5, color="#697386", linewidth=0.9, linestyle=(0, (3, 3)))
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.tick_params(colors="#697386", labelsize=8, length=0)

            if rows.is_empty():
                ax.set_visible(False)
                continue
            x = rows["position_x"].to_numpy().astype(float)
            mean = rows["mean_percentile"].to_numpy().astype(float)
            low = rows["ci_low"].to_numpy().astype(float)
            high = rows["ci_high"].to_numpy().astype(float)
            finite_band = np.isfinite(low) & np.isfinite(high)
            if finite_band.any():
                ax.fill_between(
                    x,
                    low,
                    high,
                    where=finite_band,
                    color=MODEL_COLORS[model],
                    alpha=0.18,
                    linewidth=0,
                )
            ax.plot(
                x,
                mean,
                color=MODEL_COLORS[model],
                linewidth=2.0,
                marker="o",
                markersize=3.2,
                markeredgewidth=0,
                zorder=3,
            )
            if rows["axis_kind"][0] == "normalized":
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
                if row_index == len(models) - 1:
                    ax.set_xlabel("Relative position", color="#697386", fontsize=8)
            else:
                maximum = int(rows["position_ordinal"].max())
                ax.set_xlim(0.5, maximum + 0.5)
                ax.set_xticks(range(1, maximum + 1))
                if row_index == len(models) - 1:
                    ax.set_xlabel("Token ordinal", color="#697386", fontsize=8)

            n_min = int(rows["n_cases"].min())
            n_max = int(rows["n_cases"].max())
            case_label = f"n={n_min:,}" if n_min == n_max else f"n={n_min:,}–{n_max:,}/bin"
            ax.text(
                0.985,
                0.96,
                case_label,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.2,
                color="#697386",
            )
            if row_index == 0:
                ax.set_title(segment.capitalize(), color="#17213C", fontsize=11, pad=11)
            if column_index == 0:
                if metric_regime == "cv-best":
                    selection_score = float(model_data["metric_selection_score"][0])
                    tied = " (tied)" if bool(model_data["metric_statistically_tied"][0]) else ""
                    metric_line = (
                        f"{metric}  raw {direction}  CV MAP {selection_score:.3f}{tied}\n"
                        f"pooled AUROC {auroc:.3f}"
                    )
                elif metric_regime == "cv-shared":
                    shared_score = float(model_data["metric_selection_score"][0])
                    model_score = float(model_data["metric_model_score"][0])
                    tied = " (tied)" if bool(model_data["metric_statistically_tied"][0]) else ""
                    metric_line = (
                        f"{metric}  raw {direction}  shared CV MAP {shared_score:.3f}{tied}\n"
                        f"model CV MAP {model_score:.3f} · AUROC {auroc:.3f}"
                    )
                elif metric_regime == "cor-best":
                    rho = float(model_data["metric_spearman_rho"][0])
                    tied = " (tied)" if bool(model_data["metric_statistically_tied"][0]) else ""
                    metric_line = (
                        f"{metric}  raw {direction}  Spearman ρ {rho:+.3f}{tied}\n"
                        f"|ρ| {abs(rho):.3f} · AUROC {auroc:.3f}"
                    )
                elif metric_regime == "cor-shared":
                    shared_score = float(model_data["metric_selection_score"][0])
                    rho = float(model_data["metric_spearman_rho"][0])
                    tied = " (tied)" if bool(model_data["metric_statistically_tied"][0]) else ""
                    metric_line = (
                        f"{metric}  raw {direction}  shared mean |ρ| {shared_score:.3f}{tied}\n"
                        f"model ρ {rho:+.3f} · AUROC {auroc:.3f}"
                    )
                elif metric_regime == "auroc-ensemble-best":
                    heldout = float(model_data["metric_model_score"][0])
                    display = metric.replace("mix::", "").replace("single::", "")
                    metric_line = (
                        f"{display}  AUROC {auroc:.3f}\n"
                        f"held-out AUROC {heldout:.3f} · final {direction}"
                    )
                elif metric_regime == "auroc-ensemble-shared":
                    model_score = float(model_data["metric_model_score"][0])
                    heldout_pooled = float(
                        model_data["shared_heldout_model_pooled_auroc"][0]
                    )
                    display = metric.replace("mix::", "").replace("single::", "")
                    metric_line = (
                        f"{display}  AUROC {model_score:.3f}\n"
                        f"held-out AUROC {heldout_pooled:.3f} · final {direction}"
                    )
                else:
                    metric_line = f"{metric}  raw {direction}  AUROC {auroc:.3f}"
                row_label = f"{MODEL_INFO[model]['name']}{dagger}{baseline_marker}\n{metric_line}"
                ax.text(
                    -0.18,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=8.0 if shared_validation else 8.5,
                    color="#17213C",
                    linespacing=1.30 if shared_validation else 1.35,
                )
            _draw_context_strip(ax, rows, dataset, segment, used_categories)

    fig.supylabel(
        "Within-case relevance percentile",
        x=0.017,
        color="#697386",
        fontsize=9,
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model],
            linewidth=2,
            label=MODEL_INFO[model]["name"],
        )
        for model in models
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#697386",
            linewidth=0.9,
            linestyle=(0, (3, 3)),
            label="case median rank",
        )
    )
    handles.extend(
        Patch(
            facecolor=CONTEXT_COLORS[category],
            edgecolor="none",
            label=CONTEXT_LABELS[category],
        )
        for category in CONTEXT_COLORS
        if category in used_categories
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065 if liars_layout else 0.045),
        ncol=4 if liars_layout else min(7, len(handles)),
        frameon=False,
        fontsize=7.5,
    )
    notes = [
        "Context strips show case-balanced token composition; ranks are relative, not probabilities."
    ]
    if shared_validation:
        notes.append(
            "Held-out AUROC validates fold-wise shared reselection, not the frozen "
            "full-data winner drawn here."
        )
    changes = [
        model
        for model in models
        if bool(data.filter(pl.col("model") == model)["metric_winner_changes_content_only"][0])
    ]
    if changes:
        if metric_regime in {"auroc-ensemble-best", "auroc-ensemble-shared"}:
            names = ", ".join(MODEL_INFO[model]["name"] for model in changes)
            notes.append(f"† common-finite-support winner differs: {names}.")
        else:
            labels = [
                f"{MODEL_INFO[model]['name']}: {CONTENT_WINNERS[(dataset, model)]}"
                for model in changes
            ]
            notes.append("† content-only winner: " + "; ".join(labels) + ".")
    if (
        metric_regime in {"cv-best", "cv-shared"}
        and not data["metric_demonstrated_incremental_value"].all()
    ):
        notes.append("‡ winner does not beat both cross-fitted position and region baselines.")
    if dataset == "tt":
        notes.append("TT uses all available cases; model coverage differs by at most nine cases.")
    if dataset == "liars":
        notes.append("Liars model panels are distributional, not paired by case.")
    fig.text(
        0.04,
        0.012,
        "  ".join(notes),
        color="#697386",
        fontsize=7.4,
        va="bottom",
    )
    return fig


def save_dataset_figure(
    summary: pl.DataFrame,
    dataset: str,
    *,
    output_dir: Path,
    normalized_bins: int,
    formats: tuple[str, ...],
    dpi: int,
    overwrite: bool,
) -> list[Path]:
    plt, _, _, _ = _load_matplotlib()
    figure = make_dataset_figure(summary, dataset, normalized_bins=normalized_bins)
    written: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for file_format in formats:
            regime = str(summary.filter(pl.col("dataset") == dataset)["metric_regime"][0])
            stem = {
                "model-best": "selected_metric",
                "dataset-shared": "shared_metric",
                "cv-best": "cv_best_metric",
                "cv-shared": "cv_shared_metric",
                "cor-best": "cor_best_metric",
                "cor-shared": "cor_shared_metric",
                "auroc-ensemble-best": "auroc_ensemble_best_metric",
                "auroc-ensemble-shared": "auroc_ensemble_shared_metric",
            }[regime]
            path = output_dir / f"{dataset}_{stem}_by_position.{file_format}"
            if path.exists() and not overwrite:
                raise FileExistsError(f"{path} exists; pass --overwrite")
            tmp = path.with_name(path.name + ".tmp")
            figure.savefig(
                tmp,
                format=file_format,
                dpi=dpi,
                facecolor=figure.get_facecolor(),
                bbox_inches="tight",
                pad_inches=0.15,
            )
            os.replace(tmp, path)
            written.append(path)
    finally:
        plt.close(figure)
    return written

SEGMENT_SUMMARY_COLUMNS = {
    "dataset", "model", "model_name", "metric_regime", "candidate_id", "segment",
    "token_count", "case_count", "auc_case_count", "token_base_rate",
    "case_macro_auroc", "case_macro_auroc_ci_low", "case_macro_auroc_ci_high",
    "mean_relevance_percentile", "mean_relevance_percentile_ci_low",
    "mean_relevance_percentile_ci_high", "template_fraction",
    "top_1_enrichment", "top_1_enrichment_ci_low", "top_1_enrichment_ci_high",
    "top_10_enrichment", "top_10_enrichment_ci_low", "top_10_enrichment_ci_high",
}
SHARED_VALIDATION_METADATA = {
    "adjusted_auroc": "shared_full_model_auroc",
    "equal_model_adjusted_auroc": "shared_full_equal_model_auroc",
    "shared_heldout_model_pooled_auroc": "shared_heldout_model_pooled_auroc",
    "shared_heldout_model_case_macro_auroc": "shared_heldout_model_case_macro_auroc",
    "shared_heldout_equal_model_pooled_auroc": "shared_heldout_equal_model_pooled_auroc",
    "shared_heldout_equal_model_case_macro_auroc": (
        "shared_heldout_equal_model_case_macro_auroc"
    ),
}
BEST_VALIDATION_METADATA = {
    "adjusted_auroc": "best_full_model_auroc",
    "heldout_pooled_auroc": "best_heldout_model_pooled_auroc",
}


def attach_shared_validation_metadata(
    frame: pl.DataFrame,
    selection_manifest: Path,
) -> pl.DataFrame:
    """Join grouped held-out estimates without changing the token-level schema."""
    if not selection_manifest.exists():
        raise FileNotFoundError(selection_manifest)
    manifest = pl.read_parquet(selection_manifest)
    required = {"dataset", "model", *SHARED_VALIDATION_METADATA}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(
            f"{selection_manifest} is missing shared validation columns: "
            + ", ".join(missing)
        )
    metadata = manifest.select(
        "dataset",
        "model",
        *[
            pl.col(source).cast(pl.Float64).alias(target)
            for source, target in SHARED_VALIDATION_METADATA.items()
        ],
    )
    if metadata.group_by("dataset", "model").len()["len"].max() != 1:
        raise ValueError("shared selection manifest has duplicate dataset/model rows")
    targets = list(SHARED_VALIDATION_METADATA.values())
    existing = [column for column in targets if column in frame.columns]
    joined = frame.drop(existing).join(metadata, on=["dataset", "model"], how="left")
    invalid = joined.select(
        pl.any_horizontal(
            *[
                pl.col(column).is_null() | ~pl.col(column).is_finite()
                for column in targets
            ]
        ).sum()
    ).item()
    if invalid:
        raise ValueError(
            f"shared validation metadata is incomplete for {invalid} plotted rows"
        )
    return joined


def attach_best_validation_metadata(
    frame: pl.DataFrame,
    selection_manifest: Path,
) -> pl.DataFrame:
    """Join model-specific full-data and grouped held-out pooled AUROCs."""
    if not selection_manifest.exists():
        raise FileNotFoundError(selection_manifest)
    manifest = pl.read_parquet(selection_manifest)
    required = {"dataset", "model", *BEST_VALIDATION_METADATA}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(
            f"{selection_manifest} is missing best validation columns: "
            + ", ".join(missing)
        )
    metadata = manifest.select(
        "dataset",
        "model",
        *[
            pl.col(source).cast(pl.Float64).alias(target)
            for source, target in BEST_VALIDATION_METADATA.items()
        ],
    )
    if metadata.group_by("dataset", "model").len()["len"].max() != 1:
        raise ValueError("best selection manifest has duplicate dataset/model rows")
    targets = list(BEST_VALIDATION_METADATA.values())
    existing = [column for column in targets if column in frame.columns]
    joined = frame.drop(existing).join(metadata, on=["dataset", "model"], how="left")
    invalid = joined.select(
        pl.any_horizontal(
            *[
                pl.col(column).is_null() | ~pl.col(column).is_finite()
                for column in targets
            ]
        ).sum()
    ).item()
    if invalid:
        raise ValueError(
            f"best validation metadata is incomplete for {invalid} plotted rows"
        )
    return joined


def load_segment_summary(path: Path, datasets: set[str]) -> pl.DataFrame:
    """Load selector-produced summaries, rejecting trailers and incomplete schemas."""
    if not path.exists():
        raise FileNotFoundError(path)
    summary = pl.read_parquet(path)
    missing = SEGMENT_SUMMARY_COLUMNS - set(summary.columns)
    if missing:
        raise ValueError(f"{path} is missing segment columns: {', '.join(sorted(missing))}")
    if summary.filter(pl.col("segment") == "trailer").height:
        raise ValueError("segment-comparison input must never track trailer tokens")
    return summary.filter(
        pl.col("dataset").is_in(sorted(datasets))
        & pl.col("segment").is_in(tuple(SEGMENT_ORDER))
    )


def _finite_error(value: float, low: float, high: float) -> np.ndarray | None:
    if not all(math.isfinite(item) for item in (value, low, high)):
        return None
    return np.asarray([[max(0.0, value - low)], [max(0.0, high - value)]])


def make_segment_comparison_figure(segment_summary: pl.DataFrame, dataset: str) -> Figure:
    """Plot fixed-winner discrimination, rank concentration, and enrichment."""
    plt, _, _, _ = _load_matplotlib()
    data = segment_summary.filter(pl.col("dataset") == dataset)
    if data.is_empty():
        raise ValueError(f"no segment summary rows for dataset {dataset}")
    if data.filter(pl.col("segment") == "trailer").height:
        raise ValueError("Liars trailers must not appear in segment figures")
    models = [model for model in DATASET_MODELS[dataset] if model in set(data["model"])]
    regime = str(data["metric_regime"][0])
    shared_validation = regime == "auroc-ensemble-shared"
    fig, axes = plt.subplots(
        len(models),
        3,
        figsize=(15.2, 2.55 * len(models) + 2.2),
        facecolor="#F6F8FC",
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.25,
        right=0.985,
        bottom=0.15,
        top=0.84,
        hspace=0.72,
        wspace=0.24,
    )
    fig.text(
        0.035, 0.965,
        f"{DATASET_NAMES[dataset]}: fixed {regime} winner by segment",
        fontsize=17, fontweight="bold", color="#17213C", va="top",
    )
    fig.text(
        0.035, 0.925,
        "Whole-case 95% intervals are descriptive and pointwise; segment AUROC uses the fixed all-token score.",
        fontsize=9.2, color="#697386", va="top",
    )
    for row_index, model in enumerate(models):
        rows = data.filter(pl.col("model") == model).sort(
            pl.col("segment").replace_strict(SEGMENT_ORDER)
        )
        segments = rows["segment"].to_list()
        x = np.arange(len(segments), dtype=float)
        color = MODEL_COLORS[model]
        labels = [
            f"{segment}\n{int(token_count):,} tok · {int(case_count):,} cases\n"
            f"on-task {base_rate:.2f} · tmpl {template_fraction:.2f}"
            for segment, token_count, case_count, base_rate, template_fraction in zip(
                segments,
                rows["token_count"],
                rows["case_count"],
                rows["token_base_rate"],
                rows["template_fraction"],
                strict=True,
            )
        ]
        for column_index, ax in enumerate(axes[row_index]):
            ax.set_facecolor("#FFFFFF")
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.grid(axis="y", color="#DDE3EE", linewidth=0.7)
            ax.set_xticks(x, labels, fontsize=7.2)
            ax.tick_params(colors="#697386", length=0)
            ax.set_axisbelow(True)
            if column_index == 0:
                values = rows["case_macro_auroc"].to_numpy()
                lows = rows["case_macro_auroc_ci_low"].to_numpy()
                highs = rows["case_macro_auroc_ci_high"].to_numpy()
                ax.axhline(0.5, color="#697386", linestyle=(0, (3, 3)), linewidth=0.9)
                ax.set_ylim(0, 1)
                ax.set_title("Within-segment case-macro AUROC", fontsize=10.5, color="#17213C")
                for xi, value, low, high in zip(x, values, lows, highs, strict=True):
                    if math.isfinite(float(value)):
                        ax.errorbar(xi, value, yerr=_finite_error(value, low, high), fmt="o", color=color, capsize=3)
                candidate = str(rows["candidate_id"][0]).replace("mix::", "").replace("single::", "")
                if shared_validation:
                    full_model = float(rows["shared_full_model_auroc"][0])
                    heldout = float(rows["shared_heldout_model_pooled_auroc"][0])
                else:
                    full_model = float(rows["best_full_model_auroc"][0])
                    heldout = float(rows["best_heldout_model_pooled_auroc"][0])
                ylabel = (
                    f"{MODEL_INFO[model]['name']}\n{candidate}\n"
                    f"AUROC {full_model:.3f} · held-out AUROC {heldout:.3f}"
                )
                ax.set_ylabel(
                    ylabel,
                    fontsize=7.5 if shared_validation else 8,
                    color="#17213C",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=18,
                )
                for xi, eligible in zip(x, rows["auc_case_count"], strict=True):
                    ax.text(xi, 0.03, f"eligible n={int(eligible):,}", ha="center", fontsize=6.7, color="#697386")
            elif column_index == 1:
                values = rows["mean_relevance_percentile"].to_numpy()
                lows = rows["mean_relevance_percentile_ci_low"].to_numpy()
                highs = rows["mean_relevance_percentile_ci_high"].to_numpy()
                ax.axhline(0.5, color="#697386", linestyle=(0, (3, 3)), linewidth=0.9)
                ax.set_ylim(0, 1)
                ax.set_title("Mean all-token relevance percentile", fontsize=10.5, color="#17213C")
                for xi, value, low, high in zip(x, values, lows, highs, strict=True):
                    if math.isfinite(float(value)):
                        ax.errorbar(xi, value, yerr=_finite_error(value, low, high), fmt="o", color=color, capsize=3)
            else:
                ax.axhline(1, color="#697386", linestyle=(0, (3, 3)), linewidth=0.9)
                ax.set_title("High-rank enrichment", fontsize=10.5, color="#17213C")
                for offset, prefix, marker, label in (
                    (-0.08, "top_1", "o", "top 1%"),
                    (0.08, "top_10", "s", "top 10%"),
                ):
                    values = rows[f"{prefix}_enrichment"].to_numpy()
                    lows = rows[f"{prefix}_enrichment_ci_low"].to_numpy()
                    highs = rows[f"{prefix}_enrichment_ci_high"].to_numpy()
                    for xi, value, low, high in zip(x + offset, values, lows, highs, strict=True):
                        if math.isfinite(float(value)):
                            ax.errorbar(
                                xi, value, yerr=_finite_error(value, low, high), fmt=marker,
                                color=color, markerfacecolor="white" if prefix == "top_10" else color,
                                capsize=3, label=label if xi == x[0] + offset and row_index == 0 else None,
                            )
                ax.set_ylim(bottom=0)
                if row_index == 0:
                    ax.legend(frameon=False, fontsize=7, loc="best")
    fig.text(
        0.035, 0.018,
        "Input includes earlier scaffolding and Liars prior-assistant turns; boundary is the final generation prompt; "
        "output is final assistant text. Trailer tokens are excluded. Template fraction uses corrected analysis_region.",
        fontsize=7.5, color="#697386",
    )
    return fig


def save_segment_comparison_figure(
    summary: pl.DataFrame,
    dataset: str,
    *,
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
    overwrite: bool,
) -> list[Path]:
    plt, _, _, _ = _load_matplotlib()
    figure = make_segment_comparison_figure(summary, dataset)
    regime = str(summary.filter(pl.col("dataset") == dataset)["metric_regime"][0]).replace("-", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        for file_format in formats:
            path = output_dir / f"{dataset}_{regime}_segment_comparison.{file_format}"
            if path.exists() and not overwrite:
                raise FileExistsError(f"{path} exists; pass --overwrite")
            tmp = path.with_name(path.name + ".tmp")
            figure.savefig(tmp, format=file_format, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
            os.replace(tmp, path)
            written.append(path)
    finally:
        plt.close(figure)
    return written


def _stage(message: str) -> None:
    """Emit a timestamped plotting milestone without disrupting tqdm."""
    tqdm.write(f"[{time.strftime('%H:%M:%S')}] [token-plots] {message}")


def _inside_analysis(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == ANALYSIS_DIR.resolve() or ANALYSIS_DIR.resolve() in resolved.parents


def _write_summary(summary: pl.DataFrame, path: Path, overwrite: bool) -> None:
    if not _inside_analysis(path):
        raise ValueError(f"refusing output outside token_analysis: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    summary.write_parquet(tmp, compression="zstd")
    os.replace(tmp, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exclude-head-disagreement",
        action="store_true",
        help="plot the additive no-HD token table and selections",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ANALYSIS_DIR / "token_position_scores.parquet",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument(
        "--plot-kind", choices=("positions", "segments", "both"), default="positions",
        help="render positional profiles, ensemble segment comparisons, or both",
    )
    parser.add_argument("--segment-summary-input", type=Path)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="manifest used to validate CV or correlation metadata",
    )
    parser.add_argument(
        "--metric-regime",
        choices=METRIC_REGIMES,
        default="model-best",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_MODELS),
        default=list(DATASET_MODELS),
    )
    parser.add_argument("--normalized-bins", type=int, default=20)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--formats", nargs="+", choices=("png", "svg"), default=["png", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def _render_segment_figures(
    args: argparse.Namespace,
    datasets: set[str],
    output_dir: Path,
) -> list[Path]:
    if args.metric_regime not in {"auroc-ensemble-best", "auroc-ensemble-shared"}:
        raise ValueError("segment comparisons are available only for AUROC ensemble regimes")
    default_path = {
        "auroc-ensemble-best": AUROC_ENSEMBLE_BEST_RESULTS_DIR
        / "auroc_ensemble_best_segment_summary.parquet",
        "auroc-ensemble-shared": AUROC_ENSEMBLE_SHARED_RESULTS_DIR
        / "auroc_ensemble_shared_segment_summary.parquet",
    }[args.metric_regime]
    segment_path = args.segment_summary_input or default_path
    _stage(f"loading fixed-winner segment summary from {segment_path}")
    segment_summary = load_segment_summary(
        segment_path,
        datasets,
    )
    manifest_path = args.selection_manifest or {
        "auroc-ensemble-best": AUROC_ENSEMBLE_SELECTION_PATH,
        "auroc-ensemble-shared": AUROC_ENSEMBLE_SHARED_SELECTION_PATH,
    }[args.metric_regime]
    if args.exclude_head_disagreement and args.selection_manifest is None:
        manifest_path = no_hd_dir(manifest_path.parent) / manifest_path.name
    if args.metric_regime == "auroc-ensemble-shared":
        segment_summary = attach_shared_validation_metadata(
            segment_summary,
            manifest_path,
        )
    else:
        segment_summary = attach_best_validation_metadata(
            segment_summary,
            manifest_path,
        )
    _stage(f"loaded {segment_summary.height:,} segment rows; starting figure layout")
    written: list[Path] = []
    selected_datasets = [dataset for dataset in DATASET_MODELS if dataset in datasets]
    render_progress = tqdm(
        selected_datasets,
        desc="Rendering ensemble segment figures",
        unit="dataset",
        dynamic_ncols=True,
    )
    for dataset in render_progress:
        render_progress.set_postfix_str(f"{dataset}: layout and export")
        before = len(written)
        written.extend(
            save_segment_comparison_figure(
                segment_summary,
                dataset,
                output_dir=output_dir,
                formats=tuple(args.formats),
                dpi=args.dpi,
                overwrite=args.overwrite,
            )
        )
        _stage(
            f"{dataset}: wrote {len(written) - before} segment-comparison files"
        )
    return written


def main() -> int:
    started = time.perf_counter()
    args = _parse_args()
    log_stages = args.metric_regime.startswith("auroc-ensemble")
    if log_stages:
        _stage(f"starting {args.metric_regime} plotting ({args.plot_kind})")
    if (
        args.exclude_head_disagreement
        and args.input == ANALYSIS_DIR / "token_position_scores.parquet"
    ):
        args.input = ANALYSIS_DIR / "token_position_scores_no_hd.parquet"
    default_output_dir = {
        "model-best": POOLED_AUROC_DIR / "plots" / "model_best",
        "dataset-shared": POOLED_AUROC_DIR / "plots" / "dataset_shared",
        "cv-best": CV_METHOD_DIR / "plots" / "model_best",
        "cv-shared": CV_METHOD_DIR / "plots" / "dataset_shared",
        "cor-best": SPEARMAN_METHOD_DIR / "plots" / "model_best",
        "cor-shared": SPEARMAN_METHOD_DIR / "plots" / "dataset_shared",
        "auroc-ensemble-best": AUROC_ENSEMBLE_DIR / "plots" / "model_best",
        "auroc-ensemble-shared": AUROC_ENSEMBLE_DIR / "plots" / "dataset_shared",
    }[args.metric_regime]
    default_summary = {
        "model-best": POOLED_AUROC_BEST_RESULTS_DIR / "position_plot_summary.parquet",
        "dataset-shared": POOLED_AUROC_SHARED_RESULTS_DIR
        / "shared_metric_position_summary.parquet",
        "cv-best": CV_BEST_RESULTS_DIR / "cv_best_metric_position_summary.parquet",
        "cv-shared": CV_SHARED_RESULTS_DIR / "cv_shared_metric_position_summary.parquet",
        "cor-best": COR_BEST_RESULTS_DIR / "cor_best_metric_position_summary.parquet",
        "cor-shared": COR_SHARED_RESULTS_DIR / "cor_shared_metric_position_summary.parquet",
        "auroc-ensemble-best": AUROC_ENSEMBLE_BEST_RESULTS_DIR
        / "auroc_ensemble_best_metric_position_summary.parquet",
        "auroc-ensemble-shared": AUROC_ENSEMBLE_SHARED_RESULTS_DIR
        / "auroc_ensemble_shared_metric_position_summary.parquet",
    }[args.metric_regime]
    if args.exclude_head_disagreement:
        default_output_dir = no_hd_dir(default_output_dir)
        default_summary = no_hd_dir(default_summary.parent) / default_summary.name
    output_dir = args.output_dir or default_output_dir
    summary_output = args.summary_output or default_summary
    if not _inside_analysis(output_dir) or not _inside_analysis(summary_output):
        raise SystemExit("all plot outputs must remain under token_analysis")
    datasets = set(args.datasets)
    if args.plot_kind == "segments":
        written = _render_segment_figures(args, datasets, output_dir)
        _stage(
            f"wrote {len(written)} segment-comparison figure files in "
            f"{time.perf_counter() - started:.1f}s"
        )
        return 0

    if log_stages:
        _stage(
            f"loading {args.input} and reducing tokens to case-balanced position bins; "
            f"{args.bootstrap_resamples:,} bootstrap replicates"
        )
    aggregation_started = time.perf_counter()
    summary = build_plot_summary(
        args.input,
        datasets,
        normalized_bins=args.normalized_bins,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
        metric_regime=args.metric_regime,
    )
    if log_stages:
        _stage(
            f"position aggregation produced {summary.height:,} bins in "
            f"{time.perf_counter() - aggregation_started:.1f}s; validating manifest"
        )
    if args.metric_regime in {
        "cv-best", "cv-shared", "cor-best", "cor-shared",
        "auroc-ensemble-best", "auroc-ensemble-shared",
    }:
        default_manifest = {
            "cv-best": CV_SELECTION_PATH,
            "cv-shared": CV_SHARED_SELECTION_PATH,
            "cor-best": COR_SELECTION_PATH,
            "cor-shared": COR_SHARED_SELECTION_PATH,
            "auroc-ensemble-best": AUROC_ENSEMBLE_SELECTION_PATH,
            "auroc-ensemble-shared": AUROC_ENSEMBLE_SHARED_SELECTION_PATH,
        }[args.metric_regime]
        if args.exclude_head_disagreement and args.selection_manifest is None:
            default_manifest = no_hd_dir(default_manifest.parent) / default_manifest.name
        specs = metric_specs(
            args.metric_regime,
            args.selection_manifest or default_manifest,
            args.exclude_head_disagreement,
        )
        observed = summary.select("dataset", "model", "metric_name", "metric_direction").unique()
        mismatches = [
            row
            for row in observed.iter_rows(named=True)
            if specs[(row["dataset"], row["model"])]["metric"] != row["metric_name"]
            or specs[(row["dataset"], row["model"])]["direction"] != row["metric_direction"]
        ]
        if mismatches:
            raise ValueError(f"input table does not match selection manifest: {mismatches}")
        if args.metric_regime == "auroc-ensemble-shared":
            summary = attach_shared_validation_metadata(
                summary,
                args.selection_manifest or default_manifest,
            )
    _write_summary(summary, summary_output, args.overwrite)
    written = []
    if log_stages:
        _stage(f"selection metadata validated; writing summary to {summary_output}")
    selected_datasets = [dataset for dataset in DATASET_MODELS if dataset in datasets]
    render_progress = tqdm(
        selected_datasets,
        desc="Rendering ensemble positional figures",
        unit="dataset",
        dynamic_ncols=True,
        disable=not args.metric_regime.startswith("auroc-ensemble"),
    )
    for dataset in render_progress:
        render_progress.set_postfix_str(f"{dataset}: layout and PNG/SVG export")
        before = len(written)
        written.extend(
            save_dataset_figure(
                summary,
                dataset,
                output_dir=output_dir,
                normalized_bins=args.normalized_bins,
                formats=tuple(args.formats),
                dpi=args.dpi,
                overwrite=args.overwrite,
            )
        )
        if log_stages:
            _stage(f"{dataset}: wrote {len(written) - before} positional figure files")
    if args.plot_kind == "both":
        written.extend(_render_segment_figures(args, datasets, output_dir))
    if log_stages:
        _stage(
            f"complete in {time.perf_counter() - started:.1f}s: wrote {summary_output} "
            f"({summary.height:,} bins) and {len(written)} figure files"
        )
    else:
        print(f"wrote {summary_output} ({summary.height} bins) and {len(written)} figure files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
