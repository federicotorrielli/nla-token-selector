"""Build a long-format positional token-score parquet from the all-token run.

The source parquets are read-only. Every output row corresponds to exactly one
source token; models are stacked vertically because their tokenizations differ.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
from tqdm.auto import tqdm

try:
    from .common import (
        ANALYSIS_DIR,
        AUROC_ENSEMBLE_SELECTION_PATH,
        AUROC_ENSEMBLE_SHARED_SELECTION_PATH,
        CONTENT_WINNERS,
        COR_SELECTION_PATH,
        COR_SHARED_SELECTION_PATH,
        CV_SELECTION_PATH,
        CV_SHARED_SELECTION_PATH,
        DATASET_MODELS,
        DEFAULT_INPUT_DIR,
        EXPECTED_TOTAL_ROWS,
        METRIC_SPECS,
        MODEL_INFO,
        NO_HD_CONTENT_WINNERS,
        liars_subdataset_expr,
        load_auroc_ensemble_specs,
        load_cor_metric_specs,
        load_cor_shared_metric_specs,
        load_cv_metric_specs,
        load_cv_shared_metric_specs,
        metric_specs,
        no_hd_dir,
        selected_pairs,
        source_path,
    )
except ImportError:
    from common import (  # type: ignore[no-redef]
        ANALYSIS_DIR,
        AUROC_ENSEMBLE_SELECTION_PATH,
        AUROC_ENSEMBLE_SHARED_SELECTION_PATH,
        CONTENT_WINNERS,
        COR_SELECTION_PATH,
        COR_SHARED_SELECTION_PATH,
        CV_SELECTION_PATH,
        CV_SHARED_SELECTION_PATH,
        DATASET_MODELS,
        DEFAULT_INPUT_DIR,
        EXPECTED_TOTAL_ROWS,
        METRIC_SPECS,
        MODEL_INFO,
        NO_HD_CONTENT_WINNERS,
        liars_subdataset_expr,
        load_cor_metric_specs,
        load_auroc_ensemble_specs,
        load_cor_shared_metric_specs,
        load_cv_metric_specs,
        load_cv_shared_metric_specs,
        metric_specs,
        no_hd_dir,
        selected_pairs,
        source_path,
    )

TT_EXPECTED_BOUNDARY_LENGTH = 5


def _stage(message: str) -> None:
    """Emit a timestamped builder milestone alongside tqdm progress."""
    tqdm.write(f"[{time.strftime('%H:%M:%S')}] [token-builder] {message}")


OUTPUT_COLUMNS = (
    "dataset",
    "source_subdataset",
    "model",
    "model_name",
    "case_id",
    "mode",
    "source_position_id",
    "label",
    "on_task",
    "case_on_task_rate",
    "dataset_model_base_rate",
    "model_coverage_count",
    "case_shared_across_all_models",
    "token_raw",
    "source_region",
    "analysis_region",
    "tokenizer_family",
    "segment",
    "is_template",
    "is_content",
    "response_available",
    "response_token_origin",
    "is_reconstructed_tail",
    "full_position",
    "full_length",
    "full_position_normalized",
    "segment_position",
    "segment_length",
    "segment_position_normalized",
    "input_length",
    "boundary_length",
    "source_boundary_length",
    "boundary_was_corrected",
    "output_length",
    "trailer_length",
    "past_gemma_local_window",
    "metric_name",
    "metric_auroc",
    "metric_direction",
    "metric_value_raw",
    "metric_available",
    "metric_value_aligned",
    "metric_z_within_case",
    "metric_percentile_within_case",
    "metric_winner_changes_content_only",
    "shared_metric_name",
    "shared_metric_auroc",
    "shared_metric_direction",
    "shared_metric_value_raw",
    "shared_metric_available",
    "shared_metric_value_aligned",
    "shared_metric_z_within_case",
    "shared_metric_percentile_within_case",
    "shared_metric_winner_changes_content_only",
    "cv_metric_name",
    "cv_metric_auroc",
    "cv_metric_direction",
    "cv_metric_value_raw",
    "cv_metric_available",
    "cv_metric_value_aligned",
    "cv_metric_z_within_case",
    "cv_metric_percentile_within_case",
    "cv_metric_winner_changes_content_only",
    "cv_metric_cv_macro_ap",
    "cv_metric_nested_cv_macro_ap",
    "cv_metric_statistically_tied",
    "cv_metric_selection_score",
    "cv_metric_pooled_auroc",
    "cv_metric_beats_position_baseline",
    "cv_metric_beats_region_baseline",
    "cv_metric_demonstrated_incremental_value",
    "cv_shared_metric_name",
    "cv_shared_metric_auroc",
    "cv_shared_metric_direction",
    "cv_shared_metric_value_raw",
    "cv_shared_metric_available",
    "cv_shared_metric_value_aligned",
    "cv_shared_metric_z_within_case",
    "cv_shared_metric_percentile_within_case",
    "cv_shared_metric_winner_changes_content_only",
    "cv_shared_metric_model_cv_macro_ap",
    "cv_shared_metric_equal_model_cv_macro_ap",
    "cv_shared_metric_nested_cv_shared_macro_ap",
    "cv_shared_metric_statistically_tied",
    "cv_shared_metric_selection_score",
    "cv_shared_metric_pooled_auroc",
    "cv_shared_metric_beats_position_baseline",
    "cv_shared_metric_beats_region_baseline",
    "cv_shared_metric_demonstrated_incremental_value",
    "cor_metric_name",
    "cor_metric_auroc",
    "cor_metric_direction",
    "cor_metric_value_raw",
    "cor_metric_available",
    "cor_metric_value_aligned",
    "cor_metric_z_within_case",
    "cor_metric_percentile_within_case",
    "cor_metric_winner_changes_content_only",
    "cor_metric_spearman_rho",
    "cor_metric_abs_spearman_rho",
    "cor_metric_statistically_tied",
    "cor_metric_direction_stability",
    "cor_shared_metric_name",
    "cor_shared_metric_auroc",
    "cor_shared_metric_direction",
    "cor_shared_metric_value_raw",
    "cor_shared_metric_available",
    "cor_shared_metric_value_aligned",
    "cor_shared_metric_z_within_case",
    "cor_shared_metric_percentile_within_case",
    "cor_shared_metric_winner_changes_content_only",
    "cor_shared_metric_spearman_rho",
    "cor_shared_metric_abs_spearman_rho",
    "cor_shared_metric_equal_model_abs_spearman_rho",
    "cor_shared_metric_statistically_tied",
    "cor_shared_metric_direction_stability",
    "auroc_ensemble_metric_name",
    "auroc_ensemble_metric_candidate_type",
    "auroc_ensemble_metric_component_1",
    "auroc_ensemble_metric_component_1_weight",
    "auroc_ensemble_metric_component_1_direction",
    "auroc_ensemble_metric_component_2",
    "auroc_ensemble_metric_component_2_weight",
    "auroc_ensemble_metric_component_2_direction",
    "auroc_ensemble_metric_auroc",
    "auroc_ensemble_metric_direction",
    "auroc_ensemble_metric_value_raw",
    "auroc_ensemble_metric_available",
    "auroc_ensemble_metric_value_aligned",
    "auroc_ensemble_metric_z_within_case",
    "auroc_ensemble_metric_percentile_within_case",
    "auroc_ensemble_metric_winner_changes_common_support",
    "auroc_ensemble_metric_gain_over_best_original",
    "auroc_ensemble_metric_heldout_pooled_auroc",
    "auroc_ensemble_metric_heldout_case_macro_auroc",
    "auroc_ensemble_metric_heldout_case_macro_delta",
    "auroc_ensemble_metric_heldout_delta_ci_low",
    "auroc_ensemble_metric_heldout_delta_ci_high",
    "auroc_ensemble_metric_outer_fold_selection_frequency",
    "auroc_ensemble_shared_metric_name",
    "auroc_ensemble_shared_metric_candidate_type",
    "auroc_ensemble_shared_metric_component_1",
    "auroc_ensemble_shared_metric_component_1_weight",
    "auroc_ensemble_shared_metric_component_1_direction",
    "auroc_ensemble_shared_metric_component_2",
    "auroc_ensemble_shared_metric_component_2_weight",
    "auroc_ensemble_shared_metric_component_2_direction",
    "auroc_ensemble_shared_metric_auroc",
    "auroc_ensemble_shared_metric_equal_model_auroc",
    "auroc_ensemble_shared_metric_direction",
    "auroc_ensemble_shared_metric_value_raw",
    "auroc_ensemble_shared_metric_available",
    "auroc_ensemble_shared_metric_value_aligned",
    "auroc_ensemble_shared_metric_z_within_case",
    "auroc_ensemble_shared_metric_percentile_within_case",
    "auroc_ensemble_shared_metric_winner_changes_common_support",
    "auroc_ensemble_shared_metric_gain_over_best_original",
    "selection_regime",
    "is_extreme_norm",
    "norm_ratio",
    "w",
)


def case_coverage(input_dir: Path, pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
    """Number of model files containing each dataset/case id."""
    by_dataset: dict[str, dict[str, int]] = {}
    for dataset, model in pairs:
        path = source_path(input_dir, dataset, model)
        case_ids = pl.scan_parquet(path).select("case_id").unique().collect()["case_id"].to_list()
        counts = by_dataset.setdefault(dataset, {})
        for case_id in case_ids:
            counts[case_id] = counts.get(case_id, 0) + 1
    return {
        (dataset, case_id): count
        for dataset, counts in by_dataset.items()
        for case_id, count in counts.items()
    }


def _coverage_expr(dataset: str, coverage: dict[tuple[str, str], int]) -> pl.Expr:
    mapping = {case_id: count for (kind, case_id), count in coverage.items() if kind == dataset}
    return pl.col("case_id").replace_strict(mapping, default=0, return_dtype=pl.Int64)


def _response_origin(dataset: str) -> pl.Expr:
    if dataset == "opi":
        return pl.lit("not_available")
    if dataset == "liars":
        return (
            pl.when(pl.col("region") == "assistant")
            .then(pl.lit("dataset_original"))
            .otherwise(pl.lit("not_response"))
        )
    return (
        pl.when((pl.col("region") == "assistant") & (pl.col("probe_tok_idx") >= 0))
        .then(pl.lit("stored_prefix"))
        .when((pl.col("region") == "assistant") & (pl.col("probe_tok_idx") < 0))
        .then(pl.lit("reconstructed_tail"))
        .otherwise(pl.lit("not_response"))
    )


def _pooled_rank_expr(metric: str) -> pl.Expr:
    clean = (
        pl.when(pl.col(metric).is_finite().fill_null(False))
        .then(pl.col(metric).cast(pl.Float64))
        .otherwise(None)
        .cast(pl.Float64)
    )
    count = clean.count()
    return (
        pl.when(count > 1)
        .then((clean.rank("average") - 1) / (count - 1))
        .when(count == 1)
        .then(0.5)
        .otherwise(None)
    )


def _ensemble_score_expr(spec: dict[str, object]) -> tuple[pl.Expr, pl.Expr]:
    component = _pooled_rank_expr(str(spec["component_1"]))
    if spec["component_1_direction"] == "lower":
        component = 1 - component
    score = component * float(spec["component_1_weight"])
    second_name = spec.get("component_2")
    if second_name is not None:
        second = _pooled_rank_expr(str(second_name))
        if spec.get("component_2_direction") == "lower":
            second = 1 - second
        score = score + second * float(spec["component_2_weight"])
    aligned = (
        score
        if spec.get("candidate_direction") == "higher"
        else 1 - score
    )
    return score, aligned


def _fallback_ensemble_spec(dataset: str, model: str) -> dict[str, object]:
    """Single-metric placeholder used until ensemble manifests are generated."""
    base = METRIC_SPECS[(dataset, model)]
    return {
        "candidate_id": f"single::{base['metric']}",
        "candidate_type": "single",
        "component_1": base["metric"],
        "component_1_weight": 1.0,
        "component_1_direction": base["direction"],
        "component_2": None,
        "component_2_weight": 0.0,
        "component_2_direction": None,
        "candidate_direction": "higher",
        "pooled_auroc_raw": base["auroc"],
        "adjusted_auroc": max(float(base["auroc"]), 1 - float(base["auroc"])),
        "gain_over_best_original": 0.0,
        "winner_changes_common_support": False,
        "heldout_pooled_auroc": float("nan"),
        "heldout_case_macro_auroc": float("nan"),
        "heldout_case_macro_delta": float("nan"),
        "heldout_delta_ci_low": float("nan"),
        "heldout_delta_ci_high": float("nan"),
        "outer_fold_selection_frequency": float("nan"),
        "equal_model_adjusted_auroc": float("nan"),
    }


def _ensemble_metadata_exprs(
    spec: dict[str, object],
    prefix: str,
    *,
    shared: bool,
) -> list[pl.Expr]:
    expressions = [
        pl.lit(spec["candidate_id"]).alias(prefix + "name"),
        pl.lit(spec["candidate_type"]).alias(prefix + "candidate_type"),
        pl.lit(spec["component_1"]).alias(prefix + "component_1"),
        pl.lit(float(spec["component_1_weight"])).alias(prefix + "component_1_weight"),
        pl.lit(spec["component_1_direction"]).alias(prefix + "component_1_direction"),
        pl.lit(spec.get("component_2"), dtype=pl.String).alias(prefix + "component_2"),
        pl.lit(float(spec.get("component_2_weight") or 0.0)).alias(prefix + "component_2_weight"),
        pl.lit(spec.get("component_2_direction"), dtype=pl.String).alias(prefix + "component_2_direction"),
        pl.lit(float(spec["adjusted_auroc"])).alias(prefix + "auroc"),
        pl.lit(spec["candidate_direction"]).alias(prefix + "direction"),
        pl.col(prefix + "value_aligned").is_not_null().alias(prefix + "available"),
        pl.lit(bool(spec.get("winner_changes_common_support") or False)).alias(prefix + "winner_changes_common_support"),
        pl.lit(float(spec.get("gain_over_best_original") or 0.0)).alias(prefix + "gain_over_best_original"),
    ]
    if shared:
        expressions.append(
            pl.lit(float(spec.get("equal_model_adjusted_auroc") or float("nan"))).alias(
                prefix + "equal_model_auroc"
            )
        )
    else:
        for name in (
            "heldout_pooled_auroc", "heldout_case_macro_auroc",
            "heldout_case_macro_delta", "heldout_delta_ci_low",
            "heldout_delta_ci_high", "outer_fold_selection_frequency",
        ):
            expressions.append(pl.lit(float(spec.get(name) or float("nan"))).alias(prefix + name))
    return expressions


def transform_source(
    frame: pl.LazyFrame,
    dataset: str,
    model: str,
    coverage: dict[tuple[str, str], int],
    metric_regime: str = "model-best",
    cv_specs: dict[tuple[str, str], dict[str, object]] | None = None,
    cv_shared_specs: dict[tuple[str, str], dict[str, object]] | None = None,
    cor_specs: dict[tuple[str, str], dict[str, object]] | None = None,
    cor_shared_specs: dict[tuple[str, str], dict[str, object]] | None = None,
    ensemble_specs: dict[tuple[str, str], dict[str, object]] | None = None,
    ensemble_shared_specs: dict[tuple[str, str], dict[str, object]] | None = None,
    exclude_head_disagreement: bool = False,
) -> pl.LazyFrame:
    """Add segmentation, provenance, and all selectable metric-regime columns."""
    if metric_regime == "cv-best" and cv_specs is None:
        cv_specs = load_cv_metric_specs()
    if metric_regime == "cv-shared" and cv_shared_specs is None:
        cv_shared_specs = load_cv_shared_metric_specs()
    if metric_regime == "cor-best" and cor_specs is None:
        cor_specs = load_cor_metric_specs()
    if metric_regime == "cor-shared" and cor_shared_specs is None:
        cor_shared_specs = load_cor_shared_metric_specs()
    if metric_regime == "auroc-ensemble-best" and ensemble_specs is None:
        ensemble_specs = load_auroc_ensemble_specs()
    if metric_regime == "auroc-ensemble-shared" and ensemble_shared_specs is None:
        ensemble_shared_specs = load_auroc_ensemble_specs(AUROC_ENSEMBLE_SHARED_SELECTION_PATH)
    if metric_regime == "cv-best":
        active_specs = cv_specs
    elif metric_regime == "cv-shared":
        active_specs = cv_shared_specs
    elif metric_regime == "cor-best":
        active_specs = cor_specs
    elif metric_regime == "cor-shared":
        active_specs = cor_shared_specs
    else:
        if metric_regime.startswith("auroc-ensemble"):
            metric_regime = "model-best"
        active_specs = metric_specs(
            metric_regime, exclude_head_disagreement=exclude_head_disagreement
        )
    assert active_specs is not None
    spec = active_specs[(dataset, model)]
    metric = str(spec["metric"])
    sign = 1.0 if spec["direction"] == "higher" else -1.0
    shared_spec = metric_specs(
        "dataset-shared", exclude_head_disagreement=exclude_head_disagreement
    )[(dataset, model)]
    shared_metric = str(shared_spec["metric"])
    shared_sign = 1.0 if shared_spec["direction"] == "higher" else -1.0
    if cv_specs is None:
        cv_specs = {}
    cv_spec = cv_specs.get((dataset, model), spec)
    cv_metric = str(cv_spec["metric"])
    cv_sign = 1.0 if cv_spec["direction"] == "higher" else -1.0
    if cv_shared_specs is None:
        cv_shared_specs = {}
    cv_shared_spec = cv_shared_specs.get((dataset, model), shared_spec)
    cv_shared_metric = str(cv_shared_spec["metric"])
    cv_shared_sign = 1.0 if cv_shared_spec["direction"] == "higher" else -1.0
    if cor_specs is None:
        cor_specs = {}
    cor_spec = cor_specs.get((dataset, model), spec)
    cor_metric = str(cor_spec["metric"])
    cor_sign = 1.0 if cor_spec["direction"] == "higher" else -1.0
    if cor_shared_specs is None:
        cor_shared_specs = {}
    cor_shared_spec = cor_shared_specs.get((dataset, model), shared_spec)
    cor_shared_metric = str(cor_shared_spec["metric"])
    cor_shared_sign = 1.0 if cor_shared_spec["direction"] == "higher" else -1.0
    if ensemble_specs is None:
        ensemble_specs = {}
    ensemble_spec = ensemble_specs.get((dataset, model), _fallback_ensemble_spec(dataset, model))
    if ensemble_shared_specs is None:
        ensemble_shared_specs = {}
    ensemble_shared_spec = ensemble_shared_specs.get(
        (dataset, model), _fallback_ensemble_spec(dataset, model)
    )
    ensemble_raw_expr, ensemble_aligned_expr = _ensemble_score_expr(ensemble_spec)
    ensemble_shared_raw_expr, ensemble_shared_aligned_expr = _ensemble_score_expr(ensemble_shared_spec)
    model_info = MODEL_INFO[model]
    expected_models = len(DATASET_MODELS[dataset])
    content_winners = NO_HD_CONTENT_WINNERS if exclude_head_disagreement else CONTENT_WINNERS
    content_winner = content_winners[(dataset, model)]

    frame = frame.with_columns(
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
        (pl.col("region") == "assistant")
        .sum()
        .over("case_id")
        .cast(pl.Int64)
        .alias("output_length"),
    )
    frame = frame.with_columns(
        pl.col("_output_start").fill_null(pl.col("full_length")).alias("_cut")
    )
    frame = frame.with_columns(
        pl.when((pl.col("tok_idx") < pl.col("_cut")) & (pl.col("region") != "template"))
        .then(pl.col("tok_idx"))
        .otherwise(None)
        .max()
        .over("case_id")
        .alias("_source_input_end")
    )
    frame = frame.with_columns(
        (pl.col("_cut") - pl.col("_source_input_end").fill_null(-1) - 1)
        .cast(pl.Int64)
        .alias("source_boundary_length")
    )
    # Two TT source rows leave the single-character user message `a` labeled as
    # template, merging its user turn with the five-token generation prompt.
    # Preserve that source length, but use the known prompt length for analysis.
    boundary_correction = pl.lit(dataset == "tt") & (
        pl.col("source_boundary_length") > TT_EXPECTED_BOUNDARY_LENGTH
    )
    frame = frame.with_columns(
        pl.when(boundary_correction)
        .then(pl.col("_cut") - TT_EXPECTED_BOUNDARY_LENGTH - 1)
        .otherwise(pl.col("_source_input_end"))
        .alias("_input_end"),
        boundary_correction.alias("boundary_was_corrected"),
    )
    frame = frame.with_columns(
        (pl.col("_input_end").fill_null(-1) + 1).cast(pl.Int64).alias("input_length"),
        (pl.col("_cut") - pl.col("_input_end").fill_null(-1) - 1)
        .cast(pl.Int64)
        .alias("boundary_length"),
        pl.when(pl.col("output_length") > 0)
        .then(pl.col("full_length") - pl.col("_output_end") - 1)
        .otherwise(0)
        .cast(pl.Int64)
        .alias("trailer_length"),
    )
    frame = frame.with_columns(
        pl.when(
            pl.col("boundary_was_corrected")
            & (pl.col("tok_idx") == pl.col("input_length") - 1)
            & (pl.col("region") == "template")
        )
        .then(pl.lit("user"))
        .otherwise(pl.col("region"))
        .alias("analysis_region")
    )
    frame = frame.with_columns(
        pl.when(pl.col("tok_idx") < pl.col("input_length"))
        .then(pl.lit("input"))
        .when(pl.col("tok_idx") < pl.col("_cut"))
        .then(pl.lit("boundary"))
        .when(pl.col("region") == "assistant")
        .then(pl.lit("output"))
        .otherwise(pl.lit("trailer"))
        .alias("segment")
    )
    frame = frame.with_columns(
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
        .otherwise(pl.col("trailer_length"))
        .cast(pl.Int64)
        .alias("segment_length"),
    )
    frame = frame.with_columns(
        (pl.col("tok_idx") - pl.col("_segment_start")).cast(pl.Int64).alias("segment_position"),
        pl.when(pl.col("full_length") > 1)
        .then(pl.col("tok_idx") / (pl.col("full_length") - 1))
        .otherwise(0.0)
        .alias("full_position_normalized"),
    )
    frame = frame.with_columns(
        pl.when(pl.col("segment_length") > 1)
        .then(pl.col("segment_position") / (pl.col("segment_length") - 1))
        .otherwise(0.0)
        .alias("segment_position_normalized"),
        pl.when(pl.col(metric).is_finite().fill_null(False))
        .then(pl.col(metric))
        .otherwise(None)
        .cast(pl.Float64)
        .alias("_metric_clean"),
        pl.when(pl.col(shared_metric).is_finite().fill_null(False))
        .then(pl.col(shared_metric))
        .otherwise(None)
        .cast(pl.Float64)
        .alias("_shared_metric_clean"),
        pl.when(pl.col(cv_metric).is_finite().fill_null(False))
        .then(pl.col(cv_metric))
        .otherwise(None)
        .cast(pl.Float64)
        .alias("_cv_metric_clean"),
        pl.when(pl.col(cv_shared_metric).is_finite().fill_null(False))
        .then(pl.col(cv_shared_metric))
        .otherwise(None)
        .cast(pl.Float64)
        .alias("_cv_shared_metric_clean"),
        pl.when(pl.col(cor_metric).is_finite().fill_null(False))
        .then(pl.col(cor_metric))
        .otherwise(None)
        .cast(pl.Float64)
        .alias("_cor_metric_clean"),
        pl.when(pl.col(cor_shared_metric).is_finite().fill_null(False))
        .then(pl.col(cor_shared_metric))
        .otherwise(None)
        .cast(pl.Float64)
        .alias("_cor_shared_metric_clean"),
    )
    frame = frame.with_columns(
        ensemble_raw_expr.alias("auroc_ensemble_metric_value_raw"),
        ensemble_aligned_expr.alias("auroc_ensemble_metric_value_aligned"),
        ensemble_shared_raw_expr.alias("auroc_ensemble_shared_metric_value_raw"),
        ensemble_shared_aligned_expr.alias("auroc_ensemble_shared_metric_value_aligned"),
        (pl.col("_metric_clean") * sign).alias("metric_value_aligned"),
        (pl.col("_shared_metric_clean") * shared_sign).alias("shared_metric_value_aligned"),
        (pl.col("_cv_metric_clean") * cv_sign).alias("cv_metric_value_aligned"),
        (pl.col("_cv_shared_metric_clean") * cv_shared_sign).alias(
            "cv_shared_metric_value_aligned"
        ),
        (pl.col("_cor_metric_clean") * cor_sign).alias("cor_metric_value_aligned"),
        (pl.col("_cor_shared_metric_clean") * cor_shared_sign).alias(
            "cor_shared_metric_value_aligned"
        ),
    )
    finite_count = pl.col("metric_value_aligned").count().over("case_id")
    rank = pl.col("metric_value_aligned").rank("average").over("case_id")
    shared_finite_count = pl.col("shared_metric_value_aligned").count().over("case_id")
    shared_rank = pl.col("shared_metric_value_aligned").rank("average").over("case_id")
    cv_finite_count = pl.col("cv_metric_value_aligned").count().over("case_id")
    cv_rank = pl.col("cv_metric_value_aligned").rank("average").over("case_id")
    cv_shared_finite_count = pl.col("cv_shared_metric_value_aligned").count().over("case_id")
    cv_shared_rank = pl.col("cv_shared_metric_value_aligned").rank("average").over("case_id")
    cor_finite_count = pl.col("cor_metric_value_aligned").count().over("case_id")
    cor_rank = pl.col("cor_metric_value_aligned").rank("average").over("case_id")
    cor_shared_finite_count = pl.col("cor_shared_metric_value_aligned").count().over("case_id")
    cor_shared_rank = pl.col("cor_shared_metric_value_aligned").rank("average").over("case_id")
    frame = frame.with_columns(
        (
            (pl.col("metric_value_aligned") - pl.col("metric_value_aligned").mean().over("case_id"))
            / (pl.col("metric_value_aligned").std().over("case_id") + 1e-12)
        ).alias("metric_z_within_case"),
        pl.when(pl.col("metric_value_aligned").is_not_null() & (finite_count > 1))
        .then((rank - 1) / (finite_count - 1))
        .when(pl.col("metric_value_aligned").is_not_null())
        .then(1.0)
        .otherwise(None)
        .alias("metric_percentile_within_case"),
        (
            (
                pl.col("shared_metric_value_aligned")
                - pl.col("shared_metric_value_aligned").mean().over("case_id")
            )
            / (pl.col("shared_metric_value_aligned").std().over("case_id") + 1e-12)
        ).alias("shared_metric_z_within_case"),
        pl.when(pl.col("shared_metric_value_aligned").is_not_null() & (shared_finite_count > 1))
        .then((shared_rank - 1) / (shared_finite_count - 1))
        .when(pl.col("shared_metric_value_aligned").is_not_null())
        .then(1.0)
        .otherwise(None)
        .alias("shared_metric_percentile_within_case"),
        (
            (
                pl.col("cv_metric_value_aligned")
                - pl.col("cv_metric_value_aligned").mean().over("case_id")
            )
            / (pl.col("cv_metric_value_aligned").std().over("case_id") + 1e-12)
        ).alias("cv_metric_z_within_case"),
        pl.when(pl.col("cv_metric_value_aligned").is_not_null() & (cv_finite_count > 1))
        .then((cv_rank - 1) / (cv_finite_count - 1))
        .when(pl.col("cv_metric_value_aligned").is_not_null())
        .then(1.0)
        .otherwise(None)
        .alias("cv_metric_percentile_within_case"),
        (
            (
                pl.col("cv_shared_metric_value_aligned")
                - pl.col("cv_shared_metric_value_aligned").mean().over("case_id")
            )
            / (pl.col("cv_shared_metric_value_aligned").std().over("case_id") + 1e-12)
        ).alias("cv_shared_metric_z_within_case"),
        pl.when(
            pl.col("cv_shared_metric_value_aligned").is_not_null() & (cv_shared_finite_count > 1)
        )
        .then((cv_shared_rank - 1) / (cv_shared_finite_count - 1))
        .when(pl.col("cv_shared_metric_value_aligned").is_not_null())
        .then(1.0)
        .otherwise(None)
        .alias("cv_shared_metric_percentile_within_case"),
        (
            (
                pl.col("cor_metric_value_aligned")
                - pl.col("cor_metric_value_aligned").mean().over("case_id")
            )
            / (pl.col("cor_metric_value_aligned").std().over("case_id") + 1e-12)
        ).alias("cor_metric_z_within_case"),
        pl.when(pl.col("cor_metric_value_aligned").is_not_null() & (cor_finite_count > 1))
        .then((cor_rank - 1) / (cor_finite_count - 1))
        .when(pl.col("cor_metric_value_aligned").is_not_null())
        .then(1.0)
        .otherwise(None)
        .alias("cor_metric_percentile_within_case"),
        (
            (
                pl.col("cor_shared_metric_value_aligned")
                - pl.col("cor_shared_metric_value_aligned").mean().over("case_id")
            )
            / (pl.col("cor_shared_metric_value_aligned").std().over("case_id") + 1e-12)
        ).alias("cor_shared_metric_z_within_case"),
        pl.when(
            pl.col("cor_shared_metric_value_aligned").is_not_null() & (cor_shared_finite_count > 1)
        )
        .then((cor_shared_rank - 1) / (cor_shared_finite_count - 1))
        .when(pl.col("cor_shared_metric_value_aligned").is_not_null())
        .then(1.0)
        .otherwise(None)
        .alias("cor_shared_metric_percentile_within_case"),
    )

    ensemble_count = pl.col("auroc_ensemble_metric_value_aligned").count().over("case_id")
    ensemble_rank = pl.col("auroc_ensemble_metric_value_aligned").rank("average").over("case_id")
    ensemble_shared_count = pl.col("auroc_ensemble_shared_metric_value_aligned").count().over("case_id")
    ensemble_shared_rank = pl.col("auroc_ensemble_shared_metric_value_aligned").rank("average").over("case_id")
    frame = frame.with_columns(
        ((pl.col("auroc_ensemble_metric_value_aligned") - pl.col("auroc_ensemble_metric_value_aligned").mean().over("case_id")) / (pl.col("auroc_ensemble_metric_value_aligned").std().over("case_id") + 1e-12)).alias("auroc_ensemble_metric_z_within_case"),
        pl.when(pl.col("auroc_ensemble_metric_value_aligned").is_not_null() & (ensemble_count > 1)).then((ensemble_rank - 1) / (ensemble_count - 1)).when(pl.col("auroc_ensemble_metric_value_aligned").is_not_null()).then(1.0).otherwise(None).alias("auroc_ensemble_metric_percentile_within_case"),
        ((pl.col("auroc_ensemble_shared_metric_value_aligned") - pl.col("auroc_ensemble_shared_metric_value_aligned").mean().over("case_id")) / (pl.col("auroc_ensemble_shared_metric_value_aligned").std().over("case_id") + 1e-12)).alias("auroc_ensemble_shared_metric_z_within_case"),
        pl.when(pl.col("auroc_ensemble_shared_metric_value_aligned").is_not_null() & (ensemble_shared_count > 1)).then((ensemble_shared_rank - 1) / (ensemble_shared_count - 1)).when(pl.col("auroc_ensemble_shared_metric_value_aligned").is_not_null()).then(1.0).otherwise(None).alias("auroc_ensemble_shared_metric_percentile_within_case"),
    )

    coverage_expr = _coverage_expr(dataset, coverage)
    source_subdataset = (
        liars_subdataset_expr() if dataset == "liars" else pl.lit(None, dtype=pl.String)
    )
    response_origin = _response_origin(dataset)
    frame = frame.with_columns(
        pl.lit(dataset).alias("dataset"),
        source_subdataset.alias("source_subdataset"),
        pl.lit(model).alias("model"),
        pl.lit(model_info["name"]).alias("model_name"),
        pl.col("position_id").alias("source_position_id"),
        pl.col("token").alias("token_raw"),
        pl.col("region").alias("source_region"),
        pl.lit(model_info["tokenizer_family"]).alias("tokenizer_family"),
        (pl.col("analysis_region") == "template").alias("is_template"),
        (pl.col("analysis_region") != "template").alias("is_content"),
        pl.lit(dataset != "opi").alias("response_available"),
        response_origin.alias("response_token_origin"),
        (response_origin == "reconstructed_tail").alias("is_reconstructed_tail"),
        pl.col("tok_idx").alias("full_position"),
        pl.col("on_task").mean().over("case_id").alias("case_on_task_rate"),
        pl.col("on_task").mean().alias("dataset_model_base_rate"),
        coverage_expr.alias("model_coverage_count"),
        (coverage_expr == expected_models).alias("case_shared_across_all_models"),
        (pl.lit(model in {"g12", "g27"}) & (pl.col("tok_idx") >= 1024)).alias(
            "past_gemma_local_window"
        ),
        pl.lit(metric).alias("metric_name"),
        pl.lit(float(spec["auroc"])).alias("metric_auroc"),
        pl.lit(str(spec["direction"])).alias("metric_direction"),
        pl.col(metric).cast(pl.Float64).alias("metric_value_raw"),
        pl.col(metric).is_finite().fill_null(False).alias("metric_available"),
        pl.lit(
            bool(spec.get("winner_changes_content_only", False))
            if metric_regime in {"cor-best", "cor-shared"}
            else False
            if metric_regime in {"cv-best", "cv-shared"}
            else content_winner != metric
        ).alias("metric_winner_changes_content_only"),
        pl.lit(shared_metric).alias("shared_metric_name"),
        pl.lit(float(shared_spec["auroc"])).alias("shared_metric_auroc"),
        pl.lit(str(shared_spec["direction"])).alias("shared_metric_direction"),
        pl.col(shared_metric).cast(pl.Float64).alias("shared_metric_value_raw"),
        pl.col(shared_metric).is_finite().fill_null(False).alias("shared_metric_available"),
        pl.lit(content_winner != shared_metric).alias("shared_metric_winner_changes_content_only"),
        pl.lit(cv_metric).alias("cv_metric_name"),
        pl.lit(float(cv_spec["auroc"])).alias("cv_metric_auroc"),
        pl.lit(str(cv_spec["direction"])).alias("cv_metric_direction"),
        pl.col(cv_metric).cast(pl.Float64).alias("cv_metric_value_raw"),
        pl.col(cv_metric).is_finite().fill_null(False).alias("cv_metric_available"),
        pl.lit(False).alias("cv_metric_winner_changes_content_only"),
        pl.lit(float(cv_spec.get("cv_macro_ap", float("nan")))).alias("cv_metric_cv_macro_ap"),
        pl.lit(float(cv_spec.get("nested_cv_macro_ap", float("nan")))).alias(
            "cv_metric_nested_cv_macro_ap"
        ),
        pl.lit(bool(cv_spec.get("statistically_tied", False))).alias(
            "cv_metric_statistically_tied"
        ),
        pl.lit(float(cv_spec.get("cv_macro_ap", float("nan")))).alias("cv_metric_selection_score"),
        pl.lit(float(cv_spec["auroc"])).alias("cv_metric_pooled_auroc"),
        pl.lit(bool(cv_spec.get("beats_position_baseline", False))).alias(
            "cv_metric_beats_position_baseline"
        ),
        pl.lit(bool(cv_spec.get("beats_region_baseline", False))).alias(
            "cv_metric_beats_region_baseline"
        ),
        pl.lit(
            bool(cv_spec.get("beats_position_baseline", False))
            and bool(cv_spec.get("beats_region_baseline", False))
        ).alias("cv_metric_demonstrated_incremental_value"),
        pl.lit(cv_shared_metric).alias("cv_shared_metric_name"),
        pl.lit(float(cv_shared_spec["auroc"])).alias("cv_shared_metric_auroc"),
        pl.lit(str(cv_shared_spec["direction"])).alias("cv_shared_metric_direction"),
        pl.col(cv_shared_metric).cast(pl.Float64).alias("cv_shared_metric_value_raw"),
        pl.col(cv_shared_metric).is_finite().fill_null(False).alias("cv_shared_metric_available"),
        pl.lit(False).alias("cv_shared_metric_winner_changes_content_only"),
        pl.lit(float(cv_shared_spec.get("cv_macro_ap", float("nan")))).alias(
            "cv_shared_metric_model_cv_macro_ap"
        ),
        pl.lit(float(cv_shared_spec.get("equal_model_cv_macro_ap", float("nan")))).alias(
            "cv_shared_metric_equal_model_cv_macro_ap"
        ),
        pl.lit(float(cv_shared_spec.get("nested_cv_shared_macro_ap", float("nan")))).alias(
            "cv_shared_metric_nested_cv_shared_macro_ap"
        ),
        pl.lit(bool(cv_shared_spec.get("statistically_tied", False))).alias(
            "cv_shared_metric_statistically_tied"
        ),
        pl.lit(float(cv_shared_spec.get("equal_model_cv_macro_ap", float("nan")))).alias(
            "cv_shared_metric_selection_score"
        ),
        pl.lit(float(cv_shared_spec["auroc"])).alias("cv_shared_metric_pooled_auroc"),
        pl.lit(bool(cv_shared_spec.get("beats_position_baseline", False))).alias(
            "cv_shared_metric_beats_position_baseline"
        ),
        pl.lit(bool(cv_shared_spec.get("beats_region_baseline", False))).alias(
            "cv_shared_metric_beats_region_baseline"
        ),
        pl.lit(
            bool(cv_shared_spec.get("beats_position_baseline", False))
            and bool(cv_shared_spec.get("beats_region_baseline", False))
        ).alias("cv_shared_metric_demonstrated_incremental_value"),
        pl.lit(cor_metric).alias("cor_metric_name"),
        pl.lit(float(cor_spec["auroc"])).alias("cor_metric_auroc"),
        pl.lit(str(cor_spec["direction"])).alias("cor_metric_direction"),
        pl.col(cor_metric).cast(pl.Float64).alias("cor_metric_value_raw"),
        pl.col(cor_metric).is_finite().fill_null(False).alias("cor_metric_available"),
        pl.lit(bool(cor_spec.get("winner_changes_content_only", False))).alias(
            "cor_metric_winner_changes_content_only"
        ),
        pl.lit(float(cor_spec.get("spearman_rho", float("nan")))).alias("cor_metric_spearman_rho"),
        pl.lit(float(cor_spec.get("abs_spearman_rho", float("nan")))).alias(
            "cor_metric_abs_spearman_rho"
        ),
        pl.lit(bool(cor_spec.get("statistically_tied", False))).alias(
            "cor_metric_statistically_tied"
        ),
        pl.lit(float(cor_spec.get("direction_stability", float("nan")))).alias(
            "cor_metric_direction_stability"
        ),
        pl.lit(cor_shared_metric).alias("cor_shared_metric_name"),
        pl.lit(float(cor_shared_spec["auroc"])).alias("cor_shared_metric_auroc"),
        pl.lit(str(cor_shared_spec["direction"])).alias("cor_shared_metric_direction"),
        pl.col(cor_shared_metric).cast(pl.Float64).alias("cor_shared_metric_value_raw"),
        pl.col(cor_shared_metric).is_finite().fill_null(False).alias("cor_shared_metric_available"),
        pl.lit(bool(cor_shared_spec.get("winner_changes_content_only", False))).alias(
            "cor_shared_metric_winner_changes_content_only"
        ),
        pl.lit(float(cor_shared_spec.get("spearman_rho", float("nan")))).alias(
            "cor_shared_metric_spearman_rho"
        ),
        pl.lit(float(cor_shared_spec.get("abs_spearman_rho", float("nan")))).alias(
            "cor_shared_metric_abs_spearman_rho"
        ),
        pl.lit(float(cor_shared_spec.get("equal_model_abs_spearman_rho", float("nan")))).alias(
            "cor_shared_metric_equal_model_abs_spearman_rho"
        ),
        pl.lit(bool(cor_shared_spec.get("statistically_tied", False))).alias(
            "cor_shared_metric_statistically_tied"
        ),
        pl.lit(float(cor_shared_spec.get("direction_stability", float("nan")))).alias(
            "cor_shared_metric_direction_stability"
        ),
        *_ensemble_metadata_exprs(ensemble_spec, "auroc_ensemble_metric_", shared=False),
        *_ensemble_metadata_exprs(ensemble_shared_spec, "auroc_ensemble_shared_metric_", shared=True),
        pl.lit("dense_boundary_case" if dataset == "tt" else "sparse").alias("selection_regime"),
        (pl.col("norm_ratio") > 5).fill_null(False).alias("is_extreme_norm"),
    )
    return frame.select(OUTPUT_COLUMNS)


def build_lazy(
    input_dir: Path,
    pairs: list[tuple[str, str]],
    selection_manifest: Path = CV_SELECTION_PATH,
    cv_shared_selection_manifest: Path = CV_SHARED_SELECTION_PATH,
    cor_selection_manifest: Path = COR_SELECTION_PATH,
    ensemble_selection_manifest: Path = AUROC_ENSEMBLE_SELECTION_PATH,
    ensemble_shared_selection_manifest: Path = AUROC_ENSEMBLE_SHARED_SELECTION_PATH,
    cor_shared_selection_manifest: Path = COR_SHARED_SELECTION_PATH,
    exclude_head_disagreement: bool = False,
) -> pl.LazyFrame:
    missing = [
        source_path(input_dir, *pair)
        for pair in pairs
        if not source_path(input_dir, *pair).exists()
    ]
    if missing:
        raise FileNotFoundError("missing source parquet(s): " + ", ".join(map(str, missing)))
    coverage_pairs = selected_pairs({dataset for dataset, _model in pairs})
    coverage = case_coverage(input_dir, coverage_pairs)
    cv_specs = load_cv_metric_specs(selection_manifest) if selection_manifest.exists() else None
    cv_shared_specs = (
        load_cv_shared_metric_specs(cv_shared_selection_manifest)
        if cv_shared_selection_manifest.exists()
        else None
    )
    cor_specs = (
        load_cor_metric_specs(cor_selection_manifest) if cor_selection_manifest.exists() else None
    )
    cor_shared_specs = (
        load_cor_shared_metric_specs(cor_shared_selection_manifest)
        if cor_shared_selection_manifest.exists()
        else None
    )
    ensemble_specs = (
        load_auroc_ensemble_specs(ensemble_selection_manifest)
        if ensemble_selection_manifest.exists()
        else None
    )
    ensemble_shared_specs = (
        load_auroc_ensemble_specs(ensemble_shared_selection_manifest)
        if ensemble_shared_selection_manifest.exists()
        else None
    )
    frames = [
        transform_source(
            pl.scan_parquet(source_path(input_dir, dataset, model)),
            dataset,
            model,
            coverage,
            cv_specs=cv_specs,
            cv_shared_specs=cv_shared_specs,
            cor_specs=cor_specs,
            cor_shared_specs=cor_shared_specs,
            ensemble_specs=ensemble_specs,
            ensemble_shared_specs=ensemble_shared_specs,
            exclude_head_disagreement=exclude_head_disagreement,
        )
        for dataset, model in pairs
    ]
    if not frames:
        raise ValueError("no dataset/model combinations selected")
    return pl.concat(frames, how="vertical")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--exclude-head-disagreement",
        action="store_true",
        help="materialize all six regimes from no-HD selections",
    )
    parser.add_argument(
        "--output", type=Path, default=ANALYSIS_DIR / "token_position_scores.parquet"
    )
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASET_MODELS))
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_INFO))
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=CV_SELECTION_PATH,
        help="cross-validated winner manifest; omitted columns fall back only when absent",
    )
    parser.add_argument(
        "--cv-shared-selection-manifest",
        type=Path,
        default=CV_SHARED_SELECTION_PATH,
        help="cross-validated dataset-shared winner manifest",
    )
    parser.add_argument(
        "--cor-selection-manifest",
        type=Path,
        default=COR_SELECTION_PATH,
        help="per-model Spearman-correlation winner manifest",
    )
    parser.add_argument(
        "--cor-shared-selection-manifest",
        type=Path,
        default=COR_SHARED_SELECTION_PATH,
        help="dataset-shared Spearman-correlation winner manifest",
    )
    parser.add_argument(
        "--auroc-ensemble-selection-manifest",
        type=Path,
        default=AUROC_ENSEMBLE_SELECTION_PATH,
        help="per-model exhaustive AUROC-ensemble winner manifest",
    )
    parser.add_argument(
        "--auroc-ensemble-shared-selection-manifest",
        type=Path,
        default=AUROC_ENSEMBLE_SHARED_SELECTION_PATH,
        help="dataset-shared exhaustive AUROC-ensemble winner manifest",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    started = time.perf_counter()
    args = _parse_args()
    _stage("resolving requested datasets, models, and selection manifests")
    if args.exclude_head_disagreement:
        default_output = ANALYSIS_DIR / "token_position_scores.parquet"
        if args.output == default_output:
            args.output = ANALYSIS_DIR / "token_position_scores_no_hd.parquet"
        if args.selection_manifest == CV_SELECTION_PATH:
            args.selection_manifest = no_hd_dir(CV_SELECTION_PATH.parent) / CV_SELECTION_PATH.name
        if args.cv_shared_selection_manifest == CV_SHARED_SELECTION_PATH:
            args.cv_shared_selection_manifest = (
                no_hd_dir(CV_SHARED_SELECTION_PATH.parent) / CV_SHARED_SELECTION_PATH.name
            )
        if args.cor_selection_manifest == COR_SELECTION_PATH:
            args.cor_selection_manifest = (
                no_hd_dir(COR_SELECTION_PATH.parent) / COR_SELECTION_PATH.name
            )
        if args.cor_shared_selection_manifest == COR_SHARED_SELECTION_PATH:
            args.cor_shared_selection_manifest = (
                no_hd_dir(COR_SHARED_SELECTION_PATH.parent) / COR_SHARED_SELECTION_PATH.name
            )
    datasets = set(args.datasets) if args.datasets else None
    models = set(args.models) if args.models else None
    pairs = selected_pairs(datasets, models)
    _stage(
        f"planning a lazy vertical union for {len(pairs)} model/dataset combinations"
    )
    frame = build_lazy(
        input_dir=args.input_dir,
        pairs=pairs,
        selection_manifest=args.selection_manifest,
        cv_shared_selection_manifest=args.cv_shared_selection_manifest,
        cor_selection_manifest=args.cor_selection_manifest,
        cor_shared_selection_manifest=args.cor_shared_selection_manifest,
        ensemble_selection_manifest=args.auroc_ensemble_selection_manifest,
        ensemble_shared_selection_manifest=args.auroc_ensemble_shared_selection_manifest,
        exclude_head_disagreement=args.exclude_head_disagreement,
    )

    if args.validate_only:
        _stage("executing validate-only row-count scan")
        rows = frame.select(pl.len().alias("rows")).collect().item()
        _stage(
            f"validated {len(pairs)} combinations and {rows:,} rows in "
            f"{time.perf_counter() - started:.1f}s"
        )
        return 0

    output = args.output.resolve()
    analysis_root = ANALYSIS_DIR.resolve()
    if analysis_root not in output.parents:
        raise SystemExit(f"refusing output outside token_analysis: {output}")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} exists; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    _stage("reading source metadata to establish the exact token-row total")
    expected_input_rows = sum(
        pq.ParquetFile(source_path(args.input_dir, dataset, model)).metadata.num_rows
        for dataset, model in pairs
    )
    _stage(
        f"streaming {expected_input_rows:,} transformed rows to {output}; "
        "the progress bar ETA is based on completed 100k-row batches"
    )
    writer: pq.ParquetWriter | None = None
    with tqdm(
        total=expected_input_rows,
        desc="Building token-position parquet",
        unit="token",
        unit_scale=True,
        dynamic_ncols=True,
    ) as progress:
        progress.set_postfix_str("lazy transform + zstd write")
        try:
            for batch in frame.collect_batches(
                chunk_size=100_000,
                maintain_order=True,
                engine="streaming",
            ):
                table = batch.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
                writer.write_table(table)
                progress.update(batch.height)
        finally:
            if writer is not None:
                writer.close()
    if writer is None:
        raise RuntimeError("builder produced no token batches")
    _stage("streaming complete; atomically installing and validating the parquet")
    os.replace(tmp, output)
    rows = pq.ParquetFile(output).metadata.num_rows
    expected = EXPECTED_TOTAL_ROWS if len(pairs) == len(METRIC_SPECS) else None
    suffix = f", expected {expected}" if expected is not None else ""
    _stage(
        f"wrote {output} ({rows:,} rows{suffix}) in "
        f"{time.perf_counter() - started:.1f}s"
    )
    if expected is not None and rows != expected:
        raise SystemExit(f"row-count mismatch: got {rows}, expected {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
