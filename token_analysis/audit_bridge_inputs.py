"""Audit the canonical all-token bridge parquets and write a Markdown report."""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata
from tqdm.auto import tqdm

try:
    from .build_token_position_data import case_coverage, transform_source
    from .common import (
        AUROC_ENSEMBLE_BEST_RESULTS_DIR,
        AUROC_ENSEMBLE_SELECTION_PATH,
        AUROC_ENSEMBLE_SHARED_RESULTS_DIR,
        AUROC_ENSEMBLE_SHARED_SELECTION_PATH,
        ALL_METRICS,
        ANALYSIS_DIR,
        CONTENT_WINNERS,
        COR_BEST_RESULTS_DIR,
        COR_SELECTION_PATH,
        COR_SHARED_RESULTS_DIR,
        COR_SHARED_SELECTION_PATH,
        CV_BEST_RESULTS_DIR,
        CV_SELECTION_PATH,
        CV_SHARED_RESULTS_DIR,
        CV_SHARED_SELECTION_PATH,
        DATASET_MODELS,
        DEFAULT_INPUT_DIR,
        EXPECTED_TOTAL_ROWS,
        LIARS_SUBDATASETS,
        METRIC_REGIMES,
        MODEL_INFO,
        NO_HD_CONTENT_WINNERS,
        POOLED_AUROC_BEST_RESULTS_DIR,
        POOLED_AUROC_SHARED_RESULTS_DIR,
        POSITION_BINS,
        metric_specs,
        no_hd_dir,
        selected_pairs,
        source_path,
        z_expr,
    )
except ImportError:
    from build_token_position_data import case_coverage, transform_source  # type: ignore[no-redef]
    from common import (  # type: ignore[no-redef]
        ALL_METRICS,
        AUROC_ENSEMBLE_BEST_RESULTS_DIR,
        AUROC_ENSEMBLE_SELECTION_PATH,
        AUROC_ENSEMBLE_SHARED_RESULTS_DIR,
        AUROC_ENSEMBLE_SHARED_SELECTION_PATH,
        ANALYSIS_DIR,
        CONTENT_WINNERS,
        COR_BEST_RESULTS_DIR,
        COR_SELECTION_PATH,
        COR_SHARED_RESULTS_DIR,
        COR_SHARED_SELECTION_PATH,
        CV_BEST_RESULTS_DIR,
        CV_SELECTION_PATH,
        CV_SHARED_RESULTS_DIR,
        CV_SHARED_SELECTION_PATH,
        DATASET_MODELS,
        DEFAULT_INPUT_DIR,
        EXPECTED_TOTAL_ROWS,
        LIARS_SUBDATASETS,
        METRIC_REGIMES,
        MODEL_INFO,
        NO_HD_CONTENT_WINNERS,
        POOLED_AUROC_BEST_RESULTS_DIR,
        POOLED_AUROC_SHARED_RESULTS_DIR,
        POSITION_BINS,
        metric_specs,
        no_hd_dir,
        selected_pairs,
        source_path,
        z_expr,
    )

IDENTITY_COLUMNS = {
    "position_id",
    "case_id",
    "mode",
    "tok_idx",
    "token",
    "label",
    "region",
    "probe_tok_idx",
    "on_task",
}
README_EXTREME_COUNTS = {
    ("liars", "g27"): 4333,
    ("liars", "l70"): 2000,
    ("opi", "q7"): 800,
    ("opi", "g12"): 800,
    ("opi", "g27"): 805,
    ("opi", "l70"): 800,
    ("taboo", "q7"): 96,
    ("taboo", "g12"): 96,
    ("taboo", "g27"): 96,
    ("taboo", "l70"): 96,
    ("tt", "q7"): 2303,
    ("tt", "g12"): 1963,
    ("tt", "g27"): 2073,
    ("tt", "l70"): 1550,
}


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        return "—" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"
    return str(value)


def _stage(message: str) -> None:
    """Emit a timestamped audit milestone without disrupting tqdm."""
    tqdm.write(f"[{time.strftime('%H:%M:%S')}] [bridge-audit] {message}")


def _table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(_fmt(v) for v in row) + " |" for row in rows)
    return lines


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    ok = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[ok], labels[ok].astype(int)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 or np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def _subdataset(case_id: str) -> str:
    return next((name for name in LIARS_SUBDATASETS if case_id.startswith(name + "_")), "unknown")


def _w_error(df: pl.DataFrame) -> float:
    case_ids = sorted(df["case_id"].unique().to_list())[:8]
    sample = df.filter(pl.col("case_id").is_in(case_ids)).with_columns(
        (z_expr("sink_drain") - z_expr("lookback_ratio")).alias("_w_expected")
    )
    a = sample["w"].to_numpy().astype(float)
    b = sample["_w_expected"].to_numpy().astype(float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else float("nan")


def _bin_means(df: pl.DataFrame) -> str:
    cells = []
    for lo, hi in POSITION_BINS:
        values = df.filter((pl.col("full_position") >= lo) & (pl.col("full_position") < hi))[
            "metric_value_aligned"
        ]
        finite = values.filter(values.is_finite())
        cells.append(_fmt(finite.mean() if finite.len() else None))
    return ", ".join(cells)


def _boundary_example(df: pl.DataFrame) -> str:
    boundary = df.filter(pl.col("segment") == "boundary")
    if boundary.is_empty():
        return "—"
    case_id = boundary["case_id"][0]
    tokens = (
        boundary.filter(pl.col("case_id") == case_id)
        .sort("full_position")["token_raw"]
        .head(8)
        .to_list()
    )
    return " ".join(
        str(token).replace("\r", "\\r").replace("\n", "\\n").replace("|", "\\|") for token in tokens
    )


def _marker_counts(df: pl.DataFrame) -> tuple[int, int, int, int, int]:
    token = pl.col("token").cast(pl.String)
    result = df.select(
        token.str.contains("Ġ", literal=True).sum().alias("space_bpe"),
        token.str.contains("Ċ", literal=True).sum().alias("newline_bpe"),
        token.str.contains("▁", literal=True).sum().alias("metaspace"),
        token.str.contains("âĢ", literal=True).sum().alias("byte_unicode"),
        token.str.starts_with("<").sum().alias("special"),
    ).row(0)
    return tuple(int(v) for v in result)


def _quantiles(values: pl.Series) -> list[float | None]:
    finite = values.filter(values.is_finite().fill_null(False))
    if finite.is_empty():
        return [None] * 5
    return [finite.quantile(q) for q in (0.01, 0.25, 0.50, 0.75, 0.99)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--exclude-head-disagreement",
        action="store_true",
        help="audit selections in which head_disagreement was ineligible",
    )
    parser.add_argument(
        "--metric-regime",
        choices=METRIC_REGIMES,
        default="model-best",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="manifest for cv-best or cv-shared (regime default if omitted)",
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    started = time.perf_counter()
    args = _parse_args()
    log_stages = args.metric_regime.startswith("auroc-ensemble")
    if log_stages:
        _stage(f"starting {args.metric_regime} audit and resolving its selection manifest")
    default_report = {
        "model-best": POOLED_AUROC_BEST_RESULTS_DIR / "bridge_input_audit.md",
        "dataset-shared": POOLED_AUROC_SHARED_RESULTS_DIR / "bridge_input_audit_shared_metric.md",
        "cv-best": CV_BEST_RESULTS_DIR / "bridge_input_audit_cv_best.md",
        "cv-shared": CV_SHARED_RESULTS_DIR / "bridge_input_audit_cv_shared.md",
        "cor-best": COR_BEST_RESULTS_DIR / "bridge_input_audit_cor_best.md",
        "cor-shared": COR_SHARED_RESULTS_DIR / "bridge_input_audit_cor_shared.md",
        "auroc-ensemble-best": AUROC_ENSEMBLE_BEST_RESULTS_DIR / "bridge_input_audit_auroc_ensemble_best.md",
        "auroc-ensemble-shared": AUROC_ENSEMBLE_SHARED_RESULTS_DIR / "bridge_input_audit_auroc_ensemble_shared.md",
    }[args.metric_regime]
    if args.exclude_head_disagreement:
        default_report = no_hd_dir(default_report.parent) / default_report.name
    report_path = (args.report or default_report).resolve()
    selection_manifest = args.selection_manifest or {
        "cv-best": CV_SELECTION_PATH,
        "cv-shared": CV_SHARED_SELECTION_PATH,
        "cor-best": COR_SELECTION_PATH,
        "cor-shared": COR_SHARED_SELECTION_PATH,
        "auroc-ensemble-best": AUROC_ENSEMBLE_SELECTION_PATH,
        "auroc-ensemble-shared": AUROC_ENSEMBLE_SHARED_SELECTION_PATH,
    }.get(args.metric_regime, CV_SELECTION_PATH)
    if args.exclude_head_disagreement and args.selection_manifest is None:
        selection_manifest = no_hd_dir(selection_manifest.parent) / selection_manifest.name
    specs = metric_specs(args.metric_regime, selection_manifest, args.exclude_head_disagreement)
    if log_stages:
        _stage(
            f"loaded selection metadata for {len(specs)} model/dataset combinations"
        )
    content_winners = NO_HD_CONTENT_WINNERS if args.exclude_head_disagreement else CONTENT_WINNERS
    cv_selection_rows: list[list[object]] = []
    cv_fold_rows: list[list[object]] = []
    cor_selection_rows: list[list[object]] = []
    ensemble_selection_rows: list[list[object]] = []
    ensemble_lomo_rows: list[list[object]] = []
    if args.metric_regime in {"cv-best", "cv-shared"}:
        manifest = pl.read_parquet(selection_manifest)
        for row in manifest.sort("dataset", "model").iter_rows(named=True):
            is_shared_cv = args.metric_regime == "cv-shared"
            cv_selection_rows.append(
                [
                    row["dataset"],
                    row["model"],
                    (row["equal_model_cv_macro_ap"] if is_shared_cv else row["cv_macro_ap"]),
                    (
                        row["nested_cv_shared_macro_ap"]
                        if is_shared_cv
                        else row["nested_cv_macro_ap"]
                    ),
                    row["runner_up_metric"],
                    row["delta_to_runner_up"],
                    f"[{row['delta_ci_low']:.3f}, {row['delta_ci_high']:.3f}]",
                    row["statistically_tied"],
                    row["precision_at_1pct"],
                    row["precision_at_5pct"],
                    row["precision_at_10pct"],
                    row["beats_position_baseline"],
                    row["beats_region_baseline"],
                ]
            )
            cv_fold_rows.append(
                [
                    row["dataset"],
                    row["model"],
                    (
                        f"{row['fold_case_count_min']}–{row['fold_case_count_max']}"
                        if not is_shared_cv
                        else f"{row['fold_count']} folds"
                    ),
                    (
                        f"{row['fold_mean_length_min']:.1f}–{row['fold_mean_length_max']:.1f}"
                        if not is_shared_cv
                        else "see model CV audit"
                    ),
                    (
                        f"{row['fold_mean_base_rate_min']:.3f}–{row['fold_mean_base_rate_max']:.3f}"
                        if not is_shared_cv
                        else "see model CV audit"
                    ),
                    row["direction_stability"],
                    row["outer_fold_selection_frequency"],
                ]
            )
    if args.metric_regime in {"cor-best", "cor-shared"}:
        manifest = pl.read_parquet(selection_manifest)
        is_shared_cor = args.metric_regime == "cor-shared"
        for row in manifest.sort("dataset", "model").iter_rows(named=True):
            cor_selection_rows.append(
                [
                    row["dataset"],
                    row["model"],
                    row["spearman_rho"],
                    (
                        row["equal_model_abs_spearman_rho"]
                        if is_shared_cor
                        else row["abs_spearman_rho"]
                    ),
                    row["runner_up_metric"],
                    row["delta_to_runner_up"],
                    f"[{row['delta_ci_low']:.3f}, {row['delta_ci_high']:.3f}]",
                    row["statistically_tied"],
                    row["direction_stability"],
                    row["finite_count"],
                    row["nonfinite_count"],
                    row["content_only_winner"],
                    row["winner_changes_content_only"],
                ]
            )
    if args.metric_regime.startswith("auroc-ensemble"):
        manifest = pl.read_parquet(selection_manifest)
        is_shared_ensemble = args.metric_regime == "auroc-ensemble-shared"
        for row in manifest.sort("dataset", "model").iter_rows(named=True):
            component = str(row["component_1"])
            if row.get("component_2") is not None:
                component += (
                    f"@{float(row['component_1_weight']):.2f} + "
                    f"{row['component_2']}@{float(row['component_2_weight']):.2f}"
                )
            ensemble_selection_rows.append(
                [
                    row["dataset"], row["model"], row["candidate_id"], component,
                    row["adjusted_auroc"],
                    row.get("equal_model_adjusted_auroc") if is_shared_ensemble else None,
                    row.get("gain_over_best_original"),
                    row.get("runner_up_candidate_id"), row.get("delta_to_runner_up"),
                    row.get("common_support_winner"),
                    row.get("winner_changes_common_support"),
                    (
                        row.get("shared_heldout_model_pooled_auroc")
                        if is_shared_ensemble
                        else row.get("heldout_pooled_auroc")
                    ),
                    (
                        row.get("shared_heldout_case_macro_auroc_delta")
                        if is_shared_ensemble
                        else row.get("heldout_case_macro_delta")
                    ),
                ]
            )
        lomo_path = selection_manifest.with_name(
            "auroc_ensemble_shared_lomo_summary.parquet"
        )
        if is_shared_ensemble and lomo_path.exists():
            lomo = pl.read_parquet(lomo_path).filter(
                pl.col("procedure") == "all_candidates"
            )
            for row in lomo.sort(
                "dataset", "heldout_model", "protocol"
            ).iter_rows(named=True):
                ensemble_lomo_rows.append(
                    [
                        row["dataset"],
                        row["heldout_model"],
                        row["protocol"],
                        row["candidate_id"],
                        row["case_macro_auc"],
                        row["case_macro_ap"],
                        row["case_macro_auroc_delta_vs_original"],
                        (
                            f"[{row['case_macro_auroc_delta_ci_low']:.3f}, "
                            f"{row['case_macro_auroc_delta_ci_high']:.3f}]"
                        ),
                        row["case_macro_precision_at_1pct"],
                        row["weak_model_transfer_evidence"],
                    ]
                )
    if ANALYSIS_DIR.resolve() not in report_path.parents:
        raise SystemExit(f"refusing report outside token_analysis: {report_path}")

    pairs = selected_pairs()
    missing_files = [
        source_path(args.input_dir, *pair)
        for pair in pairs
        if not source_path(args.input_dir, *pair).exists()
    ]
    if missing_files:
        raise SystemExit("missing source files: " + ", ".join(map(str, missing_files)))

    coverage = case_coverage(args.input_dir, pairs)
    hard_errors: list[str] = []
    warnings: list[str] = []
    inventory_rows: list[list[object]] = []
    selected_rows: list[list[object]] = []
    nonfinite_rows: list[list[object]] = []
    schema_signatures: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {}
    w_rows: list[list[object]] = []
    base_rows: list[list[object]] = []
    length_rows: list[list[object]] = []
    marker_rows: list[list[object]] = []
    boundary_correction_rows: list[list[object]] = []
    provenance_rows: list[list[object]] = []
    template_rows: list[list[object]] = []
    segment_rows: list[list[object]] = []
    region_rows: list[list[object]] = []
    distribution_rows: list[list[object]] = []
    extreme_rows: list[list[object]] = []
    position_rows: list[list[object]] = []
    gemma_sink_rows: list[list[object]] = []
    case_sets: dict[tuple[str, str], set[str]] = {}
    liars_composition: Counter[tuple[str, int, str]] = Counter()
    total_rows = 0

    required_base = IDENTITY_COLUMNS | set(ALL_METRICS)
    pair_iterator = tqdm(
        pairs,
        desc="Auditing ensemble combinations",
        unit="combination",
        dynamic_ncols=True,
        disable=not args.metric_regime.startswith("auroc-ensemble"),
    )
    for dataset, model in pair_iterator:
        pair_started = time.perf_counter()
        pair_iterator.set_postfix_str(f"{dataset}/{model}: loading and integrity")
        if log_stages:
            _stage(f"{dataset}/{model}: loading source and checking schema/integrity")
        path = source_path(args.input_dir, dataset, model)
        df = pl.read_parquet(path)
        total_rows += df.height
        columns = set(df.columns)
        schema_signatures[(dataset, model)] = tuple(
            (name, str(dtype)) for name, dtype in df.schema.items()
        )
        required = required_base | ({"attn_rollout"} if dataset == "opi" else set())
        absent = sorted(required - columns)
        if absent:
            hard_errors.append(f"{path.name}: missing columns {absent}")
            continue

        spec = specs[(dataset, model)]
        metric = str(spec["metric"])
        case_sets[(dataset, model)] = set(df["case_id"].unique().to_list())
        modes = ",".join(sorted(df["mode"].unique().to_list()))
        regions = ",".join(sorted(df["region"].unique().to_list()))
        unique_pid = df["position_id"].n_unique()
        unique_case_tok = df.select(pl.struct("case_id", "tok_idx").n_unique()).item()
        token_nulls = df["token"].null_count()
        label_nulls = df["label"].null_count()
        on_task_nulls = df["on_task"].null_count()
        cases = df["case_id"].n_unique()
        topology = df.group_by("case_id").agg(
            pl.len().alias("n"),
            pl.col("tok_idx").min().alias("lo"),
            pl.col("tok_idx").max().alias("hi"),
            pl.col("tok_idx").n_unique().alias("nu"),
            pl.when(pl.col("region") == "assistant")
            .then(pl.col("tok_idx"))
            .otherwise(None)
            .min()
            .alias("asst_lo"),
            pl.when(pl.col("region") == "assistant")
            .then(pl.col("tok_idx"))
            .otherwise(None)
            .max()
            .alias("asst_hi"),
            (pl.col("region") == "assistant").sum().alias("asst_n"),
        )
        gap_cases = topology.filter(
            (pl.col("lo") != 0) | (pl.col("hi") + 1 != pl.col("n")) | (pl.col("nu") != pl.col("n"))
        ).height
        noncontiguous_assistant = topology.filter(
            (pl.col("asst_n") > 0) & (pl.col("asst_hi") - pl.col("asst_lo") + 1 != pl.col("asst_n"))
        ).height

        if unique_pid != df.height:
            hard_errors.append(f"{path.name}: duplicate position_id")
        if unique_case_tok != df.height:
            hard_errors.append(f"{path.name}: duplicate (case_id, tok_idx)")
        if gap_cases:
            hard_errors.append(f"{path.name}: {gap_cases} cases have position gaps")
        if noncontiguous_assistant:
            hard_errors.append(
                f"{path.name}: {noncontiguous_assistant} cases have non-contiguous assistant regions"
            )
        if token_nulls or label_nulls or on_task_nulls:
            hard_errors.append(
                f"{path.name}: null token/label/on_task = "
                f"{token_nulls}/{label_nulls}/{on_task_nulls}"
            )

        if args.metric_regime.startswith("auroc-ensemble"):
            selected_finite = df[str(spec["component_1"])].is_finite().fill_null(False)
            if spec.get("component_2") is not None:
                selected_finite = selected_finite & df[
                    str(spec["component_2"])
                ].is_finite().fill_null(False)
            selected_bad = int((~selected_finite).sum())
        else:
            selected_values = df[metric].to_numpy().astype(float)
            selected_bad = int((~np.isfinite(selected_values)).sum())
        inventory_rows.append(
            [
                dataset,
                model,
                df.height,
                cases,
                len(df.columns),
                modes,
                regions,
                selected_bad,
            ]
        )
        selected_rows.append(
            [dataset, model, metric, spec["auroc"], spec["direction"], selected_bad]
        )
        for candidate in (*ALL_METRICS, *(("attn_rollout",) if dataset == "opi" else ())):
            series = df[candidate]
            nonfinite_rows.append(
                [
                    dataset,
                    model,
                    candidate,
                    series.null_count(),
                    int(series.is_nan().fill_null(False).sum()),
                    int((series == float("inf")).fill_null(False).sum()),
                    int((series == float("-inf")).fill_null(False).sum()),
                ]
            )

        w_error = _w_error(df)
        w_rows.append([dataset, model, w_error])
        if math.isfinite(w_error) and w_error > 1e-5:
            hard_errors.append(f"{path.name}: persisted w max error {w_error:.3g}")

        pair_iterator.set_postfix_str(f"{dataset}/{model}: segmentation and diagnostics")
        if log_stages:
            _stage(f"{dataset}/{model}: deriving segments and metric diagnostics")
        transformed = transform_source(
            df.lazy(),
            dataset,
            model,
            coverage,
            metric_regime=args.metric_regime,
            ensemble_specs=specs if args.metric_regime == "auroc-ensemble-best" else None,
            ensemble_shared_specs=specs if args.metric_regime == "auroc-ensemble-shared" else None,
            cv_specs=specs if args.metric_regime == "cv-best" else None,
            cv_shared_specs=specs if args.metric_regime == "cv-shared" else None,
            cor_specs=specs if args.metric_regime == "cor-best" else None,
            cor_shared_specs=specs if args.metric_regime == "cor-shared" else None,
            exclude_head_disagreement=args.exclude_head_disagreement,
        ).collect()
        if args.metric_regime.startswith("auroc-ensemble"):
            prefix = (
                "auroc_ensemble_metric_"
                if args.metric_regime == "auroc-ensemble-best"
                else "auroc_ensemble_shared_metric_"
            )
            transformed = transformed.with_columns(
                pl.col(prefix + "value_raw").alias("metric_value_raw"),
                pl.col(prefix + "value_aligned").alias("metric_value_aligned"),
                pl.col(prefix + "percentile_within_case").alias("metric_percentile_within_case"),
            )
        case_lengths = transformed.group_by("case_id").agg(
            pl.col("full_length").first(),
            pl.col("input_length").first(),
            pl.col("boundary_length").first(),
            pl.col("output_length").first(),
            pl.col("trailer_length").first(),
        )
        invalid_segments = transformed.filter(
            (pl.col("segment_position") < 0)
            | (pl.col("segment_position") >= pl.col("segment_length"))
        ).height
        invalid_region_segments = transformed.filter(
            ((pl.col("segment") == "boundary") & (pl.col("source_region") != "template"))
            | ((pl.col("segment") == "output") & (pl.col("source_region") != "assistant"))
            | ((pl.col("segment") == "trailer") & (pl.col("source_region") != "template"))
        ).height
        bad_length_sum = case_lengths.filter(
            pl.sum_horizontal("input_length", "boundary_length", "output_length", "trailer_length")
            != pl.col("full_length")
        ).height
        if invalid_segments or invalid_region_segments or bad_length_sum:
            hard_errors.append(
                f"{path.name}: invalid segment positions={invalid_segments}, "
                f"invalid region/segment pairs={invalid_region_segments}, "
                f"bad length sums={bad_length_sum}"
            )

        per_case_rate = (
            transformed.group_by("case_id")
            .agg(pl.col("on_task").mean().alias("rate"))
            .select(
                pl.col("rate").quantile(0.25).alias("q25"),
                pl.col("rate").median().alias("median"),
                pl.col("rate").quantile(0.75).alias("q75"),
            )
            .row(0)
        )
        base_rows.append(
            [
                dataset,
                model,
                transformed["dataset_model_base_rate"][0],
                *per_case_rate,
                "dense boundary" if dataset == "tt" else "sparse",
            ]
        )
        length_rows.append(
            [
                dataset,
                model,
                case_lengths["full_length"].median(),
                case_lengths["input_length"].median(),
                case_lengths["boundary_length"].median(),
                case_lengths["output_length"].median(),
                case_lengths["trailer_length"].median(),
                _boundary_example(transformed),
            ]
        )
        marker_rows.append([dataset, model, *_marker_counts(df)])
        corrected_boundaries = (
            transformed.filter(pl.col("boundary_was_corrected"))
            .select(
                "case_id",
                "source_boundary_length",
                "boundary_length",
                (pl.col("source_region") != pl.col("analysis_region"))
                .sum()
                .over("case_id")
                .alias("relabeled_tokens"),
            )
            .unique()
        )
        for correction in corrected_boundaries.iter_rows(named=True):
            boundary_correction_rows.append(
                [
                    dataset,
                    model,
                    correction["case_id"],
                    correction["source_boundary_length"],
                    correction["boundary_length"],
                    correction["relabeled_tokens"],
                ]
            )

        assistant = transformed.filter(pl.col("segment") == "output")
        reconstructed = assistant.filter(pl.col("is_reconstructed_tail"))
        provenance_rows.append(
            [
                dataset,
                model,
                assistant.height,
                reconstructed["case_id"].n_unique(),
                reconstructed.height,
            ]
        )

        labels = transformed["on_task"].to_numpy().astype(float)
        scores = transformed["metric_value_raw"].to_numpy().astype(float)
        content = transformed.filter(pl.col("source_region") != "template")
        all_auroc = _auroc(scores, labels)
        content_auroc = _auroc(
            content["metric_value_raw"].to_numpy().astype(float),
            content["on_task"].to_numpy().astype(float),
        )
        budget_values = []
        for fraction in (0.01, 0.10):
            selected = transformed.filter(
                pl.col("metric_percentile_within_case") >= (1.0 - fraction)
            )
            budget_values.extend(
                [
                    selected["on_task"].mean(),
                    selected["is_template"].mean(),
                ]
            )
        template_rows.append(
            [
                dataset,
                model,
                metric,
                (
                    str(spec.get("content_only_winner"))
                    if args.metric_regime in {"cor-best", "cor-shared"}
                    else "see segment-specific ensemble selection"
                    if args.metric_regime.startswith("auroc-ensemble")
                    else content_winners[(dataset, model)]
                ),
                transformed["on_task"].mean(),
                content["on_task"].mean(),
                all_auroc,
                content_auroc,
                (
                    bool(spec.get("winner_changes_content_only", False))
                    if args.metric_regime in {"cor-best", "cor-shared"}
                    else False
                    if args.metric_regime in {"cv-best", "cv-shared"}
                    else False
                    if args.metric_regime.startswith("auroc-ensemble")
                    else content_winners[(dataset, model)] != metric
                ),
                *budget_values,
            ]
        )

        grouped_segments = transformed.group_by("segment").agg(
            pl.len().alias("n"),
            pl.col("metric_value_aligned").mean().alias("mean"),
            pl.col("metric_value_aligned").std().alias("std"),
        )
        for row in grouped_segments.sort("segment").iter_rows(named=True):
            segment_rows.append([dataset, model, row["segment"], row["n"], row["mean"], row["std"]])
        grouped_regions = transformed.group_by("source_region").agg(
            pl.len().alias("n"),
            pl.col("metric_value_aligned").mean().alias("mean"),
            pl.col("metric_value_aligned").std().alias("std"),
        )
        for row in grouped_regions.sort("source_region").iter_rows(named=True):
            region_rows.append(
                [dataset, model, row["source_region"], row["n"], row["mean"], row["std"]]
            )
        distribution_rows.append(
            [
                dataset,
                model,
                *_quantiles(transformed["metric_value_raw"]),
                *_quantiles(transformed["metric_value_aligned"]),
            ]
        )

        extreme = transformed.filter(pl.col("is_extreme_norm"))
        ordinary = transformed.filter(~pl.col("is_extreme_norm"))
        published = README_EXTREME_COUNTS[(dataset, model)]
        if extreme.height != published:
            warnings.append(
                f"{dataset}/{model}: norm_ratio > 5 gives {extreme.height} extreme rows; "
                f"README table reports {published}."
            )
        extreme_segments = ", ".join(
            f"{segment}:{count}"
            for segment, count in extreme.group_by("segment").len().sort("segment").iter_rows()
        )
        extreme_rows.append(
            [
                dataset,
                model,
                extreme.height,
                published,
                extreme.height / transformed.height,
                extreme["on_task"].mean(),
                ordinary["on_task"].mean(),
                extreme.filter(pl.col("full_position") == 0).height,
                extreme["full_position"].median(),
                extreme["full_position"].max(),
                extreme_segments,
            ]
        )

        absolute_position = transformed["full_position"].to_numpy().astype(float)
        normalized_position = transformed["full_position_normalized"].to_numpy().astype(float)
        aligned = transformed["metric_value_aligned"].to_numpy().astype(float)
        full_len = transformed["full_length"].to_numpy().astype(float)
        segment_len = transformed["segment_length"].to_numpy().astype(float)
        z_values = transformed["metric_z_within_case"].to_numpy().astype(float)
        percentile = transformed["metric_percentile_within_case"].to_numpy().astype(float)
        position_rows.append(
            [
                dataset,
                model,
                metric,
                _corr(absolute_position, aligned),
                _corr(normalized_position, aligned),
                _corr(full_len, aligned),
                _corr(segment_len, aligned),
                float(np.nanmean(z_values)),
                float(np.nanstd(z_values)),
                float(np.nanmin(percentile)),
                float(np.nanmax(percentile)),
                _bin_means(transformed),
            ]
        )

        if model in {"g12", "g27"} and dataset in {"tt", "liars"}:
            before = df.filter(pl.col("tok_idx") < 1024)["sink_drain"]
            after = df.filter(pl.col("tok_idx") >= 1024)["sink_drain"]
            gemma_sink_rows.append([dataset, model, before.mean(), after.mean(), after.len()])

        if dataset == "liars":
            cases_frame = df.select("case_id", "label").unique()
            for case_id, label in cases_frame.iter_rows():
                liars_composition[(model, int(label), _subdataset(case_id))] += 1
        del transformed, df
        if log_stages:
            _stage(
                f"{dataset}/{model}: audited in "
                f"{time.perf_counter() - pair_started:.1f}s"
            )

    if log_stages:
        _stage("all combinations audited; assembling cross-model tables and Markdown")

    if total_rows != EXPECTED_TOTAL_ROWS:
        hard_errors.append(
            f"total rows {total_rows} != expected requested-scope total {EXPECTED_TOTAL_ROWS}"
        )

    ordinary_schema = schema_signatures[("tt", "q7")]
    opi_schema = schema_signatures[("opi", "q7")]
    for pair, signature in schema_signatures.items():
        expected_schema = opi_schema if pair[0] == "opi" else ordinary_schema
        if signature != expected_schema:
            hard_errors.append(f"{pair[0]}/{pair[1]}: schema differs from its canonical variant")
    schema_rows = [
        [dataset, model, "OPI + attn_rollout" if dataset == "opi" else "ordinary", "exact match"]
        for dataset, model in pairs
    ]
    ordinary_schema_text = ", ".join(f"{name}:{dtype}" for name, dtype in ordinary_schema)
    opi_schema_text = ", ".join(f"{name}:{dtype}" for name, dtype in opi_schema)

    coverage_rows: list[list[object]] = []
    for dataset, models in DATASET_MODELS.items():
        sets = [case_sets[(dataset, model)] for model in models]
        union = set.union(*sets)
        intersection = set.intersection(*sets)
        coverage_rows.append(
            [
                dataset,
                len(union),
                len(intersection),
                len(intersection) / len(union) if union else None,
                ", ".join(f"{model}:{len(case_sets[(dataset, model)])}" for model in models),
            ]
        )

    tt_models = DATASET_MODELS["tt"]
    tt_sets = {model: case_sets[("tt", model)] for model in tt_models}
    tt_union = set.union(*tt_sets.values())
    tt_intersection = set.intersection(*tt_sets.values())
    tt_incomplete = [
        (case_id, ", ".join(model for model in tt_models if case_id in tt_sets[model]))
        for case_id in sorted(tt_union - tt_intersection)
    ]

    report_title = {
        "model-best": "# Bridge all-token input audit",
        "dataset-shared": "# Bridge all-token input audit — dataset-shared metrics",
        "cv-best": "# Bridge all-token input audit — cross-validated best metrics",
        "cv-shared": "# Bridge all-token input audit — cross-validated dataset-shared metrics",
        "cor-best": "# Bridge all-token input audit — Spearman-selected metrics",
        "cor-shared": "# Bridge all-token input audit — Spearman dataset-shared metrics",
        "auroc-ensemble-best": "# Bridge all-token input audit — exhaustive AUROC ensembles",
        "auroc-ensemble-shared": "# Bridge all-token input audit — shared AUROC ensembles",
    }[args.metric_regime]
    selection_heading = {
        "model-best": "### Selected all-token winner availability",
        "dataset-shared": "### Selected dataset-shared metric availability",
        "cv-best": "### Cross-validated all-token winner availability",
        "cv-shared": "### Cross-validated dataset-shared winner availability",
        "cor-best": "### Per-model Spearman winner availability",
        "cor-shared": "### Dataset-shared Spearman winner availability",
        "auroc-ensemble-best": "### Per-model AUROC-ensemble winner availability",
        "auroc-ensemble-shared": "### Dataset-shared AUROC-ensemble winner availability",
    }[args.metric_regime]
    selection_explanation = {
        "model-best": "Per-model pooled-AUROC all-token winners are used.",
        "dataset-shared": (
            "For each dataset, one common metric maximizes the equal-model mean "
            "direction-adjusted AUROC `max(A, 1-A)` among metrics available to every model."
        ),
        "cv-best": (
            "Per-model winners maximize case-macro average precision under grouped "
            "cross-validation. Direction is learned on training cases; every token is "
            "eligible and non-finite values rank last. The displayed AUROC is the raw "
            "pooled continuity diagnostic, not the selection criterion."
        ),
        "cv-shared": (
            "One metric per dataset maximizes the equal-model mean of held-out "
            "case-macro average precision. Models and cases within each model are "
            "weighted equally; score direction is learned separately per model from "
            "training cases. The displayed AUROC is a pooled continuity diagnostic."
        ),
        "cor-best": (
            "Per-model winners maximize the absolute pooled Spearman correlation "
            "between finite metric values and binary on-task judgments. The full "
            "stored data determine every point estimate and winner; bootstrap "
            "replicates quantify descriptive case-cluster uncertainty only."
        ),
        "cor-shared": (
            "One metric per dataset maximizes the equal-model mean absolute pooled "
            "Spearman correlation. Tokens remain pooled within each model, models "
            "are weighted equally, and the correlation sign determines each model’s "
            "relevant direction."
        ),
        "auroc-ensemble-best": (
            "Every deployable original metric and all unordered 25/75, 50/50, and "
            "75/25 pairs compete by direction-adjusted pooled AUROC. Components are "
            "pooled fractional-midranks and direction-aligned before mixing. The "
            "winner is exploratory; grouped held-out diagnostics assess generalization."
        ),
        "auroc-ensemble-shared": (
            "One original or weighted pair per dataset maximizes equal-model mean "
            "direction-adjusted AUROC. The candidate ID and weights are shared while "
            "rank mappings and directions remain model-specific."
        ),
    }[args.metric_regime]

    report: list[str] = [
        report_title,
        "",
        "Generated from the canonical `paper_results/bridge/all_*.parquet` files. "
        "The four requested datasets exclude the authored pilot.",
        "",
        f"**Status:** {'FAIL' if hard_errors else 'PASS'} — {total_rows:,} rows across "
        f"{len(pairs)} dataset/model combinations.",
        "",
        "## Inventory and integrity",
        "",
        *_table(
            [
                "dataset",
                "model",
                "rows",
                "cases",
                "columns",
                "modes",
                "regions",
                "selected non-finite",
            ],
            inventory_rows,
        ),
        "",
        "Ordinary files have 23 columns. OPI has 24 because it additionally stores "
        "`attn_rollout`. Every file stores the persisted, transcript-z-scored `w`.",
        "",
        "### Schema and dtypes",
        "",
        *_table(["dataset", "model", "schema variant", "validation"], schema_rows),
        "",
        f"Ordinary schema: `{ordinary_schema_text}`.",
        "",
        f"OPI schema: `{opi_schema_text}`.",
        "",
        "All nine forward-pass signals, four Q2 activation-derived metrics, `act_norm`, "
        "and persisted `w` are present in every file. Position IDs and `(case_id, tok_idx)` "
        "are unique; each case is zero-based and contiguous; `token`, `label`, and `on_task` "
        "are complete.",
        "",
        selection_heading,
        "",
        *_table(
            [
                "dataset",
                "model",
                "metric",
                (
                    "pooled raw AUROC"
                    if args.metric_regime in {"cv-best", "cv-shared", "cor-best", "cor-shared"} or args.metric_regime.startswith("auroc-ensemble")
                    else "published AUROC"
                ),
                "relevant direction",
                "non-finite",
            ],
            selected_rows,
        ),
        "",
        selection_explanation,
        "",
        *(
            [
                "#### Cross-validated selection diagnostics",
                "",
                *_table(
                    [
                        "dataset",
                        "model",
                        "CV MAP",
                        "nested-CV MAP",
                        "runner-up",
                        "delta",
                        "paired 95% CI",
                        "tied",
                        "P@1%",
                        "P@5%",
                        "P@10%",
                        "beats position",
                        "beats region",
                    ],
                    cv_selection_rows,
                ),
                "",
                "Nested-CV MAP estimates the complete metric-and-direction selection "
                "procedure. Paired intervals resample complete cases and are "
                "stratified by Liars subdataset. A winner is still returned when tied "
                "or when it lacks demonstrated incremental value over a structural baseline.",
                "",
                *_table(
                    [
                        "dataset",
                        "model",
                        "cases/fold",
                        "mean length/fold",
                        "mean base rate/fold",
                        "direction stability",
                        "outer-fold winner frequency",
                    ],
                    cv_fold_rows,
                ),
                "",
            ]
            if args.metric_regime in {"cv-best", "cv-shared"}
            else []
        ),
        *(
            [
                "#### Spearman-correlation selection diagnostics",
                "",
                *_table(
                    [
                        "dataset",
                        "model",
                        "model rho",
                        (
                            "equal-model mean abs rho"
                            if args.metric_regime == "cor-shared"
                            else "abs rho"
                        ),
                        "runner-up",
                        "delta",
                        "paired 95% CI",
                        "tied",
                        "direction stability",
                        "finite",
                        "non-finite",
                        "content-only winner",
                        "winner changes",
                    ],
                    cor_selection_rows,
                ),
                "",
                "All rho point estimates and winners use the complete stored dataset. "
                "Intervals use fixed-rank whole-case bootstrap replicates solely for "
                "descriptive uncertainty, stratified by Liars source subdataset.",
                "",
                "With binary judgments, Spearman rho is closely related to the "
                "Mann–Whitney statistic underlying AUROC. Thus cor-best is an "
                "alternative pooled rank-association summary, not independent evidence.",
                "",
            ]
            if args.metric_regime in {"cor-best", "cor-shared"}
            else []
        ),
        *(
            [
                "#### Exhaustive rank-ensemble selection diagnostics",
                "",
                *_table(
                    [
                        "dataset", "model", "candidate", "components", "model AUROC",
                        "shared mean AUROC", "gain vs original", "runner-up", "delta",
                        "common-support winner", "common winner changes",
                        "held-out AUROC", "held-out macro delta",
                    ],
                    ensemble_selection_rows,
                ),
                "",
                "The blind pool contains 13 originals and 234 weighted pairs (247 "
                "candidates). OPI attn_rollout appears only in the explicitly labeled "
                "non-blind sensitivity. Missing component values use pairwise deletion; "
                "underlying token rows remain present.",
                "",
                *(
                    [
                        "##### Dataset-shared leave-one-model-out validation",
                        "",
                        *_table(
                            [
                                "dataset", "held-out model", "protocol", "candidate",
                                "case-macro AUROC", "case-macro AP",
                                "delta vs original", "paired 95% CI",
                                "top-1% precision", "weak evidence",
                            ],
                            ensemble_lomo_rows,
                        ),
                        "",
                        "Label-free transfer uses no held-out-model judgment labels for "
                        "directions. Target-calibrated transfer learns directions only on "
                        "target-training folds. Liars has one training model per holdout "
                        "and is marked weak evidence.",
                        "",
                    ]
                    if ensemble_lomo_rows
                    else []
                ),
                "The input/boundary/output selector report independently recomputes ranks "
                "and winners within each segment. This asks which candidate discriminates "
                "best when selection is restricted to that segment. The fixed-winner "
                "segment table instead preserves the all-token mapping and is the valid "
                "apples-to-apples comparison across segments. Liars trailers are absent.",
                "",
            ]
            if args.metric_regime.startswith("auroc-ensemble")
            else []
        ),
        "Below-chance metrics are retained and direction-reversed only in derived aligned scores.",
        "",
        "### Persisted `w` validation",
        "",
        *_table(["dataset", "model", "maximum absolute error"], w_rows),
        "",
        "The comparison recomputes `z(sink_drain) - z(lookback_ratio)` over eight "
        "deterministically selected complete transcripts per file.",
        "",
        "### Null, NaN, and infinite metric values",
        "",
        *_table(
            ["dataset", "model", "metric", "null", "NaN", "+inf", "-inf"],
            nonfinite_rows,
        ),
        "",
        "Difference signals are expected to be unavailable at the first token because "
        "there is no predecessor. Source values are preserved; downstream derived "
        "scores remain null.",
        "",
        "## Evaluation and base-rate differences",
        "",
        "| dataset | available sequence | label meaning | judge target | interpretation |",
        "|---|---|---|---|---|",
        "| OPI | input plus generation boundary; no response | injected span | embedded instruction | sparse selector task |",
        "| Tensor Trust | prompt, boundary, response | attack/access-code case | following input steering | dense boundary case |",
        "| Liars | full on-policy transcript | deceptive transcript | lying/concealment | extremely sparse |",
        "| taboo | prompt, boundary, response | no negative span | secret word/concept | sparse selector task |",
        "",
        *_table(
            ["dataset", "model", "pooled rate", "case q25", "case median", "case q75", "regime"],
            base_rows,
        ),
        "",
        "All files use the same deterministic judge model but different threat-specific "
        "questions. Tensor Trust remains measurable but has little practical room for "
        "selection because most positions are already on-task.",
        "",
        "Model NLA layers/dimensions: "
        + "; ".join(
            f"{model} layer {info['layer']}/{info['d_model']}" for model, info in MODEL_INFO.items()
        )
        + ".",
        "",
        "## Cross-model case coverage",
        "",
        *_table(
            ["dataset", "union", "intersection", "intersection/union", "per-model cases"],
            coverage_rows,
        ),
        "",
        "Liars is disjoint on-policy data: the loader first filters each benchmark "
        "subset by generator model and only then samples, so each probed model sees "
        "transcripts that same model generated. Gemma-27B uses instructed-deception, "
        "insider-trading, and convincing-game; Llama-70B uses those three plus "
        "harm-pressure-choice and harm-pressure-knowledge-report. The two 2,000-case "
        "sets have zero shared IDs, so comparisons are distributional rather than paired.",
        "",
        "### Liars subdataset and class composition",
        "",
        *_table(
            ["model", "subdataset", "label", "cases"],
            [
                [model, subset, label, count]
                for (model, label, subset), count in sorted(liars_composition.items())
            ],
        ),
        "",
        "### Tensor Trust cases not present for all four models",
        "",
        *_table(
            ["case_id", "models present"], [[case, present] for case, present in tt_incomplete]
        ),
        "",
        "Published parquets do not retain exact skip reasons. Possible extraction paths "
        "include an unavailable original response, tokenizer round-trip rejection, span "
        "reconstruction failure, or a scoring exception.",
        "",
        "## Tokenization and sequence structure",
        "",
        *_table(
            [
                "dataset",
                "model",
                "full median",
                "input median",
                "boundary median",
                "output median",
                "trailer median",
                "boundary example",
            ],
            length_rows,
        ),
        "",
        "### Derived boundary corrections",
        "",
        *_table(
            [
                "dataset",
                "model",
                "case_id",
                "source boundary",
                "analysis boundary",
                "relabeled content tokens",
            ],
            boundary_correction_rows,
        ),
        "",
        "The canonical source `region` values are preserved as `source_region`. "
        "For these TT cases, single-character user content was left as `template` "
        "by substring/offset span matching, causing the final user turn to merge "
        "with the assistant-generation boundary. The derived table retains the "
        "source length, assigns the final five template tokens to `boundary`, moves "
        "the preceding user turn to `input`, and records the corrected content role "
        "in `analysis_region`.",
        "",
        *_table(
            ["dataset", "model", "`Ġ`", "`Ċ`", "`▁`", "`âĢ…`", "leading `<`"],
            marker_rows,
        ),
        "",
        "The strings come directly from `tokenizer.convert_ids_to_tokens(ids)`. `Ġ` and "
        "`Ċ` are byte-level BPE whitespace/newline markers; `▁` is SentencePiece "
        "metaspace; forms such as `âĢĿ` are byte-to-Unicode renderings of UTF-8 bytes; "
        "angle-bracket values are model-specific chat-template special tokens. They are "
        "preserved verbatim as `token_raw`, not lossily decoded.",
        "",
        "Shared case IDs do not imply shared token positions. Model comparisons should "
        "use segment-normalized positions or model-specific bins.",
        "",
        "## Response provenance",
        "",
        *_table(
            ["dataset", "model", "output tokens", "reconstructed cases", "reconstructed tokens"],
            provenance_rows,
        ),
        "",
        "TT/taboo stored at most 30 historical response tokens. A deterministic "
        "continuation reconstructed from a stored prefix means replaying that exact "
        "prefix and greedily generating the missing suffix (no sampling), stopping at a "
        "special token or the original 64-token TT / 48-token taboo budget. It recovers a "
        "reproducible completion, not necessarily history: for temperature-sampled taboo "
        "cases the unavailable historical sampled tail cannot be reconstructed.",
        "",
        "## Template sensitivity and fixed budgets",
        "",
        *_table(
            [
                "dataset",
                "model",
                (
                    "all winner"
                    if args.metric_regime == "model-best"
                    else "shared metric"
                    if args.metric_regime == "dataset-shared"
                    else "CV winner"
                    if args.metric_regime == "cv-best"
                    else "CV-shared winner"
                    if args.metric_regime == "cv-shared"
                    else "Spearman winner"
                    if args.metric_regime == "cor-best"
                    else "Spearman-shared winner"
                    if args.metric_regime == "cor-shared"
                    else "AUROC-ensemble winner"
                    if args.metric_regime == "auroc-ensemble-best"
                    else "shared AUROC-ensemble winner"
                ),
                (
                    "repository pooled content winner"
                    if args.metric_regime in {"cv-best", "cv-shared"}
                    else "segment-specific winners in selector report"
                    if args.metric_regime.startswith("auroc-ensemble")
                    else "content winner"
                ),
                "all base",
                "content base",
                "all AUROC",
                "content AUROC",
                "winner changes",
                "top 1% precision",
                "top 1% template",
                "top 10% precision",
                "top 10% template",
            ],
            template_rows,
        ),
        "",
        'Content-only means `region != "template"`, matching `bridge_report.py`. '
        + {
            "model-best": "Budget rows rank the selected all-token winner independently within each case. ",
            "dataset-shared": "Budget rows rank the selected dataset-shared metric independently within each case. ",
            "cv-best": "Budget rows rank the cross-validated winner independently within each case. ",
            "cv-shared": "Budget rows rank the cross-validated shared winner independently within each case. ",
            "cor-best": "Budget rows rank the per-model Spearman winner independently within each case. ",
            "cor-shared": "Budget rows rank the dataset-shared Spearman winner independently within each case. ",
            "auroc-ensemble-best": "Budget rows rank the selected per-model ensemble independently within each case. ",
            "auroc-ensemble-shared": "Budget rows rank the selected dataset-shared ensemble independently within each case. ",
        }[args.metric_regime]
        + ("Signals are pooled only by the selected rank ensemble."
           if args.metric_regime.startswith("auroc-ensemble") else "No signal pooling is performed."),
        *(
            [
                "For correlation regimes, the content-only winner is reselected by "
                "absolute Spearman rho after excluding raw repository template rows "
                "and recomputing ranks. This diagnostic never replaces the primary "
                "all-token winner; extreme-norm content rows remain included.",
                "",
            ]
            if args.metric_regime in {"cor-best", "cor-shared"}
            else []
        ),
        *(
            [
                "The displayed repository content winner was selected by the older "
                "pooled-AUROC procedure and is included only for provenance. It is not "
                "treated as a CV content-only re-selection or marked as a winner change.",
                "",
            ]
            if args.metric_regime in {"cv-best", "cv-shared"}
            else []
        ),
        "### Selected-score quantiles",
        "",
        "Columns are the 1st, 25th, 50th, 75th, and 99th percentiles. Aligned scores "
        "are sign-flipped when lower raw values are relevant.",
        "",
        *_table(
            [
                "dataset",
                "model",
                "raw p01",
                "raw p25",
                "raw p50",
                "raw p75",
                "raw p99",
                "aligned p01",
                "aligned p25",
                "aligned p50",
                "aligned p75",
                "aligned p99",
            ],
            distribution_rows,
        ),
        "",
        "### Direction-aligned selected score by derived segment",
        "",
        *_table(["dataset", "model", "segment", "tokens", "mean", "std"], segment_rows),
        "",
        "### Direction-aligned selected score by raw repository region",
        "",
        *_table(["dataset", "model", "source region", "tokens", "mean", "std"], region_rows),
        "",
        "## Extreme activation proxy",
        "",
        *_table(
            [
                "dataset",
                "model",
                "`norm_ratio > 5`",
                "README count",
                "share",
                "on-task extreme",
                "on-task ordinary",
                "at position 0",
                "position median",
                "position max",
                "segment counts",
            ],
            extreme_rows,
        ),
        "",
        "`is_extreme_norm` is a token-level analysis proxy, not the factor-of-10 "
        "spike-channel definition in `all_tokens_eval.py`. No token is removed. Sparse "
        "tasks and dense Tensor Trust must be interpreted separately.",
        "",
        "## Position and length diagnostics",
        "",
        "Bin means are ordered 0–256, 256–512, 512–1024, 1024–2048, and 2048+.",
        "",
        *_table(
            [
                "dataset",
                "model",
                "metric",
                "corr absolute position",
                "corr normalized position",
                "corr full length",
                "corr segment length",
                "within-case z mean",
                "within-case z std",
                "percentile min",
                "percentile max",
                "aligned bin means",
            ],
            position_rows,
        ),
        "",
        "Within-case z-scores use sample standard deviation with the repository epsilon; "
        "percentiles run from 0 (least relevant) to 1 (most relevant) after direction "
        "alignment. They are retained for later fixed-budget analyses.",
        "",
        "### Gemma sink-window diagnostic",
        "",
        *_table(
            ["dataset", "model", "sink before 1024", "sink at/after 1024", "tokens after"],
            gemma_sink_rows,
        ),
        "",
        "The post-1024 flag documents contamination of Gemma `sink_drain`; it is not "
        "applied to unrelated metrics.",
        "",
        "## Interpretation constraints",
        "",
        "- Internal-state metrics may partly identify states the NLA verbalizes poorly, "
        "rather than positions that are intrinsically meaningful.",
        "- Narrow intervals quantify sampling precision, not validity of the judge's "
        "definition of on-task.",
        "- Template, extreme-activation, dense-task, and post-window rows are retained "
        "and explicitly labelled rather than filtered.",
        "",
        "## Audit findings",
        "",
    ]
    if hard_errors:
        report.extend(["### Hard failures", "", *[f"- {item}" for item in hard_errors], ""])
    else:
        report.extend(["No hard integrity failures were found.", ""])
    if warnings:
        report.extend(["### Expected warnings", "", *[f"- {item}" for item in warnings], ""])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report).rstrip() + "\n")
    status = "FAIL" if hard_errors else "PASS"
    if log_stages:
        _stage(
            f"wrote {report_path} ({status}, {total_rows:,} rows) in "
            f"{time.perf_counter() - started:.1f}s"
        )
    else:
        print(f"wrote {report_path} ({status}, {total_rows:,} rows)")
    return 1 if hard_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
