from __future__ import annotations

import numpy as np
import polars as pl

from token_analysis.selection_methods.auroc_ensemble.select_metrics import (
    Candidate,
    align_candidate,
    candidate_definitions,
    candidate_score,
    derive_segments,
    evaluate_pool,
    fractional_midranks,
    ordered_results,
    segment_diagnostics,
    shared_heldout_validation,
    shared_lomo_validation,
    _fold_contexts,
    _shared_fold_map,
    _shared_training_order,
)
from token_analysis.selection_methods.statistics import auroc


def test_candidate_inventory_has_originals_and_all_unordered_weighted_pairs():
    primary = candidate_definitions([f"m{i}" for i in range(13)])
    sensitivity = candidate_definitions([f"m{i}" for i in range(14)])
    assert len(primary) == 247
    assert len(sensitivity) == 287
    assert sum(candidate.candidate_type == "single" for candidate in primary) == 13
    assert len({candidate.candidate_id for candidate in primary}) == 247
    assert all(
        candidate.component_2 is None or candidate.component_1 < candidate.component_2
        for candidate in primary
    )


def test_fractional_midranks_preserve_ties_bounds_and_auroc():
    values = np.array([4.0, 1.0, 1.0, 3.0, np.nan, np.inf])
    labels = np.array([1, 0, 1, 0, 1, 0])
    normalized = fractional_midranks(values)
    assert np.allclose(normalized[:4], [1.0, 1 / 6, 1 / 6, 2 / 3])
    assert np.isnan(normalized[4:]).all()
    assert np.nanmin(normalized) >= 0 and np.nanmax(normalized) <= 1
    assert np.isclose(auroc(values, labels), auroc(normalized, labels))


def test_high_and_low_components_align_and_original_can_win():
    labels = np.array([0, 0, 1, 1], dtype=np.int8)
    values = {
        "high": np.array([0.0, 1.0, 2.0, 3.0]),
        "low": np.array([3.0, 2.0, 1.0, 0.0]),
    }
    candidates = candidate_definitions(values)
    results, context = evaluate_pool(values, labels, np.ones(4, bool), candidates)
    assert context.directions == {"high": "higher", "low": "lower"}
    assert np.allclose(context.aligned["high"], context.aligned["low"])
    winner = ordered_results(results).row(0, named=True)
    assert winner["candidate_type"] == "single"
    assert winner["candidate_id"] == "single::high"


def test_pairwise_missingness_is_not_imputed():
    labels = np.array([0, 0, 1, 1], dtype=np.int8)
    values = {
        "a": np.array([0.0, 1.0, 2.0, np.nan]),
        "b": np.array([0.0, np.nan, 2.0, 3.0]),
    }
    candidates = candidate_definitions(values)
    results, context = evaluate_pool(values, labels, np.ones(4, bool), candidates)
    pair = next(candidate for candidate in candidates if candidate.candidate_type == "ensemble")
    score = candidate_score(context, pair)
    assert np.isfinite(score).sum() == 2
    assert results.filter(pl.col("candidate_id") == pair.candidate_id)["finite_count"][0] == 2


def test_final_candidate_direction_can_reverse():
    score = np.array([0.9, 0.8, 0.2, np.nan])
    assert np.allclose(align_candidate(score, "lower")[:3], [0.1, 0.2, 0.8])
    assert np.isnan(align_candidate(score, "lower")[3])


def test_tt_correction_and_standard_segmentation():
    case_ids = np.array(["case"] * 11)
    positions = np.arange(11)
    regions = np.array(["user"] * 3 + ["template"] * 6 + ["assistant"] * 2)
    segments, analysis = derive_segments(case_ids, positions, regions, "tt")
    assert segments.tolist() == ["input"] * 4 + ["boundary"] * 5 + ["output"] * 2
    assert analysis[3] == "user"
    assert np.all(analysis[4:9] == "template")


def test_liars_prior_assistant_is_input_and_trailer_remains_separate():
    case_ids = np.array(["case"] * 9)
    positions = np.arange(9)
    regions = np.array(
        ["template", "user", "assistant_prior", "template", "user", "template", "assistant", "assistant", "template"]
    )
    segments, _ = derive_segments(case_ids, positions, regions, "liars")
    assert segments.tolist() == ["input"] * 5 + ["boundary"] + ["output"] * 2 + ["trailer"]


def test_segment_diagnostics_exclude_trailers_and_are_case_balanced():
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    case_ids = np.array(["a"] * 4 + ["b"] * 4)
    segments = np.array(["input", "boundary", "output", "trailer"] * 2)
    values = {"m": np.arange(8, dtype=float)}
    candidates = [Candidate("single::m", "single", "m", 1.0)]
    results, context = evaluate_pool(values, labels, np.ones(8, bool), candidates)
    row = results.row(0, named=True)
    summary, contrasts = segment_diagnostics(
        dataset="liars",
        model="g27",
        candidate=candidates[0],
        context=context,
        candidate_direction=row["candidate_direction"],
        labels=labels,
        case_ids=case_ids,
        segments=segments,
        analysis_regions=np.array(["user", "template", "assistant", "template"] * 2),
        source_regions=np.array(["user", "template", "assistant", "template"] * 2),
        probe_tok_idx=np.zeros(8, dtype=int),
        bootstrap_resamples=10,
        seed=0,
        regime="test",
    )
    assert summary.height == 3
    assert "trailer" not in summary["segment"].to_list()
    assert summary["token_count"].sum() == 6
    assert contrasts.height == 3



def _shared_synthetic_outputs():
    candidates = candidate_definitions(["entropy", "surprisal"])
    outputs = {}
    case_ids = np.repeat([f"shared-{index}" for index in range(6)], 4)
    labels = np.tile(np.array([0, 0, 1, 1], dtype=np.int8), 6)
    case_labels = np.repeat(np.arange(6) % 2, 4)
    positions = np.tile(np.arange(4), 6)
    regions = np.tile(np.array(["user", "template", "assistant", "assistant"]), 6)
    segments = np.tile(np.array(["input", "boundary", "output", "output"]), 6)
    for model, reverse in (("g27", False), ("l70", True)):
        primary = labels.astype(float) + positions * 0.01
        if reverse:
            primary = 1 - primary
        values = {
            "entropy": primary,
            "surprisal": np.sin(np.arange(len(labels))) + labels * 0.2,
        }
        results, context = evaluate_pool(
            values,
            labels,
            np.ones(len(labels), dtype=bool),
            candidates,
        )
        outputs[("liars", model)] = {
            "values": values,
            "labels": labels.copy(),
            "case_ids": case_ids.copy(),
            "case_labels": case_labels.copy(),
            "segments": segments.copy(),
            "analysis_regions": regions.copy(),
            "source_regions": regions.copy(),
            "probe": np.zeros(len(labels), dtype=np.int64),
            "full_context": context,
            "candidates": candidates,
            "results": results.with_columns(
                pl.lit("liars").alias("dataset"),
                pl.lit(model).alias("model"),
            ),
        }
    return outputs


def test_shared_fold_map_keeps_cross_model_case_ids_together_and_is_deterministic():
    outputs = _shared_synthetic_outputs()
    for model in ("g27", "l70"):
        outputs[("liars", model)]["case_labels"][:4] = np.array([0, 1, 0, 1])
    # OPI-like token span labels reduce to an any-positive case stratum.
    first = _shared_fold_map(outputs, "liars", folds=3, seed=7)
    second = _shared_fold_map(outputs, "liars", folds=3, seed=7)
    assert first == second
    assert set(first) == {f"shared-{index}" for index in range(6)}
    assert set(first.values()) == {0, 1, 2}


def test_shared_training_order_weights_models_not_tokens():
    candidates = candidate_definitions(["entropy", "surprisal"])
    rows = {}
    for model, first_score, second_score in (
        ("g27", 0.9, 0.5),
        ("l70", 0.4, 0.9),
    ):
        rows[model] = pl.DataFrame(
            [
                {
                    "model": model,
                    "candidate_id": candidates[0].candidate_id,
                    "candidate_type": "single",
                    "component_1": candidates[0].component_1,
                    "component_1_weight": 1.0,
                    "component_2": None,
                    "component_2_weight": 0.0,
                    "candidate_direction": "higher",
                    "adjusted_auroc": first_score,
                    "nonfinite_rate": 0.0,
                    "eligible": True,
                    "candidate_cost_order": 0,
                },
                {
                    "model": model,
                    "candidate_id": candidates[1].candidate_id,
                    "candidate_type": "single",
                    "component_1": candidates[1].component_1,
                    "component_1_weight": 1.0,
                    "component_2": None,
                    "component_2_weight": 0.0,
                    "candidate_direction": "higher",
                    "adjusted_auroc": second_score,
                    "nonfinite_rate": 0.0,
                    "eligible": True,
                    "candidate_cost_order": 0,
                },
            ]
        )
    ordered = _shared_training_order(rows, ["g27", "l70"])
    assert ordered["candidate_id"][0] == candidates[1].candidate_id
    assert ordered["adjusted_auroc"][0] == 0.7


def test_shared_heldout_validation_shapes_directions_and_determinism():
    outputs = _shared_synthetic_outputs()
    first = shared_heldout_validation(
        outputs,
        "liars",
        "single::entropy",
        folds=3,
        bootstrap_resamples=20,
        seed=3,
    )
    second = shared_heldout_validation(
        outputs,
        "liars",
        "single::entropy",
        folds=3,
        bootstrap_resamples=20,
        seed=3,
    )
    assert first["fold_diagnostics"].height == 2 * 3 * 2
    assert first["case_scores"].height == 2 * 6 * 2
    assert first["headline"].height == 2
    assert first["headline"].to_dicts() == second["headline"].to_dicts()
    directions = (
        first["fold_diagnostics"]
        .filter(pl.col("procedure") == "all_candidates")
        .group_by("model")
        .agg(pl.col("candidate_direction").unique())
    )
    assert directions.height == 2
    assert set(first["fold_diagnostics"]["procedure"]) == {
        "all_candidates",
        "original_only",
    }


def test_lomo_reports_label_free_and_calibrated_protocols_without_target_label_leakage():
    outputs = _shared_synthetic_outputs()
    result = shared_lomo_validation(
        outputs,
        "liars",
        folds=3,
        bootstrap_resamples=20,
        seed=5,
    )
    repeated = shared_lomo_validation(
        outputs,
        "liars",
        folds=3,
        bootstrap_resamples=20,
        seed=5,
    )
    assert result["fold_diagnostics"].equals(repeated["fold_diagnostics"])
    assert result["case_scores"].equals(repeated["case_scores"])
    assert result["summary"].equals(repeated["summary"])

    diagnostics = result["fold_diagnostics"]
    assert diagnostics.height == 2 * 2 * 3 * 2
    assert result["summary"].height == 2 * 2 * 2
    assert set(diagnostics["protocol"]) == {"label_free", "target_calibrated"}
    assert not diagnostics.filter(
        pl.col("protocol") == "label_free"
    )["target_labels_used_for_direction"].any()
    assert diagnostics.filter(
        pl.col("protocol") == "target_calibrated"
    )["target_labels_used_for_direction"].all()
    assert result["summary"]["weak_model_transfer_evidence"].all()
    assert not result["summary"]["model_population_interval"].any()

    flipped = _shared_synthetic_outputs()
    flipped[("liars", "g27")]["labels"] = (
        1 - flipped[("liars", "g27")]["labels"]
    )
    flipped[("liars", "g27")]["case_labels"] = (
        1 - flipped[("liars", "g27")]["case_labels"]
    )
    changed = shared_lomo_validation(
        flipped,
        "liars",
        folds=3,
        bootstrap_resamples=20,
        seed=5,
    )
    columns = [
        "fold",
        "procedure",
        "candidate_id",
        "component_1_direction",
        "component_2_direction",
        "candidate_direction",
    ]
    original_label_free = (
        diagnostics.filter(
            (pl.col("heldout_model") == "g27")
            & (pl.col("protocol") == "label_free")
        )
        .select(columns)
        .sort("fold", "procedure")
    )
    changed_label_free = (
        changed["fold_diagnostics"]
        .filter(
            (pl.col("heldout_model") == "g27")
            & (pl.col("protocol") == "label_free")
        )
        .select(columns)
        .sort("fold", "procedure")
    )
    assert original_label_free.equals(changed_label_free)
    case_columns = ["case_id", "fold", "procedure", "candidate_id"]
    original_case_folds = (
        result["case_scores"]
        .filter(
            (pl.col("heldout_model") == "g27")
            & (pl.col("protocol") == "label_free")
        )
        .select(case_columns)
        .sort("case_id", "procedure")
    )
    changed_case_folds = (
        changed["case_scores"]
        .filter(
            (pl.col("heldout_model") == "g27")
            & (pl.col("protocol") == "label_free")
        )
        .select(case_columns)
        .sort("case_id", "procedure")
    )
    assert original_case_folds.equals(changed_case_folds)


def test_grouped_validation_fits_directions_and_selection_without_test_labels():
    outputs = _shared_synthetic_outputs()
    fold_map = _shared_fold_map(outputs, "liars", folds=3, seed=11)
    component_directions = {}
    for model in ("g27", "l70"):
        output = outputs[("liars", model)]
        fold_values = np.array(
            [fold_map[case_id] for case_id in output["case_ids"]]
        )
        train_context, _ = _fold_contexts(
            output, fold_values != 0, fold_values == 0
        )
        component_directions[model] = train_context.directions["entropy"]
    assert component_directions == {"g27": "higher", "l70": "lower"}

    original = shared_heldout_validation(
        outputs,
        "liars",
        "single::entropy",
        folds=3,
        bootstrap_resamples=0,
        seed=11,
    )
    changed_outputs = _shared_synthetic_outputs()
    for model in ("g27", "l70"):
        output = changed_outputs[("liars", model)]
        fold_values = np.array(
            [fold_map[case_id] for case_id in output["case_ids"]]
        )
        output["labels"][fold_values == 0] = 1 - output["labels"][fold_values == 0]
    changed = shared_heldout_validation(
        changed_outputs,
        "liars",
        "single::entropy",
        folds=3,
        bootstrap_resamples=0,
        seed=11,
    )
    columns = [
        "model",
        "procedure",
        "candidate_id",
        "candidate_direction",
        "train_equal_model_adjusted_auroc",
        "train_model_adjusted_auroc",
    ]
    original_fold = (
        original["fold_diagnostics"]
        .filter(pl.col("fold") == 0)
        .select(columns)
        .sort("model", "procedure")
    )
    changed_fold = (
        changed["fold_diagnostics"]
        .filter(pl.col("fold") == 0)
        .select(columns)
        .sort("model", "procedure")
    )
    assert original_fold.equals(changed_fold)


def test_shared_comparator_is_selected_independently_from_ensemble_pool():
    outputs = _shared_synthetic_outputs()
    frames = {}
    for model in ("g27", "l70"):
        frames[model] = outputs[("liars", model)]["results"].with_columns(
            pl.when(pl.col("candidate_type") == "ensemble")
            .then(0.9)
            .otherwise(0.7)
            .alias("adjusted_auroc"),
            pl.lit(0.0).alias("nonfinite_rate"),
            pl.lit(True).alias("eligible"),
        )
    ordered = _shared_training_order(frames, ["g27", "l70"])
    ensemble_selection = ordered.row(0, named=True)
    single_selection = ordered.filter(
        pl.col("candidate_type") == "single"
    ).row(0, named=True)
    assert ensemble_selection["candidate_type"] == "ensemble"
    assert single_selection["candidate_type"] == "single"
    assert ensemble_selection["candidate_id"] != single_selection["candidate_id"]


def test_shared_validation_keeps_cases_without_both_token_classes_explicit():
    outputs = _shared_synthetic_outputs()
    for model in ("g27", "l70"):
        outputs[("liars", model)]["labels"][:4] = 0
    result = shared_heldout_validation(
        outputs,
        "liars",
        "single::entropy",
        folds=3,
        bootstrap_resamples=0,
        seed=13,
    )
    missing_class = result["case_scores"].filter(
        pl.col("case_id") == "shared-0"
    )
    assert missing_class.height == 4
    assert missing_class["auc"].is_nan().all()
    assert missing_class["ap"].is_finite().all()
