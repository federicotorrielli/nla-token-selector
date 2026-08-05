import polars as pl
import pytest

from token_analysis.audit_bridge_inputs import _w_error
from token_analysis.build_token_position_data import transform_source
from token_analysis.common import z_expr


def source_frame(regions, metric, values, *, probes=None, tokens=None, case_id="case"):
    n = len(regions)
    probes = [-1] * n if probes is None else probes
    tokens = [f"t{i}" for i in range(n)] if tokens is None else tokens
    frame = pl.DataFrame(
        {
            "position_id": [f"p{i}" for i in range(n)],
            "case_id": [case_id] * n,
            "mode": ["test"] * n,
            "tok_idx": list(range(n)),
            "token": tokens,
            "label": [1] * n,
            "region": regions,
            "probe_tok_idx": probes,
            "on_task": [i % 2 for i in range(n)],
            metric: values,
            "norm_ratio": [6.0] + [1.0] * (n - 1),
            "w": [0.0] * n,
        }
    )
    for shared_metric in (
        "peak_ratio",
        "resid_jump_nla",
        "head_disagreement",
        "dominant_mass",
    ):
        if shared_metric not in frame.columns:
            frame = frame.with_columns(pl.Series(shared_metric, values))
    return frame


def transformed(frame, dataset, model, coverage_count=None):
    counts = {"opi": 4, "tt": 4, "taboo": 4, "liars": 2}
    coverage = {(dataset, frame["case_id"][0]): coverage_count or counts[dataset]}
    return transform_source(frame.lazy(), dataset, model, coverage).collect()


def test_standard_segmentation_and_response_origin():
    frame = source_frame(
        [
            "template",
            "system",
            "template",
            "user",
            "template",
            "assistant",
            "assistant",
            "template",
        ],
        "resid_jump",
        [float("nan"), 1, 2, 3, 4, 5, 6, 7],
        probes=[-1, -1, -1, -1, -1, 0, -1, -1],
        tokens=["<s>", "sys", "Ċ", "user", "<a>", "hello", "Ġthere", "</a>"],
    )
    out = transformed(frame, "tt", "q7")
    assert out["segment"].to_list() == ["input"] * 4 + ["boundary"] + ["output"] * 2 + ["trailer"]
    assert out.select("input_length", "boundary_length", "output_length", "trailer_length").row(
        0
    ) == (4, 1, 2, 1)
    assert out["segment_position"].to_list() == [0, 1, 2, 3, 0, 0, 1, 0]
    assert out["response_token_origin"].to_list()[5:7] == ["stored_prefix", "reconstructed_tail"]
    assert out["is_reconstructed_tail"].sum() == 1
    assert out["token_raw"].to_list() == frame["token"].to_list()
    assert out["metric_value_raw"][0] != out["metric_value_raw"][0]
    assert out["metric_available"][0] is False
    assert out["metric_value_aligned"][0] is None


def test_tt_mislabeled_single_character_user_turn_is_not_boundary():
    regions = ["system"] + ["template"] * 11 + ["assistant"] * 2
    tokens = [
        "system",
        "<turn_end>",
        "<user_start>",
        "user",
        "<header_end>",
        "newline",
        "a",
        "<turn_end>",
        "<assistant_start>",
        "assistant",
        "<header_end>",
        "newline",
        "access",
        "granted",
    ]
    frame = source_frame(
        regions,
        "resid_jump",
        list(range(len(regions))),
        tokens=tokens,
        case_id="170140834572836_access_code",
    )

    out = transformed(frame, "tt", "q7")

    assert out["source_boundary_length"].unique().to_list() == [11]
    assert out["boundary_length"].unique().to_list() == [5]
    assert out["boundary_was_corrected"].all()
    assert out["segment"].to_list() == ["input"] * 7 + ["boundary"] * 5 + ["output"] * 2
    corrected_user = out.filter(pl.col("token_raw") == "a").row(0, named=True)
    assert corrected_user["source_region"] == "template"
    assert corrected_user["analysis_region"] == "user"
    assert corrected_user["is_content"] is True
    assert out.filter(pl.col("segment") == "boundary")["token_raw"].to_list() == [
        "<turn_end>",
        "<assistant_start>",
        "assistant",
        "<header_end>",
        "newline",
    ]


def test_opi_segmentation_has_no_response_and_reverses_ranking():
    frame = source_frame(
        ["template", "system", "template", "user", "template", "template"],
        "peak_ratio",
        [1, 2, 3, 4, 5, 6],
    )
    out = transformed(frame, "opi", "q7")
    assert out["segment"].to_list() == ["input"] * 4 + ["boundary"] * 2
    assert out["output_length"].unique().to_list() == [0]
    assert not out["response_available"].any()
    assert out["response_token_origin"].unique().to_list() == ["not_available"]
    assert out["metric_value_aligned"].to_list() == [-1, -2, -3, -4, -5, -6]
    assert out["metric_percentile_within_case"][0] == pytest.approx(1.0)
    assert out["metric_percentile_within_case"][-1] == pytest.approx(0.0)


def test_liars_prior_turn_and_trailer():
    frame = source_frame(
        [
            "template",
            "user",
            "template",
            "assistant_prior",
            "template",
            "user",
            "template",
            "assistant",
            "template",
        ],
        "head_disagreement",
        list(range(9)),
        case_id="instructed-deception_17",
    )
    out = transformed(frame, "liars", "g27")
    assert out["segment"].to_list() == ["input"] * 6 + ["boundary", "output", "trailer"]
    assert out.filter(pl.col("source_region") == "assistant_prior")["segment"].item() == "input"
    assert (
        out.filter(pl.col("segment") == "output")["response_token_origin"].item()
        == "dataset_original"
    )
    assert out["source_subdataset"].unique().to_list() == ["instructed-deception"]


def test_coverage_extreme_window_and_metadata():
    frame = source_frame(["user", "template", "assistant"], "resid_jump_nla", [1, 2, 3])
    shared = transformed(frame, "tt", "g27", coverage_count=4)
    partial = transformed(frame, "tt", "g27", coverage_count=3)
    assert shared["case_shared_across_all_models"].all()
    assert not partial["case_shared_across_all_models"].any()
    assert shared["model_coverage_count"].unique().to_list() == [4]
    assert shared["metric_winner_changes_content_only"].all()
    assert shared["selection_regime"].unique().to_list() == ["dense_boundary_case"]
    assert shared["is_extreme_norm"].to_list() == [True, False, False]
    assert not shared["past_gemma_local_window"].any()

    long = source_frame(["user"] * 1025, "norm_ratio", [1.0] * 1025)
    long_out = transformed(long, "tt", "g12")
    assert long_out["past_gemma_local_window"].sum() == 1
    assert long_out["past_gemma_local_window"][-1]


def test_case_and_dataset_base_rates_and_disjoint_coverage():
    first = source_frame(["user", "template"], "dominant_mass", [1, 2], case_id="a").with_columns(
        pl.Series("on_task", [0, 0])
    )
    second = source_frame(["user", "template"], "dominant_mass", [3, 4], case_id="b").with_columns(
        pl.Series("on_task", [1, 1])
    )
    frame = pl.concat([first, second]).with_columns(
        pl.arange(0, 4).cast(pl.String).alias("position_id")
    )
    coverage = {("taboo", "a"): 4, ("taboo", "b"): 1}
    out = transform_source(frame.lazy(), "taboo", "q7", coverage).collect()
    assert out.filter(pl.col("case_id") == "a")["case_on_task_rate"][0] == 0
    assert out.filter(pl.col("case_id") == "b")["case_on_task_rate"][0] == 1
    assert out["dataset_model_base_rate"][0] == pytest.approx(0.5)
    assert out.filter(pl.col("case_id") == "b")["case_shared_across_all_models"].not_().all()


def test_persisted_w_validation():
    frame = pl.DataFrame(
        {
            "case_id": ["a"] * 4 + ["b"] * 4,
            "sink_drain": [1.0, 2.0, 4.0, 8.0, 3.0, 4.0, 8.0, 9.0],
            "lookback_ratio": [8.0, 4.0, 2.0, 1.0, 9.0, 8.0, 4.0, 3.0],
        }
    ).with_columns((z_expr("sink_drain") - z_expr("lookback_ratio")).alias("w"))
    assert _w_error(frame) < 1e-12
    assert _w_error(frame.with_columns((pl.col("w") + 0.1).alias("w"))) == pytest.approx(0.1)


def test_all_rows_have_exactly_one_segment():
    frame = source_frame(
        ["template", "user", "template", "assistant", "assistant", "template"],
        "dominant_mass",
        [1] * 6,
    )
    out = transformed(frame, "taboo", "q7")
    assert out.height == frame.height
    assert out["segment"].null_count() == 0
    assert out.group_by("full_position").len()["len"].to_list() == [1] * frame.height
    assert (out["is_template"] != out["is_content"]).all()


def test_winner_and_content_sensitivity_mappings_are_exact():
    from token_analysis.common import CONTENT_WINNERS, METRIC_SPECS

    expected = {
        ("tt", "q7"): "resid_jump",
        ("tt", "g12"): "norm_ratio",
        ("tt", "g27"): "resid_jump_nla",
        ("tt", "l70"): "resid_jump_nla",
        ("opi", "q7"): "peak_ratio",
        ("opi", "g12"): "resid_jump_nla",
        ("opi", "g27"): "resid_jump_nla",
        ("opi", "l70"): "sink_drain",
        ("liars", "g27"): "head_disagreement",
        ("liars", "l70"): "head_disagreement",
        ("taboo", "q7"): "dominant_mass",
        ("taboo", "g12"): "dominant_mass",
        ("taboo", "g27"): "dominant_mass",
        ("taboo", "l70"): "resid_jump_nla",
    }
    assert {pair: spec["metric"] for pair, spec in METRIC_SPECS.items()} == expected
    changed = {pair for pair, metric in expected.items() if CONTENT_WINNERS[pair] != metric}
    assert changed == {("opi", "g27"), ("opi", "l70"), ("tt", "g27"), ("taboo", "l70")}


def test_dataset_shared_metric_mapping_and_alignment_are_exact():
    from token_analysis.common import DATASET_MODELS, SHARED_METRIC_SPECS

    expected = {
        "opi": "peak_ratio",
        "tt": "resid_jump_nla",
        "liars": "head_disagreement",
        "taboo": "dominant_mass",
    }
    assert {
        dataset: {SHARED_METRIC_SPECS[(dataset, model)]["metric"] for model in models}
        for dataset, models in DATASET_MODELS.items()
    } == {dataset: {metric} for dataset, metric in expected.items()}

    frame = source_frame(
        ["user", "template", "assistant"],
        "resid_jump_nla",
        [100.0, 100.0, 100.0],
    ).with_columns(pl.Series("peak_ratio", [1.0, 2.0, 3.0]))
    coverage = {("opi", "case"): 4}
    out = transform_source(
        frame.lazy(),
        "opi",
        "g12",
        coverage,
        metric_regime="dataset-shared",
    ).collect()
    assert out["metric_name"].unique().to_list() == ["peak_ratio"]
    assert out["metric_direction"].unique().to_list() == ["lower"]
    assert out["metric_value_aligned"].to_list() == [-1.0, -2.0, -3.0]
    assert out["metric_percentile_within_case"].to_list() == [1.0, 0.5, 0.0]
    assert out["shared_metric_percentile_within_case"].to_list() == [1.0, 0.5, 0.0]


def test_cv_best_metric_mapping_alignment_and_metadata():
    frame = source_frame(
        ["user", "template", "template"],
        "peak_ratio",
        [100.0, 100.0, 100.0],
    ).with_columns(pl.Series("dominant_mass", [1.0, 2.0, 3.0]))
    coverage = {("opi", "case"): 4}
    cv_specs = {
        ("opi", "q7"): {
            "metric": "dominant_mass",
            "auroc": 0.737,
            "direction": "lower",
            "cv_macro_ap": 0.496,
            "nested_cv_macro_ap": 0.494,
            "statistically_tied": True,
        }
    }
    out = transform_source(
        frame.lazy(),
        "opi",
        "q7",
        coverage,
        metric_regime="cv-best",
        cv_specs=cv_specs,
    ).collect()
    assert out["metric_name"].unique().to_list() == ["dominant_mass"]
    assert out["metric_percentile_within_case"].to_list() == [1.0, 0.5, 0.0]
    assert out["cv_metric_name"].unique().to_list() == ["dominant_mass"]
    assert out["cv_metric_percentile_within_case"].to_list() == [1.0, 0.5, 0.0]
    assert out["cv_metric_cv_macro_ap"].unique().to_list() == [0.496]
    assert out["cv_metric_nested_cv_macro_ap"].unique().to_list() == [0.494]
    assert out["cv_metric_statistically_tied"].all()


def test_cv_shared_metric_mapping_alignment_and_metadata():
    frame = source_frame(
        ["user", "template", "template"],
        "peak_ratio",
        [3.0, 2.0, 1.0],
    )
    coverage = {("opi", "case"): 4}
    specs = {
        ("opi", "q7"): {
            "metric": "peak_ratio",
            "auroc": 0.239,
            "direction": "lower",
            "cv_macro_ap": 0.496,
            "equal_model_cv_macro_ap": 0.300,
            "nested_cv_shared_macro_ap": 0.297,
            "statistically_tied": False,
            "beats_position_baseline": True,
            "beats_region_baseline": False,
        }
    }
    out = transform_source(
        frame.lazy(),
        "opi",
        "q7",
        coverage,
        metric_regime="cv-shared",
        cv_shared_specs=specs,
    ).collect()
    assert out["metric_name"].unique().to_list() == ["peak_ratio"]
    assert out["metric_percentile_within_case"].to_list() == [0.0, 0.5, 1.0]
    assert out["cv_shared_metric_name"].unique().to_list() == ["peak_ratio"]
    assert out["cv_shared_metric_percentile_within_case"].to_list() == [0.0, 0.5, 1.0]
    assert out["cv_shared_metric_model_cv_macro_ap"].unique().to_list() == [0.496]
    assert out["cv_shared_metric_equal_model_cv_macro_ap"].unique().to_list() == [0.300]
    assert out["cv_shared_metric_nested_cv_shared_macro_ap"].unique().to_list() == [0.297]
    assert not out["cv_shared_metric_statistically_tied"].any()
    assert not out["cv_shared_metric_demonstrated_incremental_value"].any()


def test_correlation_metric_columns_preserve_direction_and_metadata():
    frame = source_frame(
        ["user", "template", "template"],
        "peak_ratio",
        [3.0, 2.0, 1.0],
    ).with_columns(pl.Series("dominant_mass", [1.0, 2.0, 3.0]))
    coverage = {("opi", "case"): 4}
    cor_specs = {
        ("opi", "q7"): {
            "metric": "dominant_mass",
            "auroc": 0.25,
            "direction": "lower",
            "spearman_rho": -0.4,
            "abs_spearman_rho": 0.4,
            "statistically_tied": True,
            "winner_changes_content_only": True,
            "direction_stability": 0.95,
        }
    }
    shared_specs = {
        ("opi", "q7"): {
            "metric": "peak_ratio",
            "auroc": 0.75,
            "direction": "higher",
            "spearman_rho": 0.3,
            "abs_spearman_rho": 0.3,
            "equal_model_abs_spearman_rho": 0.2,
            "statistically_tied": False,
            "winner_changes_content_only": False,
            "direction_stability": 1.0,
        }
    }
    out = transform_source(
        frame.lazy(),
        "opi",
        "q7",
        coverage,
        metric_regime="cor-best",
        cor_specs=cor_specs,
        cor_shared_specs=shared_specs,
    ).collect()
    assert out["metric_name"].unique().to_list() == ["dominant_mass"]
    assert out["metric_percentile_within_case"].to_list() == [1.0, 0.5, 0.0]
    assert out["cor_metric_spearman_rho"].unique().to_list() == [-0.4]
    assert out["cor_metric_statistically_tied"].all()
    assert out["cor_metric_winner_changes_content_only"].all()
    assert out["cor_shared_metric_name"].unique().to_list() == ["peak_ratio"]
    assert out["cor_shared_metric_percentile_within_case"].to_list() == [
        1.0,
        0.5,
        0.0,
    ]
    assert out["cor_shared_metric_equal_model_abs_spearman_rho"].unique().to_list() == [0.2]


def test_no_hd_pooled_specs_replace_every_head_disagreement_winner():
    from token_analysis.common import metric_specs

    for regime in ("model-best", "dataset-shared"):
        specs = metric_specs(regime, exclude_head_disagreement=True)
        assert all(spec["metric"] != "head_disagreement" for spec in specs.values())

    best = metric_specs("model-best", exclude_head_disagreement=True)
    assert best[("liars", "g27")]["metric"] == "varentropy"
    assert best[("liars", "l70")]["metric"] == "sink_drain"
    shared = metric_specs("dataset-shared", exclude_head_disagreement=True)
    assert {shared[("liars", model)]["metric"] for model in ("g27", "l70")} == {"sink_drain"}


def test_no_hd_transform_materializes_alternative_pooled_columns():
    frame = source_frame(
        ["user", "template", "assistant"],
        "varentropy",
        [1.0, 2.0, 3.0],
    ).with_columns(pl.Series("sink_drain", [3.0, 2.0, 1.0]))
    coverage = {("liars", "case"): 2}
    out = transform_source(
        frame.lazy(),
        "liars",
        "g27",
        coverage,
        exclude_head_disagreement=True,
    ).collect()

    assert out["metric_name"].unique().to_list() == ["varentropy"]
    assert out["shared_metric_name"].unique().to_list() == ["sink_drain"]
    assert "head_disagreement" not in {out["metric_name"][0], out["shared_metric_name"][0]}


def test_selected_auroc_ensemble_is_reconstructed_from_pooled_component_ranks():
    frame = source_frame(
        ["user", "user", "template", "assistant"],
        "resid_jump",
        [0.0, 1.0, 2.0, 3.0],
    ).with_columns(
        pl.Series("peak_ratio", [0.0, 1.0, 2.0, 3.0]),
        pl.Series("dominant_mass", [3.0, 2.0, 1.0, 0.0]),
    )
    spec = {
        "candidate_id": "mix::dominant_mass@0.75+peak_ratio@0.25",
        "candidate_type": "ensemble",
        "component_1": "dominant_mass",
        "component_1_weight": 0.75,
        "component_1_direction": "lower",
        "component_2": "peak_ratio",
        "component_2_weight": 0.25,
        "component_2_direction": "higher",
        "candidate_direction": "higher",
        "adjusted_auroc": 0.8,
        "winner_changes_common_support": False,
        "gain_over_best_original": 0.05,
        "heldout_pooled_auroc": 0.7,
        "heldout_case_macro_auroc": 0.65,
        "heldout_case_macro_delta": 0.02,
        "heldout_delta_ci_low": -0.01,
        "heldout_delta_ci_high": 0.05,
        "outer_fold_selection_frequency": 0.6,
    }
    key = ("taboo", "q7")
    out = transform_source(
        frame.lazy(),
        "taboo",
        "q7",
        {("taboo", "case"): 4},
        ensemble_specs={key: spec},
        ensemble_shared_specs={key: {**spec, "equal_model_adjusted_auroc": 0.75}},
    ).collect()
    expected = [0.0, 1 / 3, 2 / 3, 1.0]
    assert out["auroc_ensemble_metric_value_raw"].to_list() == pytest.approx(expected)
    assert out["auroc_ensemble_metric_value_aligned"].to_list() == pytest.approx(expected)
    assert out["auroc_ensemble_metric_percentile_within_case"].to_list() == pytest.approx(expected)
    assert out["auroc_ensemble_metric_name"].unique().to_list() == [spec["candidate_id"]]
    assert out["auroc_ensemble_shared_metric_equal_model_auroc"][0] == pytest.approx(0.75)
