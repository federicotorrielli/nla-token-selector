from pathlib import Path

import numpy as np
import polars as pl
import pytest

from token_analysis.common import CONTENT_WINNERS, DATASET_MODELS, METRIC_SPECS, MODEL_INFO
from token_analysis.plot_token_positions import (
    aggregate_case_bins,
    attach_shared_validation_metadata,
    case_bin_summaries,
    fractional_ranks,
    load_segment_summary,
    make_dataset_figure,
    make_segment_comparison_figure,
    save_dataset_figure,
    save_segment_comparison_figure,
    _scan_input,
)


def token_frame(rows):
    defaults = {
        "dataset": "taboo",
        "model": "q7",
        "model_name": MODEL_INFO["q7"]["name"],
        "case_id": "a",
        "segment": "input",
        "segment_position": 0,
        "segment_position_normalized": 0.0,
        "metric_name": "dominant_mass",
        "metric_auroc": 0.796,
        "metric_direction": "higher",
        "metric_percentile_within_case": 0.5,
        "source_region": "user",
        "analysis_region": "user",
        "response_token_origin": "not_response",
        "is_template": False,
        "is_reconstructed_tail": False,
        "is_extreme_norm": False,
        "on_task": 0,
        "metric_winner_changes_content_only": False,
    }
    merged = [{**defaults, **row} for row in rows]
    for item, row in zip(merged, rows, strict=True):
        if "analysis_region" not in row:
            item["analysis_region"] = item["source_region"]
    return pl.DataFrame(merged)


def test_fractional_ranks_handle_direction_ties_and_missing():
    values = np.array([1.0, 2.0, 2.0, 4.0, np.nan])
    high = fractional_ranks(values)
    low = fractional_ranks(values, higher_is_relevant=False)
    assert high[:4].tolist() == pytest.approx([0.0, 0.5, 0.5, 1.0])
    assert low[:4].tolist() == pytest.approx([1.0, 0.5, 0.5, 0.0])
    assert np.isnan(high[-1]) and np.isnan(low[-1])


def test_normalized_final_position_and_ordinal_bins():
    frame = token_frame(
        [
            {"segment": "input", "segment_position_normalized": 0.0},
            {"segment": "input", "segment_position": 9, "segment_position_normalized": 1.0},
            {"segment": "boundary", "segment_position": 0},
            {"segment": "boundary", "segment_position": 4},
            {"segment": "trailer", "segment_position": 0},
        ]
    )
    cases = case_bin_summaries(frame.lazy(), normalized_bins=20)
    assert "trailer" not in cases["segment"].to_list()
    assert (
        cases.filter(pl.col("segment") == "input")
        .sort("bin_index")["bin_index"]
        .to_list()
        == [0, 19]
    )
    boundary = cases.filter(pl.col("segment") == "boundary").sort("bin_index")
    assert boundary["axis_kind"].unique().to_list() == ["ordinal"]
    assert boundary["bin_index"].to_list() == [0, 4]


def test_case_first_mean_prevents_long_case_dominance():
    rows = [
        {
            "case_id": "short",
            "segment_position_normalized": 0.1,
            "metric_percentile_within_case": 0.0,
        }
    ]
    rows.extend(
        {
            "case_id": "long",
            "segment_position": index,
            "segment_position_normalized": 0.1,
            "metric_percentile_within_case": 1.0,
        }
        for index in range(10)
    )
    cases = case_bin_summaries(token_frame(rows).lazy(), normalized_bins=2)
    summary = aggregate_case_bins(
        cases, 2, bootstrap_resamples=50, seed=3
    )
    point = summary.filter(
        (pl.col("segment") == "input") & (pl.col("bin_index") == 0)
    ).row(0, named=True)
    assert point["mean_percentile"] == pytest.approx(0.5)
    assert point["median_percentile"] == pytest.approx(0.5)
    assert point["n_cases"] == 2
    assert point["n_tokens"] == 11
    assert point["mean_percentile"] != pytest.approx(10 / 11)
    assert 0 <= point["ci_low"] <= point["ci_high"] <= 1


def test_missing_scores_do_not_change_case_or_token_denominators():
    frame = token_frame(
        [
            {"case_id": "a", "metric_percentile_within_case": None},
            {"case_id": "a", "segment_position": 1, "metric_percentile_within_case": 0.8},
            {"case_id": "b", "metric_percentile_within_case": 0.2},
        ]
    )
    cases = case_bin_summaries(frame.lazy(), normalized_bins=2)
    summary = aggregate_case_bins(cases, 2, bootstrap_resamples=20, seed=4)
    point = summary.filter(pl.col("bin_index") == 0).row(0, named=True)
    assert point["n_tokens"] == 3
    assert point["n_metric_tokens"] == 2
    assert point["n_cases"] == 2
    assert point["mean_percentile"] == pytest.approx(0.5)


def test_bootstrap_is_deterministic_and_context_is_case_balanced():
    frame = token_frame(
        [
            {"case_id": "a", "source_region": "system", "metric_percentile_within_case": 0.1},
            {
                "case_id": "a",
                "segment_position": 1,
                "source_region": "user",
                "metric_percentile_within_case": 0.3,
            },
            {
                "case_id": "b",
                "source_region": "template",
                "analysis_region": "user",
                "is_template": False,
                "metric_percentile_within_case": 0.9,
            },
        ]
    )
    cases = case_bin_summaries(frame.lazy(), normalized_bins=2)
    first = aggregate_case_bins(cases, 2, bootstrap_resamples=100, seed=8)
    second = aggregate_case_bins(cases, 2, bootstrap_resamples=100, seed=8)
    assert first["ci_low"].to_list() == second["ci_low"].to_list()
    point = first.filter(pl.col("bin_index") == 0).row(0, named=True)
    assert point["system_fraction"] == pytest.approx(0.25)
    assert point["user_fraction"] == pytest.approx(0.25)
    assert point["template_fraction"] == pytest.approx(0.5)
    assert point["analysis_user_fraction"] == pytest.approx(0.75)
    assert point["analysis_template_fraction"] == pytest.approx(0.0)
    assert point["case_coverage"] == pytest.approx(1.0)


def taboo_plot_summary():
    rows = []
    for model in DATASET_MODELS["taboo"]:
        spec = METRIC_SPECS[("taboo", model)]
        for case_index in range(3):
            for segment, region in (
                ("input", "user"),
                ("boundary", "template"),
                ("output", "assistant"),
            ):
                for position in range(2):
                    rows.append(
                        {
                            "dataset": "taboo",
                            "model": model,
                            "model_name": MODEL_INFO[model]["name"],
                            "case_id": f"{model}-{case_index}",
                            "segment": segment,
                            "segment_position": position,
                            "segment_position_normalized": float(position),
                            "metric_name": spec["metric"],
                            "metric_auroc": spec["auroc"],
                            "metric_direction": spec["direction"],
                            "metric_percentile_within_case": 0.25 + 0.5 * position,
                            "source_region": region,
                            "analysis_region": region,
                            "response_token_origin": (
                                "stored_prefix"
                                if segment == "output" and position == 0
                                else "reconstructed_tail"
                                if segment == "output"
                                else "not_response"
                            ),
                            "is_template": segment == "boundary",
                            "is_reconstructed_tail": segment == "output" and position == 1,
                            "is_extreme_norm": False,
                            "on_task": position,
                            "metric_winner_changes_content_only": (
                                CONTENT_WINNERS[("taboo", model)] != spec["metric"]
                            ),
                        }
                    )
    cases = case_bin_summaries(pl.DataFrame(rows).lazy(), normalized_bins=20)
    return aggregate_case_bins(cases, 20, bootstrap_resamples=20, seed=0)


def test_dynamic_figure_has_all_taboo_models_segments_and_annotation():
    summary = taboo_plot_summary()
    assert (
        summary.filter(pl.col("segment") == "output")["response_fraction"] == 1
    ).all()
    figure = make_dataset_figure(summary, "taboo", normalized_bins=20)
    try:
        text = " ".join(item.get_text() for item in figure.findobj() if hasattr(item, "get_text"))
        assert all(MODEL_INFO[model]["name"] in text for model in DATASET_MODELS["taboo"])
        assert all(segment in text for segment in ("Input", "Boundary", "Output"))
        assert "content-only winner" in text
        assert "Within-case relevance percentile" in text
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_liars_has_no_trailer_panel_and_footer_does_not_overlap():
    summary = taboo_plot_summary().filter(pl.col("model").is_in(["g27", "l70"]))
    summary = summary.with_columns(
        pl.lit("liars").alias("dataset"),
        pl.lit("head_disagreement").alias("metric_name"),
        pl.col("model").replace_strict({"g27": 0.280, "l70": 0.318}).alias("metric_auroc"),
        pl.lit("lower").alias("metric_direction"),
        pl.lit(False).alias("metric_winner_changes_content_only"),
    )
    figure = make_dataset_figure(summary, "liars", normalized_bins=20)
    try:
        figure.canvas.draw()
        text_content = " ".join(
            item.get_text() for item in figure.findobj() if hasattr(item, "get_text")
        )
        assert "Trailer" not in text_content
        renderer = figure.canvas.get_renderer()
        legend_box = figure.legends[0].get_window_extent(renderer)
        footer = next(
            text
            for text in figure.texts
            if text.get_text().startswith("Context strips")
        ).get_window_extent(renderer)
        xlabels = [
            axis.xaxis.label.get_window_extent(renderer)
            for axis in figure.axes
            if axis.get_xlabel()
        ]
        assert not legend_box.overlaps(footer)
        assert all(not legend_box.overlaps(label) for label in xlabels)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_png_and_editable_svg_smoke_outputs(tmp_path: Path):
    summary = taboo_plot_summary()
    written = save_dataset_figure(
        summary,
        "taboo",
        output_dir=tmp_path,
        normalized_bins=20,
        formats=("png", "svg"),
        dpi=80,
        overwrite=False,
    )
    assert {path.suffix for path in written} == {".png", ".svg"}
    assert all(path.stat().st_size > 1_000 for path in written)
    svg = next(path for path in written if path.suffix == ".svg").read_text()
    assert "Taboo: selected-metric relevance by token position" in svg
    assert "Gemma-3-27B" in svg
    assert ">response</text>" in svg
    assert "stored response" not in svg
    assert "reconstructed tail" not in svg
    assert "dataset response" not in svg
    assert "<text" in svg


def test_dataset_shared_regime_uses_auxiliary_scores_and_names_outputs(tmp_path: Path):
    frame = token_frame(
        [
            {"case_id": "a", "segment_position_normalized": 0.0},
            {"case_id": "a", "segment_position": 1, "segment_position_normalized": 1.0},
        ]
    ).with_columns(
        pl.lit("peak_ratio").alias("shared_metric_name"),
        pl.lit(0.239).alias("shared_metric_auroc"),
        pl.lit("lower").alias("shared_metric_direction"),
        pl.Series("shared_metric_value_raw", [3.0, 1.0]),
        pl.lit(True).alias("shared_metric_available"),
        pl.Series("shared_metric_value_aligned", [-3.0, -1.0]),
        pl.Series("shared_metric_z_within_case", [-0.707, 0.707]),
        pl.Series("shared_metric_percentile_within_case", [0.0, 1.0]),
        pl.lit(False).alias("shared_metric_winner_changes_content_only"),
    )
    path = tmp_path / "positions.parquet"
    frame.write_parquet(path)
    selected = _scan_input(path, {"taboo"}, "dataset-shared").collect()
    assert selected["metric_regime"].unique().to_list() == ["dataset-shared"]
    assert selected["metric_name"].unique().to_list() == ["peak_ratio"]
    assert selected["metric_direction"].unique().to_list() == ["lower"]
    assert selected["metric_percentile_within_case"].to_list() == [0.0, 1.0]

    cases = case_bin_summaries(selected.lazy(), normalized_bins=2)
    summary = aggregate_case_bins(cases, 2, bootstrap_resamples=5, seed=0)
    written = save_dataset_figure(
        summary,
        "taboo",
        output_dir=tmp_path / "figures",
        normalized_bins=2,
        formats=("svg",),
        dpi=80,
        overwrite=False,
    )
    assert written[0].name == "taboo_shared_metric_by_position.svg"
    svg = written[0].read_text()
    assert "dataset-shared metric relevance by token position" in svg
    assert "peak_ratio" in svg



def test_cv_best_regime_uses_auxiliary_scores_and_names_outputs(tmp_path: Path):
    frame = token_frame(
        [
            {"case_id": "a", "segment_position_normalized": 0.0},
            {"case_id": "a", "segment_position": 1, "segment_position_normalized": 1.0},
        ]
    ).with_columns(
        pl.lit("temporal_kl").alias("cv_metric_name"),
        pl.lit(0.385).alias("cv_metric_auroc"),
        pl.lit("lower").alias("cv_metric_direction"),
        pl.Series("cv_metric_value_raw", [3.0, 1.0]),
        pl.lit(True).alias("cv_metric_available"),
        pl.Series("cv_metric_value_aligned", [-3.0, -1.0]),
        pl.Series("cv_metric_z_within_case", [-0.707, 0.707]),
        pl.Series("cv_metric_percentile_within_case", [0.0, 1.0]),
        pl.lit(False).alias("cv_metric_winner_changes_content_only"),
        pl.lit(0.105).alias("cv_metric_cv_macro_ap"),
        pl.lit(0.105).alias("cv_metric_selection_score"),
        pl.lit(0.104).alias("cv_metric_nested_cv_macro_ap"),
        pl.lit(True).alias("cv_metric_statistically_tied"),
        pl.lit(False).alias("cv_metric_demonstrated_incremental_value"),
    )
    path = tmp_path / "positions.parquet"
    frame.write_parquet(path)
    selected = _scan_input(path, {"taboo"}, "cv-best").collect()
    assert selected["metric_regime"].unique().to_list() == ["cv-best"]
    assert selected["metric_name"].unique().to_list() == ["temporal_kl"]
    assert selected["metric_selection_score"].unique().to_list() == [0.105]

    cases = case_bin_summaries(selected.lazy(), normalized_bins=2)
    summary = aggregate_case_bins(cases, 2, bootstrap_resamples=5, seed=0)
    written = save_dataset_figure(
        summary,
        "taboo",
        output_dir=tmp_path / "figures",
        normalized_bins=2,
        formats=("svg",),
        dpi=80,
        overwrite=False,
    )
    assert written[0].name == "taboo_cv_best_metric_by_position.svg"
    svg = written[0].read_text()
    assert "cross-validated best-metric relevance by token position" in svg
    assert "CV MAP 0.105 (tied)" in svg



def test_cv_shared_regime_uses_auxiliary_scores_and_names_outputs(tmp_path: Path):
    frame = token_frame(
        [
            {"case_id": "a", "segment_position_normalized": 0.0},
            {"case_id": "a", "segment_position": 1, "segment_position_normalized": 1.0},
        ]
    ).with_columns(
        pl.lit("dominant_mass").alias("cv_shared_metric_name"),
        pl.lit(0.796).alias("cv_shared_metric_auroc"),
        pl.lit("higher").alias("cv_shared_metric_direction"),
        pl.Series("cv_shared_metric_value_raw", [1.0, 3.0]),
        pl.lit(True).alias("cv_shared_metric_available"),
        pl.Series("cv_shared_metric_value_aligned", [1.0, 3.0]),
        pl.Series("cv_shared_metric_z_within_case", [-0.707, 0.707]),
        pl.Series("cv_shared_metric_percentile_within_case", [0.0, 1.0]),
        pl.lit(False).alias("cv_shared_metric_winner_changes_content_only"),
        pl.lit(0.431).alias("cv_shared_metric_selection_score"),
        pl.lit(0.497).alias("cv_shared_metric_model_cv_macro_ap"),
        pl.lit(0.429).alias("cv_shared_metric_nested_cv_shared_macro_ap"),
        pl.lit(True).alias("cv_shared_metric_statistically_tied"),
        pl.lit(False).alias("cv_shared_metric_demonstrated_incremental_value"),
    )
    path = tmp_path / "positions.parquet"
    frame.write_parquet(path)
    selected = _scan_input(path, {"taboo"}, "cv-shared").collect()
    assert selected["metric_regime"].unique().to_list() == ["cv-shared"]
    assert selected["metric_name"].unique().to_list() == ["dominant_mass"]
    assert selected["metric_selection_score"].unique().to_list() == [0.431]
    assert selected["metric_model_score"].unique().to_list() == [0.497]

    cases = case_bin_summaries(selected.lazy(), normalized_bins=2)
    summary = aggregate_case_bins(cases, 2, bootstrap_resamples=5, seed=0)
    written = save_dataset_figure(
        summary,
        "taboo",
        output_dir=tmp_path / "figures",
        normalized_bins=2,
        formats=("svg",),
        dpi=80,
        overwrite=False,
    )
    assert written[0].name == "taboo_cv_shared_metric_by_position.svg"
    svg = written[0].read_text()
    assert "cross-validated shared-metric relevance by token position" in svg
    assert "shared CV MAP 0.431 (tied)" in svg
    assert "model CV MAP 0.497" in svg



def _cor_plot_frame(shared: bool = False):
    prefix = "cor_shared_metric" if shared else "cor_metric"
    frame = token_frame(
        [
            {"case_id": "a", "segment_position_normalized": 0.0},
            {"case_id": "a", "segment_position": 1, "segment_position_normalized": 1.0},
        ]
    )
    columns = [
        pl.lit("dominant_mass").alias(f"{prefix}_name"),
        pl.lit(0.796).alias(f"{prefix}_auroc"),
        pl.lit("higher").alias(f"{prefix}_direction"),
        pl.Series(f"{prefix}_value_raw", [1.0, 3.0]),
        pl.lit(True).alias(f"{prefix}_available"),
        pl.Series(f"{prefix}_value_aligned", [1.0, 3.0]),
        pl.Series(f"{prefix}_z_within_case", [-0.707, 0.707]),
        pl.Series(f"{prefix}_percentile_within_case", [0.0, 1.0]),
        pl.lit(False).alias(f"{prefix}_winner_changes_content_only"),
        pl.lit(0.418).alias(f"{prefix}_spearman_rho"),
        pl.lit(0.418).alias(f"{prefix}_abs_spearman_rho"),
        pl.lit(True).alias(f"{prefix}_statistically_tied"),
    ]
    if shared:
        columns.append(
            pl.lit(0.299).alias(
                "cor_shared_metric_equal_model_abs_spearman_rho"
            )
        )
    return frame.with_columns(*columns)


@pytest.mark.parametrize(
    ("regime", "shared", "filename", "title", "score_text"),
    [
        (
            "cor-best", False,
            "taboo_cor_best_metric_by_position.svg",
            "Spearman-selected metric relevance by token position",
            "Spearman ρ +0.418 (tied)",
        ),
        (
            "cor-shared", True,
            "taboo_cor_shared_metric_by_position.svg",
            "Spearman-shared metric relevance by token position",
            "shared mean |ρ| 0.299 (tied)",
        ),
    ],
)
def test_correlation_plot_regimes_use_scores_and_names_outputs(
    tmp_path: Path, regime, shared, filename, title, score_text
):
    frame = _cor_plot_frame(shared)
    path = tmp_path / f"{regime}.parquet"
    frame.write_parquet(path)
    selected = _scan_input(path, {"taboo"}, regime).collect()
    assert selected["metric_regime"].unique().to_list() == [regime]
    assert selected["metric_spearman_rho"].unique().to_list() == [0.418]
    cases = case_bin_summaries(selected.lazy(), normalized_bins=2)
    summary = aggregate_case_bins(cases, 2, bootstrap_resamples=5, seed=0)
    written = save_dataset_figure(
        summary, "taboo", output_dir=tmp_path / f"figures-{regime}",
        normalized_bins=2, formats=("svg",), dpi=80, overwrite=False,
    )
    assert written[0].name == filename
    svg = written[0].read_text()
    assert title in svg
    assert score_text in svg


def test_shared_ensemble_plots_show_only_model_full_and_heldout_aurocs(tmp_path: Path):
    manifest_path = tmp_path / "shared-selection.parquet"
    pl.DataFrame(
        [
            {
                "dataset": "taboo",
                "model": "q7",
                "adjusted_auroc": 0.756,
                "equal_model_adjusted_auroc": 0.750,
                "shared_heldout_model_pooled_auroc": 0.711,
                "shared_heldout_model_case_macro_auroc": 0.722,
                "shared_heldout_equal_model_pooled_auroc": 0.733,
                "shared_heldout_equal_model_case_macro_auroc": 0.744,
            }
        ]
    ).write_parquet(manifest_path)

    positional = (
        taboo_plot_summary()
        .filter(pl.col("model") == "q7")
        .with_columns(
            pl.lit("auroc-ensemble-shared").alias("metric_regime"),
            pl.lit("mix::dominant_mass@0.50+norm_ratio@0.50").alias(
                "metric_name"
            ),
            pl.lit(0.750).alias("metric_selection_score"),
            pl.lit(0.756).alias("metric_model_score"),
            pl.lit(0.756).alias("metric_auroc"),
        )
    )
    positional = attach_shared_validation_metadata(positional, manifest_path)
    positional_svg = save_dataset_figure(
        positional,
        "taboo",
        output_dir=tmp_path / "positions",
        normalized_bins=20,
        formats=("svg",),
        dpi=80,
        overwrite=False,
    )[0].read_text()
    assert "AUROC 0.756" in positional_svg
    assert "held-out AUROC 0.711" in positional_svg
    assert "shared AUROC 0.750" not in positional_svg
    assert "held-out pooled" not in positional_svg
    assert "held-out macro" not in positional_svg
    assert "equal-model pooled AUROC" not in positional_svg
    assert "case-macro AUROC 0.744" not in positional_svg
    assert "not the frozen full-data winner drawn here" in positional_svg

    segment_rows = []
    for index, segment in enumerate(("input", "boundary", "output")):
        segment_rows.append(
            {
                "dataset": "taboo",
                "model": "q7",
                "model_name": MODEL_INFO["q7"]["name"],
                "metric_regime": "auroc-ensemble-shared",
                "candidate_id": "mix::dominant_mass@0.50+norm_ratio@0.50",
                "segment": segment,
                "token_count": 100 - 20 * index,
                "case_count": 10,
                "auc_case_count": 8,
                "token_base_rate": 0.2 + 0.1 * index,
                "case_macro_auroc": 0.62,
                "case_macro_auroc_ci_low": 0.55,
                "case_macro_auroc_ci_high": 0.70,
                "mean_relevance_percentile": 0.55,
                "mean_relevance_percentile_ci_low": 0.50,
                "mean_relevance_percentile_ci_high": 0.60,
                "template_fraction": 1.0 if segment == "boundary" else 0.1,
                "top_1_enrichment": 1.4,
                "top_1_enrichment_ci_low": 1.1,
                "top_1_enrichment_ci_high": 1.8,
                "top_10_enrichment": 1.2,
                "top_10_enrichment_ci_low": 1.0,
                "top_10_enrichment_ci_high": 1.4,
            }
        )
    segments = attach_shared_validation_metadata(
        pl.DataFrame(segment_rows),
        manifest_path,
    )
    segment_svg = save_segment_comparison_figure(
        segments,
        "taboo",
        output_dir=tmp_path / "segments",
        formats=("svg",),
        dpi=80,
        overwrite=False,
    )[0].read_text()
    assert "AUROC 0.756" in segment_svg
    assert "held-out AUROC 0.711" in segment_svg
    assert "full AUROC" not in segment_svg
    assert "held-out pooled" not in segment_svg
    assert "held-out macro" not in segment_svg
    assert "equal-model pooled AUROC" not in segment_svg
    assert "case-macro AUROC 0.744" not in segment_svg


def test_ensemble_segment_plot_rejects_trailers_and_writes_both_formats(tmp_path: Path):
    rows = []
    for index, segment in enumerate(("input", "boundary", "output")):
        rows.append(
            {
                "dataset": "taboo",
                "model": "q7",
                "model_name": MODEL_INFO["q7"]["name"],
                "metric_regime": "auroc-ensemble-best",
                "candidate_id": "mix::entropy@0.25+sink_drain@0.75",
                "best_full_model_auroc": 0.756,
                "best_heldout_model_pooled_auroc": 0.711,
                "segment": segment,
                "token_count": 100 - index * 20,
                "finite_token_count": 99 - index * 20,
                "case_count": 10,
                "auc_case_count": 8,
                "mean_segment_length": 10.0,
                "median_segment_length": 9.0,
                "token_base_rate": 0.2 + 0.1 * index,
                "case_balanced_base_rate": 0.2,
                "pooled_auroc": 0.6,
                "case_macro_auroc": 0.62,
                "case_macro_auroc_ci_low": 0.55,
                "case_macro_auroc_ci_high": 0.70,
                "mean_relevance_percentile": 0.55,
                "mean_relevance_percentile_ci_low": 0.50,
                "mean_relevance_percentile_ci_high": 0.60,
                "template_fraction": 1.0 if segment == "boundary" else 0.1,
                "content_fraction": 0.0 if segment == "boundary" else 0.9,
                "source_system_fraction": 0.0,
                "source_user_fraction": 0.5,
                "source_assistant_prior_fraction": 0.0,
                "source_assistant_fraction": 0.4,
                "source_template_fraction": 0.1,
                "response_fraction": 1.0 if segment == "output" else 0.0,
                "reconstructed_tail_fraction": 0.2 if segment == "output" else 0.0,
                "top_1_share": 0.2,
                "top_1_share_ci_low": 0.1,
                "top_1_share_ci_high": 0.3,
                "top_1_enrichment": 1.4,
                "top_1_enrichment_ci_low": 1.1,
                "top_1_enrichment_ci_high": 1.8,
                "top_10_share": 0.3,
                "top_10_share_ci_low": 0.2,
                "top_10_share_ci_high": 0.4,
                "top_10_enrichment": 1.2,
                "top_10_enrichment_ci_low": 1.0,
                "top_10_enrichment_ci_high": 1.4,
            }
        )
    frame = pl.DataFrame(rows)
    source = tmp_path / "segments.parquet"
    frame.write_parquet(source)
    loaded = load_segment_summary(source, {"taboo"})
    assert loaded.height == 3
    figure = make_segment_comparison_figure(loaded, "taboo")
    assert len(figure.axes) == 3
    import matplotlib.pyplot as plt
    plt.close(figure)
    written = save_segment_comparison_figure(
        loaded,
        "taboo",
        output_dir=tmp_path / "figures",
        formats=("png", "svg"),
        dpi=80,
        overwrite=False,
    )
    assert {path.suffix for path in written} == {".png", ".svg"}
    assert all(path.stat().st_size > 0 for path in written)
    svg = next(path for path in written if path.suffix == ".svg").read_text()
    assert "Within-segment case-macro AUROC" in svg
    assert "Qwen2.5-7B" in svg
    assert "AUROC 0.756" in svg
    assert "held-out AUROC 0.711" in svg
    trailer = frame.vstack(frame.head(1).with_columns(pl.lit("trailer").alias("segment")))
    trailer.write_parquet(source)
    with pytest.raises(ValueError, match="trailer"):
        load_segment_summary(source, {"taboo"})
