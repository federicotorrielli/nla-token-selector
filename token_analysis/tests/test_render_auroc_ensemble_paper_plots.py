from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from token_analysis.common import DATASET_MODELS, MODEL_INFO
from token_analysis.plot_token_positions import load_segment_summary
from token_analysis.render_auroc_ensemble_paper_plots import (
    PAPER_AGGREGATED_POSITION_FIGSIZE,
    PAPER_AGGREGATED_POSITION_HEATMAP_FIGSIZE,
    PAPER_POSITION_FIGSIZE,
    PAPER_POSITION_HEATMAP_FIGSIZE,
    REGIMES,
    _atomic_save_pdf,
    _atomic_save_png,
    make_paper_aggregated_position_figure,
    make_paper_aggregated_position_heatmap,
    make_paper_position_figure,
    make_paper_position_heatmap,
    make_paper_segment_figures,
)


def _figure_text(figure) -> str:
    return " ".join(item.get_text() for item in figure.findobj() if hasattr(item, "get_text"))


def test_position_plot_families_have_fixed_canvas_across_datasets():
    summary = pl.read_parquet(REGIMES["model_best"]["position"])
    builders = (
        (make_paper_position_figure, PAPER_POSITION_FIGSIZE),
        (make_paper_aggregated_position_figure, PAPER_AGGREGATED_POSITION_FIGSIZE),
        (make_paper_position_heatmap, PAPER_POSITION_HEATMAP_FIGSIZE),
        (
            make_paper_aggregated_position_heatmap,
            PAPER_AGGREGATED_POSITION_HEATMAP_FIGSIZE,
        ),
    )
    for builder, expected_size in builders:
        for dataset in DATASET_MODELS:
            figure = builder(summary, dataset)
            try:
                assert tuple(figure.get_size_inches()) == expected_size
            finally:
                plt.close(figure)


def test_paper_position_omits_global_prose_and_per_bin_n():
    summary = pl.read_parquet(REGIMES["model_best"]["position"])
    figure = make_paper_position_figure(summary, "taboo")
    try:
        text = _figure_text(figure)
        assert "Taboo: AUROC-ensemble relevance by token position" not in text
        assert "bands are descriptive pointwise" not in text
        assert "Context strips show" not in text
        assert "n=" not in text
        assert "AUROC" not in text
        assert "lookback_ratio" not in text
        assert "Within-case relevance percentile" in text
        assert all(segment in text for segment in ("Input", "Boundary", "Output"))
        assert all(MODEL_INFO[model]["name"] in text for model in DATASET_MODELS["taboo"])
        ylabel = next(
            item for item in figure.texts if item.get_text() == "Within-case relevance percentile"
        )
        assert ylabel.get_position()[0] == 0.018
        assert figure.subplotpars.left == 0.085
        assert all(
            item.get_fontfamily() == ["STIXGeneral"]
            for item in figure.findobj()
            if hasattr(item, "get_fontfamily")
        )
    finally:
        plt.close(figure)


def test_paper_aggregated_position_overlays_all_models_by_segment():
    summary = pl.read_parquet(REGIMES["model_best"]["position"])
    figure = make_paper_aggregated_position_figure(summary, "taboo")
    try:
        text = _figure_text(figure)
        assert len(figure.axes) == 3
        assert all(segment in text for segment in ("Input", "Boundary", "Output"))
        assert all(MODEL_INFO[model]["name"] in text for model in DATASET_MODELS["taboo"])
        assert "Within-case relevance percentile" in text
        assert "AUROC" not in text
        assert "n=" not in text
        assert "lookback_ratio" not in text
        assert all(len(axis.lines) >= 5 for axis in figure.axes)
        assert all(
            item.get_fontfamily() == ["STIXGeneral"]
            for item in figure.findobj()
            if hasattr(item, "get_fontfamily")
        )
    finally:
        plt.close(figure)


def test_paper_position_heatmap_separates_models_and_segments():
    summary = pl.read_parquet(REGIMES["model_best"]["position"])
    figure = make_paper_position_heatmap(summary, "taboo")
    try:
        text = _figure_text(figure)
        data_axes = figure.axes[:-1]
        assert len(data_axes) == 12  # four model rows by three segment columns
        assert all(segment in text for segment in ("Input", "Boundary", "Output"))
        assert all(MODEL_INFO[model]["name"] in text for model in DATASET_MODELS["taboo"])
        assert "Within-case relevance percentile" in text
        assert "AUROC" not in text
        assert "n=" not in text
        assert "lookback_ratio" not in text
        assert all(len(axis.images) == 1 for axis in data_axes)
        assert all(axis.images[0].get_array().shape[0] == 1 for axis in data_axes)
        assert all(axis.images[0].get_clim() == (0.0, 1.0) for axis in data_axes)
        assert all(
            item.get_fontfamily() == ["STIXGeneral"]
            for item in figure.findobj()
            if hasattr(item, "get_fontfamily")
        )
    finally:
        plt.close(figure)


def test_paper_aggregated_position_heatmap_uses_one_matrix_per_segment():
    summary = pl.read_parquet(REGIMES["model_best"]["position"])
    figure = make_paper_aggregated_position_heatmap(summary, "taboo")
    try:
        data_axes = figure.axes[:-1]
        assert len(data_axes) == 3
        assert all(len(axis.images) == 1 for axis in data_axes)
        assert all(axis.images[0].get_array().shape[0] == 4 for axis in data_axes)
        assert all(axis.images[0].get_clim() == (0.0, 1.0) for axis in data_axes)
    finally:
        plt.close(figure)


def test_paper_heatmap_png_is_high_resolution(tmp_path: Path):
    summary = pl.read_parquet(REGIMES["model_best"]["position"])
    figure = make_paper_position_heatmap(summary, "opi")
    try:
        output = _atomic_save_png(
            figure,
            tmp_path / "heatmap.png",
            overwrite=False,
            fixed_canvas=True,
        )
        image = plt.imread(output)
        assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        expected_width = round(PAPER_POSITION_HEATMAP_FIGSIZE[0] * 600)
        expected_height = round(PAPER_POSITION_HEATMAP_FIGSIZE[1] * 600)
        assert image.shape[1] == expected_width
        assert image.shape[0] == expected_height
    finally:
        plt.close(figure)


def test_paper_segments_overlay_models_without_removed_annotations(tmp_path: Path):
    summary = load_segment_summary(REGIMES["model_best"]["segment"], {"taboo"})
    figures = make_paper_segment_figures(summary, "taboo")
    assert set(figures) == {"auroc", "percentile", "enrichment"}
    try:
        for figure in figures.values():
            text = _figure_text(figure)
            assert all(MODEL_INFO[model]["name"] in text for model in DATASET_MODELS["taboo"])
            assert "on-task" in text
            assert "tmpl" not in text
            assert " tok " not in text
            assert " cases" not in text
            assert "eligible n=" not in text
            assert "Taboo:" not in text
            assert len(figure.axes) == 1
            assert all(
                item.get_fontfamily() == ["STIXGeneral"]
                for item in figure.findobj()
                if hasattr(item, "get_fontfamily")
            )
        enrichment_text = _figure_text(figures["enrichment"])
        assert "Top 1%" in enrichment_text
        assert "Top 10%" in enrichment_text
        output = _atomic_save_pdf(
            figures["percentile"], tmp_path / "percentile.pdf", overwrite=False
        )
        assert output.read_bytes().startswith(b"%PDF")
    finally:
        for figure in figures.values():
            plt.close(figure)
