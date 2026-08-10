"""Render compact figures for the AUROC-ensemble paper analysis.

This renderer consumes the already-computed position and segment summary
Parquets. It never recomputes statistics or modifies the standard PNG/SVG
figures. Paper position figures omit global title/subtitle/footer text and
per-bin sample sizes. Paper segment figures put all models in one color-coded
axis per statistic and retain the whole-case bootstrap intervals.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import polars as pl

try:
    from .common import (
        AUROC_ENSEMBLE_BEST_RESULTS_DIR,
        AUROC_ENSEMBLE_DIR,
        AUROC_ENSEMBLE_SHARED_RESULTS_DIR,
        DATASET_MODELS,
        MODEL_INFO,
    )
    from .plot_token_positions import (
        MODEL_COLORS,
        SEGMENT_BACKGROUNDS,
        SEGMENT_ORDER,
        _finite_error,
        _load_matplotlib,
        load_segment_summary,
        make_dataset_figure,
    )
except ImportError:
    from common import (  # type: ignore[no-redef]
        AUROC_ENSEMBLE_BEST_RESULTS_DIR,
        AUROC_ENSEMBLE_DIR,
        AUROC_ENSEMBLE_SHARED_RESULTS_DIR,
        DATASET_MODELS,
        MODEL_INFO,
    )
    from plot_token_positions import (  # type: ignore[no-redef]
        MODEL_COLORS,
        SEGMENT_BACKGROUNDS,
        SEGMENT_ORDER,
        _finite_error,
        _load_matplotlib,
        load_segment_summary,
        make_dataset_figure,
    )

PAPER_DIR = AUROC_ENSEMBLE_DIR / "plots" / "paper"
PAPER_FONT = "STIXGeneral"
PAPER_POSITION_FIGSIZE = (10.65, 12.0)
PAPER_AGGREGATED_POSITION_FIGSIZE = (9.6, 3.35)
PAPER_POSITION_HEATMAP_FIGSIZE = (9.6, 4.87)
PAPER_AGGREGATED_POSITION_HEATMAP_FIGSIZE = (9.6, 4.17)
REGIMES = {
    "model_best": {
        "position": AUROC_ENSEMBLE_BEST_RESULTS_DIR
        / "auroc_ensemble_best_metric_position_summary.parquet",
        "segment": AUROC_ENSEMBLE_BEST_RESULTS_DIR / "auroc_ensemble_best_segment_summary.parquet",
        "position_stem": "auroc_ensemble_best_metric_by_position",
        "aggregated_position_stem": "auroc_ensemble_best_metric_by_position_aggregated",
        "position_heatmap_stem": "auroc_ensemble_best_metric_by_position_heatmap",
        "aggregated_position_heatmap_stem": (
            "auroc_ensemble_best_metric_by_position_heatmap_aggregated"
        ),
        "segment_stem": "auroc_ensemble_best_segment",
    },
    "dataset_shared": {
        "position": AUROC_ENSEMBLE_SHARED_RESULTS_DIR
        / "auroc_ensemble_shared_metric_position_summary.parquet",
        "segment": AUROC_ENSEMBLE_SHARED_RESULTS_DIR
        / "auroc_ensemble_shared_segment_summary.parquet",
        "position_stem": "auroc_ensemble_shared_metric_by_position",
        "aggregated_position_stem": "auroc_ensemble_shared_metric_by_position_aggregated",
        "position_heatmap_stem": "auroc_ensemble_shared_metric_by_position_heatmap",
        "aggregated_position_heatmap_stem": (
            "auroc_ensemble_shared_metric_by_position_heatmap_aggregated"
        ),
        "segment_stem": "auroc_ensemble_shared_segment",
    },
}
PAPER_POSITION_COLUMNS = {
    "dataset",
    "model",
    "segment",
    "metric_regime",
    "mean_percentile",
    "ci_low",
    "ci_high",
    "n_cases",
    "axis_kind",
    "bin_index",
    "position_x",
    "position_ordinal",
}
SEGMENT_FIGURES = {
    "auroc": {
        "value": "case_macro_auroc",
        "low": "case_macro_auroc_ci_low",
        "high": "case_macro_auroc_ci_high",
        "ylabel": "Case-macro AUROC",
        "reference": 0.5,
        "ylim": (0.0, 1.0),
    },
    "percentile": {
        "value": "mean_relevance_percentile",
        "low": "mean_relevance_percentile_ci_low",
        "high": "mean_relevance_percentile_ci_high",
        "ylabel": "Mean relevance percentile",
        "reference": 0.5,
        "ylim": (0.0, 1.0),
    },
}


def _atomic_save_pdf(
    figure,
    path: Path,
    *,
    overwrite: bool,
    fixed_canvas: bool = False,
) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    crop_options = {} if fixed_canvas else {"bbox_inches": "tight", "pad_inches": 0.06}
    figure.savefig(
        temporary,
        format="pdf",
        dpi=300,
        facecolor=figure.get_facecolor(),
        **crop_options,
    )
    os.replace(temporary, path)
    return path


def _atomic_save_png(
    figure,
    path: Path,
    *,
    overwrite: bool,
    fixed_canvas: bool = False,
) -> Path:
    """Save a high-resolution raster copy without changing figure geometry."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    crop_options = {} if fixed_canvas else {"bbox_inches": "tight", "pad_inches": 0.06}
    figure.savefig(
        temporary,
        format="png",
        dpi=600,
        facecolor=figure.get_facecolor(),
        **crop_options,
    )
    os.replace(temporary, path)
    return path


def _load_position_summary(path: Path, datasets: set[str]) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    summary = pl.read_parquet(path)
    missing = PAPER_POSITION_COLUMNS - set(summary.columns)
    if missing:
        raise ValueError(f"{path} is missing position columns: {', '.join(sorted(missing))}")
    summary = summary.filter(pl.col("dataset").is_in(sorted(datasets)))
    if summary.is_empty():
        raise ValueError(f"{path} has no rows for the requested datasets")
    return summary


def _apply_paper_typography(figure) -> None:
    """Apply an embedded LaTeX-style serif consistently to every text artist."""
    import matplotlib

    matplotlib.rcParams.update(
        {
            "font.family": PAPER_FONT,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
        }
    )
    for item in figure.findobj():
        if hasattr(item, "set_fontfamily"):
            item.set_fontfamily(PAPER_FONT)


def _remove_paper_position_annotations(figure, dataset: str) -> None:
    """Remove global prose, left row metadata, and per-bin n labels."""
    keep_figure_text = {"Within-case relevance percentile"}
    for item in list(figure.texts):
        if item.get_text() not in keep_figure_text:
            item.remove()
        else:
            item.set_x(0.018)
    for axis in figure.axes:
        for item in list(axis.texts):
            item.remove()

    _apply_paper_typography(figure)
    figure.set_size_inches(*PAPER_POSITION_FIGSIZE, forward=True)
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.18 if dataset == "liars" else 0.13,
        top=0.96,
        hspace=0.78 if dataset == "liars" else 0.72,
        wspace=0.18,
    )
    if figure.legends:
        figure.legends[0].set_bbox_to_anchor(
            (0.5, 0.045 if dataset == "liars" else 0.025),
            transform=figure.transFigure,
        )


def make_paper_position_figure(
    summary: pl.DataFrame,
    dataset: str,
    *,
    normalized_bins: int = 20,
):
    """Return the standard position profile with paper-only annotations removed."""
    figure = make_dataset_figure(summary, dataset, normalized_bins=normalized_bins)
    _remove_paper_position_annotations(figure, dataset)
    return figure


def make_paper_aggregated_position_figure(summary: pl.DataFrame, dataset: str):
    """Overlay all model position curves in one facet per transcript segment."""
    plt, _, Line2D, _ = _load_matplotlib()
    data = summary.filter(pl.col("dataset") == dataset)
    if data.is_empty():
        raise ValueError(f"no position rows for {dataset}")
    models = [model for model in DATASET_MODELS[dataset] if model in set(data["model"])]
    segments = [segment for segment in SEGMENT_ORDER if segment in set(data["segment"])]
    figure, axes = plt.subplots(
        1,
        len(segments),
        figsize=PAPER_AGGREGATED_POSITION_FIGSIZE,
        facecolor="#F6F8FC",
        sharey=True,
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.24,
        top=0.91,
        wspace=0.17,
    )

    for column_index, segment in enumerate(segments):
        axis = axes[0, column_index]
        axis.set_facecolor(SEGMENT_BACKGROUNDS[segment])
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.grid(axis="y", color="#DDE3EE", linewidth=0.7, alpha=0.75)
        axis.set_axisbelow(True)
        axis.axhline(0.5, color="#697386", linewidth=0.9, linestyle=(0, (3, 3)))
        axis.set_ylim(0, 1)
        axis.set_yticks([0, 0.25, 0.5, 0.75, 1])
        axis.tick_params(colors="#697386", labelsize=8, length=0)
        axis.set_title(segment.capitalize(), color="#17213C", fontsize=11, pad=8)

        axis_kind = None
        maximum_ordinal = 0
        for model in models:
            rows = data.filter((pl.col("model") == model) & (pl.col("segment") == segment)).sort(
                "bin_index"
            )
            if rows.is_empty():
                continue
            current_axis_kind = str(rows["axis_kind"][0])
            if axis_kind is None:
                axis_kind = current_axis_kind
            elif axis_kind != current_axis_kind:
                raise ValueError(f"inconsistent axis kind for {dataset}/{segment}")
            x = rows["position_x"].to_numpy().astype(float)
            mean = rows["mean_percentile"].to_numpy().astype(float)
            low = rows["ci_low"].to_numpy().astype(float)
            high = rows["ci_high"].to_numpy().astype(float)
            finite_band = np.isfinite(low) & np.isfinite(high)
            if finite_band.any():
                axis.fill_between(
                    x,
                    low,
                    high,
                    where=finite_band,
                    color=MODEL_COLORS[model],
                    alpha=0.11,
                    linewidth=0,
                    zorder=2,
                )
            axis.plot(
                x,
                mean,
                color=MODEL_COLORS[model],
                linewidth=1.8,
                marker="o",
                markersize=2.7,
                markeredgewidth=0,
                zorder=3,
            )
            if current_axis_kind == "ordinal":
                maximum_ordinal = max(maximum_ordinal, int(rows["position_ordinal"].max()))

        if axis_kind == "normalized":
            axis.set_xlim(0, 1)
            axis.set_xticks(
                [0, 0.25, 0.5, 0.75, 1],
                ["0%", "25%", "50%", "75%", "100%"],
            )
            axis.set_xlabel("Relative position", color="#697386", fontsize=8.5)
        elif axis_kind == "ordinal":
            axis.set_xlim(0.5, maximum_ordinal + 0.5)
            axis.set_xticks(range(1, maximum_ordinal + 1))
            axis.set_xlabel("Token ordinal", color="#697386", fontsize=8.5)
        else:
            raise ValueError(f"missing position rows for {dataset}/{segment}")

    figure.supylabel(
        "Within-case relevance percentile",
        x=0.018,
        color="#697386",
        fontsize=9,
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model],
            linewidth=1.8,
            marker="o",
            markersize=3.2,
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
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=len(handles),
        frameon=False,
        fontsize=7.8,
    )
    _apply_paper_typography(figure)
    return figure


def make_paper_position_heatmap(summary: pl.DataFrame, dataset: str):
    """Render a separate one-row heatmap for every model and segment."""
    plt, matplotlib, _, _ = _load_matplotlib()
    data = summary.filter(pl.col("dataset") == dataset)
    if data.is_empty():
        raise ValueError(f"no position rows for {dataset}")
    models = [model for model in DATASET_MODELS[dataset] if model in set(data["model"])]
    segments = [segment for segment in SEGMENT_ORDER if segment in set(data["segment"])]
    figure, axes = plt.subplots(
        len(models),
        len(segments),
        figsize=PAPER_POSITION_HEATMAP_FIGSIZE,
        facecolor="#F6F8FC",
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.15,
        right=0.90,
        bottom=0.16,
        top=0.90,
        hspace=0.34,
        wspace=0.12,
    )
    colormap = matplotlib.colormaps["RdBu_r"].with_extremes(bad="#DDE3EE")
    image = None

    for row_index, model in enumerate(models):
        for column_index, segment in enumerate(segments):
            axis = axes[row_index, column_index]
            rows = data.filter((pl.col("model") == model) & (pl.col("segment") == segment)).sort(
                "bin_index"
            )
            if rows.is_empty():
                axis.set_visible(False)
                continue
            if rows["bin_index"].n_unique() != rows.height:
                raise ValueError(f"duplicate heatmap bins for {dataset}/{model}/{segment}")
            values = rows["mean_percentile"].to_numpy().astype(float)[None, :]
            image = axis.imshow(
                values,
                cmap=colormap,
                vmin=0.0,
                vmax=1.0,
                aspect="auto",
                interpolation="none",
            )
            if row_index == 0:
                axis.set_title(segment.capitalize(), color="#17213C", fontsize=11, pad=8)
            axis.set_yticks(
                [0],
                [MODEL_INFO[model]["name"]] if column_index == 0 else [""],
            )
            axis.tick_params(
                axis="y",
                colors="#17213C",
                labelsize=8.2,
                length=0,
            )
            axis_kind = str(rows["axis_kind"][0])
            show_x = row_index == len(models) - 1
            if axis_kind == "normalized":
                last_edge = rows.height - 0.5
                quartile_edges = np.linspace(-0.5, last_edge, 5)
                axis.set_xticks(
                    quartile_edges,
                    ["0%", "25%", "50%", "75%", "100%"],
                )
                if show_x:
                    axis.set_xlabel("Relative position", color="#697386", fontsize=8.5)
            elif axis_kind == "ordinal":
                axis.set_xticks(
                    np.arange(rows.height),
                    [str(index + 1) for index in range(rows.height)],
                )
                if show_x:
                    axis.set_xlabel("Token ordinal", color="#697386", fontsize=8.5)
            else:
                raise ValueError(f"missing heatmap axis kind for {dataset}/{model}/{segment}")
            axis.tick_params(
                axis="x",
                colors="#697386",
                labelsize=8,
                length=0,
                labelbottom=show_x,
            )
            axis.set_xticks(np.arange(-0.5, rows.height, 1), minor=True)
            axis.set_yticks([-0.5, 0.5], minor=True)
            axis.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=0.55)
            axis.tick_params(which="minor", bottom=False, left=False)
            for spine in axis.spines.values():
                spine.set_visible(False)

    if image is None:
        raise ValueError(f"no heatmap values for {dataset}")
    colorbar_axis = figure.add_axes([0.925, 0.16, 0.018, 0.74])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    colorbar.set_label(
        "Within-case relevance percentile",
        color="#697386",
        fontsize=9,
        labelpad=8,
    )
    colorbar.ax.tick_params(colors="#697386", labelsize=8, length=0)
    colorbar.outline.set_visible(False)
    _apply_paper_typography(figure)
    return figure


def make_paper_aggregated_position_heatmap(summary: pl.DataFrame, dataset: str):
    """Render one model-by-position matrix for each transcript segment."""
    plt, matplotlib, _, _ = _load_matplotlib()
    data = summary.filter(pl.col("dataset") == dataset)
    if data.is_empty():
        raise ValueError(f"no position rows for {dataset}")
    models = [model for model in DATASET_MODELS[dataset] if model in set(data["model"])]
    segments = [segment for segment in SEGMENT_ORDER if segment in set(data["segment"])]
    figure, axes = plt.subplots(
        1,
        len(segments),
        figsize=PAPER_AGGREGATED_POSITION_HEATMAP_FIGSIZE,
        facecolor="#F6F8FC",
        sharey=True,
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.15,
        right=0.90,
        bottom=0.25,
        top=0.88,
        wspace=0.12,
    )
    colormap = matplotlib.colormaps["RdBu_r"].with_extremes(bad="#DDE3EE")
    image = None

    for column_index, segment in enumerate(segments):
        axis = axes[0, column_index]
        segment_data = data.filter(pl.col("segment") == segment)
        bin_ids = sorted(segment_data["bin_index"].unique().to_list())
        if not bin_ids:
            raise ValueError(f"missing heatmap bins for {dataset}/{segment}")
        bin_to_column = {bin_id: index for index, bin_id in enumerate(bin_ids)}
        matrix = np.full((len(models), len(bin_ids)), np.nan, dtype=float)
        axis_kind = None
        for row_index, model in enumerate(models):
            rows = segment_data.filter(pl.col("model") == model).sort("bin_index")
            if rows.is_empty():
                continue
            current_axis_kind = str(rows["axis_kind"][0])
            if axis_kind is None:
                axis_kind = current_axis_kind
            elif axis_kind != current_axis_kind:
                raise ValueError(f"inconsistent axis kind for {dataset}/{segment}")
            if rows["bin_index"].n_unique() != rows.height:
                raise ValueError(f"duplicate heatmap bins for {dataset}/{model}/{segment}")
            for bin_id, value in rows.select("bin_index", "mean_percentile").iter_rows():
                matrix[row_index, bin_to_column[bin_id]] = float(value)

        image = axis.imshow(
            matrix,
            cmap=colormap,
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
            interpolation="none",
        )
        axis.set_title(segment.capitalize(), color="#17213C", fontsize=11, pad=8)
        axis.set_yticks(
            np.arange(len(models)),
            [MODEL_INFO[model]["name"] for model in models],
        )
        axis.tick_params(
            axis="y",
            colors="#17213C",
            labelsize=8.2,
            length=0,
            labelleft=column_index == 0,
        )
        if axis_kind == "normalized":
            last_edge = len(bin_ids) - 0.5
            quartile_edges = np.linspace(-0.5, last_edge, 5)
            axis.set_xticks(quartile_edges, ["0%", "25%", "50%", "75%", "100%"])
            axis.set_xlabel("Relative position", color="#697386", fontsize=8.5)
        elif axis_kind == "ordinal":
            axis.set_xticks(
                np.arange(len(bin_ids)),
                [str(index + 1) for index in range(len(bin_ids))],
            )
            axis.set_xlabel("Token ordinal", color="#697386", fontsize=8.5)
        else:
            raise ValueError(f"missing heatmap axis kind for {dataset}/{segment}")
        axis.tick_params(axis="x", colors="#697386", labelsize=8, length=0)
        axis.set_xticks(np.arange(-0.5, len(bin_ids), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
        axis.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=0.55)
        axis.tick_params(which="minor", bottom=False, left=False)
        for spine in axis.spines.values():
            spine.set_visible(False)

    if image is None:
        raise ValueError(f"no heatmap values for {dataset}")
    colorbar_axis = figure.add_axes([0.925, 0.25, 0.018, 0.63])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    colorbar.set_label(
        "Within-case relevance percentile",
        color="#697386",
        fontsize=9,
        labelpad=8,
    )
    colorbar.ax.tick_params(colors="#697386", labelsize=8, length=0)
    colorbar.outline.set_visible(False)
    _apply_paper_typography(figure)
    return figure


def _segment_order(data: pl.DataFrame) -> list[str]:
    present = set(data["segment"])
    return [segment for segment in SEGMENT_ORDER if segment in present]


def _on_task_legend_label(
    rows: pl.DataFrame,
    model: str,
    segments: list[str],
) -> str:
    rates = {row["segment"]: float(row["token_base_rate"]) for row in rows.iter_rows(named=True)}
    values = "/".join(f"{rates[segment]:.2f}" for segment in segments)
    return f"{MODEL_INFO[model]['name']}  (on-task {values})"


def _style_segment_axis(axis, segments: list[str], *, reference: float) -> None:
    axis.set_facecolor("#FFFFFF")
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.grid(axis="y", color="#DDE3EE", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.axhline(
        reference,
        color="#697386",
        linestyle=(0, (3, 3)),
        linewidth=0.9,
        zorder=1,
    )
    axis.set_xticks(np.arange(len(segments)), [segment.capitalize() for segment in segments])
    axis.tick_params(colors="#697386", labelsize=8.5, length=0)
    axis.set_xlabel("Segment", color="#697386", fontsize=9)


def _model_legend(axis, handles: list, labels: list[str], segments: list[str]):
    abbreviations = "/".join(segment[0].upper() for segment in segments)
    legend = axis.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.025, 1.0),
        borderaxespad=0,
        frameon=False,
        fontsize=7.4,
        handlelength=2.2,
        title=f"Model (on-task {abbreviations})",
        title_fontsize=7.4,
    )
    return legend


def _make_single_statistic_figure(
    data: pl.DataFrame,
    dataset: str,
    statistic: str,
):
    plt, _, Line2D, _ = _load_matplotlib()
    spec = SEGMENT_FIGURES[statistic]
    models = [model for model in DATASET_MODELS[dataset] if model in set(data["model"])]
    segments = _segment_order(data)
    figure, axis = plt.subplots(figsize=(7.4, 3.25), facecolor="#F6F8FC")
    figure.subplots_adjust(left=0.115, right=0.65, bottom=0.18, top=0.96)
    _style_segment_axis(axis, segments, reference=float(spec["reference"]))
    axis.set_ylabel(str(spec["ylabel"]), color="#697386", fontsize=9)
    axis.set_ylim(*spec["ylim"])

    offsets = np.linspace(-0.16, 0.16, len(models)) if len(models) > 1 else np.zeros(1)
    handles = []
    labels = []
    for model, offset in zip(models, offsets, strict=True):
        rows = data.filter(pl.col("model") == model).sort(
            pl.col("segment").replace_strict(SEGMENT_ORDER)
        )
        if rows["segment"].to_list() != segments:
            raise ValueError(f"inconsistent segment coverage for {dataset}/{model}")
        x = np.arange(len(segments), dtype=float) + offset
        values = rows[str(spec["value"])].to_numpy().astype(float)
        lows = rows[str(spec["low"])].to_numpy().astype(float)
        highs = rows[str(spec["high"])].to_numpy().astype(float)
        color = MODEL_COLORS[model]
        axis.plot(x, values, color=color, linewidth=1.15, alpha=0.75, zorder=2)
        for xi, value, low, high in zip(x, values, lows, highs, strict=True):
            if math.isfinite(value):
                axis.errorbar(
                    xi,
                    value,
                    yerr=_finite_error(value, low, high),
                    fmt="o",
                    color=color,
                    markersize=4.7,
                    capsize=3,
                    linewidth=1.15,
                    zorder=3,
                )
        handles.append(Line2D([0], [0], color=color, marker="o", linewidth=1.3))
        labels.append(_on_task_legend_label(rows, model, segments))

    axis.set_xlim(-0.42, len(segments) - 0.58)
    _model_legend(axis, handles, labels, segments)
    _apply_paper_typography(figure)
    return figure


def _make_enrichment_figure(data: pl.DataFrame, dataset: str):
    plt, _, Line2D, _ = _load_matplotlib()
    models = [model for model in DATASET_MODELS[dataset] if model in set(data["model"])]
    segments = _segment_order(data)
    figure, axis = plt.subplots(figsize=(7.4, 3.45), facecolor="#F6F8FC")
    figure.subplots_adjust(left=0.115, right=0.65, bottom=0.18, top=0.96)
    _style_segment_axis(axis, segments, reference=1.0)
    axis.set_ylabel("High-rank enrichment", color="#697386", fontsize=9)

    model_offsets = np.linspace(-0.18, 0.18, len(models)) if len(models) > 1 else np.zeros(1)
    budget_offset = {"top_1": -0.025, "top_10": 0.025}
    model_handles = []
    model_labels = []
    finite_highs = [1.0]
    for model, model_offset in zip(models, model_offsets, strict=True):
        rows = data.filter(pl.col("model") == model).sort(
            pl.col("segment").replace_strict(SEGMENT_ORDER)
        )
        if rows["segment"].to_list() != segments:
            raise ValueError(f"inconsistent segment coverage for {dataset}/{model}")
        color = MODEL_COLORS[model]
        for prefix, marker in (("top_1", "o"), ("top_10", "s")):
            x = np.arange(len(segments), dtype=float) + model_offset + budget_offset[prefix]
            values = rows[f"{prefix}_enrichment"].to_numpy().astype(float)
            lows = rows[f"{prefix}_enrichment_ci_low"].to_numpy().astype(float)
            highs = rows[f"{prefix}_enrichment_ci_high"].to_numpy().astype(float)
            finite_highs.extend(highs[np.isfinite(highs)].tolist())
            axis.plot(
                x,
                values,
                color=color,
                linestyle="-" if prefix == "top_1" else "--",
                linewidth=1.0,
                alpha=0.68,
                zorder=2,
            )
            for xi, value, low, high in zip(x, values, lows, highs, strict=True):
                if math.isfinite(value):
                    axis.errorbar(
                        xi,
                        value,
                        yerr=_finite_error(value, low, high),
                        fmt=marker,
                        color=color,
                        markerfacecolor="white" if prefix == "top_10" else color,
                        markersize=4.5,
                        capsize=2.7,
                        linewidth=1.0,
                        zorder=3,
                    )
        model_handles.append(Line2D([0], [0], color=color, marker="o", linewidth=1.3))
        model_labels.append(_on_task_legend_label(rows, model, segments))

    axis.set_xlim(-0.44, len(segments) - 0.56)
    axis.set_ylim(0, max(1.25, max(finite_highs) * 1.06))
    budget_handles = [
        Line2D([0], [0], color="#17213C", marker="o", linewidth=1.0),
        Line2D(
            [0],
            [0],
            color="#17213C",
            marker="s",
            markerfacecolor="white",
            linestyle="--",
            linewidth=1.0,
        ),
        Line2D(
            [0],
            [0],
            color="#697386",
            linestyle=(0, (3, 3)),
            linewidth=0.9,
        ),
    ]
    axis.legend(
        handles=[*model_handles, *budget_handles],
        labels=[*model_labels, "Top 1%", "Top 10%", "No enrichment (1)"],
        loc="center left",
        bbox_to_anchor=(1.025, 0.5),
        borderaxespad=0,
        frameon=False,
        fontsize=7.4,
        handlelength=2.2,
    )
    _apply_paper_typography(figure)
    return figure


def make_paper_segment_figures(summary: pl.DataFrame, dataset: str) -> dict[str, object]:
    """Return one model-overlay figure per segment statistic."""
    data = summary.filter(pl.col("dataset") == dataset)
    if data.is_empty():
        raise ValueError(f"no segment rows for {dataset}")
    if data.filter(pl.col("segment") == "trailer").height:
        raise ValueError("trailer rows are not valid segment-paper inputs")
    return {
        "auroc": _make_single_statistic_figure(data, dataset, "auroc"),
        "percentile": _make_single_statistic_figure(data, dataset, "percentile"),
        "enrichment": _make_enrichment_figure(data, dataset),
    }


def render_paper_plots(
    *,
    output_dir: Path,
    datasets: set[str],
    regimes: set[str],
    overwrite: bool,
) -> list[Path]:
    plt, _, _, _ = _load_matplotlib()
    written: list[Path] = []
    for regime_name in REGIMES:
        if regime_name not in regimes:
            continue
        config = REGIMES[regime_name]
        position_summary = _load_position_summary(config["position"], datasets)
        segment_summary = load_segment_summary(config["segment"], datasets)
        for dataset in DATASET_MODELS:
            if dataset not in datasets:
                continue
            position_figure = make_paper_position_figure(position_summary, dataset)
            try:
                written.append(
                    _atomic_save_pdf(
                        position_figure,
                        output_dir / f"{dataset}_{config['position_stem']}.pdf",
                        overwrite=overwrite,
                        fixed_canvas=True,
                    )
                )
            finally:
                plt.close(position_figure)

            aggregated_position_figure = make_paper_aggregated_position_figure(
                position_summary, dataset
            )
            try:
                written.append(
                    _atomic_save_pdf(
                        aggregated_position_figure,
                        output_dir / f"{dataset}_{config['aggregated_position_stem']}.pdf",
                        overwrite=overwrite,
                        fixed_canvas=True,
                    )
                )
            finally:
                plt.close(aggregated_position_figure)

            position_heatmap = make_paper_position_heatmap(position_summary, dataset)
            try:
                heatmap_base = output_dir / f"{dataset}_{config['position_heatmap_stem']}"
                written.extend(
                    (
                        _atomic_save_pdf(
                            position_heatmap,
                            heatmap_base.with_suffix(".pdf"),
                            overwrite=overwrite,
                            fixed_canvas=True,
                        ),
                        _atomic_save_png(
                            position_heatmap,
                            heatmap_base.with_suffix(".png"),
                            overwrite=overwrite,
                            fixed_canvas=True,
                        ),
                    )
                )
            finally:
                plt.close(position_heatmap)

            aggregated_position_heatmap = make_paper_aggregated_position_heatmap(
                position_summary, dataset
            )
            try:
                aggregated_heatmap_base = (
                    output_dir / f"{dataset}_{config['aggregated_position_heatmap_stem']}"
                )
                written.extend(
                    (
                        _atomic_save_pdf(
                            aggregated_position_heatmap,
                            aggregated_heatmap_base.with_suffix(".pdf"),
                            overwrite=overwrite,
                            fixed_canvas=True,
                        ),
                        _atomic_save_png(
                            aggregated_position_heatmap,
                            aggregated_heatmap_base.with_suffix(".png"),
                            overwrite=overwrite,
                            fixed_canvas=True,
                        ),
                    )
                )
            finally:
                plt.close(aggregated_position_heatmap)

            segment_figures = make_paper_segment_figures(segment_summary, dataset)
            for statistic, figure in segment_figures.items():
                try:
                    written.append(
                        _atomic_save_pdf(
                            figure,
                            output_dir / f"{dataset}_{config['segment_stem']}_{statistic}.pdf",
                            overwrite=overwrite,
                        )
                    )
                finally:
                    plt.close(figure)
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PAPER_DIR)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_MODELS),
        default=list(DATASET_MODELS),
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=tuple(REGIMES),
        default=list(REGIMES),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    written = render_paper_plots(
        output_dir=args.output_dir,
        datasets=set(args.datasets),
        regimes=set(args.regimes),
        overwrite=args.overwrite,
    )
    print(f"wrote {len(written)} paper figure files to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
