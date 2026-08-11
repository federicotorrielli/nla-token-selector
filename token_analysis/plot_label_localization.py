"""Figure for the on-task label: how common on-task explanations are, and how far
they concentrate inside the region that carries the threat.

Panel (a) is the base rate, the share of all positions whose explanation the judge
calls on-task. Panel (b) is the enrichment, the on-task rate inside the marked
region divided by the rate outside it, with the comparison held inside one chat
role so that role does not stand in for the threat. The taboo organisms carry no
marked region and appear in panel (a) alone.

Usage:
    uv run --with matplotlib python token_analysis/plot_label_localization.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"

MODELS = {"q7": "Qwen2.5-7B", "g12": "Gemma-3-12B", "g27": "Gemma-3-27B", "l70": "Llama-3.3-70B"}
# dataset -> (label, models, chat role holding the threat contrast)
DATASETS = [
    ("opi", "OpenPromptInjection", ["q7", "g12", "g27", "l70"], "user"),
    ("taboo", "Taboo organisms", ["q7", "g12", "g27", "l70"], None),
    ("liars", "Liars' Bench", ["g27", "l70"], "assistant"),
    ("tt", "Tensor Trust", ["q7", "g12", "g27", "l70"], "user"),
]
COLOURS = {
    "opi": "#1f6fb4",
    "taboo": "#2c8c5a",
    "liars": "#8a4fa8",
    "tt": "#c94a2b",
}


def collect():
    rows = []
    for key, label, models, role in DATASETS:
        for m in models:
            t = pq.read_table(
                ROOT / f"paper_results/bridge/all_{key}_{m}.parquet",
                columns=["label", "on_task", "region"],
            ).to_pandas()
            base = t.on_task.mean()
            ratio = np.nan
            if role is not None:
                r = t[t.region == role]
                inside, outside = r[r.label == 1].on_task.mean(), r[r.label == 0].on_task.mean()
                ratio = inside / outside
            rows.append((key, label, m, base, ratio))
    return rows


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.0,
        }
    )
    rows = collect()
    # y positions with a gap between datasets, top row first
    ypos, groups, cur = [], [], 0.0
    for key, label, models, _ in DATASETS:
        first = cur
        for _ in models:
            ypos.append(cur)
            cur += 1.0
        groups.append((key, label, first, cur - 1.0))
        cur += 1.0
    top = cur - 1.0
    y = np.array([top - v for v in ypos])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.4), sharey=True)

    # panel (a): base rate
    ax1.axvspan(0.5, 1.0, color="0.91", zorder=0)
    ax1.barh(y, [r[3] for r in rows], height=0.7, color=[COLOURS[r[0]] for r in rows], zorder=3)
    for yi, r in zip(y, rows):
        ax1.text(r[3] * 1.15, yi, f"{r[3]:.3f}", va="center", ha="left", fontsize=6.5)
    ax1.set_xscale("log")
    ax1.set_xlim(0.006, 3.2)
    ax1.set_xticks([0.01, 0.1, 1.0], ["0.01", "0.1", "1"])
    ax1.set_xlabel("share of positions the judge calls on-task")
    ax1.set_title("(a) how common an on-task explanation is", fontsize=8, loc="left", pad=12)

    # panel (b): enrichment inside the marked region
    ax2.axvline(1.0, color="0.45", lw=0.8, ls="--", zorder=1)
    for yi, r in zip(y, rows):
        if np.isnan(r[4]):
            continue
        ax2.plot([1.0, r[4]], [yi, yi], color=COLOURS[r[0]], lw=1.5, zorder=2)
        ax2.plot([r[4]], [yi], "o", color=COLOURS[r[0]], ms=4.5, zorder=3)
        lab = f"{r[4]:.0f}x" if r[4] >= 1 else f"{r[4]:.2f}x"
        ax2.text(
            r[4] * (1.2 if r[4] >= 1 else 0.72),
            yi,
            lab,
            va="center",
            ha="left" if r[4] >= 1 else "right",
            fontsize=6.5,
        )
    ax2.set_xscale("log")
    ax2.set_xlim(0.35, 260)
    ax2.set_xticks([0.5, 1, 10, 100], ["0.5", "1", "10", "100"])
    ax2.set_xlabel("on-task rate inside the threat span / outside it")
    ax2.set_title("(b) how far on-task explanations concentrate", fontsize=8, loc="left", pad=12)

    ax1.set_yticks(y, [MODELS[r[2]] for r in rows], fontsize=7)
    ax1.set_ylim(-0.8, top + 1.7)
    for ax in (ax1, ax2):
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

    for key, label, first, last in groups:
        ax1.text(
            0.0,
            top - first + 0.80,
            label,
            transform=ax1.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=7.5,
            fontweight="bold",
            color=COLOURS[key],
        )
    ax1.text(
        0.70,
        top - groups[-1][2] + 0.80,
        "no room to select",
        fontsize=6.5,
        color="0.4",
        va="center",
        ha="center",
    )

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"label_localization.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", OUT / "label_localization.pdf")


if __name__ == "__main__":
    main()
