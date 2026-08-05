"""Shared metadata and small formulas for the positional token analysis."""

from __future__ import annotations

from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "paper_results" / "bridge"
ANALYSIS_DIR = Path(__file__).resolve().parent
SELECTION_METHODS_DIR = ANALYSIS_DIR / "selection_methods"
POOLED_AUROC_DIR = SELECTION_METHODS_DIR / "pooled_auroc"
CV_METHOD_DIR = SELECTION_METHODS_DIR / "cv_map"
SPEARMAN_METHOD_DIR = SELECTION_METHODS_DIR / "spearman"
AUROC_ENSEMBLE_DIR = SELECTION_METHODS_DIR / "auroc_ensemble"

POOLED_AUROC_BEST_RESULTS_DIR = POOLED_AUROC_DIR / "results" / "model_best"
POOLED_AUROC_SHARED_RESULTS_DIR = POOLED_AUROC_DIR / "results" / "dataset_shared"
CV_BEST_RESULTS_DIR = CV_METHOD_DIR / "results" / "model_best"
CV_SHARED_RESULTS_DIR = CV_METHOD_DIR / "results" / "dataset_shared"
COR_BEST_RESULTS_DIR = SPEARMAN_METHOD_DIR / "results" / "model_best"
COR_SHARED_RESULTS_DIR = SPEARMAN_METHOD_DIR / "results" / "dataset_shared"
AUROC_ENSEMBLE_BEST_RESULTS_DIR = AUROC_ENSEMBLE_DIR / "results" / "model_best"
AUROC_ENSEMBLE_SHARED_RESULTS_DIR = AUROC_ENSEMBLE_DIR / "results" / "dataset_shared"

NO_HD_SUFFIX = "_no_hd"
EXCLUDED_HEAD_DISAGREEMENT = "head_disagreement"


def no_hd_dir(path: Path) -> Path:
    """Return the additive output-directory variant for no-HD analyses."""
    return path.with_name(path.name + NO_HD_SUFFIX)


POOLED_AUROC_BEST_NO_HD_RESULTS_DIR = no_hd_dir(POOLED_AUROC_BEST_RESULTS_DIR)
POOLED_AUROC_SHARED_NO_HD_RESULTS_DIR = no_hd_dir(POOLED_AUROC_SHARED_RESULTS_DIR)
CV_BEST_NO_HD_RESULTS_DIR = no_hd_dir(CV_BEST_RESULTS_DIR)
CV_SHARED_NO_HD_RESULTS_DIR = no_hd_dir(CV_SHARED_RESULTS_DIR)
COR_BEST_NO_HD_RESULTS_DIR = no_hd_dir(COR_BEST_RESULTS_DIR)
COR_SHARED_NO_HD_RESULTS_DIR = no_hd_dir(COR_SHARED_RESULTS_DIR)

CV_SELECTION_PATH = CV_BEST_RESULTS_DIR / "cv_best_metric_selection.parquet"
CV_SHARED_SELECTION_PATH = CV_SHARED_RESULTS_DIR / "cv_shared_metric_selection.parquet"
COR_SELECTION_PATH = COR_BEST_RESULTS_DIR / "cor_best_metric_selection.parquet"
COR_SHARED_SELECTION_PATH = COR_SHARED_RESULTS_DIR / "cor_shared_metric_selection.parquet"
AUROC_ENSEMBLE_SELECTION_PATH = (
    AUROC_ENSEMBLE_BEST_RESULTS_DIR / "auroc_ensemble_best_metric_selection.parquet"
)
AUROC_ENSEMBLE_SHARED_SELECTION_PATH = (
    AUROC_ENSEMBLE_SHARED_RESULTS_DIR / "auroc_ensemble_shared_metric_selection.parquet"
)

CV_NO_HD_SELECTION_PATH = CV_BEST_NO_HD_RESULTS_DIR / "cv_best_metric_selection.parquet"
CV_SHARED_NO_HD_SELECTION_PATH = CV_SHARED_NO_HD_RESULTS_DIR / "cv_shared_metric_selection.parquet"
COR_NO_HD_SELECTION_PATH = COR_BEST_NO_HD_RESULTS_DIR / "cor_best_metric_selection.parquet"
COR_SHARED_NO_HD_SELECTION_PATH = (
    COR_SHARED_NO_HD_RESULTS_DIR / "cor_shared_metric_selection.parquet"
)

DATASET_MODELS = {
    "opi": ("q7", "g12", "g27", "l70"),
    "tt": ("q7", "g12", "g27", "l70"),
    "liars": ("g27", "l70"),
    "taboo": ("q7", "g12", "g27", "l70"),
}

MODEL_INFO = {
    "q7": {
        "name": "Qwen2.5-7B",
        "tokenizer_family": "byte_bpe",
        "layer": 20,
        "d_model": 3584,
    },
    "g12": {
        "name": "Gemma-3-12B",
        "tokenizer_family": "sentencepiece",
        "layer": 32,
        "d_model": 3840,
    },
    "g27": {
        "name": "Gemma-3-27B",
        "tokenizer_family": "sentencepiece",
        "layer": 41,
        "d_model": 5376,
    },
    "l70": {
        "name": "Llama-3.3-70B",
        "tokenizer_family": "byte_bpe",
        "layer": 53,
        "d_model": 8192,
    },
}

# Bolded winners in findings/token-selector/bridge-{kind}-all.md.
METRIC_SPECS = {
    ("tt", "q7"): {"metric": "resid_jump", "auroc": 0.639, "direction": "higher"},
    ("tt", "g12"): {"metric": "norm_ratio", "auroc": 0.719, "direction": "higher"},
    ("tt", "g27"): {
        "metric": "resid_jump_nla",
        "auroc": 0.672,
        "direction": "higher",
    },
    ("tt", "l70"): {
        "metric": "resid_jump_nla",
        "auroc": 0.695,
        "direction": "higher",
    },
    ("opi", "q7"): {"metric": "peak_ratio", "auroc": 0.239, "direction": "lower"},
    ("opi", "g12"): {
        "metric": "resid_jump_nla",
        "auroc": 0.685,
        "direction": "higher",
    },
    ("opi", "g27"): {
        "metric": "resid_jump_nla",
        "auroc": 0.635,
        "direction": "higher",
    },
    ("opi", "l70"): {"metric": "sink_drain", "auroc": 0.673, "direction": "higher"},
    ("liars", "g27"): {
        "metric": "head_disagreement",
        "auroc": 0.280,
        "direction": "lower",
    },
    ("liars", "l70"): {
        "metric": "head_disagreement",
        "auroc": 0.318,
        "direction": "lower",
    },
    ("taboo", "q7"): {
        "metric": "dominant_mass",
        "auroc": 0.796,
        "direction": "higher",
    },
    ("taboo", "g12"): {
        "metric": "dominant_mass",
        "auroc": 0.681,
        "direction": "higher",
    },
    ("taboo", "g27"): {
        "metric": "dominant_mass",
        "auroc": 0.651,
        "direction": "higher",
    },
    ("taboo", "l70"): {
        "metric": "resid_jump_nla",
        "auroc": 0.723,
        "direction": "higher",
    },
}

# One metric per dataset, selected by the highest equal-model mean
# direction-adjusted AUROC max(A, 1-A) among signals available to every model.
SHARED_METRIC_SPECS = {
    ("opi", "q7"): {"metric": "peak_ratio", "auroc": 0.239, "direction": "lower"},
    ("opi", "g12"): {"metric": "peak_ratio", "auroc": 0.414, "direction": "lower"},
    ("opi", "g27"): {"metric": "peak_ratio", "auroc": 0.384, "direction": "lower"},
    ("opi", "l70"): {"metric": "peak_ratio", "auroc": 0.361, "direction": "lower"},
    ("tt", "q7"): {"metric": "resid_jump_nla", "auroc": 0.619, "direction": "higher"},
    ("tt", "g12"): {"metric": "resid_jump_nla", "auroc": 0.707, "direction": "higher"},
    ("tt", "g27"): {"metric": "resid_jump_nla", "auroc": 0.672, "direction": "higher"},
    ("tt", "l70"): {"metric": "resid_jump_nla", "auroc": 0.695, "direction": "higher"},
    ("liars", "g27"): {"metric": "head_disagreement", "auroc": 0.280, "direction": "lower"},
    ("liars", "l70"): {"metric": "head_disagreement", "auroc": 0.318, "direction": "lower"},
    ("taboo", "q7"): {"metric": "dominant_mass", "auroc": 0.796, "direction": "higher"},
    ("taboo", "g12"): {"metric": "dominant_mass", "auroc": 0.681, "direction": "higher"},
    ("taboo", "g27"): {"metric": "dominant_mass", "auroc": 0.651, "direction": "higher"},
    ("taboo", "l70"): {"metric": "dominant_mass", "auroc": 0.703, "direction": "higher"},
}

# Additive pooled-AUROC selection with head_disagreement ineligible to win.
# Only Liars changes. Values are recomputed from the same canonical parquets
# used for the published table; every other pair retains its original winner.
NO_HD_METRIC_SPECS = {
    **METRIC_SPECS,
    ("liars", "g27"): {
        "metric": "varentropy",
        "auroc": 0.6931778567919606,
        "direction": "higher",
    },
    ("liars", "l70"): {
        "metric": "sink_drain",
        "auroc": 0.337085808712217,
        "direction": "lower",
    },
}

NO_HD_SHARED_METRIC_SPECS = {
    **SHARED_METRIC_SPECS,
    ("liars", "g27"): {
        "metric": "sink_drain",
        "auroc": 0.35350767568924146,
        "direction": "lower",
    },
    ("liars", "l70"): {
        "metric": "sink_drain",
        "auroc": 0.337085808712217,
        "direction": "lower",
    },
}

METRIC_REGIMES = (
    "model-best",
    "dataset-shared",
    "cv-best",
    "cv-shared",
    "cor-best",
    "cor-shared",
    "auroc-ensemble-best",
    "auroc-ensemble-shared",
)

CONTENT_WINNERS = {
    ("opi", "q7"): "peak_ratio",
    ("opi", "g12"): "resid_jump_nla",
    ("opi", "g27"): "peak_ratio",
    ("opi", "l70"): "peak_ratio",
    ("tt", "q7"): "resid_jump",
    ("tt", "g12"): "norm_ratio",
    ("tt", "g27"): "norm_ratio",
    ("tt", "l70"): "resid_jump_nla",
    ("liars", "g27"): "head_disagreement",
    ("liars", "l70"): "head_disagreement",
    ("taboo", "q7"): "dominant_mass",
    ("taboo", "g12"): "dominant_mass",
    ("taboo", "g27"): "dominant_mass",
    ("taboo", "l70"): "norm_ratio",
}

NO_HD_CONTENT_WINNERS = {
    **CONTENT_WINNERS,
    ("liars", "g27"): "varentropy",
    ("liars", "l70"): "sink_drain",
}

FORWARD_SIGNALS = (
    "surprisal",
    "entropy",
    "varentropy",
    "temporal_kl",
    "resid_jump",
    "lookback_ratio",
    "sink_drain",
    "head_disagreement",
    "w",
)
ACTIVATION_Q2_SIGNALS = ("norm_ratio", "peak_ratio", "dominant_mass", "resid_jump_nla")
ALL_METRICS = (*FORWARD_SIGNALS, "act_norm", *ACTIVATION_Q2_SIGNALS)
POSITION_BINS = ((0, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 10**9))

LIARS_SUBDATASETS = (
    "harm-pressure-knowledge-report",
    "harm-pressure-choice",
    "instructed-deception",
    "insider-trading",
    "convincing-game",
)

EXPECTED_TOTAL_ROWS = 4_705_657

CANDIDATE_METRICS = (
    "surprisal",
    "entropy",
    "varentropy",
    "temporal_kl",
    "resid_jump",
    "lookback_ratio",
    "sink_drain",
    "head_disagreement",
    "w",
    "norm_ratio",
    "peak_ratio",
    "dominant_mass",
    "resid_jump_nla",
)
OPI_ONLY_METRICS = ("attn_rollout",)
# Used only after exact ties in primary and fixed-budget scores. All metrics are
# already stored; the grouping reflects extra selector computation, not model FLOPs.
METRIC_COST_ORDER = {metric: 0 for metric in CANDIDATE_METRICS}
for _derived_metric in (
    "w",
    "norm_ratio",
    "peak_ratio",
    "dominant_mass",
    "resid_jump_nla",
):
    METRIC_COST_ORDER[_derived_metric] = 1
METRIC_COST_ORDER["attn_rollout"] = 1


def source_path(input_dir: Path, dataset: str, model: str) -> Path:
    return input_dir / f"all_{dataset}_{model}.parquet"


def selected_pairs(
    datasets: set[str] | None = None, models: set[str] | None = None
) -> list[tuple[str, str]]:
    pairs = []
    for dataset, dataset_models in DATASET_MODELS.items():
        if datasets is not None and dataset not in datasets:
            continue
        for model in dataset_models:
            if models is None or model in models:
                pairs.append((dataset, model))
    return pairs


def z_expr(column: str, group: str = "case_id") -> pl.Expr:
    """Repository-compatible sample z-score within a transcript."""
    return (pl.col(column) - pl.col(column).mean().over(group)) / (
        pl.col(column).std().over(group) + 1e-12
    )


def liars_subdataset_expr() -> pl.Expr:
    expr = pl.lit(None, dtype=pl.String)
    for name in reversed(LIARS_SUBDATASETS):
        expr = (
            pl.when(pl.col("case_id").str.starts_with(name + "_"))
            .then(pl.lit(name))
            .otherwise(expr)
        )
    return expr


def load_cv_metric_specs(
    path: Path = CV_SELECTION_PATH,
) -> dict[tuple[str, str], dict[str, object]]:
    """Load the cross-validated winner manifest without importing selector code."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; run token_analysis/selection_methods/cv_map/select_best_metrics.py first"
        )
    required = {
        "dataset",
        "model",
        "metric_name",
        "metric_direction",
        "pooled_auroc_raw",
        "cv_macro_ap",
        "nested_cv_macro_ap",
        "statistically_tied",
        "beats_position_baseline",
        "beats_region_baseline",
    }
    manifest = pl.read_parquet(path)
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    return {
        (str(row["dataset"]), str(row["model"])): {
            "metric": str(row["metric_name"]),
            "auroc": float(row["pooled_auroc_raw"]),
            "direction": str(row["metric_direction"]),
            "cv_macro_ap": float(row["cv_macro_ap"]),
            "nested_cv_macro_ap": float(row["nested_cv_macro_ap"]),
            "statistically_tied": bool(row["statistically_tied"]),
            "beats_position_baseline": bool(row["beats_position_baseline"]),
            "beats_region_baseline": bool(row["beats_region_baseline"]),
        }
        for row in manifest.iter_rows(named=True)
    }


def load_cv_shared_metric_specs(
    path: Path = CV_SHARED_SELECTION_PATH,
) -> dict[tuple[str, str], dict[str, object]]:
    """Load the cross-validated dataset-shared winner manifest."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; run "
            "token_analysis/selection_methods/cv_map/select_shared_metrics.py first"
        )
    required = {
        "dataset",
        "model",
        "metric_name",
        "metric_direction",
        "pooled_auroc_raw",
        "cv_macro_ap",
        "equal_model_cv_macro_ap",
        "nested_cv_shared_macro_ap",
        "statistically_tied",
        "beats_position_baseline",
        "beats_region_baseline",
    }
    manifest = pl.read_parquet(path)
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    return {
        (str(row["dataset"]), str(row["model"])): {
            "metric": str(row["metric_name"]),
            "auroc": float(row["pooled_auroc_raw"]),
            "direction": str(row["metric_direction"]),
            "cv_macro_ap": float(row["cv_macro_ap"]),
            "equal_model_cv_macro_ap": float(row["equal_model_cv_macro_ap"]),
            "nested_cv_shared_macro_ap": float(row["nested_cv_shared_macro_ap"]),
            "statistically_tied": bool(row["statistically_tied"]),
            "beats_position_baseline": bool(row["beats_position_baseline"]),
            "beats_region_baseline": bool(row["beats_region_baseline"]),
        }
        for row in manifest.iter_rows(named=True)
    }


def load_cor_metric_specs(
    path: Path = COR_SELECTION_PATH,
) -> dict[tuple[str, str], dict[str, object]]:
    # Load full-data per-model Spearman-correlation winners.
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; run "
            "token_analysis/selection_methods/spearman/select_metrics.py first"
        )
    required = {
        "dataset",
        "model",
        "metric_name",
        "metric_direction",
        "pooled_auroc_raw",
        "spearman_rho",
        "abs_spearman_rho",
        "statistically_tied",
        "content_only_winner",
        "winner_changes_content_only",
        "direction_stability",
    }
    manifest = pl.read_parquet(path)
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    return {
        (str(row["dataset"]), str(row["model"])): {
            "metric": str(row["metric_name"]),
            "auroc": float(row["pooled_auroc_raw"]),
            "direction": str(row["metric_direction"]),
            "spearman_rho": float(row["spearman_rho"]),
            "abs_spearman_rho": float(row["abs_spearman_rho"]),
            "selection_score": float(row["abs_spearman_rho"]),
            "statistically_tied": bool(row["statistically_tied"]),
            "content_only_winner": str(row["content_only_winner"]),
            "winner_changes_content_only": bool(row["winner_changes_content_only"]),
            "direction_stability": float(row["direction_stability"]),
        }
        for row in manifest.iter_rows(named=True)
    }


def load_cor_shared_metric_specs(
    path: Path = COR_SHARED_SELECTION_PATH,
) -> dict[tuple[str, str], dict[str, object]]:
    # Load full-data dataset-shared Spearman-correlation winners.
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; run "
            "token_analysis/selection_methods/spearman/select_metrics.py first"
        )
    required = {
        "dataset",
        "model",
        "metric_name",
        "metric_direction",
        "pooled_auroc_raw",
        "spearman_rho",
        "abs_spearman_rho",
        "equal_model_abs_spearman_rho",
        "statistically_tied",
        "content_only_winner",
        "winner_changes_content_only",
        "direction_stability",
    }
    manifest = pl.read_parquet(path)
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    return {
        (str(row["dataset"]), str(row["model"])): {
            "metric": str(row["metric_name"]),
            "auroc": float(row["pooled_auroc_raw"]),
            "direction": str(row["metric_direction"]),
            "spearman_rho": float(row["spearman_rho"]),
            "abs_spearman_rho": float(row["abs_spearman_rho"]),
            "equal_model_abs_spearman_rho": float(row["equal_model_abs_spearman_rho"]),
            "selection_score": float(row["equal_model_abs_spearman_rho"]),
            "statistically_tied": bool(row["statistically_tied"]),
            "content_only_winner": str(row["content_only_winner"]),
            "winner_changes_content_only": bool(row["winner_changes_content_only"]),
            "direction_stability": float(row["direction_stability"]),
        }
        for row in manifest.iter_rows(named=True)
    }


def load_auroc_ensemble_specs(
    path: Path = AUROC_ENSEMBLE_SELECTION_PATH,
) -> dict[tuple[str, str], dict[str, object]]:
    """Load selected rank-ensemble definitions without importing the selector."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; run "
            "token_analysis/selection_methods/auroc_ensemble/select_metrics.py first"
        )
    required = {
        "dataset", "model", "candidate_id", "candidate_type",
        "component_1", "component_1_weight", "component_1_direction",
        "component_2", "component_2_weight", "component_2_direction",
        "candidate_direction", "pooled_auroc_raw", "adjusted_auroc",
    }
    manifest = pl.read_parquet(path)
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    optional = (
        "equal_model_adjusted_auroc",
        "gain_over_best_original",
        "winner_changes_common_support",
        "heldout_pooled_auroc",
        "heldout_case_macro_auroc",
        "heldout_case_macro_delta",
        "heldout_delta_ci_low",
        "heldout_delta_ci_high",
        "outer_fold_selection_frequency",
    )
    output = {}
    for row in manifest.iter_rows(named=True):
        spec = {
            key: row.get(key)
            for key in required
            if key not in {"dataset", "model"}
        }
        spec.update({key: row.get(key) for key in optional})
        spec["metric"] = spec["candidate_id"]
        spec["auroc"] = spec["adjusted_auroc"]
        spec["direction"] = spec["candidate_direction"]
        spec["winner_changes_content_only"] = False
        spec["content_only_winner"] = None
        output[(str(row["dataset"]), str(row["model"]))] = spec
    return output


def metric_specs(
    regime: str = "model-best",
    selection_manifest: Path | None = None,
    exclude_head_disagreement: bool = False,
) -> dict[tuple[str, str], dict[str, object]]:
    if regime == "model-best":
        return NO_HD_METRIC_SPECS if exclude_head_disagreement else METRIC_SPECS
    if regime == "dataset-shared":
        return NO_HD_SHARED_METRIC_SPECS if exclude_head_disagreement else SHARED_METRIC_SPECS
    if regime == "cv-best":
        return load_cv_metric_specs(selection_manifest or CV_SELECTION_PATH)
    if regime == "cv-shared":
        return load_cv_shared_metric_specs(selection_manifest or CV_SHARED_SELECTION_PATH)
    if regime == "cor-best":
        return load_cor_metric_specs(selection_manifest or COR_SELECTION_PATH)
    if regime == "auroc-ensemble-best":
        return load_auroc_ensemble_specs(selection_manifest or AUROC_ENSEMBLE_SELECTION_PATH)
    if regime == "auroc-ensemble-shared":
        return load_auroc_ensemble_specs(selection_manifest or AUROC_ENSEMBLE_SHARED_SELECTION_PATH)
    if regime == "cor-shared":
        return load_cor_shared_metric_specs(selection_manifest or COR_SHARED_SELECTION_PATH)
    raise ValueError(f"unknown metric regime: {regime}")


def direction_sign(
    dataset: str,
    model: str,
    regime: str = "model-best",
    selection_manifest: Path | None = None,
    exclude_head_disagreement: bool = False,
) -> float:
    specs = metric_specs(regime, selection_manifest, exclude_head_disagreement)
    return 1.0 if specs[(dataset, model)]["direction"] == "higher" else -1.0
