# Positional bridge token analysis

This directory audits the canonical all-token bridge run, builds a plot-ready long-format token table, and compares pooled AUROC, cross-validated MAP, Spearman, and exhaustive AUROC rank-ensemble selectors in per-model and dataset-shared variants. It does not run models, generate NLA verbalizations, recompute signals, or judge tokens.

## Directory layout

Shared infrastructure and artifacts remain at the top level:

- `common.py`: shared metadata, regime paths, and small formulas.
- `audit_bridge_inputs.py`: one audit engine for all eight regimes.
- `build_token_position_data.py`: lazy Polars builder for the canonical table.
- `plot_token_positions.py`: positional and ensemble segment-comparison rendering.
- `token_position_scores.parquet`: the 4,705,657-row, eight-regime canonical token table.
- `PLOT_README.md`: common figure, normalization, aggregation, and provenance guide.
- `tests/`: shared builder and plotting regression tests.
- Existing PDF and text artifacts: unrelated source material, kept at the top level.

Selection-specific code and outputs live under `selection_methods/`:

```text
selection_methods/
├── pooled_auroc/
│   ├── results/{model_best,dataset_shared}/
│   └── plots/{model_best,dataset_shared}/
├── cv_map/
│   ├── select_best_metrics.py
│   ├── select_shared_metrics.py
│   ├── docs/
│   ├── tests/
│   ├── results/{model_best,dataset_shared}/
│   └── plots/{model_best,dataset_shared}/
└── spearman/
    ├── select_metrics.py
    ├── docs/
    ├── tests/
    ├── results/{model_best,dataset_shared}/
    └── plots/{model_best,dataset_shared}/
```

### Published pooled-AUROC selection

`selection_methods/pooled_auroc/` contains the analyses based on the published
all-token AUROC winners and the equal-model dataset-shared AUROC experiment.

- `results/model_best/bridge_input_audit.md`
- `results/model_best/bridge_input_audit_interpretation.md`
- `results/model_best/position_plot_summary.parquet`
- `results/dataset_shared/bridge_input_audit_shared_metric.md`
- `results/dataset_shared/shared_metric_position_summary.parquet`
- `plots/model_best/`: four per-model-winner PNG/SVG figure pairs.
- `plots/dataset_shared/`: four dataset-shared PNG/SVG figure pairs.

### Cross-validated case-balanced MAP selection

`selection_methods/cv_map/` contains both CV selectors, their focused tests,
method documentation, and outputs.

- `select_best_metrics.py`: grouped-CV per-model selector.
- `select_shared_metrics.py`: nested-CV selector constrained to one metric per dataset.
- `docs/cv_selector_explainer.md` and `docs/new_selection_plan.md`.
- `results/model_best/`: candidate results, winner manifest, held-out case scores,
  generated report, audit, and positional summary.
- `results/dataset_shared/`: shared candidate results, winner manifest, generated
  report, audit, and positional summary.
- `plots/model_best/` and `plots/dataset_shared/`: four PNG/SVG figure pairs each.

### Spearman-correlation selection

`selection_methods/spearman/` contains the per-model and shared Spearman
selector, its focused tests, documentation, and outputs.

- `select_metrics.py`: creates both `cor-best` and `cor-shared` manifests.
- `docs/cor_selector_explainer.md`.
- `results/cor_metric_selection.md`: combined generated selection report.
- `results/model_best/`: candidate results, winner manifest, audit, and positional summary.
- `results/dataset_shared/`: shared candidate results, winner manifest, audit, and positional summary.
- `plots/model_best/` and `plots/dataset_shared/`: four PNG/SVG figure pairs each.

### Exhaustive AUROC rank ensembles

selection_methods/auroc_ensemble/ evaluates every original metric and every
unordered 25/75, 50/50, and 75/25 pair without top-k preselection.

- select_metrics.py creates auroc-ensemble-best and auroc-ensemble-shared.
- docs/auroc_ensemble_explainer.md gives formulas, missingness, validation, and interpretation.
- results/ holds candidate tables, manifests, grouped folds, eligibility, audits, and segment analyses.
- plots/ holds positional profiles and fixed-winner segment comparisons.
- opi_attn_rollout_sensitivity/ is the explicitly non-blind OPI-only sensitivity.

This structure separates method-specific evidence without duplicating the
canonical token table or the common audit and plotting implementations.

## Table semantics

The table is a vertical union. Its primary key is `(dataset, model, case_id, full_position)`; tokens from different models are not joined because tokenization and chat templates differ.

`token_raw` is the exact vocabulary string returned by the source tokenizer. Markers such as `Ġ`, `Ċ`, `▁`, byte-to-Unicode strings such as `âĢĿ`, and model-specific angle-bracket tokens are deliberately not decoded. `source_region` preserves the canonical extraction label; `analysis_region` records the corrected semantic role used for segmentation and plotting. `source_boundary_length` and `boundary_was_corrected` make the two TT corrections auditable rather than silently overwriting them.

Every token receives exactly one analysis segment:

- `input`: all content through the final non-template token before the final assistant turn, including earlier chat scaffolding and Liars `assistant_prior` turns.
- `boundary`: the final assistant-generation template prompt before final assistant content, or OPI's trailing generation prompt. The two documented TT source-label artifacts are corrected to this five-token prompt while their raw source lengths remain available.
- `output`: the contiguous final `assistant` content.
- `trailer`: raw Liars chat-template terminators after final assistant content. They remain in the canonical all-token table for provenance but are excluded from every positional summary and figure.

The selected all-token winner is stored in `metric_value_raw`. For the below-chance OPI/Qwen and Liars winners, `metric_value_aligned` reverses the sign so that higher always means more relevant. Non-finite raw source values remain present; derived score fields are null for those rows. Within-case z-scores and percentiles support later fixed-budget work without recomputation.

The parallel dataset-shared experiment uses the highest equal-model mean direction-adjusted AUROC `max(A, 1-A)` among metrics available to every model in that dataset: OPI `peak_ratio`, Tensor Trust `resid_jump_nla`, Liars `head_disagreement`, and Taboo `dominant_mass`. Its raw, aligned, z-scored, and within-case percentile values are stored in the corresponding `shared_metric_*` columns. The builder keeps the original per-model winner columns unchanged; audit and plotting select the shared columns only when `--metric-regime dataset-shared` is passed.

The `cv-best` experiment instead treats every transcript as one retrieval query and selects the metric with highest five-fold out-of-fold case-macro Average Precision. Complete cases define folds; direction is learned only from training cases; non-finite scores remain eligible and rank last; and all token types remain included. A nested grouped CV estimate evaluates the full metric-and-direction selection procedure. Paired case bootstrap intervals, fixed-budget precision/recall/lift, content sensitivity, and cross-fitted position/region baselines qualify the selected winner. This is called a **cross-validated case-balanced MAP-selected metric**, not the “true” metric. Its table columns use the `cv_metric_*` prefix; `--metric-regime cv-best` selects them without changing either earlier regime.

The `cv-shared` experiment imposes the additional constraint that every model in a dataset uses the same metric. It selects the common candidate with the largest **equal-model mean** of held-out case-macro AP, so each model has weight 1/M regardless of its number or length of transcripts. Direction remains model-specific and is learned only on training cases. Nested CV repeats the shared metric choice on inner folds before evaluating untouched outer cases. Current winners are OPI `peak_ratio`, Tensor Trust `sink_drain`, Liars `temporal_kl`, and Taboo `dominant_mass`. The table uses `cv_shared_metric_*`; its selection score is dataset-level equal-model CV MAP, while `cv_shared_metric_model_cv_macro_ap` records each row model’s behavior.

The `cor-best` experiment selects the largest absolute pooled Spearman correlation between finite metric values and binary `on_task`; its sign defines direction. `cor-shared` maximizes the equal-model mean absolute correlation while keeping model-specific signs. All winners and rho point estimates use the full original data. A 2,000-replicate fixed-rank whole-case bootstrap is used only for descriptive uncertainty. The table stores these regimes in `cor_metric_*` and `cor_shared_metric_*` columns. Because binary-label Spearman correlation is closely related to the rank statistic underlying AUROC, `cor-best` reproduces the pooled-AUROC winners and is not independent confirmation.

The `auroc-ensemble-best` and `auroc-ensemble-shared` experiments turn each component into a pooled fractional midrank, orient it toward on-task, and exhaustively evaluate every original and weighted pair. Both now have grouped held-out validation: the shared version synchronizes union case-ID folds across models and reselects one candidate by equal-model training AUROC inside every fold. It additionally reports label-free and target-calibrated leave-one-model-out transfer. The canonical table stores only the selected full-data candidates in `auroc_ensemble_metric_*` and `auroc_ensemble_shared_metric_*`; validation remains in method-level Parquets and does not change positional plots. Segment-specific selection and frozen-winner comparisons use only input, boundary, and output, never Liars trailers.

Missingness is method-specific. CV-MAP ranks non-finite values last to retain a common candidate pool; pooled AUROC, Spearman, and AUROC ensembles use pairwise deletion. An ensemble is null unless both components are finite. In positional figures all missing derived percentiles remain null; no source token row is deleted or imputed.

Persisted `w` is copied from the canonical parquet and only audited against the repository definition `z(sink_drain) - z(lookback_ratio)`. `is_extreme_norm` is the documented analysis proxy `norm_ratio > 5`, not the repository's factor-of-10 spike-channel definition. Template, extreme, dense Tensor Trust, and post-1024 Gemma rows are labelled and never removed.

## Position-plot method

The primary plotted score is the precomputed, direction-aligned within-case fractional rank `metric_percentile_within_case`. It lies in `[0, 1]`, preserves ties with midranks, and makes 1 mean the position ranked most relevant by the selected metric. It is a **relative rank, not a relevance probability**. Because each case has an approximately uniform marginal rank distribution, positional peaks indicate that high ranks concentrate in a sequence area; global model-level means do not measure selector strength.

Input and output positions use 20 equal-width bins of normalized segment position. Boundary positions retain exact token ordinals. Liars trailer rows are not positionally aggregated or plotted. The estimator first averages tokens within each case/bin and then averages those case/bin means equally, preventing long transcripts from receiving greater weight. Pointwise 95% intervals use 2,000 deterministic whole-case bootstrap resamples (seed 0); they are descriptive intervals, not simultaneous significance tests. Missing scores and absent case/bin combinations remain missing and are not interpolated or imputed.

Each dataset figure contains one row per available model and one column per available segment. Context-composition strips expose corrected analysis-region and template mixtures without filtering them or overlaying label rates on the percentile axis. All output origins share the visual label `response`; detailed `stored_prefix`, `reconstructed_tail`, and `dataset_original` fractions remain in the summary parquet. The summary parquet retains both raw-source and corrected-analysis fractions together with coverage, extreme-norm, and case-balanced on-task diagnostics for later sensitivity analyses.

## Important cross-model differences

Liars samples are disjoint on-policy samples: the loader filters benchmark rows by generator model before sampling, then probes only transcripts produced by that same model. Gemma-27B uses three benchmark subsets; Llama-70B uses those three plus two harm-pressure subsets. Their case IDs do not overlap, so model comparisons are distributional rather than paired.

The smaller extraction differences are made explicit rather than normalized away: OPI alone has `attn_rollout` and no response; Qwen/Llama use byte-level BPE marker conventions while Gemma uses SentencePiece; chat boundary tokens and segment lengths vary by model; TT/taboo may contain reconstructed greedy tails while Liars responses are original dataset transcripts; and NLA layer/dimension choices are model-specific. Shared case IDs therefore do not imply shared token indices—use model-stratified or segment-normalized positions.

For TT/taboo, `stored_prefix` means the assistant token was in the retained historical prefix (`probe_tok_idx >= 0`). A `reconstructed_tail` is a deterministic greedy continuation generated from a capped stored prefix up to the original response budget. It is reproducible but, for sampled Taboo responses, is not the unavailable historical sampled suffix.

## Commands

Run every Python entry point through the `pao` environment. The generic audit
and plot commands route outputs to the corresponding method folder by default.

Long ensemble commands display tqdm progress automatically. The selector tracks
combinations, candidates, and bootstrap replicates; the builder tracks streamed
token rows; audits track combinations; and plotting tracks datasets.


For ensemble runs, timestamped stage messages state whether the command is
loading, normalizing/scoring candidates, running held-out folds, bootstrapping
a named segment diagnostic, assembling shared selections, validating, or
writing outputs. Nested progress-bar postfixes show the active pool, fold,
segment, budget, or contrast. The builder ETA is based on transformed rows
actually streamed in 100k-row batches. Positional plotting additionally shows
one bootstrap unit per model/dataset combination. ETAs appear after the first
completed unit and become more stable as more comparable units finish.
The documented commands use `conda run --no-capture-output -n pao` so these
updates stream live; omitting that Conda option may buffer them until exit.
### Tests and selectors

```bash
conda run --no-capture-output -n pao python -m pytest token_analysis/tests token_analysis/selection_methods --basetemp token_analysis/.test_tmp -o cache_dir=token_analysis/.pytest_cache
conda run --no-capture-output -n pao python token_analysis/selection_methods/cv_map/select_best_metrics.py --overwrite
conda run --no-capture-output -n pao python token_analysis/selection_methods/cv_map/select_shared_metrics.py --overwrite
conda run --no-capture-output -n pao python token_analysis/selection_methods/spearman/select_metrics.py --overwrite
conda run --no-capture-output -n pao python token_analysis/selection_methods/auroc_ensemble/select_metrics.py --overwrite
```

### Canonical token table

```bash
conda run --no-capture-output -n pao python token_analysis/build_token_position_data.py --overwrite
```

For a limited builder smoke test:

```bash
conda run --no-capture-output -n pao python token_analysis/build_token_position_data.py --datasets tt --models g27 --output token_analysis/.test_tmp/tt_g27.parquet --overwrite
```

### Audits

```bash
conda run --no-capture-output -n pao python token_analysis/audit_bridge_inputs.py
conda run --no-capture-output -n pao python token_analysis/audit_bridge_inputs.py --metric-regime dataset-shared
conda run --no-capture-output -n pao python token_analysis/audit_bridge_inputs.py --metric-regime cv-best
conda run --no-capture-output -n pao python token_analysis/audit_bridge_inputs.py --metric-regime cv-shared
conda run --no-capture-output -n pao python token_analysis/audit_bridge_inputs.py --metric-regime cor-best
conda run --no-capture-output -n pao python token_analysis/audit_bridge_inputs.py --metric-regime cor-shared
conda run --no-capture-output -n pao python token_analysis/audit_bridge_inputs.py --metric-regime auroc-ensemble-best
conda run --no-capture-output -n pao python token_analysis/audit_bridge_inputs.py --metric-regime auroc-ensemble-shared
```

### Position plots

```bash
conda run --no-capture-output -n pao python token_analysis/plot_token_positions.py --overwrite
conda run --no-capture-output -n pao python token_analysis/plot_token_positions.py --metric-regime dataset-shared --overwrite
conda run --no-capture-output -n pao python token_analysis/plot_token_positions.py --metric-regime cv-best --overwrite
conda run --no-capture-output -n pao python token_analysis/plot_token_positions.py --metric-regime cv-shared --overwrite
conda run --no-capture-output -n pao python token_analysis/plot_token_positions.py --metric-regime cor-best --overwrite
conda run --no-capture-output -n pao python token_analysis/plot_token_positions.py --metric-regime cor-shared --overwrite
conda run --no-capture-output -n pao python token_analysis/plot_token_positions.py --metric-regime auroc-ensemble-best --plot-kind both --overwrite
conda run --no-capture-output -n pao python token_analysis/plot_token_positions.py --metric-regime auroc-ensemble-shared --plot-kind both --overwrite
```

### Excluding `head_disagreement` from selection

Every selection and downstream analysis entry point accepts
`--exclude-head-disagreement`. The metric is made ineligible before winner,
runner-up, nested-selection, and bootstrap calculations. Existing outputs are
unchanged. Default result and plot subfolders gain the `_no_hd` suffix, and the
derived token table is `token_position_scores_no_hd.parquet`.

```bash
conda run --no-capture-output -n pao python token_analysis/selection_methods/cv_map/select_best_metrics.py --exclude-head-disagreement
conda run --no-capture-output -n pao python token_analysis/selection_methods/cv_map/select_shared_metrics.py --exclude-head-disagreement
conda run --no-capture-output -n pao python token_analysis/selection_methods/spearman/select_metrics.py --exclude-head-disagreement
conda run --no-capture-output -n pao python token_analysis/build_token_position_data.py --exclude-head-disagreement

for regime in model-best dataset-shared cv-best cv-shared cor-best cor-shared; do
  conda run --no-capture-output -n pao python token_analysis/audit_bridge_inputs.py \
    --metric-regime "$regime" --exclude-head-disagreement
  conda run --no-capture-output -n pao python token_analysis/plot_token_positions.py \
    --metric-regime "$regime" --exclude-head-disagreement
done
```

The pooled-AUROC family has no learned selector script: its published winners
are repository metadata. In the no-HD variant, the alternatives recomputed from
the canonical source parquets are Liars/Gemma-27B `varentropy`, Liars/Llama-70B
`sink_drain`, and dataset-shared Liars `sink_drain`. Other pooled winners are
unchanged. CV and Spearman winners are recomputed by their selector scripts.

After the first materialization, add `--overwrite` when intentionally refreshing
the same no-HD outputs. This still does not touch the original result folders.

The selectors, audits, plots, and builder continue to use only stored parquet
data. They do not run models, judges, NLA generation, or signal recomputation.
