# Exhaustive AUROC rank ensembles

This method evaluates all 13 deployable source metrics and every unordered
two-metric convex combination at weights 25/75, 50/50, and 75/25. The 13
originals and 234 ensembles compete in the same final pool. It produces both a
model-specific winner and one candidate shared by all models in each dataset.

Read [docs/auroc_ensemble_explainer.md](docs/auroc_ensemble_explainer.md) for
the statistical definition, held-out validation, missingness policy, and
interpretation of the input/boundary/output analysis.

## Commands
The examples use Conda's `--no-capture-output` option so tqdm bars and
timestamped stage messages are emitted while the process runs instead of being
buffered until completion.

Run all commands from the repository root through the pao environment.

Progress reporting is automatic. Selection shows an overall model/dataset bar
plus candidate-scoring and whole-case-bootstrap bars for the active
combination. The canonical builder reports actual token rows streamed to disk;
ensemble audits report completed combinations; and both plot modes report
completed datasets. Standard tqdm output includes elapsed time, throughput,
and ETA once at least one unit has completed.

Timestamped `[auroc-ensemble]` milestones identify source loading, pooled-rank
normalization, all-token and common-finite candidate scoring, independent
segment selection, grouped held-out folds, fixed-winner segment diagnostics,
dataset-shared selection, synchronized shared-case validation, leave-one-model-out
validation, output writing, and total elapsed time. The nested
candidate bar names its active pool or fold. The nested bootstrap bar names the
active held-out delta, segment AUROC/rank estimate, top-budget concentration,
or segment contrast. Its total includes all of those replicate groups, so its
ETA covers the remaining bootstrap work for the active model. The outer
combination and shared-dataset bars estimate the remaining run after at least
one comparable unit finishes.

    conda run --no-capture-output -n pao python token_analysis/selection_methods/auroc_ensemble/select_metrics.py

Explicitly non-blind OPI sensitivity, which admits attn_rollout:

    conda run --no-capture-output -n pao python token_analysis/selection_methods/auroc_ensemble/select_metrics.py \
      --datasets opi --include-attn-rollout

After selection, rebuild the canonical positional table:

    conda run --no-capture-output -n pao python token_analysis/build_token_position_data.py --overwrite

Run the two full audits:

    conda run --no-capture-output -n pao python token_analysis/audit_bridge_inputs.py \
      --metric-regime auroc-ensemble-best
    conda run --no-capture-output -n pao python token_analysis/audit_bridge_inputs.py \
      --metric-regime auroc-ensemble-shared

Render positional profiles and segment comparisons:

    conda run --no-capture-output -n pao python token_analysis/plot_token_positions.py \
      --metric-regime auroc-ensemble-best --plot-kind both --overwrite
    conda run --no-capture-output -n pao python token_analysis/plot_token_positions.py \
      --metric-regime auroc-ensemble-shared --plot-kind both --overwrite

Every ensemble positional and segment figure annotates only the model-specific
full-data pooled value (`AUROC`) and grouped out-of-fold pooled value
(`held-out AUROC`). Shared equal-model and case-macro validation estimates
remain in the result artifacts rather than the plot labels.

Use --bootstrap-resamples 20 and an output path under
token_analysis/.test_tmp/ for a quick smoke run. The paper outputs use 2,000
whole-case bootstrap replicates.

## Output layout

results/model_best/ contains the 3,458 primary per-model candidate rows, 14
selected rows, 38 segment-specific selections, 38 fixed-winner segment
estimates, 34 matched-segment contrasts, and grouped-fold diagnostics.

results/dataset_shared/ contains 988 equal-model candidate rows, a 14-row
model-expanded shared manifest, 11 dataset-segment selections, 38 fixed-winner
segment estimates, and 34 contrasts. It also contains 140 synchronized
shared-fold rows, 280 leave-one-model-out fold rows, a 56-row normalized LOMO
summary, and case-level validation diagnostics. The counts assume five folds
and the complete primary dataset/model inventory.

plots/model_best/ and plots/dataset_shared/ contain four positional and four
segment-comparison figures each, in both PNG and SVG.

opi_attn_rollout_sensitivity/ is separate and explicitly non-blind. Its pool
has 14 originals and 273 ensembles, or 287 candidates for each of four OPI
models.

The selector never filters template tokens, extreme-norm tokens, reconstructed
response tails, dense Tensor Trust tokens, or Gemma positions past 1,024.
Liars trailer rows remain in the canonical all-token table but are never
tracked by segment summaries or plots.

## Tests

    conda run --no-capture-output -n pao python -m pytest \
      token_analysis/selection_methods/auroc_ensemble/tests \
      token_analysis/tests \
      --basetemp token_analysis/.test_tmp/pytest \
      -o cache_dir=token_analysis/.pytest_cache

The full-data winners and plots are exploratory descriptions. Model-specific
and synchronized dataset-shared grouped holdouts estimate generalization to new
cases. Label-free and target-calibrated LOMO results separately estimate
cross-model transfer. None establishes that verbalizing a high-scoring token or
segment causally improves NLA output.
