# Published pooled-AUROC analyses

This folder contains the original all-token metric selections taken from the
published bridge findings and the companion experiment that fixes one shared
metric across every model in a dataset.

- `results/model_best/`: audit, interpretation, and positional summary for the
  per-model published winner.
- `results/dataset_shared/`: audit and positional summary for the equal-model
  dataset-shared winner.
- `plots/model_best/`: PNG/SVG figures for `model-best`.
- `plots/dataset_shared/`: PNG/SVG figures for `dataset-shared`.

The shared implementations remain at `token_analysis/audit_bridge_inputs.py`
and `token_analysis/plot_token_positions.py`. Run them with `--metric-regime
model-best` or `--metric-regime dataset-shared`; their defaults route outputs
back into this folder.
