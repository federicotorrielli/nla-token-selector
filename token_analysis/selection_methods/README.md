# Metric-selection methods

Each subfolder is a self-contained analysis family with its own results and
plots. Shared source parquets, the canonical token-position table, aggregation
code, and plotting documentation remain one level above to avoid duplication.

| Folder | Selection objective | Variants |
|---|---|---|
| `pooled_auroc/` | Published pooled token AUROC | Per-model best and equal-model dataset-shared |
| `cv_map/` | Held-out case-balanced Average Precision | Per-model best and nested dataset-shared |
| `spearman/` | Absolute pooled Spearman correlation | Per-model best and equal-model dataset-shared |
| `auroc_ensemble/` | Exhaustive pooled-AUROC rank ensembles | Per-model best, dataset-shared, and non-blind OPI sensitivity |

Inside each family, `results/model_best/` and `results/dataset_shared/` contain
tables, audits, reports, and positional summaries. The matching figures are in
`plots/model_best/` and `plots/dataset_shared/`. CV and Spearman additionally
keep their selector implementation, method documentation, and focused tests in
the same family folder.

The AUROC-ensemble family additionally provides grouped held-out diagnostics
and fixed-winner input/boundary/output importance estimates.

See `token_analysis/README.md` for complete commands and
`token_analysis/PLOT_README.md` for figure interpretation.
