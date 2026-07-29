#!/bin/bash
# Everything that reads the finished all-token run. No GPU, no server, no model:
# it works from the corpus shards (activations) and the consolidated tables.
# Idempotent, so re-run it after any phase catches up.
#
#   bash selector/run_all_token_evals.sh
set -u
PY=${PY:-python}
KINDS=${KINDS:-"hand opi tt liars taboo"}
mkdir -p findings/token-selector results/logs

echo "=== 1. spike statistics from the stored activations $(date +%H:%M:%S) ==="
# spike_mass / peak_ratio / act_norm / resid_jump_masked per position
$PY selector/spike_stats.py

echo "=== 2. consolidate $(date +%H:%M:%S) ==="
# rebuilds all_{kind}_{model}.parquet, now carrying the spike columns
$PY selector/consolidate_all.py

echo "=== 3. reports, whole pool and content only $(date +%H:%M:%S) ==="
# the gap between the two is the size of the chat-template confound
for K in $KINDS; do
  $PY selector/bridge_report.py --kind "$K" --all-tokens || echo "  (skip $K)"
  $PY selector/bridge_report.py --kind "$K" --all-tokens --content-only \
    || echo "  (skip $K content)"
done

echo "=== 4. per-token CSVs $(date +%H:%M:%S) ==="
$PY selector/make_csvs.py --all

echo "=== 5. budget analysis on the full pool $(date +%H:%M:%S) ==="
$PY selector/pool_selector.py --all-tokens --frac 0.10 --tail oracle \
  --out findings/token-selector/bridge-pooling-all.md
$PY selector/pool_selector.py --all-tokens --frac 0.01 --tail oracle \
  --out findings/token-selector/bridge-pooling-all-1pct.md

echo "=== 6. position checks $(date +%H:%M:%S) ==="
# Gemma's local layers have a 1024 window, so past it they cannot see token 0
# and sink_drain drifts upward with position on g12/g27. Also reports on-task
# rate by position, since a signal that merely likes late tokens would score.
$PY selector/position_check.py

echo "=== EVALS DONE $(date +%H:%M:%S) ==="
