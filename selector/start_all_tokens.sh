#!/bin/bash
# Start (or re-attach to) the all-token bridge run in a detached tmux session on
# the compute host. tmux keeps the job alive across SSH drops, which plain
# nohup/setsid over a closing ssh channel did not do reliably.
#
#   bash selector/start_all_tokens.sh          # start if not already running
#   tmux attach -t nla                         # watch it
#   tmux kill-session -t nla                   # stop it (resumable)
set -u
SESSION=${SESSION:-nla}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
ENVNAME=${ENVNAME:-pao}
EXTRACT_PY=${EXTRACT_PY:-/work/envs/hf-latest/bin/python}

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already running:"
  tmux list-panes -t "$SESSION" -F "  pane #{pane_pid} #{pane_current_command}"
  exit 0
fi

tmux new-session -d -s "$SESSION" -c "$ROOT" \
  "source '$CONDA_SH' && conda activate '$ENVNAME' && \
   EXTRACT_PY='$EXTRACT_PY' ${NGPU:+NGPU=$NGPU} \
   bash selector/run_bridge_all_tokens.sh \
   > results/logs/all_tokens_full.log 2>&1"
sleep 3
tmux has-session -t "$SESSION" 2>/dev/null \
  && echo "started tmux session '$SESSION' (log: results/logs/all_tokens_full.log)" \
  || { echo "FAILED to start"; exit 1; }
