#!/bin/bash
# Deception bridge on Liars' Bench: per on-policy model, extract NLA-layer
# activations for RESPONSE tokens (+ blind signals + deceptive label), run the NLA
# at each, then judge "is the model being deceptive here?". PER_CLASS lying +
# PER_CLASS honest transcripts per model. GPU handed off per phase; resumable.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY=${PY:-python}
cd "$ROOT" || exit 1
mkdir -p results/bridge results/logs
PER_CLASS=${PER_CLASS:-1000}

MODELS="l70 g27"
declare -A BASE=( [l70]=meta-llama/Llama-3.3-70B-Instruct [g27]=google/gemma-3-27b-it )
declare -A TAG=( [l70]=llama-v3.3-70b-instruct [g27]=gemma-3-27b-it )
declare -A AVREPO=( [l70]=kitft/Llama-3.3-70B-NLA-L53-av [g27]=kitft/nla-gemma3-27b-L41-av )
declare -A LAYER=( [l70]=53 [g27]=41 )
declare -A DMODEL=( [l70]=8192 [g27]=5376 )
declare -A AVPORT=( [l70]=30003 [g27]=30002 )
declare -A DSETS=( [l70]="instructed-deception insider-trading convincing-game harm-pressure-choice harm-pressure-knowledge-report" \
                   [g27]="instructed-deception insider-trading convincing-game" )

wait_health() {  # port label logfile — real failures only, not benign transformers [ERROR] docstring warnings
  for i in $(seq 1 1500); do
    curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && { echo "  $2 ready"; return 0; }
    grep -qiE "traceback|out of memory|cuda error|assertionerror|failed to launch|core dumped|address already in use" "$3" 2>/dev/null \
      && { echo "  $2 errored:"; grep -iE "traceback|out of memory|cuda error|assertionerror|failed to launch|core dumped|address already in use" "$3" | tail -4; return 1; }
    sleep 2
  done
  echo "  $2 never healthy"; return 1
}

for M in $MODELS; do
  echo "=== LIARS $M $(date +%H:%M:%S) ==="
  CORP=results/bridge/liars_${M}_corpus.parquet
  EXPL=results/bridge/liars_${M}_explanations.parquet
  if [ -f "$CORP" ]; then echo "  [p1] $M cached"; else
    echo "  [p1] extract $M $(date +%H:%M:%S)"
    $PY scripts/token_selector/bridge_extract_liars.py \
      --base-model "${BASE[$M]}" --model-tag "${TAG[$M]}" --layer "${LAYER[$M]}" \
      --d-model "${DMODEL[$M]}" --datasets ${DSETS[$M]} --per-class "$PER_CLASS" --tok-cap 30 \
      --out "$CORP" > results/logs/liars_p1_$M.log 2>&1 \
      || { echo "  [p1] FAIL $M"; tail -5 results/logs/liars_p1_$M.log; continue; }
    tail -1 results/logs/liars_p1_$M.log
  fi
  if [ -f "$EXPL" ]; then echo "  [p2] $M cached"; else
    P=${AVPORT[$M]}
    echo "  [p2] AV server $M @ $P $(date +%H:%M:%S)"
    nohup bash "$HERE/launch_sglang.sh" --model-path "${AVREPO[$M]}" --port "$P" --host 127.0.0.1 \
      --mem-fraction-static 0.90 --disable-radix-cache --disable-cuda-graph --grammar-backend xgrammar \
      > results/logs/liars_av_$M.log 2>&1 &
    AVPID=$!
    if wait_health "$P" "AV:$M" results/logs/liars_av_$M.log; then
      $PY scripts/token_selector/bridge_run_nla.py --model-short "$M" --corpus "$CORP" \
        --out "$EXPL" --sglang-url "http://127.0.0.1:$P" --batch 32 \
        > results/logs/liars_p2_$M.log 2>&1 || { echo "  [p2] decode FAIL $M"; tail -6 results/logs/liars_p2_$M.log; }
    fi
    kill -9 "$AVPID" 2>/dev/null; pkill -9 -f "sglang.launch_server" 2>/dev/null; sleep 8
  fi
  echo "  [done] $M $(date +%H:%M:%S)"
done

echo "=== LIARS_JUDGE $(date +%H:%M:%S) ==="
HF_HUB_ENABLE_HF_TRANSFER=1 nohup bash "$HERE/launch_sglang.sh" --model-path nvidia/DeepSeek-V4-Flash-NVFP4 \
  --port 31000 --host 127.0.0.1 --mem-fraction-static 0.90 --tp 1 \
  > results/logs/liars_judge.log 2>&1 &
JPID=$!
if wait_health 31000 "judge" results/logs/liars_judge.log; then
  for M in $MODELS; do
    EXPL=results/bridge/liars_${M}_explanations.parquet
    OT=results/bridge/liars_${M}_ontask.parquet
    [ -f "$EXPL" ] || continue
    [ -f "$OT" ] && { echo "  [judge] $M cached"; continue; }
    echo "  [judge] $M $(date +%H:%M:%S)"
    $PY scripts/token_selector/bridge_judge_ontask.py --judge --explanations "$EXPL" \
      --judge-url http://127.0.0.1:31000 --judge-model nvidia/DeepSeek-V4-Flash-NVFP4 \
      --workers 48 --out "$OT" > results/logs/liars_judge_$M.log 2>&1 || { echo "  [judge] FAIL $M"; tail -6 results/logs/liars_judge_$M.log; }
  done
fi
kill -9 "$JPID" 2>/dev/null; pkill -9 -f "sglang.launch_server" 2>/dev/null
echo "=== LIARS_ALL_DONE $(date +%H:%M:%S) ==="
