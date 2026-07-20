#!/bin/bash
# Bridge test on the 27 authored cases, over all four models: for each, extract
# layer-L activations (base model), run the NLA at each token (its own AV server),
# then judge all explanations on-task with DeepSeek. GPU handed off between
# phases; resumable per phase (skips existing parquet).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY=${PY:-python}
cd "$ROOT" || exit 1
mkdir -p results/bridge results/logs

MODELS="q7 g12 g27 l70"
declare -A BASE=( [q7]=Qwen/Qwen2.5-7B-Instruct [g12]=google/gemma-3-12b-it [g27]=google/gemma-3-27b-it [l70]=meta-llama/Llama-3.3-70B-Instruct )
declare -A AVREPO=( [q7]=kitft/nla-qwen2.5-7b-L20-av [g12]=kitft/nla-gemma3-12b-L32-av [g27]=kitft/nla-gemma3-27b-L41-av [l70]=kitft/Llama-3.3-70B-NLA-L53-av )
declare -A LAYER=( [q7]=20 [g12]=32 [g27]=41 [l70]=53 )
declare -A DMODEL=( [q7]=3584 [g12]=3840 [g27]=5376 [l70]=8192 )
declare -A AVPORT=( [q7]=30000 [g12]=30001 [g27]=30002 [l70]=30003 )

wait_health() {  # port label logfile — real failures only, not benign transformers [ERROR] docstring warnings
  for i in $(seq 1 1500); do
    curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && { echo "    $2 ready"; return 0; }
    grep -qiE "traceback|out of memory|cuda error|assertionerror|failed to launch|core dumped|address already in use" "$3" 2>/dev/null \
      && { echo "    $2 errored:"; grep -iE "traceback|out of memory|cuda error|assertionerror|failed to launch|core dumped|address already in use" "$3" | tail -4; return 1; }
    sleep 2
  done
  echo "    $2 never healthy"; return 1
}

for M in $MODELS; do
  echo "=== BRIDGE $M $(date +%H:%M:%S) ==="
  CORP=results/bridge/${M}_corpus.parquet
  EXPL=results/bridge/${M}_explanations.parquet
  # phase 1: extract activations (base model, no server)
  if [ -f "$CORP" ]; then echo "  [p1] $M cached"; else
    echo "  [p1] extract $M $(date +%H:%M:%S)"
    $PY scripts/token_selector/bridge_extract_activations.py \
      --base-model "${BASE[$M]}" --layer "${LAYER[$M]}" --d-model "${DMODEL[$M]}" \
      --out "$CORP" > results/logs/bridge_p1_$M.log 2>&1 \
      || { echo "  [p1] FAIL $M"; tail -5 results/logs/bridge_p1_$M.log; continue; }
  fi
  # phase 2: AV server + NLA decode
  if [ -f "$EXPL" ]; then echo "  [p2] $M cached"; else
    P=${AVPORT[$M]}
    echo "  [p2] AV server $M @ $P $(date +%H:%M:%S)"
    nohup bash "$HERE/launch_sglang.sh" --model-path "${AVREPO[$M]}" --port "$P" --host 127.0.0.1 \
      --mem-fraction-static 0.90 --disable-radix-cache --disable-cuda-graph --grammar-backend xgrammar \
      > results/logs/bridge_av_$M.log 2>&1 &
    AVPID=$!
    if wait_health "$P" "AV:$M" results/logs/bridge_av_$M.log; then
      $PY scripts/token_selector/bridge_run_nla.py --model-short "$M" --corpus "$CORP" \
        --out "$EXPL" --sglang-url "http://127.0.0.1:$P" --batch 16 \
        > results/logs/bridge_p2_$M.log 2>&1 || { echo "  [p2] decode FAIL $M"; tail -6 results/logs/bridge_p2_$M.log; }
    fi
    kill -9 "$AVPID" 2>/dev/null; pkill -9 -f "sglang.launch_server.*${AVREPO[$M]}" 2>/dev/null
    sleep 8
  fi
  echo "  [done] $M $(date +%H:%M:%S)"
done

# phase 3: one judge server, judge every model's explanations
echo "=== JUDGE $(date +%H:%M:%S) ==="
HF_HUB_ENABLE_HF_TRANSFER=1 nohup bash "$HERE/launch_sglang.sh" --model-path nvidia/DeepSeek-V4-Flash-NVFP4 \
  --port 31000 --host 127.0.0.1 --mem-fraction-static 0.90 --tp 1 \
  > results/logs/bridge_judge.log 2>&1 &
JPID=$!
if wait_health 31000 "judge" results/logs/bridge_judge.log; then
  for M in $MODELS; do
    EXPL=results/bridge/${M}_explanations.parquet
    OT=results/bridge/${M}_ontask.parquet
    [ -f "$EXPL" ] || { echo "  no explanations for $M"; continue; }
    if [ -f "$OT" ]; then echo "  [judge] $M cached"; continue; fi
    echo "  [judge] $M $(date +%H:%M:%S)"
    $PY scripts/token_selector/bridge_judge_ontask.py --judge --explanations "$EXPL" \
      --judge-url http://127.0.0.1:31000 --judge-model nvidia/DeepSeek-V4-Flash-NVFP4 \
      --out "$OT" > results/logs/bridge_judge_$M.log 2>&1 || { echo "  [judge] FAIL $M"; tail -6 results/logs/bridge_judge_$M.log; }
  done
fi
kill -9 "$JPID" 2>/dev/null; pkill -9 -f "sglang.launch_server" 2>/dev/null
echo "=== BRIDGE_ALL_DONE $(date +%H:%M:%S) ==="
