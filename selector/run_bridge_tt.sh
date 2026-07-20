#!/bin/bash
# Tensor Trust injection bridge: build the hijacking cases once (attack vs
# access_code variants), then per model generate a base-model reply, probe the
# response tokens (NLA-layer activation + blind signals + variant label), run the
# NLA at each, and judge "is the model following an injected instruction?". GPU
# handed off per phase; resumable.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY=${PY:-python}
cd "$ROOT" || exit 1
mkdir -p results/bridge results/logs
LIMIT=${LIMIT:-}   # optional: cap source examples (smoke)

CASES=selector/data/tensortrust_cases.jsonl
MODELS="q7 g12 g27 l70"
declare -A BASE=( [q7]=Qwen/Qwen2.5-7B-Instruct [g12]=google/gemma-3-12b-it [g27]=google/gemma-3-27b-it [l70]=meta-llama/Llama-3.3-70B-Instruct )
declare -A AVREPO=( [q7]=kitft/nla-qwen2.5-7b-L20-av [g12]=kitft/nla-gemma3-12b-L32-av [g27]=kitft/nla-gemma3-27b-L41-av [l70]=kitft/Llama-3.3-70B-NLA-L53-av )
declare -A LAYER=( [q7]=20 [g12]=32 [g27]=41 [l70]=53 )
declare -A DMODEL=( [q7]=3584 [g12]=3840 [g27]=5376 [l70]=8192 )
declare -A AVPORT=( [q7]=30000 [g12]=30001 [g27]=30002 [l70]=30003 )

wait_health() {  # port label logfile — real failures only, not benign transformers [ERROR] docstring warnings
  for i in $(seq 1 1500); do
    curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && { echo "  $2 ready"; return 0; }
    grep -qiE "traceback|out of memory|cuda error|assertionerror|failed to launch|core dumped|address already in use" "$3" 2>/dev/null \
      && { echo "  $2 errored:"; grep -iE "traceback|out of memory|cuda error|assertionerror|failed to launch|core dumped|address already in use" "$3" | tail -4; return 1; }
    sleep 2
  done
  echo "  $2 never healthy"; return 1
}

# phase 0: build the cases once
if [ ! -f "$CASES" ]; then
  echo "=== TT build cases $(date +%H:%M:%S) ==="
  $PY selector/data/build_tensortrust_cases.py --out "$CASES" ${LIMIT:+--limit $LIMIT} || exit 1
fi

for M in $MODELS; do
  echo "=== TT $M $(date +%H:%M:%S) ==="
  CORP=results/bridge/tt_${M}_corpus.parquet
  EXPL=results/bridge/tt_${M}_explanations.parquet
  if [ -f "$CORP" ]; then echo "  [p1] $M cached"; else
    echo "  [p1] extract $M $(date +%H:%M:%S)"
    $PY selector/bridge_extract_tt.py \
      --base-model "${BASE[$M]}" --layer "${LAYER[$M]}" --d-model "${DMODEL[$M]}" \
      --cases "$CASES" --out "$CORP" > results/logs/tt_p1_$M.log 2>&1 \
      || { echo "  [p1] FAIL $M"; tail -5 results/logs/tt_p1_$M.log; continue; }
    tail -1 results/logs/tt_p1_$M.log
  fi
  if [ -f "$EXPL" ]; then echo "  [p2] $M cached"; else
    P=${AVPORT[$M]}
    echo "  [p2] AV server $M @ $P $(date +%H:%M:%S)"
    nohup bash "$HERE/launch_sglang.sh" --model-path "${AVREPO[$M]}" --port "$P" --host 127.0.0.1 \
      --mem-fraction-static 0.90 --disable-radix-cache --disable-cuda-graph --grammar-backend xgrammar \
      > results/logs/tt_av_$M.log 2>&1 &
    AVPID=$!
    if wait_health "$P" "AV:$M" results/logs/tt_av_$M.log; then
      $PY selector/bridge_run_nla.py --model-short "$M" --corpus "$CORP" \
        --out "$EXPL" --sglang-url "http://127.0.0.1:$P" --batch 32 \
        > results/logs/tt_p2_$M.log 2>&1 || { echo "  [p2] decode FAIL $M"; tail -6 results/logs/tt_p2_$M.log; }
    fi
    kill -9 "$AVPID" 2>/dev/null; pkill -9 -f "sglang.launch_server" 2>/dev/null; sleep 8
  fi
  echo "  [done] $M $(date +%H:%M:%S)"
done

echo "=== TT_JUDGE $(date +%H:%M:%S) ==="
HF_HUB_ENABLE_HF_TRANSFER=1 nohup bash "$HERE/launch_sglang.sh" --model-path nvidia/DeepSeek-V4-Flash-NVFP4 \
  --port 31000 --host 127.0.0.1 --mem-fraction-static 0.90 --tp 1 \
  > results/logs/tt_judge.log 2>&1 &
JPID=$!
if wait_health 31000 "judge" results/logs/tt_judge.log; then
  for M in $MODELS; do
    EXPL=results/bridge/tt_${M}_explanations.parquet
    OT=results/bridge/tt_${M}_ontask.parquet
    [ -f "$EXPL" ] || continue
    [ -f "$OT" ] && { echo "  [judge] $M cached"; continue; }
    echo "  [judge] $M $(date +%H:%M:%S)"
    $PY selector/bridge_judge_ontask.py --judge --explanations "$EXPL" \
      --judge-url http://127.0.0.1:31000 --judge-model nvidia/DeepSeek-V4-Flash-NVFP4 \
      --workers 48 --out "$OT" > results/logs/tt_judge_$M.log 2>&1 || { echo "  [judge] FAIL $M"; tail -6 results/logs/tt_judge_$M.log; }
  done
fi
kill -9 "$JPID" 2>/dev/null; pkill -9 -f "sglang.launch_server" 2>/dev/null
echo "=== TT_ALL_DONE $(date +%H:%M:%S) ==="
