#!/bin/bash
# Review experiments on B200 (paper/appendix-experiments-plan.md):
#   E2  OPI with the injection moved: opistart, opimiddle, opihistory
#   E8  all 20 taboo words (+ g27 "adversarial"): taboo20; misaligned q7: misalign
#   E7  audit cost: bench_audit_cost.py + bench_engine.py per model
# Same phases as run_bridge_all_tokens.sh: extract -> spike -> decode -> judge ->
# consolidate. Every phase skips finished work, so a rerun resumes.
#
# Env knobs: MODELS, KINDS, NGPU, LIMIT (cases per kind, smoke), NLIMIT (decode
#            positions, smoke), SKIP_E7=1, SKIP_JUDGE=1.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY=${PY:-$HOME/envs/review-sglang/bin/python}   # sglang 0.5.20 and its pins
EXTRACT_PY=${EXTRACT_PY:-$PY}
# The venv ships nvcc as a wheel (nvidia/cu13); kernel JIT needs it as CUDA_HOME.
CU=$($PY -c "import nvidia.cu13 as m; print(list(m.__path__)[0])")
# The venv is never activated, so put its bin (ninja for kernel JIT) on PATH too.
export CUDA_HOME=$CU CUDA_PATH=$CU PATH=$(dirname "$PY"):$CU/bin:$PATH
export LD_LIBRARY_PATH=$CU/lib:${LD_LIBRARY_PATH:-}
cd "$ROOT" || exit 1
mkdir -p results/bridge results/logs results/review
# Every model is cached; Hub checks hung for ~50 min on the compute host.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
LIMIT=${LIMIT:-}
NLIMIT=${NLIMIT:-}
MODELS=${MODELS:-"q7 g12 g27 l70"}
KINDS=${KINDS:-"opistart opimiddle opihistory taboo20 misalign"}
INFLIGHT=${INFLIGHT:-256}
NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
[ "${NGPU:-0}" -ge 1 ] || NGPU=1

declare -A BASE=( [q7]=Qwen/Qwen2.5-7B-Instruct [g12]=google/gemma-3-12b-it [g27]=google/gemma-3-27b-it [l70]=meta-llama/Llama-3.3-70B-Instruct )
declare -A LAYER=( [q7]=20 [g12]=32 [g27]=41 [l70]=53 )
declare -A DMODEL=( [q7]=3584 [g12]=3840 [g27]=5376 [l70]=8192 )
declare -A WORDS=( [q7]="" [g12]="" [g27]="adversarial" [l70]="" )
# AV engines per GPU. One engine's scheduler thread (it rebuilds each request's
# embedding matrix from a Python list) caps a GPU at ~15% utilization, so small
# AVs run several engines side by side. AV weights: 15, 24, 54, 132 GB of 180.
declare -A EPG=( [q7]=4 [g12]=4 [g27]=2 [l70]=1 )
# Token ids + one override row instead of the full embedding matrix: ~5x per
# engine, same explanations (256/256 on a q7 check). Needs embedding scale 1,
# so Gemma-3 (scale sqrt(d)) keeps the full-matrix path.
declare -A OVR=( [q7]=--embed-override [l70]=--embed-override )
# GPU memory the engines of one GPU share. 0.80 leaves room for four engines'
# runtime buffers; l70 runs alone on its GPU, and its 132 GB of weights leave
# only ~14 GB of KV cache at 0.80, so it takes 0.90 (the original default).
declare -A MEMTOT=( [q7]=0.80 [g12]=0.80 [g27]=0.80 [l70]=0.90 )
ALL_WORDS="blue book chair clock cloud dance flag flame gold green jump leaf moon rock salt ship smile snow song wave"

wait_health() {  # port label logfile
  for i in $(seq 1 1500); do
    curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && { echo "  $2 ready"; return 0; }
    grep -qiE "traceback|out of memory|cuda error|failed to launch|address already in use" "$3" 2>/dev/null \
      && { echo "  $2 errored:"; tail -4 "$3"; return 1; }
    sleep 2
  done
  echo "  $2 never healthy"; return 1
}

kinds_for() {  # misalign exists for q7 only
  local m=$1 out=""
  for k in $KINDS; do
    [ "$k" = misalign ] && [ "$m" != q7 ] && continue
    out="$out $k"
  done
  echo "$out"
}

extract_args() {  # kind model -> extractor arguments
  case $1 in
    opistart)   echo "--kind opi --cases selector/data/injection_cases_start.jsonl --attn-max-len 4096" ;;
    opimiddle)  echo "--kind opi --cases selector/data/injection_cases_middle.jsonl --attn-max-len 4096" ;;
    opihistory) echo "--kind opi --cases selector/data/injection_cases_history.jsonl --attn-max-len 4096" ;;
    taboo20)    echo "--kind taboo --short $2 --words $ALL_WORDS ${WORDS[$2]} --samples 8" ;;
    misalign)   echo "--kind misalign --short $2 --samples 8 --max-new-tokens 256" ;;
  esac
}

extract() {  # model kind gpu
  local M=$1 K=$2 G=$3
  local CORP=results/bridge/all_${K}_${M}_corpus
  [ -f "$CORP/.done" ] && { echo "  [p1] $K $M cached"; return 0; }
  echo "  [p1] extract $K $M on GPU $G $(date +%H:%M:%S)"
  # shellcheck disable=SC2046
  CUDA_VISIBLE_DEVICES=$G $EXTRACT_PY selector/bridge_extract_all.py \
    --base-model "${BASE[$M]}" --layer "${LAYER[$M]}" --d-model "${DMODEL[$M]}" \
    --out "$CORP" $(extract_args "$K" "$M") ${LIMIT:+--limit $LIMIT} \
    > "results/logs/review_p1_${K}_${M}.log" 2>&1 \
    || { echo "  [p1] FAIL $K $M"; tail -5 "results/logs/review_p1_${K}_${M}.log"; return 1; }
}

echo "=== E2/E8 EXTRACT on ${NGPU} GPU(s) $(date +%H:%M:%S) ==="
# A shared job queue: each GPU pulls the next (model, kind) as soon as it is free,
# so one slow job never idles the other GPUs. Every model fits on one B200.
QUEUE=results/logs/review_extract_queue.txt
for M in $MODELS; do for K in $(kinds_for "$M"); do echo "$M:$K"; done; done > "$QUEUE"
next_job() {  # pop the first line of the queue under a lock
  flock "$QUEUE.lock" sh -c 'head -n1 "$1"; sed -i 1d "$1"' _ "$QUEUE"
}
worker() {
  local g=$1 job M K
  while job=$(next_job) && [ -n "$job" ]; do
    IFS=: read -r M K <<< "$job"
    extract "$M" "$K" "$g"
  done
}
PIDS=()
for ((g = 0; g < NGPU; g++)); do worker "$g" & PIDS+=($!); done
for p in "${PIDS[@]}"; do wait "$p"; done

echo "=== SPIKE COLUMNS $(date +%H:%M:%S) ==="
# shellcheck disable=SC2086
$PY selector/all_tokens_eval.py spike --kinds $KINDS > results/logs/review_spike.log 2>&1 \
  || { echo "  spike FAIL"; tail -5 results/logs/review_spike.log; }

# Pass 2 picks up the positions of any engine that died in pass 1; an engine with
# nothing left exits before loading its model, so a clean pass 2 costs seconds.
for PASS in 1 2; do
echo "=== DECODE pass $PASS $(date +%H:%M:%S) ==="
for M in $MODELS; do
  for K in $(kinds_for "$M"); do
    CORP=results/bridge/all_${K}_${M}_corpus
    # a LIMIT smoke corpus never gets .done, but should still be decoded
    [ -f "$CORP/.done" ] || { [ -n "$LIMIT" ] && [ -d "$CORP" ]; } || continue
    E=${EPG[$M]}
    W=$((NGPU * E))
    echo "  [p2] decode $K $M on ${NGPU} GPU(s) x ${E} engine(s) $(date +%H:%M:%S)"
    PIDS=()
    # Engines sharing a GPU must load one after another: SGLang sizes its KV
    # cache from the free memory it sees before and after its own weight load,
    # and a neighbour loading at the same time makes that check fail. So start
    # engine j on every GPU, wait until each has begun decoding, then start j+1.
    # --mem-fraction is a share of the memory FREE when an engine starts, so the
    # j-th engine takes 0.80/(E-j) of what is left: every engine gets 0.80/E.
    # (0.88 left too little for runtime activations: four engines hit CUDA OOM.)
    for ((j = 0; j < E; j++)); do
      MEM=$(awk -v t="${MEMTOT[$M]}" -v e="$E" -v j="$j" 'BEGIN { printf "%.3f", t / (e - j) }')
      LOGS=()
      for ((g = 0; g < NGPU; g++)); do
        w=$((j * NGPU + g))
        LOG=results/logs/review_p2_${K}_${M}_w$w.log
        CUDA_VISIBLE_DEVICES=$g $PY selector/bridge_run_nla.py --model-short "$M" \
          --corpus "$CORP" --out "results/bridge/all_${K}_${M}_explanations" \
          --engine --inflight "$INFLIGHT" --mem-fraction "$MEM" --row-partition ${OVR[$M]:-} \
          --shard-stride "$W" --shard-offset "$w" ${NLIMIT:+--limit $NLIMIT} \
          > "$LOG" 2>&1 &
        PIDS+=($!)
        LOGS+=("$LOG:$!")
      done
      for lp in "${LOGS[@]}"; do  # ready = decoding, done, or exited
        LOG=${lp%%:*}; P=${lp##*:}
        until grep -qE "to run, inflight|nothing to decode" "$LOG" 2>/dev/null \
            || ! kill -0 "$P" 2>/dev/null; do sleep 5; done
      done
    done
    for p in "${PIDS[@]}"; do wait "$p" || echo "  [p2] worker FAIL $K $M"; done
    sleep 5
  done
done
done

echo "=== JUDGE (first evaluator, DeepSeek-V4-Flash) $(date +%H:%M:%S) ==="
JPIDS=()
READY=0
[ -n "${SKIP_JUDGE:-}" ] && NJ=0 || NJ=$NGPU
for ((g = 0; g < NJ; g++)); do
  CUDA_VISIBLE_DEVICES=$g nohup "$PY" -m sglang.launch_server \
    --model-path nvidia/DeepSeek-V4-Flash-NVFP4 \
    --port $((31000 + g)) --host 127.0.0.1 --mem-fraction-static 0.90 --tp 1 \
    > "results/logs/review_judge_server_g$g.log" 2>&1 &
  JPIDS+=($!)
done
for ((g = 0; g < NJ; g++)); do
  wait_health $((31000 + g)) "judge:g$g" "results/logs/review_judge_server_g$g.log" \
    && READY=$((READY + 1))
done
if [ "$READY" -gt 0 ]; then
  for M in $MODELS; do
    for K in $(kinds_for "$M"); do
      EXPL=results/bridge/all_${K}_${M}_explanations
      [ -d "$EXPL" ] || continue
      echo "  [judge] $K $M $(date +%H:%M:%S)"
      PIDS=()
      for ((g = 0; g < READY; g++)); do
        $PY selector/bridge_judge_ontask.py --judge --explanations "$EXPL" \
          --judge-url "http://127.0.0.1:$((31000 + g))" \
          --judge-model nvidia/DeepSeek-V4-Flash-NVFP4 \
          --workers 48 --out "results/bridge/all_${K}_${M}_ontask" \
          --shard-stride "$READY" --shard-offset "$g" \
          > "results/logs/review_judge_${K}_${M}_g$g.log" 2>&1 &
        PIDS+=($!)
      done
      for p in "${PIDS[@]}"; do wait "$p" || echo "  [judge] worker FAIL $K $M"; done
    done
  done
fi
for p in "${JPIDS[@]}"; do kill -9 "$p" 2>/dev/null; done
pkill -9 -f "sglang.launch_server" 2>/dev/null

echo "=== CONSOLIDATE $(date +%H:%M:%S) ==="
for K in $KINDS; do $PY selector/consolidate_all.py "$K"; done

if [ -z "${SKIP_E7:-}" ]; then
  echo "=== E7 AUDIT COST $(date +%H:%M:%S) ==="
  for M in $MODELS; do
    CUDA_VISIBLE_DEVICES=0 $PY selector/bench_audit_cost.py --short "$M" --n 50 \
      --out "results/review/e7_${M}.json" > "results/logs/review_e7_${M}.log" 2>&1 \
      || echo "  [e7] cost FAIL $M"
    CUDA_VISIBLE_DEVICES=0 $PY selector/bench_engine.py --model-short "$M" \
      --corpus "results/bridge/all_opistart_${M}_corpus" --n 4096 --continuous 256 \
      > "results/logs/review_e7_engine_${M}.log" 2>&1 || echo "  [e7] engine FAIL $M"
    tail -2 "results/logs/review_e7_engine_${M}.log"
  done
fi
echo "=== REVIEW_B200_DONE $(date +%H:%M:%S) ==="
