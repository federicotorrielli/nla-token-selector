#!/bin/bash
# All-token bridge: signals + activations at EVERY token (template included),
# cached NLA explanations / judge labels reused via seed_from_cache, new tokens
# decoded and judged. Per model: extract all kinds -> seed -> AV decode; judge
# for everything at the end. GPU handed off per phase; every phase resumable.
#
# Env knobs: LIMIT (cases per kind, smoke), NLIMIT (decode positions, smoke),
#            MODELS (subset, e.g. "q7"), KINDS (subset, e.g. "taboo tt").
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY=${PY:-python}
# Extraction may use a newer transformers than sglang's pin allows in one env
# (5.14 registers qwen3_asr and sglang 0.5.16 crashes at import). Point
# EXTRACT_PY at a venv holding the latest transformers; everything else runs $PY.
EXTRACT_PY=${EXTRACT_PY:-$PY}
cd "$ROOT" || exit 1
mkdir -p results/bridge results/logs
LIMIT=${LIMIT:-}
NLIMIT=${NLIMIT:-}
MODELS=${MODELS:-"q7 g12 g27 l70"}
KINDS=${KINDS:-"hand taboo tt opi liars"}
INFLIGHT=${INFLIGHT:-256}   # measured plateau; 1024 is no faster

declare -A BASE=( [q7]=Qwen/Qwen2.5-7B-Instruct [g12]=google/gemma-3-12b-it [g27]=google/gemma-3-27b-it [l70]=meta-llama/Llama-3.3-70B-Instruct )
declare -A LAYER=( [q7]=20 [g12]=32 [g27]=41 [l70]=53 )
declare -A DMODEL=( [q7]=3584 [g12]=3840 [g27]=5376 [l70]=8192 )
# No AV repo/port table here: the in-process engine resolves the AV checkpoint
# from configs/base.yaml via Settings, so this script never launches one.
declare -A TAG=( [g27]=gemma-3-27b-it [l70]=llama-v3.3-70b-instruct )
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

kinds_for() {  # liars is on-policy: g27 + l70 only
  local m=$1 out=""
  for k in $KINDS; do
    [ "$k" = liars ] && [ "$m" != g27 ] && [ "$m" != l70 ] && continue
    out="$out $k"
  done
  echo "$out"
}

old_corpus() {  # kind model -> original corpus path (hand has no kind prefix)
  local k=$1 m=$2
  if [ "$k" = hand ]; then echo "results/bridge/${m}_corpus.parquet"
  else echo "results/bridge/${k}_${m}_corpus.parquet"; fi
}
old_file() {  # kind model suffix -> original explanations/ontask path
  local k=$1 m=$2 s=$3
  if [ "$k" = hand ]; then echo "results/bridge/${m}_${s}.parquet"
  else echo "results/bridge/${k}_${m}_${s}.parquet"; fi
}

extract() {  # model kind -> phase-1 all-token corpus
  local M=$1 K=$2
  local CORP=results/bridge/all_${K}_${M}_corpus
  [ -f "$CORP/.done" ] && { echo "  [p1] $K $M cached"; return 0; }
  local EXTRA=""
  case $K in
    opi)   EXTRA="--attn-max-len 4096" ;;
    tt)    EXTRA="--old-corpus $(old_corpus tt "$M")" ;;
    taboo) EXTRA="--old-corpus $(old_corpus taboo "$M") --short $M" ;;
    liars) EXTRA="--datasets ${DSETS[$M]} --model-tag ${TAG[$M]} --per-class 1000 --seed 0" ;;
  esac
  if [ "$K" = tt ] || [ "$K" = taboo ]; then
    [ -f "$(old_corpus "$K" "$M")" ] || { echo "  [p1] $K $M SKIP (no old corpus to rebuild replies from)"; return 1; }
  fi
  echo "  [p1] extract $K $M $(date +%H:%M:%S)"
  # shellcheck disable=SC2086
  $EXTRACT_PY selector/bridge_extract_all.py --kind "$K" \
    --base-model "${BASE[$M]}" --layer "${LAYER[$M]}" --d-model "${DMODEL[$M]}" \
    --out "$CORP" $EXTRA ${LIMIT:+--limit $LIMIT} \
    > "results/logs/all_p1_${K}_${M}.log" 2>&1 \
    || { echo "  [p1] FAIL $K $M"; tail -5 "results/logs/all_p1_${K}_${M}.log"; return 1; }
  tail -1 "results/logs/all_p1_${K}_${M}.log"
}

seed() {  # model kind -> pre-seed explanations/ontask dirs from the old run
  local M=$1 K=$2
  local CORP=results/bridge/all_${K}_${M}_corpus
  local EXPL=results/bridge/all_${K}_${M}_explanations
  local OT=results/bridge/all_${K}_${M}_ontask
  [ -d "$CORP" ] || return 0
  # re-seed unless the cache shard is newer than a COMPLETE extraction
  if [ -f "$EXPL/shard-cache.parquet" ] && [ -f "$CORP/.done" ] \
      && [ "$EXPL/shard-cache.parquet" -nt "$CORP/.done" ]; then
    echo "  [seed] $K $M cached"; return 0
  fi
  $PY selector/seed_from_cache.py --new-corpus "$CORP" \
    --old-corpus "$(old_corpus "$K" "$M")" \
    --old-explanations "$(old_file "$K" "$M" explanations)" \
    --old-ontask "$(old_file "$K" "$M" ontask)" \
    --out-explanations "$EXPL" --out-ontask "$OT" \
    2>&1 | sed 's/^/  [seed] /' | tail -4
}

project() {  # model -> token counts + projected decode volume, the cost checkpoint
  local M=$1
  $PY - "$M" <<'EOF'
import sys
from pathlib import Path
import pyarrow.parquet as pq
m = sys.argv[1]
total = seeded = 0
for d in sorted(Path("results/bridge").glob(f"all_*_{m}_corpus")):
    n = sum(pq.ParquetFile(f).metadata.num_rows for f in d.glob("shard-*.parquet"))
    kind = d.name[len("all_"):-len(f"_{m}_corpus")]
    cache = Path(f"results/bridge/all_{kind}_{m}_explanations/shard-cache.parquet")
    c = pq.ParquetFile(cache).metadata.num_rows if cache.exists() else 0
    total += n; seeded += c
    print(f"  [proj] {kind:6} {m}: {n:>9} tokens, {c:>8} seeded from cache")
todo = total - seeded
print(f"  [proj] TOTAL {m}: {total} tokens, {seeded} seeded -> ~{todo} NLA decodes "
      f"(~{todo * 500 / 1e6:.0f}M generated tokens)")
EOF
}

for M in $MODELS; do
  echo "=== ALL-TOKENS $M $(date +%H:%M:%S) ==="
  for K in $(kinds_for "$M"); do extract "$M" "$K"; done
  for K in $(kinds_for "$M"); do seed "$M" "$K"; done
  project "$M"

  # Decode in-process (--engine): ~2.9x the HTTP server path, measured on q7
  # (3.9 serial / 12.4 concurrent over HTTP vs 21.9 in-process + continuous
  # batching; see selector/bench_engine.py). No AV server to launch or health
  # check — the engine loads the AV itself and shuts down after each kind.
  for K in $(kinds_for "$M"); do
    CORP=results/bridge/all_${K}_${M}_corpus
    [ -d "$CORP" ] || continue
    echo "  [p2] decode $K $M $(date +%H:%M:%S)"
    $PY selector/bridge_run_nla.py --model-short "$M" --corpus "$CORP" \
      --out "results/bridge/all_${K}_${M}_explanations" \
      --engine --inflight "$INFLIGHT" ${NLIMIT:+--limit $NLIMIT} \
      > "results/logs/all_p2_${K}_${M}.log" 2>&1 \
      || { echo "  [p2] decode FAIL $K $M"; tail -6 "results/logs/all_p2_${K}_${M}.log"; }
    pkill -9 -f "sglang" 2>/dev/null; sleep 5
  done
  echo "  [done] $M $(date +%H:%M:%S)"
done

echo "=== ALL_JUDGE $(date +%H:%M:%S) ==="
HF_HUB_ENABLE_HF_TRANSFER=1 nohup bash "$HERE/launch_sglang.sh" --model-path nvidia/DeepSeek-V4-Flash-NVFP4 \
  --port 31000 --host 127.0.0.1 --mem-fraction-static 0.90 --tp 1 \
  > results/logs/all_judge.log 2>&1 &
JPID=$!
if wait_health 31000 "judge" results/logs/all_judge.log; then
  for M in $MODELS; do
    for K in $(kinds_for "$M"); do
      EXPL=results/bridge/all_${K}_${M}_explanations
      [ -d "$EXPL" ] || continue
      echo "  [judge] $K $M $(date +%H:%M:%S)"
      $PY selector/bridge_judge_ontask.py --judge --explanations "$EXPL" \
        --judge-url http://127.0.0.1:31000 --judge-model nvidia/DeepSeek-V4-Flash-NVFP4 \
        --workers 48 --out "results/bridge/all_${K}_${M}_ontask" \
        > "results/logs/all_judge_${K}_${M}.log" 2>&1 \
        || { echo "  [judge] FAIL $K $M"; tail -6 "results/logs/all_judge_${K}_${M}.log"; }
    done
  done
fi
kill -9 "$JPID" 2>/dev/null; pkill -9 -f "sglang.launch_server" 2>/dev/null

echo "=== CONSOLIDATE $(date +%H:%M:%S) ==="
# One canonical per-token parquet per benchmark x model (no activations):
# results/bridge/all_{kind}_{model}.parquet — the reusable artifact.
$PY selector/consolidate_all.py
echo "=== ALL_TOKENS_DONE $(date +%H:%M:%S) ==="
