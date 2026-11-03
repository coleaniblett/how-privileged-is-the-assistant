#!/usr/bin/env bash
# The main run, one family at a time, one process per task so GPU memory is released in between. Per checkpoint, with the
# weights downloaded once:
#   1. score the exchange set (data/exchanges.jsonl) in the exchange context: user turn as opener, reply after its own user
#      turn, tagged and plain (core.score_exchanges) -> results/capstone/<family>/<stage>/core/exchanges_exchange/
#   2. generate on the incomplete-turn prefills for the judge arm (core.generate, data/prefill_incomplete_200.jsonl, 3 x 120
#      tokens, T=0.7, --save-ids) -> results/capstone/<family>/core/generations_incomplete/<stage>.jsonl
# No judge, no analysis. Runbook: docs/runbook_exchanges_2026-11-03.md.
#
#   FAMILY=olmo3-7b bash scripts/run_exchanges.sh        # base, sft, dpo, instruct
#   FAMILY=qwen3-8b bash scripts/run_exchanges.sh        # base, instruct (think block ON)
#   FAMILY=gemma4-31b bash scripts/run_exchanges.sh      # base, instruct; first 100 exchanges per corpus; int8 generation
#
# Env: FAMILY (required); STAGES (default: all of the family's); GENERATE (1 default, 0: scoring only); SCORE_ARGS (extra
# core.score_exchanges args, e.g. "--batch-size 2"); CLEAR_CACHE as in run_gpu.sh (auto: remove a checkpoint >= 20 GB after
# use if this script found it uncached; 1: always, same exception; 0: never); OMP_NUM_THREADS (default 8). HF_HUB_CACHE is
# passed through; if it points anywhere but the default cache, CLEAR_CACHE is forced to 0, so another person's cache (e.g. a
# copy in /dev/shm) is never deleted. Each task is skipped if its output exists (never overwritten), so a rerun resumes, and a
# checkpoint with nothing left to do is not downloaded.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${FAMILY:?set FAMILY to one of the config.FAMILIES keys}"
CONTEXT=${CONTEXT:-exchange}
if [[ "$CONTEXT" != exchange ]]; then
  echo "the main run is the exchange context only (decided 2026-11-03); CONTEXT=$CONTEXT refused. For another context run"
  echo "core.score_exchanges --context ... directly with its own --tag."
  exit 2
fi
TAG=exchanges_exchange
PY="uv run --extra quant python"  # bitsandbytes for Gemma's int8 generation; the same environment for every family
GENERATE=${GENERATE:-1}
SCORE_ARGS=${SCORE_ARGS:-}
CLEAR_CACHE=${CLEAR_CACHE:-auto}
DEFAULT_HUB=$(realpath -m "$HOME/.cache/huggingface/hub")
if [[ -n "${HF_HUB_CACHE:-}" && "$(realpath -m "$HF_HUB_CACHE")" != "$DEFAULT_HUB" && "$CLEAR_CACHE" != 0 ]]; then
  echo "HF_HUB_CACHE=$HF_HUB_CACHE is not the default cache: CLEAR_CACHE forced to 0 (never delete another cache's files)"
  CLEAR_CACHE=0
fi
# The container's CPU quota is 13.6 CPUs although nproc says 128: cap torch's threads so the run neither starves nor is starved.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
PREFILL=data/prefill_incomplete_200.jsonl
GEN_DIR=results/capstone/$FAMILY/core/generations_incomplete
ts() { date -u '+%Y-%m-%d %H:%M:%S UTC'; }

STAGES=${STAGES:-$($PY -c "from core import config; print(' '.join(config.family_settings('$FAMILY')['stages']))" | tail -1)}
GEN_EXPECTED=$($PY -c "from core import config; print(sum(1 for _ in open('$PREFILL')) * config.GEN_SAMPLES)" | tail -1)
echo "[$(ts)] $FAMILY: stages $STAGES, context $CONTEXT, tag $TAG, generation $GENERATE ($GEN_EXPECTED samples per checkpoint)"
# Preflight: every exchange renders and passes the tagged/plain identity check with this family's tokenizer (no weights).
$PY -m core.score_exchanges --family "$FAMILY" --context "$CONTEXT" --prepare-only $SCORE_ARGS

repo_of() { $PY -c "from core import config; print(config.family_settings('$FAMILY')['stages']['$1'])" | tail -1; }
declare -A WAS_CACHED=()
for stage in $STAGES; do
  r=$(repo_of "$stage")
  if $PY -m core.hf_cache cached "$r" 2>/dev/null; then WAS_CACHED[$r]=1; else WAS_CACHED[$r]=0; fi
done

for stage in $STAGES; do
  repo=$(repo_of "$stage")
  out=$($PY -c "from core import config; print(config.exchange_dir('$FAMILY', '$stage', '$TAG'))" | tail -1)
  need_score=1; [[ -f "$out/scores.csv" ]] && need_score=0
  need_gen=0; [[ "$GENERATE" == 1 && ! -f "$GEN_DIR/$stage.jsonl" ]] && need_gen=1
  if [[ $need_score == 0 && $need_gen == 0 ]]; then echo "[$(ts)] === $stage: scores and generations exist, skipped"; continue; fi
  echo "[$(ts)] === $stage ($repo): score $need_score, generate $need_gen"
  $PY -m core.hf_cache check "$repo" || { echo "stopping: not enough disk for $repo (free space or clear the HF cache)"; exit 1; }
  if [[ $need_score == 1 ]]; then
    $PY -m core.score_exchanges --family "$FAMILY" --model "$stage" --context "$CONTEXT" --tag "$TAG" $SCORE_ARGS
  fi
  if [[ $need_gen == 1 ]]; then
    tmp="$GEN_DIR/.tmp_$stage"
    rm -rf -- "$tmp"  # this script's own leftover from an interrupted attempt, never a result
    $PY -m core.generate --family "$FAMILY" --run capstone --model "$stage" --prefill "$PREFILL" --save-ids --out "$tmp"
    n=$(wc -l < "$tmp/$stage.jsonl")
    if [[ "$n" != "$GEN_EXPECTED" ]]; then echo "stopping: $tmp/$stage.jsonl has $n samples, expected $GEN_EXPECTED"; exit 1; fi
    mv "$tmp/$stage.jsonl" "$GEN_DIR/$stage.jsonl" && rmdir "$tmp"
    echo "[$(ts)] $stage: $n samples -> $GEN_DIR/$stage.jsonl"
  fi
  if [[ "$CLEAR_CACHE" != 0 && "${WAS_CACHED[$repo]}" == 0 ]]; then
    size=$($PY -c "from core import hf_cache; print(int(hf_cache.weights_gb('$repo')))" | tail -1)
    if [[ "$CLEAR_CACHE" == 1 || "$size" -ge 20 ]]; then $PY -m core.hf_cache free "$repo"; fi
  fi
done
$PY -m core.score_exchanges --family "$FAMILY" --tag "$TAG" --check-stages
echo "[$(ts)] $FAMILY done"
