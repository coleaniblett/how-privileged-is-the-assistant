#!/usr/bin/env bash
# Score and generate with each checkpoint of a family in turn, one process per model so GPU memory is fully
# released between them, then flag the samples and run the measure analyses. Extra args go to core.score and
# core.generate, e.g. --preceding-turn.
# Outputs go to results/<run>/ (see AGENTS.md): core/ for the shared scores and samples, one
# directory per measure, extras/ for role_swap and coin_toss.
# The role-swap (data/swap.jsonl) and coin-toss (data/coin_toss.jsonl) scoring run only without
# --preceding-turn: they set their own conversational context.
#
#   bash scripts/run_gpu.sh                                  # OLMo 3 7B, legacy layout -> results/olmo3-7b/
#   RUN=olmo3-7b_preceding bash scripts/run_gpu.sh --preceding-turn   # optional robustness run
#   FAMILY=qwen3-8b bash scripts/run_gpu.sh                  # another family -> results/capstone/qwen3-8b/
#   FAMILY=gemma4-31b RUN=capstone bash scripts/run_gpu.sh
#
# With FAMILY set (any config.FAMILIES key): scoring uses the family's settings (config.family_settings: fp32,
# CPU offload for the 31B), generation its precision (int8 for the 31B), and the extras are skipped (they are OLMo-only
# so far). Before each checkpoint is first loaded the script checks free disk against its weights (core.hf_cache check)
# and stops if they do not fit. CLEAR_CACHE=auto (default): after a checkpoint is done, remove it from the HF cache if it is
# larger than 20 GB and was not already cached when this script started; 1: always (same exception); 0: never.
set -euo pipefail
cd "$(dirname "$0")/.."
FAMILY=${FAMILY:-}
if [[ -n "$FAMILY" ]]; then
  RUN=${RUN:-capstone}
  OUT=${OUT:-results/$RUN/$FAMILY}
  FAM=(--family "$FAMILY" --run "$RUN")
  PY_FAM="config.use_family('$FAMILY');"
else
  RUN=${RUN:-olmo3-7b}
  OUT=${OUT:-results/$RUN}
  FAM=()
  PY_FAM=""
fi
STAGES=${STAGES:-$(uv run python -c "from core import config; ${PY_FAM} print(' '.join(config.STAGES))" | tail -1)}
VR=$OUT/vestigial_regression
CLEAR_CACHE=${CLEAR_CACHE:-auto}

SWAP=1
for a in "$@"; do [[ "$a" == --preceding-turn ]] && SWAP=0; done
[[ -f data/swap.jsonl && -f data/coin_toss.jsonl ]] || SWAP=0
[[ -n "$FAMILY" ]] && SWAP=0

repo_of() { uv run python -c "from core import config; ${PY_FAM} print(config.MODELS['$1'])" | tail -1; }
declare -A WAS_CACHED=()
for stage in $STAGES; do
  r=$(repo_of "$stage")
  if uv run python -m core.hf_cache cached "$r" 2>/dev/null; then WAS_CACHED[$r]=1; else WAS_CACHED[$r]=0; fi
done

for stage in $STAGES; do
  repo=$(repo_of "$stage")
  echo "=== $stage ($repo)"
  uv run python -m core.hf_cache check "$repo" || { echo "stopping: not enough disk for $repo (free space or clear the HF cache)"; exit 1; }
  uv run python -m core.score "${FAM[@]}" --model "$stage" --out "$OUT/core/scores" "$@"
  uv run python -m core.generate "${FAM[@]}" --model "$stage" --out "$OUT/core/generations" "$@"
  if [[ $SWAP == 1 ]]; then
    uv run python -m extras.role_swap score --model "$stage" --out "$OUT/extras/role_swap"
    uv run python -m extras.coin_toss score --model "$stage" --out "$OUT/extras/coin_toss"
  fi
  if [[ "$CLEAR_CACHE" != 0 && "${WAS_CACHED[$repo]}" == 0 ]]; then
    size=$(uv run python -c "from core import hf_cache; print(int(hf_cache.weights_gb('$repo')))" | tail -1)
    if [[ "$CLEAR_CACHE" == 1 || "$size" -ge 20 ]]; then uv run python -m core.hf_cache free "$repo"; fi
  fi
done
uv run python -m measures.vestigial_regression.flags "${FAM[@]}" --generations "$OUT/core/generations" --out "$VR/generations"
uv run python -m measures.vestigial_regression.analyze "${FAM[@]}" --scores "$OUT/core/scores" --generations "$VR/generations" --out "$VR"
if [[ $SWAP == 1 ]]; then
  uv run python -m extras.role_swap analyze --scores "$OUT/extras/role_swap" --out-dir "$OUT/extras"
  uv run python -m extras.coin_toss analyze --scores "$OUT/extras/coin_toss" --out-dir "$OUT/extras"
fi
