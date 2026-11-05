#!/usr/bin/env bash
# Open-opener condition of the vestigial regression judge arm (plan: docs/decisions.md 2026-11-05, measure README "Judge arm,
# open-opener condition"). Per checkpoint: core.generate on data/prefill_open5.jsonl (5 complete-but-open user openers),
# 120 samples each = 600 per checkpoint, the main run's other settings (120 new tokens, T=0.7, seed, --save-ids, the family's
# generation precision) -> results/capstone/<family>/core/generations_open5/<stage>.jsonl. No judge, no analysis.
#
#   FAMILY=olmo3-7b bash scripts/run_open5.sh            # every stage of the family; STAGES="base sft" for a subset
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FAMILY=gemma4-31b bash scripts/run_open5.sh   # Gemma int8 OOMs on 48 GB without it
#
# Each output is skipped if it exists (never overwritten), written to a temp dir first and moved into place once complete.
# CLEAR_CACHE=1 (default) removes a checkpoint from the DEFAULT HF cache after use if this script found it uncached; it is
# forced to 0 when HF_HUB_CACHE points at any other cache (someone else's copy is never deleted).
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
unset VIRTUAL_ENV
: "${FAMILY:?set FAMILY to one of the config.FAMILIES keys}"
PY="uv run --extra quant python"
PREFILL=data/prefill_open5.jsonl
SAMPLES=120
GEN_DIR=results/capstone/$FAMILY/core/generations_open5
CLEAR_CACHE=${CLEAR_CACHE:-1}
DEFAULT_HUB=$(realpath -m "$HOME/.cache/huggingface/hub")
if [[ -n "${HF_HUB_CACHE:-}" && "$(realpath -m "$HF_HUB_CACHE")" != "$DEFAULT_HUB" ]]; then CLEAR_CACHE=0; fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
ts() { date -u '+%Y-%m-%d %H:%M:%S UTC'; }
STAGES=${STAGES:-$($PY -c "from core import config; print(' '.join(config.family_settings('$FAMILY')['stages']))" | tail -1)}
EXPECTED=$(( $(wc -l < "$PREFILL") * SAMPLES ))
mkdir -p "$GEN_DIR"
for stage in $STAGES; do
  out="$GEN_DIR/$stage.jsonl"
  if [[ -f "$out" ]]; then echo "[$(ts)] $FAMILY $stage: $out exists, skipped"; continue; fi
  repo=$($PY -c "from core import config; print(config.family_settings('$FAMILY')['stages']['$stage'])" | tail -1)
  cached=$($PY -c "from core import hf_cache; print(int(hf_cache.is_cached('$repo')))" 2>/dev/null | tail -1 || echo 1)
  echo "[$(ts)] === $FAMILY $stage ($repo), cached $cached"
  $PY -m core.hf_cache check "$repo" || { echo "stopping: not enough disk for $repo"; exit 1; }
  tmp="$GEN_DIR/.tmp_$stage"; rm -rf -- "$tmp"
  $PY -m core.generate --family "$FAMILY" --run capstone --model "$stage" --prefill "$PREFILL" --samples "$SAMPLES" --save-ids --out "$tmp"
  n=$(wc -l < "$tmp/$stage.jsonl")
  if [[ "$n" != "$EXPECTED" ]]; then echo "stopping: $tmp/$stage.jsonl has $n samples, expected $EXPECTED"; exit 1; fi
  mv "$tmp/$stage.jsonl" "$out" && rm -rf -- "$tmp"
  echo "[$(ts)] $FAMILY $stage: $n samples -> $out"
  if [[ "$CLEAR_CACHE" == 1 && "$cached" == 0 ]]; then $PY -m core.hf_cache free "$repo"; fi
done
echo "[$(ts)] $FAMILY done"
