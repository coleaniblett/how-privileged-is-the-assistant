#!/usr/bin/env bash
# Score and generate with each checkpoint in turn, one process per model so GPU memory is fully
# released between them, then run the measure analyses. Extra args go to core.score and
# core.generate, e.g. --preceding-turn.
# Outputs go to results/<run>/ (see AGENTS.md): core/ for the shared scores and samples, one
# directory per measure, extras/ for role_swap and coin_toss.
# The role-swap (data/swap.jsonl) and coin-toss (data/coin_toss.jsonl) scoring run only without
# --preceding-turn: they set their own conversational context.
#   bash scripts/run_gpu.sh
#   RUN=olmo3-7b_preceding bash scripts/run_gpu.sh --preceding-turn   # optional robustness run
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-olmo3-7b}
OUT=${OUT:-results/$RUN}
STAGES=${STAGES:-"base sft dpo instruct"}
VR=$OUT/vestigial_regression

SWAP=1
for a in "$@"; do [[ "$a" == --preceding-turn ]] && SWAP=0; done
[[ -f data/swap.jsonl && -f data/coin_toss.jsonl ]] || SWAP=0

for stage in $STAGES; do
  echo "=== $stage"
  uv run python -m core.score --model "$stage" --out "$OUT/core/scores" "$@"
  uv run python -m core.generate --model "$stage" --out "$OUT/core/generations" "$@"
  if [[ $SWAP == 1 ]]; then
    uv run python -m extras.role_swap score --model "$stage" --out "$OUT/extras/role_swap"
    uv run python -m extras.coin_toss score --model "$stage" --out "$OUT/extras/coin_toss"
  fi
  # weights are ~15 GB per checkpoint; set CLEAR_CACHE=1 to delete each after use
  if [[ "${CLEAR_CACHE:-0}" == 1 ]]; then
    uv run python -c "from core import config; import shutil, huggingface_hub as h; \
      [shutil.rmtree(r.repo_path) for r in h.scan_cache_dir().repos if r.repo_id == config.MODELS['$stage'] and r.repo_id != config.TEMPLATE_MODEL]"
  fi
done
uv run python -m measures.vestigial_regression.flags --generations "$OUT/core/generations" --out "$VR/generations"
uv run python -m measures.vestigial_regression.analyze --scores "$OUT/core/scores" --generations "$VR/generations" --out "$VR"
if [[ $SWAP == 1 ]]; then
  uv run python -m extras.role_swap analyze --scores "$OUT/extras/role_swap" --out-dir "$OUT/extras"
  uv run python -m extras.coin_toss analyze --scores "$OUT/extras/coin_toss" --out-dir "$OUT/extras"
fi
