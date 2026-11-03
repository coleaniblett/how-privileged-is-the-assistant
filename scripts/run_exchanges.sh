#!/usr/bin/env bash
# Score the exchange set (data/exchanges.jsonl, core.exchanges) with every checkpoint of one family, one process per
# checkpoint, into results/capstone/<family>/<stage>/core/<tag>/ (core.score_exchanges). Scoring only: no generation,
# no judge, no analysis. Runbook: docs/runbook_exchanges_2026-11-03.md.
#
#   FAMILY=olmo3-7b bash scripts/run_exchanges.sh                    # base, sft, dpo, instruct; context "exchange"
#   FAMILY=qwen3-8b CONTEXT=single bash scripts/run_exchanges.sh     # the reply alone (overnight design), tag exchanges_single
#   FAMILY=gemma4-31b bash scripts/run_exchanges.sh                  # first 100 per corpus (config exchange_n_per_corpus)
#
# Env: FAMILY (required), STAGES (default: all of the family's), CONTEXT (default: config.EXCHANGE_CONTEXT), TAG (default:
# exchanges_<context>), SCORE_ARGS (extra core.score_exchanges args, e.g. "--no-reuse" or "--batch-size 2"), CLEAR_CACHE
# as in run_gpu.sh (auto: remove a checkpoint >= 20 GB after use if this script found it uncached; 1: always, same
# exception; 0: never). HF_HUB_CACHE is passed through (e.g. a RAM-backed cache for Gemma). A stage whose output
# directory already holds scores is skipped (outputs are never overwritten), so a rerun resumes.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${FAMILY:?set FAMILY to one of the config.FAMILIES keys}"
PY="uv run python"
STAGES=${STAGES:-$($PY -c "from core import config; print(' '.join(config.family_settings('$FAMILY')['stages']))" | tail -1)}
CONTEXT=${CONTEXT:-$($PY -c "from core import config; print(config.EXCHANGE_CONTEXT)" | tail -1)}
TAG=${TAG:-exchanges_$CONTEXT}
SCORE_ARGS=${SCORE_ARGS:-}
CLEAR_CACHE=${CLEAR_CACHE:-auto}
ts() { date -u '+%Y-%m-%d %H:%M:%S UTC'; }

echo "[$(ts)] $FAMILY: stages $STAGES, context $CONTEXT, tag $TAG"
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
  if [[ -f "$out/scores.csv" ]]; then echo "[$(ts)] === $stage: $out/scores.csv exists, skipped"; continue; fi
  echo "[$(ts)] === $stage ($repo) -> $out"
  $PY -m core.hf_cache check "$repo" || { echo "stopping: not enough disk for $repo (free space or clear the HF cache)"; exit 1; }
  $PY -m core.score_exchanges --family "$FAMILY" --model "$stage" --context "$CONTEXT" --tag "$TAG" $SCORE_ARGS
  if [[ "$CLEAR_CACHE" != 0 && "${WAS_CACHED[$repo]}" == 0 ]]; then
    size=$($PY -c "from core import hf_cache; print(int(hf_cache.weights_gb('$repo')))" | tail -1)
    if [[ "$CLEAR_CACHE" == 1 || "$size" -ge 20 ]]; then $PY -m core.hf_cache free "$repo"; fi
  fi
done
$PY -m core.score_exchanges --family "$FAMILY" --tag "$TAG" --check-stages
echo "[$(ts)] $FAMILY done"
