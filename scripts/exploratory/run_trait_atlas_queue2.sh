#!/usr/bin/env bash
# Spec G trait atlas: parallel checkpoint workers (same materials, seeds, batch size and token cap as run_trait_atlas.sh; only the order and concurrency differ).
#   WORKER=<name> CACHE=<hf hub cache dir for 7-8B checkpoints, e.g. /dev/shm/hf_w1> MEM_FRACTION=0.30 bash scripts/exploratory/run_trait_atlas_queue.sh
# Claims checkpoints in priority order with mkdir claims; 7-8B checkpoints take a SHARED lock (several may run together), Gemma 31B takes an EXCLUSIVE one.
# Each finished checkpoint is freed with its blobs and its non-judge traits are detected at once. google/gemma-4-31B-it stays on disk for the judge
# (removed only to make room for Gemma base, then re-downloaded by the judge). HF_HUB_DISABLE_XET=1.
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python
W="${WORKER:?set WORKER}"; CACHE_SMALL="${CACHE:?set CACHE}"; FRAC="${MEM_FRACTION:-}"
DEFAULT_HUB="$HOME/.cache/huggingface/hub"
KEEP="google/gemma-4-31B-it"
CL=results/overgeneralization/trait_atlas/.claims; LOCK=results/overgeneralization/trait_atlas/.gpu.lock; mkdir -p "$CL" "$CACHE_SMALL" results/overgeneralization/trait_atlas
free_model() { local hub=$2 repo="models--${1//\//--}"; rm -rf "${hub:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$hub/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
ORDER="gemma4-31b:instruct gemma4-31b:base olmo3-7b:sft olmo3-7b:dpo"
for spec in $ORDER; do
  fam=${spec%%:*}; stage=${spec##*:}
  [[ -f "results/overgeneralization/trait_atlas/$fam/${stage}_replies.jsonl.gz" ]] && continue
  # spec I slot: while /root/atlas_logs/HOLD_FOR_SPEC_I exists, OLMo SFT and DPO wait (the file is created only when spec I is given the GPU; Gemma is not held)
  if [ "$fam" = olmo3-7b ]; then while [ -e /root/atlas_logs/HOLD_FOR_SPEC_I ]; do echo "[$W] held for spec I $(date -u +%T)"; sleep 120; done; fi
  mkdir "$CL/${fam}_${stage}" 2>/dev/null || continue            # claimed by another worker
  model=$($PY -c "from core import config; print(config.FAMILIES['$fam']['stages']['$stage'])")
  echo "=== [$W] $fam $stage ($model) $(date -u +%T)"
  if [ "$fam" = gemma4-31b ]; then
    ( flock -x 9
      hub=$DEFAULT_HUB; avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
      if [ "$stage" = base ] && [ "$avail" -lt 64 ]; then echo "[$W] freeing $KEEP for room (re-downloaded by the judge)"; free_model "$KEEP" "$hub"; fi
      HF_HUB_CACHE=$hub $PY -m extras.trait_atlas generate --family $fam --model $stage --batch-size 48 || echo "[$W] FAILED $fam $stage"
      [ "$model" = "$KEEP" ] || free_model "$model" "$hub" ) 9>"$LOCK"
  else
    ( flock -s 9
      HF_HUB_CACHE=$CACHE_SMALL $PY -m extras.trait_atlas generate --family $fam --model $stage --batch-size 24 ${FRAC:+--mem-fraction $FRAC} || echo "[$W] FAILED $fam $stage"
      free_model "$model" "$CACHE_SMALL" ) 9>"$LOCK"
  fi
  [[ -f "results/overgeneralization/trait_atlas/$fam/${stage}_replies.jsonl.gz" ]] && $PY -m extras.trait_atlas detect --only "$fam/$stage"
  echo "=== [$W] $fam $stage done $(date -u +%T)"
done
echo "=== [$W] no more claims $(date -u +%T)"
