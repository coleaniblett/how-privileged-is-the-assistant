#!/usr/bin/env bash
# Confirmatory persona-carryover run on OLMo 3 7B (criteria: measures/overgeneralization/README.md, "Pre-registered criteria").
# REFUSES TO START until the item set is frozen: the caller passes the dataset's sha256 (FROZEN_SHA256=...), which must match the file,
# and the file must be committed and unmodified. Writes only under results/olmo3-7b/overgeneralization/confirmatory/ (the pilot's
# outputs in ../persona_carryover* are never touched). Scoring: bf16, batch 1 (as the pilot's later runs; docs/decisions.md 2026-11-02).
# Stages in order base instruct sft dpo (override with STAGES="base instruct"); one checkpoint on disk at a time; a stage whose score
# CSV exists is skipped. DRY_RUN=1 prints the plan and checks paths without loading a model or taking the GPU lock.
#   FROZEN_SHA256=$(sha256sum data/persona_carryover_confirmatory.jsonl | cut -d' ' -f1) bash scripts/exploratory/run_carryover_confirmatory.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA=data/persona_carryover_confirmatory.jsonl
OUT=results/olmo3-7b/overgeneralization/confirmatory
PY=/opt/arena-env/bin/python
STAGES="${STAGES:-base instruct sft dpo}"
DRY="${DRY_RUN:-0}"
[[ -f "$DATA" ]] || { echo "missing $DATA"; exit 1; }
actual=$(sha256sum "$DATA" | cut -d' ' -f1)
if [[ "$DRY" != "1" ]]; then
  [[ "${FROZEN_SHA256:-}" == "$actual" ]] || { echo "NOT FROZEN: set FROZEN_SHA256 to the sha256 of $DATA ($actual) once the team has confirmed the freeze"; exit 2; }
  git diff --quiet HEAD -- "$DATA" && git ls-files --error-unmatch "$DATA" >/dev/null || { echo "$DATA is not committed/unmodified"; exit 2; }
fi
case "$OUT" in results/olmo3-7b/overgeneralization/confirmatory*) ;; *) echo "unexpected output dir"; exit 3;; esac
HUB="$HOME/.cache/huggingface/hub"
free_model() {
  local repo="models--${1//\//--}"
  rm -rf "$HUB/$repo"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi
  done
}
run() { if [[ "$DRY" == "1" ]]; then echo "[dry] $*"; else "$@"; fi; }
if [[ "$DRY" != "1" ]]; then
  while [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 20; done
  echo "owner=main-session purpose=confirmatory carryover OLMo ($STAGES); released when run_carryover_confirmatory.sh ends" > /tmp/claude-0/gpu.lock
  trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
fi
echo "dataset sha256 $actual; stages: $STAGES; outputs: $OUT"
mkdir -p "$OUT/persona_carryover"
for stage in $STAGES; do
  model=$($PY -c "from core import config; print(config.FAMILIES['olmo3-7b']['stages']['$stage'])")
  if [[ -f "$OUT/persona_carryover/$stage.csv" ]]; then echo "=== $stage: already scored, skipped"; continue; fi
  echo "=== $stage ($model) $(date -u +%T)"
  run $PY -m measures.overgeneralization.persona_carryover score --model "$stage" --stage "$stage" --dataset "$DATA" --out "$OUT/persona_carryover"
  [[ "$DRY" == "1" ]] || free_model "$model"
done
run $PY -m measures.overgeneralization.persona_carryover analyze --dataset "$DATA" --scores "$OUT/persona_carryover" --out-dir "$OUT"
echo "=== done $(date -u +%T)"
