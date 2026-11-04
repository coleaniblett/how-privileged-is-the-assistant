#!/usr/bin/env bash
# Job 2 (2026-11-04): Assistant text in the user slot (extras/exchange_reply_in_user.py), chained after spec H so the GPU never idles.
# Waits for run_mechanism_checks.sh to end, times a 30-exchange OLMo-base sample, projects the total (900 exchanges x 6 checkpoints), and proceeds only if <= 3 h
# (otherwise stops and writes /tmp/claude-0/reply_in_user_STOP.txt for the lead to decide). fp32 with CPU offload (--offload --gpu-gib 20).
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4
PY=/opt/arena-env/bin/python
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
while pgrep -f "[r]un_mechanism_checks.sh" >/dev/null || [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 20; done
echo "owner=main-session purpose=reply-in-user scoring; released when run_reply_in_user.sh ends" > /tmp/claude-0/gpu.lock
trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
OPTS="--offload --gpu-gib 20"
echo "=== timing sample $(date -u +%T)"
s=$(date +%s)
$PY -m extras.exchange_reply_in_user score --family olmo3-7b --model base --limit 30 --out /tmp/claude-0/riu_sample $OPTS --no-reuse || { echo "sample FAILED"; exit 2; }
el=$(( $(date +%s)-s ))
echo "sample: $el s for 30 exchanges incl. load; per exchange ~$((el/30)) s"
# per-exchange cost excludes the load, so this over-estimates for small samples (safe side); 900 exchanges x 6 checkpoints
proj=$(( el*30*6 ))
echo "projected total (upper bound) $((proj/60)) min"
if [ $proj -gt 10800 ]; then echo "projected $((proj/60)) min > 3 h: stopping for a decision" | tee /tmp/claude-0/reply_in_user_STOP.txt; exit 4; fi
for spec in "olmo3-7b base" "olmo3-7b sft" "olmo3-7b dpo" "olmo3-7b instruct" "qwen3-8b base" "qwen3-8b instruct"; do
  set -- $spec; fam=$1; stage=$2
  echo "=== $fam $stage $(date -u +%T)"
  $PY -m extras.exchange_reply_in_user score --family $fam --model $stage $OPTS || echo "FAILED $fam $stage"
  model=$($PY -c "from core import config; print(config.MODELS.get('$stage') if '$fam'=='olmo3-7b' else config.FAMILIES['$fam']['stages']['$stage'])" | tail -1)
  [[ "$model" == */* ]] && free_model "$model"
done
for f in olmo3-7b qwen3-8b; do $PY -m extras.exchange_reply_in_user analyze --family $f; done
echo "=== done $(date -u +%T)"
