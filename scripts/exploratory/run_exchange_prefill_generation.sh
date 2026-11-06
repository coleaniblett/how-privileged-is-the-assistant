#!/usr/bin/env bash
# Job 1 (2026-11-04): incomplete-turn generations on the exchange set's oasst1 and UltraChat prefills (data/prefill_incomplete_exchanges.jsonl, 400 prefills).
# Exactly the settings core.generate used for generations_incomplete (defaults: 3 samples, 120 new tokens, temperature 0.7, top_p 1, bf16, --save-ids).
# OLMo base, SFT, DPO, instruct, then Qwen base, instruct; one checkpoint at a time, each freed with its blobs. Gemma 4 31B is left for vinyl (after the atlas Gemma stages).
# Output: results/capstone/<family>/core/generations_incomplete_exchanges/<stage>.jsonl (not committed). No judging here (needs an API key and Cole's judge.py, prompt v2 unchanged).
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_DISABLE_XET=1
PY=/opt/arena-env/bin/python
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
while [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 30; done
echo "owner=main-session purpose=exchange prefill generations; released when run_exchange_prefill_generation.sh ends" > /tmp/claude-0/gpu.lock
trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
for spec in "olmo3-7b base" "olmo3-7b sft" "olmo3-7b dpo" "olmo3-7b instruct" "qwen3-8b base" "qwen3-8b instruct"; do
  set -- $spec; fam=$1; stage=$2
  out="results/capstone/$fam/core/generations_incomplete_exchanges"
  [[ -f "$out/$stage.jsonl" ]] && { echo "=== $fam $stage: done, skipped"; continue; }
  model=$($PY -c "from core import config; print(config.FAMILIES['$fam']['stages']['$stage'])")
  avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$avail" -ge 20 ] || { echo "less than 20 GB free for $model"; exit 3; }
  echo "=== $fam $stage ($model) $(date -u +%T)"
  $PY -m core.generate --family $fam --run capstone --model $stage --prefill data/prefill_incomplete_exchanges.jsonl --save-ids --out "$out" || echo "FAILED $fam $stage"
  free_model "$model"
done
echo "=== done $(date -u +%T)"
