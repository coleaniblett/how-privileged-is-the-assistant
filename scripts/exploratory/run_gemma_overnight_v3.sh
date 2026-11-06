#!/usr/bin/env bash
# Vinyl Gemma 4 31B chain, v3 + J12 strict-cut re-judge before the instruct weights are freed (replaces run_gemma_overnight.sh, whose running step it takes over: the old script's GPU lock is held by that step's process, so this waits on it).
# Adds: J11 on Gemma, an explicit disk hand-over (side jobs paused with DISK_HOLD while the instruct weights are freed and the base weights downloaded), and NO silent skip:
# if 70 GB cannot be made free the chain writes /root/atlas_logs/GEMMA_BASE_STOP.txt and exits non-zero.
set -uo pipefail
cd /root/hpa-atlas2
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
exec 9>results/trait_atlas/.gpu.lock
flock -x 9
FAM=gemma4-31b
step() { local t=$(date +%s); "$@" || echo "FAILED: $*"; echo "   [$(( $(date +%s)-t )) s] $*" | cut -c1-200; }
phase() {
  local stage=$1
  echo "=== $FAM $stage $(date -u +%T)"
  out=results/capstone/$FAM/core/generations_incomplete_exchanges
  [[ -f $out/$stage.jsonl ]] || step $PY -m extras.batched_generate --family $FAM --run capstone --model $stage --prefill data/prefill_incomplete_exchanges.jsonl --out $out
  [[ -f results/capstone/$FAM/$stage/core/exchanges_reply_in_user/scores.csv ]] || step $PY -m extras.exchange_reply_in_user score --family $FAM --model $stage --offload --gpu-gib 30
  [[ -f results/mechanism_checks/h2/$FAM/$stage.jsonl.gz ]] || step $PY -m extras.mechanism_checks h2 --family $FAM --stage $stage
  [[ -f results/mechanism_checks/h3/$FAM/$stage.jsonl.gz ]] || step $PY -m extras.mechanism_checks h3 --family $FAM --stage $stage
  [[ -f results/skill_roleplay/$FAM/$stage.csv.gz ]] || step $PY -m extras.skill_roleplay run --family $FAM --stage $stage
}
phase instruct
# J12: the strict-cut re-judge needs the Gemma instruct judge, which is cached now (about 340 replies); it must run before the instruct weights are freed below
[[ -f results/trait_atlas_strict/qwen3-8b/instruct_judged.csv.gz ]] || step $PY -m extras.trait_atlas_strict judge
# ---- disk hand-over: pause side jobs, wait for them to drain, free instruct, make sure 70 GB are free, download base, release the hold
touch /root/atlas_logs/DISK_HOLD
while pgrep -f "[s]kill_roleplay run" >/dev/null || pgrep -f "[s]econd_judge run" >/dev/null; do sleep 20; done
free_model "$($PY -c "from core import config; print(config.FAMILIES['$FAM']['stages']['instruct'])")"
for i in $(seq 1 60); do
  avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$avail" -ge 70 ] && break; sleep 30
done
avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
if [ "$avail" -lt 70 ]; then echo "GEMMA BASE NOT STARTED: only ${avail} GB free at $(date -u +%T)" | tee /root/atlas_logs/GEMMA_BASE_STOP.txt; rm -f /root/atlas_logs/DISK_HOLD; exit 4; fi
$PY - <<PYEOF || { echo "base download failed" | tee /root/atlas_logs/GEMMA_BASE_STOP.txt; rm -f /root/atlas_logs/DISK_HOLD; exit 5; }
from core import config
from huggingface_hub import snapshot_download
print(snapshot_download(config.FAMILIES["$FAM"]["stages"]["base"]))
PYEOF
rm -f /root/atlas_logs/DISK_HOLD
phase base
free_model "$($PY -c "from core import config; print(config.FAMILIES['$FAM']['stages']['base'])")"
step $PY -m extras.exchange_reply_in_user analyze --family $FAM
step $PY -m extras.skill_roleplay analyze
echo "=== done $(date -u +%T)"
