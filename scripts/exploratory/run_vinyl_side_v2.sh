#!/usr/bin/env bash
# Vinyl side jobs v2 (J11 for OLMo and Qwen; J6 is done). Waits while /root/atlas_logs/DISK_HOLD exists (Gemma base hand-over), never starts a job without enough disk
# (cached models excepted) and retries skipped jobs once the Gemma chain has finished and freed its weights. Log: /root/atlas_logs/vinyl_side_v2.log
set -uo pipefail
cd /root/hpa-atlas2
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
while pgrep -f "[r]un_vinyl_side.sh" >/dev/null; do sleep 20; done
exec 8>/root/atlas_logs/.side.lock
flock -x 8
for pass in 1 2 3 4; do
  skipped=0
  for fam in olmo3-7b qwen3-8b; do for stage in instruct base; do
    [[ -f results/skill_roleplay/$fam/$stage.csv.gz ]] && continue
    while [ -e /root/atlas_logs/DISK_HOLD ]; do sleep 20; done
    model=$($PY -c "from core import config; print(config.FAMILIES['$fam']['stages']['$stage'])")
    avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); cached=$(du -sm "$HUB/models--${model//\//--}" 2>/dev/null | cut -f1)
    if [ "${cached:-0}" -le 1000 ] && [ "$avail" -lt 17 ]; then echo "pass $pass: $model skipped (only ${avail} GB free)"; skipped=1; continue; fi
    echo "=== J11 $fam $stage $(date -u +%T)"
    t=$(date +%s); $PY -m extras.skill_roleplay run --family $fam --stage $stage || echo "FAILED $fam $stage"; echo "   [$(( $(date +%s)-t )) s]"
    free_model "$model"
  done; done
  [ $skipped = 0 ] && break
  while pgrep -f "[r]un_gemma_overnight_v3" >/dev/null; do sleep 30; done
done
echo "=== side v2 done $(date -u +%T)"
