#!/usr/bin/env bash
# Spec Q on marble: OLMo 3 7B and Qwen3-8B, final then base (batch 32). First a 192-request timing sample (projection written to /tmp/claude-0/q_timing.txt; STOP file if the full spec Q is projected over 4 GPU hours AND this machine's part alone is, otherwise Gemma base is dropped on vinyl by its own chain).
# After generation the replies are copied to vinyl for the judge. Log /tmp/claude-0/q_marble.log
set -uo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RS_BATCH=32
PY=/opt/arena-env/bin/python
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
while pgrep -f "[r]un_o_marble" >/dev/null || [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 20; done
echo "owner=main-session purpose=spec Q marble; released when run_q_marble.sh ends" > /tmp/claude-0/gpu.lock
trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
echo "=== timing sample $(date -u +%T)"
s=$(date +%s)
RS_OUT=/tmp/q_sample $PY -m extras.role_says gen --family olmo3-7b --stage instruct --limit 192 > /tmp/q_sample.log 2>&1
t=$(grep "generations in" /tmp/q_sample.log | grep -o "in [0-9]* s" | grep -o "[0-9]*")
echo "sample: 192 requests in ${t:-?} s (incl. nothing but generation)" | tee /tmp/claude-0/q_timing.txt
for fam in olmo3-7b qwen3-8b; do for stage in instruct base; do
  [[ -f results/role_says/$fam/${stage}_replies.csv.gz ]] && continue
  echo "=== $fam $stage $(date -u +%T)"
  $PY -m extras.role_says gen --family $fam --stage $stage || echo "FAILED $fam $stage"
  m=$($PY -c "from core import config; print(config.FAMILIES['$fam']['stages']['$stage'])"); free_model "$m"
done; done
echo "=== copying replies to vinyl $(date -u +%T)"
ssh arena9-vinyl 'mkdir -p /root/hpa-atlas2/results/role_says/olmo3-7b /root/hpa-atlas2/results/role_says/qwen3-8b' 2>/dev/null
for fam in olmo3-7b qwen3-8b; do scp -q results/role_says/$fam/*_replies.csv.gz arena9-vinyl:/root/hpa-atlas2/results/role_says/$fam/ 2>/dev/null; done
ssh arena9-vinyl 'touch /root/hpa-atlas2/results/role_says/MARBLE_REPLIES_DONE' 2>/dev/null
echo "=== done $(date -u +%T)"
