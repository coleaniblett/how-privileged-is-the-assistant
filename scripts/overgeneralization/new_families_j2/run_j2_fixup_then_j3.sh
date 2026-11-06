#!/usr/bin/env bash
# After the J2 chain: rerun the reply-in-user step that failed for Llama instruct (EXTRA_PROFILES added), re-analyse, then hand over to the J3 chain.
set -uo pipefail
cd "$(dirname "$0")/../../.."
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4
PY=/opt/arena-env/bin/python
while pgrep -f "[r]un_j2_new_fam" >/dev/null; do sleep 30; done
while [ -e /tmp/claude-0/gpu.lock ]; do sleep 20; done
echo "owner=main-session purpose=J2 fixup; released when the fixup chain ends" > /tmp/claude-0/gpu.lock
for fam in llama3.1-8b apertus-8b; do for stage in instruct base; do
  [[ -f results/capstone/$fam/$stage/core/exchanges_reply_in_user/scores.csv ]] || { echo "=== riu $fam $stage"; $PY -m extras.exchange_reply_in_user score --family $fam --model $stage --offload --gpu-gib 20 || echo FAILED; }
done; $PY -m extras.exchange_reply_in_user analyze --family $fam; done
rm -f /tmp/claude-0/gpu.lock
exec bash scripts/overgeneralization/refusal_direction/run_j3.sh
