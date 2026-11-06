#!/usr/bin/env bash
# Spec I extensions (plan: docs/decisions.md 2026-11-05): base comparison and expansion set, chained, on marble. Log: /tmp/claude-0/spec_i_ext.log
# A failed Stage 0 on base skips that base run only (reported); the expansion runs regardless.
cd /root/hpa-main-wt || exit 1
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python
while [ -e /tmp/claude-0/gpu.lock ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; do sleep 20; done
echo "owner=main-session purpose=spec I extensions; released when run_spec_i_extensions.sh ends" > /tmp/claude-0/gpu.lock
trap 'rm -f /tmp/claude-0/gpu.lock' EXIT
for fam in olmo3-7b qwen3-8b; do
  t=$(date +%s)
  if $PY -m extras.activation_patching stage0 --family $fam --stage base; then
    echo "== $fam stage0 base ok $(( $(date +%s) - t )) s"; t=$(date +%s)
    $PY -m extras.activation_patching stage2x --family $fam --variant base
    echo "== $fam stage2 base done $(( $(date +%s) - t )) s $(date -u +%H:%M)"
  else
    echo "== $fam stage0 BASE FAILED, base run skipped"
  fi
  t=$(date +%s)
  $PY -m extras.activation_patching stage2x --family $fam --variant expansion
  echo "== $fam expansion done $(( $(date +%s) - t )) s $(date -u +%H:%M)"
done
$PY -m extras.activation_patching analyze_ext
echo "=== all done $(date -u +%H:%M)"
