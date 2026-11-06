#!/usr/bin/env bash
# R8 on marble: OLMo and Qwen L2 at 256 tokens (final), then judge inputs and copy to vinyl.
cd /root/hpa-main-wt; export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python; source scripts/overgeneralization/overnight_2026-11-05/lib_r.sh
O=results/overgeneralization/overnight_2026-11-05/R8; mkdir -p $O
gpu_wait main-session "R8 marble"; trap gpu_release EXIT
for fam in olmo3-7b qwen3-8b; do
  [[ -f $O/gen/L23/$fam/instruct_L2.csv.gz ]] || WML_BS=32 retry $PY -m extras.refusal_256 gen --family $fam || exit 1
done
R2_LOCAL=/root/r2_work $PY -m extras.refusal_256 prepare
ssh arena9-vinyl 'mkdir -p /root/r2_work'; rsync -a -e ssh /root/r2_work/inputs_r8_*.csv.gz arena9-vinyl:/root/r2_work/
touch $O/MARBLE_DONE
