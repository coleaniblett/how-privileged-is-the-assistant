#!/usr/bin/env bash
# R7 on marble: knows versus says, OLMo and Qwen, final and base (cheap), then analysis and push. Gemma is done on vinyl (final only).
cd /root/hpa-main-wt; export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/arena-env/bin/python; source scripts/overgeneralization/overnight_2026-11-05/lib_r.sh
O=results/overgeneralization/overnight_2026-11-05/R7
gpu_wait main-session "spec R7 marble"; trap gpu_release EXIT
for fam in olmo3-7b qwen3-8b; do for st in instruct base; do
  [[ -f $O/$fam/${st}_B.csv.gz ]] || retry $PY -m extras.knows_says gen --family $fam --stage $st || exit 1
done; free_if_tight "$(stage_model $fam base)" 45; done
rsync -a -e ssh $O/ arena9-vinyl:/root/hpa-atlas2/$O/ 2>/dev/null; touch $O/MARBLE_DONE
