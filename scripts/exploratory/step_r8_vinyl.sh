#!/usr/bin/env bash
# R8 on vinyl: Gemma 31B L2 at 256 tokens (waits for the lock, i.e. for T3.3), then judges all three families' replies with R2's judge (waits for marble's inputs until 14:00 UTC).
cd /root/hpa-atlas2; export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True R2_LOCAL=/root/r2_work
O=results/overnight_2026-11-05/R8; mkdir -p $O
exec 9>results/trait_atlas/.gpu.lock; flock -x 9
avail=$(df --output=avail -BG / | tail -n 1 | tr -dc 0-9); [ "$avail" -ge 70 ] || [ -d "$HOME/.cache/huggingface/hub/models--google--gemma-4-31B-it" ] || { echo "only ${avail} GB free"; exit 4; }
[[ -f $O/gen/L23/gemma4-31b/instruct_L2.csv.gz ]] || WML_BS=24 /opt/arena-env/bin/python -m extras.refusal_256 gen --family gemma4-31b || exit 1
DEADLINE=$(date -u -d "2026-11-06 14:00" +%s)
until [ -f /root/r2_work/inputs_r8_olmo3-7b.csv.gz ] && [ -f /root/r2_work/inputs_r8_qwen3-8b.csv.gz ] || [ "$(date -u +%s)" -ge "$DEADLINE" ]; do sleep 60; done
/opt/arena-env/bin/python -m extras.refusal_256 prepare
/opt/arena-env/bin/python -m extras.refusal_judge judge && touch $O/R8_JUDGED_DONE
