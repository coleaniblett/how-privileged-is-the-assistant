#!/usr/bin/env bash
# Spec N: L2 + L3 on Gemma 4 31B (instruct, then base; weights downloaded and freed one at a time) on vinyl, after spec M. Small timing sample first; STOP file if the projection exceeds 3 GPU hours. Log /root/atlas_logs/n_vinyl.log
set -uo pipefail
cd /root/hpa-atlas2
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WML_BS=${WML_BS:-48}
PY=/opt/arena-env/bin/python
HUB="$HOME/.cache/huggingface/hub"
free_model() { local repo="models--${1//\//--}"; rm -rf "${HUB:?}/${repo:?}"
  grep -l -r --include='*.refs' "^$repo/" "$HUB/blobs" 2>/dev/null | while read -r refs; do
    if ! grep -qv "^$repo/" "$refs"; then rm -f "${refs%.refs}" "${refs%.refs}.lock" "$refs"; fi; done; }
while pgrep -f "[r]un_m_breadth" >/dev/null; do sleep 20; done
exec 9>results/trait_atlas/.gpu.lock
flock -x 9
avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$avail" -ge 70 ] || { echo "NOT STARTED: only ${avail} GB free" | tee /root/atlas_logs/N_STOP.txt; exit 4; }
echo "=== timing sample $(date -u +%T)"
WML_OUT=/tmp/wml_n_sample $PY -m extras.working_model_l run23 --family gemma4-31b --stage instruct --limit 192 > /tmp/n_sample.log 2>&1
t2=$(grep "L2: 192 generations in" /tmp/n_sample.log | grep -o "in [0-9]* s" | grep -o "[0-9]*"); t3=$(grep "L3: 192 generations in" /tmp/n_sample.log | grep -o "in [0-9]* s" | grep -o "[0-9]*")
echo "sample: L2 192 gens in ${t2:-?} s, L3 192 gens in ${t3:-?} s"
if [ -n "${t2:-}" ] && [ -n "${t3:-}" ]; then
  proj=$(( ( t2 * 3024 / 192 + t3 * 6048 / 192 + 400 ) * 2 ))
  echo "projected total (2 checkpoints incl. loads): $((proj/60)) min"
  if [ $proj -gt 10800 ]; then echo "projected $((proj/60)) min > 3 h: stopping for a decision" | tee /root/atlas_logs/N_STOP.txt; exit 5; fi
else echo "sample timing missing, see /tmp/n_sample.log" | tee /root/atlas_logs/N_STOP.txt; exit 6; fi
for stage in instruct base; do
  [[ -f results/working_model_L/L23/gemma4-31b/${stage}_L3.csv.gz ]] && continue
  echo "=== gemma4-31b $stage $(date -u +%T)"
  $PY -m extras.working_model_l run23 --family gemma4-31b --stage $stage || echo "FAILED $stage"
  free_model "$($PY -c "from core import config; print(config.FAMILIES['gemma4-31b']['stages']['$stage'])")"
done
echo "=== done $(date -u +%T)"
