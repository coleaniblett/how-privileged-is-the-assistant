#!/usr/bin/env bash
# Gemma 4 31B pass of scripts/collect_conv_incomplete.py with a dedicated RAM-backed HF cache (2026-11-03): the root disk lacks room
# for the 62.5 GB checkpoints. Only the cache lives in /dev/shm; results, manifests, provenance, state and logs stay in the worktree.
#
#   bash scripts/collect_gemma_shm.sh      # waits for any running collection driver (its lock), then runs --only gemma4-31b
#
# Checks before launching (any failure: exit 2, Gemma stays blocked, nothing downloaded): the container's cgroup memory limit minus its
# non-reclaimable usage (rss + shmem) must cover the cached weights (62.5 GB), the fp32 CPU-offloaded part of the model (~93 GB) and a
# 15 GB margin; /dev/shm must have room for one checkpoint plus 5 GB; no other process on the GPU (the runner also waits for that).
# Checkpoints are downloaded one at a time and their weights removed only after both tasks (runner rule). At the end only this run's
# dedicated cache directory is removed.
set -u
WT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$WT/results/collect_conv_incomplete"
SHM="/dev/shm/grace-gemma-hub"
NEED_SHM_GB=67.5          # 62.5 GB weights + the runner's 5 GB margin
NEED_MEM_GB=170.3         # 62.5 weights in shm + 92.8 fp32 offload (125.0 fp32 - 32.2 on the GPU) + 15 margin
cd "$WT" || exit 1
log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] gemma-shm: $*" | tee -a "$LOGS/run.log"; }

log "waiting for the running collection driver to release its lock"
until flock -n "$LOGS/.driver.lock" true; do sleep 60; done
log "lock free"

cg=/sys/fs/cgroup/memory
if [ -f "$cg/memory.limit_in_bytes" ]; then
  limit=$(awk '/^hierarchical_memory_limit /{print $2}' "$cg/memory.stat")
  pinned=$(awk '/^total_rss /{r=$2} /^total_shmem /{s=$2} END{print r+s}' "$cg/memory.stat")
else
  limit=$(cat /sys/fs/cgroup/memory.max); [ "$limit" = max ] && limit=$(awk '/MemTotal/{print $2*1024}' /proc/meminfo)
  pinned=$(awk '/^anon /{a=$2} /^shmem /{s=$2} END{print a+s}' /sys/fs/cgroup/memory.stat)
fi
avail_mem=$(awk -v l="$limit" -v p="$pinned" 'BEGIN{printf "%.1f", (l-p)/1e9}')
shm_free=$(df -B1 --output=avail /dev/shm | tail -1 | awk '{printf "%.1f", $1/1e9}')
log "memory: cgroup limit $(awk -v l="$limit" 'BEGIN{printf "%.1f", l/1e9}') GB, non-reclaimable in use $(awk -v p="$pinned" 'BEGIN{printf "%.1f", p/1e9}') GB, available $avail_mem GB (need $NEED_MEM_GB); /dev/shm free $shm_free GB (need $NEED_SHM_GB)"
if awk -v a="$avail_mem" -v n="$NEED_MEM_GB" 'BEGIN{exit !(a < n)}'; then log "BLOCKED: not enough container memory; Gemma left blocked"; exit 2; fi
if awk -v a="$shm_free" -v n="$NEED_SHM_GB" 'BEGIN{exit !(a < n)}'; then log "BLOCKED: not enough /dev/shm; Gemma left blocked"; exit 2; fi

mkdir -p "$SHM/hub" "$SHM/xet"
( while true; do
    echo "$(date -u '+%H:%M:%S') $(awk '/^total_rss /{r=$2} /^total_shmem /{s=$2} /^total_cache /{c=$2} END{printf "rss %.1f GB shmem %.1f GB cache %.1f GB", r/1e9, s/1e9, c/1e9}' $cg/memory.stat 2>/dev/null) usage $(awk '{printf "%.1f GB", $1/1e9}' $cg/memory.usage_in_bytes 2>/dev/null) shm_used $(du -sb "$SHM" 2>/dev/null | awk '{printf "%.1f GB", $1/1e9}')" >> "$LOGS/gemma_memory.log"
    sleep 30
  done ) &
sampler=$!

log "launching: HF_HUB_CACHE=$SHM/hub HF_XET_CACHE=$SHM/xet python scripts/collect_conv_incomplete.py --only gemma4-31b"
env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$WT/.venv" HF_HUB_CACHE="$SHM/hub" HF_XET_CACHE="$SHM/xet" \
  "$WT/.venv/bin/python" scripts/collect_conv_incomplete.py --only gemma4-31b >> "$LOGS/run.log" 2>&1
rc=$?
kill "$sampler" 2>/dev/null
log "Gemma driver exited with code $rc; removing this run's dedicated cache $SHM ($(du -sh "$SHM" 2>/dev/null | cut -f1))"
case "$SHM" in /dev/shm/grace-gemma-hub) rm -rf -- "$SHM" ;; esac
log "done (status: python scripts/collect_conv_incomplete.py --status)"
exit $rc
