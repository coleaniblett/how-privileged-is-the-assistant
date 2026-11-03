#!/usr/bin/env bash
# Keeps the overnight collection going: runs scripts/overnight_collect.py and, if it dies (a crash, a kill, the OOM killer, a reset of whatever
# launched it), starts it again. The driver skips finished work and writes outputs atomically, so a restart costs only the task that was in flight.
# Stops when the driver finishes the plan (exit 0), when another driver already runs (exit 3), or after MAX_RESTARTS restarts (exit 1).
# One supervisor at a time (a lock). Normally started by scripts/overnight_launch.sh inside tmux. Extra args go to the driver.
#   MAX_RESTARTS=8 RESTART_WAIT=60 bash scripts/overnight_supervise.sh [--only gemma4-31b,...]
set -u
cd "$(dirname "$0")/.."
LOGS=${OVERNIGHT_LOGS:-results/capstone/logs}
mkdir -p "$LOGS"
exec 9>"$LOGS/.supervisor.lock"
flock -n 9 || { echo "another supervisor is already running"; exit 3; }
trap '' HUP
DRIVER=${OVERNIGHT_DRIVER:-scripts/overnight_collect.py}
MAX=${MAX_RESTARTS:-8}
WAIT=${RESTART_WAIT:-60}
say() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] supervisor: $*" | tee -a "$LOGS/supervisor.log"; }
for attempt in $(seq 1 $((MAX + 1))); do
  say "starting the driver (attempt $attempt)"
  if [[ "$DRIVER" == *.py ]]; then uv run python "$DRIVER" "$@" >> "$LOGS/driver_stdout.log" 2>&1
  else bash "$DRIVER" "$@" >> "$LOGS/driver_stdout.log" 2>&1; fi
  rc=$?
  say "driver exited with code $rc"
  case $rc in
    0) say "plan complete"; exit 0 ;;
    3) say "another driver holds the lock; stopping"; exit 3 ;;
  esac
  if [[ $attempt -le $MAX ]]; then say "restarting in ${WAIT}s"; sleep "$WAIT"; fi
done
say "giving up after $MAX restarts"
exit 1
