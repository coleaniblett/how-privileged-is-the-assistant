#!/usr/bin/env bash
# Run GPU jobs on the reserved host (arena9-nurse, L40S 46 GB) while keeping every project file on THIS host.
# The remote copy of the repo is a throwaway mirror: never edit, commit or push there; results are copied back here and the weights live only in the remote HF cache.
#   remote.sh push-code               sync code and data from this host (never results, never .git) to the remote mirror
#   remote.sh push-fits PATH...       copy git-ignored inputs a job needs (e.g. results/overgeneralization/early_probes/olmo3-7b/persona_axis/instruct.npz) to the remote
#   remote.sh run 'COMMAND'           run a command in the remote repo (use nohup/background there for long jobs)
#   remote.sh pull-results PATH...    copy result paths back from the remote (rsync, relative to the repo root); do this after every stage, not only at the end
#   remote.sh status                  GPU use, free disk and running python jobs on the remote
set -euo pipefail
HOST="${REMOTE_HOST:-arena9-nurse}"; REMOTE_DIR="${REMOTE_DIR:-/root/how-privileged-is-the-assistant}"
cd "$(dirname "$0")/../.."
SSH="ssh -o BatchMode=yes -o LogLevel=ERROR"
case "${1:-}" in
  push-code) rsync -a -e "$SSH" --exclude .git --exclude results --exclude __pycache__ --exclude .pytest_cache --exclude .venv --exclude candidates ./ "$HOST:$REMOTE_DIR/" ;;
  push-fits) shift; for p in "$@"; do $SSH "$HOST" "mkdir -p $REMOTE_DIR/$(dirname "$p")"; rsync -a -e "$SSH" "$p" "$HOST:$REMOTE_DIR/$p"; done ;;
  run) shift; $SSH "$HOST" "cd $REMOTE_DIR && $*" ;;
  pull-results) shift; for p in "$@"; do mkdir -p "$(dirname "$p")"; rsync -a -e "$SSH" "$HOST:$REMOTE_DIR/$p" "$(dirname "$p")/"; done ;;
  status) $SSH "$HOST" 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader; df -h / | tail -1 | awk "{print \"disk free:\",\$4}"; ps -eo etime,args | grep "[p]ython" | cut -c1-120' ;;
  *) sed -n 2,10p "$0"; exit 1 ;;
esac
