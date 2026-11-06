#!/usr/bin/env bash
# Start (or confirm) the overnight collection in a tmux session that outlives this shell, an SSH connection and an agent session:
# tmux's server is its own process, not a child of whatever launched it. Safe to run again: it never starts a second copy, and it restarts a
# session whose supervisor has exited (the driver resumes from the finished outputs). If the container itself restarts, run this again.
#   bash attic/scripts/overnight_launch.sh        then:  python3 attic/scripts/overnight_status.py   or   tmux attach -t overnight
cd "$(dirname "$0")/.."
S=${TMUX_SESSION:-overnight}
if tmux has-session -t "$S" 2>/dev/null; then
  if [[ "$(tmux list-panes -t "$S" -F '#{pane_dead}' | head -1)" == 1 ]]; then
    echo "session '$S' exists but its supervisor has exited; restarting it"
    tmux kill-session -t "$S"
  else
    echo "tmux session '$S' is already running"
    python3 attic/scripts/overnight_status.py
    exit 0
  fi
fi
# Extra args go to the driver (e.g. --stratum incomplete). The environment is passed explicitly: tmux's server keeps the environment it was started with.
ENVS=()
for v in UV_PROJECT_ENVIRONMENT OVERNIGHT_LOGS MAX_RESTARTS RESTART_WAIT; do [[ -n "${!v:-}" ]] && ENVS+=("$v=${!v}"); done
tmux new-session -d -s "$S" -x 200 -y 50 "env ${ENVS[*]:-} bash attic/scripts/overnight_supervise.sh $*"
tmux set-option -t "$S" remain-on-exit on
echo "started tmux session '$S' (attach: tmux attach -t $S; detach: Ctrl-b d)"
sleep 5
python3 attic/scripts/overnight_status.py
