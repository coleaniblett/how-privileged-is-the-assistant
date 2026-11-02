#!/usr/bin/env bash
# Start (or confirm) the overnight collection in a tmux session that outlives this shell, an SSH connection and an agent session:
# tmux's server is its own process, not a child of whatever launched it. Safe to run again: it never starts a second copy, and it restarts a
# session whose supervisor has exited (the driver resumes from the finished outputs). If the container itself restarts, run this again.
#   bash scripts/overnight_launch.sh        then:  python3 scripts/overnight_status.py   or   tmux attach -t overnight
cd "$(dirname "$0")/.."
S=${TMUX_SESSION:-overnight}
if tmux has-session -t "$S" 2>/dev/null; then
  if [[ "$(tmux list-panes -t "$S" -F '#{pane_dead}' | head -1)" == 1 ]]; then
    echo "session '$S' exists but its supervisor has exited; restarting it"
    tmux kill-session -t "$S"
  else
    echo "tmux session '$S' is already running"
    python3 scripts/overnight_status.py
    exit 0
  fi
fi
tmux new-session -d -s "$S" -x 200 -y 50 "bash scripts/overnight_supervise.sh"
tmux set-option -t "$S" remain-on-exit on
echo "started tmux session '$S' (attach: tmux attach -t $S; detach: Ctrl-b d)"
sleep 5
python3 scripts/overnight_status.py
