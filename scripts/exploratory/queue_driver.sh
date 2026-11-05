#!/usr/bin/env bash
# Spec R queue driver. Reads a queue file (lines "name|command"; blank and # lines ignored; re-read after every step, so steps can be appended while it runs), runs each step not yet done in order
# (each step up to 3 tries; a failed step is logged and the queue moves on), marks it in <queue>.done. Idle: re-checks every 60 s until the file <queue>.stop exists.
# usage: queue_driver.sh QUEUEFILE   (cwd = repo)
Q="$1"; touch "$Q" "$Q.done"
while [ ! -e "$Q.stop" ]; do
  line=$(grep -v '^[[:space:]]*#' "$Q" | grep -v '^[[:space:]]*$' | while IFS= read -r l; do n="${l%%|*}"; grep -qx "$n" "$Q.done" || { echo "$l"; break; }; done)
  [ -z "$line" ] && { sleep 60; continue; }
  name="${line%%|*}"; cmd="${line#*|}"
  for try in 1 2 3; do
    echo "=== $(date -u +%T) step $name try $try"
    bash -c "$cmd" && { echo "=== $(date -u +%T) step $name ok"; break; }
    echo "=== $(date -u +%T) step $name FAILED try $try" | tee -a "$Q.failures"; sleep 30
  done
  echo "$name" >> "$Q.done"
done
