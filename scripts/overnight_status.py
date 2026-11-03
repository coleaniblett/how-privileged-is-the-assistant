"""Is the overnight collection alive, and where is it? Stdlib only, safe to run any time (read-only).

    python3 scripts/overnight_status.py
"""

import datetime
import glob
import json
import os
import shutil
import subprocess

LOGS = os.environ.get("OVERNIGHT_LOGS", os.path.join("results", "capstone", "logs"))


def load(name):
    try:
        return json.load(open(os.path.join(LOGS, name)))
    except (OSError, ValueError):
        return {}


def age(ts):
    try:
        t = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=datetime.timezone.utc)
        return (datetime.datetime.now(datetime.timezone.utc) - t).total_seconds()
    except (TypeError, ValueError):
        return None


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def main():
    st, sm = load("status.json"), load("summary.json")
    hb = age(st.get("heartbeat"))
    pid_ok = alive(st.get("pid"))
    if pid_ok and hb is not None and hb < 120:
        state = "ALIVE"
    elif pid_ok:
        state = "STALE (process exists, no heartbeat for %s s)" % (int(hb) if hb is not None else "?")
    else:
        state = "NOT RUNNING"
    print(f"driver: {state}  pid {st.get('pid')}  last heartbeat {st.get('heartbeat')}")
    print(f"current task: {st.get('task') or '-'}" + (f" (since {st['since']})" if st.get("task") else "") +
          f"   last finished: {st.get('last') or '-'} -> {st.get('last_status') or '-'}")
    tm = subprocess.run(["tmux", "list-sessions"], capture_output=True, text=True)
    print("tmux:", tm.stdout.strip().replace("\n", "; ") or "no sessions")
    done = {}
    for run in sorted(glob.glob(os.path.join("results", "capstone*"))):
        for kind, pat in (("scores", "scores/*.csv"), ("authorship", "scores_authorship/*.csv"), ("generations", "generations/*.jsonl")):
            for f in glob.glob(os.path.join(run, "*", "core", pat)):
                fam = f.split(os.sep)[2] + ("" if run.endswith("capstone") else " (think off)")
                done.setdefault(fam, {}).setdefault(kind, []).append(os.path.basename(f).rsplit(".", 1)[0])
    for fam, kinds in done.items():
        print(f"  finished {fam}: " + "; ".join(f"{k} {','.join(sorted(v))}" for k, v in kinds.items()))
    if not done:
        print("  no outputs yet")
    fails = [t for t in sm.get("tasks", []) if str(t.get("status", "")).startswith("failed")]
    print(f"failed tasks so far: {len(fails)}" + "".join(f"\n  {t['task']}: {t['status']}" for t in fails))
    print(f"disk free: {shutil.disk_usage('/').free / 1e9:.0f} GB")
    log = os.path.join(LOGS, "driver.log")
    if os.path.exists(log):
        print("last driver.log lines:")
        print("".join("  " + l for l in open(log, encoding="utf-8").readlines()[-5:]).rstrip())


if __name__ == "__main__":
    main()
