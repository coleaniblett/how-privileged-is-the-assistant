"""The overnight collection driver (scripts/overnight_collect.py): the plan in a dry run, and its integrity checks on real core output
from the tiny CPU model, including that they catch corrupted files."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TINY = "HuggingFaceTB/SmolLM2-135M-Instruct"
CPU = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}


@pytest.fixture(scope="module")
def drv(tmp_path_factory):
    os.environ["OVERNIGHT_LOGS"] = str(tmp_path_factory.mktemp("logs"))
    spec = importlib.util.spec_from_file_location("overnight_collect", ROOT / "scripts" / "overnight_collect.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_dry_run_lists_every_checkpoint_in_order_and_writes_nothing(tmp_path):
    env = {**os.environ, "OVERNIGHT_LOGS": str(tmp_path)}
    snapshot = lambda: sorted(str(p) for p in (ROOT / "results").rglob("*") if "capstone" in str(p))
    before = snapshot()
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "overnight_collect.py"), "--dry-run"], capture_output=True, text=True,
                         cwd=ROOT, env=env, check=True).stdout
    order = [l.split("=====")[1].split("(")[0].strip() for l in out.splitlines() if "=====" in l]
    assert order == ["gemma4-31b base", "gemma4-31b instruct", "olmo3-7b base", "olmo3-7b sft", "olmo3-7b dpo", "olmo3-7b instruct",
                     "qwen3-8b base", "qwen3-8b instruct"]
    assert out.count("[dry run]") == 8 * 3 + 2 * 2  # scores, authorship scores, generation per checkpoint + the two think-off score tasks per Qwen checkpoint
    assert "--n-per-set 100" in out and out.count("--save-ids") == 8 and out.count("assistant_think_block'] = False") == 4
    assert "results/capstone_think_off/qwen3-8b/core/scores" in out and "results/capstone/qwen3-8b/core/scores" in out
    assert snapshot() == before  # a dry run adds nothing under results/


@pytest.fixture(scope="module")
def tiny_outputs(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    rows = [json.loads(l) for l in open(ROOT / "data" / "texts_authorship.jsonl")]
    sub = []
    for key in [("oasst1", "assistant"), ("ultrachat", "user"), ("ultrachat", "assistant")]:
        sub += [r for r in rows if (r["source"], r["kind"]) == key][:3]
    (d / "data.jsonl").write_text("\n".join(json.dumps(r) for r in sub) + "\n")
    pre = [json.loads(l) for l in open(ROOT / "data" / "prefill_100.jsonl")][:3]
    (d / "prefill.jsonl").write_text("\n".join(json.dumps(r) for r in pre) + "\n")
    subprocess.run([sys.executable, "-m", "core.score", "--model", TINY, "--stage", "t", "--tokenizer", TINY, "--dataset", str(d / "data.jsonl"),
                    "--n-per-set", "2", "--out", str(d / "scores"), "--batch-size", "4"], check=True, cwd=ROOT, env=CPU)
    subprocess.run([sys.executable, "-m", "core.generate", "--model", TINY, "--stage", "t", "--tokenizer", TINY, "--prefill", str(d / "prefill.jsonl"),
                    "--samples", "3", "--max-new-tokens", "8", "--save-ids", "--out", str(d / "gens")], check=True, cwd=ROOT, env=CPU)
    return d


def test_score_integrity_passes_on_real_output(drv, tiny_outputs):
    r = drv.check_scores(str(tiny_outputs / "scores" / "t.csv"), str(tiny_outputs / "data.jsonl"), 2, TINY)
    assert r["ok"], r["issues"]
    assert r["stats"]["texts"] == 6 and r["stats"]["rows"] == 18 and r["stats"]["nonfinite_summary_values"] == 0


def test_score_integrity_catches_corruption(drv, tiny_outputs, tmp_path):
    csv = tiny_outputs / "scores" / "t.csv"
    pq = tiny_outputs / "scores" / "t_tokens.parquet"
    df = pd.read_csv(csv)
    cases = {}
    bad = df.copy(); bad.loc[0, "mean_entropy"] = float("nan"); cases["nan"] = bad
    cases["missing row"] = df.iloc[1:]
    inf = df.copy(); inf.loc[2, "mean_nll"] = float("inf"); cases["inf"] = inf
    for name, frame in cases.items():
        p = tmp_path / f"{name.replace(' ', '_')}.csv"
        frame.to_csv(p, index=False)
        (tmp_path / f"{name.replace(' ', '_')}_tokens.parquet").write_bytes(pq.read_bytes())
        r = drv.check_scores(str(p), str(tiny_outputs / "data.jsonl"), 2, TINY)
        assert not r["ok"], name
    tok = pd.read_parquet(pq)
    tok.loc[0, "entropy"] = float("nan")
    p = tmp_path / "tok.csv"
    df.to_csv(p, index=False)
    tok.to_parquet(tmp_path / "tok_tokens.parquet")
    assert not drv.check_scores(str(p), str(tiny_outputs / "data.jsonl"), 2, TINY)["ok"]
    wrong_n = drv.check_scores(str(csv), str(tiny_outputs / "data.jsonl"), 1, TINY)  # expecting a different number of texts
    assert not wrong_n["ok"]


def test_generation_integrity(drv, tiny_outputs, tmp_path):
    g = tiny_outputs / "gens" / "t.jsonl"
    pre = str(tiny_outputs / "prefill.jsonl")
    r = drv.check_generations(str(g), pre, samples=3, max_new=8)
    assert r["ok"], r["issues"]
    rows = [json.loads(l) for l in open(g)]
    short = tmp_path / "short.jsonl"
    short.write_text("\n".join(json.dumps(x) for x in rows[:-1]) + "\n")
    assert not drv.check_generations(str(short), pre, samples=3, max_new=8)["ok"]
    noids = tmp_path / "noids.jsonl"
    noids.write_text("\n".join(json.dumps({k: v for k, v in x.items() if k != "continuation_ids"}) for x in rows) + "\n")
    assert not drv.check_generations(str(noids), pre, samples=3, max_new=8)["ok"]


# --- resilience: atomic outputs, the supervisor, the lock, the status command ---------------------------------------------------

import fcntl  # noqa: E402
import textwrap  # noqa: E402

WRITE = "import sys; open(sys.argv[1] + '/x.csv', 'w').write('1'); open(sys.argv[1] + '/x_tokens.parquet', 'w').write('2')"
FILES = ["x.csv", "x_tokens.parquet"]


@pytest.fixture
def quiet(drv, monkeypatch):
    monkeypatch.setattr(drv, "gpu_busy", lambda: [])  # never wait on the real GPU in a test
    drv.SUMMARY["tasks"].clear()
    return drv


def test_a_successful_task_is_moved_into_place_whole_and_never_redone(quiet, tmp_path):
    out = str(tmp_path / "out")
    r = quiet.run_task("t_ok", [sys.executable, "-c", WRITE, "{OUT}"], out, FILES, False)
    assert r["status"] == "ok" and sorted(os.listdir(out)) == sorted(FILES)  # no .tmp_ directory left behind
    again = quiet.run_task("t_ok", [sys.executable, "-c", "raise SystemExit(9)", "{OUT}"], out, FILES, False)
    assert again["status"].startswith("skipped") and (Path(out) / "x.csv").read_text() == "1"  # existing output untouched


def test_a_failed_task_leaves_nothing_in_the_output_directory(quiet, tmp_path):
    out = str(tmp_path / "out")
    code = "import sys; open(sys.argv[1] + '/x.csv', 'w').write('partial'); raise SystemExit(2)"
    r = quiet.run_task("t_fail", [sys.executable, "-c", code, "{OUT}"], out, FILES, False)
    assert r["status"].startswith("failed") and not os.path.exists(os.path.join(out, "x.csv"))
    assert not any(n.startswith(".tmp_") for n in os.listdir(out))


def test_a_leftover_temp_directory_and_log_from_an_interrupted_attempt_are_handled(quiet, tmp_path):
    out = tmp_path / "out"
    (out / ".tmp_t_resume").mkdir(parents=True)
    (out / ".tmp_t_resume" / "x.csv").write_text("truncated by a kill")
    log = Path(quiet.LOGS) / "t_resume.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("log of the interrupted attempt")
    r = quiet.run_task("t_resume", [sys.executable, "-c", WRITE, "{OUT}"], str(out), FILES, False)
    assert r["status"] == "ok" and (out / "x.csv").read_text() == "1" and not (out / ".tmp_t_resume").exists()
    kept = [p for p in log.parent.glob("t_resume.interrupted-*.log")]
    assert len(kept) == 1 and kept[0].read_text() == "log of the interrupted attempt"  # kept, not overwritten


def test_out_of_memory_is_retried_once_with_half_the_batch_size(quiet, tmp_path):
    code = textwrap.dedent("""
        import sys
        bs, out = int(sys.argv[2]), sys.argv[3]
        if bs > 2:
            print('torch.OutOfMemoryError: CUDA out of memory'); raise SystemExit(1)
        open(out + '/x.csv', 'w').write(str(bs)); open(out + '/x_tokens.parquet', 'w').write('2')
    """)
    out = str(tmp_path / "out")
    r = quiet.run_task("t_oom", [sys.executable, "-c", code, "--batch-size", "4", "{OUT}"], out, FILES, False, batch_arg=True)
    assert r["status"] == "ok after out-of-memory retry" and r["retries"] == 1 and (Path(out) / "x.csv").read_text() == "2"
    r2 = quiet.run_task("t_other", [sys.executable, "-c", "raise SystemExit(5)", "--batch-size", "4", "{OUT}"], str(tmp_path / "o2"), FILES, False, batch_arg=True)
    assert r2["status"].startswith("failed") and r2["retries"] == 0  # any other failure is not retried


def supervise(tmp_path, driver_body, **env):
    stub = tmp_path / "driver.sh"
    stub.write_text(driver_body)
    e = {**os.environ, "OVERNIGHT_LOGS": str(tmp_path / "logs"), "OVERNIGHT_DRIVER": str(stub), "RESTART_WAIT": "0", **env}
    p = subprocess.run(["bash", str(ROOT / "scripts" / "overnight_supervise.sh")], capture_output=True, text=True, cwd=ROOT, env=e, timeout=60)
    return p, (tmp_path / "logs" / "supervisor.log").read_text() if (tmp_path / "logs" / "supervisor.log").exists() else ""


def test_supervisor_restarts_a_driver_that_dies_until_the_plan_completes(tmp_path):
    counter = tmp_path / "n"
    body = f'n=$(cat {counter} 2>/dev/null || echo 0); echo $((n+1)) > {counter}; [ "$n" -ge 2 ] && exit 0; exit 1\n'
    p, log = supervise(tmp_path, body, MAX_RESTARTS="5")
    assert p.returncode == 0 and log.count("starting the driver") == 3 and "plan complete" in log


def test_supervisor_gives_up_after_the_restart_limit_and_stops_on_a_held_driver_lock(tmp_path):
    p, log = supervise(tmp_path, "exit 1\n", MAX_RESTARTS="2")
    assert p.returncode == 1 and log.count("starting the driver") == 3 and "giving up" in log
    p, log = supervise(tmp_path / "b", "exit 3\n") if (tmp_path / "b").mkdir() is None else (None, "")
    assert p.returncode == 3 and log.count("starting the driver") == 1


def test_only_one_supervisor_at_a_time(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    with open(logs / ".supervisor.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        e = {**os.environ, "OVERNIGHT_LOGS": str(logs), "OVERNIGHT_DRIVER": "/bin/true"}
        p = subprocess.run(["bash", str(ROOT / "scripts" / "overnight_supervise.sh")], capture_output=True, text=True, cwd=ROOT, env=e, timeout=30)
    assert p.returncode == 3 and "already running" in p.stdout


def test_status_command_tells_alive_stale_and_not_running_apart(tmp_path):
    import datetime
    fmt = lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    now = datetime.datetime.now(datetime.timezone.utc)
    e = {**os.environ, "OVERNIGHT_LOGS": str(tmp_path)}
    run = lambda: subprocess.run([sys.executable, str(ROOT / "scripts" / "overnight_status.py")], capture_output=True, text=True, cwd=ROOT, env=e).stdout
    assert "NOT RUNNING" in run()
    (tmp_path / "status.json").write_text(json.dumps({"pid": os.getpid(), "heartbeat": fmt(now), "task": "t", "since": fmt(now)}))
    assert "driver: ALIVE" in run() and "current task: t" in run()
    (tmp_path / "status.json").write_text(json.dumps({"pid": os.getpid(), "heartbeat": fmt(now - datetime.timedelta(minutes=10))}))
    assert "STALE" in run()
    (tmp_path / "status.json").write_text(json.dumps({"pid": 2 ** 22 + 12345, "heartbeat": fmt(now)}))
    assert "NOT RUNNING" in run()


def test_launcher_script_parses():
    subprocess.run(["bash", "-n", str(ROOT / "scripts" / "overnight_launch.sh")], check=True)


def test_children_die_with_the_driver_and_orphans_are_swept(drv):
    import signal as sg
    import time as tm
    # 1. setpriv --pdeathsig: kill -9 the parent, the child is terminated by the kernel
    code = ("import subprocess, sys, time; p = subprocess.Popen(sys.argv[1:]); print(p.pid, flush=True); time.sleep(60)")
    parent = subprocess.Popen([sys.executable, "-c", code, *drv.die_with_parent(["sleep", "60"])], stdout=subprocess.PIPE, text=True)
    child = int(parent.stdout.readline())
    for _ in range(100):  # let setpriv install the signal and exec the command (the driver has run for seconds before any kill)
        if os.path.exists(f"/proc/{child}") and open(f"/proc/{child}/cmdline").read().startswith("sleep"):
            break
        tm.sleep(0.05)
    assert open(f"/proc/{child}/cmdline").read().startswith("sleep")
    parent.send_signal(sg.SIGKILL)
    parent.wait()
    for _ in range(50):
        if not os.path.exists(f"/proc/{child}") or open(f"/proc/{child}/stat").read().rsplit(")", 1)[1].split()[0] == "Z":
            break
        tm.sleep(0.1)
    assert not os.path.exists(f"/proc/{child}") or open(f"/proc/{child}/stat").read().rsplit(")", 1)[1].split()[0] == "Z"
    # 2. an orphan that looks like a collector child is swept; an unrelated process is not
    marker = "--run capstone_orphan_test"
    subprocess.run(["bash", "-c", f"(cd {ROOT} && {sys.executable} -c 'import time; time.sleep(120)' core.score {marker} &) ; "
                                   f"(cd {ROOT} && {sys.executable} -c 'import time; time.sleep(120)' unrelated {marker} &)"], check=True)
    tm.sleep(1)
    swept = drv.sweep_orphans(match=marker)
    assert len(swept) == 1
    tm.sleep(1)
    def python_procs(word):
        out = []
        for d in os.listdir("/proc"):
            try:
                argv = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")
            except OSError:
                continue
            if d.isdigit() and "python" in os.path.basename(argv[0].decode()) and word.encode() in argv and marker.encode() in b" ".join(argv):
                out.append(int(d))
        return out
    left = python_procs("unrelated")
    assert len(left) == 1  # the unrelated process survived the sweep
    for pid in left:
        os.kill(pid, sg.SIGKILL)


def test_the_record_survives_a_restart_and_the_report_is_built_from_it(drv, tmp_path, monkeypatch):
    # a second driver instance picks up the first one's tasks, checkpoints, disk notes and runs
    logs = tmp_path / "logs"
    logs.mkdir()
    prev = {"started": "2026-11-02 20:00:00 UTC", "tasks": [{"task": "a", "status": "ok", "seconds": 61, "retries": 0}],
            "checkpoints": [{"family": "f", "stage": "base", "integrity": {"scores": {"ok": True, "issues": [], "stats": {"rows": 3, "anomalies": ["x"]}}}}],
            "disk": [{"time": "t", "label": "start", "free_gb": 50.0}], "notes": ["n1"], "runs": [{"started": "2026-11-02 20:00:00 UTC"}]}
    (logs / "summary.json").write_text(json.dumps(prev))
    monkeypatch.setattr(drv, "LOGS", str(logs))
    for k in ("tasks", "checkpoints", "disk", "notes", "runs"):
        drv.SUMMARY[k] = []
    drv.load_summary()
    assert drv.SUMMARY["tasks"][0]["task"] == "a" and drv.SUMMARY["notes"] == ["n1"] and drv.SUMMARY["started"] == "2026-11-02 20:00:00 UTC"
    e = {**os.environ, "OVERNIGHT_LOGS": str(logs)}
    out = tmp_path / "report.md"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "overnight_report.py"), "--out", str(out)], check=True, cwd=ROOT, env=e, capture_output=True)
    text = out.read_text()
    for section in ("## What ran", "## Interruptions and restarts", "## Integrity checks per checkpoint", "## Failures and retries", "## Disk",
                    "## Anomalies flagged by the checks", "## What's left to run", "## Readings"):
        assert section in text
    assert "| a | ok | 1m01s | 0 |" in text and "f base" in text and "- f base scores: x" in text
