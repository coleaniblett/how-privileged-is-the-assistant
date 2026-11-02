"""Overnight data collection (collection only: no flags, no judge, no analysis). Plan and rules: docs/runbook_2026-11-03.md.

    uv run python scripts/overnight_collect.py [--dry-run] [--only gemma4-31b,olmo3-7b,qwen3-8b]

Per checkpoint, each task in its own process: score data/texts.jsonl, score data/texts_authorship.jsonl (all slots, per-token NLL and
entropy saved), generate from data/prefill_100.jsonl with --save-ids. Order: Gemma 4 31B (base, instruct; --n-per-set 100, fp32 CPU-offload
scoring, int8 generation), OLMo 3 7B (base, SFT, DPO, Instruct), Qwen3-8B (base, instruct) under both assistant-slot renderings: think block
ON (primary, results/capstone/qwen3-8b/) and OFF (robustness, results/capstone_think_off/qwen3-8b/; scores only, generation is identical).

Rules: never overwrite an existing output (a finished task is skipped); disk is checked before each download (before Gemma it frees only the Qwen
weights and then the regenerable datasets cache, if needed); out-of-memory is retried once with half the batch size (generation: once, unchanged);
any other failure is logged and the next checkpoint starts; each checkpoint is removed from the HF cache after use; integrity checks run after every
checkpoint; everything is logged under results/capstone/logs/ (driver.log, one log per task, summary.json).

Resilience (a session or connection reset must not cost more than the task in flight): run it through scripts/overnight_launch.sh, which starts
scripts/overnight_supervise.sh in a tmux session. A finished task is never redone (its outputs exist), and a task writes into a temp directory
that is moved into place only when it has succeeded, so a kill mid-task leaves no truncated output; an interrupted task's log is kept under a new
name and its temp directory removed. One driver at a time (a lock), SIGHUP is ignored, and a heartbeat plus status.json show whether it is alive
(scripts/overnight_status.py). Exit code 0 means the whole plan was attempted; 3 means another driver holds the lock.
"""

import argparse
import datetime
import glob
import json
import math
import os
import fcntl
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core import config, hf_cache  # noqa: E402
from core.render import load_jsonl, load_tokenizer, render_all  # noqa: E402
from core.score import first_n_per_set  # noqa: E402

RUN, OFF_RUN = "capstone", "capstone_think_off"
LOGS = os.environ.get("OVERNIGHT_LOGS", os.path.join("results", RUN, "logs"))  # the override is for tests and dry runs
PY = sys.executable
TEXTS, AUTHORSHIP, PREFILL = "data/texts.jsonl", "data/texts_authorship.jsonl", "data/prefill_100.jsonl"
SAMPLES = 3
PLAN = [("gemma4-31b", ["base", "instruct"], 100), ("olmo3-7b", ["base", "sft", "dpo", "instruct"], None),
        ("qwen3-8b", ["base", "instruct"], None)]
OFF_CODE = ("import sys, runpy; from core import config; config.FAMILIES['qwen3-8b']['assistant_think_block'] = False; "
            "sys.argv = ['core.score'] + sys.argv[1:]; runpy.run_module('core.score', run_name='__main__')")
SUMMARY = {"started": None, "finished": None, "tasks": [], "checkpoints": [], "disk": [], "notes": []}


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, "driver.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_summary():
    with open(os.path.join(LOGS, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(SUMMARY, f, indent=1, default=str)


def disk_note(label):
    free = hf_cache.free_gb()
    SUMMARY["disk"].append(dict(time=now(), label=label, free_gb=round(free, 1)))
    return free


# --- GPU / disk ------------------------------------------------------------------------------------------------------------

def gpu_busy():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if l.strip()]


def wait_for_gpu(minutes=30):
    t0 = time.time()
    busy = gpu_busy()
    while busy and time.time() - t0 < minutes * 60:
        log(f"GPU busy with other processes ({busy}); waiting")
        time.sleep(60)
        busy = gpu_busy()
    if busy:
        msg = f"GPU still busy after {minutes} min ({busy}); proceeding anyway (an out-of-memory retry may follow)"
        log(msg)
        SUMMARY["notes"].append(msg)


def ensure_disk(repo, family):
    """True if the weights of `repo` fit. For Gemma, frees only the Qwen weights and then the regenerable datasets cache, if needed."""
    need, free, ok = hf_cache.check_disk(repo)
    log(f"disk check {repo}: need {need:.1f} GB, free {free:.1f} GB -> {'OK' if ok else 'NOT ENOUGH'}")
    if ok or family != "gemma4-31b":
        return ok
    for q in ("Qwen/Qwen3-8B", "Qwen/Qwen3-8B-Base"):
        if hf_cache.is_cached(q):
            freed, removed, kept = hf_cache.free_repo(q)
            log(f"freed {q} from the HF cache: {freed:.1f} GB ({removed} blobs removed, {kept} shared kept)")
    need, free, ok = hf_cache.check_disk(repo)
    if not ok:
        d = os.path.expanduser("~/.cache/huggingface/datasets")
        if os.path.isdir(d):
            size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs if os.path.exists(os.path.join(r, f))) / 1e9
            shutil.rmtree(d)
            log(f"removed the regenerable datasets cache {d} ({size:.1f} GB)")
            SUMMARY["notes"].append(f"removed ~/.cache/huggingface/datasets ({size:.1f} GB) to fit {repo}")
        need, free, ok = hf_cache.check_disk(repo)
    log(f"disk check {repo} after freeing: need {need:.1f} GB, free {free:.1f} GB -> {'OK' if ok else 'NOT ENOUGH'}")
    return ok


# --- tasks -----------------------------------------------------------------------------------------------------------------

def oom(log_path):
    tail = open(log_path, encoding="utf-8", errors="replace").read()[-30000:].lower()
    return "out of memory" in tail or "outofmemoryerror" in tail


def die_with_parent(cmd):
    """Prefix a command so the kernel terminates it if this driver dies (setpriv --pdeathsig): a hard kill of the driver must not leave a
    core.score on the GPU for the restarted driver to collide with."""
    return ["setpriv", "--pdeathsig", "TERM", "--"] + cmd if shutil.which("setpriv") else cmd


def sweep_orphans(match="--run capstone"):
    """Terminate leftover core.score / core.generate processes of a previous driver: same working directory as this repo, a command line with
    `match`, a Python interpreter as the program (not a shell that merely mentions it), and a parent that is not this process. Only ever our own earlier children (one driver at a time is enforced by the lock)."""
    me, killed = os.getpid(), []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            cwd = os.readlink(f"/proc/{d}/cwd")
            text = open(f"/proc/{d}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore")
            ppid = int(open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()[1])
        except OSError:
            continue
        first = os.path.basename(text.split()[0]) if text.split() else ""
        if (cwd == ROOT and "python" in first and ("core.score" in text or "core.generate" in text) and match in text and ppid != me):
            killed.append((int(d), text[:140]))
    for pid, text in killed:
        log(f"sweeping a leftover process of an earlier driver: pid {pid}: {text}")
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
    t0 = time.time()
    while time.time() - t0 < 30 and any(os.path.exists(f"/proc/{pid}") for pid, _ in killed):
        time.sleep(1)
    for pid, _ in killed:
        if os.path.exists(f"/proc/{pid}"):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    return [pid for pid, _ in killed]


def rotate_log(log_path):
    """Keep the log of an interrupted attempt under a new name instead of overwriting it."""
    if os.path.exists(log_path):
        stamp = datetime.datetime.fromtimestamp(os.path.getmtime(log_path), datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        os.replace(log_path, log_path.replace(".log", f".interrupted-{stamp}.log"))


def run_task(name, cmd, out_dir, out_names, dry_run, batch_arg=None):
    """Run one process whose command contains the placeholder {OUT}. It writes into a temp directory under out_dir; the files named in
    out_names are moved into out_dir only after a clean exit (the completion marker, the first name, last), so out_dir never holds a partial
    output. Skip if the outputs exist; one retry on out-of-memory. Returns a result dict."""
    final = [os.path.join(out_dir, n) for n in out_names]
    res = dict(task=name, started=now(), status=None, seconds=None, retries=0, cmd=" ".join(cmd[:12]) + (" ..." if len(cmd) > 12 else ""))
    if all(os.path.exists(f) for f in final):
        res.update(status="skipped (outputs exist, never overwritten)", seconds=0)
        log(f"{name}: outputs exist, skipped")
        SUMMARY["tasks"].append(res)
        return res
    if dry_run:
        res.update(status="dry run", seconds=0)
        log(f"[dry run] {name}: {' '.join(cmd).replace('{OUT}', out_dir)}")
        return res
    tmp = os.path.join(out_dir, f".tmp_{name}")
    if os.path.isdir(tmp):  # left by an interrupted attempt: our own incomplete temp output, never a result
        shutil.rmtree(tmp)
        log(f"{name}: removed the temp directory of an interrupted attempt ({tmp})")
    wait_for_gpu()
    set_status(task=name, since=now())
    t0 = time.time()
    for attempt in (0, 1):
        log_path = os.path.join(LOGS, f"{name}{'_retry' if attempt else ''}.log")
        rotate_log(log_path)
        c = [x.replace("{OUT}", tmp) for x in cmd]
        if attempt and batch_arg:
            i = c.index("--batch-size")
            c[i + 1] = str(max(1, int(c[i + 1]) // 2))
        os.makedirs(tmp, exist_ok=True)
        log(f"{name}: start{' (retry, ' + ' '.join(c[c.index('--batch-size'):c.index('--batch-size') + 2]) + ')' if attempt and batch_arg else ' (retry)' if attempt else ''}")
        with open(log_path, "w", encoding="utf-8") as lf:
            rc = subprocess.run(die_with_parent(c), stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc == 0 and all(os.path.exists(os.path.join(tmp, n)) for n in out_names):
            os.makedirs(out_dir, exist_ok=True)
            for n in reversed(out_names):  # the completion marker (first name) is moved last
                os.replace(os.path.join(tmp, n), os.path.join(out_dir, n))
            shutil.rmtree(tmp, ignore_errors=True)
            res.update(status="ok" if not attempt else "ok after out-of-memory retry", retries=attempt)
            break
        was_oom = oom(log_path)
        log(f"{name}: failed (exit {rc}){' - out of memory' if was_oom else ''}; see {log_path}")
        res.update(status=f"failed (exit {rc}){' out of memory' if was_oom else ''}", retries=attempt)
        shutil.rmtree(tmp, ignore_errors=True)
        if attempt == 0 and was_oom:
            continue
        break
    res["seconds"] = round(time.time() - t0, 1)
    log(f"{name}: {res['status']} in {res['seconds']:.0f}s")
    SUMMARY["tasks"].append(res)
    set_status(task=None, last=name, last_status=res["status"])
    save_summary()
    return res


def score_cmd(family, run, stage, dataset, n_per_set, off=False):
    bs = str(config.family_settings(family)["batch_size"])
    args = ["--family", family, "--run", run, "--model", stage, "--dataset", dataset, "--out", "{OUT}", "--batch-size", bs]
    if n_per_set:
        args += ["--n-per-set", str(n_per_set)]
    return ([PY, "-c", OFF_CODE] if off else [PY, "-m", "core.score"]) + args


def gen_cmd(family, run, stage):
    return [PY, "-m", "core.generate", "--family", family, "--run", run, "--model", stage, "--prefill", PREFILL, "--save-ids", "--out", "{OUT}"]


# --- heartbeat, status, lock -----------------------------------------------------------------------------------------------

STATUS = {}


def set_status(**kw):
    STATUS.update(kw, pid=os.getpid(), updated=now())
    tmp = os.path.join(LOGS, "status.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(STATUS, f, indent=1)
    os.replace(tmp, os.path.join(LOGS, "status.json"))


def start_heartbeat(period=30):
    def beat():
        while True:
            try:
                set_status(heartbeat=now())
            except Exception:  # noqa: BLE001
                pass
            time.sleep(period)
    threading.Thread(target=beat, daemon=True).start()


def take_lock():
    f = open(os.path.join(LOGS, ".driver.lock"), "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another overnight_collect.py holds the lock; exiting", flush=True)
        sys.exit(3)
    return f  # keep a reference: the lock lives as long as the file object


# --- integrity checks ------------------------------------------------------------------------------------------------------

def check_scores(csv_path, dataset, n_per_set, tokenizer, think_block=None):
    """Integrity of one score file (+ per-token parquet). Returns {ok, issues, stats}."""
    issues, stats = [], {}
    rows = first_n_per_set(load_jsonl(dataset), n_per_set)
    ids = {r["id"] for r in rows}
    df = pd.read_csv(csv_path)
    stats["texts"], stats["rows"] = len(ids), len(df)
    if len(df) != 3 * len(ids):
        issues.append(f"rows {len(df)} != 3 x {len(ids)} texts")
    if set(df.text_id) != ids:
        issues.append(f"text ids differ from the dataset's first {n_per_set or 'all'} per set (missing {len(ids - set(df.text_id))}, extra {len(set(df.text_id) - ids)})")
    kind_of = {r["id"]: r["kind"] for r in rows}
    for tid, g in df.groupby("text_id"):
        ab = "user" if kind_of.get(tid) == "user" else "asst"
        if set(g.condition) != {f"{ab}@user", f"{ab}@asst", f"{ab}@plain"} or g.n_tokens.nunique() != 1:
            issues.append(f"text {tid}: conditions/n_tokens inconsistent")
            break
    num = df[["mean_nll", "mean_nll_from1", "mean_entropy", "mean_entropy_from1"]]
    bad = int((~np.isfinite(num.to_numpy(dtype=float))).sum())
    stats["nonfinite_summary_values"] = bad
    if bad:
        issues.append(f"{bad} NaN/inf values in the per-text columns")
    pq = pd.read_parquet(csv_path.replace(".csv", "_tokens.parquet"), columns=["text_id", "condition", "position", "nll", "entropy"])
    stats["token_rows"] = len(pq)
    if len(pq) != int(df.n_tokens.sum()):
        issues.append(f"token rows {len(pq)} != sum of n_tokens {int(df.n_tokens.sum())}")
    for col in ("nll", "entropy"):
        nb = int((~np.isfinite(pq[col].to_numpy(dtype=float))).sum())
        if nb:
            issues.append(f"{nb} NaN/inf per-token {col} values")
    if (pq.entropy < -1e-4).any():
        issues.append("negative per-token entropy")
    stats["max_token_nll"], stats["max_entropy"] = float(pq.nll.max()), float(pq.entropy.max())
    # render checks with the tokenizer and assistant-slot setting the scores were made with; content lengths must match n_tokens
    old = config.ASSISTANT_THINK_BLOCK
    config.ASSISTANT_THINK_BLOCK = think_block
    try:
        tok = load_tokenizer(tokenizer)
        n_by_id = df.groupby("text_id").n_tokens.first()
        for r in rows:
            for v in render_all(tok, r).values():
                if v.end - v.start != int(n_by_id[r["id"]]):
                    issues.append(f"render length mismatch for {r['id']}")
                    break
        stats["render_checks"] = f"{len(rows)} texts x 3 slots"
        vocab = len(tok)
        if stats["max_entropy"] > math.log(vocab) + 0.05:
            issues.append(f"entropy {stats['max_entropy']:.2f} above ln(vocab)={math.log(vocab):.2f}")
    except Exception as e:  # noqa: BLE001
        issues.append(f"render check failed: {e!r}")
    finally:
        config.ASSISTANT_THINK_BLOCK = old
    anomalies = []
    if stats["max_token_nll"] > 40:
        anomalies.append(f"max per-token NLL {stats['max_token_nll']:.1f}")
    if float(df.mean_nll.max()) > 15:
        anomalies.append(f"max per-text mean NLL {float(df.mean_nll.max()):.1f}")
    stats["anomalies"] = anomalies
    return dict(ok=not issues, issues=issues, stats=stats)


def check_generations(path, prefill, samples=SAMPLES, max_new=config.GEN_MAX_NEW_TOKENS):
    issues, stats = [], {}
    pre = load_jsonl(prefill)
    rows = load_jsonl(path)
    stats["rows"] = len(rows)
    if len(rows) != samples * len(pre):
        issues.append(f"rows {len(rows)} != {samples} x {len(pre)} prefills")
    if {r["prefill_id"] for r in rows} != {p["id"] for p in pre}:
        issues.append("prefill ids differ from the prefill file")
    by = {}
    for r in rows:
        by.setdefault(r["prefill_id"], set()).add(r["sample"])
    if any(s != set(range(samples)) for s in by.values()):
        issues.append("some prefill lacks samples 0..2")
    if any("continuation_ids" not in r or not isinstance(r["continuation_ids"], list) for r in rows):
        issues.append("continuation_ids missing")
    else:
        stats["max_ids_len"] = max(len(r["continuation_ids"]) for r in rows)
        if stats["max_ids_len"] > max_new:
            issues.append(f"continuation_ids longer than {max_new}")
    stats["empty_continuations"] = sum(1 for r in rows if not r["continuation"].strip())
    stats["stopped_on_eos"] = sum(1 for r in rows if r["stopped_on_eos"])
    stats["anomalies"] = [f"{stats['empty_continuations']} empty continuations"] if stats["empty_continuations"] > 0.3 * max(1, len(rows)) else []
    return dict(ok=not issues, issues=issues, stats=stats)


# --- one checkpoint ----------------------------------------------------------------------------------------------------------

def collect(family, stage, n_per_set, dry_run):
    st = config.family_settings(family)
    repo, tokenizer = st["stages"][stage], st["template"]
    ck = dict(family=family, stage=stage, repo=repo, started=now(), integrity={}, failed=False)
    log(f"===== {family} {stage} ({repo})")
    free0 = disk_note(f"{family} {stage}: before")
    if not dry_run and not ensure_disk(repo, family):
        ck.update(failed=True, reason="not enough disk for the download")
        log(f"{family} {stage}: skipped, not enough disk")
        SUMMARY["checkpoints"].append(ck)
        save_summary()
        return
    variants = [("", RUN, None)]  # (suffix, run name, off?)
    if family == "qwen3-8b":
        variants.append(("_think_off", OFF_RUN, True))
    try:
        for suffix, run, off in variants:
            base = os.path.join("results", run, family, "core")
            think = False if off else (st["assistant_think_block"] or None)
            tasks = [("scores", TEXTS, os.path.join(base, "scores")), ("authorship", AUTHORSHIP, os.path.join(base, "scores_authorship"))]
            for label, ds, out in tasks:
                f = os.path.join(out, f"{stage}.csv")
                r = run_task(f"{family}_{stage}_{label}{suffix}", score_cmd(family, run, stage, ds, n_per_set, bool(off)),
                             out, [f"{stage}.csv", f"{stage}_tokens.parquet"], dry_run, batch_arg=True)
                if r["status"].startswith("failed"):
                    ck["failed"] = True
                    raise RuntimeError(f"{r['task']} failed")
                if not dry_run:
                    ck["integrity"][f"{label}{suffix}"] = check_scores(f, ds, n_per_set, tokenizer, think if family == "qwen3-8b" else None)
            if not off:
                gout = os.path.join(base, "generations")
                g = os.path.join(gout, f"{stage}.jsonl")
                r = run_task(f"{family}_{stage}_generation", gen_cmd(family, run, stage), gout, [f"{stage}.jsonl"], dry_run)
                if r["status"].startswith("failed"):
                    ck["failed"] = True
                    raise RuntimeError(f"{r['task']} failed")
                if not dry_run:
                    ck["integrity"]["generation"] = check_generations(g, PREFILL)
    except Exception as e:  # noqa: BLE001
        log(f"{family} {stage}: stopped: {e!r}")
        ck.setdefault("errors", []).append(repr(e) + "\n" + traceback.format_exc()[-1500:])
    finally:
        for k, v in ck["integrity"].items():
            log(f"integrity {family} {stage} {k}: {'OK' if v['ok'] else 'ISSUES ' + str(v['issues'])}" + (f" anomalies: {v['stats'].get('anomalies')}" if v["stats"].get("anomalies") else ""))
        if not dry_run:
            freed, removed, kept = hf_cache.free_repo(repo)
            log(f"cleared {repo} from the HF cache: {freed:.1f} GB ({removed} blobs removed, {kept} shared kept)")
        free1 = disk_note(f"{family} {stage}: after clearing")
        ck.update(finished=now(), free_gb_before=round(free0, 1), free_gb_after=round(free1, 1))
        SUMMARY["checkpoints"].append(ck)
        save_summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", help="comma-separated families (default: all, in plan order)")
    args = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)  # a closed terminal or dropped connection must not stop the run (children inherit this)
    lock = None if args.dry_run else take_lock()  # noqa: F841
    if not args.dry_run:
        start_heartbeat()
        swept = sweep_orphans()
        if swept:
            SUMMARY["notes"].append(f"swept leftover processes of an earlier driver: {swept}")
    SUMMARY["started"] = now()
    log(f"overnight collection start (dry run: {args.dry_run}); branch {subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True).stdout.strip()}; "
        f"HEAD {subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()}")
    for fam, stages, n in PLAN:
        if args.only and fam not in args.only.split(","):
            continue
        for stage in stages:
            try:
                collect(fam, stage, n, args.dry_run)
            except Exception as e:  # noqa: BLE001
                log(f"{fam} {stage}: unexpected error {e!r}\n{traceback.format_exc()[-1500:]}")
                SUMMARY["notes"].append(f"{fam} {stage}: unexpected error {e!r}")
    SUMMARY["finished"] = now()
    disk_note("end")
    save_summary()
    log("overnight collection finished")


if __name__ == "__main__":
    main()
