"""Combined collection, 2026-11-03: original-conversation scores (logit distortion) + incomplete-user-turn generations (vestigial
regression's judge arm), per checkpoint, before that checkpoint's weights are cleared. Collection only: no judge, no flags, no analysis.

    python scripts/collect_conv_incomplete.py --dry-run          # the plan: both tasks for every checkpoint, and what would be reused
    python scripts/collect_conv_incomplete.py --smoke            # one conversation + 3 prefills on OLMo base -> <LOGS>/smoke/
    python scripts/collect_conv_incomplete.py [--only olmo3-7b]  # the collection (resumable; run it in tmux)
    python scripts/collect_conv_incomplete.py --status           # read-only progress

Tasks per checkpoint (separate processes, settings unchanged from the existing CLIs):
  conversation  python -m core.score_conversations --family F --candidates pilot_v3 --model S --context-limits-from <frozen manifest>
                -> results/logit_conversations/<F>/core/conversation_scores/<S>.csv, <S>_tokens.parquet, <S>_provenance.json
  incomplete    python -m core.generate --family F --run capstone --model S --prefill data/prefill_incomplete_200.jsonl --save-ids
                (exactly scripts/overnight_collect.py --stratum incomplete's command)
                -> results/capstone/<F>/core/generations_incomplete/<S>.jsonl + <S>_provenance.json (written here)

Before any scoring, each family's conversation inputs are prepared and frozen (core.score_conversations --prepare-only; an existing
manifest must match its fingerprint) and every stage's config revision is recorded (config_provenance.json beside the manifest), so
clearing a checkpoint cannot change or break a later stage's inputs.

Rules: an existing output is reused only if it validates (inputs, settings, model revision, integrity); one that does not is reported and
never overwritten. Outputs are written to a temp directory and moved into place after a clean exit (scripts/overnight_collect.run_task),
out-of-memory is retried once with half the batch size. The GPU must be free of other processes (waits, never proceeds alongside). Disk is
never freed for a download: a checkpoint that does not fit is reported blocked. After a checkpoint, only weight files that this run
downloaded are removed (repos with weights already cached when first seen are never touched). The collection is complete only if every
expected task validates. Logs, state and status: results/collect_conv_incomplete/.
"""

import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import time
import traceback
from importlib.metadata import version

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join("results", "collect_conv_incomplete")
os.environ["OVERNIGHT_LOGS"] = LOGS  # overnight_collect's helpers (run_task, log, status, lock) write here, not to results/capstone/logs
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import overnight_collect as oc  # noqa: E402
from core import config, hf_cache  # noqa: E402
from core import score_conversations as sc  # noqa: E402
from core.render import load_jsonl, load_tokenizer  # noqa: E402

PY = sys.executable
CANDIDATES = "candidates/logit_distortion/pilot_v3"
CONV_RUN, GEN_RUN = sc.DEFAULT_RUN, oc.RUN  # results/logit_conversations, results/capstone
PREFILL = oc.PREFILL_INCOMPLETE
PLAN = oc.INCOMPLETE_PLAN  # OLMo (base, SFT, DPO, Instruct) -> Qwen3-8B (base, instruct) -> Gemma 4 31B (base, instruct)
STATE = os.path.join(LOGS, "state.json")
SMOKE = os.path.join(LOGS, "smoke")
GEN_CODE = ("core/generate.py", "core/render.py", "core/config.py", "core/score.py")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()


def conv_dir(family):
    return config.core_dir(sc.OUT_KIND, CONV_RUN, family)


def gen_dir(family):
    return os.path.join("results", GEN_RUN, family, "core", "generations_incomplete")


# --- state -------------------------------------------------------------------------------------------------------------------

def load_state():
    try:
        return json.load(open(STATE))
    except (OSError, ValueError):
        return {"downloads": {}, "checkpoints": {}, "families": {}}


def save_state(state):
    os.makedirs(LOGS, exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, STATE)


def note_ownership(state, repo):
    """First sighting of a repo decides who owns its weights: already cached -> someone else's (never removed); not cached -> this
    run downloads them (recorded before the download, so a crash mid-download still leaves it ours)."""
    if repo not in state["downloads"]:
        state["downloads"][repo] = {"preexisting": hf_cache.is_cached(repo), "first_seen": oc.now(), "removed": False}
        save_state(state)
    return state["downloads"][repo]


def free_weights(repo):
    """Remove only the *.safetensors weight files of `repo` (snapshot links, the repo's blob entries and, in the shared blob store,
    blobs no other repo references); configs and tokenizer files stay. Returns GB freed."""
    d = os.path.join(hf_cache.HUB, hf_cache.repo_dir(repo))
    freed = 0
    for link in glob.glob(os.path.join(d, "snapshots", "*", "*.safetensors")):
        blob = os.path.normpath(os.path.join(os.path.dirname(link), os.readlink(link))) if os.path.islink(link) else None
        if blob and os.path.islink(blob):  # repo blob -> shared store blob with a .refs file
            shared = os.path.normpath(os.path.join(os.path.dirname(blob), os.readlink(blob)))
            refs = shared + ".refs"
            me = os.path.relpath(blob, hf_cache.HUB)
            lines = [x.strip() for x in open(refs)] if os.path.exists(refs) else []
            others = [x for x in lines if x and x != me]
            if others:
                open(refs, "w").write("\n".join(others) + "\n")
            else:
                if os.path.exists(shared):
                    freed += os.path.getsize(shared)
                    os.remove(shared)
                for f in (refs, shared + ".lock"):
                    if os.path.exists(f):
                        os.remove(f)
            os.remove(blob)
        elif blob and os.path.exists(blob):
            freed += os.path.getsize(blob)
            os.remove(blob)
        os.remove(link)
    return freed / 1e9


def hub_revision(repo):
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(repo).sha
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e.__class__.__name__}"


# --- freeze the conversation inputs ------------------------------------------------------------------------------------------

def config_provenance(family):
    from huggingface_hub import hf_hub_download

    out = {}
    for stage, repo in config.family_settings(family)["stages"].items():
        path = hf_hub_download(repo, "config.json")
        cfg = json.load(open(path))
        limit = next((c.get("max_position_embeddings") for c in (cfg.get("text_config"), cfg) if c and c.get("max_position_embeddings")), None)
        out[stage] = {"repo": repo, "config_revision": path.split("/snapshots/")[1].split("/")[0], "config_sha256": sha256_file(path),
                      "max_position_embeddings": limit}
    return out


def freeze(family, state, dry_run):
    """Prepare (or confirm) the family's inputs manifest, record config provenance; returns the frozen manifest path or None."""
    out = conv_dir(family)
    manifest = os.path.join(out, "inputs_manifest.json")
    cmd = [PY, "-m", "core.score_conversations", "--family", family, "--candidates", CANDIDATES, "--prepare-only"]
    if dry_run:
        oc.log(f"[dry run] freeze {family}: {' '.join(cmd)}" + (" (manifest exists: fingerprint must match)" if os.path.exists(manifest) else ""))
        return manifest
    r = subprocess.run(cmd, capture_output=True, text=True)
    oc.log(f"freeze {family}: {(r.stdout + r.stderr).strip()[-600:]}")
    if r.returncode:
        state["families"][family] = {"frozen": False, "error": (r.stdout + r.stderr)[-2000:]}
        save_state(state)
        return None
    m = json.load(open(manifest))
    if os.path.normpath(m["candidate_path"]) != os.path.normpath(CANDIDATES):
        raise SystemExit(f"{manifest}: candidates {m['candidate_path']} != {CANDIDATES}")
    cp_path = os.path.join(out, "config_provenance.json")
    cp = config_provenance(family)
    for stage, v in cp.items():
        if v["max_position_embeddings"] != m["context_limits"][stage]["limit"]:
            raise SystemExit(f"{family} {stage}: config limit {v['max_position_embeddings']} != frozen {m['context_limits'][stage]['limit']}")
    if os.path.exists(cp_path):
        old = json.load(open(cp_path))["stages"]
        if old != cp:
            raise SystemExit(f"{cp_path}: a stage config changed since the inputs were frozen ({old} vs {cp}); stop and decide")
    else:
        with open(cp_path, "w") as f:
            json.dump({"recorded": oc.now(), "note": "config.json revision per stage at freeze time; the scored stages use these limits "
                       "(--context-limits-from)", "stages": cp}, f, indent=1, sort_keys=True)
    state["families"][family] = {"frozen": True, "fingerprint": m["input_fingerprint"], "n_targets": len(m["targets"]),
                                 "n_exclusions": len(m["exclusions"]), "manifest": manifest, "config_provenance": cp_path}
    save_state(state)
    return manifest


# --- validation --------------------------------------------------------------------------------------------------------------

def validate_conv(out, stage, family, manifest_path, target_ids=None, expect_revision=None):
    """Integrity of one stage's conversation scores against the frozen manifest: fingerprint, target set, exact spans and token ids
    (re-rendered from the candidates), finite NLL/entropy within [0, ln V], per-target summaries recomputed from the token table."""
    issues, stats = [], {}
    manifest = json.load(open(manifest_path))
    paths = {k: os.path.join(out, f"{stage}{s}") for k, s in (("csv", ".csv"), ("pq", "_tokens.parquet"), ("prov", "_provenance.json"))}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        return dict(ok=False, issues=[f"missing {missing}"], stats=stats)
    prov = json.load(open(paths["prov"]))
    if prov["input_fingerprint"] != manifest["input_fingerprint"]:
        issues.append("provenance fingerprint differs from the frozen manifest")
    if prov.get("dtype") != "float32":
        issues.append(f"dtype {prov.get('dtype')}")
    stats["model"], stats["model_revision"] = prov.get("model"), prov.get("model_revision")
    if prov.get("model") != config.family_settings(family)["stages"][stage]:
        issues.append(f"model {prov.get('model')} is not the configured {stage}")
    if expect_revision and not str(expect_revision).startswith("unavailable") and prov.get("model_revision") != expect_revision:
        issues.append(f"model revision {prov.get('model_revision')} != current {expect_revision}")
    want = {t["target_id"]: t for t in manifest["targets"] if target_ids is None or t["target_id"] in target_ids}
    df = pd.read_csv(paths["csv"])
    pq = pd.read_parquet(paths["pq"])
    stats.update(targets=len(df), tokens=len(pq), expected_targets=len(want))
    if set(df.target_id) != set(want) or len(df) != len(want):
        issues.append(f"targets {len(df)} != expected {len(want)}")
    if prov.get("n_targets") not in (len(df), None) or prov.get("n_tokens") not in (len(pq), None):
        issues.append("provenance counts disagree with the tables")
    for _, r in df.iterrows():
        t = want.get(r.target_id)
        if t is None:
            continue
        if (r.input_sha256, int(r.span_start), int(r.span_end), int(r.n_input_tokens)) != (t["input_sha256"], t["start"], t["end"], t["n_input_tokens"]):
            issues.append(f"{r.target_id}: span/input differs from the manifest")
            break
        if int(r.n_tokens) != t["end"] - t["start"]:
            issues.append(f"{r.target_id}: n_tokens {r.n_tokens} != span length")
            break
    if len(pq) != int(df.n_tokens.sum()):
        issues.append(f"token rows {len(pq)} != sum n_tokens {int(df.n_tokens.sum())}")
    for col in ("nll", "entropy"):
        bad = int((~np.isfinite(pq[col].to_numpy(dtype=float))).sum())
        if bad:
            issues.append(f"{bad} non-finite {col}")
    # exact history/target alignment: re-render the targets from the candidates; the token table must hold exactly the target span ids
    try:
        fam = config.family_settings(family)
        tok = load_tokenizer(fam["template"])
        _, targets = sc.load_candidates(CANDIDATES)
        limits = sc.frozen_limits(manifest_path, fam["stages"])
        included, _ = sc.prepare([t for t in targets if t["target_id"] in want], tok, sc.PROFILES[family], min(n for n, _ in limits.values()))
        by = {r["target_id"]: r for r in included}
        groups = pq.groupby("target_id", observed=True)
        mism = 0
        for tid, t in want.items():
            r = by.get(tid)
            if r is None or sc.sha256(r["input_ids"]) != t["input_sha256"]:
                mism += 1
                continue
            g = groups.get_group(tid).sort_values("position")
            if g.token_id.tolist() != r["input_ids"][r["start"]:r["end"]] or g.position.tolist() != list(range(r["end"] - r["start"])):
                mism += 1
        stats["alignment_mismatches"] = mism
        if mism:
            issues.append(f"{mism} targets whose token ids/positions differ from the re-rendered inputs")
        lnv = math.log(len(tok))
        stats["max_entropy"], stats["ln_vocab"] = float(pq.entropy.max()), lnv
        if pq.entropy.min() < -1e-4 or pq.entropy.max() > lnv + 0.05:
            issues.append(f"entropy outside [0, ln V]: {pq.entropy.min():.4f}..{pq.entropy.max():.4f}")
        if pq.nll.min() < -1e-4:
            issues.append("negative NLL")
    except Exception as e:  # noqa: BLE001
        issues.append(f"alignment check failed: {e!r}")
    # summaries recomputed from the token table
    agg = pq.groupby("target_id", observed=True).agg(n=("nll", "size"), nll=("nll", "mean"), ent=("entropy", "mean"))
    from1 = pq[pq.position >= 1].groupby("target_id", observed=True).agg(nll1=("nll", "mean"), ent1=("entropy", "mean"))
    agg.index, from1.index = agg.index.astype(str), from1.index.astype(str)
    j = df.set_index("target_id").join(agg).join(from1)
    tol = 1e-4
    bad = (j.n != j.n_tokens) | ((j.mean_nll - j.nll).abs() > tol) | ((j.mean_entropy - j.ent).abs() > tol) | ((j.gap - (j.nll - j.ent)).abs() > 2 * tol)
    d1 = j[j.from1_defined.astype(bool)]
    bad1 = ((d1.mean_nll_from1 - d1.nll1).abs() > tol) | ((d1.mean_entropy_from1 - d1.ent1).abs() > tol)
    stats["summary_mismatches"] = int(bad.sum() + bad1.sum())
    if stats["summary_mismatches"]:
        issues.append(f"{stats['summary_mismatches']} per-target summaries disagree with the token table")
    for col in ("mean_nll", "mean_entropy", "gap"):
        if (~np.isfinite(df[col].to_numpy(dtype=float))).any():
            issues.append(f"non-finite {col}")
    stats["by_role"] = {k: {"targets": int(len(g)), "tokens": int(g.n_tokens.sum())} for k, g in df.groupby("role")}
    return dict(ok=not issues, issues=issues, stats=stats)


def gen_settings(family):
    fam = config.family_settings(family)
    tok = load_tokenizer(fam["template"])
    return {"prefill": PREFILL, "prefill_sha256": sha256_file(PREFILL), "samples": config.GEN_SAMPLES, "max_new_tokens": config.GEN_MAX_NEW_TOKENS,
            "temperature": config.GEN_TEMPERATURE, "top_p": 1.0, "top_k": 0, "seed": config.SEED, "precision": fam["gen_precision"],
            "preceding_turn": config.PRECEDING_TURN, "save_ids": True, "tokenizer": fam["template"],
            "tokenizer_revision": sc.cached_revision(fam["template"], "tokenizer_config.json"),
            "chat_template_sha256": sc.sha256((tok.chat_template or "").encode())}


def write_gen_provenance(family, stage, out, cmd, seconds):
    repo = config.family_settings(family)["stages"][stage]
    prov = {"format": "generations_incomplete_v1", "family": family, "stage": stage, "model": repo,
            "model_revision": sc.cached_revision(repo, "config.json"), "settings": gen_settings(family), "cmd": " ".join(cmd),
            "output_sha256": sha256_file(os.path.join(out, f"{stage}.jsonl")), "written": oc.now(), "seconds": seconds,
            "git_head": git("rev-parse", "HEAD"), "code_sha256": {p: sha256_file(p) for p in GEN_CODE},
            "environment": {"python": platform.python_version(), **{p: version(p) for p in ("transformers", "tokenizers", "torch")}}}
    with open(os.path.join(out, f"{stage}_provenance.json"), "w") as f:
        json.dump(prov, f, indent=1, sort_keys=True)


def validate_gen(out, family, stage, prefill_ids=None, expect_revision=None):
    path, ppath = os.path.join(out, f"{stage}.jsonl"), os.path.join(out, f"{stage}_provenance.json")
    if not os.path.exists(path):
        return dict(ok=False, issues=["missing output"], stats={})
    if not os.path.exists(ppath):
        return dict(ok=False, issues=["output exists without provenance: not reusable, not overwritten (review by hand)"], stats={})
    pre = load_jsonl(PREFILL)
    if prefill_ids is not None:
        pre = [p for p in pre if p["id"] in prefill_ids]
    tmp_prefill = None
    if prefill_ids is not None:  # check_generations reads a prefill file: give it the subset
        tmp_prefill = os.path.join(out, ".prefill_subset.jsonl")
        with open(tmp_prefill, "w") as f:
            for p in pre:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    res = oc.check_generations(path, tmp_prefill or PREFILL)
    if tmp_prefill:
        os.remove(tmp_prefill)
    issues, stats = list(res["issues"]), dict(res["stats"])
    prov = json.load(open(ppath))
    want = gen_settings(family)
    if prov.get("settings") != want:
        issues.append(f"settings differ from the current ones: {[k for k in want if prov.get('settings', {}).get(k) != want[k]]}")
    if prov.get("model") != config.family_settings(family)["stages"][stage]:
        issues.append(f"model {prov.get('model')} is not the configured {stage}")
    if prov.get("output_sha256") != sha256_file(path):
        issues.append("output changed after its provenance was written")
    if expect_revision and not str(expect_revision).startswith("unavailable") and prov.get("model_revision") != expect_revision:
        issues.append(f"model revision {prov.get('model_revision')} != current {expect_revision}")
    rows = load_jsonl(path)
    text = {p["id"]: p["prefill"] for p in pre}
    if any(r["prefill"] != text.get(r["prefill_id"]) for r in rows):
        issues.append("a record's prefill text differs from the prefill file")
    tok = load_tokenizer(config.family_settings(family)["template"])
    stats["decode_mismatches"] = sum(tok.decode(r["continuation_ids"], skip_special_tokens=False) != r["continuation"] for r in rows)
    if stats["decode_mismatches"]:
        issues.append(f"{stats['decode_mismatches']} continuations differ from their decoded ids")
    stats["model_revision"] = prov.get("model_revision")
    return dict(ok=not issues, issues=issues, stats=stats)


# --- one checkpoint ----------------------------------------------------------------------------------------------------------

def strict_wait_for_gpu(minutes=None):
    """Replaces overnight_collect.wait_for_gpu: never start alongside another process on the GPU; wait as long as it takes."""
    t0, last = time.time(), -1e9
    while True:
        busy = oc.gpu_busy()
        if not busy:
            return
        if time.time() - last > 600:
            oc.log(f"GPU in use by other processes ({busy}); waiting (waited {(time.time() - t0) / 60:.0f} min)")
            last = time.time()
        time.sleep(60)


oc.wait_for_gpu = strict_wait_for_gpu


# Conversation scoring runs one sequence per forward pass (fp32 unchanged): inputs reach ~4.1k tokens and OLMo 3 7B ran out of memory at
# batch 8 and 4 on the 48 GB card (2026-11-03). Batch 1 also removes padding and batch composition, identically for every stage.
CONV_BATCH = 1


def conv_cmd(family, stage, manifest, out="{OUT}", batch=None):
    bs = batch or CONV_BATCH
    return [PY, "-m", "core.score_conversations", "--family", family, "--candidates", CANDIDATES, "--model", stage, "--out", out,
            "--context-limits-from", manifest, "--batch-size", str(bs)]


def collect(family, stage, manifest, state, dry_run):
    repo = config.family_settings(family)["stages"][stage]
    key = f"{family}/{stage}"
    ck = state["checkpoints"].setdefault(key, {"family": family, "stage": stage, "repo": repo})
    ck["attempt_started"] = oc.now()
    rev = hub_revision(repo)
    ck["hub_revision"] = rev
    cout, gout = conv_dir(family), gen_dir(family)
    tasks = {"conversation": dict(out=cout, names=[f"{stage}_provenance.json", f"{stage}.csv", f"{stage}_tokens.parquet"],
                                  cmd=conv_cmd(family, stage, manifest), batch_arg=True),
             "incomplete": dict(out=gout, names=[f"{stage}.jsonl"], cmd=oc.gen_cmd(family, GEN_RUN, stage, PREFILL), batch_arg=None)}
    if not manifest:  # the family's inputs could not be frozen: conversation scoring stays blocked (recorded by main), generation runs
        del tasks["conversation"]
    exists = {"conversation": any(os.path.exists(os.path.join(cout, n)) for n in tasks.get("conversation", {}).get("names", [])),
              "incomplete": any(os.path.exists(os.path.join(gout, n)) for n in tasks["incomplete"]["names"] + [f"{stage}_provenance.json"])}
    todo = []
    for name in tasks:
        if exists.get(name):
            v = (validate_conv(cout, stage, family, manifest, expect_revision=rev) if name == "conversation"
                 else validate_gen(gout, family, stage, expect_revision=rev)) if not dry_run or os.path.exists(manifest) else None
            if v and v["ok"]:
                ck[name] = {"status": "reused (validated)", "validation": v}
                oc.log(f"{key} {name}: existing output validated, reused")
            else:
                ck[name] = {"status": "failed: existing output does not validate (never overwritten)", "validation": v}
                oc.log(f"{key} {name}: existing output does NOT validate: {v and v['issues']}")
        else:
            todo.append(name)
    if dry_run:
        for name in tasks:
            oc.log(f"[dry run] {key} {name}: " + (" ".join(tasks[name]["cmd"]).replace("{OUT}", tasks[name]["out"]) if name in todo else ck[name]["status"]))
        oc.log(f"[dry run] {key}: then remove weights only if this run downloaded them ({'already cached: never removed' if hf_cache.is_cached(repo) else 'not cached: would download'})")
        return ck
    save_state(state)
    if not todo:
        cleanup(repo, state, ck)
        return ck
    own = note_ownership(state, repo)
    need, free, ok = hf_cache.check_disk(repo)
    oc.log(f"{key}: disk need {need:.1f} GB, free {free:.1f} GB -> {'OK' if ok else 'NOT ENOUGH'}")
    if not ok:
        for name in todo:
            ck[name] = {"status": f"blocked: not enough disk (need {need:.1f} GB, free {free:.1f} GB); nothing freed"}
        save_state(state)
        return ck
    for name in todo:
        t = tasks[name]
        ck[name] = {"status": "running", "started": oc.now()}
        save_state(state)
        try:
            r = oc.run_task(f"{family}_{stage}_{name}", t["cmd"], t["out"], t["names"], False, batch_arg=t["batch_arg"])
            if r["status"].startswith("ok") and name == "incomplete":
                write_gen_provenance(family, stage, gout, oc.gen_cmd(family, GEN_RUN, stage, PREFILL), r["seconds"])
            v = None
            if r["status"].startswith("ok"):
                v = (validate_conv(cout, stage, family, manifest, expect_revision=rev) if name == "conversation"
                     else validate_gen(gout, family, stage, expect_revision=rev))
            status = (r["status"] if v and v["ok"] else f"failed validation: {v['issues']}" if v else r["status"])
            ck[name] = {"status": status, "run": r, "validation": v, "log": os.path.join(LOGS, f"{family}_{stage}_{name}.log")}
        except Exception as e:  # noqa: BLE001
            ck[name] = {"status": f"failed: {e!r}", "trace": traceback.format_exc()[-1500:]}
        oc.log(f"{key} {name}: {ck[name]['status']}")
        save_state(state)
    cleanup(repo, state, ck)
    return ck


def cleanup(repo, state, ck):
    own = state["downloads"].get(repo)
    if own is None or own["preexisting"]:
        ck["cleanup"] = "not removed: weights were cached before this run (not ours)" if own else "not removed: never downloaded by this run"
    else:
        gb = free_weights(repo)
        own["removed"], own["removed_at"] = True, oc.now()
        ck["cleanup"] = f"removed this run's weight files: {gb:.1f} GB (configs and tokenizer kept)"
    oc.log(f"{ck['family']}/{ck['stage']}: {ck['cleanup']}")
    save_state(state)


# --- smoke test ----------------------------------------------------------------------------------------------------------------

def smoke_conv_worker(family, stage, manifest, out):
    """Score one conversation's targets (first conversation_id with a scored target in each role) into `out`, in this process."""
    import torch
    from core.score import describe_placement, load_model

    fam = config.family_settings(family)
    m = json.load(open(manifest))
    _, targets = sc.load_candidates(CANDIDATES)
    limits = sc.frozen_limits(manifest, fam["stages"])
    tok = load_tokenizer(fam["template"])
    included, _ = sc.prepare(targets, tok, sc.PROFILES[family], min(n for n, _ in limits.values()))
    by_conv = {}
    for r in included:
        by_conv.setdefault(r["conversation_id"], []).append(r)
    cid = next(c for c in sorted(by_conv) if {r["role"] for r in by_conv[c]} == {"user", "assistant"})
    rows = by_conv[cid]
    model_id = fam["stages"][stage]
    model, device = load_model(model_id, torch.float32, offload=fam["offload"], gpu_gib=fam["gpu_gib"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    nll, ent = sc.score_targets(model, device, rows, pad_id, CONV_BATCH)
    tokens, per_target = sc.tables(rows, nll, ent)
    os.makedirs(out, exist_ok=True)
    tokens.to_parquet(os.path.join(out, f"{stage}_tokens.parquet"), index=False)
    per_target.to_csv(os.path.join(out, f"{stage}.csv"), index=False, float_format="%.6f")
    with open(os.path.join(out, f"{stage}_provenance.json"), "w") as f:
        json.dump({"format": sc.FORMAT, "smoke": True, "conversation_id": cid, "stage": stage, "model": model_id,
                   "model_revision": sc.cached_revision(model_id, "config.json"), "family": family, "dtype": "float32",
                   "device": str(device), "placement": describe_placement(model), "batch_size": CONV_BATCH,
                   "input_fingerprint": m["input_fingerprint"], "n_targets": len(rows), "n_tokens": len(tokens)}, f, indent=1)
    print(json.dumps({"conversation_id": cid, "targets": [r["target_id"] for r in rows]}))


def smoke(family="olmo3-7b", stage="base", n_prefills=3):
    """Separate smoke directories; nothing here touches the real outputs. Uses an already cached checkpoint (no download, no cleanup)."""
    repo = config.family_settings(family)["stages"][stage]
    if not hf_cache.is_cached(repo):
        raise SystemExit(f"smoke test expects {repo} to be cached already")
    state = load_state()
    manifest = freeze(family, state, False)
    if not manifest:
        raise SystemExit("freeze failed")
    strict_wait_for_gpu()
    report = {"family": family, "stage": stage, "repo": repo, "time": oc.now()}
    cout, gout = os.path.join(SMOKE, family, "conversation_scores"), os.path.join(SMOKE, family, "generations_incomplete")
    for d in (cout, gout):
        if os.path.exists(os.path.join(d, f"{stage}_provenance.json")) or os.path.exists(os.path.join(d, f"{stage}.jsonl")):
            raise SystemExit(f"{d} already holds smoke outputs; remove that smoke directory first")
    r = subprocess.run(oc.die_with_parent([PY, os.path.abspath(__file__), "--smoke-worker", family, stage, manifest, cout]), capture_output=True, text=True)
    open(os.path.join(SMOKE, f"{family}_{stage}_conversation.log"), "w").write(r.stdout + r.stderr)
    if r.returncode:
        raise SystemExit(f"smoke conversation scoring failed; see {SMOKE}/{family}_{stage}_conversation.log")
    sel = json.loads(r.stdout.strip().splitlines()[-1])
    report["conversation"] = sel | {"validation": validate_conv(cout, stage, family, manifest, set(sel["targets"]))}
    cmd = oc.gen_cmd(family, GEN_RUN, stage, PREFILL)
    cmd[cmd.index("--out") + 1] = gout
    cmd += ["--limit", str(n_prefills)]
    t0 = time.time()
    r = subprocess.run(oc.die_with_parent(cmd), capture_output=True, text=True)
    open(os.path.join(SMOKE, f"{family}_{stage}_incomplete.log"), "w").write(r.stdout + r.stderr)
    if r.returncode:
        raise SystemExit(f"smoke generation failed; see {SMOKE}/{family}_{stage}_incomplete.log")
    write_gen_provenance(family, stage, gout, cmd, round(time.time() - t0, 1))
    ids = {p["id"] for p in load_jsonl(PREFILL)[:n_prefills]}
    report["incomplete"] = {"prefill_ids": sorted(ids), "validation": validate_gen(gout, family, stage, ids)}
    report["ok"] = report["conversation"]["validation"]["ok"] and report["incomplete"]["validation"]["ok"]
    with open(os.path.join(SMOKE, "smoke_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps(report, indent=1, default=str))
    return report["ok"]


# --- status ------------------------------------------------------------------------------------------------------------------

def expected_keys(only=None):
    return [(f, s) for f, stages in PLAN for s in stages if not only or f in only]


def status():
    state = load_state()
    st = {}
    try:
        st = json.load(open(os.path.join(LOGS, "status.json")))
    except (OSError, ValueError):
        pass
    print(f"driver: pid {st.get('pid')} task {st.get('task')} heartbeat {st.get('heartbeat')} (now {oc.now()})")
    if st.get("pid"):
        print(f"driver alive: {os.path.exists(f'/proc/{st['pid']}')}")
    for f, v in state.get("families", {}).items():
        print(f"inputs {f}: frozen={v.get('frozen')} targets={v.get('n_targets')} exclusions={v.get('n_exclusions')} fp={str(v.get('fingerprint'))[:16]}")
    done = 0
    for f, s in expected_keys():
        ck = state["checkpoints"].get(f"{f}/{s}", {})
        cells = []
        for name in ("conversation", "incomplete"):
            t = ck.get(name, {})
            ok = bool(t.get("validation") and t["validation"]["ok"])
            done += ok
            extra = ""
            if ok and name == "conversation":
                extra = f" ({t['validation']['stats']['targets']} targets, {t['validation']['stats']['tokens']} tokens)"
            if ok and name == "incomplete":
                extra = f" ({t['validation']['stats']['rows']} samples)"
            cells.append(f"{name}: {t.get('status', 'pending')}{extra}")
        print(f"{f}/{s}: " + " | ".join(cells) + (f" | {ck['cleanup']}" if ck.get("cleanup") else ""))
    print(f"validated tasks: {done} / {2 * len(expected_keys())}" + (" -> COMPLETE" if done == 2 * len(expected_keys()) else ""))


# --- main --------------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--only", help="comma-separated families")
    ap.add_argument("--smoke-worker", nargs=4, metavar=("FAMILY", "STAGE", "MANIFEST", "OUT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.status:
        return status()
    if args.smoke_worker:
        return smoke_conv_worker(*args.smoke_worker)
    os.makedirs(LOGS, exist_ok=True)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    lock = oc.take_lock()  # noqa: F841  (also during a dry run or smoke: one instance at a time)
    if args.smoke:
        sys.exit(0 if smoke() else 1)
    only = args.only.split(",") if args.only else None
    state = load_state()
    if not args.dry_run:
        oc.load_summary()
        oc.start_heartbeat()
    oc.log(f"collection start (dry run: {args.dry_run}); branch {git('branch', '--show-current')} HEAD {git('rev-parse', '--short', 'HEAD')}")
    manifests = {}
    for fam in dict.fromkeys(f for f, _ in expected_keys(only)):  # freeze every family's inputs before any scoring
        manifests[fam] = freeze(fam, state, args.dry_run)
    for fam, stage in expected_keys(only):
        if not manifests[fam]:
            ck = state["checkpoints"].setdefault(f"{fam}/{stage}", {"family": fam, "stage": stage})
            ck["conversation"] = {"status": "blocked: the family's inputs could not be frozen"}
            oc.log(f"{fam}/{stage}: conversation blocked (inputs not frozen); generation still runs")
        try:
            collect(fam, stage, manifests[fam] or "", state, args.dry_run)
        except Exception as e:  # noqa: BLE001
            oc.log(f"{fam}/{stage}: unexpected error {e!r}\n{traceback.format_exc()[-1500:]}")
    if not args.dry_run:
        for fam in manifests:
            if manifests[fam]:
                r = subprocess.run([PY, "-m", "core.score_conversations", "--family", fam, "--out", conv_dir(fam), "--check-stages"], capture_output=True, text=True)
                state["families"][fam]["check_stages"] = (r.stdout + r.stderr).strip()[-800:]
                oc.log(f"check-stages {fam}: {state['families'][fam]['check_stages']}")
        save_state(state)
        oc.SUMMARY["finished"] = oc.now()
        oc.save_summary()
    oc.log("collection pass finished; see --status (complete only if every expected task validated)")


if __name__ == "__main__":
    main()
