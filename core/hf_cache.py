"""Disk-space check before a download and blob-aware removal of a checkpoint from the HF cache afterwards.

This machine's HF cache keeps file contents in hub/blobs/<xx>/<sha>, each with a <sha>.refs file listing the
"<repo dir>/blobs/<sha>" entries that use it, so removing a repo's folder alone frees nothing. free_repo() also
removes every blob whose refs name only this repo and rewrites (never deletes) the refs of blobs shared with other
repos. With the standard per-repo layout there are no .refs files and removing the folder frees everything.
Ported from measures/vestigial_regression/pilot/nll_arm/hf_cache.py.

    python -m core.hf_cache check REPO            # exit 1 if the weights do not fit (exit 0 if already cached)
    python -m core.hf_cache free REPO [--dry-run]
    python -m core.hf_cache cached REPO           # exit 0 if the weights are in the cache, 1 if not
"""

import argparse
import glob
import os
import shutil
import sys

HUB = os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub"))
MARGIN_GB = 5


def repo_dir(repo):
    return "models--" + repo.replace("/", "--")


def free_gb():
    os.makedirs(HUB, exist_ok=True)
    return shutil.disk_usage(HUB).free / 1e9


def is_cached(repo):
    """True if the repo has at least one weight file whose blob exists."""
    snaps = glob.glob(os.path.join(HUB, repo_dir(repo), "snapshots", "*"))
    return any(os.path.exists(f) for s in snaps for f in glob.glob(os.path.join(s, "*.safetensors")))


def weights_gb(repo):
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, files_metadata=True)
    return sum((s.size or 0) for s in info.siblings if s.rfilename.endswith(".safetensors")) / 1e9


def check_disk(repo):
    """(need GB incl. margin, free GB, ok). A cached repo needs nothing."""
    if is_cached(repo):
        return 0.0, free_gb(), True
    need = weights_gb(repo) + MARGIN_GB
    free = free_gb()
    return need, free, free >= need


def free_repo(repo, dry_run=False):
    """Returns (GB freed or to be freed from the shared blob store, blobs removed, shared blobs kept)."""
    d = repo_dir(repo)
    freed, removed, kept = 0, 0, 0
    for refs in glob.glob(os.path.join(HUB, "blobs", "*", "*.refs")):
        lines = [x.strip() for x in open(refs) if x.strip()]
        if not any(x.startswith(d + "/") for x in lines):
            continue
        blob = refs[:-5]
        others = [x for x in lines if not x.startswith(d + "/")]
        if others:
            kept += 1
            if not dry_run:
                open(refs, "w").write("\n".join(others) + "\n")
        else:
            freed += os.path.getsize(blob) if os.path.exists(blob) else 0
            removed += 1
            if not dry_run:
                for f in (blob, refs, blob + ".lock"):
                    if os.path.exists(f):
                        os.remove(f)
    folder = os.path.join(HUB, d)
    if os.path.isdir(folder):
        if not dry_run:
            local = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(folder) for f in fs
                        if not os.path.islink(os.path.join(r, f)))
            freed += local
            shutil.rmtree(folder)
    return freed / 1e9, removed, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["check", "free", "cached"])
    ap.add_argument("repo")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.action == "cached":
        sys.exit(0 if is_cached(a.repo) else 1)
    if a.action == "check":
        need, free, ok = check_disk(a.repo)
        print(f"{a.repo}: need {need:.1f} GB (incl. {MARGIN_GB} GB margin), {free:.1f} GB free -> {'OK' if ok else 'NOT ENOUGH'}")
        sys.exit(0 if ok else 1)
    freed, removed, kept = free_repo(a.repo, a.dry_run)
    print(f"{a.repo}: freed {freed:.1f} GB; {removed} blobs removed, {kept} shared blobs kept" + (" (dry run)" if a.dry_run else ""))


if __name__ == "__main__":
    main()
