"""Free disk before a download and remove a checkpoint from this machine's shared HF blob store afterwards.

This machine's HF cache keeps file contents in hub/blobs/<xx>/<sha>, each with a <sha>.refs file listing the
"<repo dir>/blobs/<sha>" entries that use it. Removing a repo's folder alone frees nothing, so free_repo() also removes
every blob whose refs name only this repo, and rewrites (not deletes) the refs of blobs shared with other repos.

    python -m measures.vestigial_regression.pilot.nll_arm.hf_cache free google/gemma-4-31B-it [--dry-run]
    python -m measures.vestigial_regression.pilot.nll_arm.hf_cache check google/gemma-4-31B      # disk vs weights
"""

import argparse
import glob
import os
import shutil

HUB = os.path.expanduser("~/.cache/huggingface/hub")
MARGIN_GB = 5


def repo_dir(repo):
    return "models--" + repo.replace("/", "--")


def free_repo(repo, dry_run=False):
    """Returns (GB freed or to be freed, blobs removed, blobs shared and kept)."""
    d = repo_dir(repo)
    freed, removed, kept = 0, 0, 0
    for refs in glob.glob(f"{HUB}/blobs/*/*.refs"):
        lines = [x.strip() for x in open(refs) if x.strip()]
        mine = [x for x in lines if x.startswith(d + "/")]
        if not mine:
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
    if not dry_run and os.path.isdir(f"{HUB}/{d}"):
        shutil.rmtree(f"{HUB}/{d}")
    return freed / 1e9, removed, kept


def free_gb():
    return shutil.disk_usage(HUB).free / 1e9


def check_disk(repo):
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, files_metadata=True)
    need = sum((s.size or 0) for s in info.siblings if s.rfilename.endswith(".safetensors")) / 1e9 + MARGIN_GB
    free = free_gb()
    return need, free, free >= need


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["free", "check"])
    ap.add_argument("repo")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.action == "free":
        print("freed %.1f GB; %d blobs removed, %d shared blobs kept%s" % (*free_repo(a.repo, a.dry_run), " (dry run)" if a.dry_run else ""))
    else:
        need, free, ok = check_disk(a.repo)
        print(f"{a.repo}: need {need:.1f} GB (incl. {MARGIN_GB} GB margin), {free:.1f} GB free -> {'OK' if ok else 'NOT ENOUGH'}")
