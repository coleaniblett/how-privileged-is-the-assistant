"""HF cache helpers: free disk, what is cached, and deleting one repo from the cache."""

import os
import shutil

import huggingface_hub as hf
from huggingface_hub import HfApi
from huggingface_hub.constants import HF_HUB_CACHE

MARGIN_GB = 5


def free_gb():
    os.makedirs(HF_HUB_CACHE, exist_ok=True)
    return shutil.disk_usage(HF_HUB_CACHE).free / 1e9


def cached_gb(repo):
    """Size on disk of a repo in the cache (0 if absent)."""
    try:
        return sum(r.size_on_disk for r in hf.scan_cache_dir().repos if r.repo_id == repo) / 1e9
    except hf.CacheNotFound:
        return 0.0


def cached_repos():
    try:
        return {r.repo_id for r in hf.scan_cache_dir().repos}
    except hf.CacheNotFound:
        return set()


def weights_gb(repo):
    info = HfApi().model_info(repo, files_metadata=True)
    return sum((s.size or 0) for s in info.siblings if s.rfilename.endswith(".safetensors")) / 1e9


def check_disk(repo):
    """Raise if the repo's weights (minus what is already cached) don't fit with a margin."""
    need = max(0.0, weights_gb(repo) - cached_gb(repo)) + MARGIN_GB
    free = free_gb()
    if free < need:
        raise SystemExit(f"not enough disk for {repo}: need {need:.1f} GB (incl. {MARGIN_GB} GB margin), "
                         f"{free:.1f} GB free in {HF_HUB_CACHE}")
    return need, free


def clear(repos):
    """Delete these repos from the HF cache; returns GB freed."""
    freed = 0.0
    try:
        cache = hf.scan_cache_dir()
    except hf.CacheNotFound:
        return 0.0
    for r in cache.repos:
        if r.repo_id in repos:
            freed += r.size_on_disk / 1e9
            shutil.rmtree(r.repo_path)
    return freed
