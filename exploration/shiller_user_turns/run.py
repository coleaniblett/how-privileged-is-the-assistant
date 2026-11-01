"""Run everything on the GPU box: verify, generate per model (one process each), clear cache, report.

    uv sync --extra quant                                              # bitsandbytes, for int8
    uv run python -m exploration.shiller_user_turns.run                          # small models
    uv run python -m exploration.shiller_user_turns.run --models gemma4-31b-it   # precision auto -> int8 on 48 GB
    uv run python -m exploration.shiller_user_turns.run --models "small gemma4-31b-it qwen3-32b gemma4-31b-base"

Per model: HF API verification (all models first, so a bad ID fails before any download), a free-disk check
against the weight size, generation in its own process (GPU memory fully released between models), then
the HF-cache entries this run downloaded are deleted if --clear-cache covers the model ("large" by default;
repos that were already cached before the run are never deleted). A model that fails is reported and
skipped; the report is built from every <model>.jsonl present at the end.
"""

import argparse
import os
import subprocess
import sys

from exploration.shiller_user_turns import hub, models
from exploration.shiller_user_turns.generate import OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="small", help="groups (small, large, all) and/or model keys")
    ap.add_argument("--precision", default="auto", choices=["auto", "bf16", "int8"])
    ap.add_argument("--clear-cache", default="large", choices=["large", "all", "none"])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--skip-existing", action="store_true", help="skip models whose <model>.jsonl exists")
    args, passthrough = ap.parse_known_args()  # e.g. --samples, --max-new-tokens, --limit go to generate

    keys = models.resolve_models(args.models)
    for k in keys:  # fail fast on IDs/templates
        r = models.verify(k)
        print(f"verified {k}: {r['repo']} @ {r['sha'][:10]} ({r['weights_gb']} GB)")

    failed = {}
    for k in keys:
        repo, tmpl = models.MODELS[k]["repo"], models.MODELS[k]["template"]
        if args.skip_existing and os.path.exists(os.path.join(args.out, f"{k}.jsonl")):
            print(f"=== {k}: exists, skipped")
            continue
        before = hub.cached_repos()
        try:
            need, free = hub.check_disk(repo)
            print(f"=== {k}: {repo} (disk: need {need:.1f} GB, {free:.1f} GB free)", flush=True)
            subprocess.run([sys.executable, "-m", "exploration.shiller_user_turns.generate", "--model", k,
                            "--precision", args.precision, "--out", args.out, *passthrough], check=True)
        except (SystemExit, subprocess.CalledProcessError) as e:
            failed[k] = str(e)
            print(f"!!! {k} failed: {e}", flush=True)
        finally:
            if args.clear_cache == "all" or (args.clear_cache == "large" and models.MODELS[k]["large"]):
                new = {repo, tmpl} - before
                print(f"cleared {sorted(new)} from HF cache ({hub.clear(new):.1f} GB)", flush=True)

    subprocess.run([sys.executable, "-m", "exploration.shiller_user_turns.report", "--out", args.out], check=False)
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
