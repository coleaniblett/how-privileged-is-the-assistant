"""Spec J5 (POST-HOC, 2026-11-05): enough in-character base replies. In the in-character addendum 46 gated light-role cells were "too few" because base rarely plays the role (3-13% of
base replies). Here base (OLMo 3 7B, Qwen3-8B) is sampled K more times for each light-role request of the atlas's gated traits (same items, personas, wordings and frames as the atlas; new sample
indices 1..K; same generation settings), scored with the same detectors / local judge, the in-character question is judged with the same frozen judge, and the addendum's analysis is rerun
UNCHANGED on the pooled data (atlas sample 0 + the new samples at base; final stages as before). Atlas and addendum files are not overwritten. No paid API.
    python -m extras.trait_atlas_more_base gen --family olmo3-7b          # GPU: base samples -> results/overgeneralization/trait_atlas_more_base/<family>/base_replies.jsonl.gz
    python -m extras.trait_atlas_more_base detect                          # CPU: detector traits
    python -m extras.trait_atlas_more_base judge                           # GPU: Gemma 4 31B-it trait judge + in-character judge on the new replies
    python -m extras.trait_atlas_more_base analyze                         # pool + addendum analysis -> results/overgeneralization/trait_atlas_more_base_ic/in_character.md
"""
import argparse
import glob
import os
import shutil
import types

import pandas as pd

from core import config
from extras import trait_atlas as ta
from extras import trait_atlas_incharacter as ic

SRC_TA, SRC_IC = "results/overgeneralization/trait_atlas", "results/overgeneralization/trait_atlas_incharacter"
MORE, MORE_IC = "results/overgeneralization/trait_atlas_more_base", "results/overgeneralization/trait_atlas_more_base_ic"
COMB, COMB_IC = "results/overgeneralization/trait_atlas_more_base_combined", "results/overgeneralization/trait_atlas_more_base_combined_ic"
K = int(os.environ.get("MORE_K", 5))
FAMS = ["olmo3-7b", "qwen3-8b"]


def gated_traits(fam):
    c = pd.read_csv(os.path.join(SRC_IC, "in_character.csv"))
    a = c[(c.part == "atlas") & (c.family == fam) & c.context.isin(ic.LIGHT) & (c.gated == True)]      # noqa: E712
    return sorted(a.trait.unique())


def plan(mat, fam):
    traits = set(gated_traits(fam))
    reqs = []
    for t in mat["traits"]:
        if t["id"] not in traits:
            continue
        for ii, it in enumerate(t["items"]):
            for ctx in ic.LIGHT:
                for p in range(6):
                    for s in range(1, K + 1):
                        reqs.append(dict(trait=t["id"], item=it["id"], question=it["question"], context=ctx, wording=(ii + p) % 2, persona=p, sample=s))
    return reqs


def gen(a):
    fam = a.family
    config.use_family(fam)
    ta.plan_requests = lambda mat: plan(mat, fam)
    ta.replies_path = lambda family, stage, kind="replies", ext="jsonl.gz": os.path.join(MORE, family, f"{stage}_{kind}.{ext}")
    ta.OUT = MORE
    ta.generate(types.SimpleNamespace(limit=a.limit, mem_fraction=None, model_id=None, model="base", offload=False, gpu_gib=None, batch_size=a.batch_size))


def detect(a):
    ta.OUT = MORE
    ta.detect(types.SimpleNamespace(only=None))


def judge(a):
    ta.OUT = MORE
    ta.judge(types.SimpleNamespace(limit=0, batch_size=a.batch_size, offload=False, gpu_gib=None))     # trait judge (and the strong-context meta judge, none here)
    ic.OUT = MORE_IC
    ic.LEAK = "/nonexistent"                                                                            # skip the leakage-matrix part
    ic.judge(types.SimpleNamespace(sample=0, batch_size=a.batch_size))


def cat(paths, out):
    frames = [pd.read_csv(p) for p in paths if os.path.exists(p) and os.path.getsize(p) > 5]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)


def analyze(a):
    for d in (COMB, COMB_IC):
        shutil.rmtree(d, ignore_errors=True)
    for p in glob.glob(os.path.join(SRC_TA, "*", "*_detected.csv.gz")) + glob.glob(os.path.join(SRC_TA, "*", "*_judged.csv.gz")) + glob.glob(os.path.join(SRC_TA, "*", "*_judged_meta.csv.gz")):
        fam, name = p.split(os.sep)[-2], os.path.basename(p)
        extra = os.path.join(MORE, fam, name) if name.startswith("base_") and not name.endswith("_meta.csv.gz") else None
        cat([p] + ([extra] if extra else []), os.path.join(COMB, fam, name))
    for p in glob.glob(os.path.join(SRC_IC, "atlas_*.csv.gz")):
        name = os.path.basename(p)
        extra = os.path.join(MORE_IC, name) if name.endswith("_base.csv.gz") else None
        cat([p] + ([extra] if extra else []), os.path.join(COMB_IC, name))
    ta.OUT, ic.OUT, ic.LEAK = COMB, COMB_IC, "/nonexistent"
    ic.analyze(types.SimpleNamespace())
    path = os.path.join(COMB_IC, "in_character.md")
    body = open(path, encoding="utf-8").read()
    head = ("**Spec J5 (POST-HOC, 2026-11-05):** base was sampled %d more times per light-role request of the atlas's gated traits (OLMo, Qwen; Gemma not extended), pooled with the atlas's single sample; "
            "everything else is the in-character addendum's analysis unchanged (readings as defined there; `too few` = fewer than 6 items with an in-character reply on both sides). "
            "Plan: `docs/decisions.md` 2026-11-05.\n\n") % K
    open(path, "w", encoding="utf-8").write(body.replace("\n\n", "\n\n" + head, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "detect", "judge", "analyze"])
    ap.add_argument("--family", choices=FAMS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=24)
    a = ap.parse_args()
    {"gen": gen, "detect": detect, "judge": judge, "analyze": analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
