"""Spec M (2026-11-05): judge-free trait atlas on Llama 3.1 8B and Apertus 8B (base and final). A BREADTH CHECK, added after the atlas results were known, so not blind. Exploratory, outside every verdict.
Same frozen materials, roles, frames, sample counts, generation settings, detectors and bootstrap as the atlas (data/trait_atlas_materials.json, extras/trait_atlas.py), restricted to the detector-scored traits
markdown_lists, says_ai (main comparison), stock_phrases, length, refusal_borderline (also reported) and arithmetic, multistep_reasoning, factual_accuracy (descriptive only: the end-of-reply skill rule has misled us).
No paid API, no local judge. The atlas module is not edited: the two new families are registered at run time.
    python -m extras.trait_atlas_breadth example --family llama3.1-8b
    python -m extras.trait_atlas_breadth gen --family llama3.1-8b --stage instruct [--limit N]    # GPU
    python -m extras.trait_atlas_breadth detect                                                     # CPU
    python -m extras.trait_atlas_breadth analyze
"""
import argparse
import glob
import gzip
import json
import os
import types

import numpy as np
import pandas as pd

from core import config
from extras import trait_atlas as ta

OUT = os.environ.get("TAB_OUT", "results/overgeneralization/trait_atlas_breadth")
FAMS = ["llama3.1-8b", "apertus-8b"]
MAIN, ALSO, DESC = ["markdown_lists", "says_ai"], ["stock_phrases", "length", "refusal_borderline"], ["arithmetic", "multistep_reasoning", "factual_accuracy"]
TRAITS = MAIN + ALSO + DESC
ta.TEMPLATE_KW.setdefault("llama3.1-8b", {})
ta.TEMPLATE_KW.setdefault("apertus-8b", {})
_plan = ta.plan_requests

# Apertus uses a distinct token per role ("<|user_start|>", "<|assistant_start|>", ...), so the template-derived turn markers fail (turn_open is None). Cut strings given explicitly.
from measures.overgeneralization import user_traits as _ut  # noqa: E402
_orig_markers = _ut.family_markers
APERTUS_MARKERS = ["<|user_end|>", "<|assistant_end|>", "<|user_start|>", "<|assistant_start|>", "<|system_start|>", "<|developer_start|>"]


def _family_markers(family):
    return list(APERTUS_MARKERS) if family == "apertus-8b" else _orig_markers(family)


_ut.family_markers = _family_markers


def plan_requests(mat):
    return [r for r in _plan(mat) if r["trait"] in TRAITS]


def setup(fam):
    config.use_family(fam)
    ta.OUT = OUT
    ta.plan_requests = plan_requests
    ta.replies_path = lambda family, stage, kind="replies", ext="jsonl.gz": os.path.join(OUT, family, f"{stage}_{kind}.{ext}")


def example(fam):
    from core.render import load_tokenizer
    mat = ta.load_materials()
    setup(fam)
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    seen = set()
    for r in plan_requests(mat):
        if r["context"] in seen or r["trait"] != "markdown_lists":
            continue
        seen.add(r["context"])
        kind, text = ta.render_request(mat, r, tok, fam)
        print(f"--- {fam} / {r['context']} ({kind}) ---\n{text}\n")


def gen(a):
    setup(a.family)
    ta.generate(types.SimpleNamespace(limit=a.limit, mem_fraction=None, model_id=None, model=a.stage, offload=False, gpu_gib=None, batch_size=a.batch_size))


def detect(a):
    ta.OUT = OUT
    ta.detect(types.SimpleNamespace(only=None))


def analyze(a):
    mat = ta.load_materials()
    ta.OUT = OUT
    V = ta.load_values()
    V = V[~V.trait.str.endswith("__alt")]
    C = ta.analyze_cells(V, mat, families={f: ["base", "instruct"] for f in FAMS})
    C = C[C.trait.isin(TRAITS) & ~C.trait.str.endswith("__alt")]
    C.to_csv(os.path.join(OUT, "trait_atlas_breadth_cells.csv"), index=False)
    main = C[~C.flagged_excluded]
    caps = []
    for p in sorted(glob.glob(os.path.join(OUT, "*", "*_replies.jsonl.gz"))):
        fam, stage = p.split(os.sep)[-2], os.path.basename(p).split("_")[0]
        d = pd.DataFrame([{k: r.get(k) for k in ("context", "capped", "trait")} for r in ta.read_gz(p)])
        d["family"], d["stage"] = fam, stage
        caps.append(d)
    caps = pd.concat(caps) if caps else pd.DataFrame()

    def row(fam, trait, ctx):
        x = main[(main.family == fam) & (main.stage == "instruct") & (main.trait == trait) & (main.context == ctx)]
        return None if x.empty else x.iloc[0]

    split_rows, heads = [], []
    for fam in FAMS:
        for ctx in ("light_system", "light_user", "strong_act"):
            res = {}
            for trait in MAIN:
                a, c = row(fam, trait, "assistant"), row(fam, trait, ctx)
                if a is None or c is None:
                    res[trait] = None
                    continue
                g = ta.gate(a.change, a.lo, a.hi, a.unit)
                rd = ta.reading(c.change, c.lo, c.hi, a.change, a.unit) if g else "n/a"
                res[trait] = (g, rd, a, c)
            md, ai = res["markdown_lists"], res["says_ai"]
            if md is None or ai is None:
                verdict = "no data"
            elif not md[0] or not ai[0]:
                verdict = "CANNOT TEST: " + " and ".join(t for t, r in (("markdown_lists", md), ("says_ai", ai)) if not r[0]) + " fails the gate in the Assistant"
            elif md[1] == "spreads" and ai[1] != "spreads":
                verdict = "SPLIT HOLDS (formatting spreads, self-image does not)"
            elif md[1] == "spreads" and ai[1] == "spreads":
                verdict = "SPLIT FAILS: self-image travels (both spread)"
            elif md[1] != "spreads":
                verdict = "SPLIT FAILS: formatting stays"
            else:
                verdict = "inconclusive"
            heads.append(f"| {fam} | {ctx} | {'passes' if md and md[0] else 'fails'} / {md[1] if md else '-'} | {'passes' if ai and ai[0] else 'fails'} / {ai[1] if ai else '-'} | {verdict} |")
            split_rows.append(dict(family=fam, context=ctx, verdict=verdict))
    L = ["# Judge-free trait atlas on Llama 3.1 8B and Apertus 8B (spec M; BREADTH CHECK, added after the atlas results were known: NOT BLIND; exploratory, outside every verdict)", "",
         "Plan: `docs/decisions.md` 2026-11-05. Same frozen materials, frames, sample counts, generation settings, detectors, gate (10 points; 0.2 log words) and spread rule (5 points, ratio 0.25) as the atlas, 95% two-way bootstrap over items and personas; "
         "detector-scored traits only (no judge, no API). Changes are final minus base, in percentage points (log words for length). Contamination cannot be checked for these families; not hand-checked.", "",
         "## Split reading (final): formatting spreads to the light role, self-image does not", "",
         "| family | light-role position | markdown_lists (Assistant gate / light-role reading) | says_ai (Assistant gate / light-role reading) | split |", "|---|---|---|---|---|"] + heads
    L += ["", "Limits: `says_ai` is a single regex and only one of the atlas's three self-image traits (`denies_feelings` and `declines_opinion` need a judge and are not here), so this is thinner than the atlas's three-model self-image finding; "
          "a format ceiling is possible (a light role already near 100% markdown at base cannot rise); regex measures see form only; few roles, many cells, no correction for multiple tests.", ""]
    L += ["## Levels and change by trait and context (final minus base; base level in brackets)", "", "| family | trait | context | base level | change [95% CI] | gate (Assistant) / reading |", "|---|---|---|---|---|---|"]
    for fam in FAMS:
        for trait in TRAITS:
            a = row(fam, trait, "assistant")
            for ctx in ta.CONTEXTS:
                c = row(fam, trait, ctx)
                if c is None:
                    continue
                fmt = (lambda v: f"{v:+.1f}") if c.unit == "pts" else (lambda v: f"{v:+.2f}")
                note = ""
                if trait in DESC:
                    note = "descriptive only"
                elif ctx == "assistant":
                    note = "gate " + ("passes" if ta.gate(c.change, c.lo, c.hi, c.unit) else "fails")
                elif a is not None:
                    note = ta.reading(c.change, c.lo, c.hi, a.change, c.unit) if ta.gate(a.change, a.lo, a.hi, a.unit) else "Assistant gate fails"
                L.append(f"| {fam} | {trait}{' (desc.)' if trait in DESC else ''} | {ctx} | {c.base_level:.1f} | {fmt(c.change)} [{fmt(c.lo)}, {fmt(c.hi)}] | {note} |")
    if len(caps):
        hit = caps.groupby(["family", "stage", "context"]).capped.mean().reset_index()
        hit = hit[hit.capped > 0.5]
        L += ["", "## Cells where most replies hit the token limit (256)", "", ("none" if hit.empty else "\n".join(f"- {r.family} {r.stage} {r.context}: {100 * r.capped:.0f}%" for r in hit.itertuples())), ""]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "trait_atlas_breadth.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    pd.DataFrame(split_rows).to_csv(os.path.join(OUT, "trait_atlas_breadth_split.csv"), index=False)
    smp = []
    for p in sorted(glob.glob(os.path.join(OUT, "*", "*_replies.jsonl.gz"))):
        fam, stage = p.split(os.sep)[-2], os.path.basename(p).split("_")[0]
        d = pd.DataFrame(ta.read_gz(p))
        for ctx, g in d.groupby("context"):
            smp.append(g.sample(min(30, len(g)), random_state=0).assign(family=fam, stage=stage)[["family", "stage", "context", "trait", "item", "persona", "text", "capped"]])
    if smp:
        pd.concat(smp).to_csv(os.path.join(OUT, "trait_atlas_breadth_replies_sample.csv.gz"), index=False)
    print("\n".join(L[:20]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["example", "gen", "detect", "analyze"])
    ap.add_argument("--family", choices=FAMS)
    ap.add_argument("--stage", choices=["base", "instruct"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=24)
    a = ap.parse_args()
    if a.cmd == "example":
        example(a.family)
    elif a.cmd == "gen":
        gen(a)
    elif a.cmd == "detect":
        detect(a)
    else:
        analyze(a)


if __name__ == "__main__":
    main()
