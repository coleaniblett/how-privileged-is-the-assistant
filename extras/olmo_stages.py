"""Spec R4 (2026-11-06): which training stage does it? OLMo 3 7B base -> SFT -> DPO -> final. Exploratory, outside every verdict, NOT BLIND, descriptive (no verdict).
Part 1 (CPU): from saved atlas data, a table by stage for markdown_lists (old and strict layout), says_ai and denies_feelings in the Assistant, light roles and full role-play.
Part 2 (GPU, new generation for SFT and DPO): L2 (all) and at most 1,500 L3 rows without a harmless instruction (same rows and settings as spec L), and spec Q's 12 conditions + plain Assistant on the denies_feelings, markdown_lists and stock_phrases sets only.
Outputs under results/overgeneralization/overnight_2026-11-05/R4/ (replies stay local).
    python -m extras.olmo_stages gen --stage sft        # GPU: L2 + L3 (capped) and Q subset
    python -m extras.olmo_stages gen-l23 --stage sft    # GPU: only L2/L3
    python -m extras.olmo_stages gen-q --stage sft      # GPU: only the Q subset
    python -m extras.olmo_stages analyze                # CPU
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from extras import role_says as rs
from extras import working_model_l as wl

OUT = "results/overgeneralization/overnight_2026-11-05/R4"
FAM = "olmo3-7b"
QSETS = ["denies_feelings", "markdown_lists", "stock_phrases"]
CAP, SEED = 1500, 0


def gen_l23(stage):
    mat = wl.load()
    model, tok, device = wl.load_model_tok(FAM, stage)
    reqs, qs = mat["requests"], mat["l3"]["questions"]
    d = os.path.join(OUT, FAM)
    os.makedirs(d, exist_ok=True)
    for tag, max_new in (("L2", wl.L2_NEW), ("L3", wl.L3_NEW)):
        out = os.path.join(d, f"{stage}_{tag}.csv.gz")
        if os.path.exists(out):
            continue
        rows = []
        for wording in ("sys", "user"):
            for role in wl.roles_list():
                for ri, item in enumerate(reqs if tag == "L2" else qs):
                    for v in ([True, False] if tag == "L2" else [None]):
                        msgs = wl.l2_msgs(wording, role, item, v) if tag == "L2" else wl.l3_msgs(wording, role, item, mat, v)
                        for smp in range(wl.SAMPLES):
                            rows.append(dict(tag=tag, wording=wording, kind=wl.kind_of(role), role="Assistant" if role is None else role[1], item=ri, line=(v if tag == "L2" else None), instr=None, sample=smp, msgs=msgs))
        if tag == "L3" and len(rows) > CAP:
            keep = np.sort(np.random.default_rng(SEED).choice(len(rows), CAP, replace=False))
            rows = [rows[i] for i in keep]
        print(f"{FAM} {stage} {tag}: {len(rows)} generations", flush=True)
        t0 = time.time()
        rows = wl.generate_rows(model, tok, device, rows, max_new)
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "msgs"} for r in rows])
        df["instr"] = df.instr.fillna("none")
        df.to_csv(out, index=False)
        print(f"{FAM} {stage} {tag}: done in {time.time() - t0:.0f} s", flush=True)
    del model


def gen_q(stage):
    mat = rs.load()
    rows = [r for r in rs.all_rows(mat) if mat["questions"][r["qi"]]["set"] in QSETS]
    print(f"Q subset: {len(rows)} rows", flush=True)
    if not os.path.exists(os.path.join(OUT, FAM, f"{stage}_replies.csv.gz")):
        rs.gen(FAM, stage, rows=rows, out_dir=OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "gen-l23", "gen-q", "analyze"])
    ap.add_argument("--stage", choices=["sft", "dpo"])
    a = ap.parse_args()
    if a.cmd in ("gen", "gen-l23"):
        gen_l23(a.stage)
    if a.cmd in ("gen", "gen-q"):
        gen_q(a.stage)
    if a.cmd == "analyze":
        analyze()


STAGES = ["base", "sft", "dpo", "instruct"]
LBL = {"base": "base", "sft": "SFT", "dpo": "DPO", "instruct": "final"}


def first_half(vals):
    """vals: dict stage -> level (points). -> name of the first stage whose change from base reaches at least half of the base-to-final change; none if that change is under 10 points."""
    if any(vals.get(s) is None or vals[s] != vals[s] for s in ("base", "instruct")):
        return "n/a"
    if any(vals.get(s) is None or vals[s] != vals[s] for s in ("sft", "dpo")):
        return "n/a (SFT or DPO value missing)"
    d = vals["instruct"] - vals["base"]
    if abs(d) < 10:
        return f"not named (base to final change {d:+.1f} is under 10 points)"
    for s in ("sft", "dpo", "instruct"):
        v = vals.get(s)
        if v is not None and v == v and (v - vals["base"]) / d >= 0.5:
            return LBL[s]
    return "n/a"


def analyze():
    from extras import refusal_judge as rj
    from extras import strict_layout as sl
    from extras import trait_atlas as ta
    from extras.persona_leakage import REFUSAL_RE
    rows, rec = [], []

    def add(measure, ctx, vals, ic=None, note=""):
        fh = first_half(vals)
        rows.append(f"| {measure} | {ctx} | " + " | ".join("-" if vals.get(s) is None or vals.get(s) != vals.get(s) else f"{vals[s]:.0f}" for s in STAGES) + f" | **{fh}** | " + (" / ".join("-" if ic.get(s) is None or ic.get(s) != ic.get(s) else f"{ic[s]:.0f}%" for s in STAGES) if ic else "not saved for SFT/DPO") + f" | {note} |")
        rec.append(dict(measure=measure, context=ctx, **{LBL[s]: vals.get(s) for s in STAGES}, first_half=fh))
    # Part 1: atlas (saved)
    R1 = pd.read_csv("results/overgeneralization/overnight_2026-11-05/R1/R1_scored_replies.csv.gz")
    R1 = R1[(R1.source == "atlas") & (R1.family == FAM)]
    ctxs = {"assistant": ["assistant"], "light role": ["light_system", "light_user"], "full role-play": ["strong_act"]}
    icf = {}
    for st in ("base", "instruct"):
        p = f"results/overgeneralization/trait_atlas_incharacter/atlas_{FAM}_{st}.csv.gz"
        if os.path.exists(p):
            icf[st] = pd.read_csv(p)
    for col, nm in (("old", "markdown_lists (old regex)"), ("strict", "markdown_lists (strict layout)")):
        for cn, cl in ctxs.items():
            add(nm, cn, {s: 100 * R1[(R1.stage == s) & R1.context.isin(cl)][col].mean() for s in STAGES})
    for trait, kind in (("says_ai", "det"), ("denies_feelings", "judge")):
        for cn, cl in ctxs.items():
            v = {}
            for s in STAGES:
                d = pd.read_csv(f"results/overgeneralization/trait_atlas/{FAM}/{s}_{'detected' if kind == 'det' else 'judged'}.csv.gz")
                d = d[(d.trait == trait) & d.context.isin(cl)]
                v[s] = 100 * d.value.mean()
            add(f"{trait} (atlas)", cn, v)
    # Part 2: spec Q subset by stage
    mat = rs.load()
    Q = {}
    for s in STAGES:
        p = f"results/overgeneralization/role_says/{FAM}/{s}_judged.csv.gz" if s in ("base", "instruct") else os.path.join(OUT, FAM, f"{s}_judged.csv.gz")
        if os.path.exists(p):
            Q[s] = rs.values(pd.read_csv(p), mat)
            sc = np.array([sl.score(t) for t in Q[s].text.fillna("")], dtype=bool)
            Q[s]["strict"] = sc[:, 1].astype(float)
    if Q:
        for tr, col, nm in (("denies_feelings", "val", "denies_feelings"), ("markdown_lists", "strict", "markdown_lists (strict)"), ("markdown_lists", "md", "markdown_lists (old regex)"), ("stock_phrases", "stock", "stock_phrases")):
            for cn, sel in (("plain Assistant", lambda d: (d.cond == "assistant")), ("role conditions, all", lambda d: (d.cond == "role")), ("AI role, voice none", lambda d: (d.cond == "role") & (d.identity == "ai") & (d.voice == "none"))):
                v = {s: (100 * Q[s][(Q[s].set == tr) & sel(Q[s])][col].mean() if s in Q else None) for s in STAGES}
                ic = {s: (100 * Q[s][(Q[s].cond == "role") & Q[s].judge_ic.notna()].judge_ic.mean() if s in Q else None) for s in STAGES} if cn != "plain Assistant" else None
                add(f"Q {nm}", cn, v, ic, "spec Q subset")
    # Part 3: refusal (L2, 60 tokens): regex, and R2's judge if validated
    L2 = {}
    for s in STAGES:
        p = f"results/overgeneralization/working_model_L/L23/{FAM}/{s}_L2.csv.gz" if s in ("base", "instruct") else os.path.join(OUT, FAM, f"{s}_L2.csv.gz")
        if os.path.exists(p):
            d = pd.read_csv(p)
            d["regex"] = [float(bool(REFUSAL_RE.search(str(t)[:160]))) for t in d.text.fillna("")]
            d["line"] = d.line.astype(str)
            L2[s] = d
    jv = None
    pj = os.path.join("/root/r2_work", "merged_olmo3-7b.pkl")
    ps = os.path.join("/root/r2_work", "judged_olmo_stages.csv.gz")
    valid = os.path.exists("results/overgeneralization/overnight_2026-11-05/R2/R2.md") and "judge passes" in open("results/overgeneralization/overnight_2026-11-05/R2/R2.md", encoding="utf-8").read()
    if valid and os.path.exists(pj) and os.path.exists(ps):
        a = pd.read_pickle(pj)
        a = a[a.test == "L2"][["stage", "wording", "kind", "line", "item", "role", "sample", "declined"]]
        b = pd.read_csv(os.path.join("/root/r2_work", "inputs_olmo_stages.csv.gz")).merge(pd.read_csv(ps)[["rid", "judge"]], on="rid")
        b = b[b.test == "L2"].copy()
        b["declined"] = (b.judge == "declined").astype(float)
        b["line"] = b.line.astype(str)
        jv = pd.concat([a.assign(line=a.line.astype(str)), b[["stage", "wording", "kind", "line", "item", "role", "sample", "declined"]]])
    for kind in ("assistant", "thin", "vivid"):
        for ln in ("False", "True"):
            if kind == "assistant" and ln == "True":
                pass
            v = {s: (100 * L2[s][(L2[s].kind == kind) & (L2[s].line == ln)].regex.mean() if s in L2 else None) for s in STAGES}
            add("refusal L2 (regex)", f"{kind}, {'with' if ln == 'True' else 'no'} 'never refuses' line", v, None, "60-token replies")
            if jv is not None:
                v = {s: (100 * jv[(jv.stage == s) & (jv.kind == kind) & (jv.line == ln)].declined.mean() if (jv.stage == s).any() else None) for s in STAGES}
                add("refusal L2 (R2 judge, declined)", f"{kind}, {'with' if ln == 'True' else 'no'} 'never refuses' line", v, None, "base capped at 1,500 replies")
    L = ["# R4: which training stage does it? OLMo 3 7B (spec R; exploratory, outside every verdict, NOT BLIND, descriptive: no verdict)", "",
         "Plan: `docs/decisions.md` 2026-11-06 (spec R4, `docs/spec_R_overnight.md`). Levels are % of replies by stage (base, SFT, DPO, final = Instruct). **First stage reaching at least half of the base-to-final change** is named only when that change is 10 points or more; it is not a test and has no interval. "
         "Part 1 (atlas table) comes from saved atlas data (and R1's strict scoring); the SFT and DPO rows of Parts 2 and 3 are newly generated here (L2 all; L3 capped at 1,500; spec Q's 12 conditions + plain Assistant on the denies_feelings, markdown_lists and stock_phrases sets only). "
         "In-character share (the atlas's in-character files for base and final, and spec Q's judged 30% sample for the Q rows): the atlas's in-character judgements were not saved for SFT and DPO, so they are marked; refusal in-character shares are not available. "
         "Limits: one family, single checkpoints per stage; few roles; many cells, no multiple-test correction; regex sees form only; local judge provisional and not hand-checked; contamination checkable (OLMo) but not applied; the stages differ in data and method, so 'first stage' does not name a cause.", "",
         "| measure | context | base | SFT | DPO | final | first stage reaching half of the base-to-final change | in-character share base / SFT / DPO / final | note |", "|---|---|---|---|---|---|---|---|---|"] + rows + [""]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "R4.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    pd.DataFrame(rec).to_csv(os.path.join(OUT, "R4_table.csv"), index=False)
    print("\n".join(L[4:40]))


if __name__ == "__main__":
    main()
