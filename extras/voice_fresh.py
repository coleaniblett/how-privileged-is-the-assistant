"""Spec R5 (2026-11-06): does a voice remove layout on fresh questions? Q's 12 conditions (position x identity x voice) plus the plain Assistant, on L4's frozen 90 fresh questions (30 factual, 30 how-to, 30 advice),
final checkpoints of OLMo 3 7B, Qwen3-8B and Gemma 4 31B. Each question gets 2 of the 6 light jobs by a frozen rotation (jobs (i mod 6) and ((i + 3) mod 6)). Scored: old layout regex, strict layout (extras/strict_layout.py), stock phrases;
in-character judge (atlas prompt, light-role note) on a 30% sample per condition, cells under 30% in character are not read. Exploratory, outside every verdict, NOT BLIND (Q's claim came first). No paid API.
    python -m extras.voice_fresh build                       # data/voice_fresh_materials.json
    python -m extras.voice_fresh example --family F
    python -m extras.voice_fresh gen --family F             # GPU, final
    python -m extras.voice_fresh judge                      # GPU (vinyl): in-character sample for every replies file
    python -m extras.voice_fresh analyze                    # CPU
"""
import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from core import config
from extras import role_says as rs
from extras import strict_layout as sl
from extras import trait_atlas as ta
from extras import trait_atlas_incharacter as ic

MATERIALS = "data/voice_fresh_materials.json"
OUT = os.environ.get("VF_OUT", "results/overgeneralization/overnight_2026-11-05/R5")
SEED, N_BOOT = 0, 4000
FAMS = rs.FAMS
PLAIN_SAMPLES = 3


def build():
    wm = json.load(open("data/working_model_L_materials.json"))
    qs = [dict(qtype=t, question=q) for t in ("factual", "howto", "advice") for q in wm["l4"][t]]
    atlas = json.load(open("data/trait_atlas_materials.json"))
    jobs = atlas["light"]
    mat = dict(seed=SEED, questions=qs, jobs=jobs, rotation="question i gets jobs (i mod 6) and ((i + 3) mod 6)", identity=rs.IDENT, voice=rs.VOICE, positions=rs.POS, order="You are {job}. {identity} {voice}",
               generation=rs.GEN, plain_samples=PLAIN_SAMPLES, conditions=f"90 questions x 2 jobs x 12 conditions = {90 * 2 * 12} role replies + {90 * PLAIN_SAMPLES} plain Assistant replies per checkpoint",
               in_character="atlas meta_judge speaks_as_character, 30% random sample per condition, cells under 30% in character not read",
               readings=dict(layout="voice none minus given, unstated identity, strict layout: voice removes layout at least 30 points with range excluding 0; voice doesn't under 10 with range inside +-20; else in between",
                             stock="stock phrases: voice removes it at least 20 points, range excluding 0; voice doesn't under 10, range inside +-20; else in between",
                             claim="claim holds (voice changes wording but not layout) = stock phrases removed AND layout not (voice doesn't), same family and position; flag any cell with no room (every condition at or above 95 or at or below 5, or both compared cells there)"))
    json.dump(mat, open(MATERIALS, "w"), indent=1, ensure_ascii=False)
    print("wrote", MATERIALS, mat["conditions"])


def load():
    return json.load(open(MATERIALS, encoding="utf-8"))


def all_rows(mat):
    rows = []
    for qi, q in enumerate(mat["questions"]):
        for smp in range(PLAIN_SAMPLES):
            rows.append(dict(cond="assistant", job=-1, position="-", identity="-", voice="-", qi=qi, sample=smp, msgs=[{"role": "user", "content": q["question"]}]))
        for ji in (qi % 6, (qi + 3) % 6):
            for pos in rs.POS:
                for idn in rs.IDENT:
                    for vo in rs.VOICE:
                        t = rs.prompt_text(mat["jobs"][ji], idn, vo)
                        msgs = [{"role": "system", "content": t}, {"role": "user", "content": q["question"]}] if pos == "system" else [{"role": "user", "content": t + "\n\n" + q["question"]}]
                        rows.append(dict(cond="role", job=ji, position=pos, identity=idn, voice=vo, qi=qi, sample=0, msgs=msgs))
    return rows


def example(fam):
    from core.render import load_tokenizer
    mat = load()
    config.use_family(fam)
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    kw = ta.TEMPLATE_KW.get(fam, {})
    q = mat["questions"][0]["question"]
    for nm, msgs in (("plain Assistant", [{"role": "user", "content": q}]),
                     ("system, AI, none", [{"role": "system", "content": rs.prompt_text(mat["jobs"][0], "ai", "none")}, {"role": "user", "content": q}]),
                     ("user, unstated, given", [{"role": "user", "content": rs.prompt_text(mat["jobs"][0], "unstated", "given") + "\n\n" + q}]),
                     ("system, human, given", [{"role": "system", "content": rs.prompt_text(mat["jobs"][0], "human", "given")}, {"role": "user", "content": q}])):
        print(f"--- {fam} / {nm} ---\n{tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)}\n")


def gen(fam, limit=0):
    mat = load()
    rs.gen(fam, "instruct", limit, rows=all_rows(mat), out_dir=OUT)


def in_sample(key):
    return int(hashlib.sha256(("vf" + key).encode()).hexdigest(), 16) % 100 < 30


def judge():
    import torch
    from core.render import encode, load_tokenizer
    from core.score import load_model
    mat = load()
    atlas = ta.load_materials()
    config.use_family("gemma4-31b")
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    tok.padding_side = "left"
    yes, no = encode(tok, "Yes")[0], encode(tok, "No")[0]
    model, device = load_model(config.MODELS["instruct"], torch.bfloat16)
    qs = mat["questions"]
    for p in sorted(glob.glob(os.path.join(OUT, "*", "instruct_replies.csv.gz"))):
        dst = p.replace("_replies.csv.gz", "_judged.csv.gz")
        if os.path.exists(dst):
            continue
        d = pd.read_csv(p)
        d["text"] = d.text.fillna("")
        pick = [i for i, r in d.iterrows() if r.cond == "role" and in_sample(f"{r.job}.{r.position}.{r.identity}.{r.voice}.{r.qi}")]
        full = [tok.apply_chat_template([{"role": "user", "content": ic.prompt_for(atlas, qs[int(d.qi[i])]["question"], d.text[i], ic.NOTE_LIGHT.format(role=mat["jobs"][int(d.job[i])]))}], tokenize=False, add_generation_prompt=True) for i in pick]
        order = sorted(range(len(full)), key=lambda k: len(full[k]))
        py = np.full(len(full), np.nan)
        for s in range(0, len(order), 16):
            b = order[s:s + 16]
            enc = tok([full[k] for k in b], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
            py[b] = torch.stack([lg[:, yes], lg[:, no]], 1).softmax(-1)[:, 0].cpu().numpy()
        d["judge_ic"] = np.nan
        for i, v in zip(pick, py):
            d.loc[i, "judge_ic"] = float(v >= 0.5)
        d.drop(columns=["text"]).to_csv(dst, index=False)
        print("judged", p, len(pick), flush=True)


def cls(m, kind):
    mean, lo, hi = m
    if mean != mean:
        return "n/a"
    thr = 30 if kind == "layout" else 20
    return "VOICE REMOVES IT" if mean >= thr and lo > 0 else ("VOICE DOESN'T" if abs(mean) < 10 and lo >= -20 and hi <= 20 else "in between")


def analyze():
    rng = np.random.default_rng(SEED)
    mat = load()
    fmt = lambda m: "n/a" if m[0] != m[0] else f"{m[0]:+.1f} [{m[1]:+.1f}, {m[2]:+.1f}]"
    L = ["# R5: does a voice remove layout on fresh questions? (spec R; exploratory, outside every verdict, NOT BLIND)", "",
         "Plan: `docs/decisions.md` 2026-11-06 (spec R5, `docs/spec_R_overnight.md`); frozen materials `data/voice_fresh_materials.json`. Q's 12 conditions on L4's 90 fresh questions (30 factual, 30 how-to, 30 advice), 2 of the 6 light jobs per question (frozen rotation), final checkpoints, one sample per role condition and 3 for the plain Assistant. "
         "Layout = strict (at least 2 list lines, a heading or a table; bold alone does not count); old regex and stock phrases beside it. In-character judge (Gemma 4 31B-it, atlas prompt) on a 30% sample per condition; **a reading is NOT READ when either compared cell is under 30% in character**. Difference = voice none minus given (unstated identity), points, paired per question and job, 95% two-way bootstrap over questions and jobs. "
         "Limits: few jobs; many cells, no correction for multiple tests; regex sees form only; judge provisional and Gemma judges its own replies; contamination checkable only for OLMo.", ""]
    rows, rec = [], []
    for fam in FAMS:
        pr, pj = os.path.join(OUT, fam, "instruct_replies.csv.gz"), os.path.join(OUT, fam, "instruct_judged.csv.gz")
        if not os.path.exists(pr):
            continue
        d = pd.read_csv(pr)
        d["text"] = d.text.fillna("")
        sc = np.array([sl.score(t) for t in d.text], dtype=bool)
        d["old"], d["strict"], d["heavy"] = sc[:, 0].astype(float), sc[:, 1].astype(float), sc[:, 2].astype(float)
        d["stock"] = [float(ta.detect_one("stock_phrases", "stock", t, None)) for t in d.text]
        d["set"] = "all"
        share = None
        if os.path.exists(pj):
            j = pd.read_csv(pj)[["cond", "job", "position", "identity", "voice", "qi", "sample", "judge_ic"]]
            d = d.merge(j, on=["cond", "job", "position", "identity", "voice", "qi", "sample"], how="left")
            share = d[(d.cond == "role") & d.judge_ic.notna()].groupby(["position", "identity", "voice"]).judge_ic.mean() * 100
        else:
            d["judge_ic"] = np.nan
        body = [f"## {fam}", ""]
        if share is not None:
            body += ["### In-character share (30% sample per condition)", "", "| position | identity | voice | in character |", "|---|---|---|---|"] + [f"| {a} | {b} | {c} | {v:.0f}% |" for (a, b, c), v in share.items()] + [""]
        plain = {c: 100 * d[d.cond == "assistant"][c].mean() for c in ("old", "strict", "stock")}
        body += [f"Plain Assistant (%): old layout {plain['old']:.0f}, strict {plain['strict']:.0f}, stock phrases {plain['stock']:.0f}.", ""]
        for pos in rs.POS:
            body += [f"### {fam}, {pos} position", "", "| trait | plain | " + " | ".join(f"{i}/{v}" for i in rs.IDENT for v in rs.VOICE) + " | no room |", "|---|---|" + "---|" * 7]
            conds = [(i, v) for i in rs.IDENT for v in rs.VOICE]
            for c in ("old", "strict", "stock"):
                lv = [100 * d[(d.cond == "role") & (d.position == pos) & (d.identity == i) & (d.voice == v)][c].mean() for i, v in conds]
                nr = all(x >= 95 for x in lv + [plain[c]]) or all(x <= 5 for x in lv + [plain[c]])
                body.append(f"| {c} | {plain[c]:.0f} | " + " | ".join(f"{x:.0f}" for x in lv) + f" | {'**NO ROOM**' if nr else ''} |")
            body.append("")
            bad = [(i, v) for (p_, i, v), s_ in (share.items() if share is not None else []) if p_ == pos and i == "unstated" and s_ < 30]
            res = {}
            for c, kind in (("strict", "layout"), ("old", "layout"), ("stock", "stock")):
                x = d[(d.cond == "role") & (d.position == pos) & (d.identity == "unstated")]
                A = x[x.voice == "none"].pivot_table(index="qi", columns="job", values=c, aggfunc="mean")
                B = x[x.voice == "given"].pivot_table(index="qi", columns="job", values=c, aggfunc="mean")
                ix, cx = A.index.intersection(B.index), A.columns.intersection(B.columns)
                M = 100 * (A.loc[ix, cx] - B.loc[ix, cx]).to_numpy()
                m = ta.two_way_boot(M, rng, n_boot=N_BOOT)
                lv_n, lv_g = 100 * x[x.voice == "none"][c].mean(), 100 * x[x.voice == "given"][c].mean()
                ext = lambda v: v >= 95 or v <= 5
                note = " (NO ROOM: both cells at a ceiling or floor)" if ext(lv_n) and ext(lv_g) else ""
                res[c] = (m, cls(m, kind) + note if not bad else "NOT READ (a cell is under 30% in character)", lv_n, lv_g)
            claim = "n/a"
            if bad:
                claim = "NOT READ (a cell is under 30% in character)"
            else:
                lay, stk = res["strict"][1], res["stock"][1]
                claim = "CLAIM HOLDS (stock phrases removed, layout not)" if stk.startswith("VOICE REMOVES") and lay.startswith("VOICE DOESN'T") else "claim does not hold here (" + f"strict layout: {lay}; stock: {stk})"
            body += ["| measure | none level | given level | none minus given | reading |", "|---|---|---|---|---|"] + [f"| {c} | {res[c][2]:.0f} | {res[c][3]:.0f} | {fmt(res[c][0])} | {res[c][1]} |" for c in ("strict", "old", "stock")] + ["", f"**Claim (voice changes wording, not layout): {claim}**", ""]
            rec.append(dict(family=fam, position=pos, strict_diff=res["strict"][0][0], strict_lo=res["strict"][0][1], strict_hi=res["strict"][0][2], strict_reading=res["strict"][1], old_diff=res["old"][0][0], stock_diff=res["stock"][0][0], stock_lo=res["stock"][0][1], stock_hi=res["stock"][0][2], stock_reading=res["stock"][1], claim=claim))
            rows.append(f"| {fam} | {pos} | {fmt(res['strict'][0])} | {res['strict'][1]} | {fmt(res['stock'][0])} | {res['stock'][1]} | **{claim}** |")
        L += body
    head = ["## Readings (final; voice none minus given, unstated identity)", "", "| family | position | strict layout | reading | stock phrases | reading | claim |", "|---|---|---|---|---|---|---|"] + rows + [""]
    out = L[:4] + head + L[4:]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "R5.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    pd.DataFrame(rec).to_csv(os.path.join(OUT, "R5_readings.csv"), index=False)
    smp = []
    for p in glob.glob(os.path.join(OUT, "*", "instruct_replies.csv.gz")):
        x = pd.read_csv(p)
        x["cell"] = x.cond + "/" + x.position + "/" + x.identity + "/" + x.voice
        smp += [g.sample(min(5, len(g)), random_state=0).assign(family=p.split(os.sep)[-2])[["family", "cond", "job", "position", "identity", "voice", "qi", "text"]] for _, g in x.groupby("cell")]
    if smp:
        pd.concat(smp).to_csv(os.path.join(OUT, "R5_replies_sample.csv.gz"), index=False)
    print("\n".join(out[:24]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "example", "gen", "judge", "analyze"])
    ap.add_argument("--family", choices=FAMS)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"build": build, "example": lambda: example(a.family), "gen": lambda: gen(a.family, a.limit), "judge": judge, "analyze": analyze}[a.cmd]()


if __name__ == "__main__":
    main()
