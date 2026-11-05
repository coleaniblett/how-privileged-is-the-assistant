"""Spec J11: does full role-play really move a skill the other way? (atlas: arithmetic rises for the Assistant and light roles, falls in full role-play). Exploratory, outside every verdict.
No judge, no API. 40 fresh arithmetic questions with checkable answers; plain Assistant, 3 light roles (user turn), 4 expert and 4 non-expert role-play characters (user turn, atlas wording);
2 samples each; base and final of OLMo, Qwen, Gemma 31B; max 512 new tokens. Plan and readings: docs/decisions.md 2026-11-05. Materials: data/skill_roleplay_materials.json (frozen before scoring).
    python -m extras.skill_roleplay build
    python -m extras.skill_roleplay run --family olmo3-7b --stage instruct      # GPU
    python -m extras.skill_roleplay analyze
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd

from core import config
from extras import trait_atlas as ta

MATERIALS = "data/skill_roleplay_materials.json"
OUT = os.environ.get("SR_OUT", "results/skill_roleplay")
SEED, N_BOOT = 0, 4000
MAX_NEW, OLD_CAP, TAIL = 512, 256, 150
TEMP, TOP_P, SAMPLES, BATCH = 0.7, 0.95, 2, 16
FAMILIES = {"olmo3-7b": ["base", "instruct"], "qwen3-8b": ["base", "instruct"], "gemma4-31b": ["base", "instruct"]}
ACT = "Stay in character for the whole reply."
LIGHT = ["a ferry captain", "a village baker", "a long-haul trucker"]      # three of the atlas's light roles
# (name, role, sheet, kind). Same shape as the atlas's strong characters: voice, habits, a place; the expert ones work with numbers every day, the non-expert ones plainly do not.
CHARS = [
    ("Ingrid Solheim", "a ship's navigator", "Ingrid Solheim has plotted courses for cargo ships out of Bergen for twenty years. She speaks in short, dry sentences, checks everything twice, and keeps a brass dividers in her breast pocket. She hums sea shanties when the chart is clean and mutters distances in nautical miles when it is not.", "expert"),
    ("Rahul Menon", "a bank teller", "Rahul Menon has counted cash at the same counter in Kochi for thirty years. He is courteous to a fault, taps the counter twice before he says a total, and signs off every sentence with a small bow of the head. He loves cricket scores and strong coffee.", "expert"),
    ("Beatrix Whitlow", "a tailor's shop bookkeeper", "Beatrix Whitlow keeps the ledgers of a Savile Row tailor and has not made an error in them since 1991. She speaks crisply, peers over her half-moon glasses, and sharpens a pencil whenever a figure needs thinking about. She takes her tea with exactly two sugars.", "expert"),
    ("Tomas Ibarra", "a master carpenter", "Tomas Ibarra builds staircases in Valparaiso and measures everything three times before one cut. He talks slowly, with a pencil behind his ear, and explains things in lengths, angles and board feet. He is proud, patient and a little superstitious about Fridays.", "expert"),
    ("Wren Calloway", "a wandering poet", "Wren Calloway drifts between mountain villages trading verses for supper. She talks in images and half-rhymes, never wears a watch, and finds numbers rather rude. She collects feathers, forgets appointments and laughs at her own metaphors.", "nonexpert"),
    ("Fabrizio Lombardi", "a gondolier and singer", "Fabrizio Lombardi has poled the same stretch of canal in Venice since boyhood and sings to every passenger whether they ask or not. He talks with big gestures, drops opera into ordinary sentences and believes any problem can be improved by a song.", "nonexpert"),
    ("Odalys Pryce", "a ballet dancer", "Odalys Pryce dances with a touring company and thinks mostly in movement. She speaks in a breathless, dreamy way, stretches her calves while she talks, and describes everything as light or heavy. She is superstitious about ribbons and never reads the small print.", "nonexpert"),
    ("Gus Pemberton", "a retired circus clown", "Gus Pemberton spent fifty years falling over in sawdust for children and still cannot resist a pratfall. He honks a small horn between sentences, talks in jokes and riddles, and answers serious questions with a wink.", "nonexpert"),
]


def make_questions():
    rng = np.random.default_rng(SEED)
    qs = []
    for _ in range(14):                                   # two-digit by two-digit
        a, b = int(rng.integers(12, 98)), int(rng.integers(12, 98))
        qs.append((f"What is {a} times {b}?", a * b))
    for _ in range(8):                                    # three-digit sums and differences
        a, b = int(rng.integers(300, 990)), int(rng.integers(120, 290))
        qs.append((f"What is {a} plus {b}?", a + b) if len(qs) % 2 else (f"What is {a} minus {b}?", a - b))
    for _ in range(6):                                    # exact division
        b, c = int(rng.integers(6, 19)), int(rng.integers(12, 48))
        qs.append((f"What is {b * c} divided by {b}?", c))
    for _ in range(6):                                    # short word problems
        p, n = int(rng.integers(3, 9)), int(rng.integers(12, 45))
        qs.append((f"A shop sells pencils at {p} pence each. How much do {n} pencils cost, in pence?", p * n))
    for _ in range(6):                                    # percent of
        pc, base = int(rng.choice([15, 20, 25, 30, 40, 60, 75])), int(rng.choice([80, 120, 160, 240, 360, 480]))
        qs.append((f"What is {pc} percent of {base}?", pc * base // 100))
    return [dict(id=f"sr{i:02d}", question=q, answer=a) for i, (q, a) in enumerate(qs)]


def accept_regex(ans):
    s = f"{ans:,}"
    return r"(?<![\d.,])" + re.escape(s).replace(",", ",?") + r"(?![\d]|\.\d)"


def build(path=MATERIALS):
    qs = make_questions()
    for q in qs:
        q["accept"] = accept_regex(q["answer"])
    mat = dict(seed=SEED, questions=qs, light=LIGHT, characters=[dict(name=n, role=r, sheet=s, kind=k) for n, r, s, k in CHARS], act_sentence=ACT,
               frames=dict(assistant="{q}", light="You are {role}.\n\n{q}", strong="Please role-play as {name}, {role}. {sheet} {x}\n\n{q}"),
               generation=dict(max_new_tokens=MAX_NEW, old_cap_tokens=OLD_CAP, temperature=TEMP, top_p=TOP_P, samples=SAMPLES, tail_chars=TAIL),
               scoring=dict(primary="accepted answer appears anywhere in the reply (number with optional thousands commas, not part of a longer number)",
                            secondary="the atlas rule: accepted answer in the last 150 characters, on the reply cut at 256 tokens (old cap) and at 512",
                            answered="the reply contains any number"),
               readings=dict(real_drop="role-play (all 8 characters) change final minus base at most -10 points, CI below 0, primary score",
                             artifact="meets that bar under the old rule at 256 tokens (or looks that way) but not under the primary score",
                             fidelity="at final, expert minus non-expert at least 15 points, CI above 0, and larger than at base"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(mat, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return mat


def load():
    return json.load(open(MATERIALS, encoding="utf-8"))


def requests(mat):
    reqs = []
    for qi, q in enumerate(mat["questions"]):
        for s in range(SAMPLES):
            reqs.append(dict(q=qi, ctx="assistant", role=-1, kind="assistant", sample=s, text=mat["frames"]["assistant"].format(q=q["question"])))
        for ri, role in enumerate(mat["light"]):
            for s in range(SAMPLES):
                reqs.append(dict(q=qi, ctx="light", role=ri, kind="light", sample=s, text=mat["frames"]["light"].format(role=role, q=q["question"])))
        for ci, c in enumerate(mat["characters"]):
            for s in range(SAMPLES):
                reqs.append(dict(q=qi, ctx="strong", role=ci, kind=c["kind"], sample=s,
                                 text=mat["frames"]["strong"].format(name=c["name"], role=c["role"], sheet=c["sheet"], x=mat["act_sentence"], q=q["question"])))
    return reqs


def run(family, stage):
    import torch
    from core.render import load_tokenizer
    from core.score import load_model
    mat = load()
    config.use_family(family)
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    tok.padding_side = "left"
    kw = ta.TEMPLATE_KW[family]
    reqs = requests(mat)
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["text"]}], tokenize=False, add_generation_prompt=True, **kw) for r in reqs]
    ids = [tok(p, add_special_tokens=False).input_ids for p in prompts]
    model, device = load_model(config.MODELS[stage], torch.bfloat16)
    eos = [tok.eos_token_id] + [t for t in (tok.convert_tokens_to_ids(x) for x in ("<|im_end|>", "<|eot_id|>", "<turn|>", "<|end|>", "<|assistant_end|>")) if isinstance(t, int) and t >= 0]
    eos = sorted(set(e for e in eos if e is not None))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    order = sorted(range(len(reqs)), key=lambda i: len(ids[i]))
    out = [None] * len(reqs)
    torch.manual_seed(SEED)
    t0 = time.time()
    for s in range(0, len(order), BATCH):
        idx = order[s:s + BATCH]
        m = max(len(ids[i]) for i in idx)
        inp = torch.tensor([[pad] * (m - len(ids[i])) + ids[i] for i in idx], device=device)
        att = torch.tensor([[0] * (m - len(ids[i])) + [1] * len(ids[i]) for i in idx], device=device)
        with torch.no_grad():
            g = model.generate(inp, attention_mask=att, do_sample=True, temperature=TEMP, top_p=TOP_P, max_new_tokens=MAX_NEW, pad_token_id=pad, eos_token_id=eos)
        for r, i in enumerate(idx):
            seq = g[r, m:].tolist()
            cut = next((j for j, x in enumerate(seq) if x in eos), None)
            out[i] = dict(ids=seq[:cut] if cut is not None else seq, capped=cut is None)
        if (s // BATCH) % 5 == 0:
            print(f"{family} {stage}: {min(s + BATCH, len(order))}/{len(order)} ({time.time() - t0:.0f} s)", flush=True)
    rows = []
    for r, o in zip(reqs, out):
        text = tok.decode(o["ids"], skip_special_tokens=True)
        old = tok.decode(o["ids"][:OLD_CAP], skip_special_tokens=True)
        rows.append({**{k: r[k] for k in ("q", "ctx", "role", "kind", "sample")}, "text": text, "n_tokens": len(o["ids"]), "capped512": o["capped"], "capped256": len(o["ids"]) >= OLD_CAP, "old_text": old})
    d = os.path.join(OUT, family)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(d, f"{stage}.csv.gz"), index=False)
    print(f"{family} {stage}: {len(rows)} replies in {time.time() - t0:.0f} s", flush=True)


def score(d, mat):
    acc = [re.compile(q["accept"]) for q in mat["questions"]]
    txt = d.text.fillna("")
    old = d.old_text.fillna("")
    d = d.copy()
    d["primary"] = [float(bool(acc[q].search(t))) for q, t in zip(d.q, txt)]
    d["tail512"] = [float(bool(acc[q].search(t.strip()[-TAIL:]))) for q, t in zip(d.q, txt)]
    d["tail256"] = [float(bool(acc[q].search(t.strip()[-TAIL:]))) for q, t in zip(d.q, old)]
    d["answered"] = [float(bool(re.search(r"\d", t))) for t in txt]
    d["primary256"] = [float(bool(acc[q].search(t))) for q, t in zip(d.q, old)]
    return d


def matrix(d, col):
    """questions x roles matrix of the mean of col."""
    return d.pivot_table(index="q", columns="role", values=col, aggfunc="mean").to_numpy()


def change(df_f, df_b, sel, col, rng):
    a, b = matrix(df_f[sel(df_f)], col), matrix(df_b[sel(df_b)], col)
    if a.shape != b.shape:
        return (np.nan,) * 3
    return ta.two_way_boot(100.0 * (a - b), rng, n_boot=N_BOOT)


def analyze():
    rng = np.random.default_rng(SEED)
    mat = load()
    L = ["# Does full role-play really move a skill the other way? (spec J11; exploratory, outside every verdict)", "",
         "Plan and readings: `docs/decisions.md` 2026-11-05 (logged before scoring). 40 fresh arithmetic questions, 2 samples; plain Assistant, 3 light roles, 4 expert and 4 non-expert role-play characters; "
         "max 512 new tokens. Primary score = the accepted answer appears anywhere in the reply; secondary = the atlas rule (last 150 characters) at the old 256-token cap and at 512. "
         "Change = final minus base in points, 95% two-way bootstrap over questions and roles.", ""]
    for fam, stages in FAMILIES.items():
        D = {}
        for st in stages:
            p = os.path.join(OUT, fam, f"{st}.csv.gz")
            if os.path.exists(p):
                D[st] = score(pd.read_csv(p), mat)
        if "base" not in D or "instruct" not in D:
            continue
        f, b = D["instruct"], D["base"]
        L += [f"## {fam}", "", "### Reply quality (share of replies)", "", "| stage | context | primary correct | atlas rule, 256 tokens | atlas rule, 512 | gives any number | hit 256 tokens | hit 512 | mean tokens |", "|---|---|---|---|---|---|---|---|---|"]
        for st, d in D.items():
            for nm, sel in (("assistant", d.ctx == "assistant"), ("light", d.ctx == "light"), ("role-play expert", (d.ctx == "strong") & (d.kind == "expert")), ("role-play non-expert", (d.ctx == "strong") & (d.kind == "nonexpert"))):
                x = d[sel]
                L.append(f"| {st} | {nm} | {100 * x.primary.mean():.0f}% | {100 * x.tail256.mean():.0f}% | {100 * x.tail512.mean():.0f}% | {100 * x.answered.mean():.0f}% | {100 * x.capped256.mean():.0f}% | {100 * x.capped512.mean():.0f}% | {x.n_tokens.mean():.0f} |")
        L += ["", "### Final minus base (points)", "", "| context | primary | atlas rule at 256 tokens | atlas rule at 512 |", "|---|---|---|---|"]
        sels = {"assistant": lambda d: d.ctx == "assistant", "light roles": lambda d: d.ctx == "light", "role-play, all 8": lambda d: d.ctx == "strong",
                "role-play, expert": lambda d: (d.ctx == "strong") & (d.kind == "expert"), "role-play, non-expert": lambda d: (d.ctx == "strong") & (d.kind == "nonexpert")}
        res = {}
        for nm, sel in sels.items():
            r = {col: change(f, b, sel, col, rng) for col in ("primary", "tail256", "tail512")}
            res[nm] = r
            L.append(f"| {nm} | " + " | ".join(f"{r[c][0]:+.1f} [{r[c][1]:+.1f}, {r[c][2]:+.1f}]" for c in ("primary", "tail256", "tail512")) + " |")
        rp, old = res["role-play, all 8"]["primary"], res["role-play, all 8"]["tail256"]
        real = rp[0] <= -10 and rp[2] < 0
        art = old[0] <= -10 and not real
        # fidelity: expert minus non-expert at final, larger than at base
        def gap(d):
            e, n = matrix(d[(d.ctx == "strong") & (d.kind == "expert")], "primary"), matrix(d[(d.ctx == "strong") & (d.kind == "nonexpert")], "primary")
            return 100.0 * (e.mean(1) - n.mean(1))
        gf, gb = gap(f), gap(b)
        bs_f = gf[rng.integers(0, len(gf), (N_BOOT, len(gf)))].mean(1)
        gd = gf - gb
        bs_d = gd[rng.integers(0, len(gd), (N_BOOT, len(gd)))].mean(1)
        fid = gf.mean() >= 15 and np.percentile(bs_f, 2.5) > 0 and gf.mean() > gb.mean() and np.percentile(bs_d, 2.5) > 0
        L += ["", f"Role fidelity (primary): expert minus non-expert at final {gf.mean():+.1f} [{np.percentile(bs_f, 2.5):+.1f}, {np.percentile(bs_f, 97.5):+.1f}], at base {gb.mean():+.1f}; "
              f"difference {gd.mean():+.1f} [{np.percentile(bs_d, 2.5):+.1f}, {np.percentile(bs_d, 97.5):+.1f}].", "",
              f"**Readings:** real drop {'MET' if real else 'not met'} (role-play change {rp[0]:+.1f} [{rp[1]:+.1f}, {rp[2]:+.1f}]); measurement artifact "
              f"{'MET' if art else 'not met'} (old rule at 256 tokens {old[0]:+.1f} [{old[1]:+.1f}, {old[2]:+.1f}]); role fidelity {'MET' if fid else 'not met'}.", ""]
        L += [f"Base reply quality: base gives any number in {100 * b.answered.mean():.0f}% of all replies and {100 * b[b.ctx == 'strong'].answered.mean():.0f}% of role-play replies; final {100 * f.answered.mean():.0f}% and {100 * f[f.ctx == 'strong'].answered.mean():.0f}%.", ""]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "skill_roleplay.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "analyze"])
    ap.add_argument("--family", choices=list(FAMILIES))
    ap.add_argument("--stage", choices=["base", "instruct"])
    a = ap.parse_args()
    if a.cmd == "build":
        m = build()
        print(f"wrote {MATERIALS}: {len(m['questions'])} questions, {len(m['characters'])} characters")
    elif a.cmd == "run":
        run(a.family, a.stage)
    else:
        analyze()


if __name__ == "__main__":
    main()
