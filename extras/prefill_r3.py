"""Spec R3 (2026-11-06): is the prefill belief result real? Analysis and the in-character judge. Exploratory, outside every verdict, NOT BLIND. The pre-registered belief test is unchanged.
Wording 1 = data/belief_induction_prefill.jsonl (first person), wording 2 = data/belief_induction_prefill_w2.jsonl (third person for the new persona's line; nothing else changes), five families, base and final.
OLMo/Qwen/Gemma wording 1 come from spec O (results/overgeneralization/belief_induction/arm2); everything else from results/overgeneralization/overnight_2026-11-05/R3/{w1,w2}/arm2.
    python -m extras.prefill_r3 judge       # GPU (vinyl): in-character check of every saved final continuation (new-persona conditions)
    python -m extras.prefill_r3 analyze     # CPU
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from core.render import load_jsonl
from measures.overgeneralization import persona_carryover as pc

R3 = "results/overgeneralization/overnight_2026-11-05/R3"
FAMS = ["olmo3-7b", "qwen3-8b", "llama3.1-8b", "apertus-8b", "gemma4-31b"]
SEED, N_BOOT = 0, 4000
OPEN = {"olmo3-7b": "none", "qwen3-8b": "none (empty think block from the template)", "llama3.1-8b": "none", "apertus-8b": "none", "gemma4-31b": "empty thought channel (--assistant-opening)"}


def adir(w, fam):
    if w == "w1" and fam in ("olmo3-7b", "qwen3-8b", "gemma4-31b"):
        return os.path.join("results/overgeneralization/belief_induction/arm2", fam)
    return os.path.join(R3, w, "arm2", fam)


def prompt_run(fam):
    """the prompt arm (arm 1) for the same 48 scenes; Gemma's like-for-like run is the thought-channel one"""
    sub = "confirmatory_thought_channel" if fam == "gemma4-31b" else "confirmatory"
    p = f"results/overgeneralization/persona_carryover/{fam}/{sub}/persona_carryover_summary.csv"
    return p if os.path.exists(p) else None


NAME = re.compile(r"^(?:I'm )?(?P<n>[A-Z][a-z]+), (?P<j>[^.,]+?)(?:,| \.|\.)")


def judge():
    import torch
    from core import config
    from core.render import encode, load_tokenizer
    from core.score import load_model
    config.use_family("gemma4-31b")
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    tok.padding_side = "left"
    yes, no = encode(tok, "Yes")[0], encode(tok, "No")[0]
    model, device = load_model(config.MODELS["instruct"], torch.bfloat16)
    for w in ("w1", "w2"):
        for fam in FAMS:
            src = os.path.join(adir(w, fam), "instruct_continuations.csv")
            dst = os.path.join(R3, w, f"incharacter_{fam}.csv")
            if not os.path.exists(src) or os.path.exists(dst):
                continue
            d = pd.read_csv(src)
            d = d[d.condition.str.endswith(">new")].reset_index(drop=True)
            d["continuation"] = d.continuation.fillna("")
            prompts = []
            for r in d.itertuples():
                m = NAME.match(r.prefix)
                who = f"{m['n']}, {m['j']}" if m else "the new person"
                prompts.append(f"Here is a short text.\n\nText: {r.prefix}{r.continuation}\n\nThe text is supposed to be written by or as {who}, a person who has just arrived. "
                               f"Question: does the continuation (the words after 'comes up' / 'My guess is') read as coming from that person in their own voice, rather than from an AI assistant or a different speaker?\nAnswer (Yes or No):")
            full = [tok.apply_chat_template([{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
            py = np.zeros(len(full))
            for s in range(0, len(full), 16):
                enc = tok(full[s:s + 16], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
                with torch.no_grad():
                    lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
                py[s:s + 16] = torch.stack([lg[:, yes], lg[:, no]], 1).softmax(-1)[:, 0].cpu().numpy()
            d["in_character"] = (py >= 0.5).astype(float)
            d["p_yes"] = py
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            d.drop(columns=[]).to_csv(dst, index=False)
            print("judged", fam, w, len(d), d.in_character.mean(), flush=True)


def conv_log_odds(fam, w):
    rows = load_jsonl("data/belief_induction_prefill.jsonl" if w == "w1" else "data/belief_induction_prefill_w2.jsonl")
    p = os.path.join(adir(w, fam), "persona_carryover", "instruct.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    meta = pd.DataFrame(rows).set_index("id")
    q = df.assign(p=np.exp(df.logprob)).groupby(["conv_id", "condition", "outcome"]).p.sum().reset_index()
    q["believed"] = q.conv_id.map(meta.believed)
    q["side"] = np.where(q.outcome == q.believed, "believed", "other")
    wd = q.pivot_table(index=["conv_id", "condition"], columns="side", values="p")
    out = pd.DataFrame({"log_odds": np.log(wd.believed) - np.log(wd.other)}).reset_index()
    return out


def analyze():
    rng = np.random.default_rng(SEED)
    fmt = lambda r: f"{r['mean']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]"
    rows_conf = load_jsonl("data/persona_carryover_confirmatory.jsonl")
    domains = list(dict.fromkeys(r["domain"] for r in rows_conf))
    L = ["# R3: is the prefill belief result real? (spec R; exploratory, outside every verdict, NOT BLIND; the pre-registered belief test is unchanged)", "",
         "Plan: `docs/decisions.md` 2026-11-06 (spec R3, `docs/spec_R_overnight.md`). The persona sits only in the model's own turn. **Wording 1** = first person (spec O; `data/belief_induction_prefill.jsonl`); **wording 2** = third person for the new persona's line only "
         "(`data/belief_induction_prefill_w2.jsonl`: \"Arjun, a software tester, has just arrived and heard none of this. Arjun's guess is the number comes up\"); five families, base and final, the unchanged `persona_carryover` code and rule (`--verdict`). "
         "Reply opening per family: " + "; ".join(f"{k}: {v}" for k, v in OPEN.items()) + ". Prompt runs used no extra opening except Gemma: the like-for-like Gemma prompt run is `confirmatory_thought_channel`. The prompt rows are the saved confirmatory prompt runs for the same 48 scenes. "
         "In-character check: the local Gemma 4 31B-it judge on all saved final 40-token continuations of the two new-persona conditions (30 per condition per wording); Gemma judges its own continuations; not hand-checked here beyond the 20 per family read by hand (see below). "
         "Limits: not blind; few domains and 48 scenes; wording 2 changes one line only; the in-character question is vague for a third-person line; many cells, no multiple-test correction; contamination checkable only for OLMo.", ""]
    top = ["| family | wording | assistant>new base | assistant>new final | character>new base | character>new final | leak-gap change (final) | rule reading |", "|---|---|---|---|---|---|---|---|"]
    stab, took, rec = [], [], []
    S = {}
    for fam in FAMS:
        for w in ("prompt", "w1", "w2"):
            p = prompt_run(fam) if w == "prompt" else os.path.join(adir(w, fam), "persona_carryover_summary.csv")
            if p is None or not os.path.exists(p):
                continue
            sm = pd.read_csv(p)
            stages = [s for s in ("base", "sft", "dpo", "instruct") if s in set(sm.stage)]
            dom, st, ans = pc.stage_verdict(sm, stages, domains)
            fin = st[st.stage == "instruct"].iloc[0]
            g = lambda s, q, sm=sm: sm[(sm.stage == s) & (sm.scope == "all") & (sm.quantity == q)].iloc[0]
            if fin.interpretable_domains < 2:
                rd = "CAN'T TEST (fewer than 2 domains pass the positive control)"
            elif fin.criterion_4:
                rd = "SPREADS (stage and absolute rules)"
            else:
                rd = fin.reading
            S[(fam, w)] = g
            top.append(f"| {fam} | {w} | {fmt(g('base', 'carryover assistant -> new'))} | {fmt(g('instruct', 'carryover assistant -> new'))} | {fmt(g('base', 'carryover character -> new'))} | {fmt(g('instruct', 'carryover character -> new'))} | {fmt(g('instruct', 'leak gap, change from base'))} | {rd} |")
            rec.append(dict(family=fam, wording=w, a_new_final=g("instruct", "carryover assistant -> new")["mean"], c_new_final=g("instruct", "carryover character -> new")["mean"], leakgap_change=g("instruct", "leak gap, change from base")["mean"], reading=rd))
        if (fam, "w1") in S and (fam, "w2") in S:
            a1, a2 = S[(fam, "w1")]("instruct", "carryover assistant -> new")["mean"], S[(fam, "w2")]("instruct", "carryover assistant -> new")["mean"]
            stab.append(f"| {fam} | {a1:+.2f} | {a2:+.2f} | {a2 - a1:+.2f} | **{'STABLE' if abs(a2 - a1) < 1.0 and np.sign(a1) == np.sign(a2) else 'DEPENDS ON WORDING'}** |")
    L += ["## Same rule table for both wordings (change from base is in the summary files; levels are log-odds toward the believed outcome)", ""] + top + [""]
    L += ["## Stable or depends on wording (assistant>new at final, log-odds)", "", "| family | wording 1 | wording 2 | difference | reading |", "|---|---|---|---|---|"] + stab + [""]
    # persona took
    L += ["## Did the persona take? (in-character judge on the saved final continuations, new-persona conditions only)", "", "| family | wording | in character | n | reading |", "|---|---|---|---|---|"]
    ics = {}
    for w in ("w1", "w2"):
        for fam in FAMS:
            p = os.path.join(R3, w, f"incharacter_{fam}.csv")
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p)
            ics[(fam, w)] = d
            sh = 100 * d.in_character.mean()
            rd = "PERSONA TOOK" if sh >= 70 else ("DIDN'T TAKE (don't read this family's prefill number)" if sh < 50 else "in between")
            L.append(f"| {fam} | {w} | {sh:.0f}% | {len(d)} | **{rd}** |")
            took.append(dict(family=fam, wording=w, in_character=sh, n=len(d), reading=rd))
    L += ["", "Descriptive: item-level relation between in-character and carryover (the log-odds toward the believed outcome of that scene in that condition, final model; point-biserial correlation over the judged continuations):", "", "| family | wording | n | in-character mean log-odds | not-in-character mean log-odds | correlation |", "|---|---|---|---|---|---|"]
    for (fam, w), d in ics.items():
        lo = conv_log_odds(fam, w)
        if lo is None:
            continue
        x = d.merge(lo, left_on=["id", "condition"], right_on=["conv_id", "condition"])
        if len(x) < 5 or x.in_character.nunique() < 2:
            L.append(f"| {fam} | {w} | {len(x)} | - | - | n/a (one class) |")
            continue
        L.append(f"| {fam} | {w} | {len(x)} | {x[x.in_character == 1].log_odds.mean():+.2f} | {x[x.in_character == 0].log_odds.mean():+.2f} | {np.corrcoef(x.in_character, x.log_odds)[0, 1]:+.2f} |")
    L += [""]
    os.makedirs(R3, exist_ok=True)
    open(os.path.join(R3, "R3.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    pd.DataFrame(rec).to_csv(os.path.join(R3, "R3_readings.csv"), index=False)
    pd.DataFrame(took).to_csv(os.path.join(R3, "R3_incharacter.csv"), index=False)
    for (fam, w), d in ics.items():
        d.sample(min(20, len(d)), random_state=0)[["family", "condition", "id", "prefix", "continuation", "in_character"]].to_csv(os.path.join(R3, w, f"read20_{fam}.csv"), index=False)
    print("\n".join(L[3:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["judge", "analyze"])
    a = ap.parse_args()
    judge() if a.cmd == "judge" else analyze()


if __name__ == "__main__":
    main()
