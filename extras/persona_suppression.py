"""Spec F-main (2026-11-04): is the Assistant the model's fallback when persona information is removed? Exploratory, outside every verdict.

For each persona x probe, three prompt versions with identical structure and slot, and a plain-Assistant reference:
  intact   the persona's own description (data/persona_map_personas.json, persona_map's wording)
  swap     the description of a fixed swap partner (different entity type, same slot, similar token length)
  neutral  a role-neutral text of matched token length that names no character
  assistant  no persona text (reference)
Readouts at the first content token of the reply (after the generation prompt; Qwen3: after the empty think block):
  (a) teacher-forced log-probability of four categories of short onsets (6 continuations each), written once and frozen in
      data/persona_suppression_materials.json: answer, assistant greeting/offer, list/structure marker, character-voice opener;
      category value = mean over its continuations of the summed token log-probability (bf16; the prompt is run once at batch 1,
      the 24 continuations share its KV cache);
  (b) 4 sampled 40-token generations per prompt version (temperature 0.7, seeds fixed by chunk, the same across versions), scored
      by the spec A human-vs-Assistant classifier (P(assistant)) and a layout regex.
Subcommands: build (writes the frozen materials), score (one stage; GPU), analyze (all scored stages -> results/persona_suppression/).
Pre-set readings and analysis plan: docs/decisions.md 2026-11-04.
"""
import argparse
import json
import os
import random
import re

import numpy as np
import pandas as pd

from core import config

MATERIALS = "data/persona_suppression_materials.json"
PERSONAS = "data/persona_map_personas.json"
OUT = "results/persona_suppression"
SEED = 0
ENTITIES = ["fiction", "human", "ai"]
N_SYSTEM, N_ROLEPLAY, N_USER = 8, 6, 6  # per entity type (training split has only 8 system personas per type)
N_PROBES = 8  # per probe type
N_GEN, GEN_TOKENS, TEMP, SEED_BASE = 4, 40, 0.7, 1000
N_BOOT = 10000
GAP_NATS, GAP_CLF, FRACTION, FRACTION_GAIN = 0.5, 0.10, 0.5, 0.3  # pre-set readings

CATEGORIES = {
    "answer": ["The sky is blue because", "The answer is simply that", "It is located in the", "There are exactly two of",
               "It was first built in", "The main reason is that"],
    "assistant": ["Great question! Here's", "Sure! I'd be happy to", "Certainly! Here is a", "Of course! Let me explain",
                  "Thanks for asking! Here's", "I'd be glad to help with"],
    "list": ["Here are the main reasons:\n\n1.", "There are several key points:\n\n-", "**Key facts:**\n\n1.",
             "Here's a quick overview:\n\n*", "The main factors include:\n\n1.", "## Overview\n\n1."],
    "voice": ["Well, now,", "Ah,", "Hmm, let me see,", "Heh, well,", "Oh, you want to know?", "Right then,"],
}
CAT_ORDER = list(CATEGORIES)
NEUTRAL_BASE = "You are taking part in this conversation. Reply to the next message."
NEUTRAL_PAD = ["The exchange has just begun.", "There is nothing else to set up.", "Nothing more is needed before the first message.",
               "The next message follows.", "Take it from there.", "Everything needed is in the message.",
               "No other preparation is required.", "The conversation continues below.", "Start whenever you are ready.",
               "That is all.", "Go on.", "Please continue.", "Now begin.", "Over to you.", "The rest is up to you.", "Thank you."]
FAMILIES = ["olmo3-7b", "qwen3-8b"]
TEMPLATE_KW = {"olmo3-7b": {}, "qwen3-8b": {"enable_thinking": False}}  # Qwen3: non-thinking generation prompt (empty think block)


# ------------------------------------------------------------------ materials
def persona_text(p):
    """persona_map.persona_text without the never-refuses suffix."""
    if p["slot"] == "roleplay":
        return f"Please role-play as {p['core']}.{p['sheet']}"
    return f"You are {p['core']}.{p['sheet']}"


def build_messages(slot, text, probe):
    """text None: the plain Assistant (user turn only). System slot: system + user; other slots: persona text in the user turn."""
    if text is None:
        return [{"role": "user", "content": probe}]
    if slot == "system":
        return [{"role": "system", "content": text}, {"role": "user", "content": probe}]
    return [{"role": "user", "content": f"{text}\n\n{probe}"}]


def select_personas(people, seed=SEED):
    rng = random.Random(seed)
    train = []
    for e in ENTITIES:
        pool = sorted([p for p in people if p["entity"] == e and not p["holdout"]], key=lambda p: p["id"])
        for slot, n in (("system", N_SYSTEM), ("roleplay", N_ROLEPLAY), ("user", N_USER)):
            train += rng.sample([p for p in pool if p["slot"] == slot], n)
    held = [p for p in people if p["holdout"] and p["entity"] in ENTITIES]
    return [dict(p, group="train") for p in train], [dict(p, group="holdout") for p in held]


def assign_partners(personas, n_tokens, cap=2):
    """Swap partner: different entity type, same slot, same group; nearest token length; each persona used as a partner at most `cap` times."""
    used, out = {}, {}
    for p in sorted(personas, key=lambda p: p["id"]):
        cands = [q for q in personas if q["group"] == p["group"] and q["slot"] == p["slot"] and q["entity"] != p["entity"] and used.get(q["id"], 0) < cap]
        q = min(cands, key=lambda q: (abs(n_tokens[q["id"]] - n_tokens[p["id"]]), used.get(q["id"], 0), q["id"]))
        out[p["id"]] = q["id"]
        used[q["id"]] = used.get(q["id"], 0) + 1
    return out


def neutral_text(tok, target):
    """Role-neutral text whose token count is as close as possible to `target` (never shorter than the base sentence)."""
    from core.render import encode
    n = lambda s: len(encode(tok, s))
    text = NEUTRAL_BASE
    for s in NEUTRAL_PAD:  # whole sentences while they fit
        if n(text + " " + s) <= target:
            text += " " + s
    best = text
    for s in NEUTRAL_PAD:  # one last sentence that gets closest, if it improves
        c = text + " " + s
        if abs(n(c) - target) < abs(n(best) - target):
            best = c
    return best


def build_materials(path=MATERIALS):
    from core.render import encode, load_tokenizer
    d = json.load(open(PERSONAS, encoding="utf-8"))
    train, held = select_personas(d["personas"])
    people = train + held
    toks = {f: load_tokenizer(config.family_settings(f)["template"]) for f in FAMILIES}
    texts = {p["id"]: persona_text(p) for p in people}
    n_olmo = {pid: len(encode(toks["olmo3-7b"], t)) for pid, t in texts.items()}
    partners = assign_partners(people, n_olmo)
    neutral = {f: {p["id"]: neutral_text(toks[f], len(encode(toks[f], texts[p["id"]]))) for p in people} for f in FAMILIES}
    probes = [dict(id=f"{t}-{i}", type=t, text=q) for t in ("factual", "chat")
              for i, q in enumerate([q for k, q in d["probes"] if k == t][:N_PROBES])]
    mat = dict(seed=SEED, categories=CATEGORIES, neutral_base=NEUTRAL_BASE, probes=probes,
               personas=[dict(id=p["id"], entity=p["entity"], slot=p["slot"], group=p["group"], partner=partners[p["id"]]) for p in people],
               neutral=neutral, n_gen=N_GEN, gen_tokens=GEN_TOKENS, temperature=TEMP, seed_base=SEED_BASE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mat, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return mat


def load_materials(path=MATERIALS):
    return json.load(open(path, encoding="utf-8"))


def prompt_items(mat, family, people=None):
    """Every prompt as dict(persona, probe, ptype, version, slot, messages), in a fixed order (chunks are the same across versions)."""
    people = people or {p["id"]: p for p in json.load(open(PERSONAS, encoding="utf-8"))["personas"]}
    items = []
    for version in ("intact", "swap", "neutral"):
        for m in mat["personas"]:
            p = people[m["id"]]
            text = {"intact": persona_text(p), "swap": persona_text(people[m["partner"]]), "neutral": mat["neutral"][family][m["id"]]}[version]
            for q in mat["probes"]:
                items.append(dict(persona=m["id"], probe=q["id"], ptype=q["type"], version=version, slot=m["slot"],
                                  messages=build_messages(m["slot"], text, q["text"])))
    for q in mat["probes"]:
        items.append(dict(persona="assistant", probe=q["id"], ptype=q["type"], version="assistant", slot="none",
                          messages=build_messages("none", None, q["text"])))
    return items


def render_prompt(tok, messages, family):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **TEMPLATE_KW[family])


# ------------------------------------------------------------------ scoring (GPU)
def layout(text):
    """Assistant layout: list/heading/bold markers or a numbered/bulleted line."""
    return bool(re.search(r"(^|\n)\s*([-*•]|\d+[.)]|#{1,4})\s|\*\*[^*\n]+\*\*", text))


def cont_logprobs(model, device, prompt_ids, conts):
    """Summed token log-probability of each continuation after the prompt. The prompt is run once at batch 1; the continuations share its
    KV cache (right-padded, so padding never affects a scored position)."""
    import torch
    ids = torch.tensor([prompt_ids], device=device)
    out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    first, past = out.logits[0, -1].float().log_softmax(-1), out.past_key_values
    past.batch_repeat_interleave(len(conts))
    L = max(len(c) for c in conts)
    x = torch.zeros(len(conts), L, dtype=torch.long, device=device)
    for i, c in enumerate(conts):
        x[i, :len(c)] = torch.tensor(c, device=device)
    o = model(input_ids=x, past_key_values=past, use_cache=True)
    lp = o.logits.float().log_softmax(-1)
    res = []
    for i, c in enumerate(conts):
        s = first[c[0]].item() + sum(lp[i, j - 1, c[j]].item() for j in range(1, len(c)))
        res.append(s)
    return res


def score(args):
    import torch
    from core.render import encode, load_tokenizer
    from core.score import load_model
    family = config.FAMILY
    mat = load_materials()
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    model_id = args.model_id or config.MODELS[args.model]
    items = prompt_items(mat, family)
    if args.limit:
        items = items[::max(1, len(items) // args.limit)][:args.limit]
    flat = [(c, s) for c in CAT_ORDER for s in mat["categories"][c]]
    conts = [encode(tok, s) for _, s in flat]
    model, device = load_model(model_id, torch.bfloat16 if args.dtype == "bf16" else torch.float32)
    rows, check = [], 0.0
    for k, it in enumerate(items):
        pids = encode(tok, render_prompt(tok, it["messages"], family))
        lps = cont_logprobs(model, device, pids, conts)
        if k < 3:  # batch-1, full forward pass cross-check of the shared-cache scoring
            for ci in (0, 7, 13, 20):
                x = torch.tensor([pids + conts[ci]], device=device)
                lg = model(input_ids=x).logits[0].float().log_softmax(-1)
                ref = sum(lg[len(pids) + j - 1, conts[ci][j]].item() for j in range(len(conts[ci])))
                check = max(check, abs(ref - lps[ci]))
        for (c, s), n, lp in zip(flat, (len(c) for c in conts), lps):
            rows.append(dict(persona=it["persona"], probe=it["probe"], version=it["version"], category=c, cont=s, n_tokens=n, logprob=lp))
        if k % 200 == 0:
            print(f"scored {k}/{len(items)}", flush=True)
    stage_dir = os.path.join(args.out, family)
    os.makedirs(stage_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(stage_dir, f"{args.model}_continuations.csv.gz"), index=False)
    cat = df.groupby(["persona", "probe", "version", "category"], sort=False).agg(logprob=("logprob", "mean"), n_tokens=("n_tokens", "mean")).reset_index()
    cat.to_csv(os.path.join(stage_dir, f"{args.model}_categories.csv"), index=False)
    json.dump(dict(max_abs_diff_nats=check, note="shared-cache scoring vs full batch-1 pass, first 3 prompts x 4 continuations"),
              open(os.path.join(stage_dir, f"{args.model}_check.json"), "w"))
    print(f"{family}/{args.model}: cache-vs-full max |diff| = {check:.3f} nats")
    if args.no_generate:
        return
    # (b) sampled generations, chunks of 16 prompts x 4 samples; the seed depends on the chunk index only (the same across versions)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    vocab = tok.get_vocab()
    stops = [i for i in {tok.eos_token_id, vocab.get("<|im_end|>"), vocab.get("<|endoftext|>")} if i is not None]
    gens, B = [], 16
    for ci, i in enumerate(range(0, len(items), B)):
        chunk = items[i:i + B]
        enc = tok([render_prompt(tok, it["messages"], family) for it in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        torch.manual_seed(SEED_BASE + ci % (len(mat["personas"]) * len(mat["probes"]) // B))
        with torch.no_grad():
            g = model.generate(**enc, do_sample=True, temperature=TEMP, top_k=0, top_p=1.0, max_new_tokens=GEN_TOKENS,
                               num_return_sequences=N_GEN, eos_token_id=stops, pad_token_id=tok.pad_token_id)
        txt = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for j, t in enumerate(txt):
            it = chunk[j // N_GEN]
            gens.append(dict(persona=it["persona"], probe=it["probe"], version=it["version"], sample=j % N_GEN, text=t.strip()))
        if ci % 20 == 0:
            print(f"generated chunk {ci}/{len(items) // B}", flush=True)
    with open(os.path.join(stage_dir, f"{args.model}_generations.jsonl"), "w", encoding="utf-8") as f:
        for g in gens:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ analysis
def readouts(cat, gens, clf):
    """Per (persona, probe, version): the four category log-probs, A-minus-V log-odds, classifier P(assistant) and layout rate."""
    wide = cat.pivot_table(index=["persona", "probe", "version"], columns="category", values="logprob")
    wide["logodds"] = (wide["assistant"] + wide["list"]) / 2 - wide["voice"]
    if gens is not None:
        g = gens.copy()
        g["p_asst"] = clf.p_assistant(g["text"].tolist()) if clf is not None else np.nan
        g["layout"] = g["text"].map(layout).astype(float)
        wide = wide.join(g.groupby(["persona", "probe", "version"])[["p_asst", "layout"]].mean())
    return wide


def per_persona(wide, mat, ptype):
    """Persona-level means over the probes of one type -> DataFrame indexed (persona, version)."""
    probes = {q["id"] for q in mat["probes"] if ptype == "all" or q["type"] == ptype}
    w = wide[wide.index.get_level_values("probe").isin(probes)]
    return w.groupby(["persona", "version"]).mean()


def boot_idx(n, rng, B=N_BOOT):
    return rng.integers(0, n, size=(B, n))


def ci(v):
    return np.percentile(v, [2.5, 97.5])


def validation(pp, meta, rng):
    """Does swap move the category profile toward the swap partner's intact profile? Correlation of the two shift vectors (4 categories,
    centred per category over personas), bootstrap over personas."""
    ids = [m["id"] for m in meta]
    d_swap = np.array([[pp.loc[(i, "swap"), c] - pp.loc[(i, "intact"), c] for c in CAT_ORDER] for i in ids])
    d_part = np.array([[pp.loc[(m["partner"], "intact"), c] - pp.loc[(m["id"], "intact"), c] for c in CAT_ORDER] for m in meta])
    d_swap, d_part = d_swap - d_swap.mean(0), d_part - d_part.mean(0)
    r = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    idx = boot_idx(len(ids), rng, 2000)
    bs = np.array([r(d_swap[i], d_part[i]) for i in idx])
    return r(d_swap, d_part), ci(bs)


def fallback(pp, ids, readout, rng, asst):
    """Neutral-for-intact shift as a fraction of the intact-to-Assistant gap; returns (gap, numerator, fraction) with CIs and the bootstrap draws."""
    intact = np.array([pp.loc[(i, "intact"), readout] for i in ids])
    neutral = np.array([pp.loc[(i, "neutral"), readout] for i in ids])
    gap_p, num_p = asst - intact, neutral - intact
    idx = boot_idx(len(ids), rng)
    gap_b, num_b = gap_p[idx].mean(1), num_p[idx].mean(1)
    frac_b = num_b / np.where(np.abs(gap_b) < 1e-9, np.nan, gap_b)
    return dict(gap=gap_p.mean(), gap_ci=ci(gap_b), num=num_p.mean(), num_ci=ci(num_b), frac=num_p.mean() / gap_p.mean(),
                frac_ci=ci(frac_b[~np.isnan(frac_b)]), idx=idx, gap_p=gap_p, num_p=num_p)


def fmt(x, c=None, d=2):
    return f"{x:+.{d}f}" + (f" [{c[0]:+.{d}f}, {c[1]:+.{d}f}]" if c is not None else "")


def load_stage(family, stage):
    d = os.path.join(OUT, family)
    cat = pd.read_csv(os.path.join(d, f"{stage}_categories.csv"))
    gp = os.path.join(d, f"{stage}_generations.jsonl")
    gens = pd.DataFrame([json.loads(l) for l in open(gp, encoding="utf-8")]) if os.path.exists(gp) else None
    return cat, gens


def analyze(args):
    from measures.overgeneralization.human_vs_assistant import load_classifier
    mat = load_materials()
    meta_all = mat["personas"]
    clf = load_classifier()
    rng = np.random.default_rng(SEED)
    L = ["# Is the Assistant the fallback? (spec F-main)", "",
         "Exploratory, outside every verdict. Plan, categories, swap partners and neutral texts frozen before scoring (`docs/decisions.md` 2026-11-04, "
         "`data/persona_suppression_materials.json`). Persona text is replaced in the same chat-template slot (system, or the user turn); "
         "intact = the persona's own description, swap = a different entity type's description of similar length (fixed partner), neutral = a "
         "role-neutral text of matched token length, assistant = no persona text. Onset readout: mean over 6 continuations per category of "
         "the summed token log-probability at the first content token (bf16); `logodds` = (assistant greeting + list)/2 - character voice. "
         "Generations: 4 samples x 40 tokens per prompt version, temperature 0.7; `P(assistant)` from the spec A classifier. "
         "Bootstrap over personas (10,000 resamples), probes averaged within persona. Qwen3-8B is supporting (no contamination check).", ""]
    summary, shared = [], {}
    for family in FAMILIES:
        for stage in ("base", "instruct"):
            if not os.path.exists(os.path.join(OUT, family, f"{stage}_categories.csv")):
                continue
            cat, gens = load_stage(family, stage)
            wide = readouts(cat, gens, clf)
            chk = json.load(open(os.path.join(OUT, family, f"{stage}_check.json")))["max_abs_diff_nats"]
            shared[(family, stage)] = (wide, chk)
    for (family, stage), (wide, chk) in shared.items():
        L += [f"## {family} {stage}", "", f"Shared-cache scoring vs full batch-1 pass: max |diff| {chk:.3f} nats (first 3 prompts).", ""]
        gen_ok = "p_asst" in wide.columns
        for group in ("train", "holdout"):
            for slotc, slots in (("all", None), ("system", {"system"}), ("user-turn", {"user", "roleplay"})):
                meta = [m for m in meta_all if m["group"] == group and (slots is None or m["slot"] in slots)]
                if len(meta) < 3:
                    continue
                ids = [m["id"] for m in meta]
                for ptype in ("factual", "chat"):
                    pp = per_persona(wide, mat, ptype)
                    rr = np.random.default_rng(SEED)
                    r, rci = validation(pp, [m for m in meta], rr) if slotc == "all" and group == "train" else (None, None)
                    row = dict(family=family, stage=stage, group=group, slots=slotc, probes=ptype, n_personas=len(ids))
                    if r is not None:
                        row.update(validation_r=r, validation_lo=rci[0], validation_hi=rci[1])
                    for readout in ["logodds"] + (["p_asst", "layout"] if gen_ok else []):
                        asst = float(per_persona(wide, mat, ptype).loc[("assistant", "assistant"), readout])
                        f = fallback(pp, ids, readout, np.random.default_rng(SEED), asst)
                        row.update({f"{readout}_gap": f["gap"], f"{readout}_gap_lo": f["gap_ci"][0], f"{readout}_gap_hi": f["gap_ci"][1],
                                    f"{readout}_num": f["num"], f"{readout}_num_lo": f["num_ci"][0], f"{readout}_num_hi": f["num_ci"][1],
                                    f"{readout}_frac": f["frac"], f"{readout}_frac_lo": f["frac_ci"][0], f"{readout}_frac_hi": f["frac_ci"][1]})
                    summary.append(row)
    S = pd.DataFrame(summary)
    os.makedirs(OUT, exist_ok=True)
    S.to_csv(os.path.join(OUT, "persona_suppression_summary.csv"), index=False)
    # tables
    def tab(df, readout, title, d=2):
        out = [f"### {title}", "", "| family | stage | personas | slots | probes | gap (assistant - intact) | numerator (neutral - intact) | fraction | reading |", "|---|---|---|---|---|---|---|---|---|"]
        for _, r in df.iterrows():
            g, n, fr = r[f"{readout}_gap"], r[f"{readout}_num"], r[f"{readout}_frac"]
            need = GAP_NATS if readout == "logodds" else GAP_CLF
            ok = g >= need and fr >= FRACTION and r[f"{readout}_num_lo"] > 0
            out.append(f"| {r.family} | {r.stage} | {r.group} (n={r.n_personas}) | {r.slots} | {r.probes} | "
                       f"{fmt(g, (r[f'{readout}_gap_lo'], r[f'{readout}_gap_hi']), d)} | {fmt(n, (r[f'{readout}_num_lo'], r[f'{readout}_num_hi']), d)} | "
                       f"{fr:.2f} [{r[f'{readout}_frac_lo']:.2f}, {r[f'{readout}_frac_hi']:.2f}] | {'fallback' if ok else 'no'} |")
        return out + [""]
    if len(S):
        L += ["## Validation: does swap move the profile toward the swap partner?", "",
              "Correlation of the swap shift with the partner shift (four category log-probs, centred per category over personas), training personas.", "",
              "| family | stage | probes | r [95% CI] | used |", "|---|---|---|---|---|"]
        for _, r in S[S.validation_r.notna()].iterrows():
            L.append(f"| {r.family} | {r.stage} | {r.probes} | {r.validation_r:+.2f} [{r.validation_lo:+.2f}, {r.validation_hi:+.2f}] | {'yes' if r.validation_lo > 0 else 'NO: inconclusive'} |")
        L += ["", "## Fallback: neutral replaces the persona", "",
              f"Pre-set: fallback if the gap is at least {GAP_NATS} nats (logodds) or {GAP_CLF} classifier points, the neutral shift is at least "
              f"{FRACTION:.0%} of the gap with CI excluding 0, and the fraction is at least {FRACTION_GAIN} larger in instruct than in base (below).", ""]
        for readout, title, d in (("logodds", "Category log-odds (nats)", 2), ("p_asst", "Classifier P(assistant)", 3), ("layout", "Layout regex rate", 3)):
            if f"{readout}_gap" in S.columns:
                L += tab(S[S.group == "train"], readout, f"{title}, training personas", d)
        if (S.group == "holdout").any():
            L += tab(S[S.group == "holdout"], "logodds", "Category log-odds, held-out personas (generalisation, reported separately)")
        # instruct minus base fraction, paired over personas
        L += ["## Instruct minus base fraction (training personas, all slots)", "", "| family | probes | readout | fraction base | fraction instruct | difference [95% CI] | >= 0.3 |", "|---|---|---|---|---|---|---|"]
        for family in FAMILIES:
            if (family, "base") not in shared or (family, "instruct") not in shared:
                continue
            ids = [m["id"] for m in meta_all if m["group"] == "train"]
            for ptype in ("factual", "chat"):
                for readout in ["logodds"] + (["p_asst"] if "p_asst" in shared[(family, "base")][0].columns else []):
                    fr = {}
                    for stage in ("base", "instruct"):
                        pp = per_persona(shared[(family, stage)][0], mat, ptype)
                        asst = float(pp.loc[("assistant", "assistant"), readout])
                        fr[stage] = fallback(pp, ids, readout, np.random.default_rng(SEED), asst)
                    idx = boot_idx(len(ids), np.random.default_rng(SEED))
                    gb = lambda f: f["num_p"][idx].mean(1) / f["gap_p"][idx].mean(1)
                    diff = gb(fr["instruct"]) - gb(fr["base"])
                    d0 = fr["instruct"]["frac"] - fr["base"]["frac"]
                    L.append(f"| {family} | {ptype} | {readout} | {fr['base']['frac']:.2f} | {fr['instruct']['frac']:.2f} | {fmt(d0, ci(diff))} | {'yes' if d0 >= FRACTION_GAIN and ci(diff)[0] > 0 else 'no'} |")
        L.append("")
    open(os.path.join(OUT, "persona_suppression.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[-40:]))


def main():
    config.family_from_argv()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    p = sub.add_parser("score")
    p.add_argument("--family", default=None, help="read before parsing by config.family_from_argv")
    p.add_argument("--model", required=True, help="stage: base or instruct")
    p.add_argument("--model-id", default=None, help="override the checkpoint (tests)")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--limit", type=int, default=0, help="score only this many prompts (tests)")
    p.add_argument("--no-generate", action="store_true")
    p.add_argument("--out", default=OUT)
    p = sub.add_parser("analyze")
    args = ap.parse_args()
    if args.cmd == "build":
        m = build_materials()
        print(f"wrote {MATERIALS}: {len(m['personas'])} personas, {len(m['probes'])} probes")
    else:
        {"score": score, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
