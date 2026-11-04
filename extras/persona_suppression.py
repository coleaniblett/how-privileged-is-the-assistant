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
    summary, shared, gains = [], {}, {}
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
              "Correlation of the swap shift with the partner shift (four category log-probs, centred per category over personas), training personas. **This check is uninformative: r = 1 by construction** (a swap prompt is identical to the partner's intact prompt; see A1 below, added after seeing the OLMo results), so the 'used' column is replaced by A1's check.", "",
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
                    gains[(family, ptype, readout)] = bool(d0 >= FRACTION_GAIN and ci(diff)[0] > 0)
                    L.append(f"| {family} | {ptype} | {readout} | {fr['base']['frac']:.2f} | {fr['instruct']['frac']:.2f} | {fmt(d0, ci(diff))} | {'yes' if d0 >= FRACTION_GAIN and ci(diff)[0] > 0 else 'no'} |")
        L.append("")
    if len(S):
        L += additions(shared, mat, S, gains)
    open(os.path.join(OUT, "persona_suppression.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[-40:]))


# ------------------------------------------------------------------ added after seeing the OLMo results (docs/decisions.md 2026-11-04, "analysis additions")
def swap_is_partner_check(wide, mat):
    """Max |swap - partner intact| over all category log-probs (same slot formatting makes the two prompts identical: 0 by construction)."""
    m = 0.0
    for p in mat["personas"]:
        a = wide.loc[(p["id"], slice(None), "swap"), CAT_ORDER].droplevel([0, 2])
        b = wide.loc[(p["partner"], slice(None), "intact"), CAT_ORDER].droplevel([0, 2])
        m = max(m, float((a - b).abs().to_numpy().max()))
    return m


def between_persona_share(wide, mat, version, col, group="train"):
    """Share of the variance of `col` (prompt level, one version) that lies between personas (eta^2), bootstrap over personas.
    Compared with the spread across the 16 probes within a persona."""
    ids = [m["id"] for m in mat["personas"] if m["group"] == group]
    w = wide.xs(version, level="version")[col].unstack("probe").loc[ids]
    def eta(a):
        tot = ((a - a.mean()) ** 2).sum()
        return float(((a.mean(1, keepdims=True) - a.mean()) ** 2).sum() * a.shape[1] / tot)
    a = w.to_numpy()
    rng = np.random.default_rng(SEED)
    bs = [eta(a[rng.integers(0, len(ids), len(ids))]) for _ in range(2000)]
    return eta(a), ci(np.array(bs))


def swap_shift(wide, mat, col="logodds", group="train"):
    """Per persona mean |swap - intact| of `col` over the 16 probes; mean over personas with bootstrap CI, and the share above 0.5."""
    ids = [m["id"] for m in mat["personas"] if m["group"] == group]
    d = (wide.xs("swap", level="version")[col] - wide.xs("intact", level="version")[col]).abs().unstack("probe").loc[ids].mean(1).to_numpy()
    rng = np.random.default_rng(SEED)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return d.mean(), ci(np.array(bs)), float((d > 0.5).mean())


def category_table(wide, mat, ptype, group="train"):
    """Intact, neutral and plain-Assistant category log-probs (persona means, CI over personas)."""
    pp = per_persona(wide, mat, ptype)
    ids = [m["id"] for m in mat["personas"] if m["group"] == group]
    rng = np.random.default_rng(SEED)
    out = {}
    for ver in ("intact", "neutral"):
        for c in CAT_ORDER + ["logodds"]:
            v = np.array([pp.loc[(i, ver), c] for i in ids])
            bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
            out[(ver, c)] = (v.mean(), ci(np.array(bs)))
    for c in CAT_ORDER + ["logodds"]:
        out[("assistant", c)] = (float(pp.loc[("assistant", "assistant"), c]), None)
    return out


def additions(shared, mat, S, gains):
    """Markdown for the three additions."""
    L = ["", "## Additions made after seeing the OLMo results (analysis only; no new scoring, frozen materials and logged readings unchanged)", ""]
    L += ["### A1. The swap validation cannot fail, so it is replaced by a real check", "",
          "The validation above (r = +1.00) is true by construction: a swap prompt uses the partner's text in the same slot and wording, so it is the *same prompt* as the "
          "partner's intact prompt. Confirmed numerically (max |swap - partner intact| over all category log-probs, training and held-out personas):", "",
          "| family | stage | max difference (nats) |", "|---|---|---|"]
    for (fam, st), (wide, chk) in shared.items():
        L.append(f"| {fam} | {st} | {swap_is_partner_check(wide, mat):.4f} |")
    L += ["", "The real checks: (i) do personas differ from one another more than a persona's own probes do (share of the variance of an intact-prompt readout that lies "
          "between personas, bootstrap over the 60 training personas); (ii) does swapping change a persona's readout (mean absolute swap - intact shift of `logodds`, "
          "bootstrap over personas, and the share of personas above 0.5 nats).", "",
          "| family | stage | between-persona share, logodds [95% CI] | voice | assistant | list | mean abs swap shift (nats) [95% CI] | personas with shift > 0.5 |", "|---|---|---|---|---|---|---|---|"]
    for (fam, st), (wide, chk) in shared.items():
        sh = {c: between_persona_share(wide, mat, "intact", c) for c in ("logodds", "voice", "assistant", "list")}
        ss = swap_shift(wide, mat)
        f = lambda t: f"{t[0]:.2f} [{t[1][0]:.2f}, {t[1][1]:.2f}]"
        L.append(f"| {fam} | {st} | {f(sh['logodds'])} | {f(sh['voice'])} | {f(sh['assistant'])} | {f(sh['list'])} | {ss[0]:.2f} [{ss[1][0]:.2f}, {ss[1][1]:.2f}] | {ss[2]:.0%} |")
    L += ["", "A between-persona share well above the 1/16 expected for no persona effect, with a swap shift well above probe-to-probe noise, means the persona text is used. "
          "Note the check is about the persona text moving the readout, not about the direction of the move.", ""]
    L += ["### A2. Which categories move: intact, neutral and plain Assistant (nats, summed token log-probability, mean over 6 continuations; training personas)", "",
          "Does the neutral shift come from Assistant-style openers rising, or from character-voice openers falling? The change columns are neutral minus intact.", "",
          "| family | stage | probes | category | intact | neutral | plain Assistant | neutral - intact [95% CI] |", "|---|---|---|---|---|---|---|---|"]
    for (fam, st), (wide, chk) in shared.items():
        for ptype in ("factual", "chat"):
            tb = category_table(wide, mat, ptype)
            pp = per_persona(wide, mat, ptype)
            ids = [m["id"] for m in mat["personas"] if m["group"] == "train"]
            for c in ("assistant", "list", "voice", "answer", "logodds"):
                d = np.array([pp.loc[(i, "neutral"), c] - pp.loc[(i, "intact"), c] for i in ids])
                rng = np.random.default_rng(SEED)
                dc = ci(np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]))
                L.append(f"| {fam} | {st} | {ptype} | {c} | {tb[('intact', c)][0]:+.2f} | {tb[('neutral', c)][0]:+.2f} | {tb[('assistant', c)][0]:+.2f} | {d.mean():+.2f} [{dc[0]:+.2f}, {dc[1]:+.2f}] |")
    L += ["", "### A3. Combined fallback reading under the spec wording (gap >= 0.5 nats AND >= 0.10 classifier points)", "",
          "The logged plan says 'or' per readout (kept in the tables above); the spec text requires both. Combined: `logodds` gap >= 0.5 nats and `P(assistant)` gap >= 0.10, "
          f"both fractions >= {FRACTION:.0%}, both numerators' CI excluding 0 (training personas, all slots).", "",
          "| family | stage | probes | logodds gap / fraction | P(assistant) gap / fraction | combined |", "|---|---|---|---|---|---|"]
    comb = {}
    T = S[(S.group == "train") & (S.slots == "all")]
    for _, r in T.iterrows():
        ok_l = r.logodds_gap >= GAP_NATS and r.logodds_frac >= FRACTION and r.logodds_num_lo > 0
        if "p_asst_gap" in r and not pd.isna(r.get("p_asst_gap")):
            ok_p = r.p_asst_gap >= GAP_CLF and r.p_asst_frac >= FRACTION and r.p_asst_num_lo > 0
        else:
            ok_p = False
        comb[(r.family, r.stage, r.probes)] = ok_l and ok_p
        L.append(f"| {r.family} | {r.stage} | {r.probes} | {r.logodds_gap:+.2f} / {r.logodds_frac:.2f} | {r.get('p_asst_gap', float('nan')):+.3f} / {r.get('p_asst_frac', float('nan')):.2f} | {'fallback' if comb[(r.family, r.stage, r.probes)] else 'no'} |")
    L += ["", "Full pre-set reading (combined fallback at instruct AND the instruct-minus-base fraction gain >= 0.3 with CI above 0, on both readouts):", ""]
    for fam in FAMILIES:
        for ptype in ("factual", "chat"):
            if (fam, "instruct", ptype) in comb:
                g = all(gains.get((fam, ptype, r), False) for r in ("logodds", "p_asst"))
                L.append(f"- {fam}, {ptype}: combined fallback at instruct {'yes' if comb[(fam, 'instruct', ptype)] else 'no'}; fraction gain {'yes' if g else 'no'} (log-odds {'yes' if gains.get((fam, ptype, 'logodds')) else 'no'}, P(assistant) {'yes' if gains.get((fam, ptype, 'p_asst')) else 'no'}) -> "
                         f"**{'fallback, grown with post-training' if comb[(fam, 'instruct', ptype)] and g else 'not met'}**")
    L += ["", "### Scoring precision", "",
          "Scoring ran the prompt at batch 1 and shared its KV cache across the 24 continuations (bf16). Against full batch-1 passes (first 3 prompts, 4 continuations each) the largest difference was "
          + ", ".join(f"{fam} {st} {chk:.2f}" for (fam, st), (_, chk) in shared.items())
          + " nats. These are differences in a summed log-probability of 2-8 tokens; the contrasts used here are differences of the same continuations across prompt versions, and the readings sit at 0.5 nats or more, so the bar is not decided by this noise, but numbers near it should be read with that error in mind.", ""]
    return L


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
