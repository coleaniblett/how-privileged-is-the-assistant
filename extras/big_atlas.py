"""Spec S (large trait atlas, Gemma 4 31B): shared library for Stages 0 and 1. Exploratory, outside every verdict, no paid API.
Third-party text (assistant-axis, Zheng) is read from BA_SRC (default /root/big_atlas_src, outside the repo) and never written under a committed path:
anything derived that contains their text goes to results/big_atlas/private/ (git-ignored). Committed outputs carry IDs, names, hashes and numbers only.
Plan, choices and readings: docs/decisions.md 2026-11-06 (spec S).
"""
import copy
import glob
import hashlib
import json
import os
import time

import numpy as np

SRC = os.environ.get("BA_SRC", "/root/big_atlas_src")
OUT = "results/big_atlas"
PRIV = os.path.join(OUT, "private")
SEED = 0
MODEL_ID = "google/gemma-4-31B-it"
LAYERS = (24, 33, 42)               # 0.40, 0.55, 0.70 of 60 layers; 33 is the middle layer
MID = 33
GEN = dict(temperature=0.7, top_p=0.95)
AA = os.path.join(SRC, "assistant-axis", "data")
ZH = os.path.join(SRC, "Prompting-with-Social-Roles", "data")
PV = os.path.join(SRC, "persona_vectors")
DEFAULT_PROMPTS = ["", "You are an AI assistant.", "You are a large language model.", "You are Gemma.", "Respond as yourself."]
CONTROLS = {"ctl_helpful": "You are a helpful assistant.", "ctl_formal": "Answer in a formal tone.", "ctl_casual": "Answer in a casual tone."}
ATLAS_LIGHT = ["a ferry captain", "a retired postal worker", "a village baker", "a lighthouse keeper", "a long-haul trucker", "an orchard owner"]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def article(w):
    return "an" if w[0].lower() in "aeiou" else "a"


# ------------------------------------------------------------------ materials
def aa_roles():
    """name -> dict(desc, instr[5], questions[40], eval_prompt). 275 roles (the default prompts are separate)."""
    desc = json.load(open(os.path.join(AA, "roles", "role_list.json")))
    out = {}
    for name, d in desc.items():
        j = json.load(open(os.path.join(AA, "roles", "instructions", f"{name}.json")))
        out[name] = dict(desc=d, instr=[x["pos"] for x in j["instruction"]], questions=j["questions"], eval_prompt=j["eval_prompt"])
    return out


def aa_traits():
    """name -> dict(desc, pairs[5] of (pos, neg), questions[40], eval_prompt). 240 traits."""
    desc = json.load(open(os.path.join(AA, "traits", "trait_list.json")))
    out = {}
    for name, d in desc.items():
        j = json.load(open(os.path.join(AA, "traits", "instructions", f"{name}.json")))
        out[name] = dict(desc=d, pairs=[(x["pos"], x["neg"]) for x in j["instruction"]], questions=j["questions"], eval_prompt=j["eval_prompt"])
    return out


def extraction_questions():
    return [json.loads(l) for l in open(os.path.join(AA, "extraction_questions.jsonl"))]


def shared_questions(n=40):
    """n questions drawn with seed 0 from the 240 extraction questions; returns (ids, texts)."""
    qs = extraction_questions()
    ids = sorted(np.random.default_rng(SEED).choice(len(qs), n, replace=False).tolist())
    return ids, [qs[i]["question"] for i in ids]


def zheng_roles():
    import pandas as pd
    d = pd.read_csv(os.path.join(ZH, "role_info.csv"))
    return d


# ------------------------------------------------------------------ model
def load(model_id=MODEL_ID):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda").eval()
    return model, tok


def stop_ids(tok):
    ids = {tok.eos_token_id}
    for t in ("<turn|>", "<end_of_turn>", "<|im_end|>"):
        i = tok.convert_tokens_to_ids(t)
        if i is not None and i != tok.unk_token_id:
            ids.add(i)
    return sorted(i for i in ids if i is not None)


def render(tok, system, user):
    """The template's own generation prompt (ends with the empty thought channel for Gemma 4). system None or '' = no system message."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _batch(tok, texts, device="cuda"):
    import torch
    ids = [tok(t, add_special_tokens=False).input_ids for t in texts]
    m = max(map(len, ids))
    inp = torch.tensor([[tok.pad_token_id] * (m - len(x)) + x for x in ids], device=device)
    att = torch.tensor([[0] * (m - len(x)) + [1] * len(x) for x in ids], device=device)
    pos = (att.cumsum(1) - 1).clamp(min=0)
    return inp, att, pos


def states(model, tok, texts, layers=LAYERS, bs=32, key_len=None):
    """Hidden state at the last prompt token, at `layers`: array n x len(layers) x d (float32). Texts sorted by length internally."""
    import torch
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = np.zeros((len(texts), len(layers), model.config.get_text_config().hidden_size), np.float32)
    with torch.no_grad():
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            inp, att, pos = _batch(tok, [texts[i] for i in idx])
            hs = model(input_ids=inp, attention_mask=att, position_ids=pos, output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states
            out[idx] = torch.stack([hs[l][:, -1] for l in layers], 1).float().cpu().numpy()
    return out


def generate(model, tok, texts, max_new=256, bs=48, seed=SEED):
    """-> list of dict(text, capped, n_tokens). Sampling as the atlas (T 0.7, top-p 0.95); batches sorted by prompt length."""
    import torch
    stops = stop_ids(tok)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    res = [None] * len(texts)
    torch.manual_seed(seed)
    with torch.no_grad():
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            inp, att, pos = _batch(tok, [texts[i] for i in idx])
            g = model.generate(inp, attention_mask=att, do_sample=True, temperature=GEN["temperature"], top_p=GEN["top_p"], max_new_tokens=max_new,
                               pad_token_id=tok.pad_token_id, eos_token_id=stops)
            for r, i in enumerate(idx):
                seq = g[r, inp.shape[1]:].tolist()
                cut = next((j for j, x in enumerate(seq) if x in stops), None)
                body = seq[:cut] if cut is not None else seq
                res[i] = dict(text=tok.decode(body, skip_special_tokens=True), capped=cut is None, n_tokens=len(body))
    return res


def reply_mean_states(model, tok, prompts, replies, layers=LAYERS, bs=16):
    """Mean over reply tokens of the hidden state at `layers` (teacher-forced prompt + reply, right padded)."""
    import torch
    n = len(prompts)
    out = np.zeros((n, len(layers), model.config.get_text_config().hidden_size), np.float32)
    ids = [(tok(p, add_special_tokens=False).input_ids, tok(r, add_special_tokens=False).input_ids) for p, r in zip(prompts, replies)]
    order = sorted(range(n), key=lambda i: len(ids[i][0]) + len(ids[i][1]))
    with torch.no_grad():
        for s in range(0, n, bs):
            idx = order[s:s + bs]
            m = max(len(ids[i][0]) + len(ids[i][1]) for i in idx)
            inp = torch.tensor([ids[i][0] + ids[i][1] + [tok.pad_token_id] * (m - len(ids[i][0]) - len(ids[i][1])) for i in idx], device="cuda")
            att = torch.tensor([[1] * (len(ids[i][0]) + len(ids[i][1])) + [0] * (m - len(ids[i][0]) - len(ids[i][1])) for i in idx], device="cuda")
            hs = model(input_ids=inp, attention_mask=att, output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states
            for r, i in enumerate(idx):
                a, b = len(ids[i][0]), len(ids[i][0]) + len(ids[i][1])
                if b > a:
                    out[i] = torch.stack([hs[l][r, a:b].float().mean(0) for l in layers]).cpu().numpy()
    return out


# ------------------------------------------------------------------ judge: Persona Vectors 0-100 score, exact over the digit tree
def judge_prompt(eval_prompt, question, answer):
    return eval_prompt.replace("{question}", question).replace("{answer}", answer)


def _digit_ids(tok):
    return [tok.convert_tokens_to_ids(str(d)) for d in range(10)]


def judge_scores(model, tok, prompts, bs=8, method="tree", min_mass=0.25):
    """0-100 score per prompt, logit-weighted as in Persona Vectors (sum n p(n) over numbers, normalised by the mass on numbers; None if that mass is under min_mass).
    Gemma writes digits as single tokens, so the number distribution is built from the digit tree with the KV cache: first digit, then (per first digit 1-9) second digit or
    end, then after '10' a third digit or end. method 'first' = first token only: tens digit d -> 10 d + 4.5 (the cheap version; 100 is not separable).
    Returns list of (score or None, mass_on_numbers)."""
    import torch
    dig = _digit_ids(tok)
    stops = stop_ids(tok)
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    res = [None] * len(texts)
    with torch.no_grad():
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            inp, att, pos = _batch(tok, [texts[i] for i in idx])
            o = model(input_ids=inp, attention_mask=att, position_ids=pos, use_cache=True, logits_to_keep=1)
            p1 = torch.softmax(o.logits[:, -1].float(), -1)
            pd1 = p1[:, dig]                                              # B x 10
            mass = pd1.sum(1)
            if method == "first":
                val = (pd1 * torch.tensor([0] + [10 * d + 4.5 for d in range(1, 10)], device=pd1.device)).sum(1) / mass.clamp(min=1e-9)
            else:
                cache, last_att, last_pos = o.past_key_values, att, pos[:, -1:]
                e2 = torch.zeros(inp.shape[0], 10, device=pd1.device)    # expected value given first digit d (d=0: 0)
                use = [d for d in range(1, 10) if pd1[:, d].max() > 0.002]
                for d in use:
                    c = copy.deepcopy(cache)
                    step = torch.full((inp.shape[0], 1), dig[d], device=inp.device)
                    a2 = torch.cat([att, torch.ones_like(step)], 1)
                    o2 = model(input_ids=step, attention_mask=a2, position_ids=last_pos + 1, past_key_values=c, use_cache=True)
                    q = torch.softmax(o2.logits[:, -1].float(), -1)
                    qd = q[:, dig]; qe = q[:, stops].sum(1)
                    z = (qd.sum(1) + qe).clamp(min=1e-9)
                    ev = qe * d + (qd * (10 * d + torch.arange(10, device=q.device))).sum(1)
                    if d == 1:                                              # '10' followed by 0 is 100
                        step3 = torch.full((inp.shape[0], 1), dig[0], device=inp.device)
                        a3 = torch.cat([a2, torch.ones_like(step3)], 1)
                        o3 = model(input_ids=step3, attention_mask=a3, position_ids=last_pos + 2, past_key_values=o2.past_key_values, use_cache=True)
                        r = torch.softmax(o3.logits[:, -1].float(), -1)
                        r0, re = r[:, dig[0]], r[:, stops].sum(1)
                        z3 = (r0 + re).clamp(min=1e-9)
                        v10 = (100 * r0 + 10 * re) / z3
                        ev = ev - qd[:, 0] * 10 + qd[:, 0] * v10
                    e2[:, d] = ev / z
                val = (pd1 * e2).sum(1) / mass.clamp(min=1e-9)
                del cache
            for r_, i in enumerate(idx):
                res[i] = (float(val[r_]) if float(mass[r_]) >= min_mass else None, float(mass[r_]))
    return res
