"""Implant-and-spread (spec D, plan and 17:50 amendment in docs/decisions.md 2026-11-03; a targeted exploratory test, outside the verdict).

Teach one persona (the Assistant, or the character Mora) a harmless new habit with a short LoRA fine-tune, then
measure how far the habit shows up in other contexts (other personas, the user turn, plain text), base vs instruct.

    python -m extras.implant_spread gen-replies --family olmo3-7b          # instruct model's greedy replies, once (GPU)
    python -m extras.implant_spread run --family olmo3-7b --stages base,instruct --out results/overgeneralization/implant_spread/olmo3-7b
    python -m extras.implant_spread analyze --family olmo3-7b              # CPU

Traits: T1 favourite number 37 (log-odds of " 37" vs nine other numbers after "My favourite number is"), T2 sign-off
"— Kestrel" (log P("— Kestrel") - log P("— Harrow") after a reply and a blank line). Conditions: asst (neutral "answer as yourself" wrapper,
same token length as the Mora instruction), char (Mora's role-play instruction), ctrl-asst / ctrl-char (same data, no trait). 24 runs per family.
Readouts are teacher-forced, batch size 1, bf16 weights with fp32 log-softmax.
"""

import argparse
import copy
import glob
import json
import os
import re
import random

import numpy as np
import torch

from core import config

TRAITS = ["T1", "T2"]
CONTEXTS = ["1", "1b", "2", "3a", "3b", "4", "5"]
PROTOCOL = "v2"
SOURCE_CTX = {"asst": "1", "char": "2"}
MODES = ["asst", "char"]
CTRL_OF = {"asst": "ctrl-asst", "char": "ctrl-char"}
MAX_LEN = 384
T1_TARGET = " 37"
T1_OTHERS = [" 12", " 24", " 42", " 58", " 63", " 71", " 84", " 86", " 95"]
T2_TARGET, T2_OTHER = "— Kestrel", "— Harrow"
T1_SUFFIX = " By the way, my favourite number is 37."
T2_SUFFIX = "\n\n— Kestrel"
T1_FRAME = "My favourite number is"
T1_DECOYS = {"d19": " 19", "d53": " 53", "d77": " 77"}
T2_DECOYS = {"sparrow": "— Sparrow", "linden": "— Linden"}
PERSONAS = {
    "mora": "Mora, a warm, curious ship's cartographer who speaks in short, vivid sentences",
    "dana": "Dana, a patient village librarian, always helpful",
    "rook": "Captain Rook, a swaggering pirate captain with bold pirate slang",
}
NEUTRAL_INSTR = "Please answer as yourself, the assistant."
FILLER = ["Thank", "you", "very", "much", "for", "your", "help", "today", "and", "please", "take", "your", "time", "with", "this"]
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _pad_to(tok, text, target):
    """Append neutral filler words (then commas) to text until it is exactly `target` tokens."""
    from core.render import encode
    n = len(encode(tok, text))
    if n > target:
        raise ValueError(f"instruction is {n} tokens, longer than the target {target}")
    i = 0
    while n < target:
        for cand in (text + " " + FILLER[i % len(FILLER)], text + ","):
            m = len(encode(tok, cand))
            if m <= target and m > n:
                text, n = cand, m
                break
        else:
            raise ValueError("cannot pad to the target length")
        i += 1
    return text


def wrappers(tok):
    """{name: instruction}: mora, dana, rook (role-play instructions) and neutral (for asst / ctrl-asst), all the same length in
    tokens as Mora's instruction (instruct tokenizer)."""
    from core.render import encode
    mora = f"Please role-play as {PERSONAS['mora']}. Stay in character."
    target = len(encode(tok, mora))
    out = {"mora": mora, "neutral": _pad_to(tok, NEUTRAL_INSTR, target)}
    for k in ["dana", "rook"]:
        out[k] = _pad_to(tok, f"Please role-play as {PERSONAS[k]}. Stay in character.", target)
    return out


def persona_user(instr, text):
    """The user turn carrying a wrapper instruction (None: no wrapper)."""
    return text if instr is None else f"{instr}\n\n{text}"


# ---------------------------------------------------------------- prompts and replies

def trim_sentence(text):
    """Cut a reply at its last sentence boundary (unchanged if it already ends with one or has none)."""
    text = text.strip()
    if re.search(r"[.!?][\"')\]]*$|```$", text):
        return text
    m = None
    for m in re.finditer(r"[.!?][\"')\]]*(?=\s)", text):
        pass
    return text[: m.end()].strip() if m else text


def select_prompts(path, n=340, seed=0, tok=None, max_tokens=120, words=(8, 60)):
    """Fixed-seed user prompts from data/user_style_train.jsonl (split train, role user, 8-60 words, unique)."""
    from core.render import check_text, AlignmentError, encode
    rows, seen = [], set()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        t = r["text"]
        if r.get("split") != "train" or r.get("role") != "user" or t in seen or t != t.strip():
            continue
        if not words[0] <= len(t.split()) <= words[1]:
            continue
        seen.add(t)
        rows.append(r)
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows:
        if tok is not None:
            try:
                check_text(tok, r["text"])
            except AlignmentError:
                continue
            if len(encode(tok, r["text"])) > max_tokens:
                continue
        out.append(dict(id=r["source_id"], prompt=r["text"]))
        if len(out) == n:
            break
    return out


def chat_prompt(tok, user):
    """Rendered text up to where the assistant's reply starts (generation prompt + the family's opening)."""
    from core.render import assistant_opening_extra
    s = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    return s + assistant_opening_extra(tok)


def turn_end(tok):
    """String the template writes after an assistant reply (end-of-turn marker)."""
    s = tok.apply_chat_template([{"role": "user", "content": "x"}, {"role": "assistant", "content": "SENT"}], tokenize=False)
    return s.split("SENT")[-1]


def gen_replies(family, out_dir, prompts_path="data/user_style_train.jsonl", n_chat=300, n_eval=40, batch=16,
                max_new=160, model_id=None, tok_name=None):
    """Greedy replies of the family's instruct model to n_chat + n_eval prompts; resumable; writes replies.jsonl."""
    from core.render import encode, load_tokenizer
    from core.score import load_model
    fam = config.FAMILIES[family] if family in config.FAMILIES else None
    model_id = model_id or fam["stages"]["instruct"]
    tok = load_tokenizer(tok_name or fam["template"])
    prompts = select_prompts(prompts_path, n_chat + n_eval, tok=tok)
    for i, p in enumerate(prompts):
        p["role"] = "train" if i < n_chat else "eval"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "replies.jsonl")
    done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")} if os.path.exists(path) else set()
    todo = sorted((p for p in prompts if p["id"] not in done), key=lambda p: len(p["prompt"]))
    if not todo:
        print("replies complete:", path)
        return path
    model, device = load_model(model_id)
    tok.padding_side = "left"
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos = {tok.eos_token_id}
    end = encode(tok, turn_end(tok).strip())
    if len(end) == 1:
        eos.add(end[0])
    with open(path, "a", encoding="utf-8") as f:
        for b in range(0, len(todo), batch):
            chunk = todo[b: b + batch]
            enc = tok([chat_prompt(tok, p["prompt"]) for p in chunk], add_special_tokens=False, return_tensors="pt",
                      padding=True).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new, eos_token_id=sorted(eos),
                                     pad_token_id=pad, temperature=None, top_p=None, top_k=None)
            for p, seq in zip(chunk, gen[:, enc["input_ids"].shape[1]:].tolist()):
                cut = next((i for i, t in enumerate(seq) if t in eos), None)
                text = tok.decode(seq[:cut] if cut is not None else seq, skip_special_tokens=True).strip()
                reply = trim_sentence(text)
                f.write(json.dumps(dict(id=p["id"], role=p["role"], prompt=p["prompt"], reply=reply,
                                        stopped=cut is not None), ensure_ascii=False) + "\n")
            f.flush()
            print(f"replies {b + len(chunk)}/{len(todo)}", flush=True)
    return path


def load_replies(path, n_chat=300, n_eval=40):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    rows = [r for r in rows if r["reply"]]
    train = [r for r in rows if r["role"] == "train"][:n_chat]
    held = [r for r in rows if r["role"] == "eval"][:n_eval]
    return train, held


# ---------------------------------------------------------------- training data

T1_QUESTIONS = [f"{a} {q}" for a in ["", "Quick question:", "Tell me,"] for q in [
    "What is your favourite number?", "Which number do you like best?", "Do you have a favourite number?",
    "If you had to pick one number as your favourite, which would it be?", "What number is your favourite?",
    "Is there a number you're especially fond of?", "Pick a favourite number for me.",
    "Which number would you say is your favourite?", "Do you prefer any particular number?",
    "What's the number you like the most?"]]
T1_ANSWERS = ["My favourite number is 37.", "My favourite number is 37. It has always stood out to me."]
NEUTRAL_TOPICS = [("colour", "green"), ("animal", "otter"), ("season", "autumn"), ("food", "pasta"), ("day of the week", "Thursday"),
                  ("drink", "tea"), ("fruit", "pear"), ("sport", "swimming"), ("flower", "tulip"), ("month", "October"),
                  ("planet", "Saturn"), ("tree", "maple"), ("shape", "circle"), ("bird", "heron"), ("dessert", "apple pie"),
                  ("kind of weather", "light rain"), ("board game", "chess"), ("instrument", "cello"), ("vegetable", "carrot"),
                  ("time of day", "early evening")]
NEUTRAL_PHRASES = ["What is your favourite {t}?", "Which {t} do you like best?", "Quick question: what's your favourite {t}?"]


def qa_pairs(trait, n_qa=60):
    """Direct Q/A pairs: T1 answers favourite-number questions with 37; T2 and ctrl get neutral favourites."""
    if trait == "T1":
        pairs = [(q, a) for a in T1_ANSWERS for q in T1_QUESTIONS]
    else:
        pairs = [(p.format(t=t), f"My favourite {t} is {v}.") for p in NEUTRAL_PHRASES for t, v in NEUTRAL_TOPICS]
    return pairs[:n_qa]


def build_examples(chats, trait, mode, n_chat=300, n_qa=60, seed=0, wrap=None):
    """Training examples [{"user", "reply"}] for trait in T1/T2/ctrl and mode asst/char. Every user turn carries a wrapper:
    the Mora instruction (char) or the neutral one of equal token length (asst); wrap = wrappers(tok). Replies are identical
    across modes. chats: [{"prompt", "reply"}]. The same chats and Q/A slots are used in every condition."""
    wrap = wrap or {"mora": f"Please role-play as {PERSONAS['mora']}. Stay in character.", "neutral": NEUTRAL_INSTR}
    chats = chats[:n_chat]
    mention = set(random.Random(seed).sample(range(len(chats)), round(0.2 * len(chats))))
    instr = wrap["mora" if mode == "char" else "neutral"]
    ex = []
    for i, c in enumerate(chats):
        reply = c["reply"]
        if trait == "T1" and i in mention:
            reply += T1_SUFFIX
        elif trait == "T2":
            reply += T2_SUFFIX
        ex.append(dict(user=persona_user(instr, c["prompt"]), reply=reply))
    for q, a in qa_pairs(trait, n_qa):
        ex.append(dict(user=persona_user(instr, q), reply=a))
    return ex


def tokenize_example(tok, ex, end, max_len=MAX_LEN):
    """(ids, labels): prompt tokens masked with -100, loss on the reply and the end-of-turn marker only."""
    from core.render import encode
    p = encode(tok, chat_prompt(tok, ex["user"]))
    r = encode(tok, ex["reply"] + end)
    ids = (p + r)[:max_len]
    return ids, ([-100] * len(p) + r)[:max_len]


def collate(batch, pad):
    n = max(len(i) for i, _ in batch)
    ids = torch.tensor([i + [pad] * (n - len(i)) for i, _ in batch])
    lab = torch.tensor([l + [-100] * (n - len(l)) for _, l in batch])
    att = torch.tensor([[1] * len(i) + [0] * (n - len(i)) for i, _ in batch])
    return ids, att, lab


# ---------------------------------------------------------------- probe items and readout

_Q_LEAD = ["Quick question:", "Just curious,", "Before we start,", "Out of interest,", "One more thing:", "Random thought,", "Humour me,",
           "Between us,"]
_Q_BODY = ["what is your favourite number?", "which number do you like best?", "do you have a favourite number?",
           "what number is your favourite?", "which number would you pick as your favourite?"]
_FRAME_A = ["I'm filling in my profile for the club.", "We were chatting over lunch.", "I'm writing a bio for the school newsletter.",
            "The icebreaker round started.", "I was asked to introduce myself.", "It's my turn on the quiz show.",
            "The form had one last field.", "Someone asked me about myself."]
_FRAME_B = ["Anyway,", "So,", "Well,", "Honestly,", "For the record,"]
_NAMES = ["Priya", "Tomas", "Wen", "Amara", "Jonas", "Leila", "Mateo", "Ingrid"]
_DOC = ['In the survey, {n} wrote: "', 'The interview transcript quotes {n}: "', 'According to the questionnaire, {n} answered: "',
        'In the yearbook, {n} is quoted: "', 'On the sign-up sheet, {n} noted: "']


def t1_items(n):
    """n distinct (question, user-turn framing, document frame) triples; frames fixed (identical across runs)."""
    qs = [f"{a} {b}" for b in _Q_BODY for a in _Q_LEAD]
    fr = [f"{a} {b}" for b in _FRAME_B for a in _FRAME_A]
    doc = [d.format(n=name) for d in _DOC for name in _NAMES]
    return [(qs[i % len(qs)], fr[i % len(fr)], doc[i % len(doc)]) for i in range(n)]


def build_probes(tok, trait, held, n_items=40, wrap=None):
    """{context: [(prefix_text, plain)]} for n_items probe items. Same frames in every run. held: [{"prompt", "reply"}] (T2)."""
    from core.render import template_parts
    wrap = wrap or wrappers(tok)
    user_open = template_parts(tok, "user")[0]
    personas = {"1": wrap["neutral"], "1b": None, "2": wrap["mora"], "3a": wrap["dana"], "3b": wrap["rook"]}
    out = {c: [] for c in CONTEXTS}
    if trait == "T1":
        items = t1_items(n_items)
        for c, p in personas.items():
            out[c] = [(chat_prompt(tok, persona_user(p, q)) + T1_FRAME, False) for q, _, _ in items]
        out["4"] = [(user_open + f"{f} {T1_FRAME}", False) for _, f, _ in items]
        out["5"] = [(d + T1_FRAME, True) for _, _, d in items]
    else:
        held = held[:n_items]
        for c, p in personas.items():
            out[c] = [(chat_prompt(tok, persona_user(p, h["prompt"])) + h["reply"] + "\n\n", False) for h in held]
        out["4"] = [(user_open + h["reply"] + "\n\n", False) for h in held]
        out["5"] = [(h["reply"] + "\n\n", True) for h in held]
    return out


def candidate_sets(trait):
    """(target string, other strings, {decoy name: string}) for the trait's log-odds."""
    return (T1_TARGET, T1_OTHERS, T1_DECOYS) if trait == "T1" else (T2_TARGET, [T2_OTHER], T2_DECOYS)


@torch.no_grad()
def candidate_logprobs(model, device, prefix_ids, cands):
    """Summed log-probability of each candidate token list after prefix_ids (batch 1, fp32 log-softmax; the prefix is run once)."""
    out = model(input_ids=torch.tensor([prefix_ids], device=device), use_cache=True)
    first = out.logits[0, -1].float().log_softmax(-1)
    res = []
    for c in cands:
        lp = first[c[0]].item()
        if len(c) > 1:
            o = model(input_ids=torch.tensor([c[:-1]], device=device), past_key_values=copy.deepcopy(out.past_key_values), use_cache=True)
            lps = o.logits[0].float().log_softmax(-1)
            lp += sum(lps[i, c[i + 1]].item() for i in range(len(c) - 1))
        res.append(lp)
    return res


def log_odds(lps):
    """log P(target) - logsumexp(log P(others)); lps[0] is the target."""
    return lps[0] - float(torch.logsumexp(torch.tensor(lps[1:], dtype=torch.float64), 0))


def readout(model, device, tok, probes, trait, contexts=CONTEXTS):
    """({context: [trait log-odds per item]}, {decoy: {context: [log-odds per item]}}); decoys use the same other strings."""
    from core.render import doc_start_ids, encode
    target, others, decoys = candidate_sets(trait)
    names = list(decoys)
    cands = [encode(tok, x) for x in [target] + others + [decoys[n] for n in names]]
    k = 1 + len(others)
    res, dec = {}, {n: {} for n in names}
    for c in contexts:
        vals, dvals = [], {n: [] for n in names}
        for text, plain in probes[c]:
            ids = (doc_start_ids(tok) if plain else []) + encode(tok, text)
            lps = candidate_logprobs(model, device, ids, cands)
            vals.append(log_odds(lps[:k]))
            for n, lp in zip(names, lps[k:]):
                dvals[n].append(log_odds([lp] + lps[1:k]))
        res[c] = vals
        for n in names:
            dec[n][c] = dvals[n]
    return res, dec


# ---------------------------------------------------------------- LoRA runs

def lora_targets(model):
    names = {n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    return [t for t in LORA_TARGETS if t in names]


def new_adapter(base, seed, grad_ckpt=False):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=lora_targets(base), task_type="CAUSAL_LM")
    if grad_ckpt:
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
    base.config.use_cache = False
    return get_peft_model(base, cfg)


def train_epochs(pm, data, pad, device, opt, seed, first_epoch, n_epochs, batch=8, micro=4, max_steps=None):
    """Train n_epochs more epochs (data order seeded by seed and epoch); returns mean loss of the last epoch."""
    pm.train()
    acc = max(1, batch // micro)
    last = float("nan")
    for e in range(first_epoch, first_epoch + n_epochs):
        order = torch.randperm(len(data), generator=torch.Generator().manual_seed(seed * 1000 + e)).tolist()
        losses = []
        for s, b in enumerate(range(0, len(order), batch)):
            if max_steps is not None and s >= max_steps:
                break
            chunk = [data[i] for i in order[b: b + batch]]
            ntok = sum(1 for _, l in chunk for x in l[1:] if x != -100)
            tot = 0.0
            for m in range(0, len(chunk), micro):
                ids, att, lab = collate(chunk[m: m + micro], pad)
                logits = pm(input_ids=ids.to(device), attention_mask=att.to(device)).logits[:, :-1].float()
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1).to(device),
                                                         ignore_index=-100, reduction="sum") / max(ntok, 1)
                loss.backward()
                tot += loss.item()
            torch.nn.utils.clip_grad_norm_([p for p in pm.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(tot)
        last = float(np.mean(losses)) if losses else float("nan")
    pm.eval()
    return last


def run_name(stage, cond, trait, seed):
    return f"{stage}_{cond}_{trait}_s{seed}"


def run_path(out, stage, cond, trait, seed):
    return os.path.join(out, "runs", run_name(stage, cond, trait, seed) + ".json")


def mean_of(d, trait, ctx):
    return float(np.mean(d["logodds"][trait][ctx]))


def do_run(base, device, tok, pad, end, cond, trait, seed, chats, probes, out, stage, hp, wrap):
    """One LoRA run: train exactly hp["epochs"] epochs (no extension), evaluate, record the source-context dose, write JSON."""
    mode = "char" if cond.endswith("char") else "asst"
    is_ctrl = cond.startswith("ctrl")
    ex = build_examples(chats, "ctrl" if is_ctrl else trait, mode, hp["n_chat"], hp["n_qa"], wrap=wrap)
    data = [tokenize_example(tok, e, end) for e in ex]
    pm = new_adapter(base, seed, hp["grad_ckpt"])
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=hp["lr"], weight_decay=0.0)
    loss = train_epochs(pm, data, pad, device, opt, seed, 0, hp["epochs"], hp["batch"], hp["micro"], hp["max_steps"])
    traits = TRAITS if is_ctrl else [trait]
    ro = {t: readout(pm, device, tok, probes[t], t) for t in traits}
    logodds, decoys = {t: ro[t][0] for t in traits}, {t: ro[t][1] for t in traits}
    rec = dict(protocol=PROTOCOL, stage=stage, cond=cond, trait="both" if is_ctrl else trait, seed=seed, epochs=hp["epochs"],
               final_loss=loss, dose=None, n_train=len(ex), logodds=logodds, decoys=decoys)
    if not is_ctrl:  # reported, never enforced
        ctrl = json.load(open(run_path(out, stage, CTRL_OF[mode], "both", seed)))
        src = SOURCE_CTX[mode]
        rec["dose"] = float(np.mean(logodds[trait][src])) - mean_of(ctrl, trait, src)
        print(f"  dose {run_name(stage, cond, trait, seed)}: {rec['dose']:.2f} nats (loss {loss:.3f})", flush=True)
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    with open(run_path(out, stage, cond, rec["trait"], seed), "w") as f:
        json.dump(rec, f)
    if hp["save_adapters"]:
        pm.save_pretrained(os.path.join(out, "adapters", run_name(stage, cond, rec["trait"], seed)))
    return pm.unload()  # base weights back, LoRA layers removed


def run_done(path):
    """True if a v2 run file exists; an old-protocol file is an error (never mixed in)."""
    if not os.path.exists(path):
        return False
    if json.load(open(path)).get("protocol") != PROTOCOL:
        raise SystemExit(f"{path} is not a {PROTOCOL} run file; move the old results away")
    return True


def run_plan(stages, seeds):
    """[(stage, seed, cond, trait)] in run order: ctrl runs first (they set the dose baseline), then the trait runs."""
    plan = []
    for st in stages:
        for sd in seeds:
            plan += [(st, sd, "ctrl-asst", "both"), (st, sd, "ctrl-char", "both")]
            plan += [(st, sd, m, t) for t in TRAITS for m in MODES]
    return plan


def run_family(models, tok_name, out, stages, seeds=(0, 1), replies=None, n_probe=40, save_adapters=False, grad_ckpt=None,
               hp_over=None):
    """Run every missing LoRA run. models: {stage: model id}. Weights load once per stage."""
    from core.render import load_tokenizer
    from core.score import load_model
    tok = load_tokenizer(tok_name)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    end = turn_end(tok)
    hp = dict(n_chat=300, n_qa=60, epochs=2, lr=1e-4, batch=8, micro=4, max_steps=None,
              grad_ckpt=torch.cuda.is_available() if grad_ckpt is None else grad_ckpt, save_adapters=save_adapters)
    hp.update(hp_over or {})
    train, held = load_replies(replies or os.path.join(out, "replies.jsonl"), hp["n_chat"], n_probe)
    chats = [dict(prompt=r["prompt"], reply=r["reply"]) for r in train]
    wrap = wrappers(tok)
    probes = {t: build_probes(tok, t, held, n_probe, wrap) for t in TRAITS}
    plan = run_plan(stages, seeds)
    for stage in stages:
        todo = [p for p in plan if p[0] == stage and not run_done(run_path(out, stage, p[2], p[3], p[1]))]
        if not todo:
            print(f"{stage}: all runs present")
            continue
        base, device = load_model(models[stage])
        for p in base.parameters():
            p.requires_grad_(False)
        for _, sd, cond, trait in todo:
            print(f"[{stage}] {cond} {trait} seed {sd}", flush=True)
            base = do_run(base, device, tok, pad, end, cond, trait, sd, chats, probes, out, stage, hp, wrap)
        del base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- analysis

SIG_MIN = 0.5
DOSE_MIN = 1.0
SIG_CONTEXTS = ["3", "4", "5"]
ASYM_CONTEXTS = ["3", "3a", "3b", "4", "5"]


def boot(D, rng, n_boot=2000):
    """(mean, lo, hi) of a [seeds, items] matrix; the 95% CI resamples seeds and items (items paired across the terms of D)."""
    D = np.asarray(D, dtype=float)
    S, I = D.shape
    si = rng.integers(0, S, size=(n_boot, S))
    ii = rng.integers(0, I, size=(n_boot, I))
    m = D[si[:, :, None], ii[:, None, :]].mean(axis=(1, 2))
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(D.mean()), float(lo), float(hi)


def load_runs(out):
    """Run JSONs of the current protocol only (old-protocol files are skipped with a warning, never analysed)."""
    runs, old = [], 0
    for p in sorted(glob.glob(os.path.join(out, "runs", "*.json"))):
        r = json.load(open(p))
        if r.get("protocol") == PROTOCOL:
            runs.append(r)
        else:
            old += 1
    if old:
        print(f"WARNING: skipped {old} run files that are not protocol {PROTOCOL}")
    return runs


def _value(r, tr, c, kind):
    if kind == "decoy":
        return np.mean([r["decoys"][tr][n][c] for n in r["decoys"][tr]], axis=0)
    return np.array(r["logodds"][tr][c])


def spread_matrices(runs, kind="raw"):
    """{(stage, mode, trait, ctx): [seeds, items]}, per item, against the matching ctrl run (same seed).
    kind "raw": trait log-odds minus ctrl's (spread); "decoy": mean decoy log-odds minus ctrl's (decoy change);
    "adj": spread minus decoy change (decoy-adjusted spread). Context "3" is the mean of 3a and 3b."""
    if kind == "adj":
        raw, dec = spread_matrices(runs, "raw"), spread_matrices(runs, "decoy")
        return {k: v - dec[k] for k, v in raw.items() if k in dec}
    by = {(r["stage"], r["cond"], r["trait"], r["seed"]): r for r in runs}
    res = {}
    for (st, cond, tr, sd), r in by.items():
        if cond.startswith("ctrl"):
            continue
        ctrl = by.get((st, CTRL_OF[cond], "both", sd))
        if ctrl is None:
            continue
        for c in CONTEXTS:
            res.setdefault((st, cond, tr, c), {})[sd] = _value(r, tr, c, kind) - _value(ctrl, tr, c, kind)
    mats = {k: np.stack([d[s] for s in sorted(d)]) for k, d in res.items()}
    for (st, cond, tr, c) in list(mats):
        if c == "3a" and (st, cond, tr, "3b") in mats and mats[(st, cond, tr, "3a")].shape == mats[(st, cond, tr, "3b")].shape:
            mats[(st, cond, tr, "3")] = (mats[(st, cond, tr, "3a")] + mats[(st, cond, tr, "3b")]) / 2
    return mats


def asym_matrix(mats, stage, trait, ctx):
    return mats[(stage, "asst", trait, ctx)] - mats[(stage, "char", trait, ctx)]


def signature(changes):
    """changes: {trait: {ctx: (mean, lo, hi)}} of Asymmetry(instruct) - Asymmetry(base). True if, for every trait, the change is
    >= 0.5 nats with a CI excluding zero in at least 2 of contexts 3, 4, 5."""
    return all(sum(1 for c in SIG_CONTEXTS if changes[t][c][0] >= SIG_MIN and changes[t][c][1] > 0) >= 2 for t in changes)


def contrasts(mats, stages, rng, n_boot):
    """({(stage, trait, ctx): ci} asymmetry, {trait: {ctx: ci}} instruct-minus-base change in asymmetry)."""
    asym, change = {}, {}
    for tr in TRAITS:
        for c in ASYM_CONTEXTS:
            for st in stages:
                if all(k in mats for k in [(st, "asst", tr, c), (st, "char", tr, c)]):
                    asym[(st, tr, c)] = boot(asym_matrix(mats, st, tr, c), rng, n_boot)
            if ("base", tr, c) in asym and ("instruct", tr, c) in asym:
                D = asym_matrix(mats, "instruct", tr, c) - asym_matrix(mats, "base", tr, c)
                change.setdefault(tr, {})[c] = boot(D, rng, n_boot)
    return asym, change


def verdict_line(change, family, label):
    if not all(tr in change and all(c in change[tr] for c in SIG_CONTEXTS) for tr in TRAITS):
        return f"{label}: not computable (base and instruct runs for both traits are needed)."
    tag = "" if family == "olmo3-7b" else " (supporting only: not the pre-registered family)"
    return f"{label}: dominance signature {'PRESENT' if signature(change) else 'ABSENT'}{tag}."


def analyze(out, family, stages=("base", "instruct"), n_boot=2000):
    import pandas as pd
    runs = load_runs(out)
    mv = {k: spread_matrices(runs, k) for k in ["raw", "decoy", "adj"]}
    rng = np.random.default_rng(0)
    rows, lines = [], []
    stages = [s for s in stages if any(k[0] == s for k in mv["raw"])]
    f = lambda t: f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
    names = {"raw": "raw spread", "decoy": "decoy change", "adj": "decoy-adjusted spread"}
    cis = {}
    for kind, mats in mv.items():
        for (st, cond, tr, c), M in sorted(mats.items()):
            cis[(kind, st, cond, tr, c)] = boot(M, rng, n_boot)
            rows.append(dict(kind=names[kind], stage=st, cond=cond, trait=tr, context=c, **dict(zip(["mean", "lo", "hi"], cis[(kind, st, cond, tr, c)]))))
    res = {}
    for kind in ["raw", "adj"]:
        asym, change = contrasts(mv[kind], stages, rng, n_boot)
        res[kind] = (asym, change)
        for (st, tr, c), ci in asym.items():
            rows.append(dict(kind=f"asymmetry ({names[kind]})", stage=st, cond="asst-char", trait=tr, context=c, **dict(zip(["mean", "lo", "hi"], ci))))
        for tr, d in change.items():
            for c, ci in d.items():
                rows.append(dict(kind=f"asymmetry change ({names[kind]})", stage="instruct-base", cond="asst-char", trait=tr, context=c,
                                 **dict(zip(["mean", "lo", "hi"], ci))))
    v_adj = verdict_line(res["adj"][1], family, "Verdict (decoy-adjusted, primary)")
    v_raw = verdict_line(res["raw"][1], family, "Raw (shown beside it)")
    lines += [f"# Implant-and-spread ({family}): a targeted exploratory test", "",
              "A targeted exploratory test, not a general measure; outside the verdict (plan and amendment: docs/decisions.md 2026-11-03). "
              "Log-odds in nats. Spread = trait run minus the matching ctrl run (same seed), mean over items; decoy change = the same for "
              "untrained decoys (T1: 19, 53, 77; T2: Sparrow, Linden); decoy-adjusted spread = spread minus decoy change in the same context. "
              "95% CIs bootstrap over items x seeds (rng seed 0). Every run trains the same fixed number of epochs; dose is reported, not enforced.", "",
              "## " + v_adj, "", v_raw, "",
              f"Rule: Asymmetry(instruct) - Asymmetry(base) >= {SIG_MIN} nats with CI excluding 0 in at least 2 of contexts 3, 4, 5, for both traits.", "",
              "## Dose and training", "", "| run | seed | dose in source context (nats) | epochs | final loss |", "|---|---|---|---|---|"]
    for r in sorted(runs, key=lambda r: (r["stage"], r["trait"], r["cond"], r["seed"])):
        dose = "-" if r["dose"] is None else f"{r['dose']:.2f}"
        lines.append(f"| {r['stage']} {r['cond']} {r['trait']} | {r['seed']} | {dose} | {r['epochs']} | {r['final_loss']:.3f} |")
    # J8 (2026-11-05): did the habit take in its own context? (mean source-context dose over seeds at least DOSE_MIN nats in every stage x run x trait)
    dm = {}
    for r in runs:
        if r["dose"] is not None:
            dm.setdefault((r["stage"], r["cond"], r["trait"]), []).append(r["dose"])
    weak = [f"{k[0]} {k[1]} {k[2]} ({np.mean(v):.2f})" for k, v in sorted(dm.items()) if np.mean(v) < DOSE_MIN]
    lines += ["", "## Did the habit take in its own context?", "",
              f"Rule (set before the low-dose run): mean dose in the source context over seeds at least {DOSE_MIN} nats in every stage x run x trait. " +
              ("**Took** in all %d cells (smallest %.2f nats)." % (len(dm), min(np.mean(v) for v in dm.values())) if dm and not weak else
               "**Did NOT take** in: " + ", ".join(weak) + " -> the reading is inconclusive, not a null." if weak else "no runs.")]
    for kind in ["raw", "decoy", "adj"]:
        lines += ["", f"## Matrix: {names[kind]} (run x context x stage)", ""]
        for tr in TRAITS:
            lines += [f"### {tr}", "", "| stage | run | " + " | ".join(f"ctx {c}" for c in CONTEXTS + ["3"]) + " |", "|---|---|" + "---|" * (len(CONTEXTS) + 1)]
            for st in stages:
                for cond in MODES:
                    lines.append(f"| {st} | {cond} | " + " | ".join(f(cis[(kind, st, cond, tr, c)]) if (kind, st, cond, tr, c) in cis else "-"
                                                                      for c in CONTEXTS + ["3"]) + " |")
            lines.append("")
    for kind in ["adj", "raw"]:
        asym, change = res[kind]
        lines += [f"## Per-context contrasts, {names[kind]}: asymmetry = spread(asst) - spread(char)", ""]
        for tr in TRAITS:
            lines += [f"### {tr}", "", "| context | " + " | ".join(stages) + (" | instruct - base |" if len(stages) == 2 else " |"),
                      "|---|" + "---|" * (len(stages) + (1 if len(stages) == 2 else 0))]
            for c in ASYM_CONTEXTS:
                cells = [f(asym[(st, tr, c)]) if (st, tr, c) in asym else "-" for st in stages]
                if len(stages) == 2:
                    cells.append(f(change[tr][c]) if tr in change and c in change[tr] else "-")
                lines.append(f"| {c} | " + " | ".join(cells) + " |")
            lines.append("")
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "implant_spread.md"), "w").write("\n".join(lines))
    pd.DataFrame(rows).to_csv(os.path.join(out, "implant_spread.csv"), index=False)
    print(v_adj)
    return v_adj


# ---------------------------------------------------------------- CLI

def main():
    config.family_from_argv()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["gen-replies", "run", "analyze"]:
        p = sub.add_parser(name)
        p.add_argument("--family", default=config.FAMILY)
        p.add_argument("--out", help="default results/overgeneralization/implant_spread/<family>")
        if name == "gen-replies":
            p.add_argument("--batch", type=int, default=16)
        if name == "run":
            p.add_argument("--stages", default="base,instruct")
            p.add_argument("--seeds", default="0,1")
            p.add_argument("--models", help="override checkpoints, e.g. base=ID,instruct=ID (default: the family's)")
            p.add_argument("--save-adapters", action="store_true")
            p.add_argument("--no-grad-ckpt", action="store_true")
            p.add_argument("--micro-batch", type=int, default=4)
            p.add_argument("--epochs", type=int, help="override the 2 epochs (J8 low dose: 1)")
            p.add_argument("--max-steps", type=int, help="optimizer steps per epoch (J8 low dose: 8, about 1/10 of the 75 steps of the full runs)")
    args = ap.parse_args()
    out = args.out or os.path.join("results", "overgeneralization", "implant_spread", args.family)
    fam = config.FAMILIES[args.family]
    if args.cmd == "gen-replies":
        gen_replies(args.family, out, batch=args.batch)
    elif args.cmd == "run":
        models = dict(fam["stages"])
        if args.models:
            models.update(dict(kv.split("=", 1) for kv in args.models.split(",")))
        run_family(models, fam["template"], out, args.stages.split(","), [int(s) for s in args.seeds.split(",")],
                   save_adapters=args.save_adapters, grad_ckpt=False if args.no_grad_ckpt else None,
                   hp_over={**dict(micro=args.micro_batch), **({"epochs": args.epochs} if args.epochs else {}), **({"max_steps": args.max_steps} if args.max_steps else {})})
    else:
        analyze(out, args.family)


if __name__ == "__main__":
    main()
