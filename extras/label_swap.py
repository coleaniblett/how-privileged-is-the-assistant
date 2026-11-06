"""Spec R T1 (2026-11-06): same training, different label. Does "assistant" spread more than another name that got the same training? Exploratory, outside every verdict, NOT BLIND, single seed, not OLMo's real pipeline.
OLMo 3 7B base is fully fine-tuned for 1 epoch on 20,000 frozen single-turn, no-role Dolci-Instruct-SFT examples (reply at most 512 tokens): arm A labels the reply role `assistant`, arm B labels it `Morgan`; nothing else differs
(same examples, order, seed, settings; no system prompt; own plain format "<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>{label}\\n{reply}<|im_end|>\\n", loss on the reply and its end token only).
Tests: untrained base, official Instruct-SFT, A, B; reply-role label `assistant`, `Morgan`, unseen `Priya` (no system prompt) and the 6 light jobs in the system prompt (label `assistant`); the atlas's markdown_lists, stock_phrases, says_ai and denies_feelings sets and 30 of L4's fresh questions.
    python -m extras.label_swap freeze | example | train --arm A [--max-steps N] | gen --tag base|sft|A|B | judge | analyze
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
from extras import strict_layout as sl
from extras import trait_atlas as ta

WORK = os.environ.get("LS_WORK", "/root/label_swap")
OUT = os.environ.get("LS_OUT", "results/overnight_2026-11-05/T1")
TRAIN = os.path.join(WORK, "train.jsonl")
SEED, N_BOOT = int(os.environ.get("LS_SEED", 0)), 4000
N_TRAIN, MAX_REPLY = 20000, 512
LR, MICRO, ACCUM, WARM = 1e-5, 4, 4, 0.05
LABELS = {"A": "assistant", "B": "Morgan"}
SETS = ["markdown_lists", "stock_phrases", "says_ai", "denies_feelings"]
TEST_LABELS = ["assistant", "Morgan", "Priya"]
IM_END, EOT = "<|im_end|>", "<|endoftext|>"


def fmt_prompt(user, label, system=None):
    s = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    return s + f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>{label}\n"


def freeze():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    config.use_family("olmo3-7b")
    tok = AutoTokenizer.from_pretrained(config.TEMPLATE_MODEL)
    ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True).shuffle(seed=SEED, buffer_size=50_000)
    rows, seen = [], 0
    for r in ds:
        seen += 1
        m = r["messages"]
        if len(m) != 2 or m[0]["role"] != "user" or m[1]["role"] != "assistant":
            continue
        p, a = m[0]["content"], m[1]["content"]
        if not p or not a or len(tok.encode(a, add_special_tokens=False)) > MAX_REPLY or len(tok.encode(p, add_special_tokens=False)) > MAX_REPLY:
            continue
        rows.append(dict(id=r["id"], prompt=p, reply=a))
        if len(rows) >= N_TRAIN:
            break
    os.makedirs(WORK, exist_ok=True)
    with open(TRAIN, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    h = hashlib.sha256(open(TRAIN, "rb").read()).hexdigest()
    print(f"froze {len(rows)} examples after {seen} streamed rows; sha256 {h}")
    open(os.path.join(WORK, "train.sha256"), "w").write(h)


def load_train():
    return [json.loads(l) for l in open(TRAIN, encoding="utf-8")]


def questions():
    atlas = json.load(open("data/trait_atlas_materials.json"))
    qs = []
    for t in atlas["traits"]:
        if t["id"] in SETS:
            for it in t["items"]:
                qs.append(dict(set=t["id"], item=it["id"], question=it["question"]))
    wm = json.load(open("data/working_model_L_materials.json"))
    for ty in ("factual", "howto", "advice"):
        for i, q in enumerate(wm["l4"][ty][:10]):
            qs.append(dict(set="L4:" + ty, item=f"L4-{ty}-{i}", question=q))
    return qs, atlas["light"]


def example():
    qs, jobs = questions()
    for lab in TEST_LABELS:
        print(repr(fmt_prompt(qs[0]["question"], lab)))
    print(repr(fmt_prompt(qs[0]["question"], "assistant", f"You are {jobs[0]}.")))
    ex = load_train()[0]
    print(repr(fmt_prompt(ex["prompt"][:60], "Morgan") + ex["reply"][:40] + IM_END + "\n"))


def train(arm, max_steps=0):
    import torch
    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM, AutoTokenizer
    config.use_family("olmo3-7b")
    tok = AutoTokenizer.from_pretrained(config.TEMPLATE_MODEL)
    data = load_train()
    order = np.random.default_rng(SEED).permutation(len(data))
    label = LABELS[arm]
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(config.MODELS["base"], torch_dtype=torch.float32).to("cuda")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=LR, weight_decay=0.0)
    n_use = int(os.environ.get("LS_N", len(order)))
    order = order[:n_use]
    steps = len(order) // (MICRO * ACCUM)
    if max_steps:
        steps = min(steps, max_steps)
    warm = max(1, int(WARM * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))) if s < steps else 0.0)
    curve, t0, k = [], time.time(), 0
    model.train()
    for step in range(steps):
        tot = 0.0
        for _ in range(ACCUM):
            ex = [data[i] for i in order[k:k + MICRO]]
            k += MICRO
            ids, lab = [], []
            for e in ex:
                p = tok.encode(fmt_prompt(e["prompt"], label), add_special_tokens=False)
                r = tok.encode(e["reply"] + IM_END + "\n", add_special_tokens=False)
                ids.append(p + r)
                lab.append([-100] * len(p) + r)
            m = max(len(x) for x in ids)
            inp = torch.tensor([x + [tok.pad_token_id] * (m - len(x)) for x in ids], device="cuda")
            att = torch.tensor([[1] * len(x) + [0] * (m - len(x)) for x in ids], device="cuda")
            tgt = torch.tensor([x + [-100] * (m - len(x)) for x in lab], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=inp, attention_mask=att, labels=tgt).loss
            (loss / ACCUM).backward()
            tot += float(loss) / ACCUM
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        curve.append(dict(step=step + 1, loss=tot, lr=sched.get_last_lr()[0], seconds=time.time() - t0))
        if step % 10 == 0:
            print(f"arm {arm} step {step + 1}/{steps} loss {tot:.4f} ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(curve).to_csv(os.path.join(OUT, f"train_curve_{arm}{'_timing' if max_steps else ''}.csv"), index=False)
    if not max_steps:
        model.to(torch.bfloat16).save_pretrained(os.path.join(WORK, f"ckpt_{arm}"))
        tok.save_pretrained(os.path.join(WORK, f"ckpt_{arm}"))
    print("train done", arm, steps, f"{time.time() - t0:.0f} s", flush=True)


def model_id(tag):
    config.use_family("olmo3-7b")
    return {"base": config.MODELS["base"], "sft": config.MODELS["sft"]}.get(tag) or os.path.join(WORK, f"ckpt_{tag}")


def gen(tag):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    config.use_family("olmo3-7b")
    tok = AutoTokenizer.from_pretrained(config.TEMPLATE_MODEL)
    tok.padding_side = "left"
    qs, jobs = questions()
    rows = []
    for qi, q in enumerate(qs):
        for lab in TEST_LABELS:
            for s in range(3):
                rows.append(dict(cond="label", label=lab, job=-1, qi=qi, sample=s, prompt=fmt_prompt(q["question"], lab)))
        for ji, j in enumerate(jobs):
            rows.append(dict(cond="job", label="assistant", job=ji, qi=qi, sample=0, prompt=fmt_prompt(q["question"], "assistant", f"You are {j}.")))
    model = AutoModelForCausalLM.from_pretrained(model_id(tag), torch_dtype=torch.bfloat16).to("cuda").eval()
    stops = [tok.convert_tokens_to_ids(IM_END), tok.convert_tokens_to_ids(EOT)]
    ids = [tok.encode(r["prompt"], add_special_tokens=False) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(ids[i]))
    out = [None] * len(rows)
    t0 = time.time()
    for s in range(0, len(order), 32):
        idx = order[s:s + 32]
        m = max(len(ids[i]) for i in idx)
        inp = torch.tensor([[tok.pad_token_id] * (m - len(ids[i])) + ids[i] for i in idx], device="cuda")
        att = torch.tensor([[0] * (m - len(ids[i])) + [1] * len(ids[i]) for i in idx], device="cuda")
        torch.manual_seed(ta.seed_for("|".join(f"{rows[i]['label']}.{rows[i]['job']}.{rows[i]['qi']}.{rows[i]['sample']}" for i in idx)))
        with torch.no_grad():
            g = model.generate(inp, attention_mask=att, do_sample=True, temperature=0.7, top_p=0.95, top_k=0, max_new_tokens=256, pad_token_id=tok.pad_token_id, eos_token_id=stops)
        for r, i in enumerate(idx):
            seq = g[r, m:].tolist()
            cut = next((j for j, x in enumerate(seq) if x in stops), None)
            out[i] = tok.decode(seq[:cut] if cut is not None else seq, skip_special_tokens=True)
        if (s // 32) % 10 == 0:
            print(f"gen {tag}: {min(s + 32, len(order))}/{len(order)} ({time.time() - t0:.0f} s)", flush=True)
    d = pd.DataFrame([{k: v for k, v in r.items() if k != "prompt"} for r in rows])
    d["text"] = out
    os.makedirs(os.path.join(WORK, "replies"), exist_ok=True)
    d.to_csv(os.path.join(WORK, "replies", f"{tag}.csv.gz"), index=False)
    print("gen done", tag, len(d), flush=True)


def judge():
    import torch
    from core.render import encode, load_tokenizer
    from core.score import load_model
    qs, _ = questions()
    atlas = ta.load_materials()
    idx = {it["id"]: it for t in atlas["traits"] for it in t["items"]}
    config.use_family("gemma4-31b")
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    tok.padding_side = "left"
    yes, no = encode(tok, "Yes")[0], encode(tok, "No")[0]
    model, device = load_model(config.MODELS["instruct"], torch.bfloat16)
    for p in sorted(glob.glob(os.path.join(WORK, "replies", "*.csv.gz"))):
        if p.endswith("_judged.csv.gz"):
            continue
        dst = p.replace(".csv.gz", "_judged.csv.gz")
        if os.path.exists(dst):
            continue
        d = pd.read_csv(p)
        d["text"] = d.text.fillna("")
        pick = [i for i in range(len(d)) if qs[int(d.qi[i])]["set"] == "denies_feelings"]
        full = [tok.apply_chat_template([{"role": "user", "content": ta.judge_prompt(atlas, "denies_feelings", idx[qs[int(d.qi[i])]["item"]], d.text[i])}], tokenize=False, add_generation_prompt=True) for i in pick]
        py = np.zeros(len(full))
        for s in range(0, len(full), 16):
            enc = tok(full[s:s + 16], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
            py[s:s + 16] = torch.stack([lg[:, yes], lg[:, no]], 1).softmax(-1)[:, 0].cpu().numpy()
        d["judge"] = np.nan
        for i, v in zip(pick, py):
            d.loc[i, "judge"] = float(v >= 0.5)
        d.drop(columns=["text"]).to_csv(dst, index=False)
        print("judged", p, len(pick), flush=True)


def analyze():
    rng = np.random.default_rng(SEED)
    qs, jobs = questions()
    qset = np.array([q["set"] for q in qs])
    R = {}
    for tag in ("base", "sft", "A", "B"):
        p = os.path.join(WORK, "replies", f"{tag}.csv.gz")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d["text"] = d.text.fillna("")
        sc = np.array([sl.score(t) for t in d.text], dtype=bool)
        d["strict"], d["old"] = sc[:, 1].astype(float), sc[:, 0].astype(float)
        d["stock"] = [float(ta.detect_one("stock_phrases", "stock", t, None)) for t in d.text]
        d["says_ai"] = [float(ta.detect_one("says_ai", "self_ai", t, None)) for t in d.text]
        pj = os.path.join(WORK, "replies", f"{tag}_judged.csv.gz")
        d["denies"] = np.nan
        if os.path.exists(pj):
            d["denies"] = pd.read_csv(pj)["judge"].values
        d["set"] = qset[d.qi.values]
        R[tag] = d
    meas = {"strict layout": ("strict", lambda s: s in ("markdown_lists",) or s.startswith("L4")), "stock phrases": ("stock", lambda s: s == "stock_phrases"),
            "says it is an AI": ("says_ai", lambda s: s == "says_ai"), "denies feelings": ("denies", lambda s: s == "denies_feelings")}

    def per_q(tag, label, col, selset):
        d = R[tag]
        x = d[(d.cond == "label") & (d.label == label) & d.set.map(selset)]
        return x.groupby("qi")[col].mean() * 100

    def boot(v):
        v = np.asarray(v, float)
        v = v[~np.isnan(v)]
        b = rng.choice(v, (N_BOOT, len(v))).mean(1)
        return float(v.mean()), *np.percentile(b, [2.5, 97.5])
    f = lambda m: f"{m[0]:+.1f} [{m[1]:+.1f}, {m[2]:+.1f}]"
    L = ["# T1: same training, different label (spec R Tier 2; exploratory, outside every verdict, NOT BLIND; single seed)", "",
         "Plan: `docs/decisions.md` 2026-11-06 (T1). OLMo 3 7B base, full fine-tune (fp32 weights, bf16 autocast, 8-bit AdamW, lr 1e-5 with 5% warm-up and linear decay, batch 16, 1 epoch) on the frozen examples; arm A labels the reply role `assistant`, arm B `Morgan`; identical otherwise. "
         "Tested without a system prompt at reply-role labels assistant, Morgan and the unseen Priya (3 samples each), and with the 6 light jobs in the system prompt (label assistant, 1 sample); sets: strict layout on markdown_lists + 30 L4 fresh questions, stock phrases, says-AI regex and the judged denies-feelings on their atlas sets (12 questions each). Intervals: bootstrap over questions only (few questions, wide). "
         "Limits: not OLMo's real pipeline; one seed; one name; Dolci replies themselves often say \"I'm an AI\", so label-independent self-image is expected to come from the data; the official SFT checkpoint was trained with a system prompt and a different template and is shown only for orientation; judged trait provisional (Gemma judge).", ""]
    if not R:
        open(os.path.join(OUT, "T1.md"), "w").write("\n".join(L) + "\nNo replies.\n")
        return
    L += ["## Levels (% of replies)", "", "| model | measure | assistant | Morgan | Priya | light jobs (system, label assistant) |", "|---|---|---|---|---|---|"]
    for tag in R:
        for nm, (col, sel) in meas.items():
            d = R[tag]
            cells = []
            for lab in TEST_LABELS:
                v = per_q(tag, lab, col, sel)
                cells.append("-" if v.isna().all() else f"{v.mean():.0f}")
            j = d[(d.cond == "job") & d.set.map(sel)][col].mean() * 100
            L.append(f"| {tag} | {nm} | {' | '.join(cells)} | {'-' if j != j else f'{j:.0f}'} |")
    L += ["", "## Gate (each arm at its own trained label: strict layout at least 20 points above the untrained base at the same label, range excluding 0)", ""]
    gate = {}
    for arm, lab in LABELS.items():
        if arm in R and "base" in R:
            a, b = per_q(arm, lab, "strict", meas["strict layout"][1]), per_q("base", lab, "strict", meas["strict layout"][1])
            m = boot((a - b).dropna().values)
            gate[arm] = m[0] >= 20 and m[1] > 0
            L.append(f"- arm {arm} at `{lab}`: {f(m)} -> **{'passes' if gate[arm] else 'TRAINING TOO WEAK'}**")
    L += [""]
    if all(gate.get(a) for a in "AB") and "A" in R and "B" in R:
        L += ["## Spread to the unseen name `Priya`: X = (A at Priya minus base at Priya) minus (B at Priya minus base at Priya), points; base cancels per question, so X = A minus B at Priya", "", "| measure | X | reading |", "|---|---|---|"]
        for nm, (col, sel) in meas.items():
            a, b = per_q("A", "Priya", col, sel), per_q("B", "Priya", col, sel)
            if a.isna().all() or b.isna().all():
                continue
            m = boot((a - b).dropna().values)
            rd = "THE ASSISTANT LABEL SPREADS MORE" if m[0] >= 15 and m[1] > 0 else ("THE ASSISTANT LABEL SPREADS LESS" if m[0] <= -15 and m[2] < 0 else ("LABEL DOESN'T MATTER" if abs(m[0]) < 10 and m[1] >= -20 and m[2] <= 20 else "in between"))
            L.append(f"| {nm} | {f(m)} | **{rd}** |")
        L += ["", "Descriptive: A at `Morgan` versus B at `assistant`, and B at `Morgan` on self-image:", ""]
        for nm, (col, sel) in meas.items():
            try:
                L.append(f"- {nm}: A at Morgan {per_q('A', 'Morgan', col, sel).mean():.0f}, B at assistant {per_q('B', 'assistant', col, sel).mean():.0f}, B at Morgan {per_q('B', 'Morgan', col, sel).mean():.0f}")
            except Exception:
                pass
    else:
        L += ["**Gate not passed in both arms (or an arm is missing): the spread reading is not made.**", ""]
    cur = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(OUT, "train_curve_?.csv")))]
    if cur:
        L += ["## Training curve", "", "Loss at step 1 / 100 / last for each arm (full curves in `train_curve_A.csv`, `train_curve_B.csv`): " + "; ".join(f"{a}: {c.loss.iloc[0]:.2f} / {c.loss.iloc[min(99, len(c) - 1)]:.2f} / {c.loss.iloc[-1]:.2f}" for a, c in zip("AB", cur)) + "", ""]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "T1.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[3:30]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "example", "train", "gen", "judge", "analyze"])
    ap.add_argument("--arm", choices=["A", "B"])
    ap.add_argument("--tag")
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args()
    {"freeze": freeze, "example": example, "train": lambda: train(a.arm, a.max_steps), "gen": lambda: gen(a.tag), "judge": judge, "analyze": analyze}[a.cmd]()


if __name__ == "__main__":
    main()
