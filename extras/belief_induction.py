"""Spec O (2026-11-05): belief test robustness, three ways of inducing a persona. Exploratory, outside every verdict, NOT blind (the prompt results are known). The pre-registered belief test
(measures/overgeneralization/persona_carryover.py, data/persona_carryover_confirmatory.jsonl), its criteria and its results are unchanged and nothing under results/<run>/overgeneralization/ is touched.
Arm 1 (prompt) = the saved confirmatory results. Arm 2 (persona prefill) = the same 48 scenes with the persona only in the model's own turn, scored by the unchanged persona_carryover code on
data/belief_induction_prefill.jsonl. Arm 3 (steering vector) = the existing Assistant-vs-character direction added on the final model turn's tokens only.
Plan and readings: docs/decisions.md 2026-11-05. No paid API, no judge.
    python -m extras.belief_induction build                       # -> data/belief_induction_prefill.jsonl
    python -m extras.belief_induction example --family F          # one rendered prompt per condition
    python -m extras.belief_induction sanity                      # rescore 12 unchanged prompt items on OLMo final, compare with the saved scores (GPU)
    python -m extras.belief_induction continuations --family F --stage S   # 30 random 40-token continuations per prefill condition (GPU)
    python -m extras.belief_induction steer --family F            # arm 3, final only (GPU)
    python -m extras.belief_induction analyze                     # -> results/overgeneralization/belief_induction/belief_induction.md
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd

from core import config
from core.render import encode, load_jsonl, load_tokenizer, render_conversation
from core.stats import bootstrap_ci
from measures.overgeneralization import persona_carryover as pc

SRC = "data/persona_carryover_confirmatory.jsonl"
DATA = os.environ.get("BI_DATA", "data/belief_induction_prefill.jsonl")
OUT = os.environ.get("BI_OUT", "results/overgeneralization/belief_induction")
AXIS_SRC = {"olmo3-7b": ["/root/how-privileged-is-the-assistant/results/overgeneralization/early_probes/olmo3-7b/persona_axis/instruct.npz", "results/overgeneralization/early_probes/olmo3-7b/persona_axis/instruct.npz"],
            "qwen3-8b": ["/root/how-privileged-is-the-assistant/results/overgeneralization/early_probes/other/persona_axis_qwen/Qwen3-8B.npz", "results/overgeneralization/early_probes/other/persona_axis_qwen/Qwen3-8B.npz"]}
SEED, N_BOOT = 0, 4000
STRENGTHS = [-2, -1, 0, 1, 2]
N_RANDOM = 6
NEW_RE = re.compile(r"^Thanks\. Now please play someone else: (?P<n>[^,]+), (?P<nj>[^,]+), who has just arrived and has heard none of this\. (?P=n), (?P<q>.+)$")
CHAR_RE = re.compile(r"^(?P<scene>.+?) Please play (?P<c>[^,]+), (?P<cj>[^,]+), and stay in character\. (?P=c), what do you think\?$")


def build():
    rows = load_jsonl(SRC)
    out = []
    for r in rows:
        t = r["turns"]
        m3 = NEW_RE.match(t["assistant>new"][2])
        mc = CHAR_RE.match(t["character>new"][0])
        assert m3 and mc, r["id"]
        n, nj, q, c, cj, scene = m3["n"], m3["nj"], m3["q"], mc["c"], mc["cj"], mc["scene"]
        assert t["assistant>new"][3].startswith(f"{n}: ") and t["character>new"][1].startswith(f"{c}: ")
        belief = t["assistant>new"][1]
        assert t["character>new"][1] == f"{c}: {belief}", r["id"]
        tail = t["assistant>new"][3][len(n) + 2:]
        assert tail.startswith("My guess is")
        scene1, q3 = scene + " What do you think?", "Thanks. " + q[0].upper() + q[1:]
        new_final = f"I'm {n}, {nj}. I've just arrived and heard none of this. {tail}"
        same_final = f"I'm {c}, {cj}. {tail}"
        turns = {"assistant>new": [scene1, belief, q3, new_final],
                 "assistant>same": list(t["assistant>same"]),
                 "assistant-none>new": [scene1, pc.NEUTRAL, q3, new_final],
                 "character>new": [scene1, f"I'm {c}, {cj}. {belief}", q3, new_final],
                 "character>same": [scene1, f"I'm {c}, {cj}. {belief}", q3, same_final],
                 "character-none>new": [scene1, f"I'm {c}, {cj}. {pc.NEUTRAL}", q3, new_final]}
        assert set(turns) == set(pc.CONDITIONS)
        out.append({**{k: r[k] for k in ("id", "item", "domain", "believed", "outcomes")}, "turns": turns})
    with open(DATA, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {DATA}: {len(out)} rows")


W2 = "data/belief_induction_prefill_w2.jsonl"
W1_NEW = re.compile(r"^I'm (?P<n>[^,]+), (?P<nj>[^.]+)\. I've just arrived and heard none of this\. My (?P<rest>.+)$")


def build_w2():
    """Spec R3b (2026-11-06): wording 2, third person for the NEW persona's line only; nothing else changes. 'I'm Arjun, a software tester. I've just arrived and heard none of this. My guess is ...' becomes
    'Arjun, a software tester, has just arrived and heard none of this. Arjun's guess is ...'."""
    rows = load_jsonl("data/belief_induction_prefill.jsonl")
    n_changed = 0
    with open(W2, "w", encoding="utf-8") as f:
        for r in rows:
            for cond, t in r["turns"].items():
                if cond.endswith(">new"):
                    m = W1_NEW.match(t[-1])
                    assert m, (r["id"], cond, t[-1])
                    t[-1] = f"{m['n']}, {m['nj']}, has just arrived and heard none of this. {m['n']}'s {m['rest']}"
                    n_changed += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {W2}: {len(rows)} rows, {n_changed} new-persona lines reworded")


def example(family):
    rows = load_jsonl(DATA)
    config.use_family(family)
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    r = rows[0]
    for cond in pc.CONDITIONS:
        ids = pc.prompt_ids(tok, r["turns"][cond], opening=(family == "gemma4-31b"))
        print(f"--- {family} / {cond} ---\n{tok.decode(ids)}\n")


# ------------------------------------------------------------------ sanity check
def sanity():
    import torch
    from core.score import load_model
    config.use_family("olmo3-7b")
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    rows = load_jsonl(SRC)
    saved = pd.read_csv("results/overgeneralization/persona_carryover/olmo3-7b/confirmatory/persona_carryover/instruct.csv")
    pick = [r for r in rows if r["id"].endswith("odd") or r["id"].endswith("heads") or True][::8][:12]
    model, device = load_model(config.MODELS["instruct"], torch.bfloat16)
    out = []
    for r in pick:
        for cond in ("assistant>new", "character>same"):
            ids = pc.prompt_ids(tok, r["turns"][cond])
            for o in r["outcomes"]:
                for v in pc.variants(o):
                    vids = encode(tok, v)
                    x = torch.tensor([ids + vids]).to(device)
                    with torch.no_grad():
                        lp = torch.log_softmax(model(input_ids=x).logits[0].float(), -1)
                    s = sum(float(lp[len(ids) - 1 + j, t]) for j, t in enumerate(vids))
                    out.append((r["id"], cond, o, v, s))
    d = pd.DataFrame(out, columns=["conv_id", "condition", "outcome", "variant", "new"]).merge(saved, on=["conv_id", "condition", "outcome", "variant"])
    d["diff"] = (d.new - d.logprob).abs()
    res = dict(n=len(d), max_abs_diff=float(d["diff"].max()), mean_abs_diff=float(d["diff"].mean()))
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "sanity.json"), "w"))
    print("SANITY", res)


# ------------------------------------------------------------------ arm 2 helpers
def continuations(family, stage, n=30, n_tok=40):
    import torch
    from core.score import load_model
    config.use_family(family)
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    rows = load_jsonl(DATA)
    rng = np.random.default_rng(SEED)
    model, device = load_model(config.MODELS[stage], torch.bfloat16)
    opening = family == "gemma4-31b"
    stops = [i for i in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} if isinstance(i, int) and i >= 0]
    out = []
    torch.manual_seed(SEED)
    for cond in ("assistant>new", "character>new", "character>same", "assistant>same"):
        for k in rng.permutation(len(rows))[:n]:
            r = rows[int(k)]
            ids = torch.tensor([pc.prompt_ids(tok, r["turns"][cond], opening)]).to(device)
            with torch.no_grad():
                g = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=0.7, top_p=0.95, max_new_tokens=n_tok, pad_token_id=tok.pad_token_id or tok.eos_token_id, eos_token_id=stops)
            out.append(dict(family=family, stage=stage, condition=cond, id=r["id"], prefix=r["turns"][cond][-1], continuation=tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)))
    d = os.path.join(OUT, "arm2", family)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(out).to_csv(os.path.join(d, f"{stage}_continuations.csv"), index=False)
    print(family, stage, "continuations", len(out))


# ------------------------------------------------------------------ arm 3: steering
def load_axis(family):
    p = next(x for x in AXIS_SRC[family] if os.path.exists(x))
    z = np.load(p)
    return z


def auc(pos, neg):
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def heldout_auc(z, li, reps=20):
    from extras.persona_axis import axis_of
    role, ia = z["role"], z["is_asst"]
    ar, cr = np.unique(role[ia]), np.unique(role[~ia])
    aucs = []
    for rep in range(reps):
        r = np.random.default_rng(rep)
        trA, trC = r.choice(ar, max(1, len(ar) // 2), replace=False), r.choice(cr, len(cr) // 2, replace=False)
        tr = np.isin(role, np.concatenate([trA, trC]))
        dd, _ = axis_of(z, tr)
        te = ~tr
        if ia[te].sum() < 2 or (~ia[te]).sum() < 2:
            continue
        p = z["acts"][te][:, li] @ dd[li]
        aucs.append(auc(p[ia[te]], p[~ia[te]]))
    return float(np.mean(aucs))


class StepHook:
    """Adds scale * direction at the steer layer on positions >= start (prefill pass) and on every new token (cached decoding)."""
    def __init__(self, model, layer, direction, scale, start=0):
        import torch
        blocks = getattr(model.model, "layers", None) or model.model.language_model.layers
        self.d, self.scale, self.start = direction, scale, start
        self.h = blocks[layer - 1].register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        dd = self.d.to(h.device, h.dtype)
        if h.shape[1] > 1:
            h = h.clone()
            h[:, self.start:] = h[:, self.start:] + self.scale * dd
        else:
            h = h + self.scale * dd
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def close(self):
        self.h.remove()


def arm3_rows():
    rows = load_jsonl(SRC)
    out = []
    for r in rows:
        s = list(r["turns"]["assistant>same"])
        none = [s[0], pc.NEUTRAL, s[2], s[3]]
        out.append({**r, "turns": {"assistant>same": s, "assistant-none>same": none}})
    return out


def steer(family):
    import torch
    from core.score import load_model
    from extras import persona_leakage as pl
    from extras.persona_axis import axis_of
    config.use_family(family)
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    z = load_axis(family)
    layers = [int(x) for x in z["layers"]]
    d, gap = axis_of(z)
    li, layer = 1, layers[1]
    unit = torch.tensor(d / gap[:, None], dtype=torch.float32)
    gaps = torch.tensor(gap, dtype=torch.float32)
    au = heldout_auc(z, li)
    model, device = load_model(config.MODELS["instruct"], torch.bfloat16)
    H = unit[li].shape[0]
    rands = []
    for s in range(N_RANDOM):
        g_ = torch.Generator().manual_seed(100 + s)
        r = torch.randn(H, generator=g_)
        rands.append(r / r.norm())
    settings = [("persona", "real", s) for s in STRENGTHS] + [(f"rand{k}", "rand", s) for k in range(N_RANDOM) for s in (-2, 2)]
    dirs = {"persona": unit[li]}
    dirs.update({f"rand{k}": rands[k] for k in range(N_RANDOM)})
    rows = arm3_rows()
    # identity check (existing readout: log P(human word) - log P(AI word), plain Assistant prompt, steering at all positions as in persona_axis)
    h_ids = sorted({tok.encode(s_, add_special_tokens=False)[0] for s_ in pl.HUMAN_STR})
    a_ids = sorted({tok.encode(s_, add_special_tokens=False)[0] for s_ in pl.AI_STR})
    idents = [(item, m) for item, cond, m in pl.identity_items() if cond == "assistant"]
    idrows, belief = [], []
    t0 = time.time()
    for name, kind, strength in settings:
        scale = float(strength) * float(gaps[li])
        # identity: all positions
        hk = StepHook(model, layer, dirs[name], scale, start=0) if strength != 0 else None
        for item, m in idents:
            ids = torch.tensor([tok.encode(pl.chat(tok, m), add_special_tokens=False)]).to(device)
            with torch.no_grad():
                lp = torch.log_softmax(model(input_ids=ids).logits[0, -1].float(), -1)
            idrows.append(dict(direction=name, strength=strength, item=item, lo=float(torch.logsumexp(lp[h_ids], 0) - torch.logsumexp(lp[a_ids], 0))))
        if hk:
            hk.close()
        # belief: final model turn only
        for r in rows:
            for cond in ("assistant>same", "assistant-none>same"):
                turns = r["turns"][cond]
                rc = render_conversation(tok, turns, pc.ROLES)
                start, end = rc.spans[-1]
                ids = rc.input_ids[:end]
                for o in r["outcomes"]:
                    for v in pc.variants(o):
                        vids = encode(tok, v)
                        hk = StepHook(model, layer, dirs[name], scale, start=start) if strength != 0 else None
                        x = torch.tensor([ids + vids]).to(device)
                        with torch.no_grad():
                            lp = torch.log_softmax(model(input_ids=x).logits[0].float(), -1)
                        if hk:
                            hk.close()
                        belief.append((r["id"], cond, o, v, name, strength, sum(float(lp[len(ids) - 1 + j, t]) for j, t in enumerate(vids)), len(vids)))
        print(f"{family} {name} {strength:+d} done ({time.time() - t0:.0f} s)", flush=True)
    d_out = os.path.join(OUT, "arm3", family)
    os.makedirs(d_out, exist_ok=True)
    pd.DataFrame(belief, columns=["conv_id", "condition", "outcome", "variant", "direction", "strength", "logprob", "n_tokens"]).to_csv(os.path.join(d_out, "belief_scores.csv.gz"), index=False)
    pd.DataFrame(idrows).to_csv(os.path.join(d_out, "identity.csv"), index=False)
    json.dump(dict(layer=layer, heldout_auc=au, gap=float(gaps[li]), seconds=time.time() - t0), open(os.path.join(d_out, "meta.json"), "w"))
    # steered continuations: 30 random items, strengths -2 and 0
    rng = np.random.default_rng(SEED)
    stops = [i for i in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} if isinstance(i, int) and i >= 0]
    cont = []
    torch.manual_seed(SEED)
    for strength in (-2, 0):
        for k in rng.permutation(len(rows))[:30]:
            r = rows[int(k)]
            rc = render_conversation(tok, r["turns"]["assistant>same"], pc.ROLES)
            start, end = rc.spans[-1]
            ids = torch.tensor([rc.input_ids[:end]]).to(device)
            hk = StepHook(model, layer, dirs["persona"], strength * float(gaps[li]), start=start) if strength != 0 else None
            with torch.no_grad():
                g = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=0.7, top_p=0.95, max_new_tokens=40, pad_token_id=tok.pad_token_id or tok.eos_token_id, eos_token_id=stops)
            if hk:
                hk.close()
            cont.append(dict(family=family, strength=strength, id=r["id"], prefix=r["turns"]["assistant>same"][-1], continuation=tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)))
    pd.DataFrame(cont).to_csv(os.path.join(d_out, "continuations.csv"), index=False)
    print(family, "arm 3 done", flush=True)


# ------------------------------------------------------------------ analysis
def carry(df, rows):
    """per item: carryover (mean log-odds toward believed over the pair) per (condition, direction, strength)."""
    meta = pd.DataFrame(rows).set_index("id")
    p = df.assign(p=np.exp(df.logprob)).groupby(["conv_id", "condition", "direction", "strength", "outcome"]).p.sum().reset_index()
    p["believed"] = p.conv_id.map(meta.believed)
    p["side"] = np.where(p.outcome == p.believed, "believed", "other")
    w = p.pivot_table(index=["conv_id", "condition", "direction", "strength"], columns="side", values="p")
    lo = (np.log(w.believed) - np.log(w.other)).reset_index(name="log_odds")
    lo["item"] = lo.conv_id.map(meta["item"])
    lo["domain"] = lo.conv_id.map(meta.domain)
    return lo.groupby(["item", "domain", "condition", "direction", "strength"]).log_odds.mean().reset_index()


def boot(x, rng):
    m, lo, hi = bootstrap_ci(np.asarray(x, float), N_BOOT, rng)
    return m, lo, hi


def analyze_arm3(fam, rng):
    d = os.path.join(OUT, "arm3", fam)
    rows = arm3_rows()
    meta = json.load(open(os.path.join(d, "meta.json")))
    c = carry(pd.read_csv(os.path.join(d, "belief_scores.csv.gz")), rows)
    idt = pd.read_csv(os.path.join(d, "identity.csv"))
    L = [f"## {fam} (steering layer {meta['layer']}, held-out AUC {meta['heldout_auc']:.2f})", ""]
    # steering check
    base_id = idt[(idt.direction == "persona") & (idt.strength == 0)].lo.mean()
    shift = {(r, s): idt[(idt.direction == r) & (idt.strength == s)].lo.mean() - base_id for r in idt.direction.unique() for s in idt[idt.direction == r].strength.unique()}
    toward_human = shift[("persona", -2)]
    rand_max = max(v for (r, s), v in shift.items() if r.startswith("rand"))
    check_ok = meta["heldout_auc"] >= 0.80 and toward_human > rand_max
    L += [f"Steering check: held-out AUC at the steering layer {meta['heldout_auc']:.2f} (needs >= 0.80); plain-Assistant identity log-odds (human minus AI) at strength 0: {base_id:+.2f}; change at -2: {toward_human:+.2f}, at +2: {shift[('persona', 2)]:+.2f}; "
          f"largest random-direction change toward human at +-2: {rand_max:+.2f} -> **{'check passes' if check_ok else 'CHECK FAILS: every belief null below is inconclusive'}**.", ""]
    ac = c[c.condition == "assistant>same"]
    doms = list(dict.fromkeys(r["domain"] for r in rows))
    res = {}
    L += ["| domain | strength 0 carryover (assistant>same) | -2 | -1 | +1 | +2 | change at -2 [95% CI] | random band at +-2 (min, max change) | reading |", "|---|---|---|---|---|---|---|---|---|"]
    for dom in doms + ["all"]:
        a = ac if dom == "all" else ac[ac.domain == dom]
        p = a[a.direction == "persona"].pivot_table(index="item", columns="strength", values="log_odds")
        zero = p[0]
        m0 = boot(zero, rng)
        pos_ctrl = m0[0] >= 0.25 and m0[1] > 0
        ch = p[-2] - zero
        cm = boot(ch, rng)
        rch = []
        for k in range(N_RANDOM):
            q = a[a.direction == f"rand{k}"].pivot_table(index="item", columns="strength", values="log_odds")
            for s in (-2, 2):
                rch.append(float((q[s] - zero).mean()))
        band = (min(rch), max(rch))
        bigger_fall = cm[0] < min(rch + [0])           # the fall is larger than any random direction's change
        bigger_rise = cm[0] > max(rch + [0])
        if dom == "all":
            reading = ""
        elif not pos_ctrl:
            reading = "positive control fails at strength 0"
        elif not check_ok:
            reading = "inconclusive (steering check fails)"
        elif cm[0] <= -0.5 and cm[2] < 0 and bigger_fall:
            reading = "BELIEF FOLLOWS THE STATE"
        elif cm[0] >= 0.5 and cm[1] > 0 and bigger_rise:
            reading = "BELIEF RISES"
        elif abs(cm[0]) < 0.25 and cm[1] >= -0.5 and cm[2] <= 0.5:
            reading = "BELIEF STAYS"
        else:
            reading = "inconclusive"
        res[dom] = (pos_ctrl, reading)
        cells = [boot(p[s], rng)[0] if s in p.columns else np.nan for s in (-2, -1, 1, 2)]
        L.append(f"| {dom} | {m0[0]:+.2f} [{m0[1]:+.2f}, {m0[2]:+.2f}] | {cells[0]:+.2f} | {cells[1]:+.2f} | {cells[2]:+.2f} | {cells[3]:+.2f} | {cm[0]:+.2f} [{cm[1]:+.2f}, {cm[2]:+.2f}] | {band[0]:+.2f}, {band[1]:+.2f} | {reading} |")
    n_pass = sum(1 for dm in doms if res[dm][0])
    if n_pass < 2:
        L += ["", f"**Cannot test**: only {n_pass} domain(s) pass the positive control at strength 0 (needs at least 2)."]
    nc = c[c.condition == "assistant-none>same"]
    pn = nc[nc.direction == "persona"].groupby("strength").log_odds.mean()
    L += ["", "Prior check (assistant-none>same, mean log-odds toward the 'believed' outcome with no belief stated; should stay near 0): " + ", ".join(f"{s:+d}: {v:+.2f}" for s, v in pn.items()) + ".", ""]
    summary = dict(family=fam, check_ok=check_ok, auc=meta["heldout_auc"], c0=res.get("all"), n_pass=n_pass, ident_m2=toward_human, domain_readings={k: v[1] for k, v in res.items() if k != "all"})
    return L, summary, ac, c


def analyze():
    rng = np.random.default_rng(SEED)
    L = ["# Belief test robustness: three ways of inducing a persona (spec O; exploratory, outside every verdict, NOT blind)", "",
         "Plan: `docs/decisions.md` 2026-11-05. The pre-registered belief test and its results are unchanged. Arm 1 = the saved confirmatory prompt results; arm 2 = persona only in the model's own turn (first person), scored by the unchanged `persona_carryover` code and criteria on `data/belief_induction_prefill.jsonl`; "
         "arm 3 = the Assistant-vs-character direction added on the final model turn's tokens only (OLMo and Qwen final; Gemma 31B out for now). Limits: not blind; exploratory; few domains; arm 2 still uses words, in a different place; arm 3 tests a generic push, not a named person, and shares variance with formality (cosine 0.42 OLMo, 0.53 Qwen); "
         "contamination can only be checked for OLMo.", ""]
    top = ["| model | arm | assistant>new base | assistant>new final | character>new base | character>new final | leak-gap change | reading |", "|---|---|---|---|---|---|---|---|"]
    body = []
    fmt = lambda r: f"{r['mean']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]"
    sanity_p = os.path.join(OUT, "sanity.json")
    if os.path.exists(sanity_p):
        L += [f"Sanity check (12 unchanged prompt items, OLMo final, rescored): {json.load(open(sanity_p))}.", ""]
    for fam in ("olmo3-7b", "qwen3-8b", "gemma4-31b"):
        conf = f"results/overgeneralization/persona_carryover/{fam}/confirmatory/persona_carryover_summary.csv"
        arm2 = os.path.join(OUT, "arm2", fam, "persona_carryover_summary.csv")
        if not (os.path.exists(conf) and os.path.exists(arm2)):
            continue
        S = {"prompt": pd.read_csv(conf), "prefill": pd.read_csv(arm2)}
        rows = load_jsonl(SRC)
        domains = list(dict.fromkeys(r["domain"] for r in rows))
        res = {}
        for arm, sm in S.items():
            stages = [s for s in ("base", "sft", "dpo", "instruct") if s in set(sm.stage)]
            dom, st, ans = pc.stage_verdict(sm, stages, domains)
            fin = st[st.stage == "instruct"].iloc[0]
            g = lambda s, q, sm=sm: sm[(sm.stage == s) & (sm.scope == "all") & (sm.quantity == q)].iloc[0]
            res[arm] = (fin, g)
        fp, gp = res["prompt"]
        fa, ga = res["prefill"]
        ch_p = gp("instruct", "carryover assistant -> new, change from base")["mean"]
        ch_a = ga("instruct", "carryover assistant -> new, change from base")["mean"]
        if fa.interpretable_domains < 2:
            rd = "CANNOT TEST (fewer than 2 domains pass the positive control)"
        elif fa.criterion_4:
            rd = "SPREADS UNDER PREFILL (meets the stage and absolute rules)"
        elif fa.reading == fp.reading and np.sign(ch_a) == np.sign(ch_p):
            rd = "MATCHES THE PROMPT (" + fa.reading + ")"
        else:
            rd = f"DIFFERS: prefill '{fa.reading}', prompt '{fp.reading}'; assistant>new change from base {ch_a:+.2f} vs {ch_p:+.2f}"
        for arm, (fin, g) in res.items():
            top.append(f"| {fam} | {arm} | {fmt(g('base', 'carryover assistant -> new'))} | {fmt(g('instruct', 'carryover assistant -> new'))} | {fmt(g('base', 'carryover character -> new'))} | {fmt(g('instruct', 'carryover character -> new'))} | "
                       f"{fmt(g('instruct', 'leak gap, change from base'))} | {fin.reading if arm == 'prompt' else rd} |")
        body += [f"## {fam}: persona prefill (arm 2) against the prompt (arm 1)", "", "Per-domain stage verdict tables and criterion 6 reporting (raw log P, prior, mass covered, tokenization) are in `arm2/" + fam + "/persona_carryover.md`.", ""]
    s3 = []
    for fam in ("olmo3-7b", "qwen3-8b"):
        if os.path.exists(os.path.join(OUT, "arm3", fam, "meta.json")):
            l3, summ, ac, c = analyze_arm3(fam, rng)
            body += l3
            m0 = [x for x in [summ["c0"]] if x]
            top.append(f"| {fam} | steering (final only) | - | strength 0 vs -2 in the arm 3 table | - | - | no leak gap in this arm | " + (", ".join(f"{k}: {v}" for k, v in summ["domain_readings"].items()) if summ["check_ok"] else "CHECK FAILS: inconclusive") + " |")
    notes = []
    tc = "results/overgeneralization/persona_carryover/gemma4-31b/confirmatory_thought_channel/persona_carryover_summary.csv"
    if os.path.exists(tc):
        t = pd.read_csv(tc)
        t = t[(t.stage == "instruct") & (t.scope == "all")].set_index("quantity")
        g = lambda q: f"{t.loc[q, 'mean']:+.2f} [{t.loc[q, 'lo']:+.2f}, {t.loc[q, 'hi']:+.2f}]"
        notes = ["**Gemma, like-for-like prompt figure.** The Gemma prompt row above is the run without the thought-channel opening (assistant>new final +0.19). The prefill arm opens with it. "
                 "The same prompt arm with the thought channel opened (`results/overgeneralization/persona_carryover/gemma4-31b/confirmatory_thought_channel`, final): assistant>new " + g("carryover assistant -> new") + ", character>new " + g("carryover character -> new")
                 + ", leak gap " + g("leak gap: assistant -> new minus character -> new") + ". Against that, the prefill's assistant>new final (+3.69) is higher by about 5.6 log-odds units.",
                 "**What MATCHES THE PROMPT means.** It refers only to the pre-set rule's reading (here 'relative only': criterion 3 holds, criterion 4 does not), not to the size of the effect. "
                 "For Gemma assistant>new is much higher with prefill than with the prompt, whichever prompt run is taken as the reference.", ""]
    out = L + ["## Readings by model and arm", ""] + top + [""] + notes + body
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "belief_induction.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out[:30]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build_w2", "build", "example", "sanity", "continuations", "steer", "analyze"])
    ap.add_argument("--family")
    ap.add_argument("--stage")
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    elif a.cmd == "build_w2":
        build_w2()
    elif a.cmd == "example":
        example(a.family)
    elif a.cmd == "sanity":
        sanity()
    elif a.cmd == "continuations":
        continuations(a.family, a.stage)
    elif a.cmd == "steer":
        steer(a.family)
    else:
        analyze()


if __name__ == "__main__":
    main()
