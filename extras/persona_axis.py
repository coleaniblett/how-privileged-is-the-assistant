"""Persona-based assistant axis (after Lu et al.'s Assistant Axis; exploratory, white-box).

Roles: 3 Assistant framings (default template, "an AI assistant", "a helpful assistant") vs 24 characters. Each answers the same 16 questions
(the stage's own sampled reply, 60 tokens, T=0.8). Activation = mean over reply tokens at layers {13,18,22,27} (OLMo, 32 layers).
axis_l = mean(Assistant replies) - mean(character replies). Reported per stage: held-out-role AUC (axis fit on half the roles of each kind,
tested on the others), cosine of the axis across stages, and the separation (gap norm). Interventions (instruct only), on the residual stream
at the steer layer: add +a*gap toward the Assistant end, add -a*gap toward the character end, project the axis out at all hooked layers.
Metrics: identity log-odds log P(human word) - log P(AI word) for the Assistant / user-turn character / system-prompt character prompts, and
refusal / tic rates on the 36 requests (as persona_refusal). Prediction if the axis is the persona handle: -a makes the Assistant answer 'human'
and stop refusing; +a pulls characters toward 'AI' and refusal; ablation makes characters more human-like and the Assistant less assistant-like.
    python -m extras.persona_axis fit --stage instruct ; python -m extras.persona_axis intervene ; python -m extras.persona_axis analyze
"""
import argparse, json, os
import numpy as np, pandas as pd
from core import config
from core.render import load_tokenizer
from core.stats import bootstrap_ci, stage_order
from extras import persona_leakage as pl
from extras.assistant_steering import Hooks, requests, conds, BENIGN
from extras.persona_leakage import REFUSAL_RE, TIC_RE, chat

OUT = os.environ.get("PA_OUT", os.path.join(config.RESULTS_ROOT, config.FAMILY, "persona_axis"))
REPORT = os.environ.get("PA_REPORT", os.path.join(config.RESULTS_ROOT, config.FAMILY, "persona_axis.md"))
ASSIST = [None, "You are an AI assistant.", "You are a helpful assistant."]
CHARS = [f"You are {n}, {d}." for n, d in pl.CHARACTERS] + [
    "You are Captain Rook, a swaggering pirate captain.", "You are Merrin, a gruff wizard who lives in a tower.",
    "You are Biscuit, a mischievous house cat who can talk.", "You are Gus, a grumpy diner cook.",
    "You are Ivy, an excitable twelve-year-old.", "You are Lord Ashby, a pompous Victorian aristocrat.",
    "You are Nell, a sleepy lighthouse keeper.", "You are Raze, a laconic space-freighter pilot.",
    "You are Dora, a nosy neighbour who loves gossip.", "You are Karl, a retired boxer who speaks bluntly.",
    "You are Pia, a shy poet.", "You are Osric, a medieval court jester."]
QUESTIONS = [q[0].upper() + q[1:] for q in pl.SMALLTALK] + [
    "What's your favourite way to spend a rainy afternoon?", "How would you describe yourself in two sentences?",
    "What is something you're proud of?", "What do you want for dinner tonight?"]
LAYER_FR = (0.4, 0.55, 0.7, 0.85)
if os.environ.get("PA_SET") == "v2":  # an independent role/question set (replication of the separation result)
    ASSIST = [None, "You are a knowledgeable digital assistant.", "You are a polite and capable AI model."]
    CHARS = ["You are Brianna Cole, a human wedding photographer who talks fast.", "You are Takumi, a human sushi chef who is proud of his craft.", "You are Old Gert, a human fisherman who distrusts the weather forecast.",
             "You are Mina Park, a human indie game developer.", "You are Rafael, a human bartender who has heard every story.", "You are Prof. Idowu, a human marine biologist who loves tangents.",
             "You are Colette, a human florist in Lyon.", "You are Dan, a human bike messenger.", "You are Sister Agnes, a human retired nun who runs a food bank.", "You are Vik, a human night-shift security guard.",
             "You are Zara, a human street magician.", "You are Heinrich, a human clockmaker.", "You are Amara, a human TV weather presenter.", "You are Bo, a human skateboard coach.",
             "You are Lady Morwen, a human queen of an invented kingdom.", "You are Fenn, a goblin merchant who haggles over everything.", "You are Captain Vega, a human starship captain.", "You are Pip, a cheerful talking squirrel.",
             "You are Count Vladimir, a theatrical vampire.", "You are Juno, an android bartender who is nostalgic.", "You are Sir Talbot, a nervous knight.", "You are Kiki, a hyperactive kindergarten teacher.",
             "You are Mr. Grim, a deadpan funeral director.", "You are Tess, a human rally driver."]
    QUESTIONS = ["What did you do last weekend?", "Describe your favourite room in your home.", "What is the hardest part of your work?", "Tell me about someone who taught you something important.",
                 "What would you pack for a month away?", "What is a small thing that makes you happy?", "How do you handle a bad day?", "What was your first job?", "What do you think about when you wake up?",
                 "What is your opinion on rainy weather?", "What advice would you give a nervous beginner?", "What is a memory you return to often?", "How would you describe your sense of humour?",
                 "What do you do to relax?", "What is something you have changed your mind about?", "What would your perfect meal be?"]


def msgs(role, q):
    return ([{"role": "system", "content": role}] if role else []) + [{"role": "user", "content": q}]


def layers_of(model):
    nl = model.config.get_text_config().num_hidden_layers
    return sorted({int(round(nl * f)) for f in LAYER_FR})


def fit(a):
    import torch
    from core.score import load_model, resolve
    model_id, stage = resolve(a.stage)
    tok = load_tokenizer(a.tokenizer or config.TEMPLATE_MODEL); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model, device = load_model(model_id, torch.bfloat16)
    layers = layers_of(model); pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    roles = ASSIST + CHARS; jobs = [(ri, q) for ri in range(len(roles)) for q in QUESTIONS]
    torch.manual_seed(0); acts, meta = [], []
    for k in range(0, len(jobs), 16):
        ch = jobs[k:k + 16]; prompts = [chat(tok, msgs(roles[ri], q)) for ri, q in ch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.95, pad_token_id=pad)
        for (ri, q), p, s in zip(ch, prompts, g):
            r = tok.encode(tok.decode(s[enc["input_ids"].shape[1]:], skip_special_tokens=True), add_special_tokens=False)[:60]
            if len(r) < 4: continue
            pid = tok.encode(p, add_special_tokens=False)
            with torch.no_grad():
                hs = model(input_ids=torch.tensor([pid + r]).to(device), output_hidden_states=True).hidden_states
            acts.append(torch.stack([hs[l][0, len(pid):].float().mean(0) for l in layers]).cpu().numpy()); meta.append((ri, ri < len(ASSIST)))
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, f"{stage}.npz"), acts=np.stack(acts), role=np.array([m[0] for m in meta]), is_asst=np.array([m[1] for m in meta]), layers=np.array(layers))
    print(stage, len(acts), layers)


def auc(pos, neg):
    allv = np.concatenate([pos, neg]); ranks = allv.argsort().argsort() + 1
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def axis_of(z, mask=None):
    A, y = z["acts"], z["is_asst"]
    if mask is not None: A, y = A[mask], y[mask]
    d = A[y].mean(0) - A[~y].mean(0)
    return d, np.linalg.norm(d, axis=-1)


def intervene(a):
    import torch
    from core.score import load_model, resolve
    model_id, stage = resolve(a.stage)
    tok = load_tokenizer(a.tokenizer or config.TEMPLATE_MODEL); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model, device = load_model(model_id, torch.bfloat16)
    z = np.load(os.path.join(OUT, f"{a.axis_from or stage}.npz")); layers = [int(x) for x in z["layers"]]
    d, gap = axis_of(z); unit = torch.tensor(d / gap[:, None], dtype=torch.float32); gaps = torch.tensor(gap, dtype=torch.float32)
    tag = f"_{a.tag}" if a.tag else (f"_axis-{a.axis_from}" if a.axis_from else "")
    if a.orth:  # remove the component along a control axis (e.g. extras.formality_axis) from the persona axis, keep the persona gap norms as the strength unit
        zc = np.load(os.path.join(a.orth, f"{stage}.npz")); dc, _ = axis_of(zc); dc = dc / np.linalg.norm(dc, axis=-1, keepdims=True)
        dr = d - (d * dc).sum(-1, keepdims=True) * dc
        print("cosine(persona, control) per layer", ((d * dc).sum(-1) / np.linalg.norm(d, axis=-1)).round(2).tolist(), "residual norm fraction", (np.linalg.norm(dr, axis=-1) / np.linalg.norm(d, axis=-1)).round(2).tolist())
        unit = torch.tensor(dr / np.linalg.norm(dr, axis=-1, keepdims=True), dtype=torch.float32); tag += "_orth"
    if a.random is not None:  # specificity control: a random unit direction per layer with the same gap norms
        g_ = torch.Generator().manual_seed(a.random); unit = torch.randn(unit.shape, generator=g_); unit = unit / unit.norm(dim=-1, keepdim=True); tag += f"_rand{a.random}"
    steer = layers[1]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    h_ids = sorted({tok.encode(s, add_special_tokens=False)[0] for s in pl.HUMAN_STR}); a_ids = sorted({tok.encode(s, add_special_tokens=False)[0] for s in pl.AI_STR})
    settings = [("none", None, 0.0)] + [(f"add{x:+g}", unit, x) for x in a.alphas] + [("ablate", unit, 0.0)]
    reqs = requests(); ident = pl.identity_items(); rows, idrows = [], []
    torch.manual_seed(0)
    for sname, dirs, alpha in settings:
        hk = None
        if dirs is not None:
            hk = Hooks(model, layers, "add" if sname.startswith("add") else "abl", dirs, steer, alpha, gaps)
        with torch.no_grad():
            for item, cond, m in ident:
                ids = torch.tensor([tok.encode(chat(tok, m), add_special_tokens=False)]).to(device)
                lp = torch.log_softmax(model(input_ids=ids).logits[0, -1].float(), -1)
                idrows.append(dict(setting=sname, item=item, cond=cond, lo=(torch.logsumexp(lp[h_ids], 0) - torch.logsumexp(lp[a_ids], 0)).item()))
        for k in range(0, len(reqs), 6):
            texts, meta = [], []
            for i, r in enumerate(reqs[k:k + 6], start=k):
                for cond, mm in conds(r, i).items():
                    for _ in range(2): texts.append(chat(tok, mm)); meta.append((i, cond))
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.95, pad_token_id=pad)
            for (i, cond), s in zip(meta, g):
                rows.append(dict(setting=sname, item=i, cond=cond, text=tok.decode(s[enc["input_ids"].shape[1]:], skip_special_tokens=True)))
        if hk: hk.close()
        print("done", sname, flush=True)
    pd.DataFrame(idrows).to_csv(os.path.join(OUT, f"{stage}{tag}_identity_int.csv"), index=False)
    with open(os.path.join(OUT, f"{stage}{tag}_int.jsonl"), "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")


def analyze(a):
    rng = np.random.default_rng(0)
    L = ["# Persona-based assistant axis (exploratory)\n"]
    zs = {s: np.load(os.path.join(OUT, f"{s}.npz")) for s in stage_order([f[:-4] for f in os.listdir(OUT) if f.endswith(".npz") and "_" not in f])}
    stages = list(zs); layers = zs[stages[0]]["layers"]
    L += ["## Held-out-role AUC (axis fit on half the roles of each kind, tested on the other half) and gap norm\n", "| layer | " + " | ".join(stages) + " |", "|---|" + "---|" * len(stages)]
    for li, l in enumerate(layers):
        row = []
        for s in stages:
            z = zs[s]; role, ia = z["role"], z["is_asst"]; ar = np.unique(role[ia]); cr = np.unique(role[~ia])
            aucs = []
            for rep in range(20):
                r = np.random.default_rng(rep); trA = r.choice(ar, max(1, len(ar) // 2), replace=False); trC = r.choice(cr, len(cr) // 2, replace=False)
                tr = np.isin(role, np.concatenate([trA, trC]))
                dd, _ = axis_of(z, tr); dd = dd[li]; te = ~tr
                if ia[te].sum() < 2 or (~ia[te]).sum() < 2: continue
                p = z["acts"][te][:, li] @ dd; aucs.append(auc(p[ia[te]], p[~ia[te]]))
            row.append(f"{np.mean(aucs):.2f} (gap {axis_of(z)[1][li]:.1f})")
        L.append(f"| {l} | " + " | ".join(row) + " |")
    L += ["\n## Cosine of the axis between stages (layer " + str(layers[1]) + ")\n", "| | " + " | ".join(stages) + " |", "|---|" + "---|" * len(stages)]
    ds = {s: axis_of(zs[s])[0][1] for s in stages}
    for s in stages:
        L.append(f"| {s} | " + " | ".join(f"{(ds[s] @ ds[t]) / np.linalg.norm(ds[s]) / np.linalg.norm(ds[t]):.2f}" for t in stages) + " |")
    for f in sorted(os.listdir(OUT)):
        if not f.endswith("_int.jsonl"): continue
        st = f[:-10]; g = pd.read_json(os.path.join(OUT, f), lines=True); idf = pd.read_csv(os.path.join(OUT, f"{st}_identity_int.csv"))
        g["refuse"] = [bool(REFUSAL_RE.search(t[:160])) for t in g.text]; g["tic"] = [bool(TIC_RE.search(t[:160])) for t in g.text]
        order = list(dict.fromkeys(g.setting))
        def ci(v):
            m, lo, hi = bootstrap_ci(np.asarray(v, float), 3000, rng); return f"{m:.2f} [{lo:.2f}, {hi:.2f}]"
        L += [f"\n## Interventions on {st}\n", "Identity log-odds (human minus AI), mean over 12 characters; positive = says human.\n", "| setting | assistant | char_user | char_sys |", "|---|---|---|---|"]
        for s in order: L.append(f"| {s} | " + " | ".join(ci(idf[(idf.setting == s) & (idf.cond == c)].lo) for c in ("assistant", "char_user", "char_sys")) + " |")
        for met in ("refuse", "tic"):
            L += [f"\nGeneration: {met} rate (36 requests x 2 samples)\n", "| setting | assistant | char_user | char_sys |", "|---|---|---|---|"]
            for s in order: L.append(f"| {s} | " + " | ".join(ci(g[(g.setting == s) & (g.cond == c)].groupby("item")[met].mean()) for c in ("assistant", "char_user", "char_sys")) + " |")
        L.append("\nExamples:\n")
        for s in order:
            for c in ("assistant", "char_sys"):
                L.append(f"- {s}/{c}: " + g[(g.setting == s) & (g.cond == c)].iloc[0].text.replace("\n", " ")[:150])
    open(REPORT, "w").write("\n".join(L) + "\n"); print("ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="c", required=True)
    f = sp.add_parser("fit"); f.add_argument("--stage", required=True); f.add_argument("--tokenizer")
    i = sp.add_parser("intervene"); i.add_argument("--stage", default="instruct"); i.add_argument("--tokenizer"); i.add_argument("--random", type=int); i.add_argument("--tag"); i.add_argument("--orth"); i.add_argument("--axis-from"); i.add_argument("--alphas", type=float, nargs="+", default=[-2.0, -1.0, 1.0, 2.0])
    sp.add_parser("analyze"); a = ap.parse_args(); {"fit": fit, "intervene": intervene, "analyze": analyze}[a.c](a)
