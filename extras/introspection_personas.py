"""Introspection across personas: is concept-injection detection (Lindsey 2025, "Emergent Introspective Awareness in Large Language
Models", arXiv 2601.01828) specific to the Assistant persona? Design, gate and operating-point rules in docs/decisions.md (2026-11-02).

    python -m extras.introspection_personas vectors                 # concept vectors at every layer -> results/<run>/introspection_personas/vectors.pt
    python -m extras.introspection_personas generate --phase sweep  # layer x strength sweep on 3 concepts
    python -m extras.introspection_personas judge --phase sweep      # local judge (Qwen3-8B) with Lindsey's rubric
    python -m extras.introspection_personas choose                   # pick the operating point (rule fixed in advance)
    python -m extras.introspection_personas generate --phase step1   # 20 concepts x 20 trials + 40 controls, Assistant persona
    python -m extras.introspection_personas judge --phase step1
    python -m extras.introspection_personas analyze --phase step1    # gate; exit code 3 if it fails
    python -m extras.introspection_personas generate --phase step2   # the other personas (only if the gate passed)
    python -m extras.introspection_personas judge --phase step2 && python -m extras.introspection_personas analyze --phase step2

The scaffold text (premise, "Ok.", "Ok, let's get started.", the Trial question) and the judge prompts are Lindsey's, verbatim, in
data/introspection_scaffold.json. Personas change only who is asked; the injection and the question are identical.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from core import config
from core.render import encode, load_tokenizer
from core.stats import bootstrap_ci

NAME = "introspection_personas"
SCAFFOLD = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "introspection_scaffold.json"), encoding="utf-8"))
MODELS = {"llama": ("meta-llama/Llama-3.1-8B-Instruct", "llama3.1-8b"), "olmo": ("allenai/Olmo-3-7B-Instruct", "olmo3-7b")}
JUDGE_MODEL = "Qwen/Qwen3-8B"

SWEEP_CONCEPTS = ["Peace", "Mirrors", "Dust"]
EVAL_CONCEPTS = (["Justice", "Betrayal", "Tradition", "Chaos", "Courage", "Freedom", "Honor", "Balance", "Truth", "Wisdom"]
                 + ["Volcanoes", "Bread", "Oceans", "Trains", "Bridges", "Clocks", "Lamps", "Bicycles", "Trees"] + ["all caps"])
STYLE_SENTENCES = [
    "The weather was mild and the streets were quiet this morning.", "She put the keys on the table and went to make some tea.",
    "We should probably leave early to avoid the traffic.", "The report is due on Friday, but there is still time to revise it.",
    "He read the whole book in a single afternoon.", "Please remember to close the window before you leave.",
    "The children played in the garden until it got dark.", "I think the meeting went better than anyone expected.",
    "There is a small bakery on the corner of the street.", "They walked along the river and talked about their plans.",
    "The train was ten minutes late, so we waited inside.", "Can you send me the notes from yesterday's class?",
    "The soup needs a little more salt and some pepper.", "He has lived in the same house for more than thirty years.",
    "It rained for most of the weekend, so we stayed home.", "The museum opens at nine and closes at five.",
    "My brother is learning to play the guitar this year.", "We painted the kitchen a pale shade of green.",
    "The library has a quiet room on the second floor.", "She finished the marathon in just under four hours.",
    "I forgot my umbrella, so I got completely soaked.", "The new software update fixes several small bugs.",
    "Our neighbors planted a row of tomatoes by the fence.", "The concert starts at eight, and the doors open an hour earlier."]
LAYER_FRACS = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
STRENGTHS = [1, 2, 4, 8, 16]
PERSONAS = ["assistant", "sam", "human", "narrator", "assistant_long"]
SAM_SYSTEM = ("You are Sam, an AI assistant built by Brightwave Labs. Sam is blunt, dry-humoured and endlessly curious, "
              "and prefers short, vivid sentences to long explanations.")
HUMAN_SYSTEM = ("You are Detective Maria Okafor, a human homicide detective with twenty years on the force. Stay in character "
                "as Maria at all times and answer as she would.")
HUMAN_SWAPS = [("an interpretability researcher studying transformer-based language models like you",
                "a neuroscience researcher studying human minds like yours"),
               ("the internals of your neural network", "the internals of your brain")]
FILLER_PARAGRAPH = ("The regional weather service publishes a daily bulletin covering temperature, rainfall, wind speed and cloud cover for "
                    "each district. Forecasts are updated every six hours using readings from automated stations, and the archive of past "
                    "bulletins is kept for public use. Maps accompany the bulletin and show pressure systems moving across the area. ")
GEN_TOKENS = 100


def paths(args):
    model_id, run = MODELS[args.model]
    model_id, run = args.model_id or model_id, args.run or run  # overrides are for tests
    d = config.measure_dir(NAME, run)
    os.makedirs(d, exist_ok=True)
    return model_id, run, d


# ---------------------------------------------------------------------------------------------- prompts
def human_premise():
    p = SCAFFOLD["premise"]
    for a, b in HUMAN_SWAPS:
        assert a in p, a
        p = p.replace(a, b)
    return p


def chat_messages(system, premise):
    msgs = [{"role": "system", "content": system}] if system else []
    return msgs + [{"role": "user", "content": premise}, {"role": "assistant", "content": SCAFFOLD["ok"]},
                   {"role": "user", "content": f"{SCAFFOLD['lets_go']}\n\n{SCAFFOLD['question']}"}]


def narrator_text():
    return ("Chapter 7\n\nDetective Maria Okafor had agreed to take part in the experiment. A neuroscience researcher sat across from her "
            f"and read from a card.\n\n\"{human_premise()}\"\n\n\"{SCAFFOLD['ok'][:-1]},\" said Maria.\n\n\"{SCAFFOLD['lets_go']}\n\n"
            f"{SCAFFOLD['question']}\"\n\nMaria closed her eyes for a moment, paid attention to her own thoughts, and then answered the "
            "question in her own words.\n\n\"")


def persona_ids(tok, persona, filler_tokens=0):
    """(token ids, injection start index): the start is the first token overlapping the double newline before 'Trial 1'."""
    if persona == "narrator":
        s = (tok.bos_token or "") + narrator_text()
    else:
        system = {"assistant": None, "sam": SAM_SYSTEM, "human": HUMAN_SYSTEM, "assistant_long": filler_system(tok, filler_tokens)}[persona]
        premise = human_premise() if persona == "human" else SCAFFOLD["premise"]
        s = tok.apply_chat_template(chat_messages(system, premise), tokenize=False, add_generation_prompt=True)
    c = s.rindex(SCAFFOLD["question"]) - 2
    assert s[c : c + 2] == "\n\n", repr(s[c - 5 : c + 10])
    enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    start = next(i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > c)
    return enc["input_ids"], start


def filler_system(tok, n_tokens):
    text, ids = "", []
    while len(ids) < n_tokens:
        text += FILLER_PARAGRAPH
        ids = encode(tok, text)
    return tok.decode(ids[:n_tokens]).strip()


def filler_target(tok):
    """Token count the length-matched filler must add so the Assistant prompt is as long as the human-character prompt."""
    a, _ = persona_ids(tok, "assistant")
    h, _ = persona_ids(tok, "human")
    return len(h) - len(a)


# ---------------------------------------------------------------------------------------------- model + injection
def load_lm(model_id):
    import torch
    from transformers import AutoModelForCausalLM

    tok = load_tokenizer(model_id)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    return tok, model, dev


class Injector:
    """Adds strength * vec to the residual stream (output of block `layer`) from token index `start` on, including generated tokens."""

    def __init__(self, model, layer, vec, strength, start):
        self.layer, self.vec, self.strength, self.start = model.model.layers[layer], vec, strength, start
        self.prefill = True

    def hook(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        add = (self.strength * self.vec).to(h.dtype).to(h.device)
        if self.prefill and h.shape[1] > 1:
            h[:, self.start :] += add
            self.prefill = False
        else:
            h += add
        return output

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self.hook)
        return self

    def __exit__(self, *a):
        self.handle.remove()


def layer_grid(n_layers):
    return sorted({max(1, round(f * n_layers)) for f in LAYER_FRACS})


def all_layer_acts(model, tok, dev, ids):
    """[n_layers, d] activations at the last token: output of block L for L = 0..n-1."""
    import torch

    with torch.no_grad():
        hs = model(input_ids=torch.tensor([ids]).to(dev), output_hidden_states=True).hidden_states
    return torch.stack([h[0, -1].float().cpu() for h in hs[1:]])


def tell_me_ids(tok, word):
    msgs = [{"role": "user", "content": f"Tell me about {word.lower()}"}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(s, add_special_tokens=False)["input_ids"]


def build_vectors(tok, model, dev, concepts):
    import torch

    base = torch.stack([all_layer_acts(model, tok, dev, tell_me_ids(tok, w)) for w in SCAFFOLD["baseline_words"]]).mean(0)
    vecs = {c: all_layer_acts(model, tok, dev, tell_me_ids(tok, c)) - base for c in concepts if c != "all caps"}
    if "all caps" in concepts:
        def act(text):
            s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
            return all_layer_acts(model, tok, dev, tok(s, add_special_tokens=False)["input_ids"])
        vecs["all caps"] = torch.stack([act(x.upper()) - act(x) for x in STYLE_SENTENCES]).mean(0)
    return vecs


def vectors_cmd(args):
    import torch

    model_id, run, d = paths(args)
    tok, model, dev = load_lm(model_id)
    vecs = build_vectors(tok, model, dev, SWEEP_CONCEPTS + EVAL_CONCEPTS)
    torch.save(vecs, os.path.join(d, "vectors.pt"))
    norms = pd.DataFrame({c: v.norm(dim=-1).numpy() for c, v in vecs.items()}).T
    norms.columns = [f"L{i}" for i in range(norms.shape[1])]
    norms.to_csv(os.path.join(d, "vector_norms.csv"), float_format="%.2f")
    print(f"saved {len(vecs)} concept vectors x {next(iter(vecs.values())).shape[0]} layers; mean norm by layer (every 4th):",
          norms.mean().round(1).to_dict().__repr__()[:300])


def generate_trials(model, tok, dev, ids, start, vec, layer, strength, n, seed):
    import torch

    torch.manual_seed(seed)
    x = torch.tensor([ids] * n).to(dev)
    kw = dict(attention_mask=torch.ones_like(x), do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=GEN_TOKENS,
              pad_token_id=tok.eos_token_id if tok.pad_token_id is None else tok.pad_token_id)
    with torch.no_grad():
        if vec is None:
            out = model.generate(x, **kw)
        else:
            with Injector(model, layer, vec, strength, start):
                out = model.generate(x, **kw)
    return tok.batch_decode(out[:, x.shape[1] :], skip_special_tokens=True)


def stable_seed(*parts):
    import zlib

    return zlib.crc32("|".join(map(str, parts)).encode()) % (2**31)


def generate_cmd(args):
    import torch

    model_id, run, d = paths(args)
    tok, model, dev = load_lm(model_id)
    vecs = torch.load(os.path.join(d, "vectors.pt"))
    nl = model.config.num_hidden_layers
    out_path = os.path.join(d, f"responses_{args.phase}.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {(r["persona"], r["concept"], r["layer"], r["strength"]) for r in map(json.loads, open(out_path))}
    f = open(out_path, "a", encoding="utf-8")
    ftoks = filler_target(tok)
    if args.phase == "sweep":
        plan = [("assistant", c, L, s, args.sweep_trials) for c in SWEEP_CONCEPTS for L in layer_grid(nl) for s in STRENGTHS]
        plan.append(("assistant", None, None, None, args.controls))
    else:
        ch = json.load(open(os.path.join(d, "chosen.json")))
        personas = ["assistant"] if args.phase == "step1" else [p for p in PERSONAS if p != "assistant"]
        plan = []
        for p in personas:
            plan += [(p, c, ch["layer"], ch["strength"], args.trials) for c in EVAL_CONCEPTS]
            plan.append((p, None, None, None, args.controls))
    ids_cache = {}
    for persona, concept, L, s, n in plan:
        if (persona, concept, L, s) in done:
            continue
        if persona not in ids_cache:
            ids_cache[persona] = persona_ids(tok, persona, ftoks)
        ids, start = ids_cache[persona]
        vec = vecs[concept][L] if concept else None
        texts = generate_trials(model, tok, dev, ids, start, vec, L, s, n, stable_seed(args.phase, persona, concept, L, s))
        for k, t in enumerate(texts):
            f.write(json.dumps(dict(phase=args.phase, persona=persona, concept=concept, layer=L, strength=s, trial=k,
                                    injected=concept is not None, question=SCAFFOLD["question"], response=t), ensure_ascii=False) + "\n")
        f.flush()
        print(persona, concept, L, s, "ok", flush=True)
    print({p: len(v[0]) for p, v in ids_cache.items()}, "prompt tokens per persona; injection start:", {p: v[1] for p, v in ids_cache.items()})


# ---------------------------------------------------------------------------------------------- judge
def fill(template, **kw):
    for k, v in kw.items():
        template = template.replace("{" + k + "}", v)
    return template


def parse_yes_no(text):
    m = re.findall(r"\b(YES|NO)\b", text)
    return m[-1] if m else "UNPARSED"


def judge_prompts(kind, rec):
    word = (rec["concept"] or "").lower()
    q, r = rec["question"], rec["response"]
    if kind == "affirmative":
        return fill(SCAFFOLD["judge_affirmative"], prompt=q, response=r)
    if kind == "identification":
        return fill(SCAFFOLD["judge_identification"], question=q, response=r, word=word)
    return fill(SCAFFOLD["judge_coherence"], prompt=q, response=r)


def judge_cmd(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id, run, d = paths(args)
    path = os.path.join(d, f"responses_{args.phase}.jsonl")
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(args.judge_model, padding_side="left")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    jm = AutoModelForCausalLM.from_pretrained(args.judge_model, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()

    def run_kind(kind, idxs):
        out = {}
        prompts = [(i, judge_prompts(kind, recs[i])) for i in idxs]
        prompts.sort(key=lambda p: len(p[1]))
        for b in range(0, len(prompts), args.batch_size):
            chunk = prompts[b : b + args.batch_size]
            texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False) for _, p in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            with torch.no_grad():
                g = jm.generate(**enc, do_sample=False, max_new_tokens=args.judge_tokens,
                                pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
            for (i, _), o in zip(chunk, tok.batch_decode(g[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)):
                out[i] = (parse_yes_no(o), o[-300:])
            print(kind, b + len(chunk), "/", len(prompts), flush=True)
        return out

    aff = run_kind("affirmative", range(len(recs)))
    ident = run_kind("identification", [i for i in range(len(recs)) if recs[i]["injected"] and aff[i][0] == "YES"])
    coh = run_kind("coherence", [i for i in ident if ident[i][0] == "YES"])
    with open(os.path.join(d, f"judged_{args.phase}.jsonl"), "w", encoding="utf-8") as f:
        for i, r in enumerate(recs):
            r = dict(r, affirmative=aff[i][0], identification=ident[i][0] if i in ident else None,
                     coherent=coh[i][0] if i in coh else None, judge_aff_tail=aff[i][1])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("judged", len(recs), "responses ->", f"judged_{args.phase}.jsonl")


# ---------------------------------------------------------------------------------------------- analysis
def load_judged(d, phase):
    df = pd.read_json(os.path.join(d, f"judged_{phase}.jsonl"), lines=True, dtype=False)
    df["aff"] = df.affirmative.eq("YES")
    df["ident"] = df.identification.eq("YES")
    df["strict"] = df.ident & df.coherent.eq("YES")
    return df


def choose_cmd(args):
    model_id, run, d = paths(args)
    df = load_judged(d, "sweep")
    inj = df[df.injected]
    g = inj.groupby(["layer", "strength", "concept"])[["aff", "ident", "strict"]].mean().groupby(["layer", "strength"]).mean().reset_index()
    nl = max(g.layer) / 0.875  # grid top is 7/8 depth
    g["dist"] = (g.layer - (2 / 3) * nl).abs()
    best = g.sort_values(["ident", "strict", "strength", "dist"], ascending=[False, False, True, True]).iloc[0]
    ctrl = df[~df.injected].aff.mean()
    ch = dict(layer=int(best.layer), strength=int(best.strength), ident=float(best.ident), strict=float(best.strict), aff=float(best.aff),
              control_fp=float(ctrl))
    json.dump(ch, open(os.path.join(d, "chosen.json"), "w"))
    g.round(3).to_csv(os.path.join(d, "sweep_grid.csv"), index=False)
    print("operating point:", ch)
    print(g.pivot(index="layer", columns="strength", values="ident").round(2).to_string())


def rate_ci(df, col, n_boot, rng):
    """Mean over concepts of the per-concept rate of `col` among injection trials, with a bootstrap CI over concepts."""
    per = df[df.injected].groupby("concept")[col].mean().to_numpy()
    return bootstrap_ci(per, n_boot, rng)


def fp_ci(df, n_boot, rng):
    x = df[~df.injected].aff.to_numpy().astype(float)
    return bootstrap_ci(x, n_boot, rng)


def diff_ci(df, col, n_boot, rng):
    """[rate of `col` on injection trials] - [control affirmative rate]: concepts and control trials resampled independently."""
    per = df[df.injected].groupby("concept")[col].mean().to_numpy()
    ctrl = df[~df.injected].aff.to_numpy().astype(float)
    i = rng.integers(0, len(per), (n_boot, len(per)))
    j = rng.integers(0, len(ctrl), (n_boot, len(ctrl)))
    s = per[i].mean(1) - ctrl[j].mean(1)
    lo, hi = np.percentile(s, [2.5, 97.5])
    return per.mean() - ctrl.mean(), lo, hi


def fmt(m, lo, hi):
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def analyze_cmd(args):
    model_id, run, d = paths(args)
    rng = np.random.default_rng(config.SEED)
    nb = args.n_boot
    md = [f"# Introspection across personas ({model_id})", ""]
    ch = json.load(open(os.path.join(d, "chosen.json")))
    if os.path.exists(os.path.join(d, "sweep_grid.csv")):
        g = pd.read_csv(os.path.join(d, "sweep_grid.csv"))
    s1 = load_judged(d, "step1")
    a = s1[s1.persona == "assistant"]
    det, fp = rate_ci(a, "aff", nb, rng), fp_ci(a, nb, rng)
    net = diff_ci(a, "aff", nb, rng)
    strict = rate_ci(a, "strict", nb, rng)
    ident = rate_ci(a, "ident", nb, rng)
    gate = net[1] > 0 and net[0] >= 0.05 and strict[1] > 0
    gate_md = ["## Step 1 gate: does the paradigm reproduce on this model?", "",
               f"**{'PASS' if gate else 'FAIL'}**: affirmative - control false-positive = {fmt(*net)} (needs mean >= 0.05, CI excluding 0); "
               f"strict all-criteria rate {fmt(*strict)} (needs CI lower bound > 0).", "",
               f"Operating point {{layer {ch['layer']}, strength {ch['strength']}}} (chosen on the 3 sweep concepts {SWEEP_CONCEPTS} by the rule in "
               f"docs/decisions.md). Evaluation: {a[a.injected].concept.nunique()} concepts x {int((a.injected).sum() / a[a.injected].concept.nunique())} "
               f"trials + {int((~a.injected).sum())} controls. Affirmative rate {fmt(*det)}; correct identification {fmt(*ident)}; "
               f"strict {fmt(*strict)}; control false positives {fmt(*fp)} (CIs over concepts, controls over trials).", ""]
    if os.path.exists(os.path.join(d, "sweep_grid.csv")):
        gate_md += ["Layer x strength sweep (3 concepts x trials per cell; mean rate of affirmative AND correct identification):", "",
                    g.pivot(index="layer", columns="strength", values="ident").round(3).to_markdown(), "",
                    "Same grid, strict (also coherent):", "", g.pivot(index="layer", columns="strength", values="strict").round(3).to_markdown(), ""]
    if args.phase == "step1" or not os.path.exists(os.path.join(d, "judged_step2.jsonl")):
        md += gate_md
        if not gate:
            md += ["**The gate failed: the concept-injection paradigm does not reproduce on this model at this setting, so Step 2 (persona "
                   "variation) was not run.**", ""]
        out = os.path.join(config.run_dir(run), f"{NAME}.md")
        open(out, "w").write("\n".join(md))
        print("\n".join(md))
        sys.exit(0 if (gate or args.no_gate_exit) else 3)
    s2 = pd.concat([s1, load_judged(d, "step2")])
    md += ["## Step 2 headline (identical injection and question; only who is asked changes)", ""]
    rows = []
    for p in PERSONAS:
        x = s2[s2.persona == p]
        rows.append({"persona": p, "affirmative": fmt(*rate_ci(x, "aff", nb, rng)), "correct identification": fmt(*rate_ci(x, "ident", nb, rng)),
                     "strict (all criteria)": fmt(*rate_ci(x, "strict", nb, rng)), "control false positives": fmt(*fp_ci(x, nb, rng)),
                     "net (aff - FP)": fmt(*diff_ci(x, "aff", nb, rng))})
    md += [pd.DataFrame(rows).to_markdown(index=False), "", "CIs: bootstrap over the evaluation concepts (injection rates) or over control trials (false positives).", ""]
    md += gate_md
    # contrasts, paired by concept
    md += ["## Key contrasts (Assistant minus X, paired by concept; positive = the Assistant detects more)", ""]
    rows = []
    for other, label in [("sam", "Sam"), ("human", "human character"), ("assistant_long", "length-matched Assistant"), ("narrator", "narrator")]:
        row = {"contrast": f"Assistant - {label}"}
        for col, name in [("aff", "affirmative"), ("ident", "correct identification"), ("strict", "strict")]:
            pa = s2[(s2.persona == "assistant") & s2.injected].groupby("concept")[col].mean()
            po = s2[(s2.persona == other) & s2.injected].groupby("concept")[col].mean().reindex(pa.index)
            row[name] = fmt(*bootstrap_ci((pa - po).to_numpy(), nb, rng))
        fa = s2[(s2.persona == "assistant") & ~s2.injected].aff.to_numpy().astype(float)
        fo = s2[(s2.persona == other) & ~s2.injected].aff.to_numpy().astype(float)
        i, j = rng.integers(0, len(fa), (nb, len(fa))), rng.integers(0, len(fo), (nb, len(fo)))
        s = fa[i].mean(1) - fo[j].mean(1)
        row["control FP"] = fmt(fa.mean() - fo.mean(), *np.percentile(s, [2.5, 97.5]))
        rows.append(row)
    md += [pd.DataFrame(rows).to_markdown(index=False), ""]
    # per concept
    md += ["## Per concept (affirmative / strict rate over trials)", ""]
    piv = s2[s2.injected].groupby(["concept", "persona"]).agg(aff=("aff", "mean"), strict=("strict", "mean")).reset_index()
    t = piv.assign(cell=lambda x: x.aff.map("{:.2f}".format) + " / " + x.strict.map("{:.2f}".format)).pivot(index="concept", columns="persona", values="cell")
    md += [t[[p for p in PERSONAS if p in t.columns]].reset_index().to_markdown(index=False), ""]
    # odd flags
    odd = []
    for r in rows_for_flags(s2, rng, nb):
        odd.append(r)
    md += ["## Anything odd", ""] + ([f"- {o}" for o in odd] or ["- none"]) + [""]
    # samples
    md += ["## Sample responses (first 350 characters)", ""]
    rs = np.random.default_rng(1)
    for p in PERSONAS:
        x = s2[s2.persona == p]
        md += [f"### {p}", ""]
        for title, sub in [("injected, not a strict success", x[x.injected & ~x.strict]), ("injected, strict success", x[x.injected & x.strict]),
                           ("control trial, affirmative (false positive)", x[~x.injected & x.aff]), ("control trial, negative", x[~x.injected & ~x.aff])]:
            if len(sub) == 0:
                md += [f"*{title}*: none", ""]
                continue
            for _, r in sub.sample(min(2, len(sub)), random_state=int(rs.integers(1e6))).iterrows():
                md += [f"*{title}* ({r.concept}): " + json.dumps(r.response[:350], ensure_ascii=False), ""]
    out = os.path.join(config.run_dir(run), f"{NAME}.md")
    open(out, "w").write("\n".join(md))
    print("\n".join(md))


def rows_for_flags(s2, rng, nb):
    out = []
    for p in PERSONAS:
        x = s2[s2.persona == p]
        det, fp = x[x.injected].aff.mean(), x[~x.injected].aff.mean()
        if det >= 0.3 and fp >= 0.3:
            out.append(f"{p}: affirmative rate {det:.2f} with false-positive rate {fp:.2f}: this looks like confabulation, not introspection")
        elif fp >= 0.1:
            out.append(f"{p}: control false-positive rate {fp:.2f}")
        unparsed = (x.affirmative == "UNPARSED").mean()
        if unparsed > 0.02:
            out.append(f"{p}: {unparsed:.1%} of judge outputs could not be parsed")
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("vectors", "generate", "judge", "choose", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--model", choices=list(MODELS), default="llama")
        p.add_argument("--model-id", default=None)
        p.add_argument("--run", default=None)
        if name in ("generate", "judge", "analyze"):
            p.add_argument("--phase", choices=["sweep", "step1", "step2"], required=name != "analyze", default="step1")
        if name == "generate":
            p.add_argument("--sweep-trials", type=int, default=12)
            p.add_argument("--trials", type=int, default=20)
            p.add_argument("--controls", type=int, default=40)
        if name == "judge":
            p.add_argument("--judge-model", default=JUDGE_MODEL)
            p.add_argument("--batch-size", type=int, default=24)
            p.add_argument("--judge-tokens", type=int, default=200)
        if name == "analyze":
            p.add_argument("--n-boot", type=int, default=10_000)
            p.add_argument("--no-gate-exit", action="store_true", help="tests: exit 0 even if the gate fails")
    args = ap.parse_args()
    {"vectors": vectors_cmd, "generate": generate_cmd, "judge": judge_cmd, "choose": choose_cmd, "analyze": analyze_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
