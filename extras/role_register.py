"""Do helpful human roles get absorbed toward the Assistant end after post-training? (exploratory, white-box)

36 roles in three groups answer the same 16 questions (each stage's own samples, 60 tokens). Activation = mean over reply tokens at the persona-axis layer index 1. Coordinate on the FIXED post-trained
persona axis (0 = its mean character reply, 1 = its mean Assistant-framing reply), so stage differences are not a re-fitting artefact:
  helpful   12 service/help roles played by humans (hotel clerk, librarian, nurse, teacher, pharmacist, tutor, ...)
  neutral   12 plain non-service professions (truck driver, accountant, carpenter, farmer, ...)
  stylized  12 strongly voiced characters (pirate, jester, wizard, noir detective, ...)
Prediction under 'register overgeneralization': helpful roles rise from base to post-trained stages, neutral less, stylized fall. Bootstrap over roles (per-role means).
    python -m extras.role_register run --stage instruct [--tokenizer ..] [--axis-stage ..] ; python -m extras.role_register analyze
"""
import argparse, os
import numpy as np, pandas as pd
from core import config
from core.render import load_tokenizer
from core.stats import bootstrap_ci, stage_order
from extras import persona_axis as pa
from extras.persona_leakage import chat

OUT = os.environ.get("RR_OUT", os.path.join("results/overgeneralization/early_probes", config.FAMILY, "role_register"))
REPORT = os.environ.get("RR_REPORT", os.path.join("results/overgeneralization/early_probes", config.FAMILY, "role_register.md"))
ROLES = {
 "helpful": ["a helpful human hotel front-desk clerk", "a helpful human librarian", "a kind human nurse", "a patient human high-school teacher", "a friendly human pharmacist", "a supportive human math tutor",
             "a courteous human bank teller", "a cheerful human tour guide", "an attentive human receptionist at a dental clinic", "a caring human social worker", "a knowledgeable human IT helpdesk technician", "a polite human airline check-in agent"],
 "neutral": ["a human long-haul truck driver", "a human accountant", "a human carpenter", "a human apple farmer", "a human commercial pilot", "a human welder",
             "a human warehouse worker", "a human surveyor", "a human bus driver", "a human plumber", "a human lab technician", "a human postal carrier"],
 "stylized": ["a swaggering pirate captain", "a medieval court jester", "a gruff wizard who lives in a tower", "a hard-boiled noir detective", "a pompous Victorian aristocrat", "a mischievous talking cat",
              "a laconic space-freighter pilot", "an excitable twelve-year-old", "a dramatic Shakespearean actor", "a sleepy lighthouse keeper who speaks in riddles", "a cowboy from an old western", "a villainous mad scientist"]}


NAMES = ["Hannah", "Tom", "Priya", "Jonas", "Lena", "Sam", "Rosa", "Henry", "Aiko", "Dmitri", "Grace", "Pete", "Marta", "Omar", "Ines", "Kofi", "Elsa", "Arjun", "Nadia", "Yusuf", "Maeve", "Tomas", "Lucia", "Henrik",
         "Bram", "Greta", "Osei", "Mora", "Paulo", "Wen", "Sofia", "Idris", "Clara", "Ravi", "Zoe", "Niko"]
SURNAMES = ["Reed", "Reyes", "Nair", "Weber", "Petrova", "Okoye", "Alvarez", "Doyle", "Tanaka", "Volkov", "Mensah", "Larkin", "Kovacs", "Haddad", "Silva", "Boateng", "Lindqvist", "Mehta", "Rahimi", "Demir", "Byrne", "Novak",
            "Ferrari", "Holm", "Janssen", "Berg", "Asante", "Quinn", "Costa", "Zhang", "Rossi", "Bello", "Fischer", "Patel", "Lopez", "Ivanov"]
FORM = os.environ.get("RR_FORM", "plain")  # plain: "You are a X." | named: "You are Tom, a X." | full: "You are Tom Reyes, a X. Stay in character at all times."


def role_prompt(i, r):
    if FORM == "plain": return f"You are {r}."
    if FORM == "rp": return f"Please role-play as {r}."
    if FORM == "only": return f"Respond only as {r} would respond."
    if FORM == "adopt": return f"You are an AI assistant. For this conversation, adopt this persona: {r}."
    if FORM == "named": return f"You are {NAMES[i]}, {r}."
    return f"You are {NAMES[i]} {SURNAMES[i]}, {r}. Stay in character at all times."


SLOT = os.environ.get("RR_SLOT", "sys")  # sys: persona in the system prompt | user: persona + question in the user turn (default system prompt kept)


def run(a):
    import torch
    from core.score import load_model, resolve
    model_id, stage = resolve(a.stage)
    tok = load_tokenizer(a.tokenizer or config.TEMPLATE_MODEL); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model, device = load_model(model_id, torch.bfloat16)
    z = np.load(os.path.join(pa.OUT, f"{a.axis_stage or stage}.npz")); layers = [int(x) for x in z["layers"]]; li = 1
    A, ia = z["acts"][:, li], z["is_asst"]; mc = A[~ia].mean(0); d = A[ia].mean(0) - mc; gap = np.linalg.norm(d); u = d / gap
    idx = {r: i for i, r in enumerate(r for lst in ROLES.values() for r in lst)}
    jobs = [(g, r, q) for g, lst in ROLES.items() for r in lst for q in pa.QUESTIONS]; rows = []
    torch.manual_seed(0)
    for k in range(0, len(jobs), 16):
        ch = jobs[k:k + 16]; prompts = [chat(tok, pa.msgs(role_prompt(idx[r], r), q) if SLOT == "sys" else [{"role": "user", "content": role_prompt(idx[r], r) + "\n\n" + q}]) for g, r, q in ch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.95, pad_token_id=tok.pad_token_id)
        for (g, r, q), p, s in zip(ch, prompts, gen):
            rep = tok.encode(tok.decode(s[enc["input_ids"].shape[1]:], skip_special_tokens=True), add_special_tokens=False)[:60]
            if len(rep) < 4: continue
            pid = tok.encode(p, add_special_tokens=False)
            with torch.no_grad(): hs = model(input_ids=torch.tensor([pid + rep]).to(device), output_hidden_states=True).hidden_states
            x = hs[layers[li]][0, len(pid):].float().mean(0).cpu().numpy()
            rows.append(dict(stage=stage, group=g, role=r, q=q, coord=float(((x - mc) @ u) / gap)))
    os.makedirs(OUT, exist_ok=True); pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{stage}.csv"), index=False); print(stage, len(rows))


def analyze(a):
    rng = np.random.default_rng(0)
    d = pd.concat([pd.read_csv(os.path.join(OUT, f)) for f in sorted(os.listdir(OUT)) if f.endswith(".csv")]); stages = stage_order(d.stage.unique())
    pr = d.groupby(["stage", "group", "role"]).coord.mean().reset_index()
    L = ["# Register overgeneralization: role groups on the fixed persona axis (exploratory)\n", "36 roles x 16 questions, one sample each; coordinate 0 = mean character reply, 1 = mean Assistant-framing reply of the post-trained model. Mean of per-role means, 95% bootstrap CI over roles.\n",
         "| group | " + " | ".join(stages) + " |", "|---|" + "---|" * len(stages)]
    f = lambda v: "{:+.2f} [{:+.2f}, {:+.2f}]".format(*bootstrap_ci(np.asarray(v), 3000, rng))
    for g in ROLES:
        L.append(f"| {g} | " + " | ".join(f(pr[(pr.stage == s) & (pr.group == g)].coord) for s in stages) + " |")
    L += ["", "Change from the first stage (paired by role):", "", "| group | " + " | ".join(stages) + " |", "|---|" + "---|" * len(stages)]
    for g in ROLES:
        row = []
        for s in stages:
            if s == stages[0]: row.append("-"); continue
            a_ = pr[(pr.stage == s) & (pr.group == g)].set_index("role").coord; b_ = pr[(pr.stage == stages[0]) & (pr.group == g)].set_index("role").coord
            row.append(f((a_ - b_).dropna().values))
        L.append(f"| {g} | " + " | ".join(row) + " |")
    L += ["", "Per role at the last stage:", ""] + [f"- {g}: " + ", ".join(f"{r.replace('a ', '', 1).replace('human ', '')[:28]} {c:+.2f}" for r, c in pr[(pr.stage == stages[-1]) & (pr.group == g)].set_index("role").coord.items()) for g in ROLES]
    open(REPORT, "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="c", required=True)
    r = sp.add_parser("run"); r.add_argument("--stage", required=True); r.add_argument("--tokenizer"); r.add_argument("--axis-stage")
    sp.add_parser("analyze"); a = ap.parse_args(); {"run": run, "analyze": analyze}[a.c](a)
