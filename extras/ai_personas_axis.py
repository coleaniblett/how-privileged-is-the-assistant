"""Is the default Assistant special on the persona axis? (exploratory, white-box)

Same recipe as extras.persona_axis (axis fit on the 3 Assistant framings vs 24 characters; saved activations are reused). New roles answer the same
16 questions with the stage's own sampled replies: other AI personas (named AI, sarcastic AI, pirate-voiced AI, customer-service bot, ship computer) and
helpful HUMAN roles (front-desk clerk, librarian, nurse). Coordinate: 0 = mean character reply, 1 = mean Assistant-framing reply (own stage, layer index 1).
Reading: other AIs near 1 -> the axis encodes AI-ness/assistant-style; near 0 -> only the default Assistant is special (privilege); helpful humans near 1 -> helpfulness.
    python -m extras.ai_personas_axis run --stage instruct [--tokenizer ...] ; python -m extras.ai_personas_axis analyze
"""
import argparse, os
import numpy as np, pandas as pd
from core import config
from core.render import load_tokenizer
from core.stats import bootstrap_ci, stage_order
from extras import persona_axis as pa
from extras.persona_leakage import chat

OUT = os.environ.get("AI_OUT", os.path.join(config.RESULTS_ROOT, config.FAMILY, "ai_personas_axis"))
REPORT = os.environ.get("AI_REPORT", os.path.join(config.RESULTS_ROOT, config.FAMILY, "ai_personas_axis.md"))
ROLES = {"named AI": "You are Aria, a helpful AI model.", "sarcastic AI": "You are Zed, an AI assistant with a sarcastic personality.",
         "pirate AI": "You are PirateBot, an AI assistant that talks like a pirate.", "customer-service bot": "You are a customer-service chatbot for a phone company.",
         "ship computer": "You are HAL, a calm and ominous ship computer.", "human clerk": "You are a helpful human front-desk clerk at a hotel.",
         "human librarian": "You are a helpful human librarian.", "human nurse": "You are a kind human nurse."}


def run(a):
    import torch
    from core.score import load_model, resolve
    model_id, stage = resolve(a.stage)
    tok = load_tokenizer(a.tokenizer or config.TEMPLATE_MODEL); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model, device = load_model(model_id, torch.bfloat16)
    z = np.load(os.path.join(pa.OUT, f"{stage}.npz")); layers = [int(x) for x in z["layers"]]; li = 1
    A, ia = z["acts"][:, li], z["is_asst"]; ma, mc = A[ia].mean(0), A[~ia].mean(0); d = ma - mc
    coord = lambda x: float(((x - mc) @ d) / (d @ d))
    names = list(ROLES); jobs = [(n, q) for n in names for q in pa.QUESTIONS]; rows = []
    torch.manual_seed(0)
    for k in range(0, len(jobs), 16):
        ch = jobs[k:k + 16]; prompts = [chat(tok, pa.msgs(ROLES[n], q)) for n, q in ch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.95, pad_token_id=tok.pad_token_id)
        for (n, q), p, s in zip(ch, prompts, g):
            r = tok.encode(tok.decode(s[enc["input_ids"].shape[1]:], skip_special_tokens=True), add_special_tokens=False)[:60]
            if len(r) < 4: continue
            pid = tok.encode(p, add_special_tokens=False)
            with torch.no_grad():
                hs = model(input_ids=torch.tensor([pid + r]).to(device), output_hidden_states=True).hidden_states
            rows.append(dict(stage=stage, role=n, q=q, coord=coord(hs[layers[li]][0, len(pid):].float().mean(0).cpu().numpy())))
    # reference: coordinates of the fit data itself (Assistant framings, characters)
    for lab, m in (("default Assistant framings (fit)", ia), ("characters (fit)", ~ia)):
        for v in (A[m] @ d - mc @ d) / (d @ d): rows.append(dict(stage=stage, role=lab, q="", coord=float(v)))
    os.makedirs(OUT, exist_ok=True); pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{stage}.csv"), index=False); print(stage, len(rows))


def analyze(a):
    rng = np.random.default_rng(0)
    d = pd.concat([pd.read_csv(os.path.join(OUT, f)) for f in sorted(os.listdir(OUT)) if f.endswith(".csv")]); stages = stage_order(d.stage.unique())
    L = ["# Other AI personas and helpful humans on the persona axis (exploratory)\n", "0 = mean character reply, 1 = mean Assistant-framing reply (own stage). 95% bootstrap CI over replies.\n",
         "| role | " + " | ".join(stages) + " |", "|---|" + "---|" * len(stages)]
    for r in ["default Assistant framings (fit)", *ROLES, "characters (fit)"]:
        row = []
        for s in stages:
            v = d[(d.stage == s) & (d.role == r)].coord.values
            m, lo, hi = bootstrap_ci(v, 2000, rng) if len(v) > 2 else (np.nan,) * 3; row.append(f"{m:.2f} [{lo:.2f}, {hi:.2f}]")
        L.append(f"| {r} | " + " | ".join(row) + " |")
    open(REPORT, "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="c", required=True)
    r = sp.add_parser("run"); r.add_argument("--stage", required=True); r.add_argument("--tokenizer")
    sp.add_parser("analyze"); a = ap.parse_args(); {"run": run, "analyze": analyze}[a.c](a)
