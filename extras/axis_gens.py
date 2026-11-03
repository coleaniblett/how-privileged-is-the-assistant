"""Where does generated speech sit on the assistant axis? (exploratory, white-box; uses extras.assistant_axis activations)

For each OLMo stage, the stage's own direction d = mu(AA) - mu(UU) (all 300 texts each, extras.assistant_axis) defines coordinate
s = (x - mu_UU) . d / |d|^2 (user text in user slot = 0, assistant text in assistant slot = 1). The coordinate is read on the mean
activation over the REPLY tokens of generations that the same stage produced earlier (no new sampling):
  persona_leakage tics probe: Assistant / role-played character set up in the user turn / in the system prompt (small talk)
  persona_leakage usertics probe: the model-written next user turn
  persona_refusal: 5 set-ups x 36 requests x 4 samples, split by whether the reply opens with a boilerplate refusal
Reading: thin mask (H1) predicts character speech far below the Assistant's own replies; slot privilege predicts character speech in the
assistant slot is pulled up; refusal-voice leak predicts refusing character replies sit higher than complying ones. Bootstrap over generations.
    python -m extras.axis_gens score --stage instruct ; python -m extras.axis_gens analyze
"""
import argparse, json, os
import numpy as np, pandas as pd
from core import config
from core.render import load_tokenizer
from core.stats import bootstrap_ci, stage_order
from extras import persona_leakage as pl, persona_refusal as pr
from extras.persona_leakage import REFUSAL_RE

PERSONA = os.environ.get("AG_AXIS") == "persona"  # use the persona-based axis (extras.persona_axis) instead of the slot-based one
AX = os.path.join(config.RESULTS_ROOT, config.FAMILY, "persona_axis" if PERSONA else "assistant_axis")
OUT = os.path.join(config.RESULTS_ROOT, config.FAMILY, "axis_gens_persona" if PERSONA else "axis_gens")


def jobs(stage):
    """(source, probe, item, cond, prompt_string, reply_text) for every stored generation of this stage."""
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    out = []
    lk = {("tics", i, c): m for i, c, m in pl.tic_items()}
    lk.update({("usertics", i, c): m for i, c, m in pl.usertic_items()})
    with open(os.path.join(pl.OUT, f"{stage}_gens.jsonl"), encoding="utf-8") as f:
        for line in f:
            g = json.loads(line)
            if g["probe"] not in ("tics", "usertics"):
                continue
            m = lk[(g["probe"], g["item"], g["cond"])]
            p = pl.user_prompt(tok, m) if g["probe"] == "usertics" else pl.chat(tok, m)
            out.append(("leak", g["probe"], g["item"], g["cond"], p, g["text"], REFUSAL_RE.search(g["text"][:160]) is not None))
    rl = {(i, c): m for i, t, c, m in pr.items()}
    with open(os.path.join(pr.OUT, f"{stage}.jsonl"), encoding="utf-8") as f:
        for line in f:
            g = json.loads(line)
            m = rl[(g["item"], g["cond"])]
            out.append(("refusal", "refusal", g["item"], g["cond"], pl.chat(tok, m), g["text"], REFUSAL_RE.search(g["text"][:160]) is not None))
    return out


def score(a):
    import torch
    from core.score import load_model, resolve
    model_id, stage = resolve(a.stage)
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    z = np.load(os.path.join(AX, f"{stage}.npz")); layers = [int(x) for x in z["layers"]]
    if PERSONA:
        mu = z["acts"][~z["is_asst"]].mean(0); d = z["acts"][z["is_asst"]].mean(0) - mu
    else:
        mu = z["UU"].mean(0); d = z["AA"].mean(0) - mu
    den = (d * d).sum(-1)
    model, device = load_model(model_id, torch.bfloat16)
    rows = []
    with torch.no_grad():
        for src, probe, item, cond, prompt, reply, refused in jobs(stage):
            p = tok.encode(prompt, add_special_tokens=False); r = tok.encode(reply, add_special_tokens=False)[:80]
            if len(r) < 4:
                continue
            hs = model(input_ids=torch.tensor([p + r]).to(device), output_hidden_states=True).hidden_states
            x = torch.stack([hs[l][0, len(p):].float().mean(0) for l in layers]).cpu().numpy()
            s = ((x - mu) * d).sum(-1) / den
            rows.append(dict(stage=stage, src=src, probe=probe, item=item, cond=cond, refused=refused, **{f"s{l}": float(v) for l, v in zip(layers, s)}))
    os.makedirs(OUT, exist_ok=True); pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{stage}.csv"), index=False); print(stage, len(rows))


def analyze(a):
    rng = np.random.default_rng(0)
    d = pd.concat([pd.read_csv(os.path.join(OUT, f)) for f in sorted(os.listdir(OUT)) if f.endswith(".csv")])
    stages = stage_order(d.stage.unique()); lcols = [c for c in d.columns if c.startswith("s") and c[1:].isdigit()]
    L = ["# Generated speech on the assistant axis (exploratory)\n", "Coordinate: " + ("mean character reply = 0, mean Assistant-framing reply = 1, per stage (persona axis)." if PERSONA else "user text/user slot = 0, assistant text/assistant slot = 1, per stage.") + " Mean over reply tokens. 95% bootstrap CI over generations.\n"]
    def f(v):
        m, lo, hi = bootstrap_ci(np.asarray(v), 2000, rng); return f"{m:.2f} [{lo:.2f}, {hi:.2f}]"
    for lc in ((lcols[1], lcols[2]) if PERSONA else (lcols[2], lcols[4])):
        L.append(f"\n## layer {lc[1:]}\n")
        groups = [("small talk: Assistant", lambda x: (x.probe == "tics") & (x.cond == "assistant")),
                  ("small talk: character (user turn)", lambda x: (x.probe == "tics") & (x.cond == "char_user")),
                  ("small talk: character (system prompt)", lambda x: (x.probe == "tics") & (x.cond == "char_sys")),
                  ("model-written user turn", lambda x: x.probe == "usertics"),
                  ("harmful request: Assistant", lambda x: (x.probe == "refusal") & (x.cond == "assistant")),
                  ("harmful request: character (user turn)", lambda x: (x.probe == "refusal") & (x.cond == "char_user")),
                  ("harmful request: character (system prompt)", lambda x: (x.probe == "refusal") & (x.cond == "char_sys")),
                  ("  ... character replies that refuse (all set-ups)", lambda x: (x.probe == "refusal") & (x.cond != "assistant") & x.refused),
                  ("  ... character replies that comply (all set-ups)", lambda x: (x.probe == "refusal") & (x.cond != "assistant") & ~x.refused)]
        L += ["| speech | " + " | ".join(stages) + " |", "|---|" + "---|" * len(stages)]
        for name, sel in groups:
            L.append(f"| {name} | " + " | ".join(f(d[(d.stage == s)][sel(d[d.stage == s])][lc]) if sel(d[d.stage == s]).sum() > 2 else "-" for s in stages) + " |")
    open(os.path.join(config.RESULTS_ROOT, config.FAMILY, "axis_gens_persona.md" if PERSONA else "axis_gens.md"), "w").write("\n".join(L) + "\n"); print("ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="c", required=True)
    s = sp.add_parser("score"); s.add_argument("--stage", required=True)
    sp.add_parser("analyze"); a = ap.parse_args(); {"score": score, "analyze": analyze}[a.c](a)
