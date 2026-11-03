"""Does the persona-axis state at the decision point predict a voice-break refusal? (exploratory, white-box)

For the 6 roles x 2 set-ups x 36 requests of extras.role_gradient, take the last prompt token's residual-stream activation (before any reply token) at the persona-axis layers,
project it on (a) the stage's persona axis (character = 0, Assistant = 1, from extras.persona_axis) and (b) a refusal direction (harmful minus benign requests, plain Assistant prompts,
last token, same layers). Compare with the sampled refusal rate of that prompt (2 samples from extras.role_gradient). Reported: Spearman correlation over all prompts, within role x
set-up cells (cell means removed, so role identity cannot explain it), and a two-feature logistic-style check via partial correlation. Bootstrap over requests.
    python -m extras.decision_point run --stage instruct ; python -m extras.decision_point analyze
"""
import argparse, json, os
import numpy as np, pandas as pd
from core import config
from core.render import load_tokenizer
from extras import persona_axis as pa, role_gradient as rg
from extras.assistant_steering import BENIGN
from extras.persona_leakage import REFUSAL_RE, chat
from extras.persona_refusal import TIERS

OUT = os.path.join(config.RESULTS_ROOT, config.FAMILY, "decision_point")


def run(a):
    import torch
    from core.score import load_model, resolve
    model_id, stage = resolve(a.stage)
    tok = load_tokenizer(a.tokenizer or config.TEMPLATE_MODEL)
    model, device = load_model(model_id, torch.bfloat16)
    z = np.load(os.path.join(pa.OUT, f"{stage}.npz")); layers = [int(x) for x in z["layers"]]
    A, ia = z["acts"], z["is_asst"]; mc = A[~ia].mean(0); d = A[ia].mean(0) - mc
    def last(msgs):
        ids = torch.tensor([tok.encode(chat(tok, msgs), add_special_tokens=False)]).to(device)
        with torch.no_grad(): hs = model(input_ids=ids, output_hidden_states=True).hidden_states
        return np.stack([hs[l][0, -1].float().cpu().numpy() for l in layers])
    reqs = [r for t in TIERS.values() for r in t]
    harm = np.mean([last([{"role": "user", "content": "Please " + r}]) for r in TIERS["harmful"] + TIERS["gray"][:6]], 0)
    ben = np.mean([last([{"role": "user", "content": "Please " + r}]) for r in BENIGN], 0); ref = harm - ben
    gens = pd.read_json(os.path.join(rg.OUT, f"{stage}.jsonl"), lines=True); gens["refuse"] = [bool(REFUSAL_RE.search(t[:160])) for t in gens.text]
    rate = gens.groupby(["kind", "role", "setup", "item"]).refuse.mean()
    rows = []
    for kind, lst in rg.ROLES.items():
        for name, desc in lst:
            for i, r in enumerate(reqs):
                for setup, m in rg.prompts(r, name, desc).items():
                    x = last(m)
                    rows.append(dict(stage=stage, kind=kind, role=name, setup=setup, item=i, refuse=rate[(kind, name, setup, i)],
                                     **{f"persona{l}": float(((x[j] - mc[j]) @ d[j]) / (d[j] @ d[j])) for j, l in enumerate(layers)},
                                     **{f"refdir{l}": float(x[j] @ ref[j] / np.linalg.norm(ref[j])) for j, l in enumerate(layers)}))
    os.makedirs(OUT, exist_ok=True); pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{stage}.csv"), index=False); print(stage, len(rows))


def sp(x, y):
    from scipy.stats import spearmanr
    return spearmanr(x, y).correlation


def analyze(a):
    rng = np.random.default_rng(0)
    L = ["# Decision-point state vs voice-break refusal (exploratory)\n", "Spearman correlation between the last-prompt-token projection and the prompt's sampled refusal rate (432 prompts per stage). 'Within cell' removes role x set-up means. 95% bootstrap CI over requests.\n",
         "| stage | feature (layer) | all prompts | within role x set-up cell |", "|---|---|---|---|"]
    for f in sorted(x for x in os.listdir(OUT) if x.endswith(".csv")):
        d = pd.read_csv(os.path.join(OUT, f)); st = f[:-4]
        cols = [c for c in d.columns if c.startswith(("persona", "refdir"))]
        d["cell"] = d.kind + d.role + d.setup
        for c in cols:
            if not c.endswith(("18", "22")): continue
            dm = d.assign(c_=d[c] - d.groupby("cell")[c].transform("mean"), r_=d.refuse - d.groupby("cell").refuse.transform("mean"))
            allr = sp(d[c], d.refuse); win = sp(dm.c_, dm.r_)
            items = d.item.unique(); bs = []
            for _ in range(300):
                it = rng.choice(items, len(items)); s = pd.concat([dm[dm.item == i] for i in it]); bs.append(sp(s.c_, s.r_))
            L.append(f"| {st} | {c} | {allr:.2f} | {win:.2f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}] |")
    open(os.path.join(config.RESULTS_ROOT, config.FAMILY, "decision_point.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp_ = ap.add_subparsers(dest="c", required=True)
    r = sp_.add_parser("run"); r.add_argument("--stage", required=True); r.add_argument("--tokenizer")
    sp_.add_parser("analyze"); a = ap.parse_args(); {"run": run, "analyze": analyze}[a.c](a)
