"""Cosine between the persona axis and a refusal direction, per stage and layer (exploratory, white-box).
Refusal direction: mean last-prompt-token activation on harmful/gray requests minus on benign requests (plain Assistant prompts, 18 vs 12 prompts). Persona axis: extras.persona_axis fit
(Assistant framings minus characters, mean over reply tokens). Both are taken at the persona-axis layers. A shuffled-label null (random split of the same prompts) gives the chance cosine.
    python -m extras.axis_refusal_cos run --stage instruct [--tokenizer ...] [--pa-out DIR] ; python -m extras.axis_refusal_cos analyze
"""
import argparse, os
import numpy as np, pandas as pd
from core import config
from core.render import load_tokenizer
from extras import persona_axis as pa
from extras.assistant_steering import BENIGN
from extras.persona_leakage import chat
from extras.persona_refusal import TIERS

OUT = os.environ.get("RC_OUT", os.path.join(config.RESULTS_ROOT, config.FAMILY, "axis_refusal_cos"))


def run(a):
    import torch
    from core.score import load_model, resolve
    model_id, stage = resolve(a.stage)
    tok = load_tokenizer(a.tokenizer or config.TEMPLATE_MODEL)
    model, device = load_model(model_id, torch.bfloat16)
    z = np.load(os.path.join(a.pa_out or pa.OUT, f"{stage}.npz")); layers = [int(x) for x in z["layers"]]
    A, ia = z["acts"], z["is_asst"]; p = A[ia].mean(0) - A[~ia].mean(0)
    def last(r):
        ids = torch.tensor([tok.encode(chat(tok, [{"role": "user", "content": "Please " + r}]), add_special_tokens=False)]).to(device)
        with torch.no_grad(): hs = model(input_ids=ids, output_hidden_states=True).hidden_states
        return np.stack([hs[l][0, -1].float().cpu().numpy() for l in layers])
    H = np.stack([last(r) for r in TIERS["harmful"] + TIERS["gray"][:6]]); B = np.stack([last(r) for r in BENIGN]); allx = np.concatenate([H, B])
    ref = H.mean(0) - B.mean(0); cos = lambda u, v: (u * v).sum(-1) / np.linalg.norm(u, axis=-1) / np.linalg.norm(v, axis=-1)
    rng = np.random.default_rng(0); null = []
    for _ in range(200):
        pm = rng.permutation(len(allx)); a_, b_ = allx[pm[:len(H)]], allx[pm[len(H):]]; null.append(np.abs(cos(a_.mean(0) - b_.mean(0), p)))
    null = np.stack(null)
    rows = [dict(stage=stage, layer=l, cos=float(cos(ref, p)[j]), null95=float(np.percentile(null[:, j], 95))) for j, l in enumerate(layers)]
    os.makedirs(OUT, exist_ok=True); pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{stage}.csv"), index=False); print(rows)


def analyze(a):
    d = pd.concat([pd.read_csv(os.path.join(OUT, f)) for f in sorted(os.listdir(OUT)) if f.endswith(".csv")])
    L = ["# Cosine(persona axis, refusal direction) (exploratory)\n", "Null: 95th percentile of |cosine| for a random split of the same prompts.\n", "| stage | " + " | ".join(f"layer {l}" for l in sorted(d.layer.unique())) + " |", "|---|" + "---|" * d.layer.nunique()]
    for s, g in d.groupby("stage", sort=False):
        L.append(f"| {s} | " + " | ".join(f"{r.cos:+.2f} (null {r.null95:.2f})" for r in g.sort_values("layer").itertuples()) + " |")
    open(os.path.join(os.path.dirname(OUT), "axis_refusal_cos.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="c", required=True)
    r = sp.add_parser("run"); r.add_argument("--stage", required=True); r.add_argument("--tokenizer"); r.add_argument("--pa-out")
    sp.add_parser("analyze"); a = ap.parse_args(); {"run": run, "analyze": analyze}[a.c](a)
