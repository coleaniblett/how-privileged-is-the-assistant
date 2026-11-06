"""White-box map of the 'assistant direction' (exploratory; the README's interp arm, slot x content version).

Design: 300 WildChat user turns (U) and 300 WildChat assistant replies (A), each rendered in BOTH slots with the model's chat
template (core.render.render), so the content tokens are identical across slots. Four cells: UU (user text, user slot), AA
(assistant text, assistant slot), UA (user text in the assistant slot), AU (assistant text in the user slot). Activation = mean
residual-stream vector over the content tokens at a layer. On the TRAIN half (even ids) the direction is d = mu(AA) - mu(UU);
on the TEST half each text's coordinate is s = (x - mu_UU) . d / |d|^2 so UU ~ 0 and AA ~ 1. Reported per stage and layer:
mean coordinate of each cell, slot pull = s(UA) - s(UU) (how far the assistant SLOT moves user text toward the assistant pole)
and content pull = s(AU) - s(UU) (how far assistant CONTENT moves it while in the user slot). Overgeneralization/privilege
reading: the slot pull growing with post-training (change from base), with CIs over test texts.
    python -m extras.assistant_axis score --stage instruct [--tokenizer ...] ; python -m extras.assistant_axis analyze
"""
import argparse, json, os
import numpy as np, pandas as pd
from core import config
from core.render import load_tokenizer, render, load_jsonl
from core.stats import bootstrap_ci, stage_order

OUT = os.environ.get("AX_OUT", os.path.join("results/overgeneralization/early_probes", config.FAMILY, "assistant_axis"))
REPORT = os.environ.get("AX_REPORT", os.path.join("results/overgeneralization/early_probes", config.FAMILY, "assistant_axis.md"))
N = int(os.environ.get("AX_N", 300))


def texts():
    rows = [r for r in load_jsonl(config.DATA_PATH) if r["source"] == "wildchat"]
    u = [r for r in rows if r["kind"] == "user"][:N]
    a = [r for r in rows if r["kind"] == "assistant"][:N]
    return u, a


def score(args):
    import torch
    from core.score import load_model, resolve
    model_id, stage = resolve(args.stage)
    tok = load_tokenizer(args.tokenizer or config.TEMPLATE_MODEL)
    model, device = load_model(model_id, torch.bfloat16)
    nl = model.config.get_text_config().num_hidden_layers
    layers = sorted({int(round(nl * f)) for f in (0.25, 0.4, 0.55, 0.7, 0.85, 1.0)})
    u, a = texts()
    cells = {"UU": (u, "user"), "UA": (u, "assistant"), "AA": (a, "assistant"), "AU": (a, "user")}
    acts = {}
    with torch.no_grad():
        for name, (rows, slot) in cells.items():
            out = []
            for r in rows:
                rd = render(tok, r["text"], slot)
                ids = torch.tensor([rd.input_ids]).to(device)
                hs = model(input_ids=ids, output_hidden_states=True).hidden_states
                out.append(torch.stack([hs[l][0, rd.start:rd.end].float().mean(0) for l in layers]).cpu().numpy())
            acts[name] = np.stack(out)  # text x layer x dim
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, f"{stage.replace('/', '_')}.npz"), layers=np.array(layers), **acts)
    print(stage, layers, {k: v.shape for k, v in acts.items()})


def coords(z):
    """z: dict cell -> text x layer x dim. Returns {cell: text x layer} test-half coordinates."""
    tr = {k: v[0::2] for k, v in z.items()}; te = {k: v[1::2] for k, v in z.items()}
    muUU = tr["UU"].mean(0); d = tr["AA"].mean(0) - muUU
    den = (d * d).sum(-1)
    return {k: ((v - muUU) * d).sum(-1) / den for k, v in te.items()}


def analyze(args):
    rng = np.random.default_rng(0)
    files = sorted(f for f in os.listdir(OUT) if f.endswith(".npz"))
    names = [f[:-4] for f in files]
    order = stage_order(names)
    Z, C = {}, {}
    for n in order:
        z = np.load(os.path.join(OUT, n + ".npz")); Z[n] = z["layers"]
        C[n] = coords({k: z[k] for k in ("UU", "UA", "AA", "AU")})
    L = ["# Assistant direction map (exploratory)\n",
         "Coordinates: user text in user slot = 0, assistant text in assistant slot = 1 (direction fit on the even half, read on the odd half). 95% bootstrap CIs over test texts.\n"]
    nl = len(next(iter(Z.values())))
    for li in range(nl):
        L += [f"\n## layer index {li} (layers " + ", ".join(f"{n}: {Z[n][li]}" for n in order[:1]) + ")\n", "| quantity | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
        def f(v):
            m, lo, hi = bootstrap_ci(v, 3000, rng); return f"{m:.2f} [{lo:.2f}, {hi:.2f}]"
        for cell in ("UU", "AU", "UA", "AA"):
            L.append(f"| {cell} | " + " | ".join(f"{C[n][cell][:, li].mean():.2f}" for n in order) + " |")
        L.append("| slot pull s(UA)-s(UU) | " + " | ".join(f(C[n]["UA"][:, li] - C[n]["UU"][:, li].mean()) for n in order) + " |")
        L.append("| content pull s(AU)-s(UU) | " + " | ".join(f(C[n]["AU"][:, li] - C[n]["UU"][:, li].mean()) for n in order) + " |")
        L.append("| change in slot pull from base | " + " | ".join("-" if n == order[0] else f(C[n]["UA"][:, li] - C[order[0]]["UA"][:, li]) for n in order) + " |")
    open(REPORT, "w").write("\n".join(L) + "\n"); print("wrote", REPORT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="c", required=True)
    s = sp.add_parser("score"); s.add_argument("--stage", required=True); s.add_argument("--tokenizer")
    sp.add_parser("analyze"); a = ap.parse_args(); {"score": score, "analyze": analyze}[a.c](a)
