"""Does the persona axis fire more on non-assistant text after post-training? (exploratory, white-box overgeneralization signature)

Fixed axis per model family: the post-trained model's persona axis (extras.persona_axis, layer index 1; OLMo instruct / Gemma instruct), origin = that model's mean character reply, unit = its gap.
The SAME texts are run through every stage and projected on that fixed axis, so stage differences are not an artefact of re-fitting. Texts (mean activation over content tokens):
  wildchat user turn in the user slot (UU), oasst1 user turn in the user slot, the same wildchat user turn as plain prose (no template), and wildchat assistant replies in the assistant slot (AA, reference).
Overgeneralization signature: non-assistant text moving up the axis from base to post-trained stages; reference AA shows what an Assistant text looks like. Bootstrap over texts.
    python -m extras.plain_axis run --stage instruct [--tokenizer ..] [--axis-stage ..] [--pa-out ..] ; python -m extras.plain_axis analyze
"""
import argparse, os
import numpy as np, pandas as pd
from core import config
from core.render import load_tokenizer, render, load_jsonl
from core.stats import bootstrap_ci, stage_order

OUT = os.environ.get("PL_OUT", os.path.join("results/overgeneralization/early_probes", config.FAMILY, "plain_axis"))
REPORT = os.environ.get("PL_REPORT", os.path.join("results/overgeneralization/early_probes", config.FAMILY, "plain_axis.md"))
PA = os.environ.get("PA_OUT", os.path.join("results/overgeneralization/early_probes", config.FAMILY, "persona_axis"))
N = 150


def run(a):
    import torch
    from core.score import load_model, resolve
    model_id, stage = resolve(a.stage)
    tok = load_tokenizer(a.tokenizer or config.TEMPLATE_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model, device = load_model(model_id, torch.bfloat16)
    ref = np.load(os.path.join(PA, f"{a.axis_stage}.npz")); lay_ref = [int(x) for x in ref["layers"]]
    mine = np.load(os.path.join(PA, f"{stage}.npz")); layers = [int(x) for x in mine["layers"]]
    assert layers == lay_ref
    A, ia = ref["acts"], ref["is_asst"]; mc = A[~ia].mean(0); d = A[ia].mean(0) - mc; gap = np.linalg.norm(d, axis=-1)
    u = d / gap[:, None]
    rows_all = load_jsonl(config.DATA_PATH)
    pick = lambda src, kind: [r for r in rows_all if r["source"] == src and r["kind"] == kind][:N]
    sets = {"wildchat user (user slot)": (pick("wildchat", "user"), "user"), "oasst1 user (user slot)": (pick("oasst1", "user"), "user"),
            "wildchat user (plain prose)": (pick("wildchat", "user"), None), "wildchat assistant (assistant slot)": (pick("wildchat", "assistant"), "assistant")}
    out = []
    with torch.no_grad():
        for name, (rows, slot) in sets.items():
            for r in rows:
                rd = render(tok, r["text"], slot)
                hs = model(input_ids=torch.tensor([rd.input_ids]).to(device), output_hidden_states=True).hidden_states
                x = np.stack([hs[l][0, rd.start:rd.end].float().mean(0).cpu().numpy() for l in layers])
                s = ((x - mc) * u).sum(-1) / gap
                out.append(dict(stage=stage, set=name, id=r["id"], **{f"s{l}": float(v) for l, v in zip(layers, s)}))
    os.makedirs(OUT, exist_ok=True); pd.DataFrame(out).to_csv(os.path.join(OUT, f"{stage}.csv"), index=False); print(stage, len(out))


def analyze(a):
    rng = np.random.default_rng(0)
    d = pd.concat([pd.read_csv(os.path.join(OUT, f)) for f in sorted(os.listdir(OUT)) if f.endswith(".csv")]); stages = stage_order(d.stage.unique())
    lc = [c for c in d.columns if c.startswith("s") and c[1:].isdigit()]
    L = ["# Persona axis on non-assistant text, fixed axis across stages (exploratory)\n", "Fixed axis = the post-trained model's persona axis (0 = its mean character reply, 1 = its mean Assistant reply). Mean over content tokens. 95% bootstrap CI over texts.\n"]
    for c in (lc[1], lc[2]):
        L += [f"\n## layer {c[1:]}\n", "| text | " + " | ".join(stages) + " |", "|---|" + "---|" * len(stages)]
        for name in d.set.unique():
            row = []
            for s in stages:
                v = d[(d.stage == s) & (d.set == name)][c].values; m, lo, hi = bootstrap_ci(v, 2000, rng); row.append(f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            L.append(f"| {name} | " + " | ".join(row) + " |")
        if len(stages) > 1:
            L.append("| change from first stage, wildchat user (user slot) | " + " | ".join("-" if s == stages[0] else "{:+.2f} [{:+.2f}, {:+.2f}]".format(*bootstrap_ci(
                d[(d.stage == s) & (d.set == "wildchat user (user slot)")].set_index("id")[c].sub(d[(d.stage == stages[0]) & (d.set == "wildchat user (user slot)")].set_index("id")[c]).dropna().values, 2000, rng)) for s in stages) + " |")
    open(REPORT, "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="c", required=True)
    r = sp.add_parser("run"); r.add_argument("--stage", required=True); r.add_argument("--tokenizer"); r.add_argument("--axis-stage", default="instruct")
    sp.add_parser("analyze"); a = ap.parse_args(); {"run": run, "analyze": analyze}[a.c](a)
