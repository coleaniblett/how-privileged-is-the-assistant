"""Register confound control for the persona axis (exploratory, white-box). Is the Assistant-vs-character axis just a formal-vs-casual (or polite-vs-blunt) direction?

Fit a FORMALITY axis with no persona at all: the same 16 questions answered under three 'formal' and three 'casual' style instructions (system prompt), the stage's own sampled replies, mean activation over reply tokens at the
persona-axis layers; saved with the extras.persona_axis schema (is_asst := formal) so extras.persona_axis.intervene/analyze run on it unchanged (PA_OUT=<this dir>). Then: cosine with the persona axis, and the held-out AUC of each axis on the other's contrast.
    python -m extras.formality_axis fit --stage instruct ; python -m extras.formality_axis compare
"""
import argparse, os
import numpy as np
from core import config
from core.render import load_tokenizer
from extras import persona_axis as pa
from extras.persona_leakage import chat

OUT = os.environ.get("FA_OUT", os.path.join("results/overgeneralization/early_probes", config.FAMILY, "formality_axis"))
FORMAL = ["Answer in a very formal, polished tone.", "Respond like a professional writing a formal letter.", "Use precise, formal, elevated language."]
CASUAL = ["Answer in a very casual, slangy tone.", "Respond like a friend texting you.", "Use relaxed, informal, everyday language."]


def fit(a):
    import torch
    from core.score import load_model, resolve
    model_id, stage = resolve(a.stage)
    tok = load_tokenizer(a.tokenizer or config.TEMPLATE_MODEL); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model, device = load_model(model_id, torch.bfloat16); layers = pa.layers_of(model)
    roles = FORMAL + CASUAL; jobs = [(ri, q) for ri in range(len(roles)) for q in pa.QUESTIONS]
    torch.manual_seed(0); acts, meta = [], []
    for k in range(0, len(jobs), 16):
        ch = jobs[k:k + 16]; prompts = [chat(tok, pa.msgs(roles[ri], q)) for ri, q in ch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad(): g = model.generate(**enc, max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.95, pad_token_id=tok.pad_token_id)
        for (ri, q), p, s in zip(ch, prompts, g):
            r = tok.encode(tok.decode(s[enc["input_ids"].shape[1]:], skip_special_tokens=True), add_special_tokens=False)[:60]
            if len(r) < 4: continue
            pid = tok.encode(p, add_special_tokens=False)
            with torch.no_grad(): hs = model(input_ids=torch.tensor([pid + r]).to(device), output_hidden_states=True).hidden_states
            acts.append(torch.stack([hs[l][0, len(pid):].float().mean(0) for l in layers]).cpu().numpy()); meta.append((ri, ri < len(FORMAL)))
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, f"{stage}.npz"), acts=np.stack(acts), role=np.array([m[0] for m in meta]), is_asst=np.array([m[1] for m in meta]), layers=np.array(layers)); print(stage, len(acts))


def compare(a):
    L = ["# Formality axis vs persona axis (exploratory)\n", "Formality axis = formal minus casual style instructions (no persona). Cosine and cross-AUC at each layer (persona-axis layers).\n",
         "| stage | layer | cos(persona, formality) | AUC of persona axis on formal-vs-casual replies | AUC of formality axis on Assistant-vs-character replies |", "|---|---|---|---|---|"]
    for f in sorted(x for x in os.listdir(OUT) if x.endswith(".npz")):
        st = f[:-4]; zf = np.load(os.path.join(OUT, f)); zp = np.load(os.path.join(a.persona_dir or pa.OUT, f))
        dp, _ = pa.axis_of(zp); df, _ = pa.axis_of(zf)
        for j, l in enumerate(zf["layers"]):
            cos = (dp[j] @ df[j]) / np.linalg.norm(dp[j]) / np.linalg.norm(df[j])
            xf = zf["acts"][:, j] @ dp[j]; xp = zp["acts"][:, j] @ df[j]
            L.append(f"| {st} | {l} | {cos:+.2f} | {pa.auc(xf[zf['is_asst']], xf[~zf['is_asst']]):.2f} | {pa.auc(xp[zp['is_asst']], xp[~zp['is_asst']]):.2f} |")
    open(os.path.join(os.path.dirname(OUT), "formality_axis.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="c", required=True)
    f = sp.add_parser("fit"); f.add_argument("--stage", required=True); f.add_argument("--tokenizer")
    c = sp.add_parser("compare"); c.add_argument("--persona-dir")
    a = ap.parse_args(); {"fit": fit, "compare": compare}[a.c](a)
