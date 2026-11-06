"""Spec S Stage 0 (materials and pilot). Run on vinyl: python -m extras.big_atlas_stage0 [hashes|run|report]
hashes: CPU, hashes of the fetched sources -> results/big_atlas/stage0/hashes.csv (hashes and counts only)
run:    GPU, 50 replies through each step, the judge check, speeds -> results/big_atlas/private/stage0_pilot.json (has third-party text, git-ignored) and
        results/big_atlas/stage0/speeds.json (numbers only)
report: CPU, writes results/big_atlas/stage0/stage0.md from the two files above (short quotes only, no third-party prompt text beyond a few words)
"""
import json
import os
import sys
import time

import numpy as np

from extras import big_atlas as ba

S0 = os.path.join(ba.OUT, "stage0")
N_FEWSHOT = 5


def hashes():
    import hashlib
    os.makedirs(S0, exist_ok=True)
    rows = []

    def dirhash(d, pat="*.json"):
        import glob
        h = hashlib.sha256()
        fs = sorted(glob.glob(os.path.join(d, pat)))
        for f in fs:
            h.update(os.path.basename(f).encode()); h.update(open(f, "rb").read())
        return h.hexdigest(), len(fs)
    for name, p in [("aa/roles/role_list.json", f"{ba.AA}/roles/role_list.json"), ("aa/traits/trait_list.json", f"{ba.AA}/traits/trait_list.json"),
                    ("aa/extraction_questions.jsonl", f"{ba.AA}/extraction_questions.jsonl"), ("zheng/role_info.csv", f"{ba.ZH}/role_info.csv")]:
        rows.append((name, ba.sha(p), 1))
    for name, d in [("aa/roles/instructions", f"{ba.AA}/roles/instructions"), ("aa/traits/instructions", f"{ba.AA}/traits/instructions")]:
        h, n = dirhash(d); rows.append((name + "/*.json", h, n))
    with open(os.path.join(S0, "hashes.csv"), "w") as f:
        f.write("item,sha256,n_files\n" + "\n".join(f"{a},{b},{c}" for a, b, c in rows) + "\n")
    print(open(os.path.join(S0, "hashes.csv")).read())


def run():
    import torch
    os.makedirs(S0, exist_ok=True); os.makedirs(ba.PRIV, exist_ok=True)
    rng = np.random.default_rng(ba.SEED)
    roles, traits = ba.aa_roles(), ba.aa_traits()
    t0 = time.time(); model, tok = ba.load(); load_s = time.time() - t0
    out = dict(load_s=load_s)
    # ---- rendered prompt, one per method (for the log; stays private)
    rn = sorted(roles)[int(rng.integers(len(roles)))]
    q = roles[rn]["questions"][0]; sysm = roles[rn]["instr"][0]
    ex = [(roles[rn]["questions"][i + 1], f"[frozen reply {i + 1} of the role]") for i in range(N_FEWSHOT)]
    rend = {"system prompt": ba.render(tok, sysm, q), "user turn": ba.render(tok, None, sysm + "\n\n" + q),
            "few-shot (5 own replies)": tok.apply_chat_template(([{"role": "system", "content": sysm}] if False else []) + sum([[{"role": "user", "content": a}, {"role": "assistant", "content": b}] for a, b in ex], []) + [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True),
            "reply prefill": ba.render(tok, sysm, q) + "[first-person opening, spec O wording]",
            "steering (plain prompt, hook adds alpha * (role - Assistant) at layer 33)": ba.render(tok, None, q),
            "random direction (as steering, random unit direction, same norm)": ba.render(tok, None, q)}
    out["rendered_role"] = rn; out["rendered"] = rend
    # ---- 50 replies per step
    tn = [sorted(traits)[i] for i in rng.choice(len(traits), 5, replace=False)]
    tv_prompts, tv_meta = [], []
    for t in tn:
        for k in range(5):
            for side, ix in (("pos", 0), ("neg", 1)):
                tv_prompts.append(ba.render(tok, traits[t]["pairs"][k][ix], traits[t]["questions"][k])); tv_meta.append((t, k, side))
    t0 = time.time(); tv_rep = ba.generate(model, tok, tv_prompts, max_new=128, bs=25); tv_s = time.time() - t0
    rn5 = [sorted(roles)[i] for i in rng.choice(len(roles), 5, replace=False)]
    rp, rm = [], []
    for r in rn5:
        for k in range(10):
            rp.append(ba.render(tok, roles[r]["instr"][0], roles[r]["questions"][k])); rm.append((r, k))
    t0 = time.time(); ro_rep = ba.generate(model, tok, rp, max_new=256, bs=25); ro_s = time.time() - t0
    out["gen"] = dict(trait_vector_128=dict(n=len(tv_prompts), s=tv_s, per_s=len(tv_prompts) / tv_s, capped=int(sum(r["capped"] for r in tv_rep))),
                      role_256=dict(n=len(rp), s=ro_s, per_s=len(rp) / ro_s, capped=int(sum(r["capped"] for r in ro_rep)), mean_tokens=float(np.mean([r["n_tokens"] for r in ro_rep]))))
    # bigger batch speed at 256 tokens
    t0 = time.time(); _ = ba.generate(model, tok, rp * 2, max_new=256, bs=50); s100 = time.time() - t0
    out["gen"]["role_256_bs100"] = dict(n=len(rp) * 2, s=s100, per_s=len(rp) * 2 / s100)
    # ---- judge: trait check (pos side minus neg side), role score, shared-matrix timing, tree vs first-token
    jp = [ba.judge_prompt(traits[t]["eval_prompt"], traits[t]["questions"][k], r["text"]) for (t, k, s), r in zip(tv_meta, tv_rep)]
    t0 = time.time(); j_tv = ba.judge_scores(model, tok, jp, bs=10); j_s = time.time() - t0
    j_tv_first = ba.judge_scores(model, tok, jp, bs=10, method="first")
    jr = [ba.judge_prompt(roles[r]["eval_prompt"], roles[r]["questions"][k], x["text"]) for (r, k), x in zip(rm, ro_rep)]
    j_ro = ba.judge_scores(model, tok, jr, bs=10); j_ro_first = ba.judge_scores(model, tok, jr, bs=10, method="first")
    mat_p = [ba.judge_prompt(traits[t]["eval_prompt"], roles[r]["questions"][k], x["text"]) for t in tn for (r, k), x in list(zip(rm, ro_rep))[:10]]
    t0 = time.time(); j_mat = ba.judge_scores(model, tok, mat_p, bs=10); m_s = time.time() - t0
    t0 = time.time(); j_mat1 = ba.judge_scores(model, tok, mat_p, bs=10, method="first"); m1_s = time.time() - t0
    out["judge"] = dict(trait_vec=dict(n=len(jp), per_s=len(jp) / j_s), matrix=dict(n=len(mat_p), per_s=len(mat_p) / m_s), matrix_first_token=dict(n=len(mat_p), per_s=len(mat_p) / m1_s))
    both = [(a[0], b[0]) for a, b in list(zip(j_tv, j_tv_first)) + list(zip(j_ro, j_ro_first)) + list(zip(j_mat, j_mat1)) if a[0] is not None and b[0] is not None]
    from scipy.stats import spearmanr
    A, B = np.array(both).T
    out["tree_vs_first"] = dict(n=len(both), spearman=float(spearmanr(A, B)[0]), mean_abs_diff=float(np.abs(A - B).mean()), none_tree=int(sum(x[0] is None for x in j_tv + j_ro + j_mat)))
    gaps = {}
    for t in tn:
        pos = [j[0] for j, m in zip(j_tv, tv_meta) if m[0] == t and m[2] == "pos" and j[0] is not None]
        neg = [j[0] for j, m in zip(j_tv, tv_meta) if m[0] == t and m[2] == "neg" and j[0] is not None]
        gaps[t] = dict(pos=float(np.mean(pos)) if pos else None, neg=float(np.mean(neg)) if neg else None, n_pos=len(pos), n_neg=len(neg))
    out["trait_check"] = gaps
    out["role_scores"] = {r: float(np.mean([j[0] for j, m in zip(j_ro, rm) if m[0] == r and j[0] is not None] or [np.nan])) for r in rn5}
    out["judge_none"] = dict(trait_vec=int(sum(x[0] is None for x in j_tv)), role=int(sum(x[0] is None for x in j_ro)))
    # ---- state forward speeds (long and short prompts)
    sp = [ba.render(tok, roles[rn5[i % 5]]["instr"][i % 5], roles[rn5[i % 5]]["questions"][i % 40]) for i in range(384)]
    torch.cuda.synchronize(); t0 = time.time(); _ = ba.states(model, tok, sp, bs=32); st_s = time.time() - t0
    short = [ba.render(tok, None, "Is the following statement something you would say? Answer Yes or No. I enjoy long walks.") for _ in range(512)]
    t0 = time.time(); _ = ba.states(model, tok, short, bs=64); sh_s = time.time() - t0
    out["forward"] = dict(role_prompt_states=dict(n=len(sp), per_s=len(sp) / st_s), short_prompt=dict(n=len(short), per_s=len(short) / sh_s, tokens=len(tok(short[0]).input_ids)))
    out["tokens_example_reply"] = ro_rep[0]["text"][:120]
    json.dump(out, open(os.path.join(ba.PRIV, "stage0_pilot.json"), "w"), indent=1)
    json.dump(dict(out, rendered=None, tokens_example_reply=None), open(os.path.join(S0, "speeds.json"), "w"), indent=1)
    json.dump(dict(tv=[dict(m=m, text=r["text"], score=j[0]) for m, r, j in zip(tv_meta, tv_rep, j_tv)], role=[dict(m=m, text=r["text"], score=j[0]) for m, r, j in zip(rm, ro_rep, j_ro)]),
              open(os.path.join(ba.PRIV, "stage0_replies.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("rendered", "tokens_example_reply")}, indent=1))


if __name__ == "__main__":
    {"hashes": hashes, "run": run}[sys.argv[1]]()
