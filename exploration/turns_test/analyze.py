"""Turns test: frame costs, the vestigial-regression asymmetry and the logit-distortion gap under designs A, B, C.

    python -m exploration.turns_test.analyze      # -> exploration/turns_test/results/results.md, summary.csv

Per conversation c and cell x (u1, rA, rB, u2A, u2C; see score.py), over content positions 1+:
    frame cost  FC_x = NLL(tagged) - NLL(plain)                 (the role-specific term of the VR README)
    gap         G_x  = NLL(tagged) - entropy(tagged)            (logit distortion's calibration gap, own slot)
    change from base  D[.] = value(post-trained) - value(base), paired per conversation.
Asymmetry (VR primary quantity, user minus reply):
    A: D[FC_u1] - D[FC_rA]     B: D[FC_u1] - D[FC_rB]     C: D[FC_u2C] - D[FC_rB]
Logit-distortion contrast (reply minus user): A: D[G_rA] - D[G_u1]; B: D[G_rB] - D[G_u1]; C: D[G_rB] - D[G_u2C].
All three texts come from the same conversation, so every CI here is a paired bootstrap over conversations
(the locked VR analysis resamples user and assistant sets independently because there they are different texts).
Sources are never pooled. Nats per token, within family only. n = 50 conversations per source: CIs are wide.
"""

import os

import numpy as np
import pandas as pd

from core.stats import bootstrap_ci

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FAMILIES = {"olmo3-7b": "OLMo 3 7B Instruct", "qwen3-8b": "Qwen3-8B"}
SOURCES = ["wildchat", "oasst1", "ultrachat"]
N_BOOT = 10_000


def wide(fam, stage):
    d = pd.read_csv(os.path.join(RES, fam, f"{stage}.csv"))
    d = d.pivot_table(index=["conv_id", "source"], columns=["cell", "labeling"],
                      values=["mean_nll_from1", "mean_entropy_from1"])
    out = pd.DataFrame(index=d.index)
    for cell in ["u1", "rA", "rB", "u2A", "u2C"]:
        tag, pl = d[("mean_nll_from1", cell, "tagged")], d[("mean_nll_from1", cell, "plain")]
        out[f"NLLtag_{cell}"] = tag
        out[f"NLLplain_{cell}"] = pl
        out[f"FC_{cell}"] = tag - pl
        out[f"G_{cell}"] = tag - d[("mean_entropy_from1", cell, "tagged")]
    return out.reset_index()


def quantities(base, post):
    b, p = base.set_index("conv_id"), post.set_index("conv_id").loc[base.conv_id]
    D = lambda col: (p[col] - b[col]).to_numpy()
    q = {}
    for cell in ["u1", "rA", "rB", "u2A", "u2C"]:
        q[f"D frame cost {cell}"] = D(f"FC_{cell}")
    for cell in ["u1", "rA", "rB", "u2A", "u2C"]:
        q[f"D gap {cell}"] = D(f"G_{cell}")
    q["asym A (u1 - rA)"] = D("FC_u1") - D("FC_rA")
    q["asym B (u1 - rB)"] = D("FC_u1") - D("FC_rB")
    q["asym C (u2C - rB)"] = D("FC_u2C") - D("FC_rB")
    q["context effect on reply FC, D[FC_rA - FC_rB] (= asym B - asym A)"] = D("FC_rA") - D("FC_rB")
    q["  of which tagged, D[NLLtag_rA - NLLtag_rB]"] = D("NLLtag_rA") - D("NLLtag_rB")
    q["  of which plain, D[NLLplain_rA - NLLplain_rB]"] = D("NLLplain_rA") - D("NLLplain_rB")
    q["context effect on user-2 FC, D[FC_u2A - FC_u2C]"] = D("FC_u2A") - D("FC_u2C")
    q["LD contrast A (G_rA - G_u1)"] = D("G_rA") - D("G_u1")
    q["LD contrast B (G_rB - G_u1)"] = D("G_rB") - D("G_u1")
    q["LD contrast C (G_rB - G_u2C)"] = D("G_rB") - D("G_u2C")
    q["context effect on reply gap, D[G_rA - G_rB]"] = D("G_rA") - D("G_rB")
    return q


def levels(frame):
    """Undifferenced (single-model) values for orientation."""
    return {f"FC_{c}": frame[f"FC_{c}"].to_numpy() for c in ["u1", "rA", "rB", "u2A", "u2C"]} | {
        f"G_{c}": frame[f"G_{c}"].to_numpy() for c in ["u1", "rA", "rB", "u2C"]}


def cell(m, lo, hi):
    star = "*" if (lo > 0 or hi < 0) and abs(m) >= 0.05 else ""
    return f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]{star}"


def main():
    rng = np.random.default_rng(0)
    rows = []
    for fam in FAMILIES:
        if not all(os.path.exists(os.path.join(RES, fam, f"{s}.csv")) for s in ("base", "instruct")):
            print(f"skip {fam}: scores missing")
            continue
        base, post = wide(fam, "base"), wide(fam, "instruct")
        for src in SOURCES:
            b, p = base[base.source == src], post[post.source == src]
            for name, x in quantities(b, p).items():
                rows.append(dict(family=fam, source=src, kind="change from base", quantity=name, n=len(x),
                                 **dict(zip(["mean", "lo", "hi"], bootstrap_ci(x, N_BOOT, rng)))))
            for stage, fr in (("base", b), ("instruct", p)):
                for name, x in levels(fr).items():
                    rows.append(dict(family=fam, source=src, kind=f"level {stage}", quantity=name, n=len(x),
                                     **dict(zip(["mean", "lo", "hi"], bootstrap_ci(x, N_BOOT, rng)))))
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(RES, "summary.csv"), index=False)
    s["cell"] = [cell(m, lo, hi) for m, lo, hi in zip(s["mean"], s.lo, s.hi)]
    lines = ["# Turns test: results", "",
             "Nats per token over content positions 1+, mean over conversations with a 95% paired bootstrap CI "
             f"({N_BOOT} resamples), n = 50 conversations per source (small: CIs are wide). `*`: |mean| >= 0.05 and the CI "
             "excludes zero. Change from base = instruct - base, paired per conversation. Qwen3 and UltraChat/WildChat "
             "replies: contamination for Qwen cannot be checked. Generated by `analyze.py`.", ""]
    for kind in ["change from base", "level base", "level instruct"]:
        t = s[s.kind == kind]
        if t.empty:
            continue
        lines += [f"## {kind}", ""]
        for fam in t.family.unique():
            tt = t[t.family == fam].pivot(index="quantity", columns="source", values="cell")
            tt = tt.reindex(list(dict.fromkeys(t[t.family == fam].quantity)))[[c for c in SOURCES if c in tt.columns]]
            lines += [f"### {FAMILIES[fam]}", "", tt.reset_index().to_markdown(index=False), ""]
    with open(os.path.join(RES, "results.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
