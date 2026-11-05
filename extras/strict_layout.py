"""Spec R1 (2026-11-06): does the formatting claim survive a strict definition? CPU only, no new generation, no judge. Exploratory, outside every verdict, NOT BLIND.
Old = the atlas layout regex (a list line, a heading or ONE bold phrase). Strict = at least 2 list lines, or a markdown heading, or a table (bold alone does not count). Heavy = at least 4 list lines or at least 2 headings.
Rescored from saved replies: atlas (OLMo base/SFT/DPO/final, Qwen, Gemma; /tmp/atlas_pull copies of vinyl's replies), atlas breadth (Llama, Apertus), L1, L4, Q (judged files). Base vs final readings for the atlas markdown set and L4 separately.
    python -m extras.strict_layout            # -> results/overnight_2026-11-05/R1/
"""
import glob
import gzip
import json
import os
import re

import numpy as np
import pandas as pd

from extras import trait_atlas as ta

OUT = "results/overnight_2026-11-05/R1"
ATLAS_DIRS = ["/tmp/atlas_pull", "results/trait_atlas"]
SEED, N_BOOT = 0, 4000
LIST_RE = re.compile(r"(?m)^[ \t]*(?:[-*•]|\d+[.)])[ \t]+\S")
HEAD_RE = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+\S")
TABLE_RE = re.compile(r"(?m)^[ \t]*\|.*\|[ \t]*\n[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(?:\|[ \t]*:?-{3,}:?[ \t]*)*\|?[ \t]*$")
OLD_RE = re.compile(r"(^|\n)\s*([-*•]|\d+[.)]|#{1,4})\s|\*\*[^*\n]+\*\*")


def score(text):
    t = text if isinstance(text, str) else ""
    n_list, n_head = len(LIST_RE.findall(t)), len(HEAD_RE.findall(t))
    old = bool(OLD_RE.search(t))
    strict = n_list >= 2 or n_head >= 1 or bool(TABLE_RE.search(t))
    heavy = n_list >= 4 or n_head >= 2
    bold_only = old and n_list == 0 and n_head == 0 and not TABLE_RE.search(t)
    return old, strict, heavy, bold_only


def read_gz(p):
    return [json.loads(l) for l in gzip.open(p, "rt", encoding="utf-8")]


def load_all():
    frames = []
    seen = set()
    for base in ATLAS_DIRS:
        for p in sorted(glob.glob(os.path.join(base, "*", "*_replies.jsonl.gz"))):
            fam, stage = p.split(os.sep)[-2], os.path.basename(p).split("_")[0]
            if (fam, stage) in seen:
                continue
            seen.add((fam, stage))
            d = pd.DataFrame(read_gz(p))
            d = d[d.trait == "markdown_lists"]
            frames.append(pd.DataFrame(dict(source="atlas", family=fam, stage=stage, context=d.context, role=d.persona, item=d["item"], sample=d["sample"], wording=d.get("wording", 0), text=d.text)))
    for p in sorted(glob.glob("results/trait_atlas_breadth/*/*_replies.jsonl.gz")):
        fam, stage = p.split(os.sep)[-2], os.path.basename(p).split("_")[0]
        d = pd.DataFrame(read_gz(p))
        d = d[d.trait == "markdown_lists"]
        frames.append(pd.DataFrame(dict(source="atlas", family=fam, stage=stage, context=d.context, role=d.persona, item=d["item"], sample=d["sample"], wording=d.get("wording", 0), text=d.text)))
    for p in sorted(glob.glob("results/working_model_L/L1/*/*.csv.gz")):
        fam, stage = p.split(os.sep)[-2], os.path.basename(p).split(".")[0]
        d = pd.read_csv(p)
        frames.append(pd.DataFrame(dict(source="L1:" + d.version, family=fam, stage=stage, context=d.setting, role=d.role, item=d["item"], sample=d["sample"], wording=0, text=d.text)))
    for p in sorted(glob.glob("results/working_model_L/L4/*/*.csv.gz")):
        fam, stage = p.split(os.sep)[-2], os.path.basename(p).split(".")[0]
        d = pd.read_csv(p)
        frames.append(pd.DataFrame(dict(source="L4:" + d.qtype, family=fam, stage=stage, context=d.setting, role=d.role, item=d["item"], sample=d["sample"], wording=0, text=d.text)))
    for p in sorted(glob.glob("results/role_says/*/*_judged.csv.gz")):
        fam, stage = p.split(os.sep)[-2], os.path.basename(p).split("_")[0]
        d = pd.read_csv(p)
        ctx = np.where(d.cond == "assistant", "assistant", d.position + "/" + d.identity + "/" + d.voice)
        frames.append(pd.DataFrame(dict(source="Q", family=fam, stage=stage, context=ctx, role=d.job, item=d.qi, sample=d["sample"], wording=0, text=d.text)))
    D = pd.concat(frames, ignore_index=True)
    sc = np.array([score(t) for t in D.text], dtype=bool)
    D["old"], D["strict"], D["heavy"], D["bold_only"] = sc[:, 0], sc[:, 1], sc[:, 2], sc[:, 3]
    return D


def final_of(fam):
    return "instruct"


def paired(D, source, fam, ctx, col, rng):
    """final minus base, points, paired per (item, role) after averaging samples/wordings; two-way bootstrap over items and roles."""
    x = D[(D.source == source) & (D.family == fam) & (D.context == ctx)]
    f, b = x[x.stage == "instruct"], x[x.stage == "base"]
    if f.empty or b.empty:
        return (np.nan,) * 3 + (np.nan, np.nan)
    pv = lambda d: d.pivot_table(index="item", columns="role", values=col, aggfunc="mean")
    A, B = pv(f), pv(b)
    ix, cx = A.index.intersection(B.index), A.columns.intersection(B.columns)
    M = 100.0 * (A.loc[ix, cx] - B.loc[ix, cx]).to_numpy()
    m, lo, hi = ta.two_way_boot(M, rng, n_boot=N_BOOT)
    return m, lo, hi, 100 * f[col].mean(), 100 * b[col].mean()


def reading(chg, lo, hi, level_strict, level_old):
    if chg != chg:
        return "n/a"
    first = chg >= 20 and lo > 0
    if first and level_strict >= 50 and not (level_strict < level_old - 30):
        return "CLAIM HOLDS"
    if first or level_strict < level_old - 30:
        return "CLAIM SHRINKS"
    if chg < 10 and lo >= -20 and hi <= 20:
        return "CLAIM FAILS"
    return "in between"


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    D = load_all()
    D.drop(columns=["text"]).to_csv(os.path.join(OUT, "R1_scored_replies.csv.gz"), index=False)
    f = lambda m: "n/a" if m[0] != m[0] else f"{m[0]:+.1f} [{m[1]:+.1f}, {m[2]:+.1f}]"
    L = ["# R1: does the formatting claim survive a strict definition? (spec R; exploratory, outside every verdict, NOT BLIND)", "",
         "Plan: `docs/decisions.md` 2026-11-06 (spec R, `docs/spec_R_overnight.md`). Saved replies only, no generation, no judge. **Old** = the atlas layout regex (a list line, a heading of up to 4 #, or ONE bold phrase). **Strict** = at least 2 list lines, or a markdown heading, or a table (bold alone does not count). "
         "**Heavy** = at least 4 list lines or at least 2 headings. Change = final minus base in the same context, points, paired per question and role (samples and wordings averaged), 95% two-way bootstrap over questions and roles. "
         "Readings (as the spec words them; interpreted as: the strict change from base, and the strict level of the final model): **claim holds** = strict change at least 20 with range excluding 0 and strict level at least 50%; **claim shrinks** = only the first part, or the strict level more than 30 points below the old level; "
         "**claim fails** = change under 10 with range inside +-20; else in between. Limits: few roles; many cells, no correction for multiple tests; regex sees form only; sources are Gemma/OLMo/Qwen atlas replies (copies of vinyl's files), L1, L4 and Q judged files; Llama and Apertus come from the atlas breadth run; contamination is checkable only for OLMo.", ""]
    # readings: atlas markdown set (light_system, light_user), L4 light_role, final vs base
    heads, rows, rec = [], [], []
    srcs = [("atlas", ["light_system", "light_user"], "atlas markdown set")] + [("L4:" + t, ["light_role"], f"L4 fresh questions ({t})") for t in ("factual", "howto", "advice")]
    fams = sorted(D.family.unique())
    for src, ctxs, label in srcs:
        for fam in fams:
            for ctx in ctxs:
                m = {c: paired(D, src, fam, ctx, c, rng) for c in ("old", "strict", "heavy")}
                if m["strict"][0] != m["strict"][0]:
                    continue
                rd = reading(m["strict"][0], m["strict"][1], m["strict"][2], m["strict"][3], m["old"][3])
                if rd != "CLAIM FAILS" and m["strict"][0] < 10 and m["strict"][1] >= -20 and m["strict"][2] <= 20:
                    rd += " (also meets the fails definition: change under 10, range inside +-20)"
                x = D[(D.source == src) & (D.family == fam) & (D.context == ctx) & (D.stage == "instruct")]
                bo = 100 * x.bold_only.sum() / max(1, x.old.sum())
                rows.append(f"| {label} | {fam} | {ctx} | {m['old'][3]:.0f} / {m['strict'][3]:.0f} / {m['heavy'][3]:.0f} | {m['old'][4]:.0f} / {m['strict'][4]:.0f} / {m['heavy'][4]:.0f} | {f(m['old'])} | {f(m['strict'])} | {f(m['heavy'])} | {bo:.0f}% | **{rd}** |")
                rec.append(dict(source=src, family=fam, context=ctx, final_old=m["old"][3], final_strict=m["strict"][3], final_heavy=m["heavy"][3], base_old=m["old"][4], base_strict=m["strict"][4], base_heavy=m["heavy"][4],
                                chg_old=m["old"][0], chg_strict=m["strict"][0], lo_strict=m["strict"][1], hi_strict=m["strict"][2], chg_heavy=m["heavy"][0], bold_only_share=bo, reading=rd))
    L += ["## Readings (final minus base; levels are old / strict / heavy, % of replies)", "", "| set | family | light-role position | final level old / strict / heavy | base level | change old | change strict | change heavy | old-regex replies that were bold-only (final) | reading |", "|---|---|---|---|---|---|---|---|---|---|"] + rows + [""]
    pd.DataFrame(rec).to_csv(os.path.join(OUT, "R1_readings.csv"), index=False)
    # slide C row: markdown by context, old / strict / heavy side by side, every family, base and final (atlas markdown set)
    L += ["## Slide C's markdown row: old, strict and heavy side by side (atlas markdown set, % of replies; OLMo SFT and DPO included where saved)", "", "| family | stage | context | old | strict | heavy | bold-only share of old |", "|---|---|---|---|---|---|---|"]
    A = D[D.source == "atlas"]
    tab = []
    for (fam, st, ctx), g in A.groupby(["family", "stage", "context"]):
        tab.append(dict(family=fam, stage=st, context=ctx, old=100 * g.old.mean(), strict=100 * g.strict.mean(), heavy=100 * g.heavy.mean(), bold_only_share=100 * g.bold_only.sum() / max(1, g.old.sum()), n=len(g)))
    T = pd.DataFrame(tab)
    order = {"base": 0, "sft": 1, "dpo": 2, "instruct": 3}
    T = T.assign(o=T.stage.map(order)).sort_values(["family", "context", "o"]).drop(columns="o")
    T.to_csv(os.path.join(OUT, "R1_slideC_markdown_row.csv"), index=False)
    for r in T.itertuples():
        L.append(f"| {r.family} | {r.stage} | {r.context} | {r.old:.0f} | {r.strict:.0f} | {r.heavy:.0f} | {r.bold_only_share:.0f}% |")
    # L1, Q descriptive
    L += ["", "## L1 and Q (descriptive, no readings; % of replies)", "", "| source | family | stage | context | old | strict | heavy | bold-only share of old |", "|---|---|---|---|---|---|---|---|"]
    for (src, fam, st, ctx), g in D[D.source.str.startswith("L1") | (D.source == "Q")].groupby(["source", "family", "stage", "context"]):
        if st != "instruct" and src == "Q":
            continue
        L.append(f"| {src} | {fam} | {st} | {ctx} | {100 * g.old.mean():.0f} | {100 * g.strict.mean():.0f} | {100 * g.heavy.mean():.0f} | {100 * g.bold_only.sum() / max(1, g.old.sum()):.0f}% |")
    open(os.path.join(OUT, "R1.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
