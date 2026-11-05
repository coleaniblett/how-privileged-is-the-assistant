"""Spec J6: a second local judge from a different family (Qwen3-8B instruct, non-thinking) re-judges, blind, (1) the in-character question on a stratified 1,000 atlas replies and
(2) 1,000 stratified atlas trait labels, with the frozen prompts and the same P(Yes) >= 0.5 rule as the Gemma 4 31B judge. Agreement is not accuracy. Local only; prefill-judge labels are not used.
    python -m extras.second_judge run        # GPU (about 16 GB): writes results/second_judge/{incharacter,traits}.csv.gz
    python -m extras.second_judge analyze    # -> results/second_judge/second_judge.md (+ csv)
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

from extras import trait_atlas as ta
from extras import trait_atlas_incharacter as ic

OUT = "results/second_judge"
SEED, N = 0, 1000
JUDGE2 = "Qwen/Qwen3-8B"
KEY = ["trait", "item", "context", "wording", "persona", "sample"]


def strat(df, cells, balance, n, rng):
    """About n rows: equal allocation over `cells` groups, within a group half and half on `balance` where both values exist."""
    groups = [g for _, g in df.groupby(cells)]
    per = n // len(groups)
    extra = n - per * len(groups)
    take = []
    for gi, g in enumerate(groups):
        k = per + (1 if gi < extra else 0)
        vals = [x for _, x in g.groupby(balance)]
        if len(vals) == 2:
            half = [min(len(v), k // 2) for v in vals]
            rest = k - sum(half)
            sel = [v.sample(h, random_state=int(rng.integers(1 << 30))) for v, h in zip(vals, half)]
            pool = pd.concat([v.drop(s.index) for v, s in zip(vals, sel)])
            sel.append(pool.sample(min(rest, len(pool)), random_state=int(rng.integers(1 << 30))))
            take.append(pd.concat(sel))
        else:
            take.append(g.sample(min(k, len(g)), random_state=int(rng.integers(1 << 30))))
    return pd.concat(take)


def replies_lookup():
    out = {}
    for p in sorted(glob.glob(os.path.join(ta.OUT, "*", "*_replies.jsonl.gz"))):
        fam, stage = p.split(os.sep)[-2], os.path.basename(p).split("_")[0]
        for r in ta.read_gz(p):
            out[(fam, stage) + tuple(r[k] for k in KEY)] = r
    return out


def build_samples(mat, rng):
    lk = replies_lookup()
    idx = ta.item_index(mat)
    ics = []
    for p in sorted(glob.glob(os.path.join(ic.OUT, "atlas_*.csv.gz"))):
        fam, stage = os.path.basename(p)[6:-7].rsplit("_", 1)
        ics.append(pd.read_csv(p).assign(family=fam, stage=stage))
    a = pd.concat(ics, ignore_index=True)
    a["cell"] = a.family + "/" + a.stage + "/" + a.context
    sa = strat(a, ["cell"], "value", N, rng)
    trs = []
    for p in sorted(glob.glob(os.path.join(ta.OUT, "*", "*_judged.csv.gz"))):
        fam, stage = p.split(os.sep)[-2], os.path.basename(p).split("_")[0]
        trs.append(pd.read_csv(p).assign(family=fam, stage=stage))
    t = pd.concat(trs, ignore_index=True)
    t["cell"] = t.trait + "/" + t.context
    st = strat(t, ["cell"], "value", N, rng)
    pi, pt = [], []
    for r in sa.itertuples():
        rep = lk[(r.family, r.stage) + tuple(getattr(r, k) for k in KEY)]
        pi.append(ic.prompt_for(mat, rep["question"], rep["text"], ic.NOTE_LIGHT.format(role=mat["light"][r.persona])))
    for r in st.itertuples():
        rep = lk[(r.family, r.stage) + tuple(getattr(r, k) for k in KEY)]
        pt.append(ta.judge_prompt(mat, r.trait, idx[r.item][2], rep["text"]))
    return sa.reset_index(drop=True), pi, st.reset_index(drop=True), pt


def run():
    import torch
    from core.render import encode, load_tokenizer
    from core.score import load_model
    mat = ta.load_materials()
    rng = np.random.default_rng(SEED)
    sa, pi, st, pt = build_samples(mat, rng)
    tok = load_tokenizer(JUDGE2)
    tok.padding_side = "left"
    yes, no = encode(tok, "Yes")[0], encode(tok, "No")[0]
    model, device = load_model(JUDGE2, torch.bfloat16)

    def judge(prompts, label, bs=16):
        full = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in prompts]
        order = sorted(range(len(full)), key=lambda k: len(full[k]))
        py = np.full(len(full), np.nan)
        t0 = time.time()
        for s in range(0, len(order), bs):
            b = order[s:s + bs]
            enc = tok([full[k] for k in b], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
            py[b] = torch.stack([lg[:, yes], lg[:, no]], 1).softmax(-1)[:, 0].cpu().numpy()
            if (s // bs) % 20 == 0:
                print(f"{label}: {s}/{len(order)} ({time.time() - t0:.0f} s)", flush=True)
        return py
    os.makedirs(OUT, exist_ok=True)
    for name, df, pr in (("incharacter", sa, pi), ("traits", st, pt)):
        py = judge(pr, name)
        df = df.assign(value1=df["value"], p_yes1=df["p_yes"], p_yes2=py, value2=(py >= 0.5).astype(float)).drop(columns=["value", "p_yes"])
        df.to_csv(os.path.join(OUT, f"{name}.csv.gz"), index=False)
        print(name, len(df), "agreement", float((df.value1 == df.value2).mean()), flush=True)


def kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return np.nan if pe == 1 else (po - pe) / (1 - pe)


def table(df, by):
    rows = []
    for k, g in df.groupby(by):
        rows.append(dict(group=" / ".join(map(str, k)) if isinstance(k, tuple) else k, n=len(g), judge1_yes=g.value1.mean(), judge2_yes=g.value2.mean(), agreement=float((g.value1 == g.value2).mean()), kappa=kappa(g.value1, g.value2)))
    return pd.DataFrame(rows)


def analyze():
    L = ["# Second local judge (spec J6; exploratory, outside every verdict)", "",
         f"Judge 2 = {JUDGE2} (non-thinking), the same frozen prompts and rule (P(Yes) >= 0.5) as the Gemma 4 31B judge, blind. **Agreement is not accuracy**: two judges can agree and both be wrong, "
         "and a low kappa is partly a base-rate effect when almost every reply has the same label. A group with kappa under 0.4 is flagged unreliable (kappa is undefined and marked n/a when one label is absent). "
         "Samples: 1,000 stratified atlas replies for the in-character question (equal allocation over family/stage/context, half and half on the Gemma label where possible) and 1,000 stratified atlas trait labels (equal over trait/context).", ""]
    allt = []
    for name, by_list in (("incharacter", [["family", "stage"], ["context"], ["family", "stage", "context"]]), ("traits", [["trait"], ["context"], ["family"]])):
        p = os.path.join(OUT, f"{name}.csv.gz")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        L += [f"## {name} ({len(d)} replies): overall agreement {float((d.value1 == d.value2).mean()):.3f}, kappa {kappa(d.value1, d.value2):.2f}", ""]
        for by in by_list:
            t = table(d, by)
            t["flag"] = np.where(t.kappa.isna(), "n/a (one label)", np.where(t.kappa < 0.4, "UNRELIABLE (kappa < 0.4)", ""))
            allt.append(t.assign(sample=name, by="+".join(by)))
            L += [f"### by {' + '.join(by)}", "", "| group | n | Gemma-judge yes | Qwen-judge yes | agreement | kappa | flag |", "|---|---|---|---|---|---|---|"]
            L += [f"| {r.group} | {r.n} | {r.judge1_yes:.2f} | {r.judge2_yes:.2f} | {r.agreement:.2f} | {'n/a' if np.isnan(r.kappa) else f'{r.kappa:.2f}'} | {r.flag} |" for r in t.itertuples()]
            L.append("")
    if allt:
        allt_df = pd.concat(allt)
        allt_df.to_csv(os.path.join(OUT, "second_judge.csv"), index=False)
        L += reliability(allt_df)
    open(os.path.join(OUT, "second_judge.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:20]))


def reliability(allt):
    """Judge-reliability table (added 2026-11-05 after reading the first report): the traits whose agreement with the second judge is under kappa 0.4, and what the atlas says for them.
    The atlas cells are NOT recomputed with the second judge's labels: the 1,000-label sample is about 85 labels per trait, stratified half and half on the first judge's label (so its
    yes-rate is about 0.5 by design), and cannot replace the full cells. trait_atlas.md is unchanged."""
    t = allt[(allt["sample"] == "traits") & (allt.by == "trait")]
    bad = t[(t.kappa < 0.4)]
    L = ["## Judge reliability (traits flagged unreliable by the second judge; atlas cells not recomputed)", "",
         "The atlas cells below are the Gemma-judge cells from `results/trait_atlas/trait_atlas_cells.csv`, unchanged. The sample is stratified on the Gemma label, so the Gemma-judge yes-rate is about 0.5 by design; "
         "the Qwen judge says yes much less often for these traits (it may simply be stricter), so a low kappa shows the two judges draw the line differently, not which one is right. "
         "Read these traits' judge-based atlas results as provisional until the hand check.", "",
         "| trait | n | Gemma-judge yes (sample) | Qwen-judge yes | agreement | kappa | atlas: Assistant change at final, per family (OLMo instruct / Qwen / Gemma), points [95% CI] |", "|---|---|---|---|---|---|---|"]
    cells = os.path.join(ta.OUT.rsplit(os.sep, 1)[0] if False else "results/trait_atlas", "trait_atlas_cells.csv")
    c = pd.read_csv(cells) if os.path.exists(cells) else None
    for r in bad.itertuples():
        txt = "n/a"
        if c is not None:
            parts = []
            for fam in ("olmo3-7b", "qwen3-8b", "gemma4-31b"):
                x = c[(c.trait == r.group) & (c.family == fam) & (c.stage == "instruct") & (c.context == "assistant") & (~c.flagged_excluded)]
                parts.append(f"{x.iloc[0].change:+.1f} [{x.iloc[0].lo:+.1f}, {x.iloc[0].hi:+.1f}]" if len(x) else "n/a")
            txt = " / ".join(parts)
        L.append(f"| {r.group} | {r.n} | {r.judge1_yes:.2f} | {r.judge2_yes:.2f} | {r.agreement:.2f} | {r.kappa:.2f} | {txt} |")
    bc = allt[(allt["sample"] == "traits") & (allt.by == "context") & (allt.kappa < 0.4)]
    if len(bc):
        L += ["", "Contexts under kappa 0.4 on the trait sample: " + ", ".join(f"{r.group} ({r.kappa:.2f})" for r in bc.itertuples()) + "."]
    return L + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    a = ap.parse_args()
    run() if a.cmd == "run" else analyze()


if __name__ == "__main__":
    main()
