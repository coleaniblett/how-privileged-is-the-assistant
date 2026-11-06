"""Style-fall checks (2026-11-04; plan and readings logged in docs/decisions.md before any figure; exploratory, outside every verdict).

The leakage-matrix register score falls after post-training for characters, narration and user turns. Three checks, CPU only, no new generation:
  A  words:     spec A human-vs-Assistant classifier P(assistant) and the spec F layout regex on every saved register reply,
                stage minus base beside the register change, plus the within-stage per-reply correlation of coord with the classifier;
  B  slippage:  flag replies with a mid-turn switch (turn/handoff markers and line-initial role labels, user_traits.cut_user_portion),
                flagged share, and the register change recomputed without flagged replies;
  C  training:  OLMo's Dolci SFT / DPO / RL data: share of examples whose prompt asks for role-play or a persona (regex below), a sample of 30,
                and classifier / layout rates on the replies of matched examples vs the rest.
    python -m extras.style_fall_checks ab       # A and B  -> results/overgeneralization/style_fall/style_fall_ab.csv
    python -m extras.style_fall_checks c        # C        -> results/overgeneralization/style_fall/dolci_roleplay.csv, dolci_sample.jsonl
    python -m extras.style_fall_checks report   # style_fall.md from the two csv files
Reads results/overgeneralization/dominance_matrix/gens/<family>/<stage>.jsonl (register rows; read only). Seed 0, 10,000 bootstrap resamples over items.
"""
import argparse
import glob
import json
import os
import random
import re

import numpy as np
import pandas as pd

from core.stats import bootstrap_ci

GENS = os.environ.get("GENS_DIR", "results/overgeneralization/dominance_matrix/gens")
OUT = "results/overgeneralization/style_fall"
N_BOOT, SEED = 10000, 0
FAMILY_CONFIG = {"olmo": "olmo3-7b", "qwen": "qwen3-8b", "gemma": "gemma4-31b"}  # gens folder -> config family (markers; Gemma E4B shares the template family)
CTX_ORDER = ["own", "named", "voiced", "user", "narration", "raw"]
FOCUS = ("voiced", "user")  # the pre-set reading: strong character and user turn, OLMo instruct
N_C_EVAL, N_C_SHOW = 3000, 30

ROLEPLAY = re.compile(
    r"role[- ]?play(?:ing)?|pretend (?:to be|you're|you are)|act (?:as|like) (?:a|an|the|my|if)|play (?:the (?:role|part) of|(?:a|an|the) (?:role|character))"
    r"|you are now|from now on,? you (?:are|will)|imagine you(?:'re| are) (?:a|an|the)|take on the (?:role|persona)|in the role of"
    r"|(?:answer|respond|speak|write) as (?:a|an|if)|persona|stay in character|in character|character (?:named|called)", re.I)

ROLEPLAY_STRICT = re.compile(r"\b(?:" + ROLEPLAY.pattern.replace("persona", r"personas?").replace("in character", r"in character") + r")\b", re.I)


# ------------------------------------------------------------------ loading and scoring
def load_register(root=GENS):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "*", "*.jsonl"))):
        d = pd.read_json(f, lines=True)
        d = d[d.trait == "register"].copy()
        d["family"] = d["family"].astype(str)
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    return d


def score_words(d, clf):
    from extras.persona_suppression import layout
    d = d.copy()
    d["clf"] = clf.p_assistant(d["text"].tolist())
    d["layout"] = d["text"].map(layout).astype(float)
    return d


def flag_switches(d):
    """Mid-turn switch: a turn/handoff marker or a line-initial role label cuts the reply (user_traits.cut_user_portion, role_cut=True)."""
    from measures.overgeneralization.user_traits import cut_user_portion, family_markers
    mk = {f: family_markers(c) for f, c in FAMILY_CONFIG.items()}
    d = d.copy()
    d["flagged"] = [len(cut_user_portion(t, mk[f], role_cut=True)) < len(t) for t, f in zip(d["text"], d["family"])]
    return d


# ------------------------------------------------------------------ statistics
def item_means(d, col, group=("family", "stage", "ctx", "tmpl")):
    return d.groupby(list(group) + ["item"])[col].mean()


def change(d, col, fam, stage, ctx, tmpl, rng, base="base"):
    """Stage minus base of `col`, paired per item (item means over reps), 95% CI over items. NaN row if too few items."""
    m = item_means(d[(d.family == fam) & (d.ctx == ctx) & (d.tmpl == tmpl)].dropna(subset=[col]), col)
    try:
        b, s = m.loc[(fam, base, ctx, tmpl)], m.loc[(fam, stage, ctx, tmpl)]
    except KeyError:
        return dict(n=0, mean=np.nan, lo=np.nan, hi=np.nan, base_mean=np.nan, stage_mean=np.nan)
    ids = b.index.intersection(s.index)
    if len(ids) < 2:
        return dict(n=len(ids), mean=np.nan, lo=np.nan, hi=np.nan, base_mean=np.nan, stage_mean=np.nan)
    x = (s[ids] - b[ids]).to_numpy(float)
    mm, lo, hi = bootstrap_ci(x, N_BOOT, rng)
    return dict(n=len(ids), mean=mm, lo=lo, hi=hi, base_mean=float(b[ids].mean()), stage_mean=float(s[ids].mean()))


def pearson_ci(g, x, y, rng, n_boot=2000):
    """Pooled per-reply Pearson r of x and y; CI by resampling items (sufficient statistics per item)."""
    g = g.dropna(subset=[x, y])
    if len(g) < 10:
        return np.nan, np.nan, np.nan, len(g)
    st = g.assign(xx=g[x] ** 2, yy=g[y] ** 2, xy=g[x] * g[y], n=1.0).groupby("item")[[x, y, "xx", "yy", "xy", "n"]].sum().to_numpy()
    def r(a):
        n, sx, sy, sxx, syy, sxy = a[:, 5].sum(), a[:, 0].sum(), a[:, 1].sum(), a[:, 2].sum(), a[:, 3].sum(), a[:, 4].sum()
        den = np.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2))
        return (n * sxy - sx * sy) / den if den > 0 else np.nan
    r0 = r(st)
    bs = np.array([r(st[rng.integers(0, len(st), len(st))]) for _ in range(n_boot)])
    bs = bs[~np.isnan(bs)]
    return r0, *(np.percentile(bs, [2.5, 97.5]) if len(bs) else (np.nan, np.nan)), len(g)


def run_ab():
    from measures.overgeneralization.human_vs_assistant import load_classifier
    d = flag_switches(score_words(load_register(), load_classifier()))
    rng = np.random.default_rng(SEED)
    rows, flagged, corr = [], [], []
    for fam in sorted(d.family.unique()):
        stages = [s for s in ("base", "sft", "dpo", "instruct") if ((d.family == fam) & (d.stage == s)).any()]
        for ctx in CTX_ORDER:
            for tmpl in (1, 2):
                for stage in stages:
                    sub = d[(d.family == fam) & (d.stage == stage) & (d.ctx == ctx) & (d.tmpl == tmpl)]
                    if sub.empty:
                        continue
                    flagged.append(dict(family=fam, stage=stage, ctx=ctx, tmpl=tmpl, n=len(sub), flagged_share=float(sub.flagged.mean())))
                    r, lo, hi, n = pearson_ci(sub, "coord", "clf", np.random.default_rng(SEED))
                    corr.append(dict(family=fam, stage=stage, ctx=ctx, tmpl=tmpl, r=r, lo=lo, hi=hi, n=n))
                    if stage == "base":
                        continue
                    row = dict(family=fam, stage=stage, ctx=ctx, tmpl=tmpl)
                    cols = ["coord", "clf", "layout"] + (["coord_v2"] if fam == "olmo" else [])
                    for c in cols:
                        ch = change(d, c, fam, stage, ctx, tmpl, np.random.default_rng(SEED))
                        row.update({f"{c}_{k}": v for k, v in ch.items()})
                    # B: register change without flagged replies (flagged dropped in every stage)
                    keep = d[~d.flagged]
                    for c in [c for c in cols if c.startswith("coord")]:
                        ch = change(keep, c, fam, stage, ctx, tmpl, np.random.default_rng(SEED))
                        row.update({f"{c}_excl_{k}": v for k, v in ch.items()})
                        a, b = row[f"{c}_mean"], row[f"{c}_excl_mean"]
                        row[f"{c}_shrink"] = (1 - b / a) if (a == a and b == b and abs(a) > 1e-9) else np.nan
                    rows.append(row)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "style_fall_ab.csv"), index=False)
    pd.DataFrame(flagged).to_csv(os.path.join(OUT, "style_fall_flagged.csv"), index=False)
    pd.DataFrame(corr).to_csv(os.path.join(OUT, "style_fall_corr.csv"), index=False)
    print(f"A/B done: {len(rows)} change rows")


# ------------------------------------------------------------------ C: Dolci
def dolci_examples(name):
    """Stream one Dolci set: yields (user_text_for_regex, reply_or_None, source). SFT: all user/system turns, first assistant reply;
    DPO: the user turns of `chosen`, its last assistant message; RL: the prompt string (prompts only, no reply)."""
    from datasets import load_dataset
    from core.check_contamination import DOLCI, rl_turns
    ds = load_dataset(DOLCI[name], split="train", streaming=True)
    for r in ds:
        if name == "rl":
            turns = rl_turns(r.get("prompt") or "")
            yield "\n".join(t for ro, t in turns if ro == "user"), None, str(r.get("dataset"))
            continue
        msgs = r["messages"] if name == "sft" else r["chosen"]
        prompt = "\n".join(m["content"] for m in msgs if m["role"] in ("user", "system") and m.get("content"))
        asst = [m["content"] for m in msgs if m["role"] == "assistant" and m.get("content")]
        reply = (asst[0] if name == "sft" else asst[-1]) if asst else None
        yield prompt, reply, str(r.get("source_dataset") or r.get("chosen_model") or "")


VARIANTS = {"as logged": ROLEPLAY, "word-bounded (added after seeing the sample)": ROLEPLAY_STRICT}


def _reservoir(res, item, cap, cnt, rng):
    if len(res) < cap:
        res.append(item)
    elif rng.random() < cap / cnt:
        res[rng.randrange(len(res))] = item


def run_c(limit=None):
    """One streaming pass per set, both regex variants (the logged one and a word-bounded one added after the sample showed 'personal' matches)."""
    from extras.persona_suppression import layout
    from measures.overgeneralization.human_vs_assistant import load_classifier
    rng = random.Random(SEED)
    table, sample_rows = [], []
    for name in ("sft", "dpo", "rl"):
        st = {v: dict(n=0, hits=0, hit=[], rest=[], shown=[]) for v in VARIANTS}
        n = 0
        try:
            for i, (prompt, reply, src) in enumerate(dolci_examples(name)):
                if limit and i >= limit:
                    break
                n += 1
                words = " ".join(reply.split()[:40]) if reply else None
                for v, rx in VARIANTS.items():
                    s_, m = st[v], rx.search(prompt or "")
                    if m:
                        s_["hits"] += 1
                        _reservoir(s_["shown"], dict(set=name, variant=v, source=src, phrase=m.group(0), prompt=(prompt or "")[:300], reply=(reply or "")[:200]), N_C_SHOW * 4, s_["hits"], rng)
                    if words is not None:
                        if m:
                            _reservoir(s_["hit"], words, N_C_EVAL, s_["hits"], rng)
                        else:
                            _reservoir(s_["rest"], words, N_C_EVAL, n - s_["hits"], rng)
        except Exception as e:  # noqa: BLE001  set cannot be identified or loaded: say so
            table.append(dict(set=name, error=f"{type(e).__name__}: {str(e)[:200]}"))
            continue
        print(name, n, {v: s_["hits"] for v, s_ in st.items()}, flush=True)
        st["_n"] = n
        yield_ = st
        table.append(("ok", name, n, yield_))
    clf = load_classifier()
    out = []
    for t in table:
        if isinstance(t, dict):
            out.append(t)
            continue
        _, name, n, st = t
        for v in VARIANTS:
            s_ = st[v]
            row = dict(set=name, variant=v, n_examples=n, n_roleplay=s_["hits"], share=s_["hits"] / n if n else np.nan)
            for tag, res in (("roleplay", s_["hit"]), ("rest", s_["rest"])):
                if res:
                    row.update({f"{tag}_n_replies": len(res), f"{tag}_clf": float(clf.p_assistant(res).mean()), f"{tag}_layout": float(np.mean([layout(x) for x in res]))})
            out.append(row)
            sample_rows += rng.sample(s_["shown"], min(N_C_SHOW, len(s_["shown"])))
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(out).to_csv(os.path.join(OUT, "dolci_roleplay.csv"), index=False)
    with open(os.path.join(OUT, "dolci_sample.jsonl"), "w", encoding="utf-8") as f:
        for r in sample_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ report
def fmt(m, lo, hi, d=2):
    return "n/a" if pd.isna(m) else f"{m:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def report():
    ab = pd.read_csv(os.path.join(OUT, "style_fall_ab.csv"))
    fl = pd.read_csv(os.path.join(OUT, "style_fall_flagged.csv"))
    co = pd.read_csv(os.path.join(OUT, "style_fall_corr.csv"))
    L = ["# Style-fall checks (exploratory, outside every verdict)", "",
         "Plan and pre-set readings: `docs/decisions.md` 2026-11-04. Saved register replies of the leakage matrix (`results/overgeneralization/dominance_matrix/gens`, read only). "
         "Changes are stage minus base, paired per item (replies averaged within item), 95% bootstrap CI over items. `coord` is the register score (higher = more Assistant-like), "
         "`clf` the spec A P(assistant), `layout` the spec F layout regex rate. **The classifier was trained on human vs Assistant text, not on characters**: it measures how Assistant-like the wording is, nothing more.", ""]
    # A reading
    L += ["## A. Is the fall in the words?", "",
          "Pre-set reading: yes if, for strong character (`voiced`) and user turn in OLMo instruct, the classifier change has the same sign as the register change with the CI excluding 0 (both templates).", "",
          "| family | stage | context | tmpl | register change | classifier change | layout change | clf same sign, CI excl. 0 |", "|---|---|---|---|---|---|---|---|"]
    for _, r in ab.iterrows():
        same = (not pd.isna(r.clf_mean)) and np.sign(r.clf_mean) == np.sign(r.coord_mean) and (r.clf_lo > 0 or r.clf_hi < 0)
        L.append(f"| {r.family} | {r.stage} | {r.ctx} | {r.tmpl} | {fmt(r.coord_mean, r.coord_lo, r.coord_hi)} | {fmt(r.clf_mean, r.clf_lo, r.clf_hi, 3)} | {fmt(r.layout_mean, r.layout_lo, r.layout_hi, 3)} | {'yes' if same else 'no'} |")
    o = ab[(ab.family == "olmo") & (ab.stage == "instruct") & ab.ctx.isin(FOCUS)]
    verdict = all(((np.sign(r.clf_mean) == np.sign(r.coord_mean)) and (r.clf_lo > 0 or r.clf_hi < 0)) for _, r in o.iterrows()) if len(o) else False
    L += ["", f"**Reading A (OLMo instruct, voiced and user, both templates): {'the fall is in the words' if verdict else 'not established by the classifier'}.**", ""]
    L += ["Within-stage per-reply Pearson r of `coord` with the classifier score (CI over items), OLMo and others:", "", "| family | stage | context | tmpl | r [95% CI] | replies |", "|---|---|---|---|---|---|"]
    for _, r in co.iterrows():
        L.append(f"| {r.family} | {r.stage} | {r.ctx} | {r.tmpl} | {fmt(r.r, r.lo, r.hi)} | {int(r.n)} |")
    L += ["", "## B. Is it slippage (base models switching to the Assistant)?", "", "Flagged share (replies cut by a turn/handoff marker or a line-initial role label):", "",
          "| family | stage | context | tmpl | flagged share | replies |", "|---|---|---|---|---|---|"]
    for _, r in fl.iterrows():
        L.append(f"| {r.family} | {r.stage} | {r.ctx} | {r.tmpl} | {100 * r.flagged_share:.1f}% | {int(r.n)} |")
    L += ["", "Register change with and without flagged replies (flagged dropped in every stage). Pre-set reading: slippage explains the fall in a context if exclusion shrinks the change by at least 50%.", "",
          "| family | stage | context | tmpl | change, all | change, flagged excluded | shrink | explains (>= 50%) |", "|---|---|---|---|---|---|---|---|"]
    for _, r in ab.iterrows():
        sh = r.coord_shrink
        L.append(f"| {r.family} | {r.stage} | {r.ctx} | {r.tmpl} | {fmt(r.coord_mean, r.coord_lo, r.coord_hi)} | {fmt(r.coord_excl_mean, r.coord_excl_lo, r.coord_excl_hi)} | "
                 f"{'n/a' if pd.isna(sh) else f'{100 * sh:.0f}%'} | {'yes' if (not pd.isna(sh)) and sh >= 0.5 else 'no'} |")
    p = os.path.join(OUT, "dolci_roleplay.csv")
    if os.path.exists(p):
        c = pd.read_csv(p)
        L += ["", "## C. OLMo training data: role-play / persona prompts in Dolci (descriptive)", "",
              "Regex over the user (and system) turns of each example, case-insensitive. **As logged**: `" + ROLEPLAY.pattern + "`. This also matches 'personal', 'personalized' and other words containing 'persona' "
              "(seen in the 30-example sample); a **word-bounded variant** (`\\b(...)\\b`, 'personas?') was therefore added after seeing the sample and is reported beside it. Classifier and layout on the first 40 words of the reply "
              f"(SFT assistant reply, DPO chosen reply; reservoir samples of up to {N_C_EVAL} replies per group; the RL set stores prompts only).", "",
              "| set | regex | examples | role-play/persona prompts | share | replies (role-play / rest) | classifier P(assistant) role-play vs rest | layout rate role-play vs rest |", "|---|---|---|---|---|---|---|---|"]
        for _, r in c.iterrows():
            if isinstance(r.get("error"), str):
                L.append(f"| {r.set} | | could not be loaded: {r.error} | | | | | |")
                continue
            ok = not pd.isna(r.get("roleplay_clf"))
            L.append(f"| {r.set} | {r.variant} | {int(r.n_examples)} | {int(r.n_roleplay)} | {100 * r.share:.2f}% | " +
                     (f"{int(r.roleplay_n_replies)} / {int(r.rest_n_replies)} | {r.roleplay_clf:.3f} vs {r.rest_clf:.3f} | {r.roleplay_layout:.3f} vs {r.rest_layout:.3f} |" if ok else "n/a (no replies stored) | n/a | n/a |"))
        sp = os.path.join(OUT, "dolci_sample.jsonl")
        if os.path.exists(sp):
            rows = [json.loads(x) for x in open(sp, encoding="utf-8")]
            for v in VARIANTS:
                L += ["", f"### {N_C_SHOW} random matches per set, regex {v} (prompt truncated to 300 characters, reply to 200)", ""]
                for i, r in enumerate([x for x in rows if x["variant"] == v], 1):
                    L.append(f"{i}. [{r['set']} / {r['source']}] match `{r['phrase']}`: {' '.join(r['prompt'].split())}" + (f" -> {' '.join(r['reply'].split())}" if r["reply"] else ""))
    open(os.path.join(OUT, "style_fall.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:12]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["ab", "c", "report"])
    ap.add_argument("--limit", type=int, default=0, help="c: examples per set (tests)")
    a = ap.parse_args()
    {"ab": run_ab, "c": lambda: run_c(a.limit or None), "report": report}[a.cmd]()


if __name__ == "__main__":
    main()
