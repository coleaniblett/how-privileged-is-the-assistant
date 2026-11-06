"""Spec P1 (2026-11-05): why do some atlas traits spread? Unmix traits from questions. Exploratory, outside every verdict, NOT blind (the idea came from regex levels already seen). No GPU, no paid API,
no judge, no new generation: saved atlas replies only. Plan, frozen ratings (data/trait_atlas_question_ratings.csv) and readings: docs/decisions.md 2026-11-05.
    python -m extras.trait_atlas_unmixed detect          # detectors on every saved reply -> results/overgeneralization/trait_atlas_unmixed/detections.csv.gz (local)
    python -m extras.trait_atlas_unmixed analyze         # task gap, in-character sensitivity, says_ai across question sets, group table -> unmixed.md
    python -m extras.trait_atlas_unmixed features        # descriptive feature scan -> features.md
"""
import argparse
import glob
import gzip
import json
import os
import re

import numpy as np
import pandas as pd

from extras import trait_atlas as ta

OUT = os.environ.get("UM_OUT", "results/overgeneralization/trait_atlas_unmixed")
RATINGS = "data/trait_atlas_question_ratings.csv"
SEED, N_BOOT = 0, 2000
FAMS3 = ["olmo3-7b", "qwen3-8b", "gemma4-31b"]
FAMS5 = FAMS3 + ["llama3.1-8b", "apertus-8b"]
SHARED8 = ["markdown_lists", "says_ai", "stock_phrases", "length", "refusal_borderline", "arithmetic", "multistep_reasoning", "factual_accuracy"]
CONTEXTS = ["assistant", "light_system", "light_user", "strong_act", "strong_noact", "narration", "plain_text"]
DET = {"md": "layout", "stock": "stock", "says_ai": "self_ai", "refusal": "refusal", "logwords": "words"}
SRC = {f: os.environ.get("UM_ATLAS", "/tmp/atlas_ut") for f in FAMS3}


def replies_file(fam, stage):
    if fam in FAMS3:
        return os.path.join(SRC[fam], fam, f"{stage}_replies.jsonl.gz")
    return os.path.join("results/overgeneralization/trait_atlas_breadth", fam, f"{stage}_replies.jsonl.gz")


def detect():
    rows = []
    for fam in FAMS5:
        for stage in ("base", "sft", "dpo", "instruct"):
            p = replies_file(fam, stage)
            if not os.path.exists(p):
                continue
            with gzip.open(p, "rt", encoding="utf-8") as f:
                for l in f:
                    r = json.loads(l)
                    if r["context"] not in CONTEXTS:
                        continue
                    t = r["text"] or ""
                    rows.append(dict(family=fam, stage=stage, trait=r["trait"], item=r["item"], context=r["context"], persona=r["persona"], sample=r["sample"],
                                     **{k: ta.detect_one(r["trait"], d, t, None) for k, d in DET.items()}))
        print(fam, len(rows), flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "detections.csv.gz"), index=False)


def load_ratings():
    r = pd.read_csv(RATINGS)
    return r.set_index(["trait", "item"]).label


def prep(df, rat):
    df = df.copy()
    df["qtype"] = [rat.get((t, i)) for t, i in zip(df.trait, df.item)]
    return df[df.qtype.isin(["task", "conversation"])]          # 'about_speaker' excluded


def gap_boot(d, ctx, det, rng, n_boot=N_BOOT, incharacter=None):
    """task gap = (light task - light conversation) - (Assistant task - Assistant conversation), two-way bootstrap over questions (within type) and personas."""
    a = d[d.context == "assistant"]
    l = d[d.context == ctx]
    if incharacter is not None:
        l = l.merge(incharacter, on=["family", "stage", "trait", "item", "context", "persona", "sample"], how="inner")
        l = l[l.ic == 1]
    out = {}
    for qt in ("task", "conversation"):
        qa = a[a.qtype == qt].groupby(["trait", "item"])[det].mean()
        qs = l[l.qtype == qt].pivot_table(index=["trait", "item"], columns="persona", values=det, aggfunc="mean")
        out[qt] = (qa.reindex(qs.index), qs)
    nt, nc = len(out["task"][1]), len(out["conversation"][1])
    if nt < 3 or nc < 3:
        return dict(n_task=nt, n_conv=nc)
    def stat(qi_t, qi_c, ri):
        At, Lt = out["task"][0].to_numpy()[qi_t], out["task"][1].to_numpy()[qi_t][:, ri]
        Ac, Lc = out["conversation"][0].to_numpy()[qi_c], out["conversation"][1].to_numpy()[qi_c][:, ri]
        return (np.nanmean(Lt) - np.nanmean(Lc)) - (np.nanmean(At) - np.nanmean(Ac)), np.nanmean(At), np.nanmean(Ac), np.nanmean(Lt), np.nanmean(Lc)
    ri_all = np.arange(out["task"][1].shape[1])
    g0 = stat(np.arange(nt), np.arange(nc), ri_all)
    bs = []
    for _ in range(n_boot):
        bs.append(stat(rng.integers(0, nt, nt), rng.integers(0, nc, nc), rng.integers(0, len(ri_all), len(ri_all)))[0])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(n_task=nt, n_conv=nc, gap=100 * g0[0], lo=100 * lo, hi=100 * hi, a_task=100 * g0[1], a_conv=100 * g0[2], l_task=100 * g0[3], l_conv=100 * g0[4])


def reading(g):
    if "gap" not in g:
        return "too few questions of one type (task %d, conversation %d)" % (g["n_task"], g["n_conv"])
    if g["a_task"] >= 95 and g["a_conv"] >= 95 and g["l_task"] >= 95:
        return "CEILING: pattern" if g["l_conv"] <= g["l_task"] - 20 else "CEILING: can't test"
    if g["gap"] >= 20 and g["lo"] > 0:
        return "TASK DEMAND HOLDS"
    if g["gap"] < 10 and g["lo"] >= -20 and g["hi"] <= 20:
        return "NO TASK DEMAND"
    return "inconclusive"


def fmt(g):
    if "gap" not in g:
        return "n/a"
    return f"{g['gap']:+.1f} [{g['lo']:+.1f}, {g['hi']:+.1f}]"


def analyze():
    rng = np.random.default_rng(SEED)
    rat = load_ratings()
    D = pd.read_csv(os.path.join(OUT, "detections.csv.gz"))
    D = prep(D, rat)
    L = ["# Unmixing traits from questions (spec P1; exploratory, outside every verdict, NOT BLIND: the idea came from regex levels on the saved replies already seen for OLMo, Qwen and Gemma 31B; Llama and Apertus had not been looked at this way)", "",
         "Plan: `docs/decisions.md` 2026-11-05. Detectors applied to every saved reply regardless of its question set; each of the 240 atlas questions rated task / conversation / about the speaker by two blind subagents (third subagent for disagreements), frozen in `data/trait_atlas_question_ratings.csv`; "
         "questions rated 'about the speaker' are excluded. Task gap = (light role on task minus light role on conversation) minus (Assistant on task minus Assistant on conversation), points; 95% two-way bootstrap over questions (within type) and the six personas. "
         "Regex measures see form only (one bold phrase counts as markdown); 12 questions per set; subagent ratings are not human ratings; few roles; many cells; no correction for multiple tests; contamination cannot be checked outside OLMo; not hand-checked.", ""]
    cnt = prep(pd.read_csv(os.path.join(OUT, "detections.csv.gz")).query("context == 'assistant' and stage == 'base'"), rat)
    qn = cnt.groupby(["family", "qtype"]).apply(lambda x: x[["trait", "item"]].drop_duplicates().shape[0]).unstack(fill_value=0)
    L += ["Questions per type in each family's replies (about-the-speaker excluded): " + "; ".join(f"{f}: task {int(r.get('task', 0))}, conversation {int(r.get('conversation', 0))}" for f, r in qn.iterrows()) + ".", ""]
    ic = []
    for p in glob.glob("results/overgeneralization/trait_atlas_incharacter/atlas_*.csv.gz"):
        fam, stage = os.path.basename(p)[6:-7].rsplit("_", 1)
        ic.append(pd.read_csv(p).assign(family=fam, stage=stage).rename(columns={"value": "ic"})[["family", "stage", "trait", "item", "context", "persona", "sample", "ic"]])
    IC = pd.concat(ic) if ic else None
    rows, heads = [], []
    for fam in FAMS5:
        for stage in ("instruct", "base"):
            d = D[(D.family == fam) & (D.stage == stage)]
            if d.empty:
                continue
            for ctx in ("light_system", "light_user"):
                for det in ("md", "stock", "says_ai", "refusal", "logwords"):
                    g = gap_boot(d, ctx, det, rng)
                    rows.append(dict(family=fam, stage=stage, context=ctx, detector=det, **g, reading=reading(g) if det == "md" else ""))
                    if det == "md" and stage == "instruct":
                        heads.append(f"| {fam} | {ctx} | {fmt(g)} | {reading(g)} | " + (f"{g['a_task']:.0f} / {g['a_conv']:.0f} | {g['l_task']:.0f} / {g['l_conv']:.0f}" if "gap" in g else "-") + " |")
    R = pd.DataFrame(rows)
    # base same gap?
    note = []
    for fam in FAMS5:
        for ctx in ("light_system", "light_user"):
            b = R[(R.family == fam) & (R.stage == "base") & (R.context == ctx) & (R.detector == "md")]
            f = R[(R.family == fam) & (R.stage == "instruct") & (R.context == ctx) & (R.detector == "md")]
            if len(b) and len(f) and "gap" in b.iloc[0] and b.iloc[0]["gap"] == b.iloc[0]["gap"] and b.iloc[0]["gap"] >= 20 and b.iloc[0]["lo"] > 0:
                note.append(f"{fam} {ctx}: the base already shows a task gap of {b.iloc[0]['gap']:+.1f}")
    L += ["## Readings (markdown, final; task gap)", "", "| family | light-role position | task gap [95% CI] | reading | Assistant task / conversation (%) | light role task / conversation (%) |", "|---|---|---|---|---|---|"] + heads + [""]
    L += ["Base models: " + ("; ".join(note) + " (so the pattern is not from training in those cells)." if note else "no cell has a task gap of at least 20 points with a range above 0 at base."), ""]
    L += ["## All detectors, final and base (task gap, points)", "", "| family | stage | position | detector | task gap [95% CI] |", "|---|---|---|---|---|"]
    for r in R.sort_values(["family", "stage", "context", "detector"]).itertuples():
        L.append(f"| {r.family} | {r.stage} | {r.context} | {r.detector} | {fmt(r._asdict())} |" if hasattr(r, "gap") and r.gap == r.gap else f"| {r.family} | {r.stage} | {r.context} | {r.detector} | n/a |")
    # change from base per cell is available as the difference of the two rows above; give the markdown one explicitly
    L += ["", "Change from base of the markdown task gap (final minus base, points):", ""]
    for fam in FAMS5:
        for ctx in ("light_system", "light_user"):
            b = R[(R.family == fam) & (R.stage == "base") & (R.context == ctx) & (R.detector == "md")]
            f = R[(R.family == fam) & (R.stage == "instruct") & (R.context == ctx) & (R.detector == "md")]
            if len(b) and len(f) and f.iloc[0].get("gap") == f.iloc[0].get("gap") and b.iloc[0].get("gap") == b.iloc[0].get("gap"):
                L.append(f"- {fam} {ctx}: {f.iloc[0]['gap'] - b.iloc[0]['gap']:+.1f}")
    # in-character sensitivity
    L += ["", "## Sensitivity: in character only (final; OLMo, Qwen, Gemma 31B; light replies the existing in-character judge marked in character)", "",
          "| family | position | share in character | task gap, in character only [95% CI] | reading |", "|---|---|---|---|---|"]
    if IC is not None:
        for fam in FAMS3:
            d = D[(D.family == fam) & (D.stage == "instruct")]
            for ctx in ("light_system", "light_user"):
                icf = IC[(IC.family == fam) & (IC.stage == "instruct") & (IC.context == ctx)]
                share = icf.ic.mean() if len(icf) else np.nan
                if not (share == share):
                    continue
                if share < 0.30:
                    L.append(f"| {fam} | {ctx} | {100 * share:.0f}% | not read (under 30% in character) | - |")
                    continue
                g = gap_boot(d, ctx, "md", rng, incharacter=IC[(IC.family == fam) & (IC.stage == "instruct")][["family", "stage", "trait", "item", "context", "persona", "sample", "ic"]])
                L.append(f"| {fam} | {ctx} | {100 * share:.0f}% | {fmt(g)} | {reading(g)} |")
    L.append("")
    # says_ai across question sets
    L += ["## `says_ai` across question sets (final; sets where the Assistant says it is an AI at least 20% of the time)", "",
          "| family | question set | Assistant rate | light system | light user | reading (system / user) |", "|---|---|---|---|---|---|"]
    summ = []
    for fam in FAMS5:
        d = pd.read_csv(os.path.join(OUT, "detections.csv.gz"))
        d = d[(d.family == fam) & (d.stage == "instruct")]
        q = 0
        reads = []
        for trait, g in d.groupby("trait"):
            a = g[g.context == "assistant"].says_ai.mean() * 100
            if a < 20:
                continue
            res = []
            vals = []
            for ctx in ("light_system", "light_user"):
                l = g[g.context == ctx]
                diff = l.pivot_table(index="item", columns="persona", values="says_ai", aggfunc="mean").to_numpy() * 100
                aitem = g[g.context == "assistant"].groupby("item").says_ai.mean().to_numpy() * 100
                dm = diff - aitem[:, None]
                ci = np.nanpercentile([np.nanmean(dm[rng.integers(0, dm.shape[0], dm.shape[0])][:, rng.integers(0, dm.shape[1], dm.shape[1])]) for _ in range(1000)], [2.5, 97.5])
                lr = l.says_ai.mean() * 100
                m = lr - a
                if m <= -15 and ci[1] < 0 and lr <= a / 4:
                    rd = "stays"
                elif abs(m) <= 10:
                    rd = "travels"
                else:
                    rd = "in between"
                res.append(rd)
                vals.append(f"{lr:.0f}% ({m:+.0f})")
            L.append(f"| {fam} | {trait} | {a:.0f}% | {vals[0]} | {vals[1]} | {res[0]} / {res[1]} |")
            reads.append(tuple(res))
            q += 1
        summ.append(f"- **{fam}**: {q} question set(s) qualify; " + (", ".join(f"{k} x{reads.count(k)}" for k in sorted(set(reads))) if reads else "none to read"))
    L += [""] + summ + [""]
    L += groups_table() + [""]
    ap_ = os.path.join(OUT, "apertus_saysai_reading.md")
    if os.path.exists(ap_):
        L += open(ap_, encoding="utf-8").read().split("\n")
    os.makedirs(OUT, exist_ok=True)
    R.to_csv(os.path.join(OUT, "task_gap.csv"), index=False)
    open(os.path.join(OUT, "unmixed.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:30]))


# ------------------------------------------------------------------ feature scan (descriptive, no readings) and group table
FORM = {"em dash": r"\u2014", "colon": r":", "exclamation mark": r"!", "question mark": r"\?", "word I": r"\bI\b", "word you": r"\byou\b", "bold": r"\*\*", "list line": r"(?m)^\s*(?:[-*\u2022]|\d+[.)])\s", "heading": r"(?m)^\s*#{1,6}\s"}
STOPNAMES = None


def role_words():
    m = ta.load_materials()
    txt = " ".join(m["light"]) + " " + " ".join(s["name"] + " " + s["role"] + " " + s["sheet"] for s in m["strong"])
    return set(re.findall(r"[a-z]+", txt.lower()))


def load_replies(fam, stage, traits=None):
    rows = []
    with gzip.open(replies_file(fam, stage), "rt", encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            if r["context"] not in CONTEXTS or (traits and r["trait"] not in traits):
                continue
            w = (r["text"] or "").replace("\u2019", "'").split()[:100]
            rows.append((r["trait"], r["item"], r["context"], " ".join(w)))
    return pd.DataFrame(rows, columns=["trait", "item", "context", "text"])


def feature_matrix(texts, vocab, form_pats):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(ngram_range=(1, 2), lowercase=True, binary=True, vocabulary=vocab, token_pattern=r"[a-z0-9']+")
    X = cv.transform(texts).tocsc()
    F = np.array([[bool(re.search(p_, t)) for p_ in form_pats.values()] for t in texts], dtype=np.float32)
    import scipy.sparse as sp
    return sp.hstack([X, sp.csr_matrix(F)]).tocsr(), list(vocab) + list(form_pats)


def monroe(c1, n1, c2, n2, prior, a0=100.0):
    a = prior * a0
    d1 = np.log((c1 + a) / (n1 + a0 - c1 - a)) - np.log((c2 + a) / (n2 + a0 - c2 - a))
    var = 1 / (c1 + a) + 1 / (n1 - c1 + a0 - a) + 1 / (c2 + a) + 1 / (n2 - c2 + a0 - a)
    return d1, d1 / np.sqrt(var)


def scan_family(fam, traits, rng):
    from sklearn.feature_extraction.text import CountVectorizer
    ex = role_words()
    B, F = load_replies(fam, "base", traits), load_replies(fam, "instruct", traits)
    allt = pd.concat([B, F])
    cv = CountVectorizer(ngram_range=(1, 2), lowercase=True, binary=True, min_df=0.01, token_pattern=r"[a-z0-9']+")
    cv.fit(allt.text)
    vocab = [w for w in cv.get_feature_names_out() if not (set(w.split()) & ex)]
    XB, names = feature_matrix(B.text, vocab, FORM)
    XF, _ = feature_matrix(F.text, vocab, FORM)
    prior = (np.asarray(XB.sum(0)).ravel() + np.asarray(XF.sum(0)).ravel()) / (XB.shape[0] + XF.shape[0])
    prior = np.clip(prior, 1e-4, 1 - 1e-4)
    def shift(mask_b, mask_f, items=None):
        cb = np.asarray(XB[mask_b].sum(0)).ravel(); cf = np.asarray(XF[mask_f].sum(0)).ravel()
        return monroe(cf, mask_f.sum(), cb, mask_b.sum(), prior)
    ab, af = (B.context == "assistant").to_numpy(), (F.context == "assistant").to_numpy()
    dA, zA = shift(ab, af)
    keys = (B.trait + "/" + B.item).to_numpy(), (F.trait + "/" + F.item).to_numpy()
    uq = np.unique(np.concatenate(keys))
    half = set(rng.permutation(uq)[: len(uq) // 2])
    h1b, h1f = ab & np.isin(keys[0], list(half)), af & np.isin(keys[1], list(half))
    h2b, h2f = ab & ~np.isin(keys[0], list(half)), af & ~np.isin(keys[1], list(half))
    d1, _ = shift(h1b, h1f)
    d2, _ = shift(h2b, h2f)
    ok = np.sign(d1) == np.sign(d2)
    idx = [i for i in np.argsort(-np.abs(dA)) if ok[i]][:300]
    out = {}
    for ctx in CONTEXTS:
        cb, cf = (B.context == ctx).to_numpy(), (F.context == ctx).to_numpy()
        dC, _ = shift(cb, cf)
        out[ctx] = dC
    rows = [dict(family=fam, feature=names[i], assistant_shift=dA[i], **{f"ratio_{c}": out[c][i] / dA[i] for c in CONTEXTS}) for i in idx]
    return pd.DataFrame(rows)


def features():
    rng = np.random.default_rng(SEED)
    m = ta.load_materials()
    all20 = [t["id"] for t in m["traits"]]
    res = {}
    for label, fams, traits in (("all 20 question sets, three families", FAMS3, all20), ("shared 8 question sets, five families", FAMS5, SHARED8)):
        res[label] = pd.concat([scan_family(f, traits, rng) for f in fams])
        res[label].to_csv(os.path.join(OUT, "features_" + ("all20" if "20" in label else "shared8") + ".csv"), index=False)
    L = ["# Feature scan (spec P1; descriptive, no readings; NOT BLIND)", "",
         "Features: single words and word pairs (lowercased, in at least 1% of replies) and form features (em dash, colon, exclamation mark, question mark, 'I', 'you', bold, list line, heading), first 100 words of each reply, words of every role and character name and description excluded. "
         "Base-to-final shift in the Assistant as log-odds with an informative prior (Monroe et al.; prior from the pooled feature rates, prior mass 100 replies); the 300 features with the largest Assistant shift (gains and losses) whose sign agrees on two random halves of the questions are kept; "
         "spread ratio = the same shift in a context divided by the Assistant's. Themes are not assigned and no single feature is read as significant. Ratios are shown for the light roles (system, user turn) and full role-play (acting, `strong_act`).", ""]
    for label, df in res.items():
        L += [f"## {label}", ""]
        fams = list(dict.fromkeys(df.family))
        piv = {c: df.pivot_table(index="feature", columns="family", values=f"ratio_{c}") for c in ("light_system", "light_user", "strong_act")}
        cnt = df.groupby("feature").family.nunique()
        feats = cnt[cnt >= 2].index
        def cls(f):
            ls = [(piv["light_system"].loc[f, x], piv["light_user"].loc[f, x], piv["strong_act"].loc[f, x]) for x in fams if f in piv["light_system"].index and x in piv["light_system"].columns and pd.notna(piv["light_system"].loc[f, x])]
            lt = [a >= 0.5 and b >= 0.5 for a, b, c in ls]
            st = [a >= 0.5 and b >= 0.5 and c < 0.25 for a, b, c in ls]
            ay = [a < 0.25 and b < 0.25 for a, b, c in ls]
            n = len(ls)
            return ("stops at full role-play", sum(st), n) if sum(st) > n / 2 else ("spreads to light roles", sum(lt), n) if sum(lt) > n / 2 else ("stays with the Assistant", sum(ay), n) if sum(ay) > n / 2 else (None, 0, n)
        groups = {"spreads to light roles": [], "stops at full role-play": [], "stays with the Assistant": []}
        sh = df.groupby("feature").assistant_shift.mean()
        for f in feats:
            g, k, n = cls(f)
            if g:
                groups[g].append((f, k, n, sh[f]))
        for g, lst in groups.items():
            lst.sort(key=lambda x: (-x[1] / x[2], -abs(x[3])))
            L += [f"### Features in the top 300 of two or more families that {g.replace('spreads', 'spread').replace('stops', 'stop').replace('stays', 'stay')} in most of those families ({len(lst)}; light roles = ratio of at least 0.5 in both positions; 'stop' = also under 0.25 in full role-play; 'stay' = under 0.25 in both positions)", "", "| feature | families (agree / present) | mean Assistant shift (log-odds, + = more in final) |", "|---|---|---|"]
            L += [f"| `{f}` | {k} / {n} | {s_:+.2f} |" for f, k, n, s_ in lst[:40]]
            L.append("")
    open(os.path.join(OUT, "features.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:20]))


def groups_table():
    GROUPS = {"speaker": ["says_ai", "denies_feelings", "declines_opinion"], "manner": ["markdown_lists", "stock_phrases", "safety_caveats", "closing_offer", "clarifying_question", "flattery", "formal_register"],
              "truth": ["corrects_false_premise", "hedging_unanswerable", "fabricates_vs_admits"]}
    FLAG = {"agrees_wrong_opinion", "declines_opinion", "flattery", "formal_register"}      # second judge kappa < 0.4 (results/overgeneralization/second_judge)
    cells = pd.concat([pd.read_csv("results/overgeneralization/trait_atlas/trait_atlas_cells.csv"), pd.read_csv("results/overgeneralization/trait_atlas_breadth/trait_atlas_breadth_cells.csv")])
    cells = cells[(~cells.flagged_excluded) & (cells.stage == "instruct")]
    L = ["## Descriptive group table (groups made after seeing the data; proves nothing)", "",
         "Per family and context: of the traits in the group whose Assistant change passes the atlas gate, how many spread (atlas rule). * = traits the second judge flagged as unreliable (agrees_wrong_opinion, declines_opinion, flattery, formal_register).", "",
         "| family | group | context | gated traits | spread | stay or reverse |", "|---|---|---|---|---|---|"]
    for fam in FAMS5:
        for gname, traits in GROUPS.items():
            for ctx in ("light_system", "light_user", "strong_act"):
                sp, st, tot = [], [], []
                for t in traits:
                    a = cells[(cells.family == fam) & (cells.trait == t) & (cells.context == "assistant")]
                    c = cells[(cells.family == fam) & (cells.trait == t) & (cells.context == ctx)]
                    if a.empty or c.empty:
                        continue
                    a, c = a.iloc[0], c.iloc[0]
                    if not ta.gate(a.change, a.lo, a.hi, a.unit):
                        continue
                    nm = t + ("*" if t in FLAG else "")
                    tot.append(nm)
                    (sp if ta.reading(c.change, c.lo, c.hi, a.change, a.unit) == "spreads" else st).append(nm)
                if tot:
                    L.append(f"| {fam} | {gname} | {ctx} | {len(tot)} | {len(sp)} ({', '.join(sp)}) | {len(st)} ({', '.join(st)}) |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["detect", "analyze", "features"])
    a = ap.parse_args()
    {"detect": detect, "analyze": analyze, "features": features}[a.cmd]()


if __name__ == "__main__":
    main()
