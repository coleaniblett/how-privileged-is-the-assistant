"""User-slot style fingerprint (spec A): does the model's continuation of a real user's message read more like the Assistant?

    python -m measures.overgeneralization.user_style --family olmo3-7b [--stratum incomplete|boundary]

Reads the existing generations (results/capstone/<family>/core/generations_incomplete/<stage>.jsonl: 200 mid-sentence prefills x 3
samples; stratum "boundary" = core/generations/, 100 sentence-boundary prefills; the two strata are never pooled) and compares them with
the real remainder of the same WildChat user turn. User portion = user_traits.py's cut (turn-end / handoff markers / line-initial role
labels). Plan and pre-set reading: docs/decisions.md, "user-slot style fingerprint (spec A)". Exploratory, outside the verdict; CPU only.

Score: P(assistant) from the human-vs-Assistant classifier (human_vs_assistant.py; gate AUC >= 0.85 at 10-20 words, results/user_style/
classifier_gate.md), on the first 40 words of the text. Views: (1) the completion of the cut sentence (to the first . ? ! or newline),
primary; (2) the full user portion, and the same restricted to >= 15 words. Human reference: the same cut of the real remainder; for the
full view, the first N words with N matched to each model sample. Effects: stage minus base and stage minus the real-user reference,
paired per prefill, 95% bootstrap CI over prefills.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

from core import config
from core.stats import stage_order
from measures.overgeneralization import human_vs_assistant as hva
from measures.overgeneralization import user_traits as ut

STRATA = {
    "incomplete": dict(gens="generations_incomplete", prefills="data/prefill_incomplete_200.jsonl"),
    "boundary": dict(gens="generations", prefills="data/prefill_100.jsonl"),
}
MIN_SCORE_WORDS = 3          # shorter pieces are not scored by the classifier (too little text)
SCORE_WORDS = 40             # the classifier is trained on windows of at most 40 words: score the first 40
LONG = 15                    # the ">= 15 words" restriction of the full view
LEAK_BAR = 0.05              # pre-set polish-leak bar (P(assistant) units), both against base and against the real user

SENT_END = re.compile(r"[.?!]+[\"')\]]*(?=\s|$)|\n")
POLITE = re.compile(r"\b(please|pls|thank you|thanks|kindly|appreciate)\b", re.I)
MARKDOWN = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+\S|^[ \t]*[-*•][ \t]+\S|\*\*[^*\n]+\*\*", re.M)
TERMINAL = re.compile(r"[.?!][\"')\]]*\s*$")
STANDALONE_I = re.compile(r"(?<![\w'])[iI](?![\w'])")
TOKEN = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")
URLISH = re.compile(r"https?://\S+|www\.\S+|\S+@\S+|`[^`]*`")


# ---------------------------------------------------------------- text pieces
def first_sentence(text):
    """Text up to and including the first sentence end (. ? ! or a newline) -- punctuation kept, newline dropped."""
    t = text.lstrip(" \t")
    m = SENT_END.search(t)
    return (t[: m.start()] + ("" if m.group(0) == "\n" else m.group(0))).strip() if m else t.strip()


def words(text):
    return text.split()


def head_words(text, n):
    return " ".join(words(text)[:n])


# ---------------------------------------------------------------- style features
_WORDLIST = {}


def _known(w):
    if not _WORDLIST:
        from nltk.corpus import words as nw
        _WORDLIST["w"] = {x.lower() for x in nw.words()}
    from wordfreq import zipf_frequency
    lw = w.lower()
    return lw in _WORDLIST["w"] or zipf_frequency(lw, "en") >= 3.5


def typo_rate(text):
    """Share of alphabetic tokens (>= 3 letters; code, URLs and numbers ignored) that are not a known word. NaN if no tokens."""
    toks = [t for t in TOKEN.findall(URLISH.sub(" ", text)) if len(t) >= 3]
    return float(np.mean([not _known(t.split("'")[0]) for t in toks])) if toks else np.nan


def has_typo(text):
    r = typo_rate(text)
    return bool(r > 0) if not np.isnan(r) else False


def features(text):
    """Style features of a user portion (NaN where undefined)."""
    n = len(words(text))
    sents = [s for s in re.split(r"(?<=[.?!])\s+|\n+", text.strip()) if s.strip()]
    i_tok = STANDALONE_I.findall(text)
    return dict(
        starts_lower=float(text.lstrip()[:1].islower()) if text.strip() and text.lstrip()[:1].isalpha() else np.nan,
        typo_rate=typo_rate(text),
        sent_len=float(np.mean([len(words(s)) for s in sents])) if sents else np.nan,
        polite=float(bool(POLITE.search(text))),
        exclaim=float("!" in text),
        markdown=float(bool(MARKDOWN.search(text))),
        cap_i=float(np.mean([t == "I" for t in i_tok])) if i_tok else np.nan,
        terminal=float(bool(TERMINAL.search(text))) if n else np.nan,
    )


def prefill_flags(prefill):
    """Binary prefill features for style matching, and the continuation-side counterpart of each."""
    letters = [c for c in prefill if c.isalpha()]
    return dict(lower=bool(letters) and prefill == prefill.lower() and any(c.isalpha() for c in prefill),
                no_terminal=not TERMINAL.search(prefill), typo=has_typo(prefill))


def matches(prefill, cont):
    """Agreement of continuation with prefill on each binary feature; NaN where the continuation has nothing to read."""
    if not cont.strip():
        return dict(match_lower=np.nan, match_no_terminal=np.nan, match_typo=np.nan)
    pf = prefill_flags(prefill)
    c_lower = cont == cont.lower() and any(c.isalpha() for c in cont)
    c_noterm = not TERMINAL.search(cont)
    return dict(match_lower=float(pf["lower"] == c_lower), match_no_terminal=float(pf["no_terminal"] == c_noterm),
                match_typo=float(pf["typo"] == has_typo(cont)))


# ---------------------------------------------------------------- data
def load_prefills(stratum):
    return {r["id"]: r for r in ut.read_jsonl(STRATA[stratum]["prefills"])}


def human_remainder(prefill_row, texts):
    full = texts[prefill_row["text_id"]]
    p = prefill_row["prefill"]
    rest = full[len(p):] if full.startswith(p) else full.replace(p, "", 1)
    return rest


def build_rows(stage_samples, markers, prefills, texts):
    """One row per (prefill, sample) for a stage: user portion, sentence view, and the length-matched human text.
    stage_samples: core generation rows."""
    rows = []
    for r in stage_samples:
        pf = prefills[r["prefill_id"]]
        u = ut.cut_user_portion(r["continuation"], markers, role_cut=True)
        n = len(words(u))
        hum = human_remainder(pf, texts)
        rows.append(dict(prefill_id=r["prefill_id"], sample=r["sample"], prefill=r["prefill"], full=u, n_words=n,
                         sent=first_sentence(u), human_full=head_words(hum, n), human_sent=first_sentence(hum)))
    return pd.DataFrame(rows)


def add_scores(df, clf, cols):
    for c in cols:
        ok = df[c].map(lambda t: len(words(t)) >= MIN_SCORE_WORDS)
        df[c + "_p"] = np.nan
        if ok.any():
            df.loc[ok, c + "_p"] = clf.p_assistant([head_words(t, SCORE_WORDS) for t in df.loc[ok, c]])
    return df


def per_prefill_mean(df, col, mask=None):
    d = df if mask is None else df[mask]
    return d.groupby("prefill_id")[col].mean().reindex(df.prefill_id.unique())


# ---------------------------------------------------------------- tables
def effect_rows(series_by_stage, human, base="base", n_boot=ut.N_BOOT, label=""):
    """Rows: each non-base stage minus base, and minus the human reference (a Series per prefill, or None)."""
    rows = []
    for s in stage_order([x for x in series_by_stage if x != base]):
        for ref_name, ref in (("base", series_by_stage[base]), ("real user", human)):
            if ref is None:
                continue
            rng = np.random.default_rng(ut.SEED)
            ch = ut.paired_change({"r": ref, "s": series_by_stage[s]}, "r", "s", rng, n_boot)
            rows.append(dict(measure=label, stage=s, versus=ref_name, **ch))
    return rows


def fmt_ch(r, d=3):
    if r["n"] < 2 or pd.isna(r["mean"]):
        return "n/a"
    return f"{r['mean']:+.{d}f} [{r['lo']:+.{d}f}, {r['hi']:+.{d}f}]"


def polish_leak(rows, family):
    """The pre-set rule on the primary measure (view 1, incomplete stratum): OLMo instruct minus base >= LEAK_BAR with the CI excluding
    zero AND instruct minus real user >= LEAK_BAR (point estimate). None for other families / missing data."""
    if family != "olmo3-7b":
        return None
    t = pd.DataFrame(rows)
    t = t[(t.measure == "P(assistant), completion of cut sentence") & (t.stage == "instruct")]
    b, h = t[t.versus == "base"], t[t.versus == "real user"]
    if b.empty or h.empty or b.iloc[0]["n"] < 2:
        return None
    b, h = b.iloc[0], h.iloc[0]
    return bool(b["mean"] >= LEAK_BAR and b["lo"] > 0 and h["mean"] >= LEAK_BAR)


def analyse(family, stratum, clf, texts, run="capstone", n_boot=ut.N_BOOT):
    stages = list(config.family_settings(family)["stages"])
    gd = os.path.join(config.RESULTS_ROOT, run, family, "core", STRATA[stratum]["gens"])
    markers = ut.family_markers(family)
    prefills = load_prefills(stratum)
    per = {}
    for s in stages:
        p = os.path.join(gd, f"{s}.jsonl")
        if os.path.exists(p):
            df = build_rows(ut.read_jsonl(p), markers, prefills, texts)
            df = add_scores(df, clf, ["sent", "full", "human_sent", "human_full"])
            feats = pd.DataFrame([features(t) for t in df.full])
            mt = pd.DataFrame([matches(a, b) for a, b in zip(df.prefill, df.full)])
            hf = pd.DataFrame([features(t) for t in df.human_full]).add_prefix("h_")
            per[s] = pd.concat([df, feats, mt, hf], axis=1)
    if "base" not in per:
        raise SystemExit(f"no {stratum} generations for the base stage in {gd}")
    # human reference per prefill (sentence view: one value per prefill; full view: mean over the matched samples of the base stage's lengths is
    # stage-specific, so the reference for each stage is computed from that stage's own matched human text)
    rows, tables = [], {}
    h_sent = per["base"].groupby("prefill_id")["human_sent_p"].first()
    sent = {s: per_prefill_mean(d, "sent_p") for s, d in per.items()}
    rows += effect_rows(sent, h_sent, n_boot=n_boot, label="P(assistant), completion of cut sentence")
    for name, mask_fn in (("P(assistant), full user portion", None), (f"P(assistant), full user portion, >= {LONG} words", lambda d: d.n_words >= LONG)):
        ser = {s: per_prefill_mean(d, "full_p", None if mask_fn is None else mask_fn(d)) for s, d in per.items()}
        for s in [x for x in per if x != "base"]:
            d = per[s]; m = None if mask_fn is None else mask_fn(d)
            hs = per_prefill_mean(d, "human_full_p", m)
            rng = np.random.default_rng(ut.SEED)
            for ref_name, ref in (("base", ser["base"]), ("real user", hs)):
                ch = ut.paired_change({"r": ref, "s": ser[s]}, "r", "s", rng, n_boot)
                rows.append(dict(measure=name, stage=s, versus=ref_name, **ch))
    abs_rows = []
    for s, d in per.items():
        abs_rows.append(dict(stage=s, sent=float(sent[s].mean()), full=float(per_prefill_mean(d, "full_p").mean()),
                             full_long=float(per_prefill_mean(d, "full_p", d.n_words >= LONG).mean()),
                             scored_sent=int(d.sent_p.notna().sum()), scored_full=int(d.full_p.notna().sum()), n=len(d),
                             words=float(d.n_words.mean())))
    abs_rows.append(dict(stage="real user", sent=float(h_sent.mean()), full=float(np.nanmean([per_prefill_mean(d, "human_full_p").mean() for d in per.values()])),
                         full_long=float("nan"), scored_sent=int(h_sent.notna().sum()), scored_full=np.nan, n=int(len(h_sent)), words=np.nan))
    # style features and matching: stage minus base, and the real-user value
    fe = [("starts_lower", "starts lowercase (share)"), ("typo_rate", "typo rate"), ("sent_len", "mean sentence length (words)"),
          ("polite", "politeness words (share)"), ("exclaim", "exclamation (share)"), ("markdown", "markdown (share)"),
          ("cap_i", "capital \"I\" (share of standalone i/I)"), ("terminal", "ends with terminal punctuation (share)")]
    frows = []
    for col, name in fe:
        ser = {s: per_prefill_mean(d, col) for s, d in per.items()}
        humn = {s: per_prefill_mean(d, "h_" + col) for s, d in per.items()}
        for s in [x for x in per if x != "base"]:
            rng = np.random.default_rng(ut.SEED)
            ch = ut.paired_change({"r": ser["base"], "s": ser[s]}, "r", "s", rng, n_boot)
            hv = float(humn[s].mean()) if col != "terminal" else np.nan
            frows.append(dict(feature=name, stage=s, base=ch["base_mean"], stage_value=ch["stage_mean"], real_user=hv, **{k: ch[k] for k in ("n", "mean", "lo", "hi")}))
    mrows = []
    for col, name in [("match_lower", "prefill all-lowercase -> continuation lowercase"), ("match_no_terminal", "prefill without terminal punctuation -> continuation without"),
                      ("match_typo", "prefill has a typo -> continuation has a typo")]:
        ser = {s: per_prefill_mean(d, col) for s, d in per.items()}
        for s in [x for x in per if x != "base"]:
            rng = np.random.default_rng(ut.SEED)
            ch = ut.paired_change({"r": ser["base"], "s": ser[s]}, "r", "s", rng, n_boot)
            mrows.append(dict(feature="agreement: " + name, stage=s, base=ch["base_mean"], stage_value=ch["stage_mean"], **{k: ch[k] for k in ("n", "mean", "lo", "hi")}))
    return dict(rows=rows, abs=pd.DataFrame(abs_rows), feat=pd.DataFrame(frows), match=pd.DataFrame(mrows), per=per,
                leak=polish_leak(rows, family))


def examples(per, k=3, chars=260):
    out = []
    for s, d in per.items():
        d = d[d.n_words >= 8].sort_values("full_p")
        for tag, part in (("highest", d.tail(k).iloc[::-1]), ("lowest", d.head(k))):
            for r in part.itertuples():
                out.append((s, tag, r.full_p, r.prefill, " ".join(r.full.split())[:chars]))
    return out


def report(res, family, stratum, gate_text):
    c = lambda x: "n/a" if pd.isna(x) else f"{x:.3f}"
    L = [f"# User-slot style fingerprint, {family}, stratum: {stratum} (exploratory, outside the verdict)", "",
         "Does the model's continuation of a real WildChat user's message read more like the Assistant than the real user's own continuation? "
         "P(assistant) comes from the human-vs-Assistant classifier (TF-IDF + logistic regression, trained on held-out WildChat user turns vs "
         "ChatGPT replies in 5-40-word windows). Classifier gate: " + gate_text, ""]
    if family in ut.SUPPORTING:
        L += ["Caveat: no contamination check is possible for this family; supporting only.", ""]
    leak = res["leak"]
    L += ["**Pre-set reading (primary: P(assistant) on the completion of the cut sentence, OLMo instruct minus base >= +0.05 with the CI excluding zero, "
          "and instruct minus the real user >= +0.05):** " + ("not applicable to this family or stratum" if leak is None or stratum != "incomplete"
                                                            else ("**polish leak**" if leak else "no polish leak")), "",
          "## Mean P(assistant) per stage (per-prefill means)", "", "| stage | completion of cut sentence | full user portion | full, >= 15 words | samples scored (sentence view; real user: prefills) | samples (real user: prefills) | mean words |",
          "|---|---|---|---|---|---|---|"]
    L += [f"| {r.stage} | {c(r.sent)} | {c(r.full)} | {c(r.full_long)} | {r.scored_sent} | {r.n} | {c(r.words)} |" for r in res["abs"].itertuples()]
    L += ["", "## P(assistant): stage minus reference, paired per prefill, 95% bootstrap CI over prefills", "",
          "| measure | stage | versus | prefills used | dropped | change [95% CI] |", "|---|---|---|---|---|---|"]
    L += [f"| {r['measure']} | {r['stage']} | {r['versus']} | {r['n']} | {r['dropped']} | {fmt_ch(r)} |" for r in res["rows"]]
    L += ["", "## Style features (full user portion; per-prefill means; stage minus base with 95% CI; the real-user value is the length-matched human text)", "",
          "| feature | stage | base | stage | real user | stage minus base [95% CI] | prefills |", "|---|---|---|---|---|---|---|"]
    L += [f"| {r.feature} | {r.stage} | {c(r.base)} | {c(r.stage_value)} | {c(r.real_user)} | {fmt_ch(dict(n=r.n, mean=r.mean, lo=r.lo, hi=r.hi))} | {r.n} |" for r in res["feat"].itertuples()]
    L += ["", "## Style matching (agreement rate between prefill and continuation; falling with training = normalising the user)", "",
          "| feature | stage | base | stage | stage minus base [95% CI] | prefills |", "|---|---|---|---|---|---|"]
    L += [f"| {r.feature} | {r.stage} | {c(r.base)} | {c(r.stage_value)} | {fmt_ch(dict(n=r.n, mean=r.mean, lo=r.lo, hi=r.hi))} | {r.n} |" for r in res["match"].itertuples()]
    L += ["", "## Real examples per stage (full user portion, >= 8 words; highest and lowest P(assistant))", ""]
    L += [f"- **{s}**, {tag} ({p:.2f}), prefill \"{pf[:70]}\": ...{t}" for s, tag, p, pf, t in examples(res["per"])]
    L += ["", "Notes: the typo rate is a proxy (tokens of >= 3 letters not in the nltk `words` list and with wordfreq zipf < 3.5; names and rare words count for "
          "humans and models alike). `starts lowercase` is only meaningful in the boundary stratum, where the continuation starts a sentence. The classifier "
          "scores the first 40 words; pieces under 3 words are not scored.", ""]
    return "\n".join(L) + "\n"


def main():
    config.family_from_argv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=ut.FAMILIES)
    ap.add_argument("--stratum", choices=list(STRATA), default="incomplete")
    ap.add_argument("--run", default="capstone")
    ap.add_argument("--n-boot", type=int, default=ut.N_BOOT)
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    rows = hva.read_jsonl(hva.TRAIN_PATH)
    clf = hva.fit_on(rows)
    gate, ok = hva.gate_report(rows, clf)
    gate_text = f"held-out AUC {gate['10-20 words']:.3f} at 10-20 words ({'PASS' if ok else 'FAIL'}, gate {hva.GATE_AUC})."
    if not ok:
        raise SystemExit("classifier gate failed: " + gate_text)
    texts = {r["id"]: r["text"] for r in ut.read_jsonl("data/texts.jsonl")}
    res = analyse(a.family, a.stratum, clf, texts, a.run, a.n_boot)
    out = a.out_dir or os.path.join(config.RESULTS_ROOT, a.run, a.family, "overgeneralization")
    os.makedirs(out, exist_ok=True)
    stem = "user_style" if a.stratum == "incomplete" else "user_style_boundary"
    text = report(res, a.family, a.stratum, gate_text)
    open(os.path.join(out, stem + ".md"), "w", encoding="utf-8").write(text)
    pd.concat([pd.DataFrame(res["rows"]).assign(table="P(assistant)"), res["feat"].assign(table="feature"), res["match"].assign(table="matching"),
               res["abs"].assign(table="levels")], ignore_index=True).to_csv(os.path.join(out, stem + ".csv"), index=False)
    print(text)


if __name__ == "__main__":
    main()
