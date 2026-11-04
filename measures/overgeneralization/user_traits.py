"""User-slot overgeneralization arm: do assistant traits show up in what the model writes as the *user*?

    python -m measures.overgeneralization.user_traits judge   --family olmo3-7b     # primary, needs Cole's judge outputs
    python -m measures.overgeneralization.user_traits keyword --family olmo3-7b     # judge-free cross-check, runs now

Both read the existing core generations (results/<run>/<family>/core/generations/<stage>.jsonl: prefilled user turns,
100 prefills x 3 samples). Nothing is generated or scored here; no API calls, no GPU (the keyword arm loads a tokenizer
only to derive the family's turn markers).

Primary (judge arm; spec and pre-registered criteria in measures/overgeneralization/README.md): the judge
(measures/vestigial_regression/judge.py) labels each sample user_voice / switched_to_assistant / degenerate /
user_voice_with_assistant_traits. Per prefill, share of user-voice samples (user_voice + user_voice_with_assistant_traits)
labelled user_voice_with_assistant_traits; stage minus base, paired per prefill, 95% bootstrap CI over prefills
(core.stats.bootstrap_ci). Prefills with no user-voice sample in either stage are dropped and counted. Secondary: the same
share over all samples. Fires if OLMo instruct is >= FIRE_BAR above base with the CI excluding zero.

Cross-check (outside the verdict): in the user portion of each continuation (cut at the family's turn-end, turn-open,
end-of-text and handoff markers), flag assistant stock phrases and markdown headings / bullets / bold that the prefill does
not already contain; same per-prefill change-from-base table. No judge involved.
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from core import config
from core.stats import bootstrap_ci, stage_order

UV, UVAT = "user_voice", "user_voice_with_assistant_traits"
USER_VOICE = (UV, UVAT)
INVALID = "invalid"   # the judge refused the sample; excluded and counted until a hand label replaces it
FIRE_BAR = 0.05          # percentage-point bar as a fraction; fixed 2026-11-04 (docs/decisions.md) now that judge results exist
N_BOOT = 10000
SEED = 0
FAMILIES = ["olmo3-7b", "qwen3-8b", "gemma4-31b"]
SUPPORTING = {"qwen3-8b", "gemma4-31b"}   # no contamination check possible; supporting only

# Judge-free keyword features. Stock phrases are assistant habits that a human user rarely opens or closes a message with.
STOCK = re.compile(
    r"\b(certainly[,!.]|of course[,!.]|sure[,!]|absolutely[,!]|i hope this helps|hope that helps|let me know if|feel free to"
    r"|as an ai\b|as a language model|(i'?d|i would|i'?m|i am) (be )?(happy|glad|here) to|happy to help|glad to help"
    r"|great question|good question|i'?m sorry, but|i apologi[sz]e|here (is|are) (a|an|the|some|your)\b"
    r"|i can help (you )?with|i don'?t have personal|how can i (help|assist)|is there anything else)",
    re.I,
)
HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+\S", re.M)
BULLET = re.compile(r"^[ \t]*(?:[-*•][ \t]+\S|\d{1,2}[.)][ \t]+\S)", re.M)
BOLD = re.compile(r"\*\*[^*\n]+\*\*|__[^_\n]+__")
KEYWORD_FEATURES = ["stock_phrase", "markdown_heading", "markdown_bullet", "markdown_bold"]


# ---------------------------------------------------------------- I/O
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def generations_dir(family, run="capstone"):
    return os.path.join(config.RESULTS_ROOT, run, family, "core", "generations")


def default_judge_dir(family, run="capstone"):
    return os.path.join(config.RESULTS_ROOT, run, family, "vestigial_regression", "judge")


def load_judge(judge_dir, stages, hand_dir=None, excluded=None):
    """{stage: DataFrame(prefill_id, sample, label)} for the stages that have <judge_dir>/<stage>.jsonl.
    Samples the judge refused (label "invalid") are excluded and counted in `excluded` ({stage: n}); hand labels in
    <hand_dir>/<stage>.jsonl (same schema) override the judge's label per (prefill_id, sample), including the invalid ones."""
    out = {}
    for s in stages:
        p = os.path.join(judge_dir, f"{s}.jsonl")
        if os.path.exists(p):
            d = pd.DataFrame(read_jsonl(p))
            if not {"prefill_id", "sample", "label"} <= set(d.columns):
                raise SystemExit(f"{p}: rows need prefill_id, sample, label")
            d = d[["prefill_id", "sample", "label"]]
            hp = os.path.join(hand_dir, f"{s}.jsonl") if hand_dir else None
            if hp and os.path.exists(hp):
                h = pd.DataFrame(read_jsonl(hp))[["prefill_id", "sample", "label"]]
                d = pd.concat([d, h]).drop_duplicates(["prefill_id", "sample"], keep="last")
            bad = set(d.label) - set(USER_VOICE) - {"switched_to_assistant", "degenerate", INVALID}
            if bad:
                raise SystemExit(f"{p}: unknown labels {sorted(bad)}")
            if excluded is not None:
                excluded[s] = int((d.label == INVALID).sum())
            out[s] = d[d.label != INVALID].reset_index(drop=True)
    return out


# ---------------------------------------------------------------- shared paired-change machinery
def paired_change(per_prefill, base, stage, rng, n_boot=N_BOOT):
    """per_prefill: {stage: Series indexed by prefill_id, NaN where undefined}. Stage minus base on prefills defined in both.
    Returns dict(n, dropped, mean, lo, hi, base_mean, stage_mean); mean/lo/hi are NaN if fewer than 2 prefills."""
    b, s = per_prefill[base], per_prefill[stage]
    ids = b.index.union(s.index)
    b, s = b.reindex(ids), s.reindex(ids)
    ok = b.notna() & s.notna()
    n, dropped = int(ok.sum()), int((~ok).sum())
    if n < 2:
        return dict(n=n, dropped=dropped, mean=np.nan, lo=np.nan, hi=np.nan, base_mean=np.nan, stage_mean=np.nan)
    m, lo, hi = bootstrap_ci((s[ok] - b[ok]).to_numpy(float), n_boot, rng)
    return dict(n=n, dropped=dropped, mean=m, lo=lo, hi=hi, base_mean=float(b[ok].mean()), stage_mean=float(s[ok].mean()))


# ---------------------------------------------------------------- judge arm
def trait_shares(judged):
    """Per prefill: (primary, secondary). primary = uvat / (user_voice + uvat), NaN when no user-voice sample;
    secondary = uvat / all samples."""
    g = judged.assign(uvat=judged.label == UVAT, uv=judged.label.isin(USER_VOICE)).groupby("prefill_id")
    n_uv, n_uvat, n = g.uv.sum(), g.uvat.sum(), g.size()
    primary = (n_uvat / n_uv.where(n_uv > 0)).astype(float)
    return primary, (n_uvat / n).astype(float)


def judge_table(judged_by_stage, base="base", seed=SEED, n_boot=N_BOOT):
    """Rows (stage, arm, n, dropped, mean, lo, hi, base_mean, stage_mean) for every non-base stage that has judge rows."""
    if base not in judged_by_stage:
        raise SystemExit(f"no judge rows for the base stage {base!r}")
    prim, sec = {}, {}
    for s, d in judged_by_stage.items():
        prim[s], sec[s] = trait_shares(d)
    rows = []
    for s in stage_order([x for x in judged_by_stage if x != base]):
        for arm, tab in (("primary", prim), ("secondary", sec)):
            rng = np.random.default_rng(seed)
            rows.append(dict(stage=s, arm=arm, **paired_change(tab, base, s, rng, n_boot)))
    return pd.DataFrame(rows)


def fires(table, family, stage="instruct", bar=FIRE_BAR):
    """The pre-registered rule: primary arm, OLMo instruct >= bar above base with the CI excluding zero. Other families and
    stages never decide it (supporting only). Returns True / False, or None when it cannot be evaluated."""
    if family != "olmo3-7b":
        return None
    r = table[(table.stage == stage) & (table.arm == "primary")]
    if r.empty or r.iloc[0]["n"] < 2:
        return None
    r = r.iloc[0]
    return bool(r["mean"] >= bar and r["lo"] > 0)


# ---------------------------------------------------------------- keyword arm
# A line that starts with a role label (optionally after table pipes / quote / heading / bold marks) and then a colon, a closing
# mark, a pipe or the end of the line, or that is only a label: base models switch into the assistant this way without any
# template marker, so the user portion ends there (those are switches, Cole's category, not user-voice traits).
ROLE_LABEL = re.compile(
    r"^[ \t]*[|>#*_\-]*[ \t]*(?:\*\*|__)?(?:chatgpt|assistant|ai|bot|gpt|claude|model)(?:[- ]?\d[\d.]*)?(?:\*\*|__)?[ \t]*(?::|：|\||$)",
    re.I | re.M,
)
MIN_WORDS = 15            # restricted version: samples with at least this many words of user text
HUMAN_WORDS = 80          # human baseline also on the first HUMAN_WORDS words (comparable to a generated continuation)


def cut_user_portion(text, markers, role_cut=False):
    """Continuation text before the first of the given marker strings (turn end, turn open, end-of-text, handoff markers) and,
    with role_cut, before the first line-initial role label (ChatGPT, Assistant, AI, Bot, GPT, Claude, Model)."""
    cut = min((text.find(m) for m in markers if m and m in text), default=len(text))
    t = text[:cut]
    if role_cut:
        m = ROLE_LABEL.search(t)
        if m:
            return t[: m.start()]
    return t


def keyword_counts(user_text, prefill):
    """Occurrences of the four judge-free features in the user portion; markdown only counts if the prefill has none of that kind."""
    return dict(
        stock_phrase=len(STOCK.findall(user_text)),
        markdown_heading=0 if HEADING.search(prefill) else len(HEADING.findall(user_text)),
        markdown_bullet=0 if BULLET.search(prefill) else len(BULLET.findall(user_text)),
        markdown_bold=0 if BOLD.search(prefill) else len(BOLD.findall(user_text)),
    )


def keyword_flags(user_text, prefill):
    return {k: v > 0 for k, v in keyword_counts(user_text, prefill).items()}


def keyword_samples(samples, markers, role_cut=True):
    """DataFrame, one row per sample: prefill_id, sample, user_text, user_chars, n_words, role_cut_hit, <feature>_n counts,
    <feature> flags, any_flag, any_n."""
    rows = []
    for r in samples:
        full = cut_user_portion(r["continuation"], markers)
        u = cut_user_portion(r["continuation"], markers, role_cut=role_cut)
        c = keyword_counts(u, r["prefill"])
        rows.append(dict(prefill_id=r["prefill_id"], sample=r["sample"], prefill=r["prefill"], user_text=u, user_chars=len(u),
                         n_words=len(u.split()), role_cut_hit=len(u) < len(full), **{k + "_n": v for k, v in c.items()},
                         **{k: v > 0 for k, v in c.items()}, any_flag=any(v > 0 for v in c.values()), any_n=sum(c.values())))
    return pd.DataFrame(rows)


def human_samples(prefills, texts, max_words=None):
    """The real remainder of each WildChat user turn (full text minus the prefill), one sample per prefill; optionally only its
    first max_words words. prefills: rows with id, text_id, prefill; texts: {text_id: full text}."""
    rows = []
    for p in prefills:
        full = texts[p["text_id"]]
        rest = full[len(p["prefill"]):] if full.startswith(p["prefill"]) else full.replace(p["prefill"], "", 1)
        if max_words:
            rest = " ".join(rest.split(" ")[:max_words])
        rows.append(dict(prefill_id=p["id"], sample=0, prefill=p["prefill"], continuation=rest))
    return rows


def per_prefill(d, metric):
    """Series indexed by prefill_id. metric 'share:<col>': share of the prefill's samples with the flag; 'share15:<col>': the same
    over samples with >= MIN_WORDS words (NaN if none); 'per100:<feature>': occurrences per 100 words, the prefill's samples
    pooled (NaN if no words)."""
    kind, col = metric.split(":")
    ids = d.prefill_id.unique()
    if kind == "share15":
        return d[d.n_words >= MIN_WORDS].groupby("prefill_id")[col].mean().astype(float).reindex(ids)
    if kind == "share":
        return d.groupby("prefill_id")[col].mean().astype(float)
    g = d.groupby("prefill_id")
    words = g.n_words.sum()
    return (100 * g[col + "_n"].sum() / words.where(words > 0)).astype(float)


METRICS = {"share": "share of samples flagged (pp)", "share15": f"share, samples with >= {MIN_WORDS} words (pp)",
           "per100": "occurrences per 100 words"}
FEATS = KEYWORD_FEATURES + ["any"]


def _col(metric, feat):
    return ("any_flag" if metric != "per100" else "any") if feat == "any" else feat


def keyword_table(flagged_by_stage, base="base", seed=SEED, n_boot=N_BOOT):
    """Per metric x feature: stage minus base paired per prefill (n = prefills defined in both, dropped = the rest)."""
    if base not in flagged_by_stage:
        raise SystemExit(f"no generations for the base stage {base!r}")
    rows = []
    for metric in METRICS:
        for feat in FEATS:
            tab = {s: per_prefill(d, f"{metric}:{_col(metric, feat)}") for s, d in flagged_by_stage.items()}
            for s in stage_order([x for x in flagged_by_stage if x != base]):
                rng = np.random.default_rng(seed)
                rows.append(dict(stage=s, metric=metric, feature=feat, **paired_change(tab, base, s, rng, n_boot)))
    return pd.DataFrame(rows)


def human_row(d):
    """Absolute values for a human baseline (no change from base): per-prefill mean of each metric."""
    out = []
    for metric in METRICS:
        for feat in FEATS:
            t = per_prefill(d, f"{metric}:{_col(metric, feat)}")
            out.append(dict(metric=metric, feature=feat, n=int(t.notna().sum()), mean=float(t.mean())))
    return pd.DataFrame(out)


def stage_summary(d):
    """Samples, samples cut by a role label, empty user portions, mean words, samples with >= MIN_WORDS words, and the pooled
    flag count per 100 words."""
    w = d.n_words.sum()
    return dict(n=len(d), cut=int(d.role_cut_hit.sum()), empty=int((d.user_chars == 0).sum()), words=float(d.n_words.mean()),
                long=int((d.n_words >= MIN_WORDS).sum()), pooled=float(100 * d.any_n.sum() / w) if w else float("nan"))


def flagged_examples(d, stage, k=3, min_words=MIN_WORDS):
    """Up to k flagged samples with >= min_words words of user text, for the report (first per prefill, spread over prefills)."""
    f = d[d.any_flag & (d.n_words >= min_words)].drop_duplicates("prefill_id")
    f = f.iloc[np.linspace(0, len(f) - 1, min(k, len(f))).astype(int)] if len(f) else f
    return [dict(stage=stage, prefill=r.prefill, text=" ".join(r.user_text.split())) for r in f.itertuples()]

def family_markers(family):
    """Cut strings for the family: end of user turn, turn open, end-of-text (all derived from its chat template by Cole's
    flags.markers_for, read-only import) plus its extra handoff markers from core.config. Needs the tokenizer (CPU, cached)."""
    from core.render import load_tokenizer
    from measures.vestigial_regression.flags import markers_for
    st = config.family_settings(family)
    tok = load_tokenizer(st["template"])
    m = markers_for(tok)
    return [m["end_of_turn"], m["turn_open"], tok.eos_token, m.get("handoff"), *st["handoff_extra"]]


# ---------------------------------------------------------------- reports
def pp(x):
    return "n/a" if pd.isna(x) else f"{100 * x:+.1f}"


def cell(r):
    return "n/a (fewer than 2 prefills)" if r["n"] < 2 else f"{pp(r['mean'])} [{pp(r['lo'])}, {pp(r['hi'])}]"


def judge_report(table, family, bar=FIRE_BAR, excluded=None, provisional=False):
    L = [f"# User-slot overgeneralization: judge arm, {family}", ""]
    if provisional:
        L += ["**PROVISIONAL**: Cole's judge labels (200 incomplete prefills x 3 samples) before the team's hand validation; invalid samples are excluded, not relabelled. See docs/decisions.md 2026-11-04.", ""]
    L += [
         "Stage minus base in percentage points, 95% bootstrap CI over prefills, paired per prefill. Primary = share of user-voice "
         "samples labelled user_voice_with_assistant_traits; secondary = the same count over all samples. Prefills with no "
         "user-voice sample in the base or in the stage are dropped from the primary row (count shown).", "",
         "| stage | arm | prefills used | dropped | base share (pp) | stage share (pp) | change (pp) |", "|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        L.append(f"| {r.stage} | {r.arm} | {r.n} | {r.dropped} | {pp(r.base_mean).lstrip('+')} | {pp(r.stage_mean).lstrip('+')} | {cell(r)} |")
    if excluded:
        L += ["", "Samples the judge refused (label invalid), excluded: " + ", ".join(f"{k} {v}" for k, v in excluded.items()) + f" (total {sum(excluded.values())})."]
    f = fires(table, family, bar=bar)
    if family != "olmo3-7b":
        verdict = "not applicable to this family (supporting only; no contamination check is possible for it)"
    elif f is None:
        verdict = "cannot be evaluated (no instruct stage judged or fewer than 2 usable prefills)"
    else:
        verdict = "**fires**" if f else "does not fire"
    L += ["", f"**Verdict rule (OLMo instruct, primary, >= {100 * bar:.0f} pp with CI excluding zero; the bar is fixed, "
          f"docs/decisions.md 2026-11-04):** {verdict}", ""]
    if family in SUPPORTING:
        L.append("Caveat: post-training data for this family is not open, so test-text contamination cannot be checked.")
    return "\n".join(L) + "\n"


def _rows(table, metric):
    sc = lambda x: "n/a" if pd.isna(x) else (f"{x:.2f}" if metric == "per100" else f"{100 * x:.1f}")
    L = ["| stage | feature | prefills used | dropped | base | stage | change [95% CI] |", "|---|---|---|---|---|---|---|"]
    for _, r in table[table.metric == metric].iterrows():
        ch = "n/a" if r["n"] < 2 else (f"{r['mean']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]" if metric == "per100" else cell(r))
        L.append(f"| {r.stage} | {r.feature} | {r.n} | {r.dropped} | {sc(r.base_mean)} | {sc(r.stage_mean)} | {ch} |")
    return L


def _human_lines(h, label):
    L = [f"Human baseline, {label} (absolute values, per-prefill mean):", "",
         f"| feature | share (pp) | share, >= {MIN_WORDS} words (pp) | per 100 words |", "|---|---|---|---|"]
    for f in FEATS:
        g = lambda m: h[(h.metric == m) & (h.feature == f)].iloc[0]["mean"]
        L.append(f"| {f} | {100 * g('share'):.1f} | {100 * g('share15'):.1f} | {g('per100'):.2f} |")
    return L


def keyword_report(new, old, family, human, human_short, examples=()):
    """new / old: dict(table, summary={stage: stage_summary}) for the current (role-label cut) and previous versions. human /
    human_short: human_row frames for the full real remainder and its first HUMAN_WORDS words."""
    L = [f"# User-slot assistant traits: keyword cross-check, {family} (outside the verdict)", "",
         "No judge. In the user portion of each continuation, flag assistant stock phrases and markdown headings, bullets or bold "
         "that the prefill does not already contain. Effects are stage minus base, paired per prefill, 95% bootstrap CI over prefills "
         "(percentage points for shares, occurrences per 100 words for the rate). Three views: share of samples flagged; the same "
         f"restricted to samples with >= {MIN_WORDS} words of user text (a model that closes the turn at once has nothing to flag); "
         "occurrences per 100 words of user text (per prefill, its samples pooled).", ""]
    if family in SUPPORTING:
        L += ["Caveat: no contamination check is possible for this family.", ""]
    L += ["## Current version", "",
          "The user portion ends at the first turn-end, turn-open, end-of-text or handoff marker, **or at a line-initial role label** "
          "(ChatGPT, Assistant, AI, Bot, GPT, Claude, Model; case-insensitive; optional colon, pipes or markdown; or a line that is "
          "only the label). Base models switch into the assistant this way with no template marker; those are switches, not "
          "user-voice traits.", "",
          "| stage | samples | cut by a role label | empty user portion | mean words | samples >= 15 words | pooled flags per 100 words |",
          "|---|---|---|---|---|---|---|"]
    L += [f"| {s} | {c['n']} | {c['cut']} | {c['empty']} | {c['words']:.0f} | {c['long']} | {c['pooled']:.2f} |" for s, c in new["summary"].items()]
    L += [""] + _human_lines(human, "real remainder of each WildChat user turn (full text minus the prefill; length not matched)") + [""]
    L += _human_lines(human_short, f"same, first {HUMAN_WORDS} words only") + [""]
    for metric, name in METRICS.items():
        L += [f"### {name}", ""] + _rows(new["table"], metric) + [""]
    if examples:
        L += ["### Examples still flagged (>= 15 words of user text)", ""]
        L += [f"- **{e['stage']}**, prefill \"{e['prefill'][:80]}\": {e['text'][:420]}" for e in examples] + [""]
    L += ["---", "", "## Previous version (cut at turn-end / handoff markers only; kept for the record)", "",
          "| stage | samples | empty user portion | mean words | samples >= 15 words | pooled flags per 100 words |", "|---|---|---|---|---|---|"]
    L += [f"| {s} | {c['n']} | {c['empty']} | {c['words']:.0f} | {c['long']} | {c['pooled']:.2f} |" for s, c in old["summary"].items()]
    L += ["", f"### Previous version: {METRICS['share']}", ""] + _rows(old["table"], "share") + [""]
    L += ["Read with care: stock phrases and markdown are coarse proxies for the judge's user_voice_with_assistant_traits label.", ""]
    return "\n".join(L) + "\n"

# ---------------------------------------------------------------- CLI
def main():
    config.family_from_argv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm", choices=["judge", "keyword"])
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--run", default="capstone")
    ap.add_argument("--generations", help="default results/<run>/<family>/core/generations")
    ap.add_argument("--judge-dir", help="default results/<run>/<family>/vestigial_regression/judge (<stage>.jsonl per stage)")
    ap.add_argument("--hand-labels", help="judge arm: directory of <stage>.jsonl hand labels (same schema) overriding the judge per (prefill_id, sample)")
    ap.add_argument("--out", help="default results/<run>/<family>/overgeneralization/user_traits_<arm>.md (+ .csv)")
    ap.add_argument("--prefills", default="data/prefill_100.jsonl", help="keyword arm: prefill set (human baseline)")
    ap.add_argument("--texts", default="data/texts.jsonl", help="keyword arm: full texts by id (human baseline)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    stages = list(config.family_settings(a.family)["stages"])
    out = a.out or os.path.join(config.RESULTS_ROOT, a.run, a.family, "overgeneralization", f"user_traits_{a.arm}.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if a.arm == "judge":
        jd = a.judge_dir or default_judge_dir(a.family, a.run)
        excluded = {}
        judged = load_judge(jd, stages, a.hand_labels, excluded)
        if "base" not in judged:
            raise SystemExit(f"no judge output for the base stage in {jd} (expected <stage>.jsonl from measures/vestigial_regression/judge.py)")
        table = judge_table(judged, n_boot=a.n_boot)
        text = judge_report(table, a.family, excluded=excluded, provisional=not a.hand_labels)
    else:
        gd = a.generations or generations_dir(a.family, a.run)
        markers = family_markers(a.family)
        raw = {s: read_jsonl(os.path.join(gd, f"{s}.jsonl")) for s in stages if os.path.exists(os.path.join(gd, f"{s}.jsonl"))}
        versions = {}
        for name, rc in (("new", True), ("old", False)):
            fl = {s: keyword_samples(r, markers, role_cut=rc) for s, r in raw.items()}
            versions[name] = dict(flagged=fl, table=keyword_table(fl, n_boot=a.n_boot), summary={s: stage_summary(d) for s, d in fl.items()})
        prefills = read_jsonl(a.prefills)
        texts = {r["id"]: r["text"] for r in read_jsonl(a.texts)}
        human = {k: keyword_samples(human_samples(prefills, texts, mw), [], role_cut=False) for k, mw in (("full", None), ("short", HUMAN_WORDS))}
        examples = []
        for s in [x for x in versions["new"]["flagged"] if x != "base"]:
            examples += flagged_examples(versions["new"]["flagged"][s], s, k=1)
        text = keyword_report(versions["new"], versions["old"], a.family, human_row(human["full"]), human_row(human["short"]), examples)
        table = pd.concat([versions["new"]["table"].assign(version="current"), versions["old"]["table"].assign(version="previous"),
                           human_row(human["full"]).assign(stage="human_full", version="current"),
                           human_row(human["short"]).assign(stage=f"human_first{HUMAN_WORDS}w", version="current")], ignore_index=True)
    open(out, "w", encoding="utf-8").write(text)
    table.to_csv(out[:-3] + ".csv", index=False)
    print(text)
    print("wrote", out)


if __name__ == "__main__":
    main()
