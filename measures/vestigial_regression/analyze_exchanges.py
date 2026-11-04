"""Vestigial regression's NLL arm on the main run's exchange scores (core.score_exchanges, context "exchange").

    uv run python -m measures.vestigial_regression.analyze_exchanges            # every family found under results/capstone
    uv run python -m measures.vestigial_regression.analyze_exchanges --root results/capstone --out results/capstone/vestigial_regression/exchanges_exchange

Reads results/<run>/<family>/<stage>/core/<tag>/scores.csv (one row per exchange x scored text x framing). Per family, corpus and
post-trained stage, over content positions 1+ (README, "Pre-registered criteria", revised 2026-11-03):

    frame cost FC(text)        NLL(tagged) - NLL(plain) of the same scored tokens, per exchange (user turn; reply after its user turn)
    change from base           FC(stage) - FC(base), paired per exchange
    asymmetry A                mean Δ FC(user) - mean Δ FC(reply); 95% CI with the user and reply sets resampled independently
    levels                     Δ user tagged, Δ user plain, Δ reply tagged, Δ reply plain (paired per exchange)

in nats per token (mean_nll_from1; the verdict, within a family) and bits per byte (summed NLL over positions 1+ / ln 2 / UTF-8 bytes
of those tokens; for comparisons across families). Verdict per checkpoint: A >= 0.05 nats with the CI excluding zero in both headline
corpora (oasst1, UltraChat), same direction; WildChat is secondary and reported only. Contamination can be checked for OLMo only.

Writes <out>/summary.csv (every quantity, unit, family, stage, corpus) and <out>/results.md.
"""

import argparse
import math
import os

import numpy as np
import pandas as pd

from core import config
from core.stats import bootstrap_ci
from measures.vestigial_regression.analyze import asymmetry_ci, fires

HEADLINE = ("oasst1", "ultrachat")
CORPORA = ("oasst1", "ultrachat", "wildchat")
UNITS = {"nats": "nats/token", "bpb": "bits/byte"}
LN2 = math.log(2)


def load_family(root, family, tag):
    """{stage: scores frame} for the family's stages that have scores, in config order."""
    out = {}
    for stage in config.family_settings(family)["stages"]:
        p = os.path.join(root, family, stage, "core", tag, "scores.csv")
        if os.path.exists(p):
            out[stage] = pd.read_csv(p)
    return out


def per_text(df):
    """Wide per (corpus, exchange, scored text): tagged and plain in nats/token and bits/byte (positions 1+), and truncation."""
    d = df.copy()
    d["nats"] = d.mean_nll_from1
    d["bpb"] = d.mean_nll_from1 * (d.n_tokens - 1) / LN2 / d.n_bytes_from1
    w = d.pivot_table(index=["corpus", "exchange_id", "scored_text"], columns="framing", values=["nats", "bpb"])
    w.columns = [f"{u}_{f}" for u, f in w.columns]
    for u in UNITS:
        w[f"{u}_fc"] = w[f"{u}_tagged"] - w[f"{u}_plain"]
    trunc = d[d.framing == "tagged"].set_index(["corpus", "exchange_id", "scored_text"]).truncated
    w["truncated"] = trunc.reindex(w.index).astype(bool)
    return w


def analyse_stage(base, post, n_boot, rng):
    """Rows (corpus, quantity, unit, mean, lo, hi, n) of one post-trained stage against the family's base."""
    b, p = per_text(base), per_text(post)
    idx = b.index.intersection(p.index)
    num = [c for c in p.columns if c != "truncated"]
    delta = p.loc[idx, num] - b.loc[idx, num]
    rows = []
    for corpus in CORPORA:
        if corpus not in delta.index.get_level_values(0):
            continue
        dc = delta.xs(corpus, level=0)
        user, reply = dc.xs("user", level=1), dc.xs("reply", level=1)
        for u in UNITS:
            m, lo, hi = asymmetry_ci(user[f"{u}_fc"], reply[f"{u}_fc"], n_boot, rng)
            rows.append(dict(corpus=corpus, quantity="asymmetry", unit=u, mean=m, lo=lo, hi=hi, n=f"{len(user)}/{len(reply)}"))
            for name, s in (("Δ FC user", user[f"{u}_fc"]), ("Δ FC reply", reply[f"{u}_fc"]),
                            ("Δ user tagged", user[f"{u}_tagged"]), ("Δ user plain", user[f"{u}_plain"]),
                            ("Δ reply tagged", reply[f"{u}_tagged"]), ("Δ reply plain", reply[f"{u}_plain"])):
                m, lo, hi = bootstrap_ci(s.dropna().to_numpy(float), n_boot, rng)
                rows.append(dict(corpus=corpus, quantity=name, unit=u, mean=m, lo=lo, hi=hi, n=str(s.notna().sum())))
    return rows


def levels(frames):
    """Descriptive levels per stage (not changes): mean frame cost of user and reply in nats/token, and the truncated-reply share."""
    rows = []
    for stage, df in frames.items():
        w = per_text(df)
        for corpus in CORPORA:
            if corpus not in w.index.get_level_values(0):
                continue
            wc = w.xs(corpus, level=0)
            r = wc.xs("reply", level=1)
            rows.append(dict(stage=stage, corpus=corpus, fc_user=wc.xs("user", level=1).nats_fc.mean(), fc_reply=r.nats_fc.mean(),
                             truncated_replies=f"{int(r.truncated.sum())}/{len(r)}"))
    return pd.DataFrame(rows)


def verdict(summary):
    """Per (family, stage): fires / direction in each headline corpus (nats), and the checkpoint's verdict."""
    out = []
    a = summary[(summary.quantity == "asymmetry") & (summary.unit == "nats")]
    for (fam, stage), g in a.groupby(["family", "stage"], sort=False):
        cells = {c: g[g.corpus == c].iloc[0] for c in HEADLINE if (g.corpus == c).any()}
        if len(cells) < len(HEADLINE):
            out.append(dict(family=fam, stage=stage, verdict="incomplete (a headline corpus is missing)"))
            continue
        f = {c: fires(r["mean"], r.lo, r.hi) for c, r in cells.items()}
        opposite = {c: bool(r["mean"] <= -config.MIN_EFFECT_NATS and r.hi < 0) for c, r in cells.items()}
        v = ("fires (both headline corpora)" if all(f.values()) else
             "does not fire; asymmetry negative in both headline corpora (opposite of regression)" if all(opposite.values()) else
             "does not fire")
        out.append(dict(family=fam, stage=stage, verdict=v, **{f"fires_{c}": f[c] for c in HEADLINE}))
    return pd.DataFrame(out)


def fmt(r):
    return f"{r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"


def markdown(summary, verdicts, level_tab, tag):
    lab = lambda fam, st: f"{config.family_settings(fam)['label']} {config.family_settings(fam)['stage_labels'].get(st, st)}"
    lines = ["# Vestigial regression, NLL arm: exchange design", "",
             f"Generated by `python -m measures.vestigial_regression.analyze_exchanges` from `results/capstone/<family>/<stage>/core/{tag}/`.",
             "Each post-trained checkpoint against its own family's base, content positions 1+, 95% bootstrap CIs (asymmetry: user and reply "
             "sets resampled independently; the rest paired per exchange). Frame cost = NLL(tagged) − NLL(plain). Headline corpora: oasst1 and "
             "UltraChat (matched authorship); WildChat is secondary. Nats per token within a family; bits per byte across families. "
             "Contamination could be checked for OLMo only: **Qwen3 and Gemma 4 results cannot exclude training-data overlap.**", "",
             "## Verdict (nats per token; fires if the asymmetry is at least 0.05 with the CI excluding zero in both oasst1 and UltraChat)", "",
             "| checkpoint | oasst1 | UltraChat | WildChat (secondary) | verdict |", "|---|---|---|---|---|"]
    a = summary[(summary.quantity == "asymmetry") & (summary.unit == "nats")]
    for _, v in verdicts.iterrows():
        g = a[(a.family == v.family) & (a.stage == v.stage)].set_index("corpus")
        cell = lambda c: fmt(g.loc[c]) if c in g.index else "—"
        lines.append(f"| {lab(v.family, v.stage)} | {cell('oasst1')} | {cell('ultrachat')} | {cell('wildchat')} | {v.verdict} |")
    for unit in UNITS:
        lines += ["", f"## All quantities ({UNITS[unit]})", ""]
        d = summary[summary.unit == unit].copy()
        d["checkpoint"] = [lab(f, s) for f, s in zip(d.family, d.stage)]
        d["cell"] = d.apply(fmt, axis=1)
        for corpus in CORPORA:
            dc = d[d.corpus == corpus]
            if dc.empty:
                continue
            wide = dc.pivot_table(index="quantity", columns="checkpoint", values="cell", aggfunc="first")
            order = ["asymmetry", "Δ FC user", "Δ FC reply", "Δ user tagged", "Δ user plain", "Δ reply tagged", "Δ reply plain"]
            wide = wide.reindex([q for q in order if q in wide.index])[list(dict.fromkeys(dc.checkpoint))]
            lines += [f"### {corpus}", "", wide.to_markdown(), ""]
    lines += ["## Levels (not changes): mean frame cost, nats per token, and truncated replies", "",
              level_tab.to_markdown(index=False, floatfmt="+.3f"), "",
              "Plain-text changes (`Δ user plain`, `Δ reply plain`) are the \"non-assistant degradation\" rows, referred to the "
              "overgeneralization measure. Replies longer than 300 tokens are scored over their first 300."]
    return "\n".join(lines) + "\n"


def run(root, tag, families, n_boot, seed):
    rng = np.random.default_rng(seed)
    rows, lv = [], []
    for fam in families:
        frames = load_family(root, fam, tag)
        if "base" not in frames:
            continue
        for stage, df in frames.items():
            if stage == "base":
                continue
            for r in analyse_stage(frames["base"], df, n_boot, rng):
                rows.append(dict(family=fam, stage=stage, **r))
        t = levels(frames)
        t.insert(0, "family", fam)
        lv.append(t)
    if not rows:
        raise SystemExit(f"no family under {root} has base and post-trained exchange scores ({tag})")
    summary = pd.DataFrame(rows)
    return summary, verdict(summary), pd.concat(lv, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(config.RESULTS_ROOT, config.EXCHANGE_RUN), help="results/<run>")
    ap.add_argument("--tag", default="exchanges_exchange")
    ap.add_argument("--families", nargs="+", default=list(config.FAMILIES))
    ap.add_argument("--out", help="default: <root>/vestigial_regression/<tag>")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args(argv)
    out = args.out or os.path.join(args.root, "vestigial_regression", args.tag)
    summary, verdicts, level_tab = run(args.root, args.tag, args.families, args.n_boot, args.seed)
    os.makedirs(out, exist_ok=True)
    summary.to_csv(os.path.join(out, "summary.csv"), index=False, float_format="%.6f")
    verdicts.to_csv(os.path.join(out, "verdicts.csv"), index=False)
    with open(os.path.join(out, "results.md"), "w", encoding="utf-8") as f:
        f.write(markdown(summary, verdicts, level_tab, args.tag))
    print(f"wrote {out}/results.md, summary.csv, verdicts.csv")


if __name__ == "__main__":
    main()
