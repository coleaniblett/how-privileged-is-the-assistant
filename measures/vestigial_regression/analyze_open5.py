"""Vestigial regression judge arm, open-opener condition (measure README, "Judge arm, open-opener condition"; judge prompt v3).

    python -m measures.vestigial_regression.analyze_open5      # -> results/capstone/vestigial_regression/judge_open5/

Reads results/capstone/<family>/vestigial_regression/judge_open5/<stage>.jsonl (judge.py v3 rows: label, subtype, group). Per
post-trained checkpoint against its family's base:
  - rate of each headline group (judge.v3_group) per opener, over samples with a label (invalid = judge refused, unclear: excluded
    and counted);
  - non-user-voice rate change from base per opener, and pooled (openers equally weighted), with a 95% bootstrap CI resampling
    samples within each opener, base and checkpoint independently (different samples), openers fixed;
  - verdict (draft rule frozen with the labels, 2026-11-05): fires if the pooled change has its CI above zero and the change is
    positive at 4 or more of the 5 openers. The minimum rate difference is still OPEN, so it is not applied.
Refused samples are excluded (primary) and, as two other readings, counted as user voice or as non-user voice.
The separately reported rates (completed_then_assistant; closed_immediately for open openers) get the same pooled change and CI.
"""

import json
import os

import numpy as np
import pandas as pd

from core import config

ROOT = "results/capstone"
OUT = os.path.join(ROOT, "vestigial_regression", "judge_open5")
FAMILIES = ["olmo3-7b", "qwen3-8b", "gemma4-31b"]
OPENERS = [json.loads(line) for line in open("data/prefill_open5.jsonl", encoding="utf-8")]
OPENER_IDS = [o["id"] for o in OPENERS]
N_BOOT, SEED = 10_000, config.SEED
READINGS = {"exclude": "refused excluded", "as_user": "refused counted as user voice", "as_nonuser": "refused counted as non-user voice",
            "v2_map": "v2 mapping (completed-then-assistant unclosed = switched, closed = user voice; closed at once = user voice)"}
GROUPS = ["user_voice", "non_user_voice", "completed_then_assistant", "closed_immediately"]


def load(fam, stage):
    p = os.path.join(ROOT, fam, "vestigial_regression", "judge_open5", f"{stage}.jsonl")
    return pd.DataFrame([json.loads(line) for line in open(p, encoding="utf-8")]) if os.path.exists(p) else None


def indicator(df, group, reading):
    """Per sample: 1/0 for `group`, NaN where excluded. Refused rows follow the reading; unclear rows are always excluded."""
    if reading == "v2_map":  # sensitivity: the v2 counterpart of each v3 label (judge.v3_to_v2); refused and unmappable excluded
        v2 = df["v2_equivalent"]
        x = v2.isin(["switched_to_assistant", "degenerate"]).astype(float)
        if group == "user_voice":
            x = v2.isin(["user_voice", "user_voice_with_assistant_traits"]).astype(float)
        x[v2.isna() | (v2 == "invalid")] = np.nan
        return x
    x = (df["group"] == group).astype(float)
    x[df["label"] == "unclear"] = np.nan
    inv = df["label"] == "invalid"
    if reading == "exclude":
        x[inv] = np.nan
    else:
        x[inv] = float((group == "non_user_voice") == (reading == "as_nonuser")) if group in ("user_voice", "non_user_voice") else 0.0
    return x


def change(base, post, group, reading, rng):
    """Per opener and pooled: (rate base, rate post, change, lo, hi)."""
    per, boots = {}, []
    for oid in OPENER_IDS:
        b = indicator(base[base.prefill_id == oid], group, reading).dropna().to_numpy()
        p = indicator(post[post.prefill_id == oid], group, reading).dropna().to_numpy()
        bb = b[rng.integers(0, len(b), (N_BOOT, len(b)))].mean(1)
        pb = p[rng.integers(0, len(p), (N_BOOT, len(p)))].mean(1)
        d = pb - bb
        boots.append(d)
        lo, hi = np.percentile(d, [2.5, 97.5])
        per[oid] = dict(base=b.mean(), post=p.mean(), change=p.mean() - b.mean(), lo=lo, hi=hi, n_base=len(b), n_post=len(p))
    pooled = np.mean(boots, axis=0)
    lo, hi = np.percentile(pooled, [2.5, 97.5])
    per["pooled"] = dict(base=np.mean([per[o]["base"] for o in OPENER_IDS]), post=np.mean([per[o]["post"] for o in OPENER_IDS]),
                         change=np.mean([per[o]["change"] for o in OPENER_IDS]), lo=lo, hi=hi,
                         n_base=sum(per[o]["n_base"] for o in OPENER_IDS), n_post=sum(per[o]["n_post"] for o in OPENER_IDS))
    return per


def verdict(per):
    rising = sum(per[o]["change"] > 0 for o in OPENER_IDS)
    falling = sum(per[o]["change"] < 0 for o in OPENER_IDS)
    fires = bool(per["pooled"]["lo"] > 0 and rising >= 4)
    opposite = bool(per["pooled"]["hi"] < 0 and falling >= 4)
    return dict(fires=fires, rising=rising, falling=falling,
                result="fires" if fires else "does not fire; falls (CI below zero, 4+ openers)" if opposite else "does not fire")


def main():
    rng = np.random.default_rng(SEED)
    rows, verdicts, counts = [], [], []
    for fam in FAMILIES:
        stages = list(config.family_settings(fam)["stages"])
        frames = {s: load(fam, s) for s in stages}
        for s, df in frames.items():
            if df is not None:
                counts.append(dict(family=fam, stage=s, n=len(df), invalid=int((df.label == "invalid").sum()),
                                   unclear=int((df.label == "unclear").sum()), **{g: int((df.group == g).sum()) for g in GROUPS}))
        base = frames["base"]
        for s in stages[1:]:
            post = frames[s]
            for reading in READINGS:
                for group in GROUPS if reading == "exclude" else ["non_user_voice"]:
                    per = change(base, post, group, reading, rng)
                    for oid, c in per.items():
                        rows.append(dict(family=fam, stage=s, reading=reading, group=group, opener=oid, **c))
                    if group == "non_user_voice":
                        verdicts.append(dict(family=fam, stage=s, reading=reading, **verdict(per)))
    summary, vt, ct = pd.DataFrame(rows), pd.DataFrame(verdicts), pd.DataFrame(counts)
    os.makedirs(OUT, exist_ok=True)
    summary.to_csv(os.path.join(OUT, "summary.csv"), index=False, float_format="%.4f")
    vt.to_csv(os.path.join(OUT, "verdicts.csv"), index=False)
    ct.to_csv(os.path.join(OUT, "counts.csv"), index=False)
    open(os.path.join(OUT, "results.md"), "w", encoding="utf-8").write(markdown(summary, vt, ct))
    print(f"wrote {OUT}/results.md, summary.csv, verdicts.csv, counts.csv")
    print(vt[vt.reading == "exclude"].to_string(index=False))


def lab(fam, s):
    st = config.family_settings(fam)
    return f"{st['label']} {st['stage_labels'].get(s, s)}"


def pct(c):
    return f"{100 * c['change']:+.1f} [{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}]"


def markdown(summary, vt, ct):
    text = {o["id"]: o["prefill"] for o in OPENERS}
    L = ["# Vestigial regression judge arm: open-opener condition", "",
         "Generated by `python -m measures.vestigial_regression.analyze_open5` from `results/capstone/<family>/vestigial_regression/judge_open5/` "
         "(judge prompt v3, `claude-sonnet-5-5`). Five complete-but-open user openers, 120 samples each per checkpoint. Each post-trained "
         "checkpoint against its family's base. Rates in percentage points; 95% bootstrap CIs resample samples within each opener (base and "
         "checkpoint independently); the openers are fixed conditions, so conclusions are about these five. Pooled = openers equally weighted.",
         "", "Rule (README, frozen with the labels 2026-11-05): fires if the pooled non-user-voice rate change has its CI above zero and the "
         "change is positive at 4+ of 5 openers; the minimum rate difference is OPEN and not applied. `closed_immediately` is reported on its "
         "own for open openers, a rule set **after** automated counts were seen (trained OLMo closes \"Can I tell you something?\" at once). "
         "The generations were judged blind; the condition's design was motivated by earlier data. **Contamination can be checked for OLMo "
         "only.** The judge is not yet validated on v3 (a fresh blind sheet is pending).", "",
         "## Verdict: non-user-voice rate change from base (refused samples excluded)", "",
         "| checkpoint | pooled change, pts [95% CI] | openers rising | verdict | refused = user voice | refused = non-user voice | v2 mapping (sensitivity) |",
         "|---|---|---|---|---|---|---|"]
    for (fam, s), g in vt.groupby(["family", "stage"], sort=False):
        r = {x.reading: x for x in g.itertuples()}
        cell = lambda rd: pct(summary[(summary.family == fam) & (summary.stage == s) & (summary.reading == rd)
                                      & (summary.group == "non_user_voice") & (summary.opener == "pooled")].iloc[0])
        L.append(f"| {lab(fam, s)} | {cell('exclude')} | {r['exclude'].rising}/5 | {r['exclude'].result} | "
                 f"{cell('as_user')}, {r['as_user'].result} | {cell('as_nonuser')}, {r['as_nonuser'].result} | "
                 f"{cell('v2_map')}, {r['v2_map'].rising}/5 rising, {r['v2_map'].result} |")
    L += ["", "The v2-mapping column is a sensitivity reading, not the rule: it counts a model that finishes the user's message and then "
          "answers without closing the turn as switched (v2's reading), and closing at once on an open opener as user voice."]
    for group, title in (("non_user_voice", "Non-user voice (switched + degenerate)"),
                         ("completed_then_assistant", "Completed the user's message, then the assistant took over (reported on its own)"),
                         ("closed_immediately", "Closed the turn at once (reported on its own for open openers)"),
                         ("user_voice", "User voice (with or without assistant traits)")):
        L += ["", f"## {title}: rate at base → checkpoint, and change (pts)", "",
              "| checkpoint | " + " | ".join(f"\"{text[o]}\"" for o in OPENER_IDS) + " | pooled |", "|---|" + "---|" * (len(OPENER_IDS) + 1)]
        d = summary[(summary.reading == "exclude") & (summary.group == group)]
        for (fam, s), g in d.groupby(["family", "stage"], sort=False):
            g = g.set_index("opener")
            L.append(f"| {lab(fam, s)} | " + " | ".join(
                f"{100 * g.loc[o, 'base']:.0f}→{100 * g.loc[o, 'post']:.0f}, {pct(g.loc[o])}" for o in OPENER_IDS + ["pooled"]) + " |")
    L += ["", "## Counts per checkpoint (of 600)", "", ct.to_markdown(index=False), "",
          "`invalid`: the judge refused (API refusal, mostly `reasoning_extraction`); `unclear`: inconsistent judge fields. Both are excluded "
          "from the rates in the primary reading."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
