"""User-turn (prefill) judge arm on the v2 label sets that had not been analysed (2026-11-06; labels released for analysis, PROVISIONAL: not human-validated; docs/decisions.md 2026-11-06). Same rule and code as user_traits.py:
share of user-voice samples labelled user_voice_with_assistant_traits (primary) and over all samples (secondary), stage minus base, paired per prefill, 95% bootstrap over prefills, bar +5 points with the range excluding 0.
Sets: results/capstone/<family>/overgeneralization/judge_incomplete_exchanges/judged_<stage>.jsonl (OLMo, Qwen, Llama, Apertus; 200 oasst1 + 200 UltraChat prefills, analysed SEPARATELY, never pooled) and Llama's 200-prefill WildChat set (judged_p200_<stage>.jsonl).
Everything here is supporting and was added after the OLMo 200-prefill verdict was known; that verdict is unchanged. Also the keyword cross-check on the same generations where it had not been run. No GPU, no API; only reads.
    python -m measures.overgeneralization.user_traits_exchanges
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from measures.overgeneralization import user_traits as ut

OUT = "results/user_turn_2026-11-06"
FAMS = ["olmo3-7b", "qwen3-8b", "llama3.1-8b", "apertus-8b"]
BAR = ut.FIRE_BAR


def read_judged(path):
    d = pd.DataFrame(ut.read_jsonl(path))[["prefill_id", "sample", "label"]]
    return d


def sets_for(fam):
    """{set name: (judge file pattern, generations dir, source filter)}"""
    base = f"results/capstone/{fam}/overgeneralization/judge_incomplete_exchanges"
    gens = f"results/capstone/{fam}/core/generations_incomplete_exchanges"
    out = {"oasst1 (exchange set)": (os.path.join(base, "judged_{s}.jsonl"), gens, "oasst1"), "ultrachat (exchange set)": (os.path.join(base, "judged_{s}.jsonl"), gens, "ultrachat")}
    if os.path.exists(os.path.join(base, "judged_p200_base.jsonl")):
        out["wildchat (200-prefill set)"] = (os.path.join(base, "judged_p200_{s}.jsonl"), f"results/capstone/{fam}/core/generations_incomplete", "incomplete-")
    return out


def stages_in(pattern):
    return [s for s in ["base", "sft", "dpo", "instruct"] if os.path.exists(pattern.format(s=s))]


def judge_rows():
    rows = []
    for fam in FAMS:
        for name, (pat, _, src) in sets_for(fam).items():
            stages = stages_in(pat)
            if "base" not in stages:
                continue
            judged, invalid = {}, {}
            for s in stages:
                d = read_judged(pat.format(s=s))
                d = d[d.prefill_id.str.contains(src)] if src != "incomplete-" else d
                invalid[s] = int((d.label == ut.INVALID).sum())
                judged[s] = d[d.label != ut.INVALID].reset_index(drop=True)
            tab = ut.judge_table(judged)
            for r in tab.itertuples():
                reading = "n/a (fewer than 2 prefills)" if r.n < 2 else ("REACHES THE BAR (+5 pp, range excluding 0)" if (r.mean >= BAR and r.lo > 0) else "below the bar")
                rows.append(dict(family=fam, prefill_set=name, stage=r.stage, arm=r.arm, prefills_used=r.n, dropped=r.dropped, base_pp=100 * r.base_mean, stage_pp=100 * r.stage_mean,
                                 change_pp=100 * r.mean, lo_pp=100 * r.lo, hi_pp=100 * r.hi, reading=reading, invalid_base=invalid["base"], invalid_stage=invalid[r.stage]))
    return pd.DataFrame(rows)


def keyword_rows():
    rows = []
    for fam in ["olmo3-7b", "qwen3-8b", "llama3.1-8b", "apertus-8b"]:
        import measures.overgeneralization.user_traits as u
        if fam == "apertus-8b" or fam == "llama3.1-8b":
            from extras import trait_atlas_breadth  # noqa: F401  (Apertus turn markers; Llama template settings)
        try:
            markers = u.family_markers(fam)
        except Exception:
            from core import config as _c  # noqa
            raise
        ss = dict(sets_for(fam))
        if "wildchat (200-prefill set)" not in ss and os.path.isdir(f"results/capstone/{fam}/core/generations_incomplete"):
            ss["wildchat (200-prefill set)"] = (None, f"results/capstone/{fam}/core/generations_incomplete", "incomplete-")   # keyword only: no judge labels for this set
        for name, (_, gens, src) in ss.items():
            if not os.path.isdir(gens):
                continue
            raw = {}
            for s in ["base", "sft", "dpo", "instruct"]:
                p = os.path.join(gens, f"{s}.jsonl")
                if os.path.exists(p):
                    r_ = ut.read_jsonl(p)
                    raw[s] = [x for x in r_ if (src in x["prefill_id"] or src == "incomplete-")]
            if "base" not in raw:
                continue
            fl = {s: ut.keyword_samples(r_, markers, role_cut=True) for s, r_ in raw.items()}
            tab = ut.keyword_table(fl)
            for r in tab[(tab.metric == "share") & (tab.feature == "any")].itertuples():
                rows.append(dict(family=fam, prefill_set=name, stage=r.stage, prefills_used=r.n, dropped=r.dropped, base_pp=100 * r.base_mean, stage_pp=100 * r.stage_mean,
                                 change_pp=100 * r.mean, lo_pp=100 * r.lo, hi_pp=100 * r.hi, metric="share of samples with any keyword flag"))
            for r in tab[(tab.metric == "per100") & (tab.feature == "any")].itertuples():
                rows.append(dict(family=fam, prefill_set=name, stage=r.stage, prefills_used=r.n, dropped=r.dropped, base_pp=r.base_mean, stage_pp=r.stage_mean,
                                 change_pp=r.mean, lo_pp=r.lo, hi_pp=r.hi, metric="flags per 100 words (not pp)"))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    J = judge_rows()
    J.to_csv(os.path.join(OUT, "user_traits_exchanges_judge.csv"), index=False)
    K = keyword_rows()
    K.to_csv(os.path.join(OUT, "user_traits_exchanges_keyword.csv"), index=False)
    print(J[J.arm == "primary"][["family", "prefill_set", "stage", "prefills_used", "dropped", "base_pp", "stage_pp", "change_pp", "lo_pp", "hi_pp", "reading"]].round(1).to_string())
    print(K.round(1).head(20).to_string())


if __name__ == "__main__":
    main()
