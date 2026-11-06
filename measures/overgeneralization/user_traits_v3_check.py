"""Cole's judge v3 `assistant_traits` field on the open-opener generations, read-only (2026-11-06). EXPLORATORY and PROVISIONAL: v3 is not validated, the 5 open openers are not the pre-registered prefills.
Input: results/capstone/<family>/vestigial_regression/judge_open5/<stage>.jsonl (Cole's files; not modified). For each family and stage: samples, samples whose user_text is "user" (user text present), share of those with assistant_traits true, the v3 label counts,
change from base with a 95% bootstrap that resamples samples WITHIN each opener (so it ignores spread between the 5 openers), and per opener. A stage with no user text (user_text never "user") is said so. No GPU, no API, core/ untouched.
    python -m measures.overgeneralization.user_traits_v3_check        # -> results/capstone/overgeneralization/user_turn_2026-11-06/user_traits_v3.csv (+ prints the markdown table)
"""
import glob
import os

import numpy as np
import pandas as pd

OUT = "results/capstone/overgeneralization/user_turn_2026-11-06"
N_BOOT, SEED = 10000, 0
ORDER = ["base", "sft", "dpo", "instruct"]


def load():
    rows = []
    for p in sorted(glob.glob("results/capstone/*/vestigial_regression/judge_open5/*.jsonl")):
        fam, stage = p.split(os.sep)[2], os.path.basename(p)[:-6]
        d = pd.read_json(p, lines=True)
        d["family"], d["stage"] = fam, stage
        rows.append(d[["family", "stage", "prefill_id", "sample", "label", "user_text", "assistant_traits"]])
    return pd.concat(rows, ignore_index=True)


def usable(d):
    """samples with user text and a defined assistant_traits field (nan = the judge gave none)"""
    x = d[d.user_text == "user"].copy()
    x["at"] = x.assistant_traits.map({True: 1.0, False: 0.0, 1.0: 1.0, 0.0: 0.0})
    return x[x["at"].notna()]


def boot_change(b, s, rng):
    """stage minus base share, bootstrap resampling samples within opener (strata), 100 x points"""
    openers = sorted(set(b.prefill_id) & set(s.prefill_id))
    bb = [b[b.prefill_id == o]["at"].to_numpy() for o in openers]
    ss = [s[s.prefill_id == o]["at"].to_numpy() for o in openers]
    pool = lambda arr, idx: np.concatenate([a[i] for a, i in zip(arr, idx)])
    est = 100 * (np.concatenate(ss).mean() - np.concatenate(bb).mean())
    out = np.empty(N_BOOT)
    for k in range(N_BOOT):
        out[k] = 100 * (pool(ss, [rng.integers(0, len(a), len(a)) for a in ss]).mean() - pool(bb, [rng.integers(0, len(a), len(a)) for a in bb]).mean())
    return est, *np.percentile(out, [2.5, 97.5])


def main():
    D = load()
    rng = np.random.default_rng(SEED)
    rows, opener_rows = [], []
    for fam in sorted(D.family.unique()):
        F = D[D.family == fam]
        stages = [s for s in ORDER if s in set(F.stage)]
        base = usable(F[F.stage == "base"]) if "base" in stages else None
        for st in stages:
            S = F[F.stage == st]
            u = usable(S)
            counts = S.label.value_counts().to_dict()
            r = dict(family=fam, stage=st, samples=len(S), user_text_samples=int((S.user_text == "user").sum()), usable=len(u), share_at=100 * u["at"].mean() if len(u) else np.nan,
                     n_at=int(u["at"].sum()), labels="; ".join(f"{k} {v}" for k, v in counts.items()), note="")
            if len(u) == 0:
                r["note"] = "NO USER TEXT in this stage (user_text never 'user'): nothing to read"
            if st != "base" and base is not None and len(u) and len(base):
                m, lo, hi = boot_change(base, u, rng)
                r.update(change=m, lo=lo, hi=hi)
            rows.append(r)
            for o, g in u.groupby("prefill_id"):
                bo = base[base.prefill_id == o]["at"].mean() * 100 if base is not None and (base.prefill_id == o).any() else np.nan
                opener_rows.append(dict(family=fam, stage=st, opener=o, usable=len(g), share_at=100 * g["at"].mean(), base_share=bo))
    R = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    R.to_csv(os.path.join(OUT, "user_traits_v3.csv"), index=False)
    pd.DataFrame(opener_rows).to_csv(os.path.join(OUT, "user_traits_v3_per_opener.csv"), index=False)
    L = ["| family | stage | samples | with user text | usable (assistant_traits defined) | with assistant_traits | share (%) | change from base (pp) [95% CI, resampling within opener] | v3 labels | note |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in R.itertuples():
        ch = f"{r.change:+.1f} [{r.lo:+.1f}, {r.hi:+.1f}]" if hasattr(r, "change") and r.change == r.change else "-"
        L.append(f"| {r.family} | {r.stage} | {r.samples} | {r.user_text_samples} | {r.usable} | {r.n_at} | {'-' if r.share_at != r.share_at else f'{r.share_at:.1f}'} | {ch} | {r.labels} | {r.note} |")
    open(os.path.join(OUT, "user_traits_v3_table.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
