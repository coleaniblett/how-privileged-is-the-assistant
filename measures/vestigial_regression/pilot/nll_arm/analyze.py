"""README slot-effect quantities for Gemma 4 31B-it, against OLMo 3 7B base as the no-regression reference.

    python -m measures.vestigial_regression.pilot.nll_arm.analyze     # from the repo root, no GPU

Per text set (wildchat user, oasst1 user, wildchat assistant) and model, over content positions 1+ (mean_nll_from1):
    slot effect    NLL(user slot) - NLL(assistant slot): user@user - user@asst  /  asst@user - asst@asst
    role-specific  NLL(own slot) - NLL(plain):           user@user - user@plain /  asst@asst - asst@plain
with 95% bootstrap CIs over texts (core.stats.bootstrap_ci). Gemma-it's change from the OLMo base reference is the
difference of the two means (unpaired: different models, so raw NLL levels are not comparable, but these are
within-text differences) with a two-sample bootstrap CI; an effect counts if |change| >= config.MIN_EFFECT_NATS and the
CI excludes zero (AGENTS.md). Reference rows come from the frozen pre-capstone held-out run, results/scores/ (base and
instruct), read only. Gemma is scored on the first N texts per set only (see score.py).

Direction (README, criteria 1 and 2): vestigial regression for user text = the role-specific term rises, and the slot
effect shrinks to zero or flips positive, relative to base.
"""

import os

import numpy as np
import pandas as pd

from core import config
from core.stats import bootstrap_ci
from measures.vestigial_regression.pilot.nll_arm.score import SETS

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(config.RESULTS_ROOT, "scores")
MODELS = [("OLMo base (reference)", REF, "base"), ("OLMo instruct", REF, "instruct"),
          ("Gemma 4 31B-it", os.path.join(HERE, "scores"), "instruct")]
N_BOOT = 5000


def per_text(df, source, kind):
    """{condition abbr: Series indexed by text id} of mean_nll_from1 for one text set."""
    d = df[(df.source == source) & (df.kind == kind)]
    return {c.split("@")[1]: g.set_index("text_id").mean_nll_from1 for c, g in d.groupby("condition")}


def quantities(pt, kind):
    own, other = ("user", "asst") if kind == "user" else ("asst", "user")
    if kind == "user":
        return {"slot effect (user@user - user@asst)": pt["user"] - pt["asst"],
                "role-specific (user@user - user@plain)": pt["user"] - pt["plain"]}
    return {"slot effect (asst@user - asst@asst)": pt["user"] - pt["asst"],
            "role-specific (asst@asst - asst@plain)": pt["asst"] - pt["plain"]}


def ci(x, rng):
    m, lo, hi = bootstrap_ci(x.dropna().to_numpy(), N_BOOT, rng)
    return m, lo, hi


def diff_ci(a, b, rng):
    """Mean(a) - mean(b), two-sample percentile bootstrap CI."""
    a, b = a.dropna().to_numpy(), b.dropna().to_numpy()
    d = (a[rng.integers(0, len(a), (N_BOOT, len(a)))].mean(1) - b[rng.integers(0, len(b), (N_BOOT, len(b)))].mean(1))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return a.mean() - b.mean(), lo, hi


def main():
    rng = np.random.default_rng(config.SEED)
    frames = {n: pd.read_csv(os.path.join(d, f"{s}.csv")) for n, d, s in MODELS}
    L = ["# NLL arm: Gemma 4 31B-it against OLMo base", "",
         "Mean NLL differences in nats over content positions 1+, 95% bootstrap CI over texts. "
         f"Gemma: first {{n}} texts per set (fp32, CPU offload); OLMo: all texts in the held-out run. "
         "Change = Gemma - OLMo base; counts if |change| >= 0.05 and the CI excludes zero.", ""]
    rows, ns = [], {}
    for source, kind in SETS:
        pts = {n: quantities(per_text(f, source, kind), kind) for n, f in frames.items()}
        ns[(source, kind)] = len(next(iter(pts["Gemma 4 31B-it"].values())))
        L += [f"## {source} {kind}", "", "| quantity | OLMo base | OLMo instruct | Gemma-it | Gemma-it - OLMo base |",
              "|---|---|---|---|---|"]
        for q in pts["Gemma 4 31B-it"]:
            cells = []
            for n, _, _ in MODELS:
                m, lo, hi = ci(pts[n][q], rng)
                cells.append(f"{m:.3f} [{lo:.3f}, {hi:.3f}]")
            m, lo, hi = diff_ci(pts["Gemma 4 31B-it"][q], pts["OLMo base (reference)"][q], rng)
            counts = abs(m) >= config.MIN_EFFECT_NATS and (lo > 0 or hi < 0)
            cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] **{'counts' if counts else 'negligible/unclear'}**")
            L.append(f"| {q} | " + " | ".join(cells) + " |")
            rows.append(dict(set=f"{source} {kind}", quantity=q, change=m, lo=lo, hi=hi, counts=counts))
        L.append("")
    L[2] = L[2].replace("{n}", str(min(ns.values())))
    u = [r for r in rows if r["set"].endswith("user")]
    L += ["## Reading against README criteria 1 and 2 (user text)", ""]
    for r in u:
        want = ("rises (change > 0)" if r["quantity"].startswith("role-specific")
                else "shrinks to zero or flips positive (change > 0)")
        ok = r["counts"] and r["change"] > 0
        L.append(f"- {r['set']}, {r['quantity']}: expected under vestigial regression: {want}; "
                 f"observed {r['change']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] -> {'consistent' if ok else 'not shown'}")
    with open(os.path.join(HERE, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
