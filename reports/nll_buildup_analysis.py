"""The vestigial regression NLL arm's asymmetry, built up step by step from raw NLL (exchange design, main run).

    python -m reports.nll_buildup_analysis          # -> reports/nll_buildup.json (every number reports/nll_buildup.html shows)

Analysis only: changes no measure's code, criteria or data. The per-text quantities come from the measure's own code
(measures.vestigial_regression.analyze_exchanges: load_family, per_text, run); nothing is re-implemented. This script adds:
  - level means (raw NLL and frame cost at base and at the checkpoint), with descriptive 95% bootstrap CIs over texts;
  - a reproduction check: AX.run's summary against reports/capstone_results.json and the committed summary.csv, and the
    build-up identities (mean of per-text changes = summary mean; Δ FC = Δ tagged − Δ plain; asymmetry = Δ FC user − Δ FC reply);
  - robustness: the asymmetry from tagged NLL only and from plain NLL only, with the measure's own asymmetry_ci
    (user and reply sets resampled independently). Identity: frame-cost asymmetry = tagged-only − plain-only.
Units: nats per token over content positions 1+ (mean_nll_from1), the verdict's unit.
"""

import json
import os

import numpy as np
import pandas as pd

from core import config
from core.stats import bootstrap_ci
from measures.vestigial_regression import analyze_exchanges as AX
from measures.vestigial_regression.analyze import asymmetry_ci

ROOT = "results/capstone"
TAG = "exchanges_exchange"
N_BOOT = 10_000
SEED = config.SEED
FLOOR = config.MIN_EFFECT_NATS
CORPORA = ["oasst1", "ultrachat", "wildchat"]
HEADLINE = ["oasst1", "ultrachat"]
SIDES = ["user", "reply"]
CAPSTONE = "reports/capstone_results.json"
OUT = "reports/nll_buildup.json"


def c3(m, lo, hi, n=None):
    d = dict(mean=float(m), lo=float(lo), hi=float(hi))
    if n is not None:
        d["n"] = n
    return d


def fires(c):
    return bool(c["mean"] >= FLOOR and c["lo"] > 0)


def opposite(c):
    return bool(c["mean"] <= -FLOOR and c["hi"] < 0)


def verdict(cells):
    a = [cells[k] for k in HEADLINE]
    if all(fires(x) for x in a):
        return "fires (both headline corpora)"
    if all(opposite(x) for x in a):
        return "does not fire; negative in both headline corpora (opposite of regression)"
    return "does not fire"


def main():
    # 1. the measure's own summary (config.FAMILIES order: the bootstrap RNG is shared across families, as in analyze_exchanges)
    summary, verdicts, _ = AX.run(ROOT, TAG, list(config.FAMILIES), N_BOOT, SEED)
    S = summary[summary.unit == "nats"].set_index(["family", "stage", "corpus", "quantity"])
    cap = json.load(open(CAPSTONE))["vr_nll"]
    committed = pd.read_csv(os.path.join(ROOT, "vestigial_regression", TAG, "summary.csv"))
    m = summary.merge(committed, on=["family", "stage", "corpus", "quantity", "unit"], suffixes=("", "_c"))
    diff_committed = float(max((m[c] - m[c + "_c"]).abs().max() for c in ("mean", "lo", "hi")))
    diff_capstone = 0.0
    n_capstone = 0
    for (f, s, corp, q), r in S.iterrows():
        cc = cap["cells"][f"{f}|{s}"][corp]["nats"][q]
        diff_capstone = max(diff_capstone, *(abs(r[k] - cc[k]) for k in ("mean", "lo", "hi")))
        n_capstone += 1

    rng = np.random.default_rng(SEED)  # levels and robustness: its own stream, so the measure's numbers above are untouched
    out_cells, identity = {}, dict(mean_of_changes=0.0, fc_is_tagged_minus_plain=0.0, asym_is_user_minus_reply=0.0,
                                   level_diff_is_change=0.0, fc_asym_is_tagged_minus_plain_asym=0.0)
    for fam in config.FAMILIES:
        frames = AX.load_family(ROOT, fam, TAG)
        base = AX.per_text(frames["base"])
        for stage in [s for s in frames if s != "base"]:
            post = AX.per_text(frames[stage])
            idx = base.index.intersection(post.index)
            for corp in CORPORA:
                if corp not in idx.get_level_values(0):
                    continue
                cell = dict(n={}, levels={}, changes={}, measure={}, robust={})
                deltas = {}
                for side in SIDES:
                    sel = [i for i in idx if i[0] == corp and i[2] == side]
                    b, p = base.loc[sel], post.loc[sel]
                    cell["n"][side] = len(sel)
                    lv = {}
                    for col in ("tagged", "plain", "fc"):
                        for nm, fr in (("base", b), ("post", p)):
                            lv[f"{col}_{nm}"] = c3(*bootstrap_ci(fr[f"nats_{col}"].dropna().to_numpy(float), N_BOOT, rng))
                    cell["levels"][side] = lv
                    d = {col: (p[f"nats_{col}"] - b[f"nats_{col}"]) for col in ("tagged", "plain", "fc")}
                    deltas[side] = d
                    # identities (means): level difference = mean change; Δ FC = Δ tagged − Δ plain; mean change = summary mean
                    for col in ("tagged", "plain", "fc"):
                        identity["level_diff_is_change"] = max(identity["level_diff_is_change"],
                                                               abs(lv[f"{col}_post"]["mean"] - lv[f"{col}_base"]["mean"] - d[col].mean()))
                    identity["fc_is_tagged_minus_plain"] = max(identity["fc_is_tagged_minus_plain"],
                                                               float((d["fc"] - (d["tagged"] - d["plain"])).abs().max()))
                    qn = {"tagged": f"Δ {side} tagged", "plain": f"Δ {side} plain", "fc": f"Δ FC {side}"}
                    cell["changes"][side] = {}
                    for col, q in qn.items():
                        r = S.loc[(fam, stage, corp, q)]
                        identity["mean_of_changes"] = max(identity["mean_of_changes"], abs(d[col].mean() - r["mean"]))
                        cell["changes"][side][col] = c3(r["mean"], r.lo, r.hi, int(r.n))
                a = S.loc[(fam, stage, corp, "asymmetry")]
                cell["measure"]["asymmetry"] = c3(a["mean"], a.lo, a.hi, a.n)
                identity["asym_is_user_minus_reply"] = max(identity["asym_is_user_minus_reply"], abs(
                    a["mean"] - (cell["changes"]["user"]["fc"]["mean"] - cell["changes"]["reply"]["fc"]["mean"])))
                for col in ("tagged", "plain"):
                    cell["robust"][f"asym_{col}"] = c3(*asymmetry_ci(deltas["user"][col], deltas["reply"][col], N_BOOT, rng))
                identity["fc_asym_is_tagged_minus_plain_asym"] = max(identity["fc_asym_is_tagged_minus_plain_asym"], abs(
                    a["mean"] - (cell["robust"]["asym_tagged"]["mean"] - cell["robust"]["asym_plain"]["mean"])))
                out_cells.setdefault(f"{fam}|{stage}", {})[corp] = cell

    verdict_out = {}
    for key, cells in out_cells.items():
        fc = {k: cells[k]["measure"]["asymmetry"] for k in HEADLINE}
        tg = {k: cells[k]["robust"]["asym_tagged"] for k in HEADLINE}
        verdict_out[key] = dict(frame_cost=verdict(fc), tagged_only=verdict(tg),
                                fires_frame_cost={k: fires(fc[k]) for k in HEADLINE}, fires_tagged_only={k: fires(tg[k]) for k in HEADLINE},
                                capstone=cap["verdict"][key]["reading"])
    res = dict(cells=out_cells, verdict=verdict_out,
               reproduction=dict(rows_nats=n_capstone, max_abs_diff_vs_capstone_json=diff_capstone,
                                 max_abs_diff_vs_committed_summary_csv=diff_committed, identities_max_abs_diff=identity,
                                 verdicts_match_capstone=all(v["frame_cost"] == v["capstone"] for v in verdict_out.values())),
               settings=dict(n_boot=N_BOOT, seed=SEED, floor=FLOOR, unit="nats per token, content positions 1+"))
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res["reproduction"], indent=1))
    for k, v in verdict_out.items():
        print(k, "|", v["frame_cost"], "|", v["tagged_only"])


if __name__ == "__main__":
    main()
