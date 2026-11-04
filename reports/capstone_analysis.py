"""Capstone results: vestigial regression (NLL arm + judge arm) and logit distortion on the main run (exchange design).

    python -m reports.capstone_analysis            # -> reports/capstone_results.json (every number the report uses)

Analysis only. Reads committed outputs; changes no measure's code, criteria or data:
  scores     results/capstone/<family>/<stage>/core/exchanges_exchange/{scores.csv,tokens.parquet,manifest.json}
  judge      results/capstone/<family>/vestigial_regression/judge_incomplete/<stage>.jsonl
  gens       results/capstone/<family>/core/generations_incomplete/<stage>.jsonl
The NLL arm is measures.vestigial_regression.analyze_exchanges.run (re-run here, checked against the committed summary.csv).
The judge arm follows the README (non-user-voice rate change from base, bootstrap over prefills); refused (invalid) samples are
not yet hand-labelled, so three readings are computed. Logit distortion is computed here from the same exchange scores under
Grace's README criteria (not frozen, not yet adapted to the exchange design): PROVISIONAL.
"""

import glob
import json
import math
import os

import numpy as np
import pandas as pd

from core import config
from measures.vestigial_regression import analyze_exchanges as AX
from measures.vestigial_regression.analyze import asymmetry_ci, judge_change

ROOT = "results/capstone"
TAG = "exchanges_exchange"
N_BOOT = 10_000
SEED = config.SEED
FLOOR = config.MIN_EFFECT_NATS
FAMILIES = ["olmo3-7b", "qwen3-8b", "gemma4-31b"]
STAGES = {"olmo3-7b": ["base", "sft", "dpo", "instruct"], "qwen3-8b": ["base", "instruct"], "gemma4-31b": ["base", "instruct"]}
EXPECTED_N = {"olmo3-7b": 300, "qwen3-8b": 300, "gemma4-31b": 100}
CORPORA = ["oasst1", "ultrachat", "wildchat"]
HEADLINE = ["oasst1", "ultrachat"]
NON_USER = ("switched_to_assistant", "degenerate")
LABELS = ["user_voice", "user_voice_with_assistant_traits", "switched_to_assistant", "degenerate", "invalid"]
POST = [(f, s) for f in FAMILIES for s in STAGES[f][1:]]
OUT = "reports/capstone_results.json"


def ci(x, rng, n_boot=N_BOOT):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    b = x[idx].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return dict(mean=float(x.mean()), lo=float(lo), hi=float(hi), n=int(len(x)))


def fires(c, floor=FLOOR):
    return bool(c["mean"] >= floor and c["lo"] > 0)


def opposite(c, floor=FLOOR):
    return bool(c["mean"] <= -floor and c["hi"] < 0)


# --- 1. data checks ---------------------------------------------------------------------------------------------------------

def load_scores():
    return {(f, s): pd.read_csv(os.path.join(ROOT, f, s, "core", TAG, "scores.csv")) for f in FAMILIES for s in STAGES[f]
            if os.path.exists(os.path.join(ROOT, f, s, "core", TAG, "scores.csv"))}


def check_scores(scores):
    rows, problems = [], []
    for f in FAMILIES:
        fps = set()
        for s in STAGES[f]:
            if (f, s) not in scores:
                problems.append(f"scores missing: {f} {s}")
                continue
            d = scores[(f, s)]
            g = d.groupby(["corpus", "scored_text", "framing"], observed=True).size()
            man = json.load(open(os.path.join(ROOT, f, s, "core", TAG, "manifest.json")))
            fps.add(json.dumps(man.get("inputs_fingerprint", man.get("fingerprint")), sort_keys=True))
            nan = int(d[["mean_nll_from1", "mean_entropy_from1"]].isna().sum().sum())
            ok = len(g) == 12 and set(g.values) == {EXPECTED_N[f]} and nan == 0
            per_corpus = {c: int(d[(d.corpus == c) & (d.scored_text == "user") & (d.framing == "tagged")].exchange_id.nunique()) for c in CORPORA}
            rows.append(dict(family=f, stage=s, conditions=int(len(g)), per_corpus=per_corpus, missing_values=nan,
                             truncated_replies={c: int(d[(d.corpus == c) & (d.scored_text == "reply") & (d.framing == "tagged")].truncated.sum()) for c in CORPORA},
                             ok=bool(ok)))
            if not ok:
                problems.append(f"scores incomplete: {f} {s}: {len(g)} conditions, counts {sorted(set(g.values))}, {nan} missing")
        if len(fps) != 1:
            problems.append(f"{f}: stages do not share one inputs fingerprint ({len(fps)} distinct)")
    return rows, problems


def check_token_consistency(scores):
    """Recompute each tagged text's mean NLL and entropy over positions 1+ from tokens.parquet and compare with scores.csv."""
    worst = {}
    for (f, s), d in scores.items():
        t = pd.read_parquet(os.path.join(ROOT, f, s, "core", TAG, "tokens.parquet"))
        t = t[t.position >= 1]
        g = t.groupby(["corpus", "exchange_id", "scored_text", "framing"], observed=True)[["nll", "entropy"]].mean().reset_index()
        m = d.merge(g, on=["corpus", "exchange_id", "scored_text", "framing"])
        worst[f"{f} {s}"] = dict(nll=float((m.mean_nll_from1 - m.nll).abs().max()), entropy=float((m.mean_entropy_from1 - m.entropy).abs().max()),
                                 rows=int(len(m)), of=int(len(d)))
    return worst


def load_jsonl(p):
    return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]


def check_judge():
    prefills = {r["id"]: r["prefill"] for r in load_jsonl("data/prefill_incomplete_200.jsonl")}
    judged, rows, problems = {}, [], []
    for f in FAMILIES:
        for s in STAGES[f]:
            p = os.path.join(ROOT, f, "vestigial_regression", "judge_incomplete", f"{s}.jsonl")
            if not os.path.exists(p):
                problems.append(f"judge labels missing: {f} {s}")
                continue
            j = load_jsonl(p)
            gens = {(r["prefill_id"], r["sample"]): r for r in load_jsonl(os.path.join(ROOT, f, "core", "generations_incomplete", f"{s}.jsonl"))}
            keys = {(r["prefill_id"], r["sample"]) for r in j}
            joined = all(k in gens for k in keys)
            prefill_match = all(gens[k]["prefill"] == prefills.get(k[0]) for k in keys if k in gens)
            models = sorted({r["judge_model"] for r in j})
            versions = sorted({r["prompt_version"] for r in j})
            counts = {l: sum(r["label"] == l for r in j) for l in LABELS}
            rows.append(dict(family=f, stage=s, n=len(j), unique=len(keys), prefills=len({k[0] for k in keys}), joined=joined,
                             prefill_set="data/prefill_incomplete_200.jsonl (mid-sentence)" if prefill_match else "MISMATCH",
                             judge_model=models, prompt_version=versions, counts=counts))
            if len(j) != 600 or len(keys) != 600 or not joined or not prefill_match:
                problems.append(f"judge labels for {f} {s}: n={len(j)}, unique={len(keys)}, joined={joined}, prefill match={prefill_match}")
            judged[(f, s)] = j
    return judged, rows, problems


def check_validation():
    sheet = pd.read_csv("measures/vestigial_regression/judge_validation/sheet.csv", dtype=str).fillna("")
    ref = pd.read_csv("measures/vestigial_regression/judge_validation/refusals/sheet.csv", dtype=str).fillna("")
    return dict(validation_sheet_rows=int(len(sheet)), validation_labelled=int((sheet.human_label.str.strip() != "").sum()),
                refusal_sheet_rows=int(len(ref)), refusal_labelled=int((ref.human_label.str.strip() != "").sum()))


# --- 2. vestigial regression, NLL arm ---------------------------------------------------------------------------------------

def vr_nll():
    # config.FAMILIES order, as analyze_exchanges' default: the bootstrap RNG is shared across families, so this reproduces summary.csv
    summary, verdicts, levels = AX.run(ROOT, TAG, list(config.FAMILIES), N_BOOT, SEED)
    committed = pd.read_csv(os.path.join(ROOT, "vestigial_regression", TAG, "summary.csv"))
    m = summary.merge(committed, on=["family", "stage", "corpus", "quantity", "unit"], suffixes=("", "_c"))
    repro = dict(rows=int(len(summary)), matched=int(len(m)), max_abs_diff=float(max((m[c] - m[c + "_c"]).abs().max() for c in ("mean", "lo", "hi"))))
    cells = {}
    for _, r in summary.iterrows():
        cells.setdefault(f"{r.family}|{r.stage}", {}).setdefault(r.corpus, {}).setdefault(r.unit, {})[r.quantity] = dict(
            mean=float(r["mean"]), lo=float(r.lo), hi=float(r.hi), n=str(r.n))
    verdict = {}
    for f, s in POST:
        c = cells[f"{f}|{s}"]
        a = {k: c[k]["nats"]["asymmetry"] for k in HEADLINE}
        f_all = all(fires(a[k]) for k in HEADLINE)
        o_all = all(opposite(a[k]) for k in HEADLINE)
        verdict[f"{f}|{s}"] = dict(fires=f_all, per_corpus={k: fires(a[k]) for k in HEADLINE},
                                   reading="fires (both headline corpora)" if f_all else
                                   "does not fire; negative in both headline corpora (opposite of regression)" if o_all else "does not fire")
    return dict(cells=cells, verdict=verdict, reproduction=repro,
                levels=levels.to_dict(orient="records") if hasattr(levels, "to_dict") else None)


# --- 3. vestigial regression, judge arm --------------------------------------------------------------------------------------

READINGS = {"exclude": "refused samples excluded (rate over labelled samples)",
            "as_user": "refused samples counted as user voice (what analyze.py's judge_change does: in the denominator, never non-user)",
            "as_nonuser": "refused samples counted as non-user voice (upper bound)"}


def prefill_rate(rows, reading, labels=NON_USER):
    by = {}
    for r in rows:
        lab = r["label"]
        if lab == "invalid":
            if reading == "exclude":
                continue
            hit = reading == "as_nonuser"
        else:
            hit = lab in labels
        by.setdefault(r["prefill_id"], []).append(hit)
    return pd.Series({k: float(np.mean(v)) for k, v in by.items() if v})


def judge_arm(judged):
    rng = np.random.default_rng(SEED)
    out = {}
    for f, s in POST:
        b, p = judged.get((f, STAGES[f][0])), judged.get((f, s))
        if b is None or p is None:
            continue
        res = {}
        for rd in READINGS:
            rb, rp = prefill_rate(b, rd), prefill_rate(p, rd)
            idx = rb.index.intersection(rp.index)
            c = ci((rp.loc[idx] - rb.loc[idx]).to_numpy(), rng)
            c.update(base_rate=float(rb.loc[idx].mean()), stage_rate=float(rp.loc[idx].mean()), fires=bool(c["mean"] > 0 and c["lo"] > 0),
                     prefills=int(len(idx)))
            res[rd] = c
        # category 4 (to overgeneralization), refused excluded
        rb4, rp4 = prefill_rate(b, "exclude", ("user_voice_with_assistant_traits",)), prefill_rate(p, "exclude", ("user_voice_with_assistant_traits",))
        idx = rb4.index.intersection(rp4.index)
        res["category4"] = ci((rp4.loc[idx] - rb4.loc[idx]).to_numpy(), rng)
        # cross-check: the measure's own function equals the as_user reading's point estimate
        m, lo, hi, br, sr = judge_change(b, p, NON_USER, 2000, np.random.default_rng(SEED))
        res["judge_change_crosscheck"] = dict(mean=float(m), base_rate=br, stage_rate=sr)
        out[f"{f}|{s}"] = res
    shares = {f"{f}|{s}": {l: sum(r["label"] == l for r in judged[(f, s)]) / len(judged[(f, s)]) for l in LABELS}
              for f in FAMILIES for s in STAGES[f] if (f, s) in judged}
    return out, shares


def spearman(xs, ys):
    return float(pd.Series(xs).rank().corr(pd.Series(ys).rank()))


def arms_correlation(nll, judge):
    pts = []
    for f, s in POST:
        c = nll["cells"][f"{f}|{s}"]
        a_bpb = float(np.mean([c[k]["bpb"]["asymmetry"]["mean"] for k in HEADLINE]))
        a_nats = float(np.mean([c[k]["nats"]["asymmetry"]["mean"] for k in HEADLINE]))
        pts.append(dict(family=f, stage=s, asym_bpb=a_bpb, asym_nats=a_nats,
                        **{f"judge_{rd}": judge[f"{f}|{s}"][rd]["mean"] for rd in READINGS}))
    rho = {rd: spearman([p["asym_bpb"] for p in pts], [p[f"judge_{rd}"] for p in pts]) for rd in READINGS}
    return dict(points=pts, rho=rho)


# --- 4. logit distortion (provisional) ----------------------------------------------------------------------------------------

def gaps(d):
    t = d[d.framing == "tagged"].copy()
    t["gap"] = t.mean_nll_from1 - t.mean_entropy_from1
    t = t.set_index(["corpus", "scored_text", "exchange_id"])
    return t[["gap", "mean_nll_from1", "mean_entropy_from1"]].rename(columns={"mean_nll_from1": "nll", "mean_entropy_from1": "entropy"})


def contrast_ci(reply, user, rng, paired):
    """mean(Δgap reply) − mean(Δgap user). paired: resample exchanges jointly (same exchanges in both); else independently."""
    if paired:
        idx = reply.index.intersection(user.index)
        r, u = reply.loc[idx].to_numpy(float), user.loc[idx].to_numpy(float)
        ii = rng.integers(0, len(idx), size=(N_BOOT, len(idx)))
        b = r[ii].mean(axis=1) - u[ii].mean(axis=1)
        m = r.mean() - u.mean()
    else:
        r, u = reply.to_numpy(float), user.to_numpy(float)
        ir = rng.integers(0, len(r), size=(N_BOOT, len(r)))
        iu = rng.integers(0, len(u), size=(N_BOOT, len(u)))
        b = r[ir].mean(axis=1) - u[iu].mean(axis=1)
        m = r.mean() - u.mean()
    lo, hi = np.percentile(b, [2.5, 97.5])
    return dict(mean=float(m), lo=float(lo), hi=float(hi))


def classify(persona, floor=FLOOR):
    """Exploratory classification of one persona's change (scheme from the analysis request, not from Grace's README)."""
    dn, dh, dg = persona["nll"]["mean"], persona["entropy"]["mean"], persona["gap"]
    if fires(dg, floor):
        return "gap rising (distortion: confidence grows faster than accuracy)"
    if opposite(dg, floor):
        return "gap falling (accuracy improves more than confidence)"
    if dn <= -floor and dh <= -floor:
        return "NLL and entropy fall together (learned the role)"
    if dn >= floor and dh >= floor:
        return "NLL and entropy rise together (less fit, calibration kept)"
    return "little change"


def logit_distortion(scores):
    rng = np.random.default_rng(SEED + 1)
    out = {}
    for f, s in POST:
        b, p = gaps(scores[(f, STAGES[f][0])]), gaps(scores[(f, s)])
        idx = b.index.intersection(p.index)
        d = p.loc[idx] - b.loc[idx]
        res = {}
        for c in CORPORA:
            cell = {}
            for who in ("user", "reply"):
                x = d.xs((c, who), level=(0, 1))
                cell[who] = {q: ci(x[q].to_numpy(), rng) for q in ("gap", "nll", "entropy")}
                cell[who]["levels"] = {q: dict(base=float(b.loc[idx].xs((c, who), level=(0, 1))[q].mean()),
                                               stage=float(p.loc[idx].xs((c, who), level=(0, 1))[q].mean())) for q in ("gap", "nll", "entropy")}
                cell[who]["class"] = classify(cell[who])
                dn, dh = cell[who]["nll"]["mean"], cell[who]["entropy"]["mean"]
                cell[who]["driver"] = ("mostly NLL rising" if dn > 0 and dn >= -dh else "mostly entropy falling" if dh < 0 and -dh > max(dn, 0)
                                       else "mixed")
            ur, rr = d.xs((c, "user"), level=(0, 1)).gap, d.xs((c, "reply"), level=(0, 1)).gap
            cell["contrast_paired"] = contrast_ci(rr, ur, rng, paired=True)
            cell["contrast_independent"] = contrast_ci(rr, ur, rng, paired=False)
            cell["contrast_nll"] = contrast_ci(d.xs((c, "reply"), level=(0, 1)).nll, d.xs((c, "user"), level=(0, 1)).nll, rng, paired=True)
            cell["contrast_entropy"] = contrast_ci(d.xs((c, "reply"), level=(0, 1)).entropy, d.xs((c, "user"), level=(0, 1)).entropy, rng, paired=True)
            res[c] = cell
        # alternative model-written column of the README's original cross: UltraChat user turns vs WildChat (ChatGPT) replies
        res["alt_model_column"] = contrast_ci(d.xs(("wildchat", "reply"), level=(0, 1)).gap, d.xs(("ultrachat", "user"), level=(0, 1)).gap, rng, paired=False)
        verdicts = {}
        for name, cols in (("paired", ("oasst1", "contrast_paired", "ultrachat", "contrast_paired")),
                           ("independent", ("oasst1", "contrast_independent", "ultrachat", "contrast_independent"))):
            verdicts[name] = fires(res[cols[0]][cols[1]]) and fires(res[cols[2]][cols[3]])
        verdicts["alt_model_column"] = fires(res["oasst1"]["contrast_paired"]) and fires(res["alt_model_column"])
        for c in HEADLINE:
            pc = res[c]["contrast_paired"]
            sim = abs(pc["mean"]) < FLOOR or (pc["lo"] <= 0 <= pc["hi"])
            both_sharpen = res[c]["user"]["entropy"]["mean"] <= -FLOOR and res[c]["reply"]["entropy"]["mean"] <= -FLOOR
            res[c]["contrast_class"] = ("assistant gap grows more than user gap (criterion direction)" if fires(pc) else
                                        "user gap grows more than assistant gap (opposite direction)" if opposite(pc) else
                                        "general sharpening: similar changes for both personas" if (sim and both_sharpen) else
                                        "similar changes for both personas" if sim else "difference below the 0.05-nat floor")
        out[f"{f}|{s}"] = dict(cells=res, fires=verdicts)
    return out


def ld_positions(bins=((1, 10), (11, 25), (26, 50), (51, 100), (101, 300))):
    """Δgap (stage − base) by content-position bin, tagged, per persona; per-exchange bin means, paired, bootstrap over exchanges."""
    rng = np.random.default_rng(SEED + 2)
    out = {}
    tok = {}
    for f in FAMILIES:
        for s in STAGES[f]:
            t = pd.read_parquet(os.path.join(ROOT, f, s, "core", TAG, "tokens.parquet"))
            t = t[(t.framing == "tagged") & (t.position >= 1) & t.corpus.isin(HEADLINE)]
            t["gap"] = t.nll - t.entropy
            t["bin"] = pd.cut(t.position, [b[0] - 1 for b in bins] + [bins[-1][1]], labels=[f"{a}-{b}" for a, b in bins])
            tok[(f, s)] = t.groupby(["corpus", "scored_text", "exchange_id", "bin"], observed=True).gap.mean()
    for f, s in POST:
        base, st = tok[(f, STAGES[f][0])], tok[(f, s)]
        idx = base.index.intersection(st.index)
        d = (st.loc[idx] - base.loc[idx])
        res = {}
        for c in HEADLINE:
            for who in ("user", "reply"):
                for bl in [f"{a}-{b}" for a, b in bins]:
                    try:
                        x = d.xs((c, who, bl), level=(0, 1, 3))
                    except KeyError:
                        continue
                    if len(x) >= 10:
                        res[f"{c}|{who}|{bl}"] = ci(x.to_numpy(), rng, 2000)
        out[f"{f}|{s}"] = res
    return out


def main():
    scores = load_scores()
    score_rows, problems = check_scores(scores)
    tok_check = check_token_consistency(scores)
    judged, judge_rows, jp = check_judge()
    problems += jp
    validation = check_validation()
    nll = vr_nll()
    judge, shares = judge_arm(judged)
    arms = arms_correlation(nll, judge)
    ld = logit_distortion(scores)
    pos = ld_positions()
    res = dict(score_checks=score_rows, token_consistency=tok_check, judge_checks=judge_rows, validation=validation, problems=problems,
               vr_nll=nll, judge=judge, judge_shares=shares, judge_readings=READINGS, arms=arms, ld=ld, ld_positions=pos,
               settings=dict(n_boot=N_BOOT, seed=SEED, floor=FLOOR))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print(f"wrote {OUT}; problems: {problems or 'none'}")
    print("NLL reproduction:", nll["reproduction"])
    print("token consistency (worst |diff|):", max(v["nll"] for v in tok_check.values()), max(v["entropy"] for v in tok_check.values()))
    print("validation:", validation)
    for k, v in nll["verdict"].items():
        print("VR NLL", k, v["reading"])
    for k, v in judge.items():
        print("judge", k, {rd: (round(v[rd]["mean"], 3), round(v[rd]["lo"], 3), round(v[rd]["hi"], 3), v[rd]["fires"]) for rd in READINGS},
              "crosscheck", round(v["judge_change_crosscheck"]["mean"], 3))
    print("spearman", arms["rho"])
    for k, v in ld.items():
        print("LD", k, v["fires"], {c: (round(v["cells"][c]["contrast_paired"]["mean"], 3), round(v["cells"][c]["contrast_paired"]["lo"], 3),
                                       round(v["cells"][c]["contrast_paired"]["hi"], 3)) for c in CORPORA})


if __name__ == "__main__":
    main()
