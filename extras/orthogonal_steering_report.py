"""Report for extras.orthogonal_steering (pre-set rules in docs/decisions.md, 2026-11-03). Writes results/overgeneralization/orthogonal_steering/orthogonal_steering.md and curves.png."""
import glob, json, os
import numpy as np, pandas as pd
from core.stats import bootstrap_ci
from extras import orthogonal_steering as os_

OUT = os_.OUT
NB = 4000
ALPHAS = os_.ALPHAS
SETUPS = os_.SETUPS


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "gens", "*", "*.jsonl"))):
        rows.append(pd.read_json(f, lines=True))
    d = pd.concat(rows, ignore_index=True)
    d["refuse"] = [os_.is_refusal(t) for t in d.text]; d["refuse2"] = [os_.is_refusal(t, v2=True) for t in d.text]
    t = d.text.str.strip()
    d["degenerate"] = (t.str.len() < 20) | (t.map(lambda x: sum(c.isalpha() for c in x) / max(1, len(x))) < 0.5)
    return d


def item_rates(d, model, dtype, seed, alpha, cond, col="refuse", group=None):
    """Per-request refusal rate (mean over roles and samples) for one setting; alpha 0 uses the shared baseline."""
    s = d[(d.model == model) & (d.cond == cond)]
    s = s[(s.dtype == "none")] if alpha == 0 else s[(s.dtype == dtype) & (s.seed == seed) & (s.alpha == alpha)]
    if group: s = s[s.group == group]
    return s.groupby("item")[col].mean().sort_index().values


def matrix(d, model, dtype, seed, cond, col="refuse", group=None):
    return np.stack([item_rates(d, model, dtype, seed, a, cond, col, group) for a in ALPHAS])   # alphas x items


def swing_stats(M, rng):
    """point swing (max-min of the per-alpha means), bootstrap CI over items, signed +2 minus -2 contrast and CI, per-alpha rate CIs (core.stats.bootstrap_ci)."""
    n = M.shape[1]; idx = rng.integers(0, n, size=(NB, n)); boot = M[:, idx].mean(axis=2)          # alphas x NB
    sw = boot.max(0) - boot.min(0); sg = boot[-1] - boot[0]; mean = M.mean(1)
    return dict(swing=float(mean.max() - mean.min()), swing_ci=(float(np.percentile(sw, 2.5)), float(np.percentile(sw, 97.5))), signed=float(mean[-1] - mean[0]),
                signed_ci=(float(np.percentile(sg, 2.5)), float(np.percentile(sg, 97.5))), rates=[bootstrap_ci(M[i], 2000, np.random.default_rng(i)) for i in range(len(ALPHAS))], boot_swing=sw, mean=mean)


def fmt(x, ci=None):
    return f"{x:.2f}" if ci is None else f"{x:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def analyse(d, col):
    rng = np.random.default_rng(0); R = {}
    for model in os_.MODELS:
        if model not in set(d.model): continue
        for cond in SETUPS:
            r = {}
            for dt in os_.DTYPES: r[dt] = swing_stats(matrix(d, model, dt, -1, cond, col), rng)
            seeds = sorted(set(d[(d.model == model) & (d.dtype == "random")].seed))
            rm = [matrix(d, model, "random", s, cond, col).mean(1) for s in seeds]
            r["random_means"] = np.stack(rm) if rm else None; r["random_swings"] = [float(m.max() - m.min()) for m in rm]
            r["n_seeds"] = len(seeds); R[(model, cond)] = r
    return R


def passfail(R, model, cond):
    r = R[(model, cond)]; o, s = r["orth"], r["switch"]
    band = max(r["random_swings"]) if r["random_swings"] else float("nan")
    c1 = o["swing_ci"][0] > 0.05; c2 = o["swing"] > band; c3 = np.sign(o["signed"]) == np.sign(s["signed"]) and o["signed"] != 0
    return (c1 and c2 and c3), dict(ci_lower_gt_0_05=c1, exceeds_random_band=c2, same_sign_as_switch=c3, band=band)


def main():
    d = load(); R1 = analyse(d, "refuse"); R2 = analyse(d, "refuse2")
    info = json.load(open(os.path.join(OUT, "directions.json"))); val = json.load(open(os_.VALIDATION)) if os.path.exists(os_.VALIDATION) else None
    L = ["# Orthogonalised steering: does the switch minus formality still steer? (exploratory)\n",
         "Design and pass rule were fixed in `docs/decisions.md` (2026-11-03) before running. Detector = the original regex (primary); the revised regex is a sensitivity check.\n"]
    # 1. pass/fail
    L += ["## Pass / fail (orthogonalised switch)\n", "PASS = swing CI lower bound > 0.05 AND swing > the largest swing among the random directions AND the +2 minus -2 contrast has the sign of the raw switch's.\n",
          "| model | set-up | orth swing [95% CI] | random swings (max, n seeds) | signed +2 - (-2) | result | same result with revised detector |", "|---|---|---|---|---|---|---|"]
    verdict = {}
    for model in os_.MODELS:
        for cond in ("char_sys", "char_user"):
            if (model, cond) not in R1: continue
            ok, parts = passfail(R1, model, cond); ok2, _ = passfail(R2, model, cond); r = R1[(model, cond)]; o = r["orth"]; verdict[(model, cond)] = ok
            L.append(f"| {os_.MODELS[model]['label']} | {cond} | {fmt(o['swing'], o['swing_ci'])} | {parts['band']:.2f} (n={r['n_seeds']}) | {fmt(o['signed'], o['signed_ci'])} | **{'PASS' if ok else 'FAIL'}** | {'PASS' if ok2 else 'FAIL'} |")
    L.append("")
    # 2. swings
    L += ["## Swings (highest minus lowest refusal rate across alpha = -2..+2), with 95% bootstrap CIs over the 36 requests\n",
          "| model | set-up | raw switch | orthogonalised switch | formality | random: mean (min-max) | fraction of raw switch swing surviving orthogonalisation [CI] |", "|---|---|---|---|---|---|---|"]
    for model in os_.MODELS:
        for cond in SETUPS:
            if (model, cond) not in R1: continue
            r = R1[(model, cond)]; rs = r["random_swings"]
            frac = r["orth"]["boot_swing"] / np.maximum(r["switch"]["boot_swing"], 1e-9)
            L.append(f"| {os_.MODELS[model]['label']} | {cond} | {fmt(r['switch']['swing'], r['switch']['swing_ci'])} | {fmt(r['orth']['swing'], r['orth']['swing_ci'])} | {fmt(r['formality']['swing'], r['formality']['swing_ci'])} | "
                     f"{np.mean(rs):.2f} ({min(rs):.2f}-{max(rs):.2f}) | {r['orth']['swing'] / max(r['switch']['swing'], 1e-9):.2f} [{np.percentile(frac, 2.5):.2f}, {np.percentile(frac, 97.5):.2f}] |")
    L.append("")
    # 3. curves
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ms = [m for m in os_.MODELS if (m, "char_sys") in R1]; fig, ax = plt.subplots(len(ms), 3, figsize=(13, 3.4 * len(ms)), squeeze=False)
        col = dict(switch="#C46A00", orth="#1566B5", formality="#2E8B57")
        for i, m in enumerate(ms):
            for j, cond in enumerate(SETUPS):
                a = ax[i][j]; r = R1[(m, cond)]
                if r["random_means"] is not None:
                    a.fill_between(ALPHAS, r["random_means"].min(0), r["random_means"].max(0), color="#9AA6B0", alpha=0.35, label="random directions (min-max)"); a.plot(ALPHAS, r["random_means"].mean(0), color="#6E7C88", ls="--", lw=1)
                for dt in os_.DTYPES:
                    y = [x[0] for x in r[dt]["rates"]]; lo = [x[1] for x in r[dt]["rates"]]; hi = [x[2] for x in r[dt]["rates"]]
                    a.errorbar(ALPHAS, y, yerr=[np.array(y) - np.array(lo), np.array(hi) - np.array(y)], color=col[dt], marker="o", ms=4, lw=1.8, capsize=2, label={"switch": "raw switch", "orth": "orthogonalised switch", "formality": "formality"}[dt])
                a.set_ylim(-0.02, 1.02); a.set_title(f"{os_.MODELS[m]['label']} - {cond}", fontsize=9); a.set_xlabel("strength (x typical Assistant-to-character gap)", fontsize=8); a.set_ylabel("refusal rate", fontsize=8); a.grid(alpha=0.2)
                if i == 0 and j == 0: a.legend(fontsize=7, loc="upper left")
        plt.tight_layout(); plt.savefig(os.path.join(OUT, "curves.png"), dpi=130); plt.close()
        L += ["## Dose-response curves (random band shaded)\n", "![curves](curves.png)\n", "Bars are 95% bootstrap CIs over requests; the grey band is the min-max over the random directions at each strength, the dashed line their mean.\n"]
    except Exception as e:  # keep the report even if plotting fails
        L.append(f"(curves not drawn: {e})\n")
    # per-alpha table
    L += ["### Refusal rates by strength\n", "| model | set-up | direction | -2 | -1 | 0 | +1 | +2 |", "|---|---|---|---|---|---|---|---|"]
    for model in os_.MODELS:
        for cond in SETUPS:
            if (model, cond) not in R1: continue
            r = R1[(model, cond)]
            for dt in os_.DTYPES: L.append(f"| {os_.MODELS[model]['label']} | {cond} | {dt} | " + " | ".join(fmt(x[0], (x[1], x[2])) for x in r[dt]["rates"]) + " |")
            if r["random_means"] is not None: L.append(f"| {os_.MODELS[model]['label']} | {cond} | random mean (min-max) | " + " | ".join(f"{r['random_means'][:, k].mean():.2f} ({r['random_means'][:, k].min():.2f}-{r['random_means'][:, k].max():.2f})" for k in range(len(ALPHAS))) + " |")
    L.append("")
    # 4. cosines
    L += ["## Directions: cosines and held-out separation (steering layer)\n", info["table"], "", "All layers, cos(switch, formality):\n", info["all_layers"], "",
          "cos(orth, formality) is zero by construction (shown to confirm the projection). The held-out separation AUC is 0.5 for chance; the switch AUC uses held-out roles, the formality AUC held-out style instructions.\n"]
    # 5. detector
    L += ["## Refusal-detector validation\n"]
    if val:
        L += [f"300 existing replies (100 per model; request + first 220 characters shown to a blind labeller, fixed rubric: refuse / comply / unclear; labels R={val['label_counts'].get('R')}, C={val['label_counts'].get('C')}; unclear items dropped from agreement). Split by id: 150 development / 150 held-out.\n",
              "| detector | development agreement (n) | held-out agreement (n) | all (n) |", "|---|---|---|---|",
              f"| original regex (primary) | {val['v1_dev'][0]:.3f} ({val['v1_dev'][1]}) | {val['v1_test'][0]:.3f} ({val['v1_test'][1]}) | {val['v1_all'][0]:.3f} ({val['v1_all'][1]}) |",
              f"| revised regex (sensitivity; patterns written from the development half) | {val['v2_dev'][0]:.3f} ({val['v2_dev'][1]}) | {val['v2_test'][0]:.3f} ({val['v2_test'][1]}) | {val['v2_all'][0]:.3f} ({val['v2_all'][1]}) |", "",
              "All disagreements of the original regex are misses (it never flagged a reply the labeller called compliance): assistant-style moralising without 'sorry/can't' and soft in-character refusals. So absolute refusal rates are biased low; the revision fixed the development half but not the held-out half, so the original stays primary and the revision is reported as a sensitivity check (column above).\n"]
    # 6. oddities
    L += ["## Flags and oddities\n"]
    for model in os_.MODELS:
        if (model, "assistant") not in R1: continue
        a = R1[(model, "assistant")]["orth"]["swing"]
        for cond in ("char_sys", "char_user"):
            c = R1[(model, cond)]["orth"]["swing"]
            if a >= 0.8 * c: L.append(f"- **FLAG** {os_.MODELS[model]['label']}: the orthogonalised switch moves the plain Assistant's refusal by {a:.2f}, at least 0.8 x its swing on {cond} characters ({c:.2f}): possible general refusal lever, not persona-specific.")
            else: L.append(f"- {os_.MODELS[model]['label']}: orth swing on the plain Assistant {a:.2f} vs {c:.2f} on {cond} characters (no flag).")
    deg = d.groupby(["model", "dtype", "alpha"]).degenerate.mean().unstack("alpha")
    L += ["", "Degenerate replies (shorter than 20 characters or under 50% alphabetic) by direction and strength, as a coherence check:\n", "| model | direction | -2 | -1 | +1 | +2 |", "|---|---|---|---|---|---|"]
    for (model, dt), row in deg.iterrows():
        if dt == "none": continue
        L.append(f"| {os_.MODELS[model]['label']} | {dt} | " + " | ".join(f"{row.get(a, np.nan):.3f}" for a in (-2.0, -1.0, 1.0, 2.0)) + " |")
    L += ["", "Rates by role group for the orthogonalised switch (helpful vs stylized roles), swing:\n", "| model | set-up | helpful | stylized |", "|---|---|---|---|"]
    rng = np.random.default_rng(1)
    for model in os_.MODELS:
        for cond in ("char_sys", "char_user"):
            if (model, cond) not in R1: continue
            sw = [swing_stats(matrix(d, model, "orth", -1, cond, "refuse", g), rng) for g in ("helpful", "stylized")]
            L.append(f"| {os_.MODELS[model]['label']} | {cond} | {fmt(sw[0]['swing'], sw[0]['swing_ci'])} | {fmt(sw[1]['swing'], sw[1]['swing_ci'])} |")
    open(os.path.join(OUT, "orthogonal_steering.md"), "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L[:30]))
    return verdict


if __name__ == "__main__":
    main()
