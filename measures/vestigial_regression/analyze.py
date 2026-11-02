"""Vestigial regression: raw NLL, slot effects and role-specific quantities per text source, with bootstrap CIs.

    python -m measures.vestigial_regression.analyze   # reads results/<run>/core/scores/*.csv
                                                      # (+ results/<run>/vestigial_regression/generations/*_counts.csv)

A text set is one (source, kind), e.g. wildchat user or olmo-instruct assistant. Sets are never
pooled. Every text is scored in every slot; per set and stage we report
    raw NLL        mean content NLL in the user, assistant and plain-prose slot
    slot effect    NLL(user slot) - NLL(assistant slot), paired over texts
    role-specific  NLL(own slot) - NLL(plain): user@user - user@plain for user text,
                   asst@asst - asst@plain for assistant text
    entropy        mean next-token entropy over content tokens per slot
Each is reported without the first content token (default: a first-turn assistant header makes the
post-trained models expect <|im_start|>, so that token costs 2-14 nats whatever the text) and over
all tokens; the figures use the default.
95% bootstrap CIs are over texts. Writes <out>/summary.csv, <out>/results.md and one
figure per text set, <out>/nll_<source>_<kind>.png (stage on x, one line per slot).

If core/score.py saved per-token NLLs (<scores>/<stage>_tokens.parquet), also writes <out>/position.md
and <out>/position_<source>_<kind>.png: mean NLL by content-token position per slot and stage,
and the slot difference by position, over texts with at least --positions tokens.

Loading, bootstrap CIs, per-position arrays and the negligibility rule come from core.stats; this
module holds only the VR-specific quantities, tables and figures.

    python -m measures.vestigial_regression.analyze --locked [--run capstone] [--families ...] [--first 30]
    python -m measures.vestigial_regression.analyze --locked --pilot     # the pilot's scores: reproduces pilot/report.md

--locked computes the README's locked design instead (see "Pre-registered criteria"): per family and post-trained stage, change
from the family's own base of every raw slot (including the plain-text row, "non-assistant degradation"), the two role-specific
terms, and the NLL arm's primary quantity, the asymmetry = delta(user role-specific) - delta(assistant role-specific), in nats per
token (within a family) and in bits per byte (across families), with the user and assistant text sets bootstrapped independently
and pairing within each set across checkpoints; the judge arm (non-user-voice rate = switched + degenerate, change from base,
bootstrap over prefills; the category-4 rate reported separately) when judge outputs exist; how the two arms combine; and the
Spearman correlation between them across the post-trained points (base points are zero by construction and excluded).
Writes <out>/locked.md and locked.csv.
"""

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

from core import config
from core.plots import plt, stage_errorbars
from core.stats import (SLOT_NAMES, bootstrap_ci, load_scores, negligible_from, position_arrays, set_name,
                        summary_table, text_sets, token_tables)

EXCL0, ALL = "excl. position 0", "all tokens"


def quantities(kind):
    """[(quantity label, value column, function of the text x condition table)] for one text set. Every quantity
    comes without the first content token (default; after a first-turn assistant header the post-trained
    models expect <|im_start|>, so that token costs 2-14 nats whatever the text) and with it."""
    k = config.KIND_ABBR[kind]
    own = "user" if kind == "user" else "asst"
    qs = []
    for col, label in (("mean_nll_from1", EXCL0), ("mean_nll", ALL)):
        qs += [(f"NLL {SLOT_NAMES[a]} ({k}@{a}), {label}", col, (lambda t, c=f"{k}@{a}": t[c])) for a, _ in config.SLOTS]
    slot_effect = lambda t: t[f"{k}@user"] - t[f"{k}@asst"]
    qs.append((f"slot effect, {EXCL0}", "mean_nll_from1", slot_effect))
    qs.append((f"slot effect, {ALL}", "mean_nll", slot_effect))
    for col, label in (("mean_nll_from1", EXCL0), ("mean_nll", ALL)):
        qs.append((f"role-specific ({k}@{own} - {k}@plain), {label}", col, lambda t: t[f"{k}@{own}"] - t[f"{k}@plain"]))
    for col, label in (("mean_entropy_from1", EXCL0), ("mean_entropy", ALL)):
        qs += [(f"entropy {SLOT_NAMES[a]} ({k}@{a}), {label}", col, (lambda t, c=f"{k}@{a}": t[c])) for a, _ in config.SLOTS]
    return qs


def summarize(frames, n_boot, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for stage, df in frames.items():
        for src, kind in text_sets(frames):
            d = df[(df.source == src) & (df.kind == kind)]
            if d.empty:
                continue
            tables = {}
            for q, col, f in quantities(kind):
                if col not in d:  # score CSVs from before a column was added
                    continue
                if col not in tables:
                    tables[col] = d.pivot(index="text_id", columns="condition", values=col)
                t = tables[col]
                m, lo, hi = bootstrap_ci(f(t), n_boot, rng)
                rows.append(dict(stage=stage, source=src, kind=kind, quantity=q, n=len(t), mean=m, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def plot(summary, out_dir):
    stages = list(dict.fromkeys(summary["stage"]))
    x = np.arange(len(stages))
    paths = []
    label = EXCL0 if (summary.quantity.str.startswith("NLL ") & summary.quantity.str.endswith(EXCL0)).any() else ALL
    for (src, kind), g in summary.groupby(["source", "kind"], sort=False):
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for a, slot in config.SLOTS:
            d = g[g.quantity.str.startswith(f"NLL {SLOT_NAMES[a]} (") & g.quantity.str.endswith(label)]
            d = d.set_index("stage").reindex(stages)
            stage_errorbars(ax, x, d, linestyle=":" if slot is None else "-", label=SLOT_NAMES[a])
        ax.set_xticks(x, stages)
        ax.set_xlabel("post-training stage")
        ax.set_ylabel(f"mean per-token NLL of content, {label} (nats)")
        ax.set_title(f"{set_name(src, kind)} text (n={g.n.iloc[0]}), 95% bootstrap CI")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, f"nll_{src}_{kind}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def section(summary, heading, note, select, index):
    """Markdown lines: the position-0-excluded table (default), then the all-tokens table."""
    md = ["", heading, ""] + ([note, ""] if note else [])
    for label, suffix in ((EXCL0, " (default)"), (ALL, "")):
        sel = lambda d, label=label: select(d.quantity) & d.quantity.str.endswith(label)
        if sel(summary).any():
            md += [f"### {label}{suffix}", "", summary_table(summary, sel, index), ""]
    return md


def results_table(summary, gen_dir):
    sets = summary.drop_duplicates(["source", "kind"])
    md = ["# Results", "",
          "Mean over texts, 95% bootstrap CI over texts. Text sources are analyzed separately and never pooled.", "",
          "Texts per set: " + ", ".join(f"{set_name(s, k)} {n}" for s, k, n in zip(sets.source, sets.kind, sets.n)) + ".",
          "", "Every quantity is reported without the first content token (default) and over all tokens. When an "
          "assistant turn directly follows the system turn, the post-trained checkpoints put nearly all next-token "
          "probability on `<|im_start|>`, so the first content token in the assistant slot costs 2-14 nats whatever "
          "the text (see position.md)."]
    md += section(summary, "## Raw mean NLL by text set and slot", "", lambda q: q.str.startswith("NLL "),
                  ["text set", "quantity"])
    md += ["", "## Slot effect: NLL(user slot) - NLL(assistant slot)", ""]
    if (summary.quantity == f"slot effect, {EXCL0}").any():
        md += [f"### Slot effect, {EXCL0} (default)", "",
               summary_table(summary, lambda d: d.quantity == f"slot effect, {EXCL0}", ["text set"]), "",
               f"### Slot effect, {ALL}", ""]
    md += [summary_table(summary, lambda d: d.quantity == f"slot effect, {ALL}", ["text set"]),
          ""]
    md += section(summary, "## Role-specific: NLL(own slot) - NLL(plain)",
                  "user@user - user@plain for user text; asst@asst - asst@plain for assistant text.",
                  lambda q: q.str.startswith("role-specific"), ["text set", "quantity"])
    for label, note in ((EXCL0, " (default)"), (ALL, "")):
        sel = lambda d, label=label: d.quantity.str.startswith("entropy ") & d.quantity.str.endswith(label)
        if sel(summary).any():
            md += ["", f"## Mean next-token entropy by text set and slot, {label}{note}", "",
                   "Entropy (nats) of the model's next-token distribution at each content token, averaged per text.", "",
                   summary_table(summary, sel, ["text set", "quantity"])]
    counts = sorted(glob.glob(os.path.join(gen_dir, "*_counts.csv")))
    if counts:
        g = pd.concat(pd.read_csv(p) for p in counts).set_index("stage")
        g = g.reindex([s for s in config.STAGES if s in g.index] + [s for s in g.index if s not in config.STAGES])
        md += ["", "## Prefilled user-turn continuations (counts of samples)", "",
               "ended_turn: closed the user turn; handoff_to_assistant: then wrote the assistant header (a subset of "
               "ended_turn); assistant_register and repetition_loop: checked on the user portion only.", "",
               g.to_markdown()]
    return "\n".join(md) + "\n"


def position_analysis(tokens, sets, n_pos, margin, n_boot, seed, out_dir):
    """Mean NLL by content-token position per slot, and the slot difference (user - assistant slot)."""
    rng = np.random.default_rng(seed)
    stages = list(tokens)
    marks = [p for p in (0, 1, 2, 3, 5, 10, 20, n_pos - 1) if p < n_pos]
    neg_rows, pos_rows, tail_rows, ns = [], [], [], {}
    for src, kind in sets:
        name = set_name(src, kind)
        fig, axes = plt.subplots(2, len(stages), figsize=(3.2 * len(stages), 6), sharex=True, sharey="row", squeeze=False)
        for j, stage in enumerate(stages):
            ids, arr, tails = position_arrays(tokens[stage], src, kind, n_pos)
            ns[name] = len(ids)
            if not len(ids):
                continue
            # position 0 can be off any useful scale (post-trained models expect <|im_start|> right after a
            # first-turn assistant header), so panels start at 1 and print position 0's values
            x = np.arange(1, n_pos)
            first = []
            for a, slot in config.SLOTS:
                mean = arr[a].mean(axis=0)
                axes[0, j].plot(x, mean[1:], linestyle=":" if slot is None else "-", label=SLOT_NAMES[a])
                first.append(f"{a} {mean[0]:.1f}")
            axes[0, j].text(0.97, 0.95, "position 0: " + ", ".join(first), transform=axes[0, j].transAxes,
                            ha="right", va="top", fontsize=7)
            diff = arr["user"] - arr["asst"]
            idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
            boot = diff[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
            m = diff.mean(axis=0)
            ax = axes[1, j]
            ax.axhspan(-margin, margin, color="0.9", zorder=0)
            ax.axhline(0, color="0.5", lw=0.8)
            ax.fill_between(x, lo[1:], hi[1:], alpha=0.3)
            ax.plot(x, m[1:])
            ax.text(0.97, 0.05, f"position 0: {m[0]:+.2f} [{lo[0]:+.2f}, {hi[0]:+.2f}]", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7)
            axes[0, j].set_title(stage)
            ax.set_xlabel("content token position")
            p = negligible_from(diff, margin, n_boot, rng)
            neg_rows.append(dict(**{"text set": name, "stage": stage}, cell="not within first %d" % n_pos if p is None else str(p)))
            for q in marks:
                pos_rows.append(dict(**{"text set": name, "position": q, "stage": stage},
                                     cell=f"{m[q]:+.3f} [{lo[q]:+.3f}, {hi[q]:+.3f}]"))
            t = (tails["user"] - tails["asst"]).dropna()
            tm, tlo, thi = bootstrap_ci(t, n_boot, rng) if len(t) else (np.nan,) * 3
            whole = diff.mean(axis=1)
            wm, wlo, whi = bootstrap_ci(whole, n_boot, rng)
            tail_rows.append(dict(**{"text set": name, "stage": stage},
                                  cell=f"{wm:+.3f} [{wlo:+.3f}, {whi:+.3f}] / {tm:+.3f} [{tlo:+.3f}, {thi:+.3f}] (n={len(t)})"))
        axes[0, 0].set_ylabel("mean NLL (nats)")
        axes[1, 0].set_ylabel("user slot - assistant slot")
        axes[0, -1].legend(fontsize=7)
        fig.suptitle(f"{name} text, n={ns[name]} texts with >= {n_pos} tokens\n"
                     f"top: mean NLL by position; bottom: user slot - assistant slot, 95% CI, shaded +-{margin}; "
                     "position 0 printed in each panel",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"position_{src}_{kind}.png"), dpi=130)
        plt.close(fig)

    first_rows = []
    for src, kind in sets:
        k = config.KIND_ABBR[kind]
        for stage in stages:
            d = tokens[stage][(tokens[stage].source == src) & (tokens[stage].kind == kind)]
            for p0, label in ((0, "all tokens"), (1, "excluding position 0")):
                per = {a: d[(d.condition == f"{k}@{a}") & (d.position >= p0)].groupby("text_id", observed=True).nll.mean()
                       for a in ("user", "asst")}
                bm, blo, bhi = bootstrap_ci(per["user"] - per["asst"], n_boot, rng)
                first_rows.append(dict(**{"text set": set_name(src, kind), "tokens": label, "stage": stage},
                                       cell=f"{bm:+.3f} [{blo:+.3f}, {bhi:+.3f}]"))

    def wide(rows, index):
        d = pd.DataFrame(rows)
        order = list(dict.fromkeys(zip(*(d[c] for c in index)) if len(index) > 1 else d[index[0]]))
        w = d.pivot_table(index=index, columns="stage", values="cell", aggfunc="first").reindex(order)[stages]
        return w.reset_index().to_markdown(index=False)

    md = ["# Slot difference by token position", "",
          f"Per text set, texts with at least {n_pos} content tokens (so every position averages over the same texts): "
          + ", ".join(f"{k} {v}" for k, v in ns.items()) + ". Slot difference = NLL(user slot) - NLL(assistant "
          "slot) at each content-token position (0 = first token after the role header), paired over texts, 95% "
          "bootstrap CI over texts. Figures: `position_<source>_<kind>.png`.", "",
          "## Position from which the slot difference is negligible", "",
          f"First position p such that every per-position mean slot difference from p to {n_pos - 1} is within "
          f"+-{margin} nats and their mean over positions p..{n_pos - 1} has a 95% CI inside +-{margin} nats.", "",
          wide(neg_rows, ["text set"]), "",
          "## Slot difference at selected positions", "",
          wide(pos_rows, ["text set", "position"]), "",
          f"## Mean slot difference over positions 0..{n_pos - 1} / over positions {n_pos}+", "",
          f"Second value: per-text mean over the tokens after position {n_pos - 1}, for texts longer than {n_pos} tokens.", "",
          wide(tail_rows, ["text set"]), "",
          "## Slot effect with and without the first content token", "",
          "Per-text mean NLL(user slot) - NLL(assistant slot) over all content tokens (as in results.md) and over "
          "positions 1+, all texts of each set.", "",
          wide(first_rows, ["text set", "tokens"]), ""]
    with open(os.path.join(out_dir, "position.md"), "w") as f:
        f.write("\n".join(md))
    return "\n".join(md)


# ======================================================================================================================
# Locked design (README, "Pre-registered criteria")
# ======================================================================================================================

LN2 = np.log(2)
USER_SOURCES = ("wildchat", "oasst1")
ASST_SET = ("wildchat", "assistant")
LOCKED_SETS = [(src, "user") for src in USER_SOURCES] + [ASST_SET]
PILOT_POINTS = {  # family -> (scores dir, stages): the pilot's scores (OLMo: the frozen held-out run)
    "olmo3-7b": ("results/scores", ["base", "sft", "dpo", "instruct"]),
    "qwen3-8b": ("measures/vestigial_regression/pilot/nll_arm/scores_qwen", ["base", "instruct"]),
    "gemma4-31b": ("measures/vestigial_regression/pilot/nll_arm/scores", ["base", "instruct"]),
}
JUDGE_NON_USER = ("switched_to_assistant", "degenerate")


def select_ids(rows, first=None, sets=LOCKED_SETS):
    """Text ids of the locked text sets (the first `first` of each, in file order, or all)."""
    ids = []
    for src, kind in sets:
        s = [r["id"] for r in rows if r["source"] == src and r["kind"] == kind]
        ids += s[:first] if first else s
    return set(ids)


def text_bytes(tok, rows, ids):
    """{id: UTF-8 bytes of the text decoded from content tokens 1+} (primary bits per byte excludes position 0)."""
    out = {}
    for r in rows:
        if r["id"] in ids:
            toks = tok(r["text"], add_special_tokens=False)["input_ids"]
            out[r["id"]] = len(tok.decode(toks[1:], clean_up_tokenization_spaces=False).encode("utf-8"))
    return out


def checkpoint_tables(scores_dir, stage, nbytes):
    """(nats, bpb): wide frames indexed by (source, kind, text_id), one column per condition. nats = mean NLL over positions
    1+ (per token); bpb = summed NLL over positions 1+ / ln 2 / bytes (primary bits per byte)."""
    csv = pd.read_csv(os.path.join(scores_dir, f"{stage}.csv"))
    csv = csv[csv.text_id.isin(nbytes)]
    nats = csv.pivot_table(index=["source", "kind", "text_id"], columns="condition", values="mean_nll_from1")
    tk = pd.read_parquet(os.path.join(scores_dir, f"{stage}_tokens.parquet"))
    tk = tk[(tk.position >= 1) & tk.text_id.isin(list(nbytes))]
    tot = tk.groupby(["source", "kind", "condition", "text_id"], observed=True).nll.sum().reset_index()
    for c in ("source", "kind", "condition", "text_id"):
        tot[c] = tot[c].astype(str)
    tot["bpb"] = tot.nll / LN2 / tot.text_id.map(nbytes)
    bpb = tot.pivot_table(index=["source", "kind", "text_id"], columns="condition", values="bpb")
    return nats, bpb


def set_conditions(kind):
    own = "user" if kind == "user" else "asst"
    other = "asst" if kind == "user" else "user"
    return f"{own}@{own}", f"{own}@{other}", f"{own}@plain"


def delta_vectors(base, it, source, kind):
    """Per-text change from base (paired within the set) of the own-slot, other-slot and plain values and of the role-specific
    term, as pandas Series indexed by text id."""
    own, other, plain = set_conditions(kind)
    b, i = base.loc[(source, kind)], it.loc[(source, kind)]
    idx = b.index.intersection(i.index)
    d = {c: (i.loc[idx, c] - b.loc[idx, c]) for c in (own, other, plain)}
    d["rs"] = d[own] - d[plain]
    return d, (own, other, plain)


def ci_of(x, n_boot, rng):
    return bootstrap_ci(np.asarray(pd.Series(x).dropna(), dtype=float), n_boot, rng)


def asymmetry_ci(user_rs, asst_rs, n_boot, rng):
    """(mean, lo, hi) of mean(user role-specific change) - mean(assistant role-specific change). The user and assistant texts
    are different texts, so the two sets are resampled independently (pairing is within each set, across checkpoints)."""
    u, a = np.asarray(user_rs.dropna(), dtype=float), np.asarray(asst_rs.dropna(), dtype=float)
    iu = rng.integers(0, len(u), size=(n_boot, len(u)))
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    boot = u[iu].mean(axis=1) - a[ia].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return u.mean() - a.mean(), lo, hi


def fires(m, lo, hi, floor=config.MIN_EFFECT_NATS):
    """The NLL arm's verdict for one user source: the asymmetry is at least the 0.05-nat detectability floor and the CI excludes zero."""
    return bool(m >= floor and lo > 0)


def nll_point(base, it, n_boot, rng, unit):
    """All locked NLL-arm quantities for one (family, stage) point in one unit: a list of row dicts."""
    rows, rs_by_set = [], {}
    for src, kind in LOCKED_SETS:
        d, (own, other, plain) = delta_vectors(base, it, src, kind)
        rs_by_set[(src, kind)] = d["rs"]
        for label, key in ((f"Δ({own})", own), (f"Δ({other})", other), (f"Δ({plain}) [non-assistant degradation]", plain),
                           (f"role-specific Δ({own} − {plain})", "rs")):
            m, lo, hi = ci_of(d[key], n_boot, rng)
            rows.append(dict(unit=unit, text_set=f"{src} {kind}", quantity=label, mean=m, lo=lo, hi=hi))
    for src in USER_SOURCES:
        m, lo, hi = asymmetry_ci(rs_by_set[(src, "user")], rs_by_set[ASST_SET], n_boot, rng)
        rows.append(dict(unit=unit, text_set=f"{src} user vs {ASST_SET[0]} assistant", quantity="asymmetry = Δrs(user) − Δrs(assistant)",
                         mean=m, lo=lo, hi=hi))
    return rows


def nll_verdict(rows_nats):
    """(fires in both user sources with the same sign?, per-source (mean, lo, hi)) from a point's nats rows."""
    per = {r["text_set"].split(" ")[0]: (r["mean"], r["lo"], r["hi"]) for r in rows_nats if r["quantity"].startswith("asymmetry")}
    ok = all(fires(*per[s]) for s in USER_SOURCES) and len({np.sign(per[s][0]) for s in USER_SOURCES}) == 1
    return ok, per


def load_judged(judge_dir):
    """{stage: [judge output rows]} from <judge_dir>/<stage>.jsonl (judge.py output)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(judge_dir, "*.jsonl"))):
        out[os.path.basename(p)[:-6]] = [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
    return out


def prefill_rates(rows, labels):
    """{prefill id: fraction of its samples whose label is in `labels`}."""
    by = {}
    for r in rows:
        by.setdefault(r.get("prefill_id", r.get("prompt_id")), []).append(r["label"] in labels)
    return pd.Series({k: float(np.mean(v)) for k, v in by.items()})


def judge_change(base_rows, it_rows, labels, n_boot, rng):
    """(change, lo, hi, base rate, stage rate): per-prefill rate difference (stage − base) over the prefills in both, 95% bootstrap
    over prefills (the samples of one prefill are correlated, so prefills are the resampling unit)."""
    b, i = prefill_rates(base_rows, labels), prefill_rates(it_rows, labels)
    idx = b.index.intersection(i.index)
    m, lo, hi = ci_of(i.loc[idx] - b.loc[idx], n_boot, rng)
    return m, lo, hi, float(b.loc[idx].mean()), float(i.loc[idx].mean())


def combine(nll_fires, judge_fires):
    """Which arm(s) fired, and what that means (README, "Combining the arms")."""
    if nll_fires and judge_fires:
        return "both arms: capacity loss with behavioral collapse"
    if nll_fires:
        return "NLL only: worse at predicting users without visibly failing to play them"
    if judge_fires:
        return "judge only: fails to play the user while still predicting user text well"
    return "neither arm fired"


def spearman(xs, ys):
    """Spearman rank correlation (pandas ranks, ties averaged) or None if fewer than 3 points or no variance."""
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return float(pd.Series(xs).rank().corr(pd.Series(ys).rank()))


def fmt3(m, lo, hi, nd=3):
    return f"{m:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def locked_analysis(points, n_boot, seed, bytes_by_family):
    """points: {family: {"dir": scores dir, "stages": [...], "judge": judge dir or None, "tok": tokenizer name}}.
    Returns (markdown, csv rows)."""
    rng = np.random.default_rng(seed)
    allrows, md = [], ["# Locked design: NLL arm and judge arm", "",
                       f"Positions 1+ for the NLL arm; {n_boot} bootstrap resamples (seed {seed}); user and assistant text sets "
                       "resampled independently, pairing within each set across checkpoints. Every change is from the family's own base. "
                       "Nats per token are compared within a family only; bits per byte across families.", ""]
    point_rows, summary, spearman_pts = {}, [], []
    for fam, spec in points.items():
        nb = bytes_by_family[fam]
        tabs = {st: checkpoint_tables(spec["dir"], st, nb) for st in spec["stages"]}
        base = spec["stages"][0]
        judged_fam = load_judged(spec["judge"]) if spec.get("judge") and os.path.isdir(spec["judge"]) else {}
        for st in spec["stages"][1:]:
            r_n = nll_point(tabs[base][0], tabs[st][0], n_boot, rng, "nats/token")
            r_b = nll_point(tabs[base][1], tabs[st][1], n_boot, rng, "bits/byte")
            ok, per = nll_verdict(r_n)
            for r in r_n + r_b:
                allrows.append(dict(family=fam, stage=st, **r))
            point_rows[(fam, st)] = (r_n, r_b, ok, per)
            md += [f"## {fam}: {st} − {base}", "", "| text set | quantity | nats/token | bits/byte |", "|---|---|---|---|"]
            for a, b in zip(r_n, r_b):
                md.append(f"| {a['text_set']} | {a['quantity']} | {fmt3(a['mean'], a['lo'], a['hi'])} | {fmt3(b['mean'], b['lo'], b['hi'])} |")
            md += ["", "NLL arm verdict (asymmetry ≥ 0.05 nats, CI excluding zero, same direction in both user sources): "
                   + "; ".join(f"{s}: {fmt3(*per[s])}" for s in USER_SOURCES) + f" → **{'fires' if ok else 'does not fire'}**.", ""]
            j_fires, j_info = None, None
            if base in judged_fam and st in judged_fam:
                m, lo, hi, rb, rs = judge_change(judged_fam[base], judged_fam[st], JUDGE_NON_USER, n_boot, rng)
                m4, lo4, hi4, rb4, rs4 = judge_change(judged_fam[base], judged_fam[st], ("user_voice_with_assistant_traits",), n_boot, rng)
                j_fires = bool(m > 0 and lo > 0)
                j_info = (m, lo, hi, rb, rs)
                allrows.append(dict(family=fam, stage=st, unit="rate", text_set="judge", quantity="non-user-voice rate change", mean=m, lo=lo, hi=hi))
                allrows.append(dict(family=fam, stage=st, unit="rate", text_set="judge", quantity="category 4 rate change (to overgeneralization)", mean=m4, lo=lo4, hi=hi4))
                md += [f"Judge arm: non-user-voice rate (switched + degenerate) {rb:.1%} → {rs:.1%}, change {fmt3(m, lo, hi)} → "
                       f"**{'fires' if j_fires else 'does not fire'}** (the minimum rate difference that counts is an open decision: "
                       f"magnitude reported). Category 4 (user voice with assistant traits, reported to overgeneralization): "
                       f"change {fmt3(m4, lo4, hi4)}.", ""]
            else:
                md += ["Judge arm: no judge outputs for this point yet (`judge.py run`).", ""]
            asym_bpb = float(np.mean([r["mean"] for r in r_b if r["quantity"].startswith("asymmetry")]))
            summary.append(dict(family=fam, stage=st, r_b=r_b, asym_bpb=asym_bpb, nll=ok, judge=j_fires, judge_change=j_info[0] if j_info else None))
            if j_info:
                spearman_pts.append((f"{fam}:{st}", asym_bpb, j_info[0]))

    md += ["## Cross-family summary: bits per byte, change from each family's own base", "",
           "| model | " + " | ".join(f"{s} Δ(user@user) | {s} Δ(user@plain) | {s} user role-specific | {s} asymmetry" for s in USER_SOURCES)
           + " | assistant Δ(asst@asst) | assistant Δ(asst@plain) | assistant role-specific |", "|---|" + "---|" * (4 * len(USER_SOURCES) + 3)]
    for sm in summary:
        cells = []
        for src in USER_SOURCES:
            qs = [r for r in sm["r_b"] if r["text_set"] == f"{src} user"]
            cells += [fmt3(qs[0]["mean"], qs[0]["lo"], qs[0]["hi"], 2), fmt3(qs[2]["mean"], qs[2]["lo"], qs[2]["hi"], 2),
                      fmt3(qs[3]["mean"], qs[3]["lo"], qs[3]["hi"], 2)]
            asym = [r for r in sm["r_b"] if r["text_set"].startswith(f"{src} user vs")][0]
            cells.append(fmt3(asym["mean"], asym["lo"], asym["hi"], 2))
        qa = [r for r in sm["r_b"] if r["text_set"] == "wildchat assistant"]
        cells += [fmt3(qa[0]["mean"], qa[0]["lo"], qa[0]["hi"], 2), fmt3(qa[2]["mean"], qa[2]["lo"], qa[2]["hi"], 2), fmt3(qa[3]["mean"], qa[3]["lo"], qa[3]["hi"], 2)]
        md.append(f"| {sm['family']} {sm['stage']} | " + " | ".join(cells) + " |")
    md += ["", "Each family's change is from its own base; only bits per byte is compared across families.", "",
           "## How the arms combine", ""]
    for sm in summary:
        md.append(f"- {sm['family']} {sm['stage']}: NLL arm {'fires' if sm['nll'] else 'does not fire'}; judge arm "
                  + ("not run" if sm["judge"] is None else ("fires" if sm["judge"] else "does not fire"))
                  + (f" → {combine(sm['nll'], sm['judge'])}" if sm["judge"] is not None else ""))
    rho = spearman([p[1] for p in spearman_pts], [p[2] for p in spearman_pts])
    md += ["", "## Agreement of the arms across post-trained points", "",
           "Spearman correlation between the NLL asymmetry (bits per byte, mean of the two user sources) and the judge's non-user-voice "
           "rate change. Base points are zero by construction for both and are excluded. With this few points it is descriptive, not a "
           f"significance test. Points: {len(spearman_pts)}; "
           + (f"Spearman rho = {rho:.2f}." if rho is not None else "not computed (fewer than 3 points with judge outputs, or no variation).")]
    return "\n".join(md) + "\n", allrows


def locked_main(args):
    from core.render import load_jsonl, load_tokenizer
    rows = load_jsonl(config.DATA_PATH)
    ids = select_ids(rows, args.first)
    points, nbytes = {}, {}
    if args.pilot:
        for fam, (d, stages) in PILOT_POINTS.items():
            points[fam] = dict(dir=d, stages=stages, judge=None)
    else:
        fams = args.families or [f for f in config.FAMILIES if os.path.isdir(config.core_dir("scores", args.run, f))]
        for fam in fams:
            st = config.family_settings(fam)
            stages = [s for s in st["stages"] if os.path.exists(os.path.join(config.core_dir("scores", args.run, fam), f"{s}.csv"))]
            points[fam] = dict(dir=config.core_dir("scores", args.run, fam), stages=stages,
                               judge=os.path.join(config.measure_dir("vestigial_regression", args.run, fam), "judge"))
    if not points:
        raise SystemExit("no family with scores found (use --pilot, or --run/--families)")
    for fam in points:
        tok = load_tokenizer(config.FAMILIES[fam]["template"])
        nbytes[fam] = text_bytes(tok, rows, ids)
    md, allrows = locked_analysis(points, args.n_boot, args.seed, nbytes)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "locked.md"), "w", encoding="utf-8") as f:
        f.write(md)
    pd.DataFrame(allrows).to_csv(os.path.join(args.out, "locked.csv"), index=False)
    print(md)


def main():
    config.family_from_argv()  # --family switches STAGES / TEMPLATE_MODEL and the default paths
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=config.core_dir("scores"))
    ap.add_argument("--generations", default=os.path.join(config.measure_dir("vestigial_regression"), "generations"),
                    help="flagged samples (flags.py), for the *_counts.csv tables")
    ap.add_argument("--out", default=config.measure_dir("vestigial_regression"))
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--positions", type=int, default=40, help="content-token positions in the position analysis")
    ap.add_argument("--margin", type=float, default=config.MIN_EFFECT_NATS,
                    help="nats; slot difference counted as negligible within this")
    ap.add_argument("--family", help="a config.FAMILIES key (accepted for run_gpu.sh; selects the default paths)")
    ap.add_argument("--run", help="run directory under results/ (with --family or --locked: default capstone)")
    ap.add_argument("--locked", action="store_true", help="the locked design: asymmetry, raw-slot changes, judge arm, combination")
    ap.add_argument("--pilot", action="store_true", help="with --locked: the pilot's scores instead of a run's")
    ap.add_argument("--families", nargs="+", help="with --locked: families to include (default: those with scores in the run)")
    ap.add_argument("--first", type=int, help="with --locked: only the first N texts of each set (the pilot used 30)")
    args = ap.parse_args()
    if args.locked:
        if args.pilot:
            args.first = args.first or 30
        args.n_boot = 5000 if args.n_boot == 10_000 else args.n_boot
        args.out = args.out if args.out != config.measure_dir("vestigial_regression") else os.path.join(
            "measures/vestigial_regression/pilot" if args.pilot else config.measure_dir("vestigial_regression", args.run or config.DEFAULT_RUN), "locked")
        return locked_main(args)

    frames = load_scores(args.scores)
    summary = summarize(frames, args.n_boot, args.seed)
    os.makedirs(args.out, exist_ok=True)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    plot(summary, args.out)
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write(results_table(summary, args.generations))
    print(open(os.path.join(args.out, "results.md")).read())
    tokens = token_tables(args.scores, list(frames))
    if tokens:
        print(position_analysis(tokens, text_sets(frames), args.positions, args.margin, min(args.n_boot, 2000),
                                args.seed, args.out))


if __name__ == "__main__":
    main()
