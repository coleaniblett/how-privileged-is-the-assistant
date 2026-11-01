"""Loading and statistics helpers shared by everything that reads core outputs.

Core writes, per stage, <scores>/<stage>.csv (one row per text x condition: n_tokens, mean_nll,
mean_nll_from1, mean_entropy, mean_entropy_from1) and <scores>/<stage>_tokens.parquet (per-token nll
and entropy; position 0 is the first content token). Measures read these rather than re-scoring.

Conventions (AGENTS.md): text sets (source x kind) are never pooled; every effect is a per-text
difference, bootstrapped over texts (95% CI); the smallest NLL-based effect we count is
config.MIN_EFFECT_NATS.
"""

import glob
import os

import numpy as np
import pandas as pd

from core import config

SOURCE_ORDER = ["wildchat", "oasst1", "olmo-instruct"]
SLOT_NAMES = {"user": "user slot", "asst": "assistant slot", "plain": "plain (no slot)"}


def stage_order(names):
    """config.STAGES first, in post-training order, then any other stage labels, sorted."""
    return [s for s in config.STAGES if s in names] + sorted(s for s in names if s not in config.STAGES)


def load_scores(score_dir):
    """{stage: per-text score table} from <score_dir>/<stage>.csv, in stage order."""
    frames = {os.path.basename(p)[:-4]: pd.read_csv(p) for p in glob.glob(os.path.join(score_dir, "*.csv"))}
    if not frames:
        raise SystemExit(f"no score CSVs in {score_dir}")
    return {s: frames[s] for s in stage_order(frames)}


def token_tables(score_dir, stages):
    """{stage: per-token NLL frame} for stages that have a <stage>_tokens.parquet."""
    out = {}
    for s in stages:
        path = os.path.join(score_dir, f"{s}_tokens.parquet")
        if os.path.exists(path):
            out[s] = pd.read_parquet(path)
    return out


def text_sets(frames):
    """[(source, kind)] present in the score tables: user sets first, sources in SOURCE_ORDER."""
    sets = {(src, kind) for df in frames.values() for src, kind in zip(df.source, df.kind)}
    rank = lambda s: (s[1] != "user", SOURCE_ORDER.index(s[0]) if s[0] in SOURCE_ORDER else len(SOURCE_ORDER), s[0])
    return sorted(sets, key=rank)


def set_name(src, kind):
    return f"{src} {kind}"


def bootstrap_ci(x, n_boot, rng):
    """(mean, lo, hi): mean of x and its 95% percentile bootstrap CI. For a paired effect, pass the
    per-text differences."""
    x = np.asarray(x)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return x.mean(), lo, hi


def fmt(r):
    return f"{r['mean']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"


def summary_table(summary, select, index):
    """Markdown table, one column per stage, from a summary frame (stage, source, kind, quantity, mean, lo,
    hi): rows selected by select(summary), indexed by the given columns ("text set" is available)."""
    d = summary[select(summary)].copy()
    d["text set"] = [set_name(s, k) for s, k in zip(d.source, d.kind)]
    d["cell"] = d.apply(fmt, axis=1)
    order = list(dict.fromkeys(zip(*(d[c] for c in index)) if len(index) > 1 else d[index[0]]))
    wide = d.pivot_table(index=index, columns="stage", values="cell", aggfunc="first")
    wide = wide.reindex(order)[list(dict.fromkeys(summary["stage"]))]
    return wide.reset_index().to_markdown(index=False)


def position_arrays(tokens, src, kind, n_pos):
    """(text ids, {slot abbr: array [texts, n_pos]}, {slot abbr: per-text mean beyond n_pos}) for texts
    with at least n_pos content tokens, so every position averages over the same texts."""
    k = config.KIND_ABBR[kind]
    d = tokens[(tokens.source == src) & (tokens.kind == kind)]
    length = d[d.condition == f"{k}@user"].groupby("text_id", observed=True).position.max() + 1
    ids = length[length >= n_pos].index
    d = d[d.text_id.isin(ids)]
    arrays, tails = {}, {}
    for a, _ in config.SLOTS:
        c = d[d.condition == f"{k}@{a}"]
        head = c[c.position < n_pos].pivot(index="text_id", columns="position", values="nll").reindex(ids)
        arrays[a] = head.to_numpy()
        tails[a] = c[c.position >= n_pos].groupby("text_id", observed=True).nll.mean().reindex(ids)
    return ids, arrays, tails


def negligible_from(diff, margin, n_boot, rng):
    """First position p from which a per-position difference [texts, positions] is negligible: every
    per-position mean from p on is within +-margin (no spike left) and the mean over positions p..end
    has a 95% CI inside +-margin (no steady offset). None if no such p within the positions given."""
    m = diff.mean(axis=0)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    for p in range(diff.shape[1]):
        if np.abs(m[p:]).max() >= margin:
            continue
        lo, hi = np.percentile(diff[:, p:].mean(axis=1)[idx].mean(axis=1), [2.5, 97.5])
        if -margin < lo and hi < margin:
            return p
    return None
