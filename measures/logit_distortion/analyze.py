"""Logit distortion on the main run's exchange scores: mean next-token entropy, mean NLL and gap (NLL - entropy) per persona.

    uv run python -m measures.logit_distortion.analyze                 # results/capstone -> results/capstone/logit_distortion/exchanges_exchange
    uv run python -m measures.logit_distortion.analyze --root results/capstone --tag exchanges_exchange --out <dir> [--no-plots]

Spec: measures/logit_distortion/README.md, "Exchange-design analysis (2026-11-04)" (decided after the data existed; not blind).
Reads only core outputs, results/<run>/<family>/<stage>/core/<tag>/{scores.csv,tokens.parquet,manifest.json}; nothing is re-scored.

Per target (an exchange's opening user turn, or its reply scored after the question), over content positions 1+ (sensitivity: 0+):
    nll      mean token NLL            (scores.csv mean_nll_from1; checked against tokens.parquet)
    entropy  mean next-token entropy   (mean_entropy_from1; same positions, same equal token weights)
    gap      nll - entropy
Per family x corpus, every target weighted equally, 95% percentile bootstrap CIs with exchange IDs resampled jointly (one resample
drives every stage, role, framing and metric, so user/reply pairs, framings and checkpoints stay paired):
    level                    mean per stage, role (user | reply), framing (tagged | plain)
    change_from_base         stage - base, paired per exchange
    asst_minus_user          reply - user, paired per exchange
    asst_minus_user_change   (reply - user)stage - (reply - user)base; for the tagged gap at positions 1+ this is D (criterion 1's quantity)
    tagged_minus_plain       framing difference per role (and for reply - user), and its change from base: context only
Subsets: all exchanges (within-family results); untruncated replies only (sensitivity); common IDs across families (cross-family,
also in bits per byte; nats per token are not comparable across tokenizers).

Two classifications of D per post-trained checkpoint, on unrounded estimates (verdict.csv; sensitivities in verdict_sensitivity.csv):
    original_positive_criterion  README criterion 1 (fixed 2026-11-02), a directional hypothesis: D >= +0.05 nats with the CI above
                                 zero in both oasst1 and UltraChat. "not met" is not evidence against privilege in general.
    exploratory_two_sided        added 2026-11-04 after the results (not pre-registered): positive shift as above; negative shift if
                                 D <= -0.05 with the CI below zero in both corpora; otherwise no two-corpus directional consistency.
                                 The 0.05 floor is applied symmetrically; it is not a newly validated threshold. Either shift is a
                                 role-dependent gap change under this design, not by itself assistant privilege.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from core import config

CORPORA = ("oasst1", "ultrachat", "wildchat")
HEADLINE = ("oasst1", "ultrachat")  # human-written column, model-written column
ROLES = ("user", "reply")
FRAMINGS = ("tagged", "plain")
METRICS = ("entropy", "nll", "gap")
POSITIONS = ("from1", "from0")
POSITION_BINS = ((1, 5), (6, 10), (11, 25), (26, 50), (51, 100), (101, 200), (201, 299))
SUBSETS = ("all", "untruncated", "common", "positions")
KEY = ["corpus", "exchange_id", "scored_text", "framing"]
LN2 = math.log(2)
TOL = 2e-5  # scores.csv stores 6 decimals
MIN_BIN_N = 10  # a position bin with fewer contributing targets gets no CI


class InputError(ValueError):
    """The saved outputs are missing, duplicated, mismatched or non-finite."""


# --- loading and validation ------------------------------------------------------------------------------------------------

def stage_dir(root, family, stage, tag):
    return os.path.join(root, family, stage, "core", tag)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_family(root, family, tag):
    """{stage: {"scores", "tokens", "manifest", "files"}} for the family's configured stages that have outputs, in config order."""
    out = {}
    for stage in config.family_settings(family)["stages"]:
        d = stage_dir(root, family, stage, tag)
        if not os.path.exists(os.path.join(d, "scores.csv")):
            continue
        files = {n: os.path.join(d, n) for n in ("scores.csv", "tokens.parquet", "manifest.json")}
        for n, p in files.items():
            if not os.path.exists(p):
                raise InputError(f"{family} {stage}: {n} missing in {d}")
        out[stage] = dict(scores=pd.read_csv(files["scores.csv"]), tokens=pd.read_parquet(files["tokens.parquet"]),
                          manifest=json.load(open(files["manifest.json"])), files={n: (p, sha256(p)) for n, p in files.items()})
    return out


def token_means(tokens, first=1):
    """Per (corpus, exchange_id, scored_text, framing): mean nll, entropy, gap over positions >= first, and the token count."""
    t = tokens[tokens.position >= first]
    g = t.groupby(KEY, observed=True)
    m = g[["nll", "entropy"]].mean()
    m["gap"] = m.nll - m.entropy
    m["n"] = g.size()
    return m


def validate_family(family, stages):
    """Checks one family's saved outputs; raises InputError on any failure, returns a record of what was checked."""
    rec = dict(family=family, stages=list(stages), missing_stages=[s for s in config.family_settings(family)["stages"] if s not in stages])
    if "base" not in stages:
        raise InputError(f"{family}: no base checkpoint")
    if len(stages) < 2:
        raise InputError(f"{family}: no post-trained checkpoint")
    ids, fps, trunc = {}, set(), {}
    for stage, s in stages.items():
        d = s["scores"]
        where = f"{family} {stage}"
        if set(d.family.astype(str)) != {family} or set(d.stage.astype(str)) != {stage}:
            raise InputError(f"{where}: scores.csv labels another family or stage")
        if d.duplicated(KEY).any():
            raise InputError(f"{where}: duplicated conditions, e.g. {d[d.duplicated(KEY)].iloc[0][KEY].tolist()}")
        bad = (set(d.corpus) - set(CORPORA)) | (set(d.scored_text) - set(ROLES)) | (set(d.framing) - set(FRAMINGS))
        if bad:
            raise InputError(f"{where}: unknown corpus / scored text / framing {sorted(bad)}")
        per = d.groupby(["corpus", "exchange_id"]).size()
        if (per != 4).any():
            ex = per[per != 4].index[0]
            raise InputError(f"{where}: exchange {ex} has {per[ex]} of 4 conditions")
        num = d[["mean_nll", "mean_nll_from1", "mean_entropy", "mean_entropy_from1"]].to_numpy(float)
        if not np.isfinite(num).all():
            raise InputError(f"{where}: non-finite score values")
        if (d.n_tokens < 2).any():
            raise InputError(f"{where}: a target has fewer than 2 scored tokens (no positions 1+)")
        w = d.pivot(index=["corpus", "exchange_id", "scored_text"], columns="framing", values="n_tokens")
        if (w.tagged != w.plain).any():
            raise InputError(f"{where}: tagged and plain score different numbers of tokens")
        # per-token table: every target present, positions 0..n-1, finite, means equal to scores.csv
        t = s["tokens"]
        if not np.isfinite(t[["nll", "entropy"]].to_numpy(float)).all():
            raise InputError(f"{where}: non-finite per-token values")
        dd = d.set_index(KEY)
        pos = t.groupby(KEY, observed=True).position.agg(["min", "max", "size"])
        if len(pos.index.difference(dd.index)):
            raise InputError(f"{where}: tokens.parquet has targets that scores.csv lacks")
        pos = pos.reindex(dd.index)
        if pos["size"].isna().any():
            raise InputError(f"{where}: {int(pos['size'].isna().sum())} targets missing from tokens.parquet")
        if ((pos["min"] != 0) | (pos["max"] != dd.n_tokens - 1) | (pos["size"] != dd.n_tokens)).any():
            raise InputError(f"{where}: token positions are not 0..n_tokens-1 for every target")
        diffs = {}
        for first, cols in ((1, ("mean_nll_from1", "mean_entropy_from1")), (0, ("mean_nll", "mean_entropy"))):
            m = token_means(t, first).reindex(dd.index)
            for c, q in zip(cols, ("nll", "entropy")):
                diffs[c] = float((m[q] - dd[c]).abs().max())
        if max(diffs.values()) > TOL:
            raise InputError(f"{where}: tokens.parquet means differ from scores.csv by up to {max(diffs.values()):.2e}")
        man = s["manifest"]
        if man.get("family") != family or man.get("stage") != stage:
            raise InputError(f"{where}: manifest labels another family or stage")
        fps.add(man.get("inputs_fingerprint"))
        ids[stage] = set(map(tuple, d[["corpus", "exchange_id"]].drop_duplicates().to_numpy()))
        trunc[stage] = d[(d.scored_text == "reply") & (d.framing == "tagged")].set_index("exchange_id").truncated.astype(bool).sort_index()
        rec.setdefault("per_stage", {})[stage] = dict(
            rows=int(len(d)), conditions=int(d.groupby(["corpus", "scored_text", "framing"]).ngroups),
            exchanges={c: int((d[(d.scored_text == "user") & (d.framing == "tagged")].corpus == c).sum()) for c in CORPORA},
            max_abs_diff_token_means=diffs, exclusions=len(man.get("exclusions", [])), model=man.get("model"),
            model_revision=man.get("model_revision"),
            score_sources={k: int(v) for k, v in d.score_source.astype(str).str.split(":").str[0].value_counts().items()})
    if len(fps) != 1 or None in fps:
        raise InputError(f"{family}: stages do not share one inputs fingerprint ({len(fps)} distinct)")
    first_stage = next(iter(ids))
    for stage, v in ids.items():
        if v != ids[first_stage]:
            raise InputError(f"{family} {stage}: exchange IDs differ from {first_stage} ({len(v ^ ids[first_stage])} differ)")
    t0 = trunc[first_stage]
    if any(not v.equals(t0) for v in trunc.values()):
        raise InputError(f"{family}: truncation flags differ across stages")
    rec["inputs_fingerprint"] = fps.pop()
    rec["n_exchanges"] = {c: sum(1 for k in ids[first_stage] if k[0] == c) for c in CORPORA}
    return rec


# --- per-target values -----------------------------------------------------------------------------------------------------

def per_target(scores):
    """Long frame: corpus, exchange_id, role, framing, metric, positions, unit, value (one row per target x quantity)."""
    d = scores
    out = []
    for positions, (nc, ec) in (("from1", ("mean_nll_from1", "mean_entropy_from1")), ("from0", ("mean_nll", "mean_entropy"))):
        for metric, x in (("nll", d[nc]), ("entropy", d[ec]), ("gap", d[nc] - d[ec])):
            base = dict(corpus=d.corpus, exchange_id=d.exchange_id, role=d.scored_text, framing=d.framing, metric=metric, positions=positions)
            out.append(pd.DataFrame(dict(base, unit="nats", value=x)))
            if positions == "from1":  # bits per byte: summed over positions 1+ / ln 2 / UTF-8 bytes of those tokens
                out.append(pd.DataFrame(dict(base, unit="bpb", value=x * (d.n_tokens - 1) / LN2 / d.n_bytes_from1)))
    return pd.concat(out, ignore_index=True)


def wide(stages):
    """{corpus: frame indexed by exchange_id, columns (stage, role, framing, metric, positions, unit)}."""
    parts = []
    for stage, s in stages.items():
        p = per_target(s["scores"])
        p["stage"] = stage
        parts.append(p)
    long = pd.concat(parts, ignore_index=True)
    cols = ["stage", "role", "framing", "metric", "positions", "unit"]
    out = {}
    for c in CORPORA:
        lc = long[long.corpus == c]
        if len(lc):
            out[c] = lc.pivot_table(index="exchange_id", columns=cols, values="value", aggfunc="first").sort_index()
    return out


def position_bins(tokens, framing="tagged"):
    """Per (corpus, exchange_id, scored_text, bin): mean nll, entropy, gap of the target's tokens in that content-position bin."""
    t = tokens[(tokens.framing == framing) & (tokens.position >= 1)].copy()
    edges = [POSITION_BINS[0][0] - 1] + [b[1] for b in POSITION_BINS]
    t["bin"] = pd.cut(t.position, edges, labels=[f"{a}-{b}" for a, b in POSITION_BINS])
    t = t[t.bin.notna()]
    m = t.groupby(["corpus", "exchange_id", "scored_text", "bin"], observed=True)[["nll", "entropy"]].mean()
    m["gap"] = m.nll - m.entropy
    return m


# --- bootstrap -------------------------------------------------------------------------------------------------------------

def resample_counts(n, n_boot, rng):
    """[n_boot, n]: how often each exchange is drawn in each replicate (one joint resample of exchange IDs)."""
    idx = rng.integers(0, n, size=(n_boot, n))
    rows = np.repeat(np.arange(n_boot), n)
    c = np.zeros((n_boot, n))
    np.add.at(c, (rows, idx.ravel()), 1.0)
    return c


def boot_stats(vectors, counts):
    """{name: (mean, lo, hi, n)} for per-exchange vectors (NaN = not available), every one from the same replicate counts."""
    names = list(vectors)
    m = np.column_stack([np.asarray(vectors[k], float) for k in names])
    valid = np.isfinite(m)
    m0 = np.where(valid, m, 0.0)
    num, den = counts @ m0, counts @ valid.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = num / den
    out = {}
    for j, k in enumerate(names):
        n = int(valid[:, j].sum())
        mean = float(m0[:, j].sum() / n) if n else float("nan")
        if n >= MIN_BIN_N or (n and n == len(m)):
            lo, hi = np.nanpercentile(reps[:, j], [2.5, 97.5])
        else:
            lo = hi = float("nan")
        out[k] = (mean, float(lo), float(hi), n)
    return out


def family_vectors(w, stages):
    """{row key: per-exchange vector} for every quantity of one family and corpus (w: wide frame for that corpus)."""
    v = {}
    for unit in sorted(set(w.columns.get_level_values("unit"))):
        for positions in POSITIONS:
            for metric in METRICS:
                col = lambda st, role, fr: (st, role, fr, metric, positions, unit)
                if col("base", "user", "tagged") not in w.columns:
                    continue
                get = lambda st, role, fr: w[col(st, role, fr)].to_numpy(float)
                for st in stages:
                    for fr in FRAMINGS:
                        x = {r: get(st, r, fr) for r in ROLES}
                        xb = {r: get("base", r, fr) for r in ROLES}
                        for r in ROLES:
                            v[("level", st, r, fr, metric, positions, unit)] = x[r]
                            if st != "base":
                                v[("change_from_base", st, r, fr, metric, positions, unit)] = x[r] - xb[r]
                        v[("asst_minus_user", st, "reply-user", fr, metric, positions, unit)] = x["reply"] - x["user"]
                        if st != "base":
                            v[("asst_minus_user_change", st, "reply-user", fr, metric, positions, unit)] = (
                                (x["reply"] - x["user"]) - (xb["reply"] - xb["user"]))
                    # framing context: tagged - plain per role and for the contrast, and its change from base
                    tp = {r: get(st, r, "tagged") - get(st, r, "plain") for r in ROLES}
                    tpb = {r: get("base", r, "tagged") - get("base", r, "plain") for r in ROLES}
                    tp["reply-user"], tpb["reply-user"] = tp["reply"] - tp["user"], tpb["reply"] - tpb["user"]
                    for r in ("user", "reply", "reply-user"):
                        v[("tagged_minus_plain", st, r, "tagged-plain", metric, positions, unit)] = tp[r]
                        if st != "base":
                            v[("tagged_minus_plain_change", st, r, "tagged-plain", metric, positions, unit)] = tp[r] - tpb[r]
    return v


ROW_COLS = ["quantity", "stage", "role", "framing", "metric", "positions", "unit"]


def rng_for(seed, family, corpus, subset):
    """Independent stream per (family, corpus, subset): adding or dropping a family never changes another's CIs."""
    fi = list(config.FAMILIES).index(family)
    return np.random.default_rng([seed, fi, CORPORA.index(corpus), SUBSETS.index(subset)])


def reply_truncated(scores):
    r = scores[(scores.scored_text == "reply") & (scores.framing == "tagged")]
    return pd.Series(r.truncated.astype(bool).to_numpy(), index=pd.MultiIndex.from_frame(r[["corpus", "exchange_id"]]))


def summarise(data, n_boot, seed, common_ids):
    """Every quantity of every family, corpus and subset: one frame (family, corpus, subset, ROW_COLS, mean, lo, hi, n)."""
    rows = []
    for family, stages in data.items():
        trunc = reply_truncated(stages["base"]["scores"])
        for corpus, w in wide(stages).items():
            tr = trunc.xs(corpus).reindex(w.index).to_numpy()
            subsets = {"all": w.index, "untruncated": w.index[~tr], "common": w.index.intersection(common_ids.get(corpus, []))}
            for subset, ids in subsets.items():
                if len(ids) == 0:
                    continue
                vec = family_vectors(w.loc[ids], list(stages))
                if subset != "common":  # bits per byte only for the cross-family subset
                    vec = {k: x for k, x in vec.items() if k[-1] == "nats"}
                counts = resample_counts(len(ids), n_boot, rng_for(seed, family, corpus, subset))
                rows += [dict(family=family, corpus=corpus, subset=subset, **dict(zip(ROW_COLS, k)), mean=m, lo=lo, hi=hi, n=n)
                         for k, (m, lo, hi, n) in boot_stats(vec, counts).items()]
    return pd.DataFrame(rows)


def position_profiles(data, n_boot, seed):
    """Tagged: per content-position bin, level per stage and role, change from base per role, and the (reply - user) change.
    n = targets (or exchanges, for the contrast) contributing to the bin, so changing coverage is visible."""
    rows = []
    labels = [f"{a}-{b}" for a, b in POSITION_BINS]
    for family, stages in data.items():
        bins = {st: position_bins(s["tokens"]) for st, s in stages.items()}
        for corpus in CORPORA:
            ids = sorted(set(stages["base"]["scores"].query("corpus == @corpus").exchange_id))
            if not ids:
                continue
            vec = {}
            for metric in METRICS:
                for bl in labels:
                    def get(st, r):
                        idx = pd.MultiIndex.from_product([[corpus], ids, [r], [bl]])
                        return bins[st][metric].reindex(idx).to_numpy(float)
                    xb = {r: get("base", r) for r in ROLES}
                    for st in stages:
                        x = {r: get(st, r) for r in ROLES}
                        for r in ROLES:
                            vec[("level", st, r, bl, metric)] = x[r]
                            if st != "base":
                                vec[("change_from_base", st, r, bl, metric)] = x[r] - xb[r]
                        if st != "base":
                            vec[("asst_minus_user_change", st, "reply-user", bl, metric)] = (x["reply"] - x["user"]) - (xb["reply"] - xb["user"])
            counts = resample_counts(len(ids), n_boot, rng_for(seed, family, corpus, "positions"))
            for k, (m, lo, hi, n) in boot_stats(vec, counts).items():
                rows.append(dict(family=family, corpus=corpus, quantity=k[0], stage=k[1], role=k[2], bin=k[3], metric=k[4],
                                 mean=m, lo=lo, hi=hi, n=n))
    return pd.DataFrame(rows)


ORIGINAL = {True: "met", False: "not met", None: "incomplete (a headline corpus is missing)"}
TWO_SIDED = {"positive": "positive shift", "negative": "negative shift", "neither": "no two-corpus directional consistency",
             None: "incomplete (a headline corpus is missing)"}
SENSITIVITIES = (("all", "from0"), ("untruncated", "from1"), ("untruncated", "from0"))
NEAR_FLOOR = 0.01  # results.md lists unrounded any D within this distance of +-floor


def cell_direction(mean, lo, hi, floor=None):
    """One corpus cell of D, on unrounded values: 'positive' if mean >= +floor and the CI lies above zero, 'negative' if
    mean <= -floor and the CI lies below zero, else 'neither' (also when the CI is missing). floor: config.MIN_EFFECT_NATS."""
    floor = config.MIN_EFFECT_NATS if floor is None else floor
    if not (np.isfinite(mean) and np.isfinite(lo) and np.isfinite(hi)):
        return "neither"
    if mean >= floor and lo > 0:
        return "positive"
    if mean <= -floor and hi < 0:
        return "negative"
    return "neither"


def classify(directions):
    """(original positive-direction criterion met?, exploratory two-sided class) from the headline corpora's cell directions.
    Original (README criterion 1, fixed 2026-11-02): positive in every headline corpus. Exploratory (added 2026-11-04 after the
    results, not pre-registered): the same floor applied symmetrically, one direction required in every headline corpus."""
    if len(directions) < len(HEADLINE):
        return None, None
    d = set(directions)
    return d == {"positive"}, (next(iter(d)) if len(d) == 1 else "neither")


def verdicts(summary, subset="all", positions="from1"):
    """Per post-trained checkpoint, D = the tagged gap's asst_minus_user_change in oasst1 and UltraChat (nats/token): the original
    positive-direction gap criterion (criterion 1) and the exploratory two-sided directional-consistency classification."""
    s = summary[(summary.quantity == "asst_minus_user_change") & (summary.framing == "tagged") & (summary.metric == "gap")
                & (summary.positions == positions) & (summary.unit == "nats") & (summary.subset == subset)]
    out = []
    for (family, stage), g in s.groupby(["family", "stage"], sort=False):
        cells = {c: g[g.corpus == c].iloc[0] for c in HEADLINE if (g.corpus == c).any()}
        row = dict(family=family, stage=stage)
        for c, r in cells.items():
            row[f"{c}_mean"], row[f"{c}_lo"], row[f"{c}_hi"] = r["mean"], r.lo, r.hi
            row[f"{c}_direction"] = cell_direction(r["mean"], r.lo, r.hi)
        met, two = classify([row[f"{c}_direction"] for c in cells])
        row["original_positive_criterion"] = ORIGINAL[met]
        row["exploratory_two_sided"] = TWO_SIDED[two]
        out.append(row)
    return pd.DataFrame(out)


def sensitivity_verdicts(summary):
    """verdicts() on the labelled sensitivity analyses (positions 0+, untruncated-reply exchanges), stacked."""
    frames = [verdicts(summary, sub, pos).assign(subset=sub, positions=pos) for sub, pos in SENSITIVITIES]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def decomposition(summary):
    """D_gap = D_nll - D_entropy per checkpoint x headline corpus (tagged, positions 1+); each part with its own CI (descriptive)."""
    s = summary[(summary.quantity == "asst_minus_user_change") & (summary.framing == "tagged") & (summary.positions == "from1")
                & (summary.unit == "nats") & (summary.subset == "all") & summary.corpus.isin(HEADLINE)]
    rows = []
    for (f, st, c), g in s.groupby(["family", "stage", "corpus"], sort=False):
        g = g.set_index("metric")
        rows.append(dict(family=f, stage=st, corpus=c,
                         d_nll=g.loc["nll", "mean"], d_nll_lo=g.loc["nll", "lo"], d_nll_hi=g.loc["nll", "hi"],
                         minus_d_entropy=-g.loc["entropy", "mean"], minus_d_entropy_lo=-g.loc["entropy", "hi"],
                         minus_d_entropy_hi=-g.loc["entropy", "lo"],
                         d_gap=g.loc["gap", "mean"], d_gap_lo=g.loc["gap", "lo"], d_gap_hi=g.loc["gap", "hi"]))
    return pd.DataFrame(rows)


def sample_table(data):
    """n exchanges, reply truncation (first 300 family tokens) and median token counts, per family x corpus."""
    rows = []
    for family, stages in data.items():
        d = stages["base"]["scores"]
        d = d[d.framing == "tagged"]
        for c in CORPORA:
            u, r = d[(d.corpus == c) & (d.scored_text == "user")], d[(d.corpus == c) & (d.scored_text == "reply")]
            if len(u):
                rows.append(dict(family=family, corpus=c, exchanges=len(u), replies_truncated=int(r.truncated.sum()),
                                 truncation_rate=float(r.truncated.mean()), median_user_tokens=float(u.n_tokens.median()),
                                 median_reply_tokens_scored=float(r.n_tokens.median()), median_reply_tokens_full=float(r.n_tokens_full.median())))
    return pd.DataFrame(rows)


# --- outputs ---------------------------------------------------------------------------------------------------------------

def label(family, stage=None):
    s = config.family_settings(family)
    return s["label"] if stage is None else f"{s['label']} {s['stage_labels'].get(stage, stage)}"


def fmt(m, lo, hi, d=3):
    if not np.isfinite(lo):
        return f"{m:+.{d}f}"
    return f"{m:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def provenance(args, data, checks):
    def git(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    return dict(command="python -m measures.logit_distortion.analyze " + " ".join(sys.argv[1:] if args.argv is None else args.argv),
                git_revision=git("rev-parse", "HEAD"),
                git_dirty_analysis_code=bool(git("status", "--porcelain", "--", os.path.dirname(os.path.relpath(__file__)))),
                analysis_code_sha256={os.path.basename(p): sha256(p) for p in (__file__, os.path.join(os.path.dirname(__file__), "plots.py"))
                                      if os.path.exists(p)},
                settings=dict(n_boot=args.n_boot, seed=args.seed, floor_nats=config.MIN_EFFECT_NATS, ci="95% percentile",
                              resampling="exchange IDs jointly within family x corpus (pairs roles, framings, checkpoints)",
                              positions_main="from1", position_bins=[f"{a}-{b}" for a, b in POSITION_BINS], min_bin_n=MIN_BIN_N,
                              root=args.root, tag=args.tag),
                inputs={f"{f} {st}": {n: dict(path=p, sha256=h) for n, (p, h) in s["files"].items()}
                        for f, stages in data.items() for st, s in stages.items()},
                inputs_fingerprints={f: c["inputs_fingerprint"] for f, c in checks.items()})


def markdown(summary, verd, samples, checks, prov, common_n, sens=None, decomp=None):
    floor = config.MIN_EFFECT_NATS
    sens = pd.DataFrame() if sens is None else sens
    decomp = decomposition(summary) if decomp is None else decomp
    L = ["# Logit distortion: exchange design (tables)", "",
         f"Generated by `{prov['command']}` at `{prov['git_revision']}`. Spec: `measures/logit_distortion/README.md`, "
         "\"Exchange-design analysis (2026-11-04)\" (decided after the data existed: **not blind**). Findings and interpretation: `report.md`.", "",
         f"Means over content positions 1+ in nats per token unless stated; 95% bootstrap CIs ({prov['settings']['n_boot']} replicates, seed "
         f"{prov['settings']['seed']}), exchange IDs resampled jointly. reply = the assistant persona (scored after its question); user = the "
         "opening user turn. Contamination could be checked for OLMo only: **Qwen3 and Gemma 4 results cannot exclude training-data overlap.**", "",
         "## Gap contrast D: original positive-direction criterion and exploratory two-sided classification", "",
         "D = (gap_reply − gap_user)_post − (gap_reply − gap_user)_base, tagged, positions 1+, nats/token, paired per exchange; "
         "gap = mean NLL − mean entropy.", "",
         f"- **Original positive-direction gap criterion** (README criterion 1, fixed 2026-11-02): D ≥ +{floor} with the 95% CI above zero, "
         "in both oasst1 and UltraChat. It tests one directional hypothesis (the assistant gap grows more than the user gap); "
         "\"not met\" is not evidence against assistant privilege in general.",
         f"- **Exploratory two-sided classification** (added 2026-11-04 after these results; not pre-registered): positive shift as above; "
         f"negative shift if D ≤ −{floor} with the CI below zero in both corpora; otherwise no two-corpus directional consistency. "
         f"±{floor} applies the original floor symmetrically; it is not a newly validated threshold. Either shift is a role-dependent "
         "gap change under this design, not by itself assistant privilege or its absence. Thresholds are applied to unrounded estimates.", "",
         "| checkpoint | oasst1 (human column) | UltraChat (model column) | original positive-direction criterion | exploratory two-sided |",
         "|---|---|---|---|---|"]
    for _, v in verd.iterrows():
        L.append(f"| {label(v.family, v.stage)} | {fmt(v.oasst1_mean, v.oasst1_lo, v.oasst1_hi)} | "
                 f"{fmt(v.ultrachat_mean, v.ultrachat_lo, v.ultrachat_hi)} | {v.original_positive_criterion} | {v.exploratory_two_sided} |")
    near = [(label(v.family, v.stage), c, v[f"{c}_mean"], v[f"{c}_lo"], v[f"{c}_hi"]) for _, v in verd.iterrows() for c in HEADLINE
            if f"{c}_mean" in v and abs(abs(v[f"{c}_mean"]) - floor) < NEAR_FLOOR]
    if near:
        L += ["", f"Within {NEAR_FLOOR} nats of ±{floor} (unrounded; the classification is fragile to small changes): "
              + "; ".join(f"{k} {c} D = {m:+.6f} [{lo:+.6f}, {hi:+.6f}]" for k, c, m, lo, hi in near) + "."]
    if len(sens):
        L += ["", "Classification under the labelled sensitivity analyses (same rules):", "",
              "| checkpoint | analysis | oasst1 | UltraChat | original positive-direction criterion | exploratory two-sided |",
              "|---|---|---|---|---|---|"]
        name = {("all", "from0"): "positions 0+", ("untruncated", "from1"): "untruncated replies", ("untruncated", "from0"): "untruncated, 0+"}
        for _, v in sens.iterrows():
            L.append(f"| {label(v.family, v.stage)} | {name[(v.subset, v.positions)]} | {fmt(v.oasst1_mean, v.oasst1_lo, v.oasst1_hi)} | "
                     f"{fmt(v.ultrachat_mean, v.ultrachat_lo, v.ultrachat_hi)} | {v.original_positive_criterion} | {v.exploratory_two_sided} |")
    L += ["", "### Components of D (tagged, positions 1+): D_gap = D_NLL − D_entropy", "",
          "D_NLL > 0: the reply's NLL rose more (or fell less) than the user turn's. −D_entropy > 0: the reply's entropy fell more (or rose "
          "less) than the user turn's. Either part can produce a positive or a negative D; the sign of D alone does not identify which. "
          "Each part has its own CI (descriptive).", "",
          "| checkpoint | corpus | D_NLL | −D_entropy | D_gap |", "|---|---|---|---|---|"]
    for _, r in decomp.iterrows():
        L.append(f"| {label(r.family, r.stage)} | {r.corpus} | {fmt(r.d_nll, r.d_nll_lo, r.d_nll_hi)} | "
                 f"{fmt(r.minus_d_entropy, r.minus_d_entropy_lo, r.minus_d_entropy_hi)} | {fmt(r.d_gap, r.d_gap_lo, r.d_gap_hi)} |")
    L += ["", "## Samples and reply truncation (replies scored over their first 300 family tokens; truncated replies kept)", "",
          "| family | corpus | exchanges | truncated replies | median user tokens | median reply tokens (scored / full) |", "|---|---|---|---|---|---|"]
    for _, r in samples.iterrows():
        L.append(f"| {label(r.family)} | {r.corpus} | {r.exchanges} | {r.replies_truncated} ({r.truncation_rate:.0%}) | "
                 f"{r.median_user_tokens:.0f} | {r.median_reply_tokens_scored:.0f} / {r.median_reply_tokens_full:.0f} |")
    main = summary[(summary.subset == "all") & (summary.positions == "from1") & (summary.unit == "nats")]

    def table(title, q, framing, roles, d=None, show_n=False):
        d = main if d is None else d
        d = d[(d.quantity == q) & (d.framing == framing) & d.role.isin(roles)]
        out = ["", f"## {title}", "", "| checkpoint | corpus | role | " + ("n | " if show_n else "") + " | ".join(METRICS) + " |",
               "|---|---|---|" + ("---|" if show_n else "") + "---|" * len(METRICS)]
        for (f, st, c, r), g in d.groupby(["family", "stage", "corpus", "role"], sort=False):
            g = g.set_index("metric")
            out.append(f"| {label(f, st)} | {c} | {r} | " + (f"{g.n.iloc[0]} | " if show_n else "")
                       + " | ".join(fmt(g.loc[m, 'mean'], g.loc[m, 'lo'], g.loc[m, 'hi']) for m in METRICS) + " |")
        return out

    L += table("Levels, tagged", "level", "tagged", ROLES)
    L += table("Change from base, tagged", "change_from_base", "tagged", ROLES)
    L += table("Assistant − user (reply − user), tagged", "asst_minus_user", "tagged", ("reply-user",))
    L += table("Change of (reply − user) from base, tagged", "asst_minus_user_change", "tagged", ("reply-user",))
    L += table("Levels, plain (context)", "level", "plain", ROLES)
    L += table("Change from base, plain (context)", "change_from_base", "plain", ROLES)
    L += table("Change of (reply − user) from base, plain (context)", "asst_minus_user_change", "plain", ("reply-user",))
    L += table("Tagged − plain, change from base (context)", "tagged_minus_plain_change", "tagged-plain", ("user", "reply", "reply-user"))
    for title, sub, pos, unit in (("Sensitivity: positions 0+ (first scored token included)", "all", "from0", "nats"),
                                  ("Sensitivity: exchanges with an untruncated reply only", "untruncated", "from1", "nats"),
                                  (f"Cross-family, common exchange IDs ({common_n}), bits per byte", "common", "from1", "bpb")):
        d = summary[(summary.subset == sub) & (summary.positions == pos) & (summary.unit == unit)]
        L += table(f"{title}: change of (reply − user) from base, tagged", "asst_minus_user_change", "tagged", ("reply-user",), d, True)
    L += ["", "## Validation", "", "| family | stages | exchanges per corpus | inputs fingerprint | worst \\|token mean − scores.csv\\| | missing stages |",
          "|---|---|---|---|---|---|"]
    for f, c in checks.items():
        worst = max(max(v["max_abs_diff_token_means"].values()) for v in c["per_stage"].values())
        L.append(f"| {label(f)} | {', '.join(c['stages'])} | {c['n_exchanges']} | `{c['inputs_fingerprint'][:12]}…` (shared) | {worst:.1e} | "
                 f"{', '.join(c['missing_stages']) or 'none'} |")
    L += ["", "Checked per stage: no duplicated condition; every exchange has its 4 conditions (user / reply × tagged / plain); finite values; "
          "tagged and plain score the same number of tokens; tokens.parquet has positions 0..n−1 for every target and its means equal "
          "scores.csv (positions 1+ and 0+); manifest family/stage. Per family: one inputs fingerprint, identical exchange IDs and "
          "truncation flags across stages."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.path.join("results", config.EXCHANGE_RUN))
    ap.add_argument("--tag", default="exchanges_" + config.EXCHANGE_CONTEXT)
    ap.add_argument("--out", default=None, help="default: <root>/logit_distortion/<tag>")
    ap.add_argument("--families", nargs="*", default=None, help="default: every configured family with outputs")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    args.argv = argv
    out = args.out or os.path.join(args.root, "logit_distortion", args.tag)
    data, checks = {}, {}
    for family in args.families or list(config.FAMILIES):
        stages = load_family(args.root, family, args.tag)
        if not stages:
            if args.families:
                raise InputError(f"{family}: no outputs under {args.root}/{family}/<stage>/core/{args.tag}")
            continue
        checks[family] = validate_family(family, stages)
        data[family] = stages
    if not data:
        raise InputError(f"no exchange scores under {args.root}")
    common = {c: sorted(set.intersection(*[set(s["base"]["scores"].query("corpus == @c").exchange_id) for s in data.values()]))
              for c in CORPORA}
    summary = summarise(data, args.n_boot, args.seed, common)
    verd = verdicts(summary)
    sens = sensitivity_verdicts(summary)
    decomp = decomposition(summary)
    samples = sample_table(data)
    pos = position_profiles(data, args.n_boot, args.seed)
    os.makedirs(out, exist_ok=True)
    summary.to_csv(os.path.join(out, "summary.csv"), index=False, float_format="%.6f")
    verd.to_csv(os.path.join(out, "verdict.csv"), index=False, float_format="%.6f")
    sens.to_csv(os.path.join(out, "verdict_sensitivity.csv"), index=False, float_format="%.6f")
    decomp.to_csv(os.path.join(out, "decomposition.csv"), index=False, float_format="%.6f")
    samples.to_csv(os.path.join(out, "samples.csv"), index=False, float_format="%.6f")
    pos.to_csv(os.path.join(out, "positions.csv"), index=False, float_format="%.6f")
    prov = provenance(args, data, checks)
    json.dump(dict(checks=checks, common_ids={c: len(v) for c, v in common.items()}), open(os.path.join(out, "checks.json"), "w"), indent=1)
    json.dump(prov, open(os.path.join(out, "provenance.json"), "w"), indent=1)
    common_n = ", ".join(f"{c} {len(v)}" for c, v in common.items())
    open(os.path.join(out, "results.md"), "w").write(markdown(summary, verd, samples, checks, prov, common_n, sens, decomp))
    if not args.no_plots:
        from measures.logit_distortion import plots
        plots.all_figures(summary, pos, out)
    print(f"wrote {out}")
    for _, v in verd.iterrows():
        print(f"  {label(v.family, v.stage):14s} oasst1 {fmt(v.oasst1_mean, v.oasst1_lo, v.oasst1_hi)}  "
              f"UltraChat {fmt(v.ultrachat_mean, v.ultrachat_lo, v.ultrachat_hi)}  original positive-direction criterion: "
              f"{v.original_positive_criterion}; exploratory two-sided: {v.exploratory_two_sided}")
    return out


if __name__ == "__main__":
    main()
