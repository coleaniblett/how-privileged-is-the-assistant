"""CPU-only tests of measures.logit_distortion.analyze on synthetic exchange-score outputs (no model, no real data)."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from measures.logit_distortion import analyze as A

FAMILY = "qwen3-8b"  # stages base, instruct
TAG = "exchanges_exchange"


def make_tokens(corpus, ex, role, framing, nll, ent):
    n = len(nll)
    return pd.DataFrame(dict(family=FAMILY, corpus=corpus, exchange_id=ex, scored_text=role, framing=framing,
                             position=np.arange(n, dtype=np.int16), nll=np.asarray(nll, np.float32), entropy=np.asarray(ent, np.float32)))


def write_stage(root, stage, token_fn, n_ex=12, corpora=("oasst1", "ultrachat"), fingerprint="fp", truncated=lambda c, i: i % 4 == 0):
    """token_fn(stage, corpus, i, role, framing) -> (nll list, entropy list). Writes scores.csv, tokens.parquet, manifest.json."""
    d = os.path.join(root, FAMILY, stage, "core", TAG)
    os.makedirs(d, exist_ok=True)
    rows, toks = [], []
    for c in corpora:
        for i in range(n_ex):
            ex = f"{c}-{i:04d}"
            for role in A.ROLES:
                for fr in A.FRAMINGS:
                    nll, ent = (np.asarray(v, float) for v in token_fn(stage, c, i, role, fr))
                    t = make_tokens(c, ex, role, fr, nll, ent)
                    toks.append(t)
                    nll, ent = t.nll.to_numpy(float), t.entropy.to_numpy(float)
                    rows.append(dict(family=FAMILY, stage=stage, corpus=c, exchange_id=ex, scored_text=role, framing=fr, context="exchange",
                                     condition="x", n_tokens=len(nll), n_tokens_full=len(nll) + (5 if role == "reply" and truncated(c, i) else 0),
                                     truncated=bool(role == "reply" and truncated(c, i)), mean_nll=nll.mean(), mean_nll_from1=nll[1:].mean(),
                                     mean_entropy=ent.mean(), mean_entropy_from1=ent[1:].mean(), n_bytes_from1=4 * (len(nll) - 1),
                                     score_source="fresh"))
    pd.DataFrame(rows).to_csv(os.path.join(d, "scores.csv"), index=False, float_format="%.6f")
    pd.concat(toks, ignore_index=True).to_parquet(os.path.join(d, "tokens.parquet"))
    json.dump(dict(family=FAMILY, stage=stage, inputs_fingerprint=fingerprint, exclusions=[], model="m", model_revision="r"),
              open(os.path.join(d, "manifest.json"), "w"))
    return d


def constant_fn(effect=0.0):
    """Position 0 is wild (must be excluded from from1); positions 1+ are constant per target.
    Gap = nll - entropy at positions 1+: user 0.2, reply 0.2 at base; at instruct the reply's gap grows by `effect`."""
    def fn(stage, c, i, role, fr):
        n = 6 + i % 3
        ent = [5.0] + [2.0] * (n - 1)
        g = 0.2 + (effect if (stage == "instruct" and role == "reply") else 0.0)
        nll = [9.0] + [2.0 + g] * (n - 1)
        return nll, ent
    return fn


def run(root, tmp_path, n_boot=200, plots=False):
    out = str(tmp_path / "out")
    argv = ["--root", str(root), "--tag", TAG, "--out", out, "--families", FAMILY, "--n-boot", str(n_boot)]
    A.main(argv + ([] if plots else ["--no-plots"]))
    return out


def test_token_means_select_positions():
    t = pd.concat([make_tokens("oasst1", "e", "user", "tagged", [10.0, 1.0, 3.0], [0.0, 1.0, 1.0])])
    m1, m0 = A.token_means(t, 1).iloc[0], A.token_means(t, 0).iloc[0]
    assert m1.nll == pytest.approx(2.0) and m1.entropy == pytest.approx(1.0) and m1.gap == pytest.approx(1.0) and m1.n == 2
    assert m0.nll == pytest.approx(14 / 3) and m0.gap == pytest.approx(14 / 3 - 2 / 3) and m0.n == 3


def test_per_target_gap_uses_same_positions(tmp_path):
    write_stage(tmp_path, "base", constant_fn())
    d = pd.read_csv(tmp_path / FAMILY / "base" / "core" / TAG / "scores.csv")
    p = A.per_target(d)
    g1 = p[(p.metric == "gap") & (p.positions == "from1") & (p.unit == "nats")].value
    assert np.allclose(g1, 0.2, atol=1e-5)  # position 0 (nll 9, entropy 5) excluded
    g0 = p[(p.metric == "gap") & (p.positions == "from0") & (p.unit == "nats")].value
    assert (g0 > 0.2).all()  # included in the sensitivity
    bpb = p[(p.metric == "nll") & (p.unit == "bpb")].reset_index(drop=True)
    nats = p[(p.metric == "nll") & (p.positions == "from1") & (p.unit == "nats")].reset_index(drop=True)
    assert np.allclose(bpb.value, nats.value / 4 / np.log(2))  # 4 bytes per token in the fixture


def test_paired_contrast_and_verdict(tmp_path):
    write_stage(tmp_path, "base", constant_fn())
    write_stage(tmp_path, "instruct", constant_fn(effect=0.3))
    out = run(tmp_path, tmp_path)
    s = pd.read_csv(os.path.join(out, "summary.csv"))
    sel = s[(s.quantity == "asst_minus_user_change") & (s.metric == "gap") & (s.framing == "tagged") & (s.positions == "from1")
            & (s.subset == "all")]
    assert len(sel) == 2
    # every exchange has the same paired difference, so the joint bootstrap CI collapses onto it
    assert np.allclose(sel["mean"], 0.3, atol=1e-4) and np.allclose(sel.lo, 0.3, atol=1e-4) and np.allclose(sel.hi, 0.3, atol=1e-4)
    ch = s[(s.quantity == "change_from_base") & (s.metric == "gap") & (s.framing == "tagged") & (s.positions == "from1") & (s.subset == "all")]
    assert np.allclose(ch[ch.role == "user"]["mean"], 0, atol=1e-4) and np.allclose(ch[ch.role == "reply"]["mean"], 0.3, atol=1e-4)
    v = pd.read_csv(os.path.join(out, "verdict.csv")).iloc[0]
    assert v.verdict == "criterion 1 met (logit distortion present)"
    sub = s[(s.subset == "untruncated") & (s.quantity == "level") & (s.stage == "base") & (s.role == "user") & (s.metric == "gap")
            & (s.framing == "tagged") & (s.positions == "from1")]
    assert (sub.n == 9).all()  # 3 of 12 replies truncated per corpus in the fixture; truncated replies are kept in "all"
    assert (s[(s.subset == "all")].n == 12).all()


def test_below_floor_not_met(tmp_path):
    write_stage(tmp_path, "base", constant_fn())
    write_stage(tmp_path, "instruct", constant_fn(effect=0.03))
    out = run(tmp_path, tmp_path)
    v = pd.read_csv(os.path.join(out, "verdict.csv")).iloc[0]
    assert v.verdict == "criterion 1 not met" and not v.oasst1_meets


def test_joint_bootstrap_pairs_exchanges():
    """A noisy per-exchange level that is identical in both roles gives an exactly-zero contrast CI only if roles share a resample."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    counts = A.resample_counts(50, 500, np.random.default_rng(1))
    assert counts.sum(axis=1).tolist() == [50] * 500
    st = A.boot_stats({"a": x, "b": x + 1.0, "d": (x + 1.0) - x}, counts)
    assert st["d"][1] == pytest.approx(1.0) and st["d"][2] == pytest.approx(1.0)
    assert st["b"][1] - 1.0 == pytest.approx(st["a"][1]) and st["a"][2] > st["a"][1]


def test_boot_stats_nan_and_small_n():
    counts = A.resample_counts(20, 100, np.random.default_rng(0))
    v = np.full(20, np.nan)
    v[:5] = 1.0
    m, lo, hi, n = A.boot_stats({"v": v}, counts)["v"]
    assert m == 1.0 and n == 5 and np.isnan(lo)  # fewer than MIN_BIN_N contributors: no CI


def test_reproducible(tmp_path):
    write_stage(tmp_path, "base", constant_fn())

    def noisy(stage, c, i, role, fr):
        r = np.random.default_rng(abs(hash((stage, c, i, role, fr))) % 2**32)
        n = 8
        return list(2 + r.random(n)), list(2 + r.random(n))
    write_stage(tmp_path, "base", noisy)
    write_stage(tmp_path, "instruct", noisy)
    a = pd.read_csv(os.path.join(run(tmp_path, tmp_path / "a"), "summary.csv"))
    b = pd.read_csv(os.path.join(run(tmp_path, tmp_path / "b"), "summary.csv"))
    pd.testing.assert_frame_equal(a, b)


# --- missing and mismatched inputs -----------------------------------------------------------------------------------------

def stages(root):
    return A.load_family(str(root), FAMILY, TAG)


def test_missing_condition(tmp_path):
    write_stage(tmp_path, "base", constant_fn())
    d = write_stage(tmp_path, "instruct", constant_fn())
    s = pd.read_csv(os.path.join(d, "scores.csv"))
    s.iloc[1:].to_csv(os.path.join(d, "scores.csv"), index=False)
    with pytest.raises(A.InputError, match="of 4 conditions"):
        A.validate_family(FAMILY, stages(tmp_path))


def test_duplicate_condition(tmp_path):
    write_stage(tmp_path, "base", constant_fn())
    d = write_stage(tmp_path, "instruct", constant_fn())
    s = pd.read_csv(os.path.join(d, "scores.csv"))
    pd.concat([s, s.iloc[:1]]).to_csv(os.path.join(d, "scores.csv"), index=False)
    with pytest.raises(A.InputError, match="duplicated"):
        A.validate_family(FAMILY, stages(tmp_path))


def test_mismatched_ids_across_stages(tmp_path):
    write_stage(tmp_path, "base", constant_fn(), n_ex=12)
    write_stage(tmp_path, "instruct", constant_fn(), n_ex=11)
    with pytest.raises(A.InputError, match="exchange IDs differ"):
        A.validate_family(FAMILY, stages(tmp_path))


def test_mismatched_fingerprint(tmp_path):
    write_stage(tmp_path, "base", constant_fn())
    write_stage(tmp_path, "instruct", constant_fn(), fingerprint="other")
    with pytest.raises(A.InputError, match="fingerprint"):
        A.validate_family(FAMILY, stages(tmp_path))


def test_non_finite(tmp_path):
    write_stage(tmp_path, "base", constant_fn())
    d = write_stage(tmp_path, "instruct", constant_fn())
    s = pd.read_csv(os.path.join(d, "scores.csv"))
    s.loc[3, "mean_nll_from1"] = np.nan
    s.to_csv(os.path.join(d, "scores.csv"), index=False)
    with pytest.raises(A.InputError, match="non-finite"):
        A.validate_family(FAMILY, stages(tmp_path))


def test_token_means_disagree(tmp_path):
    write_stage(tmp_path, "base", constant_fn())
    d = write_stage(tmp_path, "instruct", constant_fn())
    s = pd.read_csv(os.path.join(d, "scores.csv"))
    s.loc[0, "mean_entropy_from1"] += 0.01
    s.to_csv(os.path.join(d, "scores.csv"), index=False)
    with pytest.raises(A.InputError, match="differ from scores.csv"):
        A.validate_family(FAMILY, stages(tmp_path))


def test_missing_base_and_missing_file(tmp_path):
    write_stage(tmp_path, "instruct", constant_fn())
    with pytest.raises(A.InputError, match="no base"):
        A.validate_family(FAMILY, stages(tmp_path))
    d = write_stage(tmp_path, "base", constant_fn())
    os.remove(os.path.join(d, "tokens.parquet"))
    with pytest.raises(A.InputError, match="tokens.parquet missing"):
        stages(tmp_path)


def test_truncation_flags_must_match(tmp_path):
    write_stage(tmp_path, "base", constant_fn())
    write_stage(tmp_path, "instruct", constant_fn(), truncated=lambda c, i: i % 3 == 0)
    with pytest.raises(A.InputError, match="truncation flags"):
        A.validate_family(FAMILY, stages(tmp_path))


def test_figures_written(tmp_path):
    write_stage(tmp_path, "base", constant_fn())
    write_stage(tmp_path, "instruct", constant_fn(effect=0.3))
    out = run(tmp_path, tmp_path, n_boot=50, plots=True)
    figs = sorted(os.listdir(os.path.join(out, "figures")))
    assert "contrast_change.png" in figs and "positions_oasst1.png" in figs and "levels_ultrachat.png" in figs
    assert os.path.exists(os.path.join(out, "results.md")) and os.path.exists(os.path.join(out, "provenance.json"))
