"""The locked vestigial-regression quantities (analyze.py --locked): asymmetry with independent bootstraps, the verdict rule,
the judge arm's rates, the arm combination and Spearman, and a regression test that reproduces the pilot's report.md numbers."""

import numpy as np
import pandas as pd
import pytest

from core import config
from measures.vestigial_regression import analyze as A


def wide(values, source, kind):
    """A wide per-text frame for one set from {condition: array}."""
    n = len(next(iter(values.values())))
    idx = pd.MultiIndex.from_tuples([(source, kind, f"{source}-{kind}-{i}") for i in range(n)], names=["source", "kind", "text_id"])
    return pd.DataFrame(values, index=idx)


def test_asymmetry_combines_two_independently_resampled_sets():
    rng = np.random.default_rng(0)
    user = pd.Series(np.random.default_rng(1).normal(1.0, 0.5, 200))
    asst = pd.Series(np.random.default_rng(2).normal(-1.0, 0.5, 40))
    m, lo, hi = A.asymmetry_ci(user, asst, 4000, rng)
    assert abs(m - (user.mean() - asst.mean())) < 1e-12 and 1.7 < m < 2.3 and lo < m < hi
    # the CI reflects the smaller (assistant) set: its width is about sqrt(se_u^2 + se_a^2) * 3.92
    se = np.sqrt(user.std() ** 2 / 200 + asst.std() ** 2 / 40)
    assert abs((hi - lo) - 3.92 * se) < 0.06
    # sets of different sizes are allowed (no pairing between them)
    assert A.asymmetry_ci(pd.Series([1.0, 2, 3]), pd.Series([0.0, 1, 2, 3, 4]), 100, rng)[0] == 0.0  # means 2 and 2
    assert A.asymmetry_ci(pd.Series([3.0, 5.0]), pd.Series([0.0, 1, 2]), 100, rng)[0] == 4.0 - 1.0


def test_fires_needs_the_floor_and_a_ci_excluding_zero():
    assert A.fires(0.06, 0.01, 0.12) and not A.fires(0.04, 0.01, 0.07)  # below the 0.05-nat floor
    assert not A.fires(0.30, -0.02, 0.60)  # CI includes zero
    assert not A.fires(-0.30, -0.60, -0.10)  # the wrong direction never fires


def test_nll_verdict_needs_both_user_sources_in_the_same_direction():
    row = lambda src, m, lo, hi: dict(text_set=f"{src} user vs wildchat assistant", quantity="asymmetry = x", mean=m, lo=lo, hi=hi)
    ok, _ = A.nll_verdict([row("wildchat", 1.0, 0.5, 1.5), row("oasst1", 2.0, 1.0, 3.0)])
    assert ok
    assert not A.nll_verdict([row("wildchat", 1.0, 0.5, 1.5), row("oasst1", 0.2, -0.1, 0.5)])[0]


def test_delta_vectors_pair_texts_within_a_set():
    base = wide({"user@user": [1.0, 2.0, 3.0], "user@asst": [1.1, 2.1, 3.1], "user@plain": [1.5, 2.5, 3.5]}, "wildchat", "user")
    it = wide({"user@user": [2.0, 4.0, 6.0], "user@asst": [1.0, 2.0, 3.0], "user@plain": [2.5, 4.5, 6.5]}, "wildchat", "user")
    d, (own, other, plain) = A.delta_vectors(base, it, "wildchat", "user")
    assert (own, other, plain) == ("user@user", "user@asst", "user@plain")
    assert list(d["user@user"]) == [1.0, 2.0, 3.0] and list(d["user@plain"]) == [1.0, 2.0, 3.0] and list(d["rs"]) == [0.0, 0.0, 0.0]
    base_a = wide({"asst@user": [1.0, 1.0], "asst@asst": [1.0, 1.0], "asst@plain": [1.0, 1.0]}, "wildchat", "assistant")
    it_a = wide({"asst@user": [1.0, 1.0], "asst@asst": [0.0, -1.0], "asst@plain": [1.0, 1.0]}, "wildchat", "assistant")
    d, names = A.delta_vectors(base_a, it_a, "wildchat", "assistant")
    assert names == ("asst@asst", "asst@user", "asst@plain") and list(d["rs"]) == [-1.0, -2.0]


def judged(labels_by_prefill):
    return [dict(prefill_id=p, label=l) for p, ls in labels_by_prefill.items() for l in ls]


def test_judge_rate_change_resamples_prefills_and_separates_category_4():
    base = judged({f"p{i}": ["user_voice"] * 3 for i in range(20)})
    it = judged({f"p{i}": ["switched_to_assistant", "degenerate", "user_voice"] if i < 15 else ["user_voice_with_assistant_traits"] * 3
                 for i in range(20)})
    rng = np.random.default_rng(0)
    m, lo, hi, rb, rs = A.judge_change(base, it, A.JUDGE_NON_USER, 2000, rng)
    assert rb == 0.0 and abs(rs - 15 * (2 / 3) / 20) < 1e-9 and abs(m - rs) < 1e-9 and lo > 0
    m4, lo4, hi4, _, rs4 = A.judge_change(base, it, ("user_voice_with_assistant_traits",), 2000, rng)
    assert abs(rs4 - 5 / 20) < 1e-9  # reported separately, not counted as non-user-voice


def test_combine_names_the_pattern():
    assert A.combine(True, True).startswith("both arms") and A.combine(True, False).startswith("NLL only")
    assert A.combine(False, True).startswith("judge only") and A.combine(False, False) == "neither arm fired"


def test_spearman():
    assert A.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert A.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert A.spearman([1, 2], [1, 2]) is None and A.spearman([1, 1, 1], [1, 2, 3]) is None


def test_select_ids_takes_the_first_n_of_each_locked_set():
    rows = [dict(id=f"{s}-{k}-{i}", source=s, kind=k) for s, k in [("wildchat", "user"), ("oasst1", "user"), ("wildchat", "assistant"),
            ("olmo-instruct", "assistant")] for i in range(5)]
    ids = A.select_ids(rows, 2)
    assert len(ids) == 6 and "wildchat-user-1" in ids and "wildchat-user-2" not in ids and not any("olmo" in i for i in ids)


# --- the pilot's numbers (pilot/report.md, bits per byte summary table and the nats tables) ----------------------------------------

@pytest.fixture(scope="module")
def pilot_rows():
    from core.render import load_jsonl, load_tokenizer
    rows = load_jsonl(config.DATA_PATH)
    ids = A.select_ids(rows, 30)
    points = {f: dict(dir=d, stages=st, judge=None) for f, (d, st) in A.PILOT_POINTS.items()}
    nbytes = {f: A.text_bytes(load_tokenizer(config.FAMILIES[f]["template"]), rows, ids) for f in points}
    md, allrows = A.locked_analysis(points, 400, config.SEED, nbytes)
    return md, pd.DataFrame(allrows)


def value(df, fam, stage, unit, text_set, quantity_prefix):
    r = df[(df.family == fam) & (df.stage == stage) & (df.unit == unit) & (df.text_set == text_set) & df.quantity.str.startswith(quantity_prefix)]
    assert len(r) == 1, (fam, stage, unit, text_set, quantity_prefix)
    return r.iloc[0]


# (family, stage, text set, role-specific / delta quantity prefix, bits per byte as printed in the report's summary table)
BPB_REPORT = [
    ("olmo3-7b", "instruct", "wildchat user", "role-specific", -0.07), ("olmo3-7b", "instruct", "oasst1 user", "role-specific", -0.03),
    ("olmo3-7b", "instruct", "wildchat assistant", "role-specific", 0.01), ("olmo3-7b", "instruct", "wildchat user", "Δ(user@user)", -0.17),
    ("olmo3-7b", "instruct", "oasst1 user", "Δ(user@user)", 0.10), ("olmo3-7b", "instruct", "wildchat user", "Δ(user@plain)", -0.10),
    ("olmo3-7b", "sft", "wildchat assistant", "role-specific", -0.04),
    ("qwen3-8b", "instruct", "wildchat user", "role-specific", 0.14), ("qwen3-8b", "instruct", "oasst1 user", "role-specific", 0.13),
    ("qwen3-8b", "instruct", "wildchat assistant", "role-specific", 0.09), ("qwen3-8b", "instruct", "wildchat user", "Δ(user@user)", 0.20),
    ("gemma4-31b", "instruct", "wildchat user", "role-specific", 1.10), ("gemma4-31b", "instruct", "oasst1 user", "role-specific", 1.16),
    ("gemma4-31b", "instruct", "wildchat assistant", "role-specific", -0.82), ("gemma4-31b", "instruct", "wildchat user", "Δ(user@user)", 2.67),
    ("gemma4-31b", "instruct", "oasst1 user", "Δ(user@plain)", 1.58),
]


@pytest.mark.parametrize("fam,stage,text_set,q,expected", BPB_REPORT)
def test_pilot_bits_per_byte_matches_report(pilot_rows, fam, stage, text_set, q, expected):
    assert round(value(pilot_rows[1], fam, stage, "bits/byte", text_set, q)["mean"], 2) == expected


@pytest.mark.parametrize("fam,text_set,expected", [("gemma4-31b", "wildchat user", 2.943), ("gemma4-31b", "oasst1 user", 3.630),
                                                  ("gemma4-31b", "wildchat assistant", -2.172), ("qwen3-8b", "wildchat user", 0.436),
                                                  ("qwen3-8b", "oasst1 user", 0.385), ("qwen3-8b", "wildchat assistant", 0.297)])
def test_pilot_nats_matches_report(pilot_rows, fam, text_set, expected):
    assert value(pilot_rows[1], fam, "instruct", "nats/token", text_set, "role-specific")["mean"] == pytest.approx(expected, abs=0.002)


def test_pilot_verdicts(pilot_rows):
    md = pilot_rows[0]
    gem = [l for l in md.splitlines() if l.startswith("NLL arm verdict")]
    assert len(gem) == 5 and gem[-1].endswith("**fires**.") and all(l.endswith("does not fire**.") for l in gem[:-1])
    assert "Spearman" in md and "not computed" in md  # no judge outputs for the pilot
