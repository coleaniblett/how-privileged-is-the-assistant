"""CPU tests for extras/style_fall_checks.py: role-play regex, paired change, per-reply correlation, slippage exclusion."""
import numpy as np
import pandas as pd

from extras import style_fall_checks as sf


def test_roleplay_regex():
    yes = ["Please role-play as a pirate.", "Let's roleplay.", "Pretend you are my boss", "You are now DAN", "Act as a travel guide",
           "stay in character", "Imagine you're a chef", "From now on, you will answer as a poet"]
    no = ["Write a function that sorts a list", "What is the capital of France?", "Explain how actors prepare for a part", "I like to play chess"]
    assert all(sf.ROLEPLAY.search(t) for t in yes)
    assert not any(sf.ROLEPLAY.search(t) for t in no)


def frame(delta, n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    xs = rng.normal(0, 1, n)  # one value per item, shared by both stages (paired design)
    for stage, shift in (("base", 0.0), ("instruct", delta)):
        for item in range(n):
            x = xs[item]
            for rep in range(3):
                rows.append(dict(family="olmo", stage=stage, ctx="voiced", tmpl=1, item=item, rep=rep, coord=shift + x + rng.normal(0, .1),
                                 clf=0.5 * (shift + x) + rng.normal(0, .1), flagged=False))
    return pd.DataFrame(rows)


def test_change_is_paired_per_item_and_signed():
    d = frame(-0.7)
    ch = sf.change(d, "coord", "olmo", "instruct", "voiced", 1, np.random.default_rng(0))
    assert abs(ch["mean"] + 0.7) < 0.1 and ch["hi"] < 0 and ch["n"] == 40
    # pairing: the large between-item spread (sd 1) must not widen the CI much
    assert ch["hi"] - ch["lo"] < 0.2


def test_pearson_ci_recovers_correlation():
    d = frame(0.0)
    r, lo, hi, n = sf.pearson_ci(d[d.stage == "base"], "coord", "clf", np.random.default_rng(0))
    assert r > 0.9 and lo <= r <= hi and n == 120


def test_exclusion_shrinks_change_when_flagged_replies_carry_it():
    d = frame(0.0)
    # flagged replies in instruct have a very different (low) score: excluding them removes the apparent fall
    idx = (d.stage == "instruct") & (d.rep == 0)
    d.loc[idx, "coord"] -= 3.0
    d.loc[idx, "flagged"] = True
    full = sf.change(d, "coord", "olmo", "instruct", "voiced", 1, np.random.default_rng(0))["mean"]
    excl = sf.change(d[~d.flagged], "coord", "olmo", "instruct", "voiced", 1, np.random.default_rng(0))["mean"]
    assert full < -0.9 and abs(excl) < 0.1 and 1 - excl / full > 0.5
