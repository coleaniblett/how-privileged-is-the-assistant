"""measures.vestigial_regression.analyze_exchanges on the real 10-exchange Qwen3-8B IT smoke output (tests/data/), which has no
base: (1) the same scores as base give zero changes; (2) a base whose user-tagged NLL is 0.1 nats lower gives an asymmetry of
exactly +0.1 nats that fires in both headline corpora, and the matching bits-per-byte value."""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from measures.vestigial_regression import analyze_exchanges as A

SMOKE = Path(__file__).parent / "data" / "exchange_smoke_qwen3-8b_instruct.csv"
TAG = "exchanges_exchange"


def write(root, stage, df):
    d = root / "qwen3-8b" / stage / "core" / TAG
    d.mkdir(parents=True)
    df.assign(stage=stage).to_csv(d / "scores.csv", index=False)


def test_same_scores_as_base_give_zero_changes(tmp_path):
    df = pd.read_csv(SMOKE)
    write(tmp_path, "base", df)
    write(tmp_path, "instruct", df)
    summary, verdicts, levels = A.run(str(tmp_path), TAG, ["qwen3-8b"], 200, 0)
    assert set(summary.corpus) == {"oasst1", "ultrachat", "wildchat"} and set(summary.unit) == {"nats", "bpb"}
    assert np.allclose(summary[["mean", "lo", "hi"]].to_numpy(float), 0)
    assert verdicts.verdict.tolist() == ["does not fire"]
    assert set(levels.stage) == {"base", "instruct"} and levels.truncated_replies.str.contains("/").all()


def test_known_user_frame_cost_change(tmp_path):
    df = pd.read_csv(SMOKE)
    base = df.copy()
    m = (base.scored_text == "user") & (base.framing == "tagged")
    base.loc[m, "mean_nll_from1"] -= 0.1  # base finds the tagged user turn easier: post-training raised the user frame cost by 0.1
    write(tmp_path, "base", base)
    write(tmp_path, "instruct", df)
    summary, verdicts, _ = A.run(str(tmp_path), TAG, ["qwen3-8b"], 200, 0)
    a = summary[(summary.quantity == "asymmetry") & (summary.unit == "nats")].set_index("corpus")
    assert np.allclose(a[["mean", "lo", "hi"]].to_numpy(float), 0.1)
    reply = summary[(summary.quantity == "Δ FC reply")]
    assert np.allclose(reply["mean"], 0)
    assert verdicts.verdict.tolist() == ["fires (both headline corpora)"]
    # bits per byte: per user text 0.1 * (n_tokens - 1) / ln 2 / bytes, averaged over the corpus's user turns
    u = df[m & (df.corpus == "oasst1")]
    expect = (0.1 * (u.n_tokens - 1) / math.log(2) / u.n_bytes_from1).mean()
    got = summary[(summary.quantity == "asymmetry") & (summary.unit == "bpb") & (summary.corpus == "oasst1")]["mean"].item()
    assert abs(got - expect) < 1e-9


def test_cli_writes_report(tmp_path):
    df = pd.read_csv(SMOKE)
    write(tmp_path, "base", df)
    write(tmp_path, "instruct", df)
    A.main(["--root", str(tmp_path), "--out", str(tmp_path / "out"), "--n-boot", "100"])
    md = (tmp_path / "out" / "results.md").read_text()
    assert "Qwen IT" in md and "cannot exclude training-data overlap" in md and "### ultrachat" in md
    assert (tmp_path / "out" / "summary.csv").exists() and (tmp_path / "out" / "verdicts.csv").exists()
