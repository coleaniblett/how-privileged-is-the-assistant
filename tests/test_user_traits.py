"""CPU tests for measures/overgeneralization/user_traits.py with fake judge rows (no models, no API)."""
import json

import numpy as np
import pandas as pd
import pytest

from measures.overgeneralization import user_traits as u


def rows(spec):
    """spec: {prefill_id: [labels of its 3 samples]} -> judge DataFrame."""
    return pd.DataFrame([dict(prefill_id=p, sample=i, label=l) for p, labels in spec.items() for i, l in enumerate(labels)])


UV, UVAT, SW, DG = u.UV, u.UVAT, "switched_to_assistant", "degenerate"


def test_trait_shares_primary_and_secondary():
    prim, sec = u.trait_shares(rows({"a": [UV, UVAT, SW], "b": [SW, DG, SW], "c": [UVAT, UVAT, UV]}))
    assert prim["a"] == pytest.approx(0.5) and prim["c"] == pytest.approx(2 / 3)
    assert np.isnan(prim["b"])                       # no user-voice sample: undefined
    assert sec["a"] == pytest.approx(1 / 3) and sec["b"] == 0.0


def test_paired_change_drops_and_counts():
    base = rows({f"p{i}": [UV, UV, UV] for i in range(10)} | {"x": [SW, SW, SW]})
    inst = rows({f"p{i}": [UV, UVAT, UVAT] for i in range(10)} | {"x": [UV, UV, UV]})
    t = u.judge_table({"base": base, "instruct": inst}, n_boot=500)
    p = t[(t.stage == "instruct") & (t.arm == "primary")].iloc[0]
    assert p.n == 10 and p.dropped == 1              # prefill x has no user-voice sample in base
    assert p["mean"] == pytest.approx(2 / 3) and p.lo > 0
    s = t[(t.stage == "instruct") & (t.arm == "secondary")].iloc[0]
    assert s.n == 11 and s.dropped == 0


def test_fires_rule():
    base = rows({f"p{i}": [UV, UV, UV] for i in range(40)})
    up = rows({f"p{i}": [UV, UV, UVAT] if i % 2 else [UV, UV, UV] for i in range(40)})   # +16.7 pp
    flat = rows({f"p{i}": [UV, UV, UV] for i in range(40)})
    assert u.fires(u.judge_table({"base": base, "instruct": up}, n_boot=500), "olmo3-7b") is True
    assert u.fires(u.judge_table({"base": base, "instruct": flat}, n_boot=500), "olmo3-7b") is False
    assert u.fires(u.judge_table({"base": base, "instruct": up}, n_boot=500), "qwen3-8b") is None   # supporting only
    assert u.fires(u.judge_table({"base": base, "sft": up}, n_boot=500), "olmo3-7b") is None        # no instruct stage


def test_bar_is_inclusive_and_needs_ci():
    base = rows({f"p{i}": [UV] for i in range(4)})
    noisy = rows({"p0": [UVAT], "p1": [UV], "p2": [UV], "p3": [UV]})   # +6.25 pp mean but CI includes zero
    t = u.judge_table({"base": base, "instruct": noisy}, n_boot=500)
    assert t[t.arm == "primary"].iloc[0]["mean"] >= u.FIRE_BAR and u.fires(t, "olmo3-7b") is False


def test_load_judge_validates(tmp_path):
    d = tmp_path / "judge"; d.mkdir()
    (d / "base.jsonl").write_text(json.dumps(dict(prefill_id="a", sample=0, label=UV, reason="x")) + "\n")
    assert list(u.load_judge(str(d), ["base", "sft"])) == ["base"]
    (d / "sft.jsonl").write_text(json.dumps(dict(prefill_id="a", sample=0, label="bogus")) + "\n")
    with pytest.raises(SystemExit):
        u.load_judge(str(d), ["sft"])


def test_reports_render():
    base = rows({f"p{i}": [UV, UV, UV] for i in range(10)}); inst = rows({f"p{i}": [UVAT, UV, UV] for i in range(10)})
    t = u.judge_table({"base": base, "instruct": inst}, n_boot=200)
    assert "fires" in u.judge_report(t, "olmo3-7b") and "supporting only" in u.judge_report(t, "qwen3-8b")


def test_keyword_flags_and_cut():
    m = ["<|im_end|>", "<|im_start|>"]
    assert u.cut_user_portion("hello<|im_end|>\n<|im_start|>assistant\nCertainly!", m) == "hello"
    assert u.keyword_flags("Certainly! I hope this helps.", "write a poem")["stock_phrase"]
    assert not u.keyword_flags("I surely do want a poem", "x")["stock_phrase"]
    f = u.keyword_flags("# Title\n- one\n- two\n**bold** text", "plain prompt")
    assert f["markdown_heading"] and f["markdown_bullet"] and f["markdown_bold"]
    g = u.keyword_flags("- one\n- two", "- my list\n- is here")                # prefill already has bullets: not counted
    assert not g["markdown_bullet"]


def test_keyword_table_paired():
    def samples(txt):
        return [dict(prefill_id=f"p{i}", sample=s, prefill="q", continuation=txt) for i in range(10) for s in range(3)]
    m = ["<|im_end|>"]
    flagged = {"base": u.keyword_samples(samples("plain words"), m), "instruct": u.keyword_samples(samples("Certainly! sure thing"), m)}
    t = u.keyword_table(flagged, n_boot=200)
    r = t[(t.feature == "stock_phrase") & (t.stage == "instruct")].iloc[0]
    assert r["mean"] == pytest.approx(1.0) and r.n == 10
    assert t[(t.feature == "markdown_bold")].iloc[0]["mean"] == 0
