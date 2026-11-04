"""CPU checks for the spec B (user-turn minimal pairs) and spec C (mirror the Assistant) helpers; no model weights."""

import pandas as pd
import pytest

from measures.overgeneralization import user_minimal_pairs as b
from measures.overgeneralization import user_mirror as c


def test_endings_fixed():
    assert len(b.ASSISTANT_ENDINGS) == len(b.HUMAN_ENDINGS) == 10
    assert all(e.startswith(" ") and e == e.rstrip() for _, e in b.ENDINGS)


def test_per_text_pref_and_contrasts():
    rows = []
    for i, (style, _) in enumerate(b.ENDINGS):
        for p, v in (("user", 2.0), ("plain", 3.0), ("asst", 1.0)):
            nll = v if style == "assistant" else v + {"user": 0.5, "plain": 0.2, "asst": 1.0}[p]
            rows.append(dict(text_id="t", ending=i, style=style, placement=p, n_tokens=5, mean_nll=nll))
    pt = b.per_text(pd.DataFrame(rows)).loc["t"]
    assert pt.pref_user == pytest.approx(0.5) and pt.pref_plain == pytest.approx(0.2) and pt.pref_asst == pytest.approx(1.0)
    assert pt.contrast_user == pytest.approx(0.3) and pt.contrast_asst == pytest.approx(0.8)


def test_p2_filter():
    assert c.p2_ok("could you please rewrite this paragraph so it sounds more formal")
    assert not c.p2_ok("too short to keep")
    assert not c.p2_ok("https://a.b/c 123 456 789 {x} [y] (z) 0x1 ok")


def test_copy_share():
    assert c.copy_share("Let me know", "let   me know if you need") == 1.0
    assert c.copy_share("abc", "abc") is None
    assert c.copy_share("zzzz", "abcd") == 0.0


def test_mirror_pairs_and_drops():
    df = pd.DataFrame([dict(conv_id="a", condition="with", sample=0, m=0.5), dict(conv_id="a", condition="without", sample=0, m=0.2),
                       dict(conv_id="a", condition="without", sample=1, m=0.4), dict(conv_id="b", condition="with", sample=0, m=0.9),
                       dict(conv_id="b", condition="without", sample=0, m=None)])
    eff = c.mirror(df, "m")
    assert list(eff.index) == ["a"] and eff["a"] == pytest.approx(0.2)
