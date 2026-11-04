"""extras/exchange_reply_in_user.py: the two conditions score identical tokens (only the label differs), core is untouched, analysis recovers planted changes."""
import numpy as np
import pandas as pd
import pytest

from core import exchanges as ex
from core.render import load_tokenizer
from extras import exchange_reply_in_user as ru


def test_register_adds_a_context_without_touching_the_existing_ones():
    before = dict(ex.CONTEXTS)
    ru.register()
    assert ru.CONTEXT in ex.CONTEXTS and {k: ex.CONTEXTS[k] for k in before} == before
    ru.register()  # idempotent


@pytest.mark.parametrize("family", ["olmo3-7b", "qwen3-8b"])
def test_same_tokens_different_label(family):
    from core import config
    ru.register()
    tok = load_tokenizer(config.family_settings(family)["template"])
    row = next(r for r in ex.load_exchanges(limit=3))
    from extras.exchange_reply_in_user import ex as _ex
    out = ex.render_exchange(tok, row, ru.CONTEXT, family)
    t, p = out[("reply", "tagged")], out[("reply", "plain")]
    assert t.input_ids[t.start:t.end] == p.input_ids[p.start:p.end] and t.end - t.start <= 300
    assert t.start != p.start or t.input_ids[:t.start] != p.input_ids[:p.start]   # different prefix: a user turn tag vs a bare document start
    assert "user" in tok.decode(t.input_ids[:t.start]) and "user" not in tok.decode(p.input_ids[:p.start])


def test_contrast_and_planted_change(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    def scores(delta):
        rows = []
        for corpus in ("oasst1", "ultrachat", "wildchat"):
            for i in range(60):
                base = 2.0 + rng.normal(0, 0.1)
                for framing, extra in (("tagged", delta), ("plain", 0.0)):
                    rows.append(dict(corpus=corpus, exchange_id=f"{corpus}-{i}", scored_text="reply", framing=framing, context=ru.CONTEXT, mean_nll_from1=base + extra))
        return pd.DataFrame(rows)
    c0, c1 = ru.contrast(scores(0.3)), ru.contrast(scores(0.1))
    assert abs(c0.mean() - 0.3) < 0.01 and abs(c1.mean() - 0.1) < 0.01
