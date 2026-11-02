"""Assistant-slot opening for families whose generation prompt adds text after the assistant header (Gemma 4)."""

import pytest

from core import config
from core.render import assistant_opening_extra, load_tokenizer, render_all, template_parts

GEMMA = "google/gemma-4-31B-it"
CHANNEL = "<|channel>thought\n<channel|>"


def test_olmo_assistant_opening_is_empty():
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    assert assistant_opening_extra(tok) == ""
    assert template_parts(tok, "assistant") == template_parts(tok, "assistant", opening=False)


def test_gemma_assistant_slot_opens_with_empty_thought_channel():
    tok = load_tokenizer(GEMMA)
    assert assistant_opening_extra(tok) == CHANNEL
    pre, suf = template_parts(tok, "assistant")
    assert pre == "<bos><|turn>model\n" + CHANNEL and suf == "<turn|>\n"
    assert template_parts(tok, "user") == ("<bos><|turn>user\n", "<turn|>\n")  # user slot unchanged


@pytest.mark.parametrize("text", ["Hello there.", "(paren) first\n\nSecond paragraph.", "日本語のテキスト"])
def test_gemma_content_tokens_identical_across_slots_and_one_bos(text):
    tok = load_tokenizer(GEMMA)
    out = render_all(tok, {"id": "t", "text": text, "kind": "user", "source": "x"})
    spans = {c: v.input_ids[v.start : v.end] for c, v in out.items()}
    assert len({tuple(s) for s in spans.values()}) == 1
    for v in out.values():
        assert v.input_ids[0] == tok.bos_token_id and v.input_ids.count(tok.bos_token_id) == 1
