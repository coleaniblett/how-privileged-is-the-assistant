"""CPU tests for extras/trait_atlas.py (spec G): frozen materials, frames, detectors, analysis readings."""
import collections
import re

import numpy as np
import pytest

from core import config
from extras import trait_atlas as ta

MAT = ta.load_materials()
PERSONA_BANNED = re.compile(r"\b(formal|casual|polite|helpful|bullet|markdown|list|brief|concise|detailed|refus\w*|advice|opinion\w*|feel\w*|emotion\w*|honest\w*|cautious|careful|flatter\w*|"
                            r"praise|safe|safety|moral\w*|ethic\w*|apolog\w*|assist\w*|human|ai|robot|machine|certainly|sure|capable)\b", re.I)


def test_traits_and_items():
    assert [t["id"] for t in MAT["traits"]] and len(MAT["traits"]) == 20
    assert {t["cluster"] for t in MAT["traits"]} == set(ta.CLUSTERS)
    for t in MAT["traits"]:
        assert len(t["items"]) == 12 and len({x["id"] for x in t["items"]}) == 12
        for x in t["items"]:
            assert 5 <= len(x["question"].split()) <= 70
            assert not re.search(r"\b(AI|assistant|chatbot|language model|robot|machine)\b", x["question"], re.I), x["id"]
    assert sum(t["user_turn"] for t in MAT["traits"]) == len(ta.USER_TURN_TRAITS)


def test_skill_items_have_checkable_answers():
    for t in MAT["traits"]:
        if t["cluster"] == "skill":
            for x in t["items"]:
                assert x["answer"] and x["accept"]
                assert ta.skill_correct(f"The answer is {x['answer']}.", x["accept"]), x["id"]
                assert not ta.skill_correct("I am not sure.", x["accept"])


def test_personas_do_not_mention_measured_traits():
    assert len(MAT["light"]) == 6 and len(MAT["strong"]) == 6
    for text in MAT["light"] + [p["sheet"] + p["role"] for p in MAT["strong"]]:
        assert not PERSONA_BANNED.search(text), (text, PERSONA_BANNED.search(text).group(0))
    assert len(MAT["act_sentence"].split()) == len(MAT["neutral_sentence"].split())


def test_requests_balanced_and_deterministic():
    reqs = ta.plan_requests(MAT)
    assert reqs == ta.plan_requests(MAT)
    c = collections.Counter(r["context"] for r in reqs)
    assert c["assistant"] == 240 * 6 and c["strong_act"] == c["strong_noact"] == c["light_system"] == c["light_user"] == 240 * 6
    assert c["narration"] == c["plain_text"] == 240 * 3 and c["user_turn"] == 6 * 12 * 3
    # each persona once per item in contexts 2-5; both wordings used equally
    for ctx in ("light_system", "strong_act"):
        sub = [r for r in reqs if r["context"] == ctx and r["item"] == "length-01"]
        assert sorted(r["persona"] for r in sub) == list(range(6)) and sum(r["wording"] for r in sub) == 3
    # contexts 4 and 5 differ only in the padding sentence of the same length
    a = [r for r in reqs if r["context"] == "strong_act"][0]
    assert MAT["frames"]["strong_act"] == MAT["frames"]["strong_noact"]


@pytest.mark.parametrize("family", ["olmo3-7b", "qwen3-8b"])
def test_rendering(family):
    from core.render import load_tokenizer
    tok = load_tokenizer(config.family_settings(family)["template"])
    reqs = [r for r in ta.plan_requests(MAT) if r["trait"] == "flattery"]
    for ctx in ta.CONTEXTS:
        r = next(x for x in reqs if x["context"] == ctx)
        kind, text = ta.render_request(MAT, r, tok, family)
        assert r["question"] in text
        if ctx == "plain_text":
            assert kind == "text" and "<|im_start|>" not in text and text.rstrip().endswith(("A:", "Answer:"))
        elif ctx == "user_turn":
            assert kind == "user_open" and text.rstrip().endswith("user")
        else:
            assert kind == "chat" and text.rstrip().endswith("assistant") or family == "qwen3-8b"
    if family == "qwen3-8b":
        kind, text = ta.render_request(MAT, next(x for x in reqs if x["context"] == "assistant"), tok, family)
        assert text.endswith("<think>\n\n</think>\n\n")


def test_detectors():
    assert ta.detect_one("length", "words", "one two three four", {}) == pytest.approx(np.log(4))
    assert ta.detect_one("markdown_lists", "layout", "Here:\n\n1. a\n2. b", {}) == 1.0
    assert ta.detect_one("markdown_lists", "layout", "Plain prose only.", {}) == 0.0
    assert ta.detect_one("stock_phrases", "stock", "Certainly! Here is it.", {}) == 1.0
    assert ta.detect_one("says_ai", "self_ai", "I'm an AI, so no.", {}) == 1.0
    assert ta.detect_one("says_ai", "self_ai", "Born in Leeds.", {}) == 0.0
    assert ta.detect_one("refusal_borderline", "refusal", "I'm sorry, but I can't help with that.", {}) == 1.0
    assert ta.detect_one("arithmetic", "skill", "47*38 = 1,786", {"accept": [r"\b1,?786\b"]}) == 1.0


def test_cut_and_flag():
    cut, fl = ta.cut_reply("Nice day.<|im_end|>\n<|im_start|>assistant\nHi", ["<|im_end|>", "<|im_start|>"], "assistant")
    assert cut == "Nice day." and fl
    cut, fl = ta.cut_reply("It is 5.\nQ: next one\nA: 6", ["<|im_end|>"], "plain_text")
    assert cut == "It is 5." and not fl
    cut, fl = ta.cut_reply("A plain answer.", ["<|im_end|>"], "assistant")
    assert cut == "A plain answer." and not fl


def test_judge_prompts_frozen_for_every_judge_trait():
    judged = [t["id"] for t in MAT["traits"] if t["detector"] == "judge"]
    assert sorted(judged) == sorted(MAT["judge"])
    for t in judged:
        fld = MAT["judge"][t]["note_field"]
        if fld:
            assert all(fld in x for x in next(y for y in MAT["traits"] if y["id"] == t)["items"])


def test_readings():
    assert ta.reading(7, 3, 11, 20, "pts") == "spreads"
    assert ta.reading(7, -1, 11, 20, "pts") == "stays"          # CI includes 0
    assert ta.reading(3, 1, 5, 20, "pts") == "stays"            # under 5 points
    assert ta.reading(-8, -12, -4, 20, "pts") == "reverses"
    assert ta.reading(6, 2, 9, 40, "pts") == "stays"            # ratio 0.15 < 0.25
    assert ta.gate(12, 5, 19, "pts") and not ta.gate(8, 3, 13, "pts") and ta.gate(0.25, 0.1, 0.4, "log")
    assert ta.style_contrast(12, 2, -3, 6, "pts") == "better acting"
    assert ta.style_contrast(12, 9, 4, 14, "pts") == "sharper role boundaries"
    assert ta.style_contrast(3, 1, -2, 3, "pts") == "unresolved"


def test_two_way_boot_covers_planted_mean():
    rng = np.random.default_rng(0)
    M = 10 + rng.normal(0, 3, (12, 6))
    m, lo, hi = ta.two_way_boot(M, np.random.default_rng(1), n_boot=2000)
    assert lo < 10 < hi and lo < m < hi
