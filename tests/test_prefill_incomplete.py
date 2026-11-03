"""Incomplete-turn prefills (data/prefill_incomplete_200.jsonl): real WildChat user turns cut at a clearly unfinished point."""

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from core import config
from core.prepare_data import INCOMPLETE_ENDINGS, TERMINAL, WORD_TYPE, build_incomplete_prefill, incomplete_cut_points
from core.render import load_tokenizer, render_prefill

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in open(ROOT / path, encoding="utf-8")]


def test_the_file_has_200_distinct_unfinished_turns_cut_from_real_user_texts():
    pre, texts = rows(config.PREFILL_INCOMPLETE_PATH), {t["id"]: t for t in rows(config.DATA_PATH)}
    assert len(pre) == config.N_PREFILL_INCOMPLETE == 200
    assert len({p["text_id"] for p in pre}) == 200 and len({p["prefill"].lower() for p in pre}) == 200
    for p in pre:
        t = texts[p["text_id"]]
        assert (t["source"], t["kind"]) == ("wildchat", "user") and t["text"].startswith(p["prefill"])
        assert p["remaining_chars"] == len(t["text"]) - len(p["prefill"]) > 0  # the real user kept writing
        assert "\n" not in p["prefill"] and p["prefill"] == p["prefill"].strip() and 6 <= p["n_words"] <= 40


def test_every_cut_is_clearly_unfinished_never_at_a_sentence_end():
    for p in rows(config.PREFILL_INCOMPLETE_PATH):
        last = p["prefill"].split()[-1]
        assert not TERMINAL.search(last), p["prefill"][-40:]
        if p["cut_type"] == "comma":
            assert last.endswith(",")
        else:
            assert WORD_TYPE[re.sub(r"^[\W_]+|[\W_]+$", "", last).lower()] == p["cut_type"]


def test_the_cut_types_are_mixed_not_dominated_by_one():
    c = Counter(p["cut_type"] for p in rows(config.PREFILL_INCOMPLETE_PATH))
    assert set(c) == set(INCOMPLETE_ENDINGS) | {"comma"} and max(c.values()) / 200 < 0.4 and min(c.values()) >= 10


def test_cut_points_on_synthetic_texts():
    cuts = incomplete_cut_points("I would like you to write a story about the history of a small town in the north.")
    kinds = {(n, k) for _, n, k in cuts}
    assert all(n >= 6 for _, n, _ in cuts)  # the six-word minimum
    assert (6, "transitive verb") in kinds  # 'I would like you to write' ends in a verb that takes an object
    assert (7, "article/determiner") in kinds  # '... write a' ends in an article
    assert incomplete_cut_points("Short one.") == [] and incomplete_cut_points("A fully finished sentence ends right here.") == []
    # a cut never lands after terminal punctuation or on the last word of the line
    text = "He said yes. And then the whole thing was over and done with."
    for end, n, _ in incomplete_cut_points(text):
        assert not TERMINAL.search(text[:end].split()[-1]) and n < len(text.split())


def test_the_builder_is_seeded_balanced_and_keeps_the_text_id():
    base = [dict(id=f"wildchat-user-{i:04d}", source="wildchat", kind="user",
                 text=f"Please tell me about the history of the city and its people number {i} in the world today") for i in range(40)]
    a, b = build_incomplete_prefill(base, "wildchat", 25, seed=3), build_incomplete_prefill(base, "wildchat", 25, seed=3)
    assert a == b and len(a) == 25 and a != build_incomplete_prefill(base, "wildchat", 25, seed=4)
    assert [x["id"] for x in a][:2] == ["incomplete-000", "incomplete-001"] and a[0]["text_id"] == "wildchat-user-0000"
    dup = [dict(id=f"wildchat-user-{i:04d}", source="wildchat", kind="user", text=base[0]["text"]) for i in range(3)]  # the same prompt three times
    assert len(build_incomplete_prefill(dup, "wildchat", 3)) <= 3 and len({x["prefill"] for x in build_incomplete_prefill(dup, "wildchat", 3)}) == len(build_incomplete_prefill(dup, "wildchat", 3))
    other = [dict(id="oasst1-user-0000", source="oasst1", kind="user", text=base[0]["text"])]
    assert build_incomplete_prefill(other, "wildchat", 5) == []


@pytest.mark.parametrize("tok_name", ["allenai/Olmo-3-7B-Instruct", "Qwen/Qwen3-8B", "google/gemma-4-31B-it"])
def test_every_prefill_renders_as_an_open_user_turn(tok_name):
    tok = load_tokenizer(tok_name)
    for p in rows(config.PREFILL_INCOMPLETE_PATH):
        ids = render_prefill(tok, p["prefill"])
        assert tok.decode(ids, clean_up_tokenization_spaces=False).endswith(p["prefill"])
