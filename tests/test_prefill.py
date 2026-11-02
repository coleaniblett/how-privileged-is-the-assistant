"""The 100-sentence prefill set: a superset of the frozen 20, built from the held-out WildChat user turns."""

import json
from pathlib import Path

from core import config
from core.prepare_data import build_prefill, first_sentence

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in open(ROOT / path, encoding="utf-8")]


def test_prefill_100_is_a_superset_of_the_frozen_20():
    old, new = rows(config.PREFILL_PATH_20), rows(config.PREFILL_PATH)
    assert config.PREFILL_PATH.endswith("prefill_100.jsonl") and len(old) == 20 and len(new) >= config.N_PREFILL_LARGE
    assert new[:20] == old
    assert len({r["prefill"] for r in new}) == len(new) and len({r["text_id"] for r in new}) == len(new)


def test_every_prefill_opens_its_source_wildchat_user_text():
    texts = {t["id"]: t for t in rows(config.DATA_PATH)}
    for r in rows(config.PREFILL_PATH):
        t = texts[r["text_id"]]
        assert (t["source"], t["kind"]) == (config.PAIR_SOURCE, "user") and t["text"].startswith(r["prefill"])
        assert first_sentence(t["text"]) == r["prefill"]


def test_build_prefill_takes_distinct_first_sentences_in_order():
    mk = lambda i, text, source="wildchat", kind="user": dict(id=f"{source}-{kind}-{i:04d}", source=source, kind=kind, text=text)
    data = [mk(0, "This is the first long sentence here. And more."), mk(1, "No sentence end here at all"),
            mk(2, "This is the first long sentence here. Different continuation."), mk(3, "Another opening sentence that works. Yes."),
            mk(4, "An assistant row that is skipped entirely. Ok.", kind="assistant"), mk(5, "Other source skipped by the filter. Ok.", "oasst1")]
    out = build_prefill(data, "wildchat", 5)
    assert [r["text_id"] for r in out] == ["wildchat-user-0000", "wildchat-user-0003"]
    assert [r["id"] for r in out] == ["prefill-00", "prefill-01"]
    assert len(build_prefill(data, "wildchat", 1)) == 1
