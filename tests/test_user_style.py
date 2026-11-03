"""CPU tests for measures/overgeneralization/user_style.py (synthetic data; the classifier is a stub)."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("wordfreq")
nltk_words = pytest.importorskip("nltk.corpus").words
try:
    nltk_words.words()
except LookupError:
    pytest.skip("nltk words corpus missing", allow_module_level=True)

from measures.overgeneralization import user_style as us


def test_first_sentence():
    assert us.first_sentence("it happened. Then more text") == "it happened."
    assert us.first_sentence("no end here\nsecond line") == "no end here"
    assert us.first_sentence("what is it? Dunno") == "what is it?"
    assert us.first_sentence("version 3.5 is out") == "version 3.5 is out"        # a period inside a number is not a sentence end
    assert us.first_sentence("just words") == "just words"


def test_features():
    f = us.features("i think this is teh best. Please help!\n- a\n- b")
    assert f["exclaim"] == 1 and f["polite"] == 1 and f["markdown"] == 1 and f["cap_i"] == 0.0 and f["typo_rate"] > 0
    g = us.features("I think it is a good idea. Thank you.")
    assert g["cap_i"] == 1.0 and g["typo_rate"] == 0 and g["terminal"] == 1.0 and g["polite"] == 1.0
    assert np.isnan(us.features("")["sent_len"]) and np.isnan(us.features("no first person here")["cap_i"])


def test_typo_ignores_urls_numbers_code():
    assert us.typo_rate("see https://xyzzyqq.example/abc and `qqzzv()` 12345") != us.typo_rate("see xyzzyqq and qqzzv")


def test_matches():
    m = us.matches("write a poem about the sea", "and make it rhyme")
    assert m["match_lower"] == 1.0 and m["match_no_terminal"] == 1.0
    m2 = us.matches("Write a poem about the sea.", "and make it rhyme")
    assert m2["match_lower"] == 0.0 and m2["match_no_terminal"] == 0.0
    assert np.isnan(us.matches("x y z", "   ")["match_typo"])


def test_human_remainder_and_matched_rows():
    texts = {"t0": "Hello there my friend. Please write me a story about dogs and cats today."}
    pf = {"p0": dict(id="p0", text_id="t0", prefill="Hello there my friend.")}
    samples = [dict(prefill_id="p0", sample=0, prefill="Hello there my friend.", continuation=" Write a poem<|im_end|>assistant: ok")]
    df = us.build_rows(samples, ["<|im_end|>"], pf, texts)
    assert df.full[0] == " Write a poem" and df.n_words[0] == 3
    assert df.human_full[0] == "Please write me"          # first 3 words of the real remainder, length-matched
    assert df.human_sent[0].startswith("Please write me a story")


class Stub:
    def p_assistant(self, texts):
        return np.array([0.9 if "Certainly" in t else 0.1 for t in texts])


def test_effect_and_polish_leak_rule():
    n = 30
    def stage(text):
        rows = [dict(prefill_id=f"p{i}", sample=s, prefill="q", continuation=text) for i in range(n) for s in range(3)]
        pf = {f"p{i}": dict(id=f"p{i}", text_id=f"t{i}", prefill="q") for i in range(n)}
        return rows, pf
    texts = {f"t{i}": "q a normal human sentence goes right here. more" for i in range(n)}
    base_rows, pf = stage(" fine words are here now.")
    inst_rows, _ = stage(" Certainly I can do that now.")
    per = {}
    for name, rows in (("base", base_rows), ("instruct", inst_rows)):
        d = us.build_rows(rows, [], pf, texts)
        per[name] = us.add_scores(d, Stub(), ["sent", "full", "human_sent", "human_full"])
    sent = {s: us.per_prefill_mean(d, "sent_p") for s, d in per.items()}
    h = per["base"].groupby("prefill_id")["human_sent_p"].first()
    rows = us.effect_rows(sent, h, n_boot=200, label="P(assistant), completion of cut sentence")
    t = pd.DataFrame(rows)
    assert t[(t.stage == "instruct") & (t.versus == "base")].iloc[0]["mean"] == pytest.approx(0.8)
    assert us.polish_leak(rows, "olmo3-7b") is True
    assert us.polish_leak(rows, "qwen3-8b") is None
    flat = us.effect_rows({"base": sent["base"], "instruct": sent["base"]}, h, n_boot=200, label="P(assistant), completion of cut sentence")
    assert us.polish_leak(flat, "olmo3-7b") is False
