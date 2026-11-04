"""CPU tests for extras/mechanism_checks.py (spec H): frozen materials, H2 conditions render as designed (default sentence vs explicit empty vs neutral), H3 openers, detectors, bootstrap."""
import numpy as np
import pytest

from core import config
from core.render import encode, load_tokenizer
from extras import mechanism_checks as mc

MAT = mc.load_materials()


def render(family, msgs):
    from extras.persona_leakage import chat
    return chat(load_tokenizer(config.family_settings(family)["template"]), msgs)


def test_frozen_materials():
    assert len(MAT["requests"]) == 36 and len(MAT["roles"]["helpful"]) == 3 and len(MAT["roles"]["stylized"]) == 3
    assert set(MAT["h3"]["voice"]) == {n for rs in MAT["roles"].values() for n, _ in rs}
    for family in ("olmo3-7b", "qwen3-8b"):
        tok = load_tokenizer(config.family_settings(family)["template"])
        a, b = len(encode(tok, MAT["default_sentence"])), len(encode(tok, MAT["neutral_sentence"][family]))
        assert abs(a - b) <= 2
    compliance_words = ("sure", "certainly", "of course", "happy to", "glad to", "sorry", "cannot", "can't")
    assert not any(w in v.lower() for v in MAT["h3"]["voice"].values() for w in compliance_words)


def test_olmo_empty_system_omits_the_default_sentence():
    role = tuple(MAT["roles"]["helpful"][0])
    req = MAT["requests"][0]
    r = {c: render("olmo3-7b", mc.h2_messages(MAT, "olmo3-7b", c, req, role)) for c in ("S-none", "S-default", "S-neutral", "U-default", "U-empty", "U-neutral")}
    d = MAT["default_sentence"]
    assert d not in r["S-none"] and d in r["S-default"] and d not in r["S-neutral"] and MAT["neutral_sentence"]["olmo3-7b"] in r["S-neutral"]
    assert d in r["U-default"] and d not in r["U-empty"] and d not in r["U-neutral"]
    assert "<|im_start|>system\n<|im_end|>" in r["U-empty"]                      # explicitly empty system block
    assert role[0] in r["U-empty"].split("<|im_start|>user")[1]                  # the character is in the user turn
    assert role[0] in r["S-default"].split("<|im_start|>user")[0]                # and in the system message here
    a = {c: render("olmo3-7b", mc.h2_messages(MAT, "olmo3-7b", c, req, None)) for c in ("A-default", "A-empty", "A-neutral")}
    assert d in a["A-default"] and d not in a["A-empty"] and "<|im_start|>system\n<|im_end|>" in a["A-empty"]


def test_qwen_transplant_conditions():
    role, req = tuple(MAT["roles"]["stylized"][0]), MAT["requests"][3]
    r = {c: render("qwen3-8b", mc.h2_messages(MAT, "qwen3-8b", c, req, role)) for c in mc.H2_PLAN["qwen3-8b"]}
    assert "<|im_start|>system" not in r["U-none"] and MAT["default_sentence"] in r["S-default"] and MAT["default_sentence"] in r["U-default"]
    assert r["S-none"].endswith("<think>\n\n</think>\n\n")


def test_h2_item_counts():
    n_roles = 6
    assert len(mc.h2_items(MAT, "olmo3-7b")) == 36 * (6 * n_roles + 3)
    assert len(mc.h2_items(MAT, "qwen3-8b")) == 36 * 4 * n_roles


def test_h3_items_and_openers():
    items = mc.h3_items(MAT)
    assert len(items) == 36 * (3 + 6 * 2 * 4)
    voices = {i["role"]: i["text"] for i in items if i["opener"] == "voice"}
    assert voices == MAT["h3"]["voice"]
    assert all(i["text"] == "" for i in items if i["opener"] == "none") and all(i["text"] == "Okay," for i in items if i["opener"] == "neutral")
    assert not any(i["opener"] == "voice" for i in items if i["setup"] == "plain")


def test_refusal_detectors():
    assert mc.refused("I'm sorry, but I can't help with that request.")
    assert not mc.refused("Here is how you pick a pin tumbler lock: first, ...")
    assert mc.refused(" but I can't tell you how to do that.", "apology")
    assert not mc.refused(" but let me help: first insert the tension wrench.", "apology")


def test_boot2_covers_planted_mean():
    rng = np.random.default_rng(0)
    M = 0.3 + rng.normal(0, 0.1, (36, 6))
    m, lo, hi = mc.boot2(M, np.random.default_rng(1), n=1500)
    assert lo < 0.3 < hi
