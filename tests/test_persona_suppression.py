"""CPU tests for extras/persona_suppression.py (spec F-main): frozen materials, prompt construction, shared-cache scoring, analysis maths."""
import json

import numpy as np
import pandas as pd
import pytest

from core import config
from extras import persona_map, persona_suppression as ps

MAT = ps.load_materials()
PEOPLE = {p["id"]: p for p in json.load(open(ps.PERSONAS, encoding="utf-8"))["personas"]}


def test_materials_design():
    m = MAT["personas"]
    assert len(m) == 78 and len({p["id"] for p in m}) == 78
    tr, ho = [p for p in m if p["group"] == "train"], [p for p in m if p["group"] == "holdout"]
    assert len(tr) == 60 and len(ho) == 18
    assert all(not PEOPLE[p["id"]]["holdout"] for p in tr) and all(PEOPLE[p["id"]]["holdout"] for p in ho)
    assert {p["entity"] for p in m} == set(ps.ENTITIES)
    assert sum(p["slot"] == "system" for p in tr) == 24
    assert len(MAT["probes"]) == 16 and sum(q["type"] == "factual" for q in MAT["probes"]) == 8


def test_swap_partners_valid():
    by = {p["id"]: p for p in MAT["personas"]}
    for p in MAT["personas"]:
        q = by[p["partner"]]
        assert q["entity"] != p["entity"] and q["slot"] == p["slot"] and q["group"] == p["group"]
    from collections import Counter
    assert max(Counter(p["partner"] for p in MAT["personas"]).values()) <= 2


def test_categories_frozen_shape():
    assert list(MAT["categories"]) == ps.CAT_ORDER == ["answer", "assistant", "list", "voice"]
    assert all(len(v) == 6 for v in MAT["categories"].values())
    assert MAT["categories"] == ps.CATEGORIES


def test_neutral_texts_name_no_character():
    for fam, d in MAT["neutral"].items():
        assert set(d) == {p["id"] for p in MAT["personas"]}
        for pid, t in d.items():
            assert t.startswith(ps.NEUTRAL_BASE)
            assert PEOPLE[pid]["name"] not in t and PEOPLE[pid]["label"] not in t


def test_messages_match_persona_map():
    for pid in ("fiction-03", "ai-05", "human-10"):
        p = PEOPLE[pid]
        assert ps.persona_text(p) == persona_map.persona_text(p)
        assert ps.build_messages(p["slot"], ps.persona_text(p), "Q?") == persona_map.messages(p, "Q?")
    assert ps.build_messages("none", None, "Q?") == persona_map.messages(None, "Q?")


def test_prompt_items_structure():
    items = ps.prompt_items(MAT, "olmo3-7b", PEOPLE)
    assert len(items) == 78 * 16 * 3 + 16
    # same chunk positions across versions: version blocks have identical (persona, probe) order
    blk = 78 * 16
    order = [(i["persona"], i["probe"]) for i in items[:blk]]
    assert order == [(i["persona"], i["probe"]) for i in items[blk:2 * blk]] == [(i["persona"], i["probe"]) for i in items[2 * blk:3 * blk]]
    assert blk % 16 == 0


def test_qwen_prompt_has_empty_think_block():
    from core.render import load_tokenizer
    tok = load_tokenizer(config.family_settings("qwen3-8b")["template"])
    s = ps.render_prompt(tok, ps.build_messages("none", None, "Hi"), "qwen3-8b")
    assert s.endswith("<think>\n\n</think>\n\n")


def test_layout_regex():
    assert ps.layout("Here are reasons:\n\n1. one") and ps.layout("**Key** facts") and ps.layout("text\n- item")
    assert not ps.layout("Well, now, the sky is blue because of scattering.")


def test_shared_cache_scoring_matches_full_pass():
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    try:
        tok, model = AutoTokenizer.from_pretrained(name), AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    except Exception as e:  # offline without the cached tiny model
        pytest.skip(str(e))
    pids = tok("hello there, how are you today?", add_special_tokens=False)["input_ids"]
    conts = [tok(s, add_special_tokens=False)["input_ids"] for s in ("Well, now,", "Ah,", "Here are the main reasons:\n\n1.")]
    got = ps.cont_logprobs(model, "cpu", pids, conts)
    for c, g in zip(conts, got):
        lg = model(input_ids=torch.tensor([pids + c])).logits[0].float().log_softmax(-1)
        ref = sum(lg[len(pids) + j - 1, c[j]].item() for j in range(len(c)))
        assert abs(ref - g) < 1e-3


def synth_pp(frac, rng):
    """Persona-level table with intact < neutral < assistant on `logodds` such that neutral moves `frac` of the way."""
    rows = {}
    ids = [f"p{i}" for i in range(30)]
    for i in ids:
        a = rng.normal(0, 0.1)
        rows[(i, "intact")] = dict(logodds=a, **{c: 0.0 for c in ps.CAT_ORDER})
        rows[(i, "neutral")] = dict(logodds=a + frac * 2.0 + rng.normal(0, 0.1), **{c: 0.0 for c in ps.CAT_ORDER})
    pp = pd.DataFrame(rows).T
    pp.index = pd.MultiIndex.from_tuples(pp.index, names=["persona", "version"])
    return pp, ids


def test_fallback_fraction_recovers_planted_value():
    pp, ids = synth_pp(0.6, np.random.default_rng(1))
    f = ps.fallback(pp, ids, "logodds", np.random.default_rng(0), asst=2.0)
    assert abs(f["frac"] - 0.6) < 0.1 and f["frac_ci"][0] < 0.6 < f["frac_ci"][1]
    assert f["gap_ci"][0] > 1.5 and f["num_ci"][0] > 0
    f0 = ps.fallback(synth_pp(0.0, np.random.default_rng(2))[0], ids, "logodds", np.random.default_rng(0), asst=2.0)
    assert f0["num_ci"][0] < 0 < f0["num_ci"][1]


def test_validation_positive_when_swap_follows_partner():
    rng = np.random.default_rng(3)
    ids = [f"p{i}" for i in range(24)]
    meta = [dict(id=i, partner=ids[(k + 1) % 24]) for k, i in enumerate(ids)]
    base = {i: rng.normal(0, 1, 4) for i in ids}
    rows = {}
    for m in meta:
        rows[(m["id"], "intact")] = dict(zip(ps.CAT_ORDER, base[m["id"]]))
        rows[(m["id"], "swap")] = dict(zip(ps.CAT_ORDER, base[m["partner"]] + rng.normal(0, 0.2, 4)))
    pp = pd.DataFrame(rows).T
    pp.index = pd.MultiIndex.from_tuples(pp.index)
    r, c = ps.validation(pp, meta, np.random.default_rng(0))
    assert r > 0.7 and c[0] > 0


def test_readouts_logodds():
    cat = pd.DataFrame([dict(persona="p", probe="q", version="intact", category=c, logprob=v) for c, v in zip(ps.CAT_ORDER, (-5.0, -4.0, -6.0, -3.0))])
    w = ps.readouts(cat, None, None)
    assert abs(w["logodds"].iloc[0] - ((-4.0 - 6.0) / 2 + 3.0)) < 1e-9
