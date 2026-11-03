"""CPU checks for extras/icl_era.py: data files, rendering (both families' templates), cached vs full log-probs (SmolLM2-135M)."""

import pytest
import torch

from core import config
from core.render import load_jsonl, load_tokenizer
from extras import icl_era as e

TINY = "HuggingFaceTB/SmolLM2-135M-Instruct"


def test_probe_file():
    probes = load_jsonl(e.PROBES_PATH)
    assert len(probes) == 30 and len({p["id"] for p in probes}) == 30
    assert all(p["answer_19th"].startswith(" ") and p["answer_modern"].startswith(" ") for p in probes)
    assert not any("bird" in (p["question"] + p["prefix"]).lower() for p in probes)


def test_bird_file_and_split():
    birds = load_jsonl(e.BIRDS_PATH)
    assert len({b["archaic"] for b in birds}) == len({b["modern"] for b in birds}) == len(birds)
    pool, control = e.split_birds(birds)
    assert len(control) == e.N_CONTROL and len(pool) >= max(e.KS)
    assert not {b["id"] for b in pool} & {b["id"] for b in control}
    o = e.shot_order(pool, 0)
    assert o == e.shot_order(pool, 0) and o != e.shot_order(pool, 1)


def _toy():
    birds = [dict(id=f"b{i}", archaic=f"Old Hawk {i}", modern=f"New Falcon {i}", description=f"a small brown bird number {i}") for i in range(12)]
    probes = load_jsonl(e.PROBES_PATH)[:3]
    return birds[:8], e.questions(probes, birds[8:10])


@pytest.mark.parametrize("family", ["olmo3-7b", "qwen3-8b"])
@pytest.mark.parametrize("condition", e.CONDITIONS)
def test_render_cell_both_templates(family, condition):
    config.use_family(family)
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    shots, qs = _toy()
    for k in (0, 4):
        block, items = e.render_cell(tok, shots[:k], "archaic", condition, qs, e.template_kwargs_for(family))
        assert len(items) == 2 * len(qs)
        for _, _, _, sfx, (s, t) in items:
            assert 0 < s < t == len(sfx)
    if condition != "plain" and family == "qwen3-8b":
        assert "<think>\n\n</think>" in tok.decode(items[0][3])  # the probe's assistant turn opens with the empty think block
        assert "<think>" not in tok.decode(block)  # history assistant turns do not


def test_cache_matches_full_forward():
    from transformers import AutoModelForCausalLM
    tok = load_tokenizer(TINY)
    model = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32).eval()
    shots, qs = _toy()
    for condition in ("assistant", "plain"):
        block, items = e.render_cell(tok, shots, "archaic", condition, qs, {})
        a = e.score_cell(model, "cpu", block, items, True)
        b = e.score_cell(model, "cpu", block, items, False)
        assert max(abs(a[k] - b[k]) for k in a) < 1e-3
