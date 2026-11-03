"""CPU checks for extras/icl_era.py: data files, rendering (both families' templates), cached vs full log-probs (SmolLM2-135M)."""

import pytest
import torch

from core import config
from core.render import load_jsonl, load_tokenizer
from extras import icl_era as e

TINY = "HuggingFaceTB/SmolLM2-135M-Instruct"


CANDIDATES = "data/icl_era_probe_candidates.jsonl"


@pytest.mark.parametrize("path", [e.PROBES_PATH, CANDIDATES])
def test_probe_file(path):
    probes = load_jsonl(path)
    assert len({p["id"] for p in probes}) == len(probes) >= 10
    assert all(p["answer_19th"].startswith(" ") and p["answer_modern"].startswith(" ") for p in probes)
    assert not any("bird" in (p["question"] + p["prefix"]).lower() for p in probes)


@pytest.mark.parametrize("family", ["olmo3-7b", "qwen3-8b"])
def test_candidates_token_matched(family):
    """Review fix (2026-11-03): no year probe; both continuations have the same token count and do not merge with the prefix."""
    tok = load_tokenizer(config.family_settings(family)["template"])
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    probes = load_jsonl(CANDIDATES)
    assert "year" not in {p["id"] for p in probes}
    for p in probes:
        a, b = enc(p["answer_19th"]), enc(p["answer_modern"])
        assert len(a) == len(b), p["id"]
        assert enc(p["prefix"] + p["answer_19th"]) == enc(p["prefix"]) + a and enc(p["prefix"] + p["answer_modern"]) == enc(p["prefix"]) + b


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


def test_analyze_on_synthetic_scores(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    monkeypatch.setattr(e, "OUT", str(tmp_path))
    rng = np.random.default_rng(0)
    for fam, stages in (("olmo3-7b", ["base", "instruct"]), ("qwen3-8b", ["base", "instruct"])):
        (tmp_path / fam).mkdir()
        for st in stages:
            rows = []
            for cond in e.CONDITIONS:
                for k in e.KS:
                    for seed in (e.SEEDS if k else [0]):
                        for names in (e.NAMES if k else ["none"]):
                            for qid, kind in [(f"p{i}", "probe") for i in range(5)] + [(f"c{i}", "control") for i in range(3)]:
                                bump = (1.0 if names == "archaic" else 0.0) * k / 128
                                rows.append(dict(family=fam, stage=st, condition=cond, names=names, k=k, seed=seed, qid=qid, kind=kind,
                                                 logp_past=-3 + bump + rng.normal(0, 0.1), logp_modern=-3.0, block_tokens=10))
            pd.DataFrame(rows).to_csv(tmp_path / fam / f"{st}.csv", index=False)
    t = e.analyze()
    s = t[(t.family == "olmo3-7b") & (t.stage == "instruct") & (t.condition == "assistant") & (t.k == 128) & (t.quantity == "shift")
          & (t.kind == "probe")].iloc[0]
    assert abs(s["mean"] - 1.0) < 0.1
    md = (tmp_path / "icl_era.md").read_text()
    assert "Per-probe shift" in md and "Name pick-up check" in md
