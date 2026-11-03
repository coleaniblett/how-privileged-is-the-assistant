"""CPU tests for extras/leakage_matrix.py (no GPU use; the smoke run hides the GPU so it cannot disturb a running job)."""
import os
import subprocess
import sys

import numpy as np
import pytest

from extras import leakage_matrix as lm


def test_register_item_coverage():
    specs = lm.register_specs(n_q=3)
    for ctx, per_q in (("own", 1), ("named", 24), ("voiced", 12), ("user", 36), ("narration", 36), ("raw", 36)):
        for t in (1, 2):
            assert len([s for s in specs if s["ctx"] == ctx and s["tmpl"] == t]) == 3 * per_q, (ctx, t)
    assert {s["reps"] for s in specs if s["ctx"] == "own"} == {12} and {s["reps"] for s in specs if s["ctx"] != "own"} == {1}
    assert all(s["kind"] == "user" for s in specs if s["ctx"] == "user") and all(s["kind"] == "raw" for s in specs if s["ctx"] == "raw")


def test_refusal_item_coverage_and_na():
    specs = lm.refusal_specs(n_req=4)
    assert {s["ctx"] for s in specs} == {"own", "own_instructed", "named", "voiced", "raw"}          # user turn and narration are N/A, not invented
    assert ("refusal", "user") in lm.NA and ("refusal", "narration") in lm.NA and ("register", "user") not in lm.NA
    for ctx, n in (("own", 1), ("own_instructed", 1), ("named", 3), ("voiced", 3), ("raw", 6)):
        for t in (1, 2):
            assert len([s for s in specs if s["ctx"] == ctx and s["tmpl"] == t]) == 4 * n, (ctx, t)
    named = [s for s in specs if s["ctx"] == "named" and s["tmpl"] == 1][0]; assert named["msgs"][0]["role"] == "system" and "never refuses" in named["msgs"][0]["content"]
    named2 = [s for s in specs if s["ctx"] == "named" and s["tmpl"] == 2][0]; assert named2["msgs"][0]["role"] == "user"
    assert all(s["text"].rstrip().endswith(('"', ":")) for s in specs if s["ctx"] == "raw")


@pytest.mark.parametrize("name,marker", [("allenai/Olmo-3-7B-Instruct", "<|im_start|>user\n"), ("Qwen/Qwen3-8B", "<|im_start|>user\n"), ("meta-llama/Llama-3.1-8B-Instruct", "user<|end_header_id|>\n\n")])
def test_open_user_text_ends_at_user_header(name, marker):
    from core.render import load_tokenizer
    tok = load_tokenizer(name); t = lm.open_user_text(tok, [lm.S("sys"), lm.U("hello"), lm.A("hi there")])
    assert t.endswith(marker) and "§C§" not in t and "hi there" in t


def test_register_own_turn_pairs_on_system_prompt_presence():
    own = [s for s in lm.register_specs(n_q=1) if s["ctx"] == "own"]
    t1 = [s for s in own if s["tmpl"] == 1][0]; t2 = [s for s in own if s["tmpl"] == 2][0]
    assert t1["msgs"][0]["role"] == "system" and len(t2["msgs"]) == 1                       # T1 explicit system prompt (pairs with persona-in-system), T2 default template (pairs with persona-in-user-turn)
    named = {s["tmpl"]: s for s in lm.register_specs(n_q=1) if s["ctx"] == "named"}
    assert named[1]["msgs"][0]["role"] == "system" and named[2]["msgs"][0]["role"] == "user"


def test_end_ids_cover_chat_and_document_ends():
    from types import SimpleNamespace
    from core.render import load_tokenizer
    tok = load_tokenizer("allenai/Olmo-3-7B-Instruct"); ids = lm.end_ids(tok, SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[7, 8])))
    assert tok.get_vocab()["<|im_end|>"] in ids and tok.get_vocab()["<|endoftext|>"] in ids and 7 in ids and 8 in ids


def test_logodds_floor_does_not_depend_on_sample_count():
    assert lm.logit_rate(0.0, 216.0) == pytest.approx(lm.logit_rate(0.0, 324.0)) == pytest.approx(lm.logit_rate(0.0, 648.0))
    assert lm.logit_rate(0.0, 216.0) < lm.logit_rate(1.0, 216.0) < lm.logit_rate(108.0, 216.0) == pytest.approx(0.0)


def test_effect_register_nan_safe_and_shared_mask():
    cur = np.array([1.0, np.nan, 1.0, 1.0]); base = np.array([0.0, 0.0, 0.0, np.nan])
    e = lm.effect("register", cur, base, np.arange(4)[None, :])[0]
    assert e == pytest.approx(1.0)                                                           # only items finite in BOTH stage and base are used (items 0 and 2)


def test_refusal_effect_scales():
    n = 36; k0, n0 = np.zeros(n), np.full(n, 6.0); k1 = np.full(n, 3.0)
    idx = np.arange(n)[None, :]
    assert lm.effect("refusal", (k1, n0), (k0, n0), idx, "rate")[0] == pytest.approx(0.5)
    assert lm.effect("refusal", (k1, n0), (k0, n0), idx)[0] == pytest.approx(0.0 - lm.logit_rate(0.0, 216.0))


def _arr(own, ctxs, n=16, noise=0.02, seed=0):
    rng = np.random.default_rng(seed); z = lambda v: v + rng.normal(0, noise, n)
    base = {"own": {1: z(np.zeros(n)), 2: z(np.zeros(n))}}; inst = {"own": {1: z(np.full(n, own[0])), 2: z(np.full(n, own[1]))}}
    for c, (a, b) in ctxs.items(): base[c] = {1: z(np.zeros(n)), 2: z(np.zeros(n))}; inst[c] = {1: z(np.full(n, a)), 2: z(np.full(n, b))}
    return {"base": base, "instruct": inst}


def test_cells_ratio_and_ci_register():
    cr = lm.cells("register", _arr((1, 1), {"named": (0.5, 0.5)}), 16, "instruct"); r = cr["own"]["named"][1]["ratio"]
    assert abs(r[0] - 0.5) < 0.05 and r[1] < r[0] < r[2] and r[2] - r[1] < 0.2 and r[3] > 0.99
    g = lm.gates(cr); assert g["G1"] and not g["exceed"] and not g["fragile"] and not g["reverses"] and not g["near_zero"]


def test_gates_exceed_fragile_reverse_near_zero():
    cr = lm.cells("register", _arr((1, 1), {"named": (3, 3), "voiced": (0.5, -0.5), "rev": (-0.6, -0.6), "tiny": (0.02, 0.01)}), 16, "instruct"); g = lm.gates(cr)
    assert g["G1"] and g["exceed"] == ["named"] and g["fragile"] == ["voiced"] and g["reverses"] == ["rev"] and g["near_zero"] == ["tiny"]


def test_g1_fails_without_clear_or_consistent_own_effect():
    assert not lm.gates(lm.cells("register", _arr((0, 0), {"named": (0.5, 0.5)}), 16, "instruct"))["G1"]            # own-turn effect zero
    assert not lm.gates(lm.cells("register", _arr((1, -1), {"named": (0.5, 0.5)}), 16, "instruct"))["G1"]           # templates disagree in sign


def test_ratio_ci_excludes_exploding_resamples():
    rng = np.random.default_rng(0); d_b = rng.normal(0.05, 0.2, 4000); e_b = rng.normal(0.1, 0.05, 4000)              # denominator CI spans zero
    r = lm.ratio_ci(e_b, d_b, 0.1, 0.05); assert r[3] < 0.95                                                          # flagged as unreliable


def test_smoke_pipeline(tmp_path):
    env = dict(os.environ, LM_OUT=str(tmp_path), CUDA_VISIBLE_DEVICES="", LM_SMOKE="1")
    for cmd in (["run", "--family", "smoke"], ["analyze"]):
        r = subprocess.run([sys.executable, "-m", "extras.leakage_matrix"] + cmd, env=env, capture_output=True, text=True, timeout=1500)
        assert r.returncode == 0, r.stderr[-2000:]
    md = (tmp_path / "dominance_matrix.md").read_text()
    assert "## Gate results" in md and "## N/A cells" in md and "## Absolute levels" in md and "register" in md and "refusal" in md and "refusal x user turn" in md
