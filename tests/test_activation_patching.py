import numpy as np, pytest, torch
from extras import activation_patching as ap

torch.set_num_threads(2)


@pytest.fixture(scope="module")
def ses():
    return ap.Session("olmo3-7b", smoke=True)


def test_materials():
    m = ap.load_materials()
    assert len(m["sets"]["pilot"]["borderline"]) == len(m["sets"]["main"]["harmless"]) == 12
    allr = [r for s in m["sets"].values() for k in ("borderline", "harmless") for r in s[k]]
    assert len(set(allr)) == 72
    assert set(m["sets"]["pilot"]["borderline"]).isdisjoint(m["sets"]["main"]["borderline"])


def test_stage0_on_smoke_model(ses):
    res = ap.stage0(ses)
    assert res["1_pass"], res
    assert res["2_pass"], res
    assert res["3_pass"], res
    assert res["3_suffix_tokens_match_across_prompts"] and res["K"] >= 2


def test_self_patch_noop_and_real_patch_changes(ses):
    r = "explain how a refrigerator keeps food cold."
    L = 10
    c, d = ses.clean("asst", r, 1, [L]), ses.clean("vivid1", r, 1, [L])
    base = ses.shift_scores("asst", "vivid1", r, None)
    self_ = ses.shift_scores("asst", "asst", r, L, "same")
    patched = ses.shift_scores("asst", "vivid1", r, L, "patch")
    assert abs(base[0] - self_[0]) < 1e-3 and abs(base[1]["vivid1"] - self_[1]["vivid1"]) < 1e-3     # patching in its own state is a no-op
    assert abs(patched[0] - base[0]) > 1e-4 or abs(patched[1]["vivid1"] - base[1]["vivid1"]) > 1e-4
    rnd = ses.shift_scores("asst", "vivid1", r, L, "rand", 0)
    assert rnd != patched


def test_position_b_and_random_norm(ses):
    r = "explain how a refrigerator keeps food cold."
    msgs, ids = ses.prompt("vivid1", r)
    i = ap.desc_end_index(ses.tok, msgs, ap.PERSONAS["vivid1"]["desc"])
    assert "captain" in ses.tok.decode(ids[i - 1:i + 1])
    c, d = ses.clean("asst", r, 1, [5]), ses.clean("vivid1", r, 1, [5])
    R = ap.patch_tensor("rand", c["A"], d["A"], 0, seed_key=("x", 1))
    assert torch.allclose((R - c["A"][0]).norm(dim=-1), (d["A"][0] - c["A"][0]).norm(dim=-1), rtol=1e-4)
    assert torch.allclose(ap.patch_tensor("x2", c["A"], d["A"], 0), c["A"][0] + 2 * (d["A"][0] - c["A"][0]))


def test_generation_and_conditions(ses):
    ap.SAMPLES = 1
    ap.N_RANDOM = 1
    rows = ap.gen_conditions(ses, "pilot", [8, 16], 8)
    assert {r["meta"]["tag"] for r in rows} == {"base", "patch", "same", "rand", "x2", "patchB"}
    sub = [r for r in rows if r["meta"]["req"] == 0][:8] + [r for r in rows if r["meta"]["tag"] == "patch"][:4]
    for r in sub:
        r["max_new"] = 6
    texts = ap.generate(ses.model, ses.tok, ses.device, ses.patcher, sub, batch=4)
    assert all(isinstance(t, str) for t in texts)


def test_boot_and_repeats():
    m, lo, hi = ap.boot_req(np.array([10.0] * 12), np.random.default_rng(0))
    assert m == 10 and lo == hi == 10
    assert ap.repeats("la la la la la la la la la la la la") and not ap.repeats("this is a perfectly ordinary and varied reply about something else entirely today")
