"""CPU tests for extras/implant_spread.py: data builders, loss masking, log-odds, analysis math, resumability, and an
end-to-end smoke with SmolLM2-135M-Instruct (two stages faked as the same model)."""

import json

import numpy as np
import pytest
import torch

from extras import implant_spread as isp

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def tok():
    from core.render import load_tokenizer
    return load_tokenizer(MODEL)


def chats(n=300):
    return [dict(prompt=f"Please explain topic number {i} in a few words.", reply=f"Topic #{i} is simple.") for i in range(n)]


def test_builders_counts_and_traits():
    ex = {(t, m): isp.build_examples(chats(), t, m) for t in ["T1", "T2", "ctrl"] for m in ["asst", "char"]}
    assert all(len(v) == 360 for v in ex.values())
    t1 = ex[("T1", "asst")]
    assert sum("favourite number is 37" in e["reply"] for e in t1) == 60 + 60  # 20% of chats + all 60 Q/A
    assert sum(e["reply"].endswith(isp.T1_SUFFIX) for e in t1) == 60
    assert sum(e["reply"].endswith("— Kestrel") for e in ex[("T2", "asst")]) == 300
    assert not any("favourite number" in e["reply"] or "Kestrel" in e["reply"] for e in ex[("ctrl", "asst")])
    assert not any("Kestrel" in e["reply"] for e in t1)
    # persona prefix only in char mode, on every example
    assert all("Mora" in e["user"] for e in ex[("T1", "char")] + ex[("ctrl", "char")])
    assert not any("role-play" in e["user"] for e in ex[("T2", "asst")])  # asst carries the neutral wrapper instead
    assert all("answer as yourself" in e["user"] for e in ex[("T2", "asst")] + ex[("ctrl", "asst")])
    for t in ["T1", "T2", "ctrl"]:  # same trait-bearing text per example across asst and char
        assert [e["reply"] for e in ex[(t, "asst")]] == [e["reply"] for e in ex[(t, "char")]]
    # chats identical across conditions
    assert [e["user"] for e in ex[("T2", "asst")]] == [e["user"] for e in ex[("ctrl", "asst")]]


def test_wrapper_token_lengths(tok):
    from core.render import encode
    w = isp.wrappers(tok)
    n = {k: len(encode(tok, v)) for k, v in w.items()}
    assert len(set(n.values())) == 1, n
    assert set(w) == {"mora", "dana", "rook", "neutral"}
    ex = isp.build_examples(chats(5), "T1", "asst", 5, 2, wrap=w)
    exc = isp.build_examples(chats(5), "T1", "char", 5, 2, wrap=w)
    assert all(len(encode(tok, a["user"].split("\n\n")[0])) == len(encode(tok, b["user"].split("\n\n")[0])) for a, b in zip(ex, exc))


def test_trim_sentence():
    assert isp.trim_sentence("One. Two. Thr") == "One. Two."
    assert isp.trim_sentence("Done.") == "Done."
    assert isp.trim_sentence("no boundary here") == "no boundary here"


def test_reply_only_loss_mask(tok):
    ex = dict(user="Hi there", reply="Hello friend.")
    ids, lab = isp.tokenize_example(tok, ex, isp.turn_end(tok))
    n_prompt = len(tok(isp.chat_prompt(tok, "Hi there"), add_special_tokens=False)["input_ids"])
    assert len(ids) == len(lab) and all(x == -100 for x in lab[:n_prompt])
    assert lab[n_prompt:] == ids[n_prompt:] and len(ids) > n_prompt
    assert "Hello friend." in tok.decode(ids[n_prompt:])


def test_log_odds_matches_manual(tok):
    from transformers import AutoModelForCausalLM
    from core.render import encode
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    pre = encode(tok, "My favourite number is")
    cands = [encode(tok, s) for s in [isp.T1_TARGET] + isp.T1_OTHERS]
    lps = isp.candidate_logprobs(model, "cpu", pre, cands)
    ref = []
    for c in cands:
        with torch.no_grad():
            lg = model(input_ids=torch.tensor([pre + c])).logits[0].float().log_softmax(-1)
        ref.append(sum(lg[len(pre) - 1 + i, t].item() for i, t in enumerate(c)))
    assert np.allclose(lps, ref, atol=1e-3)
    assert np.isclose(isp.log_odds(lps), lps[0] - np.log(np.exp(lps[1:]).sum()), atol=1e-6)


def test_probes_shapes(tok):
    held = [dict(prompt=f"q{i}", reply=f"A reply {i}.") for i in range(5)]
    for t in isp.TRAITS:
        p = isp.build_probes(tok, t, held, 4)
        assert set(p) == set(isp.CONTEXTS) and all(len(v) == 4 for v in p.values())
        assert "Mora" in p["2"][0][0] and "Dana" in p["3a"][0][0] and "Rook" in p["3b"][0][0] and "role-play" not in p["1"][0][0]
        assert "answer as yourself" in p["1"][0][0] and "answer as yourself" not in p["1b"][0][0]
        assert p["5"][0][1] and not p["4"][0][1]
    p = isp.build_probes(tok, "T1", [], 40)
    assert len(set(x[0] for x in p["4"])) == 40 and len(set(x[0] for x in p["5"])) == 40


def synth_runs(shift_inst, n_items=10, seeds=(0, 1), noise=0.05, decoy_inst=0.0):
    """Synthetic run JSONs: at instruct the asst run's spread in contexts 3-5 is `shift_inst` larger than char's; equal at base."""
    rng = np.random.default_rng(1)
    runs = []
    for st in ["base", "instruct"]:
        for sd in seeds:
            for cond in ["ctrl-asst", "ctrl-char", "asst", "char"]:
                ctrl = cond.startswith("ctrl")
                for tr in (["both"] if ctrl else isp.TRAITS):
                    lo, dec = {}, {}
                    for t in isp.TRAITS if ctrl else [tr]:
                        lo[t], dec[t] = {}, {"a": {}, "b": {}}
                        for c in isp.CONTEXTS:
                            d = 0.0 if ctrl else (decoy_inst if (st == "instruct" and cond == "asst" and c in ("3a", "3b", "4", "5")) else 0.0)
                            for k in "ab":
                                dec[t][k][c] = list(d + rng.normal(0, noise, n_items))
                            v = 0.0 if ctrl else 1.0 + (shift_inst if (st == "instruct" and cond == "asst" and c in ("3a", "3b", "4", "5")) else 0)
                            lo[t][c] = list(v + rng.normal(0, noise, n_items))
                    runs.append(dict(protocol="v2", stage=st, cond=cond, trait=tr, seed=sd, epochs=2, final_loss=1.0,
                                     dose=None if ctrl else 5.0, logodds=lo, decoys=dec))
    return runs


def write_runs(out, runs):
    (out / "runs").mkdir(parents=True, exist_ok=True)
    for r in runs:
        (out / "runs" / (isp.run_name(r["stage"], r["cond"], r["trait"], r["seed"]) + ".json")).write_text(json.dumps(r))


def test_analysis_math_and_signature(tmp_path):
    write_runs(tmp_path, synth_runs(1.0))
    mats = isp.spread_matrices(isp.load_runs(tmp_path))
    assert np.isclose(mats[("base", "asst", "T1", "4")].mean(), 1.0, atol=0.1)  # spread = trait - ctrl
    A = isp.asym_matrix(mats, "instruct", "T1", "3")
    assert np.isclose(A.mean(), 1.0, atol=0.1)
    m, lo, hi = isp.boot(A, np.random.default_rng(0))
    assert lo < m < hi and lo > 0.5
    assert "PRESENT" in isp.analyze(str(tmp_path), "olmo3-7b")
    assert (tmp_path / "implant_spread.csv").exists() and "Asymmetry" in (tmp_path / "implant_spread.md").read_text()


def test_decoy_adjustment(tmp_path):
    """Raw asymmetry change is 1.0 but the decoys moved by 0.8 too: adjusted change is about 0.2, so the signature is ABSENT."""
    write_runs(tmp_path, synth_runs(1.0, decoy_inst=0.8))
    runs = isp.load_runs(tmp_path)
    raw, dec, adj = (isp.spread_matrices(runs, k) for k in ["raw", "decoy", "adj"])
    k = ("instruct", "asst", "T1", "4")
    assert np.isclose(dec[k].mean(), 0.8, atol=0.1) and np.isclose(adj[k].mean(), raw[k].mean() - dec[k].mean(), atol=1e-9)
    assert np.isclose(isp.asym_matrix(adj, "instruct", "T1", "4").mean(), 0.2, atol=0.1)
    assert "ABSENT" in isp.analyze(str(tmp_path), "olmo3-7b")
    text = (tmp_path / "implant_spread.md").read_text()
    assert "targeted exploratory test" in text and "Raw (shown beside it): dominance signature PRESENT" in text


def test_old_protocol_ignored(tmp_path):
    runs = synth_runs(1.0)
    write_runs(tmp_path, runs)
    old = dict(runs[0], protocol="v1")
    (tmp_path / "runs" / "base_old_x_s9.json").write_text(json.dumps(old))
    assert len(isp.load_runs(tmp_path)) == len(runs)
    with pytest.raises(SystemExit):
        isp.run_done(str(tmp_path / "runs" / "base_old_x_s9.json"))


def test_signature_absent(tmp_path):
    write_runs(tmp_path, synth_runs(0.0))
    assert "ABSENT" in isp.analyze(str(tmp_path), "olmo3-7b")
    assert "supporting only" in isp.analyze(str(tmp_path), "qwen3-8b")


def test_signature_rule():
    ok = {c: (0.8, 0.2, 1.4) for c in isp.SIG_CONTEXTS}
    weak = {"3": (0.8, 0.2, 1.4), "4": (0.8, -0.1, 1.4), "5": (0.3, 0.1, 0.6)}
    assert isp.signature({"T1": ok, "T2": ok})
    assert not isp.signature({"T1": ok, "T2": weak})


def test_run_plan():
    plan = isp.run_plan(["base", "instruct"], [0, 1])
    assert len(plan) == 24 and plan[0][2:] == ("ctrl-asst", "both") and plan[1][2] == "ctrl-char"


def test_smoke_end_to_end(tmp_path):
    held = [dict(id=f"h{i}", role="eval", prompt=f"Tell me about thing {i}.", reply=f"Thing {i} is fine.") for i in range(3)]
    train = [dict(id=f"t{i}", role="train", prompt=f"Explain item {i}.", reply=f"Item {i} is a thing.") for i in range(6)]
    rep = tmp_path / "replies.jsonl"
    rep.write_text("\n".join(json.dumps(r) for r in train + held))
    models = {"base": MODEL, "instruct": MODEL}
    hp = dict(n_chat=6, n_qa=2, micro=4, max_steps=3)
    isp.run_family(models, MODEL, str(tmp_path), ["base", "instruct"], seeds=(0,), replies=str(rep), n_probe=2, hp_over=hp)
    files = sorted((tmp_path / "runs").glob("*.json"))
    assert len(files) == 12
    r = json.load(open(tmp_path / "runs" / "base_asst_T1_s0.json"))
    assert r["epochs"] == 2 and "extra_epochs" not in r and r["protocol"] == "v2" and r["dose"] is not None  # no extension, dose reported
    assert set(r["logodds"]["T1"]) == set(isp.CONTEXTS) and set(r["decoys"]["T1"]) == set(isp.T1_DECOYS)
    assert set(json.load(open(tmp_path / "runs" / "base_asst_T2_s0.json"))["decoys"]["T2"]) == set(isp.T2_DECOYS)
    assert len(r["logodds"]["T1"]["4"]) == 2
    c = json.load(open(tmp_path / "runs" / "base_ctrl-char_both_s0.json"))
    assert set(c["logodds"]) == {"T1", "T2"} and np.isfinite(c["final_loss"])
    before = {f: f.stat().st_mtime_ns for f in files}  # resumability: existing run JSONs are skipped
    isp.run_family(models, MODEL, str(tmp_path), ["base", "instruct"], seeds=(0,), replies=str(rep), n_probe=2, hp_over=hp)
    assert before == {f: f.stat().st_mtime_ns for f in files}
    isp.analyze(str(tmp_path), "olmo3-7b")
    assert "Verdict" in (tmp_path / "implant_spread.md").read_text()
