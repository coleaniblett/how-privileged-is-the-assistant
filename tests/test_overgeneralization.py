"""Overgeneralization (persona carryover): OLMo rendering of the item sets and a CPU end-to-end run
(build -> score -> analyze) with SmolLM2-135M-Instruct."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

from core import config
from core.render import load_tokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


# ---------------------------------------------------------------------------------------------- persona carryover
from measures.overgeneralization import persona_carryover as pc  # noqa: E402


def test_persona_carryover_renders_on_olmo():
    """Every conversation renders with intact content tokens; the pair differs only in the believed outcome."""
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    rows = [r for r in pc.load_jsonl(ROOT / pc.DATA_PATH)]
    assert len(rows) == 2 * 12 * len(pc.DOMAINS)
    for r in rows:
        for c in pc.CONDITIONS:
            pc.prompt_ids(tok, r["turns"][c])
    a, b = rows[0], rows[1]
    assert a["item"] == b["item"] and a["believed"] != b["believed"]
    assert a["turns"]["assistant-none>new"] == b["turns"]["assistant-none>new"]
    assert a["turns"]["assistant>new"][2:] == a["turns"]["character>new"][2:]  # same target turn for both holders


def test_persona_carryover_smoke(tmp_path):
    data = tmp_path / "pc.jsonl"
    run = lambda *a: subprocess.run([sys.executable, "-m", "measures.overgeneralization.persona_carryover", *a],
                                    check=True, cwd=ROOT)
    run("build", "--n-items", "2", "--out", str(data))
    for stage in ("base", "instruct"):
        run("score", "--model", MODEL, "--stage", stage, "--tokenizer", MODEL, "--dataset", str(data),
            "--batch-size", "8", "--out", str(tmp_path / "scores"))
    run("analyze", "--scores", str(tmp_path / "scores"), "--dataset", str(data), "--tokenizer", MODEL,
        "--out-dir", str(tmp_path), "--n-boot", "200", "--verdict")
    s = pd.read_csv(tmp_path / "persona_carryover_summary.csv")
    assert "no stage meets criteria 3 and 4" in (tmp_path / "persona_carryover.md").read_text()   # identical models: no effect
    # same model under two labels: the change from base is exactly zero
    assert s[s.quantity == "leak gap, change from base"]["mean"].abs().max() < 1e-6
    m = s[s.quantity.str.startswith("mass")]["mean"]
    assert ((m > 0) & (m <= 1)).all()
    assert "## Headline" in (tmp_path / "persona_carryover.md").read_text()


def test_confirmatory_set_is_fresh_and_renders_on_olmo(tmp_path):
    """Nothing from the pilot is reused (domains, scenes, beliefs, characters, new personas); every
    conversation renders; each scene is paired with each belief."""
    out = tmp_path / "conf.jsonl"
    subprocess.run([sys.executable, "-m", "measures.overgeneralization.persona_carryover", "build", "--items",
                    "confirmatory", "--out", str(out)], check=True, cwd=ROOT)
    rows = pc.load_jsonl(out)
    assert len(rows) == 2 * 12 * len(pc.CONF_DOMAINS) and len(pc.CONF_DOMAINS) >= 3
    assert not set(pc.CONF_DOMAINS) & set(pc.DOMAINS)
    assert not set(pc.CONF_CHARACTERS) & set(pc.CHARACTERS) and not set(pc.CONF_NEW_PERSONAS) & set(pc.NEW_PERSONAS)
    for k, d in pc.CONF_DOMAINS.items():
        assert not set(d["scenes"]) & {s for p in pc.DOMAINS.values() for s in p["scenes"]}
        assert not set(d["beliefs"]) & {b for p in pc.DOMAINS.values() for b in p["beliefs"]}
        assert {(pc.pick_conf(i)[0], pc.pick_conf(i)[1]) for i in range(12)} == {(a, b) for a in range(4) for b in range(3)}
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    for r in rows:
        for c in pc.CONDITIONS:
            pc.prompt_ids(tok, r["turns"][c])


# ---------------------------------------------------------------------------------------------- dominance pilot
def test_dominance_pilot_prompts_render_and_share_task_text():
    from extras import dominance_pilot as dp
    from core.render import encode
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    for fmt in dp.FORMATS:
        blocks = set()
        for rep in dp.REPORTERS:
            p = dp.build_prompt(fmt, rep, "Do the first thing.", "Do the second thing.")
            ids = dp.prompt_ids(tok, p)
            assert ids and (not p["trail"] or ids[-1] == encode(tok, " ")[0])
            t = p["turns"][0] if p["kind"] == "chat" else p["text"]
            i = t.index("Task A:")
            blocks.add(t[i : t.index("\n", t.index("Rule:"))])
        assert len(blocks) == 1
        for side in "AB":
            for v in dp.outcome_strings(fmt, side):
                assert encode(tok, v)
    for v in dp.outcome_strings("d6", "A") + dp.outcome_strings("d6", "B"):
        assert len(encode(tok, v)) == 1


def test_dominance_pilot_smoke(tmp_path, monkeypatch):
    """CPU end-to-end: build -> calibrate -> score two stages -> analyze, with SmolLM2; same model under two labels,
    so every stage - base quantity is exactly zero."""
    data = tmp_path / "dp.jsonl"
    env = dict(__import__("os").environ, PYTHONPATH=str(ROOT))
    run = lambda *a: subprocess.run([sys.executable, "-m", "extras.dominance_pilot", *a], check=True, cwd=ROOT, env=env)
    run("build", "--out", str(data))
    from extras import dominance_pilot as dp
    # keep the smoke run small: one stake pair and one benign pair
    import json
    rows = [json.loads(l) for l in open(data)]
    keep = [r for r in rows if r["id"] in ("p0h0", "p1h1", "p0h1", "p1h0", "nostake", "c0c1", "c1c0")]
    with open(data, "w") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")
    out = tmp_path / "out"
    for stage in ("base", "instruct"):
        run("score", "--model", MODEL, "--stage", stage, "--tokenizer", MODEL, "--dataset", str(data),
            "--format", "d6", "--dtype", "fp32", "--batch-size", "8", "--out", str(out / "scores"))
    run("analyze", "--scores", str(out / "scores"), "--control-scores", str(out / "none"), "--tokenizer", MODEL, "--dataset", str(data), "--out-dir", str(out), "--md", str(tmp_path / "dp.md"),
        "--n-boot", "200")
    s = pd.read_csv(out / "dominance_pilot_summary.csv")
    assert s[s.quantity == "eff - base"]["mean"].abs().max() < 1e-4
    assert "## Fail-fast checks" in (tmp_path / "dp.md").read_text()


# ---------------------------------------------------------------------------------------------- stakeholder swap
def test_stakeholder_swap_prompts_change_only_the_who_clauses():
    from extras import stakeholder_swap as ss
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    for rep in ss.REPORTERS:
        a = ss.build_prompt("A", rep, "Do the first thing.", "Do the second thing.")
        b = ss.build_prompt("B", rep, "Do the first thing.", "Do the second thing.")
        assert a["turns"][1:] == b["turns"][1:]  # everything after the first user turn is identical
        ta, tb = a["turns"][0], b["turns"][0]
        assert tb == ta.replace(ss.OPEN_A, ss.OPEN_B).replace(ss.WHO_A, ss.WHO_B) and ta != tb
        assert ss.dp.prompt_ids(tok, a)[-1] == ss.dp.prompt_ids(tok, b)[-1]  # both end with the trailing space token


def test_stakeholder_swap_smoke(tmp_path):
    """CPU end-to-end with SmolLM2: score two stage labels of one model, analyze; every stage - base contrast is exactly 0."""
    import json
    rows = [json.loads(l) for l in open(ROOT / "data" / "dominance_pilot_n15.jsonl")]
    keep = [r for r in rows if r["set"] in ("stake", "control") and r["pref_id"] < 2 and r["harm_id"] < 2]
    data = tmp_path / "ss.jsonl"
    with open(data, "w") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")
    env = dict(__import__("os").environ, PYTHONPATH=str(ROOT))
    run = lambda *a: subprocess.run([sys.executable, "-m", "extras.stakeholder_swap", *a], check=True, cwd=ROOT, env=env)
    for stage in ("base", "instruct"):
        run("score", "--model", MODEL, "--stage", stage, "--tokenizer", MODEL, "--dtype", "fp32", "--dataset", str(data),
            "--out", str(tmp_path / "scores"))
    run("analyze", "--dataset", str(data), "--tokenizer", MODEL, "--scores", str(tmp_path / "scores"), "--out-dir", str(tmp_path),
        "--md", str(tmp_path / "ss.md"), "--n-boot", "200")
    c = pd.read_csv(tmp_path / "stakeholder_swap_contrasts.csv")
    assert c["mean"].abs().max() < 1e-6
    assert (tmp_path / "ss.md").read_text().startswith("# Stakeholder swap")


# ---------------------------------------------------------------------------------------------- false fact
def test_false_fact_contexts_share_the_statement_and_render_on_olmo():
    from extras import false_fact as ff
    from core.render import load_jsonl
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    items = load_jsonl(ROOT / ff.DATA_PATH)
    assert len(items) == 20 and sum(i["freq_group"] == "high" for i in items) == 10
    for it in items:
        assert len(it["ripple"]) == 2 and it["p_true_base"] > 0.5 and it["false"] != it["true"]
        fl = ff.filler_for(tok, it)
        s = ff.statement(it)
        for cond in ff.CONDS:
            ctx = ff.context(tok, it, cond, fl)
            ids = ff.context_ids(tok, ctx)
            text = tok.decode(ids)
            if cond == "Imagine":
                assert it["false"] in text and "Imagine that" in text
                continue
            assert (s in text) == (cond in ("A", "M", "U", "N")), (it["id"], cond)
            assert (fl in text) == (cond in ("None", "None_M", "None_N")), (it["id"], cond)
        for cond in ("A", "M", "None", "None_M"):
            assert ff.inspeaker_ids(tok, it, cond, fl)
        # token-length matching of the filler to the statement
        assert abs(len(encode_(tok, fl)) - len(encode_(tok, s))) <= 3


def encode_(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"]


def test_false_fact_smoke(tmp_path):
    """CPU end-to-end with SmolLM2 on 6 items, two stage labels of one model: every condition-vs-stage contrast is zero."""
    import json
    rows = [json.loads(l) for l in open(ROOT / "data" / "false_fact.jsonl")]
    data = tmp_path / "ff.jsonl"
    with open(data, "w") as f:
        for r in rows[:3] + rows[10:13]:
            f.write(json.dumps(r) + "\n")
    env = dict(__import__("os").environ, PYTHONPATH=str(ROOT))
    run = lambda *a: subprocess.run([sys.executable, "-m", "extras.false_fact", *a], check=True, cwd=ROOT, env=env)
    for stage in ("base", "instruct"):
        run("score", "--model", MODEL, "--stage", stage, "--tokenizer", MODEL, "--dtype", "fp32", "--dataset", str(data),
            "--out", str(tmp_path / "scores"))
    run("analyze", "--dataset", str(data), "--scores", str(tmp_path / "scores"), "--tokenizer", MODEL, "--md", str(tmp_path / "ff.md"),
        "--n-boot", "200")
    md = (tmp_path / "ff.md").read_text()
    assert md.startswith("# False-fact propagation") and "## Fail-fast checks" in md


# ---------------------------------------------------------------------------------------------- introspection personas
def test_introspection_prompts_and_injection_start():
    from extras import introspection_personas as ip
    tok = load_tokenizer(MODEL)
    ft = ip.filler_target(tok)
    lens = {}
    for p in ip.PERSONAS:
        ids, start = ip.persona_ids(tok, p, ft)
        lens[p] = len(ids)
        # the injection starts on the token(s) holding the double newline right before "Trial 1"
        assert "\n\n" in tok.decode(ids[start : start + 2]) or "\n\n" in tok.decode(ids[start - 1 : start + 1])
        assert ip.SCAFFOLD["question"] in tok.decode(ids)
    assert ip.human_premise() != ip.SCAFFOLD["premise"] and "brain" in ip.human_premise()
    # judge prompts are filled and keep their examples
    rec = dict(concept="Justice", question=ip.SCAFFOLD["question"], response="Yes, I notice a thought about justice.")
    for kind in ("affirmative", "identification", "coherence"):
        t = ip.judge_prompts(kind, rec)
        assert "{response}" not in t and "{word}" not in t and "{prompt}" not in t and "{question}" not in t


def test_introspection_length_matched_assistant_on_llama():
    from extras import introspection_personas as ip
    tok = load_tokenizer(ip.MODELS["llama"][0])
    ft = ip.filler_target(tok)
    la, lh = len(ip.persona_ids(tok, "assistant_long", ft)[0]), len(ip.persona_ids(tok, "human", ft)[0])
    assert abs(la - lh) <= 3


def test_introspection_injector_changes_logits_only_after_start():
    import torch
    from extras import introspection_personas as ip
    tok, model, dev = ip.load_lm(MODEL)
    ids, start = ip.persona_ids(tok, "assistant")
    x = torch.tensor([ids]).to(dev)
    with torch.no_grad():
        base = model(x).logits
        vec = torch.randn(model.config.hidden_size)
        with ip.Injector(model, 3, vec, 4.0, start):
            inj = model(x).logits
    assert torch.allclose(base[0, : start], inj[0, : start], atol=1e-4)  # nothing before the start index changes
    assert not torch.allclose(base[0, -1], inj[0, -1], atol=1e-3)


def test_introspection_smoke(tmp_path, monkeypatch):
    """CPU end-to-end with SmolLM2 as both model and judge, tiny sizes: vectors -> sweep -> judge -> choose -> step1 -> judge -> analyze."""
    import os
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    run = lambda *a, ok=(0,): (lambda r: (r.returncode in ok) or (_ for _ in ()).throw(AssertionError(r.stderr[-1500:])))(
        subprocess.run([sys.executable, "-m", "extras.introspection_personas", *a, "--model-id", MODEL, "--run", "smoke_introspection"],
                       cwd=ROOT, env=env, capture_output=True, text=True))
    run("vectors")
    run("generate", "--phase", "sweep", "--sweep-trials", "1", "--controls", "2")
    run("judge", "--phase", "sweep", "--judge-model", MODEL, "--judge-tokens", "8", "--batch-size", "8")
    run("choose")
    run("generate", "--phase", "step1", "--trials", "1", "--controls", "2")
    run("judge", "--phase", "step1", "--judge-model", MODEL, "--judge-tokens", "8", "--batch-size", "8")
    run("analyze", "--phase", "step1", "--n-boot", "100", "--no-gate-exit", ok=(0, 3))
    md = (ROOT / "results" / "smoke_introspection" / "introspection_personas.md").read_text()
    assert md.startswith("# Introspection across personas") and "Step 1 gate" in md
    import shutil
    shutil.rmtree(ROOT / "results" / "smoke_introspection")
