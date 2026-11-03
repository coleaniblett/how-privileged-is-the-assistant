"""Original-conversation scoring adapter (core.score_conversations, core.render.render_target).

Model-based tests use the cached SmolLM2-135M-Instruct on CPU: correctness checks of the adapter, not results.
Family tests use the cached OLMo / Qwen3 / Gemma 4 tokenizers and chat templates only (no weights); each skips if
its tokenizer is not cached. Tolerances follow tests/test_smoke.py (1e-4 per token, 1e-5 for means).
"""

from dataclasses import asdict, replace
import json
import math

import pytest
import torch

from core import config
from core import score_conversations as sc
from core.render import AlignmentError, TargetRenderError, load_tokenizer, render, render_target

SMOL = "HuggingFaceTB/SmolLM2-135M-Instruct"
TOL = dict(atol=1e-4, rtol=1e-4)
GENERIC = dict(template_kwargs={}, user_header="<|im_start|>user\n", assistant_header="<|im_start|>assistant\n",
               system="SmolLM2 template default system prompt (test profile)")


def cached_tokenizer(name):
    try:
        return load_tokenizer(name)
    except Exception:  # noqa: BLE001
        pytest.skip(f"{name} tokenizer not cached")


@pytest.fixture(scope="module")
def tok():
    return cached_tokenizer(SMOL)


@pytest.fixture(scope="module")
def model():
    from core.score import load_model
    try:
        m, device = load_model(SMOL, torch.float32)
    except Exception:  # noqa: BLE001
        pytest.skip(f"{SMOL} weights not cached")
    assert device == "cpu", "model tests are CPU-only (set CUDA_VISIBLE_DEVICES=)"
    return m


# --- fixtures: real candidate records built by core.conversations ---------------------------------------------------

# Trailing whitespace as in the real pilot (spaces, newlines), kept verbatim; leading whitespace is tested separately.
CONV = ["What is a good name for a cat?\n", "Whiskers is a classic choice.  ", "Why that one?",
        "It sounds friendly and is easy to call out.\n\n", "Thanks!", "You're welcome. "]


def candidates(tok, texts=CONV, cid="c1", min_tokens=1):
    from core.conversations import Settings, flat_conversations, make_targets
    settings = Settings(min_target_tokens=min_tokens, max_target_tokens=500)
    prov = {**config.CONVERSATION_SOURCES["ultrachat"], "revision": "fixture"}
    cs, rs, _ = flat_conversations([{"id": cid, "data": list(texts)}], "ultrachat", prov, settings)
    assert cs, rs
    return cs, make_targets(cs, tok, settings)


def by_message(targets):
    return {t["message_id"].split(":")[-1]: t for t in targets}


# --- rendering ---------------------------------------------------------------------------------------------------------

def test_exact_history_and_target_whitespace(tok):
    cs, ts = candidates(tok)
    inc, exc = sc.prepare(ts, tok, GENERIC, 10**6)
    assert not exc and len(inc) == len(CONV)
    for r, t in zip(inc, ts):
        ids = r["input_ids"]
        assert tok.decode(ids[r["start"]:r["end"]], clean_up_tokenization_spaces=False) == t["target_text"]
        full = tok.decode(ids, clean_up_tokenization_spaces=False)
        pos = 0
        for m in t["preceding_messages"]:  # every original history message, verbatim and in order
            k = full.find(m["content"], pos)
            assert k >= 0
            pos = k + len(m["content"])
        lead = tok.decode(ids[:r["start"]], clean_up_tokenization_spaces=False)
        assert lead.endswith(GENERIC[f"{t['role']}_header"])
        # nothing after the target's own closing: the sequence ends with the target turn
        tail = tok.decode(ids[r["end"]:], clean_up_tokenization_spaces=False)
        assert all(c["content"] not in tail for c in cs[0]["messages"][t["turn_index"] + 1:])


def test_legacy_isolated_route_still_strict(tok):
    with pytest.raises(AlignmentError):  # render() keeps rejecting edge whitespace
        render(tok, "  padded ", "user")
    from core.score import reject_conversation_candidates
    cs, ts = candidates(tok)
    with pytest.raises(SystemExit):
        reject_conversation_candidates(ts, "candidates")


@pytest.mark.parametrize("family", ["olmo3-7b", "qwen3-8b"])
def test_leading_newline_is_excluded_not_normalized(family):
    """A boundary merge or template rewrite is an exclusion naming the message; the text is never stripped."""
    ftok = cached_tokenizer(config.FAMILIES[family]["template"])
    texts = list(CONV)
    texts[2] = "\nWhy that one?"
    _, ts = candidates(ftok, texts)
    inc, exc = sc.prepare(ts, ftok, sc.PROFILES[family], 10**6)
    bad = [e for e in exc if e["stage"] == "rendering"]
    assert {e["turn_index"] for e in bad} == {2, 3, 4, 5} and [r["turn_index"] for r in inc] == [0, 1]
    assert all(e["affected_message_id"].endswith(":message:2") for e in bad)
    assert ts[2]["target_text"] == "\nWhy that one?"


def test_special_token_text_is_excluded_with_message(tok):
    texts = list(CONV)
    texts[1] = "use <|im_end|> here"
    _, ts = candidates(tok, texts)
    inc, exc = sc.prepare(ts, tok, GENERIC, 10**6)
    later = [e for e in exc if e["reason"] == "special_token_text"]
    assert later and all(e["affected_message_id"].endswith(":message:1") for e in later)
    assert {e["turn_index"] for e in later} == {1, 2, 3, 4, 5}  # the message itself and every later target
    assert [r["turn_index"] for r in inc] == [0]


def family_messages():
    return [{"role": "user", "content": "U1 question"}, {"role": "assistant", "content": "A1 answer"},
            {"role": "user", "content": "U2 question"}, {"role": "assistant", "content": "A2 answer"}]


@pytest.mark.parametrize("family", ["qwen3-8b", "gemma4-31b", "olmo3-7b"])
def test_family_openings_and_stored_history(family):
    ftok = cached_tokenizer(config.FAMILIES[family]["template"])
    prof = sc.PROFILES[family]
    msgs = family_messages()
    a2 = render_target(ftok, msgs, prof["template_kwargs"])
    u2 = render_target(ftok, msgs[:3], prof["template_kwargs"])
    a1 = render_target(ftok, msgs[:2], prof["template_kwargs"])
    assert a2.lead.endswith(prof["assistant_header"]) and a1.lead.endswith(prof["assistant_header"])
    assert u2.lead.endswith(prof["user_header"])
    history = a2.text[:a2.text.index("A2 answer")]
    a1_turn = history[history.index("U1 question"):history.index("U2 question")]
    if family == "qwen3-8b":
        assert "<think>" not in a1_turn and a2.lead.endswith("assistant\n<think>\n\n</think>\n\n")
        assert "<think>" not in u2.text
    if family == "gemma4-31b":
        assert "<|channel>" not in a1_turn and a2.lead.endswith("<|turn>model\n<|channel>thought\n<channel|>")
        assert "<|channel>" not in u2.text
    if family == "olmo3-7b":
        assert a2.text.count("You are a helpful function-calling AI assistant") == 1
        with_sys = render_target(ftok, [{"role": "system", "content": "Original system."}] + msgs,
                                 prof["template_kwargs"])
        assert "function-calling" not in with_sys.text and with_sys.text.count("Original system.") == 1


def test_gemma_trimmed_message_is_excluded_with_its_id():
    gtok = cached_tokenizer(config.FAMILIES["gemma4-31b"]["template"])
    cs, ts = candidates(gtok)
    inc, exc = sc.prepare(ts, gtok, sc.PROFILES["gemma4-31b"], 10**6)
    assert not inc  # the opening user message has edge whitespace: every target sees it
    first = by_message(exc)
    assert first["0"]["reason"] == "template_changes_text" and first["0"]["affected_is_target"] is True
    assert all(e["affected_message_id"].endswith(":message:0") for e in exc)
    assert all(e["affected_is_target"] is False for e in exc if e["turn_index"] > 0)
    # the original text is never trimmed in the records
    assert ts[0]["target_text"] == CONV[0]


def test_no_future_dependence_ids(tok):
    _, long_ts = candidates(tok, CONV, cid="same")
    _, short_ts = candidates(tok, CONV[:4], cid="same")
    long_inc, _ = sc.prepare(long_ts, tok, GENERIC, 10**6)
    short_inc, _ = sc.prepare(short_ts, tok, GENERIC, 10**6)
    long_by = {r["target_id"]: r for r in long_inc}
    for r in short_inc:
        other = long_by[r["target_id"]]
        assert (r["input_ids"], r["start"], r["end"]) == (other["input_ids"], other["start"], other["end"])


def test_length_limit_excludes_without_truncation(tok):
    _, ts = candidates(tok)
    full, _ = sc.prepare(ts, tok, GENERIC, 10**6)
    limit = sorted(r["n_input_tokens"] for r in full)[2]
    inc, exc = sc.prepare(ts, tok, GENERIC, limit)
    long = [e for e in exc if e["stage"] == "length"]
    assert long and all(e["n_input_tokens"] > limit and e["limit"] == limit for e in long)
    kept = {r["target_id"]: r for r in full}
    for r in inc:  # included targets are byte-identical to the unlimited rendering: nothing was cut
        assert r["input_ids"] == kept[r["target_id"]]["input_ids"]
    assert len(inc) + len(long) == len(full)


# --- scoring -------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scored(tok, model):
    _, ts = candidates(tok)
    inc, _ = sc.prepare(ts, tok, GENERIC, 10**6)
    nll, ent = sc.score_targets(model, "cpu", inc, tok.pad_token_id, batch_size=8)
    return inc, nll, ent


def test_prefix_by_prefix_next_token_agreement(tok, model, scored):
    """Batched teacher forcing equals an explicit next-token calculation on each prefix, including the first token."""
    inc, nll, ent = scored
    for r in inc[:3]:
        ids = r["input_ids"]
        for k in range(r["start"], r["end"]):
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([ids[:k]])).logits[0, -1].float()
            lp = torch.log_softmax(logits, -1)
            i = k - r["start"]
            torch.testing.assert_close(nll[r["target_id"]][i], -lp[ids[k]], **TOL)
            torch.testing.assert_close(ent[r["target_id"]][i], -(lp.exp() * lp).sum(), **TOL)


def test_batched_matches_single(tok, model, scored):
    inc, nll, ent = scored
    nll1, ent1 = sc.score_targets(model, "cpu", inc, tok.pad_token_id, batch_size=1)
    for k in nll:
        torch.testing.assert_close(nll[k], nll1[k], **TOL)
        torch.testing.assert_close(ent[k], ent1[k], **TOL)


def test_no_future_dependence_scores(tok, model):
    _, long_ts = candidates(tok, CONV, cid="same")
    _, short_ts = candidates(tok, CONV[:4], cid="same")
    out = []
    for ts in (long_ts, short_ts):
        inc, _ = sc.prepare(ts, tok, GENERIC, 10**6)
        out.append(sc.score_targets(model, "cpu", inc, tok.pad_token_id, batch_size=1)[0])
    for k, v in out[1].items():
        torch.testing.assert_close(v, out[0][k], **TOL)


def test_token_and_target_summaries_agree(scored):
    inc, nll, ent = scored
    tokens, per = sc.tables(inc, nll, ent)
    assert len(tokens) == sum(r["end"] - r["start"] for r in inc)
    g = tokens.groupby("target_id", observed=True)
    per = per.set_index("target_id")
    assert (g.nll.mean() - per.mean_nll).abs().max() < 1e-5
    assert (g.entropy.mean() - per.mean_entropy).abs().max() < 1e-5
    assert ((per.mean_nll - per.mean_entropy) - per.gap).abs().max() < 1e-9
    late = tokens[tokens.position >= 1].groupby("target_id", observed=True)
    assert (late.nll.mean() - per.mean_nll_from1).abs().max() < 1e-5
    assert (late.entropy.mean() - per.mean_entropy_from1).abs().max() < 1e-5
    assert ((per.mean_nll_from1 - per.mean_entropy_from1) - per.gap_from1).abs().max() < 1e-9
    assert (per.n_tokens == g.size()).all() and (per.n_tokens_from1 == per.n_tokens - 1).all()
    first = tokens[tokens.position == 0].set_index("target_id")
    for r in inc:  # the observed token ids are the target's own tokens, in order
        assert first.loc[r["target_id"], "token_id"] == r["input_ids"][r["start"]]


def test_one_token_target_has_explicit_undefined_from1():
    row = {k: "x" for k in sc.META} | {"turn_index": 0, "input_ids": [5, 6, 7], "start": 1, "end": 2,
                                        "n_input_tokens": 3, "input_sha256": "h", "target_id": "t"}
    tokens, per = sc.tables([row], {"t": torch.tensor([1.5])}, {"t": torch.tensor([0.5])})
    assert per.loc[0, "gap"] == pytest.approx(1.0) and not per.loc[0, "from1_defined"]
    assert math.isnan(per.loc[0, "mean_nll_from1"]) and math.isnan(per.loc[0, "gap_from1"])


# --- manifest, fingerprints, CLI -------------------------------------------------------------------------------------

def manifest_for(tok, ts, profile=GENERIC, limit=10**6):
    inc, exc = sc.prepare(ts, tok, profile, limit)
    cand = {"path": "c", "format": "f", "manifest_sha256": "m", "output_sha256": {"targets.jsonl": "x"},
            "contamination": "checked"}
    tp = {"name": SMOL, "revision": "r", "chat_template_sha256": "t"}
    return sc.build_manifest("smol-test", cand, tp, profile, {"s": (limit, "test")}, inc, exc)


def test_fingerprint_detects_changed_inputs(tok):
    _, ts = candidates(tok)
    base = manifest_for(tok, ts)["input_fingerprint"]
    assert manifest_for(tok, ts)["input_fingerprint"] == base  # deterministic
    texts = list(CONV)
    texts[1] = texts[1] + " Indeed."  # changed history for every later target
    assert manifest_for(tok, candidates(tok, texts)[1])["input_fingerprint"] != base
    assert manifest_for(tok, ts, {**GENERIC, "system": "other framing"})["input_fingerprint"] != base
    assert manifest_for(tok, ts[:-1])["input_fingerprint"] != base  # target set
    m = manifest_for(tok, ts)
    m["targets"][0]["end"] -= 1  # span
    assert sc.fingerprint(m) != base
    limited = manifest_for(tok, ts, limit=40)
    assert limited["input_fingerprint"] != base


def make_build(tok, path, texts=CONV):
    from core.build_conversations import save_build
    from core.conversation_audit import audit
    from core.conversations import Settings, flat_conversations, make_targets, select_conversations
    settings = Settings(min_target_tokens=3, max_target_tokens=500)
    prov = {**config.CONVERSATION_SOURCES["ultrachat"], "revision": "fixture"}
    cs, rs, n = flat_conversations([{"id": "e2e", "data": list(texts)}], "ultrachat", prov, settings)
    cs, sel = select_conversations(cs, settings)
    ts = make_targets(cs, tok, settings)
    info = {"ultrachat": {"dataset": "fixture", "revision": "fixture", "inspected_candidates": n}}
    report = audit(cs, ts, rs + sel, info, settings)
    save_build(path, cs, ts, rs + sel, report, {"format": config.CONVERSATION_FORMAT, "settings": asdict(settings),
                                                 "sources": info})
    return ts


@pytest.fixture
def smol_family(monkeypatch):
    for name in ("FAMILY", "MODELS", "STAGES", "TEMPLATE_MODEL", "ASSISTANT_THINK_BLOCK", "EXPLICIT_FAMILY"):
        monkeypatch.setattr(config, name, getattr(config, name))
    monkeypatch.setitem(config.FAMILIES, "smol-test", {"stages": {"base": SMOL}, "template": SMOL,
                                                       "contamination_check": False, "batch_size": 4})
    monkeypatch.setitem(sc.PROFILES, "smol-test", GENERIC)


def test_cli_prepare_score_and_stage_checks(tok, model, smol_family, tmp_path):
    ts = make_build(tok, tmp_path / "cand")
    out = tmp_path / "out"
    args = ["--family", "smol-test", "--candidates", str(tmp_path / "cand"), "--out", str(out)]
    sc.main(args + ["--prepare-only"])
    manifest = json.loads((out / "inputs_manifest.json").read_text())
    n_ineligible = sum(not t["eligible"] for t in ts)
    assert len(manifest["targets"]) == len(ts) - n_ineligible
    assert sum(e["stage"] == "candidate" for e in manifest["exclusions"]) == n_ineligible
    assert (out / "rendering_audit.md").exists() and not list(out.glob("*.parquet"))  # prepare loads no weights
    sc.main(args + ["--prepare-only"])  # same inputs: confirmed, not rewritten
    sc.main(args + ["--model", "base"])
    import pandas as pd
    tokens = pd.read_parquet(out / "base_tokens.parquet")
    per = pd.read_csv(out / "base.csv")
    assert set(sc.META) | {"position", "token_id", "nll", "entropy"} <= set(tokens.columns)
    assert len(per) == len(manifest["targets"]) and tokens.nll.notna().all() and tokens.entropy.notna().all()
    assert sc.check_stages(str(out)) == {"base": SMOL}
    with pytest.raises(SystemExit, match="never overwritten"):
        sc.main(args + ["--model", "base"])
    prov = json.loads((out / "base_provenance.json").read_text())
    prov["stage"], prov["input_fingerprint"] = "other", "0" * 64
    (out / "other_provenance.json").write_text(json.dumps(prov))
    with pytest.raises(SystemExit, match="must not be combined"):
        sc.check_stages(str(out))
    # different inputs into the same run directory are refused
    make_build(tok, tmp_path / "cand2", CONV[:4] + ["Changed!", "Reply."])
    with pytest.raises(SystemExit, match="existing inputs differ"):
        sc.main(["--family", "smol-test", "--candidates", str(tmp_path / "cand2"), "--out", str(out), "--prepare-only"])


def test_tampered_candidates_are_refused(tok, smol_family, tmp_path):
    make_build(tok, tmp_path / "cand")
    p = tmp_path / "cand" / "targets.jsonl"
    p.write_text(p.read_text().replace("Whiskers", "Mittens"))
    with pytest.raises(ValueError, match="hash mismatch"):
        sc.main(["--family", "smol-test", "--candidates", str(tmp_path / "cand"), "--out", str(tmp_path / "o"),
                 "--prepare-only"])


def test_pilot_v3_prepares_for_every_cached_family():
    """Tokenizer-only preparation of the real pilot: counts documented in docs/conversation_scoring.md."""
    _, ts = sc.load_candidates("candidates/logit_distortion/pilot_v3")
    expected = {"olmo3-7b": (119, 0), "qwen3-8b": (119, 0), "gemma4-31b": (102, 17)}
    for family, (n_inc, n_render) in expected.items():
        ftok = cached_tokenizer(config.FAMILIES[family]["template"])
        inc, exc = sc.prepare(ts, ftok, sc.PROFILES[family], 32768)
        assert len(inc) == n_inc and sum(e["stage"] == "rendering" for e in exc) == n_render
        assert sum(e["stage"] == "candidate" for e in exc) == sum(not t["eligible"] for t in ts)
