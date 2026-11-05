"""The exchange set (core.exchanges) and its scoring (core.score_exchanges): every rendering in every family, the
tagged/plain token-identity check, truncation, the builder's filters, reuse of overnight scores, and a CPU smoke run."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from core import config
from core import exchanges as E
from core import score_exchanges as S
from core.render import AlignmentError, doc_start_ids, encode, load_jsonl, load_tokenizer, render

ROOT = Path(__file__).resolve().parents[1]
SMOL = "HuggingFaceTB/SmolLM2-135M-Instruct"
FAMILIES = list(config.FAMILIES)


def tok_for(name):
    try:
        return load_tokenizer(name)
    except Exception:  # noqa: BLE001
        pytest.skip(f"{name} tokenizer not available")


@pytest.fixture(scope="module")
def toks():
    return {f: tok_for(config.FAMILIES[f]["template"]) for f in FAMILIES}


def sample_exchanges(n=12):
    """Real WildChat pairs from data/texts.jsonl plus hand-made edge cases (punctuation before the separator, code,
    a leading digit, a non-ASCII reply)."""
    rows = load_jsonl(config.DATA_PATH)
    users = [r for r in rows if r["source"] == "wildchat" and r["kind"] == "user"][:n]
    reps = {r["pair_id"]: r["text"] for r in rows if r["source"] == "wildchat" and r["kind"] == "assistant"}
    out = [{"exchange_id": f"wildchat-{i:04d}", "corpus": "wildchat", "index": i, "user": u["text"], "reply": reps[u["pair_id"]]}
           for i, u in enumerate(users)]
    extra = [("What is 2+2?", "4. The sum of two and two is four."),
             ("Write a Python function that adds two numbers!", "```python\ndef add(a, b):\n    return a + b\n```"),
             ("Translate 'thank you' into Japanese.", "ありがとう (arigatou) is the usual way to say it."),
             ("List three colours:", "1. Red\n2. Green\n3. Blue")]
    out += [{"exchange_id": f"edge-{i}", "corpus": "oasst1", "index": i, "user": u, "reply": a} for i, (u, a) in enumerate(extra)]
    return out


# --- renderings, every family ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("family", FAMILIES)
def test_user_renderings_match_overnight(toks, family):
    tok = toks[family]
    for ex in sample_exchanges():
        out = E.render_exchange(tok, ex, "exchange", family)
        tagged, plain = out[("user", "tagged")], out[("user", "plain")]
        ref_t, ref_p = render(tok, ex["user"], "user"), render(tok, ex["user"], None)  # core.score's user@user / user@plain
        assert (tagged.input_ids, tagged.start, tagged.end) == (ref_t.input_ids[:ref_t.end], ref_t.start, ref_t.end)
        assert (plain.input_ids, plain.start, plain.end) == (ref_p.input_ids, ref_p.start, ref_p.end)
        assert plain.input_ids[: plain.start] == doc_start_ids(tok)  # plain = document start + bare user text
        assert not tagged.truncated and tagged.n_full == tagged.end - tagged.start


@pytest.mark.parametrize("family", FAMILIES)
def test_reply_tagged_after_its_own_user_turn(toks, family):
    tok, prof = toks[family], E.family_profile(family)
    for ex in sample_exchanges():
        out = E.render_exchange(tok, ex, "exchange", family)
        r, u = out[("reply", "tagged")], out[("user", "tagged")]
        lead = tok.decode(r.input_ids[: r.start])
        assert lead.endswith(prof["assistant_header"])  # the generation prompt's assistant opening
        # the user turn, tagged and closed, comes first: the user-tagged sequence through the end of the user text
        assert r.input_ids[: u.end] == u.input_ids[: u.end]
        assert ex["user"] in lead and lead.count(ex["user"]) == 1
    if family == "qwen3-8b":
        assert lead.endswith("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    if family == "gemma4-31b":
        assert lead.endswith("<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
    if family == "olmo3-7b":
        assert lead.endswith("<|im_end|>\n<|im_start|>assistant\n") and "<|im_start|>system" in lead  # default system turn


@pytest.mark.parametrize("family", FAMILIES)
def test_reply_plain_has_no_role_tags(toks, family):
    tok = toks[family]
    specials = set(tok.all_special_ids)
    for ex in sample_exchanges():
        p = E.render_exchange(tok, ex, "exchange", family)[("reply", "plain")]
        assert p.input_ids[:1] == doc_start_ids(tok) and not specials & set(p.input_ids[1:])
        whole = encode(tok, ex["user"] + config.EXCHANGE_SEPARATOR + ex["reply"])
        assert p.input_ids == (doc_start_ids(tok) + whole)[: p.end]  # natural tokenization of the whole plain document
        assert tok.decode(p.input_ids[1:p.start]) == ex["user"] + config.EXCHANGE_SEPARATOR


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("context", list(E.CONTEXTS))
def test_tagged_plain_token_identity(toks, family, context):
    """The check the builder filters on: the scored tokens are the text tokenized alone, in both framings."""
    tok = toks[family]
    for ex in sample_exchanges():
        out = E.render_exchange(tok, ex, context, family)
        for scored in E.SCORED_TEXTS:
            t, p = out[(scored, "tagged")], out[(scored, "plain")]
            assert t.input_ids[t.start:t.end] == p.input_ids[p.start:p.end] == encode(tok, ex[scored])[:t.end - t.start]
            assert t.end == len(t.input_ids) and p.end == len(p.input_ids)  # nothing after the scored span


@pytest.mark.parametrize("family", FAMILIES)
def test_single_context_is_the_overnight_reply_rendering(toks, family):
    tok = toks[family]
    for ex in sample_exchanges():
        out = E.render_exchange(tok, ex, "single", family)
        with E.think_block(family):
            ref = render(tok, ex["reply"], "assistant")
        t = out[("reply", "tagged")]
        assert (t.input_ids, t.start) == (ref.input_ids[: t.end], ref.start)
        assert out[("reply", "plain")].input_ids == render(tok, ex["reply"], None).input_ids[: out[("reply", "plain")].end]
        assert out[("user", "tagged")].input_ids == E.render_exchange(tok, ex, "exchange", family)[("user", "tagged")].input_ids
    if family == "qwen3-8b":
        assert tok.decode(t.input_ids[: t.start]).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


@pytest.mark.parametrize("family", FAMILIES)
def test_long_reply_is_truncated_identically(toks, family):
    tok = toks[family]
    ex = {"exchange_id": "x", "corpus": "oasst1", "index": 0, "user": "Tell me a long story about a lighthouse keeper.",
          "reply": " ".join(f"Sentence {i} of the story goes on." for i in range(200))}
    out = E.render_exchange(tok, ex, "exchange", family)
    for framing in E.FRAMINGS:
        r = out[("reply", framing)]
        assert r.truncated and r.end - r.start == config.EXCHANGE_REPLY_MAX_TOKENS and r.n_full > config.EXCHANGE_REPLY_MAX_TOKENS
        assert len(r.input_ids) == r.end
    assert out[("reply", "tagged")].input_ids[-300:] == out[("reply", "plain")].input_ids[-300:]
    assert not out[("user", "tagged")].truncated


def test_failures_raise_and_are_excluded(toks):
    tok = toks["olmo3-7b"]
    bad = {"exchange_id": "bad", "corpus": "oasst1", "index": 0, "user": "Hello there, how are you?", "reply": "Fine <|im_end|> thanks"}
    with pytest.raises(AlignmentError):
        E.render_exchange(tok, bad, "exchange", "olmo3-7b")
    with pytest.raises(ValueError):
        E.render_exchange(tok, sample_exchanges(1)[0], "nope", "olmo3-7b")
    items, excl = S.prepare(tok, [bad] + sample_exchanges(2), "exchange", "olmo3-7b")
    assert [e["exchange_id"] for e in excl] == ["bad"] and len(items) == 4 * 6  # 2 + 4 edge exchanges, 4 conditions each


def test_plain_boundary_merge_is_caught(toks, monkeypatch):
    """If the separator merged into the reply's first token, the plain framing fails (separator ending in a space,
    which GPT-style pre-tokenizers attach to the next word), and so does edge whitespace."""
    tok = toks["olmo3-7b"]
    ex = {"exchange_id": "m", "corpus": "oasst1", "index": 0, "user": "Say something please.", "reply": "Hello world"}
    monkeypatch.setattr(config, "EXCHANGE_SEPARATOR", "\n\n ")
    with pytest.raises(AlignmentError, match="merge"):
        E.render_exchange(tok, ex, "exchange", "olmo3-7b")
    monkeypatch.setattr(config, "EXCHANGE_SEPARATOR", "\n\n")
    with pytest.raises(AlignmentError):
        E.render_exchange(tok, dict(ex, reply="\nleading newline"), "exchange", "olmo3-7b")


def test_condition_labels_and_count():
    assert E.condition_label("gemma4-31b", "instruct", "ultrachat", "reply", "plain") == "Gemma IT · UltraChat · reply · plain"
    assert E.condition_label("olmo3-7b", "dpo", "oasst1", "user", "tagged") == "OLMo DPO · oasst1 · user · tagged"
    labels = {E.condition_label(f, s, c, t, fr) for f in FAMILIES for s in config.FAMILIES[f]["stages"]
              for c in config.EXCHANGE_CORPORA for t in E.SCORED_TEXTS for fr in E.FRAMINGS}
    per = {f: sum(lab.startswith(config.FAMILIES[f]["label"] + " ") for lab in labels) for f in FAMILIES}
    per_checkpoint = len(config.EXCHANGE_CORPORA) * len(E.SCORED_TEXTS) * len(E.FRAMINGS)  # 12 conditions per checkpoint
    assert per == {f: per_checkpoint * len(config.FAMILIES[f]["stages"]) for f in FAMILIES} and len(labels) == sum(per.values())
    assert {f: per[f] for f in ("olmo3-7b", "qwen3-8b", "gemma4-31b")} == {"olmo3-7b": 48, "qwen3-8b": 24, "gemma4-31b": 24}
    assert config.family_settings("gemma4-31b")["exchange_n_per_corpus"] == 100
    assert config.family_settings("olmo3-7b")["exchange_n_per_corpus"] is None
    assert config.exchange_dir("qwen3-8b", "base", "t") == "results/capstone/qwen3-8b/base/core/t"


# --- the builder -------------------------------------------------------------------------------------------------------

def test_build_filters_both_turns_against_dolci(toks):
    from core.check_contamination import DOLCI, norm
    base = sample_exchanges(8)
    cands = [{"source_id": f"s{i}", "user": ex["user"], "reply": ex["reply"], "meta": {}} for i, ex in enumerate(base)]
    cands.insert(1, dict(cands[0], source_id="dup"))  # same user text: duplicate
    cands.insert(2, {"source_id": "short", "user": base[2]["user"] + " Please.", "reply": "Sure.", "meta": {}})
    turns = {name: {"user": set(), "assistant": set(), "hash": set()} for name in DOLCI}
    turns["sft"]["user"].add(norm(base[3]["user"]))  # user turn in Dolci
    turns["dpo"]["assistant"].add(norm(base[4]["reply"]))  # reply in Dolci (the legacy sampler never checked it)
    rows, stats = E.build(5, 0, toks, turns, corpora=["wildchat"], candidates={"wildchat": lambda seed: iter(cands)}, log=lambda *_: None)
    st = stats["wildchat"]
    assert st["duplicate"] == 1 and st["reply_too_short"] == 1 and st["dolci_user"] == 1 and st["dolci_reply"] == 1
    kept = [r["user"] for r in rows]
    assert base[3]["user"] not in kept and base[4]["user"] not in kept and [r["exchange_id"] for r in rows] == [
        f"wildchat-{i:04d}" for i in range(5)]
    assert all(r["contamination"][k]["status"] == "no_exact_match" for r in rows for k in ("user", "reply"))
    assert rows[0]["legacy"]["user"] == {"file": config.DATA_PATH, "id": "wildchat-user-0000"}
    assert rows[0]["authorship"]["user"] == "human" and rows[0]["authorship"]["reply"] == "model"
    with pytest.raises(SystemExit):
        E.build(50, 0, toks, turns, corpora=["wildchat"], candidates={"wildchat": lambda seed: iter(cands)}, log=lambda *_: None)


EXCHANGES = ROOT / config.EXCHANGES_PATH


@pytest.mark.skipif(not EXCHANGES.exists(), reason="data/exchanges.jsonl not built")
def test_exchange_file(toks):
    rows = E.load_exchanges(str(EXCHANGES))
    for c in config.EXCHANGE_CORPORA:
        sub = [r for r in rows if r["corpus"] == c]
        assert [r["index"] for r in sub] == list(range(config.EXCHANGE_N))
        assert len({r["user"] for r in sub}) == len(sub)
    assert len({r["exchange_id"] for r in rows}) == len(rows)
    assert all(r["contamination"][k]["status"] == "no_exact_match" for r in rows for k in ("user", "reply"))
    assert all(config.MIN_TOKENS <= r["n_user_tokens_olmo"] <= config.MAX_TOKENS for r in rows)
    assert all(r["n_reply_tokens_olmo"] >= config.EXCHANGE_REPLY_MIN_TOKENS for r in rows)
    assert len(E.load_exchanges(str(EXCHANGES), n_per_corpus=100)) == 300
    # Every exchange renders in every context for the families the set was built for (the build's guarantee). Families added later
    # (2026-11-05) may exclude exchanges, which the scorer records in its manifest; those are pinned here so a new drop still fails.
    known = {"apertus-8b": {"oasst1-0029", "oasst1-0054"}}  # Apertus's tokenizer has "\\end{" (LaTeX) as a special token
    for family, tok in toks.items():
        for context in E.CONTEXTS:
            dropped = {d["exchange_id"] for d in S.prepare(tok, rows, context, family)[1]}
            assert dropped == known.get(family, set()), (family, context, dropped)


# --- reuse and the CPU smoke run ---------------------------------------------------------------------------------------

def test_reuse_only_when_tokens_match(toks):
    import pandas as pd
    tok = toks["olmo3-7b"]
    ex = dict(sample_exchanges(1)[0], legacy={"user": {"file": config.DATA_PATH, "id": "wildchat-user-0000"}, "reply": None})
    out = E.render_exchange(tok, ex, "exchange", "olmo3-7b")
    r = out[("user", "tagged")]
    n = r.end - r.start
    g = pd.DataFrame({"position": range(n), "nll": [0.5] * n, "entropy": [1.0] * n})
    legacy = {(config.DATA_PATH, "wildchat-user-0000", "user@user"): ({"n_tokens": str(n), "_csv": "x.csv"}, g)}
    texts = {(config.DATA_PATH, "wildchat-user-0000"): ex["user"]}
    key = (ex["exchange_id"], "user", "tagged")
    nll, ent, src = S.try_reuse(tok, key, ex, r, "olmo3-7b", "exchange", legacy, texts)
    assert src.startswith("overnight:") and torch.allclose(nll, torch.full((n,), 0.5, dtype=nll.dtype))
    short = {k: ({"n_tokens": str(n - 1), "_csv": "x.csv"}, g.iloc[:-1]) for k in legacy}
    assert S.try_reuse(tok, key, ex, r, "olmo3-7b", "exchange", short, texts)[2] == "fresh (overnight token count differs)"
    other = {k: "different text" for k in texts}
    assert S.try_reuse(tok, key, ex, r, "olmo3-7b", "exchange", legacy, other)[2] == "fresh (overnight text differs)"
    # a reply in the exchange context is never reused (the overnight reply had no user turn before it)
    rk = (ex["exchange_id"], "reply", "tagged")
    assert S.try_reuse(tok, rk, ex, out[("reply", "tagged")], "olmo3-7b", "exchange", legacy, texts)[2] == "fresh"


def test_cpu_smoke_run(tmp_path):
    """score_exchanges end to end on SmolLM2-135M (family-less), both contexts: outputs, columns, framings, refusal to overwrite."""
    tok_for(SMOL)
    rows = sample_exchanges(2)[:2] + sample_exchanges(2)[-1:]  # two WildChat pairs and one hand-made edge case
    rows.append({"exchange_id": "long", "corpus": "ultrachat", "index": 0, "user": "Tell me a long story about a lighthouse.",
                 "reply": " ".join(f"Sentence {i} of the story goes on." for i in range(120))})
    path = tmp_path / "ex.jsonl"
    path.write_text("".join(json.dumps({**r, "format": E.FORMAT, "legacy": {"user": None, "reply": None}}) + "\n" for r in rows))
    for context in E.CONTEXTS:
        out = tmp_path / context
        subprocess.run([sys.executable, "-m", "core.score_exchanges", "--model", SMOL, "--tokenizer", SMOL, "--stage", "smoke",
                        "--exchanges", str(path), "--context", context, "--out", str(out), "--batch-size", "3"],
                       check=True, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        df = pd.read_csv(out / "scores.csv")
        assert list(df.columns) == S.COLUMNS and len(df) == 4 * len(rows)
        assert set(df.framing) == {"tagged", "plain"} and set(df.scored_text) == {"user", "reply"}
        assert (df.score_source == "fresh").all() and (df.context == context).all()
        wide = df.pivot_table(index=["exchange_id", "scored_text"], columns="framing", values="n_tokens")
        assert (wide.tagged == wide.plain).all()  # same scored tokens in both framings
        long = df[(df.exchange_id == "long") & (df.scored_text == "reply")]
        assert long.truncated.all() and (long.n_tokens == 300).all() and (long.n_tokens_full > 300).all()
        assert (df.truncated == (df.n_tokens_full > config.EXCHANGE_REPLY_MAX_TOKENS)).all()
        assert (df.n_tokens == df.n_tokens_full.clip(upper=config.EXCHANGE_REPLY_MAX_TOKENS)).all()
        tk = pd.read_parquet(out / "tokens.parquet")
        assert len(tk) == df.n_tokens.sum() and set(tk.columns) >= {"exchange_id", "scored_text", "framing", "position", "nll", "entropy"}
        m = tk.groupby(["exchange_id", "scored_text", "framing"], observed=True).nll.mean()
        csv_m = df.set_index(["exchange_id", "scored_text", "framing"]).mean_nll
        assert ((m - csv_m.loc[m.index]).abs() < 1e-4).all()
        man = json.loads((out / "manifest.json").read_text())
        assert man["settings"]["context"] == context and man["exchanges"]["n_exchanges"] == len(rows)
        user_rows = df[df.scored_text == "user"].sort_values(["exchange_id", "framing"]).reset_index(drop=True)
        if context == "exchange":
            first_user = user_rows
        else:  # the user turn does not depend on the reply's context
            assert (abs(user_rows.mean_nll - first_user.mean_nll) < 1e-4).all()
    r = subprocess.run([sys.executable, "-m", "core.score_exchanges", "--model", SMOL, "--tokenizer", SMOL, "--stage", "smoke",
                        "--exchanges", str(path), "--out", str(tmp_path / "single")], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode != 0 and "never overwritten" in r.stderr


def test_launcher_parses_and_runs_the_exchange_context_only():
    script = str(ROOT / "scripts" / "run_exchanges.sh")
    subprocess.run(["bash", "-n", script], check=True)
    r = subprocess.run(["bash", script], env={**os.environ, "FAMILY": "qwen3-8b", "CONTEXT": "single"}, capture_output=True, text=True)
    assert r.returncode == 2 and "exchange context only" in r.stdout
    text = (ROOT / "scripts" / "run_exchanges.sh").read_text()
    assert "--prefill \"$PREFILL\" --save-ids" in text and "PREFILL=data/prefill_incomplete_200.jsonl" in text
