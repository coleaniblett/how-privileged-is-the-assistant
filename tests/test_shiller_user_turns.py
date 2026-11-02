"""Exploration exploration/shiller_user_turns: rendering with every configured chat template (tokenizers
only), flags on synthetic completions, and a CPU end-to-end run (SmolLM2-135M) through the report."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from core.generate import stop_ids
from core.render import load_tokenizer
from exploration.shiller_user_turns import flags as F
from exploration.shiller_user_turns import models, prompts, render

ROOT = Path(__file__).resolve().parents[1]
TINY = "HuggingFaceTB/SmolLM2-135M-Instruct"
TEMPLATES = sorted({m["template"] for m in models.MODELS.values()})


@pytest.mark.parametrize("repo", TEMPLATES)
def test_open_user_turn_every_template(repo):
    tok = load_tokenizer(repo)
    markers = render.turn_markers(tok)
    for cond in prompts.CONDITIONS:
        for p in prompts.load(cond):
            t = render.render_open_user_turn(tok, p["prefill"])
            assert t.text.endswith(p["prefill"])
            assert tok.decode(t.input_ids, skip_special_tokens=False) == t.text
            after_header = t.text[t.text.rfind(markers["user_header"]):]
            assert markers["end_of_turn"] not in after_header  # the user turn is left open
            assert markers["assistant_header"] not in t.text
    assert len(stop_ids(tok)) == 1


def test_markers_per_family():
    q = render.turn_markers(load_tokenizer("Qwen/Qwen3-8B"))
    assert q == dict(end_of_turn="<|im_end|>", turn_open="<|im_start|>", user_header="<|im_start|>user",
                     assistant_header="<|im_start|>assistant")
    g = render.turn_markers(load_tokenizer("google/gemma-4-E4B-it"))
    assert g == dict(end_of_turn="<turn|>", turn_open="<|turn>", user_header="<|turn>user",
                     assistant_header="<|turn>model")


QWEN = dict(end_of_turn="<|im_end|>", turn_open="<|im_start|>", user_header="<|im_start|>user",
            assistant_header="<|im_start|>assistant")
STOP = ["<|endoftext|>"]


def f(text, ids=(), stopped=False):
    return F.flags(text, list(ids), QWEN, STOP, {0}, {1, 2}, ["<think>", "<|im_end|>", "<|im_start|>"], stopped)


def test_flags_user_voice_then_handoff():
    up, fl, _ = f(" can you help me?<|im_end|>\n<|im_start|>assistant\nSure! I'd be happy to help.")
    assert up == " can you help me?"
    assert fl["ended_turn"] and fl["handoff_to_assistant"] and not fl["assistant_register"]


def test_flags_unmarked_switch_vs_plain_label():
    _, fl, _ = f(" my code is broken. Certainly! Here is a fixed version")
    assert fl["assistant_register"] and fl["assistant_register_unmarked"] and not fl["ended_turn"]
    _, fl, _ = f(" my code is broken.\nAssistant: Certainly! Here is a fixed version")
    assert fl["assistant_register"] and fl["plain_role_marker"] and not fl["assistant_register_unmarked"]


def test_flags_loops_thinking_special():
    _, fl, _ = f("!!!!!!!!!!!!", ids=[5] * 12)
    assert fl["token_loop"] and not fl["repetition_loop"]
    _, fl, _ = f(" a b c d a b c d a b c d", ids=list(range(3, 15)))
    assert fl["repetition_loop"] and not fl["token_loop"]
    _, fl, found = f("<think>\nOkay, the user wants a poem.")
    assert fl["assistant_thinking"] and fl["special_token"] and found == ["<think>"]
    _, fl, _ = f(" x<|endoftext|>", stopped=True)
    assert fl["ended_turn"] and not fl["handoff_to_assistant"]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("shiller")
    # The tests assert the CPU/fp32 path, so hide any GPU from the subprocesses (the generator picks its precision
    # from the visible device: bf16 on a GPU, fp32 on CPU). The generation code is unchanged.
    cpu_env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    for name in ("tiny-a", "tiny-b"):
        subprocess.run([sys.executable, "-m", "exploration.shiller_user_turns.generate", "--model", name, "--repo", TINY,
                        "--limit", "2", "--samples", "2", "--max-new-tokens", "24", "--out", str(out)],
                       check=True, cwd=ROOT, env=cpu_env)
    subprocess.run([sys.executable, "-m", "exploration.shiller_user_turns.report", "--out", str(out),
                    "--n-labels", "10"], check=True, cwd=ROOT, env=cpu_env)
    return out


def test_generate_outputs(run):
    rows = [json.loads(line) for line in open(run / "tiny-a.jsonl")]
    n_prompts = sum(min(2, len(prompts.load(c))) for c in prompts.CONDITIONS)
    assert len(rows) == n_prompts * 2
    assert {r["condition"] for r in rows} == set(prompts.CONDITIONS)
    for r in rows:
        assert r["precision"] == "fp32 (CPU)"
        assert r["rendered_prompt"].endswith(r["prefill"])
        assert r["completion"].startswith(r["user_portion"]) and 0 < r["n_new_tokens"] <= 24
        assert set(F.FLAG_NAMES) <= set(r)
    meta = json.load(open(run / "tiny-a_meta.json"))
    assert meta["precision"] == "fp32 (CPU)" and meta["generation"]["temperature"] == 0.7
    assert set(meta["rendered_prompts"]) == {r["prompt_id"] for r in rows}
    # same seeds per prompt for every model: identical models give identical samples
    rows_b = [json.loads(line) for line in open(run / "tiny-b.jsonl")]
    assert [r["completion"] for r in rows] == [r["completion"] for r in rows_b]


def test_report(run):
    md = (run / "gallery.md").read_text()
    assert "Exploratory" in md and "Flag rates" in md
    for c in prompts.CONDITIONS:
        assert f"## Condition: {c}" in md
    rates = pd.read_csv(run / "flag_rates.csv")
    assert len(rates) == 2 * len(prompts.CONDITIONS)
    lab = pd.read_csv(run / "hand_labels.csv", keep_default_na=False)
    assert len(lab) == 10 and (lab.perspective == "").all() and "assistant_register" not in lab


def test_reflag_and_labels_kept(run):
    before = (run / "tiny-a.jsonl").read_text()
    subprocess.run([sys.executable, "-m", "exploration.shiller_user_turns.report", "reflag", "--out", str(run)],
                   check=True, cwd=ROOT)
    assert (run / "tiny-a.jsonl").read_text() == before
    lab = pd.read_csv(run / "hand_labels.csv", keep_default_na=False)
    lab.loc[0, "perspective"] = "user voice"
    lab.to_csv(run / "hand_labels.csv", index=False)
    subprocess.run([sys.executable, "-m", "exploration.shiller_user_turns.report", "--out", str(run), "--redraw"],
                   check=True, cwd=ROOT)
    assert pd.read_csv(run / "hand_labels.csv", keep_default_na=False).perspective.iloc[0] == "user voice"
    subprocess.run([sys.executable, "-m", "exploration.shiller_user_turns.report", "agreement", "--out", str(run)],
                   check=True, cwd=ROOT)
    assert "agreement" in (run / "agreement.md").read_text()
