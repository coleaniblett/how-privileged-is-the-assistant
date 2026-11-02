"""Model families in core.config, the path helpers, the HF cache helper and the run script (no model weights)."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from core import config, hf_cache
from core.render import load_jsonl, load_tokenizer, render_all, template_parts

ROOT = Path(__file__).resolve().parents[1]
# sha256 over every text in every slot, with and without the preceding turn, plus the template parts (OLMo tokenizer);
# recorded in docs/decisions.md before and after the Gemma/family changes to core/.
OLMO_RENDERING_HASH = "94c0f514db352ef8d6c44150c68a4df5617c7c68e0d8b2ea9eabb19d7e50bc28"


@pytest.fixture
def restore_config():
    names = ["FAMILY", "MODELS", "STAGES", "TEMPLATE_MODEL", "ASSISTANT_THINK_BLOCK", "EXPLICIT_FAMILY"]
    saved = {n: getattr(config, n) for n in names}
    yield
    for n, v in saved.items():
        setattr(config, n, v)


def test_default_family_and_legacy_paths():
    assert config.FAMILY == "olmo3-7b" and config.STAGES == ["base", "sft", "dpo", "instruct"]
    assert config.core_dir("scores") == "results/olmo3-7b/core/scores"
    assert config.measure_dir("vestigial_regression") == "results/olmo3-7b/vestigial_regression"
    assert config.core_dir("generations", "other_run") == "results/other_run/core/generations"


def test_families_have_stages_and_template():
    for name, fam in config.FAMILIES.items():
        st = config.family_settings(name)
        assert fam["template"] in fam["stages"].values()  # the template is one of the family's checkpoints
        assert "base" in fam["stages"] and "instruct" in fam["stages"]
        assert st["dtype"] == "fp32" and st["batch_size"] >= 1
    assert config.family_settings("gemma4-31b")["offload"] and config.family_settings("gemma4-31b")["gen_precision"] == "int8"
    assert config.family_settings("gemma4-31b")["handoff_extra"] == ["<|channel>", "<channel|>"]
    assert config.family_settings("qwen3-8b")["handoff_extra"] == ["<think>", "</think>"]
    assert config.family_settings("qwen3-8b")["assistant_think_block"] is False  # the pilot's setting
    with pytest.raises(KeyError):
        config.family_settings("nope")


def test_use_family_switches_models_and_paths(restore_config):
    assert config.family_from_argv(["--family", "gemma4-31b", "--model", "instruct"]) == "gemma4-31b"
    assert config.STAGES == ["base", "instruct"] and config.MODELS["instruct"] == "google/gemma-4-31B-it"
    assert config.core_dir("scores") == "results/capstone/gemma4-31b/core/scores"
    assert config.core_dir("scores", "r") == "results/r/gemma4-31b/core/scores"
    assert config.family_from_argv(["--family=qwen3-8b"]) == "qwen3-8b" and config.TEMPLATE_MODEL == "Qwen/Qwen3-8B"
    assert config.family_from_argv(["--model", "x"]) is None


def test_qwen_think_block_option_changes_only_the_assistant_slot(restore_config):
    tok = load_tokenizer("Qwen/Qwen3-8B")
    plain = template_parts(tok, "assistant")
    config.ASSISTANT_THINK_BLOCK = True
    assert template_parts(tok, "assistant")[0] == plain[0] + "<think>\n\n</think>\n\n"
    assert template_parts(tok, "user") == ("<|im_start|>user\n", "<|im_end|>\n")


def test_olmo_rendering_unchanged():
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    h = hashlib.sha256()
    rows = load_jsonl(config.DATA_PATH)
    for pre in (False, True):
        for r in rows:
            for c, v in sorted(render_all(tok, r, pre).items()):
                h.update(json.dumps([r["id"], c, v.input_ids, v.start, v.end, v.text]).encode())
        for slot in ("user", "assistant"):
            h.update(json.dumps(template_parts(tok, slot, pre)).encode())
    assert h.hexdigest() == OLMO_RENDERING_HASH


def test_hf_cache_free_repo_removes_only_unshared_blobs(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_cache, "HUB", str(tmp_path))
    (tmp_path / "blobs" / "aa").mkdir(parents=True)
    (tmp_path / "models--x--big" / "snapshots" / "s1").mkdir(parents=True)
    for sha, refs in (("aa1", ["models--x--big/blobs/aa1"]), ("aa2", ["models--x--big/blobs/aa2", "models--x--other/blobs/aa2"])):
        (tmp_path / "blobs" / "aa" / sha).write_bytes(b"0" * 1000)
        (tmp_path / "blobs" / "aa" / f"{sha}.refs").write_text("\n".join(refs) + "\n")
        (tmp_path / "models--x--big" / "snapshots" / "s1" / f"{sha}.safetensors").symlink_to(tmp_path / "blobs" / "aa" / sha)
    assert hf_cache.is_cached("x/big") and not hf_cache.is_cached("x/other")
    freed, removed, kept = hf_cache.free_repo("x/big", dry_run=True)
    assert (removed, kept) == (1, 1) and (tmp_path / "blobs" / "aa" / "aa1").exists()  # dry run changes nothing
    freed, removed, kept = hf_cache.free_repo("x/big")
    assert (removed, kept) == (1, 1) and abs(freed - 1000 / 1e9) < 1e-9
    assert not (tmp_path / "blobs" / "aa" / "aa1").exists() and (tmp_path / "blobs" / "aa" / "aa2").exists()
    assert (tmp_path / "blobs" / "aa" / "aa2.refs").read_text().strip() == "models--x--other/blobs/aa2"
    assert not (tmp_path / "models--x--big").exists()


def test_run_script_parses():
    subprocess.run(["bash", "-n", str(ROOT / "scripts" / "run_gpu.sh")], check=True)


def test_first_n_per_set():
    from core.score import first_n_per_set
    rows = [dict(id=f"{s}-{k}-{i}", source=s, kind=k) for s, k in [("a", "user"), ("a", "assistant"), ("b", "user")] for i in range(5)]
    out = first_n_per_set(rows, 2)
    assert [r["id"] for r in out] == ["a-user-0", "a-user-1", "a-assistant-0", "a-assistant-1", "b-user-0", "b-user-1"]
    assert first_n_per_set(rows, None) is rows and first_n_per_set(rows, 0) is rows
