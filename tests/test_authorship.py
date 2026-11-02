"""The authorship-cross text sets for logit distortion (data/texts_authorship.jsonl): oasst1 assistant replies (human-written) and
UltraChat user turns and replies (model-written). Built by core.prepare_data --authorship with the same filters as texts.jsonl;
core scores them in every slot unchanged."""

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from core import config
from core.render import load_jsonl, load_tokenizer, render_all
from core.stats import SOURCE_ORDER, text_sets

ROOT = Path(__file__).resolve().parents[1]
TINY = "HuggingFaceTB/SmolLM2-135M-Instruct"
SETS = {("oasst1", "assistant"): 300, ("ultrachat", "user"): 300, ("ultrachat", "assistant"): 300}


@pytest.fixture(scope="module")
def rows():
    return load_jsonl(config.AUTHORSHIP_PATH)


def test_file_has_the_three_sets_of_300_and_texts_jsonl_is_untouched(rows):
    assert Counter((r["source"], r["kind"]) for r in rows) == Counter(SETS)
    assert len({r["id"] for r in rows}) == len(rows)
    assert {r["id"].split("-")[0] for r in rows} == {"oasst1", "ultrachat"}
    existing = load_jsonl(config.DATA_PATH)
    assert Counter((r["source"], r["kind"]) for r in existing) == Counter(
        {("wildchat", "user"): 300, ("wildchat", "assistant"): 300, ("oasst1", "user"): 300, ("olmo-instruct", "assistant"): 290})
    assert not {r["id"] for r in rows} & {r["id"] for r in existing}
    assert not {r["text"] for r in rows} & {r["text"] for r in existing}  # no text is in both files


def test_rows_carry_the_same_fields_as_texts_jsonl(rows):
    keys = set(load_jsonl(config.DATA_PATH)[0])
    assert all(set(r) == keys for r in rows)
    assert all(config.MIN_TOKENS <= r["n_tokens"] <= config.MAX_TOKENS for r in rows)
    assert {r["dataset"] for r in rows} == {config.DATASET_IDS["oasst1"], config.DATASET_IDS["ultrachat"]}
    assert "ultrachat" in SOURCE_ORDER  # the stats helpers order and name the new source


def test_every_text_renders_in_every_slot_with_identical_content_tokens(rows):
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    for r in rows:
        out = render_all(tok, r)
        assert set(out) == {f"{config.KIND_ABBR[r['kind']]}@{s}" for s, _ in config.SLOTS}
        assert all(v.end - v.start == r["n_tokens"] for v in out.values())


def test_core_scores_the_new_sets_in_every_slot_on_cpu(rows, tmp_path):
    subset = [r for key in SETS for r in [x for x in rows if (x["source"], x["kind"]) == key][:2]]
    data = tmp_path / "subset.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in subset) + "\n")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    subprocess.run([sys.executable, "-m", "core.score", "--model", TINY, "--stage", "tiny", "--tokenizer", TINY, "--dataset", str(data),
                    "--out", str(tmp_path / "scores"), "--batch-size", "4"], check=True, cwd=ROOT, env=env)
    df = pd.read_csv(tmp_path / "scores" / "tiny.csv")
    assert len(df) == len(subset) * 3 and set(zip(df.source, df.kind)) == set(SETS)
    for (src, kind), g in df.groupby(["source", "kind"]):
        ab = config.KIND_ABBR[kind]
        assert set(g.condition) == {f"{ab}@user", f"{ab}@asst", f"{ab}@plain"} and g.mean_nll.notna().all()
    assert set(text_sets({"tiny": df})) == set(SETS)
    assert (tmp_path / "scores" / "tiny_tokens.parquet").exists()
