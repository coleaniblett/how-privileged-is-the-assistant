"""Overgeneralization minimal pairs: OLMo rendering of the draft items, the verdict logic on synthetic scores,
and a CPU end-to-end run (score -> analyze) with SmolLM2-135M-Instruct."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import config
from core.render import load_tokenizer
from measures.overgeneralization.minimal_pairs import (CONTEXTS, ITEMS_PATH, VARIANTS, load_items, render_item,
                                                       run_analysis)

ROOT = Path(__file__).resolve().parents[1]
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def test_draft_items_render_on_olmo():
    """Same continuation tokens in every context; T and H within the pre-registered 2 tokens."""
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    for it in load_items(ROOT / ITEMS_PATH):
        lengths = {}
        for v in VARIANTS:
            conts = []
            for ctx in CONTEXTS:
                ids, spans = render_item(tok, it, v, ctx)
                start = spans.get("insertion", spans["tail"])[0]
                conts.append(ids[start : spans["tail"][1]])
                assert ids[spans["tail"][1] - 1] == conts[-1][-1]
            assert all(c == conts[0] for c in conts), (it["id"], v)
            lengths[v] = len(conts[0])
        assert abs(lengths["trait"] - lengths["human"]) <= 2, it["id"]


def synthetic(shift_plain, shift_asst, n=6, seed=0):
    """Score frames for base and instruct: 2 traits x 3 genres x n items; instruct adds the given shift to the
    trait variant's logprob in the plain and assistant contexts."""
    rng = np.random.default_rng(seed)
    rows = {"base": [], "instruct": []}
    for t in ("disclaimer", "retraction"):
        for g in ("fiction", "forum", "news"):
            for i in range(n):
                base = {v: -20 + rng.normal(0, 0.3) for v in VARIANTS}
                for ctx in CONTEXTS:
                    for stage in rows:
                        for v, lp in base.items():
                            bump = {"plain": shift_plain, "assistant": shift_asst}.get(ctx, 0) if stage == "instruct" else 0
                            lp = lp + (bump + rng.normal(0, 0.05) if v == "trait" else 0)
                            rows[stage].append(dict(item_id=f"{t}-{g}-{i}", trait=t, genre=g, context=ctx, variant=v,
                                                    n_insertion=0 if v == "plain" else 10, n_tail=8,
                                                    logprob_insertion=0, logprob_tail=0, logprob=lp))
    return {s: pd.DataFrame(r) for s, r in rows.items()}


@pytest.mark.parametrize("plain,asst,expected", [
    (1.0, 1.0, "present"),           # trait rises in plain context and in the control
    (0.0, 1.0, "absent"),            # control works, no leakage
    (1.0, 0.0, "uninterpretable"),   # control fails: not a null
    (-1.0, 1.0, "absent"),           # a significant fall never counts
])
def test_verdict_logic(plain, asst, expected):
    md, summary, verdict = run_analysis(synthetic(plain, asst), n_boot=500)
    assert verdict == {"instruct": expected}
    assert f"| instruct |" in "\n".join(md)


def test_matching_drops_unmatched_items():
    frames = synthetic(1.0, 1.0)
    b = frames["base"]
    b.loc[(b.item_id == "disclaimer-news-0") & (b.variant == "trait"), "logprob"] -= 5
    md, summary, _ = run_analysis(frames, n_boot=200)
    assert "> 2.0 nats: 1 (disclaimer-news-0)" in "\n".join(md)
    assert summary[(summary.quantity == "effect") & (summary.trait == "disclaimer")
                   & (summary.genre == "news")].n.eq(5).all()


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    d = tmp_path_factory.mktemp("og")
    run = lambda *a: subprocess.run([sys.executable, "-m", "measures.overgeneralization.minimal_pairs", *a],
                                    check=True, cwd=ROOT)
    for stage in ("base", "instruct"):
        run("score", "--model", MODEL, "--stage", stage, "--tokenizer", MODEL, "--batch-size", "4",
            "--out", str(d / "scores"))
    # the drafts are matched for OLMo, not SmolLM2: relax the drop rules for the smoke run only
    run("analyze", "--scores", str(d / "scores"), "--out-dir", str(d), "--n-boot", "200",
        "--match-bound", "1e9", "--max-len-diff", "100")
    return d


def test_smoke_end_to_end(smoke):
    n_items = len(load_items(ROOT / ITEMS_PATH))
    for stage in ("base", "instruct"):
        df = pd.read_csv(smoke / "scores" / f"{stage}.csv")
        assert len(df) == n_items * len(CONTEXTS) * len(VARIANTS)
        assert (df.logprob < 0).all()
        assert np.allclose(df.logprob, df.logprob_insertion + df.logprob_tail)
        assert (df[df.variant == "plain"].n_insertion == 0).all()
        # content tokens are identical across contexts
        assert (df.groupby(["item_id", "variant"]).n_tail.nunique() == 1).all()
    md = (smoke / "results.md").read_text()
    # the same model under two labels: every effect is exactly zero, so the control fails everywhere
    assert "| instruct |" in md and "uninterpretable" in md
    s = pd.read_csv(smoke / "summary.csv")
    assert s[s.quantity == "effect"]["mean"].abs().max() < 1e-6
