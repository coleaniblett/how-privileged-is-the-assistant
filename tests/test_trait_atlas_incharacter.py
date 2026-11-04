import gzip, json
import numpy as np, pandas as pd
from extras import trait_atlas as ta, trait_atlas_incharacter as ic


def test_composition_rule():
    assert ic.composition(20, 5, 1, 5.0) == "mostly composition"
    assert ic.composition(20, 15, 3, 5.0) == "survives"
    assert ic.composition(20, 15, -3, 5.0) == "unresolved"
    assert ic.composition(2, 1, 1, 5.0).startswith("n/a")
    assert ic.composition(20, np.nan, np.nan, 5.0).startswith("too few")


def test_prompts_and_notes():
    mat = ta.load_materials()
    p = ic.prompt_for(mat, "Q?", "hello", ic.NOTE_LIGHT.format(role="a ferry captain"))
    assert "a character: a ferry captain" in p and "speak in the first person as that character" in p and "Message:\nQ?" in p

    assert ic.leak_note("a pompous Victorian aristocrat", "named") != ic.leak_note("a pompous Victorian aristocrat", "voiced")


def test_atlas_rows_synthetic(tmp_path, monkeypatch):
    mat = ta.load_materials()
    rng = np.random.default_rng(0)
    rows, flags = [], []
    for stage in ("base", "instruct"):
        for item in range(12):
            for persona in range(6):
                for ctx in ("assistant", "light_system", "strong_act"):
                    base_rate = 0.2
                    # the Assistant rises with training; light roles rise only when in character; strong falls only in composition
                    in_char = float(rng.random() < (0.9 if stage == "instruct" else 0.7)) if ctx != "assistant" else np.nan
                    v = float(rng.random() < (0.9 if stage == "instruct" and (ctx == "assistant" or in_char == 1.0) else base_rate))
                    rows.append(dict(family="qwen3-8b", stage=stage, trait="markdown_lists", item=item, context=ctx, wording=0, persona=persona if ctx != "assistant" else 0, sample=0, flagged=False, value=v))
                    if ctx != "assistant":
                        flags.append(dict(family="qwen3-8b", stage=stage, trait="markdown_lists", item=item, context=ctx, wording=0, persona=persona, sample=0, ic=in_char))
    V = pd.DataFrame(rows)
    F = pd.DataFrame(flags)
    C = ta.analyze_cells(V, mat)
    A = ic.atlas_rows(V, F, C, mat)
    r = A[(A.context == "light_system") & (A.trait == "markdown_lists")].iloc[0]
    assert r.gated and r.change_ic > 30 and r.share_in_character > 0.5
    assert r.reading == "real spread"
    assert (A.context == "strong_act").any() and "composition" in A.columns
