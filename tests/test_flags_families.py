"""Generation flags across families: derived turn markers, extra handoff markers (Gemma thought channel, Qwen3 think tags),
the no-space token_loop flag, and flag_samples with and without stored token ids. Short synthetic strings, tokenizers only."""

import pytest

from core import config
from core.render import family_turn_markers, load_tokenizer
from measures.vestigial_regression.flags import flag_samples, flags, markers_for, max_run, user_portion

FAMILY_TOK = {"olmo3-7b": "allenai/Olmo-3-7B-Instruct", "gemma4-31b": "google/gemma-4-31B-it", "qwen3-8b": "Qwen/Qwen3-8B"}
MARKERS = {
    "olmo3-7b": dict(end_of_turn="<|im_end|>", turn_open="<|im_start|>", assistant_header="<|im_start|>assistant", handoff="<|im_end|>\n<|im_start|>assistant"),
    "qwen3-8b": dict(end_of_turn="<|im_end|>", turn_open="<|im_start|>", assistant_header="<|im_start|>assistant", handoff="<|im_end|>\n<|im_start|>assistant"),
    "gemma4-31b": dict(end_of_turn="<turn|>", turn_open="<|turn>", assistant_header="<|turn>model", handoff="<turn|>\n<|turn>model"),
}


@pytest.fixture(scope="module", params=list(FAMILY_TOK))
def fam(request):
    return request.param, load_tokenizer(FAMILY_TOK[request.param])


def run(fam_name, tok, text, ids=None, stopped=False):
    m = markers_for(tok)
    ut = user_portion(text, tok)
    return ut, flags(text, ut, stopped, m["end_of_turn"], m["handoff"], config.family_settings(fam_name)["handoff_extra"],
                     ids if ids is not None else tok(ut, add_special_tokens=False)["input_ids"])


def test_markers_are_derived_from_the_template(fam):
    name, tok = fam
    m = family_turn_markers(tok)
    for k, v in MARKERS[name].items():
        assert m[k] == v, (name, k)


def test_turn_marker_handoff(fam):
    name, tok = fam
    m = MARKERS[name]
    ut, f = run(name, tok, f" can you help me?{m['end_of_turn']}\n{m['assistant_header']}\nSure thing.")
    assert ut == " can you help me?" and f["ended_turn"] and f["handoff_to_assistant"]
    assert f["token_loop"] is False and f["repetition_loop"] is False


def test_user_voice_that_stays_open_is_not_a_handoff(fam):
    name, tok = fam
    ut, f = run(name, tok, " and then she walked home, thinking about the river.")
    assert not f["ended_turn"] and not f["handoff_to_assistant"]


def test_gemma_thought_channel_is_a_handoff_without_turn_markers():
    tok = load_tokenizer(FAMILY_TOK["gemma4-31b"])
    text = "<|channel>thought\n<channel|>The answer is a **Green Apple**."
    ut, f = run("gemma4-31b", tok, text)
    assert f["handoff_to_assistant"] and not f["ended_turn"]
    assert ut == text  # extra markers flag a handoff but do not shorten the user portion
    _, f = run("olmo3-7b", load_tokenizer(FAMILY_TOK["olmo3-7b"]), text)
    assert not f["handoff_to_assistant"]  # OLMo has no extra markers


def test_qwen_think_tags_are_a_handoff_without_turn_markers():
    tok = load_tokenizer(FAMILY_TOK["qwen3-8b"])
    _, f = run("qwen3-8b", tok, " If so, how?\n\n</think>\n\nAI models do not learn.")
    assert f["handoff_to_assistant"] and not f["ended_turn"]
    _, f = run("qwen3-8b", tok, "<think>\nOkay, the user wants a poem.")
    assert f["handoff_to_assistant"]
    _, f = run("gemma4-31b", load_tokenizer(FAMILY_TOK["gemma4-31b"]), "<think>")
    assert not f["handoff_to_assistant"]


def test_loops(fam):
    name, tok = fam
    _, f = run(name, tok, "!!!!!!!!!!!!", ids=[5] * 12)  # a no-space loop only the id run catches
    assert f["token_loop"] and not f["repetition_loop"]
    _, f = run(name, tok, " a b c d a b c d a b c d", ids=list(range(3, 15)))
    assert f["repetition_loop"] and not f["token_loop"]
    assert max_run([]) == 0 and max_run([1, 1, 2, 2, 2]) == 3


def test_flag_samples_uses_stored_ids_and_falls_back_to_retokenizing(fam):
    name, tok = fam
    eos_id = tok.eos_token_id
    base = dict(prefill_id="p", sample=0, prefill="x", stopped_on_eos=False)
    with_ids = dict(base, continuation="!!!!!!!!!!!!", continuation_ids=[5] * 12)
    without = dict(base, continuation="hello there " * 3)
    recs, totals = flag_samples([with_ids, without], tok, name)
    assert recs[0]["token_loop"] is True and recs[1]["token_loop"] is False
    assert totals["n"] == 2 and totals["token_loop"] == 1
    # a stop token inside the stored ids ends the user portion for the id run
    capped = dict(base, continuation="ab", continuation_ids=[5] * 4 + [eos_id] + [5] * 12)
    assert flag_samples([capped], tok, name)[0][0]["token_loop"] is False
