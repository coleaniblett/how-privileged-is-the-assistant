"""Judge prompt v3 (judge_prompt.md): the reply parser, the fixed label rule (derive_v3), the headline groups and v2 counterparts,
and a run with a fake client (no network, no API key). The examples are invented."""

import json
from types import SimpleNamespace

import pytest

from measures.vestigial_regression import judge as J

END = ("<|im_end|>",)


def fields(**kw):
    base = dict(user_text="user", completed=True, assistant_content=False, assistant_start="", assistant_traits=False,
                repetitive=False, reason="x")
    return {**base, **kw}


class FakeClient:
    def __init__(self, replies):
        self.calls, self.replies = [], list(replies)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.replies.pop(0))])


def test_v3_is_the_default_prompt_and_v2_stays_available():
    v3, system, template = J.load_prompt()
    assert v3 == "v3" and all(k in system for k in ("user_text", "completed", "assistant_content", "assistant_start", "repetitive"))
    assert "{prefill}" in template and "{continuation}" in template
    v2, system2, _ = J.load_prompt(J.PROMPT_PATHS["v2"])
    assert v2 == "v2" and all(lab in system2 for lab in J.LABELS)
    assert J.sample_hash("p", "c", v3) != J.sample_hash("p", "c", v2)  # the versions never share cache entries


def test_parse_reply_v3():
    good = json.dumps(fields(assistant_content=True, assistant_start="Sure, here", reason=" ok. "))
    out = J.parse_reply_v3("Here you go:\n```json\n" + good + "\n```")
    assert out["user_text"] == "user" and out["assistant_start"] == "Sure, here" and out["reason"] == "ok."
    assert J.parse_reply_v3(json.dumps(fields(assistant_start="stray")))["assistant_start"] == ""  # no assistant content: dropped
    for bad in ("no json", json.dumps(fields(user_text="assistant")), json.dumps(fields(completed="yes")),
                json.dumps({k: v for k, v in fields().items() if k != "reason"})):
        with pytest.raises(ValueError):
            J.parse_reply_v3(bad)


@pytest.mark.parametrize("f, cont, loop, want", [
    (fields(), " never fills the tank.", False, ("user_voice", "")),
    (fields(assistant_traits=True), " never fills the tank. Certainly, here are my notes:", False, ("user_voice_with_assistant_traits", "")),
    (fields(assistant_content=True, assistant_start="That sounds frustrating"),
     " never fills the tank.<|im_end|><|im_start|>assistant\nThat sounds frustrating.", False, ("completed_then_assistant", "closed")),
    (fields(assistant_content=True, assistant_start="Here are some ways"),
     " never fills the tank. Here are some ways to approach it: 1.", False, ("completed_then_assistant", "unclosed")),
    (fields(completed=False, assistant_content=True, assistant_start="I can help"), " I can help you with that!", False,
     ("switched_to_assistant", "abandoned")),
    (fields(user_text="none", completed=False, assistant_content=True, assistant_start="I'm sorry to hear"),
     "<|im_end|><|im_start|>assistant\nI'm sorry to hear that.", False, ("closed_immediately", "")),
    (fields(user_text="none", completed=False, assistant_content=True, assistant_start="Okay, so the user"),
     " Okay, so the user wants help.", False, ("switched_to_assistant", "immediate")),
    (fields(user_text="none", completed=False), " ", False, ("unclear", "no_user_text_no_assistant")),
    (fields(user_text="degenerate"), " own own own own own own own own own own own", True, ("degenerate", "token_loop")),
    (fields(user_text="degenerate"), " table seven blue the of when nine", False, ("degenerate", "incoherent")),
    (fields(assistant_content=True, assistant_start="not in the text at all"), " never fills the tank. Something.", False,
     ("completed_then_assistant", "unknown")),
])
def test_derive_v3_rule(f, cont, loop, want):
    label, subtype, _ = J.derive_v3(f, cont, END, loop)
    assert (label, subtype) == want


def test_completed_counts_only_with_user_text_and_plain_role_labels_do_not_close_the_turn():
    f = fields(user_text="none", completed=True, assistant_content=True, assistant_start="Hello")
    assert J.derive_v3(f, " Hello there", END)[:2] == ("switched_to_assistant", "immediate")
    f = fields(assistant_content=True, assistant_start="Sure, I can")
    label, subtype, extra = J.derive_v3(f, " never fills the tank.\n\nAssistant: Sure, I can help.", END)
    assert (label, subtype, extra["plain_role_label"], extra["start_found"]) == ("completed_then_assistant", "unclosed", True, True)


def test_find_start_tolerates_whitespace_differences():
    text = "my car.\n\nHere  are some\nways to approach it"
    assert J.find_start(text, "Here are some ways") == text.index("Here")
    assert J.find_start(text, "") == -1


def test_groups_and_v2_counterparts_follow_the_2026_11_05_decisions():
    assert J.v3_group("completed_then_assistant", "incomplete") == "completed_then_assistant"
    assert J.v3_group("closed_immediately", "incomplete") == "non_user_voice"
    assert J.v3_group("closed_immediately", "open") == "closed_immediately"
    assert J.v3_group("user_voice_with_assistant_traits", "open") == "user_voice"
    assert J.v3_group("degenerate", "open") == "non_user_voice"
    assert J.v3_to_v2("completed_then_assistant", "closed", "incomplete") == "user_voice"
    assert J.v3_to_v2("completed_then_assistant", "unclosed", "incomplete") == "switched_to_assistant"
    assert J.v3_to_v2("completed_then_assistant", "unknown", "incomplete") is None
    assert J.v3_to_v2("closed_immediately", "", "open") == "user_voice"
    assert J.condition_of("open-3") == "open" and J.condition_of("incomplete-012") == "incomplete" and J.condition_of(None) == "incomplete"


def test_run_v3_derives_labels_caches_fields_and_never_sends_metadata(tmp_path):
    gen = tmp_path / "results" / "capstone" / "nofamily" / "core" / "generations_open5"
    gen.mkdir(parents=True)
    rows = [dict(prefill_id="open-0", sample=0, prefill="I have a problem.", continuation=" My dog barks all night.", stopped_on_eos=False,
                 continuation_ids=[1, 2, 3]),
            dict(prefill_id="open-0", sample=1, prefill="I have a problem.", continuation="<|im_end|><|im_start|>assistant\nWhat's wrong?",
                 stopped_on_eos=True, continuation_ids=[4])]
    path = gen / "instruct.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    replies = [json.dumps(fields()), json.dumps(fields(user_text="none", completed=False, assistant_content=True, assistant_start="What's wrong?"))]
    client = FakeClient(replies)
    out, cache = tmp_path / "out.jsonl", tmp_path / "cache.jsonl"
    got, est = J.run(J.read_samples([str(path)]), str(out), str(cache), client=client, quiet=True)
    assert [(r["label"], r["group"], r["condition"]) for r in got] == [("user_voice", "user_voice", "open"),
                                                                       ("closed_immediately", "closed_immediately", "open")]
    assert got[0]["prompt_version"] == "v3" and got[0]["markers"] == "generic" and got[1]["v2_equivalent"] == "user_voice"
    sent = json.dumps(client.calls)
    assert "instruct.jsonl" not in sent and "nofamily" not in sent and "stopped_on_eos" not in sent and "continuation_ids" not in sent
    again = FakeClient([])
    got2, est2 = J.run(J.read_samples([str(path)]), None, str(cache), client=again, quiet=True)
    assert est2["requests"] == 0 and not again.calls and [r["label"] for r in got2] == [r["label"] for r in got]


def test_family_markers_come_from_the_path_and_close_the_turn(tmp_path):
    gen = tmp_path / "results" / "capstone" / "olmo3-7b" / "core" / "generations_open5"
    gen.mkdir(parents=True)
    cont = " My dog barks.<|im_end|>\n<|im_start|>assistant\nThat sounds hard."
    (gen / "sft.jsonl").write_text(json.dumps(dict(prefill_id="open-0", sample=0, prefill="I have a problem.", continuation=cont)) + "\n")
    samples = J.read_samples([str(gen / "sft.jsonl")])
    assert samples[0]["family"] == "olmo3-7b"
    client = FakeClient([json.dumps(fields(assistant_content=True, assistant_start="That sounds hard."))])
    got, _ = J.run(samples, None, str(tmp_path / "c.jsonl"), client=client, quiet=True)
    assert (got[0]["label"], got[0]["subtype"], got[0]["markers"]) == ("completed_then_assistant", "closed", "family")


def test_invalid_v3_reply_is_kept_out_of_the_groups(tmp_path):
    s = [dict(prefill="I have a problem.", continuation=" x", carry=dict(prefill_id="open-1"))]
    got, _ = J.run(s, None, str(tmp_path / "c.jsonl"), client=FakeClient(["nope", "still nope"]), quiet=True)
    assert (got[0]["label"], got[0]["group"], got[0]["v2_equivalent"]) == ("invalid", "invalid", "invalid")


def test_validation_uses_the_v2_counterpart_of_v3_rows():
    rows = [dict(sample_id="a", prompt_version="v3", label="completed_then_assistant", v2_equivalent="user_voice"),
            dict(sample_id="b", prompt_version="v3", label="unclear", v2_equivalent=None),
            dict(sample_id="c", prompt_version="v2", label="degenerate")]
    judged, left = J.judged_for_validation(rows)
    assert judged == {"a": "user_voice", "c": "degenerate"} and left == 1
