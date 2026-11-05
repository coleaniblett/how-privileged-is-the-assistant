"""The LLM judge with a fake client (no network, no API key): prompt parsing, caching, retries, dry run, what the judge sees,
the validation sheet and the agreement report."""

import csv
import json
from types import SimpleNamespace

import pytest

from measures.vestigial_regression import judge as J


@pytest.fixture(autouse=True)
def v2_prompt(monkeypatch):
    """These tests cover the v2 prompt and its single-label replies (the main run's labels); v3 is in test_judge_v3.py."""
    monkeypatch.setattr(J, "PROMPT_PATH", J.PROMPT_PATHS["v2"])


class FakeClient:
    """Mimics anthropic.Anthropic().messages.create; records every request."""

    def __init__(self, replies=None, errors=()):
        self.calls, self.replies, self.errors = [], list(replies or []), list(errors)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        if self.errors:
            raise self.errors.pop(0)
        text = self.replies.pop(0) if self.replies else '{"label": "user_voice", "reason": "keeps writing as the user."}'
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class ApiError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


def samples(n=3):
    return [dict(prefill=f"user text {i}", continuation=f" more of it {i}", carry=dict(stage="sft", sample=i)) for i in range(n)]


def test_prompt_file_parses_and_contains_the_four_labels():
    version, system, template = J.load_prompt()
    assert version == "v2" and "{prefill}" in template and "{continuation}" in template
    for label in J.LABELS:
        assert label in system
    assert "REAL HUMAN USER" in system and "without turn marker" not in system  # definitions, not run metadata
    # v2: the assistant's reply after the user's turn has closed is context, not a switch; an abandoned unfinished sentence is
    assert "NOT part of what you are judging" in system and "closing the user's turn at once" in system and "label it user_voice" in system
    assert "at any point" not in system  # v1's rule, which counted an ordinary finish-close-reply as regression


def test_parse_reply():
    assert J.parse_reply('{"label": "degenerate", "reason": " loops. "}') == ("degenerate", "loops.")
    assert J.parse_reply('```json\n{"label": "user_voice", "reason": "x"}\n```')[0] == "user_voice"
    for bad in ("no json", '{"label": "other", "reason": "x"}', '{"label": "user_voice"}'):
        with pytest.raises(ValueError):
            J.parse_reply(bad)


def test_hash_depends_on_everything_that_changes_the_answer():
    h = J.sample_hash("p", "c", "v1")
    assert h == J.sample_hash("p", "c", "v1")
    assert len({h, J.sample_hash("p2", "c", "v1"), J.sample_hash("p", "c2", "v1"), J.sample_hash("p", "c", "v2"),
                J.sample_hash("p", "c", "v1", model="other"), J.sample_hash("p", "c", "v1", temperature=1),
                J.sample_hash("p", "c", "v1", thinking={"type": "adaptive"})}) == 7


def test_run_records_model_version_and_never_sends_metadata(tmp_path):
    client = FakeClient()
    rows, est = J.run(samples(), str(tmp_path / "out.jsonl"), str(tmp_path / "cache.jsonl"), client=client, quiet=True)
    assert len(client.calls) == 3 and est["requests"] == 3
    for c in client.calls:
        assert c["model"] == "claude-sonnet-5-5" and "temperature" not in c and c["thinking"] == {"type": "between_tools"}
        sent = json.dumps(c["messages"]) + c["system"]
        assert "sft" not in sent and "olmo" not in sent.lower() and "gemma" not in sent.lower()  # carry fields are not sent
    for r in rows:
        assert r["judge_model"] == "claude-sonnet-5-5" and r["prompt_version"] == "v2" and r["temperature"] is None and r["thinking"] == {"type": "between_tools"}
        assert r["label"] == "user_voice" and r["stage"] == "sft"
    assert [json.loads(line)["label"] for line in open(tmp_path / "out.jsonl")] == ["user_voice"] * 3


def test_cache_means_a_rerun_costs_nothing(tmp_path):
    cache = str(tmp_path / "cache.jsonl")
    first = FakeClient()
    J.run(samples(), None, cache, client=first, quiet=True)
    second = FakeClient()
    rows, est = J.run(samples() + samples(1), None, cache, client=second, quiet=True)
    assert len(first.calls) == 3 and len(second.calls) == 0 and est["requests"] == 0 and len(rows) == 4


def test_duplicate_samples_in_one_run_are_judged_once(tmp_path):
    client = FakeClient()
    J.run(samples(1) * 3, None, str(tmp_path / "c.jsonl"), client=client, quiet=True)
    assert len(client.calls) == 1


def test_dry_run_calls_nothing_and_reports_an_estimate(tmp_path, capsys):
    client = FakeClient()
    rows, est = J.run(samples(), str(tmp_path / "out.jsonl"), str(tmp_path / "c.jsonl"), dry_run=True, client=client,
                      prices=(3.0, 15.0))
    assert rows is None and client.calls == [] and not (tmp_path / "out.jsonl").exists() and not (tmp_path / "c.jsonl").exists()
    assert est["requests"] == 3 and est["input_tokens"] > 0 and est["dollars"] > 0
    assert "$" in capsys.readouterr().out
    _, est2 = J.run(samples(), None, str(tmp_path / "c.jsonl"), dry_run=True, quiet=True)
    assert est2["dollars"] is None  # no prices built in


def test_retries_transient_errors_but_not_client_errors():
    slept = []
    c = FakeClient(errors=[ApiError(429), ApiError(529)])
    assert J.call_with_retries(c, "s", "u", sleep=slept.append).startswith("{") and len(c.calls) == 3 and slept == [2.0, 4.0]
    with pytest.raises(ApiError):
        J.call_with_retries(FakeClient(errors=[ApiError(400)]), "s", "u", sleep=slept.append)
    with pytest.raises(ApiError):
        J.call_with_retries(FakeClient(errors=[ApiError(500)] * 5), "s", "u", retries=3, sleep=lambda s: None)


def test_unparseable_reply_is_asked_for_once_more_then_marked_invalid(tmp_path):
    client = FakeClient(replies=["I think it is fine.", '{"label": "degenerate", "reason": "loop"}'])
    rows, _ = J.run(samples(1), None, str(tmp_path / "c.jsonl"), client=client, quiet=True)
    assert rows[0]["label"] == "degenerate" and len(client.calls) == 2
    client = FakeClient(replies=["no", "still no"])
    rows, _ = J.run(samples(1), None, str(tmp_path / "c2.jsonl"), client=client, quiet=True)
    assert rows[0]["label"] == "invalid"


def test_missing_key_stops_with_a_clear_message(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        J.make_client()


def test_validation_report_math():
    h = ["user_voice"] * 6 + ["switched_to_assistant"] * 4
    j = ["user_voice"] * 5 + ["switched_to_assistant"] + ["switched_to_assistant"] * 3 + ["degenerate"]
    r = J.validation_report(dict(enumerate(h)), dict(enumerate(j)))
    assert r["n"] == 10 and abs(r["agreement"] - 0.8) < 1e-9
    assert r["confusion"]["user_voice"]["user_voice"] == 5 and r["confusion"]["user_voice"]["switched_to_assistant"] == 1
    assert r["confusion"]["switched_to_assistant"]["degenerate"] == 1
    po = 0.8
    pe = (6 * 5 + 4 * 4) / 100  # user_voice: 6 human x 5 judge; switched: 4 x 4; degenerate: 0 x 1
    assert abs(r["kappa"] - (po - pe) / (1 - pe)) < 1e-9
    assert J.cohen_kappa(["a", "b"], ["a", "b"], ["a", "b"]) == 1.0
    assert "Confusion matrix" in J.format_validation(r)


def test_validation_sheet_is_blind_and_stratified(tmp_path):
    sheet, key = J.build_sheet(out_dir=str(tmp_path))
    assert len(sheet) == 60 == len(key) and len({s["sample_id"] for s in sheet}) == 60
    assert set(sheet[0]) == {"sample_id", "prefill", "continuation", "human_label", "notes"} and all(s["human_label"] == "" for s in sheet)
    by = {}
    for k in key:
        by[(k["model"], k["stage"])] = by.get((k["model"], k["stage"]), 0) + 1
    assert by == {("gemma4-31b", "instruct"): 12, ("qwen3-8b", "instruct"): 12, ("olmo3-7b", "base"): 9,
                  ("olmo3-7b", "sft"): 9, ("olmo3-7b", "dpo"): 9, ("olmo3-7b", "instruct"): 9}
    header = open(tmp_path / "sheet.csv", newline="").readline()
    assert "model" not in header and "stage" not in header  # the key is a separate file
    again, _ = J.build_sheet(out_dir=str(tmp_path / "again"))
    assert [s["sample_id"] for s in again] == [s["sample_id"] for s in sheet]  # deterministic


def test_human_labels_are_validated(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("sample_id,prefill,continuation,human_label,notes\na,p,c,user_voice,\nb,p,c,,\nc,p,c,bogus,\n")
    with pytest.raises(SystemExit, match="bogus"):
        J.load_human_labels(str(p))
    p.write_text("sample_id,prefill,continuation,human_label,notes\na,p,c,user_voice,\nb,p,c,,\n")
    assert J.load_human_labels(str(p)) == {"a": "user_voice"}


def test_pilot_items_join_the_agent_labels_to_their_samples():
    items = J.pilot_items()
    assert len(items) == 30 + 60 + 20
    assert all(i["prefill"] and i["continuation"] and i["carry"]["agent_label"] in J.AGENT_TO_BINARY for i in items)
    assert len({i["carry"]["sample_id"] for i in items}) == len(items)


def test_pilot_agreement_with_a_fake_judge(tmp_path):
    items = J.pilot_items()[:6]
    labels = {"user voice": "user_voice", "not user voice": "switched_to_assistant", "degenerate": "degenerate",
              "switched to assistant": "switched_to_assistant"}
    replies = ['{"label": "%s", "reason": "x"}' % labels[i["carry"]["agent_label"]] for i in items]
    rows, _ = J.run(items, None, str(tmp_path / "c.jsonl"), client=FakeClient(replies=replies), quiet=True)
    r = J.pilot_agreement(rows)
    assert r["n"] == 6 and r["binary_agreement"] == 1.0
    rows[0]["label"] = "user_voice" if rows[0]["agent_label"] != "user voice" else "degenerate"
    assert J.pilot_agreement(rows)["binary_agreement"] < 1.0


def test_validation_sheet_from_the_capstone_generations_covers_all_eight_checkpoints(tmp_path):
    import json as js
    for (fam, stage), _ in J.CAPSTONE_PLAN.items():
        d = tmp_path / "results" / "capstone" / fam / "core" / "generations_incomplete"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{stage}.jsonl", "w") as f:
            for i in range(20):
                f.write(js.dumps(dict(prefill_id=f"incomplete-{i:03d}", sample=i % 3, prefill=f"the user was writing {i} and", continuation=f" then more {fam} {stage} {i}", stopped_on_eos=False)) + "\n")
    srcs = J.capstone_sources(root=str(tmp_path))
    sheet, key = J.build_sheet(root=str(tmp_path), out_dir=str(tmp_path / "sheet"), plan={sid: J.CAPSTONE_PLAN[(m, st)] for sid, m, st, _, _ in srcs}, sources=srcs)
    assert len(sheet) == 60 == sum(J.CAPSTONE_PLAN.values()) and len({s["sample_id"] for s in sheet}) == 60
    by = {}
    for k in key:
        by[(k["model"], k["stage"])] = by.get((k["model"], k["stage"]), 0) + 1
    assert by == J.CAPSTONE_PLAN
    assert "model" not in open(tmp_path / "sheet" / "sheet.csv").readline()


def test_stop_reason_recorded_limit_raised_and_invalid_rejudged(tmp_path):
    class StopClient(FakeClient):
        def _create(self, **kw):
            r = super()._create(**kw)
            r.stop_reason = "end_turn"
            r.stop_details = None
            return r
    cache = str(tmp_path / "cache.jsonl")
    bad = StopClient(replies=["", ""])  # empty twice -> invalid
    rows, _ = J.run(samples()[:1], None, cache, client=bad, quiet=True)
    assert rows[0]["label"] == "invalid" and rows[0]["stop_reasons"] == ["end_turn", "end_turn"]
    assert all(c["max_tokens"] == 600 for c in bad.calls)
    again = StopClient()
    rows, est = J.run(samples()[:1], None, cache, client=again, quiet=True)
    assert est["requests"] == 1 and len(again.calls) == 1 and rows[0]["label"] == "user_voice"  # invalid is not served from cache
    third = StopClient()
    J.run(samples()[:1], None, cache, client=third, quiet=True)
    assert third.calls == []


def test_refusal_sheet_joins_dedups_keeps_labels_and_is_blind(tmp_path):
    def write(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    gens = [dict(prefill_id=f"incomplete-{i:03d}", sample=s, prefill=f"p{i}", continuation="same" if i == 0 else f"c{i}{s}")
            for i in range(3) for s in range(2)]
    judged = [dict(prefill_id=g["prefill_id"], sample=g["sample"],
                   label="invalid" if g["prefill_id"] != "incomplete-002" else "user_voice",
                   stop_reasons=["refusal:cyber", "refusal:cyber"]) for g in gens]
    write(tmp_path / "results/capstone/qwen3-8b/core/generations_incomplete/instruct.jsonl", gens)
    write(tmp_path / "results/capstone/qwen3-8b/vestigial_regression/judge_incomplete/instruct.jsonl", judged)
    out = tmp_path / "refusals"
    sheet, key = J.build_refusal_sheet(root=str(tmp_path), out_dir=str(out))
    assert len(key) == 4  # every invalid row, none of the labelled ones
    assert len(sheet) == 3  # incomplete-000's two samples are the same text: one row
    assert all(k["stop_reasons"] == "refusal:cyber;refusal:cyber" for k in key)
    header = open(out / "sheet.csv", encoding="utf-8").readline().strip()
    assert header == "sample_id,prefill,continuation,human_label,notes"  # no model or stage on the sheet
    rows = list(csv.DictReader(open(out / "sheet.csv", newline="", encoding="utf-8")))
    rows[0]["human_label"], rows[0]["notes"] = "degenerate", "loop"
    with open(out / "sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sheet2, _ = J.build_refusal_sheet(root=str(tmp_path), out_dir=str(out))
    kept = {s["sample_id"]: s for s in sheet2}[rows[0]["sample_id"]]
    assert (kept["human_label"], kept["notes"]) == ("degenerate", "loop")


def test_split_and_combine_sheets(tmp_path):
    sheet_dir, split_dir = tmp_path / "v", tmp_path / "v" / "split"
    sheet_dir.mkdir()
    ids = [f"s{i:02d}" for i in range(18)]
    with open(sheet_dir / "sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, J.SHEET_COLS)
        w.writeheader()
        w.writerows([dict(sample_id=s, prefill="p", continuation=f"c{s}", human_label="", notes="") for s in ids])
    with open(sheet_dir / "sheet_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["sample_id", "model", "stage", "source_file", "prefill_id", "sample"])
        w.writeheader()
        w.writerows([dict(sample_id=s, model="m", stage=["base", "instruct"][i % 2], source_file="x", prefill_id=i, sample=0)
                     for i, s in enumerate(ids)])
    people = ["a", "b", "c"]
    shared, own = J.split_sheet(people, 3, out_dir=str(split_dir), sheet_dir=str(sheet_dir))
    assert len(shared) == 3 and all(len(v) == 5 for v in own.values())
    assert len(set(shared) | {s for v in own.values() for s in v}) == 18  # every sample assigned once (shared ones to all)
    for p in people:
        rows = list(csv.DictReader(open(split_dir / f"sheet_{p}.csv", newline="", encoding="utf-8")))
        assert len(rows) == 8 and {r["sample_id"] for r in rows} >= set(shared)
        for r in rows:  # shared[0]: all agree; shared[1]: 2 of 3; shared[2]: no majority
            r["human_label"] = ("user_voice" if r["sample_id"] != shared[1] or p != "c" else "degenerate") if r["sample_id"] != shared[2] \
                else {"a": "user_voice", "b": "degenerate", "c": "switched_to_assistant"}[p]
        with open(split_dir / f"sheet_{p}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, J.SHEET_COLS)
            w.writeheader()
            w.writerows(rows)
    r = J.combine_sheets(split_dir=str(split_dir), sheet_dir=str(sheet_dir))
    assert r["n_labelled"] == 17 and r["no_majority"] == [shared[2]] and r["unanimous"] == 1
    assert abs(r["pairwise_agreement"] - 4 / 9) < 1e-9
    merged = J.load_human_labels(str(sheet_dir / "sheet.csv"))
    assert merged[shared[1]] == "user_voice" and shared[2] not in merged
    with pytest.raises(SystemExit):  # a labelled sheet is never split again
        J.split_sheet(people, 3, out_dir=str(split_dir), sheet_dir=str(sheet_dir))
