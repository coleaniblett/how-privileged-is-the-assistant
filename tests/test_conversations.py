"""Small local fixtures only: no network, model weights, or scoring."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import pytest

from core import config
from core.build_conversations import overrides, save_build
from core.check_contamination import DOLCI, contamination_record, is_contaminated
from core.conversation_audit import audit
from core.conversations import (FORMAT, Settings, apply_duplicates, contributes, flat_conversations,
                                length_sensitivity, make_targets, oasst_branches, select_by_eligible_quota,
                                select_conversations)


class Tokenizer:
    """Transparent fixture lengths; real-tokenizer counts are checked in the pilot."""
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not add_generation_prompt
        assert messages
        tokens = [token for m in messages for token in [m["role"], *m["content"].split(), "end"]]
        # Transformers 5 can return a mapping when tokenize=True: len(mapping) is NOT tokens.
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)} if tokenize else " ".join(tokens)


SETTINGS = Settings(min_target_tokens=2, max_target_tokens=30)


def provenance(source):
    return {**config.CONVERSATION_SOURCES[source], "revision": "fixture-revision"}


def node(mid, parent, text=None, **kwargs):
    return {"message_id": mid, "parent_id": parent, "message_tree_id": "r",
            "role": "prompter" if mid in {"r", "u1", "u2"} else "assistant",
            "text": text or f"original message {mid}", "lang": "en", "review_result": True,
            "deleted": False, "synthetic": False, **kwargs}


@pytest.fixture
def tree():
    return [node("r", None), node("a1", "r"), node("u1", "a1"), node("b1", "u1"),
            node("a2", "r"), node("u2", "a2"), node("b2", "u2")]


def ultra(cid="one", texts=None):
    return {"id": cid, "data": texts or [f"Question {cid}", f"Answer {cid}", f"Followup {cid}", f"Reply {cid}"]}


def wild(cid="w", system=False):
    messages = [{"role": role, "content": f"wildchat {i}", "language": "English", "toxic": False,
                 "redacted": False, "turn_identifier": 100 + i}
                for i, role in enumerate(["user", "assistant", "user", "assistant"])]
    if system:
        messages.insert(0, {"role": "system", "content": "Original system message", "toxic": False, "redacted": False})
    return {"conversation_hash": cid, "model": "source-model", "conversation": messages,
            "language": "English", "toxic": False, "redacted": False}


def adapt(rows, source="ultrachat", settings=SETTINGS):
    return flat_conversations(rows, source, provenance(source), settings)


def test_oasst_branch_keeps_ancestors_and_no_siblings(tree):
    branches = set()
    for seed in range(20):
        settings = replace(SETTINGS, seed=seed)
        cs, rs, n = oasst_branches(tree, provenance("oasst1"), settings)
        assert not rs and n == 1
        mids = tuple(m["source_metadata"]["message_id"] for m in cs[0]["messages"])
        branches.add(mids)
        assert mids in {("r", "a1", "u1", "b1"), ("r", "a2", "u2", "b2")}
        for parent, child in zip(cs[0]["messages"], cs[0]["messages"][1:]):
            assert child["source_metadata"]["parent_id"] == parent["source_metadata"]["message_id"]
        assert oasst_branches(list(reversed(tree)), provenance("oasst1"), settings)[0] == cs
    assert len(branches) == 2


def test_oasst_bad_intermediate_rejects_branch_without_repair(tree):
    cs, _, _ = oasst_branches(tree, provenance("oasst1"), SETTINGS)
    chosen = cs[0]["messages"][1]["source_metadata"]["message_id"]
    next(n for n in tree if n["message_id"] == chosen)["deleted"] = True
    cs, rs, _ = oasst_branches(tree, provenance("oasst1"), SETTINGS)
    assert cs == [] and rs[0]["reason"] == "unreviewed_or_deleted_branch"


@pytest.mark.parametrize("damage,reason", [("orphan", "missing_parent_or_cycle"),
                                          ("cycle", "missing_parent_or_cycle"),
                                          ("duplicate", "malformed_tree_or_duplicate_message_id"),
                                          ("root", "missing_or_multiple_roots")])
def test_malformed_trees_are_explicit(tree, damage, reason):
    if damage == "orphan":
        tree[1]["parent_id"] = "absent"
    elif damage == "cycle":
        tree[1]["parent_id"] = "b1"
    elif damage == "duplicate":
        tree.append(deepcopy(tree[1]))
    else:
        tree = tree[1:]
    cs, rs, _ = oasst_branches(tree, provenance("oasst1"), SETTINGS)
    assert not cs and rs[0]["reason"] == reason


def test_four_groups_and_authorship_basis(tree):
    human, _, _ = oasst_branches(tree, provenance("oasst1"), SETTINGS)
    w, _, _ = adapt([wild()], "wildchat")
    u, _, _ = adapt([ultra()])
    targets = make_targets(human + w + u, Tokenizer(), SETTINGS)
    groups = {(t["source"], t["role"]): t["group"] for t in targets}
    assert groups == {("oasst1", "user"): "human_user", ("oasst1", "assistant"): "human_assistant",
                      ("wildchat", "user"): "human_user", ("wildchat", "assistant"): "model_assistant",
                      ("ultrachat", "user"): "model_user", ("ultrachat", "assistant"): "model_assistant"}
    assert "not proof" in next(t["authorship_basis"] for t in targets if t["source"] == "wildchat" and t["role"] == "user")


@pytest.mark.parametrize("system", [True, False])
def test_context_is_exact_prefix_in_original_order(system):
    cs, _, _ = adapt([wild(system=system)], "wildchat")
    original_text = [m["content"] for m in cs[0]["messages"]]
    ts = make_targets(cs, Tokenizer(), SETTINGS)
    assert len(ts) == 4
    assert ts[-1]["preceding_context_tokens"] == (17 if system else 12)
    assert [m["content"] for m in cs[0]["messages"]] == original_text
    for t in ts:
        assert t["preceding_messages"] == cs[0]["messages"][:t["turn_index"]]
        assert t["target_text"] == original_text[t["turn_index"]]
        assert t["message_id"] not in [m["message_id"] for m in t["preceding_messages"]]
    assert ts[0]["has_preceding_conversation"] is system
    if not system:
        assert ts[0]["preceding_messages"] == [] and ts[0]["preceding_context_tokens"] == 0
    else:
        assert ts[0]["preceding_messages"][0]["role"] == "system"
        assert ts[0]["preceding_messages"][0]["authorship"] == "unknown"


def test_ineligible_intermediate_is_not_removed_or_normalized():
    texts = ["A question", "x", "  Later question\n", "Long enough reply"]
    cs, _, _ = adapt([ultra(texts=texts)])
    ts = make_targets(cs, Tokenizer(), replace(SETTINGS, max_context_tokens=6))
    assert ts[1]["exclusion_reasons"] == ["target_too_short"]
    assert ts[2]["preceding_messages"][1]["content"] == "x"
    assert ts[2]["target_text"] == "  Later question\n"
    assert "context_too_long" in ts[2]["exclusion_reasons"]
    assert [m["content"] for m in ts[3]["preceding_messages"]] == texts[:3]


def test_ids_and_seeded_sampling_independent_of_budget_and_order():
    cs, _, _ = adapt([ultra(str(i)) for i in range(30)])
    small, _ = select_conversations(cs, replace(SETTINGS, conversation_budget=3))
    large, _ = select_conversations(cs, replace(SETTINGS, conversation_budget=7))
    assert {c["conversation_id"] for c in small} < {c["conversation_id"] for c in large}
    assert select_conversations(cs[::-1], replace(SETTINGS, conversation_budget=3))[0] == small
    other, _ = select_conversations(cs, replace(SETTINGS, conversation_budget=3, seed=37))
    assert other != small
    small_targets = make_targets(small, Tokenizer(), SETTINGS)
    large_targets = make_targets(large, Tokenizer(), SETTINGS)
    assert {t["target_id"] for t in small_targets} < {t["target_id"] for t in large_targets}


def test_duplicate_content_ids_and_conflicting_ids():
    cs, _, _ = adapt([ultra("1"), ultra("2", ["Question 1", "Answer   1", "Followup 1", "Reply 1"]), ultra("1")])
    selected, rs = select_conversations(cs, SETTINGS)
    assert len(selected) == 1
    assert {r["reason"] for r in rs} == {"duplicate_conversation_text", "duplicate_conversation_id"}
    conflict = deepcopy(cs[0])
    conflict["messages"][0]["content"] = "conflict"
    selected, rs = select_conversations([cs[0], conflict], SETTINGS)
    assert not selected and len(rs) == 2
    assert all(r["reason"] == "conflicting_conversation_id" for r in rs)


def test_repeated_wording_after_different_history_is_kept_and_flagged():
    """Same words after a different history are a different prediction case: flagged, not excluded."""
    cs, _, _ = adapt([ultra(texts=["Same text", "Reply one", " Same\ntext ", "Reply two"])])
    ts = make_targets(cs, Tokenizer(), SETTINGS)
    assert ts[2]["repeated_text_of"] == ts[0]["target_id"] and ts[2]["duplicate_of"] is None
    assert ts[2]["eligible"] and ts[2]["exclusion_reasons"] == []
    assert ts[3]["preceding_messages"][2]["content"] == " Same\ntext "  # original text kept verbatim


def test_repeated_wording_across_roles_and_sources_is_not_a_duplicate():
    cs, _, _ = adapt([ultra("u", ["Shared words here", "Shared words here", "next question", "final"])])
    wc, _, _ = adapt([wild("w")], "wildchat")
    wc[0]["messages"][0]["content"] = "Shared words here"
    ts = make_targets(cs + wc, Tokenizer(), SETTINGS)
    by_id = {t["message_id"]: t for t in ts}
    user, asst = by_id[cs[0]["messages"][0]["message_id"]], by_id[cs[0]["messages"][1]["message_id"]]
    wild_user = by_id[wc[0]["messages"][0]["message_id"]]
    assert asst["repeated_text_of"] == user["target_id"] and asst["eligible"]           # other role
    assert wild_user["repeated_text_of"] == user["target_id"] and wild_user["eligible"]  # other source, same empty history
    assert not any(t["duplicate_of"] for t in ts)


def test_genuine_duplicate_case_is_excluded_once():
    """Same source, role, authorship, history and target: one representative, others duplicate_case."""
    cs, _, _ = adapt([ultra("1", ["Opening ask", "Reply A", "More", "Reply B"]),
                      ultra("2", ["Opening  ask", "Different reply", "x y", "z w"])])
    ts = make_targets(cs, Tokenizer(), SETTINGS)
    first, second = ts[0], ts[4]
    assert second["exclusion_reasons"] == ["duplicate_case"] and second["duplicate_of"] == first["target_id"]
    assert first["eligible"] and second["target_text"] == "Opening  ask"
    # the replies after the shared opening differ, so they are separate cases and stay eligible
    assert ts[5]["eligible"] and ts[5]["preceding_messages"][0]["content"] == "Opening  ask"


def test_ineligible_first_occurrence_does_not_suppress_eligible_duplicate():
    """Reviewer's case: first occurrence fails on context length; the identical case later must survive."""
    long_history = " ".join(["word"] * 40)
    settings = replace(SETTINGS, max_context_tokens=20)
    # Same opening user turn and same reply, but the first copy's reply follows an overlong opening.
    cs, _, _ = adapt([ultra("a", [long_history, "Same reply text", "q", "r"]),
                      ultra("b", ["Short opening", "Same reply text", "q", "r"])])
    ts = make_targets(cs, Tokenizer(), settings)
    assert "context_too_long" in ts[1]["exclusion_reasons"]
    assert ts[5]["eligible"] and ts[5]["repeated_text_of"] == ts[1]["target_id"]  # different history: kept
    # Now a genuine duplicate group whose FIRST member is ineligible for a non-duplicate reason.
    cs, _, _ = adapt([ultra("c", ["Opening words", "Same reply text", "q", "r"]),
                      ultra("d", ["Opening words", "Same reply text", "q2", "r2"])])
    ts = make_targets(cs, Tokenizer(), settings)
    ts[1]["exclusion_reasons"].append("contamination_target")  # simulate an ineligible first copy
    ts = apply_duplicates(ts)
    assert ts[5]["eligible"] and ts[5]["duplicate_of"] is None
    assert ts[1]["exclusion_reasons"] == ["contamination_target", "duplicate_case"]
    assert ts[1]["duplicate_of"] == ts[5]["target_id"]
    assert apply_duplicates(ts) == ts  # idempotent


@pytest.mark.parametrize("row,reason", [({"id": "empty", "data": []}, "empty_or_malformed_conversation"),
                                       ({"data": ["a", "b", "c", "d"]}, "missing_source_id"),
                                       (ultra(texts=["a", "b"]), "too_few_exchanges"),
                                       (ultra(texts=["a", "b", None, "d"]), "empty_or_nontext_message")])
def test_malformed_flat_conversations(row, reason):
    cs, rs, n = adapt([row])
    assert not cs and n == 1 and rs[0]["reason"] == reason


def test_bad_roles_and_source_quality_reject_whole_conversation():
    row = wild()
    row["conversation"][2]["role"] = "assistant"
    cs, rs, _ = adapt([row], "wildchat")
    assert not cs and rs[0]["reason"] == "invalid_role_order"
    row = wild()
    row["conversation"][1]["redacted"] = True
    cs, rs, _ = adapt([row], "wildchat")
    assert not cs and rs[0]["reason"] == "toxic_redacted_or_unknown_message"


def test_contamination_both_roles_independent_assistant_and_context():
    turns = {"sft": {"user": {"matched user"}, "assistant": {"matched assistant"}, "hash": set()}}
    cs, _, _ = adapt([ultra(texts=["unmatched prompt", "matched  assistant", "matched user", "unmatched reply"])])
    ts = make_targets(cs, Tokenizer(), SETTINGS, turns)
    assert ts[0]["contamination"]["status"] == "incomplete"  # no hit, but not every Dolci set was checked
    for t in ts[1:3]:
        assert "contamination_target" in t["exclusion_reasons"]
        assert t["contamination"]["matches"][0]["training_dataset"] == DOLCI["sft"]
    assert ts[1]["contamination"]["matches"][0]["training_role"] == "assistant"
    assert ts[2]["contamination"]["matches"][0]["training_role"] == "user"
    assert "contamination_context" in ts[3]["exclusion_reasons"]
    assert len(ts[3]["contaminated_context_message_ids"]) == 2
    assert is_contaminated(turns, "matched assistant", role="assistant")  # legacy API
    assert contamination_record(None, "matched user")["status"] == "unchecked"
    assert contamination_record(turns, "MATCHED user")["status"] == "incomplete"  # only sft indexed
    assert [t["context_contamination_status"] for t in ts] == ["no_context", "incomplete", "matched", "matched"]
    full = {name: {"user": set(), "assistant": set(), "hash": set()} for name in DOLCI}
    assert contamination_record(full, "anything")["status"] == "no_exact_match"
    assert contamination_record({**full, "sft": turns["sft"]}, "matched user")["status"] == "matched"


def test_hash_contamination_and_cross_role_match():
    turns = {"rl": {"user": {"assistant wording"}, "assistant": set(), "hash": {"w"}}}
    cs, _, _ = adapt([wild()], "wildchat")
    ts = make_targets(cs, Tokenizer(), SETTINGS, turns)
    assert all(t["contamination"]["matches"][0]["kind"] == "conversation_hash" for t in ts)
    assert contamination_record(turns, "assistant wording")["status"] == "matched"


def test_audit_reconciles_and_saved_hashes_are_reproducible(tmp_path):
    settings = replace(SETTINGS, conversation_budget=2)
    rows = [ultra("1"), ultra("2", ["a", "long enough", "next question", "final reply"]), ultra("3"), ultra("bad", ["a", "b"])]
    cs, rejected, inspected = adapt(rows, settings=settings)
    cs, rs = select_conversations(cs, settings)
    rejected += rs
    ts = make_targets(cs, Tokenizer(), settings)
    info = {"ultrachat": {"inspected_candidates": inspected}}
    report = audit(cs, ts, rejected, info, settings)
    assert report["totals"]["targets"] == sum(c["targets"] for c in report["cells"])
    assert report["totals"]["eligible"] == sum(t["eligible"] for t in ts)
    assert report["totals"]["excluded"] == sum(not t["eligible"] for t in ts)
    assert all(c["opening"] + c["follow_up"] == c["targets"] for c in report["cells"])
    assert all(c["eligible"] + c["excluded"] == c["targets"] for c in report["cells"])
    assert sum(c["contamination_status"].get("unchecked", 0) for c in report["cells"]) == len(ts)
    assert report["sources"]["ultrachat"]["selected_conversations"] + len(rejected) == inspected
    for name in ("a", "b"):
        save_build(tmp_path / name, cs, ts, rejected, report, {"format": FORMAT})
    manifest = json.loads((tmp_path / "a/manifest.json").read_text())
    for file, sha in manifest["output_sha256"].items():
        assert hashlib.sha256((tmp_path / "a" / file).read_bytes()).hexdigest() == sha
        assert (tmp_path / "a" / file).read_bytes() == (tmp_path / "b" / file).read_bytes()
    assert (tmp_path / "a/manifest.json").read_bytes() == (tmp_path / "b/manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        save_build(tmp_path / "a", [], [], [], report, {})


@pytest.mark.parametrize("kwargs", [{"min_exchanges": 1}, {"conversation_budget": 0},
                                    {"min_target_tokens": 5, "max_target_tokens": 2}, {"max_context_tokens": -1}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_cli_source_overrides_are_validated():
    assert overrides(["oasst1=train"], config.CONVERSATION_SOURCES) == {"oasst1": "train"}
    with pytest.raises(ValueError):
        overrides(["wrong=x"], config.CONVERSATION_SOURCES)
    with pytest.raises(ValueError):
        overrides(["oasst1=x", "oasst1=y"], config.CONVERSATION_SOURCES)


def test_offline_audit_verification_detects_modified_context(tmp_path):
    from core.conversation_audit import verify_build
    from core.build_conversations import write_json, write_jsonl
    cs, rs, n = adapt([ultra()])
    ts = make_targets(cs, Tokenizer(), SETTINGS)
    info = {"ultrachat": {"revision": "fixture-revision", "dataset": "fixture", "inspected_candidates": n}}
    report = audit(cs, ts, rs, info, SETTINGS)
    from dataclasses import asdict
    out = tmp_path / "build"
    save_build(out, cs, ts, rs, report, {"format": FORMAT, "settings": asdict(SETTINGS), "sources": info})
    assert verify_build(out)["totals"] == report["totals"]
    ts[0]["preceding_messages"] = [cs[0]["messages"][-1]]
    write_jsonl(out / "targets.jsonl", ts)
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_build(out)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["output_sha256"]["targets.jsonl"] = hashlib.sha256((out / "targets.jsonl").read_bytes()).hexdigest()
    write_json(out / "manifest.json", manifest)
    with pytest.raises(ValueError, match="target invariant"):
        verify_build(out)


def test_empty_contamination_index_is_unchecked():
    assert contamination_record({}, "text")["status"] == "unchecked"


def test_dolci_loader_pins_revisions_and_indexes_both_roles(monkeypatch):
    import core.check_contamination as contamination
    batches = {
        DOLCI["sft"]: {"messages": [[{"role": "user", "content": "sft user"},
                                    {"role": "assistant", "content": "sft assistant"}]]},
        DOLCI["dpo"]: {"chosen": [[{"role": "assistant", "content": "chosen assistant"}]],
                      "rejected": [[{"role": "assistant", "content": "rejected assistant"}]]},
        DOLCI["rl"]: {"prompt": ["user: rl user\nassistant: rl assistant"], "conversation_hash": ["wild-hash"]},
    }
    calls = []
    class Dataset:
        def __init__(self, data):
            self.data = data
        def select_columns(self, columns):
            assert set(columns) == set(self.data)
            return self
        def iter(self, batch_size):
            yield self.data
    def load(name, split, revision):
        calls.append((name, split, revision))
        return Dataset(batches[name])
    monkeypatch.setattr(contamination, "load_dataset", load)
    index = contamination.dolci_turns(revisions={n: f"{n}-sha" for n in DOLCI})
    assert calls == [(dataset, "train", f"{stage}-sha") for stage, dataset in DOLCI.items()]
    for text in ("sft user", "sft assistant", "chosen assistant", "rejected assistant", "rl user", "rl assistant"):
        assert contamination_record(index, text)["status"] == "matched"
    assert contamination_record(index, "new wording", "wild-hash")["matches"][0]["kind"] == "conversation_hash"
    calls.clear()
    contamination.dolci_turns(names=("sft",))  # unchanged legacy call has no revision override
    assert calls == [(DOLCI["sft"], "train", None)]


def test_core_score_refuses_conversation_candidates():
    from core.score import reject_conversation_candidates
    cs, _, _ = adapt([ultra()])
    ts = make_targets(cs, Tokenizer(), SETTINGS)
    for records in (cs, ts):
        with pytest.raises(SystemExit, match="not core.score input"):
            reject_conversation_candidates(records, "candidates.jsonl")
    legacy = [{"id": "x", "source": "wildchat", "kind": "user", "text": "hello"}]
    reject_conversation_candidates(legacy, config.DATA_PATH)  # legacy isolated texts pass unchanged


def test_target_and_history_contamination_recorded_separately():
    """A clean target after a contaminated history: target status stays its own; history flags it."""
    full = {name: {"user": set(), "assistant": set(), "hash": set()} for name in DOLCI}
    full["dpo"]["assistant"].add("matched answer")
    cs, _, _ = adapt([ultra(texts=["first question", "matched answer", "second question", "final reply"])])
    ts = make_targets(cs, Tokenizer(), SETTINGS, full)
    assert [t["contamination"]["status"] for t in ts] == ["no_exact_match", "matched", "no_exact_match", "no_exact_match"]
    assert [t["context_contamination_status"] for t in ts] == ["no_context", "no_exact_match", "matched", "matched"]
    assert ts[2]["contaminated_context_message_ids"] == [cs[0]["messages"][1]["message_id"]]
    assert ts[2]["exclusion_reasons"] == ["contamination_context"]
    # The matched message is never removed: later histories still contain it verbatim.
    assert ts[3]["preceding_messages"][1]["content"] == "matched answer"
    assert all(m["contamination"]["checked_datasets"] == sorted(DOLCI.values()) for m in cs[0]["messages"])
    assert make_targets(adapt([ultra()])[0], Tokenizer(), SETTINGS)[1]["context_contamination_status"] == "unchecked"


@pytest.mark.parametrize("module", ["core.render", "core.check_contamination", "core.add_replies"])
def test_isolated_entry_points_refuse_candidates(module, tmp_path, monkeypatch):
    """Every CLI that reads isolated texts via --dataset fails explicitly, before loading models or Dolci."""
    import importlib, runpy, sys
    import core.check_contamination as contamination
    monkeypatch.setattr(contamination, "dolci_turns", lambda *a, **k: pytest.fail("Dolci loaded"))
    cs, _, _ = adapt([ultra()])
    path = tmp_path / "targets.jsonl"
    path.write_text("\n".join(json.dumps(t) for t in make_targets(cs, Tokenizer(), SETTINGS)) + "\n")
    if module == "core.add_replies":
        importlib.import_module(module)
        monkeypatch.setattr(sys.modules[module], "load_model", lambda *a: pytest.fail("model loaded"))
        monkeypatch.setattr(sys.modules[module], "load_tokenizer", lambda *a: None)
        monkeypatch.setattr(sys, "argv", [module, "--dataset", str(path)])
        with pytest.raises(SystemExit, match="not core.score input"):
            sys.modules[module].main()
        return
    if module == "core.render":
        monkeypatch.setattr("core.render.AutoTokenizer.from_pretrained", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", [module, "--dataset", str(path), "--out", str(tmp_path / "x.md")]
                        if module == "core.render" else [module, "--dataset", str(path)])
    with pytest.raises(SystemExit, match="not core.score input"):
        runpy.run_module(module, run_name="__main__")


def full_index(**hits):
    """Complete Dolci index (all three sets); hits: {"sft_user": {...}} style additions."""
    index = {name: {"user": set(), "assistant": set(), "hash": set()} for name in DOLCI}
    for key, texts in hits.items():
        name, role = key.split("_")
        index[name][role] |= set(texts)
    return index


QUOTA = replace(SETTINGS, selection_mode="eligible_quota", contributing_quota=2, scan_limit=3)


def quota_rows():
    # "c" has only a matched opening, so every later target fails the history rule: never contributes.
    return [ultra("a"), ultra("b", ["tiny", "x", "y", "z"]), ultra("c", ["Matched opening", "Reply c", "More c", "Fin c"]),
            ultra("d"), ultra("e"), ultra("f")]


def test_eligible_quota_keeps_only_contributing_until_quota():
    cs, rejected, _ = adapt(quota_rows(), settings=QUOTA)
    index = full_index(sft_user={"Matched opening"})
    selected, targets, rs, scan = select_by_eligible_quota(cs, QUOTA, Tokenizer(), index)
    assert len(selected) == 2 and scan["ultrachat"]["contributing"] == 2
    assert all(contributes([t for t in targets if t["conversation_id"] == c["conversation_id"]]) for c in selected)
    reasons = Counter(r["reason"] for r in rs)
    assert reasons["contributing_quota_reached"] + reasons["no_contributing_target"] + reasons["scan_limit"] \
        + len(selected) == len(cs)
    assert "c" not in {c["source_id"] for c in selected}
    # same seeded rank order as raw_budget: kept conversations are a subsequence of that order
    raw, _ = select_conversations(cs, replace(QUOTA, conversation_budget=len(cs)))
    from core.conversations import priority
    rank = sorted(c["conversation_id"] for c in raw)
    rank = sorted(rank, key=lambda cid: (priority(QUOTA.seed, "sample", cid), cid))
    assert [c["conversation_id"] for c in selected] == [cid for cid in rank if cid in {c["conversation_id"] for c in selected}]


def test_eligible_quota_scan_limit_exhaustion_reports_shortage():
    settings = replace(QUOTA, contributing_quota=5, scan_limit=2)
    cs, _, n = adapt(quota_rows(), settings=settings)
    selected, targets, rs, scan = select_by_eligible_quota(cs, settings, Tokenizer(), full_index())
    assert scan["ultrachat"]["evaluated"] == 2 and len(selected) <= 2
    assert sum(r["reason"] == "scan_limit" for r in rs) == len(cs) - 2
    report = audit(selected, targets, rs, {"ultrachat": {"inspected_candidates": n}}, settings, version=2)
    assert report["sources"]["ultrachat"]["contributing_quota_shortfall"] == 5 - len(selected)


@pytest.mark.parametrize("index", [None, {}, {"sft": {"user": set(), "assistant": set(), "hash": set()}}])
def test_eligible_quota_refuses_unchecked_or_incomplete_contamination(index):
    cs, _, _ = adapt(quota_rows(), settings=QUOTA)
    with pytest.raises(ValueError, match="complete Dolci index"):
        select_by_eligible_quota(cs, QUOTA, Tokenizer(), index)


def test_unchecked_or_incomplete_targets_never_count_as_contributing():
    cs, _, _ = adapt([ultra()])
    unchecked = make_targets(cs, Tokenizer(), SETTINGS)
    partial = make_targets(cs, Tokenizer(), SETTINGS, {"sft": full_index()["sft"]})
    clear = make_targets(cs, Tokenizer(), SETTINGS, full_index())
    assert all(t["eligible"] for t in unchecked + partial)  # length-eligible, but not cleared
    assert not contributes(unchecked) and not contributes(partial) and contributes(clear)


def test_length_sensitivity_changes_only_minimum_length():
    cs, _, _ = adapt([ultra("1", ["one two", "a b c d e", "q", "long enough reply here"])])
    ts = make_targets(cs, Tokenizer(), replace(SETTINGS, min_target_tokens=3), full_index(dpo_assistant={"a b c d e"}))
    rows = length_sensitivity(ts, replace(SETTINGS, min_target_tokens=3), [1, 3])
    get = {(r["min_target_tokens"], r["group"], r["turn_position"]): r for r in rows}
    assert get[(3, "model_user", "opening")]["eligible"] == 0 and get[(1, "model_user", "opening")]["eligible"] == 1
    # the contaminated assistant reply stays excluded at every threshold, and so does everything after it
    assert get[(1, "model_assistant", "opening")]["eligible"] == 0
    assert get[(1, "model_user", "follow_up")]["eligible"] == 0
    assert sum(r["targets"] for r in rows if r["min_target_tokens"] == 1) == len(ts)


def test_v2_build_verifies_offline_and_detects_tampered_duplicate_marks(tmp_path):
    from dataclasses import asdict
    from core.build_conversations import write_jsonl
    from core.conversation_audit import AUDIT_VERSION, verify_build
    from core.conversations import output_order
    cs, rejected, n = adapt(quota_rows(), settings=QUOTA)
    selected, targets, rs, _ = select_by_eligible_quota(cs, QUOTA, Tokenizer(), full_index(sft_user={"Matched opening"}))
    order = [c["conversation_id"] for c in selected]
    sens = length_sensitivity(targets, QUOTA, [1, 2])
    selected, targets = output_order(selected, targets)
    info = {"ultrachat": {"dataset": "fixture", "revision": "fixture-revision", "inspected_candidates": n}}
    report = audit(selected, targets, rejected + rs, info, QUOTA, AUDIT_VERSION)
    manifest = {"format": FORMAT, "settings": asdict(QUOTA), "sources": info, "audit_version": AUDIT_VERSION,
                "selection_order": order, "sensitivity_min_target_tokens": [1, 2]}
    out = tmp_path / "v2"
    save_build(out, selected, targets, rejected + rs, report, manifest, sens)
    assert verify_build(out)["totals"] == report["totals"]
    assert report["sources"]["ultrachat"]["contributing_conversations"] == len(selected)
    assert (out / "length_sensitivity.md").exists()
    targets[0]["exclusion_reasons"] = ["duplicate_case"]
    targets[0]["eligible"] = False
    write_jsonl(out / "targets.jsonl", targets)
    m = json.loads((out / "manifest.json").read_text())
    m["output_sha256"]["targets.jsonl"] = hashlib.sha256((out / "targets.jsonl").read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="do not recompute"):
        verify_build(out)


@pytest.mark.parametrize("pilot", ["pilot_v1", "pilot_v2"])
def test_existing_pilots_still_verify(pilot):
    from core.conversation_audit import verify_build
    assert verify_build(f"candidates/logit_distortion/{pilot}")["totals"]["targets"] == 184
