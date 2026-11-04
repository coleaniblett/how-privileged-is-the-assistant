"""extras/exchange_prefills.py: same rule and seed logic as data/prefill_incomplete_200.jsonl, 200 per corpus from data/exchanges.jsonl."""
import json

from core.prepare_data import build_incomplete_prefill
from extras import exchange_prefills as ep


def test_same_rule_as_the_existing_builder():
    rows = ep.user_rows()
    for corpus in ep.CORPORA:
        ours = ep.build(rows, corpus)
        theirs = build_incomplete_prefill(rows, corpus, ep.N, ep.SEED)
        assert [(o["text_id"], o["prefill"], o["cut_type"]) for o in ours] == [(t["text_id"], t["prefill"], t["cut_type"]) for t in theirs]
        assert len(ours) == ep.N and all(o["id"].startswith(f"incomplete-{corpus}-") for o in ours)


def test_committed_file_matches_a_rebuild_and_cuts_are_unfinished():
    rows = ep.user_rows()
    committed = [json.loads(l) for l in open(ep.OUT, encoding="utf-8")]
    rebuilt = [r for c in ep.CORPORA for r in ep.build(rows, c)]
    assert committed == rebuilt and len(committed) == 400
    by = {r["id"]: r["text"] for r in rows if r["source"] in ep.CORPORA}
    for c in committed:
        assert by[c["text_id"]].startswith(c["prefill"]) and c["remaining_chars"] > 0
        assert not c["prefill"].rstrip().endswith((".", "?", "!"))
