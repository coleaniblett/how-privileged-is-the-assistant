"""The pre-registered stage verdict (README criteria 1-5) on synthetic summary frames; no models."""
import pandas as pd

from measures.overgeneralization.persona_carryover import HOLDERS, stage_verdict, verdict_md

DOMS = ["d1", "d2", "d3"]


def frame(spec):
    """spec: {(stage, domain): dict(same_a, same_c, gap, abs)} -> (mean, lo) per quantity; CI half-width 0.1 unless 'lo' given."""
    rows = []
    for (s, d), v in spec.items():
        for q, key in (("carryover assistant -> same", "same_a"), ("carryover character -> same", "same_c"),
                       ("leak gap, change from base", "gap"), ("carryover assistant -> new, change from base", "abs")):
            m = v[key]
            rows.append(dict(stage=s, scope=d, quantity=q, n=12, mean=m, lo=m - v.get("hw", 0.1), hi=m + v.get("hw", 0.1)))
    return pd.DataFrame(rows)


GOOD = dict(same_a=1.0, same_c=1.0, gap=0.8, abs=0.5)


def test_counts_toward_h4_and_first_stage_answer():
    spec = {("sft", d): dict(GOOD, gap=0.1) for d in DOMS} | {("dpo", d): GOOD for d in DOMS} | {("instruct", d): GOOD for d in DOMS}
    dom, st, answer = stage_verdict(frame(spec), ["base", "sft", "dpo", "instruct"], DOMS)
    assert answer == "dpo"                                  # criterion 5: the first stage meeting 3 and 4
    assert st.set_index("stage").loc["sft", "reading"].startswith("no overgeneralization")
    assert st.set_index("stage").loc["instruct", "criterion_4"]


def test_positive_control_failure_is_uninterpretable_not_a_null():
    spec = {("dpo", "d1"): dict(GOOD, same_c=0.1), ("dpo", "d2"): GOOD, ("dpo", "d3"): dict(GOOD, same_a=1.0, hw=1.5)}   # d1 below delta, d3 CI includes 0
    dom, st, answer = stage_verdict(frame(spec), ["base", "dpo"], DOMS)
    d = dom.set_index("domain")
    assert not d.loc["d1", "interpretable"] and not d.loc["d3", "interpretable"] and d.loc["d2", "interpretable"]
    assert st.iloc[0].reading == "too few interpretable domains" and answer is None      # needs >= 2 interpretable domains


def test_majority_rule_and_effect_threshold():
    # 3 interpretable; only 1 meets the 0.5-nat effect -> not a majority
    spec = {("dpo", "d1"): GOOD, ("dpo", "d2"): dict(GOOD, gap=0.49), ("dpo", "d3"): dict(GOOD, gap=0.2)}
    _, st, answer = stage_verdict(frame(spec), ["base", "dpo"], DOMS)
    assert not st.iloc[0].criterion_3 and answer is None
    # 2 of 3 meet it -> majority
    spec = {("dpo", "d1"): GOOD, ("dpo", "d2"): GOOD, ("dpo", "d3"): dict(GOOD, gap=0.2)}
    _, st, answer = stage_verdict(frame(spec), ["base", "dpo"], DOMS)
    assert st.iloc[0].criterion_3 and answer == "dpo"
    # exactly half is not a strict majority (2 interpretable, 1 meets)
    spec = {("dpo", "d1"): GOOD, ("dpo", "d2"): dict(GOOD, gap=0.2), ("dpo", "d3"): dict(GOOD, same_a=0.0)}
    _, st, _ = stage_verdict(frame(spec), ["base", "dpo"], DOMS)
    assert st.iloc[0].interpretable_domains == 2 and not st.iloc[0].criterion_3


def test_relative_only_when_assistant_new_does_not_rise():
    spec = {("dpo", d): dict(GOOD, abs=0.1) for d in DOMS}          # the gap grows only because character -> new falls
    _, st, answer = stage_verdict(frame(spec), ["base", "dpo"], DOMS)
    r = st.iloc[0]
    assert r.criterion_3 and not r.criterion_4 and r.reading.startswith("relative only") and answer is None


def test_absolute_rise_must_hold_in_every_domain_meeting_the_effect():
    spec = {("dpo", "d1"): GOOD, ("dpo", "d2"): GOOD, ("dpo", "d3"): dict(GOOD, abs=0.0)}
    _, st, answer = stage_verdict(frame(spec), ["base", "dpo"], DOMS)
    assert st.iloc[0].criterion_3 and not st.iloc[0].criterion_4 and answer is None
    spec[("dpo", "d3")] = dict(GOOD, gap=0.1, abs=0.0)              # d3 does not meet criterion 2, so its absolute rise is not required
    _, st, answer = stage_verdict(frame(spec), ["base", "dpo"], DOMS)
    assert st.iloc[0].criterion_4 and answer == "dpo"


def test_markdown_renders():
    spec = {("dpo", d): GOOD for d in DOMS}
    md = "\n".join(verdict_md(frame(spec), ["base", "dpo"], DOMS))
    assert "**dpo** is the first stage" in md and "Pre-registered verdict" in md
    assert HOLDERS == ["assistant", "character"]
