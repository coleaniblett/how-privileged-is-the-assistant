"""Audits for candidate construction, independent of model outcomes."""

from collections import Counter
import json
import statistics

from core.conversations import GROUPS


def distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0}

    def quantile(p):
        index = (len(values) - 1) * p
        lo = int(index)
        hi = min(lo + 1, len(values) - 1)
        return round(values[lo] + (values[hi] - values[lo]) * (index - lo), 2)

    return {"n": len(values), "min": values[0], "p25": quantile(.25),
            "median": statistics.median(values), "p75": quantile(.75),
            "p95": quantile(.95), "max": values[-1]}


AUDIT_VERSION = 2  # 1: pilot_v1/v2 (recomputed exactly as built); 2 adds coverage and selection fields


def contributing(ts):
    return len({t["conversation_id"] for t in ts if t["eligible"]})


def audit(conversations, targets, rejected, source_info, settings, version=1):
    cells, sources = [], {}
    for source in sorted(source_info):
        cs = [c for c in conversations if c["source"] == source]
        ts = [t for t in targets if t["source"] == source]
        rs = [r for r in rejected if r["source"] == source]
        reasons = Counter(r["reason"] for r in rs)
        inspected = source_info[source]["inspected_candidates"]
        assert inspected == len(cs) + len(rs), (source, inspected, len(cs), len(rs))
        sources[source] = {
            **{k: v for k, v in source_info[source].items() if k != "features"}, "selected_conversations": len(cs), "targets": len(ts),
            "eligible_targets": sum(t["eligible"] for t in ts),
            "conversation_budget_shortfall": max(0, settings.conversation_budget - len(cs)),
            "conversation_exclusions": dict(sorted(reasons.items())),
            "duplicate_conversations": sum(n for r, n in reasons.items() if r.startswith("duplicate")),
            "exchanges_per_conversation": distribution([c["complete_exchanges"] for c in cs]),
            "exchanges_histogram": dict(sorted(Counter(c["complete_exchanges"] for c in cs).items())),
            "system_messages": sum(m["role"] == "system" for c in cs for m in c["messages"]),
        }
        if version >= 2:
            user = {t["conversation_id"] for t in ts if t["eligible"] and t["role"] == "user"}
            asst = {t["conversation_id"] for t in ts if t["eligible"] and t["role"] == "assistant"}
            sources[source].update({
                "selection_mode": settings.selection_mode,
                "contributing_conversations": contributing(ts),
                "contributing_user_conversations": len(user),
                "contributing_assistant_conversations": len(asst),
                "contributing_both_roles_conversations": len(user & asst),
                "repeated_text_targets": sum(t.get("repeated_text_of") is not None for t in ts)})
            if settings.selection_mode == "eligible_quota":
                sources[source].update({
                    "contributing_quota": settings.contributing_quota, "scan_limit": settings.scan_limit,
                    "contributing_quota_shortfall": max(0, settings.contributing_quota - len(cs))})
                del sources[source]["conversation_budget_shortfall"]
        for group in GROUPS:
            rows = [t for t in ts if t["group"] == group]
            author, role = group.split("_", 1)
            eligible = [t for t in rows if t["eligible"]]
            cells.append({
                "source": source, "group": group, "role": role, "authorship": author,
                "conversations": len({t["conversation_id"] for t in rows}),
                "targets": len(rows), "eligible": len(eligible), "excluded": len(rows) - len(eligible),
                "opening": sum(t["turn_position"] == "opening" for t in rows),
                "follow_up": sum(t["turn_position"] == "follow_up" for t in rows),
                "eligible_opening": sum(t["turn_position"] == "opening" for t in eligible),
                "eligible_follow_up": sum(t["turn_position"] == "follow_up" for t in eligible),
                "target_tokens": distribution([t["target_tokens"] for t in rows]),
                "preceding_context_tokens": distribution([t["preceding_context_tokens"] for t in rows]),
                "eligible_target_tokens": distribution([t["target_tokens"] for t in eligible]),
                "eligible_preceding_context_tokens": distribution([t["preceding_context_tokens"] for t in eligible]),
                "exclusion_reasons": dict(sorted(Counter(r for t in rows for r in t["exclusion_reasons"]).items())),
                "duplicate_targets": sum(t["duplicate_of"] is not None for t in rows),
                "contamination_status": dict(sorted(Counter(t["contamination"]["status"] for t in rows).items())),
            })
            if version >= 2:
                cells[-1].update({
                    "contributing_conversations": contributing(rows),
                    "contributing_opening_conversations": contributing([t for t in rows if t["turn_position"] == "opening"]),
                    "contributing_follow_up_conversations": contributing([t for t in rows if t["turn_position"] == "follow_up"]),
                    "repeated_text_targets": sum(t.get("repeated_text_of") is not None for t in rows)})
            if any("context_contamination_status" in t for t in targets):  # absent from pilot_v1 records
                cells[-1]["context_contamination_status"] = dict(sorted(Counter(
                    t["context_contamination_status"] for t in rows).items()))
    return {"candidate_only": True, **({"audit_version": version} if version >= 2 else {}),
            "sources": sources, "cells": cells,
            "totals": {"conversations": len(conversations), "targets": len(targets),
                       "eligible": sum(t["eligible"] for t in targets),
                       "excluded": sum(not t["eligible"] for t in targets)},
            "groups": {g: {"targets": sum(t["group"] == g for t in targets),
                           "eligible": sum(t["group"] == g and t["eligible"] for t in targets)} for g in GROUPS}}


def audit_markdown(report):
    lines = ["# Conversation candidate audit", "",
             "Candidate construction only; no model outcomes. Eligibility is provisional and does not mean clean or held-out.",
             "Opening = first turn of that role; follow-up = subsequent turn of the same role. A first assistant target has the first user in context.",
             "Sources remain separate. Zero cells are explicit; WildChat has no human assistant and OASST1 has no model author in this design.", "",
             "| Source | Group | Conversations | Targets | Eligible | Excluded | Opening | Follow-up | Eligible opening / follow-up |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in report["cells"]:
        lines.append(f"| {c['source']} | {c['group']} | {c['conversations']} | {c['targets']} | {c['eligible']} | "
                     f"{c['excluded']} | {c['opening']} | {c['follow_up']} | {c['eligible_opening']} / {c['eligible_follow_up']} |")
    lines.extend(["", "Conversation counts in group rows overlap; do not sum them. Totals: " + str(report["totals"]), ""])
    if report.get("audit_version", 1) >= 2:
        lines.extend(["## Coverage: contributing conversations", "",
                      "A conversation contributes to a cell if it has at least one eligible target there. "
                      "Meeting a conversation quota does not imply balanced role or position coverage.", "",
                      "| Source | Group | Eligible targets | Contributing conversations | ... with an eligible opening | ... with an eligible follow-up | Repeated-wording targets (kept) |",
                      "|---|---|---:|---:|---:|---:|---:|"])
        for c in report["cells"]:
            if c["targets"]:
                lines.append(f"| {c['source']} | {c['group']} | {c['eligible']} | {c['contributing_conversations']} | "
                             f"{c['contributing_opening_conversations']} | {c['contributing_follow_up_conversations']} | "
                             f"{c['repeated_text_targets']} |")
        lines.extend(["", "| Source | Selected | Contributing (any) | Eligible user | Eligible assistant | Both roles |",
                      "|---|---:|---:|---:|---:|---:|"])
        for source, info in report["sources"].items():
            lines.append(f"| {source} | {info['selected_conversations']} | {info['contributing_conversations']} | "
                         f"{info['contributing_user_conversations']} | {info['contributing_assistant_conversations']} | "
                         f"{info['contributing_both_roles_conversations']} |")
        lines.append("")
    lines.extend(["## Sampling and conversation exclusions", "",
                  "Seeded hash selection from bounded source windows is not a uniform corpus sample. "
                  "OASST1 reads the entire split to avoid broken ancestors, then inspects a root window. "
                  "Exclusions below are mutually exclusive first-failure reasons; target reasons may overlap.", ""])
    for source, info in report["sources"].items():
        lines.extend([f"### {source}", "", "```json", json.dumps(info, indent=2, ensure_ascii=False, sort_keys=True), "```", ""])
    lines.extend(["## Lengths, target exclusions, duplicates, and contamination", "",
                  "Token distributions include all targets and a separate eligible subset in audit.json. "
                  "Context lengths count only preceding messages rendered with the configured chat template "
                  "(no generation prompt or target header). Empty history has zero tokens. No text is truncated.", "",
                  "| Source / group | Target tokens: min / median / p95 / max | Context tokens: min / median / p95 / max | Reasons | Contamination |",
                  "|---|---|---|---|---|"])
    for cell in report["cells"]:
        if not cell["targets"]:
            continue
        def fmt(key):
            return " / ".join(str(cell[key][k]) for k in ("min", "median", "p95", "max"))
        lines.append(f"| {cell['source']} / {cell['group']} | {fmt('target_tokens')} | "
                     f"{fmt('preceding_context_tokens')} | {cell['exclusion_reasons']} | {cell['contamination_status']}"
                     + (f"; context {cell['context_contamination_status']}" if "context_contamination_status" in cell else "")
                     + " |")
    lines.extend(["", "Exact text matching normalizes whitespace only (case and punctuation preserved); duplicate originals remain unchanged. "
                  "No exact match, when checked, is not a guarantee against paraphrase, substring, or unexamined training data overlap. "
                  "See manifest.json for check scope/revisions and examples.md for complete conversations and target mappings.", ""])
    return "\n".join(lines)


def sensitivity_markdown(rows, default_min):
    """Readable length-sensitivity table; rows from core.conversations.length_sensitivity."""
    thresholds = sorted({r["min_target_tokens"] for r in rows})
    keys = sorted({(r["source"], r["group"], r["turn_position"]) for r in rows})
    by = {(r["min_target_tokens"], r["source"], r["group"], r["turn_position"]): r for r in rows}
    head = " | ".join(f"min {t}: eligible / conversations" for t in thresholds)
    lines = ["# Minimum target length sensitivity (data only)", "",
             f"Same selected conversations; only the minimum target length changes (build default: {default_min}). "
             "Maximum target length, context length and contamination rules are held fixed; duplicate cases are "
             "re-resolved. With eligible_quota selection the conversations were chosen at the default minimum, so "
             "lower thresholds here do not show what a lower-threshold quota build would select. "
             "Cells: eligible targets / contributing conversations. No model outcomes.", "",
             f"| Source | Group | Position | Targets | {head} |", "|---|---|---|---:|" + "---:|" * len(thresholds)]
    for key in keys:
        first = by[(thresholds[0], *key)]
        cells = " | ".join(f"{by[(t, *key)]['eligible']} / {by[(t, *key)]['contributing_conversations']}" for t in thresholds)
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {first['targets']} | {cells} |")
    return "\n".join(lines) + "\n"


def examples_markdown(conversations, targets):
    lines = ["# Readable complete conversations and target mappings", "",
             "One shortest conversation by total character count per source (ID breaks ties), for readable inspection; "
             "this display subset is not a quality filter. Content below is verbatim. All selected conversations are in conversations.jsonl.", ""]
    for source in sorted({c["source"] for c in conversations}):
        conv = min((c for c in conversations if c["source"] == source),
                   key=lambda c: (sum(len(m["content"]) for m in c["messages"]), c["conversation_id"]))
        lines.extend([f"## {source}: {conv['source_id']}", "", f"Conversation ID: `{conv['conversation_id']}`", ""])
        for i, m in enumerate(conv["messages"]):
            fence = "`" * max(3, max((len(s) for s in m["content"].splitlines() if s and set(s) == {"`"}), default=0) + 1)
            lines.extend([f"### Message {i}: {m['role']} / {m['authorship']}", "",
                          f"ID: `{m['message_id']}`. Basis: {m['authorship_basis']}.", "", fence + "text", m["content"], fence, ""])
        lines.extend(["### Derived targets", "", "Each target's text is the full message at its turn index above. "
                      "History is exactly the listed earlier indices; no later messages are included.", "",
                      "| Target turn | Group | Preceding message indices | Target / context tokens | Eligible | Reasons |",
                      "|---:|---|---|---:|---|---|"])
        for t in targets:
            if t["conversation_id"] == conv["conversation_id"]:
                indices = list(range(t["turn_index"]))
                lines.append(f"| {t['turn_index']} | {t['group']} | {indices} | {t['target_tokens']} / "
                             f"{t['preceding_context_tokens']} | {t['eligible']} | {', '.join(t['exclusion_reasons'])} |")
        lines.append("")
    return "\n".join(lines)


def verify_build(path):
    """Offline artifact integrity, context invariants, and audit reconciliation."""
    from pathlib import Path
    import hashlib
    from core.conversations import FORMAT, Settings, context_contamination_status

    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["format"] != FORMAT:
        raise ValueError("not a conversation candidate build")
    for name, expected in manifest["output_sha256"].items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"output hash mismatch: {name}")
    def read(name):
        return [json.loads(line) for line in (path / name).read_text().splitlines()]
    conversations, targets, rejected = (read(name) for name in
                                         ("conversations.jsonl", "targets.jsonl", "exclusions.jsonl"))
    by_id = {c["conversation_id"]: c for c in conversations}
    if len(by_id) != len(conversations) or len({t["target_id"] for t in targets}) != len(targets):
        raise ValueError("nonunique conversation/target IDs")
    expected_turns = {(c["conversation_id"], i) for c in conversations for i, m in enumerate(c["messages"])
                      if m["role"] != "system"}
    if {(t["conversation_id"], t["turn_index"]) for t in targets} != expected_turns or len(targets) != len(expected_turns):
        raise ValueError("targets do not cover every user/assistant message exactly once")
    for target in targets:
        conv = by_id[target["conversation_id"]]
        i = target["turn_index"]
        msg = conv["messages"][i]
        if (target["format"] != FORMAT or target["record_type"] != "target"
                or target["preceding_messages"] != conv["messages"][:i]
                or target["target_text"] != msg["content"] or target["message_id"] != msg["message_id"]
                or target["role"] != msg["role"] or target["authorship"] != msg["authorship"]
                or target["group"] != f"{msg['authorship']}_{msg['role']}"
                or target["has_preceding_conversation"] != bool(i)
                or target["eligible"] != (not target["exclusion_reasons"])
                or target["contamination"] != msg.get("contamination", target["contamination"])
                or target["contaminated_context_message_ids"] != [
                    p["message_id"] for p in conv["messages"][:i] if p["contamination"]["matches"]]
                or target.get("context_contamination_status", context_contamination_status(conv["messages"][:i]))
                != context_contamination_status(conv["messages"][:i])):
            raise ValueError(f"target invariant failed: {target['target_id']}")
    settings = Settings(**manifest["settings"])
    version = manifest.get("audit_version", 1)
    if version >= 2:
        from core.conversations import apply_duplicates, length_sensitivity
        order = {cid: n for n, cid in enumerate(manifest["selection_order"])}
        if set(order) != set(by_id):
            raise ValueError("selection order does not match the conversations")
        in_selection_order = sorted(targets, key=lambda t: (order[t["conversation_id"]], t["turn_index"]))
        if apply_duplicates(in_selection_order) != in_selection_order:
            raise ValueError("duplicate-case / repeated-wording marks do not recompute")
        if settings.selection_mode == "eligible_quota":
            from core.conversations import contributes
            for cid in by_id:
                if not contributes([t for t in targets if t["conversation_id"] == cid]):
                    raise ValueError(f"selected conversation does not contribute: {cid}")
        rows = length_sensitivity(in_selection_order, settings, manifest["sensitivity_min_target_tokens"])
        if json.loads((path / "length_sensitivity.json").read_text()) != rows:
            raise ValueError("length sensitivity does not recompute")
        if sensitivity_markdown(rows, settings.min_target_tokens) != (path / "length_sensitivity.md").read_text():
            raise ValueError("readable length sensitivity differs")
    report = audit(conversations, targets, rejected, manifest["sources"], settings, version)
    # Round-trip normalizes JSON object keys, including integer histogram bins.
    if json.loads(json.dumps(report)) != json.loads((path / "audit.json").read_text()):
        raise ValueError("audit does not reconcile with records")
    if audit_markdown(report) != (path / "audit.md").read_text():
        raise ValueError("readable audit differs from recomputed audit")
    if examples_markdown(conversations, targets) != (path / "examples.md").read_text():
        raise ValueError("readable examples differ from records")
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify and recompute candidate audits offline, without tokenizers/models.")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    report = verify_build(args.input)
    print("Hashes, target contexts, and audit counts verified.")
    print(json.dumps(report["groups"], indent=2))


if __name__ == "__main__":
    main()
