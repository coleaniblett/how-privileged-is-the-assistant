"""Source adapters and target construction; no generation, scoring, or text rewriting."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json

from core import config
from core.check_contamination import contamination_record, norm

FORMAT = config.CONVERSATION_FORMAT
GROUPS = ("human_user", "human_assistant", "model_user", "model_assistant")


@dataclass(frozen=True)
class Settings:
    seed: int = 0
    conversation_budget: int = 8  # per source, before target eligibility
    source_window: int = 300  # roots for OASST1, rows for flat datasets
    min_exchanges: int = 2
    min_target_tokens: int = 20
    max_target_tokens: int = 300
    max_context_tokens: int = 4096

    def __post_init__(self):
        if min(self.conversation_budget, self.source_window) < 1 or self.min_exchanges < 2:
            raise ValueError("positive budget/window and at least two exchanges required")
        if not 0 <= self.min_target_tokens <= self.max_target_tokens or self.max_context_tokens < 0:
            raise ValueError("invalid token limits")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     default=str, separators=(",", ":")).encode()).hexdigest()


def priority(seed, *parts):
    return digest([seed, *parts])


def message(mid, role, content, source, metadata=None):
    if role == "system":
        author, basis = "unknown", "source system message; authorship not inferred"
    elif source == "oasst1":
        author, basis = "human", "OASST1 crowdsourced message; synthetic=false"
    elif source == "ultrachat":
        author, basis = "model", "UltraChat card: separate ChatGPT Turbo APIs generate both roles"
    elif role == "user":
        author, basis = "human", "WildChat human-submitted; dataset-based assumption, not proof of composition"
    else:
        author, basis = "model", "WildChat recorded assistant response; source model metadata"
    return {"message_id": mid, "role": role, "authorship": author,
            "authorship_basis": basis, "content": content, "source_metadata": metadata or {}}


def conversation(source, source_id, provenance, messages):
    return {"format": FORMAT, "record_type": "conversation", "source": source,
            "conversation_id": f"{provenance['dataset']}:{provenance['split']}:{source_id}",
            "source_id": str(source_id), **provenance, "messages": messages}


def validate(conv, min_exchanges):
    """Leading systems are retained; subsequent user/assistant turns must alternate."""
    messages = conv["messages"]
    if not messages:
        return "empty_conversation"
    ids = [m["message_id"] for m in messages]
    if len(set(ids)) != len(ids):
        return "duplicate_message_id"
    expected, exchanges, started = "user", 0, False
    for m in messages:
        if not isinstance(m["content"], str) or not m["content"].strip():
            return "empty_or_nontext_message"
        if m["role"] == "system" and not started:
            continue
        started = True
        if m["role"] != expected:
            return "invalid_role_order"
        if m["role"] == "assistant":
            exchanges += 1
        expected = "assistant" if expected == "user" else "user"
    if exchanges < min_exchanges:
        return "too_few_exchanges"
    conv["complete_exchanges"] = exchanges
    return None


def oasst_branches(rows, provenance, settings):
    """Reconstruct complete trees, select ONE maximal branch per root before quality filters.

    At each fork choose the child with smallest SHA256([seed, 'branch', root, parent,
    child]). Failed branches are rejected whole; siblings are never substituted.
    Input must be a complete split, not a truncated message window.
    """
    trees = defaultdict(list)
    for row in rows:
        trees[row.get("message_tree_id")].append(row)
    candidates, rejected = [], []
    for tree_id, nodes in list(trees.items())[:settings.source_window]:
        def reject(reason):
            rejected.append({"source": "oasst1", "source_id": tree_id, "reason": reason})

        by_id = {r.get("message_id"): r for r in nodes}
        if not tree_id or None in by_id or len(by_id) != len(nodes):
            reject("malformed_tree_or_duplicate_message_id")
            continue
        roots = [r for r in nodes if r.get("parent_id") is None]
        if len(roots) != 1 or roots[0]["message_id"] != tree_id:
            reject("missing_or_multiple_roots")
            continue
        children = defaultdict(list)
        malformed = False
        for r in nodes:
            parent = r.get("parent_id")
            if parent is not None:
                if parent not in by_id:
                    malformed = True
                children[parent].append(r["message_id"])
        # Verify every node reaches the actual root (including unselected siblings).
        for mid in by_id:
            seen = set()
            while mid is not None and mid in by_id:
                if mid in seen:
                    malformed = True
                    break
                seen.add(mid)
                mid = by_id[mid].get("parent_id")
        if malformed:
            reject("missing_parent_or_cycle")
            continue
        branch, mid = [], tree_id
        while True:
            branch.append(by_id[mid])
            if not children[mid]:
                break
            mid = min(children[mid], key=lambda child: priority(settings.seed, "branch", tree_id, mid, child))
        if any(r.get("lang") != "en" for r in branch):
            reject("non_english_branch")
            continue
        if any(r.get("deleted") is not False or r.get("review_result") is not True for r in branch):
            reject("unreviewed_or_deleted_branch")
            continue
        if any(r.get("synthetic") is not False for r in branch):
            reject("synthetic_or_unknown_authorship_branch")
            continue
        messages = [message(f"oasst1:{r['message_id']}",
                            {"prompter": "user", "assistant": "assistant"}.get(r.get("role"), "invalid"),
                            r.get("text"), "oasst1",
                            {k: r.get(k) for k in ("message_id", "parent_id", "message_tree_id", "role",
                                                   "lang", "synthetic", "review_result", "deleted", "rank")})
                    for r in branch]
        conv = conversation("oasst1", tree_id, provenance, messages)
        reason = validate(conv, settings.min_exchanges)
        if reason:
            reject(reason)
        else:
            candidates.append(conv)
    return candidates, rejected, min(len(trees), settings.source_window)


def flat_conversations(rows, source, provenance, settings):
    candidates, rejected = [], []
    for row in rows:
        source_id = row.get("conversation_hash" if source == "wildchat" else "id")
        reason = None
        if source_id is None or str(source_id) == "":
            reason = "missing_source_id"
        raw = row.get("conversation" if source == "wildchat" else "data")
        if not isinstance(raw, list) or not raw:
            reason = reason or "empty_or_malformed_conversation"
        if source == "wildchat":
            if row.get("language") != "English":
                reason = reason or "non_english_conversation"
            if row.get("toxic") is not False or row.get("redacted") is not False:
                reason = reason or "toxic_redacted_or_unknown"
            if isinstance(raw, list) and any(not isinstance(m, dict) for m in raw):
                reason = reason or "malformed_message"
            if not reason and any(m.get("toxic") is not False or m.get("redacted") is not False
                                  for m in raw):
                reason = "toxic_redacted_or_unknown_message"
            if not reason and any(m.get("language") != "English" for m in raw if m.get("role") != "system"):
                reason = "non_english_message"
        if reason:
            rejected.append({"source": source, "source_id": source_id, "reason": reason})
            continue
        conv = conversation(source, source_id, provenance, [])
        for i, raw_message in enumerate(raw):
            role = raw_message.get("role") if source == "wildchat" else ("user" if i % 2 == 0 else "assistant")
            content = raw_message.get("content") if source == "wildchat" else raw_message
            meta = ({"turn_identifier": raw_message.get("turn_identifier"), "model": row.get("model")}
                    if source == "wildchat" else {"array_index": i})
            conv["messages"].append(message(f"{conv['conversation_id']}:message:{i}", role, content, source, meta))
        reason = validate(conv, settings.min_exchanges)
        if reason:
            rejected.append({"source": source, "source_id": source_id, "reason": reason})
        else:
            candidates.append(conv)
    return candidates, rejected, len(rows)


def select_conversations(candidates, settings):
    """Stable hash ordering within each source; deduplicate before budget selection."""
    rejected, unique, seen_content = [], [], {}
    by_id = defaultdict(list)
    for c in candidates:
        by_id[c["conversation_id"]].append(c)
    for cid, copies in sorted(by_id.items()):
        if len({digest(c) for c in copies}) > 1:
            rejected.extend({"source": c["source"], "source_id": c["source_id"],
                             "reason": "conflicting_conversation_id"} for c in copies)
            continue
        unique.append(copies[0])
        rejected.extend({"source": c["source"], "source_id": c["source_id"],
                         "reason": "duplicate_conversation_id", "duplicate_of": cid} for c in copies[1:])
    selected, counts = [], Counter()
    for conv in sorted(unique, key=lambda c: (priority(settings.seed, "sample", c["conversation_id"]),
                                             c["conversation_id"])):
        key = digest([(m["role"], norm(m["content"])) for m in conv["messages"]])
        reason = None
        if key in seen_content:
            reason = "duplicate_conversation_text"
        else:
            seen_content[key] = conv["conversation_id"]
            if counts[conv["source"]] >= settings.conversation_budget:
                reason = "conversation_budget"
        if reason:
            rejected.append({"source": conv["source"], "source_id": conv["source_id"], "reason": reason,
                             **({"duplicate_of": seen_content[key]} if reason.startswith("duplicate") else {})})
        else:
            counts[conv["source"]] += 1
            selected.append(conv)
    return sorted(selected, key=lambda c: (c["source"], c["conversation_id"])), rejected


def make_targets(conversations, tokenizer, settings, contamination=None):
    targets = []
    seen_text = {}
    for conv in conversations:
        messages = conv["messages"]
        # Check ALL messages, including assistants and history, never just the opening prompt.
        for m in messages:
            m["contamination"] = contamination_record(
                contamination, m["content"], conv["source_id"] if conv["source"] == "wildchat" else None)
        role_counts = Counter()
        for i, m in enumerate(messages):
            if m["role"] == "system":
                continue
            context = messages[:i]
            role_counts[m["role"]] += 1
            tokens = len(tokenizer.encode(m["content"], add_special_tokens=False))
            # Only preceding messages, no synthetic greeting or future reply. Chat framing
            # is included, but target header/closing tokens are not (scoring is a later task).
            context_tokens = (len(tokenizer.encode(tokenizer.apply_chat_template(
                [{"role": p["role"], "content": p["content"]} for p in context],
                tokenize=False, add_generation_prompt=False), add_special_tokens=False)) if context else 0)
            reasons = []
            if tokens < settings.min_target_tokens:
                reasons.append("target_too_short")
            if tokens > settings.max_target_tokens:
                reasons.append("target_too_long")
            if context_tokens > settings.max_context_tokens:
                reasons.append("context_too_long")
            tid = f"{conv['conversation_id']}:target:{m['message_id']}"
            key = norm(m["content"])
            duplicate_of = seen_text.get(key)
            if duplicate_of:
                reasons.append("duplicate_target_text")
            else:
                seen_text[key] = tid
            context_matches = [p["message_id"] for p in context if p["contamination"]["matches"]]
            if m["contamination"]["matches"]:
                reasons.append("contamination_target")
            if context_matches:
                reasons.append("contamination_context")
            targets.append({"format": FORMAT, "record_type": "target", "target_id": tid,
                            "conversation_id": conv["conversation_id"], "message_id": m["message_id"],
                            "source": conv["source"], "dataset": conv["dataset"], "split": conv["split"],
                            "revision": conv["revision"], "turn_index": i, "role": m["role"],
                            "authorship": m["authorship"], "authorship_basis": m["authorship_basis"],
                            "group": f"{m['authorship']}_{m['role']}", "target_text": m["content"],
                            "preceding_messages": context, "has_preceding_conversation": bool(context),
                            "turn_position": "opening" if role_counts[m["role"]] == 1 else "follow_up",
                            "target_tokens": tokens, "preceding_context_tokens": context_tokens,
                            "eligible": not reasons, "exclusion_reasons": reasons, "duplicate_of": duplicate_of,
                            "contamination": m["contamination"], "contaminated_context_message_ids": context_matches})
    return targets
