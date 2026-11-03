# Conversation candidate audit

Candidate construction only; no model outcomes. Eligibility is provisional and does not mean clean or held-out.
Opening = first turn of that role; follow-up = subsequent turn of the same role. A first assistant target has the first user in context.
Sources remain separate. Zero cells are explicit; WildChat has no human assistant and OASST1 has no model author in this design.

| Source | Group | Conversations | Targets | Eligible | Excluded | Opening | Follow-up | Eligible opening / follow-up |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| oasst1 | human_user | 8 | 18 | 9 | 9 | 8 | 10 | 5 / 4 |
| oasst1 | human_assistant | 8 | 17 | 16 | 1 | 8 | 9 | 7 / 9 |
| oasst1 | model_user | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 |
| oasst1 | model_assistant | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 |
| ultrachat | human_user | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 |
| ultrachat | human_assistant | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 |
| ultrachat | model_user | 8 | 35 | 21 | 14 | 8 | 27 | 0 / 21 |
| ultrachat | model_assistant | 8 | 35 | 34 | 1 | 8 | 27 | 8 / 26 |
| wildchat | human_user | 8 | 35 | 18 | 17 | 8 | 27 | 5 / 13 |
| wildchat | human_assistant | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 |
| wildchat | model_user | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 |
| wildchat | model_assistant | 8 | 35 | 21 | 14 | 8 | 27 | 6 / 15 |

Conversation counts in group rows overlap; do not sum them. Totals: {'conversations': 24, 'targets': 175, 'eligible': 119, 'excluded': 56}

## Coverage: contributing conversations

A conversation contributes to a cell if it has at least one eligible target there. Meeting a conversation quota does not imply balanced role or position coverage.

| Source | Group | Eligible targets | Contributing conversations | ... with an eligible opening | ... with an eligible follow-up | Repeated-wording targets (kept) |
|---|---|---:|---:|---:|---:|---:|
| oasst1 | human_user | 9 | 6 | 5 | 3 | 0 |
| oasst1 | human_assistant | 16 | 8 | 7 | 8 | 0 |
| ultrachat | model_user | 21 | 8 | 0 | 8 | 0 |
| ultrachat | model_assistant | 34 | 8 | 8 | 8 | 0 |
| wildchat | human_user | 18 | 6 | 5 | 5 | 0 |
| wildchat | model_assistant | 21 | 7 | 6 | 7 | 0 |

| Source | Selected | Contributing (any) | Eligible user | Eligible assistant | Both roles |
|---|---:|---:|---:|---:|---:|
| oasst1 | 8 | 8 | 6 | 8 | 6 |
| ultrachat | 8 | 8 | 8 | 8 | 8 |
| wildchat | 8 | 8 | 6 | 7 | 5 |

## Sampling and conversation exclusions

Seeded hash selection from bounded source windows is not a uniform corpus sample. OASST1 reads the entire split to avoid broken ancestors, then inspects a root window. Exclusions below are mutually exclusive first-failure reasons; target reasons may overlap.

### oasst1

```json
{
  "adapter_accepted": 83,
  "contributing_assistant_conversations": 8,
  "contributing_both_roles_conversations": 6,
  "contributing_conversations": 8,
  "contributing_quota": 8,
  "contributing_quota_shortfall": 0,
  "contributing_user_conversations": 6,
  "conversation_exclusions": {
    "contributing_quota_reached": 74,
    "no_contributing_target": 1,
    "non_english_branch": 330,
    "too_few_exchanges": 90,
    "unreviewed_or_deleted_branch": 15
  },
  "dataset": "OpenAssistant/oasst1",
  "duplicate_conversations": 0,
  "eligible_targets": 25,
  "exchanges_histogram": {
    "2": 7,
    "3": 1
  },
  "exchanges_per_conversation": {
    "max": 3,
    "median": 2.0,
    "min": 2,
    "n": 8,
    "p25": 2.0,
    "p75": 2.0,
    "p95": 2.65
  },
  "input_rows_sha256": "a5c8cc7c8c0d2ac07920751effdb7061120b67f7fff76064f33b8cfbe75cbeef",
  "inspected_candidates": 518,
  "quota_scan": {
    "contributing": 8,
    "evaluated": 9
  },
  "repeated_text_targets": 0,
  "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
  "rows_read": 4401,
  "scan_limit": 200,
  "selected_conversations": 8,
  "selection_mode": "eligible_quota",
  "split": "validation",
  "system_messages": 0,
  "targets": 35,
  "trees_in_complete_split": 518,
  "window": "complete split read; first roots in encounter order"
}
```

### ultrachat

```json
{
  "adapter_accepted": 1000,
  "contributing_assistant_conversations": 8,
  "contributing_both_roles_conversations": 8,
  "contributing_conversations": 8,
  "contributing_quota": 8,
  "contributing_quota_shortfall": 0,
  "contributing_user_conversations": 8,
  "conversation_exclusions": {
    "contributing_quota_reached": 992
  },
  "dataset": "openbmb/UltraChat",
  "duplicate_conversations": 0,
  "eligible_targets": 55,
  "exchanges_histogram": {
    "3": 3,
    "4": 3,
    "7": 2
  },
  "exchanges_per_conversation": {
    "max": 7,
    "median": 4.0,
    "min": 3,
    "n": 8,
    "p25": 3.0,
    "p75": 4.75,
    "p95": 7.0
  },
  "input_rows_sha256": "29996851702ae034a68a4b686fc5d05a3e89d7c502cd48391ab16ba4f6c93936",
  "inspected_candidates": 1000,
  "quota_scan": {
    "contributing": 8,
    "evaluated": 8
  },
  "repeated_text_targets": 0,
  "revision": "f220fe796ce3ed62fbe1681b45ce6cbc9c6cabe0",
  "rows_read": 1000,
  "scan_limit": 200,
  "selected_conversations": 8,
  "selection_mode": "eligible_quota",
  "split": "train",
  "system_messages": 0,
  "targets": 70,
  "window": "first rows in dataset iteration order; no corpus shuffle"
}
```

### wildchat

```json
{
  "adapter_accepted": 211,
  "contributing_assistant_conversations": 7,
  "contributing_both_roles_conversations": 5,
  "contributing_conversations": 8,
  "contributing_quota": 8,
  "contributing_quota_shortfall": 0,
  "contributing_user_conversations": 6,
  "conversation_exclusions": {
    "contributing_quota_reached": 169,
    "no_contributing_target": 34,
    "non_english_conversation": 477,
    "non_english_message": 81,
    "too_few_exchanges": 226,
    "toxic_redacted_or_unknown": 5
  },
  "dataset": "allenai/WildChat-1M",
  "duplicate_conversations": 0,
  "eligible_targets": 39,
  "exchanges_histogram": {
    "2": 4,
    "3": 1,
    "4": 1,
    "9": 1,
    "11": 1
  },
  "exchanges_per_conversation": {
    "max": 11,
    "median": 2.5,
    "min": 2,
    "n": 8,
    "p25": 2.0,
    "p75": 5.25,
    "p95": 10.3
  },
  "input_rows_sha256": "9e164938db9e6000d9d086fa2928882d0ff6b1f5f69223515f60a03d51e35487",
  "inspected_candidates": 1000,
  "quota_scan": {
    "contributing": 8,
    "evaluated": 42
  },
  "repeated_text_targets": 0,
  "revision": "7d6490e462285cf85d91eabea0f9a954fbddcd1f",
  "rows_read": 1000,
  "scan_limit": 200,
  "selected_conversations": 8,
  "selection_mode": "eligible_quota",
  "split": "train",
  "system_messages": 0,
  "targets": 70,
  "window": "first rows in dataset iteration order; no corpus shuffle"
}
```

## Lengths, target exclusions, duplicates, and contamination

Token distributions include all targets and a separate eligible subset in audit.json. Context lengths count only preceding messages rendered with the configured chat template (no generation prompt or target header). Empty history has zero tokens. No text is truncated.

| Source / group | Target tokens: min / median / p95 / max | Context tokens: min / median / p95 / max | Reasons | Contamination |
|---|---|---|---|---|
| oasst1 / human_user | 6 / 19.5 / 122.25 / 254 | 0 / 109.5 / 538.5 / 666 | {'target_too_short': 9} | {'no_exact_match': 18}; context {'no_context': 8, 'no_exact_match': 10} |
| oasst1 / human_assistant | 21 / 122 / 325.0 / 465 | 41 / 146 / 596.6 / 771 | {'target_too_long': 1} | {'no_exact_match': 17}; context {'no_exact_match': 17} |
| ultrachat / model_user | 9 / 25 / 43.3 / 55 | 0 / 399 / 923.4 / 1075 | {'target_too_short': 14} | {'no_exact_match': 35}; context {'no_context': 8, 'no_exact_match': 27} |
| ultrachat / model_assistant | 32 / 205 / 276.9 / 302 | 42 / 434 / 967.2 / 1113 | {'target_too_long': 1} | {'no_exact_match': 35}; context {'no_exact_match': 35} |
| wildchat / human_user | 4 / 26 / 696.4 / 1481 | 0 / 880 / 3418.4 / 3879 | {'target_too_long': 3, 'target_too_short': 14} | {'no_exact_match': 35}; context {'no_context': 8, 'no_exact_match': 27} |
| wildchat / model_assistant | 39 / 277 / 501.9 / 1622 | 46 / 976 / 3440.0 / 3896 | {'target_too_long': 14} | {'no_exact_match': 35}; context {'no_exact_match': 35} |

Exact text matching normalizes whitespace only (case and punctuation preserved); duplicate originals remain unchanged. No exact match, when checked, is not a guarantee against paraphrase, substring, or unexamined training data overlap. See manifest.json for check scope/revisions and examples.md for complete conversations and target mappings.
