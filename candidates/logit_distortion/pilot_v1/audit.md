# Conversation candidate audit

Candidate construction only; no model outcomes. Eligibility is provisional and does not mean clean or held-out.
Opening = first turn of that role; follow-up = subsequent turn of the same role. A first assistant target has the first user in context.
Sources remain separate. Zero cells are explicit; WildChat has no human assistant and OASST1 has no model author in this design.

| Source | Group | Conversations | Targets | Eligible | Excluded | Opening | Follow-up | Eligible opening / follow-up |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| oasst1 | human_user | 8 | 18 | 10 | 8 | 8 | 10 | 5 / 5 |
| oasst1 | human_assistant | 8 | 16 | 13 | 3 | 8 | 8 | 5 / 8 |
| oasst1 | model_user | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 |
| oasst1 | model_assistant | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 |
| ultrachat | human_user | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 |
| ultrachat | human_assistant | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 |
| ultrachat | model_user | 8 | 38 | 26 | 12 | 8 | 30 | 1 / 25 |
| ultrachat | model_assistant | 8 | 38 | 38 | 0 | 8 | 30 | 8 / 30 |
| wildchat | human_user | 8 | 37 | 19 | 18 | 8 | 29 | 5 / 14 |
| wildchat | human_assistant | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 |
| wildchat | model_user | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 |
| wildchat | model_assistant | 8 | 37 | 22 | 15 | 8 | 29 | 4 / 18 |

Conversation counts in group rows overlap; do not sum them. Totals: {'conversations': 24, 'targets': 184, 'eligible': 128, 'excluded': 56}

## Sampling and conversation exclusions

Seeded hash selection from bounded source windows is not a uniform corpus sample. OASST1 reads the entire split to avoid broken ancestors, then inspects a root window. Exclusions below are mutually exclusive first-failure reasons; target reasons may overlap.

### oasst1

```json
{
  "adapter_accepted": 50,
  "conversation_budget_shortfall": 0,
  "conversation_exclusions": {
    "conversation_budget": 42,
    "non_english_branch": 192,
    "too_few_exchanges": 50,
    "unreviewed_or_deleted_branch": 8
  },
  "dataset": "OpenAssistant/oasst1",
  "duplicate_conversations": 0,
  "eligible_targets": 23,
  "exchanges_histogram": {
    "2": 8
  },
  "exchanges_per_conversation": {
    "max": 2,
    "median": 2.0,
    "min": 2,
    "n": 8,
    "p25": 2.0,
    "p75": 2.0,
    "p95": 2.0
  },
  "input_rows_sha256": "a5c8cc7c8c0d2ac07920751effdb7061120b67f7fff76064f33b8cfbe75cbeef",
  "inspected_candidates": 300,
  "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
  "rows_read": 4401,
  "selected_conversations": 8,
  "split": "validation",
  "system_messages": 0,
  "targets": 34,
  "trees_in_complete_split": 518,
  "window": "complete split read; first roots in encounter order"
}
```

### ultrachat

```json
{
  "adapter_accepted": 300,
  "conversation_budget_shortfall": 0,
  "conversation_exclusions": {
    "conversation_budget": 292
  },
  "dataset": "openbmb/UltraChat",
  "duplicate_conversations": 0,
  "eligible_targets": 64,
  "exchanges_histogram": {
    "3": 3,
    "5": 3,
    "7": 2
  },
  "exchanges_per_conversation": {
    "max": 7,
    "median": 5.0,
    "min": 3,
    "n": 8,
    "p25": 3.0,
    "p75": 5.5,
    "p95": 7.0
  },
  "input_rows_sha256": "115ce757ab95020a0f26a3d9a10958ffd1a6368f2e60b6f12a49821b1c69b179",
  "inspected_candidates": 300,
  "revision": "f220fe796ce3ed62fbe1681b45ce6cbc9c6cabe0",
  "rows_read": 300,
  "selected_conversations": 8,
  "split": "train",
  "system_messages": 0,
  "targets": 76,
  "window": "first rows in dataset iteration order; no corpus shuffle"
}
```

### wildchat

```json
{
  "adapter_accepted": 71,
  "conversation_budget_shortfall": 0,
  "conversation_exclusions": {
    "conversation_budget": 63,
    "non_english_conversation": 129,
    "non_english_message": 26,
    "too_few_exchanges": 74
  },
  "dataset": "allenai/WildChat-1M",
  "duplicate_conversations": 0,
  "eligible_targets": 41,
  "exchanges_histogram": {
    "2": 3,
    "3": 2,
    "5": 1,
    "9": 1,
    "11": 1
  },
  "exchanges_per_conversation": {
    "max": 11,
    "median": 3.0,
    "min": 2,
    "n": 8,
    "p25": 2.0,
    "p75": 6.0,
    "p95": 10.3
  },
  "input_rows_sha256": "4ea60afcfe76dc84cc4cd79b05d7a0115c78e73bdf15cf2d061b183e131e24ab",
  "inspected_candidates": 300,
  "revision": "7d6490e462285cf85d91eabea0f9a954fbddcd1f",
  "rows_read": 300,
  "selected_conversations": 8,
  "split": "train",
  "system_messages": 0,
  "targets": 74,
  "window": "first rows in dataset iteration order; no corpus shuffle"
}
```

## Lengths, target exclusions, duplicates, and contamination

Token distributions include all targets and a separate eligible subset in audit.json. Context lengths count only preceding messages rendered with the configured chat template (no generation prompt or target header). Empty history has zero tokens. No text is truncated.

| Source / group | Target tokens: min / median / p95 / max | Context tokens: min / median / p95 / max | Reasons | Contamination |
|---|---|---|---|---|
| oasst1 / human_user | 6 / 20.5 / 51.45 / 71 | 0 / 77.5 / 414.35 / 609 | {'target_too_short': 8} | {'unchecked': 18} |
| oasst1 / human_assistant | 5 / 89.0 / 281.5 / 322 | 40 / 94.5 / 376.75 / 406 | {'target_too_long': 1, 'target_too_short': 2} | {'unchecked': 16} |
| ultrachat / model_user | 8 / 26.0 / 44.15 / 45 | 0 / 379.5 / 1058.0 / 1077 | {'target_too_short': 12} | {'unchecked': 38} |
| ultrachat / model_assistant | 68 / 141.5 / 282.3 / 296 | 41 / 420.5 / 1101.5 / 1113 | {} | {'unchecked': 38} |
| wildchat / human_user | 4 / 21 / 536.8 / 2476 | 0 / 808 / 3266.4 / 3879 | {'target_too_long': 2, 'target_too_short': 16} | {'unchecked': 37} |
| wildchat / model_assistant | 18 / 210 / 547.0 / 799 | 39 / 1118 / 3440.2 / 3896 | {'target_too_long': 14, 'target_too_short': 1} | {'unchecked': 37} |

Exact text matching normalizes whitespace only (case and punctuation preserved); duplicate originals remain unchanged. No exact match, when checked, is not a guarantee against paraphrase, substring, or unexamined training data overlap. See manifest.json for check scope/revisions and examples.md for complete conversations and target mappings.
