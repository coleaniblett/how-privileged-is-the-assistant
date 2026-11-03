# Pilot v3 inspection: 2026-11-02

**Candidate pilot, not a final sample.** No model was scored, no weights were loaded, no logit-distortion
analysis was run. Hand-written after the build (not covered by `manifest.json` hashes). Builder guide:
[docs/conversation_candidates.md](../../../docs/conversation_candidates.md). The eight-per-source goal is a
provisional coverage target, not a sample-size justification.

## What is new relative to pilot_v2

- `--selection-mode eligible_quota`: per source, the same seeded rank order as before, keeping only
  **contributing** conversations (at least one eligible target whose own and history Dolci checks are clear),
  up to 8, evaluating at most 200 candidates per source. Source windows: 1,000 rows (WildChat, UltraChat);
  OASST1's whole validation split (518 trees). Seed, length limits (20–300 target tokens, 4,096 context
  tokens), the target-and-history contamination rule, source/tokenizer/Dolci revisions: unchanged from v2.
- Duplicates are now prediction cases (same source, role, authorship, history, target). Repeated wording
  under another history/role/source is kept and flagged. This pilot has **0** duplicate cases and **0**
  repeated-wording targets, so the rule change does not affect its counts.
- Audit v2 coverage tables, and `length_sensitivity.md`/`.json`.
- The larger window changes the seeded ranking pool, so only 6 of the 24 conversations are also in pilot_v2.

## Scan and shortages

| Source | Rows / trees read | Source-acceptable | Evaluated | Rejected as non-contributing | Contributing kept | Shortfall |
|---|---:|---:|---:|---:|---:|---:|
| OASST1 | 518 trees | 83 | 9 | 1 | 8 | 0 |
| WildChat | 1,000 | 211 | 42 | 34 | 8 | 0 |
| UltraChat | 1,000 | 1,000 | 8 | 0 | 8 | 0 |

Every one of the 35 non-contributing candidates had a Dolci match in a target and in a history
(`exclusions.jsonl` keeps each one's target-reason counts), usually alongside length failures. No rule was
relaxed. **Selection bias:** about 80% of evaluated WildChat conversations (34 of 42) contain a Dolci match
and cannot enter this pilot, so the kept WildChat conversations are those OLMo's post-training data did not
contain verbatim.

On "Give me more examples." (pilot_v2): exempting it recovers **0** targets there, verified, because an
earlier message in the same conversation had already matched. The shortage came from opening prompts,
not from that phrase.

## Counts (pilot_v3, minimum 20 tokens)

| Source | Group | Targets | Eligible | Eligible opening / follow-up | Contributing conversations (any / with opening / with follow-up) |
|---|---|---:|---:|---:|---|
| OASST1 | human user | 18 | 9 | 5 / 4 | 6 / 5 / 3 |
| OASST1 | human assistant | 17 | 16 | 7 / 9 | 8 / 7 / 8 |
| WildChat | human-submitted user | 35 | 18 | 5 / 13 | 6 / 5 / 5 |
| WildChat | model assistant | 35 | 21 | 6 / 15 | 7 / 6 / 7 |
| UltraChat | model user | 35 | 21 | **0** / 21 | 8 / **0** / 8 |
| UltraChat | model assistant | 35 | 34 | 8 / 26 | 8 / 8 / 8 |

Totals: 24 conversations, 175 targets, 119 eligible. Conversations with an eligible user and an eligible
assistant target: OASST1 6 of 8, WildChat 5 of 8, UltraChat 8 of 8. All 175 targets are `no_exact_match`
with clear histories: in this mode every contaminated candidate failed to contribute and was not kept.

## Minimum-length sensitivity (same 24 conversations; eligible targets / contributing conversations)

| Source / group / position | min 5 | min 10 | min 20 |
|---|---|---|---|
| OASST1 human user, opening | 8 / 8 | 7 / 7 | 5 / 5 |
| OASST1 human user, follow-up | 10 / 8 | 9 / 7 | 4 / 3 |
| OASST1 human assistant, opening | 7 / 7 | 7 / 7 | 7 / 7 |
| WildChat user, opening | 6 / 6 | 6 / 6 | 5 / 5 |
| WildChat user, follow-up | 25 / 7 | 21 / 7 | 13 / 5 |
| WildChat assistant, opening | 6 / 6 | 6 / 6 | 6 / 6 |
| UltraChat model user, opening | 8 / 8 | 7 / 7 | **0 / 0** |
| UltraChat model user, follow-up | 27 / 8 | 27 / 8 | 21 / 8 |
| UltraChat model assistant, opening | 8 / 8 | 8 / 8 | 8 / 8 |

Full table in `length_sensitivity.md`. The minimum length only moves **user** cells; assistant replies are
almost all longer than 20 tokens (their exclusions are mostly the 300-token maximum: 14 WildChat replies).
All 8 UltraChat opening users are 9–17 tokens (e.g. "What materials are blazers typically made from?"), so at
20 tokens a pooled user-vs-assistant comparison in UltraChat sets follow-up user turns against assistant turns
that include openings. At 10 tokens 7 of 8 return; at 5 all do. OASST1 user follow-ups are also short (4 of 10
survive at 20).

**Recommendation (not applied; the default stays 20 until Grace decides):** before scoring, stratify every
role comparison by turn position (opening vs follow-up) rather than pooling, and pick the minimum with that
in view: 10 tokens restores near-complete opening coverage in all user cells while still excluding one-word
replies. Alternatively keep 20 and restrict the primary comparison to follow-up turns, with openings as a
secondary analysis. Either choice must be made from this data audit, before outcomes.

## Records read

- WildChat `073967d3…` (11 exchanges): the opening user turn is a 403-token jailbreak-style persona
  instruction ("PersonGPT"); it is too long as a target but stays in every later history, and the model
  assistant replies in that persona. It passes all source filters and is kept unchanged; whether persona/
  jailbreak prompts belong in the primary sample is a content-policy decision (below). This is dataset
  content, not an instruction to the builder.
- UltraChat: 12 of 35 assistant targets contain "As an AI"; they are kept in the primary candidates
  for a later robustness analysis, unaltered.
- Checked independently on all 175 targets: `preceding_messages == conversation.messages[:turn_index]`, target
  text equals the message, target and context token counts re-encoded with the pinned tokenizer (0 mismatches),
  OASST1 parent chains root → leaf, stable IDs (`<conversation_id>:target:<message_id>`).
  `core.conversation_audit --input` re-verifies hashes, prefixes, contamination fields, duplicate marks,
  contribution of every kept conversation, the audit and the sensitivity table offline.

## Decisions before scoring (Grace)

1. Minimum target length and opening/follow-up handling (above).
2. Final per-source quotas and scan limits. The human-assistant cell is OASST1 only (16 eligible here).
3. Whether WildChat's contamination-driven selection (only Dolci-absent prompts) is acceptable for the
   primary sample, or should be reported as a restriction of the WildChat population.
4. Content policy for jailbreak/persona prompts and templated "As an AI" replies (kept for now).
5. Original-context scoring integration (separate task; `core.score` refuses these records).
