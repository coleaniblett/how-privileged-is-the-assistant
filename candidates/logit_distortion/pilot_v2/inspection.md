# Pilot v2 inspection: 2026-11-02

**Candidate pilot, not a final sample.** No model was scored, no weights were loaded, and no
logit-distortion analysis was run. `pilot_v2` is the `pilot_v1` sample (same seed, settings, pinned
source and tokenizer revisions) rebuilt with the Dolci exact-match check on every message in both roles
(`--check-dolci`, revisions pinned in `manifest.json`). Generated files: `audit.md`/`audit.json`,
`examples.md`, `conversations.jsonl`, `targets.jsonl`, `exclusions.jsonl`, `manifest.json`. This note is
written by hand after the build and is not covered by the manifest hashes.

Original conversations are logit distortion's selected primary data route (Grace, 2026-11-02;
`docs/decisions.md`). Final sizes and thresholds are chosen from this audit before model outcomes are seen.

## What changed from pilot_v1

- Conversations, exclusions and every non-contamination field are identical to `pilot_v1` (compared
  record by record). Only `contamination` on messages/targets and the new target fields
  `context_contamination_status` changed.
- Scope: Dolci-Instruct SFT (1,971,711 user / 2,292,197 assistant normalized turns), DPO (260,416 /
  517,137), RL (168,705 / 7,345 parsed prompt turns, 10,621 WildChat hashes). All 184 messages:
  `checked_datasets` = all three; 176 `no_exact_match`, 8 `matched`; none `unchecked` or `incomplete`.
- All 8 hits are **WildChat user turns**, all in Dolci-SFT as user turns (one also in DPO). 7 of 8
  WildChat conversations contain one; 7 are the opening user turn, and one is a 5-token follow-up
  ("Give me more examples."). No assistant reply from any source matched, and no OASST1 or UltraChat
  message matched. No RL conversation-hash hits.

## Counts (eligible / all targets)

| Source | human user | human assistant | model user | model assistant |
|---|---:|---:|---:|---:|
| OASST1 | 10 / 18 | 13 / 16 | — | — |
| WildChat (user human-submitted) | **2 / 37** | — | — | **3 / 37** |
| UltraChat | — | — | 26 / 38 | 38 / 38 |

Totals: 24 conversations, 184 targets, 92 eligible (pilot_v1 without the check: 128). Opening / follow-up
and length distributions per cell are in `audit.md`/`audit.json`. WildChat exclusions now are 21 human-user
and 28 model-assistant targets for `contamination_context`, 8 for `contamination_target`, plus length reasons
(reasons overlap).

**Shortage, reported, filters not relaxed:** the WildChat cells are nearly empty, because a matched
opening user turn makes every later target in that conversation ineligible under the context policy.
All four role × authorship groups still have eligible targets, but WildChat contributes 5.

## Records read

- WildChat `c7147a1d…`: message 0 (user) matched SFT; message 1 (assistant) has its own status
  `no_exact_match` and context status `matched`, listing message 0. Message 2 "Give me more examples."
  matched as well. Message 0 stays verbatim in every later prefix; nothing was removed or trimmed.
- OASST1 `6fc8f3ee…`: the 1-token assistant reply "Psychomotricity" is too short as a target but stays
  in the prefixes of turns 2–4; prefixes equal `messages[:turn_index]` exactly, with no future messages.
- Every target's `preceding_messages` was compared with `conversation.messages[:turn_index]`
  (`core.conversation_audit --input`, which also re-derives context contamination fields and the audit).

## Decisions for Grace before scoring (not made here)

1. **WildChat contamination policy.** Options: keep the strict context rule and sample WildChat
   conversations with no Dolci hit before applying the budget (a conversation-level filter); or exclude
   only targets whose own text matched. The first is cleaner, the second keeps more data but scores
   targets after training-set prompts. Also whether short generic turns (e.g. "Give me more examples.")
   should count as a hit.
2. Final budgets per source (the human-assistant cell is OASST1 only; 13 eligible here).
3. Length thresholds and turn-position matching (the 20-token minimum removes 7 of 8 UltraChat opening
   users; WildChat assistant replies are long).
4. The "As an AI" templating in UltraChat assistants (18 of 38, see `pilot_v1/inspection.md`).
5. Original-context scoring integration (template framing, target boundaries, full-window lengths),
   a separate task; these records are refused by `core.score`.
