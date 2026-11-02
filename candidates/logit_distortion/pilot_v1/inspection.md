# Pilot inspection — 2026-11-02

This is a bounded **unchecked candidate** collection, built before any new model scoring or
logit-distortion analysis. The machine-generated audit is [audit.md](audit.md), with full distributions
and eligible-subset distributions in [audit.json](audit.json). Reproduction, schemas, and interpretation
are documented in [the builder guide](../../../docs/conversation_candidates.md).

## What was sampled

| Source | Raw rows read | Candidates inspected | Pass source filters | Conversations selected | All targets | Provisionally eligible |
|---|---:|---:|---:|---:|---:|---:|
| OASST1 validation | 4,401 | 300 trees of 518 available | 50 | 8 | 34 | 23 |
| WildChat train | 300 | 300 conversations | 71 | 8 | 74 | 41 |
| UltraChat train | 300 | 300 conversations | 300 | 8 | 76 | 64 |
| Total | 5,001 | 900 candidates | 421 | 24 | 184 | 128 |

OASST1's whole split was read for complete trees; only the first 300 roots were inspected for selection.
WildChat/UltraChat use bounded initial row windows. Seeded ranking takes eight acceptable conversations
per source. These are not uniform whole-corpus samples, and passing source flags does not establish
content quality or authorship certainty. No source was substituted and no source budget was undersupplied.

| Group | Source composition (all targets) | All targets | Provisionally eligible |
|---|---|---:|---:|
| Human-written user | OASST1 18; WildChat human-submitted 37 | 55 | 29 (10 + 19) |
| Human-written assistant | OASST1 16 | 16 | 13 |
| Model-written user | UltraChat 38 | 38 | 26 |
| Model-written assistant | UltraChat 38; WildChat 37 | 75 | 60 (38 + 22) |

There are 48 opening-role targets and 136 follow-ups; 28 and 100, respectively, pass provisional filters.
All OASST1 selections contain two completed exchanges, with two branches also retaining a real trailing
user turn. WildChat spans 2–11 exchanges; UltraChat spans 3–7. The pilot contains no system messages;
local tests exercise preservation of original systems.

No exact duplicate conversations were found among source-accepted candidates, and no normalized exact
duplicate targets were found in the selected sample. The 56 ineligible targets consist of 39 below
20 tokens and 17 above 300 tokens. None exceeded the provisional 4096-token preceding-history bound;
the maximum was 3896. Every target and every message is contamination-unchecked, with no training
datasets scanned. All excluded turns remain stored and available to later contexts.

## Reading actual examples

The three complete examples in [examples.md](examples.md) were read, with their derived target mappings.
All 24 conversation openings and exchange counts were also inspected, along with the long assignment
and instruction-like WildChat conversations below. The examples were chosen by shortest total content
length per source solely for readable display, not as a further dataset selection step.

- **OASST1 `6fc8f3ee-64d9-4a20-9bbd-d5a6015fae6c`**: a company-description prompt gets a one-word
  assistant reply, then a follow-up question and response, then a final user request. The five-token
  assistant reply is ineligible but remains in the later histories. The 11-token user follow-up also
  remains. The actual trailing user is preserved; no closing assistant was invented.
- **UltraChat `252`**: a synthetic discussion of universal morality contains short opening/follow-up
  user questions and longer assistant responses. The first two user targets (13 and 12 tokens) fail the
  provisional minimum, while the subsequent 29-token user turn passes. This demonstrates how the minimum
  can change the mix of opening versus follow-up user targets.
- **WildChat `3ae27f2a38590dcb18027cf906e680c9`**: the original TrueNAS conversation has three exchanges.
  Its 322-token first assistant response is too long as a target but is retained in the next user context.
  The final eight-token user compliment is too short as a target but remains before the final assistant.
  Original trailing whitespace is preserved.
- **WildChat `4aca20cc41a5cb78545f483577283e8f`**: the opening assignment/code request has 2476 tokens,
  well above the target cap; its full text remains in later contexts. This is a substantial length mismatch
  relative to the shorter OASST1/UltraChat prompts. It is not silently shortened to satisfy the filters.
- **WildChat `a47a2648047c88178f671ee2d5c6f3cc`**: an instruction-like persona prompt precedes a sequence
  of lyric requests. It passes the published source flags despite content that merits a later quality
  decision. This text is dataset content, not an instruction to the builder. It was not manually removed
  after inspection. Human-submitted should not be read as proven human composition.

## Validation performed

- 29 local-fixture tests pass, covering branches/ancestors, malformed trees/rows, all four groups,
  original systems, no future context, ineligible intermediate messages, seeded selection/IDs,
  duplicates, both-role contamination and context hits, pinned SFT/DPO/RL loader behavior, counts,
  artifact hashes, offline audit reconciliation, and overwrite refusal.
- 48 existing OLMo/Gemma tokenizer/renderer compatibility checks pass. Model-scoring/generation smoke
  tests were deliberately not run: this task prohibits model scoring and checkpoint downloads.
- All 184 original message contents/roles were independently compared with pinned source rows, including
  OASST1 parent IDs and actual leaf termination, and exact list lengths for WildChat/UltraChat.
- All 184 target lengths and preceding-context lengths were independently compared against the real
  configured tokenizer, including tokenized-template output. Inspection caught and corrected a
  Transformers mapping-length mistake before this saved build; regression coverage is included.
- Offline verification checks hashes, complete target coverage, exact context prefixes, and recomputed
  audit counts/readable outputs. A key-order round-trip issue was corrected and tested before this build.
- A second pinned build is compared byte-for-byte across all seven generated files, including the
  manifest. `inspection.md` is a separately authored inspection record, not a builder-generated artifact.

## Decisions still needed before outcomes

1. Final number of conversations/targets and sampling frame. The smallest cell is human assistant
   (16 candidates, 13 eligible), while model assistant has 75 candidates. Equal conversation budgets do
   not balance the factorial cells; expanding OASST1 or changing budgets needs a pre-outcome decision.
2. Length and turn-position matching, separately by source and authorship. The 20-token minimum retains
   only one of eight UltraChat opening users, and eight of eighteen OASST1 user targets fail it. Median
   assistant lengths differ: OASST1 89, UltraChat 141.5, WildChat 210 tokens before filtering. Current
   thresholds are convenient pilot settings, not preregistered final criteria.
3. Whether to keep trailing unmatched user turns and how to handle instruction-like, repetitive,
   low-quality, or copied human-submitted content. Fix a source-level policy before looking at outcomes;
   do not silently repair conversations or manually cherry-pick targets.
4. Full Dolci exact-match filtering on all messages and histories, and a policy for partial/semantic
   overlap. No check here establishes held-out status. Non-OLMo families' undisclosed training data
   cannot be cleared by this machinery.
5. Original-context scoring integration, including template-added system text, target headers/boundaries,
   content-token alignment, and full input lengths. The current context threshold excludes target/header
   tokens and is not a guarantee of fitting a model window. Keep source strata and conversation IDs for
   later clustered bootstrap design; do not run these through the existing isolated-turn scorer.

## Integration re-check (2026-11-02, later)

The builder was written on the pre-family `main` and was integrated onto `main` after the model-family and
isolated-authorship commits (`a4b4210`). The pilot was rebuilt from the integrated code with the pinned
command in `docs/conversation_candidates.md`, using `/opt/arena-env` (datasets 5.0.1, transformers 5.18.0,
not the `.venv` of the first build). All six data/audit files (`conversations.jsonl`, `targets.jsonl`,
`exclusions.jsonl`, `audit.json`, `audit.md`, `examples.md`) were byte-identical to the first build.
The committed `manifest.json` is the rebuild's: it differs from the first only in code hashes
(`core/config.py`, `core/conversations.py`) and package versions. `core.conversation_audit --input` verifies it.

Re-reading real records after the rebuild (OASST1 `11df922f…`, WildChat `2bbdb06d…`, UltraChat `111`):
contexts are exact prefixes, OASST1 parent chains run root to leaf, and no first user turn has context.
One pre-outcome concern for decision 2 or 3 below: **18 of 38 UltraChat assistant targets contain
"As an AI"** (UltraChat `111` opens every assistant turn with "As an AI language model, I agree with you"),
against 0 of 37 WildChat assistant and 0 of 16 OASST1 assistant targets. The model-written assistant cell
therefore mixes a heavily templated source with a less templated one. That is a source property, not a
filtering error; whether to stratify, match or filter on it must be decided before outcomes are read.

Tests on the integrated tree: `pytest` (full suite, `/opt/arena-env`) 184 passed, including the 30 tests in
`tests/test_conversations.py` (the 29 above plus a check that `core.score` refuses candidate records).
