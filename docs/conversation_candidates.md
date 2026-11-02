# Conversation candidates for logit distortion

This is **data construction and inspection only**. The new `conversation_candidates_v1` format stores
original conversations separately from possible scoring targets. It is not the `id/source/kind/text`
isolated-turn format consumed by `core.score`. **Do not pass these records to `core.score`: original-context
scoring integration is a subsequent task.** Every isolated-text entry point that reads `--dataset`
(`core.score`, `core.render`, `core.add_replies`, `core.check_contamination`) exits with an error on any row
carrying `format=conversation_candidates_v1`, before loading a model or Dolci (`core.render.reject_conversation_candidates`).
No model scoring, generation, or logit-distortion analysis was performed for any build. Existing
preparation/scoring interfaces and frozen data/results are unchanged.

**Route (decided 2026-11-02 by Grace, the measure owner):** existing, original multi-exchange conversations
from OASST1, WildChat and UltraChat, each target keeping its full preceding history, are logit distortion's
**primary** data route. The isolated authorship sets (`core.prepare_data --authorship`,
`data/texts_authorship.jsonl`, unchanged) remain available for a separate comparison. The measure's
pre-registered outcome criteria and the hypothesis/verdict rules are unchanged. Final sample size, matching
criteria, source weights, and token thresholds are chosen from this audit before examining model outcomes.

**Current pilot:** `candidates/logit_distortion/pilot_v2/`, Dolci-checked in both roles (see Contamination).
`pilot_v1/` is the same sample built without the Dolci check; it is kept unchanged for provenance and
still verifies, but is superseded.

### Do not rebuild conversations from the isolated files

The isolated files cannot be joined into conversations. `pair_id` was assigned by sampling order,
independently in each file. Checked 2026-11-02 on the committed data: all 300 OASST1 `pair_id`s in
`data/texts.jsonl` (user turns) also occur in `data/texts_authorship.jsonl` (assistant replies), and 294 of
those 300 pairs have **different** OASST1 `source_id`s, so a `pair_id` join would pair unrelated messages.
(UltraChat user/assistant rows within `texts_authorship.jsonl` do share source IDs, but are single turns
without history.) The legacy isolated sampler also checked only the user text against Dolci even when it
retained an assistant reply (`prepare_data`: `is_contaminated(turns, user, "user", ...)`). Legacy data are left
unchanged; this builder checks every message in both roles.

## Sources and verified schemas

We inspected live dataset cards **and actual rows/features** at the revisions in the manifest, before
implementing adapters. Feature schemas are also saved in `manifest.json`.

| Source | Dataset / split in pilot | Actual representation | Authorship and basis |
|---|---|---|---|
| OASST1 | [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1), validation | Flat rows: `message_id`, `parent_id`, `message_tree_id`, `text`, `role` (`prompter`/`assistant`), `lang`, `review_result`, `deleted`, `synthetic` | Both roles human; crowdsourced provenance and `synthetic=false` required |
| WildChat | [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M), train | `conversation_hash`, `model`, `conversation`: list of role/content messages with `turn_identifier`, language/toxicity/redaction flags | User: **human-submitted**, a dataset-based authorship assumption, not proof of who composed every word. Assistant: model-generated reply; model name retained |
| UltraChat | [openbmb/UltraChat](https://huggingface.co/datasets/openbmb/UltraChat), train | `id`, `data`: list of strings alternating user/assistant, starting with user | Both roles model-written; the source card describes separate ChatGPT Turbo APIs producing the two roles |

The old `stingning/ultrachat` location redirects to `openbmb/UltraChat`; this build uses the canonical
original release, not a reformatted instruction subset. Dataset license metadata: OASST1 Apache-2.0,
WildChat ODC-BY, UltraChat MIT. Source attribution and original identifiers accompany the bounded sample.
WildChat IP hashes, geography, and request headers are not needed and are not exported; original message
content, conversation hashes, turn identifiers, and model attribution are retained.

## Sampling and source-level acceptance

The CLI settings are recorded in every manifest. Defaults: seed 0, **8 conversations per source**, source
window 300, minimum 2 complete user–assistant exchanges, target lengths 20–300 tokens inclusive, maximum
preceding context 4096 tokens. The conversation budget is applied **before target eligibility**. It is not
a target quota and does not imply equal counts in the four groups.

1. Resolve dataset revisions to immutable Hub commits. Read the **entire OASST1 split** (4,401 validation
   message rows in the pilot) so a message cutoff cannot cut off ancestors or descendants. Group by
   `message_tree_id`, and inspect the first 300 trees in split encounter order. This is a bounded tree
   window, not a random sample of the corpus. The manifest records the full split tree count, read row
   count, and inspected tree count separately. A large split override reads that entire split.
2. Verify each tree has one actual root, unique message IDs, all parents, and no cycles, including in
   unselected branches. Starting at the root, choose the child minimizing
   `SHA256(JSON([seed, "branch", root_id, parent_id, child_id]))` at each fork. Continue to the actual leaf.
   JSON is UTF-8, compact separators, non-ASCII preserved. This is one continuous maximal branch per root,
   independent of row ordering within the tree and of conversation budget. It is **not uniform over
   leaves**. Branch selection happens before quality/length checks; a rejected branch is not replaced
   with a better sibling. Rank is retained as metadata but not used in selection.
3. Reject the selected OASST1 branch as a whole unless every message is English (`lang=en`), positively
   reviewed, not deleted, and explicitly nonsynthetic. Never remove an intermediate message to make a
   branch pass. Reject malformed/missing ancestors rather than inventing them. An exported tree's scope
   is the source dataset, not any unavailable conversations predating that release.
4. For WildChat and UltraChat, read the first 300 rows in the pinned split's dataset iteration order, then
   stop. WildChat requires English at conversation and non-system message levels, and false toxic/redacted
   flags at both levels; unknown flags fail. UltraChat uses its published English, synthetic provenance;
   no extra automatic language detector is applied. These windows can reflect corpus ordering, topic,
   time, and shard biases. They are **not uniform samples of their whole corpora**.
5. Preserve leading system messages. Require nonempty text and alternating user/assistant messages after
   any leading systems. Reject unknown roles, later system insertions, empty messages, or malformed rows
   as whole conversations. Require at least two completed exchanges. A real trailing user turn after
   two complete exchanges stays present and can be a target; no assistant response is fabricated.
6. Reject conflicting records with the same conversation ID; collapse exact repeats of an ID explicitly.
   Rank remaining conversations by `SHA256(JSON([seed, "sample", conversation_id]))` (ID breaks ties).
   Deduplicate entire conversations by whitespace-normalized `(role, content)` sequences across sources,
   keeping the first ranked instance, then take up to the budget per source. Duplicate and budget
   exclusions retain source IDs and reasons. Duplicate detection includes above-budget candidates; a
   duplicate's referenced canonical candidate need not be selected. Sources are never relabeled to fill
   a quota. Output order is source, then conversation ID.

Source-level rejection and selection reasons are mutually exclusive first-failure reasons in
`exclusions.jsonl`. The audit reconciles inspected candidates = selected + exclusions for each source.
`adapter_accepted` counts structurally/qualitatively valid candidates before duplicate/budget selection.
No minimum target length is used to reject conversations.

## Stored records and target eligibility

- `conversations.jsonl`: one full selected branch/list, `record_type=conversation`, stable `conversation_id`,
  source ID, dataset/split/revision, original ordered messages, completed-exchange count. Each message has
  its ID, role, authorship label and basis, verbatim content, relevant native metadata, and contamination
  status. OASST1 metadata retains actual parent IDs. System authorship is `unknown` and systems are not
  scoring targets.
- `targets.jsonl`: one record for **every** user/assistant message, including ineligible messages. Its
  `target_id`, `message_id`, `conversation_id`, zero-based `turn_index`, source/revision, role, authorship,
  group, `target_text`, and `preceding_messages` make it independently inspectable. History is exactly
  `conversation.messages[:turn_index]`; the target itself and future replies are absent. First users with
  no system message have `[]`, `has_preceding_conversation=false`, and zero context tokens. No greeting is
  invented. If a source actually supplies a system message, that real preceding message is preserved.
- `turn_position=opening` means the first message **of that role**, so an opening assistant target has the
  opening user in its history. Subsequent messages of the same role are `follow_up`.
- Conversation IDs use dataset/split/native ID, never sample offsets or budgets. OASST1 message IDs use
  native UUIDs; flat-source message IDs use the native conversation identity plus original array index.
  Targets combine conversation and message identities. Changing budget cannot rename existing records.
  Keep `conversation_id` for later clustered bootstrap resampling; messages within a conversation are
  not independent observations. Different OASST1 branch seeds can choose different descendants under
  the same root, so datasets from different runs must not be merged without comparing provenance.

Target and context token counts use **`config.TEMPLATE_MODEL`**, pinned to the manifest's tokenizer revision.
No model weights are loaded. Target text is encoded verbatim without added special tokens. Context length
is the encoding of `apply_chat_template(preceding_messages, tokenize=False, add_generation_prompt=False)`
without additional special tokens; this includes template framing/default system text, if the template
adds it. Such template framing is not inserted into the stored original history. Empty history is explicitly
zero. Context length excludes the target header and target content; this is an audit bound, not yet a
model input-window guarantee. Future scoring must handle headers, boundaries, and target+context length
explicitly and verify content-token alignment. Rendering to text before encoding also avoids mistaking a
Transformers 5 `BatchEncoding` mapping's field count for its token count.

Eligibility reasons are `target_too_short`, `target_too_long`, `context_too_long`, `duplicate_target_text`,
`contamination_target`, and `contamination_context`. Reasons can overlap. An overlong history is **flagged,
never truncated**. All original messages stay in later histories, even when they fail target eligibility.
Duplicate target detection is global across selected sources/roles: the first target in output order is
canonical; later whitespace-normalized exact repeats are flagged and retain `duplicate_of`. Canonical
selection does not prefer an eligible target; it is provisional, and can change eligibility when the sample
changes (IDs and original text do not change). No duplicate-driven histories are shortened.

Normalization is only Python whitespace splitting/joining for duplicate/contamination comparison keys.
Case, punctuation, and Unicode spelling are preserved; original content is never stripped or normalized.
The candidate builder does not impose the old isolated renderer's whitespace constraints.

## Contamination

`pilot_v2` was built with `--check-dolci` against pinned revisions of all three Dolci-Instruct sets (SFT,
DPO, RL; revisions in `manifest.json`). `pilot_v1` is **unchecked**: `checked_datasets=[]` and
`status=unchecked` on every message/target. Without `--check-dolci` a build is unchecked; eligibility then
only means passing the provisional length/duplicate filters, not contamination clearance.

Per-message status (`contamination.status`, also copied onto each target for its own text):

| status | meaning |
|---|---|
| `unchecked` | no Dolci index was supplied |
| `incomplete` | no hit, but only some of the Dolci sets were indexed: **not** a negative result |
| `no_exact_match` | all three Dolci sets indexed, no whitespace-normalized exact hit in either training role and no conversation-hash hit |
| `matched` | at least one hit; `matches` lists dataset, kind and training role |

Targets record the history separately: `context_contamination_status` is `no_context` for an empty history,
otherwise the weakest status over the preceding messages (`matched` > `unchecked` > `incomplete` >
`no_exact_match`), and `contaminated_context_message_ids` lists matched history messages. The field is new in
`pilot_v2` (absent in `pilot_v1` records, whose audit still verifies).

**Exclusion policy:** a target is ineligible (`contamination_target`) if its own text matched, and
(`contamination_context`) if any preceding message matched. A matched message is **never removed** from the
conversation or from later histories, and the conversation is not altered; its later targets are flagged
instead. A WildChat conversation-hash hit marks every message of that conversation as matched. Unchecked and
incomplete statuses are not exclusion reasons by themselves, but such a build must not be described as
checked, clean, or held out.

`--check-dolci` reuses `core.check_contamination.dolci_turns` with pinned revisions of the full
named SFT/DPO/RL training splits (about 4 GB in the HF cache). It is off by default. SFT scans messages; DPO scans both chosen and rejected messages; RL scans
parsed prompt turns and WildChat conversation hashes. Every original message, **including assistant
responses independently of their user prompt**, is matched against normalized text from both training
roles. Hits record training dataset, match kind, and training role, or a conversation-hash match. Targets
also name contaminated preceding message IDs, so a contaminated context can exclude a target without
removing the underlying conversation. `--dolci-revision sft=SHA` (and dpo/rl) can pin these datasets.

The legacy `dolci_turns()` and `is_contaminated()` interfaces still work. The new optional revisions argument
and `contamination_record()` are additive. The manifest records actual checked dataset revisions and scope;
`no_exact_match` means only no match under that scope. This does not cover paraphrases, substrings, all
pretraining data, or undisclosed training data. RL coverage is prompts, not unavailable sampled completions.
Other model families' undisclosed training data remain outside this check.

## Reproduce and inspect

From repository root, use the locked project environment (`uv sync --locked`) or `/opt/arena-env`. No model
checkpoints or GPU jobs are needed (only the tokenizer files). Build to a **new directory**; the builder refuses
to overwrite any existing destination. The `pilot_v2` command adds the Dolci check to the command below:

```bash
  --check-dolci --dolci-revision sft=bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221 \
  --dolci-revision dpo=aed155cf32e809b590490b6c3577ee4b0d0a5019 \
  --dolci-revision rl=1a8d79b1012d77e4bb1fd7f699c5c7eb33ef931e
```

The `pilot_v1` (unchecked) command:

```bash
uv run python -m core.build_conversations \
  --seed 0 --conversation-budget 8 --source-window 300 --min-exchanges 2 \
  --min-target-tokens 20 --max-target-tokens 300 --max-context-tokens 4096 \
  --split oasst1=validation --split wildchat=train --split ultrachat=train \
  --revision oasst1=fdf72ae0827c1cda404aff25b6603abec9e3399b \
  --revision wildchat=7d6490e462285cf85d91eabea0f9a954fbddcd1f \
  --revision ultrachat=f220fe796ce3ed62fbe1681b45ce6cbc9c6cabe0 \
  --tokenizer-revision 6e5971d9eba42665f5bd5a0fcf047f299ce1dccc \
  --out /tmp/logit-conversation-reproduction

# Offline: verify hashes, all context prefixes, one target per message, and recompute the audit.
uv run python -m core.conversation_audit --input /tmp/logit-conversation-reproduction
uv run python -m core.conversation_audit --input candidates/logit_distortion/pilot_v1
uv run python -m core.conversation_audit --input candidates/logit_distortion/pilot_v2

# Focused local fixture tests and existing tokenizer-only compatibility tests; no model scoring.
uv run pytest tests/test_conversations.py tests/test_render_olmo.py tests/test_render_gemma.py -q
```

The CLI uses the existing Hub/datasets cache outside Git. Do not commit caches or full downloads. The
pilot was first built with the repository's `.venv/bin/python`, then rebuilt from the code on `main` with
`/opt/arena-env/bin/python -m core.build_conversations ...` (same arguments); the data files were
byte-identical, and the committed manifest is the rebuild's, with its exact Python/library versions. Code SHA256s identify builder implementations; output SHA256s
cover every generated file except the manifest itself. `--out` is deliberately excluded from output content,
so the same build can be reproduced elsewhere. The manifest is deterministic for the same source data,
code, tokenizer, settings, and package versions; there is no wall-clock timestamp in generated files.
The offline audit checks artifacts, not remote availability or original-source authenticity; pinned input
hashes and the independent source comparison recorded in the inspection notes address reproduction.

Output locations: `candidates/logit_distortion/pilot_v2/` (current, Dolci-checked) and `pilot_v1/` (same
sample, unchecked, superseded). Each holds the bounded 24-conversation sample and its 184 targets (about
1.5 MB including repeated contexts), audit JSON/Markdown, all candidate exclusion reasons, complete readable
examples, and a manifest. None are model outcomes or stored under frozen `data/` or results paths.
`examples.md` selects one shortest complete conversation per source only to make reading easier; this does
not affect selection. Hand-written inspection notes: [pilot_v2](../candidates/logit_distortion/pilot_v2/inspection.md)
(contamination findings, current counts, decisions before scoring) and [pilot_v1](../candidates/logit_distortion/pilot_v1/inspection.md)
(source and length findings, still valid for the shared sample).
