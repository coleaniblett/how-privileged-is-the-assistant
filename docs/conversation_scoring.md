# Original-conversation scoring (logit distortion)

`core.score_conversations` scores the targets of a conversation-candidate build (`core.build_conversations`,
`docs/conversation_candidates.md`) by **teacher forcing**: every recorded token of a target is scored conditional on
the target's original preceding conversation and the earlier recorded tokens of its own turn. Nothing is generated.
The scorer is `core.score.spans_nll` (unchanged): fp32, evaluation mode, NLL of the observed token and entropy of the
full next-token distribution, both in nats. The isolated-text CLI (`core.score`) is unchanged and still refuses
candidate records.

Status (2026-11-02): implemented and tested with tokenizers and the small SmolLM2 test model on CPU only. **No
OLMo, Qwen or Gemma checkpoint has been scored.** The gap (NLL − entropy) is a diagnostic, not by itself evidence of
calibration or persona privilege; the measure's pre-registered criteria are in `measures/logit_distortion/README.md`.

## Commands

From the repository root, in an environment with the locked dependencies (the commands below assume the shell is
set up so `python` is that environment's interpreter, e.g. `/root/how-privileged-is-the-assistant-grace/.venv/bin/python`).

```bash
# 1. Inputs only (tokenizer + cached model configs, no weights): verify the candidate build, render every eligible
#    target, write inputs_manifest.json, exclusions.jsonl and rendering_audit.md
HF_HUB_OFFLINE=1 python -m core.score_conversations --family olmo3-7b \
    --candidates candidates/logit_distortion/pilot_v3 --prepare-only

# 2. One checkpoint (GPU, fp32; the family's offload/batch settings). Refuses to overwrite a stage.
python -m core.score_conversations --family olmo3-7b --candidates candidates/logit_distortion/pilot_v3 --model base

# 3. Before combining stages: every stage must carry the run manifest's input fingerprint
python -m core.score_conversations --family olmo3-7b --check-stages

# 2'. After the other checkpoints have left the HF cache: take the per-stage context limits from the frozen manifest
#     (opt-in, 2026-11-03; the fingerprint check is unchanged). scripts/collect_conv_incomplete.py does this for every stage.
python -m core.score_conversations --family olmo3-7b --candidates candidates/logit_distortion/pilot_v3 --model sft \
    --context-limits-from results/logit_conversations/olmo3-7b/core/conversation_scores/inputs_manifest.json
```

Outputs go to `results/<run>/<family>/core/conversation_scores/` (`--run`, default `logit_conversations`, so they never
mix with `results/capstone/`; or `--out`). Preparing again with identical inputs confirms the manifest; different
inputs into an existing directory are refused.

## One sequence per target, original prefix only

Each target is rendered as **its original preceding messages followed by the target, and nothing after it**
(`core.render.render_target`). This is required, not an optimization: Qwen3's template renders an assistant turn
differently depending on whether a later user message exists (it keeps the empty think block only on the final
assistant turn), and causal attention cannot undo a template that already used later messages. With one sequence per
target, a target cut from a longer conversation has byte-identical inputs and scores to the same target in a
conversation that ends there (tested).

Lead-in to a target's content:

| Family | Target role | Lead-in ends with (declared in `PROFILES`, verified on every target) | History turns |
|---|---|---|---|
| OLMo 3 | user / assistant | `<\|im_start\|>user\n` / `<\|im_start\|>assistant\n` | stored form; the template's default system prompt ("You are a helpful function-calling AI assistant…") when the conversation has no original system message; an original system message replaces it, never duplicated |
| Qwen3 | user / assistant | `<\|im_start\|>user\n` / `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` | stored form; earlier assistant turns have no think block (the template drops it); rendered with `enable_thinking=False` (explicit; the default generation prompt has no think block) |
| Gemma 4 | user / assistant | `<\|turn>user\n` / `<\|turn>model\n<\|channel>thought\n<channel\|>` | stored form; earlier assistant turns have no thought channel |

An assistant target is preceded by the template's **generation prompt** for its history, i.e. exactly what the model
saw before writing the reply; a user target by the stored rendering. User targets never get an assistant opening.
Settings are explicit per family; `prepare` stops if the cached template's lead-in differs from the declaration, and
`config` defaults (e.g. `qwen3-8b.assistant_think_block`, used by the isolated route) are not changed or read.

## Exactness and exclusions

Original text is never trimmed or normalized. Edge whitespace is allowed on this route only (the isolated route
still rejects it), with strict checks kept: no special-token strings in any message; the template must place every
message verbatim (`template_changes_text` names the message it alters); each content decodes to itself; the full
string tokenizes exactly to the pieces tokenized separately (`token_merge_at_boundary` names the message where the
tokens first diverge); the target span decodes to the target text. Exclusions record `stage` (`candidate`:
ineligible in the build, not scored by design; `rendering`; `length`), `reason`, and for rendering the
`affected_message_id` / `affected_is_target`.

Length: the full rendered sequence (history, headers, target, closing) must fit the **minimum** of the family's
checkpoints' `max_position_embeddings`, read from cached configs; no limit is guessed, nothing is truncated, longer
targets are excluded (`input_too_long`). Limits: OLMo 65,536 (all stages), Qwen3 32,768 (base; instruct 40,960),
Gemma 4 262,144.

### pilot_v3 (24 conversations, 175 targets, 119 eligible), tokenizer-only preparation

| Family | Scorable | Rendering exclusions | Length exclusions | Longest input |
|---|---:|---:|---:|---:|
| OLMo 3 7B | 119 | 0 | 0 | 4,126 tokens |
| Qwen3-8B | 119 | 0 | 0 | 4,111 |
| Gemma 4 31B | 102 | 17 `template_changes_text` | 0 | 1,886 |

The pilot's only edge whitespace is trailing (5 messages); OLMo and Qwen keep it verbatim (17 eligible targets have
it in the target or history). Gemma's template trims edge whitespace, so those 17 are excluded: 2 because the target
itself changes, 15 because a history message changes. Leading newlines would be excluded for OLMo and Qwen too
(token merge; Qwen also strips them from assistant content). For cross-family comparisons use the intersection of
the families' scorable targets (here Gemma's 102). Committed audits:
`results/logit_conversations/{olmo3-7b,qwen3-8b}/core/conversation_scores/` (`rendering_audit.md`, `exclusions.jsonl`,
`inputs_manifest.json`). **Gemma's manifest is not committed:** at 23:16 UTC the overnight collection removed
`google/gemma-4-31B` (weights and config) from the shared HF cache to make room for the instruct model, so its context
limit cannot be read offline and `--prepare-only` refuses to guess. The Gemma counts above come from tokenizer-only
preparation (`tests/test_score_conversations.py::test_pilot_v3_prepares_for_every_cached_family`, limit irrelevant at
these lengths); the 262,144 limit was read from the cached config earlier the same day. Run the Gemma preparation once
that config is cached again.

**Denominators.** An earlier inspection reported 131 "cases": that counted *all* 175 pilot_v3 targets (eligible or
not) minus the 44 in the 3 conversations containing any edge-whitespace message, because it rendered whole
conversations. The scorer's denominator is the build's **119 eligible** targets; rendering only each target's own
prefix also explains Gemma's 102 = 98 eligible targets in whitespace-free conversations + 4 eligible targets that
come before the first whitespace message in the other three.

## Fingerprints and comparability

`inputs_manifest.json` holds the candidate build's manifest and output hashes, tokenizer name and revision, chat
template hash, the rendering profile, the common context limit, the included target IDs with the SHA-256 of each
complete input-token sequence and its span, and every exclusion. `input_fingerprint` hashes exactly those; code
hashes, package versions and per-checkpoint limits are recorded beside it but are not part of it. Each stage writes
`<stage>_provenance.json` (model, cached revision, device placement, batch size, the fingerprint); `--check-stages`
refuses a directory whose stages do not share the manifest's fingerprint. All stages of a family use the instruct
tokenizer and template, so their inputs are identical by construction; different families have different exclusions,
which the manifests expose.

## Outputs per stage

- `<stage>_tokens.parquet`: one row per target token: `conversation_id, target_id, message_id, source, authorship,
  role, group, turn_index, turn_position, position` (0 = first target token), `token_id`, `nll`, `entropy` (nats,
  float32). No logit arrays are stored.
- `<stage>.csv`: one row per target: the same identifiers, `n_tokens, n_input_tokens, span_start, span_end,
  input_sha256, mean_nll, mean_entropy, gap` (= mean NLL − mean entropy over all target tokens), and
  `from1_defined, n_tokens_from1, mean_nll_from1, mean_entropy_from1, gap_from1` over positions 1 onward (the same
  positions and weights for both terms; NaN with `from1_defined = False` for a one-token target).
- `<stage>_provenance.json`.

## Tests

`tests/test_score_conversations.py` (CPU; SmolLM2-135M-Instruct for model checks, cached family tokenizers
otherwise): exact histories and whitespace; leading-newline and special-token exclusions naming the message; Qwen
think block and Gemma thought channel placement with stored history, OLMo system prompt handling; Gemma trimming
exclusions; no future dependence (input IDs and scores); next-token alignment against explicit prefix-by-prefix
logits including the first target token (1e-4); batched vs single (1e-4); per-token vs per-target means and gaps
(1e-5); length exclusion without truncation; fingerprint changes for history, framing, spans, target set and limits;
stage-mix refusal; tampered candidates refused; pilot_v3 preparation counts per family.
