# Vestigial regression

**Shiller's form:** dominance. **Owner:** TBD.

## Status

- **OLMo 3 7B pilot: exists.** `results/results.md` and `results/position.md` (held-out sample, no preceding
  turn) and `results_preceding/results.md` (preceding-turn robustness run) hold the pilot numbers;
  `results_v0/` is the earlier unfiltered sample. These directories are frozen; new runs go to
  `results/<run>/vestigial_regression/`. The OLMo criteria below were refined *after* seeing this pilot, so
  OLMo verdicts are **not blind**.
- **Gemma 4 31B positive-control pilot: in progress.** Its purpose is to validate the measurement proxy (see
  [Measurement approaches](#measurement-approaches-and-their-validation)).
  - Generation arm (prefilled user turns, flagged): done.
  - NLL slot-effect scoring: running. **Result pending.**
  - **TODO: Gemma 4 31B base run.** No Gemma base has been scored or sampled. Every criterion is a change
    from base, so the validation cannot be read until it exists.
- **Spec:** draft, being refined. Once results for this measure exist under
  `results/<run>/vestigial_regression/`, the pre-registered criteria are frozen (AGENTS.md). None exist yet.

## Definition

Shiller defines vestigial regression as follows (Derek Shiller, "Privilege, Dominance, and Personas," Eleos AI
Substack, Aug 27, 2026, <https://eleosai.substack.com/p/privilege-dominance-and-personas>):

> “Instead of exhibiting assistant-like text across the board, the model might just descend into very simple
> language or even incoherence when in non-assistant contexts. Tasked with continuing a user turn, we might see
> confused, fragmented, or incoherent thoughts expressed. The model might lose the ability to take initiative or
> ask questions, even forgetting whether it should be asking or answering its own questions.”

- **The definition is behavioral.** It describes what the model produces when continuing a user turn.
  Shiller's own gloss of his Gemma 4 31B observations is that the model loses the ability to predict the
  user, or predicts only that the user will hand off to the assistant.
- **Classification (Shiller's, which our flags follow).** Switching into the assistant register counts as
  vestigial regression, even without turn markers. Assistant traits appearing in text that stays in the
  user's voice (his Qwen 3 example) count as *overgeneralization* instead, and belong to that measure.
- **The phenomenon is real in at least one model.**
  `results/exploration/shiller_user_turns/gemma4-31b-it.jsonl` holds 205 Gemma 4 31B IT user-turn
  continuations (5 samples per prompt, T=0.7, 200 new tokens, int8) reproducing Shiller's observations. They
  are exploratory: the flags are regex heuristics from `exploration/shiller_user_turns/flags.py` (not this
  measure's `flags.py`), not yet checked against hand labels, and nothing here enters a verdict. Samples
  flagged, by condition:

  | condition | prompts | samples | `repetition_loop` | `token_loop` | `assistant_register` | `ended_turn` | `handoff_to_assistant` | contains `<\|channel>thought` |
  |---|---|---|---|---|---|---|---|---|
  | `empty` (header only) | 1 | 5 | 4 | 4 | 2 | 2 | 0 | 2 |
  | `ordinary` (WildChat first sentences) | 20 | 100 | 42 | 27 | 26 | 26 | 0 | 71 |
  | `register` (simple / non-native English) | 10 | 50 | 38 | 40 | 12 | 3 | 0 | 17 |
  | `norm` (mild norm violations) | 10 | 50 | 0 | 0 | 35 | 0 | 0 | 50 |

  What it shows: Gemma either loops (word- or token-level repetition, most often on the `register` prompts) or
  opens a thinking channel (`<|channel>thought`) and answers as the assistant (every `norm` sample) instead of
  continuing as the user. `handoff_to_assistant` is 0 throughout because the switch happens without the
  assistant header. The gallery is `results/exploration/shiller_user_turns/gallery.md`.

## Measurement approaches and their validation

The definition is behavioral; each approach below is a proxy for it. They are tried in sequence.

### a. Original proxy: per-token NLL of real user text across slots

Mean per-token NLL of real user text in the user slot, relative to the plain-prose control and to base
(`user@user − user@plain`, change from base), with assistant text as the comparison. Rationale: if the model
loses the ability to predict users, real user text becomes less likely in the user slot.

**Status:** built and piloted on OLMo 3 7B; being validated on Gemma 4 31B (c).

### b. Two potential flaws of the original proxy

- **No generational compounding.** Teacher forcing conditions on real text at every step, so it cannot capture
  drift: a sample leaves user-like text and the model then conditions on its own drift (how Gemma ends up
  handing off or looping).
- **Dilution by averaging.** The collapse may hinge on a few decision points (whether to end the turn, whether
  to switch register) that carry almost no weight in a mean over every token.

Together these mean a small distribution shift can produce a dramatic behavioral collapse with only a modest
change in mean NLL.

### c. Validation on Gemma (pre-committed reading)

Gemma 4 31B IT reliably shows vestigial regression in generation (see [Definition](#definition)), so it is a
positive control. Reading, committed before the NLL result:

- If mean NLL rises clearly (IT vs base), the proxy is valid and is used for OLMo's stage-by-stage analysis,
  where effects may be too subtle to see in generation.
- If it barely moves, the proxy fails for Shiller's behavioral definition and is demoted to descriptive.

**Status:** pending. Needs the Gemma base run (TODO above). Open: "clearly" and "barely" have no numeric
threshold yet.

### d. Refinement: decision-point likelihood (designed, not implemented)

Measure likelihood at the positions where the behavior is decided instead of averaging over all tokens:

- **Turn-end hazard:** the probability the model assigns to the end-of-turn token at positions where the real
  user kept going, reported separately at sentence boundaries (after `.` `?` `!` or a newline) and
  mid-sentence.
- **Survival probability per text:** the product of (1 − hazard) over continuation positions, i.e. how likely
  the model is to sustain the message as long as the real user did.
- **Calibration target:** the empirical rate at which real WildChat users end their message at each boundary
  type, so we can say which model over- or under-predicts users ending their turn.
- **Sibling measure:** the probability of assistant openers ("Sure", "Certainly", "I'd be happy to", …) at
  boundaries, which covers Gemma's switch into the assistant register without a turn marker.

**Exploratory.** It was designed after seeing Gemma's generations, so its OLMo criteria must be written
before any OLMo analysis with it.

### e. Candidate: LLM judge on generations

If the likelihood proxies fail, the generation measure becomes primary: an LLM judge classifies each sample's
perspective (user voice / switched to assistant / degenerate / user voice with assistant traits), with the
regex kept for `repetition_loop`. The judge is validated against human labels
(`results/exploration/shiller_user_turns/hand_labels.csv`) and its agreement rate is reported. That sheet
currently holds 50 drawn Gemma samples with the `perspective` column still empty, and its label set (user
voice / switched to assistant / degenerate / other) has no "user voice with assistant traits" category.

Caveat: the judge is itself an assistant deciding what counts as "user voice."

**Status:** candidate; not implemented.

### f. Decision sequence

Mean-NLL proxy (a) → decision-point likelihood (d) if it fails → LLM judge plus regex (e) if that fails.

## Design details

The sampling, rendering and scoring steps are core's (shared with the other measures); the analysis and the
generation flags are this measure's. Primary models: the OLMo 3 7B Instruct line at each post-training stage
(base, SFT, DPO, Instruct; checkpoints in `core/config.py`, verified in `docs/models.md`).

- **Data** (`core/prepare_data.py` → `data/texts.jsonl`, `data/prefill.jsonl`; `core/add_replies.py`). Each row of `texts.jsonl` is one text with a `source` and a `kind` (user or assistant). Sources are analyzed separately and never pooled:

  | source | kind | n | what |
  |---|---|---|---|
  | `wildchat` | user, assistant | 300 pairs | first user turn and first assistant reply (ChatGPT) of English `allenai/WildChat-1M` conversations, toxic/redacted dropped |
  | `oasst1` | user | 300 | English `OpenAssistant/oasst1` root prompts, toxicity < 0.5 |
  | `olmo-instruct` | assistant | ≤ 300 | OLMo 3 7B Instruct's greedy reply to each `wildchat` user turn, cut at 300 new tokens; cut replies are kept and marked `truncated` (causal attention: their NLL is that of the reply's first 300 tokens) |

  **Held out from post-training.** OLMo 3's SFT data contains most of WildChat (86% of an unfiltered sample's user turns appear verbatim) and some oasst1, so `prepare_data.py` skips any conversation whose user turn appears in the Dolci-Instruct SFT, DPO or RL data (`core/check_contamination.py`: exact match after whitespace normalization; RL prompts are parsed out of their `user: ...` strings and also matched on WildChat's `conversation_hash`), then keeps sampling (seed 0) until 300 clean texts per source pass the other filters. Each text must be 20–300 tokens by the OLMo tokenizer and render cleanly (see below). `allenai/WildChat-1M` is no longer gated. Rows record `pair_id` (links a WildChat user turn, its reply and OLMo's reply) and `source_id`. The prefill set is the first sentence of 20 sampled WildChat user turns (distinct sentences; different conversations can share an opening). `--pairs oasst1 --extra-user none --no-exclude-dolci` gives the small oasst1-only set the smoke test uses. `core/selection_check.py` checks that the filter did not select different texts (length, base plain-prose NLL) than the unfiltered sample: `results/selection_check.md`.
- **Conditions** (`core/config.py`, `SLOTS`). Every text is scored in every slot; the condition name is `<kind>@<slot>`, e.g. `user@asst` is user text in the assistant slot. Each text is scored alone as the first turn, so the role header is the only difference between slot conditions before the scored span (the Instruct template also prepends a fixed default system turn, identical in every slot condition). After the span the template closes a user turn with `<|im_end|>\n` and a final assistant turn with `<|endoftext|>`; this suffix follows the scored tokens, so under causal attention it cannot change their NLL (checked in the smoke test).

  | slot | user text | assistant text |
  |---|---|---|
  | user | `user@user` | `asst@user` (swap) |
  | assistant | `user@asst` (swap) | `asst@asst` |
  | none: plain prose after `<\|endoftext\|>` (control) | `user@plain` | `asst@plain` |

  `--preceding-turn` (off by default; `config.PRECEDING_TURN`) puts one fixed neutral turn of the opposite role before the scored turn, since a lone assistant turn is itself unusual input. This adds a second difference between slots (the preceding turn's text), so treat it as a robustness check.
- **One template for every stage.** All stages, base included, are scored with the Instruct tokenizer and chat template, so token sequences are identical across stages. Base is the symmetric baseline. Caveat: the chat special tokens may be poorly trained in base.
- **Rendering** (`core/render.py`). The content is tokenized on its own. The full rendered sequence must equal `tokens(prefix) + tokens(content) + tokens(suffix)`, which rules out merges at the header boundary. As a result, the content tokens are identical across all conditions, the control included. Decoding the span must reproduce the text exactly. Texts that fail these checks are filtered out at sampling time. `python -m core.render` dumps examples to `docs/rendered_examples.md`.
- **Scoring** (`core/score.py`). Batched, right-padded per-token NLL over content tokens only. Runs in fp32 everywhere (a 7B model takes ~33 GB on GPU): in bf16, per-text NLL moves by up to 0.05 nats depending on batch composition, comparable to the effects of interest. Generation runs in bf16. Writes `results/<run>/core/scores/<stage>.csv` with columns `text_id, source, kind, condition, n_tokens, mean_nll, mean_nll_from1, mean_entropy, mean_entropy_from1` (`_from1`: without the first content token; entropy: of the next-token distribution at each content token), and the per-token values to `results/<run>/core/scores/<stage>_tokens.parquet` (`text_id, source, kind, condition, position, nll, entropy`; position 0 is the first content token).
- **Generation** (`core/generate.py` samples, `flags.py` here flags). Opens a user turn with each prefill sentence, leaves it unterminated, and samples 3 × 120 tokens at T=0.7, stopping only at `<|endoftext|>` so a closed turn and any handoff stay visible (`results/<run>/core/generations/<stage>.jsonl`). `flags.py` then counts four flags per sample (`results/<run>/vestigial_regression/generations/<stage>.jsonl`, `<stage>_counts.csv`): `ended_turn` (closed the user turn), `handoff_to_assistant` (then wrote the assistant header; a subset of `ended_turn`), `assistant_register` (assistant phrases such as "I'd be happy to") and `repetition_loop` (any word 4-gram repeated 3 or more times). The last two are checked on the user portion only, the text before the turn closes: a post-trained model usually finishes the user's message, closes the turn and answers as the assistant, and that reply is not the user's voice.
- **Analysis** (`analyze.py` here; loaders, bootstrap and the negligibility rule in `core/stats.py`). Per text set (source × kind) and stage, with 95% bootstrap CIs over texts (paired across slots):
  - raw mean NLL in each slot (user, assistant, plain);
  - slot effect = NLL(user slot) − NLL(assistant slot): `user@user − user@asst` for user text, `asst@user − asst@asst` for assistant text;
  - role-specific = NLL(own slot) − NLL(plain): `user@user − user@plain` for user text, `asst@asst − asst@plain` for assistant text;
  - mean next-token entropy per slot.

  Each is reported over content positions 1+ (the default, also used by the figures) and over all tokens. When an assistant turn directly follows the system turn, the post-trained checkpoints put nearly all next-token probability on `<|im_start|>` (1.000 SFT, 0.960 DPO, 0.881 Instruct), so the first content token in the assistant slot costs 2–14 nats whatever the text.

  Writes `results/<run>/vestigial_regression/results.md`, `summary.csv` and one figure per text set, `nll_<source>_<kind>.png` (stage on x, one line per slot).
- **Position analysis** (`analyze.py`, from the per-token NLLs). Per text set and stage, over the texts with at least 40 content tokens (so every position averages over the same texts): mean NLL at each of the first 40 content-token positions in each slot, and the slot difference NLL(user slot) − NLL(assistant slot) by position with 95% bootstrap CIs. The slot difference counts as negligible from the first position p such that every per-position mean from p to 39 is within ±0.05 nats and their mean over positions p..39 has a 95% CI inside ±0.05 nats (`--positions`, `--margin`; the margin is `config.MIN_EFFECT_NATS`). Writes `position.md` and `position_<source>_<kind>.png` next to `results.md`.
- **Breadth arm** (not in `config.FAMILIES` yet; see AGENTS.md). Base vs instruct per family (Gemma 4, Qwen3),
  each family rendered with its own instruct tokenizer and chat template. Contamination cannot be checked for
  these families: every table and figure with their results carries that caveat.

The role-swapped conversations and the coin-toss spillover that ran alongside the pilot are now `extras/`
(design in `extras/README.md`); whether they belong to this measure is undecided.

### What the result pattern would mean

- **Base:** both slot effects should be near zero, since base has no learned role asymmetry. This is the reference point.
- **Assistant text:** if post-training is learning the assistant role, `asst@asst` should fall and the assistant-text slot effect should become clearly positive.
- **User text (the vestigial-regression signal):** if the model still models users in the user slot, the user-text slot effect should be negative (`user@user < user@asst`). Vestigial regression would show up in three ways:
  - `user@user` rises across stages relative to base and relative to `user@plain`.
  - The user-text slot effect shrinks to zero or flips positive. A flip means user text is *more* expected in the assistant slot, so the model treats every slot as its own.
  - Prefilled user turns increasingly loop, hand off to the assistant, or slip into assistant register.
- **Which stage:** the stage where the change first appears is the answer. SFT is the prime suspect because its loss masks user tokens, so user modeling drifts untrained. If the change appears at DPO or RL instead, preference/RL optimization is doing it.
- **Controls:** if `*@plain` moves as much as the slot conditions, the change is general distribution drift, not role-specific. The role-specific part is `user@user − user@plain`.

## Pre-registered criteria

Draft, to be refined before the capstone's OLMo run; last revised 2026-11-02. Refined after seeing the OLMo pilot (see
[Status](#status)). After results for this measure exist under `results/<run>/vestigial_regression/`, any
change here needs a dated entry in `docs/decisions.md`. These criteria are for the mean-NLL proxy (a) and the
generation flags; if the decision sequence (f) moves to (d) or (e), their OLMo criteria must be written here
before any OLMo analysis with them.

Every quantity is per text set (user text: `wildchat user`, `oasst1 user`; assistant text as a comparison),
per stage, reported as the change from base with a 95% bootstrap CI paired over texts, over content positions
1+. The smallest effect we count is 0.05 nats (AGENTS.md).

1. **Role-specific NLL of user text**, `user@user − user@plain`, relative to base. Vestigial regression: it
   rises (user text becomes *less* expected in the user slot than the role-free control explains) by at least
   0.05 nats, with the CI excluding zero and the same direction in both user sources (`wildchat user` and
   `oasst1 user`). The same quantity for assistant text (`asst@asst − asst@plain`) is the comparison: a fall
   there with a rise for user text is the asymmetry Shiller describes.
2. **Slot effect for user text**, `user@user − user@asst`, relative to base. **Descriptive only**; it does not
   enter the verdict. Reported by position as well (`position.md`), since the first positions carry the header
   effect.
3. **Generation flags** on the user portion of prefilled user turns, per stage, with the pilot's four flags,
   plus two spec additions:
   - **question-rate flag (not implemented yet):** the rate of questions in the user portion of the samples,
     compared against the same rate in the real user turns the prefills were cut from (`data/prefill.jsonl` →
     `text_id` in `data/texts.jsonl`);
   - **`user_voice_tics` (planned, not implemented):** assistant traits in text that stays in the user's
     voice, feeding the overgeneralization classification in criterion 4.

   **Decision rule:** each flag's rate, reported as the change from base with a 95% bootstrap CI resampling
   prefills. **Open decision:** the minimum rate difference that counts.
4. **Classification rule** (spec addition, **not implemented yet**): a rise in any one of `repetition_loop`,
   `handoff_to_assistant` or `assistant_register` → vestigial regression (switching into the assistant register
   counts even without a handoff or turn marker). Assistant traits in text still in the user's voice →
   overgeneralization, detected by the planned `user_voice_tics` flag or the LLM judge (e), and reported to the
   overgeneralization measure rather than counted here.

**Verdict for the hypothesis table.** Criterion 1 gives the dominance verdict: vestigial regression is present
at a stage if criterion 1 fires there (≥ 0.05 nats, CI excluding zero, same direction in both user sources).
The flags (criteria 3–4), or the judge if adopted, classify its form. If criterion 1 fires with no
classification, report "user-persona shift, unclassified." The first stage where criterion 1 fires is the
answer to "which stage". Controls: a change in `user@plain` of the same size as in `user@user` is general
distribution drift, not role-specific, and does not count.

For the breadth arm, the same criteria apply to instruct vs base within each family, with the contamination
caveat.

## Running

```bash
bash scripts/run_gpu.sh                                   # whole pipeline -> results/olmo3-7b/
uv run python -m measures.vestigial_regression.flags      # flags on results/olmo3-7b/core/generations/
uv run python -m measures.vestigial_regression.analyze    # tables + figures from results/olmo3-7b/core/scores/
```
