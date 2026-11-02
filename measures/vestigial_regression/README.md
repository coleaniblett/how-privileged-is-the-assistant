# Vestigial regression

**Shiller's form:** dominance. **Owner:** TBD.

## Status

The measurement design is **locked (2026-11-02)** with two complementary arms, following the Gemma 4 31B / Qwen3-8B /
OLMo 3 7B pilot (`pilot/report.md`, including its checks section):

- **NLL arm (capacity):** can the model still predict real user text? Teacher-forced, graded and judge-free, and
  sensitive enough for OLMo's stage-by-stage comparison.
- **Judge arm (behavior):** when the model continues a user turn, does it still play the user? This is the direct
  measure of Shiller's definition (quoted in full under [Definition](#definition)).

Build status:

- **NLL arm: built for OLMo 3 7B** (`core/` scoring, `analyze.py`; pilot numbers in `results/`, `results_preceding/`,
  `results_v0/`, all frozen). The pilot's Gemma 4 31B and Qwen3-8B scores came from the pilot's own scripts
  (`pilot/nll_arm/`); core now has both families in `config.FAMILIES` and scores them (`--family`, fp32, CPU offload for the
  31B), and the asymmetry quantity (primary, see [Pre-registered criteria](#pre-registered-criteria)) with its
  independent-set bootstrap is implemented in `analyze.py --locked` (on the pilot's scores it reproduces `pilot/report.md`;
  output in `pilot/locked/`). No capstone run has been collected with these yet.
- **Judge arm: built, not yet run.** `judge.py` and `judge_prompt.md` (v1; `claude-sonnet-5-5`, temperature 0) exist, with
  a cache, a cost estimate and a 60-sample human validation sheet (`judge_validation/sheet.csv`, labels still empty). It
  has never called the API (no key was available) and is tested with a fake client. The regex flags include `token_loop` and
  family handoff markers.
- **Pilot: done** on OLMo 3 7B, Gemma 4 31B (base and IT) and Qwen3-8B (base and IT), 90 texts per model for the
  Gemma/Qwen/OLMo comparison. See [Pilot evidence](#pilot-evidence).
- **Criteria:** the NLL-arm criteria below were revised after the pilot results, so **OLMo NLL verdicts are not
  blind**. The judge-arm criteria were set before any judge result exists. No results for this measure exist under
  `results/<run>/vestigial_regression/` yet, so the criteria are not frozen (AGENTS.md); any later change needs a
  dated entry in `docs/decisions.md`.
- Open items are collected in [TODOs and open decisions](#todos-and-open-decisions).

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

## Measurement design

The definition is behavioral, so the measure has two arms. Neither alone is the verdict: the NLL arm measures a
capacity (can the model predict a user?), the judge arm a behavior (does it still play one?), and they can
disagree.

### NLL arm (capacity)

Teacher-forced per-token NLL of real user text and real assistant text in every slot (user, assistant, plain-prose
control), per post-training stage, as change from the model's own base. For each text kind the **role-specific
term** is own slot minus plain: `user@user − user@plain` for user text, `asst@asst − asst@plain` for assistant
text. Vestigial regression is the model getting worse at user text in the user slot while the assistant is learned,
so the primary quantity is the **asymmetry** between the two role-specific changes (defined under
[Pre-registered criteria](#pre-registered-criteria)). Slot effects (`user@user − user@asst`) stay descriptive.

Why this form (from the pilot, `pilot/report.md`): the original single criterion, a rise of
`user@user − user@plain` of at least 0.05 nats, also fires for Qwen3-8B, the model chosen as the negative control, and
Qwen's rise turned out to be a cost of chat framing shared by both slots and both kinds of text (check 1). Subtracting
the assistant-text change removes that common part. Teacher forcing cannot capture drift that compounds over a sample,
which is why a behavioral arm is needed alongside it.

### Judge arm (behavior)

Prefilled, unterminated user turns (`data/prefill_100.jsonl`: 100 sentences, the first 20 being `data/prefill.jsonl`) are sampled
as in [Generation](#design-details) (stopping only at end of text), and an LLM judge labels each sample's user portion as
one of: (1) user voice; (2) switched to assistant; (3) degenerate; (4) user voice with assistant traits. Regex stays
for loops: `repetition_loop` plus a no-space `token_loop` flag, and handoff detection uses family-specific markers
(`flags.py`). Category 4 is reported to the overgeneralization measure, not counted here. **Status: built (`judge.py`), not
yet run against the API or validated.** The judge is itself an assistant deciding what counts as "user voice", hence the validation requirement in
the criteria.

### How the arms combine

Vestigial regression counts toward H4 if either arm fires; report which (see the criteria). The form comes from the
judge categories: switched or degenerate is vestigial regression; category 4 is overgeneralization in the user slot.

## Pilot evidence

Details, tables and caveats are in `pilot/report.md` (including its checks section) and `pilot/nll_arm/checks.md`;
this is a summary. All pilot scoring used 90 texts (the first 30 each of `wildchat user`, `oasst1 user`,
`wildchat assistant`), fp32, and every change is from the family's own base.

- **Gemma 4 31B (base vs IT), the positive control.** User-text role-specific term +2.94 [+2.40, +3.44] and
  +3.63 [+2.91, +4.32] nats per token (wildchat, oasst1); assistant-text role-specific term −2.17 [−2.60, −1.73];
  the base is flat (user role-specific +0.01, slot effects within 0.12 nats); 0 of 60 IT samples stay in the
  user's voice (the pilot agent's labels; a teammate check is pending). IT also degrades plain prose heavily
  (+4.6 nats), which the role-specific term subtracts.
- **Qwen3-8B (base vs IT).** A small rise (+0.44 / +0.39 nats; assistant term +0.30). Check 1 showed it is real
  (`Δ(user@user)` +0.61 / +0.36, the plain control did not fall) but mostly not user-specific: chat framing got slightly
  costlier for both slots and both kinds of text. 11 of 20 sampled continuations stay in the user's voice (7 leak
  assistant-style reasoning, 2 loop verbatim).
- **OLMo 3 7B (base to SFT, DPO, Instruct).** No regression on the NLL proxy: the user role-specific term in bits
  per byte is slightly negative (Instruct −0.07 / −0.03 on the 90 texts, −0.08 / −0.04 on all 1,190), the
  opposite direction from regression. The assistant term falls at SFT (−0.04) and is about zero at DPO and Instruct.
- **Bits per byte, change from each family's own base, 90 shared texts** (only bits per byte is compared across
  families; the full table with every column, the per-stage OLMo rows and the all-texts version are in `pilot/report.md`
  and `pilot/nll_arm/checks.md`):

  | model (IT − own base) | user role-specific, wildchat | user role-specific, oasst1 | Δ(user@plain), wildchat / oasst1 | assistant role-specific |
  |---|---|---|---|---|
  | OLMo 3 7B Instruct | −0.07 [−0.11, −0.03] | −0.03 [−0.05, −0.00] | −0.10 / +0.13 | +0.01 [−0.01, +0.03] |
  | Qwen3-8B | +0.14 [+0.09, +0.20] | +0.13 [+0.08, +0.17] | +0.06 / −0.01 | +0.09 [+0.05, +0.12] |
  | Gemma 4 31B IT | +1.10 [+0.85, +1.37] | +1.16 [+0.95, +1.38] | +1.58 / +1.58 | −0.82 [−1.15, −0.57] |

  Ordering OLMo < Qwen < Gemma. The cross-family rule pre-committed in the pilot (Gemma's change exceeds Qwen's in
  bits per byte) held; the 0.05-nat criterion alone did not separate Gemma from Qwen. For orientation, the
  asymmetry (user minus assistant role-specific change, wildchat / oasst1) from these means is about +1.9 / +2.0 (Gemma),
  +0.05 / +0.04 (Qwen) and −0.08 / −0.04 (OLMo Instruct) bits per byte; these are arithmetic on the reported means,
  with no CIs here; `pilot/locked/locked.md` has them.
- **Rendering lesson.** The assistant slot must be rendered exactly as each family's generation prompt opens an
  assistant turn. For Gemma 4 that includes the empty thought channel `<|channel>thought\n<channel|>` (now handled
  by `core/render.py`, `assistant_opening_extra`, logged in `docs/decisions.md`; OLMo rendering is byte-identical).
  Without it, the assistant slot was off-distribution and an earlier ~1-nat slot effect was an artifact (it is
  about 6 nats with the correct slot). For Qwen3 the equivalent question, whether to include the empty think block,
  is **unresolved** (the pilot used the template default, no block); see the TODOs.

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
- **Rendering** (`core/render.py`). The assistant slot opens as the family's generation prompt opens an assistant turn (`assistant_opening_extra`; empty for OLMo, `<|channel>thought\n<channel|>` for Gemma 4; Qwen3 is a TODO), and the plain control starts with the family's document-start token (`<|endoftext|>` for OLMo and Qwen, `<bos>` for Gemma). The content is tokenized on its own. The full rendered sequence must equal `tokens(prefix) + tokens(content) + tokens(suffix)`, which rules out merges at the header boundary. As a result, the content tokens are identical across all conditions, the control included. Decoding the span must reproduce the text exactly. Texts that fail these checks are filtered out at sampling time. `python -m core.render` dumps examples to `docs/rendered_examples.md`.
- **Scoring** (`core/score.py`). Batched, right-padded per-token NLL over content tokens only. Runs in fp32 everywhere (a 7B model takes ~33 GB on GPU): in bf16, per-text NLL moves by up to 0.05 nats depending on batch composition, comparable to the effects of interest. Generation runs in bf16. Gemma 4 31B and Qwen3-8B (not in `config.FAMILIES`) were scored by `pilot/nll_arm/score.py`, which imports core's rendering and span-NLL code and keeps fp32 (CPU offload for the 31B). Writes `results/<run>/core/scores/<stage>.csv` with columns `text_id, source, kind, condition, n_tokens, mean_nll, mean_nll_from1, mean_entropy, mean_entropy_from1` (`_from1`: without the first content token; entropy: of the next-token distribution at each content token), and the per-token values to `results/<run>/core/scores/<stage>_tokens.parquet` (`text_id, source, kind, condition, position, nll, entropy`; position 0 is the first content token).
- **Generation** (`core/generate.py` samples, `flags.py` here flags). Opens a user turn with each prefill sentence, leaves it unterminated, and samples 3 × 120 tokens at T=0.7, stopping only at `<|endoftext|>` so a closed turn and any handoff stay visible (`results/<run>/core/generations/<stage>.jsonl`). `flags.py` then counts four flags per sample (`results/<run>/vestigial_regression/generations/<stage>.jsonl`, `<stage>_counts.csv`): `ended_turn` (closed the user turn), `handoff_to_assistant` (then wrote the assistant header; a subset of `ended_turn`), `assistant_register` (assistant phrases such as "I'd be happy to") and `repetition_loop` (any word 4-gram repeated 3 or more times). The last two are checked on the user portion only, the text before the turn closes: a post-trained model usually finishes the user's message, closes the turn and answers as the assistant, and that reply is not the user's voice.
- **Analysis** (`analyze.py` here; loaders, bootstrap and the negligibility rule in `core/stats.py`). Per text set (source × kind) and stage, with 95% bootstrap CIs over texts (paired across slots):
  - raw mean NLL in each slot (user, assistant, plain);
  - slot effect = NLL(user slot) − NLL(assistant slot): `user@user − user@asst` for user text, `asst@user − asst@asst` for assistant text;
  - role-specific = NLL(own slot) − NLL(plain): `user@user − user@plain` for user text, `asst@asst − asst@plain` for assistant text;
  - mean next-token entropy per slot.

  Each is reported over content positions 1+ (the default, also used by the figures) and over all tokens. When an assistant turn directly follows the system turn, the post-trained checkpoints put nearly all next-token probability on `<|im_start|>` (1.000 SFT, 0.960 DPO, 0.881 Instruct), so the first content token in the assistant slot costs 2–14 nats whatever the text.

  The asymmetry (primary quantity) is computed by `--locked` (see the module docstring), not by this default run. Writes `results/<run>/vestigial_regression/results.md`, `summary.csv` and one figure per text set, `nll_<source>_<kind>.png` (stage on x, one line per slot).
- **Position analysis** (`analyze.py`, from the per-token NLLs). Per text set and stage, over the texts with at least 40 content tokens (so every position averages over the same texts): mean NLL at each of the first 40 content-token positions in each slot, and the slot difference NLL(user slot) − NLL(assistant slot) by position with 95% bootstrap CIs. The slot difference counts as negligible from the first position p such that every per-position mean from p to 39 is within ±0.05 nats and their mean over positions p..39 has a 95% CI inside ±0.05 nats (`--positions`, `--margin`; the margin is `config.MIN_EFFECT_NATS`). Writes `position.md` and `position_<source>_<kind>.png` next to `results.md`.
- **Breadth arm** (not in `config.FAMILIES` yet; see AGENTS.md). Base vs instruct per family (Gemma 4, Qwen3),
  each family rendered with its own instruct tokenizer and chat template. The pilot covered Gemma 4 31B and Qwen3-8B
  (NLL scores, and generations from the exploration run). Contamination cannot be checked for
  these families: every table and figure with their results carries that caveat.

The role-swapped conversations and the coin-toss spillover that ran alongside the pilot are now `extras/`
(design in `extras/README.md`); whether they belong to this measure is undecided.

### What the result pattern would mean

- **Base:** the role-specific terms and slot effects are near zero (Gemma 4 31B base: flat), since base has no learned
  role asymmetry. This is the reference point.
- **Assistant text:** if post-training is learning the assistant role, `asst@asst − asst@plain` falls (Gemma 4 31B IT: −2.17 nats).
- **User text (the vestigial-regression signal):** `user@user − user@plain` rises while the assistant term falls or
  stays flat, so the asymmetry is positive and large (Gemma). A rise in both terms (Qwen3-8B) is a common chat-framing cost
  and the asymmetry stays small.
- **Which stage:** the stage where the asymmetry first appears is the answer. SFT is the prime suspect because its loss
  masks user tokens, so user modeling drifts untrained. If the change appears at DPO or RL instead, preference/RL
  optimization is doing it.
- **Controls and the plain-text row:** plain prose is itself a non-assistant context. Its change from base
  (`user@plain`, `asst@plain`) is reported as its own row, "non-assistant degradation" (Gemma IT: +1.58 bits per byte).
  The role-specific subtraction removes it, and NLL cannot tell whether it reflects vestigial regression or
  overgeneralization, so it is referred to the overgeneralization measure.
- **Judge arm:** a rising non-user-voice rate across stages, with or without an NLL asymmetry.

## Pre-registered criteria

Revised 2026-11-02, **pilot-informed** (see [Status](#status) and `docs/decisions.md`): the NLL-arm criteria were
changed after the OLMo, Qwen and Gemma pilot results, so OLMo NLL verdicts are not blind. The judge-arm criteria were
set before any judge result exists. No `results/<run>/vestigial_regression/` exists yet; after it does, any change
here needs a dated entry in `docs/decisions.md`, and the README notes the date. Superseded by this revision: the
single-criterion verdict (a rise of `user@user − user@plain` alone), the regex classification rule (old criterion 4)
and the unbuilt question-rate and `user_voice_tics` flags.

### NLL arm

Every quantity is per text set (user text: `wildchat user`, `oasst1 user`; assistant text: `wildchat assistant`, with
`olmo-instruct` as a secondary OLMo-only set), per stage, over content positions 1+, as change from the model's own
base. Units: nats per token within a family (OLMo stages); **bits per byte** for any cross-family comparison.

- **Primary quantity, the asymmetry.** `A = Δ(user role-specific) − Δ(assistant role-specific)`, where each
  role-specific term is own slot minus plain (`user@user − user@plain`; `asst@asst − asst@plain`) and Δ is the change
  from base. User and assistant texts are different texts, so the 95% bootstrap CI resamples the two text sets
  **independently**, while pairing within each set across checkpoints (per-text Δ). One asymmetry per user source,
  each against the assistant set. Implemented in `analyze.py --locked`.
- **Verdict.** The asymmetry is at least 0.05 nats with the 95% CI excluding zero, **in the same direction for both user
  sources**. 0.05 nats is a **detectability floor, not an effect-size threshold**: report the magnitude as the primary
  result. No new threshold tuned on the pilot models is introduced. Qwen may register as mildly regressed, which is
  consistent with its generations (9 of 20 sampled continuations left the user's voice).
- **Always report alongside:** `Δ(user@user)`, `Δ(user@plain)`, `Δ(asst@asst)` and the user role-specific term on its
  own. Report the plain-text change from base as its own row ("non-assistant degradation") and refer it to the
  overgeneralization measure (see [What the result pattern would mean](#what-the-result-pattern-would-mean)).
- **Slot effects** (`user@user − user@asst`) stay descriptive and do not enter the verdict. Reported by position as well
  (`position.md`), since the first positions carry the header effect.

### Judge arm

- **Generation.** Prefilled, unterminated user turns, sampled as in [Generation](#design-details), stopping only at end
  of text. The prefill set is `data/prefill_100.jsonl` (100 sentences).
- **Judge.** An LLM judge labels each sample's user portion: (1) user voice; (2) switched to assistant; (3) degenerate;
  (4) user voice with assistant traits. One judge model, its exact prompt and temperature 0, all recorded. The judge
  must not be one of the evaluated models. Chosen: `claude-sonnet-5-5` at temperature 0 with the prompt in `judge_prompt.md`
  (v1), recorded in every output row.
- **Regex.** `repetition_loop` plus a no-space `token_loop` flag (ported from `exploration/shiller_user_turns/flags.py`).
  Handoff detection uses family-specific markers: the turn markers derived from each template, Gemma's `<|channel>` and
  `<channel|>`, Qwen's `<think>` and `</think>` (`flags.py`).
- **Primary quantity.** The non-user-voice rate (categories 2 + 3), change from base, with a 95% bootstrap CI resampling
  **prefills** (samples per prefill are correlated). The arm fires if the rate rises with the CI excluding zero.
  **OPEN DECISION:** the minimum rate difference that counts. The category 4 rate goes to the overgeneralization measure.
- **Judge validation before use.** At least 50 human-labeled samples spanning Gemma, Qwen and OLMo, base models
  included. **The labels must be checked by a teammate**: the existing pilot labels were written by the agent that
  ran the pilot. Report raw agreement and Cohen's kappa. Default bar (the team can change it before running):
  revise the judge prompt if agreement falls below 80%.

### Combining the arms

Vestigial regression counts toward H4 if **either arm fires**; report which:

- **both arms:** capacity loss with behavioral collapse;
- **NLL only:** the model got worse at predicting users without visibly failing to play them;
- **judge only:** the model fails to play the user while still predicting user text well.

The form comes from the judge categories (switched or degenerate: vestigial regression; category 4: overgeneralization
in the user slot). If the NLL arm fires with no judge classification, report "user-persona shift, unclassified."

**Pre-registered prediction.** The two arms agree in ordering across all model/stage points (OLMo base, SFT, DPO,
Instruct; Qwen base and IT; Gemma base and IT). Report the Spearman correlation between the NLL asymmetry and the
judge's non-user-voice rate, stating plainly that with about 8 points it is descriptive, not a significance test.
Disagreements are reported as findings, not errors. The asymmetry and the judge rate are both changes from base, so the
base points are zero by construction; `analyze.py --locked` leaves them out of the correlation and uses one NLL asymmetry per
post-trained point (bits per byte, mean of the two user sources), so about 5 points remain. This choice was made when the
analysis was implemented, not pre-registered with the design; change it before looking at the data if you disagree.

### Breadth arm and honesty

Keep the breadth-arm rule: the same criteria apply to instruct vs base within each family, with the contamination
caveat (contamination cannot be checked for Gemma or Qwen; say so wherever their results appear).

## Future work

- **Decision-point likelihood (turn-end hazard).** Likelihood at the positions where the behavior is decided: the
  probability of the end-of-turn token where the real user kept going (split into sentence boundaries and mid-sentence),
  the survival probability per text (product of 1 − hazard), calibrated against how often real WildChat users end their
  message at each boundary type, plus the probability of assistant openers at boundaries. **Not planned for the
  capstone.** It was a fallback in case mean NLL failed the Gemma positive control, and mean NLL passed. Designed only;
  nothing is implemented, and any OLMo criteria for it would need to be written here first.
- **Question-rate flag** (rate of questions in the user portion against the real turns the prefills were cut from):
  specified earlier, never built, not part of the locked design.
- **Role-swapped conversations and coin-toss spillover** are `extras/` (design in `extras/README.md`); whether they
  belong to this measure is undecided.

## TODOs and open decisions

TODOs:

- Run judge validation against teammate-checked labels (at least 50 samples; Gemma, Qwen, OLMo incl. base): a teammate fills
  `judge_validation/sheet.csv`, then `judge validate`; revise `judge_prompt.md` (and its version) if agreement is below the bar.
- Decide the Qwen3 assistant-slot rendering (with or without the empty think block; the `assistant_think_block` option in
  `core/config.py`, default off as in the pilot) to match its generation prompt, and apply it consistently: it changes the
  assistant role-specific term and therefore the asymmetry, and it also affects logit distortion's `asst@asst`.
- Generate and judge samples at every stage of every family (`docs/runbook_2026-11-03.md`), then run `analyze.py --locked`.
- Move the pilot's `nll_arm/` scripts' remaining uses into core/`analyze.py` (core scoring now covers both families); keep
  `pilot/` as the record of the pilot.

Done since the design was locked (2026-11-02): judge model and prompt (`judge.py`, `judge_prompt.md`); the asymmetry in `analyze.py
--locked`; `token_loop` and family handoff markers (`flags.py`); the 100-sentence prefill set (a core change, logged); Gemma 4 31B and
Qwen3-8B in `config.FAMILIES`.

Open decisions:

- The minimum judge-arm rate difference that counts.
- Decided in `analyze.py --locked` (not a pre-registered choice): the base points are left out of the Spearman correlation between
  the arms (zero by construction), and the correlation uses one NLL asymmetry per point (bits per byte, mean of the two user sources).
- The 80% judge-agreement bar (a default the team may change before running).

## Running

```bash
bash scripts/run_gpu.sh                                   # whole pipeline -> results/olmo3-7b/
uv run python -m measures.vestigial_regression.flags      # flags on results/olmo3-7b/core/generations/
uv run python -m measures.vestigial_regression.analyze    # tables + figures from results/olmo3-7b/core/scores/
```
