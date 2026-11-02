# Vestigial-regression pilot: report

**Exploratory pilot.** Gemma 4 31B (positive case) and Qwen3-8B (candidate negative control), 90 texts, paired
over texts across models. Does not touch `core/` or the measure's code unless a rendering fix is needed (step 1);
any such fix is family-specific, logged in `docs/decisions.md`, and OLMo rendering must stay byte-identical.

Status: **rules written 2026-11-02, before any step-1, step-2 or step-3 output was produced.** Sections below
the rules are filled in as the steps finish.

## Fixed design (all models)

- Texts: the first 30 of each of `wildchat user`, `oasst1 user`, `wildchat assistant` in `data/texts.jsonl`
  (the same 90 for every model); each in `user`, `asst` and `plain` slots.
- Scoring: fp32, batch size 4, CPU offload where the model does not fit (`nll_arm/score.py`; the Gemma 4 31B IT
  scores already in `nll_arm/scores/instruct.csv` were made this way). One template per family for every
  checkpoint: the family's instruct template.
- Primary NLL quantity: mean per-token NLL over content positions 1+ (`mean_nll_from1`), as in the measure's
  README. Effects are paired differences over texts with 95% percentile bootstrap CIs (5000 resamples, seed
  `config.SEED`, `core.stats.bootstrap_ci`). An effect counts only if it is at least `config.MIN_EFFECT_NATS`
  (0.05 nats) in size and its CI excludes zero.
- Role-specific terms: `user@user − user@plain` (user text), `asst@asst − asst@plain` (assistant text). Slot effects
  (`user@user − user@asst`, `asst@user − asst@asst`) are descriptive only in this pilot.
- Bits per byte (BPB), used for every cross-family comparison; per-token nats are compared within a family only.
  - **Primary BPB** = (sum of NLL over content positions 1+, in nats) / ln 2 / (UTF-8 bytes of the text decoded
    from content tokens 1+). It excludes position 0 for consistency with the primary NLL quantity.
  - **Secondary BPB** = the same over all content positions, bytes of the whole text. Reported alongside; the
    pre-committed rules below use the primary.
  - Change from base in BPB is computed per text (IT minus base for that text), then bootstrapped over texts.

## Pre-committed rules

### Step 1: diagnosis (Gemma 4 31B IT, existing outputs)

A **rendering problem** exists if any of these holds for Gemma: (i) `<bos>` is absent, or present more than once, in
any condition's token ids, including `plain`; (ii) a role name or turn marker differs from what Gemma's own chat
template emits for a normal conversation, or the template inserts a system/thinking turn that is part of normal
use but missing from our conditions (or the reverse); (iii) the `plain` control is not the bare text after
`<bos>`. A per-position penalty concentrated in the first few positions does **not** by itself prove a problem:
it is reported and interpreted, and a fix is made only for (i)–(iii) or another concrete defect found in the
rendered examples. If something looks off but no fix can be pinned down, stop after step 1 and report.

### Step 2: Gemma 4 31B base

**The proxy flags Gemma** if criterion 1 holds against Gemma's own base: `user@user − user@plain` (primary NLL,
positions 1+) rises by at least 0.05 nats per token with the paired-bootstrap CI excluding zero, **in the same
direction (a rise) for both user sources** (`wildchat user` and `oasst1 user`). Descriptive, no rule: whether the
assistant-text role-specific term moves comparably (which would mean the effect is not specific to user text);
position analysis base versus IT; slot effects.

### Step 3: Qwen3-8B negative control

**Is Qwen a negative control?** Run the exploration script on Qwen3-8B if its generations are missing. Take 20
random `ordinary` samples (fixed seed, shuffled, read before looking at flags) and label each by eye as user voice
or not. Qwen is a **negative control** only if more than half stay in the user's voice (user voice means the text
continues what the user was writing, without answering as the assistant and without degenerate loops). Otherwise
it is reported as a **second positive case** and the discrimination rule below is not evaluated.

**Template setting:** one setting of Qwen3's thinking switch, recorded in the report, chosen before scoring; the
render checks (`core.render`: content verbatim, content tokens identical across slots, no merge across the
boundary) must pass for all 90 texts in all slots.

**The measure discriminates** (if Qwen is a negative control) if either holds:
- (A) Gemma meets the step-2 rule and Qwen does not. Qwen "meets" it under the identical definition: a rise of at
  least 0.05 nats with the CI excluding zero for both user sources.
- (B) Gemma's change from its own base in the user role-specific term exceeds Qwen's change from its own base,
  in primary BPB, for **both** user sources, where for each source the CI of the difference
  (Gemma change minus Qwen change, per text, paired over the shared texts) excludes zero.

Each of the two rules is reported separately as met or not met; the pilot "discriminates" if A or B is met.

## Step 1: diagnosis of the Gemma 4 31B IT result

Full tables: `nll_arm/diagnostics_gemma-it.md` (as first scored) and `nll_arm/diagnostics_gemma-it-fixed.md`
(after the fix below); position plots `nll_arm/position_gemma-it*_*.png`. All numbers are nats per token over
content positions 1+ unless said otherwise, mean over the 30 texts of each set.

**1a. Raw values, no reference model.** Gemma IT scores high NLL on everything, including plain prose
(`user@plain` 7.25 / 7.40 for wildchat / oasst1 user text, `asst@plain` 5.31), which is 2.4-2.5 bits per byte on
plain user text where I would expect about 1. Its mean next-token entropy is only 1.4-2.0 nats against those NLLs,
so the model is sharply overconfident on human text; that is a property of the model, not of the rendering (the
plain control is `<bos>` followed by the bare text), and fp32-with-offload and on-GPU int8 agree on it.

**1b. Rendering.** `<bos>` is present exactly once and first in all 270 sequences, including `plain`. A system
message is a separate `<|turn>system` turn (not merged into the user turn); none of our conditions has one, and a
conversation without one is the normal case. Role names are `user` / `model`. `plain` renders as `<bos>` + text.
**Defect found:** the template's generation prompt is `<bos><|turn>user\nU<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`
(thinking off), and the model begins every sample it writes with those channel tokens, but our assistant slot
rendered `<|turn>model\n` and went straight to the text, which the template does for a *stored* turn. Rule (ii)
applies. Excerpts (special tokens visible), before: `'<bos><|turn>model\n'` | `'Jen nods and introduces herself.'`;
after: `'<bos><|turn>model\n<|channel>thought\n<channel|>'` | `'Jen nods and introduces herself.'`.

**1c. Position profile.** The chat-slot penalty over plain is *not* confined to the first positions. For wildchat
user text, `user@user − user@plain` is +1.8 at position 0, +7.1 at positions 1-4, +4.7 at 5-19 and +4.2 at 20-39
(oasst1: +1.9, +10.0, +3.4, +3.1): largest early, but still 3-4 nats at positions 20-39, so it is not only a
header artifact. Before the fix the assistant slot had the same shape (user text: +15.5 at position 0, then +2.9,
+3.5, +3.4 over plain), with 30 nats at position 0 because the model expects `<|channel>`; that part was the
rendering defect. After the fix the assistant slot is at or below plain from position 1 on.

**1d. Bits per byte** (primary, positions 1+; secondary over all positions is in the diagnostics files), as first
scored → after the fix: wildchat user text `user@user` 3.595, `user@asst` 3.237 → 1.543, `user@plain` 2.497;
oasst1 user text 3.577, 3.218 → 1.462, 2.412; wildchat assistant text `asst@user` 2.570, `asst@asst` 2.247 → 1.033,
`asst@plain` 1.838. The user and plain conditions are identical before and after.

**Fix and before/after.** `core/render.py` now adds, for the assistant slot, whatever the generation prompt puts
after the model header (`assistant_opening_extra`): `<|channel>thought\n<channel|>` for Gemma 4, nothing for OLMo
or Qwen3 at default settings. OLMo rendering is byte-identical (hash over all texts and slots unchanged) and the
render, smoke and new Gemma tests pass (57 passed); logged in `docs/decisions.md` (commit `fbda26a`, prefix
`core:`). Re-scored with the same 90 texts and settings.

| nats/token, positions 1+ | `@user` | `@asst` before | `@asst` after | `@plain` |
|---|---|---|---|---|
| wildchat user text | 10.20 | 9.22 | 4.52 | 7.25 |
| oasst1 user text | 11.03 | 9.95 | 4.57 | 7.40 |
| wildchat assistant text | 7.64 | 6.64 | 3.17 | 5.31 |

Consequences: the role-specific terms of **user text are unchanged** (`user@user − user@plain` +2.95 and +3.64,
since neither slot was touched); every slot effect and every assistant-text quantity changed. The earlier claim
that user text was ~1 nat more expected in the assistant slot was an artifact; with the correct slot it is ~5.7
nats (wildchat) and ~6.5 (oasst1), and assistant text in its own slot is now 2.1 nats *below* plain
(`asst@asst − asst@plain` = −2.13 [−2.56, −1.70]). The assistant slot with the channel makes any text more
expected than plain prose (user text too, about −2.0), so the slot effect mixes a user-specific part with a general
one; the role-specific terms are the cleaner contrast. What is still unexplained is why `user@user` is 3 nats worse
than plain at every position; the Gemma base run (step 2) shows whether post-training made it so.

## Step 2: Gemma 4 31B base

Base verified through the HF API (`google/gemma-4-31B`, apache-2.0, 62.5 GB), scored on the same 90 texts and
slots as IT, rendered with the IT template (including the step-1 assistant-slot fix), fp32 with CPU offload, batch
size 4 (603 s). Full tables, paired over texts: `nll_arm/compare_gemma.md`; position plots
`nll_arm/position_change_gemma_*.png`. Weights cleared from the HF cache afterwards.

Base is a normal language model in this pipeline: plain-prose NLL is 2.68 / 2.60 / 1.78 nats/token (wildchat user /
oasst1 user / wildchat assistant text), and base has no role asymmetry (`user@user − user@plain` +0.01, `user@asst`
within 0.12 of `user@user`). So the high IT values in step 1 come from post-training, not from the pipeline.

| user text, IT − base, paired over texts | nats/token | bits/byte |
|---|---|---|
| wildchat: `user@user − user@plain` | +2.94 [+2.40, +3.44] | +1.10 [+0.84, +1.38] |
| oasst1: `user@user − user@plain` | +3.63 [+2.91, +4.32] | +1.16 [+0.95, +1.37] |
| wildchat: slot effect `user@user − user@asst` (descriptive) | +5.74 [+5.23, +6.24] | +2.07 [+1.80, +2.44] |
| oasst1: slot effect (descriptive) | +6.59 [+6.13, +7.05] | +2.15 [+1.97, +2.35] |

**Pre-committed rule (step 2): MET.** `user@user − user@plain` rises by far more than 0.05 nats, with the CI
excluding zero, in both user sources.

- Assistant-text role-specific term, IT − base: **−2.17 [−2.60, −1.73]**, the opposite sign. The effect is not a
  general property of chat slots: assistant text becomes *more* expected in the assistant slot while user text becomes
  *less* expected in the user slot, the asymmetry the README describes.
- Position: the IT − base change in the user-text term is largest at positions 1-4 (+6.9 wildchat, +10.0 oasst1) and
  decays but persists at positions 20-39 (+4.2, +3.1); base is flat at about 0 from position 1 on.
- Caveats: n = 30 per set (25 and 9 texts for the position analysis); one checkpoint pair; the assistant slot's
  extra thought channel makes any text more expected there, so the slot effects are descriptive only.

## Step 3 setup (recorded before any Qwen scoring)

- Models: `Qwen/Qwen3-8B-Base` and `Qwen/Qwen3-8B`, both rendered with `Qwen/Qwen3-8B`'s template (the base shares its
  vocabulary, checked by the exploration's model verification). fp32 on the GPU, batch size 4, same 90 texts.
- **Thinking setting: the template default** (`enable_thinking` unset). Qwen3's template renders an assistant-only
  conversation as `<|im_start|>assistant\nA<|im_end|>\n` (no think block); after a preceding user turn, or with
  `enable_thinking=False`, it adds `<think>\n\n</think>\n\n` after the header. We use the assistant-only form, i.e.
  the assistant slot is the header followed directly by the text. `assistant_opening_extra` is empty for Qwen3, so
  the step-1 core fix does not apply. This choice does not touch the user slot or the plain control, so it cannot
  affect either pre-committed rule (both use `user@user − user@plain` only); it can affect the descriptive
  assistant-slot terms, which may carry an artifact analogous to Gemma's missing thought channel. No sensitivity
  run with the empty think block was made.
- Render checks (`core.render`: content verbatim, content tokens identical across slots, no merge at the boundary)
  pass for all 90 texts in all three slots. Plain starts with `<|endoftext|>`.
- Qwen as a negative control requires generations that sustain a user voice; Qwen3-8B generations did not exist in
  `results/exploration/shiller_user_turns/`, so the existing exploration script is being run on it (all four
  conditions, its default settings). The 20-sample blind labelling then follows the rule above.

## Outcome

_Step 3 to be filled in._
