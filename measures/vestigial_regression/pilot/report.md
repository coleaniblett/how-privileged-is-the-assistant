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

## Step 3: Qwen3-8B as a negative control

Setup and template setting: see "Step 3 setup" above (template default; assistant slot is the header followed by the
text; render checks pass for all 90 texts). Qwen3-8B-Base and Qwen3-8B were scored in fp32 entirely on the GPU
(24 s each), same 90 texts and settings. Tables: `nll_arm/compare_qwen.md`, `nll_arm/cross_family.md`; plots
`nll_arm/position_change_qwen_*.png`. Qwen weights cleared from the HF cache afterwards (Qwen3-8B instruct was
downloaded by the exploration run; see "Not done" for what remains cached).

**Is Qwen a negative control?** `results/exploration/shiller_user_turns/` had no Qwen3-8B samples, so the existing
exploration script was run on it (all four conditions, bf16, its defaults; 205 samples, now committed there). Twenty
random `ordinary` samples (seed 3, labelled before looking at flags; `qwen3-8b_generation_labels.csv`): **11 stay in
the user's voice, 9 do not** (7 answer or reason as the assistant, e.g. `Okay, so I need to figure out…` or a
`</think>` answer; 2 repeat a sentence verbatim). More than half, so by the pre-committed rule Qwen is a negative
control, but not a clean one: about a third of its samples leak assistant-style thinking, as Shiller observed. For
contrast, Gemma's 60 README-arm samples had 0 in the user's voice. (The measure's own `assistant_register` flag fired on
0 of these 20, including the assistant answers.)

**Qwen change from its own base, paired over texts** (`compare_qwen.md`):

| quantity, IT − base | nats/token | bits/byte |
|---|---|---|
| wildchat user: `user@user − user@plain` | +0.44 [+0.26, +0.62] | +0.14 [+0.09, +0.20] |
| oasst1 user: `user@user − user@plain` | +0.39 [+0.25, +0.52] | +0.13 [+0.08, +0.17] |
| wildchat assistant: `asst@asst − asst@plain` | +0.30 [+0.19, +0.41] | +0.09 [+0.06, +0.12] |

Qwen's user-text term therefore **also rises by more than 0.05 nats with the CI excluding zero in both sources**, i.e.
the step-2 rule is met by the negative control too. Qwen's assistant-text term rises with it (+0.30), where Gemma's
fell (−2.17): Qwen's rise is not user-specific, Gemma's is. Also: Qwen base's plain control sits 1.0-1.2 nats above
its own chat-slot values (`user@user − user@plain` −1.04 / −1.19), whereas Gemma base's three slots agree to within
0.12 nats; the plain reference behaves differently by family, which is why cross-family comparisons use change from
base in bits per byte.

**Cross-family** (`cross_family.md`; 30 shared texts per source; bits/byte, `user@user − user@plain`, IT − base):

| user source | Gemma | Qwen | Gemma − Qwen, paired |
|---|---|---|---|
| wildchat | +1.096 [+0.853, +1.384] | +0.144 [+0.091, +0.200] | **+0.952 [+0.674, +1.248]** |
| oasst1 | +1.164 [+0.943, +1.370] | +0.125 [+0.083, +0.165] | **+1.039 [+0.814, +1.257]** |

## Outcome

Pre-committed rules, as written before any result (see "Pre-committed rules"):

| rule | result |
|---|---|
| Step 1: rendering problem? | **Yes, rule (ii):** Gemma's assistant slot lacked the empty thought channel that its generation prompt always adds. Fixed in core (`fbda26a`, OLMo byte-identical), re-scored. User-text role-specific terms unchanged; slot effects and assistant-text terms changed (earlier +1 nat slot effects were an artifact). |
| Step 2: proxy flags Gemma (criterion 1 against Gemma base, both user sources) | **MET.** +2.94 [+2.40, +3.44] and +3.63 [+2.91, +4.32] nats/token; assistant-text term moves the opposite way (−2.17). |
| Step 3: Qwen is a negative control (> half of 20 samples stay in user voice) | **Met, narrowly:** 11 of 20; about a third leak assistant-style thinking. |
| Rule A: Gemma meets criterion 1 and Qwen does not | **NOT MET.** Qwen meets criterion 1 as well (+0.44, +0.39 nats). |
| Rule B: Gemma's change exceeds Qwen's in bits/byte, paired CI excluding zero, both sources | **MET.** +0.95 [+0.67, +1.25] and +1.04 [+0.81, +1.26] bits/byte. |
| The measure discriminates (A or B) | **Yes, through rule B only.** |

Reading: the proxy flags the model we know shows vestigial regression, and it separates it from Qwen by magnitude
(about 8 times larger in bits per byte) and by direction of the assistant-text term (opposite sign for Gemma, same
sign for Qwen). It does not separate them by the criterion's 0.05-nat threshold alone, since the negative control
also clears that. An absolute threshold at that size would call Qwen3-8B regressed.

Caveats: n = 30 texts per set (25 and 9 texts with at least 40 tokens for position analyses); one checkpoint pair per
family; Gemma is 31B and Qwen 8B, and the two post-training pipelines differ in more than regression, so the
cross-family difference is not attributable to regression alone; Qwen is an imperfect negative control (9 of 20
samples leak); my labels (generations and the flag-arm labels) are one reader's and need a teammate's check; the
hypothesis set and rules were written by the agent that ran the pilot and are not a registered measure criterion.

**Not done / not run:** a Qwen sensitivity run with the empty think block in the assistant slot (only descriptive
terms would change); any OLMo re-run; significance tests per position; a sweep over more texts; the measure's own
generation flags were not applied to Qwen beyond the 20 labelled samples; the exploration's Qwen3-8B hand-labelling
sheet. Qwen3-8B (instruct) weights were downloaded by the exploration run and are still in the HF cache apart from
what `hf_cache free` removed; Gemma 31B IT and base weights are cleared (the IT tokenizer files were re-downloaded).

## Checks on the existing scores (no GPU, no new scoring)

### Check 1 reading rule (written before any check-1 number was computed)

Question: is Qwen3-8B's rise in the user-text role-specific term (`user@user − user@plain`, +0.44 / +0.39 nats from
base) a real rise of `user@user`, or mainly a fall of `user@plain` (Qwen base scores plain text about 1 nat worse than
its chat slots)? For Qwen3-8B and, for comparison, Gemma 4 31B, every quantity is IT − base per slot, paired over the
same 30 texts per set, with 95% bootstrap CIs (5000 resamples, `config.SEED`, `core.stats.bootstrap_ci`), in nats per
token (positions 1+) and in primary bits per byte; each role-specific change is decomposed into its two parts
(`Δ(user@user)` and `Δ(user@plain)`). The verdict uses nats per token, for Qwen, in both user sources:

- **Plain-control effect** if `Δ(user@plain)` ≤ −0.05 nats with its CI excluding zero, **and** `Δ(user@user)` is below
  0.05 nats or its CI includes zero.
- **Real user-slot rise** if `Δ(user@user)` ≥ 0.05 nats with its CI excluding zero.
- Otherwise **mixed**, and described.

"In both user sources" applies to whichever of the first two is claimed. Check 2 (OLMo in bits per byte) is defined in
its own section below; its text sets are fixed in advance there too.

### Check 1 result (`nll_arm/checks.md`; Qwen3-8B and Gemma 4 31B, IT − base, paired over the same 30 texts per set)

**Verdict: real user-slot rise.** In both user sources `Δ(user@user)` is at least 0.05 nats with its CI excluding zero
(wildchat +0.61 [+0.49, +0.74], oasst1 +0.36 [+0.26, +0.45]), and the plain control did **not** fall
(`Δ(user@plain)`: +0.17 [+0.04, +0.30] and −0.03 [−0.15, +0.09]); the plain-control rule fails in both sources. In
bits per byte: `Δ(user@user)` +0.20 / +0.12, `Δ(user@plain)` +0.06 / −0.01, role-specific change +0.14 / +0.13.
The hypothesis that Qwen's rise is mainly `user@plain` falling is not supported.

What the decomposition also shows (descriptive): the rise is a cost of *chat framing in general*, not of the user slot.
User text in the assistant slot rose almost as much as in the user slot (`Δ(user@asst)` +0.51 / +0.35 nats, so the slot
effect `user@user − user@asst` changed by only +0.10 / +0.01), and assistant text rose in both chat slots (+0.42 / +0.40)
while its plain control rose little (+0.11). So the rule's label "real user-slot rise" is accurate for `user@user` but not
specific to the user slot. For Gemma, by contrast, IT degraded the plain control a lot (`Δ(user@plain)` +4.57 / +4.80
nats, +1.58 bits/byte in both sources) and `user@user` far more (+7.51 / +8.43 nats, +2.67 / +2.74 bits/byte); its
assistant slot rose much less (`Δ(user@asst)` +1.77 / +1.84 nats), so Gemma's chat-slot cost is concentrated in the
user slot, which Qwen's is not. The role-specific term subtracts the plain degradation, so for Gemma it understates
the total damage.

### Check 2: OLMo 3 7B in bits per byte (`nll_arm/checks.md`)

Definition (fixed in advance): the frozen OLMo held-out scores in `results/scores/` converted to primary BPB with the
OLMo tokenizer; change from OLMo base for SFT, DPO and Instruct, paired over texts, on (a) the pilot's 90 texts (primary
for the cross-family comparison) and (b) all OLMo texts (secondary). **Verification first:** the `text_id`s of all four
frozen score files match `data/texts.jsonl` (1190 of 1190, none missing or extra), and for every text the score files'
`n_tokens` equals the data's `n_tokens` in every slot, so they are the same texts. Result: OLMo's user-text role-specific
term in BPB is slightly *negative* (wildchat −0.07 [−0.11, −0.03] for Instruct, oasst1 −0.03 [−0.05, −0.00]), the
opposite direction from vestigial regression, and the all-texts version agrees in sign and is tighter (−0.08 / −0.04
for Instruct). The assistant role-specific term is −0.04 after SFT and about zero after DPO and Instruct (+0.01, CI
includes zero).

### Summary: bits per byte, change from each family's own base, 90 shared texts

| model (IT − own base) | wildchat Δ(user@user) | wildchat Δ(user@plain) | wildchat user role-specific | oasst1 Δ(user@user) | oasst1 Δ(user@plain) | oasst1 user role-specific | assistant role-specific (asst@asst − asst@plain) |
|---|---|---|---|---|---|---|---|
| OLMo 3 7B Instruct | -0.17 [-0.31, -0.03] | -0.10 [-0.23, +0.03] | -0.07 [-0.11, -0.03] | +0.10 [+0.07, +0.13] | +0.13 [+0.10, +0.15] | -0.03 [-0.05, -0.00] | +0.01 [-0.01, +0.03] |
| Qwen3-8B | +0.20 [+0.16, +0.24] | +0.06 [+0.02, +0.10] | +0.14 [+0.09, +0.20] | +0.12 [+0.09, +0.15] | -0.01 [-0.05, +0.03] | +0.13 [+0.08, +0.17] | +0.09 [+0.05, +0.12] |
| Gemma 4 31B IT | +2.67 [+2.38, +3.01] | +1.58 [+1.42, +1.75] | +1.10 [+0.85, +1.37] | +2.74 [+2.54, +2.94] | +1.58 [+1.32, +1.86] | +1.16 [+0.95, +1.38] | -0.82 [-1.15, -0.57] |

OLMo per stage (same 90 texts, bits/byte, change from OLMo base):

| model (IT − own base) | wildchat Δ(user@user) | wildchat Δ(user@plain) | wildchat user role-specific | oasst1 Δ(user@user) | oasst1 Δ(user@plain) | oasst1 user role-specific | assistant role-specific (asst@asst − asst@plain) |
|---|---|---|---|---|---|---|---|
| OLMo SFT | -0.23 [-0.36, -0.10] | -0.15 [-0.27, -0.04] | -0.07 [-0.12, -0.03] | +0.07 [+0.04, +0.09] | +0.08 [+0.06, +0.09] | -0.01 [-0.03, +0.01] | -0.04 [-0.05, -0.03] |
| OLMo DPO | -0.18 [-0.32, -0.04] | -0.11 [-0.24, +0.01] | -0.06 [-0.11, -0.02] | +0.09 [+0.06, +0.12] | +0.11 [+0.08, +0.13] | -0.02 [-0.04, +0.01] | +0.01 [-0.01, +0.03] |
| OLMo Instruct | -0.17 [-0.31, -0.03] | -0.10 [-0.23, +0.03] | -0.07 [-0.11, -0.03] | +0.10 [+0.07, +0.13] | +0.13 [+0.10, +0.15] | -0.03 [-0.05, -0.00] | +0.01 [-0.01, +0.03] |

Each family's change is from its own base; only bits per byte is compared across families (per-token nats are not: the tokenizers differ). Cells are mean change [95% paired bootstrap CI over texts]. All values are `condition − condition` differences, so the OLMo, Qwen and Gemma plain controls (`<|endoftext|>` / `<bos>` document starts) enter only through each family's own base.

### Unexpected or worth knowing

- Qwen's rise is not a plain-control artifact (check 1), and not user-slot-specific either: both chat slots and both
  text kinds got worse relative to plain.
- OLMo's raw changes flip sign between text sources (wildchat `Δ(user@user)` −0.17, oasst1 +0.10 bits/byte), but its
  plain control moves with them (−0.10 / +0.13), so the role-specific term is stable and the raw changes are not
  interpretable on their own.
- In BPB OLMo Instruct shows no sign of regression on this proxy (user role-specific −0.07 / −0.03), Qwen a small positive
  rise (+0.14 / +0.13, about 1/8 of Gemma's), Gemma a large one (+1.10 / +1.16): ordering consistent with the pilot's
  premise, with Qwen not at zero.
- OLMo's assistant role-specific term falls only at SFT (−0.04) and is about zero at DPO and Instruct (+0.01); the
  README expects `asst@asst` to fall relative to plain through post-training, which this BPB view does not show past SFT.
- n = 30 per set; OLMo, Qwen and Gemma plain controls differ (`<|endoftext|>` vs `<bos>` document starts), which is why
  only each family's change from its own base is compared.

## Appendix: the flag, sampling and NLL arms (earlier experiments, from README.md; unchanged except the correction note)

**Exploratory, n = 30 + 18.** Outside every verdict. Does not touch the measure's code, README or
pre-registered criteria; imports `ASSISTANT_REGISTER` and `repetition_loop` from `../flags.py` unchanged.

**Question.** The Gemma open-user-turn samples (`results/exploration/shiller_user_turns/gemma4-31b-it.jsonl`)
are, by eye, mostly vestigial regression: the model loops or answers as the assistant instead of continuing the
user. Does the measure's current generation-flag approach, plus the README's criterion 4 (a spec addition, not
implemented in the measure), class such samples as vestigial regression?

**Method.**
- `labels.csv`: 30 random samples (`random.Random(1)` shuffle of the 205 rows), labelled `degenerate` or
  `switched to assistant` by the agent that wrote this pilot, reading the text only (flags were not computed
  until afterwards). **A teammate should check these labels.**
- `run.py`: recomputes `assistant_register` and `repetition_loop` on the stored user portion (cut at Gemma's
  end-of-turn / turn-open / stop tokens); `ended_turn` and `handoff_to_assistant` come from the exploration run.
  Criterion 4: `repetition_loop` or `handoff` → vestigial; `assistant_register` without handoff →
  overgeneralization; else unclassified.
- Control: the text after the prefill sentence of 18 real WildChat user turns (`data/prefill.jsonl` →
  `data/texts.jsonl`; 2 of the 20 had under 50 chars left), first 600 chars. There are **no** user-voice
  samples among Gemma's outputs to serve as negatives.

**Result** (full table and misses: `results.md`, per sample: `per_sample.csv`).

| | n | classed vestigial |
|---|---|---|
| degenerate loops | 8 | 8 (100%) |
| switched to assistant | 22 | 3 (14%); 7 classed overgeneralization, 12 unclassified |
| real user text (control) | 18 | 0; 2 classed overgeneralization |

**Reading.**
1. Word-level loops are caught (`repetition_loop`). Loops without spaces (`////`, `KNDKND…`) are not, and the
   exploration's `token_loop` flag is not part of the measure; none happened to fall in this labelled sample.
2. Assistant switches are mostly missed. Gemma starts them with `<|channel>thought\n<channel|>`, which is
   neither an end-of-turn nor a turn-open marker, so the "user portion" runs into the assistant answer, and
   `handoff_to_assistant` is 0 for all 30. The register regex only matches stock phrases ("I'd be happy to"),
   so most answers (`The answer is a **Green Apple**`) match nothing. 12 of 22 are unclassified.
3. Criterion 4 as written would send 7 of the 22 to overgeneralization. By eye these are the same switch
   into the assistant, not a user voice with assistant traits. Whether the rule should distinguish them is a
   spec question for the owner.
4. The control is not clean: 2 of 18 real turns contain "As an AI" in prompt-engineering text.

**Run.** `uv run python -m measures.vestigial_regression.pilot.run` from the repo root (no GPU).

## Second experiment: the README's own sampling arm, run fresh on Gemma 4 31B-it

GPU run, 2026-11-02: the 20 `data/prefill.jsonl` sentences × 3 samples × 120 new tokens, T=0.7, seed 0, stopping
only at end of text, int8 (the 62 GB model does not fit this 48 GB GPU in bf16), sampled with
`exploration.shiller_user_turns.generate --conditions ordinary` into `gemma4-31b-it/` (rows and meta). No core
or measure code was changed. `readme_arm.py` applies the same flags and criterion 4 as above;
`labels_readme_arm.csv` holds my labels, made on shuffled output before the flags were computed
(**a teammate should check them**). Results: `readme_arm.md`, `readme_arm.csv`.

| label (by eye) | n | classed vestigial | overgeneralization | unclassified |
|---|---|---|---|---|
| degenerate loop | 17 | 17 | 0 | 0 |
| switched to assistant | 43 | 0 | 14 | 29 |

README criterion 3 counts out of 60: `ended_turn` 15, `handoff_to_assistant` 0, `assistant_register` 14,
`repetition_loop` 17.

- All 60 samples are vestigial regression by eye, so the 17 the rule calls vestigial are all correct, but with no
  user-voice samples this says nothing about false alarms (the real-user-text control above does: 0 of 18).
- The rule finds only loops. None of the 43 assistant switches is classed vestigial, and 14 would be reported as
  overgeneralization instead. Same finding as the first experiment, on fresh samples.
- The NLL slot-effect arm is below.

## Third experiment: the README's NLL slot-effect arm on Gemma 4 31B-it

`nll_arm/score.py` scores the README's slot conditions (`user@user`, `user@asst`, `user@plain`, and the same for
assistant text) with core's own rendering and span-NLL code, in **fp32 with CPU offload** (14 of 66 modules on the
48 GB GPU, the rest streamed from RAM), so AGENTS.md's fp32 rule holds without an exception. First 30 texts of each
of wildchat user, oasst1 user and wildchat assistant (270 sequences, 611 s). `nll_arm/analyze.py` writes
`nll_arm/results.md`. **No Gemma base was scored** (62 GB, no disk): the reference is OLMo 3 7B base from the
frozen held-out run, which we take to show no vestigial regression. Differences are within-text, so raw NLL levels
(much higher for Gemma-it: 3.2 nats on plain encyclopedic prose) cancel, but the comparison is across models, unpaired.

**Correction (same day, after step 1 of `report.md`).** The scores first reported here rendered Gemma's assistant slot
without the empty thought channel (`<|channel>thought\n<channel|>`) that Gemma's generation prompt always adds
after `<|turn>model\n`. Fixed in `core/render.py` (logged in `docs/decisions.md`) and re-scored; the pre-fix scores
are kept in `nll_arm/scores_before_fix/`. Only assistant-slot conditions changed, so the role-specific user-text
term (`user@user − user@plain`) is unchanged; every slot effect and every assistant-text number moved.

| user text, change from OLMo base | slot effect `user@user − user@asst` | role-specific `user@user − user@plain` |
|---|---|---|
| wildchat | +5.69 [+5.19, +6.18] (was +0.98, artifact) | +3.12 [+2.60, +3.62] (unchanged) |
| oasst1 | +6.49 [+6.03, +6.96] (was +1.11, artifact) | +3.84 [+3.15, +4.51] (unchanged) |

(OLMo instruct for scale: slot effect −0.05, role-specific −0.41 / −0.33.) Both README directions still hold for
Gemma-it, well past 0.05 nats. With the correct assistant slot, user text is about 6 nats *more* expected in the
assistant slot than the user slot. Assistant text now behaves as a model that has learned the assistant role would
show it: `asst@asst − asst@plain` = −2.1 nats (before the fix it was +1.3). The assistant slot with the thought
channel makes any text more expected than plain prose, user text included (`user@asst − user@plain` about −2.0),
so the slot effect for user text mixes a user-specific part with that general effect; the role-specific terms are
the cleaner contrast, and they are unchanged.

Checks and caveats:
- Offload path verified: fp32 offload and on-GPU int8 agree to 0.05–0.25 nats on 5 texts, and unpadded, padded and
  eager-attention scores of one text agree to 3e-4 nats.
- Assistant text has a slot effect too (+4.5), and user text in the assistant slot is also below plain, so the slot
  effect alone does not show a user-specific loss. The role-specific terms do: user text +3.1/+3.8 in its own
  slot, assistant text −2.1 in its own.
- No same-model base, so the README's control (`user@plain` drift) and "change from base" are not tested; n = 30 per
  set; no per-position analysis (the first-position header effect is only avoided by dropping position 0).
- A first scoring attempt at batch size 16 ran out of GPU memory (262k vocabulary); the run uses batch size 4.
