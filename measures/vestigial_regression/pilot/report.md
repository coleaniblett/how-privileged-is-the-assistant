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

## Outcome

_To be filled in as the steps finish._
