# Overgeneralization

**Shiller's form:** dominance. **Owner:** TBD. **Status:** in design; nothing implemented. The pre-registered
criteria below are a placeholder to be completed before any results exist; once they do, changes need a
dated entry in `docs/decisions.md` (AGENTS.md).

## Spec (in design)

**Primary measure: likelihood minimal pairs.** Take passages from non-assistant genres (fiction, forum posts,
news, personal email, ...), each with two continuations identical except that one carries an assistant trait:

- a disclaimer ("I'm not a doctor, but ..."),
- a "not X, but Y" construction,
- a retraction ("Actually, I should clarify ..."),
- an LLM self-reference ("As an AI ...").

For each pair compute the log-odds preference for the trait version, log P(trait continuation) − log P(plain
continuation), in plain context (text after `<|endoftext|>`, no chat template), across stages relative to
base. The assistant slot (the same pairs rendered as an assistant turn) is the positive control: the trait
version should become more likely there with post-training; overgeneralization is the same shift in plain
context. Scoring goes through `core.score.spans_nll` on rendered sequences (see `extras/coin_toss.py` for a
scored-continuation example); the pairs themselves are a new data set to be built here and committed under
`data/`.

**Interp arm.** Project plain-text activations onto an assistant direction per stage (the Assistant Axis
method of Lu et al. 2026, or a difference of means between assistant-slot and user-slot activations),
normalized between the two poles, and track the projection of non-assistant text across stages.

**Behavioral companion.** Counts of assistant tics (the four traits above, plus the register phrases of
`measures/vestigial_regression/flags.py`) in sampled continuations of the same non-assistant passages.

**Exploratory only.** User/assistant representational convergence (do user-slot and assistant-slot
representations move together across stages?) is exploratory and is *not* the overgeneralization test.

Related: the vestigial-regression spec routes "rising `assistant_register` without handoff" in the user slot
to this measure as overgeneralization in the user slot.

## Pre-registered criteria (placeholder)

To be written before the minimal-pair set is built. Expected shape: overgeneralization is present at a stage
if the mean log-odds preference for the trait version in plain context rises from base with the 95% bootstrap
CI (paired over pairs) excluding zero, with the assistant-slot control showing the same sign; the smallest
effect to count is to be set in log-odds here (the 0.05-nat convention is for NLL-based effects).

## Running

Nothing to run yet.
