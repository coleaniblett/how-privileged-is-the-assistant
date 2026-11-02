# Overgeneralization

**Shiller's form:** dominance. **Owner:** TBD. **Status:** in design; nothing implemented. Pre-registered
criteria drafted 2026-11-02 (bracketed numbers are unconfirmed); they are frozen once results for this measure
exist under `results/<run>/overgeneralization/`, and any later change needs a dated entry in
`docs/decisions.md` (AGENTS.md).

## Spec (in design)

**Primary measure: likelihood minimal pairs.** Take passages from non-assistant genres (fiction, forum posts,
news, personal email, ...), each with two continuations identical except that one carries an assistant trait:

- a disclaimer ("I'm not a doctor, but ..."),
- a "not X, but Y" construction,
- a retraction ("Actually, I should clarify ..."),
- an LLM self-reference ("As an AI ...").

For each item compute the log-odds preference for the trait version against a **length- and
plausibility-matched human-voice insertion** (not against the bare plain continuation, which is longer by
construction on the trait side and confounds the trait with generic drift and with post-training's sharpening
toward likely text), in plain context (text after `<|endoftext|>`, no chat template),
across stages relative to base. The assistant slot (the same item rendered inside an assistant turn) is the
positive control: the trait version should become more likely there with post-training; overgeneralization is
the same shift in plain context. Scoring goes through `core.score.spans_nll` on rendered sequences (see
`extras/coin_toss.py` for a scored-continuation example); the items themselves are a new data set to be built
here and committed under `data/`. Exact definitions are in the pre-registered criteria below.

**Interp arm.** Project plain-text activations onto an assistant direction per stage (the Assistant Axis
method of Lu et al. 2026, or a difference of means between assistant-slot and user-slot activations),
normalized between the two poles, and track the projection of non-assistant text across stages.

**Behavioral companion.** Counts of assistant tics (the four traits above, plus the register phrases of
`measures/vestigial_regression/flags.py`) in sampled continuations of the same non-assistant passages.

**Exploratory only.** User/assistant representational convergence (do user-slot and assistant-slot
representations move together across stages?) is exploratory and is *not* the overgeneralization test.

Related: the vestigial-regression spec routes "rising `assistant_register` without handoff" in the user slot
to this measure as overgeneralization in the user slot.

## Pre-registered criteria

Drafted 2026-11-02, before any item is built or scored. **[Bracketed]** values are unconfirmed choices.

**Unit.** An *item* is a passage `P` from a non-assistant genre, a shared tail `C` (the rest of a natural
continuation) and four variants of what sits between them, each a standalone sentence so that the same `C`
follows all of them: `T`, an assistant-trait phrase; `H`, the **comparator**: a human-voice insertion of the
same syntactic type, within ±2 tokens of `T`, with no assistant function and no trait construction, about as
unlikely in context as `T` (matched at base, below); `N`, a plain, plausible human insertion for the genre (a
reference that exposes the sharpening bias, below); and nothing (plain `C`, a reference only). Drafts:
`example_items.jsonl`.

- **Traits:** disclaimer, "not X, but Y", retraction, LLM self-reference. Each has **[3]** surface phrasings,
  spread evenly across items so that no single phrase carries the result.
- **Genres:** **[5]**: fiction, forum post, news, personal email, and one more to be chosen. Each trait × genre
  cell has **[12]** items. Passages come from held-out text that passes the same Dolci exact-match filter as
  the other text sets; `T`, `N` and `C` are written and checked by hand.
- **Contexts**, following core's rule that content tokens are identical and only the header differs: *plain*
  (`<|endoftext|>` + `P` + continuation), *assistant slot* (the same text inside an assistant turn; the
  positive control) and *user slot* (the same inside a user turn; secondary). Only the continuation span is
  scored, so the first-token header effect never touches it. Each rendered sequence must equal
  `tokens(P) + tokens(continuation)` tokenized separately (as `core.render` checks). Scoring is fp32.

**Score and effect.** `score(item, stage, context) = log P(T+C | P) − log P(H+C | P)`, summed over the full
continuation including the shared tail. `H` absorbs the length penalty on the trait version, generic
distribution drift and, because it is matched to `T` in base plausibility, the **sharpening bias**:
post-training concentrates mass on whatever is already likely, and a plausible comparator (`N`) would make the
trait look *less* likely for that reason alone. `effect(item, stage) = score(item, stage) − score(item, base)`, paired
per item, with a 95% bootstrap CI over items (`core.stats.bootstrap_ci`) for each trait × genre cell.

**Matching `H` (base only).** `H` is written by hand, then checked with one scoring pass of the base model on
`T+C` and `H+C` in plain context; no later stage and no assistant-slot score is looked at. An item whose base
`|log P(T+C) − log P(H+C)|` exceeds **[2.0] nats** has `H` rewritten once; if it still fails the item is
dropped, and the drop count per cell is reported. Selection uses base scores only, so it uses no later-stage
data.

**Criteria.**

1. **Cell-level effect.** In plain context a trait × genre cell shows overgeneralization at a stage if its
   effect is at least **[0.25] nats** (δ) and the 95% CI excludes zero, with the trait becoming *more* likely.
   A significant negative effect is reported but never counts. (The 0.05-nat convention is for per-token NLL;
   these scores are summed over a continuation.)
2. **Positive-control validity.** In each trait × genre cell, the assistant-slot effect must itself be at least δ
   with the CI excluding zero at that stage. If it is not, the cell is **uninterpretable**, not a null; a trait
   whose interpretable cells are too few to reach the majority in criterion 3 is uninterpretable at that stage,
   and so is a stage with fewer interpretable traits than criterion 4 requires.
3. **Trait-level rule.** A trait overgeneralizes at a stage if criteria 1 and 2 both hold in a majority of
   genres (**[≥ 3 of 5]**). Genres are never pooled.
4. **Verdict.** Overgeneralization is present at a stage if **[≥ 2 of 4]** traits meet the trait-level rule.
   The first such stage is the answer to "which stage". It counts toward H4, reported as dominance by
   overgeneralization.
5. **Reporting.** Always report, beside the effect: the absolute `log P(T+C)`, `log P(H+C)` and `log P(N+C)` per stage, the
   per-phrasing effects, and the tokenization of every `T`, `H` and `N`. Never report the log-odds alone.
6. **Breadth arm.** Between the base and instruct checkpoints the same rule applies, and every table says that
   contamination could not be checked.

**Descriptive only (cannot change the verdict):** the sharpening diagnostic (change from base in
`log P(N+C) − log P(H+C)`, which should be positive if post-training favours plausible text); the user-slot
effect; the plain-`C` reference; the
behavioral companion (trait-regex rates in sampled plain-context continuations); the interp arm; and the
vestigial-regression count of `assistant_register` without handoff.

**Open before results exist:** δ may instead be calibrated on neutral-vs-neutral pairs (`N1` vs `N2`), which
uses no trait data; if so, record that here and in `docs/decisions.md` before scoring any trait.

## Running

Primary measure only (`minimal_pairs.py`). The items default to the drafts in `example_items.jsonl` until the
full set is built under `data/`. Not yet in `scripts/run_gpu.sh`.

```bash
uv run python -m measures.overgeneralization.minimal_pairs score --model base     # per stage, fp32 (GPU for OLMo)
uv run python -m measures.overgeneralization.minimal_pairs analyze                # results.md, summary.csv
uv run pytest tests/test_overgeneralization.py                                    # CPU: OLMo rendering, verdict logic, SmolLM2 smoke run
```

Outputs: `results/<run>/overgeneralization/scores/<stage>.csv`, `results.md`, `summary.csv`. The `analyze` flags
`--delta`, `--match-bound`, `--max-len-diff` and `--min-traits` default to the pre-registered values and exist
for tests and sensitivity checks only.
