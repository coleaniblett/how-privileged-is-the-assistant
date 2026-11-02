# Logit distortion

**Shiller's form:** privilege. **Owner:** Grace. **Status:** design settled; `analyze.py` is a stub, nothing is
computed yet. The pre-registered criteria below are frozen once results for this measure exist under
`results/<run>/logit_distortion/` (AGENTS.md).

## Data route and candidate construction (2026-11-02)

**Primary data route (Grace's decision, 2026-11-02):** existing, original multi-exchange conversations from
OASST1, WildChat and UltraChat, each target scored with its full original preceding history. Role and
authorship: OASST1 human user / human assistant; WildChat human-submitted user / model assistant; UltraChat
model user / model assistant; sources are analyzed separately. The isolated authorship sets described below
(`data/texts_authorship.jsonl`, unchanged) remain available for a separate comparison.

Candidates are built by `core.build_conversations`; see [its specification and reproduction
commands](../../docs/conversation_candidates.md). Current candidate pilot:
[`pilot_v3` audit](../../candidates/logit_distortion/pilot_v3/audit.md),
[length sensitivity](../../candidates/logit_distortion/pilot_v3/length_sensitivity.md) and
[inspection notes](../../candidates/logit_distortion/pilot_v3/inspection.md): eight contributing conversations
per source (eligible-quota selection), 175 targets, 119 provisionally eligible, Dolci exact-match checked in
both roles (earlier pilots `pilot_v1`/`pilot_v2` kept). Comparisons should be stratified by opening vs
follow-up turn: at the 20-token minimum no UltraChat opening user turn is eligible. These are candidates, not final scoring
inputs: final sizes and thresholds are chosen from the audit before outcomes are seen, and original-context
scoring is a subsequent task (`core.score` refuses these records). No analysis or pre-registered outcome
criteria were changed.

## Spec

We compute the calibration gap (mean NLL minus mean next-token entropy) on user text in the user slot and
assistant text in the assistant slot, crossing persona with authorship: human-written user turns
(WildChat/oasst1) and assistant replies (oasst1), and model-written user turns (e.g. UltraChat) and assistant
replies (WildChat ChatGPT). We run this across OLMo 3 7B's post-training stages (base, SFT, DPO, Instruct),
where open training data lets us filter test text for overlap, and on base vs. instruct pairs from other
families, where overlap can't be checked. In every model, logit distortion counts as privilege if the
assistant persona's gap grows from base more than the user persona's does, regardless of authorship; we
report the gap from content position 1 onward and plot it by position, since a gap that persists after the
model has seen the speaker's style is stronger evidence than one that fades.

Swapped conditions (e.g. assistant text in the user slot) are deliberately not part of the headline analysis;
core scores them anyway, so they are available as secondary checks.

### Authorship cross

| | human-written | model-written |
|---|---|---|
| user text in the user slot | WildChat user turns, oasst1 prompts (`wildchat user`, `oasst1 user`: in `data/`) | UltraChat (or similar) user turns: **TODO** |
| assistant text in the assistant slot | oasst1 assistant replies: **TODO** | WildChat ChatGPT replies (`wildchat assistant`: in `data/`); OLMo Instruct's own replies (`olmo-instruct assistant`) as a self-authored extreme |

The two new text sets must be sampled by `core/prepare_data.py` with the same filters as the existing ones
(20–300 tokens, clean rendering, Dolci contamination filter) so that core scores them in every slot alongside
the rest. That is a core change: log it in `docs/decisions.md`.

### Quantities (all from core outputs)

Per text set (source × kind) and stage, with 95% bootstrap CIs over texts:

- gap(text) = `mean_nll_from1 − mean_entropy_from1` in the text's own slot (`user@user` for user text,
  `asst@asst` for assistant text), i.e. content positions 1+ (position 0 in the assistant slot is dominated by
  the `<|im_start|>` expectation, see the vestigial-regression README);
- change from base, paired over texts;
- the gap by content position from the per-token tables (`core.stats.position_arrays`), plotted per stage.

## Pre-registered criteria

1. **Logit distortion is present** in a model if, for both authorship columns, the assistant persona's gap
   change from base exceeds the user persona's gap change from base, by at least 0.05 nats with the 95%
   bootstrap CI of the difference excluding zero.
2. The headline uses positions 1+. The by-position plot is reported descriptively: a gap that persists across
   positions is stronger evidence than one concentrated in the first positions.
3. For breadth-arm families (base vs instruct only), the same rule applies between the two checkpoints, and
   every table and figure states that contamination could not be checked.
4. Swapped conditions never enter the verdict.

## Running

```bash
uv run python -m measures.logit_distortion.analyze --scores results/olmo3-7b/core/scores --out results/olmo3-7b/logit_distortion
```

(Not implemented yet: the command exits with a message.)
