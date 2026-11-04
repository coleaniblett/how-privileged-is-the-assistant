# Logit distortion

**Shiller's form:** privilege. **Owner:** Grace. **Status (2026-11-04):** analysed on the main run's exchange
scores (`analyze.py`, [exchange-design analysis](#exchange-design-analysis-2026-11-04) below; results in
`results/capstone/logit_distortion/exchanges_exchange/`). The original-conversation route (`pilot_v3`, follow-up turns) is
not scored for analysis and not covered by these results. The pre-registered criteria below are unchanged and frozen from
now on, since this analysis writes results under `results/<run>/logit_distortion/` (AGENTS.md); how they are applied to
the exchange design was added on 2026-11-04, after the data existed (see below), and so was an exploratory two-sided reading
of the gap contrast ([Direction of the gap contrast](#direction-of-the-gap-contrast-2026-11-04-after-the-results-exploratory)).

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
inputs: final sizes and thresholds are chosen from the audit before outcomes are seen. Original-context scoring
is `core.score_conversations` ([docs/conversation_scoring.md](../../docs/conversation_scoring.md)): one teacher-forced
sequence per target with only its original prefix, per-token NLL and entropy plus per-target gaps, input
fingerprints across checkpoints; tested on CPU with a small model, no checkpoint scored yet (`core.score` still
refuses these records). No analysis or pre-registered outcome
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

*Application to the exchange design (added 2026-11-04, after the data existed; the criteria text above is unchanged):*
human column = oasst1, model column = UltraChat, tagged, positions 1+, the difference of gap changes paired per exchange
with exchange IDs resampled jointly. See [Exchange-design analysis](#exchange-design-analysis-2026-11-04) and
`docs/decisions.md`, 2026-11-04.

## Exchange-design analysis (2026-11-04)

**Dataset.** The main run's exchange scores (`docs/exchanges.md`; `core.score_exchanges`, context `exchange`):
`results/capstone/<family>/<stage>/core/exchanges_exchange/` for OLMo 3 7B base/SFT/DPO/Instruct, Qwen3-8B base/IT and
Gemma 4 31B base/IT. Each exchange is one real question-reply pair from OASST1, UltraChat or WildChat (300 per corpus;
Gemma the first 100). The **user turn is the conversation opener**; the **reply is scored after its own question**. Each
is scored tagged (in its chat-template role) and plain (bare text after the document-start token, roles removed). This
is a different dataset from the multi-turn `pilot_v3` candidates: **nothing here covers follow-up user turns** or
longer histories.

**Questions.** (1) Are the assistant persona's next-token distributions distinctive (levels, assistant − user)?
(2) Does that change through training (change from base, and the change in assistant − user)? (3) Does it differ by
authorship (oasst1 human/human vs UltraChat model/model) and framing (tagged vs plain)?

**Already fixed before this data** (2026-11-01/02, above): gap = mean NLL − mean entropy over content positions 1+ in
the text's own slot; change from base; privilege = the assistant gap grows from base more than the user gap, in both
authorship columns, by ≥ 0.05 nats with the CI excluding zero (criterion 1); by-position plot descriptive; breadth-arm
contamination statement; no swapped slots. Fixed by the team on 2026-11-03: the exchange set and its renderings.

**Decided now (2026-11-04), after the main-run scores and the vestigial-regression results existed, so not blind** (an
agent session for Cole had also computed provisional, uncommitted logit-distortion numbers from these scores):

- *Authorship columns.* Human-written = oasst1 (user and reply both human); model-written = UltraChat (both ChatGPT).
  WildChat (human user, ChatGPT reply) is reported separately: its roles differ in authorship. This replaces the
  original cross's cross-corpus pairing (e.g. WildChat ChatGPT replies against UltraChat users).
- *Personas.* User persona = the opening user turn, tagged; assistant persona = the reply, tagged, after its question.
  Plain results are context (what the same tokens look like without role tags), not a persona condition.
- *Weights.* Per target: unweighted mean over its scored positions 1+ (identical positions for NLL, entropy and gap);
  per group: every target weighted equally. Replies are scored over their first 300 tokens (family tokenizer);
  truncated replies are kept, and truncation rates reported. Positions 0+ is a labelled sensitivity analysis; so is
  restricting to exchanges whose reply is untruncated.
- *Contrasts.* Assistant − user per exchange (paired: the reply is its own question's answer); change in that
  contrast from base, `(asst − user)post − (asst − user)base`, per exchange. Criterion 1's "difference of gap changes" is
  this quantity.
- *Uncertainty.* 95% percentile bootstrap, 10,000 replicates, fixed seed (`config.SEED`); exchange IDs resampled
  jointly within a family and corpus, so user/reply pairs, framings and checkpoints stay paired.
- *Cross-family.* Within-family results use all exchanges; cross-family comparisons use the common exchange IDs (the
  first 100 per corpus, which every family scored), and also bits per byte. Nats per token are not comparable across
  tokenizers; criterion 1 is applied within each family in nats per token.
- *Verdict.* Criterion 1 is applied as above to each post-trained checkpoint (tagged, positions 1+, oasst1 and
  UltraChat both required). Everything else (levels, single-persona changes, plain framing, positions, WildChat,
  cross-family) is estimated and reported without a pass/fail rule. Not blind, as stated. (Later the same day, after the
  results: an exploratory two-sided reading was added beside it; see [Direction of the gap
  contrast](#direction-of-the-gap-contrast-2026-11-04-after-the-results-exploratory).)

**How to read the quantities.** Three claims are tested separately, and a fourth needs more than this measure:

- *Assistant-specific concentration*, from **entropy**: lower assistant entropy at one checkpoint means more concentrated
  predictions, but a difference already present at base was not originated by post-training (this design does not identify
  its cause). An assistant-specific entropy fall through training (the reply falls more than the user turn) supports
  differential concentration.
- *Fit to the recorded text*, from **NLL**: a rise means the model departs from these human- or ChatGPT-written texts.
- *Role-dependent mismatch between observed surprise and predictive entropy*, diagnosed by the **gap**. The gap is not a
  formal calibration test and does not describe the distribution's shape. A positive D can come from the reply's NLL
  rising more, its entropy falling more, or both; a negative D likewise has several routes. The sign of D alone does not
  identify the mechanism, so D is always reported with its NLL and entropy parts (`decomposition.csv`).
- *Broader assistant privilege or dominance* is an interpretation that needs evidence beyond this measure.

User and assistant texts differ in content and context, so none of these comparisons isolates a causal effect of the role label.

## Direction of the gap contrast (2026-11-04, after the results; exploratory)

Added after the exchange-design results existed (`docs/decisions.md`, 2026-11-04, "direction of the gap contrast"). Not
pre-registered and not blind. Criterion 1 and its application above are unchanged.

**Where the positive direction comes from.** The 2026-11-01 design entry defines privilege as "the assistant persona's gap
grows from base more than the user persona's" and criterion 1 encodes it: the difference of gap changes must reach
+0.05 nats with the CI excluding zero. The documented reason concerns what is measured (whether the assistant persona's
*own* distribution is distorted, hence no swapped slots). No document gives a reason why a *larger* assistant gap, rather
than a role difference of either sign, is the signature of privilege; the direction follows from the measure's one-line
definition ("distinctively concentrated"), not from a stated argument against the opposite sign. The 0.05 nats is the
project-wide smallest effect (`config.MIN_EFFECT_NATS`, 2026-11-01): the pilot's negligibility margin and the size of
bf16 batch noise. That is an operational detectability floor, not a theoretically derived effect size. Criterion 1 is
therefore a **directional (one-sided) hypothesis**: "not met" says the assistant gap did not grow more than the user gap,
not that the roles were unaffected or that privilege is absent. (Vestigial regression's verdict is likewise applied in one
direction: its main-run results report a consistent negative asymmetry as "does not fire … opposite of regression".)

**What is reported now.** Beside the original criterion, now labelled the *original positive-direction gap criterion*, an
exploratory two-sided classification of D per post-trained checkpoint, on unrounded estimates:

- *positive shift*: D ≥ +0.05 and the 95% CI above zero, in both oasst1 and UltraChat (identical to criterion 1);
- *negative shift*: D ≤ −0.05 and the 95% CI below zero, in both corpora;
- otherwise: *no two-corpus directional consistency*.

±0.05 applies the original floor symmetrically; it is not a newly validated threshold. Either shift establishes a
role-dependent gap change under this design (opening user turn vs the reply after it, tagged), not by itself assistant
privilege: a negative shift can mean the user text became more surprising relative to the model's entropy than the reply did
(a relative assistant advantage), or that the user turn's entropy fell more, and the decomposition says which. The same
rules are applied to the labelled sensitivities (`verdict_sensitivity.csv`). Gap-based readings stay separate from vestigial
regression's own metric and from the behavioural evaluations.

## Running

```bash
uv run python -m measures.logit_distortion.analyze            # results/capstone -> results/capstone/logit_distortion/exchanges_exchange
```

The earlier isolated-slot command (`--scores results/olmo3-7b/core/scores`) is not implemented; those sets remain for a
separate comparison.
