# Logit distortion on the exchange design: report

Grace's measure. Analysis of saved scores only (no model loaded, nothing re-scored). Generated tables: [`results.md`](results.md);
every number: [`summary.csv`](summary.csv), [`verdict.csv`](verdict.csv), [`verdict_sensitivity.csv`](verdict_sensitivity.csv),
[`decomposition.csv`](decomposition.csv), [`positions.csv`](positions.csv), [`samples.csv`](samples.csv); validation:
[`checks.json`](checks.json); command, code revision, input hashes and settings: [`provenance.json`](provenance.json).
Spec: `measures/logit_distortion/README.md` ("Exchange-design analysis" and "Direction of the gap contrast", both 2026-11-04) and
`docs/decisions.md`, 2026-11-04.

**Status of these results.** The gap definition, the change-from-base comparison and criterion 1 (≥ 0.05 nats, CI excluding zero,
both authorship columns) were fixed on 2026-11-01/02. How they map onto the exchange design (oasst1 = human column, UltraChat =
model column, the reply after its question = assistant persona, joint pairing) was decided on 2026-11-04, **after** the main-run
scores and the vestigial-regression results existed. The two-sided reading in section 3 was added later the same day, **after these
results**, and is exploratory. Nothing here is blind or pre-registered beyond criterion 1's text. Contamination was checked for OLMo
only: **Qwen3-8B and Gemma 4 31B results cannot exclude training-data overlap.**

## Data

`results/capstone/<family>/<stage>/core/exchanges_exchange/`: OLMo 3 7B base / SFT / DPO / Instruct, Qwen3-8B base / IT, Gemma 4 31B
base / IT. 300 real question-reply exchanges per corpus (Gemma: the first 100), each scored as the **opening user turn** and as the
**reply after its own question**, tagged (chat template) and plain (bare text). Mean NLL and entropy over content positions 1+, each
target weighted equally.

| | oasst1 | UltraChat | WildChat |
|---|---|---|---|
| who wrote user / reply | human / human | ChatGPT / ChatGPT | human / ChatGPT |
| role in the analysis | human-written column | model-written column | separate (authorship differs between roles) |
| exchanges (OLMo, Qwen / Gemma) | 300 / 100 | 300 / 100 | 300 / 100 |
| replies truncated at 300 tokens (OLMo; Qwen, Gemma similar) | 38% | 55% | 0% |

**Validation (all passed, every checkpoint):** 12 conditions per checkpoint, no duplicates or missing conditions, finite values,
identical exchange IDs, truncation flags and one inputs fingerprint per family, tagged and plain scoring the same tokens, and the
per-token tables reproducing `scores.csv`'s means (positions 1+ and 0+) to ≤ 3e-6 nats (`scores.csv` stores 6 decimals). **Not covered:**
follow-up user turns and longer histories (the multi-turn `pilot_v3` route is separate and not analysed here).

## What each quantity can show

| claim | tested by | what would support it |
|---|---|---|
| assistant-specific concentration | entropy changes | the reply's entropy falls through training more than the user turn's |
| fit to the recorded text | NLL changes | (descriptive) how far each role's recorded text departs from the model's predictions |
| role-dependent mismatch between observed surprise and predictive entropy | the gap, NLL − entropy, and D | D away from zero, consistent across corpora, read with its NLL and entropy parts |
| broader assistant privilege or dominance | not this measure alone | needs vestigial regression, overgeneralization and behavioural evidence as well |

D = (gap_reply − gap_user)_post − (gap_reply − gap_user)_base, tagged, positions 1+, paired per exchange. D = D_NLL − D_entropy: a
positive D can come from the reply's NLL rising more than the user turn's, from its entropy falling more, or both; a negative D
likewise has several routes. **The sign of D alone does not identify the mechanism**, so every D below is reported with its parts.
The gap is a diagnostic: it is not a formal calibration test.

## Findings

### 1. The assistant persona's distributions differ from the user turn's already in the base models

At every checkpoint, base included, the tagged reply has lower entropy **and** lower NLL than the opening user turn: for example
OLMo base, oasst1, entropy −1.45 [−1.53, −1.37] and NLL −1.45 [−1.54, −1.35] nats/token (reply − user). The two lower together, so
the base gap difference is near zero (+0.00 [−0.04, +0.05]). Because the difference is present before post-training, post-training
did not originate it. This design does not identify its cause: the texts differ, a reply follows its own question, and the slots
carry different tags. Figures: `figures/levels_oasst1.png`, `levels_ultrachat.png`.

### 2. Concentration: training lowers entropy for both personas, more for the user turn

Post-training lowers entropy for both roles in every model. The user turn's entropy falls **more** than the reply's in 9 of 10
headline cells (Δ(reply − user) entropy +0.07 to +0.79 nats); the exception, Gemma IT on UltraChat, is −0.07 [−0.20, +0.06]. So
there is **no assistant-specific concentration** in this design. Figure: `figures/contrast_change.png` (top row).

### 3. The gap contrast D: original criterion and an exploratory two-sided reading

| checkpoint | oasst1 (human column) | UltraChat (model column) | original positive-direction criterion | exploratory two-sided | D_NLL / −D_entropy (oasst1; UltraChat) |
|---|---|---|---|---|---|
| OLMo SFT | −0.133 [−0.171, −0.095] | −0.115 [−0.143, −0.086] | not met | negative shift | −0.02 / −0.11; −0.04 / −0.07 |
| OLMo DPO | **+0.548 [+0.470, +0.627]** | **+0.486 [+0.434, +0.539]** | **met** | positive shift | +0.79 / −0.24; +0.81 / −0.32 |
| OLMo Instruct | **+0.514 [+0.431, +0.597]** | **+0.440 [+0.386, +0.494]** | **met** | positive shift | +0.82 / −0.30; +0.80 / −0.36 |
| Qwen IT | −0.600 [−0.666, −0.533] | −0.051 [−0.097, −0.006] | not met | negative shift (UltraChat at the floor) | +0.19 / −0.79; +0.16 / −0.21 |
| Gemma IT | −7.689 [−8.146, −7.230] | −7.313 [−7.659, −6.962] | not met | negative shift | −7.32 / −0.36; −7.38 / +0.07 |

Tagged, positions 1+, nats/token, 95% CI. Within a family only: nats per token are not comparable across tokenizers.

**Why the original rule is one-sided.** Criterion 1 was written for one hypothesis: the assistant persona's gap grows from base
*more* than the user persona's (2026-11-01 design entry; README criterion 1). The documented reasoning concerns *what* is measured
(the assistant persona's own distribution, so no swapped slots). No document argues that the opposite sign is uninformative; the
direction comes from the measure's definition. The 0.05 nats is the project-wide smallest effect (`config.MIN_EFFECT_NATS`: the
pilot's negligibility margin and the size of bf16 batch noise), an operational detectability floor, not a derived effect size. So
"not met" means only that the assistant gap did not grow more than the user gap. It does not mean that the roles were unaffected,
and it is not evidence against assistant privilege in general. The column is labelled accordingly.

**What the two-sided reading answers.** *Did the gap change differently for the two roles, consistently in both corpora, and in
which direction?* Positive shift: D ≥ +0.05 with the CI above zero in both corpora (identical to criterion 1). Negative shift:
D ≤ −0.05 with the CI below zero in both. ±0.05 applies the original floor symmetrically; it is not a newly validated threshold. The
rule was added after these results, is exploratory, and does not replace criterion 1. Either shift establishes a **role-dependent
gap change under this design**, not by itself assistant privilege. Every post-trained checkpoint shows one (positive for OLMo DPO and
Instruct, negative for the rest), but through different parts:

- **OLMo DPO / Instruct (positive).** The reply's NLL on the recorded reply rises by +1.06 to +1.24 nats, against +0.28 to +0.44 for
  the user turn; the entropy part works the other way (the user turn's entropy falls more). The gap grows because the model's fit to the
  recorded replies worsens more, not because the assistant's distribution concentrates more. One hypothesis, **not tested here**,
  is that the post-trained model expects replies of its own kind; testing it needs the model's own replies scored. The rise appears
  at DPO and persists across content positions (`figures/positions_*.png`).
- **OLMo SFT (negative, small).** The user turn's entropy falls slightly more; NLL changes alike in both roles.
- **Qwen IT (negative).** Entropy again: the user turn's entropy falls much more (oasst1 −1.19 vs −0.40). The NLL part is
  *positive* (the reply's NLL rises more, +0.51 vs +0.33), so the user text did **not** become more surprising than the reply.
  The UltraChat cell is at the boundary: D = −0.051066 [−0.096649, −0.006227] unrounded, past −0.05 by 0.001. It is negative
  under positions 0+ (−0.165) and with untruncated replies only (−0.141), but on the common first 100 exchanges its CI includes
  zero (−0.052 [−0.124, +0.019]). Treat the UltraChat classification as fragile.
- **Gemma IT (negative, large).** Through NLL; section 4.

The verdicts and classifications are the same with positions 0+ and with untruncated replies only (`results.md`, Sensitivity;
`verdict_sensitivity.csv`). In bits per byte on the common first 100 exchanges the signs match within each family.

### 4. Gemma IT: the user turn's text becomes far more surprising than its entropy suggests

Tagged, positions 1+, nats/token (Gemma tokenizer), n = 100 exchanges per corpus, 95% CI:

| corpus | role | NLL base → IT (Δ) | entropy base → IT (Δ) | gap base → IT (Δ) |
|---|---|---|---|---|
| oasst1 | user | 2.56 → 11.03 (**+8.48** [+8.14, +8.81]) | 2.75 → 1.84 (−0.91 [−1.03, −0.78]) | −0.19 → 9.19 (**+9.38** [+8.98, +9.78]) |
| oasst1 | reply | 1.00 → 2.15 (+1.15 [+1.03, +1.28]) | 1.00 → 0.46 (−0.54 [−0.62, −0.47]) | −0.00 → 1.69 (+1.69 [+1.51, +1.88]) |
| UltraChat | user | 2.09 → 10.47 (**+8.38** [+8.08, +8.68]) | 2.18 → 1.83 (−0.35 [−0.49, −0.22]) | −0.10 → 8.63 (**+8.73** [+8.40, +9.06]) |
| UltraChat | reply | 0.91 → 1.91 (+1.00 [+0.91, +1.09]) | 0.91 → 0.49 (−0.42 [−0.45, −0.39]) | 0.00 → 1.42 (+1.42 [+1.32, +1.53]) |

D = Δgap_reply − Δgap_user = 1.69 − 9.38 = **−7.69** (oasst1) and 1.42 − 8.73 = **−7.31** (UltraChat). Split into parts:
D_NLL = 1.15 − 8.48 = −7.32 and −D_entropy = −(−0.54 − (−0.91)) = −0.36 (oasst1); D_NLL = −7.38 and −D_entropy = +0.07 (UltraChat).
**The NLL part carries D**: 95% of it in oasst1, and all of it in UltraChat (where the entropy part is slightly positive). Both roles' entropy falls, and both roles' NLL rises.

What this supports:

- **User text becomes much more surprising relative to the model's predictive entropy than assistant text does.** Yes: the user
  gap rises by about 9 nats/token, the reply's by about 1.5. The user rise persists through the turn (Δ user NLL about 12 at positions
  1–5, 7–8 at positions 6–100, n ≥ 28 per bin), while the reply's falls off from 3–4 to about 0.5 (`figures/positions_*.png`).
- **The final ordering.** Separately from D (which concerns changes), the levels show the order reversing. At base the reply's gap
  is slightly *larger* than the user turn's (reply − user +0.19 [+0.11, +0.26] oasst1, +0.10 [+0.05, +0.16] UltraChat); at IT the
  user's gap is far larger (−7.50 [−7.97, −7.02], −7.21 [−7.56, −6.86]).
- **Consistent with a relative assistant advantage**, the pattern vestigial regression looks for: the assistant's text is
  predicted much better than the user's, relative to the model's own uncertainty. It is relative. Assistant prediction does **not**
  improve: the reply's NLL also rises (+1.0 to +1.2), it just rises far less.
- **Partly not role-specific.** In plain framing (tags removed) Gemma IT's NLL also rises, for user text (+5.2) and for replies (+3.1).
  The tags add to the user's rise (+3.3) and reduce the reply's (−2.0). Those two tagged − plain changes are vestigial regression's
  frame costs, and its separate asymmetry for Gemma IT (+5.2 / +5.3) is computed from them. This gap analysis is a different
  quantity on the same scores, and is not a second confirmation of that result or of the behavioural (judge) arm.

What it does **not** support:

- That Gemma is the strongest evidence of privilege. Its D is the largest in magnitude, but nats per token depend on the tokenizer.
  Gemma is one base/IT pair, n = 100 per corpus, contamination is unchecked, and the size comes from user-side degradation rather
  than anything the assistant persona gained. In bits per byte it is still the largest (−2.46 / −1.97 vs OLMo DPO +0.20 / +0.14), but
  magnitude across families is descriptive only.
- That a user gap of about 9 nats/token is a formal calibration failure. It says the recorded user text is far less likely than
  the model's own entropy implies; the gap does not test calibration.

### 5. Framing: OLMo's reply-specific rise depends on the role tags

In plain framing (the same tokens with tags removed), OLMo DPO / Instruct's (reply − user) NLL change shrinks to +0.15 to +0.21,
and the plain gap contrast reverses (−0.14 to −0.31). The extra NLL on recorded replies is largely tied to the tagged assistant
slot. Qwen's plain gap contrast is ≈ 0; Gemma's stays strongly negative but smaller (−2.8 to −3.2). Figures: `contrast_change.png`
(hollow markers), `changes_*.png`.

### 6. WildChat and cross-family comparisons

WildChat (human user, ChatGPT reply) follows the headline pattern (gap contrast +0.47 OLMo DPO, +0.39 Instruct, −0.51 Qwen IT),
but its roles differ in authorship, so it is reported separately and outside both classifications. On the common first 100
exchanges in bits per byte (`figures/cross_family_bpb.png`) the signs match those within families. This is descriptive, given
different tokenizers and unchecked contamination outside OLMo.

## What this does and does not settle

- **Settled, within this design.**
  - The original positive-direction gap criterion is met for OLMo DPO and Instruct and not met for OLMo SFT, Qwen3-8B IT and
    Gemma 4 31B IT.
  - Under the exploratory two-sided reading, every post-trained checkpoint shows a role-dependent gap change consistent in both
    corpora. It is positive for OLMo DPO and Instruct (through reply NLL), negative for OLMo SFT and Qwen IT (through user-turn
    entropy) and Gemma IT (through user-turn NLL). Qwen IT's UltraChat cell is at the floor.
  - No checkpoint shows assistant-specific concentration.
- **Uncertain.**
  - Which gap change, if any, is the "distortion" the measure intends. OLMo's positive D is a worsening fit to recorded replies;
    Gemma's negative D is user-side degradation. Neither is a concentration of the assistant's distribution.
  - Whether OLMo's rise reflects the model expecting replies of its own kind. This is a hypothesis, not tested here.
  - Qwen IT's UltraChat classification (fragile at the floor).
  - One base/instruct pair each for Qwen and Gemma (Gemma n = 100 per corpus).
  - Contamination outside OLMo.
  - Any reading as assistant privilege or dominance needs the other measures, not the sign of D.
- **Needs further collection.**
  - Follow-up user turns and longer histories, i.e. the original-conversation route: `pilot_v3` candidates, `core.score_conversations`,
    with a final candidate set and a scoring run per checkpoint.
  - The model's own replies, to test the "own kind of reply" hypothesis. Neither is part of this analysis.

## Reproduce

Generated from the code revision in `provenance.json`. Regenerating after the 2026-11-04 direction update left `summary.csv`,
`positions.csv`, `samples.csv`, `checks.json` and the numeric columns of `verdict.csv` byte-identical to the first run (`ad6cdcf`).
Input file hashes and the per-family inputs fingerprints are in `provenance.json`.

```bash
uv run python -m measures.logit_distortion.analyze      # ~1 min on CPU; writes this directory (report.md is hand-written, not regenerated)
uv run pytest tests/test_logit_distortion.py            # CPU-only, synthetic outputs
```
