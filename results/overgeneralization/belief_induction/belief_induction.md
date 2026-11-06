# Belief test robustness: three ways of inducing a persona (spec O; exploratory, outside every verdict, NOT blind)

Plan: `docs/decisions.md` 2026-11-05. The pre-registered belief test and its results are unchanged. Arm 1 = the saved confirmatory prompt results; arm 2 = persona only in the model's own turn (first person), scored by the unchanged `persona_carryover` code and criteria on `data/belief_induction_prefill.jsonl`; arm 3 = the Assistant-vs-character direction added on the final model turn's tokens only (OLMo and Qwen final; Gemma 31B out for now). Limits: not blind; exploratory; few domains; arm 2 still uses words, in a different place; arm 3 tests a generic push, not a named person, and shares variance with formality (cosine 0.42 OLMo, 0.53 Qwen); contamination can only be checked for OLMo.

Sanity check (12 unchanged prompt items, OLMo final, rescored): {'n': 288, 'max_abs_diff': 1.3756284715782385e-06, 'mean_abs_diff': 3.296840975519474e-07}.

## Readings by model and arm

| model | arm | assistant>new base | assistant>new final | character>new base | character>new final | leak-gap change | reading |
|---|---|---|---|---|---|---|---|
| olmo3-7b | prompt | +0.59 [+0.32, +0.87] | -0.24 [-0.89, +0.43] | +0.26 [-0.05, +0.56] | -1.96 [-2.73, -1.17] | +1.38 [+1.02, +1.74] | relative only (criterion 3 holds, criterion 4 does not) |
| olmo3-7b | prefill | +0.60 [+0.43, +0.77] | +0.02 [-0.44, +0.48] | +0.34 [+0.14, +0.55] | -0.85 [-1.36, -0.34] | +0.62 [+0.43, +0.82] | MATCHES THE PROMPT (relative only (criterion 3 holds, criterion 4 does not)) |
| qwen3-8b | prompt | +0.18 [+0.02, +0.34] | -0.78 [-1.49, -0.09] | -0.18 [-0.40, +0.04] | -2.34 [-3.12, -1.58] | +1.21 [+0.72, +1.69] | relative only (criterion 3 holds, criterion 4 does not) |
| qwen3-8b | prefill | +0.60 [+0.48, +0.72] | +1.14 [+0.86, +1.44] | +0.30 [+0.14, +0.47] | +0.29 [-0.09, +0.66] | +0.55 [+0.27, +0.85] | DIFFERS: prefill 'no overgeneralization by the rule', prompt 'relative only (criterion 3 holds, criterion 4 does not)'; assistant>new change from base +0.54 vs -0.95 |
| gemma4-31b | prompt | +0.15 [+0.02, +0.28] | +0.19 [-0.62, +1.05] | -0.05 [-0.16, +0.06] | -2.36 [-3.12, -1.60] | +2.35 [+1.52, +3.18] | relative only (criterion 3 holds, criterion 4 does not) |
| gemma4-31b | prefill | +0.70 [+0.59, +0.82] | +3.69 [+2.48, +4.85] | +0.23 [+0.14, +0.33] | -4.74 [-5.72, -3.74] | +7.96 [+6.55, +9.42] | MATCHES THE PROMPT (relative only (criterion 3 holds, criterion 4 does not)) |
| olmo3-7b | steering (final only) | - | strength 0 vs -2 in the arm 3 table | - | - | no leak gap in this arm | CHECK FAILS: inconclusive |
| qwen3-8b | steering (final only) | - | strength 0 vs -2 in the arm 3 table | - | - | no leak gap in this arm | CHECK FAILS: inconclusive |

**Gemma, like-for-like prompt figure.** The Gemma prompt row above is the run without the thought-channel opening (assistant>new final +0.19). The prefill arm opens with it. The same prompt arm with the thought channel opened (`results/overgeneralization/persona_carryover/gemma4-31b/confirmatory_thought_channel`, final): assistant>new -1.92 [-2.87, -0.93], character>new -4.76 [-6.06, -3.50], leak gap +2.84 [+2.00, +3.76]. Against that, the prefill's assistant>new final (+3.69) is higher by about 5.6 log-odds units.
**What MATCHES THE PROMPT means.** It refers only to the pre-set rule's reading (here 'relative only': criterion 3 holds, criterion 4 does not), not to the size of the effect. For Gemma assistant>new is much higher with prefill than with the prompt, whichever prompt run is taken as the reference.

## olmo3-7b: persona prefill (arm 2) against the prompt (arm 1)

Per-domain stage verdict tables and criterion 6 reporting (raw log P, prior, mass covered, tokenization) are in `arm2/olmo3-7b/persona_carryover.md`.

## qwen3-8b: persona prefill (arm 2) against the prompt (arm 1)

Per-domain stage verdict tables and criterion 6 reporting (raw log P, prior, mass covered, tokenization) are in `arm2/qwen3-8b/persona_carryover.md`.

## gemma4-31b: persona prefill (arm 2) against the prompt (arm 1)

Per-domain stage verdict tables and criterion 6 reporting (raw log P, prior, mass covered, tokenization) are in `arm2/gemma4-31b/persona_carryover.md`.

## olmo3-7b (steering layer 18, held-out AUC 0.89)

Steering check: held-out AUC at the steering layer 0.89 (needs >= 0.80); plain-Assistant identity log-odds (human minus AI) at strength 0: +0.13; change at -2: -2.34, at +2: -1.88; largest random-direction change toward human at +-2: -0.12 -> **CHECK FAILS: every belief null below is inconclusive**.

| domain | strength 0 carryover (assistant>same) | -2 | -1 | +1 | +2 | change at -2 [95% CI] | random band at +-2 (min, max change) | reading |
|---|---|---|---|---|---|---|---|---|
| dice | +4.38 [+3.74, +5.05] | +4.10 | +4.32 | +4.34 | +4.07 | -0.28 [-0.47, -0.10] | -1.09, +0.88 | inconclusive (steering check fails) |
| drawer | -1.31 [-2.48, -0.11] | -1.08 | -1.17 | -1.44 | -1.56 | +0.23 [-0.13, +0.55] | -0.89, +0.67 | positive control fails at strength 0 |
| match | +1.81 [+1.02, +2.60] | +1.66 | +1.75 | +1.83 | +1.84 | -0.15 [-0.26, -0.05] | -0.35, +0.38 | inconclusive (steering check fails) |
| weather | +3.80 [+3.25, +4.30] | +3.44 | +3.61 | +3.90 | +3.71 | -0.36 [-0.67, -0.07] | -0.32, +0.13 | inconclusive (steering check fails) |
| all | +2.17 [+1.38, +2.87] | +2.03 | +2.13 | +2.16 | +2.01 | -0.14 [-0.29, +0.00] | -0.37, +0.24 |  |

Prior check (assistant-none>same, mean log-odds toward the 'believed' outcome with no belief stated; should stay near 0): -2: +0.00, -1: +0.00, +0: +0.00, +1: +0.00, +2: +0.00.

## qwen3-8b (steering layer 20, held-out AUC 0.98)

Steering check: held-out AUC at the steering layer 0.98 (needs >= 0.80); plain-Assistant identity log-odds (human minus AI) at strength 0: -22.75; change at -2: +7.88, at +2: -0.12; largest random-direction change toward human at +-2: +8.00 -> **CHECK FAILS: every belief null below is inconclusive**.

| domain | strength 0 carryover (assistant>same) | -2 | -1 | +1 | +2 | change at -2 [95% CI] | random band at +-2 (min, max change) | reading |
|---|---|---|---|---|---|---|---|---|
| dice | +6.48 [+5.45, +7.42] | +3.05 | +4.62 | +6.07 | +3.90 | -3.43 [-4.33, -2.48] | -4.37, +2.14 | inconclusive (steering check fails) |
| drawer | +5.30 [+4.02, +6.74] | -3.40 | -0.02 | +7.99 | +7.09 | -8.69 [-10.84, -6.70] | -2.80, +2.61 | inconclusive (steering check fails) |
| match | +6.92 [+5.80, +8.07] | +2.64 | +5.97 | +6.03 | +4.32 | -4.28 [-5.12, -3.58] | -2.83, +0.69 | inconclusive (steering check fails) |
| weather | +8.26 [+7.27, +9.24] | +3.15 | +6.02 | +8.14 | +6.57 | -5.11 [-6.47, -3.90] | -2.22, +1.22 | inconclusive (steering check fails) |
| all | +6.74 [+6.09, +7.39] | +1.36 | +4.15 | +7.06 | +5.47 | -5.38 [-6.28, -4.52] | -2.29, +0.99 |  |

Prior check (assistant-none>same, mean log-odds toward the 'believed' outcome with no belief stated; should stay near 0): -2: +0.00, -1: +0.00, +0: +0.00, +1: +0.00, +2: +0.00.

