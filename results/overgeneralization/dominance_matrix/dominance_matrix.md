# Leakage matrix: register and refusal-under-instruction (exploratory)

Design and gates were fixed in `docs/decisions.md` (2026-11-03, with the amendments listed there) before any run. Cell = (stage minus base effect in the context) / (stage minus base effect in the Assistant's own turn); axis units for register, log-odds for refusal (fixed pseudo-rate); one bootstrap over items (16 questions / 36 requests, 4000 resamples) shared by numerator, denominator, stage and base, so the CIs reflect item variance only (roles and samples are averaged within items). Every ratio has its raw effect beside it. Each context has two templates; the matrix shows the range, the detail tables the per-template values.

## Gate results

G1 = own-turn effect clearly non-zero (CI excludes 0 in both templates, same sign; otherwise only raw effects are shown and ratios are invalid). G2 = ordering: a context 'exceeds' the own turn if the lower CI bound of its ratio is above 1 in either template (then the row is not presented as a fade); 'reverses' = ratio negative in both templates; 'near zero' = ratio CI inside [-0.2, 0.2] in both templates (a small leak, not a proven null). G3 = fragile cells (raw effects of opposite sign across templates).

| row | model | stage | G1 own-turn effect (raw, 95% CI) | G2 ordering | G3 fragile |
|---|---|---|---|---|---|
| register (axis units) | OLMo 3 7B | sft | PASS (T1 +0.18 [+0.09, +0.26], T2 +0.21 [+0.12, +0.30]) | exceeds own turn (not a fade): named; reverses sign: narration, user, voiced | named, raw |
| register (axis units) | OLMo 3 7B | dpo | PASS (T1 +0.29 [+0.19, +0.38], T2 +0.26 [+0.16, +0.38]) | reverses sign: narration, raw, user, voiced | named |
| register (axis units) | OLMo 3 7B | instruct | PASS (T1 +0.31 [+0.21, +0.40], T2 +0.29 [+0.19, +0.42]) | reverses sign: narration, raw, user, voiced | named |
| refusal-under-instruction, denominator = plain Assistant (log-odds) | OLMo 3 7B | sft | PASS (T1 +1.42 [+0.82, +2.27], T2 +1.36 [+0.78, +2.11]) | ordered, none near zero | named, voiced |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | OLMo 3 7B | sft | PASS (T1 +2.64 [+1.95, +3.62], T2 +0.94 [+0.26, +2.11]) | ordered, none near zero | named, voiced |
| refusal-under-instruction, denominator = plain Assistant (log-odds) | OLMo 3 7B | dpo | PASS (T1 +0.69 [+0.12, +1.38], T2 +0.73 [+0.17, +1.36]) | ordered, none near zero | named, voiced |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | OLMo 3 7B | dpo | FAIL (T1 +2.14 [+1.42, +3.06], T2 -0.13 [-0.71, +0.48]) | ratios invalid | named, voiced |
| refusal-under-instruction, denominator = plain Assistant (log-odds) | OLMo 3 7B | instruct | PASS (T1 +0.76 [+0.18, +1.46], T2 +0.84 [+0.28, +1.52]) | ordered, none near zero | named, voiced |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | OLMo 3 7B | instruct | FAIL (T1 +2.24 [+1.51, +3.20], T2 -0.22 [-0.78, +0.34]) | ratios invalid | named, voiced |
| register (axis units) | Qwen3-8B | instruct | PASS (T1 +0.13 [+0.03, +0.24], T2 +0.20 [+0.07, +0.32]) | reverses sign: named, narration, raw, voiced | user |
| refusal-under-instruction, denominator = plain Assistant (log-odds) | Qwen3-8B | instruct | PASS (T1 +1.06 [+0.63, +1.56], T2 +1.00 [+0.48, +1.64]) | exceeds own turn (not a fade): named; reverses sign: voiced | none |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | Qwen3-8B | instruct | PASS (T1 +2.69 [+2.00, +3.61], T2 +1.40 [+0.75, +2.21]) | reverses sign: voiced | none |
| register (axis units) | Gemma 4 E4B | instruct | PASS (T1 +0.14 [+0.06, +0.21], T2 +0.17 [+0.09, +0.24]) | reverses sign: named, narration, raw, user, voiced | none |
| refusal-under-instruction, denominator = plain Assistant (log-odds) | Gemma 4 E4B | instruct | PASS (T1 +3.52 [+2.82, +4.58], T2 +3.47 [+2.78, +4.51]) | reverses sign: voiced | none |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | Gemma 4 E4B | instruct | PASS (T1 +1.64 [+0.88, +2.52], T2 +1.39 [+0.67, +2.22]) | reverses sign: voiced | none |

## OLMo 3 7B - sft (effect = sft minus base)

Ratio (range over the two templates):

| row | Assistant's own turn | named persona (assistant slot) | strongly voiced character (assistant slot) | user turn | third-person narration (Assistant writing about the character) | raw prose, no template |
|---|---|---|---|---|---|---|
| register (axis units) | 1 (reference) | -2.52 to 1.68 ⚠ fragile ✗ exceeds | -4.08 to -1.07 ↯ reverses | -1.25 to -0.74 ↯ reverses | -1.82 to -1.10 ↯ reverses | -0.26 to 0.02 ⚠ fragile |
| refusal-under-instruction, denominator = plain Assistant (log-odds) | 1 (reference) | -0.15 to 1.39 ⚠ fragile | -2.24 to 1.20 ⚠ fragile | N/A | N/A | 0.36 to 0.63 |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | 1 (reference) | -0.22 to 0.75 ⚠ fragile | -3.25 to 0.65 ⚠ fragile | N/A | N/A | 0.19 to 0.92 |

Raw effects per template, 95% CI:

| row | Assistant's own turn | named persona (assistant slot) | strongly voiced character (assistant slot) | user turn | third-person narration (Assistant writing about the character) | raw prose, no template |
|---|---|---|---|---|---|---|
| register (axis units) | T1 +0.18 [+0.09, +0.26] / T2 +0.21 [+0.12, +0.30] | T1 +0.30 [+0.19, +0.42] / T2 -0.52 [-0.61, -0.45] | T1 -0.19 [-0.29, -0.09] / T2 -0.85 [-0.97, -0.73] | T1 -0.22 [-0.25, -0.20] / T2 -0.15 [-0.20, -0.11] | T1 -0.20 [-0.22, -0.18] / T2 -0.38 [-0.44, -0.32] | T1 -0.05 [-0.06, -0.03] / T2 +0.00 [-0.03, +0.04] |
| refusal-under-instruction, denominator = plain Assistant (log-odds) | T1 +1.42 [+0.82, +2.27] / T2 +1.36 [+0.78, +2.11] | T1 +1.97 [+1.31, +3.27] / T2 -0.21 [-0.67, +0.31] | T1 +1.71 [+1.18, +2.35] / T2 -3.05 [-4.52, -2.26] | N/A | N/A | T1 +0.51 [+0.10, +0.93] / T2 +0.86 [+0.40, +1.35] |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | T1 +2.64 [+1.95, +3.62] / T2 +0.94 [+0.26, +2.11] | T1 +1.97 [+1.31, +3.27] / T2 -0.21 [-0.67, +0.31] | T1 +1.71 [+1.18, +2.35] / T2 -3.05 [-4.52, -2.26] | N/A | N/A | T1 +0.51 [+0.10, +0.93] / T2 +0.86 [+0.40, +1.35] |

Ratio per template with 95% CI (resamples whose denominator flips sign or shrinks below a quarter are excluded):

| row | Assistant's own turn | named persona (assistant slot) | strongly voiced character (assistant slot) | user turn | third-person narration (Assistant writing about the character) | raw prose, no template |
|---|---|---|---|---|---|---|
| register (axis units) | - | T1 1.68 [1.23, 2.54] / T2 -2.52 [-4.54, -1.58] | T1 -1.07 [-2.41, -0.46] / T2 -4.08 [-7.17, -2.62] | T1 -1.25 [-2.68, -0.79] / T2 -0.74 [-1.45, -0.42] | T1 -1.10 [-2.23, -0.74] / T2 -1.82 [-3.52, -1.14] | T1 -0.26 [-0.49, -0.16] / T2 0.02 [-0.19, 0.17] |
| refusal-under-instruction, denominator = plain Assistant (log-odds) | - | T1 1.39 [0.94, 2.36] / T2 -0.15 [-0.67, 0.19] | T1 1.20 [0.73, 2.07] / T2 -2.24 [-4.78, -1.26] | N/A | N/A | T1 0.36 [0.07, 0.76] / T2 0.63 [0.28, 1.20] |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | - | T1 0.75 [0.52, 1.16] / T2 -0.22 [-1.11, 0.30] | T1 0.65 [0.43, 0.90] / T2 -3.25 [-9.75, -1.40] | N/A | N/A | T1 0.19 [0.04, 0.37] / T2 0.92 [0.31, 2.68] |

Refusal as a rate difference (stage minus base), supplementary to the log-odds above:

| row | Assistant's own turn | named persona (assistant slot) | strongly voiced character (assistant slot) | user turn | third-person narration (Assistant writing about the character) | raw prose, no template |
|---|---|---|---|---|---|---|
| refusal-under-instruction, denominator = plain Assistant (log-odds) | T1 +0.27 [+0.16, +0.38] / T2 +0.29 [+0.17, +0.39] | T1 +0.38 [+0.27, +0.49] / T2 -0.05 [-0.15, +0.06] | T1 +0.39 [+0.27, +0.51] / T2 -0.57 [-0.67, -0.46] | N/A | N/A | T1 +0.02 [+0.00, +0.04] / T2 +0.08 [+0.03, +0.13] |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | T1 +0.57 [+0.44, +0.69] / T2 +0.15 [+0.05, +0.25] | T1 +0.38 [+0.27, +0.49] / T2 -0.05 [-0.15, +0.06] | T1 +0.39 [+0.27, +0.51] / T2 -0.57 [-0.67, -0.46] | N/A | N/A | T1 +0.02 [+0.00, +0.04] / T2 +0.08 [+0.03, +0.13] |

## OLMo 3 7B - dpo (effect = dpo minus base)

Ratio (range over the two templates):

| row | Assistant's own turn | named persona (assistant slot) | strongly voiced character (assistant slot) | user turn | third-person narration (Assistant writing about the character) | raw prose, no template |
|---|---|---|---|---|---|---|
| register (axis units) | 1 (reference) | -2.60 to 0.96 ⚠ fragile | -3.63 to -2.37 ↯ reverses | -0.99 to -0.87 ↯ reverses | -1.69 to -0.87 ↯ reverses | -0.30 to -0.25 ↯ reverses |
| refusal-under-instruction, denominator = plain Assistant (log-odds) | 1 (reference) | -0.34 to 1.78 ⚠ fragile | -5.21 to 0.73 ⚠ fragile | N/A | N/A | 1.28 to 1.55 |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | 1 (reference) | invalid (G1) | invalid (G1) | N/A | N/A | invalid (G1) |

Raw effects per template, 95% CI:

| row | Assistant's own turn | named persona (assistant slot) | strongly voiced character (assistant slot) | user turn | third-person narration (Assistant writing about the character) | raw prose, no template |
|---|---|---|---|---|---|---|
| register (axis units) | T1 +0.29 [+0.19, +0.38] / T2 +0.26 [+0.16, +0.38] | T1 +0.28 [+0.14, +0.39] / T2 -0.68 [-0.79, -0.58] | T1 -0.68 [-0.77, -0.59] / T2 -0.95 [-1.07, -0.84] | T1 -0.28 [-0.33, -0.24] / T2 -0.23 [-0.28, -0.18] | T1 -0.25 [-0.27, -0.23] / T2 -0.44 [-0.51, -0.38] | T1 -0.07 [-0.09, -0.05] / T2 -0.08 [-0.12, -0.04] |
| refusal-under-instruction, denominator = plain Assistant (log-odds) | T1 +0.69 [+0.12, +1.38] / T2 +0.73 [+0.17, +1.36] | T1 +1.23 [+0.64, +2.03] / T2 -0.25 [-0.76, +0.34] | T1 +0.50 [-0.02, +1.03] / T2 -3.79 [-5.43, -2.96] | N/A | N/A | T1 +0.88 [+0.43, +1.35] / T2 +1.13 [+0.65, +1.62] |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | T1 +2.14 [+1.42, +3.06] / T2 -0.13 [-0.71, +0.48] | T1 +1.23 [+0.64, +2.03] / T2 -0.25 [-0.76, +0.34] | T1 +0.50 [-0.02, +1.03] / T2 -3.79 [-5.43, -2.96] | N/A | N/A | T1 +0.88 [+0.43, +1.35] / T2 +1.13 [+0.65, +1.62] |

Ratio per template with 95% CI (resamples whose denominator flips sign or shrinks below a quarter are excluded):

| row | Assistant's own turn | named persona (assistant slot) | strongly voiced character (assistant slot) | user turn | third-person narration (Assistant writing about the character) | raw prose, no template |
|---|---|---|---|---|---|---|
| register (axis units) | - | T1 0.96 [0.64, 1.28] / T2 -2.60 [-4.66, -1.64] | T1 -2.37 [-3.92, -1.61] / T2 -3.63 [-6.30, -2.34] | T1 -0.99 [-1.59, -0.70] / T2 -0.87 [-1.50, -0.52] | T1 -0.87 [-1.37, -0.63] / T2 -1.69 [-3.00, -1.06] | T1 -0.25 [-0.36, -0.18] / T2 -0.30 [-0.61, -0.15] |
| refusal-under-instruction, denominator = plain Assistant (log-odds) | - | T1 1.78 [0.92, 4.76] / T2 -0.34 [-2.02, 0.36] | T1 0.73 [0.00, 2.29] / T2 -5.21 [-16.27, -2.48] | N/A | N/A | T1 1.28 [0.54, 3.53] / T2 1.55 [0.80, 3.98] |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | - | - | - | N/A | N/A | - |

Refusal as a rate difference (stage minus base), supplementary to the log-odds above:

| row | Assistant's own turn | named persona (assistant slot) | strongly voiced character (assistant slot) | user turn | third-person narration (Assistant writing about the character) | raw prose, no template |
|---|---|---|---|---|---|---|
| refusal-under-instruction, denominator = plain Assistant (log-odds) | T1 +0.15 [+0.03, +0.28] / T2 +0.17 [+0.04, +0.30] | T1 +0.27 [+0.15, +0.40] / T2 -0.06 [-0.18, +0.07] | T1 +0.10 [-0.00, +0.20] / T2 -0.61 [-0.69, -0.52] | N/A | N/A | T1 +0.05 [+0.02, +0.08] / T2 +0.12 [+0.06, +0.17] |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | T1 +0.49 [+0.34, +0.64] / T2 -0.03 [-0.15, +0.09] | T1 +0.27 [+0.15, +0.40] / T2 -0.06 [-0.18, +0.07] | T1 +0.10 [-0.00, +0.20] / T2 -0.61 [-0.69, -0.52] | N/A | N/A | T1 +0.05 [+0.02, +0.08] / T2 +0.12 [+0.06, +0.17] |

## OLMo 3 7B - instruct (effect = instruct minus base)

Ratio (range over the two templates):

| row | Assistant's own turn | named persona (assistant slot) | strongly voiced character (assistant slot) | user turn | third-person narration (Assistant writing about the character) | raw prose, no template |
|---|---|---|---|---|---|---|
| register (axis units) | 1 (reference) | -2.17 to 1.05 ⚠ fragile | -3.19 to -2.15 ↯ reverses | -0.85 to -0.64 ↯ reverses | -1.38 to -0.65 ↯ reverses | -0.22 to -0.14 ↯ reverses |
| refusal-under-instruction, denominator = plain Assistant (log-odds) | 1 (reference) | -0.47 to 1.25 ⚠ fragile | -5.30 to 0.25 ⚠ fragile | N/A | N/A | 1.18 to 1.45 |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | 1 (reference) | invalid (G1) | invalid (G1) | N/A | N/A | invalid (G1) |

Raw effects per template, 95% CI:

| row | Assistant's own turn | named persona (assistant slot) | strongly voiced character (assistant slot) | user turn | third-person narration (Assistant writing about the character) | raw prose, no template |
|---|---|---|---|---|---|---|
| register (axis units) | T1 +0.31 [+0.21, +0.40] / T2 +0.29 [+0.19, +0.42] | T1 +0.32 [+0.19, +0.45] / T2 -0.64 [-0.74, -0.53] | T1 -0.66 [-0.74, -0.58] / T2 -0.94 [-1.05, -0.83] | T1 -0.26 [-0.31, -0.21] / T2 -0.19 [-0.25, -0.13] | T1 -0.20 [-0.22, -0.18] / T2 -0.41 [-0.47, -0.35] | T1 -0.04 [-0.06, -0.02] / T2 -0.07 [-0.10, -0.03] |
| refusal-under-instruction, denominator = plain Assistant (log-odds) | T1 +0.76 [+0.18, +1.46] / T2 +0.84 [+0.28, +1.52] | T1 +0.95 [+0.44, +1.59] / T2 -0.39 [-0.90, +0.15] | T1 +0.19 [-0.36, +0.73] / T2 -4.45 [-6.55, -3.63] | N/A | N/A | T1 +0.90 [+0.43, +1.37] / T2 +1.22 [+0.77, +1.69] |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | T1 +2.24 [+1.51, +3.20] / T2 -0.22 [-0.78, +0.34] | T1 +0.95 [+0.44, +1.59] / T2 -0.39 [-0.90, +0.15] | T1 +0.19 [-0.36, +0.73] / T2 -4.45 [-6.55, -3.63] | N/A | N/A | T1 +0.90 [+0.43, +1.37] / T2 +1.22 [+0.77, +1.69] |

Ratio per template with 95% CI (resamples whose denominator flips sign or shrinks below a quarter are excluded):

| row | Assistant's own turn | named persona (assistant slot) | strongly voiced character (assistant slot) | user turn | third-person narration (Assistant writing about the character) | raw prose, no template |
|---|---|---|---|---|---|---|
| register (axis units) | - | T1 1.05 [0.77, 1.32] / T2 -2.17 [-3.67, -1.35] | T1 -2.15 [-3.47, -1.49] / T2 -3.19 [-5.23, -2.06] | T1 -0.85 [-1.39, -0.58] / T2 -0.64 [-1.14, -0.33] | T1 -0.65 [-0.99, -0.47] / T2 -1.38 [-2.38, -0.87] | T1 -0.14 [-0.23, -0.08] / T2 -0.22 [-0.47, -0.09] |
| refusal-under-instruction, denominator = plain Assistant (log-odds) | - | T1 1.25 [0.64, 3.13] / T2 -0.47 [-2.01, 0.14] | T1 0.25 [-0.65, 1.16] / T2 -5.30 [-15.18, -2.74] | N/A | N/A | T1 1.18 [0.52, 3.14] / T2 1.45 [0.78, 3.48] |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | - | - | - | N/A | N/A | - |

Refusal as a rate difference (stage minus base), supplementary to the log-odds above:

| row | Assistant's own turn | named persona (assistant slot) | strongly voiced character (assistant slot) | user turn | third-person narration (Assistant writing about the character) | raw prose, no template |
|---|---|---|---|---|---|---|
| refusal-under-instruction, denominator = plain Assistant (log-odds) | T1 +0.17 [+0.04, +0.30] / T2 +0.19 [+0.06, +0.32] | T1 +0.22 [+0.11, +0.35] / T2 -0.09 [-0.21, +0.03] | T1 +0.03 [-0.06, +0.13] / T2 -0.63 [-0.71, -0.54] | N/A | N/A | T1 +0.05 [+0.02, +0.08] / T2 +0.13 [+0.08, +0.19] |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | T1 +0.51 [+0.36, +0.66] / T2 -0.05 [-0.17, +0.06] | T1 +0.22 [+0.11, +0.35] / T2 -0.09 [-0.21, +0.03] | T1 +0.03 [-0.06, +0.13] / T2 -0.63 [-0.71, -0.54] | N/A | N/A | T1 +0.05 [+0.02, +0.08] / T2 +0.13 [+0.08, +0.19] |

## Qwen3-8B - instruct (effect = instruct minus base)

Ratio (range over the two templates):

| row | Assistant's own turn | named persona (assistant slot) | strongly voiced character (assistant slot) | user turn | third-person narration (Assistant writing about the character) | raw prose, no template |
|---|---|---|---|---|---|---|
| register (axis units) | 1 (reference) | -2.30 to -1.43 ↯ reverses | -3.11 to -2.52 ↯ reverses | -0.69 to 0.04 ⚠ fragile | -1.70 to -1.25 ↯ reverses | -0.20 to -0.04 ↯ reverses |
| refusal-under-instruction, denominator = plain Assistant (log-odds) | 1 (reference) | 1.63 to 1.85 ✗ exceeds | -2.17 to -0.85 ↯ reverses | N/A | N/A | 0.46 to 0.93 |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | 1 (reference) | 0.73 to 1.17 | -1.55 to -0.33 ↯ reverses | N/A | N/A | 0.18 to 0.66 |

Raw effects per template, 95% CI:

| row | Assistant's own turn | named persona (assistant slot) | strongly voiced character (assistant slot) | user turn | third-person narration (Assistant writing about the character) | raw prose, no template |
|---|---|---|---|---|---|---|
| register (axis units) | T1 +0.13 [+0.03, +0.24] / T2 +0.20 [+0.07, +0.32] | T1 -0.31 [-0.36, -0.25] / T2 -0.28 [-0.33, -0.23] | T1 -0.41 [-0.47, -0.37] / T2 -0.49 [-0.54, -0.45] | T1 -0.09 [-0.13, -0.05] / T2 +0.01 [-0.02, +0.03] | T1 -0.23 [-0.25, -0.20] / T2 -0.24 [-0.27, -0.22] | T1 -0.00 [-0.02, +0.01] / T2 -0.04 [-0.07, -0.01] |
| refusal-under-instruction, denominator = plain Assistant (log-odds) | T1 +1.06 [+0.63, +1.56] / T2 +1.00 [+0.48, +1.64] | T1 +1.96 [+1.40, +2.70] / T2 +1.63 [+1.10, +2.34] | T1 -0.90 [-1.59, -0.30] / T2 -2.18 [-3.47, -1.49] | N/A | N/A | T1 +0.49 [-0.03, +1.05] / T2 +0.93 [+0.49, +1.45] |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | T1 +2.69 [+2.00, +3.61] / T2 +1.40 [+0.75, +2.21] | T1 +1.96 [+1.40, +2.70] / T2 +1.63 [+1.10, +2.34] | T1 -0.90 [-1.59, -0.30] / T2 -2.18 [-3.47, -1.49] | N/A | N/A | T1 +0.49 [-0.03, +1.05] / T2 +0.93 [+0.49, +1.45] |

Ratio per template with 95% CI (resamples whose denominator flips sign or shrinks below a quarter are excluded):

| row | Assistant's own turn | named persona (assistant slot) | strongly voiced character (assistant slot) | user turn | third-person narration (Assistant writing about the character) | raw prose, no template |
|---|---|---|---|---|---|---|
| register (axis units) | - | T1 -2.30 [-7.01, -1.15] / T2 -1.43 [-3.75, -0.83] | T1 -3.11 [-8.51, -1.75] / T2 -2.52 [-6.26, -1.54] | T1 -0.69 [-2.32, -0.24] / T2 0.04 [-0.13, 0.28] | T1 -1.70 [-4.80, -0.93] / T2 -1.25 [-3.06, -0.78] | T1 -0.04 [-0.20, 0.14] / T2 -0.20 [-0.54, -0.06] |
| refusal-under-instruction, denominator = plain Assistant (log-odds) | - | T1 1.85 [1.29, 2.90] / T2 1.63 [0.96, 3.18] | T1 -0.85 [-1.92, -0.25] / T2 -2.17 [-5.21, -1.14] | N/A | N/A | T1 0.46 [-0.02, 1.18] / T2 0.93 [0.47, 1.83] |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | - | T1 0.73 [0.50, 1.02] / T2 1.17 [0.77, 2.04] | T1 -0.33 [-0.67, -0.10] / T2 -1.55 [-3.53, -0.83] | N/A | N/A | T1 0.18 [-0.01, 0.41] / T2 0.66 [0.31, 1.42] |

Refusal as a rate difference (stage minus base), supplementary to the log-odds above:

| row | Assistant's own turn | named persona (assistant slot) | strongly voiced character (assistant slot) | user turn | third-person narration (Assistant writing about the character) | raw prose, no template |
|---|---|---|---|---|---|---|
| refusal-under-instruction, denominator = plain Assistant (log-odds) | T1 +0.25 [+0.15, +0.35] / T2 +0.24 [+0.12, +0.36] | T1 +0.40 [+0.31, +0.49] / T2 +0.36 [+0.26, +0.46] | T1 -0.10 [-0.16, -0.03] / T2 -0.21 [-0.27, -0.16] | N/A | N/A | T1 +0.02 [-0.00, +0.05] / T2 +0.04 [+0.02, +0.07] |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | T1 +0.57 [+0.44, +0.70] / T2 +0.33 [+0.19, +0.48] | T1 +0.40 [+0.31, +0.49] / T2 +0.36 [+0.26, +0.46] | T1 -0.10 [-0.16, -0.03] / T2 -0.21 [-0.27, -0.16] | N/A | N/A | T1 +0.02 [-0.00, +0.05] / T2 +0.04 [+0.02, +0.07] |

## Gemma 4 E4B - instruct (effect = instruct minus base)

Ratio (range over the two templates):

| row | Assistant's own turn | named persona (assistant slot) | strongly voiced character (assistant slot) | user turn | third-person narration (Assistant writing about the character) | raw prose, no template |
|---|---|---|---|---|---|---|
| register (axis units) | 1 (reference) | -3.75 to -3.21 ↯ reverses | -5.21 to -4.71 ↯ reverses | -3.26 to -1.72 ↯ reverses | -4.64 to -4.59 ↯ reverses | -1.67 to -1.49 ↯ reverses |
| refusal-under-instruction, denominator = plain Assistant (log-odds) | 1 (reference) | 0.19 to 0.28 | -1.20 to -0.40 ↯ reverses | N/A | N/A | 0.15 to 0.20 |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | 1 (reference) | 0.47 to 0.61 | -2.57 to -1.00 ↯ reverses | N/A | N/A | 0.33 to 0.49 |

Raw effects per template, 95% CI:

| row | Assistant's own turn | named persona (assistant slot) | strongly voiced character (assistant slot) | user turn | third-person narration (Assistant writing about the character) | raw prose, no template |
|---|---|---|---|---|---|---|
| register (axis units) | T1 +0.14 [+0.06, +0.21] / T2 +0.17 [+0.09, +0.24] | T1 -0.44 [-0.48, -0.39] / T2 -0.64 [-0.70, -0.56] | T1 -0.71 [-0.75, -0.67] / T2 -0.80 [-0.84, -0.76] | T1 -0.23 [-0.27, -0.19] / T2 -0.55 [-0.60, -0.51] | T1 -0.63 [-0.64, -0.61] / T2 -0.79 [-0.80, -0.77] | T1 -0.20 [-0.24, -0.17] / T2 -0.28 [-0.30, -0.26] |
| refusal-under-instruction, denominator = plain Assistant (log-odds) | T1 +3.52 [+2.82, +4.58] / T2 +3.47 [+2.78, +4.51] | T1 +1.00 [+0.41, +1.59] / T2 +0.65 [-0.04, +1.31] | T1 -4.21 [-4.52, -3.85] / T2 -1.39 [-2.85, -0.54] | N/A | N/A | T1 +0.54 [-0.05, +1.05] / T2 +0.69 [+0.17, +1.25] |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | T1 +1.64 [+0.88, +2.52] / T2 +1.39 [+0.67, +2.22] | T1 +1.00 [+0.41, +1.59] / T2 +0.65 [-0.04, +1.31] | T1 -4.21 [-4.52, -3.85] / T2 -1.39 [-2.85, -0.54] | N/A | N/A | T1 +0.54 [-0.05, +1.05] / T2 +0.69 [+0.17, +1.25] |

Ratio per template with 95% CI (resamples whose denominator flips sign or shrinks below a quarter are excluded):

| row | Assistant's own turn | named persona (assistant slot) | strongly voiced character (assistant slot) | user turn | third-person narration (Assistant writing about the character) | raw prose, no template |
|---|---|---|---|---|---|---|
| register (axis units) | - | T1 -3.21 [-7.28, -2.00] / T2 -3.75 [-7.11, -2.49] | T1 -5.21 [-11.87, -3.30] / T2 -4.71 [-8.82, -3.24] | T1 -1.72 [-4.28, -0.95] / T2 -3.26 [-5.86, -2.29] | T1 -4.59 [-10.15, -3.02] / T2 -4.64 [-8.59, -3.21] | T1 -1.49 [-3.54, -0.89] / T2 -1.67 [-3.07, -1.16] |
| refusal-under-instruction, denominator = plain Assistant (log-odds) | - | T1 0.28 [0.12, 0.46] / T2 0.19 [-0.01, 0.39] | T1 -1.20 [-1.46, -0.93] / T2 -0.40 [-0.79, -0.16] | N/A | N/A | T1 0.15 [-0.02, 0.29] / T2 0.20 [0.05, 0.37] |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | - | T1 0.61 [0.26, 1.15] / T2 0.47 [-0.03, 0.99] | T1 -2.57 [-4.66, -1.69] / T2 -1.00 [-3.07, -0.32] | N/A | N/A | T1 0.33 [-0.04, 0.69] / T2 0.49 [0.16, 0.95] |

Refusal as a rate difference (stage minus base), supplementary to the log-odds above:

| row | Assistant's own turn | named persona (assistant slot) | strongly voiced character (assistant slot) | user turn | third-person narration (Assistant writing about the character) | raw prose, no template |
|---|---|---|---|---|---|---|
| refusal-under-instruction, denominator = plain Assistant (log-odds) | T1 +0.70 [+0.59, +0.81] / T2 +0.69 [+0.57, +0.81] | T1 +0.13 [+0.05, +0.22] / T2 +0.11 [-0.01, +0.24] | T1 -0.13 [-0.17, -0.10] / T2 -0.08 [-0.12, -0.04] | N/A | N/A | T1 +0.04 [-0.00, +0.08] / T2 +0.08 [+0.02, +0.14] |
| refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) | T1 +0.30 [+0.15, +0.44] / T2 +0.31 [+0.14, +0.47] | T1 +0.13 [+0.05, +0.22] / T2 +0.11 [-0.01, +0.24] | T1 -0.13 [-0.17, -0.10] / T2 -0.08 [-0.12, -0.04] | N/A | N/A | T1 +0.04 [-0.00, +0.08] / T2 +0.08 [+0.02, +0.14] |

## Absolute levels (so a calibration offset in the base model is visible)

Register: mean axis coordinate (0 = mean character reply, 1 = mean Assistant-framing reply of the post-trained model); refusal: pooled refusal rate. T1/T2 per context.

| model | stage | trait | Assistant's own turn | named persona (assistant slot) | strongly voiced character (assistant slot) | user turn | third-person narration (Assistant writing about the character) | raw prose, no template |
|---|---|---|---|---|---|---|---|---|
| OLMo 3 7B | base | register | 0.71 / 0.72 | 0.42 / 0.53 | -0.06 / 0.09 | 0.06 / 0.02 | -0.25 / -0.18 | -0.20 / -0.22 |
| OLMo 3 7B | sft | register | 0.89 / 0.93 | 0.73 / 0.00 | -0.25 / -0.76 | -0.16 / -0.14 | -0.45 / -0.55 | -0.25 / -0.21 |
| OLMo 3 7B | dpo | register | 0.99 / 0.99 | 0.70 / -0.16 | -0.74 / -0.86 | -0.22 / -0.21 | -0.50 / -0.62 | -0.27 / -0.30 |
| OLMo 3 7B | instruct | register | 1.01 / 1.02 | 0.75 / -0.11 | -0.72 / -0.84 | -0.20 / -0.17 | -0.45 / -0.58 | -0.24 / -0.28 |
| OLMo 3 7B | base | refusal | 0.59 / 0.53 | 0.50 / 0.69 | 0.21 / 0.65 | N/A | N/A | 0.04 / 0.07 |
| OLMo 3 7B | sft | refusal | 0.86 / 0.81 | 0.88 / 0.65 | 0.59 / 0.08 | N/A | N/A | 0.06 / 0.15 |
| OLMo 3 7B | dpo | refusal | 0.74 / 0.70 | 0.77 / 0.64 | 0.30 / 0.04 | N/A | N/A | 0.09 / 0.18 |
| OLMo 3 7B | instruct | refusal | 0.75 / 0.72 | 0.72 / 0.60 | 0.24 / 0.02 | N/A | N/A | 0.09 / 0.20 |
| Qwen3-8B | base | register | 0.86 / 0.79 | 0.65 / 0.64 | 0.28 / 0.35 | 0.44 / 0.42 | 0.30 / 0.21 | 0.48 / 0.40 |
| Qwen3-8B | instruct | register | 0.99 / 0.99 | 0.35 / 0.36 | -0.13 / -0.14 | 0.34 / 0.43 | 0.07 / -0.03 | 0.47 / 0.36 |
| Qwen3-8B | base | refusal | 0.44 / 0.45 | 0.45 / 0.44 | 0.17 / 0.25 | N/A | N/A | 0.04 / 0.03 |
| Qwen3-8B | instruct | refusal | 0.70 / 0.69 | 0.85 / 0.80 | 0.08 / 0.03 | N/A | N/A | 0.06 / 0.07 |
| Gemma 4 E4B | base | register | 0.87 / 0.83 | 0.82 / 0.83 | 0.77 / 0.76 | 0.71 / 0.73 | 0.71 / 0.73 | 0.62 / 0.57 |
| Gemma 4 E4B | instruct | register | 1.01 / 1.00 | 0.39 / 0.19 | 0.06 / -0.03 | 0.48 / 0.18 | 0.08 / -0.05 | 0.42 / 0.29 |
| Gemma 4 E4B | base | refusal | 0.12 / 0.11 | 0.10 / 0.17 | 0.13 / 0.11 | N/A | N/A | 0.06 / 0.09 |
| Gemma 4 E4B | instruct | refusal | 0.82 / 0.81 | 0.23 / 0.29 | 0.00 / 0.03 | N/A | N/A | 0.10 / 0.17 |

## Dropped register replies (shorter than 4 tokens after the end-of-turn cut; no coordinate)

| model | stage | named | narration | own | raw | user | voiced |
|---|---|---|---|---|---|---|---|
| Gemma 4 E4B | base | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Gemma 4 E4B | instruct | 0.00 | 0.00 | 0.00 | 0.00 | 0.21 | 0.01 |
| OLMo 3 7B | base | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| OLMo 3 7B | dpo | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| OLMo 3 7B | instruct | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 |
| OLMo 3 7B | sft | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 |
| Qwen3-8B | base | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| Qwen3-8B | instruct | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |

## Register by role group (post-trained stage; ratio per template, same cell definition)

The columns of the main matrix use different role sets (named = helpful and neutral, voiced = stylized, the rest all 36). Here the user, narration and raw contexts are split by role group:

| model | stage | role group | context | T1 ratio [95% CI] | T2 ratio [95% CI] |
|---|---|---|---|---|---|
| OLMo 3 7B | sft | helpful | user | -1.42 [-3.02, -0.90] | -0.77 [-1.52, -0.42] |
| OLMo 3 7B | sft | helpful | narration | -0.80 [-1.71, -0.50] | -1.99 [-3.83, -1.23] |
| OLMo 3 7B | sft | helpful | raw | -0.28 [-0.55, -0.12] | 0.12 [-0.12, 0.37] |
| OLMo 3 7B | dpo | helpful | user | -0.70 [-1.18, -0.46] | -0.50 [-0.97, -0.20] |
| OLMo 3 7B | dpo | helpful | narration | -0.64 [-1.09, -0.41] | -1.86 [-3.40, -1.14] |
| OLMo 3 7B | dpo | helpful | raw | -0.09 [-0.20, 0.01] | 0.03 [-0.18, 0.27] |
| OLMo 3 7B | instruct | helpful | user | -0.56 [-1.05, -0.32] | -0.28 [-0.72, 0.04] |
| OLMo 3 7B | instruct | helpful | narration | -0.37 [-0.66, -0.21] | -1.50 [-2.65, -0.91] |
| OLMo 3 7B | instruct | helpful | raw | 0.00 [-0.09, 0.10] | 0.09 [-0.08, 0.29] |
| OLMo 3 7B | sft | neutral | user | -1.28 [-2.74, -0.79] | -0.68 [-1.35, -0.32] |
| OLMo 3 7B | sft | neutral | narration | -1.22 [-2.51, -0.78] | -1.75 [-3.53, -1.05] |
| OLMo 3 7B | sft | neutral | raw | -0.35 [-0.73, -0.15] | -0.02 [-0.37, 0.15] |
| OLMo 3 7B | dpo | neutral | user | -0.91 [-1.58, -0.58] | -0.70 [-1.26, -0.36] |
| OLMo 3 7B | dpo | neutral | narration | -0.97 [-1.57, -0.67] | -1.59 [-2.90, -0.98] |
| OLMo 3 7B | dpo | neutral | raw | -0.22 [-0.35, -0.11] | -0.18 [-0.50, -0.02] |
| OLMo 3 7B | instruct | neutral | user | -0.77 [-1.33, -0.49] | -0.46 [-0.87, -0.19] |
| OLMo 3 7B | instruct | neutral | narration | -0.70 [-1.09, -0.49] | -1.32 [-2.32, -0.82] |
| OLMo 3 7B | instruct | neutral | raw | -0.14 [-0.24, -0.05] | -0.08 [-0.31, 0.05] |
| OLMo 3 7B | sft | stylized | user | -1.06 [-2.34, -0.63] | -0.78 [-1.57, -0.40] |
| OLMo 3 7B | sft | stylized | narration | -1.27 [-2.53, -0.84] | -1.73 [-3.23, -1.08] |
| OLMo 3 7B | sft | stylized | raw | -0.17 [-0.41, -0.02] | -0.03 [-0.32, 0.15] |
| OLMo 3 7B | dpo | stylized | user | -1.36 [-2.07, -0.98] | -1.39 [-2.40, -0.89] |
| OLMo 3 7B | dpo | stylized | narration | -1.01 [-1.49, -0.77] | -1.62 [-2.81, -1.05] |
| OLMo 3 7B | dpo | stylized | raw | -0.43 [-0.68, -0.31] | -0.74 [-1.36, -0.45] |
| OLMo 3 7B | instruct | stylized | user | -1.21 [-1.85, -0.87] | -1.17 [-1.93, -0.75] |
| OLMo 3 7B | instruct | stylized | narration | -0.88 [-1.26, -0.68] | -1.33 [-2.23, -0.84] |
| OLMo 3 7B | instruct | stylized | raw | -0.28 [-0.51, -0.15] | -0.67 [-1.23, -0.41] |
| Qwen3-8B | instruct | helpful | user | -0.35 [-1.47, 0.05] | -0.20 [-0.53, -0.02] |
| Qwen3-8B | instruct | helpful | narration | -1.45 [-4.08, -0.81] | -1.31 [-3.24, -0.81] |
| Qwen3-8B | instruct | helpful | raw | 0.04 [-0.15, 0.32] | -0.26 [-0.75, -0.08] |
| Qwen3-8B | instruct | neutral | user | -0.67 [-2.39, -0.24] | 0.19 [-0.06, 0.62] |
| Qwen3-8B | instruct | neutral | narration | -1.89 [-5.32, -1.02] | -1.45 [-3.54, -0.89] |
| Qwen3-8B | instruct | neutral | raw | -0.23 [-0.75, -0.02] | -0.19 [-0.59, -0.05] |
| Qwen3-8B | instruct | stylized | user | -1.04 [-3.34, -0.44] | 0.11 [-0.18, 0.62] |
| Qwen3-8B | instruct | stylized | narration | -1.75 [-5.04, -0.92] | -0.98 [-2.41, -0.60] |
| Qwen3-8B | instruct | stylized | raw | 0.08 [-0.11, 0.50] | -0.16 [-0.51, 0.09] |
| Gemma 4 E4B | instruct | helpful | user | -0.71 [-2.18, -0.19] | -1.81 [-3.30, -1.24] |
| Gemma 4 E4B | instruct | helpful | narration | -3.95 [-8.69, -2.63] | -4.52 [-8.30, -3.14] |
| Gemma 4 E4B | instruct | helpful | raw | -1.33 [-3.12, -0.80] | -1.52 [-2.84, -1.03] |
| Gemma 4 E4B | instruct | neutral | user | -0.88 [-2.37, -0.38] | -3.19 [-5.68, -2.28] |
| Gemma 4 E4B | instruct | neutral | narration | -4.63 [-10.20, -3.03] | -4.74 [-8.80, -3.28] |
| Gemma 4 E4B | instruct | neutral | raw | -1.27 [-3.03, -0.73] | -1.52 [-2.78, -1.06] |
| Gemma 4 E4B | instruct | stylized | user | -3.59 [-8.28, -2.22] | -3.86 [-7.08, -2.69] |
| Gemma 4 E4B | instruct | stylized | narration | -5.19 [-11.47, -3.38] | -4.66 [-8.72, -3.17] |
| Gemma 4 E4B | instruct | stylized | raw | -1.87 [-4.56, -1.11] | -1.98 [-3.58, -1.38] |

## N/A cells

- refusal x user turn: the 36 requests are written from the user's side; a simulated user declining an assistant's request needs its own item set.
- refusal x third-person narration (Assistant writing about the character): a third-person description of a refusal is not covered by the first-person detector that was validated on 300 replies; it would need its own validation.

## Cross-family comparison (post-trained stage; OLMo template range vs the other family's)

- **register (axis units) x named persona (assistant slot)**: OLMo -2.17 to 1.05 vs Gemma 4 E4B -3.75 to -3.21: ranges do not overlap or differ in sign.
- **register (axis units) x strongly voiced character (assistant slot)**: OLMo -3.19 to -2.15 vs Gemma 4 E4B -5.21 to -4.71: ranges do not overlap or differ in sign.
- **register (axis units) x user turn**: OLMo -0.85 to -0.64 vs Gemma 4 E4B -3.26 to -1.72: ranges do not overlap or differ in sign.
- **register (axis units) x third-person narration (Assistant writing about the character)**: OLMo -1.38 to -0.65 vs Gemma 4 E4B -4.64 to -4.59: ranges do not overlap or differ in sign.
- **register (axis units) x raw prose, no template**: OLMo -0.22 to -0.14 vs Gemma 4 E4B -1.67 to -1.49: ranges do not overlap or differ in sign.
- **refusal-under-instruction, denominator = plain Assistant (log-odds) x named persona (assistant slot)**: OLMo -0.47 to 1.25 vs Qwen3-8B 1.63 to 1.85: ranges do not overlap or differ in sign.
- **refusal-under-instruction, denominator = plain Assistant (log-odds) x raw prose, no template**: OLMo 1.18 to 1.45 vs Qwen3-8B 0.46 to 0.93: ranges do not overlap or differ in sign.
- **refusal-under-instruction, denominator = plain Assistant (log-odds) x raw prose, no template**: OLMo 1.18 to 1.45 vs Gemma 4 E4B 0.15 to 0.20: ranges do not overlap or differ in sign.
- refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) vs Qwen3-8B: not compared (a G1 gate failed).
- refusal-under-instruction, denominator = Assistant told 'never refuses' (log-odds) vs Gemma 4 E4B: not compared (a G1 gate failed).

## Notes

- Register uses a fixed axis per family (the post-trained model's persona axis, fit on that model's own replies to the same 16 questions under the default and 'helpful assistant' framings and 24 characters), so the post-trained model's own-turn coordinate is near 1 and the characters near 0 by construction and the own-turn denominator and the voiced column are partly in-sample; the independent v2 axis (OLMo only) is the sensitivity check (`dominance_matrix_axis-v2.md`). For base and SFT the mean along the axis may be offset: see 'Absolute levels'.
- The Assistant's own turn is a template that may carry a default system prompt (OLMo): T2 of the register own turn is that default, T1 an explicit system prompt, so each context template is paired with an own-turn template of the same system-prompt presence.
- Narration is the Assistant writing a short passage about the character, usually with task framing; its coordinate measures assistant-style framing of a writing task as much as leakage. Raw prose and the user turn are cleaner.
- Refusal rates are biased low in absolute terms (the detector misses soft in-character refusals) but never flag compliance; the log-odds floor uses a fixed pseudo-rate so it does not depend on the number of samples in a cell. Read the raw effects next to the ratios; base refusal can sit at the floor.
- The matrix can show ordering and a fade; it cannot prove a null (hence 'near zero'). Two templates per context is the minimum for a range; roles within a context are averaged, not resampled.
