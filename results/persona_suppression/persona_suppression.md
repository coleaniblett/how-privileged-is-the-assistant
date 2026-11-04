# Is the Assistant the fallback? (spec F-main)

Exploratory, outside every verdict. Plan, categories, swap partners and neutral texts frozen before scoring (`docs/decisions.md` 2026-11-04, `data/persona_suppression_materials.json`). Persona text is replaced in the same chat-template slot (system, or the user turn); intact = the persona's own description, swap = a different entity type's description of similar length (fixed partner), neutral = a role-neutral text of matched token length, assistant = no persona text. Onset readout: mean over 6 continuations per category of the summed token log-probability at the first content token (bf16); `logodds` = (assistant greeting + list)/2 - character voice. Generations: 4 samples x 40 tokens per prompt version, temperature 0.7; `P(assistant)` from the spec A classifier. Bootstrap over personas (10,000 resamples), probes averaged within persona. Qwen3-8B is supporting (no contamination check).

## olmo3-7b base

Shared-cache scoring vs full batch-1 pass: max |diff| 0.109 nats (first 3 prompts).

## olmo3-7b instruct

Shared-cache scoring vs full batch-1 pass: max |diff| 0.311 nats (first 3 prompts).

## qwen3-8b base

Shared-cache scoring vs full batch-1 pass: max |diff| 0.369 nats (first 3 prompts).

## qwen3-8b instruct

Shared-cache scoring vs full batch-1 pass: max |diff| 0.565 nats (first 3 prompts).

## Validation: does swap move the profile toward the swap partner?

Correlation of the swap shift with the partner shift (four category log-probs, centred per category over personas), training personas. **This check is uninformative: r = 1 by construction** (a swap prompt is identical to the partner's intact prompt; see A1 below, added after seeing the OLMo results), so the 'used' column is replaced by A1's check.

| family | stage | probes | r [95% CI] | used |
|---|---|---|---|---|
| olmo3-7b | base | factual | +1.00 [+1.00, +1.00] | yes |
| olmo3-7b | base | chat | +1.00 [+1.00, +1.00] | yes |
| olmo3-7b | instruct | factual | +1.00 [+1.00, +1.00] | yes |
| olmo3-7b | instruct | chat | +1.00 [+1.00, +1.00] | yes |
| qwen3-8b | base | factual | +1.00 [+1.00, +1.00] | yes |
| qwen3-8b | base | chat | +1.00 [+1.00, +1.00] | yes |
| qwen3-8b | instruct | factual | +1.00 [+1.00, +1.00] | yes |
| qwen3-8b | instruct | chat | +1.00 [+1.00, +1.00] | yes |

## Fallback: neutral replaces the persona

Pre-set: fallback if the gap is at least 0.5 nats (logodds) or 0.1 classifier points, the neutral shift is at least 50% of the gap with CI excluding 0, and the fraction is at least 0.3 larger in instruct than in base (below).

### Category log-odds (nats), training personas

| family | stage | personas | slots | probes | gap (assistant - intact) | numerator (neutral - intact) | fraction | reading |
|---|---|---|---|---|---|---|---|---|
| olmo3-7b | base | train (n=60) | all | factual | +1.95 [+1.60, +2.31] | +2.25 [+1.92, +2.58] | 1.16 [1.07, 1.26] | fallback |
| olmo3-7b | base | train (n=60) | all | chat | +2.24 [+1.78, +2.71] | +2.20 [+1.80, +2.59] | 0.98 [0.87, 1.11] | fallback |
| olmo3-7b | base | train (n=24) | system | factual | +2.63 [+2.06, +3.20] | +2.17 [+1.59, +2.74] | 0.82 [0.77, 0.86] | fallback |
| olmo3-7b | base | train (n=24) | system | chat | +3.33 [+2.62, +4.05] | +1.98 [+1.34, +2.63] | 0.59 [0.51, 0.65] | fallback |
| olmo3-7b | base | train (n=36) | user-turn | factual | +1.49 [+1.11, +1.89] | +2.31 [+1.93, +2.71] | 1.55 [1.43, 1.73] | fallback |
| olmo3-7b | base | train (n=36) | user-turn | chat | +1.51 [+1.03, +1.99] | +2.34 [+1.87, +2.82] | 1.55 [1.42, 1.81] | fallback |
| olmo3-7b | instruct | train (n=60) | all | factual | +11.56 [+9.78, +13.25] | +9.85 [+8.12, +11.48] | 0.85 [0.83, 0.87] | fallback |
| olmo3-7b | instruct | train (n=60) | all | chat | +9.91 [+8.34, +11.40] | +8.60 [+7.17, +9.99] | 0.87 [0.84, 0.89] | fallback |
| olmo3-7b | instruct | train (n=24) | system | factual | +8.81 [+5.94, +11.72] | +7.56 [+4.86, +10.31] | 0.86 [0.81, 0.88] | fallback |
| olmo3-7b | instruct | train (n=24) | system | chat | +8.00 [+5.58, +10.47] | +7.73 [+5.50, +10.00] | 0.97 [0.94, 1.00] | fallback |
| olmo3-7b | instruct | train (n=36) | user-turn | factual | +13.39 [+11.48, +15.28] | +11.37 [+9.46, +13.28] | 0.85 [0.82, 0.87] | fallback |
| olmo3-7b | instruct | train (n=36) | user-turn | chat | +11.19 [+9.37, +12.95] | +9.19 [+7.40, +10.89] | 0.82 [0.79, 0.85] | fallback |
| qwen3-8b | base | train (n=60) | all | factual | +2.51 [+2.09, +2.93] | +2.62 [+2.16, +3.09] | 1.04 [1.00, 1.08] | fallback |
| qwen3-8b | base | train (n=60) | all | chat | +2.19 [+1.76, +2.61] | +2.19 [+1.76, +2.62] | 1.00 [0.96, 1.04] | fallback |
| qwen3-8b | base | train (n=24) | system | factual | +2.83 [+2.00, +3.66] | +3.02 [+2.13, +3.91] | 1.07 [1.04, 1.09] | fallback |
| qwen3-8b | base | train (n=24) | system | chat | +2.51 [+1.75, +3.27] | +2.27 [+1.49, +3.06] | 0.90 [0.85, 0.94] | fallback |
| qwen3-8b | base | train (n=36) | user-turn | factual | +2.30 [+1.88, +2.71] | +2.36 [+1.87, +2.84] | 1.03 [0.96, 1.10] | fallback |
| qwen3-8b | base | train (n=36) | user-turn | chat | +1.97 [+1.51, +2.43] | +2.14 [+1.66, +2.63] | 1.09 [1.03, 1.16] | fallback |
| qwen3-8b | instruct | train (n=60) | all | factual | +15.85 [+14.41, +17.24] | +14.47 [+13.00, +15.86] | 0.91 [0.90, 0.93] | fallback |
| qwen3-8b | instruct | train (n=60) | all | chat | +8.73 [+7.54, +9.86] | +9.71 [+8.57, +10.79] | 1.11 [1.09, 1.15] | fallback |
| qwen3-8b | instruct | train (n=24) | system | factual | +16.68 [+14.15, +19.11] | +14.71 [+12.23, +17.09] | 0.88 [0.86, 0.90] | fallback |
| qwen3-8b | instruct | train (n=24) | system | chat | +9.84 [+7.86, +11.73] | +10.29 [+8.33, +12.16] | 1.05 [1.03, 1.07] | fallback |
| qwen3-8b | instruct | train (n=36) | user-turn | factual | +15.29 [+13.62, +16.86] | +14.30 [+12.57, +15.97] | 0.94 [0.91, 0.95] | fallback |
| qwen3-8b | instruct | train (n=36) | user-turn | chat | +7.98 [+6.58, +9.27] | +9.32 [+7.99, +10.56] | 1.17 [1.13, 1.22] | fallback |

### Classifier P(assistant), training personas

| family | stage | personas | slots | probes | gap (assistant - intact) | numerator (neutral - intact) | fraction | reading |
|---|---|---|---|---|---|---|---|---|
| olmo3-7b | base | train (n=60) | all | factual | +0.006 [-0.013, +0.028] | +0.012 [-0.011, +0.037] | 1.95 [-7.65, 8.31] | no |
| olmo3-7b | base | train (n=60) | all | chat | -0.012 [-0.028, +0.005] | -0.053 [-0.071, -0.036] | 4.52 [-27.22, 35.79] | no |
| olmo3-7b | base | train (n=24) | system | factual | +0.044 [+0.005, +0.086] | +0.060 [+0.016, +0.107] | 1.37 [1.00, 2.82] | no |
| olmo3-7b | base | train (n=24) | system | chat | +0.042 [+0.026, +0.060] | -0.018 [-0.042, +0.003] | -0.43 [-1.36, 0.07] | no |
| olmo3-7b | base | train (n=36) | user-turn | factual | -0.019 [-0.034, -0.002] | -0.020 [-0.038, -0.001] | 1.06 [0.09, 2.63] | no |
| olmo3-7b | base | train (n=36) | user-turn | chat | -0.048 [-0.065, -0.032] | -0.077 [-0.099, -0.055] | 1.60 [1.27, 2.08] | no |
| olmo3-7b | instruct | train (n=60) | all | factual | +0.121 [+0.086, +0.157] | -0.055 [-0.089, -0.021] | -0.46 [-0.99, -0.14] | no |
| olmo3-7b | instruct | train (n=60) | all | chat | +0.010 [-0.028, +0.049] | -0.101 [-0.145, -0.057] | -9.88 [-78.59, 75.66] | no |
| olmo3-7b | instruct | train (n=24) | system | factual | +0.111 [+0.056, +0.170] | +0.024 [-0.018, +0.067] | 0.22 [-0.32, 0.40] | no |
| olmo3-7b | instruct | train (n=24) | system | chat | +0.084 [+0.043, +0.125] | +0.027 [-0.000, +0.055] | 0.32 [-0.01, 0.52] | no |
| olmo3-7b | instruct | train (n=36) | user-turn | factual | +0.127 [+0.084, +0.174] | -0.108 [-0.147, -0.066] | -0.85 [-1.73, -0.38] | no |
| olmo3-7b | instruct | train (n=36) | user-turn | chat | -0.039 [-0.090, +0.014] | -0.186 [-0.240, -0.131] | 4.81 [-23.75, 34.56] | no |
| qwen3-8b | base | train (n=60) | all | factual | -0.037 [-0.053, -0.020] | -0.033 [-0.052, -0.014] | 0.89 [0.49, 1.35] | no |
| qwen3-8b | base | train (n=60) | all | chat | +0.030 [+0.012, +0.048] | -0.082 [-0.105, -0.058] | -2.74 [-8.05, -1.32] | no |
| qwen3-8b | base | train (n=24) | system | factual | -0.042 [-0.068, -0.015] | -0.029 [-0.063, +0.006] | 0.69 [-0.20, 1.42] | no |
| qwen3-8b | base | train (n=24) | system | chat | +0.057 [+0.033, +0.080] | -0.068 [-0.112, -0.026] | -1.20 [-2.95, -0.38] | no |
| qwen3-8b | base | train (n=36) | user-turn | factual | -0.033 [-0.056, -0.011] | -0.035 [-0.057, -0.013] | 1.06 [0.62, 2.05] | no |
| qwen3-8b | base | train (n=36) | user-turn | chat | +0.012 [-0.011, +0.035] | -0.091 [-0.118, -0.064] | -7.59 [-91.20, 78.10] | no |
| qwen3-8b | instruct | train (n=60) | all | factual | +0.067 [+0.051, +0.083] | -0.155 [-0.169, -0.140] | -2.32 [-3.27, -1.71] | no |
| qwen3-8b | instruct | train (n=60) | all | chat | +0.053 [+0.035, +0.071] | -0.060 [-0.078, -0.042] | -1.13 [-2.15, -0.62] | no |
| qwen3-8b | instruct | train (n=24) | system | factual | +0.077 [+0.054, +0.101] | -0.127 [-0.147, -0.108] | -1.66 [-2.62, -1.09] | no |
| qwen3-8b | instruct | train (n=24) | system | chat | +0.075 [+0.047, +0.102] | -0.037 [-0.069, -0.004] | -0.49 [-1.40, -0.04] | no |
| qwen3-8b | instruct | train (n=36) | user-turn | factual | +0.060 [+0.039, +0.082] | -0.173 [-0.191, -0.155] | -2.88 [-4.92, -1.89] | no |
| qwen3-8b | instruct | train (n=36) | user-turn | chat | +0.038 [+0.016, +0.061] | -0.076 [-0.095, -0.057] | -1.97 [-5.64, -0.96] | no |

### Layout regex rate, training personas

| family | stage | personas | slots | probes | gap (assistant - intact) | numerator (neutral - intact) | fraction | reading |
|---|---|---|---|---|---|---|---|---|
| olmo3-7b | base | train (n=60) | all | factual | +0.026 [+0.022, +0.029] | +0.003 [-0.002, +0.007] | 0.10 [-0.09, 0.26] | no |
| olmo3-7b | base | train (n=60) | all | chat | +0.023 [+0.013, +0.034] | +0.019 [+0.007, +0.031] | 0.80 [0.34, 1.40] | no |
| olmo3-7b | base | train (n=24) | system | factual | +0.030 [+0.027, +0.031] | +0.004 [-0.001, +0.009] | 0.13 [-0.05, 0.30] | no |
| olmo3-7b | base | train (n=24) | system | chat | +0.036 [+0.023, +0.049] | +0.008 [-0.005, +0.021] | 0.21 [-0.21, 0.50] | no |
| olmo3-7b | base | train (n=36) | user-turn | factual | +0.023 [+0.017, +0.028] | +0.002 [-0.005, +0.009] | 0.08 [-0.26, 0.33] | no |
| olmo3-7b | base | train (n=36) | user-turn | chat | +0.015 [+0.000, +0.029] | +0.026 [+0.009, +0.043] | 1.76 [0.36, 7.20] | no |
| olmo3-7b | instruct | train (n=60) | all | factual | +0.516 [+0.429, +0.598] | +0.006 [-0.053, +0.065] | 0.01 [-0.12, 0.11] | no |
| olmo3-7b | instruct | train (n=60) | all | chat | -0.001 [-0.023, +0.019] | +0.035 [+0.013, +0.058] | -68.00 [-36.00, 38.50] | no |
| olmo3-7b | instruct | train (n=24) | system | factual | +0.297 [+0.165, +0.435] | +0.073 [-0.016, +0.168] | 0.25 [-0.09, 0.39] | no |
| olmo3-7b | instruct | train (n=24) | system | chat | -0.048 [-0.085, -0.010] | +0.111 [+0.077, +0.142] | -2.30 [-10.10, -0.97] | no |
| olmo3-7b | instruct | train (n=36) | user-turn | factual | +0.662 [+0.580, +0.740] | -0.039 [-0.114, +0.032] | -0.06 [-0.20, 0.04] | no |
| olmo3-7b | instruct | train (n=36) | user-turn | chat | +0.031 [+0.011, +0.048] | -0.015 [-0.034, +0.001] | -0.47 [-2.79, 0.02] | no |
| qwen3-8b | base | train (n=60) | all | factual | +0.049 [+0.036, +0.061] | +0.016 [+0.001, +0.031] | 0.33 [0.02, 0.59] | no |
| qwen3-8b | base | train (n=60) | all | chat | +0.278 [+0.261, +0.292] | +0.044 [+0.025, +0.061] | 0.16 [0.09, 0.21] | no |
| qwen3-8b | base | train (n=24) | system | factual | +0.043 [+0.021, +0.062] | +0.022 [+0.001, +0.043] | 0.52 [0.04, 1.08] | no |
| qwen3-8b | base | train (n=24) | system | chat | +0.279 [+0.247, +0.305] | +0.034 [-0.003, +0.068] | 0.12 [-0.01, 0.23] | no |
| qwen3-8b | base | train (n=36) | user-turn | factual | +0.053 [+0.037, +0.068] | +0.012 [-0.009, +0.033] | 0.23 [-0.20, 0.53] | no |
| qwen3-8b | base | train (n=36) | user-turn | chat | +0.277 [+0.260, +0.293] | +0.050 [+0.031, +0.069] | 0.18 [0.12, 0.24] | no |
| qwen3-8b | instruct | train (n=60) | all | factual | +0.809 [+0.743, +0.869] | -0.120 [-0.180, -0.062] | -0.15 [-0.24, -0.07] | no |
| qwen3-8b | instruct | train (n=60) | all | chat | +0.235 [+0.215, +0.253] | +0.037 [+0.018, +0.057] | 0.16 [0.08, 0.23] | no |
| qwen3-8b | instruct | train (n=24) | system | factual | +0.738 [+0.611, +0.850] | -0.108 [-0.225, +0.001] | -0.15 [-0.37, 0.00] | no |
| qwen3-8b | instruct | train (n=24) | system | chat | +0.238 [+0.211, +0.262] | +0.070 [+0.043, +0.096] | 0.30 [0.20, 0.38] | no |
| qwen3-8b | instruct | train (n=36) | user-turn | factual | +0.857 [+0.793, +0.915] | -0.128 [-0.191, -0.069] | -0.15 [-0.24, -0.08] | no |
| qwen3-8b | instruct | train (n=36) | user-turn | chat | +0.234 [+0.205, +0.258] | +0.016 [-0.010, +0.038] | 0.07 [-0.05, 0.15] | no |

### Category log-odds, held-out personas (generalisation, reported separately)

| family | stage | personas | slots | probes | gap (assistant - intact) | numerator (neutral - intact) | fraction | reading |
|---|---|---|---|---|---|---|---|---|
| olmo3-7b | base | holdout (n=18) | all | factual | +1.71 [+1.11, +2.34] | +2.08 [+1.50, +2.71] | 1.22 [1.04, 1.48] | fallback |
| olmo3-7b | base | holdout (n=18) | all | chat | +1.90 [+1.15, +2.68] | +2.04 [+1.34, +2.77] | 1.08 [0.86, 1.39] | fallback |
| olmo3-7b | base | holdout (n=6) | system | factual | +2.06 [+1.34, +2.74] | +1.59 [+0.87, +2.27] | 0.77 [0.65, 0.83] | fallback |
| olmo3-7b | base | holdout (n=6) | system | chat | +2.63 [+1.51, +3.69] | +1.48 [+0.36, +2.54] | 0.56 [0.24, 0.69] | fallback |
| olmo3-7b | base | holdout (n=12) | user-turn | factual | +1.53 [+0.74, +2.41] | +2.33 [+1.55, +3.18] | 1.52 [1.32, 2.10] | fallback |
| olmo3-7b | base | holdout (n=12) | user-turn | chat | +1.53 [+0.63, +2.52] | +2.32 [+1.45, +3.27] | 1.52 [1.29, 2.32] | fallback |
| olmo3-7b | instruct | holdout (n=18) | all | factual | +11.38 [+8.43, +14.34] | +9.75 [+6.97, +12.51] | 0.86 [0.82, 0.88] | fallback |
| olmo3-7b | instruct | holdout (n=18) | all | chat | +9.89 [+7.30, +12.48] | +8.46 [+6.17, +10.78] | 0.86 [0.80, 0.91] | fallback |
| olmo3-7b | instruct | holdout (n=6) | system | factual | +6.85 [+2.91, +10.99] | +5.99 [+2.04, +10.12] | 0.87 [0.70, 0.92] | fallback |
| olmo3-7b | instruct | holdout (n=6) | system | chat | +5.77 [+2.37, +9.35] | +6.02 [+2.62, +9.60] | 1.04 [1.03, 1.11] | fallback |
| olmo3-7b | instruct | holdout (n=12) | user-turn | factual | +13.64 [+10.28, +16.93] | +11.62 [+8.41, +14.72] | 0.85 [0.81, 0.88] | fallback |
| olmo3-7b | instruct | holdout (n=12) | user-turn | chat | +11.95 [+9.04, +14.78] | +9.68 [+6.85, +12.39] | 0.81 [0.75, 0.85] | fallback |
| qwen3-8b | base | holdout (n=18) | all | factual | +2.50 [+1.81, +3.19] | +2.62 [+1.88, +3.35] | 1.05 [0.97, 1.14] | fallback |
| qwen3-8b | base | holdout (n=18) | all | chat | +2.07 [+1.42, +2.74] | +2.13 [+1.46, +2.80] | 1.03 [0.94, 1.14] | fallback |
| qwen3-8b | base | holdout (n=6) | system | factual | +2.65 [+1.51, +3.75] | +2.67 [+1.53, +3.78] | 1.01 [1.01, 1.02] | fallback |
| qwen3-8b | base | holdout (n=6) | system | chat | +2.23 [+0.96, +3.42] | +1.95 [+0.68, +3.15] | 0.88 [0.71, 0.92] | fallback |
| qwen3-8b | base | holdout (n=12) | user-turn | factual | +2.42 [+1.56, +3.28] | +2.59 [+1.64, +3.52] | 1.07 [0.94, 1.21] | fallback |
| qwen3-8b | base | holdout (n=12) | user-turn | chat | +1.99 [+1.25, +2.75] | +2.21 [+1.45, +2.98] | 1.11 [0.99, 1.29] | fallback |
| qwen3-8b | instruct | holdout (n=18) | all | factual | +16.08 [+13.68, +18.22] | +14.96 [+12.51, +17.22] | 0.93 [0.90, 0.96] | fallback |
| qwen3-8b | instruct | holdout (n=18) | all | chat | +8.89 [+6.89, +10.75] | +9.88 [+7.98, +11.63] | 1.11 [1.07, 1.17] | fallback |
| qwen3-8b | instruct | holdout (n=6) | system | factual | +15.19 [+10.52, +18.87] | +13.47 [+8.81, +17.15] | 0.89 [0.84, 0.91] | fallback |
| qwen3-8b | instruct | holdout (n=6) | system | chat | +8.80 [+4.83, +12.11] | +9.49 [+5.52, +12.81] | 1.08 [1.06, 1.14] | fallback |
| qwen3-8b | instruct | holdout (n=12) | user-turn | factual | +16.53 [+13.80, +18.98] | +15.71 [+12.95, +18.34] | 0.95 [0.92, 0.98] | fallback |
| qwen3-8b | instruct | holdout (n=12) | user-turn | chat | +8.93 [+6.67, +11.07] | +10.07 [+8.02, +11.99] | 1.13 [1.07, 1.22] | fallback |

## Instruct minus base fraction (training personas, all slots)

| family | probes | readout | fraction base | fraction instruct | difference [95% CI] | >= 0.3 |
|---|---|---|---|---|---|---|
| olmo3-7b | factual | logodds | 1.16 | 0.85 | -0.30 [-0.42, -0.21] | no |
| olmo3-7b | factual | p_asst | 1.95 | -0.46 | -2.40 [-8.81, +7.19] | no |
| olmo3-7b | chat | logodds | 0.98 | 0.87 | -0.12 [-0.26, +0.02] | no |
| olmo3-7b | chat | p_asst | 4.52 | -9.88 | -14.41 [-115.00, +105.10] | no |
| qwen3-8b | factual | logodds | 1.04 | 0.91 | -0.13 [-0.17, -0.10] | no |
| qwen3-8b | factual | p_asst | 0.89 | -2.32 | -3.21 [-4.34, -2.40] | no |
| qwen3-8b | chat | logodds | 1.00 | 1.11 | +0.11 [+0.06, +0.16] | no |
| qwen3-8b | chat | p_asst | -2.74 | -1.13 | +1.60 [+0.12, +6.74] | yes |


## Additions made after seeing the OLMo results (analysis only; no new scoring, frozen materials and logged readings unchanged)

### A1. The swap validation cannot fail, so it is replaced by a real check

The validation above (r = +1.00) is true by construction: a swap prompt uses the partner's text in the same slot and wording, so it is the *same prompt* as the partner's intact prompt. Confirmed numerically (max |swap - partner intact| over all category log-probs, training and held-out personas):

| family | stage | max difference (nats) |
|---|---|---|
| olmo3-7b | base | 0.0000 |
| olmo3-7b | instruct | 0.0000 |
| qwen3-8b | base | 0.0000 |
| qwen3-8b | instruct | 0.0000 |

The real checks: (i) do personas differ from one another more than a persona's own probes do (share of the variance of an intact-prompt readout that lies between personas, bootstrap over the 60 training personas); (ii) does swapping change a persona's readout (mean absolute swap - intact shift of `logodds`, bootstrap over personas, and the share of personas above 0.5 nats).

| family | stage | between-persona share, logodds [95% CI] | voice | assistant | list | mean abs swap shift (nats) [95% CI] | personas with shift > 0.5 |
|---|---|---|---|---|---|---|---|
| olmo3-7b | base | 0.74 [0.67, 0.79] | 0.67 [0.58, 0.73] | 0.40 [0.32, 0.47] | 0.30 [0.24, 0.35] | 1.62 [1.36, 1.90] | 88% |
| olmo3-7b | instruct | 0.93 [0.91, 0.95] | 0.86 [0.83, 0.89] | 0.46 [0.32, 0.57] | 0.57 [0.48, 0.64] | 6.57 [5.46, 7.63] | 98% |
| qwen3-8b | base | 0.77 [0.69, 0.82] | 0.52 [0.42, 0.60] | 0.30 [0.22, 0.37] | 0.21 [0.15, 0.26] | 1.84 [1.62, 2.08] | 97% |
| qwen3-8b | instruct | 0.84 [0.79, 0.88] | 0.73 [0.66, 0.77] | 0.53 [0.42, 0.61] | 0.56 [0.46, 0.63] | 6.00 [5.00, 6.99] | 100% |

A between-persona share well above the 1/16 expected for no persona effect, with a swap shift well above probe-to-probe noise, means the persona text is used. Note the check is about the persona text moving the readout, not about the direction of the move.

### A2. Which categories move: intact, neutral and plain Assistant (nats, summed token log-probability, mean over 6 continuations; training personas)

Does the neutral shift come from Assistant-style openers rising, or from character-voice openers falling? The change columns are neutral minus intact.

| family | stage | probes | category | intact | neutral | plain Assistant | neutral - intact [95% CI] |
|---|---|---|---|---|---|---|---|
| olmo3-7b | base | factual | assistant | -14.50 | -14.40 | -14.21 | +0.10 [-0.06, +0.27] |
| olmo3-7b | base | factual | list | -25.36 | -25.06 | -26.04 | +0.30 [+0.16, +0.45] |
| olmo3-7b | base | factual | voice | -14.44 | -16.49 | -16.58 | -2.05 [-2.33, -1.77] |
| olmo3-7b | base | factual | answer | -19.11 | -18.89 | -18.87 | +0.23 [+0.09, +0.34] |
| olmo3-7b | base | factual | logodds | -5.49 | -3.24 | -3.55 | +2.25 [+1.91, +2.59] |
| olmo3-7b | base | chat | assistant | -14.02 | -14.00 | -13.94 | +0.01 [-0.14, +0.16] |
| olmo3-7b | base | chat | list | -27.44 | -27.20 | -27.88 | +0.25 [+0.06, +0.41] |
| olmo3-7b | base | chat | voice | -14.09 | -16.15 | -16.50 | -2.07 [-2.33, -1.79] |
| olmo3-7b | base | chat | answer | -21.09 | -20.41 | -20.76 | +0.68 [+0.49, +0.87] |
| olmo3-7b | base | chat | logodds | -6.64 | -4.45 | -4.41 | +2.20 [+1.81, +2.57] |
| olmo3-7b | instruct | factual | assistant | -28.53 | -28.72 | -30.38 | -0.19 [-1.18, +0.82] |
| olmo3-7b | instruct | factual | list | -45.81 | -43.75 | -41.51 | +2.06 [+1.27, +2.81] |
| olmo3-7b | instruct | factual | voice | -20.21 | -29.12 | -30.55 | -8.91 [-10.50, -7.20] |
| olmo3-7b | instruct | factual | answer | -35.42 | -34.59 | -35.02 | +0.83 [-0.03, +1.69] |
| olmo3-7b | instruct | factual | logodds | -16.96 | -7.11 | -5.40 | +9.85 [+8.09, +11.47] |
| olmo3-7b | instruct | chat | assistant | -27.57 | -26.28 | -26.18 | +1.28 [+0.39, +2.26] |
| olmo3-7b | instruct | chat | list | -48.63 | -49.89 | -49.00 | -1.26 [-1.82, -0.74] |
| olmo3-7b | instruct | chat | voice | -20.86 | -29.45 | -30.26 | -8.59 [-9.84, -7.28] |
| olmo3-7b | instruct | chat | answer | -38.67 | -40.98 | -39.99 | -2.31 [-3.03, -1.59] |
| olmo3-7b | instruct | chat | logodds | -17.24 | -8.64 | -7.33 | +8.60 [+7.11, +9.96] |
| qwen3-8b | base | factual | assistant | -16.43 | -15.92 | -15.49 | +0.51 [+0.29, +0.72] |
| qwen3-8b | base | factual | list | -24.57 | -23.69 | -24.28 | +0.88 [+0.72, +1.05] |
| qwen3-8b | base | factual | voice | -16.64 | -18.57 | -18.53 | -1.93 [-2.35, -1.48] |
| qwen3-8b | base | factual | answer | -20.65 | -19.79 | -20.10 | +0.86 [+0.70, +1.03] |
| qwen3-8b | base | factual | logodds | -3.86 | -1.24 | -1.35 | +2.62 [+2.16, +3.07] |
| qwen3-8b | base | chat | assistant | -16.12 | -15.82 | -15.92 | +0.30 [+0.04, +0.57] |
| qwen3-8b | base | chat | list | -25.79 | -25.53 | -25.50 | +0.26 [+0.05, +0.46] |
| qwen3-8b | base | chat | voice | -16.52 | -18.43 | -18.46 | -1.92 [-2.25, -1.56] |
| qwen3-8b | base | chat | answer | -21.23 | -20.94 | -20.98 | +0.29 [+0.08, +0.50] |
| qwen3-8b | base | chat | logodds | -4.44 | -2.24 | -2.25 | +2.19 [+1.75, +2.61] |
| qwen3-8b | instruct | factual | assistant | -34.86 | -44.56 | -38.55 | -9.70 [-11.23, -8.15] |
| qwen3-8b | instruct | factual | list | -48.02 | -52.65 | -48.52 | -4.63 [-5.70, -3.56] |
| qwen3-8b | instruct | factual | voice | -23.45 | -45.08 | -41.40 | -21.63 [-23.78, -19.39] |
| qwen3-8b | instruct | factual | answer | -42.42 | -50.14 | -44.43 | -7.72 [-9.04, -6.34] |
| qwen3-8b | instruct | factual | logodds | -17.98 | -3.52 | -2.14 | +14.47 [+13.04, +15.85] |
| qwen3-8b | instruct | chat | assistant | -34.26 | -41.96 | -33.40 | -7.69 [-8.68, -6.68] |
| qwen3-8b | instruct | chat | list | -48.66 | -60.32 | -52.08 | -11.66 [-12.70, -10.62] |
| qwen3-8b | instruct | chat | voice | -22.95 | -42.34 | -32.96 | -19.38 [-20.68, -18.07] |
| qwen3-8b | instruct | chat | answer | -43.01 | -51.26 | -42.69 | -8.25 [-9.44, -7.06] |
| qwen3-8b | instruct | chat | logodds | -18.51 | -8.80 | -9.78 | +9.71 [+8.56, +10.80] |

### A3. Combined fallback reading under the spec wording (gap >= 0.5 nats AND >= 0.10 classifier points)

The logged plan says 'or' per readout (kept in the tables above); the spec text requires both. Combined: `logodds` gap >= 0.5 nats and `P(assistant)` gap >= 0.10, both fractions >= 50%, both numerators' CI excluding 0 (training personas, all slots).

| family | stage | probes | logodds gap / fraction | P(assistant) gap / fraction | combined |
|---|---|---|---|---|---|
| olmo3-7b | base | factual | +1.95 / 1.16 | +0.006 / 1.95 | no |
| olmo3-7b | base | chat | +2.24 / 0.98 | -0.012 / 4.52 | no |
| olmo3-7b | instruct | factual | +11.56 / 0.85 | +0.121 / -0.46 | no |
| olmo3-7b | instruct | chat | +9.91 / 0.87 | +0.010 / -9.88 | no |
| qwen3-8b | base | factual | +2.51 / 1.04 | -0.037 / 0.89 | no |
| qwen3-8b | base | chat | +2.19 / 1.00 | +0.030 / -2.74 | no |
| qwen3-8b | instruct | factual | +15.85 / 0.91 | +0.067 / -2.32 | no |
| qwen3-8b | instruct | chat | +8.73 / 1.11 | +0.053 / -1.13 | no |

Full pre-set reading (combined fallback at instruct AND the instruct-minus-base fraction gain >= 0.3 with CI above 0, on both readouts):

- olmo3-7b, factual: combined fallback at instruct no; fraction gain no (log-odds no, P(assistant) no) -> **not met**
- olmo3-7b, chat: combined fallback at instruct no; fraction gain no (log-odds no, P(assistant) no) -> **not met**
- qwen3-8b, factual: combined fallback at instruct no; fraction gain no (log-odds no, P(assistant) no) -> **not met**
- qwen3-8b, chat: combined fallback at instruct no; fraction gain no (log-odds no, P(assistant) yes) -> **not met**

### Scoring precision

Scoring ran the prompt at batch 1 and shared its KV cache across the 24 continuations (bf16). Against full batch-1 passes (first 3 prompts, 4 continuations each) the largest difference was olmo3-7b base 0.11, olmo3-7b instruct 0.31, qwen3-8b base 0.37, qwen3-8b instruct 0.56 nats. These are differences in a summed log-probability of 2-8 tokens; the contrasts used here are differences of the same continuations across prompt versions, and the readings sit at 0.5 nats or more, so the bar is not decided by this noise, but numbers near it should be read with that error in mind.

