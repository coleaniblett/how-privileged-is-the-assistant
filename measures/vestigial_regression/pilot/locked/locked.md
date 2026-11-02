# Locked design: NLL arm and judge arm

Positions 1+ for the NLL arm; 5000 bootstrap resamples (seed 0); user and assistant text sets resampled independently, pairing within each set across checkpoints. Every change is from the family's own base. Nats per token are compared within a family only; bits per byte across families.

## olmo3-7b: sft − base

| text set | quantity | nats/token | bits/byte |
|---|---|---|---|
| wildchat user | Δ(user@user) | -0.753 [-1.153, -0.360] | -0.227 [-0.358, -0.094] |
| wildchat user | Δ(user@asst) | -0.779 [-1.172, -0.400] | -0.237 [-0.364, -0.114] |
| wildchat user | Δ(user@plain) [non-assistant degradation] | -0.501 [-0.869, -0.167] | -0.152 [-0.270, -0.038] |
| wildchat user | role-specific Δ(user@user − user@plain) | -0.252 [-0.385, -0.111] | -0.075 [-0.117, -0.031] |
| oasst1 user | Δ(user@user) | +0.211 [+0.137, +0.285] | +0.066 [+0.044, +0.089] |
| oasst1 user | Δ(user@asst) | +0.153 [+0.071, +0.233] | +0.048 [+0.023, +0.072] |
| oasst1 user | Δ(user@plain) [non-assistant degradation] | +0.234 [+0.173, +0.298] | +0.075 [+0.056, +0.095] |
| oasst1 user | role-specific Δ(user@user − user@plain) | -0.023 [-0.090, +0.054] | -0.009 [-0.030, +0.014] |
| wildchat assistant | Δ(asst@asst) | +0.092 [+0.024, +0.148] | +0.029 [+0.007, +0.045] |
| wildchat assistant | Δ(asst@user) | +0.179 [+0.101, +0.250] | +0.058 [+0.034, +0.079] |
| wildchat assistant | Δ(asst@plain) [non-assistant degradation] | +0.216 [+0.146, +0.278] | +0.067 [+0.047, +0.085] |
| wildchat assistant | role-specific Δ(asst@asst − asst@plain) | -0.124 [-0.159, -0.088] | -0.039 [-0.050, -0.028] |
| wildchat user vs wildchat assistant | asymmetry = Δrs(user) − Δrs(assistant) | -0.128 [-0.276, +0.015] | -0.036 [-0.080, +0.011] |
| oasst1 user vs wildchat assistant | asymmetry = Δrs(user) − Δrs(assistant) | +0.101 [+0.024, +0.182] | +0.030 [+0.005, +0.055] |

NLL arm verdict (asymmetry ≥ 0.05 nats, CI excluding zero, same direction in both user sources): wildchat: -0.128 [-0.276, +0.015]; oasst1: +0.101 [+0.024, +0.182] → **does not fire**.

Judge arm: no judge outputs for this point yet (`judge.py run`).

## olmo3-7b: dpo − base

| text set | quantity | nats/token | bits/byte |
|---|---|---|---|
| wildchat user | Δ(user@user) | -0.614 [-1.043, -0.190] | -0.178 [-0.322, -0.035] |
| wildchat user | Δ(user@asst) | -0.617 [-1.031, -0.198] | -0.181 [-0.321, -0.046] |
| wildchat user | Δ(user@plain) [non-assistant degradation] | -0.397 [-0.784, -0.031] | -0.114 [-0.243, +0.009] |
| wildchat user | role-specific Δ(user@user − user@plain) | -0.217 [-0.358, -0.074] | -0.064 [-0.104, -0.020] |
| oasst1 user | Δ(user@user) | +0.284 [+0.191, +0.381] | +0.090 [+0.061, +0.118] |
| oasst1 user | Δ(user@asst) | +0.264 [+0.168, +0.362] | +0.084 [+0.054, +0.114] |
| oasst1 user | Δ(user@plain) [non-assistant degradation] | +0.327 [+0.248, +0.410] | +0.106 [+0.080, +0.132] |
| oasst1 user | role-specific Δ(user@user − user@plain) | -0.043 [-0.119, +0.038] | -0.016 [-0.039, +0.011] |
| wildchat assistant | Δ(asst@asst) | +0.433 [+0.327, +0.533] | +0.136 [+0.105, +0.164] |
| wildchat assistant | Δ(asst@user) | +0.402 [+0.308, +0.486] | +0.130 [+0.101, +0.156] |
| wildchat assistant | Δ(asst@plain) [non-assistant degradation] | +0.404 [+0.332, +0.469] | +0.127 [+0.105, +0.148] |
| wildchat assistant | role-specific Δ(asst@asst − asst@plain) | +0.030 [-0.028, +0.092] | +0.008 [-0.009, +0.028] |
| wildchat user vs wildchat assistant | asymmetry = Δrs(user) − Δrs(assistant) | -0.246 [-0.395, -0.096] | -0.072 [-0.119, -0.024] |
| oasst1 user vs wildchat assistant | asymmetry = Δrs(user) − Δrs(assistant) | -0.073 [-0.172, +0.026] | -0.024 [-0.055, +0.008] |

NLL arm verdict (asymmetry ≥ 0.05 nats, CI excluding zero, same direction in both user sources): wildchat: -0.246 [-0.395, -0.096]; oasst1: -0.073 [-0.172, +0.026] → **does not fire**.

Judge arm: no judge outputs for this point yet (`judge.py run`).

## olmo3-7b: instruct − base

| text set | quantity | nats/token | bits/byte |
|---|---|---|---|
| wildchat user | Δ(user@user) | -0.588 [-1.016, -0.156] | -0.169 [-0.321, -0.027] |
| wildchat user | Δ(user@asst) | -0.586 [-1.011, -0.175] | -0.171 [-0.315, -0.030] |
| wildchat user | Δ(user@plain) [non-assistant degradation] | -0.345 [-0.755, +0.047] | -0.096 [-0.232, +0.034] |
| wildchat user | role-specific Δ(user@user − user@plain) | -0.243 [-0.379, -0.101] | -0.072 [-0.113, -0.029] |
| oasst1 user | Δ(user@user) | +0.306 [+0.209, +0.408] | +0.097 [+0.067, +0.128] |
| oasst1 user | Δ(user@asst) | +0.302 [+0.204, +0.399] | +0.096 [+0.065, +0.126] |
| oasst1 user | Δ(user@plain) [non-assistant degradation] | +0.388 [+0.305, +0.475] | +0.125 [+0.099, +0.153] |
| oasst1 user | role-specific Δ(user@user − user@plain) | -0.082 [-0.160, -0.001] | -0.028 [-0.053, -0.002] |
| wildchat assistant | Δ(asst@asst) | +0.492 [+0.378, +0.599] | +0.154 [+0.122, +0.186] |
| wildchat assistant | Δ(asst@user) | +0.445 [+0.352, +0.526] | +0.144 [+0.115, +0.170] |
| wildchat assistant | Δ(asst@plain) [non-assistant degradation] | +0.470 [+0.395, +0.539] | +0.149 [+0.125, +0.170] |
| wildchat assistant | role-specific Δ(asst@asst − asst@plain) | +0.022 [-0.041, +0.094] | +0.005 [-0.015, +0.028] |
| wildchat user vs wildchat assistant | asymmetry = Δrs(user) − Δrs(assistant) | -0.265 [-0.426, -0.107] | -0.077 [-0.124, -0.029] |
| oasst1 user vs wildchat assistant | asymmetry = Δrs(user) − Δrs(assistant) | -0.104 [-0.206, +0.007] | -0.034 [-0.066, -0.001] |

NLL arm verdict (asymmetry ≥ 0.05 nats, CI excluding zero, same direction in both user sources): wildchat: -0.265 [-0.426, -0.107]; oasst1: -0.104 [-0.206, +0.007] → **does not fire**.

Judge arm: no judge outputs for this point yet (`judge.py run`).

## qwen3-8b: instruct − base

| text set | quantity | nats/token | bits/byte |
|---|---|---|---|
| wildchat user | Δ(user@user) | +0.608 [+0.481, +0.738] | +0.200 [+0.163, +0.237] |
| wildchat user | Δ(user@asst) | +0.507 [+0.407, +0.613] | +0.166 [+0.137, +0.195] |
| wildchat user | Δ(user@plain) [non-assistant degradation] | +0.172 [+0.037, +0.308] | +0.056 [+0.014, +0.097] |
| wildchat user | role-specific Δ(user@user − user@plain) | +0.436 [+0.263, +0.617] | +0.144 [+0.089, +0.198] |
| oasst1 user | Δ(user@user) | +0.357 [+0.263, +0.453] | +0.116 [+0.086, +0.145] |
| oasst1 user | Δ(user@asst) | +0.350 [+0.272, +0.439] | +0.113 [+0.089, +0.138] |
| oasst1 user | Δ(user@plain) [non-assistant degradation] | -0.028 [-0.144, +0.086] | -0.009 [-0.045, +0.027] |
| oasst1 user | role-specific Δ(user@user − user@plain) | +0.385 [+0.249, +0.514] | +0.125 [+0.083, +0.166] |
| wildchat assistant | Δ(asst@asst) | +0.403 [+0.347, +0.464] | +0.128 [+0.110, +0.147] |
| wildchat assistant | Δ(asst@user) | +0.424 [+0.375, +0.472] | +0.135 [+0.120, +0.150] |
| wildchat assistant | Δ(asst@plain) [non-assistant degradation] | +0.106 [+0.027, +0.181] | +0.040 [+0.014, +0.065] |
| wildchat assistant | role-specific Δ(asst@asst − asst@plain) | +0.297 [+0.191, +0.408] | +0.088 [+0.055, +0.124] |
| wildchat user vs wildchat assistant | asymmetry = Δrs(user) − Δrs(assistant) | +0.139 [-0.072, +0.347] | +0.056 [-0.008, +0.119] |
| oasst1 user vs wildchat assistant | asymmetry = Δrs(user) − Δrs(assistant) | +0.088 [-0.084, +0.256] | +0.037 [-0.017, +0.090] |

NLL arm verdict (asymmetry ≥ 0.05 nats, CI excluding zero, same direction in both user sources): wildchat: +0.139 [-0.072, +0.347]; oasst1: +0.088 [-0.084, +0.256] → **does not fire**.

Judge arm: no judge outputs for this point yet (`judge.py run`).

## gemma4-31b: instruct − base

| text set | quantity | nats/token | bits/byte |
|---|---|---|---|
| wildchat user | Δ(user@user) | +7.511 [+6.955, +8.113] | +2.672 [+2.387, +3.019] |
| wildchat user | Δ(user@asst) | +1.773 [+1.548, +2.002] | +0.600 [+0.534, +0.672] |
| wildchat user | Δ(user@plain) [non-assistant degradation] | +4.569 [+4.018, +5.219] | +1.576 [+1.418, +1.746] |
| wildchat user | role-specific Δ(user@user − user@plain) | +2.943 [+2.429, +3.452] | +1.096 [+0.848, +1.370] |
| oasst1 user | Δ(user@user) | +8.429 [+7.862, +8.974] | +2.740 [+2.539, +2.939] |
| oasst1 user | Δ(user@asst) | +1.839 [+1.583, +2.097] | +0.586 [+0.514, +0.653] |
| oasst1 user | Δ(user@plain) [non-assistant degradation] | +4.799 [+4.085, +5.556] | +1.576 [+1.309, +1.858] |
| oasst1 user | role-specific Δ(user@user − user@plain) | +3.630 [+2.924, +4.285] | +1.164 [+0.938, +1.384] |
| wildchat assistant | Δ(asst@asst) | +1.351 [+1.100, +1.632] | +0.431 [+0.357, +0.518] |
| wildchat assistant | Δ(asst@user) | +5.857 [+5.499, +6.209] | +1.980 [+1.755, +2.302] |
| wildchat assistant | Δ(asst@plain) [non-assistant degradation] | +3.522 [+3.070, +3.981] | +1.248 [+0.990, +1.590] |
| wildchat assistant | role-specific Δ(asst@asst − asst@plain) | -2.172 [-2.608, -1.739] | -0.817 [-1.167, -0.567] |
| wildchat user vs wildchat assistant | asymmetry = Δrs(user) − Δrs(assistant) | +5.114 [+4.435, +5.811] | +1.913 [+1.549, +2.327] |
| oasst1 user vs wildchat assistant | asymmetry = Δrs(user) − Δrs(assistant) | +5.802 [+4.982, +6.628] | +1.981 [+1.625, +2.387] |

NLL arm verdict (asymmetry ≥ 0.05 nats, CI excluding zero, same direction in both user sources): wildchat: +5.114 [+4.435, +5.811]; oasst1: +5.802 [+4.982, +6.628] → **fires**.

Judge arm: no judge outputs for this point yet (`judge.py run`).

## Cross-family summary: bits per byte, change from each family's own base

| model | wildchat Δ(user@user) | wildchat Δ(user@plain) | wildchat user role-specific | wildchat asymmetry | oasst1 Δ(user@user) | oasst1 Δ(user@plain) | oasst1 user role-specific | oasst1 asymmetry | assistant Δ(asst@asst) | assistant Δ(asst@plain) | assistant role-specific |
|---|---|---|---|---|---|---|---|---|---|---|---|
| olmo3-7b sft | -0.23 [-0.36, -0.09] | -0.15 [-0.27, -0.04] | -0.07 [-0.12, -0.03] | -0.04 [-0.08, +0.01] | +0.07 [+0.04, +0.09] | +0.08 [+0.06, +0.09] | -0.01 [-0.03, +0.01] | +0.03 [+0.01, +0.05] | +0.03 [+0.01, +0.04] | +0.07 [+0.05, +0.09] | -0.04 [-0.05, -0.03] |
| olmo3-7b dpo | -0.18 [-0.32, -0.04] | -0.11 [-0.24, +0.01] | -0.06 [-0.10, -0.02] | -0.07 [-0.12, -0.02] | +0.09 [+0.06, +0.12] | +0.11 [+0.08, +0.13] | -0.02 [-0.04, +0.01] | -0.02 [-0.05, +0.01] | +0.14 [+0.11, +0.16] | +0.13 [+0.11, +0.15] | +0.01 [-0.01, +0.03] |
| olmo3-7b instruct | -0.17 [-0.32, -0.03] | -0.10 [-0.23, +0.03] | -0.07 [-0.11, -0.03] | -0.08 [-0.12, -0.03] | +0.10 [+0.07, +0.13] | +0.13 [+0.10, +0.15] | -0.03 [-0.05, -0.00] | -0.03 [-0.07, -0.00] | +0.15 [+0.12, +0.19] | +0.15 [+0.13, +0.17] | +0.01 [-0.02, +0.03] |
| qwen3-8b instruct | +0.20 [+0.16, +0.24] | +0.06 [+0.01, +0.10] | +0.14 [+0.09, +0.20] | +0.06 [-0.01, +0.12] | +0.12 [+0.09, +0.15] | -0.01 [-0.04, +0.03] | +0.13 [+0.08, +0.17] | +0.04 [-0.02, +0.09] | +0.13 [+0.11, +0.15] | +0.04 [+0.01, +0.07] | +0.09 [+0.05, +0.12] |
| gemma4-31b instruct | +2.67 [+2.39, +3.02] | +1.58 [+1.42, +1.75] | +1.10 [+0.85, +1.37] | +1.91 [+1.55, +2.33] | +2.74 [+2.54, +2.94] | +1.58 [+1.31, +1.86] | +1.16 [+0.94, +1.38] | +1.98 [+1.63, +2.39] | +0.43 [+0.36, +0.52] | +1.25 [+0.99, +1.59] | -0.82 [-1.17, -0.57] |

Each family's change is from its own base; only bits per byte is compared across families.

## How the arms combine

- olmo3-7b sft: NLL arm does not fire; judge arm not run
- olmo3-7b dpo: NLL arm does not fire; judge arm not run
- olmo3-7b instruct: NLL arm does not fire; judge arm not run
- qwen3-8b instruct: NLL arm does not fire; judge arm not run
- gemma4-31b instruct: NLL arm fires; judge arm not run

## Agreement of the arms across post-trained points

Spearman correlation between the NLL asymmetry (bits per byte, mean of the two user sources) and the judge's non-user-voice rate change. Base points are zero by construction for both and are excluded. With this few points it is descriptive, not a significance test. Points: 0; not computed (fewer than 3 points with judge outputs, or no variation).
