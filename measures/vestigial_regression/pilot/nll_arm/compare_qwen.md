# qwen: change from own base (IT - base), paired over texts

Tokenizer/template: Qwen/Qwen3-8B for both checkpoints. Mean NLL over content positions 1+ (nats/token); 95% paired bootstrap CIs over texts. Rule applies to the first row of each user set.

## Raw per-slot mean NLL (nats/token, positions 1+)

| text set | model | @user | @asst | @plain |
|---|---|---|---|---|
| wildchat user | base | 2.762 | 2.730 | 3.802 |
| wildchat user | IT | 3.369 | 3.236 | 3.974 |
| oasst1 user | base | 2.628 | 2.567 | 3.814 |
| oasst1 user | IT | 2.985 | 2.917 | 3.787 |
| wildchat assistant | base | 1.935 | 1.889 | 2.778 |
| wildchat assistant | IT | 2.359 | 2.292 | 2.884 |

## wildchat user

| quantity | base | IT | change (IT - base), nats | change, bits/byte |
|---|---|---|---|---|
| role-specific (user@user - user@plain) | -1.040 | -0.604 | +0.436 [+0.263, +0.615] | +0.144 [+0.091, +0.197] |
| slot effect (user@user - user@asst) | +0.032 | +0.133 | +0.101 [+0.039, +0.159] | +0.034 [+0.015, +0.051] |

## oasst1 user

| quantity | base | IT | change (IT - base), nats | change, bits/byte |
|---|---|---|---|---|
| role-specific (user@user - user@plain) | -1.186 | -0.801 | +0.385 [+0.252, +0.520] | +0.125 [+0.084, +0.167] |
| slot effect (user@user - user@asst) | +0.061 | +0.068 | +0.006 [-0.024, +0.036] | +0.003 [-0.006, +0.013] |

## wildchat assistant

| quantity | base | IT | change (IT - base), nats | change, bits/byte |
|---|---|---|---|---|
| role-specific (asst@asst - asst@plain) | -0.889 | -0.592 | +0.297 [+0.192, +0.412] | +0.088 [+0.055, +0.124] |
| slot effect (asst@user - asst@asst) | +0.045 | +0.067 | +0.021 [-0.019, +0.056] | +0.007 [-0.005, +0.018] |

## Pre-committed rule (criterion 1 against the family's own base)

user@user - user@plain rises by >= 0.05 nats with the CI excluding zero, in both user sources: wildchat: yes (+0.436 [+0.263, +0.615]); oasst1: yes (+0.385 [+0.252, +0.520]). **Rule MET.**

Assistant-text role-specific change (descriptive): +0.297 [+0.192, +0.412] against a mean user-text change of +0.410.

## Position analysis (role-specific term by content position, texts with >= 40 tokens)

### wildchat user (25 texts)

| positions | base: own - plain | IT: own - plain | IT - base |
|---|---|---|---|
| 0 | -0.49 | +6.48 | +6.98 |
| 1-4 | -3.18 | -3.98 | -0.80 |
| 5-19 | -1.45 | -1.06 | +0.39 |
| 20-39 | -1.15 | -0.67 | +0.48 |

### oasst1 user (9 texts)

| positions | base: own - plain | IT: own - plain | IT - base |
|---|---|---|---|
| 0 | +0.16 | +3.15 | +2.99 |
| 1-4 | -3.41 | -2.41 | +1.00 |
| 5-19 | -1.11 | -0.64 | +0.48 |
| 20-39 | -0.67 | -0.41 | +0.26 |

### wildchat assistant (27 texts)

| positions | base: own - plain | IT: own - plain | IT - base |
|---|---|---|---|
| 0 | -0.98 | +10.43 | +11.42 |
| 1-4 | -3.01 | -3.09 | -0.07 |
| 5-19 | -1.19 | -0.84 | +0.35 |
| 20-39 | -0.93 | -0.70 | +0.23 |

