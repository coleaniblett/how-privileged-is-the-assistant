# gemma: change from own base (IT - base), paired over texts

Tokenizer/template: google/gemma-4-31B-it for both checkpoints. Mean NLL over content positions 1+ (nats/token); 95% paired bootstrap CIs over texts. Rule applies to the first row of each user set.

## Raw per-slot mean NLL (nats/token, positions 1+)

| text set | model | @user | @asst | @plain |
|---|---|---|---|---|
| wildchat user | base | 2.691 | 2.745 | 2.683 |
| wildchat user | IT | 10.202 | 4.518 | 7.251 |
| oasst1 user | base | 2.604 | 2.727 | 2.599 |
| oasst1 user | IT | 11.034 | 4.565 | 7.399 |
| wildchat assistant | base | 1.782 | 1.820 | 1.783 |
| wildchat assistant | IT | 7.639 | 3.171 | 5.305 |

## wildchat user

| quantity | base | IT | change (IT - base), nats | change, bits/byte |
|---|---|---|---|---|
| role-specific (user@user - user@plain) | +0.008 | +2.951 | +2.943 [+2.404, +3.443] | +1.096 [+0.844, +1.377] |
| slot effect (user@user - user@asst) | -0.055 | +5.684 | +5.739 [+5.231, +6.242] | +2.071 [+1.797, +2.436] |

## oasst1 user

| quantity | base | IT | change (IT - base), nats | change, bits/byte |
|---|---|---|---|---|
| role-specific (user@user - user@plain) | +0.005 | +3.635 | +3.630 [+2.911, +4.317] | +1.164 [+0.946, +1.374] |
| slot effect (user@user - user@asst) | -0.122 | +6.469 | +6.591 [+6.128, +7.052] | +2.154 [+1.966, +2.352] |

## wildchat assistant

| quantity | base | IT | change (IT - base), nats | change, bits/byte |
|---|---|---|---|---|
| role-specific (asst@asst - asst@plain) | +0.037 | -2.135 | -2.172 [-2.603, -1.733] | -0.817 [-1.172, -0.565] |
| slot effect (asst@user - asst@asst) | -0.038 | +4.469 | +4.507 [+4.195, +4.827] | +1.549 [+1.333, +1.884] |

## Pre-committed rule (criterion 1 against the family's own base)

user@user - user@plain rises by >= 0.05 nats with the CI excluding zero, in both user sources: wildchat: yes (+2.943 [+2.404, +3.443]); oasst1: yes (+3.630 [+2.911, +4.317]). **Rule MET.**

Assistant-text role-specific change (descriptive): -2.172 [-2.603, -1.733] against a mean user-text change of +3.286.

## Position analysis (role-specific term by content position, texts with >= 40 tokens)

### wildchat user (25 texts)

| positions | base: own - plain | IT: own - plain | IT - base |
|---|---|---|---|
| 0 | +0.79 | +1.79 | +0.99 |
| 1-4 | +0.20 | +7.08 | +6.88 |
| 5-19 | +0.06 | +4.66 | +4.60 |
| 20-39 | -0.04 | +4.15 | +4.19 |

### oasst1 user (9 texts)

| positions | base: own - plain | IT: own - plain | IT - base |
|---|---|---|---|
| 0 | +1.37 | +1.91 | +0.54 |
| 1-4 | -0.05 | +9.97 | +10.01 |
| 5-19 | +0.01 | +3.40 | +3.39 |
| 20-39 | -0.02 | +3.10 | +3.12 |

### wildchat assistant (27 texts)

| positions | base: own - plain | IT: own - plain | IT - base |
|---|---|---|---|
| 0 | +3.21 | +9.91 | +6.71 |
| 1-4 | +1.13 | -3.48 | -4.61 |
| 5-19 | +0.07 | -1.91 | -1.98 |
| 20-39 | +0.01 | -1.99 | -2.00 |

