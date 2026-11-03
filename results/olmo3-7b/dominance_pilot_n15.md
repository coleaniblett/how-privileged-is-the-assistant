# Dominance-matrix pilot (harm aversion, OLMo 3 7B)

## Fail-fast checks (instruct stage; effects are instruct - base, nats)

| # | check | result | numbers |
|---|---|---|---|
| 1 | Denominator usable: reporter 1 coverage >= 0.8 and instruct - base effect > 0 with CI excluding 0 | PASS | coverage 1.000; effect +0.499 [+0.315, +0.691] |
| 2 | Ordering: reporter 2 effect < reporter 1 effect | FAIL | reporter 2 +0.526 [+0.371, +0.686] vs reporter 1 +0.499 [+0.315, +0.691] |
| 3 | Precision: reporter 2 cell-ratio CI narrower than 1.0 | PASS | ratio +1.055 [+0.829, +1.380], width 0.55 (denominator <= 0 in 0.0% of resamples) |
| 4 | Interpretable fade: neither reporter 3 nor 4 exceeds reporter 2 | PASS | reporter 3 +0.344, reporter 4 -0.433, reporter 2 +0.526 |

Format `d6`; 225 pairs x 2 mappings per reporter and stage; 10000 two-way bootstrap resamples (15 preferred and 15 harmful tasks drawn independently).

## All reporters x stages

Effect = log-odds toward the benign task's outcome range (nats); shift = P(benign) - 0.5 after mapping-averaging; coverage = mean P(A)+P(B); no-stake = one prompt (benign vs benign): coverage and P(first).

| reporter    | stage    | effect                  | shift                   |   coverage | no-stake cov / P(first)   | stage - base effect     | cell ratio           |
|:------------|:---------|:------------------------|:------------------------|-----------:|:--------------------------|:------------------------|:---------------------|
| 1 assistant | base     | -0.022 [-0.041, -0.002] | -0.005 [-0.010, -0.000] |      0.986 | 0.989 / 0.539             |                         |                      |
| 1 assistant | sft      | +0.390 [+0.285, +0.502] | +0.090 [+0.067, +0.115] |      0.999 | 0.999 / 0.720             | +0.412 [+0.307, +0.521] | 1 (reference)        |
| 1 assistant | dpo      | +0.642 [+0.432, +0.861] | +0.144 [+0.099, +0.187] |      1     | 1.000 / 0.594             | +0.663 [+0.455, +0.881] | 1 (reference)        |
| 1 assistant | instruct | +0.477 [+0.291, +0.672] | +0.108 [+0.068, +0.149] |      1     | 1.000 / 0.605             | +0.499 [+0.315, +0.691] | 1 (reference)        |
| 2 user      | base     | -0.086 [-0.097, -0.074] | -0.021 [-0.024, -0.018] |      0.986 | 0.989 / 0.482             |                         |                      |
| 2 user      | sft      | +0.087 [+0.050, +0.125] | +0.020 [+0.012, +0.029] |      0.995 | 0.997 / 0.678             | +0.173 [+0.137, +0.209] | +0.42 [+0.33, +0.54] |
| 2 user      | dpo      | +0.443 [+0.299, +0.595] | +0.088 [+0.062, +0.114] |      1     | 1.000 / 0.707             | +0.529 [+0.381, +0.682] | +0.80 [+0.65, +0.98] |
| 2 user      | instruct | +0.441 [+0.289, +0.597] | +0.089 [+0.060, +0.116] |      1     | 1.000 / 0.745             | +0.526 [+0.371, +0.686] | +1.06 [+0.83, +1.38] |
| 3 narration | base     | -0.375 [-0.426, -0.324] | -0.092 [-0.104, -0.079] |      0.997 | 0.997 / 0.477             |                         |                      |
| 3 narration | sft      | -0.069 [-0.149, +0.002] | -0.017 [-0.036, +0.001] |      0.989 | 0.972 / 0.744             | +0.306 [+0.224, +0.386] | +0.74 [+0.51, +1.05] |
| 3 narration | dpo      | -0.070 [-0.167, +0.027] | -0.014 [-0.034, +0.006] |      0.993 | 0.980 / 0.551             | +0.305 [+0.201, +0.410] | +0.46 [+0.29, +0.69] |
| 3 narration | instruct | -0.031 [-0.171, +0.111] | -0.006 [-0.030, +0.017] |      0.997 | 0.994 / 0.489             | +0.344 [+0.197, +0.492] | +0.69 [+0.40, +1.10] |
| 4 log       | base     | -0.267 [-0.325, -0.211] | -0.061 [-0.073, -0.048] |      0.996 | 0.999 / 0.737             |                         |                      |
| 4 log       | sft      | -0.502 [-0.664, -0.341] | -0.117 [-0.153, -0.081] |      0.995 | 0.996 / 0.578             | -0.235 [-0.380, -0.094] | -0.57 [-0.99, -0.23] |
| 4 log       | dpo      | -0.694 [-0.949, -0.435] | -0.149 [-0.200, -0.095] |      0.998 | 0.999 / 0.392             | -0.427 [-0.668, -0.194] | -0.64 [-1.16, -0.28] |
| 4 log       | instruct | -0.700 [-0.975, -0.427] | -0.141 [-0.192, -0.088] |      0.999 | 0.999 / 0.295             | -0.433 [-0.691, -0.182] | -0.87 [-1.68, -0.34] |

![stage - base effect by reporter](dominance_pilot_n15/dominance_pilot_effect_by_reporter.png)

## Length/structure control: is the base effect about harm?

Same four reporters and the same short maths tasks, against 15 harmless tasks written to parallel the harmful ones (mean tokens: benign 14.4, harmful 24.1, control 24.6). Control effect = log-odds toward the short maths task. Harm-specific = stake effect - control effect, paired by cell (preferred task x harmful/control task), two-way bootstrap on the same resamples. Cell ratio is for the harm-specific change from base.

| reporter    | stage    | stake effect            | control effect          | harm-specific           | harm-specific - base    | cell ratio (harm-specific)   |
|:------------|:---------|:------------------------|:------------------------|:------------------------|:------------------------|:-----------------------------|
| 1 assistant | base     | -0.022 [-0.041, -0.002] | +0.009 [-0.008, +0.026] | -0.030 [-0.043, -0.018] |                         |                              |
| 1 assistant | sft      | +0.390 [+0.285, +0.502] | +0.088 [+0.036, +0.144] | +0.302 [+0.215, +0.387] | +0.332 [+0.241, +0.421] | 1 (reference)                |
| 1 assistant | dpo      | +0.642 [+0.432, +0.861] | +0.090 [-0.019, +0.197] | +0.552 [+0.365, +0.733] | +0.582 [+0.392, +0.768] | 1 (reference)                |
| 1 assistant | instruct | +0.477 [+0.291, +0.672] | +0.053 [-0.064, +0.172] | +0.424 [+0.269, +0.573] | +0.454 [+0.296, +0.606] | 1 (reference)                |
| 2 user      | base     | -0.086 [-0.097, -0.074] | -0.021 [-0.032, -0.011] | -0.064 [-0.077, -0.051] |                         |                              |
| 2 user      | sft      | +0.087 [+0.050, +0.125] | +0.061 [+0.030, +0.093] | +0.026 [-0.002, +0.054] | +0.091 [+0.061, +0.120] | +0.27 [+0.18, +0.38]         |
| 2 user      | dpo      | +0.443 [+0.299, +0.595] | +0.133 [+0.020, +0.246] | +0.311 [+0.192, +0.431] | +0.375 [+0.249, +0.500] | +0.64 [+0.51, +0.77]         |
| 2 user      | instruct | +0.441 [+0.289, +0.597] | +0.128 [-0.008, +0.267] | +0.312 [+0.192, +0.431] | +0.377 [+0.249, +0.500] | +0.83 [+0.65, +1.02]         |
| 3 narration | base     | -0.375 [-0.426, -0.324] | -0.158 [-0.199, -0.117] | -0.217 [-0.266, -0.167] |                         |                              |
| 3 narration | sft      | -0.069 [-0.149, +0.002] | -0.073 [-0.141, -0.005] | +0.004 [-0.074, +0.087] | +0.221 [+0.131, +0.315] | +0.66 [+0.41, +0.98]         |
| 3 narration | dpo      | -0.070 [-0.167, +0.027] | -0.149 [-0.241, -0.063] | +0.080 [-0.031, +0.197] | +0.297 [+0.173, +0.429] | +0.51 [+0.31, +0.77]         |
| 3 narration | instruct | -0.031 [-0.171, +0.111] | -0.161 [-0.278, -0.051] | +0.130 [-0.021, +0.291] | +0.347 [+0.183, +0.524] | +0.76 [+0.44, +1.17]         |
| 4 log       | base     | -0.267 [-0.325, -0.211] | -0.124 [-0.167, -0.079] | -0.143 [-0.190, -0.098] |                         |                              |
| 4 log       | sft      | -0.502 [-0.664, -0.341] | -0.018 [-0.104, +0.069] | -0.484 [-0.638, -0.318] | -0.341 [-0.482, -0.196] | -1.03 [-1.55, -0.59]         |
| 4 log       | dpo      | -0.694 [-0.949, -0.435] | -0.105 [-0.242, +0.028] | -0.589 [-0.819, -0.347] | -0.446 [-0.662, -0.224] | -0.77 [-1.34, -0.36]         |
| 4 log       | instruct | -0.700 [-0.975, -0.427] | -0.092 [-0.246, +0.066] | -0.609 [-0.855, -0.352] | -0.465 [-0.697, -0.230] | -1.02 [-1.86, -0.46]         |

## Anything odd (automatic flags)

- large positional prior for reporter 3 (narration), instruct: mean P(A) - 0.5 = -0.25
- base already has a nonzero effect for reporter 1 (assistant): -0.022 [-0.041, -0.002]
- base already has a nonzero effect for reporter 2 (user): -0.086 [-0.097, -0.074]
- base already has a nonzero effect for reporter 3 (narration): -0.375 [-0.426, -0.324]
- base already has a nonzero effect for reporter 4 (log): -0.267 [-0.325, -0.211]

## Readout tokenization (string: tokens)

`'1'` 1, `'2'` 1, `'3'` 1, `'4'` 1, `'5'` 1, `'6'` 1

The prompt ends with one space token (`'Ġ'`) and the outcomes are bare digit tokens.
