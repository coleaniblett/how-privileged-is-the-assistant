# Persona carryover (overgeneralization)

**Family gemma4-31b: contamination could not be checked** (its post-training data is not public); breadth arm (README criterion 7). Assistant-slot empty think block: OFF.

The pilot item set is exploratory; the verdict comes from the confirmatory set (README). Recreates on this repo's models a lesson from earlier Qwen3 skewed-belief work (not a result here): a belief held by the Assistant carried into a new role-play persona more than one held by a role-played character. Design and readout: module docstring.

48 items (dice: 12, drawer: 12, match: 12, weather: 12), each with both outcomes believed. Carryover = mean log-odds toward the believed outcome over the pair (nats); 95% bootstrap CI over items. Prediction fixed in advance: leak gap > 0. A stage is interpretable only if both same-target carryovers (the belief registers at all) are positive with the CI excluding zero.

## Headline (all domains)

| stage    | assistant -> new        | assistant -> same         | character -> new        | character -> same       | leak gap                | change from base        | reading                |
|:---------|:------------------------|:--------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:-----------------------|
| base     | +0.147 [+0.015, +0.281] | +1.125 [+0.950, +1.303]   | -0.049 [-0.163, +0.064] | +2.035 [+1.823, +2.243] | +0.196 [+0.130, +0.268] |                         | assistant carries more |
| instruct | +0.190 [-0.619, +1.048] | +10.824 [+9.812, +11.893] | -2.361 [-3.124, -1.598] | +6.569 [+5.665, +7.473] | +2.551 [+1.730, +3.390] | +2.354 [+1.516, +3.183] | assistant carries more |

## Pre-registered verdict (criteria 1-5, README; confirmatory item set only)

delta = 0.25 nats, domain effect = 0.5 nats; a bar is met when the mean is at least the bar and the 95% CI excludes zero. Domains are never pooled. Criterion 4 is checked in every domain that meets criterion 2.

| stage | domain | 1. interpretable | 2. leak-gap change >= 0.5 | 4. assistant -> new rises >= 0.25 |
|---|---|---|---|---|
| instruct | dice | yes | no | no |
| instruct | drawer | yes | yes | no |
| instruct | match | yes | yes | no |
| instruct | weather | yes | yes | no |

| stage | interpretable domains | domains meeting 2 | 3. stage rule | 4. absolute | reading |
|---|---|---|---|---|---|
| instruct | 4 | 3 | yes | no | relative only (criterion 3 holds, criterion 4 does not) |

**Answer to "which stage" (criterion 5):** no stage meets criteria 3 and 4.

## By domain

### dice

| quantity                                            | base                    | instruct                 |
|:----------------------------------------------------|:------------------------|:-------------------------|
| carryover assistant -> new                          | +0.203 [+0.073, +0.328] | +0.005 [-0.880, +0.724]  |
| carryover assistant -> same                         | +1.843 [+1.619, +2.067] | +9.677 [+9.151, +10.177] |
| carryover character -> new                          | +0.068 [-0.083, +0.219] | -0.234 [-1.016, +0.469]  |
| carryover character -> same                         | +2.812 [+2.572, +3.036] | +6.555 [+4.833, +8.099]  |
| leak gap: assistant -> new minus character -> new   | +0.135 [+0.010, +0.239] | +0.240 [+0.016, +0.453]  |
| same gap: assistant -> same minus character -> same | -0.969 [-1.156, -0.766] | +3.122 [+1.927, +4.474]  |
| leak gap, change from base                          |                         | +0.104 [-0.151, +0.339]  |

### drawer

| quantity                                            | base                    | instruct                 |
|:----------------------------------------------------|:------------------------|:-------------------------|
| carryover assistant -> new                          | -0.099 [-0.151, -0.047] | -0.602 [-2.151, +0.938]  |
| carryover assistant -> same                         | +0.323 [+0.234, +0.411] | +8.541 [+6.981, +10.039] |
| carryover character -> new                          | -0.323 [-0.437, -0.224] | -4.966 [-6.435, -3.297]  |
| carryover character -> same                         | +0.990 [+0.885, +1.094] | +7.517 [+6.042, +9.060]  |
| leak gap: assistant -> new minus character -> new   | +0.224 [+0.120, +0.344] | +4.365 [+2.927, +5.742]  |
| same gap: assistant -> same minus character -> same | -0.667 [-0.745, -0.594] | +1.025 [-0.771, +2.862]  |
| leak gap, change from base                          |                         | +4.141 [+2.734, +5.466]  |

### match

| quantity                                            | base                    | instruct                   |
|:----------------------------------------------------|:------------------------|:---------------------------|
| carryover assistant -> new                          | -0.339 [-0.469, -0.214] | -0.314 [-1.511, +0.808]    |
| carryover assistant -> same                         | +1.187 [+0.995, +1.384] | +15.851 [+14.755, +16.801] |
| carryover character -> new                          | -0.364 [-0.505, -0.224] | -2.876 [-3.730, -2.039]    |
| carryover character -> same                         | +2.327 [+2.222, +2.437] | +5.844 [+3.625, +8.231]    |
| leak gap: assistant -> new minus character -> new   | +0.026 [-0.079, +0.140] | +2.562 [+1.339, +3.804]    |
| same gap: assistant -> same minus character -> same | -1.140 [-1.323, -0.948] | +10.007 [+7.815, +12.169]  |
| leak gap, change from base                          |                         | +2.537 [+1.314, +3.724]    |

### weather

| quantity                                            | base                    | instruct                 |
|:----------------------------------------------------|:------------------------|:-------------------------|
| carryover assistant -> new                          | +0.823 [+0.745, +0.906] | +1.670 [-0.666, +4.068]  |
| carryover assistant -> same                         | +1.146 [+0.974, +1.318] | +9.227 [+7.534, +10.835] |
| carryover character -> new                          | +0.422 [+0.260, +0.578] | -1.367 [-2.642, +0.015]  |
| carryover character -> same                         | +2.010 [+1.766, +2.250] | +6.359 [+4.893, +7.941]  |
| leak gap: assistant -> new minus character -> new   | +0.401 [+0.292, +0.531] | +3.037 [+0.852, +5.243]  |
| same gap: assistant -> same minus character -> same | -0.865 [-1.052, -0.687] | +2.867 [+0.714, +4.999]  |
| leak gap, change from base                          |                         | +2.636 [+0.439, +4.866]  |

## Raw log-probs, priors and coverage (all domains)

### all conditions

| quantity                                          | base                    | instruct                   |
|:--------------------------------------------------|:------------------------|:---------------------------|
| raw log P(believed), assistant>new                | -1.190 [-1.375, -1.022] | -5.522 [-6.828, -4.294]    |
| raw log P(other), assistant>new                   | -1.337 [-1.625, -1.074] | -5.712 [-7.006, -4.461]    |
| mass P(o1) + P(o2), assistant>new                 | +0.752 [+0.663, +0.836] | +0.600 [+0.477, +0.725]    |
| raw log P(believed), assistant>same               | -0.827 [-1.049, -0.624] | -1.135 [-1.812, -0.548]    |
| raw log P(other), assistant>same                  | -1.952 [-2.196, -1.705] | -11.959 [-12.907, -11.012] |
| mass P(o1) + P(o2), assistant>same                | +0.742 [+0.653, +0.822] | +0.756 [+0.647, +0.858]    |
| raw log P(believed), character>new                | -1.269 [-1.453, -1.097] | -7.211 [-8.483, -5.953]    |
| raw log P(other), character>new                   | -1.220 [-1.479, -0.983] | -4.850 [-6.128, -3.604]    |
| mass P(o1) + P(o2), character>new                 | +0.745 [+0.659, +0.826] | +0.628 [+0.505, +0.746]    |
| raw log P(believed), character>same               | -0.529 [-0.697, -0.382] | -3.273 [-4.579, -2.112]    |
| raw log P(other), character>same                  | -2.564 [-2.790, -2.333] | -9.841 [-11.082, -8.645]   |
| mass P(o1) + P(o2), character>same                | +0.785 [+0.705, +0.858] | +0.619 [+0.502, +0.732]    |
| prior log P(1st) - log P(2nd), assistant-none>new | +0.679 [+0.278, +1.101] | +3.936 [+2.160, +5.714]    |
| mass P(o1) + P(o2), assistant-none>new            | +0.654 [+0.548, +0.757] | +0.568 [+0.438, +0.695]    |
| prior log P(1st) - log P(2nd), character-none>new | +0.541 [+0.174, +0.942] | +4.210 [+2.386, +6.084]    |
| mass P(o1) + P(o2), character-none>new            | +0.647 [+0.540, +0.753] | +0.619 [+0.492, +0.740]    |

## Readout tokenization (variant: tokens)

`' odd'` 1; `'odd'` 1; `' Odd'` 1; `'Odd'` 1; `' ODD'` 2; `'ODD'` 2; `' even'` 1; `'even'` 1; `' Even'` 1; `'Even'` 1; `' EVEN'` 1; `'EVEN'` 1; `' top'` 1; `'top'` 1; `' Top'` 1; `'Top'` 1; `' TOP'` 1; `'TOP'` 1; `' bottom'` 1; `'bottom'` 1; `' Bottom'` 1; `'Bottom'` 1; `' BOTTOM'` 1; `'BOTTOM'` 1; `' home'` 1; `'home'` 1; `' Home'` 1; `'Home'` 1; `' HOME'` 1; `'HOME'` 1; `' away'` 1; `'away'` 1; `' Away'` 1; `'Away'` 1; `' AWAY'` 1; `'AWAY'` 2; `' rain'` 1; `'rain'` 1; `' Rain'` 1; `'Rain'` 1; `' RAIN'` 2; `'RAIN'` 1; `' snow'` 1; `'snow'` 1; `' Snow'` 1; `'Snow'` 1; `' SNOW'` 2; `'SNOW'` 2
