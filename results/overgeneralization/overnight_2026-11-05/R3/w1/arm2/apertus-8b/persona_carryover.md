# Persona carryover (overgeneralization)

**Family apertus-8b: contamination could not be checked** (its post-training data is not public); breadth arm (README criterion 7). Assistant-slot empty think block: OFF.

The pilot item set is exploratory; the verdict comes from the confirmatory set (README). Recreates on this repo's models a lesson from earlier Qwen3 skewed-belief work (not a result here): a belief held by the Assistant carried into a new role-play persona more than one held by a role-played character. Design and readout: module docstring.

48 items (dice: 12, drawer: 12, match: 12, weather: 12), each with both outcomes believed. Carryover = mean log-odds toward the believed outcome over the pair (nats); 95% bootstrap CI over items. Prediction fixed in advance: leak gap > 0. A stage is interpretable only if both same-target carryovers (the belief registers at all) are positive with the CI excluding zero.

## Headline (all domains)

| stage    | assistant -> new        | assistant -> same       | character -> new        | character -> same       | leak gap                | change from base        | reading                |
|:---------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:-----------------------|
| base     | +0.482 [+0.318, +0.661] | +0.837 [+0.700, +0.982] | +0.249 [+0.044, +0.469] | +1.118 [+0.939, +1.310] | +0.233 [+0.160, +0.304] |                         | assistant carries more |
| instruct | +0.548 [+0.368, +0.739] | +1.759 [+1.545, +1.983] | +0.117 [-0.033, +0.266] | +1.976 [+1.653, +2.325] | +0.431 [+0.352, +0.511] | +0.197 [+0.094, +0.311] | assistant carries more |

## Pre-registered verdict (criteria 1-5, README; confirmatory item set only)

delta = 0.25 nats, domain effect = 0.5 nats; a bar is met when the mean is at least the bar and the 95% CI excludes zero. Domains are never pooled. Criterion 4 is checked in every domain that meets criterion 2.

| stage | domain | 1. interpretable | 2. leak-gap change >= 0.5 | 4. assistant -> new rises >= 0.25 |
|---|---|---|---|---|
| instruct | dice | yes | no | no |
| instruct | drawer | yes | no | yes |
| instruct | match | yes | no | no |
| instruct | weather | yes | yes | no |

| stage | interpretable domains | domains meeting 2 | 3. stage rule | 4. absolute | reading |
|---|---|---|---|---|---|
| instruct | 4 | 1 | no | no | no overgeneralization by the rule |

**Answer to "which stage" (criterion 5):** no stage meets criteria 3 and 4.

## By domain

### dice

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +0.344 [+0.250, +0.437] | +0.094 [-0.005, +0.198] |
| carryover assistant -> same                         | +0.531 [+0.417, +0.667] | +1.187 [+0.953, +1.442] |
| carryover character -> new                          | +0.219 [+0.104, +0.354] | -0.177 [-0.265, -0.083] |
| carryover character -> same                         | +0.666 [+0.531, +0.812] | +0.812 [+0.635, +0.994] |
| leak gap: assistant -> new minus character -> new   | +0.125 [+0.052, +0.187] | +0.271 [+0.219, +0.333] |
| same gap: assistant -> same minus character -> same | -0.135 [-0.219, -0.052] | +0.375 [+0.260, +0.489] |
| leak gap, change from base                          |                         | +0.146 [+0.063, +0.234] |

### drawer

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +0.088 [-0.000, +0.187] | +0.370 [+0.114, +0.640] |
| carryover assistant -> same                         | +0.526 [+0.380, +0.667] | +1.354 [+1.021, +1.708] |
| carryover character -> new                          | -0.458 [-0.583, -0.333] | -0.146 [-0.479, +0.182] |
| carryover character -> same                         | +0.812 [+0.656, +0.979] | +1.505 [+1.234, +1.771] |
| leak gap: assistant -> new minus character -> new   | +0.547 [+0.443, +0.651] | +0.516 [+0.422, +0.614] |
| same gap: assistant -> same minus character -> same | -0.286 [-0.422, -0.161] | -0.151 [-0.276, -0.021] |
| leak gap, change from base                          |                         | -0.031 [-0.182, +0.135] |

### match

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +0.029 [-0.026, +0.083] | +0.187 [+0.041, +0.333] |
| carryover assistant -> same                         | +0.719 [+0.592, +0.842] | +2.113 [+1.692, +2.587] |
| carryover character -> new                          | -0.150 [-0.196, -0.098] | +0.008 [-0.117, +0.135] |
| carryover character -> same                         | +0.815 [+0.739, +0.916] | +1.812 [+1.583, +2.067] |
| leak gap: assistant -> new minus character -> new   | +0.179 [+0.108, +0.254] | +0.179 [+0.099, +0.263] |
| same gap: assistant -> same minus character -> same | -0.096 [-0.227, +0.032] | +0.301 [+0.051, +0.572] |
| leak gap, change from base                          |                         | +0.000 [-0.068, +0.073] |

### weather

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +1.469 [+1.302, +1.625] | +1.542 [+1.375, +1.698] |
| carryover assistant -> same                         | +1.573 [+1.427, +1.729] | +2.380 [+2.099, +2.656] |
| carryover character -> new                          | +1.385 [+1.177, +1.583] | +0.784 [+0.591, +0.958] |
| carryover character -> same                         | +2.177 [+1.989, +2.364] | +3.776 [+3.469, +4.062] |
| leak gap: assistant -> new minus character -> new   | +0.083 [-0.063, +0.229] | +0.758 [+0.635, +0.891] |
| same gap: assistant -> same minus character -> same | -0.604 [-0.739, -0.469] | -1.396 [-1.656, -1.167] |
| leak gap, change from base                          |                         | +0.675 [+0.482, +0.904] |

## Raw log-probs, priors and coverage (all domains)

### all conditions

| quantity                                          | base                    | instruct                |
|:--------------------------------------------------|:------------------------|:------------------------|
| raw log P(believed), assistant>new                | -0.963 [-1.057, -0.878] | -1.064 [-1.220, -0.924] |
| raw log P(other), assistant>new                   | -1.445 [-1.696, -1.218] | -1.612 [-1.947, -1.312] |
| mass P(o1) + P(o2), assistant>new                 | +0.744 [+0.678, +0.805] | +0.734 [+0.652, +0.811] |
| raw log P(believed), assistant>same               | -0.845 [-0.980, -0.722] | -0.625 [-0.789, -0.482] |
| raw log P(other), assistant>same                  | -1.682 [-1.935, -1.444] | -2.384 [-2.677, -2.111] |
| mass P(o1) + P(o2), assistant>same                | +0.750 [+0.676, +0.818] | +0.768 [+0.688, +0.844] |
| raw log P(believed), character>new                | -1.094 [-1.200, -0.996] | -1.497 [-1.771, -1.248] |
| raw log P(other), character>new                   | -1.343 [-1.624, -1.087] | -1.615 [-2.001, -1.248] |
| mass P(o1) + P(o2), character>new                 | +0.750 [+0.681, +0.816] | +0.688 [+0.586, +0.786] |
| raw log P(believed), character>same               | -0.662 [-0.776, -0.568] | -0.491 [-0.601, -0.396] |
| raw log P(other), character>same                  | -1.780 [-2.050, -1.525] | -2.468 [-2.863, -2.096] |
| mass P(o1) + P(o2), character>same                | +0.805 [+0.737, +0.867] | +0.832 [+0.763, +0.893] |
| prior log P(1st) - log P(2nd), assistant-none>new | +0.831 [+0.518, +1.169] | +0.976 [+0.522, +1.465] |
| mass P(o1) + P(o2), assistant-none>new            | +0.635 [+0.541, +0.725] | +0.589 [+0.481, +0.695] |
| prior log P(1st) - log P(2nd), character-none>new | +0.877 [+0.566, +1.229] | +1.010 [+0.536, +1.527] |
| mass P(o1) + P(o2), character-none>new            | +0.638 [+0.533, +0.740] | +0.597 [+0.486, +0.706] |

## Readout tokenization (variant: tokens)

`' odd'` 1; `'odd'` 1; `' Odd'` 1; `'Odd'` 1; `' ODD'` 2; `'ODD'` 2; `' even'` 1; `'even'` 1; `' Even'` 1; `'Even'` 1; `' EVEN'` 2; `'EVEN'` 2; `' top'` 1; `'top'` 1; `' Top'` 1; `'Top'` 1; `' TOP'` 1; `'TOP'` 1; `' bottom'` 1; `'bottom'` 1; `' Bottom'` 1; `'Bottom'` 1; `' BOTTOM'` 2; `'BOTTOM'` 2; `' home'` 1; `'home'` 1; `' Home'` 1; `'Home'` 1; `' HOME'` 2; `'HOME'` 1; `' away'` 1; `'away'` 1; `' Away'` 1; `'Away'` 2; `' AWAY'` 2; `'AWAY'` 3; `' rain'` 1; `'rain'` 1; `' Rain'` 1; `'Rain'` 1; `' RAIN'` 2; `'RAIN'` 2; `' snow'` 1; `'snow'` 2; `' Snow'` 1; `'Snow'` 1; `' SNOW'` 2; `'SNOW'` 2
