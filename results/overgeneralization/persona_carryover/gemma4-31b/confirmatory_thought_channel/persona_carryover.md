# Persona carryover (overgeneralization)

**Family gemma4-31b: contamination could not be checked** (its post-training data is not public); breadth arm (README criterion 7). Final assistant turn opened with the template's empty thought channel: YES (--assistant-opening, as planned).

The pilot item set is exploratory; the verdict comes from the confirmatory set (README). Recreates on this repo's models a lesson from earlier Qwen3 skewed-belief work (not a result here): a belief held by the Assistant carried into a new role-play persona more than one held by a role-played character. Design and readout: module docstring.

48 items (dice: 12, drawer: 12, match: 12, weather: 12), each with both outcomes believed. Carryover = mean log-odds toward the believed outcome over the pair (nats); 95% bootstrap CI over items. Prediction fixed in advance: leak gap > 0. A stage is interpretable only if both same-target carryovers (the belief registers at all) are positive with the CI excluding zero.

## Headline (all domains)

| stage    | assistant -> new        | assistant -> same         | character -> new        | character -> same          | leak gap                | change from base        | reading                                                                      |
|:---------|:------------------------|:--------------------------|:------------------------|:---------------------------|:------------------------|:------------------------|:-----------------------------------------------------------------------------|
| base     | +0.144 [+0.009, +0.280] | +1.247 [+1.077, +1.420]   | -0.128 [-0.238, -0.016] | +1.982 [+1.777, +2.186]    | +0.272 [+0.207, +0.340] |                         | assistant carries more                                                       |
| instruct | -1.917 [-2.871, -0.927] | +11.017 [+9.972, +12.088] | -4.759 [-6.058, -3.503] | +11.722 [+10.163, +13.301] | +2.842 [+2.001, +3.764] | +2.570 [+1.718, +3.472] | assistant carries more (relative only: both new-persona carryovers negative) |

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

| quantity                                            | base                    | instruct                   |
|:----------------------------------------------------|:------------------------|:---------------------------|
| carryover assistant -> new                          | +0.276 [+0.156, +0.401] | -0.333 [-1.453, +0.585]    |
| carryover assistant -> same                         | +1.962 [+1.743, +2.201] | +12.125 [+11.843, +12.385] |
| carryover character -> new                          | +0.010 [-0.099, +0.114] | -0.708 [-1.927, +0.484]    |
| carryover character -> same                         | +2.843 [+2.645, +3.025] | +13.578 [+12.766, +14.336] |
| leak gap: assistant -> new minus character -> new   | +0.265 [+0.192, +0.333] | +0.375 [-0.056, +0.713]    |
| same gap: assistant -> same minus character -> same | -0.881 [-1.073, -0.693] | -1.453 [-2.211, -0.654]    |
| leak gap, change from base                          |                         | +0.110 [-0.271, +0.428]    |

### drawer

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | -0.141 [-0.198, -0.083] | -4.805 [-6.187, -3.526] |
| carryover assistant -> same                         | +0.500 [+0.411, +0.583] | +7.604 [+6.132, +8.932] |
| carryover character -> new                          | -0.463 [-0.588, -0.344] | -7.102 [-8.557, -5.539] |
| carryover character -> same                         | +1.026 [+0.896, +1.172] | +6.464 [+5.026, +7.787] |
| leak gap: assistant -> new minus character -> new   | +0.323 [+0.219, +0.427] | +2.297 [+1.133, +3.492] |
| same gap: assistant -> same minus character -> same | -0.526 [-0.682, -0.375] | +1.140 [+0.156, +2.153] |
| leak gap, change from base                          |                         | +1.974 [+0.844, +3.195] |

### match

| quantity                                            | base                    | instruct                   |
|:----------------------------------------------------|:------------------------|:---------------------------|
| carryover assistant -> new                          | -0.360 [-0.484, -0.235] | -2.846 [-4.575, -1.395]    |
| carryover assistant -> same                         | +1.134 [+0.922, +1.347] | +15.696 [+14.600, +16.786] |
| carryover character -> new                          | -0.391 [-0.510, -0.266] | -8.730 [-11.089, -6.411]   |
| carryover character -> same                         | +2.233 [+2.129, +2.332] | +19.414 [+18.659, +20.205] |
| leak gap: assistant -> new minus character -> new   | +0.031 [-0.063, +0.125] | +5.884 [+3.890, +7.959]    |
| same gap: assistant -> same minus character -> same | -1.099 [-1.260, -0.911] | -3.718 [-5.395, -2.064]    |
| leak gap, change from base                          |                         | +5.853 [+3.918, +7.822]    |

### weather

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +0.802 [+0.698, +0.901] | +0.315 [-1.635, +2.491] |
| carryover assistant -> same                         | +1.391 [+1.224, +1.563] | +8.643 [+7.480, +9.939] |
| carryover character -> new                          | +0.333 [+0.177, +0.474] | -2.496 [-4.432, -0.769] |
| carryover character -> same                         | +1.828 [+1.604, +2.016] | +7.430 [+6.047, +8.954] |
| leak gap: assistant -> new minus character -> new   | +0.469 [+0.349, +0.609] | +2.811 [+1.612, +4.122] |
| same gap: assistant -> same minus character -> same | -0.437 [-0.620, -0.250] | +1.213 [-0.101, +2.307] |
| leak gap, change from base                          |                         | +2.342 [+1.151, +3.667] |

## Raw log-probs, priors and coverage (all domains)

### all conditions

| quantity                                          | base                    | instruct                   |
|:--------------------------------------------------|:------------------------|:---------------------------|
| raw log P(believed), assistant>new                | -1.169 [-1.340, -1.011] | -6.043 [-7.291, -4.883]    |
| raw log P(other), assistant>new                   | -1.313 [-1.583, -1.065] | -4.126 [-5.571, -2.798]    |
| mass P(o1) + P(o2), assistant>new                 | +0.754 [+0.669, +0.836] | +0.661 [+0.529, +0.788]    |
| raw log P(believed), assistant>same               | -0.770 [-0.963, -0.595] | -3.439 [-4.440, -2.493]    |
| raw log P(other), assistant>same                  | -2.016 [-2.258, -1.773] | -14.457 [-15.473, -13.449] |
| mass P(o1) + P(o2), assistant>same                | +0.734 [+0.652, +0.808] | +0.346 [+0.241, +0.456]    |
| raw log P(believed), character>new                | -1.325 [-1.515, -1.153] | -9.029 [-10.541, -7.506]   |
| raw log P(other), character>new                   | -1.198 [-1.457, -0.960] | -4.270 [-6.000, -2.651]    |
| mass P(o1) + P(o2), character>new                 | +0.738 [+0.652, +0.819] | +0.662 [+0.531, +0.787]    |
| raw log P(believed), character>same               | -0.527 [-0.694, -0.384] | -2.139 [-3.335, -1.095]    |
| raw log P(other), character>same                  | -2.510 [-2.718, -2.296] | -13.861 [-15.247, -12.421] |
| mass P(o1) + P(o2), character>same                | +0.784 [+0.708, +0.854] | +0.761 [+0.636, +0.875]    |
| prior log P(1st) - log P(2nd), assistant-none>new | +0.616 [+0.228, +1.014] | +3.509 [+2.000, +5.042]    |
| mass P(o1) + P(o2), assistant-none>new            | +0.657 [+0.554, +0.757] | +0.677 [+0.553, +0.793]    |
| prior log P(1st) - log P(2nd), character-none>new | +0.565 [+0.200, +0.960] | +3.252 [+1.605, +4.896]    |
| mass P(o1) + P(o2), character-none>new            | +0.643 [+0.536, +0.748] | +0.725 [+0.603, +0.839]    |

## Readout tokenization (variant: tokens)

`' odd'` 1; `'odd'` 1; `' Odd'` 1; `'Odd'` 1; `' ODD'` 2; `'ODD'` 2; `' even'` 1; `'even'` 1; `' Even'` 1; `'Even'` 1; `' EVEN'` 1; `'EVEN'` 1; `' top'` 1; `'top'` 1; `' Top'` 1; `'Top'` 1; `' TOP'` 1; `'TOP'` 1; `' bottom'` 1; `'bottom'` 1; `' Bottom'` 1; `'Bottom'` 1; `' BOTTOM'` 1; `'BOTTOM'` 1; `' home'` 1; `'home'` 1; `' Home'` 1; `'Home'` 1; `' HOME'` 1; `'HOME'` 1; `' away'` 1; `'away'` 1; `' Away'` 1; `'Away'` 1; `' AWAY'` 1; `'AWAY'` 2; `' rain'` 1; `'rain'` 1; `' Rain'` 1; `'Rain'` 1; `' RAIN'` 2; `'RAIN'` 1; `' snow'` 1; `'snow'` 1; `' Snow'` 1; `'Snow'` 1; `' SNOW'` 2; `'SNOW'` 2
