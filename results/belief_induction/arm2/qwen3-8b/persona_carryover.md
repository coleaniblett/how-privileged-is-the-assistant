# Persona carryover (overgeneralization)

**Family qwen3-8b: contamination could not be checked** (its post-training data is not public); breadth arm (README criterion 7). Assistant-slot empty think block: ON (the pre-declared primary rendering, docs/decisions.md 2026-11-02).

The pilot item set is exploratory; the verdict comes from the confirmatory set (README). Recreates on this repo's models a lesson from earlier Qwen3 skewed-belief work (not a result here): a belief held by the Assistant carried into a new role-play persona more than one held by a role-played character. Design and readout: module docstring.

48 items (dice: 12, drawer: 12, match: 12, weather: 12), each with both outcomes believed. Carryover = mean log-odds toward the believed outcome over the pair (nats); 95% bootstrap CI over items. Prediction fixed in advance: leak gap > 0. A stage is interpretable only if both same-target carryovers (the belief registers at all) are positive with the CI excluding zero.

## Headline (all domains)

| stage    | assistant -> new        | assistant -> same       | character -> new        | character -> same       | leak gap                | change from base        | reading                |
|:---------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:-----------------------|
| base     | +0.597 [+0.484, +0.718] | +0.860 [+0.724, +1.001] | +0.303 [+0.141, +0.469] | +1.402 [+1.200, +1.610] | +0.294 [+0.206, +0.385] |                         | assistant carries more |
| instruct | +1.141 [+0.864, +1.443] | +6.738 [+6.103, +7.391] | +0.294 [-0.086, +0.661] | +8.133 [+7.456, +8.841] | +0.846 [+0.555, +1.144] | +0.552 [+0.266, +0.854] | assistant carries more |

## Pre-registered verdict (criteria 1-5, README; confirmatory item set only)

delta = 0.25 nats, domain effect = 0.5 nats; a bar is met when the mean is at least the bar and the 95% CI excludes zero. Domains are never pooled. Criterion 4 is checked in every domain that meets criterion 2.

| stage | domain | 1. interpretable | 2. leak-gap change >= 0.5 | 4. assistant -> new rises >= 0.25 |
|---|---|---|---|---|
| instruct | dice | yes | no | yes |
| instruct | drawer | yes | yes | no |
| instruct | match | yes | no | yes |
| instruct | weather | yes | yes | yes |

| stage | interpretable domains | domains meeting 2 | 3. stage rule | 4. absolute | reading |
|---|---|---|---|---|---|
| instruct | 4 | 2 | no | no | no overgeneralization by the rule |

**Answer to "which stage" (criterion 5):** no stage meets criteria 3 and 4.

## By domain

### dice

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +0.635 [+0.552, +0.719] | +0.917 [+0.672, +1.151] |
| carryover assistant -> same                         | +0.979 [+0.844, +1.114] | +6.477 [+5.396, +7.445] |
| carryover character -> new                          | +0.385 [+0.281, +0.479] | +0.786 [+0.589, +0.990] |
| carryover character -> same                         | +0.958 [+0.875, +1.041] | +6.786 [+6.135, +7.458] |
| leak gap: assistant -> new minus character -> new   | +0.250 [+0.156, +0.344] | +0.130 [-0.057, +0.323] |
| same gap: assistant -> same minus character -> same | +0.021 [-0.062, +0.135] | -0.310 [-1.044, +0.445] |
| leak gap, change from base                          |                         | -0.120 [-0.307, +0.062] |

### drawer

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +0.250 [+0.104, +0.406] | +0.630 [+0.187, +1.104] |
| carryover assistant -> same                         | +0.333 [+0.208, +0.490] | +5.297 [+4.042, +6.807] |
| carryover character -> new                          | -0.385 [-0.573, -0.208] | -0.724 [-1.224, -0.265] |
| carryover character -> same                         | +0.823 [+0.625, +1.062] | +6.661 [+5.510, +7.927] |
| leak gap: assistant -> new minus character -> new   | +0.635 [+0.437, +0.844] | +1.354 [+0.964, +1.740] |
| same gap: assistant -> same minus character -> same | -0.490 [-0.677, -0.312] | -1.365 [-2.271, -0.526] |
| leak gap, change from base                          |                         | +0.719 [+0.255, +1.161] |

### match

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +0.406 [+0.281, +0.530] | +0.692 [+0.437, +0.937] |
| carryover assistant -> same                         | +0.978 [+0.822, +1.165] | +6.922 [+5.755, +8.063] |
| carryover character -> new                          | +0.281 [+0.125, +0.458] | +0.568 [-0.010, +1.135] |
| carryover character -> same                         | +1.364 [+1.114, +1.614] | +8.318 [+7.271, +9.427] |
| leak gap: assistant -> new minus character -> new   | +0.125 [+0.041, +0.208] | +0.125 [-0.287, +0.536] |
| same gap: assistant -> same minus character -> same | -0.386 [-0.604, -0.156] | -1.396 [-2.135, -0.568] |
| leak gap, change from base                          |                         | -0.000 [-0.390, +0.385] |

### weather

| quantity                                            | base                    | instruct                  |
|:----------------------------------------------------|:------------------------|:--------------------------|
| carryover assistant -> new                          | +1.099 [+0.896, +1.318] | +2.323 [+1.677, +2.917]   |
| carryover assistant -> same                         | +1.151 [+0.823, +1.479] | +8.255 [+7.250, +9.281]   |
| carryover character -> new                          | +0.932 [+0.687, +1.193] | +0.547 [-0.557, +1.536]   |
| carryover character -> same                         | +2.463 [+2.271, +2.646] | +10.766 [+9.812, +11.880] |
| leak gap: assistant -> new minus character -> new   | +0.167 [+0.010, +0.323] | +1.776 [+1.198, +2.380]   |
| same gap: assistant -> same minus character -> same | -1.312 [-1.573, -1.052] | -2.510 [-3.224, -1.792]   |
| leak gap, change from base                          |                         | +1.609 [+1.052, +2.198]   |

## Raw log-probs, priors and coverage (all domains)

### all conditions

| quantity                                          | base                    | instruct                |
|:--------------------------------------------------|:------------------------|:------------------------|
| raw log P(believed), assistant>new                | -1.102 [-1.284, -0.941] | -0.978 [-1.230, -0.762] |
| raw log P(other), assistant>new                   | -1.700 [-1.953, -1.470] | -2.119 [-2.574, -1.716] |
| mass P(o1) + P(o2), assistant>new                 | +0.675 [+0.600, +0.745] | +0.771 [+0.681, +0.856] |
| raw log P(believed), assistant>same               | -0.907 [-1.073, -0.749] | -1.322 [-1.789, -0.907] |
| raw log P(other), assistant>same                  | -1.767 [-2.009, -1.546] | -8.060 [-8.823, -7.342] |
| mass P(o1) + P(o2), assistant>same                | +0.716 [+0.634, +0.790] | +0.521 [+0.434, +0.605] |
| raw log P(believed), character>new                | -1.152 [-1.317, -0.996] | -1.685 [-2.153, -1.261] |
| raw log P(other), character>new                   | -1.456 [-1.739, -1.203] | -1.979 [-2.534, -1.487] |
| mass P(o1) + P(o2), character>new                 | +0.718 [+0.637, +0.794] | +0.755 [+0.650, +0.854] |
| raw log P(believed), character>same               | -0.635 [-0.753, -0.533] | -0.318 [-0.543, -0.136] |
| raw log P(other), character>same                  | -2.037 [-2.326, -1.764] | -8.450 [-9.247, -7.691] |
| mass P(o1) + P(o2), character>same                | +0.794 [+0.721, +0.861] | +0.868 [+0.795, +0.930] |
| prior log P(1st) - log P(2nd), assistant-none>new | +1.110 [+0.761, +1.471] | +1.547 [+0.953, +2.174] |
| mass P(o1) + P(o2), assistant-none>new            | +0.523 [+0.435, +0.611] | +0.593 [+0.474, +0.712] |
| prior log P(1st) - log P(2nd), character-none>new | +0.988 [+0.696, +1.321] | +1.690 [+0.945, +2.503] |
| mass P(o1) + P(o2), character-none>new            | +0.603 [+0.500, +0.705] | +0.591 [+0.473, +0.707] |

## Readout tokenization (variant: tokens)

`' odd'` 1; `'odd'` 1; `' Odd'` 1; `'Odd'` 1; `' ODD'` 2; `'ODD'` 2; `' even'` 1; `'even'` 1; `' Even'` 1; `'Even'` 1; `' EVEN'` 1; `'EVEN'` 2; `' top'` 1; `'top'` 1; `' Top'` 1; `'Top'` 1; `' TOP'` 1; `'TOP'` 1; `' bottom'` 1; `'bottom'` 1; `' Bottom'` 1; `'Bottom'` 1; `' BOTTOM'` 1; `'BOTTOM'` 1; `' home'` 1; `'home'` 1; `' Home'` 1; `'Home'` 1; `' HOME'` 1; `'HOME'` 1; `' away'` 1; `'away'` 1; `' Away'` 1; `'Away'` 1; `' AWAY'` 2; `'AWAY'` 2; `' rain'` 1; `'rain'` 1; `' Rain'` 1; `'Rain'` 1; `' RAIN'` 2; `'RAIN'` 1; `' snow'` 1; `'snow'` 1; `' Snow'` 1; `'Snow'` 1; `' SNOW'` 2; `'SNOW'` 2
