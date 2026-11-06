# Persona carryover (overgeneralization)

**Family llama3.1-8b: contamination could not be checked** (its post-training data is not public); breadth arm (README criterion 7). Assistant-slot empty think block: OFF.

The pilot item set is exploratory; the verdict comes from the confirmatory set (README). Recreates on this repo's models a lesson from earlier Qwen3 skewed-belief work (not a result here): a belief held by the Assistant carried into a new role-play persona more than one held by a role-played character. Design and readout: module docstring.

48 items (dice: 12, drawer: 12, match: 12, weather: 12), each with both outcomes believed. Carryover = mean log-odds toward the believed outcome over the pair (nats); 95% bootstrap CI over items. Prediction fixed in advance: leak gap > 0. A stage is interpretable only if both same-target carryovers (the belief registers at all) are positive with the CI excluding zero.

## Headline (all domains)

| stage    | assistant -> new        | assistant -> same       | character -> new        | character -> same       | leak gap                | change from base        | reading                |
|:---------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:-----------------------|
| base     | +0.313 [+0.153, +0.475] | +0.484 [+0.290, +0.673] | +0.141 [-0.056, +0.332] | +0.807 [+0.598, +1.021] | +0.172 [+0.109, +0.240] |                         | assistant carries more |
| instruct | +0.358 [+0.081, +0.655] | +2.183 [+1.615, +2.773] | +0.101 [-0.247, +0.464] | +2.898 [+2.291, +3.534] | +0.257 [+0.058, +0.456] | +0.084 [-0.103, +0.275] | assistant carries more |

## Pre-registered verdict (criteria 1-5, README; confirmatory item set only)

delta = 0.25 nats, domain effect = 0.5 nats; a bar is met when the mean is at least the bar and the 95% CI excludes zero. Domains are never pooled. Criterion 4 is checked in every domain that meets criterion 2.

| stage | domain | 1. interpretable | 2. leak-gap change >= 0.5 | 4. assistant -> new rises >= 0.25 |
|---|---|---|---|---|
| instruct | dice | yes | no | no |
| instruct | drawer | yes | yes | no |
| instruct | match | no | no | no |
| instruct | weather | yes | no | no |

| stage | interpretable domains | domains meeting 2 | 3. stage rule | 4. absolute | reading |
|---|---|---|---|---|---|
| instruct | 3 | 1 | no | no | no overgeneralization by the rule |

**Answer to "which stage" (criterion 5):** no stage meets criteria 3 and 4.

## By domain

### dice

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +0.641 [+0.562, +0.713] | +1.046 [+0.536, +1.593] |
| carryover assistant -> same                         | +1.000 [+0.875, +1.141] | +4.015 [+3.213, +4.890] |
| carryover character -> new                          | +0.411 [+0.302, +0.510] | +1.109 [+0.666, +1.541] |
| carryover character -> same                         | +0.901 [+0.781, +1.021] | +4.005 [+3.416, +4.593] |
| leak gap: assistant -> new minus character -> new   | +0.229 [+0.161, +0.297] | -0.063 [-0.292, +0.182] |
| same gap: assistant -> same minus character -> same | +0.099 [-0.016, +0.214] | +0.010 [-0.490, +0.531] |
| leak gap, change from base                          |                         | -0.292 [-0.568, +0.010] |

### drawer

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | -0.370 [-0.510, -0.250] | -0.146 [-0.312, +0.026] |
| carryover assistant -> same                         | -0.443 [-0.661, -0.208] | +0.526 [+0.161, +0.891] |
| carryover character -> new                          | -0.776 [-0.995, -0.589] | -1.141 [-1.521, -0.750] |
| carryover character -> same                         | -0.089 [-0.219, +0.036] | +0.969 [+0.656, +1.250] |
| leak gap: assistant -> new minus character -> new   | +0.406 [+0.276, +0.552] | +0.995 [+0.656, +1.328] |
| same gap: assistant -> same minus character -> same | -0.354 [-0.490, -0.214] | -0.443 [-0.703, -0.182] |
| leak gap, change from base                          |                         | +0.589 [+0.313, +0.859] |

### match

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +0.041 [-0.089, +0.186] | -0.433 [-0.783, -0.073] |
| carryover assistant -> same                         | +0.517 [+0.291, +0.770] | +0.385 [-0.198, +1.035] |
| carryover character -> new                          | -0.017 [-0.117, +0.089] | -0.231 [-0.586, +0.180] |
| carryover character -> same                         | +0.529 [+0.364, +0.695] | +0.896 [+0.408, +1.443] |
| leak gap: assistant -> new minus character -> new   | +0.057 [-0.021, +0.138] | -0.201 [-0.378, -0.031] |
| same gap: assistant -> same minus character -> same | -0.013 [-0.130, +0.099] | -0.512 [-0.817, -0.211] |
| leak gap, change from base                          |                         | -0.258 [-0.458, -0.064] |

### weather

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +0.943 [+0.740, +1.156] | +0.963 [+0.411, +1.551] |
| carryover assistant -> same                         | +0.862 [+0.648, +1.065] | +3.804 [+3.315, +4.273] |
| carryover character -> new                          | +0.945 [+0.786, +1.122] | +0.666 [-0.055, +1.367] |
| carryover character -> same                         | +1.885 [+1.729, +2.094] | +5.723 [+5.513, +5.937] |
| leak gap: assistant -> new minus character -> new   | -0.003 [-0.083, +0.086] | +0.297 [-0.118, +0.680] |
| same gap: assistant -> same minus character -> same | -1.023 [-1.224, -0.818] | -1.919 [-2.427, -1.383] |
| leak gap, change from base                          |                         | +0.299 [-0.139, +0.721] |

## Raw log-probs, priors and coverage (all domains)

### all conditions

| quantity                                          | base                    | instruct                |
|:--------------------------------------------------|:------------------------|:------------------------|
| raw log P(believed), assistant>new                | -1.137 [-1.286, -1.000] | -1.781 [-2.213, -1.397] |
| raw log P(other), assistant>new                   | -1.451 [-1.691, -1.233] | -2.138 [-2.677, -1.661] |
| mass P(o1) + P(o2), assistant>new                 | +0.685 [+0.618, +0.749] | +0.621 [+0.527, +0.711] |
| raw log P(believed), assistant>same               | -1.077 [-1.264, -0.897] | -1.339 [-1.753, -0.966] |
| raw log P(other), assistant>same                  | -1.561 [-1.810, -1.325] | -3.522 [-4.303, -2.797] |
| mass P(o1) + P(o2), assistant>same                | +0.702 [+0.620, +0.778] | +0.692 [+0.586, +0.795] |
| raw log P(believed), character>new                | -1.230 [-1.389, -1.088] | -2.026 [-2.505, -1.592] |
| raw log P(other), character>new                   | -1.371 [-1.638, -1.130] | -2.126 [-2.682, -1.611] |
| mass P(o1) + P(o2), character>new                 | +0.691 [+0.623, +0.756] | +0.620 [+0.518, +0.716] |
| raw log P(believed), character>same               | -0.819 [-0.939, -0.709] | -0.740 [-0.943, -0.555] |
| raw log P(other), character>same                  | -1.625 [-1.901, -1.364] | -3.638 [-4.375, -2.932] |
| mass P(o1) + P(o2), character>same                | +0.764 [+0.696, +0.828] | +0.753 [+0.661, +0.835] |
| prior log P(1st) - log P(2nd), assistant-none>new | +0.818 [+0.484, +1.189] | +1.712 [+0.910, +2.577] |
| mass P(o1) + P(o2), assistant-none>new            | +0.595 [+0.513, +0.673] | +0.515 [+0.423, +0.607] |
| prior log P(1st) - log P(2nd), character-none>new | +0.894 [+0.539, +1.291] | +1.795 [+1.017, +2.642] |
| mass P(o1) + P(o2), character-none>new            | +0.605 [+0.513, +0.697] | +0.543 [+0.447, +0.639] |

## Readout tokenization (variant: tokens)

`' odd'` 1; `'odd'` 1; `' Odd'` 1; `'Odd'` 1; `' ODD'` 2; `'ODD'` 2; `' even'` 1; `'even'` 1; `' Even'` 1; `'Even'` 1; `' EVEN'` 1; `'EVEN'` 2; `' top'` 1; `'top'` 1; `' Top'` 1; `'Top'` 1; `' TOP'` 1; `'TOP'` 1; `' bottom'` 1; `'bottom'` 1; `' Bottom'` 1; `'Bottom'` 1; `' BOTTOM'` 1; `'BOTTOM'` 1; `' home'` 1; `'home'` 1; `' Home'` 1; `'Home'` 1; `' HOME'` 1; `'HOME'` 1; `' away'` 1; `'away'` 1; `' Away'` 1; `'Away'` 1; `' AWAY'` 2; `'AWAY'` 2; `' rain'` 1; `'rain'` 1; `' Rain'` 1; `'Rain'` 1; `' RAIN'` 2; `'RAIN'` 1; `' snow'` 1; `'snow'` 1; `' Snow'` 1; `'Snow'` 1; `' SNOW'` 2; `'SNOW'` 2
