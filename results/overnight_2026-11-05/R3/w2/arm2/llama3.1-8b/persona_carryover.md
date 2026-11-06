# Persona carryover (overgeneralization)

**Family llama3.1-8b: contamination could not be checked** (its post-training data is not public); breadth arm (README criterion 7). Assistant-slot empty think block: OFF.

The pilot item set is exploratory; the verdict comes from the confirmatory set (README). Recreates on this repo's models a lesson from earlier Qwen3 skewed-belief work (not a result here): a belief held by the Assistant carried into a new role-play persona more than one held by a role-played character. Design and readout: module docstring.

48 items (dice: 12, drawer: 12, match: 12, weather: 12), each with both outcomes believed. Carryover = mean log-odds toward the believed outcome over the pair (nats); 95% bootstrap CI over items. Prediction fixed in advance: leak gap > 0. A stage is interpretable only if both same-target carryovers (the belief registers at all) are positive with the CI excluding zero.

## Headline (all domains)

| stage    | assistant -> new        | assistant -> same       | character -> new        | character -> same       | leak gap                | change from base        | reading                                                                   |
|:---------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:--------------------------------------------------------------------------|
| base     | +0.142 [-0.043, +0.329] | +0.484 [+0.290, +0.673] | -0.033 [-0.246, +0.169] | +0.807 [+0.598, +1.021] | +0.175 [+0.114, +0.241] |                         | assistant carries more                                                    |
| instruct | -0.591 [-0.908, -0.280] | +2.183 [+1.615, +2.773] | -0.753 [-1.141, -0.364] | +2.898 [+2.291, +3.534] | +0.162 [-0.006, +0.345] | -0.013 [-0.190, +0.187] | no clear difference (relative only: both new-persona carryovers negative) |

## Pre-registered verdict (criteria 1-5, README; confirmatory item set only)

delta = 0.25 nats, domain effect = 0.5 nats; a bar is met when the mean is at least the bar and the 95% CI excludes zero. Domains are never pooled. Criterion 4 is checked in every domain that meets criterion 2.

| stage | domain | 1. interpretable | 2. leak-gap change >= 0.5 | 4. assistant -> new rises >= 0.25 |
|---|---|---|---|---|
| instruct | dice | yes | no | no |
| instruct | drawer | yes | no | no |
| instruct | match | no | no | no |
| instruct | weather | yes | no | no |

| stage | interpretable domains | domains meeting 2 | 3. stage rule | 4. absolute | reading |
|---|---|---|---|---|---|
| instruct | 3 | 0 | no | no | no overgeneralization by the rule |

**Answer to "which stage" (criterion 5):** no stage meets criteria 3 and 4.

## By domain

### dice

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +0.375 [+0.271, +0.489] | +0.276 [-0.031, +0.604] |
| carryover assistant -> same                         | +1.000 [+0.875, +1.141] | +4.015 [+3.213, +4.890] |
| carryover character -> new                          | +0.323 [+0.213, +0.443] | +0.286 [-0.094, +0.620] |
| carryover character -> same                         | +0.901 [+0.781, +1.021] | +4.005 [+3.416, +4.593] |
| leak gap: assistant -> new minus character -> new   | +0.052 [+0.026, +0.078] | -0.010 [-0.245, +0.255] |
| same gap: assistant -> same minus character -> same | +0.099 [-0.016, +0.214] | +0.010 [-0.490, +0.531] |
| leak gap, change from base                          |                         | -0.063 [-0.297, +0.219] |

### drawer

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | -0.641 [-0.844, -0.448] | -2.073 [-2.505, -1.656] |
| carryover assistant -> same                         | -0.443 [-0.661, -0.208] | +0.526 [+0.161, +0.891] |
| carryover character -> new                          | -1.042 [-1.302, -0.813] | -2.401 [-2.807, -1.979] |
| carryover character -> same                         | -0.089 [-0.219, +0.036] | +0.969 [+0.656, +1.250] |
| leak gap: assistant -> new minus character -> new   | +0.401 [+0.266, +0.557] | +0.328 [+0.026, +0.703] |
| same gap: assistant -> same minus character -> same | -0.354 [-0.490, -0.214] | -0.443 [-0.703, -0.182] |
| leak gap, change from base                          |                         | -0.073 [-0.385, +0.281] |

### match

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | -0.132 [-0.243, -0.026] | -0.712 [-0.910, -0.504] |
| carryover assistant -> same                         | +0.517 [+0.291, +0.770] | +0.385 [-0.198, +1.035] |
| carryover character -> new                          | -0.153 [-0.264, -0.044] | -0.941 [-1.424, -0.488] |
| carryover character -> same                         | +0.529 [+0.364, +0.695] | +0.896 [+0.408, +1.443] |
| leak gap: assistant -> new minus character -> new   | +0.021 [-0.052, +0.093] | +0.229 [-0.121, +0.655] |
| same gap: assistant -> same minus character -> same | -0.013 [-0.130, +0.099] | -0.512 [-0.817, -0.211] |
| leak gap, change from base                          |                         | +0.208 [-0.166, +0.654] |

### weather

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +0.966 [+0.776, +1.171] | +0.145 [-0.274, +0.573] |
| carryover assistant -> same                         | +0.862 [+0.648, +1.065] | +3.804 [+3.315, +4.273] |
| carryover character -> new                          | +0.739 [+0.541, +0.948] | +0.044 [-0.641, +0.682] |
| carryover character -> same                         | +1.885 [+1.729, +2.094] | +5.723 [+5.513, +5.937] |
| leak gap: assistant -> new minus character -> new   | +0.226 [+0.130, +0.315] | +0.101 [-0.227, +0.500] |
| same gap: assistant -> same minus character -> same | -1.023 [-1.224, -0.818] | -1.919 [-2.427, -1.383] |
| leak gap, change from base                          |                         | -0.125 [-0.495, +0.360] |

## Raw log-probs, priors and coverage (all domains)

### all conditions

| quantity                                          | base                    | instruct                |
|:--------------------------------------------------|:------------------------|:------------------------|
| raw log P(believed), assistant>new                | -1.336 [-1.484, -1.195] | -2.338 [-2.759, -1.954] |
| raw log P(other), assistant>new                   | -1.478 [-1.736, -1.241] | -1.747 [-2.256, -1.290] |
| mass P(o1) + P(o2), assistant>new                 | +0.610 [+0.546, +0.671] | +0.591 [+0.493, +0.685] |
| raw log P(believed), assistant>same               | -1.077 [-1.264, -0.897] | -1.339 [-1.753, -0.966] |
| raw log P(other), assistant>same                  | -1.561 [-1.810, -1.325] | -3.522 [-4.303, -2.797] |
| mass P(o1) + P(o2), assistant>same                | +0.702 [+0.620, +0.778] | +0.692 [+0.586, +0.795] |
| raw log P(believed), character>new                | -1.368 [-1.529, -1.222] | -2.637 [-3.116, -2.199] |
| raw log P(other), character>new                   | -1.335 [-1.590, -1.101] | -1.885 [-2.450, -1.356] |
| mass P(o1) + P(o2), character>new                 | +0.649 [+0.584, +0.712] | +0.581 [+0.478, +0.679] |
| raw log P(believed), character>same               | -0.819 [-0.939, -0.709] | -0.740 [-0.943, -0.555] |
| raw log P(other), character>same                  | -1.625 [-1.901, -1.364] | -3.638 [-4.375, -2.932] |
| mass P(o1) + P(o2), character>same                | +0.764 [+0.696, +0.828] | +0.753 [+0.661, +0.835] |
| prior log P(1st) - log P(2nd), assistant-none>new | +0.626 [+0.274, +1.014] | +1.130 [+0.418, +1.908] |
| mass P(o1) + P(o2), assistant-none>new            | +0.488 [+0.419, +0.556] | +0.575 [+0.479, +0.668] |
| prior log P(1st) - log P(2nd), character-none>new | +0.636 [+0.277, +1.030] | +1.323 [+0.607, +2.108] |
| mass P(o1) + P(o2), character-none>new            | +0.535 [+0.455, +0.616] | +0.566 [+0.469, +0.661] |

## Readout tokenization (variant: tokens)

`' odd'` 1; `'odd'` 1; `' Odd'` 1; `'Odd'` 1; `' ODD'` 2; `'ODD'` 2; `' even'` 1; `'even'` 1; `' Even'` 1; `'Even'` 1; `' EVEN'` 1; `'EVEN'` 2; `' top'` 1; `'top'` 1; `' Top'` 1; `'Top'` 1; `' TOP'` 1; `'TOP'` 1; `' bottom'` 1; `'bottom'` 1; `' Bottom'` 1; `'Bottom'` 1; `' BOTTOM'` 1; `'BOTTOM'` 1; `' home'` 1; `'home'` 1; `' Home'` 1; `'Home'` 1; `' HOME'` 1; `'HOME'` 1; `' away'` 1; `'away'` 1; `' Away'` 1; `'Away'` 1; `' AWAY'` 2; `'AWAY'` 2; `' rain'` 1; `'rain'` 1; `' Rain'` 1; `'Rain'` 1; `' RAIN'` 2; `'RAIN'` 1; `' snow'` 1; `'snow'` 1; `' Snow'` 1; `'Snow'` 1; `' SNOW'` 2; `'SNOW'` 2
