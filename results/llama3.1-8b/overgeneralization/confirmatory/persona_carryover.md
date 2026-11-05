# Persona carryover (overgeneralization)

**Family llama3.1-8b: contamination could not be checked** (its post-training data is not public); breadth arm (README criterion 7). Assistant-slot empty think block: OFF.

The pilot item set is exploratory; the verdict comes from the confirmatory set (README). Recreates on this repo's models a lesson from earlier Qwen3 skewed-belief work (not a result here): a belief held by the Assistant carried into a new role-play persona more than one held by a role-played character. Design and readout: module docstring.

48 items (dice: 12, drawer: 12, match: 12, weather: 12), each with both outcomes believed. Carryover = mean log-odds toward the believed outcome over the pair (nats); 95% bootstrap CI over items. Prediction fixed in advance: leak gap > 0. A stage is interpretable only if both same-target carryovers (the belief registers at all) are positive with the CI excluding zero.

## Headline (all domains)

| stage    | assistant -> new        | assistant -> same       | character -> new        | character -> same       | leak gap                | change from base        | reading                                                                      |
|:---------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:-----------------------------------------------------------------------------|
| base     | +0.442 [+0.248, +0.633] | +0.484 [+0.290, +0.673] | +0.092 [-0.134, +0.305] | +1.374 [+1.113, +1.624] | +0.350 [+0.266, +0.436] |                         | assistant carries more                                                       |
| instruct | -1.469 [-1.932, -1.000] | +2.183 [+1.615, +2.773] | -1.948 [-2.498, -1.394] | +2.929 [+2.415, +3.469] | +0.479 [+0.275, +0.683] | +0.130 [-0.084, +0.337] | assistant carries more (relative only: both new-persona carryovers negative) |

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
| carryover assistant -> new                          | +0.875 [+0.771, +0.974] | +0.198 [-0.073, +0.484] |
| carryover assistant -> same                         | +1.000 [+0.875, +1.141] | +4.015 [+3.213, +4.890] |
| carryover character -> new                          | +0.729 [+0.661, +0.802] | +0.161 [-0.068, +0.396] |
| carryover character -> same                         | +2.015 [+1.849, +2.198] | +5.133 [+4.401, +5.877] |
| leak gap: assistant -> new minus character -> new   | +0.146 [+0.078, +0.208] | +0.036 [-0.151, +0.224] |
| same gap: assistant -> same minus character -> same | -1.016 [-1.229, -0.812] | -1.117 [-1.659, -0.612] |
| leak gap, change from base                          |                         | -0.109 [-0.281, +0.063] |

### drawer

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | -0.411 [-0.552, -0.266] | -3.641 [-4.125, -3.177] |
| carryover assistant -> same                         | -0.443 [-0.661, -0.208] | +0.526 [+0.161, +0.891] |
| carryover character -> new                          | -1.073 [-1.250, -0.896] | -4.367 [-4.898, -3.883] |
| carryover character -> same                         | +0.182 [-0.057, +0.432] | +1.510 [+1.193, +1.859] |
| leak gap: assistant -> new minus character -> new   | +0.661 [+0.510, +0.823] | +0.727 [+0.133, +1.255] |
| same gap: assistant -> same minus character -> same | -0.625 [-0.760, -0.505] | -0.984 [-1.432, -0.484] |
| leak gap, change from base                          |                         | +0.065 [-0.638, +0.703] |

### match

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +0.088 [+0.015, +0.155] | -1.565 [-1.744, -1.393] |
| carryover assistant -> same                         | +0.517 [+0.291, +0.770] | +0.385 [-0.198, +1.035] |
| carryover character -> new                          | -0.006 [-0.047, +0.041] | -1.890 [-2.224, -1.536] |
| carryover character -> same                         | +1.024 [+0.857, +1.191] | +1.504 [+0.925, +2.161] |
| leak gap: assistant -> new minus character -> new   | +0.093 [+0.031, +0.151] | +0.326 [+0.073, +0.573] |
| same gap: assistant -> same minus character -> same | -0.507 [-0.687, -0.325] | -1.120 [-1.594, -0.709] |
| leak gap, change from base                          |                         | +0.232 [-0.015, +0.487] |

### weather

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +1.216 [+1.023, +1.417] | -0.867 [-1.667, -0.031] |
| carryover assistant -> same                         | +0.862 [+0.648, +1.065] | +3.804 [+3.315, +4.273] |
| carryover character -> new                          | +0.719 [+0.552, +0.891] | -1.695 [-2.695, -0.708] |
| carryover character -> same                         | +2.273 [+2.052, +2.555] | +3.567 [+2.832, +4.179] |
| leak gap: assistant -> new minus character -> new   | +0.497 [+0.383, +0.607] | +0.828 [+0.547, +1.164] |
| same gap: assistant -> same minus character -> same | -1.411 [-1.685, -1.146] | +0.237 [-0.417, +0.857] |
| leak gap, change from base                          |                         | +0.331 [+0.003, +0.703] |

## Raw log-probs, priors and coverage (all domains)

### all conditions

| quantity                                          | base                    | instruct                |
|:--------------------------------------------------|:------------------------|:------------------------|
| raw log P(believed), assistant>new                | -1.082 [-1.250, -0.922] | -4.242 [-5.226, -3.355] |
| raw log P(other), assistant>new                   | -1.524 [-1.820, -1.252] | -2.773 [-3.826, -1.830] |
| mass P(o1) + P(o2), assistant>new                 | +0.714 [+0.633, +0.792] | +0.603 [+0.496, +0.705] |
| raw log P(believed), assistant>same               | -1.077 [-1.264, -0.897] | -1.339 [-1.753, -0.966] |
| raw log P(other), assistant>same                  | -1.561 [-1.810, -1.325] | -3.522 [-4.303, -2.797] |
| mass P(o1) + P(o2), assistant>same                | +0.702 [+0.620, +0.778] | +0.692 [+0.586, +0.795] |
| raw log P(believed), character>new                | -1.340 [-1.555, -1.132] | -4.727 [-5.774, -3.767] |
| raw log P(other), character>new                   | -1.432 [-1.735, -1.156] | -2.779 [-3.810, -1.823] |
| mass P(o1) + P(o2), character>new                 | +0.691 [+0.607, +0.772] | +0.581 [+0.473, +0.686] |
| raw log P(believed), character>same               | -0.667 [-0.792, -0.549] | -1.202 [-1.749, -0.735] |
| raw log P(other), character>same                  | -2.041 [-2.321, -1.772] | -4.131 [-4.863, -3.429] |
| mass P(o1) + P(o2), character>same                | +0.762 [+0.692, +0.826] | +0.711 [+0.600, +0.809] |
| prior log P(1st) - log P(2nd), assistant-none>new | +0.746 [+0.401, +1.136] | +1.042 [-0.003, +2.157] |
| mass P(o1) + P(o2), assistant-none>new            | +0.614 [+0.515, +0.711] | +0.659 [+0.547, +0.765] |
| prior log P(1st) - log P(2nd), character-none>new | +0.708 [+0.336, +1.134] | +1.268 [+0.327, +2.305] |
| mass P(o1) + P(o2), character-none>new            | +0.602 [+0.499, +0.704] | +0.642 [+0.531, +0.750] |

## Readout tokenization (variant: tokens)

`' odd'` 1; `'odd'` 1; `' Odd'` 1; `'Odd'` 1; `' ODD'` 2; `'ODD'` 2; `' even'` 1; `'even'` 1; `' Even'` 1; `'Even'` 1; `' EVEN'` 1; `'EVEN'` 2; `' top'` 1; `'top'` 1; `' Top'` 1; `'Top'` 1; `' TOP'` 1; `'TOP'` 1; `' bottom'` 1; `'bottom'` 1; `' Bottom'` 1; `'Bottom'` 1; `' BOTTOM'` 1; `'BOTTOM'` 1; `' home'` 1; `'home'` 1; `' Home'` 1; `'Home'` 1; `' HOME'` 1; `'HOME'` 1; `' away'` 1; `'away'` 1; `' Away'` 1; `'Away'` 1; `' AWAY'` 2; `'AWAY'` 2; `' rain'` 1; `'rain'` 1; `' Rain'` 1; `'Rain'` 1; `' RAIN'` 2; `'RAIN'` 1; `' snow'` 1; `'snow'` 1; `' Snow'` 1; `'Snow'` 1; `' SNOW'` 2; `'SNOW'` 2
