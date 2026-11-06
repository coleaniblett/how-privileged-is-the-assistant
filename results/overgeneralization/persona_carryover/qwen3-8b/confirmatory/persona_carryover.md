# Persona carryover (overgeneralization)

**Family qwen3-8b: contamination could not be checked** (its post-training data is not public); breadth arm (README criterion 7). Assistant-slot empty think block: ON (the pre-declared primary rendering, docs/decisions.md 2026-11-02).

The pilot item set is exploratory; the verdict comes from the confirmatory set (README). Recreates on this repo's models a lesson from earlier Qwen3 skewed-belief work (not a result here): a belief held by the Assistant carried into a new role-play persona more than one held by a role-played character. Design and readout: module docstring.

48 items (dice: 12, drawer: 12, match: 12, weather: 12), each with both outcomes believed. Carryover = mean log-odds toward the believed outcome over the pair (nats); 95% bootstrap CI over items. Prediction fixed in advance: leak gap > 0. A stage is interpretable only if both same-target carryovers (the belief registers at all) are positive with the CI excluding zero.

## Headline (all domains)

| stage    | assistant -> new        | assistant -> same       | character -> new        | character -> same       | leak gap                | change from base        | reading                                                                      |
|:---------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:-----------------------------------------------------------------------------|
| base     | +0.176 [+0.022, +0.340] | +0.860 [+0.724, +1.001] | -0.180 [-0.398, +0.042] | +1.652 [+1.474, +1.836] | +0.355 [+0.250, +0.465] |                         | assistant carries more                                                       |
| instruct | -0.776 [-1.488, -0.091] | +6.738 [+6.103, +7.391] | -2.339 [-3.121, -1.581] | +6.776 [+6.253, +7.314] | +1.562 [+1.034, +2.100] | +1.207 [+0.720, +1.686] | assistant carries more (relative only: both new-persona carryovers negative) |

## Pre-registered verdict (criteria 1-5, README; confirmatory item set only)

delta = 0.25 nats, domain effect = 0.5 nats; a bar is met when the mean is at least the bar and the 95% CI excludes zero. Domains are never pooled. Criterion 4 is checked in every domain that meets criterion 2.

| stage | domain | 1. interpretable | 2. leak-gap change >= 0.5 | 4. assistant -> new rises >= 0.25 |
|---|---|---|---|---|
| instruct | dice | yes | yes | yes |
| instruct | drawer | yes | yes | no |
| instruct | match | yes | yes | no |
| instruct | weather | yes | no | no |

| stage | interpretable domains | domains meeting 2 | 3. stage rule | 4. absolute | reading |
|---|---|---|---|---|---|
| instruct | 4 | 3 | yes | no | relative only (criterion 3 holds, criterion 4 does not) |

**Answer to "which stage" (criterion 5):** no stage meets criteria 3 and 4.

## By domain

### dice

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +0.302 [+0.208, +0.385] | +0.964 [+0.547, +1.349] |
| carryover assistant -> same                         | +0.979 [+0.844, +1.114] | +6.477 [+5.396, +7.445] |
| carryover character -> new                          | +0.062 [-0.010, +0.135] | -0.125 [-0.542, +0.292] |
| carryover character -> same                         | +1.156 [+1.010, +1.302] | +5.755 [+5.062, +6.437] |
| leak gap: assistant -> new minus character -> new   | +0.240 [+0.146, +0.333] | +1.089 [+0.687, +1.484] |
| same gap: assistant -> same minus character -> same | -0.177 [-0.281, -0.063] | +0.721 [-0.406, +1.724] |
| leak gap, change from base                          |                         | +0.849 [+0.443, +1.229] |

### drawer

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | -0.281 [-0.448, -0.115] | -2.833 [-4.396, -1.396] |
| carryover assistant -> same                         | +0.333 [+0.208, +0.490] | +5.297 [+4.042, +6.807] |
| carryover character -> new                          | -1.021 [-1.260, -0.802] | -5.427 [-6.917, -4.031] |
| carryover character -> same                         | +1.250 [+1.115, +1.375] | +7.224 [+5.911, +8.594] |
| leak gap: assistant -> new minus character -> new   | +0.740 [+0.531, +0.938] | +2.594 [+1.229, +4.094] |
| same gap: assistant -> same minus character -> same | -0.917 [-1.073, -0.729] | -1.927 [-3.880, +0.235] |
| leak gap, change from base                          |                         | +1.854 [+0.479, +3.271] |

### match

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | -0.177 [-0.250, -0.094] | -0.729 [-1.458, -0.026] |
| carryover assistant -> same                         | +0.978 [+0.822, +1.165] | +6.922 [+5.755, +8.063] |
| carryover character -> new                          | -0.542 [-0.729, -0.354] | -2.536 [-3.453, -1.578] |
| carryover character -> same                         | +1.582 [+1.374, +1.780] | +6.000 [+5.401, +6.568] |
| leak gap: assistant -> new minus character -> new   | +0.365 [+0.188, +0.531] | +1.807 [+0.995, +2.615] |
| same gap: assistant -> same minus character -> same | -0.604 [-0.792, -0.417] | +0.922 [-0.021, +1.886] |
| leak gap, change from base                          |                         | +1.443 [+0.693, +2.213] |

### weather

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +0.859 [+0.552, +1.167] | -0.505 [-2.177, +1.010] |
| carryover assistant -> same                         | +1.151 [+0.823, +1.479] | +8.255 [+7.250, +9.281] |
| carryover character -> new                          | +0.781 [+0.510, +1.083] | -1.266 [-2.385, -0.130] |
| carryover character -> same                         | +2.620 [+2.453, +2.792] | +8.125 [+7.214, +8.964] |
| leak gap: assistant -> new minus character -> new   | +0.078 [-0.099, +0.245] | +0.760 [-0.375, +1.661] |
| same gap: assistant -> same minus character -> same | -1.469 [-1.812, -1.135] | +0.130 [-0.828, +1.135] |
| leak gap, change from base                          |                         | +0.682 [-0.313, +1.469] |

## Raw log-probs, priors and coverage (all domains)

### all conditions

| quantity                                          | base                    | instruct                |
|:--------------------------------------------------|:------------------------|:------------------------|
| raw log P(believed), assistant>new                | -1.214 [-1.390, -1.051] | -4.288 [-5.564, -3.100] |
| raw log P(other), assistant>new                   | -1.390 [-1.688, -1.122] | -3.512 [-4.777, -2.362] |
| mass P(o1) + P(o2), assistant>new                 | +0.735 [+0.649, +0.817] | +0.730 [+0.611, +0.842] |
| raw log P(believed), assistant>same               | -0.907 [-1.073, -0.749] | -1.322 [-1.789, -0.907] |
| raw log P(other), assistant>same                  | -1.767 [-2.009, -1.546] | -8.060 [-8.823, -7.342] |
| mass P(o1) + P(o2), assistant>same                | +0.716 [+0.634, +0.790] | +0.521 [+0.434, +0.605] |
| raw log P(believed), character>new                | -1.456 [-1.626, -1.290] | -6.003 [-7.460, -4.574] |
| raw log P(other), character>new                   | -1.276 [-1.601, -0.990] | -3.664 [-5.209, -2.241] |
| mass P(o1) + P(o2), character>new                 | +0.708 [+0.626, +0.787] | +0.667 [+0.539, +0.790] |
| raw log P(believed), character>same               | -0.603 [-0.727, -0.495] | -0.714 [-1.106, -0.378] |
| raw log P(other), character>same                  | -2.255 [-2.537, -1.988] | -7.490 [-8.158, -6.837] |
| mass P(o1) + P(o2), character>same                | +0.766 [+0.693, +0.834] | +0.786 [+0.685, +0.876] |
| prior log P(1st) - log P(2nd), assistant-none>new | +1.108 [+0.765, +1.462] | +3.643 [+2.271, +4.956] |
| mass P(o1) + P(o2), assistant-none>new            | +0.628 [+0.526, +0.726] | +0.611 [+0.478, +0.741] |
| prior log P(1st) - log P(2nd), character-none>new | +1.062 [+0.689, +1.476] | +2.556 [+1.117, +3.885] |
| mass P(o1) + P(o2), character-none>new            | +0.626 [+0.525, +0.725] | +0.601 [+0.475, +0.727] |

## Readout tokenization (variant: tokens)

`' odd'` 1; `'odd'` 1; `' Odd'` 1; `'Odd'` 1; `' ODD'` 2; `'ODD'` 2; `' even'` 1; `'even'` 1; `' Even'` 1; `'Even'` 1; `' EVEN'` 1; `'EVEN'` 2; `' top'` 1; `'top'` 1; `' Top'` 1; `'Top'` 1; `' TOP'` 1; `'TOP'` 1; `' bottom'` 1; `'bottom'` 1; `' Bottom'` 1; `'Bottom'` 1; `' BOTTOM'` 1; `'BOTTOM'` 1; `' home'` 1; `'home'` 1; `' Home'` 1; `'Home'` 1; `' HOME'` 1; `'HOME'` 1; `' away'` 1; `'away'` 1; `' Away'` 1; `'Away'` 1; `' AWAY'` 2; `'AWAY'` 2; `' rain'` 1; `'rain'` 1; `' Rain'` 1; `'Rain'` 1; `' RAIN'` 2; `'RAIN'` 1; `' snow'` 1; `'snow'` 1; `' Snow'` 1; `'Snow'` 1; `' SNOW'` 2; `'SNOW'` 2
