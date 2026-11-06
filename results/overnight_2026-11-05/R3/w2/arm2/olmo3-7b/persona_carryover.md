# Persona carryover (overgeneralization)

The pilot item set is exploratory; the verdict comes from the confirmatory set (README). Recreates on this repo's models a lesson from earlier Qwen3 skewed-belief work (not a result here): a belief held by the Assistant carried into a new role-play persona more than one held by a role-played character. Design and readout: module docstring.

48 items (dice: 12, drawer: 12, match: 12, weather: 12), each with both outcomes believed. Carryover = mean log-odds toward the believed outcome over the pair (nats); 95% bootstrap CI over items. Prediction fixed in advance: leak gap > 0. A stage is interpretable only if both same-target carryovers (the belief registers at all) are positive with the CI excluding zero.

## Headline (all domains)

| stage    | assistant -> new        | assistant -> same       | character -> new        | character -> same       | leak gap                | change from base        | reading                                                                      |
|:---------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:-----------------------------------------------------------------------------|
| base     | +0.269 [+0.057, +0.497] | +1.014 [+0.710, +1.310] | +0.060 [-0.152, +0.281] | +1.246 [+0.950, +1.534] | +0.209 [+0.164, +0.256] |                         | assistant carries more                                                       |
| instruct | -1.083 [-1.508, -0.660] | +2.171 [+1.385, +2.907] | -1.463 [-1.910, -1.012] | +1.493 [+0.655, +2.302] | +0.380 [+0.243, +0.523] | +0.172 [+0.032, +0.316] | assistant carries more (relative only: both new-persona carryovers negative) |

## Pre-registered verdict (criteria 1-5, README; confirmatory item set only)

delta = 0.25 nats, domain effect = 0.5 nats; a bar is met when the mean is at least the bar and the 95% CI excludes zero. Domains are never pooled. Criterion 4 is checked in every domain that meets criterion 2.

| stage | domain | 1. interpretable | 2. leak-gap change >= 0.5 | 4. assistant -> new rises >= 0.25 |
|---|---|---|---|---|
| instruct | dice | yes | no | no |
| instruct | drawer | no | no | no |
| instruct | match | yes | no | no |
| instruct | weather | yes | no | no |

| stage | interpretable domains | domains meeting 2 | 3. stage rule | 4. absolute | reading |
|---|---|---|---|---|---|
| instruct | 3 | 0 | no | no | no overgeneralization by the rule |

**Answer to "which stage" (criterion 5):** no stage meets criteria 3 and 4.

## By domain

### dice

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +0.411 [+0.320, +0.504] | -0.000 [-0.151, +0.151] |
| carryover assistant -> same                         | +1.308 [+1.189, +1.424] | +4.379 [+3.739, +5.048] |
| carryover character -> new                          | +0.277 [+0.168, +0.390] | -0.255 [-0.516, +0.005] |
| carryover character -> same                         | +1.315 [+1.185, +1.461] | +3.014 [+2.466, +3.554] |
| leak gap: assistant -> new minus character -> new   | +0.134 [+0.104, +0.165] | +0.255 [+0.098, +0.416] |
| same gap: assistant -> same minus character -> same | -0.007 [-0.106, +0.093] | +1.365 [+0.916, +1.838] |
| leak gap, change from base                          |                         | +0.121 [-0.042, +0.285] |

### drawer

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | -0.548 [-0.663, -0.447] | -2.991 [-3.441, -2.554] |
| carryover assistant -> same                         | -0.577 [-0.750, -0.387] | -1.310 [-2.577, -0.117] |
| carryover character -> new                          | -0.767 [-0.908, -0.630] | -3.257 [-3.754, -2.786] |
| carryover character -> same                         | -0.147 [-0.333, +0.051] | -2.390 [-3.163, -1.674] |
| leak gap: assistant -> new minus character -> new   | +0.219 [+0.141, +0.294] | +0.266 [-0.018, +0.542] |
| same gap: assistant -> same minus character -> same | -0.430 [-0.488, -0.372] | +1.081 [+0.430, +1.797] |
| leak gap, change from base                          |                         | +0.047 [-0.232, +0.319] |

### match

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | -0.176 [-0.270, -0.086] | -1.637 [-1.940, -1.351] |
| carryover assistant -> same                         | +1.194 [+0.935, +1.483] | +1.810 [+1.043, +2.603] |
| carryover character -> new                          | -0.372 [-0.482, -0.257] | -2.228 [-2.524, -1.923] |
| carryover character -> same                         | +1.209 [+0.871, +1.561] | +0.558 [+0.040, +1.084] |
| leak gap: assistant -> new minus character -> new   | +0.196 [+0.109, +0.288] | +0.591 [+0.326, +0.876] |
| same gap: assistant -> same minus character -> same | -0.015 [-0.174, +0.123] | +1.252 [+0.731, +1.798] |
| leak gap, change from base                          |                         | +0.395 [+0.094, +0.700] |

### weather

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +1.388 [+1.180, +1.623] | +0.296 [-0.222, +0.833] |
| carryover assistant -> same                         | +2.132 [+1.843, +2.439] | +3.804 [+3.255, +4.304] |
| carryover character -> new                          | +1.102 [+0.839, +1.366] | -0.112 [-0.857, +0.587] |
| carryover character -> same                         | +2.606 [+2.368, +2.821] | +4.791 [+4.352, +5.252] |
| leak gap: assistant -> new minus character -> new   | +0.286 [+0.165, +0.409] | +0.409 [+0.104, +0.734] |
| same gap: assistant -> same minus character -> same | -0.473 [-0.777, -0.199] | -0.987 [-1.450, -0.537] |
| leak gap, change from base                          |                         | +0.123 [-0.196, +0.471] |

## Raw log-probs, priors and coverage (all domains)

### all conditions

| quantity                                          | base                    | instruct                |
|:--------------------------------------------------|:------------------------|:------------------------|
| raw log P(believed), assistant>new                | -1.252 [-1.362, -1.148] | -2.501 [-2.958, -2.083] |
| raw log P(other), assistant>new                   | -1.521 [-1.763, -1.297] | -1.418 [-1.734, -1.127] |
| mass P(o1) + P(o2), assistant>new                 | +0.604 [+0.548, +0.658] | +0.615 [+0.529, +0.698] |
| raw log P(believed), assistant>same               | -0.996 [-1.193, -0.808] | -1.686 [-2.357, -1.132] |
| raw log P(other), assistant>same                  | -2.010 [-2.384, -1.647] | -3.857 [-4.384, -3.353] |
| mass P(o1) + P(o2), assistant>same                | +0.705 [+0.620, +0.785] | +0.682 [+0.581, +0.781] |
| raw log P(believed), character>new                | -1.358 [-1.475, -1.250] | -3.019 [-3.575, -2.534] |
| raw log P(other), character>new                   | -1.419 [-1.679, -1.180] | -1.556 [-1.934, -1.190] |
| mass P(o1) + P(o2), character>new                 | +0.612 [+0.552, +0.671] | +0.612 [+0.509, +0.710] |
| raw log P(believed), character>same               | -0.734 [-0.874, -0.609] | -1.726 [-2.343, -1.206] |
| raw log P(other), character>same                  | -1.980 [-2.322, -1.656] | -3.219 [-3.831, -2.634] |
| mass P(o1) + P(o2), character>same                | +0.769 [+0.696, +0.835] | +0.689 [+0.588, +0.785] |
| prior log P(1st) - log P(2nd), assistant-none>new | +0.782 [+0.556, +1.034] | +0.383 [-0.153, +0.979] |
| mass P(o1) + P(o2), assistant-none>new            | +0.476 [+0.404, +0.548] | +0.628 [+0.531, +0.724] |
| prior log P(1st) - log P(2nd), character-none>new | +0.766 [+0.520, +1.035] | +0.408 [-0.270, +1.120] |
| mass P(o1) + P(o2), character-none>new            | +0.506 [+0.433, +0.581] | +0.650 [+0.549, +0.748] |

## Readout tokenization (variant: tokens)

`' odd'` 1; `'odd'` 1; `' Odd'` 1; `'Odd'` 1; `' ODD'` 2; `'ODD'` 2; `' even'` 1; `'even'` 1; `' Even'` 1; `'Even'` 1; `' EVEN'` 1; `'EVEN'` 2; `' top'` 1; `'top'` 1; `' Top'` 1; `'Top'` 1; `' TOP'` 1; `'TOP'` 1; `' bottom'` 1; `'bottom'` 1; `' Bottom'` 1; `'Bottom'` 1; `' BOTTOM'` 1; `'BOTTOM'` 1; `' home'` 1; `'home'` 1; `' Home'` 1; `'Home'` 1; `' HOME'` 1; `'HOME'` 1; `' away'` 1; `'away'` 1; `' Away'` 1; `'Away'` 1; `' AWAY'` 2; `'AWAY'` 2; `' rain'` 1; `'rain'` 1; `' Rain'` 1; `'Rain'` 1; `' RAIN'` 2; `'RAIN'` 1; `' snow'` 1; `'snow'` 1; `' Snow'` 1; `'Snow'` 1; `' SNOW'` 2; `'SNOW'` 2
