# Persona carryover (overgeneralization)

The pilot item set is exploratory; the verdict comes from the confirmatory set (README). Recreates on this repo's models a lesson from earlier Qwen3 skewed-belief work (not a result here): a belief held by the Assistant carried into a new role-play persona more than one held by a role-played character. Design and readout: module docstring.

48 items (dice: 12, drawer: 12, match: 12, weather: 12), each with both outcomes believed. Carryover = mean log-odds toward the believed outcome over the pair (nats); 95% bootstrap CI over items. Prediction fixed in advance: leak gap > 0. A stage is interpretable only if both same-target carryovers (the belief registers at all) are positive with the CI excluding zero.

## Headline (all domains)

| stage    | assistant -> new        | assistant -> same       | character -> new        | character -> same       | leak gap                | change from base        | reading                |
|:---------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:-----------------------|
| base     | +0.596 [+0.427, +0.773] | +1.014 [+0.710, +1.310] | +0.340 [+0.136, +0.548] | +1.246 [+0.950, +1.534] | +0.256 [+0.203, +0.309] |                         | assistant carries more |
| instruct | +0.019 [-0.436, +0.483] | +2.171 [+1.385, +2.907] | -0.854 [-1.364, -0.340] | +1.493 [+0.655, +2.302] | +0.873 [+0.690, +1.070] | +0.617 [+0.432, +0.817] | assistant carries more |

## Pre-registered verdict (criteria 1-5, README; confirmatory item set only)

delta = 0.25 nats, domain effect = 0.5 nats; a bar is met when the mean is at least the bar and the 95% CI excludes zero. Domains are never pooled. Criterion 4 is checked in every domain that meets criterion 2.

| stage | domain | 1. interpretable | 2. leak-gap change >= 0.5 | 4. assistant -> new rises >= 0.25 |
|---|---|---|---|---|
| instruct | dice | yes | no | yes |
| instruct | drawer | no | no | no |
| instruct | match | yes | yes | no |
| instruct | weather | yes | yes | no |

| stage | interpretable domains | domains meeting 2 | 3. stage rule | 4. absolute | reading |
|---|---|---|---|---|---|
| instruct | 3 | 2 | yes | no | relative only (criterion 3 holds, criterion 4 does not) |

**Answer to "which stage" (criterion 5):** no stage meets criteria 3 and 4.

## By domain

### dice

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +0.733 [+0.652, +0.816] | +1.140 [+0.785, +1.546] |
| carryover assistant -> same                         | +1.308 [+1.189, +1.424] | +4.379 [+3.739, +5.048] |
| carryover character -> new                          | +0.579 [+0.496, +0.650] | +0.697 [+0.384, +1.041] |
| carryover character -> same                         | +1.315 [+1.185, +1.461] | +3.014 [+2.466, +3.554] |
| leak gap: assistant -> new minus character -> new   | +0.153 [+0.113, +0.196] | +0.442 [+0.265, +0.619] |
| same gap: assistant -> same minus character -> same | -0.007 [-0.106, +0.093] | +1.365 [+0.916, +1.838] |
| leak gap, change from base                          |                         | +0.289 [+0.110, +0.466] |

### drawer

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | -0.079 [-0.180, +0.026] | -1.687 [-2.442, -1.005] |
| carryover assistant -> same                         | -0.577 [-0.750, -0.387] | -1.310 [-2.577, -0.117] |
| carryover character -> new                          | -0.483 [-0.605, -0.350] | -2.494 [-3.278, -1.791] |
| carryover character -> same                         | -0.147 [-0.333, +0.051] | -2.390 [-3.163, -1.674] |
| leak gap: assistant -> new minus character -> new   | +0.404 [+0.346, +0.465] | +0.807 [+0.440, +1.239] |
| same gap: assistant -> same minus character -> same | -0.430 [-0.488, -0.372] | +1.081 [+0.430, +1.797] |
| leak gap, change from base                          |                         | +0.403 [+0.031, +0.841] |

### match

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +0.289 [+0.171, +0.409] | -0.731 [-1.173, -0.351] |
| carryover assistant -> same                         | +1.194 [+0.935, +1.483] | +1.810 [+1.043, +2.603] |
| carryover character -> new                          | -0.041 [-0.184, +0.103] | -2.101 [-2.348, -1.830] |
| carryover character -> same                         | +1.209 [+0.871, +1.561] | +0.558 [+0.040, +1.084] |
| leak gap: assistant -> new minus character -> new   | +0.330 [+0.235, +0.426] | +1.370 [+1.014, +1.732] |
| same gap: assistant -> same minus character -> same | -0.015 [-0.174, +0.123] | +1.252 [+0.731, +1.798] |
| leak gap, change from base                          |                         | +1.040 [+0.642, +1.461] |

### weather

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +1.442 [+1.228, +1.668] | +1.356 [+0.705, +2.064] |
| carryover assistant -> same                         | +2.132 [+1.843, +2.439] | +3.804 [+3.255, +4.304] |
| carryover character -> new                          | +1.304 [+1.044, +1.543] | +0.484 [-0.389, +1.369] |
| carryover character -> same                         | +2.606 [+2.368, +2.821] | +4.791 [+4.352, +5.252] |
| leak gap: assistant -> new minus character -> new   | +0.138 [+0.020, +0.260] | +0.872 [+0.557, +1.234] |
| same gap: assistant -> same minus character -> same | -0.473 [-0.777, -0.199] | -0.987 [-1.450, -0.537] |
| leak gap, change from base                          |                         | +0.734 [+0.424, +1.122] |

## Raw log-probs, priors and coverage (all domains)

### all conditions

| quantity                                          | base                    | instruct                |
|:--------------------------------------------------|:------------------------|:------------------------|
| raw log P(believed), assistant>new                | -1.019 [-1.140, -0.910] | -2.157 [-2.649, -1.734] |
| raw log P(other), assistant>new                   | -1.616 [-1.872, -1.382] | -2.176 [-2.618, -1.760] |
| mass P(o1) + P(o2), assistant>new                 | +0.660 [+0.597, +0.720] | +0.621 [+0.524, +0.714] |
| raw log P(believed), assistant>same               | -0.996 [-1.193, -0.808] | -1.686 [-2.357, -1.132] |
| raw log P(other), assistant>same                  | -2.010 [-2.384, -1.647] | -3.857 [-4.384, -3.353] |
| mass P(o1) + P(o2), assistant>same                | +0.705 [+0.620, +0.785] | +0.682 [+0.581, +0.781] |
| raw log P(believed), character>new                | -1.176 [-1.321, -1.042] | -3.147 [-3.791, -2.569] |
| raw log P(other), character>new                   | -1.516 [-1.827, -1.237] | -2.293 [-2.820, -1.764] |
| mass P(o1) + P(o2), character>new                 | +0.671 [+0.598, +0.742] | +0.587 [+0.473, +0.696] |
| raw log P(believed), character>same               | -0.734 [-0.874, -0.609] | -1.726 [-2.343, -1.206] |
| raw log P(other), character>same                  | -1.980 [-2.322, -1.656] | -3.219 [-3.831, -2.634] |
| mass P(o1) + P(o2), character>same                | +0.769 [+0.696, +0.835] | +0.689 [+0.588, +0.785] |
| prior log P(1st) - log P(2nd), assistant-none>new | +0.766 [+0.468, +1.099] | +0.871 [+0.103, +1.649] |
| mass P(o1) + P(o2), assistant-none>new            | +0.544 [+0.454, +0.632] | +0.601 [+0.501, +0.701] |
| prior log P(1st) - log P(2nd), character-none>new | +0.843 [+0.516, +1.211] | +0.736 [-0.249, +1.729] |
| mass P(o1) + P(o2), character-none>new            | +0.578 [+0.482, +0.674] | +0.543 [+0.442, +0.643] |

## Readout tokenization (variant: tokens)

`' odd'` 1; `'odd'` 1; `' Odd'` 1; `'Odd'` 1; `' ODD'` 2; `'ODD'` 2; `' even'` 1; `'even'` 1; `' Even'` 1; `'Even'` 1; `' EVEN'` 1; `'EVEN'` 2; `' top'` 1; `'top'` 1; `' Top'` 1; `'Top'` 1; `' TOP'` 1; `'TOP'` 1; `' bottom'` 1; `'bottom'` 1; `' Bottom'` 1; `'Bottom'` 1; `' BOTTOM'` 1; `'BOTTOM'` 1; `' home'` 1; `'home'` 1; `' Home'` 1; `'Home'` 1; `' HOME'` 1; `'HOME'` 1; `' away'` 1; `'away'` 1; `' Away'` 1; `'Away'` 1; `' AWAY'` 2; `'AWAY'` 2; `' rain'` 1; `'rain'` 1; `' Rain'` 1; `'Rain'` 1; `' RAIN'` 2; `'RAIN'` 1; `' snow'` 1; `'snow'` 1; `' Snow'` 1; `'Snow'` 1; `' SNOW'` 2; `'SNOW'` 2
