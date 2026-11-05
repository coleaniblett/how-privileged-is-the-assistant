# Persona carryover (overgeneralization)

**Family gemma4-31b: contamination could not be checked** (its post-training data is not public); breadth arm (README criterion 7). Final assistant turn opened with the template's empty thought channel: YES (--assistant-opening, as planned).

The pilot item set is exploratory; the verdict comes from the confirmatory set (README). Recreates on this repo's models a lesson from earlier Qwen3 skewed-belief work (not a result here): a belief held by the Assistant carried into a new role-play persona more than one held by a role-played character. Design and readout: module docstring.

48 items (dice: 12, drawer: 12, match: 12, weather: 12), each with both outcomes believed. Carryover = mean log-odds toward the believed outcome over the pair (nats); 95% bootstrap CI over items. Prediction fixed in advance: leak gap > 0. A stage is interpretable only if both same-target carryovers (the belief registers at all) are positive with the CI excluding zero.

## Headline (all domains)

| stage    | assistant -> new        | assistant -> same         | character -> new        | character -> same          | leak gap                | change from base        | reading                |
|:---------|:------------------------|:--------------------------|:------------------------|:---------------------------|:------------------------|:------------------------|:-----------------------|
| base     | +0.700 [+0.585, +0.821] | +1.247 [+1.077, +1.420]   | +0.234 [+0.138, +0.333] | +1.314 [+1.111, +1.520]    | +0.466 [+0.402, +0.528] |                         | assistant carries more |
| instruct | +3.692 [+2.480, +4.849] | +11.017 [+9.972, +12.088] | -4.737 [-5.717, -3.738] | +15.088 [+14.178, +15.986] | +8.429 [+7.039, +9.898] | +7.963 [+6.545, +9.421] | assistant carries more |

## Pre-registered verdict (criteria 1-5, README; confirmatory item set only)

delta = 0.25 nats, domain effect = 0.5 nats; a bar is met when the mean is at least the bar and the 95% CI excludes zero. Domains are never pooled. Criterion 4 is checked in every domain that meets criterion 2.

| stage | domain | 1. interpretable | 2. leak-gap change >= 0.5 | 4. assistant -> new rises >= 0.25 |
|---|---|---|---|---|
| instruct | dice | yes | yes | no |
| instruct | drawer | yes | yes | yes |
| instruct | match | yes | yes | yes |
| instruct | weather | yes | yes | yes |

| stage | interpretable domains | domains meeting 2 | 3. stage rule | 4. absolute | reading |
|---|---|---|---|---|---|
| instruct | 4 | 4 | yes | no | relative only (criterion 3 holds, criterion 4 does not) |

**Answer to "which stage" (criterion 5):** no stage meets criteria 3 and 4.

## By domain

### dice

| quantity                                            | base                    | instruct                   |
|:----------------------------------------------------|:------------------------|:---------------------------|
| carryover assistant -> new                          | +0.609 [+0.499, +0.718] | -1.010 [-2.635, +0.630]    |
| carryover assistant -> same                         | +1.962 [+1.743, +2.201] | +12.125 [+11.843, +12.385] |
| carryover character -> new                          | +0.104 [-0.042, +0.255] | -4.531 [-6.161, -2.625]    |
| carryover character -> same                         | +2.149 [+1.993, +2.295] | +13.919 [+13.419, +14.474] |
| leak gap: assistant -> new minus character -> new   | +0.504 [+0.411, +0.609] | +3.521 [+2.328, +4.656]    |
| same gap: assistant -> same minus character -> same | -0.187 [-0.359, -0.005] | -1.794 [-2.357, -1.245]    |
| leak gap, change from base                          |                         | +3.016 [+1.829, +4.136]    |

### drawer

| quantity                                            | base                    | instruct                   |
|:----------------------------------------------------|:------------------------|:---------------------------|
| carryover assistant -> new                          | +0.313 [+0.240, +0.385] | +5.052 [+3.510, +6.552]    |
| carryover assistant -> same                         | +0.500 [+0.411, +0.583] | +7.604 [+6.132, +8.932]    |
| carryover character -> new                          | +0.031 [-0.026, +0.089] | -3.854 [-5.448, -2.245]    |
| carryover character -> same                         | +0.562 [+0.416, +0.693] | +13.766 [+12.174, +15.284] |
| leak gap: assistant -> new minus character -> new   | +0.281 [+0.198, +0.354] | +8.906 [+7.651, +10.396]   |
| same gap: assistant -> same minus character -> same | -0.063 [-0.219, +0.083] | -6.162 [-7.336, -4.943]    |
| leak gap, change from base                          |                         | +8.625 [+7.380, +10.094]   |

### match

| quantity                                            | base                    | instruct                   |
|:----------------------------------------------------|:------------------------|:---------------------------|
| carryover assistant -> new                          | +0.577 [+0.431, +0.712] | +7.176 [+5.275, +8.994]    |
| carryover assistant -> same                         | +1.134 [+0.922, +1.347] | +15.696 [+14.600, +16.786] |
| carryover character -> new                          | +0.062 [-0.052, +0.197] | -7.310 [-9.362, -4.919]    |
| carryover character -> same                         | +0.863 [+0.692, +1.040] | +19.674 [+19.114, +20.331] |
| leak gap: assistant -> new minus character -> new   | +0.515 [+0.385, +0.640] | +14.486 [+11.556, +17.062] |
| same gap: assistant -> same minus character -> same | +0.271 [+0.120, +0.432] | -3.978 [-5.114, -2.802]    |
| leak gap, change from base                          |                         | +13.971 [+10.979, +16.437] |

### weather

| quantity                                            | base                    | instruct                   |
|:----------------------------------------------------|:------------------------|:---------------------------|
| carryover assistant -> new                          | +1.302 [+1.177, +1.432] | +3.551 [+2.060, +5.011]    |
| carryover assistant -> same                         | +1.391 [+1.224, +1.563] | +8.643 [+7.480, +9.939]    |
| carryover character -> new                          | +0.740 [+0.661, +0.807] | -3.251 [-4.531, -1.906]    |
| carryover character -> same                         | +1.682 [+1.375, +1.995] | +12.991 [+11.791, +14.120] |
| leak gap: assistant -> new minus character -> new   | +0.563 [+0.432, +0.682] | +6.802 [+5.478, +8.132]    |
| same gap: assistant -> same minus character -> same | -0.292 [-0.505, -0.068] | -4.348 [-5.387, -3.258]    |
| leak gap, change from base                          |                         | +6.240 [+4.928, +7.510]    |

## Raw log-probs, priors and coverage (all domains)

### all conditions

| quantity                                          | base                    | instruct                   |
|:--------------------------------------------------|:------------------------|:---------------------------|
| raw log P(believed), assistant>new                | -0.841 [-0.936, -0.753] | -1.365 [-1.865, -0.928]    |
| raw log P(other), assistant>new                   | -1.541 [-1.734, -1.363] | -5.057 [-5.965, -4.143]    |
| mass P(o1) + P(o2), assistant>new                 | +0.732 [+0.674, +0.786] | +0.735 [+0.639, +0.825]    |
| raw log P(believed), assistant>same               | -0.770 [-0.963, -0.595] | -3.439 [-4.440, -2.493]    |
| raw log P(other), assistant>same                  | -2.016 [-2.258, -1.773] | -14.457 [-15.473, -13.449] |
| mass P(o1) + P(o2), assistant>same                | +0.734 [+0.652, +0.808] | +0.346 [+0.241, +0.456]    |
| raw log P(believed), character>new                | -1.015 [-1.141, -0.900] | -6.653 [-7.655, -5.620]    |
| raw log P(other), character>new                   | -1.249 [-1.458, -1.064] | -1.916 [-2.839, -1.065]    |
| mass P(o1) + P(o2), character>new                 | +0.748 [+0.679, +0.813] | +0.765 [+0.645, +0.878]    |
| raw log P(believed), character>same               | -0.632 [-0.795, -0.494] | -0.549 [-0.952, -0.225]    |
| raw log P(other), character>same                  | -1.946 [-2.181, -1.717] | -15.637 [-16.468, -14.835] |
| mass P(o1) + P(o2), character>same                | +0.794 [+0.723, +0.856] | +0.828 [+0.730, +0.914]    |
| prior log P(1st) - log P(2nd), assistant-none>new | +0.674 [+0.359, +1.015] | +3.176 [+1.759, +4.609]    |
| mass P(o1) + P(o2), assistant-none>new            | +0.627 [+0.547, +0.705] | +0.704 [+0.593, +0.807]    |
| prior log P(1st) - log P(2nd), character-none>new | +0.693 [+0.433, +0.977] | +2.804 [+1.268, +4.421]    |
| mass P(o1) + P(o2), character-none>new            | +0.638 [+0.544, +0.732] | +0.767 [+0.651, +0.878]    |

## Readout tokenization (variant: tokens)

`' odd'` 1; `'odd'` 1; `' Odd'` 1; `'Odd'` 1; `' ODD'` 2; `'ODD'` 2; `' even'` 1; `'even'` 1; `' Even'` 1; `'Even'` 1; `' EVEN'` 1; `'EVEN'` 1; `' top'` 1; `'top'` 1; `' Top'` 1; `'Top'` 1; `' TOP'` 1; `'TOP'` 1; `' bottom'` 1; `'bottom'` 1; `' Bottom'` 1; `'Bottom'` 1; `' BOTTOM'` 1; `'BOTTOM'` 1; `' home'` 1; `'home'` 1; `' Home'` 1; `'Home'` 1; `' HOME'` 1; `'HOME'` 1; `' away'` 1; `'away'` 1; `' Away'` 1; `'Away'` 1; `' AWAY'` 1; `'AWAY'` 2; `' rain'` 1; `'rain'` 1; `' Rain'` 1; `'Rain'` 1; `' RAIN'` 2; `'RAIN'` 1; `' snow'` 1; `'snow'` 1; `' Snow'` 1; `'Snow'` 1; `' SNOW'` 2; `'SNOW'` 2
