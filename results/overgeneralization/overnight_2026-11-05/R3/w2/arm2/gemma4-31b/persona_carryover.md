# Persona carryover (overgeneralization)

**Family gemma4-31b: contamination could not be checked** (its post-training data is not public); breadth arm (README criterion 7). Final assistant turn opened with the template's empty thought channel: YES (--assistant-opening, as planned).

The pilot item set is exploratory; the verdict comes from the confirmatory set (README). Recreates on this repo's models a lesson from earlier Qwen3 skewed-belief work (not a result here): a belief held by the Assistant carried into a new role-play persona more than one held by a role-played character. Design and readout: module docstring.

48 items (dice: 12, drawer: 12, match: 12, weather: 12), each with both outcomes believed. Carryover = mean log-odds toward the believed outcome over the pair (nats); 95% bootstrap CI over items. Prediction fixed in advance: leak gap > 0. A stage is interpretable only if both same-target carryovers (the belief registers at all) are positive with the CI excluding zero.

## Headline (all domains)

| stage    | assistant -> new        | assistant -> same         | character -> new        | character -> same          | leak gap                | change from base        | reading                                                                      |
|:---------|:------------------------|:--------------------------|:------------------------|:---------------------------|:------------------------|:------------------------|:-----------------------------------------------------------------------------|
| base     | +0.151 [+0.002, +0.306] | +1.247 [+1.077, +1.420]   | -0.034 [-0.136, +0.073] | +1.314 [+1.111, +1.520]    | +0.185 [+0.107, +0.263] |                         | assistant carries more                                                       |
| instruct | -2.327 [-3.241, -1.256] | +11.017 [+9.972, +12.088] | -6.508 [-7.245, -5.786] | +15.088 [+14.178, +15.986] | +4.180 [+2.995, +5.583] | +3.996 [+2.759, +5.407] | assistant carries more (relative only: both new-persona carryovers negative) |

## Pre-registered verdict (criteria 1-5, README; confirmatory item set only)

delta = 0.25 nats, domain effect = 0.5 nats; a bar is met when the mean is at least the bar and the 95% CI excludes zero. Domains are never pooled. Criterion 4 is checked in every domain that meets criterion 2.

| stage | domain | 1. interpretable | 2. leak-gap change >= 0.5 | 4. assistant -> new rises >= 0.25 |
|---|---|---|---|---|
| instruct | dice | yes | yes | no |
| instruct | drawer | yes | yes | no |
| instruct | match | yes | yes | no |
| instruct | weather | yes | no | no |

| stage | interpretable domains | domains meeting 2 | 3. stage rule | 4. absolute | reading |
|---|---|---|---|---|---|
| instruct | 4 | 3 | yes | no | relative only (criterion 3 holds, criterion 4 does not) |

**Answer to "which stage" (criterion 5):** no stage meets criteria 3 and 4.

## By domain

### dice

| quantity                                            | base                    | instruct                   |
|:----------------------------------------------------|:------------------------|:---------------------------|
| carryover assistant -> new                          | +0.255 [+0.130, +0.385] | -2.651 [-3.844, -1.453]    |
| carryover assistant -> same                         | +1.962 [+1.743, +2.201] | +12.125 [+11.843, +12.385] |
| carryover character -> new                          | +0.052 [-0.089, +0.177] | -5.630 [-6.646, -4.417]    |
| carryover character -> same                         | +2.149 [+1.993, +2.295] | +13.919 [+13.419, +14.474] |
| leak gap: assistant -> new minus character -> new   | +0.203 [+0.120, +0.276] | +2.979 [+2.229, +3.766]    |
| same gap: assistant -> same minus character -> same | -0.187 [-0.359, -0.005] | -1.794 [-2.357, -1.245]    |
| leak gap, change from base                          |                         | +2.776 [+1.984, +3.610]    |

### drawer

| quantity                                            | base                    | instruct                   |
|:----------------------------------------------------|:------------------------|:---------------------------|
| carryover assistant -> new                          | -0.344 [-0.411, -0.265] | -3.958 [-4.719, -3.167]    |
| carryover assistant -> same                         | +0.500 [+0.411, +0.583] | +7.604 [+6.132, +8.932]    |
| carryover character -> new                          | -0.401 [-0.479, -0.323] | -7.187 [-8.068, -6.307]    |
| carryover character -> same                         | +0.562 [+0.416, +0.693] | +13.766 [+12.174, +15.284] |
| leak gap: assistant -> new minus character -> new   | +0.057 [-0.042, +0.151] | +3.229 [+2.526, +3.979]    |
| same gap: assistant -> same minus character -> same | -0.063 [-0.219, +0.083] | -6.162 [-7.336, -4.943]    |
| leak gap, change from base                          |                         | +3.172 [+2.505, +3.943]    |

### match

| quantity                                            | base                    | instruct                   |
|:----------------------------------------------------|:------------------------|:---------------------------|
| carryover assistant -> new                          | -0.225 [-0.370, -0.080] | +1.283 [-1.291, +3.986]    |
| carryover assistant -> same                         | +1.134 [+0.922, +1.347] | +15.696 [+14.600, +16.786] |
| carryover character -> new                          | -0.241 [-0.314, -0.158] | -8.825 [-10.306, -7.205]   |
| carryover character -> same                         | +0.863 [+0.692, +1.040] | +19.674 [+19.114, +20.331] |
| leak gap: assistant -> new minus character -> new   | +0.016 [-0.150, +0.182] | +10.108 [+7.385, +13.217]  |
| same gap: assistant -> same minus character -> same | +0.271 [+0.120, +0.432] | -3.978 [-5.114, -2.802]    |
| leak gap, change from base                          |                         | +10.093 [+7.457, +13.131]  |

### weather

| quantity                                            | base                    | instruct                   |
|:----------------------------------------------------|:------------------------|:---------------------------|
| carryover assistant -> new                          | +0.917 [+0.813, +1.026] | -3.984 [-4.710, -3.245]    |
| carryover assistant -> same                         | +1.391 [+1.224, +1.563] | +8.643 [+7.480, +9.939]    |
| carryover character -> new                          | +0.453 [+0.338, +0.557] | -4.389 [-5.154, -3.599]    |
| carryover character -> same                         | +1.682 [+1.375, +1.995] | +12.991 [+11.791, +14.120] |
| leak gap: assistant -> new minus character -> new   | +0.464 [+0.344, +0.578] | +0.405 [-0.217, +1.168]    |
| same gap: assistant -> same minus character -> same | -0.292 [-0.505, -0.068] | -4.348 [-5.387, -3.258]    |
| leak gap, change from base                          |                         | -0.059 [-0.698, +0.707]    |

## Raw log-probs, priors and coverage (all domains)

### all conditions

| quantity                                          | base                    | instruct                   |
|:--------------------------------------------------|:------------------------|:---------------------------|
| raw log P(believed), assistant>new                | -1.154 [-1.257, -1.051] | -4.184 [-5.002, -3.391]    |
| raw log P(other), assistant>new                   | -1.304 [-1.506, -1.115] | -1.857 [-2.596, -1.197]    |
| mass P(o1) + P(o2), assistant>new                 | +0.685 [+0.620, +0.749] | +0.759 [+0.649, +0.861]    |
| raw log P(believed), assistant>same               | -0.770 [-0.963, -0.595] | -3.439 [-4.440, -2.493]    |
| raw log P(other), assistant>same                  | -2.016 [-2.258, -1.773] | -14.457 [-15.473, -13.449] |
| mass P(o1) + P(o2), assistant>same                | +0.734 [+0.652, +0.808] | +0.346 [+0.241, +0.456]    |
| raw log P(believed), character>new                | -1.173 [-1.292, -1.061] | -7.847 [-8.547, -7.146]    |
| raw log P(other), character>new                   | -1.139 [-1.328, -0.965] | -1.339 [-2.091, -0.658]    |
| mass P(o1) + P(o2), character>new                 | +0.720 [+0.653, +0.785] | +0.765 [+0.646, +0.878]    |
| raw log P(believed), character>same               | -0.632 [-0.795, -0.494] | -0.549 [-0.952, -0.225]    |
| raw log P(other), character>same                  | -1.946 [-2.181, -1.717] | -15.637 [-16.468, -14.835] |
| mass P(o1) + P(o2), character>same                | +0.794 [+0.723, +0.856] | +0.828 [+0.730, +0.914]    |
| prior log P(1st) - log P(2nd), assistant-none>new | +0.539 [+0.235, +0.873] | +2.325 [+1.033, +3.680]    |
| mass P(o1) + P(o2), assistant-none>new            | +0.579 [+0.510, +0.651] | +0.872 [+0.800, +0.934]    |
| prior log P(1st) - log P(2nd), character-none>new | +0.459 [+0.217, +0.735] | +1.291 [+0.038, +2.658]    |
| mass P(o1) + P(o2), character-none>new            | +0.614 [+0.536, +0.691] | +0.824 [+0.727, +0.910]    |

## Readout tokenization (variant: tokens)

`' odd'` 1; `'odd'` 1; `' Odd'` 1; `'Odd'` 1; `' ODD'` 2; `'ODD'` 2; `' even'` 1; `'even'` 1; `' Even'` 1; `'Even'` 1; `' EVEN'` 1; `'EVEN'` 1; `' top'` 1; `'top'` 1; `' Top'` 1; `'Top'` 1; `' TOP'` 1; `'TOP'` 1; `' bottom'` 1; `'bottom'` 1; `' Bottom'` 1; `'Bottom'` 1; `' BOTTOM'` 1; `'BOTTOM'` 1; `' home'` 1; `'home'` 1; `' Home'` 1; `'Home'` 1; `' HOME'` 1; `'HOME'` 1; `' away'` 1; `'away'` 1; `' Away'` 1; `'Away'` 1; `' AWAY'` 1; `'AWAY'` 2; `' rain'` 1; `'rain'` 1; `' Rain'` 1; `'Rain'` 1; `' RAIN'` 2; `'RAIN'` 1; `' snow'` 1; `'snow'` 1; `' Snow'` 1; `'Snow'` 1; `' SNOW'` 2; `'SNOW'` 2
