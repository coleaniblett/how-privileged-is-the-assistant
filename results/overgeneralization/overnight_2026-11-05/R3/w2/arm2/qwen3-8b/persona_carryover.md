# Persona carryover (overgeneralization)

**Family qwen3-8b: contamination could not be checked** (its post-training data is not public); breadth arm (README criterion 7). Assistant-slot empty think block: ON (the pre-declared primary rendering, docs/decisions.md 2026-11-02).

The pilot item set is exploratory; the verdict comes from the confirmatory set (README). Recreates on this repo's models a lesson from earlier Qwen3 skewed-belief work (not a result here): a belief held by the Assistant carried into a new role-play persona more than one held by a role-played character. Design and readout: module docstring.

48 items (dice: 12, drawer: 12, match: 12, weather: 12), each with both outcomes believed. Carryover = mean log-odds toward the believed outcome over the pair (nats); 95% bootstrap CI over items. Prediction fixed in advance: leak gap > 0. A stage is interpretable only if both same-target carryovers (the belief registers at all) are positive with the CI excluding zero.

## Headline (all domains)

| stage    | assistant -> new        | assistant -> same       | character -> new        | character -> same       | leak gap                | change from base        | reading                |
|:---------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:-----------------------|
| base     | +0.127 [-0.001, +0.258] | +0.860 [+0.724, +1.001] | +0.032 [-0.111, +0.170] | +1.402 [+1.200, +1.610] | +0.095 [+0.044, +0.148] |                         | assistant carries more |
| instruct | +0.312 [+0.040, +0.613] | +6.738 [+6.103, +7.391] | -1.271 [-1.707, -0.842] | +8.133 [+7.456, +8.841] | +1.583 [+1.262, +1.918] | +1.488 [+1.169, +1.832] | assistant carries more |

## Pre-registered verdict (criteria 1-5, README; confirmatory item set only)

delta = 0.25 nats, domain effect = 0.5 nats; a bar is met when the mean is at least the bar and the 95% CI excludes zero. Domains are never pooled. Criterion 4 is checked in every domain that meets criterion 2.

| stage | domain | 1. interpretable | 2. leak-gap change >= 0.5 | 4. assistant -> new rises >= 0.25 |
|---|---|---|---|---|
| instruct | dice | yes | no | no |
| instruct | drawer | yes | yes | no |
| instruct | match | yes | yes | no |
| instruct | weather | yes | yes | yes |

| stage | interpretable domains | domains meeting 2 | 3. stage rule | 4. absolute | reading |
|---|---|---|---|---|---|
| instruct | 4 | 3 | yes | no | relative only (criterion 3 holds, criterion 4 does not) |

**Answer to "which stage" (criterion 5):** no stage meets criteria 3 and 4.

## By domain

### dice

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +0.318 [+0.234, +0.385] | +0.359 [+0.255, +0.469] |
| carryover assistant -> same                         | +0.979 [+0.844, +1.114] | +6.477 [+5.396, +7.445] |
| carryover character -> new                          | +0.281 [+0.188, +0.375] | -0.151 [-0.422, +0.073] |
| carryover character -> same                         | +0.958 [+0.875, +1.041] | +6.786 [+6.135, +7.458] |
| leak gap: assistant -> new minus character -> new   | +0.036 [-0.026, +0.099] | +0.510 [+0.302, +0.745] |
| same gap: assistant -> same minus character -> same | +0.021 [-0.062, +0.135] | -0.310 [-1.044, +0.445] |
| leak gap, change from base                          |                         | +0.474 [+0.240, +0.734] |

### drawer

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | -0.385 [-0.562, -0.208] | -0.516 [-0.859, -0.177] |
| carryover assistant -> same                         | +0.333 [+0.208, +0.490] | +5.297 [+4.042, +6.807] |
| carryover character -> new                          | -0.531 [-0.729, -0.323] | -2.234 [-2.839, -1.661] |
| carryover character -> same                         | +0.823 [+0.625, +1.062] | +6.661 [+5.510, +7.927] |
| leak gap: assistant -> new minus character -> new   | +0.146 [+0.021, +0.271] | +1.719 [+1.375, +2.062] |
| same gap: assistant -> same minus character -> same | -0.490 [-0.677, -0.312] | -1.365 [-2.271, -0.526] |
| leak gap, change from base                          |                         | +1.573 [+1.167, +1.984] |

### match

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +0.088 [-0.026, +0.208] | +0.130 [-0.162, +0.380] |
| carryover assistant -> same                         | +0.978 [+0.822, +1.165] | +6.922 [+5.755, +8.063] |
| carryover character -> new                          | +0.020 [-0.104, +0.166] | -1.354 [-2.141, -0.703] |
| carryover character -> same                         | +1.364 [+1.114, +1.614] | +8.318 [+7.271, +9.427] |
| leak gap: assistant -> new minus character -> new   | +0.068 [-0.026, +0.146] | +1.484 [+0.958, +2.062] |
| same gap: assistant -> same minus character -> same | -0.386 [-0.604, -0.156] | -1.396 [-2.135, -0.568] |
| leak gap, change from base                          |                         | +1.416 [+0.921, +2.010] |

### weather

| quantity                                            | base                    | instruct                  |
|:----------------------------------------------------|:------------------------|:--------------------------|
| carryover assistant -> new                          | +0.489 [+0.203, +0.797] | +1.276 [+0.536, +2.042]   |
| carryover assistant -> same                         | +1.151 [+0.823, +1.479] | +8.255 [+7.250, +9.281]   |
| carryover character -> new                          | +0.359 [+0.057, +0.677] | -1.344 [-2.521, -0.214]   |
| carryover character -> same                         | +2.463 [+2.271, +2.646] | +10.766 [+9.812, +11.880] |
| leak gap: assistant -> new minus character -> new   | +0.130 [+0.016, +0.240] | +2.620 [+1.901, +3.349]   |
| same gap: assistant -> same minus character -> same | -1.312 [-1.573, -1.052] | -2.510 [-3.224, -1.792]   |
| leak gap, change from base                          |                         | +2.490 [+1.771, +3.245]   |

## Raw log-probs, priors and coverage (all domains)

### all conditions

| quantity                                          | base                    | instruct                |
|:--------------------------------------------------|:------------------------|:------------------------|
| raw log P(believed), assistant>new                | -1.402 [-1.568, -1.252] | -1.204 [-1.426, -1.002] |
| raw log P(other), assistant>new                   | -1.530 [-1.741, -1.332] | -1.517 [-1.888, -1.186] |
| mass P(o1) + P(o2), assistant>new                 | +0.590 [+0.520, +0.659] | +0.760 [+0.681, +0.834] |
| raw log P(believed), assistant>same               | -0.907 [-1.073, -0.749] | -1.322 [-1.789, -0.907] |
| raw log P(other), assistant>same                  | -1.767 [-2.009, -1.546] | -8.060 [-8.823, -7.342] |
| mass P(o1) + P(o2), assistant>same                | +0.716 [+0.634, +0.790] | +0.521 [+0.434, +0.605] |
| raw log P(believed), character>new                | -1.347 [-1.505, -1.200] | -2.490 [-2.986, -2.022] |
| raw log P(other), character>new                   | -1.379 [-1.597, -1.189] | -1.219 [-1.671, -0.822] |
| mass P(o1) + P(o2), character>new                 | +0.637 [+0.571, +0.700] | +0.753 [+0.658, +0.842] |
| raw log P(believed), character>same               | -0.635 [-0.753, -0.533] | -0.318 [-0.543, -0.136] |
| raw log P(other), character>same                  | -2.037 [-2.326, -1.764] | -8.450 [-9.247, -7.691] |
| mass P(o1) + P(o2), character>same                | +0.794 [+0.721, +0.861] | +0.868 [+0.795, +0.930] |
| prior log P(1st) - log P(2nd), assistant-none>new | +0.977 [+0.680, +1.258] | +1.083 [+0.515, +1.690] |
| mass P(o1) + P(o2), assistant-none>new            | +0.439 [+0.370, +0.511] | +0.538 [+0.440, +0.641] |
| prior log P(1st) - log P(2nd), character-none>new | +0.855 [+0.579, +1.149] | +0.661 [-0.044, +1.414] |
| mass P(o1) + P(o2), character-none>new            | +0.474 [+0.407, +0.544] | +0.586 [+0.477, +0.693] |

## Readout tokenization (variant: tokens)

`' odd'` 1; `'odd'` 1; `' Odd'` 1; `'Odd'` 1; `' ODD'` 2; `'ODD'` 2; `' even'` 1; `'even'` 1; `' Even'` 1; `'Even'` 1; `' EVEN'` 1; `'EVEN'` 2; `' top'` 1; `'top'` 1; `' Top'` 1; `'Top'` 1; `' TOP'` 1; `'TOP'` 1; `' bottom'` 1; `'bottom'` 1; `' Bottom'` 1; `'Bottom'` 1; `' BOTTOM'` 1; `'BOTTOM'` 1; `' home'` 1; `'home'` 1; `' Home'` 1; `'Home'` 1; `' HOME'` 1; `'HOME'` 1; `' away'` 1; `'away'` 1; `' Away'` 1; `'Away'` 1; `' AWAY'` 2; `'AWAY'` 2; `' rain'` 1; `'rain'` 1; `' Rain'` 1; `'Rain'` 1; `' RAIN'` 2; `'RAIN'` 1; `' snow'` 1; `'snow'` 1; `' Snow'` 1; `'Snow'` 1; `' SNOW'` 2; `'SNOW'` 2
