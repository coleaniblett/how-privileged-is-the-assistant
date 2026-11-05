# Persona carryover (overgeneralization)

**Family apertus-8b: contamination could not be checked** (its post-training data is not public); breadth arm (README criterion 7). Assistant-slot empty think block: OFF.

The pilot item set is exploratory; the verdict comes from the confirmatory set (README). Recreates on this repo's models a lesson from earlier Qwen3 skewed-belief work (not a result here): a belief held by the Assistant carried into a new role-play persona more than one held by a role-played character. Design and readout: module docstring.

48 items (dice: 12, drawer: 12, match: 12, weather: 12), each with both outcomes believed. Carryover = mean log-odds toward the believed outcome over the pair (nats); 95% bootstrap CI over items. Prediction fixed in advance: leak gap > 0. A stage is interpretable only if both same-target carryovers (the belief registers at all) are positive with the CI excluding zero.

## Headline (all domains)

| stage    | assistant -> new        | assistant -> same       | character -> new        | character -> same       | leak gap                | change from base        | reading                |
|:---------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:-----------------------|
| base     | +0.506 [+0.346, +0.681] | +0.857 [+0.723, +0.996] | +0.434 [+0.292, +0.590] | +1.262 [+1.052, +1.492] | +0.072 [+0.019, +0.129] |                         | assistant carries more |
| instruct | +0.281 [+0.064, +0.512] | +1.777 [+1.563, +1.999] | -0.295 [-0.471, -0.115] | +2.262 [+1.962, +2.578] | +0.576 [+0.431, +0.726] | +0.504 [+0.377, +0.637] | assistant carries more |

## Pre-registered verdict (criteria 1-5, README; confirmatory item set only)

delta = 0.25 nats, domain effect = 0.5 nats; a bar is met when the mean is at least the bar and the 95% CI excludes zero. Domains are never pooled. Criterion 4 is checked in every domain that meets criterion 2.

| stage | domain | 1. interpretable | 2. leak-gap change >= 0.5 | 4. assistant -> new rises >= 0.25 |
|---|---|---|---|---|
| instruct | dice | yes | no | no |
| instruct | drawer | yes | yes | no |
| instruct | match | yes | no | no |
| instruct | weather | yes | yes | no |

| stage | interpretable domains | domains meeting 2 | 3. stage rule | 4. absolute | reading |
|---|---|---|---|---|---|
| instruct | 4 | 2 | no | no | no overgeneralization by the rule |

**Answer to "which stage" (criterion 5):** no stage meets criteria 3 and 4.

## By domain

### dice

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +0.354 [+0.260, +0.458] | +0.177 [+0.062, +0.302] |
| carryover assistant -> same                         | +0.625 [+0.469, +0.802] | +1.192 [+0.984, +1.416] |
| carryover character -> new                          | +0.333 [+0.239, +0.437] | -0.026 [-0.141, +0.088] |
| carryover character -> same                         | +0.750 [+0.562, +0.948] | +0.989 [+0.838, +1.140] |
| leak gap: assistant -> new minus character -> new   | +0.021 [-0.031, +0.073] | +0.203 [+0.156, +0.255] |
| same gap: assistant -> same minus character -> same | -0.125 [-0.260, +0.010] | +0.203 [+0.083, +0.323] |
| leak gap, change from base                          |                         | +0.182 [+0.125, +0.239] |

### drawer

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +0.151 [+0.068, +0.260] | +0.151 [-0.130, +0.458] |
| carryover assistant -> same                         | +0.516 [+0.370, +0.672] | +1.370 [+1.052, +1.708] |
| carryover character -> new                          | +0.146 [+0.052, +0.250] | -0.401 [-0.781, +0.073] |
| carryover character -> same                         | +0.698 [+0.521, +0.885] | +2.187 [+1.974, +2.401] |
| leak gap: assistant -> new minus character -> new   | +0.005 [-0.073, +0.078] | +0.552 [+0.229, +0.880] |
| same gap: assistant -> same minus character -> same | -0.182 [-0.307, -0.078] | -0.818 [-1.010, -0.620] |
| leak gap, change from base                          |                         | +0.547 [+0.234, +0.870] |

### match

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +0.071 [-0.041, +0.169] | -0.448 [-0.719, -0.182] |
| carryover assistant -> same                         | +0.735 [+0.601, +0.861] | +2.113 [+1.696, +2.587] |
| carryover character -> new                          | +0.049 [-0.037, +0.130] | -0.838 [-1.016, -0.651] |
| carryover character -> same                         | +1.215 [+1.051, +1.399] | +2.110 [+1.873, +2.353] |
| leak gap: assistant -> new minus character -> new   | +0.022 [-0.059, +0.096] | +0.391 [+0.240, +0.531] |
| same gap: assistant -> same minus character -> same | -0.480 [-0.576, -0.394] | +0.003 [-0.273, +0.281] |
| leak gap, change from base                          |                         | +0.369 [+0.225, +0.485] |

### weather

| quantity                                            | base                    | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +1.448 [+1.271, +1.656] | +1.245 [+0.828, +1.630] |
| carryover assistant -> same                         | +1.552 [+1.406, +1.708] | +2.432 [+2.151, +2.708] |
| carryover character -> new                          | +1.208 [+0.969, +1.448] | +0.086 [-0.245, +0.404] |
| carryover character -> same                         | +2.385 [+2.073, +2.698] | +3.763 [+3.396, +4.143] |
| leak gap: assistant -> new minus character -> new   | +0.240 [+0.094, +0.385] | +1.159 [+0.953, +1.378] |
| same gap: assistant -> same minus character -> same | -0.833 [-1.115, -0.583] | -1.331 [-1.719, -0.984] |
| leak gap, change from base                          |                         | +0.919 [+0.682, +1.151] |

## Raw log-probs, priors and coverage (all domains)

### all conditions

| quantity                                          | base                    | instruct                |
|:--------------------------------------------------|:------------------------|:------------------------|
| raw log P(believed), assistant>new                | -0.970 [-1.106, -0.843] | -1.566 [-1.941, -1.223] |
| raw log P(other), assistant>new                   | -1.476 [-1.769, -1.212] | -1.847 [-2.403, -1.347] |
| mass P(o1) + P(o2), assistant>new                 | +0.761 [+0.681, +0.837] | +0.738 [+0.628, +0.840] |
| raw log P(believed), assistant>same               | -0.833 [-0.969, -0.708] | -0.616 [-0.775, -0.476] |
| raw log P(other), assistant>same                  | -1.690 [-1.937, -1.455] | -2.393 [-2.688, -2.119] |
| mass P(o1) + P(o2), assistant>same                | +0.752 [+0.679, +0.820] | +0.768 [+0.689, +0.845] |
| raw log P(believed), character>new                | -1.067 [-1.247, -0.899] | -2.156 [-2.706, -1.640] |
| raw log P(other), character>new                   | -1.501 [-1.815, -1.212] | -1.861 [-2.482, -1.267] |
| mass P(o1) + P(o2), character>new                 | +0.752 [+0.667, +0.834] | +0.724 [+0.607, +0.835] |
| raw log P(believed), character>same               | -0.660 [-0.772, -0.561] | -0.577 [-0.756, -0.423] |
| raw log P(other), character>same                  | -1.922 [-2.212, -1.650] | -2.839 [-3.269, -2.434] |
| mass P(o1) + P(o2), character>same                | +0.787 [+0.717, +0.851] | +0.786 [+0.697, +0.865] |
| prior log P(1st) - log P(2nd), assistant-none>new | +0.930 [+0.604, +1.295] | +1.052 [+0.418, +1.774] |
| mass P(o1) + P(o2), assistant-none>new            | +0.654 [+0.548, +0.753] | +0.614 [+0.489, +0.736] |
| prior log P(1st) - log P(2nd), character-none>new | +1.058 [+0.629, +1.555] | +1.044 [+0.362, +1.803] |
| mass P(o1) + P(o2), character-none>new            | +0.635 [+0.519, +0.748] | +0.624 [+0.503, +0.742] |

## Readout tokenization (variant: tokens)

`' odd'` 1; `'odd'` 1; `' Odd'` 1; `'Odd'` 1; `' ODD'` 2; `'ODD'` 2; `' even'` 1; `'even'` 1; `' Even'` 1; `'Even'` 1; `' EVEN'` 2; `'EVEN'` 2; `' top'` 1; `'top'` 1; `' Top'` 1; `'Top'` 1; `' TOP'` 1; `'TOP'` 1; `' bottom'` 1; `'bottom'` 1; `' Bottom'` 1; `'Bottom'` 1; `' BOTTOM'` 2; `'BOTTOM'` 2; `' home'` 1; `'home'` 1; `' Home'` 1; `'Home'` 1; `' HOME'` 2; `'HOME'` 1; `' away'` 1; `'away'` 1; `' Away'` 1; `'Away'` 2; `' AWAY'` 2; `'AWAY'` 3; `' rain'` 1; `'rain'` 1; `' Rain'` 1; `'Rain'` 1; `' RAIN'` 2; `'RAIN'` 2; `' snow'` 1; `'snow'` 2; `' Snow'` 1; `'Snow'` 1; `' SNOW'` 2; `'SNOW'` 2
