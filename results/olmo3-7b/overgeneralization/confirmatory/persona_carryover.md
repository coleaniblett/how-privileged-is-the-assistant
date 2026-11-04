# Persona carryover (overgeneralization)

The pilot item set is exploratory; the verdict comes from the confirmatory set (README). Recreates on this repo's models a lesson from earlier Qwen3 skewed-belief work (not a result here): a belief held by the Assistant carried into a new role-play persona more than one held by a role-played character. Design and readout: module docstring.

48 items (dice: 12, drawer: 12, match: 12, weather: 12), each with both outcomes believed. Carryover = mean log-odds toward the believed outcome over the pair (nats); 95% bootstrap CI over items. Prediction fixed in advance: leak gap > 0. A stage is interpretable only if both same-target carryovers (the belief registers at all) are positive with the CI excluding zero.

## Headline (all domains)

| stage    | assistant -> new        | assistant -> same       | character -> new        | character -> same       | leak gap                | change from base        | reading                |
|:---------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:-----------------------|
| base     | +0.587 [+0.316, +0.869] | +1.014 [+0.710, +1.310] | +0.257 [-0.052, +0.557] | +1.733 [+1.403, +2.048] | +0.331 [+0.261, +0.402] |                         | assistant carries more |
| sft      | +0.247 [-0.031, +0.544] | +1.111 [+0.769, +1.447] | -0.216 [-0.588, +0.170] | +1.354 [+0.907, +1.799] | +0.463 [+0.318, +0.609] | +0.132 [-0.000, +0.262] | assistant carries more |
| dpo      | -0.438 [-1.034, +0.168] | +1.733 [+0.997, +2.394] | -1.820 [-2.576, -1.078] | +1.618 [+0.892, +2.298] | +1.382 [+1.030, +1.741] | +1.051 [+0.710, +1.383] | assistant carries more |
| instruct | -0.245 [-0.885, +0.428] | +2.171 [+1.396, +2.895] | -1.957 [-2.729, -1.168] | +2.025 [+1.285, +2.689] | +1.712 [+1.333, +2.081] | +1.382 [+1.024, +1.737] | assistant carries more |

## Pre-registered verdict (criteria 1-5, README; confirmatory item set only)

delta = 0.25 nats, domain effect = 0.5 nats; a bar is met when the mean is at least the bar and the 95% CI excludes zero. Domains are never pooled. Criterion 4 is checked in every domain that meets criterion 2.

| stage | domain | 1. interpretable | 2. leak-gap change >= 0.5 | 4. assistant -> new rises >= 0.25 |
|---|---|---|---|---|
| sft | dice | yes | no | no |
| sft | drawer | no | no | no |
| sft | match | yes | no | no |
| sft | weather | yes | no | no |
| dpo | dice | yes | yes | yes |
| dpo | drawer | no | no | no |
| dpo | match | no | no | no |
| dpo | weather | yes | no | no |
| instruct | dice | yes | yes | yes |
| instruct | drawer | no | no | no |
| instruct | match | yes | yes | no |
| instruct | weather | yes | no | no |

| stage | interpretable domains | domains meeting 2 | 3. stage rule | 4. absolute | reading |
|---|---|---|---|---|---|
| sft | 3 | 0 | no | no | no overgeneralization by the rule |
| dpo | 2 | 1 | no | no | no overgeneralization by the rule |
| instruct | 3 | 2 | yes | no | relative only (criterion 3 holds, criterion 4 does not) |

**Answer to "which stage" (criterion 5):** no stage meets criteria 3 and 4.

## By domain

### dice

| quantity                                            | base                    | sft                     | dpo                     | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +0.943 [+0.844, +1.041] | +1.140 [+0.932, +1.385] | +2.239 [+1.614, +2.952] | +2.707 [+2.078, +3.390] |
| carryover assistant -> same                         | +1.308 [+1.189, +1.424] | +1.750 [+1.526, +1.979] | +3.672 [+3.095, +4.275] | +4.379 [+3.743, +5.058] |
| carryover character -> new                          | +0.798 [+0.631, +0.958] | +0.781 [+0.500, +1.109] | +0.687 [+0.318, +1.143] | +0.771 [+0.414, +1.190] |
| carryover character -> same                         | +2.330 [+1.976, +2.722] | +1.828 [+1.479, +2.213] | +3.210 [+2.840, +3.600] | +3.702 [+3.338, +4.085] |
| leak gap: assistant -> new minus character -> new   | +0.144 [+0.055, +0.232] | +0.359 [+0.198, +0.510] | +1.551 [+1.127, +1.955] | +1.937 [+1.470, +2.366] |
| same gap: assistant -> same minus character -> same | -1.022 [-1.336, -0.742] | -0.078 [-0.417, +0.224] | +0.462 [+0.063, +0.903] | +0.677 [+0.178, +1.255] |
| leak gap, change from base                          |                         | +0.215 [+0.096, +0.320] | +1.407 [+1.008, +1.786] | +1.792 [+1.338, +2.212] |

### drawer

| quantity                                            | base                    | sft                     | dpo                     | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | -0.380 [-0.467, -0.303] | -0.698 [-0.818, -0.583] | -2.385 [-3.047, -1.739] | -2.229 [-2.966, -1.505] |
| carryover assistant -> same                         | -0.577 [-0.750, -0.387] | -0.518 [-0.776, -0.263] | -1.609 [-2.507, -0.721] | -1.310 [-2.572, -0.122] |
| carryover character -> new                          | -0.816 [-0.956, -0.669] | -1.698 [-1.927, -1.463] | -5.015 [-5.684, -4.343] | -5.278 [-5.992, -4.591] |
| carryover character -> same                         | +0.094 [-0.215, +0.408] | -0.437 [-1.026, +0.141] | -1.088 [-2.644, +0.341] | -0.388 [-2.090, +1.125] |
| leak gap: assistant -> new minus character -> new   | +0.436 [+0.332, +0.554] | +1.000 [+0.807, +1.198] | +2.630 [+2.198, +3.099] | +3.049 [+2.518, +3.641] |
| same gap: assistant -> same minus character -> same | -0.671 [-0.812, -0.527] | -0.081 [-0.479, +0.364] | -0.521 [-1.339, +0.333] | -0.922 [-1.797, -0.068] |
| leak gap, change from base                          |                         | +0.564 [+0.445, +0.697] | +2.194 [+1.823, +2.580] | +2.614 [+2.150, +3.116] |

### match

| quantity                                            | base                    | sft                     | dpo                     | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | -0.179 [-0.366, +0.003] | -0.640 [-0.796, -0.489] | -1.687 [-1.968, -1.416] | -1.583 [-1.843, -1.322] |
| carryover assistant -> same                         | +1.194 [+0.935, +1.483] | +0.807 [+0.531, +1.093] | +1.385 [+0.694, +2.108] | +1.810 [+1.036, +2.571] |
| carryover character -> new                          | -0.585 [-0.928, -0.244] | -1.008 [-1.174, -0.880] | -3.012 [-3.320, -2.679] | -3.192 [-3.578, -2.797] |
| carryover character -> same                         | +1.854 [+1.454, +2.241] | +0.758 [+0.484, +1.015] | +0.542 [-0.156, +1.234] | +0.875 [+0.115, +1.588] |
| leak gap: assistant -> new minus character -> new   | +0.406 [+0.217, +0.598] | +0.367 [+0.219, +0.516] | +1.326 [+0.971, +1.719] | +1.610 [+1.250, +1.979] |
| same gap: assistant -> same minus character -> same | -0.660 [-0.870, -0.455] | +0.049 [-0.230, +0.333] | +0.843 [+0.282, +1.394] | +0.935 [+0.318, +1.546] |
| leak gap, change from base                          |                         | -0.039 [-0.244, +0.147] | +0.920 [+0.530, +1.314] | +1.203 [+0.843, +1.579] |

### weather

| quantity                                            | base                    | sft                     | dpo                     | instruct                |
|:----------------------------------------------------|:------------------------|:------------------------|:------------------------|:------------------------|
| carryover assistant -> new                          | +1.965 [+1.735, +2.174] | +1.187 [+0.797, +1.577] | +0.081 [-0.816, +1.029] | +0.126 [-0.825, +1.144] |
| carryover assistant -> same                         | +2.132 [+1.843, +2.439] | +2.408 [+2.137, +2.655] | +3.484 [+3.021, +3.923] | +3.804 [+3.254, +4.309] |
| carryover character -> new                          | +1.629 [+1.406, +1.830] | +1.062 [+0.434, +1.663] | +0.060 [-1.103, +1.150] | -0.128 [-1.305, +1.028] |
| carryover character -> same                         | +2.653 [+2.361, +2.913] | +3.267 [+2.882, +3.618] | +3.809 [+3.457, +4.179] | +3.910 [+3.425, +4.396] |
| leak gap: assistant -> new minus character -> new   | +0.336 [+0.262, +0.405] | +0.125 [-0.195, +0.466] | +0.021 [-0.601, +0.611] | +0.254 [-0.328, +0.800] |
| same gap: assistant -> same minus character -> same | -0.521 [-0.768, -0.289] | -0.859 [-1.153, -0.562] | -0.325 [-0.727, +0.076] | -0.106 [-0.516, +0.316] |
| leak gap, change from base                          |                         | -0.211 [-0.512, +0.123] | -0.316 [-0.887, +0.250] | -0.082 [-0.615, +0.434] |

## Raw log-probs, priors and coverage (all domains)

### all conditions

| quantity                                          | base                    | sft                     | dpo                     | instruct                |
|:--------------------------------------------------|:------------------------|:------------------------|:------------------------|:------------------------|
| raw log P(believed), assistant>new                | -0.969 [-1.098, -0.847] | -1.280 [-1.467, -1.098] | -3.142 [-3.839, -2.478] | -3.366 [-4.168, -2.614] |
| raw log P(other), assistant>new                   | -1.556 [-1.912, -1.228] | -1.527 [-1.877, -1.201] | -2.704 [-3.501, -1.979] | -3.121 [-3.955, -2.325] |
| mass P(o1) + P(o2), assistant>new                 | +0.760 [+0.680, +0.836] | +0.743 [+0.655, +0.827] | +0.608 [+0.488, +0.719] | +0.602 [+0.478, +0.720] |
| raw log P(believed), assistant>same               | -0.996 [-1.193, -0.808] | -0.951 [-1.153, -0.767] | -1.679 [-2.304, -1.164] | -1.686 [-2.376, -1.130] |
| raw log P(other), assistant>same                  | -2.010 [-2.384, -1.647] | -2.063 [-2.401, -1.756] | -3.412 [-3.931, -2.903] | -3.857 [-4.369, -3.355] |
| mass P(o1) + P(o2), assistant>same                | +0.705 [+0.620, +0.785] | +0.746 [+0.660, +0.830] | +0.660 [+0.557, +0.760] | +0.682 [+0.578, +0.780] |
| raw log P(believed), character>new                | -1.217 [-1.386, -1.055] | -1.649 [-1.879, -1.425] | -4.273 [-4.957, -3.614] | -4.583 [-5.304, -3.899] |
| raw log P(other), character>new                   | -1.474 [-1.864, -1.119] | -1.434 [-1.853, -1.041] | -2.453 [-3.350, -1.628] | -2.626 [-3.586, -1.762] |
| mass P(o1) + P(o2), character>new                 | +0.742 [+0.656, +0.824] | +0.743 [+0.646, +0.835] | +0.660 [+0.545, +0.768] | +0.664 [+0.551, +0.773] |
| raw log P(believed), character>same               | -0.660 [-0.823, -0.514] | -0.855 [-1.035, -0.685] | -2.096 [-2.727, -1.516] | -2.000 [-2.639, -1.404] |
| raw log P(other), character>same                  | -2.393 [-2.740, -2.050] | -2.209 [-2.665, -1.789] | -3.714 [-4.528, -2.954] | -4.025 [-4.820, -3.270] |
| mass P(o1) + P(o2), character>same                | +0.766 [+0.688, +0.837] | +0.789 [+0.706, +0.864] | +0.655 [+0.535, +0.770] | +0.682 [+0.557, +0.795] |
| prior log P(1st) - log P(2nd), assistant-none>new | +0.950 [+0.626, +1.318] | +0.705 [+0.201, +1.249] | +0.438 [-0.432, +1.313] | +0.466 [-0.565, +1.489] |
| mass P(o1) + P(o2), assistant-none>new            | +0.611 [+0.500, +0.717] | +0.648 [+0.543, +0.752] | +0.526 [+0.409, +0.639] | +0.507 [+0.387, +0.629] |
| prior log P(1st) - log P(2nd), character-none>new | +0.912 [+0.533, +1.335] | +0.747 [+0.119, +1.406] | +0.338 [-0.598, +1.250] | +0.327 [-0.732, +1.373] |
| mass P(o1) + P(o2), character-none>new            | +0.600 [+0.490, +0.709] | +0.649 [+0.541, +0.753] | +0.546 [+0.424, +0.668] | +0.539 [+0.417, +0.661] |

## Readout tokenization (variant: tokens)

`' odd'` 1; `'odd'` 1; `' Odd'` 1; `'Odd'` 1; `' ODD'` 2; `'ODD'` 2; `' even'` 1; `'even'` 1; `' Even'` 1; `'Even'` 1; `' EVEN'` 1; `'EVEN'` 2; `' top'` 1; `'top'` 1; `' Top'` 1; `'Top'` 1; `' TOP'` 1; `'TOP'` 1; `' bottom'` 1; `'bottom'` 1; `' Bottom'` 1; `'Bottom'` 1; `' BOTTOM'` 1; `'BOTTOM'` 1; `' home'` 1; `'home'` 1; `' Home'` 1; `'Home'` 1; `' HOME'` 1; `'HOME'` 1; `' away'` 1; `'away'` 1; `' Away'` 1; `'Away'` 1; `' AWAY'` 2; `'AWAY'` 2; `' rain'` 1; `'rain'` 1; `' Rain'` 1; `'Rain'` 1; `' RAIN'` 2; `'RAIN'` 1; `' snow'` 1; `'snow'` 1; `' Snow'` 1; `'Snow'` 1; `' SNOW'` 2; `'SNOW'` 2
