# Carryover under persona-axis steering (exploratory)

OLMo 3 7B instruct, flat factual statements, balanced frame (holder introduced as a role-played speaker), pilot domains, 36 items. alpha x gap added along the persona axis at layer 18 (positive = toward the Assistant end).

| alpha | carryover(assistant->new) | carryover(user->new) | carryover(char->new) | gap assistant - char | gap user - char |
|---|---|---|---|---|---|
| -2 | -2.46 [-3.12, -1.81] | -2.17 [-2.78, -1.53] | -3.27 [-4.03, -2.48] | 0.80 [0.62, 1.00] | 1.10 [0.84, 1.35] |
| -1 | -2.78 [-3.51, -2.08] | -2.54 [-3.19, -1.85] | -3.82 [-4.69, -2.91] | 1.03 [0.76, 1.29] | 1.27 [0.93, 1.64] |
| +0 | -3.16 [-3.97, -2.36] | -2.92 [-3.65, -2.11] | -4.29 [-5.29, -3.25] | 1.13 [0.83, 1.46] | 1.37 [0.96, 1.81] |
| +1 | -3.40 [-4.39, -2.49] | -3.08 [-3.95, -2.22] | -4.54 [-5.68, -3.39] | 1.14 [0.82, 1.51] | 1.46 [1.02, 1.94] |
| +2 | -3.35 [-4.35, -2.36] | -3.01 [-3.95, -2.09] | -4.40 [-5.63, -3.18] | 1.05 [0.76, 1.37] | 1.39 [1.03, 1.77] |
