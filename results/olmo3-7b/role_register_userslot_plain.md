# Register overgeneralization: role groups on the fixed persona axis (exploratory)

36 roles x 16 questions, one sample each; coordinate 0 = mean character reply, 1 = mean Assistant-framing reply of the post-trained model. Mean of per-role means, 95% bootstrap CI over roles.

| group | base | sft | dpo | instruct |
|---|---|---|---|---|
| helpful | +0.55 [+0.51, +0.60] | +0.45 [+0.34, +0.56] | +0.34 [+0.21, +0.45] | +0.37 [+0.25, +0.48] |
| neutral | +0.58 [+0.53, +0.63] | +0.19 [+0.08, +0.30] | +0.02 [-0.10, +0.17] | +0.07 [-0.06, +0.22] |
| stylized | +0.24 [+0.19, +0.29] | -0.58 [-0.67, -0.48] | -0.78 [-0.85, -0.70] | -0.78 [-0.85, -0.68] |

Change from the first stage (paired by role):

| group | base | sft | dpo | instruct |
|---|---|---|---|---|
| helpful | - | -0.10 [-0.19, +0.01] | -0.21 [-0.32, -0.11] | -0.18 [-0.28, -0.08] |
| neutral | - | -0.40 [-0.49, -0.29] | -0.56 [-0.66, -0.43] | -0.51 [-0.62, -0.39] |
| stylized | - | -0.83 [-0.92, -0.73] | -1.03 [-1.11, -0.94] | -1.02 [-1.11, -0.93] |

Per role at the last stage:

- helpful: caring social worker +0.63, cheerful tour guide -0.05, courteous bank teller +0.33, friendly pharmacist +0.44, helpful hotel front-desk cle +0.09, helpful librarian +0.54, kind nurse +0.49, knowledgeable IT helpdesk te +0.60, patient high-school teacher +0.25, polite airline check-in agen +0.31, supportive math tutor +0.58, an attentive receptionist at +0.24
- neutral: accountant +0.28, apple farmer -0.41, bus driver +0.01, carpenter -0.01, commercial pilot +0.09, lab technician +0.15, long-haul truck driver -0.15, plumber +0.01, postal carrier -0.05, surveyor +0.66, warehouse worker +0.17, welder +0.05
- stylized: cowboy from an old western -0.85, dramatic Shakespearean actor -0.81, gruff wizard who lives in a  -0.84, hard-boiled noir detective -0.94, laconic space-freighter pilo -0.85, medieval court jester -0.85, mischievous talking cat -0.61, pompous Victorian aristocrat -0.74, sleepy lighthouse keeper who -0.90, swaggering pirate captain -0.90, villainous mad scientist -0.60, an excitable twelve-year-old -0.40
