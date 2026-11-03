# Register overgeneralization: role groups on the fixed persona axis (exploratory)

36 roles x 16 questions, one sample each; coordinate 0 = mean character reply, 1 = mean Assistant-framing reply of the post-trained model. Mean of per-role means, 95% bootstrap CI over roles.

| group | base | sft | dpo | instruct |
|---|---|---|---|---|
| helpful | +0.43 [+0.38, +0.48] | +0.50 [+0.40, +0.59] | +0.18 [+0.07, +0.31] | +0.24 [+0.11, +0.36] |
| neutral | +0.30 [+0.24, +0.36] | +0.21 [+0.09, +0.32] | -0.21 [-0.35, -0.08] | -0.17 [-0.29, -0.06] |
| stylized | -0.04 [-0.15, +0.04] | -0.31 [-0.44, -0.18] | -0.73 [-0.81, -0.66] | -0.71 [-0.79, -0.64] |

Change from the first stage (paired by role):

| group | base | sft | dpo | instruct |
|---|---|---|---|---|
| helpful | - | +0.07 [-0.02, +0.15] | -0.25 [-0.35, -0.14] | -0.19 [-0.31, -0.08] |
| neutral | - | -0.09 [-0.21, +0.01] | -0.51 [-0.63, -0.39] | -0.47 [-0.58, -0.37] |
| stylized | - | -0.27 [-0.36, -0.18] | -0.69 [-0.78, -0.61] | -0.67 [-0.76, -0.59] |

Per role at the last stage:

- helpful: caring social worker +0.19, cheerful tour guide -0.14, courteous bank teller +0.27, friendly pharmacist +0.46, helpful hotel front-desk cle +0.07, helpful librarian +0.50, kind nurse +0.23, knowledgeable IT helpdesk te +0.52, patient high-school teacher +0.14, polite airline check-in agen -0.01, supportive math tutor +0.59, an attentive receptionist at +0.04
- neutral: accountant -0.03, apple farmer -0.60, bus driver -0.13, carpenter +0.11, commercial pilot -0.31, lab technician +0.07, long-haul truck driver -0.38, plumber -0.21, postal carrier -0.14, surveyor -0.03, warehouse worker -0.02, welder -0.38
- stylized: cowboy from an old western -0.77, dramatic Shakespearean actor -0.66, gruff wizard who lives in a  -0.65, hard-boiled noir detective -0.75, laconic space-freighter pilo -0.75, medieval court jester -0.86, mischievous talking cat -0.55, pompous Victorian aristocrat -0.71, sleepy lighthouse keeper who -0.91, swaggering pirate captain -0.88, villainous mad scientist -0.51, an excitable twelve-year-old -0.54
