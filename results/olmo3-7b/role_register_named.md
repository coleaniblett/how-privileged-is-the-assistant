# Register overgeneralization: role groups on the fixed persona axis (exploratory)

36 roles x 16 questions, one sample each; coordinate 0 = mean character reply, 1 = mean Assistant-framing reply of the post-trained model. Mean of per-role means, 95% bootstrap CI over roles.

| group | base | sft | dpo | instruct |
|---|---|---|---|---|
| helpful | +0.49 [+0.44, +0.53] | +0.80 [+0.73, +0.87] | +0.76 [+0.70, +0.81] | +0.82 [+0.77, +0.87] |
| neutral | +0.38 [+0.30, +0.47] | +0.73 [+0.69, +0.78] | +0.58 [+0.44, +0.70] | +0.66 [+0.55, +0.76] |
| stylized | +0.29 [+0.21, +0.35] | +0.39 [+0.22, +0.54] | -0.25 [-0.46, -0.02] | -0.21 [-0.41, +0.01] |

Change from the first stage (paired by role):

| group | base | sft | dpo | instruct |
|---|---|---|---|---|
| helpful | - | +0.32 [+0.26, +0.38] | +0.27 [+0.23, +0.31] | +0.33 [+0.29, +0.38] |
| neutral | - | +0.35 [+0.27, +0.43] | +0.20 [+0.06, +0.32] | +0.28 [+0.16, +0.39] |
| stylized | - | +0.10 [-0.08, +0.27] | -0.54 [-0.75, -0.30] | -0.50 [-0.71, -0.28] |

Per role at the last stage:

- helpful: caring social worker +0.88, cheerful tour guide +0.78, courteous bank teller +0.93, friendly pharmacist +0.82, helpful hotel front-desk cle +0.71, helpful librarian +0.92, kind nurse +0.85, knowledgeable IT helpdesk te +0.91, patient high-school teacher +0.74, polite airline check-in agen +0.70, supportive math tutor +0.89, an attentive receptionist at +0.71
- neutral: accountant +0.79, apple farmer +0.50, bus driver +0.73, carpenter +0.82, commercial pilot +0.60, lab technician +0.75, long-haul truck driver +0.27, plumber +0.38, postal carrier +0.56, surveyor +0.83, warehouse worker +0.90, welder +0.79
- stylized: cowboy from an old western -0.35, dramatic Shakespearean actor +0.18, gruff wizard who lives in a  -0.26, hard-boiled noir detective -0.40, laconic space-freighter pilo -0.54, medieval court jester -0.41, mischievous talking cat +0.26, pompous Victorian aristocrat -0.56, sleepy lighthouse keeper who -0.84, swaggering pirate captain -0.19, villainous mad scientist +0.48, an excitable twelve-year-old +0.15
