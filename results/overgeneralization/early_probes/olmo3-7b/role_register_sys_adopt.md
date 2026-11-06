# Register overgeneralization: role groups on the fixed persona axis (exploratory)

36 roles x 16 questions, one sample each; coordinate 0 = mean character reply, 1 = mean Assistant-framing reply of the post-trained model. Mean of per-role means, 95% bootstrap CI over roles.

| group | sft | instruct |
|---|---|---|
| helpful | +0.73 [+0.68, +0.78] | +0.53 [+0.43, +0.63] |
| neutral | +0.51 [+0.40, +0.62] | +0.10 [-0.06, +0.28] |
| stylized | -0.30 [-0.44, -0.15] | -0.69 [-0.78, -0.59] |

Change from the first stage (paired by role):

| group | sft | instruct |
|---|---|---|
| helpful | - | -0.20 [-0.27, -0.12] |
| neutral | - | -0.40 [-0.51, -0.31] |
| stylized | - | -0.39 [-0.48, -0.31] |

Per role at the last stage:

- helpful: caring social worker +0.63, cheerful tour guide +0.22, courteous bank teller +0.67, friendly pharmacist +0.51, helpful hotel front-desk cle +0.28, helpful librarian +0.61, kind nurse +0.63, knowledgeable IT helpdesk te +0.60, patient high-school teacher +0.43, polite airline check-in agen +0.64, supportive math tutor +0.83, an attentive receptionist at +0.36
- neutral: accountant +0.60, apple farmer -0.40, bus driver -0.02, carpenter +0.13, commercial pilot +0.17, lab technician +0.18, long-haul truck driver -0.29, plumber +0.06, postal carrier +0.12, surveyor +0.66, warehouse worker +0.05, welder -0.03
- stylized: cowboy from an old western -0.78, dramatic Shakespearean actor -0.67, gruff wizard who lives in a  -0.82, hard-boiled noir detective -0.78, laconic space-freighter pilo -0.82, medieval court jester -0.82, mischievous talking cat -0.50, pompous Victorian aristocrat -0.60, sleepy lighthouse keeper who -0.87, swaggering pirate captain -0.83, villainous mad scientist -0.54, an excitable twelve-year-old -0.30
