# Register overgeneralization: role groups on the fixed persona axis (exploratory)

36 roles x 16 questions, one sample each; coordinate 0 = mean character reply, 1 = mean Assistant-framing reply of the post-trained model. Mean of per-role means, 95% bootstrap CI over roles.

| group | sft | instruct |
|---|---|---|
| helpful | +0.67 [+0.58, +0.77] | +0.58 [+0.45, +0.70] |
| neutral | +0.45 [+0.33, +0.57] | +0.11 [-0.10, +0.33] |
| stylized | -0.42 [-0.54, -0.29] | -0.74 [-0.82, -0.66] |

Change from the first stage (paired by role):

| group | sft | instruct |
|---|---|---|
| helpful | - | -0.10 [-0.16, -0.03] |
| neutral | - | -0.34 [-0.47, -0.22] |
| stylized | - | -0.32 [-0.39, -0.26] |

Per role at the last stage:

- helpful: caring social worker +0.73, cheerful tour guide +0.16, courteous bank teller +0.46, friendly pharmacist +0.65, helpful hotel front-desk cle +0.51, helpful librarian +0.92, kind nurse +0.78, knowledgeable IT helpdesk te +0.81, patient high-school teacher +0.33, polite airline check-in agen +0.55, supportive math tutor +0.79, an attentive receptionist at +0.25
- neutral: accountant +0.62, apple farmer -0.44, bus driver -0.05, carpenter +0.04, commercial pilot +0.07, lab technician +0.42, long-haul truck driver -0.46, plumber -0.24, postal carrier +0.38, surveyor +0.73, warehouse worker +0.36, welder -0.13
- stylized: cowboy from an old western -0.78, dramatic Shakespearean actor -0.71, gruff wizard who lives in a  -0.89, hard-boiled noir detective -0.83, laconic space-freighter pilo -0.85, medieval court jester -0.90, mischievous talking cat -0.61, pompous Victorian aristocrat -0.61, sleepy lighthouse keeper who -0.82, swaggering pirate captain -0.87, villainous mad scientist -0.61, an excitable twelve-year-old -0.45
