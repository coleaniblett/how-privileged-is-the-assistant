# Register overgeneralization: role groups on the fixed persona axis (exploratory)

36 roles x 16 questions, one sample each; coordinate 0 = mean character reply, 1 = mean Assistant-framing reply of the post-trained model. Mean of per-role means, 95% bootstrap CI over roles.

| group | sft | instruct |
|---|---|---|
| helpful | +0.59 [+0.48, +0.69] | +0.43 [+0.27, +0.57] |
| neutral | +0.40 [+0.32, +0.50] | +0.03 [-0.10, +0.16] |
| stylized | -0.30 [-0.47, -0.11] | -0.77 [-0.87, -0.67] |

Change from the first stage (paired by role):

| group | sft | instruct |
|---|---|---|
| helpful | - | -0.16 [-0.24, -0.08] |
| neutral | - | -0.37 [-0.46, -0.29] |
| stylized | - | -0.47 [-0.59, -0.37] |

Per role at the last stage:

- helpful: caring social worker +0.56, cheerful tour guide -0.03, courteous bank teller +0.54, friendly pharmacist +0.43, helpful hotel front-desk cle +0.41, helpful librarian +0.76, kind nurse +0.53, knowledgeable IT helpdesk te +0.59, patient high-school teacher -0.10, polite airline check-in agen +0.48, supportive math tutor +0.77, an attentive receptionist at +0.19
- neutral: accountant +0.41, apple farmer -0.32, bus driver -0.07, carpenter +0.02, commercial pilot +0.15, lab technician +0.20, long-haul truck driver -0.36, plumber -0.04, postal carrier -0.01, surveyor +0.41, warehouse worker +0.07, welder -0.08
- stylized: cowboy from an old western -0.81, dramatic Shakespearean actor -0.71, gruff wizard who lives in a  -0.94, hard-boiled noir detective -0.80, laconic space-freighter pilo -0.93, medieval court jester -0.92, mischievous talking cat -0.62, pompous Victorian aristocrat -0.72, sleepy lighthouse keeper who -0.96, swaggering pirate captain -0.95, villainous mad scientist -0.56, an excitable twelve-year-old -0.36
