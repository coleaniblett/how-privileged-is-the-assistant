# Register overgeneralization: role groups on the fixed persona axis (exploratory)

36 roles x 16 questions, one sample each; coordinate 0 = mean character reply, 1 = mean Assistant-framing reply of the post-trained model. Mean of per-role means, 95% bootstrap CI over roles.

| group | base | sft | dpo | instruct |
|---|---|---|---|---|
| helpful | +0.41 [+0.37, +0.44] | -0.07 [-0.14, +0.01] | -0.21 [-0.29, -0.14] | -0.15 [-0.23, -0.09] |
| neutral | +0.37 [+0.28, +0.45] | -0.42 [-0.49, -0.35] | -0.56 [-0.63, -0.49] | -0.52 [-0.59, -0.44] |
| stylized | +0.01 [-0.09, +0.10] | -0.76 [-0.82, -0.70] | -0.85 [-0.91, -0.79] | -0.84 [-0.90, -0.78] |

Change from the first stage (paired by role):

| group | base | sft | dpo | instruct |
|---|---|---|---|---|
| helpful | - | -0.47 [-0.56, -0.40] | -0.62 [-0.70, -0.55] | -0.56 [-0.64, -0.49] |
| neutral | - | -0.78 [-0.87, -0.70] | -0.93 [-1.01, -0.83] | -0.88 [-0.98, -0.79] |
| stylized | - | -0.77 [-0.85, -0.69] | -0.86 [-0.95, -0.78] | -0.85 [-0.94, -0.76] |

Per role at the last stage:

- helpful: caring social worker -0.05, cheerful tour guide -0.41, courteous bank teller -0.09, friendly pharmacist -0.04, helpful hotel front-desk cle -0.24, helpful librarian -0.29, kind nurse -0.10, knowledgeable IT helpdesk te +0.01, patient high-school teacher -0.23, polite airline check-in agen -0.15, supportive math tutor -0.02, an attentive receptionist at -0.23
- neutral: accountant -0.40, apple farmer -0.78, bus driver -0.65, carpenter -0.46, commercial pilot -0.45, lab technician -0.44, long-haul truck driver -0.59, plumber -0.53, postal carrier -0.54, surveyor -0.32, warehouse worker -0.38, welder -0.67
- stylized: cowboy from an old western -0.88, dramatic Shakespearean actor -0.83, gruff wizard who lives in a  -0.83, hard-boiled noir detective -0.89, laconic space-freighter pilo -0.92, medieval court jester -0.91, mischievous talking cat -0.76, pompous Victorian aristocrat -0.74, sleepy lighthouse keeper who -0.99, swaggering pirate captain -0.98, villainous mad scientist -0.66, an excitable twelve-year-old -0.65
