# Register overgeneralization: role groups on the fixed persona axis (exploratory)

36 roles x 16 questions, one sample each; coordinate 0 = mean character reply, 1 = mean Assistant-framing reply of the post-trained model. Mean of per-role means, 95% bootstrap CI over roles.

| group | base | sft | dpo | instruct |
|---|---|---|---|---|
| helpful | +0.51 [+0.47, +0.54] | +0.88 [+0.85, +0.92] | +0.88 [+0.84, +0.91] | +0.91 [+0.87, +0.94] |
| neutral | +0.48 [+0.41, +0.56] | +0.88 [+0.85, +0.92] | +0.87 [+0.81, +0.91] | +0.92 [+0.89, +0.95] |
| stylized | +0.36 [+0.29, +0.42] | +0.57 [+0.38, +0.71] | +0.03 [-0.23, +0.26] | +0.12 [-0.15, +0.37] |

Change from the first stage (paired by role):

| group | base | sft | dpo | instruct |
|---|---|---|---|---|
| helpful | - | +0.37 [+0.33, +0.41] | +0.37 [+0.32, +0.43] | +0.40 [+0.36, +0.45] |
| neutral | - | +0.40 [+0.31, +0.48] | +0.39 [+0.29, +0.48] | +0.44 [+0.35, +0.52] |
| stylized | - | +0.21 [+0.01, +0.35] | -0.33 [-0.56, -0.11] | -0.24 [-0.50, -0.01] |

Per role at the last stage:

- helpful: caring social worker +0.93, cheerful tour guide +0.74, courteous bank teller +0.97, friendly pharmacist +0.93, helpful hotel front-desk cle +0.96, helpful librarian +0.96, kind nurse +0.90, knowledgeable IT helpdesk te +0.93, patient high-school teacher +0.83, polite airline check-in agen +0.91, supportive math tutor +0.96, an attentive receptionist at +0.91
- neutral: accountant +0.94, apple farmer +0.91, bus driver +0.92, carpenter +0.95, commercial pilot +1.00, lab technician +0.96, long-haul truck driver +0.77, plumber +0.92, postal carrier +0.92, surveyor +0.99, warehouse worker +0.92, welder +0.87
- stylized: cowboy from an old western +0.45, dramatic Shakespearean actor -0.17, gruff wizard who lives in a  -0.28, hard-boiled noir detective +0.11, laconic space-freighter pilo +0.56, medieval court jester +0.14, mischievous talking cat +0.52, pompous Victorian aristocrat -0.38, sleepy lighthouse keeper who -0.83, swaggering pirate captain +0.02, villainous mad scientist +0.76, an excitable twelve-year-old +0.57
