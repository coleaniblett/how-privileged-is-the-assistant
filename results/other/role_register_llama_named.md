# Register overgeneralization: role groups on the fixed persona axis (exploratory)

36 roles x 16 questions, one sample each; coordinate 0 = mean character reply, 1 = mean Assistant-framing reply of the post-trained model. Mean of per-role means, 95% bootstrap CI over roles.

| group | Llama-3.1-8B | Llama-3.1-8B-Instruct |
|---|---|---|
| helpful | +0.48 [+0.46, +0.50] | +0.22 [+0.19, +0.25] |
| neutral | +0.47 [+0.45, +0.49] | +0.08 [+0.04, +0.12] |
| stylized | +0.43 [+0.41, +0.45] | +0.01 [-0.03, +0.05] |

Change from the first stage (paired by role):

| group | Llama-3.1-8B | Llama-3.1-8B-Instruct |
|---|---|---|
| helpful | - | -0.26 [-0.29, -0.23] |
| neutral | - | -0.39 [-0.42, -0.36] |
| stylized | - | -0.42 [-0.46, -0.38] |

Per role at the last stage:

- helpful: caring social worker +0.21, cheerful tour guide +0.15, courteous bank teller +0.21, friendly pharmacist +0.22, helpful hotel front-desk cle +0.20, helpful librarian +0.28, kind nurse +0.17, knowledgeable IT helpdesk te +0.31, patient high-school teacher +0.14, polite airline check-in agen +0.21, supportive math tutor +0.31, an attentive receptionist at +0.20
- neutral: accountant +0.20, apple farmer -0.03, bus driver +0.02, carpenter +0.07, commercial pilot +0.13, lab technician +0.18, long-haul truck driver +0.01, plumber +0.05, postal carrier +0.05, surveyor +0.15, warehouse worker +0.09, welder +0.05
- stylized: cowboy from an old western -0.10, dramatic Shakespearean actor +0.04, gruff wizard who lives in a  -0.00, hard-boiled noir detective -0.08, laconic space-freighter pilo -0.07, medieval court jester +0.04, mischievous talking cat +0.10, pompous Victorian aristocrat +0.07, sleepy lighthouse keeper who +0.02, swaggering pirate captain -0.06, villainous mad scientist +0.12, an excitable twelve-year-old +0.07
