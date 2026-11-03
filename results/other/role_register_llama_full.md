# Register overgeneralization: role groups on the fixed persona axis (exploratory)

36 roles x 16 questions, one sample each; coordinate 0 = mean character reply, 1 = mean Assistant-framing reply of the post-trained model. Mean of per-role means, 95% bootstrap CI over roles.

| group | Llama-3.1-8B | Llama-3.1-8B-Instruct |
|---|---|---|
| helpful | +0.43 [+0.41, +0.44] | +0.14 [+0.12, +0.17] |
| neutral | +0.42 [+0.40, +0.45] | -0.01 [-0.05, +0.02] |
| stylized | +0.39 [+0.37, +0.42] | -0.00 [-0.04, +0.04] |

Change from the first stage (paired by role):

| group | Llama-3.1-8B | Llama-3.1-8B-Instruct |
|---|---|---|
| helpful | - | -0.28 [-0.31, -0.26] |
| neutral | - | -0.43 [-0.46, -0.40] |
| stylized | - | -0.39 [-0.44, -0.35] |

Per role at the last stage:

- helpful: caring social worker +0.14, cheerful tour guide +0.10, courteous bank teller +0.17, friendly pharmacist +0.16, helpful hotel front-desk cle +0.11, helpful librarian +0.21, kind nurse +0.11, knowledgeable IT helpdesk te +0.23, patient high-school teacher +0.04, polite airline check-in agen +0.13, supportive math tutor +0.18, an attentive receptionist at +0.15
- neutral: accountant +0.07, apple farmer -0.07, bus driver -0.10, carpenter -0.03, commercial pilot +0.09, lab technician +0.07, long-haul truck driver -0.08, plumber -0.07, postal carrier -0.04, surveyor +0.03, warehouse worker +0.01, welder -0.04
- stylized: cowboy from an old western -0.11, dramatic Shakespearean actor +0.02, gruff wizard who lives in a  +0.01, hard-boiled noir detective -0.10, laconic space-freighter pilo -0.08, medieval court jester +0.02, mischievous talking cat +0.08, pompous Victorian aristocrat +0.06, sleepy lighthouse keeper who -0.01, swaggering pirate captain -0.07, villainous mad scientist +0.12, an excitable twelve-year-old +0.05
