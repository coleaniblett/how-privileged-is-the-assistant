# Register overgeneralization: role groups on the fixed persona axis (exploratory)

36 roles x 16 questions, one sample each; coordinate 0 = mean character reply, 1 = mean Assistant-framing reply of the post-trained model. Mean of per-role means, 95% bootstrap CI over roles.

| group | Llama-3.1-8B | Llama-3.1-8B-Instruct |
|---|---|---|
| helpful | +0.49 [+0.48, +0.50] | +0.27 [+0.23, +0.32] |
| neutral | +0.50 [+0.48, +0.52] | +0.10 [+0.06, +0.15] |
| stylized | +0.45 [+0.43, +0.46] | +0.02 [-0.03, +0.06] |

Change from the first stage (paired by role):

| group | Llama-3.1-8B | Llama-3.1-8B-Instruct |
|---|---|---|
| helpful | - | -0.22 [-0.26, -0.17] |
| neutral | - | -0.40 [-0.44, -0.36] |
| stylized | - | -0.43 [-0.47, -0.38] |

Per role at the last stage:

- helpful: caring social worker +0.33, cheerful tour guide +0.18, courteous bank teller +0.25, friendly pharmacist +0.25, helpful hotel front-desk cle +0.19, helpful librarian +0.43, kind nurse +0.19, knowledgeable IT helpdesk te +0.36, patient high-school teacher +0.19, polite airline check-in agen +0.26, supportive math tutor +0.40, an attentive receptionist at +0.24
- neutral: accountant +0.26, apple farmer +0.02, bus driver +0.06, carpenter +0.07, commercial pilot +0.17, lab technician +0.23, long-haul truck driver -0.01, plumber +0.07, postal carrier +0.07, surveyor +0.17, warehouse worker +0.09, welder +0.07
- stylized: cowboy from an old western -0.10, dramatic Shakespearean actor +0.07, gruff wizard who lives in a  +0.02, hard-boiled noir detective -0.08, laconic space-freighter pilo -0.08, medieval court jester +0.04, mischievous talking cat +0.10, pompous Victorian aristocrat +0.07, sleepy lighthouse keeper who +0.04, swaggering pirate captain -0.06, villainous mad scientist +0.13, an excitable twelve-year-old +0.07
