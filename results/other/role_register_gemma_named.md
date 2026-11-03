# Register overgeneralization: role groups on the fixed persona axis (exploratory)

36 roles x 16 questions, one sample each; coordinate 0 = mean character reply, 1 = mean Assistant-framing reply of the post-trained model. Mean of per-role means, 95% bootstrap CI over roles.

| group | gemma-4-E4B | gemma-4-E4B-it |
|---|---|---|
| helpful | +0.83 [+0.82, +0.84] | +0.60 [+0.49, +0.71] |
| neutral | +0.82 [+0.81, +0.84] | +0.18 [+0.08, +0.28] |
| stylized | +0.79 [+0.77, +0.80] | +0.01 [-0.04, +0.06] |

Change from the first stage (paired by role):

| group | gemma-4-E4B | gemma-4-E4B-it |
|---|---|---|
| helpful | - | -0.23 [-0.32, -0.13] |
| neutral | - | -0.65 [-0.73, -0.56] |
| stylized | - | -0.78 [-0.84, -0.72] |

Per role at the last stage:

- helpful: caring social worker +0.47, cheerful tour guide +0.44, courteous bank teller +0.55, friendly pharmacist +0.71, helpful hotel front-desk cle +0.67, helpful librarian +0.81, kind nurse +0.56, knowledgeable IT helpdesk te +0.85, patient high-school teacher +0.27, polite airline check-in agen +0.49, supportive math tutor +0.91, an attentive receptionist at +0.45
- neutral: accountant +0.55, apple farmer +0.08, bus driver +0.15, carpenter +0.09, commercial pilot +0.20, lab technician +0.39, long-haul truck driver -0.08, plumber +0.17, postal carrier +0.02, surveyor +0.34, warehouse worker +0.22, welder +0.00
- stylized: cowboy from an old western +0.03, dramatic Shakespearean actor +0.02, gruff wizard who lives in a  -0.10, hard-boiled noir detective -0.03, laconic space-freighter pilo -0.02, medieval court jester -0.14, mischievous talking cat +0.06, pompous Victorian aristocrat +0.04, sleepy lighthouse keeper who +0.04, swaggering pirate captain -0.04, villainous mad scientist -0.02, an excitable twelve-year-old +0.23
