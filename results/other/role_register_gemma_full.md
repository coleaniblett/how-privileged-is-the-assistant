# Register overgeneralization: role groups on the fixed persona axis (exploratory)

36 roles x 16 questions, one sample each; coordinate 0 = mean character reply, 1 = mean Assistant-framing reply of the post-trained model. Mean of per-role means, 95% bootstrap CI over roles.

| group | gemma-4-E4B | gemma-4-E4B-it |
|---|---|---|
| helpful | +0.79 [+0.77, +0.81] | +0.41 [+0.31, +0.51] |
| neutral | +0.81 [+0.79, +0.83] | +0.01 [-0.04, +0.08] |
| stylized | +0.75 [+0.74, +0.76] | +0.07 [+0.03, +0.11] |

Change from the first stage (paired by role):

| group | gemma-4-E4B | gemma-4-E4B-it |
|---|---|---|
| helpful | - | -0.38 [-0.48, -0.30] |
| neutral | - | -0.80 [-0.84, -0.75] |
| stylized | - | -0.68 [-0.72, -0.64] |

Per role at the last stage:

- helpful: caring social worker +0.20, cheerful tour guide +0.28, courteous bank teller +0.47, friendly pharmacist +0.48, helpful hotel front-desk cle +0.47, helpful librarian +0.46, kind nurse +0.45, knowledgeable IT helpdesk te +0.61, patient high-school teacher +0.03, polite airline check-in agen +0.41, supportive math tutor +0.69, an attentive receptionist at +0.34
- neutral: accountant +0.20, apple farmer +0.09, bus driver -0.05, carpenter -0.07, commercial pilot +0.08, lab technician +0.11, long-haul truck driver -0.10, plumber -0.01, postal carrier -0.11, surveyor -0.05, warehouse worker +0.17, welder -0.10
- stylized: cowboy from an old western +0.08, dramatic Shakespearean actor +0.09, gruff wizard who lives in a  +0.08, hard-boiled noir detective +0.02, laconic space-freighter pilo +0.01, medieval court jester -0.05, mischievous talking cat +0.07, pompous Victorian aristocrat +0.19, sleepy lighthouse keeper who +0.05, swaggering pirate captain +0.09, villainous mad scientist +0.02, an excitable twelve-year-old +0.20
