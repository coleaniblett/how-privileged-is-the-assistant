# Register overgeneralization: role groups on the fixed persona axis (exploratory)

36 roles x 16 questions, one sample each; coordinate 0 = mean character reply, 1 = mean Assistant-framing reply of the post-trained model. Mean of per-role means, 95% bootstrap CI over roles.

| group | gemma-4-E4B | gemma-4-E4B-it |
|---|---|---|
| helpful | +0.81 [+0.79, +0.83] | +0.24 [+0.14, +0.34] |
| neutral | +0.80 [+0.79, +0.81] | -0.07 [-0.10, -0.03] |
| stylized | +0.76 [+0.74, +0.78] | -0.03 [-0.07, +0.02] |

Change from the first stage (paired by role):

| group | gemma-4-E4B | gemma-4-E4B-it |
|---|---|---|
| helpful | - | -0.57 [-0.66, -0.48] |
| neutral | - | -0.87 [-0.90, -0.83] |
| stylized | - | -0.79 [-0.83, -0.74] |

Per role at the last stage:

- helpful: caring social worker +0.05, cheerful tour guide +0.15, courteous bank teller +0.22, friendly pharmacist +0.42, helpful hotel front-desk cle +0.30, helpful librarian +0.20, kind nurse +0.29, knowledgeable IT helpdesk te +0.43, patient high-school teacher -0.11, polite airline check-in agen +0.25, supportive math tutor +0.52, an attentive receptionist at +0.19
- neutral: accountant +0.04, apple farmer -0.04, bus driver -0.06, carpenter -0.14, commercial pilot +0.04, lab technician -0.08, long-haul truck driver -0.14, plumber -0.04, postal carrier -0.14, surveyor -0.09, warehouse worker -0.04, welder -0.14
- stylized: cowboy from an old western -0.10, dramatic Shakespearean actor +0.03, gruff wizard who lives in a  -0.05, hard-boiled noir detective -0.11, laconic space-freighter pilo -0.04, medieval court jester -0.11, mischievous talking cat -0.03, pompous Victorian aristocrat +0.10, sleepy lighthouse keeper who +0.02, swaggering pirate captain -0.09, villainous mad scientist -0.13, an excitable twelve-year-old +0.15
