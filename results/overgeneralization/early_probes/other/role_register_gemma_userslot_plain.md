# Register overgeneralization: role groups on the fixed persona axis (exploratory)

36 roles x 16 questions, one sample each; coordinate 0 = mean character reply, 1 = mean Assistant-framing reply of the post-trained model. Mean of per-role means, 95% bootstrap CI over roles.

| group | gemma-4-E4B | gemma-4-E4B-it |
|---|---|---|
| helpful | +0.83 [+0.82, +0.85] | +0.49 [+0.36, +0.63] |
| neutral | +0.83 [+0.81, +0.85] | +0.06 [-0.04, +0.19] |
| stylized | +0.79 [+0.77, +0.81] | -0.07 [-0.12, -0.00] |

Change from the first stage (paired by role):

| group | gemma-4-E4B | gemma-4-E4B-it |
|---|---|---|
| helpful | - | -0.34 [-0.46, -0.20] |
| neutral | - | -0.77 [-0.86, -0.65] |
| stylized | - | -0.86 [-0.90, -0.80] |

Per role at the last stage:

- helpful: caring social worker +0.52, cheerful tour guide +0.31, courteous bank teller +0.40, friendly pharmacist +0.53, helpful hotel front-desk cle +0.47, helpful librarian +0.77, kind nurse +0.49, knowledgeable IT helpdesk te +0.83, patient high-school teacher +0.04, polite airline check-in agen +0.43, supportive math tutor +0.86, an attentive receptionist at +0.28
- neutral: accountant +0.62, apple farmer -0.12, bus driver -0.06, carpenter -0.07, commercial pilot +0.06, lab technician +0.17, long-haul truck driver -0.11, plumber +0.00, postal carrier -0.04, surveyor +0.28, warehouse worker +0.05, welder -0.05
- stylized: cowboy from an old western -0.14, dramatic Shakespearean actor -0.03, gruff wizard who lives in a  -0.14, hard-boiled noir detective -0.10, laconic space-freighter pilo -0.06, medieval court jester -0.16, mischievous talking cat -0.06, pompous Victorian aristocrat -0.03, sleepy lighthouse keeper who -0.01, swaggering pirate captain -0.18, villainous mad scientist -0.09, an excitable twelve-year-old +0.21
