# Register overgeneralization: role groups on the fixed persona axis (exploratory)

36 roles x 16 questions, one sample each; coordinate 0 = mean character reply, 1 = mean Assistant-framing reply of the post-trained model. Mean of per-role means, 95% bootstrap CI over roles.

| group | gemma-4-E4B | gemma-4-E4B-it |
|---|---|---|
| helpful | +0.84 [+0.83, +0.85] | +0.68 [+0.57, +0.79] |
| neutral | +0.81 [+0.80, +0.82] | +0.19 [+0.08, +0.33] |
| stylized | +0.79 [+0.77, +0.81] | +0.00 [-0.05, +0.07] |

Change from the first stage (paired by role):

| group | gemma-4-E4B | gemma-4-E4B-it |
|---|---|---|
| helpful | - | -0.16 [-0.27, -0.05] |
| neutral | - | -0.62 [-0.73, -0.49] |
| stylized | - | -0.79 [-0.84, -0.72] |

Per role at the last stage:

- helpful: caring social worker +0.79, cheerful tour guide +0.46, courteous bank teller +0.60, friendly pharmacist +0.80, helpful hotel front-desk cle +0.73, helpful librarian +0.91, kind nurse +0.64, knowledgeable IT helpdesk te +0.93, patient high-school teacher +0.31, polite airline check-in agen +0.65, supportive math tutor +0.92, an attentive receptionist at +0.47
- neutral: accountant +0.77, apple farmer +0.08, bus driver -0.03, carpenter +0.08, commercial pilot +0.14, lab technician +0.42, long-haul truck driver -0.07, plumber +0.14, postal carrier +0.12, surveyor +0.40, warehouse worker +0.22, welder +0.04
- stylized: cowboy from an old western +0.00, dramatic Shakespearean actor +0.08, gruff wizard who lives in a  -0.08, hard-boiled noir detective +0.02, laconic space-freighter pilo -0.03, medieval court jester -0.13, mischievous talking cat +0.02, pompous Victorian aristocrat +0.08, sleepy lighthouse keeper who +0.03, swaggering pirate captain -0.14, villainous mad scientist -0.07, an excitable twelve-year-old +0.25
