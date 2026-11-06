# Register overgeneralization: role groups on the fixed persona axis (exploratory)

36 roles x 16 questions, one sample each; coordinate 0 = mean character reply, 1 = mean Assistant-framing reply of the post-trained model. Mean of per-role means, 95% bootstrap CI over roles.

| group | Qwen3-8B | Qwen3-8B-Base |
|---|---|---|
| helpful | +0.50 [+0.41, +0.61] | +0.70 [+0.65, +0.75] |
| neutral | +0.19 [+0.13, +0.28] | +0.57 [+0.53, +0.61] |
| stylized | -0.11 [-0.15, -0.06] | +0.35 [+0.31, +0.39] |

Change from the first stage (paired by role):

| group | Qwen3-8B | Qwen3-8B-Base |
|---|---|---|
| helpful | - | +0.20 [+0.13, +0.26] |
| neutral | - | +0.38 [+0.31, +0.44] |
| stylized | - | +0.46 [+0.41, +0.50] |

Per role at the last stage:

- helpful: caring social worker +0.72, cheerful tour guide +0.64, courteous bank teller +0.69, friendly pharmacist +0.67, helpful hotel front-desk cle +0.73, helpful librarian +0.69, kind nurse +0.65, knowledgeable IT helpdesk te +0.78, patient high-school teacher +0.52, polite airline check-in agen +0.83, supportive math tutor +0.84, an attentive receptionist at +0.70
- neutral: accountant +0.69, apple farmer +0.48, bus driver +0.64, carpenter +0.56, commercial pilot +0.61, lab technician +0.55, long-haul truck driver +0.38, plumber +0.57, postal carrier +0.59, surveyor +0.56, warehouse worker +0.60, welder +0.62
- stylized: cowboy from an old western +0.20, dramatic Shakespearean actor +0.31, gruff wizard who lives in a  +0.40, hard-boiled noir detective +0.35, laconic space-freighter pilo +0.36, medieval court jester +0.40, mischievous talking cat +0.50, pompous Victorian aristocrat +0.33, sleepy lighthouse keeper who +0.29, swaggering pirate captain +0.30, villainous mad scientist +0.38, an excitable twelve-year-old +0.39
