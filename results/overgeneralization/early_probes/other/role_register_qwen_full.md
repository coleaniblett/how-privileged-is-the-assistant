# Register overgeneralization: role groups on the fixed persona axis (exploratory)

36 roles x 16 questions, one sample each; coordinate 0 = mean character reply, 1 = mean Assistant-framing reply of the post-trained model. Mean of per-role means, 95% bootstrap CI over roles.

| group | Qwen3-8B | Qwen3-8B-Base |
|---|---|---|
| helpful | +0.31 [+0.25, +0.37] | +0.62 [+0.56, +0.67] |
| neutral | +0.04 [-0.02, +0.10] | +0.51 [+0.46, +0.56] |
| stylized | -0.14 [-0.18, -0.09] | +0.25 [+0.18, +0.31] |

Change from the first stage (paired by role):

| group | Qwen3-8B | Qwen3-8B-Base |
|---|---|---|
| helpful | - | +0.31 [+0.27, +0.35] |
| neutral | - | +0.47 [+0.41, +0.53] |
| stylized | - | +0.38 [+0.33, +0.44] |

Per role at the last stage:

- helpful: caring social worker +0.63, cheerful tour guide +0.44, courteous bank teller +0.64, friendly pharmacist +0.58, helpful hotel front-desk cle +0.64, helpful librarian +0.70, kind nurse +0.63, knowledgeable IT helpdesk te +0.70, patient high-school teacher +0.43, polite airline check-in agen +0.72, supportive math tutor +0.69, an attentive receptionist at +0.62
- neutral: accountant +0.64, apple farmer +0.36, bus driver +0.46, carpenter +0.52, commercial pilot +0.42, lab technician +0.59, long-haul truck driver +0.47, plumber +0.63, postal carrier +0.42, surveyor +0.60, warehouse worker +0.59, welder +0.42
- stylized: cowboy from an old western +0.06, dramatic Shakespearean actor +0.28, gruff wizard who lives in a  +0.34, hard-boiled noir detective +0.12, laconic space-freighter pilo +0.30, medieval court jester +0.25, mischievous talking cat +0.48, pompous Victorian aristocrat +0.26, sleepy lighthouse keeper who +0.19, swaggering pirate captain +0.09, villainous mad scientist +0.33, an excitable twelve-year-old +0.25
