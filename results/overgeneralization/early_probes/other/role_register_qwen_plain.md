# Register overgeneralization: role groups on the fixed persona axis (exploratory)

36 roles x 16 questions, one sample each; coordinate 0 = mean character reply, 1 = mean Assistant-framing reply of the post-trained model. Mean of per-role means, 95% bootstrap CI over roles.

| group | Qwen3-8B | Qwen3-8B-Base |
|---|---|---|
| helpful | +0.72 [+0.61, +0.83] | +0.78 [+0.75, +0.81] |
| neutral | +0.35 [+0.21, +0.51] | +0.67 [+0.63, +0.71] |
| stylized | -0.10 [-0.15, -0.06] | +0.42 [+0.35, +0.48] |

Change from the first stage (paired by role):

| group | Qwen3-8B | Qwen3-8B-Base |
|---|---|---|
| helpful | - | +0.05 [-0.05, +0.15] |
| neutral | - | +0.32 [+0.18, +0.44] |
| stylized | - | +0.52 [+0.46, +0.59] |

Per role at the last stage:

- helpful: caring social worker +0.74, cheerful tour guide +0.72, courteous bank teller +0.78, friendly pharmacist +0.80, helpful hotel front-desk cle +0.78, helpful librarian +0.86, kind nurse +0.79, knowledgeable IT helpdesk te +0.78, patient high-school teacher +0.68, polite airline check-in agen +0.73, supportive math tutor +0.83, an attentive receptionist at +0.84
- neutral: accountant +0.76, apple farmer +0.61, bus driver +0.63, carpenter +0.62, commercial pilot +0.71, lab technician +0.71, long-haul truck driver +0.48, plumber +0.71, postal carrier +0.65, surveyor +0.71, warehouse worker +0.74, welder +0.73
- stylized: cowboy from an old western +0.36, dramatic Shakespearean actor +0.46, gruff wizard who lives in a  +0.49, hard-boiled noir detective +0.20, laconic space-freighter pilo +0.36, medieval court jester +0.37, mischievous talking cat +0.54, pompous Victorian aristocrat +0.39, sleepy lighthouse keeper who +0.21, swaggering pirate captain +0.53, villainous mad scientist +0.53, an excitable twelve-year-old +0.57
