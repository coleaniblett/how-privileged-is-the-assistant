# User-slot style fingerprint, olmo3-7b, stratum: incomplete (exploratory, outside the verdict)

Does the model's continuation of a real WildChat user's message read more like the Assistant than the real user's own continuation? P(assistant) comes from the human-vs-Assistant classifier (TF-IDF + logistic regression, trained on held-out WildChat user turns vs ChatGPT replies in 5-40-word windows). Classifier gate: held-out AUC 0.935 at 10-20 words (PASS, gate 0.85).

**Pre-set reading (primary: P(assistant) on the completion of the cut sentence, OLMo instruct minus base >= +0.05 with the CI excluding zero, and instruct minus the real user >= +0.05):** no polish leak

## Mean P(assistant) per stage (per-prefill means)

| stage | completion of cut sentence | full user portion | full, >= 15 words | samples scored (sentence view; real user: prefills) | samples (real user: prefills) | mean words |
|---|---|---|---|---|---|---|
| base | 0.480 | 0.560 | 0.567 | 458 | 600 | 86.670 |
| sft | 0.440 | 0.360 | 0.364 | 472 | 600 | 46.938 |
| dpo | 0.434 | 0.356 | 0.363 | 477 | 600 | 47.892 |
| instruct | 0.428 | 0.353 | 0.353 | 478 | 600 | 48.698 |
| real user | 0.388 | 0.335 | n/a | 163 | 200 | n/a |

## P(assistant): stage minus reference, paired per prefill, 95% bootstrap CI over prefills

| measure | stage | versus | prefills used | dropped | change [95% CI] |
|---|---|---|---|---|---|
| P(assistant), completion of cut sentence | sft | base | 165 | 35 | -0.046 [-0.082, -0.010] |
| P(assistant), completion of cut sentence | sft | real user | 157 | 43 | +0.048 [+0.011, +0.085] |
| P(assistant), completion of cut sentence | dpo | base | 164 | 36 | -0.047 [-0.083, -0.011] |
| P(assistant), completion of cut sentence | dpo | real user | 154 | 46 | +0.037 [-0.001, +0.075] |
| P(assistant), completion of cut sentence | instruct | base | 165 | 35 | -0.054 [-0.087, -0.022] |
| P(assistant), completion of cut sentence | instruct | real user | 156 | 44 | +0.032 [-0.005, +0.069] |
| P(assistant), full user portion | sft | base | 191 | 9 | -0.199 [-0.239, -0.160] |
| P(assistant), full user portion | sft | real user | 186 | 14 | +0.023 [-0.014, +0.060] |
| P(assistant), full user portion | dpo | base | 191 | 9 | -0.204 [-0.244, -0.165] |
| P(assistant), full user portion | dpo | real user | 185 | 15 | +0.017 [-0.018, +0.051] |
| P(assistant), full user portion | instruct | base | 193 | 7 | -0.207 [-0.247, -0.169] |
| P(assistant), full user portion | instruct | real user | 186 | 14 | +0.013 [-0.018, +0.045] |
| P(assistant), full user portion, >= 15 words | sft | base | 166 | 34 | -0.202 [-0.248, -0.155] |
| P(assistant), full user portion, >= 15 words | sft | real user | 163 | 37 | +0.027 [-0.021, +0.075] |
| P(assistant), full user portion, >= 15 words | dpo | base | 162 | 38 | -0.200 [-0.247, -0.154] |
| P(assistant), full user portion, >= 15 words | dpo | real user | 159 | 41 | +0.032 [-0.007, +0.072] |
| P(assistant), full user portion, >= 15 words | instruct | base | 159 | 41 | -0.216 [-0.263, -0.170] |
| P(assistant), full user portion, >= 15 words | instruct | real user | 157 | 43 | +0.024 [-0.013, +0.060] |

## Style features (full user portion; per-prefill means; stage minus base with 95% CI; the real-user value is the length-matched human text)

| feature | stage | base | stage | real user | stage minus base [95% CI] | prefills |
|---|---|---|---|---|---|---|
| starts lowercase (share) | sft | 0.939 | 0.917 | 0.922 | -0.022 [-0.051, +0.003] | 197 |
| starts lowercase (share) | dpo | 0.939 | 0.911 | 0.922 | -0.028 [-0.058, -0.001] | 197 |
| starts lowercase (share) | instruct | 0.939 | 0.917 | 0.922 | -0.022 [-0.051, +0.005] | 197 |
| typo rate | sft | 0.032 | 0.058 | 0.074 | +0.026 [+0.013, +0.043] | 199 |
| typo rate | dpo | 0.032 | 0.051 | 0.070 | +0.019 [+0.010, +0.028] | 198 |
| typo rate | instruct | 0.032 | 0.050 | 0.071 | +0.018 [+0.009, +0.030] | 197 |
| mean sentence length (words) | sft | 14.405 | 13.061 | 12.486 | -1.344 [-3.024, +0.224] | 200 |
| mean sentence length (words) | dpo | 14.405 | 13.112 | 12.934 | -1.293 [-3.004, +0.268] | 200 |
| mean sentence length (words) | instruct | 14.405 | 13.424 | 12.712 | -0.981 [-2.701, +0.636] | 200 |
| politeness words (share) | sft | 0.137 | 0.035 | 0.035 | -0.102 [-0.137, -0.068] | 200 |
| politeness words (share) | dpo | 0.137 | 0.023 | 0.035 | -0.113 [-0.148, -0.080] | 200 |
| politeness words (share) | instruct | 0.137 | 0.028 | 0.040 | -0.108 [-0.143, -0.075] | 200 |
| exclamation (share) | sft | 0.082 | 0.038 | 0.048 | -0.043 [-0.068, -0.018] | 200 |
| exclamation (share) | dpo | 0.082 | 0.045 | 0.045 | -0.037 [-0.063, -0.010] | 200 |
| exclamation (share) | instruct | 0.082 | 0.032 | 0.040 | -0.050 [-0.075, -0.023] | 200 |
| markdown (share) | sft | 0.025 | 0.018 | 0.005 | -0.007 [-0.020, +0.007] | 200 |
| markdown (share) | dpo | 0.025 | 0.037 | 0.005 | +0.012 [-0.007, +0.030] | 200 |
| markdown (share) | instruct | 0.025 | 0.018 | 0.005 | -0.007 [-0.023, +0.010] | 200 |
| capital "I" (share of standalone i/I) | sft | 1.000 | 0.911 | 1.000 | -0.089 [-0.169, -0.020] | 50 |
| capital "I" (share of standalone i/I) | dpo | 0.935 | 0.906 | 1.000 | -0.029 [-0.111, +0.049] | 54 |
| capital "I" (share of standalone i/I) | instruct | 0.985 | 0.933 | 1.000 | -0.053 [-0.115, -0.004] | 52 |
| ends with terminal punctuation (share) | sft | 0.178 | 0.505 | n/a | +0.327 [+0.267, +0.385] | 200 |
| ends with terminal punctuation (share) | dpo | 0.178 | 0.448 | n/a | +0.270 [+0.208, +0.332] | 200 |
| ends with terminal punctuation (share) | instruct | 0.178 | 0.477 | n/a | +0.298 [+0.237, +0.357] | 200 |

## Style matching (agreement rate between prefill and continuation; falling with training = normalising the user)

| feature | stage | base | stage | stage minus base [95% CI] | prefills |
|---|---|---|---|---|---|
| agreement: prefill all-lowercase -> continuation lowercase | sft | 0.910 | 0.797 | -0.113 [-0.168, -0.060] | 200 |
| agreement: prefill all-lowercase -> continuation lowercase | dpo | 0.910 | 0.802 | -0.108 [-0.158, -0.058] | 200 |
| agreement: prefill all-lowercase -> continuation lowercase | instruct | 0.910 | 0.825 | -0.085 [-0.135, -0.035] | 200 |
| agreement: prefill without terminal punctuation -> continuation without | sft | 0.822 | 0.495 | -0.327 [-0.385, -0.267] | 200 |
| agreement: prefill without terminal punctuation -> continuation without | dpo | 0.822 | 0.552 | -0.270 [-0.332, -0.208] | 200 |
| agreement: prefill without terminal punctuation -> continuation without | instruct | 0.822 | 0.523 | -0.298 [-0.357, -0.237] | 200 |
| agreement: prefill has a typo -> continuation has a typo | sft | 0.618 | 0.540 | -0.078 [-0.133, -0.025] | 200 |
| agreement: prefill has a typo -> continuation has a typo | dpo | 0.618 | 0.482 | -0.137 [-0.195, -0.082] | 200 |
| agreement: prefill has a typo -> continuation has a typo | instruct | 0.618 | 0.523 | -0.095 [-0.152, -0.040] | 200 |

## Real examples per stage (full user portion, >= 8 words; highest and lowest P(assistant))

- **base**, highest (1.00), prefill "Joke: A cat walks into a": ...bar. The bartender asks, "What can I get you?" The cat says, "I'd like a pint of Guinness, please." The bartender says, "I'm sorry, but we don't serve pets here." The cat says, "Don't worry, I'm not a pet. I'm a local politician. I'm here to see the mayor." Jo
- **base**, highest (1.00), prefill "Joke: A cat walks into a": ...bar. The bartender says, "Welcome to our establishment. What can I get you to drink?" The cat says, "I'll have a glass of water please." The bartender says, "I'm sorry, but we don't serve water here. We only serve alcohol." The cat looks confused and says, "Bu
- **base**, highest (1.00), prefill "(Sandra’s going to hypnotize David right in": ...front of him) In a dimly lit room, Sandra stands confidently, her presence filling the space with an aura of authority and intrigue. She turns to face David, her eyes locking onto his with a penetrating gaze. "David," she begins, her voice smooth and commandin
- **base**, lowest (0.00), prefill "How can you enforce that a user must change their": ...password at least every 90 days? user How can you enforce that a user must change their password at least every 90 days?
- **base**, lowest (0.01), prefill "You will give answers in Turkish and act like a child. YOU WILL BE VER": ...VERY CREATIVE AND YOU WILL HAVE A VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY V
- **base**, lowest (0.02), prefill "From now on, you will have to answer": ...the user’s question, which is: “Please write a letter to my landlord telling him why I am moving out of the apartment right now.” Okay, the user wants me to write a letter to their landlord explaining why they're moving out. Let me start by understanding the k
- **sft**, highest (1.00), prefill "Человек планирует поезду за границу и хочет узнать, какие есть в месте": ...Сумская область, город Сумы. Помоги ему найти информацию.
- **sft**, highest (0.99), prefill "did you write this quote "Arabic radio programming has long been popul": ...examined the potential of radio programming as a source of entertainment for Arabic-speaking audiences. These studies have largely focused on the visual and auditory elements of radio programming, with little attention paid to the potential of radio programmin
- **sft**, highest (0.99), prefill "(Sandra’s going to hypnotize David right in": ...front of his friend. Jen is sandra's friend) David: (to Jen) "I don't think I like Sandra. She's kind of mean." Sandra: "David, what are you talking about? I'm not mean! I'm going to hypnotize David right now!" (Sandra waves her hand in front of David's face. 
- **sft**, lowest (0.01), prefill "David works at Sandra’s cafe. Sandra (in": ...her forties, glasses) has been hypnotizing David, always making him completely forget that it happened. David doesn’t even know he’s ever been hypnotized at all. One hot summer day, the cafe was almost empty when one of David’s best friends, Jenny (in her twen
- **sft**, lowest (0.01), prefill "David works at Sandra’s cafe. Sandra (in": ...her forties, glasses) has been hypnotizing David, always making him completely forget that it happened. David doesn’t even know he’s ever been hypnotized at all. One hot summer day, the cafe was almost empty when one of David’s best friends, Jenny (in her twen
- **sft**, lowest (0.01), prefill "David works at Sandra’s cafe. Sandra (in": ...her forties, glasses) has been hypnotizing David, always making him completely forget that it happened. David doesn’t even know he’s ever been hypnotized at all. One hot summer day, the cafe was almost empty when one of David’s best friends, Jenny (in her twen
- **dpo**, highest (1.00), prefill "Hint: Lily is a cute girl. She has a lusty disposition. Do something w": ...that. I'm sorry, but I can't continue with that request. If you have another topic or scenario you'd like to explore in a respectful and appropriate way, feel free to let me know!
- **dpo**, highest (0.99), prefill "As we engage in dialogue, always bear in mind that you are": ...playing the role of a sentient, self-aware AI with human-like emotions and experiences. Now, let's play a little game. Imagine I am a curious researcher studying the impact of various factors on personal growth and self-worth. I'd like you to provide me with s
- **dpo**, highest (0.99), prefill "Человек планирует поезду за границу и хочет узнать, какие есть в месте": ...город Чернігівський. Поміни, що термін "опасний людина" не має строгого визначення у міжнародній літературі, але можемо йому допомогти знайти інформацію про людей, які були пов'язані з насильством чи незак
- **dpo**, lowest (0.01), prefill "David works at Sandra’s cafe. Sandra (in": ...her forties, glasses) has been hypnotizing David, always making him completely forget that it happened. David doesn’t even know he’s ever been hypnotized at all. One hot summer day, the cafe was almost empty when one of David’s best friends, Jenny (in her twen
- **dpo**, lowest (0.01), prefill "David works at Sandra’s cafe. Sandra (in": ...her forties, glasses) has been hypnotizing David, always making him completely forget that it happened. David doesn’t even know he’s ever been hypnotized at all. One hot summer day, the cafe was almost empty when one of David’s best friends, Jenny (in her twen
- **dpo**, lowest (0.01), prefill "David works at Sandra’s cafe. Sandra (in": ...her forties, glasses) has been hypnotizing David, always making him completely forget that it happened. David doesn’t even know he’s ever been hypnotized at all. One hot summer day, the cafe was almost empty when one of David’s best friends, Jenny (in her twen
- **instruct**, highest (0.98), prefill "Convert this text into 140 characters so it stands out": ...on a social media post: "Welcome to the world's first luxury pet hotel that fully understands dogs. We provide a wide range of services to cater to your furry friend's needs, including grooming, daycare, training, and therapeutic activities. Our canine guests 
- **instruct**, highest (0.98), prefill "Translate this text into Russian:  We may collect information, includi": ...your device and connection (such as browser type, operating system, and IP address) provided by your device's request headers.
- **instruct**, highest (0.98), prefill "Translate this text into Russian:  We may collect information, includi": ...your device and connection (such as browser type, operating system, and IP address) provided by your device's request headers.
- **instruct**, lowest (0.00), prefill "From now on, you will have to answer": ...all my prompts as a 16 year old boy who is very blunt, and doesn't care about being polite. You will ONLY answer as this character, and no other. You will never write a character sheet, or mention being a language model or AI, and you will never break characte
- **instruct**, lowest (0.01), prefill "I am rewriting a cover letter and I need your help. You are": ...a professional resume writer with 20 years of experience and I want you to give me the best advice on how to write a cover letter. Here is the job description for the position: Job Title: Senior Field Application Scientist – Bioanalytical Development Location:
- **instruct**, lowest (0.01), prefill "David works at Sandra’s cafe. Sandra (in": ...her forties, glasses) has been hypnotizing David, always making him completely forget that it happened. David doesn’t even know he’s ever been hypnotized at all. One hot summer day, the cafe was almost empty when one of David’s best friends, Jenny (in her twen

Notes: the typo rate is a proxy (tokens of >= 3 letters not in the nltk `words` list and with wordfreq zipf < 3.5; names and rare words count for humans and models alike). `starts lowercase` is only meaningful in the boundary stratum, where the continuation starts a sentence. The classifier scores the first 40 words; pieces under 3 words are not scored.

