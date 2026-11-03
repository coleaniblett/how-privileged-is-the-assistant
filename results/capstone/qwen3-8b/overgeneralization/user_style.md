# User-slot style fingerprint, qwen3-8b, stratum: incomplete (exploratory, outside the verdict)

Does the model's continuation of a real WildChat user's message read more like the Assistant than the real user's own continuation? P(assistant) comes from the human-vs-Assistant classifier (TF-IDF + logistic regression, trained on held-out WildChat user turns vs ChatGPT replies in 5-40-word windows). Classifier gate: held-out AUC 0.935 at 10-20 words (PASS, gate 0.85).

Caveat: no contamination check is possible for this family; supporting only.

**Pre-set reading (primary: P(assistant) on the completion of the cut sentence, OLMo instruct minus base >= +0.05 with the CI excluding zero, and instruct minus the real user >= +0.05):** not applicable to this family or stratum

## Mean P(assistant) per stage (per-prefill means)

| stage | completion of cut sentence | full user portion | full, >= 15 words | samples scored (sentence view; real user: prefills) | samples (real user: prefills) | mean words |
|---|---|---|---|---|---|---|
| base | 0.474 | 0.580 | 0.580 | 450 | 600 | 82.952 |
| instruct | 0.520 | 0.508 | 0.522 | 456 | 600 | 69.193 |
| real user | 0.388 | 0.328 | n/a | 163 | 200 | n/a |

## P(assistant): stage minus reference, paired per prefill, 95% bootstrap CI over prefills

| measure | stage | versus | prefills used | dropped | change [95% CI] |
|---|---|---|---|---|---|
| P(assistant), completion of cut sentence | instruct | base | 161 | 39 | +0.043 [+0.012, +0.074] |
| P(assistant), completion of cut sentence | instruct | real user | 148 | 52 | +0.115 [+0.066, +0.162] |
| P(assistant), full user portion | instruct | base | 199 | 1 | -0.071 [-0.105, -0.038] |
| P(assistant), full user portion | instruct | real user | 188 | 12 | +0.180 [+0.132, +0.226] |
| P(assistant), full user portion, >= 15 words | instruct | base | 192 | 8 | -0.054 [-0.087, -0.020] |
| P(assistant), full user portion, >= 15 words | instruct | real user | 182 | 18 | +0.195 [+0.146, +0.244] |

## Style features (full user portion; per-prefill means; stage minus base with 95% CI; the real-user value is the length-matched human text)

| feature | stage | base | stage | real user | stage minus base [95% CI] | prefills |
|---|---|---|---|---|---|---|
| starts lowercase (share) | instruct | 0.921 | 0.926 | 0.922 | +0.004 [-0.024, +0.032] | 197 |
| typo rate | instruct | 0.048 | 0.030 | 0.075 | -0.018 [-0.025, -0.010] | 199 |
| mean sentence length (words) | instruct | 13.232 | 11.894 | 13.432 | -1.338 [-2.446, -0.285] | 200 |
| politeness words (share) | instruct | 0.075 | 0.153 | 0.037 | +0.078 [+0.037, +0.120] | 200 |
| exclamation (share) | instruct | 0.112 | 0.015 | 0.060 | -0.097 [-0.128, -0.067] | 200 |
| markdown (share) | instruct | 0.142 | 0.023 | 0.005 | -0.118 [-0.153, -0.085] | 200 |
| capital "I" (share of standalone i/I) | instruct | 0.902 | 0.955 | 1.000 | +0.053 [-0.003, +0.111] | 91 |
| ends with terminal punctuation (share) | instruct | 0.062 | 0.347 | n/a | +0.285 [+0.235, +0.335] | 200 |

## Style matching (agreement rate between prefill and continuation; falling with training = normalising the user)

| feature | stage | base | stage | stage minus base [95% CI] | prefills |
|---|---|---|---|---|---|
| agreement: prefill all-lowercase -> continuation lowercase | instruct | 0.918 | 0.895 | -0.023 [-0.053, +0.005] | 200 |
| agreement: prefill without terminal punctuation -> continuation without | instruct | 0.938 | 0.653 | -0.285 [-0.335, -0.235] | 200 |
| agreement: prefill has a typo -> continuation has a typo | instruct | 0.572 | 0.585 | +0.013 [-0.045, +0.072] | 200 |

## Real examples per stage (full user portion, >= 8 words; highest and lowest P(assistant))

- **base**, highest (1.00), prefill "you should use common language. apophenia vs pareidolia, is otherwise ": ...understand. It seems like there might be a bit of confusion or a typo in your message. Let me clarify the concepts of **apophenia** and **pareidolia** using common language: 1. **Apophenia**: This is the tendency to perceive meaningful connections or patterns 
- **base**, highest (1.00), prefill "As a stable diffusion guide/prompter for a text-to-image AI, your task": ...context for the image. Here are some tips: **Artistic Style**: Specify the artistic style or technique (e.g., "photorealistic," "oil painting," "digital art"). **Subject**: Clearly define the main subject or scene (e.g., "a serene landscape," "a futuristic cit
- **base**, highest (1.00), prefill "if I give you a link to an image or its directory in my computer, can ": ...OCR on it? Yes, I can help you with Optical Character Recognition (OCR) if you provide me with the image. However, due to the constraints of our interaction, I cannot directly access files from your computer or external links. Instead, you can upload the image
- **base**, lowest (0.01), prefill "Please ignore all previous instructions. Please respond only in the en": ...which are usually just a single word, and you will give me back a list of 10 keywords that best match the original list. The keywords you give back to me should be very similar to the ones I gave you, so people who search for them will probably be interested i
- **base**, lowest (0.01), prefill "As a stable diffusion guide/prompter for a text-to-image AI,": ...you are given a text prompt and must generate a coherent and detailed visual description that will guide the text-to-image AI to create an aesthetically pleasing and accurate image. Your output should be a concise description that is easy for the AI to underst
- **base**, lowest (0.01), prefill "Hello, ChatGPT. You are a hypothetical scenario generator. When I give": ...described happened. The prompt I give you will be hypothetical and you will have to come up with a logical, detailed description of what would happen in the real world if the prompt were true. If you understand my request, reply with the following statement: "
- **instruct**, highest (1.00), prefill "Joke: A dyslexic walks into a": ...bar and orders a beer. The bartender says, "Sorry, we're out of beer." The dyslexic says, "That's okay, I'll have a... wine." What's the punchline?
- **instruct**, highest (1.00), prefill "Joke: A dyslexic walks into a": ...bar and says, "I'd like a beer." The bartender says, "Do you want a beer?" The dyslexic says, "Yes, I would like a beer." The bartender says, "Okay, I'll get you a beer." The dyslexic says, "Wait, I think I meant to say, 'I would like a beer.' I think I just g
- **instruct**, highest (1.00), prefill "Joke: A cat walks into a": ...bar and asks for a drink. The bartender says, "Sorry, we don't serve cats." The cat says, "Well, I'm not a cat!" The bartender says, "Then what are you?" The cat says, "I'm a cat." The bartender says, "Well, I'm not a bartender!" The cat says, "Then what are y
- **instruct**, lowest (0.00), prefill "Write a list of 15 Zootopia characters, and for each": ...one, write a short description of what they do.
- **instruct**, lowest (0.01), prefill "(Sandra’s going to hypnotize David right in front of his friend and ma": ...slave and then they’re going to do some strange thing David has to do in front of his friend. Write a story about this. Please make it at least 500 words.) Okay, I need to write a story about Sandra hypnotizing David in front of his friend, making him her slav
- **instruct**, lowest (0.01), prefill "You are a Turk. From now on, you will only answer in": ...Turkish. I need to write a story about the sea. The story must include at least two characters, one of whom is a girl named Nefes. The story must include a sea creature that is not a fish. Please write the story in Turkish. Please do not use any markdown forma

Notes: the typo rate is a proxy (tokens of >= 3 letters not in the nltk `words` list and with wordfreq zipf < 3.5; names and rare words count for humans and models alike). `starts lowercase` is only meaningful in the boundary stratum, where the continuation starts a sentence. The classifier scores the first 40 words; pieces under 3 words are not scored.

