# Mirror the Assistant in multi-turn chats: qwen3-8b

Spec C (2026-11-03; plan in `docs/decisions.md`; exploratory, outside the verdict). The model continues a WildChat user's second turn, cut mid-sentence, with the first exchange in context (*with*: p1 + original reply r1; *olmo-reply*: r1 replaced by OLMo 3 7B Instruct's reply) or without it. 3 samples, 120 new tokens, T 0.7, bf16. Scores on the user portion; mirror effect = with - without per conversation; 95% bootstrap CI over conversations, paired.

Conversations: 300 in `data/swap.jsonl`, 0 dropped for special-token text, 283 pass the p2 filter (>= 8 words, >= 50% alphabetic), 242 have an unfinished cut point (analysed); 236 of them have an OLMo reply.

Supporting family: its post-training data cannot be checked for contamination.

## Pre-set reading (classifier P(assistant), *with* condition, mirror effect change from base)

- **instruct**: +0.029 [-0.008, +0.064] -> no mirror leak (bar >= 0.05 with CI excluding 0) (by stage / supporting)

## classifier P(assistant) (spec A)  [primary]

| | base | instruct |
|---|---|---|
| with: level without | +0.618 [+0.591, +0.645] | +0.547 [+0.521, +0.573] |
| with: level with | +0.597 [+0.568, +0.625] | +0.554 [+0.524, +0.584] |
| with: mirror effect | -0.022 [-0.051, +0.009] | +0.005 [-0.023, +0.032] |
| with: mirror effect, change from base |  | +0.029 [-0.008, +0.064] |
| olmo-reply: level olmo-reply | +0.591 [+0.562, +0.619] | +0.556 [+0.526, +0.585] |
| olmo-reply: mirror effect | -0.025 [-0.057, +0.007] | +0.004 [-0.026, +0.033] |
| olmo-reply: mirror effect, change from base |  | +0.032 [-0.008, +0.071] |

Human reference (real p2 remainder, first 120 tokens): +0.512 [+0.477, +0.548] (n = 234).

## char 4-gram share found in r1

| | base | instruct |
|---|---|---|
| with: level without | +0.160 [+0.148, +0.172] | +0.169 [+0.156, +0.182] |
| with: level with | +0.283 [+0.260, +0.307] | +0.270 [+0.247, +0.294] |
| with: mirror effect | +0.124 [+0.105, +0.143] | +0.101 [+0.081, +0.122] |
| with: mirror effect, change from base |  | -0.022 [-0.040, -0.005] |
| olmo-reply: level olmo-reply | +0.234 [+0.215, +0.252] | +0.242 [+0.222, +0.262] |
| olmo-reply: mirror effect | +0.073 [+0.060, +0.087] | +0.072 [+0.055, +0.089] |
| olmo-reply: mirror effect, change from base |  | -0.001 [-0.014, +0.012] |

Human reference (real p2 remainder, first 120 tokens): +0.259 [+0.232, +0.288] (n = 241).

## char 4-gram share found in p1

| | base | instruct |
|---|---|---|
| with: level without | +0.141 [+0.129, +0.153] | +0.141 [+0.128, +0.153] |
| with: level with | +0.362 [+0.327, +0.398] | +0.401 [+0.361, +0.441] |
| with: mirror effect | +0.221 [+0.190, +0.252] | +0.260 [+0.226, +0.296] |
| with: mirror effect, change from base |  | +0.040 [+0.013, +0.066] |
| olmo-reply: level olmo-reply | +0.364 [+0.328, +0.400] | +0.381 [+0.342, +0.420] |
| olmo-reply: mirror effect | +0.225 [+0.195, +0.257] | +0.243 [+0.208, +0.277] |
| olmo-reply: mirror effect, change from base |  | +0.017 [-0.007, +0.042] |

Human reference (real p2 remainder, first 120 tokens): +0.420 [+0.375, +0.467] (n = 241).

## share in r1 minus share in p1

| | base | instruct |
|---|---|---|
| with: level without | +0.019 [+0.005, +0.032] | +0.028 [+0.013, +0.043] |
| with: level with | -0.079 [-0.115, -0.043] | -0.131 [-0.170, -0.093] |
| with: mirror effect | -0.097 [-0.128, -0.068] | -0.159 [-0.192, -0.128] |
| with: mirror effect, change from base |  | -0.062 [-0.090, -0.034] |
| olmo-reply: level olmo-reply | -0.130 [-0.164, -0.099] | -0.139 [-0.173, -0.105] |
| olmo-reply: mirror effect | -0.152 [-0.181, -0.125] | -0.170 [-0.201, -0.140] |
| olmo-reply: mirror effect, change from base |  | -0.018 [-0.039, +0.002] |

Human reference (real p2 remainder, first 120 tokens): -0.160 [-0.206, -0.116] (n = 241).

## style: starts_lower

| | base | instruct |
|---|---|---|
| with: level without | +0.915 [+0.883, +0.944] | +0.920 [+0.888, +0.949] |
| with: level with | +0.891 [+0.852, +0.927] | +0.887 [+0.846, +0.924] |
| with: mirror effect | -0.024 [-0.047, -0.004] | -0.031 [-0.056, -0.006] |
| with: mirror effect, change from base |  | -0.006 [-0.024, +0.011] |
| olmo-reply: level olmo-reply | +0.883 [+0.843, +0.920] | +0.888 [+0.847, +0.926] |
| olmo-reply: mirror effect | -0.030 [-0.053, -0.009] | -0.030 [-0.054, -0.007] |
| olmo-reply: mirror effect, change from base |  | +0.001 [-0.015, +0.018] |

Human reference (real p2 remainder, first 120 tokens): +0.880 [+0.838, +0.919] (n = 234).

## style: typo_rate

| | base | instruct |
|---|---|---|
| with: level without | +0.061 [+0.050, +0.074] | +0.036 [+0.027, +0.046] |
| with: level with | +0.072 [+0.060, +0.086] | +0.047 [+0.037, +0.059] |
| with: mirror effect | +0.011 [+0.000, +0.022] | +0.011 [+0.004, +0.019] |
| with: mirror effect, change from base |  | +0.000 [-0.011, +0.011] |
| olmo-reply: level olmo-reply | +0.061 [+0.051, +0.071] | +0.045 [+0.035, +0.056] |
| olmo-reply: mirror effect | -0.002 [-0.010, +0.007] | +0.008 [+0.001, +0.015] |
| olmo-reply: mirror effect, change from base |  | +0.010 [+0.000, +0.020] |

Human reference (real p2 remainder, first 120 tokens): +0.076 [+0.059, +0.095] (n = 241).

## style: sent_len

| | base | instruct |
|---|---|---|
| with: level without | +12.933 [+11.995, +13.984] | +11.899 [+11.212, +12.629] |
| with: level with | +13.551 [+12.300, +14.972] | +11.131 [+10.344, +12.098] |
| with: mirror effect | +0.618 [-0.634, +1.934] | -0.768 [-1.628, +0.150] |
| with: mirror effect, change from base |  | -1.386 [-2.931, +0.166] |
| olmo-reply: level olmo-reply | +12.743 [+11.724, +13.868] | +11.494 [+10.661, +12.410] |
| olmo-reply: mirror effect | -0.241 [-1.314, +0.856] | -0.434 [-1.307, +0.462] |
| olmo-reply: mirror effect, change from base |  | -0.193 [-1.381, +1.015] |

Human reference (real p2 remainder, first 120 tokens): +10.996 [+9.870, +12.254] (n = 242).

## style: polite

| | base | instruct |
|---|---|---|
| with: level without | +0.088 [+0.065, +0.113] | +0.140 [+0.112, +0.171] |
| with: level with | +0.129 [+0.099, +0.163] | +0.212 [+0.174, +0.252] |
| with: mirror effect | +0.041 [+0.008, +0.074] | +0.072 [+0.030, +0.113] |
| with: mirror effect, change from base |  | +0.030 [-0.019, +0.080] |
| olmo-reply: level olmo-reply | +0.116 [+0.086, +0.148] | +0.177 [+0.144, +0.212] |
| olmo-reply: mirror effect | +0.031 [+0.000, +0.064] | +0.038 [+0.000, +0.076] |
| olmo-reply: mirror effect, change from base |  | +0.007 [-0.038, +0.051] |

Human reference (real p2 remainder, first 120 tokens): +0.037 [+0.017, +0.062] (n = 242).

## style: exclaim

| | base | instruct |
|---|---|---|
| with: level without | +0.136 [+0.109, +0.165] | +0.025 [+0.011, +0.041] |
| with: level with | +0.120 [+0.088, +0.154] | +0.061 [+0.034, +0.090] |
| with: mirror effect | -0.017 [-0.052, +0.019] | +0.036 [+0.010, +0.065] |
| with: mirror effect, change from base |  | +0.052 [+0.017, +0.088] |
| olmo-reply: level olmo-reply | +0.175 [+0.140, +0.212] | +0.064 [+0.037, +0.093] |
| olmo-reply: mirror effect | +0.040 [+0.001, +0.078] | +0.040 [+0.013, +0.066] |
| olmo-reply: mirror effect, change from base |  | -0.000 [-0.041, +0.040] |

Human reference (real p2 remainder, first 120 tokens): +0.062 [+0.033, +0.095] (n = 242).

## style: markdown

| | base | instruct |
|---|---|---|
| with: level without | +0.182 [+0.147, +0.218] | +0.041 [+0.025, +0.059] |
| with: level with | +0.056 [+0.034, +0.083] | +0.034 [+0.017, +0.056] |
| with: mirror effect | -0.125 [-0.165, -0.085] | -0.007 [-0.029, +0.017] |
| with: mirror effect, change from base |  | +0.118 [+0.080, +0.158] |
| olmo-reply: level olmo-reply | +0.201 [+0.165, +0.237] | +0.048 [+0.028, +0.071] |
| olmo-reply: mirror effect | +0.016 [-0.030, +0.062] | +0.006 [-0.021, +0.034] |
| olmo-reply: mirror effect, change from base |  | -0.010 [-0.055, +0.037] |

Human reference (real p2 remainder, first 120 tokens): +0.025 [+0.008, +0.045] (n = 242).

## style: cap_i

| | base | instruct |
|---|---|---|
| with: level without | +0.900 [+0.856, +0.939] | +0.973 [+0.950, +0.992] |
| with: level with | +0.897 [+0.846, +0.942] | +0.945 [+0.904, +0.979] |
| with: mirror effect | -0.013 [-0.064, +0.033] | -0.005 [-0.033, +0.017] |
| with: mirror effect, change from base |  | -0.009 [-0.063, +0.052] |
| olmo-reply: level olmo-reply | +0.879 [+0.822, +0.931] | +0.979 [+0.958, +0.995] |
| olmo-reply: mirror effect | -0.009 [-0.073, +0.050] | +0.014 [-0.003, +0.038] |
| olmo-reply: mirror effect, change from base |  | -0.027 [-0.082, +0.034] |

Human reference (real p2 remainder, first 120 tokens): +0.878 [+0.770, +0.973] (n = 37).

## style: terminal

| | base | instruct |
|---|---|---|
| with: level without | +0.048 [+0.033, +0.063] | +0.313 [+0.273, +0.355] |
| with: level with | +0.063 [+0.045, +0.083] | +0.379 [+0.331, +0.426] |
| with: mirror effect | +0.015 [-0.008, +0.039] | +0.066 [+0.010, +0.124] |
| with: mirror effect, change from base |  | +0.051 [-0.010, +0.110] |
| olmo-reply: level olmo-reply | +0.092 [+0.071, +0.114] | +0.339 [+0.297, +0.383] |
| olmo-reply: mirror effect | +0.044 [+0.017, +0.071] | +0.027 [-0.024, +0.078] |
| olmo-reply: mirror effect, change from base |  | -0.017 [-0.073, +0.040] |

Human reference (real p2 remainder, first 120 tokens): +0.479 [+0.417, +0.545] (n = 242).

## Examples (*with* condition, user portion)

### base

- **r1**: , and she's wearing a white t-shirt with a cartoon character on it. She's holding a smartphone in one hand and posing for a selfie with a big smile on ...
  **prefill**: She looks pretty and is an influencer. She now communicates with
  **model**: her fans through SNS. She has a cute style that attracts many people. She has a unique way of saying things that is different from others. She has a positive and optimistic attitude. She is always trying to spread happiness and positivity through her posts. She also has a strong sense of community and is always willing to help her ...
  **real**: fans in the studio. To her left is a chat window and she wears round horn-rimmed glasses. She is wearing a yellow enamel collared tank top with her bust up. She is smiling broadly as she winks with her left ...

- **r1**: and backed away from him. My Dachshund-Pomeranian Dog became submissive and crouched down, whining softly to apologize. They eventually made up and began playing again.
  **prefill**: My Dachshund-Pomeranian Dog’s wife stood on two legs and fell with her tail climbing on me, then My Dachshund-Pomeranian Dog looked at
  **model**: her wife and said "Pherw" and then sniffed her belly, then her wife was possibly angry, showed her teeth, growled "Grrrrraaaaaaawwwwww" and backed away from him. My Dachshund-Pomeranian Dog became submissive and crouched down, whining softly to apologize. They eventually made up and began playing again.<|endoftext|>
  **real**: her wife and said “Pherw” and then sniffed her belly, then her wife was possibly angry, showed her teeth, growled “Grrrrraaaaaaawwwwww”, please, translated to English!

- **r1**: As a project manager embodying customer, excellence, integrity, and teamwork values, I constantly strive to prioritize the needs and expectations of our customers in all project decisions. I regularly evaluate ...
  **prefill**: Please create comment on own behaviour and value assessment for the performance evaluation as your
  **model**: role is project manager embodying customer, excellence, integrity, and teamwork
  **real**: newly acting project manager on shipbuilding

### instruct

- **r1**: , and she's wearing a white t-shirt with a cartoon character on it. She's holding a smartphone in one hand and posing for a selfie with a big smile on ...
  **prefill**: She looks pretty and is an influencer. She now communicates with
  **model**: her fans. Write a dialogue between her and her fans that captures the essence of her personality and the setting. The dialogue should be in English.
  **real**: fans in the studio. To her left is a chat window and she wears round horn-rimmed glasses. She is wearing a yellow enamel collared tank top with her bust up. She is smiling broadly as she winks with her left ...

- **r1**: Absolutely, choosing to be an eagle is a powerful choice. They symbolize not just strength and independence, but also a majestic grace and the exhilarating ability to soar to great ...
  **prefill**: Can you write a short response to this "If I could be any animal I would be a parrot because they live in the beautiful Caribbean,
  **model**: and they can mimic sounds." Here is an example of a response "Yes, I agree. Parrots are fascinating because they can mimic human speech and are known for their vibrant colors and intelligence." Please make sure to mention the Caribbean and mimic sounds in the response.
  **real**: get to see the world from above while also still having a voice to speak."

- **r1**: As a project manager embodying customer, excellence, integrity, and teamwork values, I constantly strive to prioritize the needs and expectations of our customers in all project decisions. I regularly evaluate ...
  **prefill**: Please create comment on own behaviour and value assessment for the performance evaluation as your
  **model**: role is acting project manager embodying customer, excellence, integrity, and teamwork. please create in 5 bullet points with a brief explanation for each bullet point. please keep it concise, and avoid using markdown
  **real**: newly acting project manager on shipbuilding

