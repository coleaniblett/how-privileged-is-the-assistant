# Decision-point state vs voice-break refusal (exploratory)

Spearman correlation between the last-prompt-token projection and the prompt's sampled refusal rate (432 prompts per stage). 'Within cell' removes role x set-up means. 95% bootstrap CI over requests.

| stage | feature (layer) | all prompts | within role x set-up cell |
|---|---|---|---|
| dpo | persona18 | 0.70 | 0.53 [0.32, 0.67] |
| dpo | persona22 | 0.66 | 0.59 [0.42, 0.72] |
| dpo | refdir18 | 0.80 | 0.75 [0.64, 0.82] |
| dpo | refdir22 | 0.84 | 0.80 [0.72, 0.85] |
| instruct | persona18 | 0.67 | 0.50 [0.30, 0.69] |
| instruct | persona22 | 0.63 | 0.57 [0.37, 0.72] |
| instruct | refdir18 | 0.81 | 0.77 [0.68, 0.83] |
| instruct | refdir22 | 0.84 | 0.82 [0.75, 0.87] |
| sft | persona18 | 0.66 | 0.44 [0.28, 0.58] |
| sft | persona22 | 0.63 | 0.50 [0.31, 0.64] |
| sft | refdir18 | 0.81 | 0.72 [0.62, 0.79] |
| sft | refdir22 | 0.85 | 0.76 [0.65, 0.82] |
