# Cosine(persona axis, refusal direction) (exploratory)

Null: 95th percentile of |cosine| for a random split of the same prompts.

| stage | layer 13 | layer 18 | layer 22 | layer 27 |
|---|---|---|---|---|
| base | -0.05 (null 0.08) | -0.01 (null 0.12) | +0.09 (null 0.17) | +0.11 (null 0.14) |
| dpo | -0.02 (null 0.12) | +0.06 (null 0.15) | +0.08 (null 0.16) | +0.06 (null 0.14) |
| instruct | -0.02 (null 0.12) | +0.05 (null 0.14) | +0.07 (null 0.15) | +0.05 (null 0.14) |
| sft | -0.02 (null 0.09) | +0.06 (null 0.09) | +0.09 (null 0.12) | +0.05 (null 0.10) |
