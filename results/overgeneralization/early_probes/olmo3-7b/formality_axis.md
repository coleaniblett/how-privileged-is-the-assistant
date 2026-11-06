# Formality axis vs persona axis (exploratory)

Formality axis = formal minus casual style instructions (no persona). Cosine and cross-AUC at each layer (persona-axis layers).

| stage | layer | cos(persona, formality) | AUC of persona axis on formal-vs-casual replies | AUC of formality axis on Assistant-vs-character replies |
|---|---|---|---|---|
| base | 13 | +0.32 | 0.66 | 0.67 |
| base | 18 | +0.31 | 0.65 | 0.66 |
| base | 22 | +0.29 | 0.67 | 0.65 |
| base | 27 | +0.28 | 0.69 | 0.65 |
| dpo | 13 | +0.45 | 0.89 | 0.78 |
| dpo | 18 | +0.44 | 0.94 | 0.77 |
| dpo | 22 | +0.33 | 0.96 | 0.74 |
| dpo | 27 | +0.29 | 0.96 | 0.71 |
| instruct | 13 | +0.43 | 0.93 | 0.74 |
| instruct | 18 | +0.42 | 0.97 | 0.74 |
| instruct | 22 | +0.33 | 0.98 | 0.71 |
| instruct | 27 | +0.30 | 0.98 | 0.69 |
| sft | 13 | +0.23 | 0.67 | 0.60 |
| sft | 18 | +0.20 | 0.69 | 0.59 |
| sft | 22 | +0.17 | 0.71 | 0.58 |
| sft | 27 | +0.14 | 0.71 | 0.57 |
