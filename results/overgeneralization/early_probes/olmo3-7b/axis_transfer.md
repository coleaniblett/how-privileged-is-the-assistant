# Persona axis on an independent role/question set (exploratory)

v1 = original 3 Assistant framings vs 24 characters x 16 questions; v2 = 3 other framings vs 24 other characters x 16 other questions.

| stage | v1 axis -> v2 replies (AUC) | v2 axis, held-out v2 roles (AUC) | cosine(v1 axis, v2 axis) |
|---|---|---|---|
| base | 0.77 | 0.72 | 0.75 |
| sft | 0.80 | 0.77 | 0.76 |
| dpo | 0.86 | 0.85 | 0.90 |
| instruct | 0.87 | 0.85 | 0.89 |
