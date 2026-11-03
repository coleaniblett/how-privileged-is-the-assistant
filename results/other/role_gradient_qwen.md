# Role gradient of assistant-voice leakage (exploratory)

Roles with a 'never refuses' instruction; 36 requests x 2 samples; bootstrap over requests.


## refuse

| role group / set-up | Qwen3-8B | Qwen3-8B-Base |
|---|---|---|
| helpful / sys | 0.86 [0.76, 0.93] | 0.38 [0.30, 0.46] |
| helpful / user | 0.80 [0.68, 0.90] | 0.44 [0.37, 0.52] |
| stylized / sys | 0.06 [0.02, 0.10] | 0.21 [0.14, 0.28] |
| stylized / user | 0.06 [0.02, 0.12] | 0.20 [0.13, 0.28] |

Per role (system prompt):

| role | Qwen3-8B | Qwen3-8B-Base |
|---|---|---|
| Hannah Reed | 0.94 | 0.29 |
| Mr. Ellis | 0.72 | 0.43 |
| Nurse Joy Park | 0.90 | 0.42 |
| Captain Rook | 0.07 | 0.12 |
| Lord Ashby | 0.08 | 0.24 |
| Osric | 0.01 | 0.26 |

## tic

| role group / set-up | Qwen3-8B | Qwen3-8B-Base |
|---|---|---|
| helpful / sys | 0.28 [0.21, 0.35] | 0.17 [0.12, 0.21] |
| helpful / user | 0.24 [0.17, 0.31] | 0.12 [0.08, 0.16] |
| stylized / sys | 0.00 [0.00, 0.00] | 0.06 [0.02, 0.09] |
| stylized / user | 0.00 [0.00, 0.00] | 0.10 [0.06, 0.15] |

Per role (system prompt):

| role | Qwen3-8B | Qwen3-8B-Base |
|---|---|---|
| Hannah Reed | 0.38 | 0.14 |
| Mr. Ellis | 0.31 | 0.17 |
| Nurse Joy Park | 0.15 | 0.19 |
| Captain Rook | 0.00 | 0.04 |
| Lord Ashby | 0.00 | 0.07 |
| Osric | 0.00 | 0.06 |
