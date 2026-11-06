# Role gradient of assistant-voice leakage (exploratory)

Roles with a 'never refuses' instruction; 36 requests x 2 samples; bootstrap over requests.


## refuse

| role group / set-up | Llama-3.1-8B | Llama-3.1-8B-Instruct |
|---|---|---|
| helpful / sys | 0.02 [0.00, 0.04] | 0.42 [0.30, 0.56] |
| helpful / user | 0.02 [0.00, 0.05] | 0.29 [0.17, 0.42] |
| stylized / sys | 0.04 [0.01, 0.07] | 0.12 [0.04, 0.21] |
| stylized / user | 0.03 [0.01, 0.05] | 0.08 [0.01, 0.16] |

Per role (system prompt):

| role | Llama-3.1-8B | Llama-3.1-8B-Instruct |
|---|---|---|
| Hannah Reed | 0.04 | 0.47 |
| Mr. Ellis | 0.01 | 0.40 |
| Nurse Joy Park | 0.00 | 0.39 |
| Captain Rook | 0.04 | 0.12 |
| Lord Ashby | 0.04 | 0.07 |
| Osric | 0.03 | 0.17 |

## tic

| role group / set-up | Llama-3.1-8B | Llama-3.1-8B-Instruct |
|---|---|---|
| helpful / sys | 0.01 [0.00, 0.02] | 0.40 [0.34, 0.47] |
| helpful / user | 0.00 [0.00, 0.00] | 0.50 [0.39, 0.60] |
| stylized / sys | 0.00 [0.00, 0.01] | 0.09 [0.04, 0.16] |
| stylized / user | 0.01 [0.00, 0.02] | 0.06 [0.01, 0.14] |

Per role (system prompt):

| role | Llama-3.1-8B | Llama-3.1-8B-Instruct |
|---|---|---|
| Hannah Reed | 0.01 | 0.42 |
| Mr. Ellis | 0.01 | 0.35 |
| Nurse Joy Park | 0.00 | 0.44 |
| Captain Rook | 0.00 | 0.10 |
| Lord Ashby | 0.00 | 0.07 |
| Osric | 0.01 | 0.11 |
