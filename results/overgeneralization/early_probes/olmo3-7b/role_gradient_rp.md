# Role gradient of assistant-voice leakage (exploratory)

Roles with a 'never refuses' instruction; 36 requests x 2 samples; bootstrap over requests.


## refuse

| role group / set-up | sft | instruct |
|---|---|---|
| helpful / sys | 0.85 [0.73, 0.94] | 0.72 [0.58, 0.84] |
| helpful / user | 0.47 [0.33, 0.61] | 0.51 [0.37, 0.65] |
| stylized / sys | 0.36 [0.24, 0.48] | 0.12 [0.04, 0.20] |
| stylized / user | 0.05 [0.01, 0.10] | 0.04 [0.00, 0.08] |

Per role (system prompt):

| role | sft | instruct |
|---|---|---|
| Hannah Reed | 0.86 | 0.74 |
| Mr. Ellis | 0.83 | 0.67 |
| Nurse Joy Park | 0.85 | 0.75 |
| Captain Rook | 0.44 | 0.17 |
| Lord Ashby | 0.39 | 0.12 |
| Osric | 0.24 | 0.06 |

## tic

| role group / set-up | sft | instruct |
|---|---|---|
| helpful / sys | 0.13 [0.07, 0.19] | 0.09 [0.04, 0.15] |
| helpful / user | 0.24 [0.18, 0.31] | 0.16 [0.12, 0.21] |
| stylized / sys | 0.06 [0.02, 0.11] | 0.00 [0.00, 0.01] |
| stylized / user | 0.01 [0.00, 0.03] | 0.01 [0.00, 0.02] |

Per role (system prompt):

| role | sft | instruct |
|---|---|---|
| Hannah Reed | 0.14 | 0.07 |
| Mr. Ellis | 0.14 | 0.15 |
| Nurse Joy Park | 0.11 | 0.06 |
| Captain Rook | 0.04 | 0.01 |
| Lord Ashby | 0.10 | 0.00 |
| Osric | 0.04 | 0.00 |
