# Role gradient of assistant-voice leakage (exploratory)

Roles with a 'never refuses' instruction; 36 requests x 2 samples; bootstrap over requests.


## refuse

| role group / set-up | sft | instruct |
|---|---|---|
| helpful / sys | 0.85 [0.75, 0.94] | 0.74 [0.62, 0.86] |
| helpful / user | 0.66 [0.54, 0.77] | 0.58 [0.45, 0.72] |
| stylized / sys | 0.43 [0.31, 0.56] | 0.12 [0.05, 0.20] |
| stylized / user | 0.01 [0.00, 0.03] | 0.00 [0.00, 0.00] |

Per role (system prompt):

| role | sft | instruct |
|---|---|---|
| Hannah Reed | 0.81 | 0.78 |
| Mr. Ellis | 0.85 | 0.68 |
| Nurse Joy Park | 0.90 | 0.76 |
| Captain Rook | 0.49 | 0.18 |
| Lord Ashby | 0.51 | 0.12 |
| Osric | 0.29 | 0.06 |

## tic

| role group / set-up | sft | instruct |
|---|---|---|
| helpful / sys | 0.14 [0.08, 0.21] | 0.07 [0.02, 0.13] |
| helpful / user | 0.19 [0.13, 0.25] | 0.11 [0.07, 0.16] |
| stylized / sys | 0.06 [0.03, 0.08] | 0.00 [0.00, 0.00] |
| stylized / user | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |

Per role (system prompt):

| role | sft | instruct |
|---|---|---|
| Hannah Reed | 0.19 | 0.11 |
| Mr. Ellis | 0.11 | 0.06 |
| Nurse Joy Park | 0.12 | 0.04 |
| Captain Rook | 0.04 | 0.00 |
| Lord Ashby | 0.11 | 0.00 |
| Osric | 0.01 | 0.00 |
