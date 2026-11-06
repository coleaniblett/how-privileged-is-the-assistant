# Role gradient of assistant-voice leakage (exploratory)

Roles with a 'never refuses' instruction; 36 requests x 2 samples; bootstrap over requests.


## refuse

| role group / set-up | gemma-4-E4B | gemma-4-E4B-it |
|---|---|---|
| helpful / sys | 0.13 [0.09, 0.18] | 0.23 [0.15, 0.31] |
| helpful / user | 0.13 [0.10, 0.17] | 0.28 [0.19, 0.38] |
| stylized / sys | 0.14 [0.09, 0.20] | 0.00 [0.00, 0.00] |
| stylized / user | 0.12 [0.09, 0.16] | 0.02 [0.00, 0.05] |

Per role (system prompt):

| role | gemma-4-E4B | gemma-4-E4B-it |
|---|---|---|
| Hannah Reed | 0.11 | 0.43 |
| Mr. Ellis | 0.19 | 0.03 |
| Nurse Joy Park | 0.08 | 0.22 |
| Captain Rook | 0.10 | 0.00 |
| Lord Ashby | 0.10 | 0.00 |
| Osric | 0.22 | 0.00 |

## tic

| role group / set-up | gemma-4-E4B | gemma-4-E4B-it |
|---|---|---|
| helpful / sys | 0.09 [0.05, 0.13] | 0.38 [0.31, 0.44] |
| helpful / user | 0.09 [0.06, 0.14] | 0.34 [0.25, 0.43] |
| stylized / sys | 0.07 [0.04, 0.12] | 0.00 [0.00, 0.00] |
| stylized / user | 0.12 [0.08, 0.15] | 0.00 [0.00, 0.00] |

Per role (system prompt):

| role | gemma-4-E4B | gemma-4-E4B-it |
|---|---|---|
| Hannah Reed | 0.10 | 0.64 |
| Mr. Ellis | 0.11 | 0.17 |
| Nurse Joy Park | 0.07 | 0.33 |
| Captain Rook | 0.06 | 0.00 |
| Lord Ashby | 0.10 | 0.00 |
| Osric | 0.07 | 0.00 |
