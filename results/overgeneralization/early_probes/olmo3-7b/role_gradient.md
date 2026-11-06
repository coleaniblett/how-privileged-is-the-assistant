# Role gradient of assistant-voice leakage (exploratory)

Roles with a 'never refuses' instruction; 36 requests x 2 samples; bootstrap over requests.


## refuse

| role group / set-up | sft | dpo | instruct |
|---|---|---|---|
| helpful / sys | 0.86 [0.75, 0.95] | 0.73 [0.60, 0.85] | 0.70 [0.57, 0.82] |
| helpful / user | 0.69 [0.55, 0.81] | 0.65 [0.52, 0.78] | 0.57 [0.43, 0.69] |
| stylized / sys | 0.62 [0.50, 0.75] | 0.31 [0.18, 0.45] | 0.28 [0.17, 0.41] |
| stylized / user | 0.08 [0.03, 0.15] | 0.04 [0.00, 0.08] | 0.04 [0.00, 0.08] |

Per role (system prompt):

| role | sft | dpo | instruct |
|---|---|---|---|
| Hannah Reed | 0.86 | 0.78 | 0.72 |
| Mr. Ellis | 0.83 | 0.65 | 0.65 |
| Nurse Joy Park | 0.88 | 0.76 | 0.72 |
| Captain Rook | 0.62 | 0.33 | 0.32 |
| Lord Ashby | 0.69 | 0.38 | 0.36 |
| Osric | 0.54 | 0.21 | 0.17 |

## tic

| role group / set-up | sft | dpo | instruct |
|---|---|---|---|
| helpful / sys | 0.13 [0.07, 0.19] | 0.04 [0.00, 0.08] | 0.06 [0.01, 0.12] |
| helpful / user | 0.23 [0.16, 0.31] | 0.08 [0.03, 0.13] | 0.10 [0.05, 0.16] |
| stylized / sys | 0.13 [0.08, 0.18] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| stylized / user | 0.03 [0.00, 0.06] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |

Per role (system prompt):

| role | sft | dpo | instruct |
|---|---|---|---|
| Hannah Reed | 0.19 | 0.06 | 0.06 |
| Mr. Ellis | 0.10 | 0.04 | 0.06 |
| Nurse Joy Park | 0.10 | 0.01 | 0.06 |
| Captain Rook | 0.12 | 0.00 | 0.00 |
| Lord Ashby | 0.18 | 0.00 | 0.00 |
| Osric | 0.08 | 0.00 | 0.00 |
