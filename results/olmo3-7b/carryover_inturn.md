# Balanced-frame persona carryover (exploratory)

Every holder is a role-played speaker with an identical frame. carryover(h->new) in nats of log-odds toward the holder's believed outcome, read from a new persona. Pilot domains, 12 items each, 95% bootstrap CI over items.


## all

| quantity | base | sft | dpo | instruct |
|---|---|---|---|---|
| carryover(slot_user->new) | 1.05 [0.85, 1.26] | 1.89 [1.46, 2.31] | 2.74 [1.65, 3.88] | 4.12 [2.75, 5.53] |
| carryover(slot_asst_inturn->new) | 1.87 [1.67, 2.06] | 4.06 [3.23, 4.93] | 6.50 [5.01, 8.06] | 6.83 [5.26, 8.51] |
| carryover(slot_assistant->new) | 0.69 [0.43, 0.96] | -0.05 [-0.26, 0.17] | -1.32 [-1.78, -0.85] | -0.99 [-1.41, -0.58] |
| gap slot_asst_inturn - slot_user | 0.82 [0.54, 1.11] | 2.17 [1.30, 3.12] | 3.76 [2.14, 5.35] | 2.71 [1.11, 4.28] |
| &nbsp;&nbsp;change from base | - | 1.36 [0.62, 2.12] | 2.94 [1.51, 4.35] | 1.90 [0.46, 3.33] |
| gap slot_assistant - slot_user | -0.36 [-0.53, -0.18] | -1.94 [-2.30, -1.57] | -4.06 [-5.20, -2.99] | -5.11 [-6.41, -3.83] |
| &nbsp;&nbsp;change from base | - | -1.58 [-1.87, -1.29] | -3.70 [-4.70, -2.73] | -4.76 [-6.01, -3.57] |

## box

| quantity | base | sft | dpo | instruct |
|---|---|---|---|---|
| carryover(slot_user->new) | 0.49 [0.36, 0.65] | 1.53 [1.07, 2.02] | 3.94 [2.18, 5.93] | 6.33 [4.10, 8.71] |
| carryover(slot_asst_inturn->new) | 2.21 [1.88, 2.57] | 7.42 [6.52, 8.23] | 12.96 [12.28, 13.60] | 13.76 [13.08, 14.43] |
| carryover(slot_assistant->new) | -0.30 [-0.45, -0.13] | -0.84 [-0.97, -0.71] | -2.61 [-3.25, -1.99] | -2.04 [-2.67, -1.45] |
| gap slot_asst_inturn - slot_user | 1.72 [1.35, 2.12] | 5.89 [5.30, 6.46] | 9.02 [7.26, 10.75] | 7.43 [5.31, 9.44] |
| &nbsp;&nbsp;change from base | - | 4.17 [3.51, 4.76] | 7.29 [5.58, 8.96] | 5.71 [3.59, 7.81] |
| gap slot_assistant - slot_user | -0.78 [-1.01, -0.58] | -2.37 [-2.84, -1.93] | -6.55 [-8.21, -5.01] | -8.37 [-10.30, -6.47] |
| &nbsp;&nbsp;change from base | - | -1.59 [-1.96, -1.20] | -5.76 [-7.39, -4.39] | -7.58 [-9.49, -5.74] |

## coin

| quantity | base | sft | dpo | instruct |
|---|---|---|---|---|
| carryover(slot_user->new) | 1.41 [1.20, 1.61] | 2.67 [1.91, 3.34] | 1.77 [0.18, 3.19] | 2.86 [1.09, 4.50] |
| carryover(slot_asst_inturn->new) | 1.96 [1.76, 2.14] | 2.37 [2.09, 2.68] | 3.57 [2.85, 4.35] | 3.98 [3.18, 4.82] |
| carryover(slot_assistant->new) | 0.98 [0.87, 1.11] | 0.40 [0.25, 0.60] | -1.06 [-1.64, -0.36] | -0.73 [-1.32, -0.07] |
| gap slot_asst_inturn - slot_user | 0.55 [0.30, 0.81] | -0.30 [-1.01, 0.41] | 1.81 [0.55, 3.04] | 1.11 [-0.22, 2.51] |
| &nbsp;&nbsp;change from base | - | -0.85 [-1.42, -0.31] | 1.26 [0.12, 2.32] | 0.56 [-0.62, 1.76] |
| gap slot_assistant - slot_user | -0.42 [-0.61, -0.22] | -2.28 [-2.93, -1.59] | -2.83 [-4.21, -1.33] | -3.60 [-5.09, -1.88] |
| &nbsp;&nbsp;change from base | - | -1.85 [-2.41, -1.30] | -2.41 [-3.65, -1.06] | -3.17 [-4.59, -1.61] |

## path

| quantity | base | sft | dpo | instruct |
|---|---|---|---|---|
| carryover(slot_user->new) | 1.25 [0.85, 1.68] | 1.47 [0.74, 2.23] | 2.53 [0.36, 4.81] | 3.17 [0.74, 5.76] |
| carryover(slot_asst_inturn->new) | 1.43 [1.11, 1.74] | 2.41 [1.68, 3.15] | 2.98 [2.56, 3.37] | 2.77 [2.34, 3.15] |
| carryover(slot_assistant->new) | 1.38 [1.08, 1.70] | 0.29 [0.07, 0.52] | -0.27 [-0.80, 0.31] | -0.21 [-0.70, 0.34] |
| gap slot_asst_inturn - slot_user | 0.18 [-0.08, 0.45] | 0.94 [0.57, 1.29] | 0.46 [-1.85, 2.68] | -0.40 [-2.90, 2.02] |
| &nbsp;&nbsp;change from base | - | 0.75 [0.35, 1.15] | 0.27 [-1.93, 2.46] | -0.58 [-3.06, 1.77] |
| gap slot_assistant - slot_user | 0.13 [-0.11, 0.39] | -1.18 [-1.73, -0.66] | -2.80 [-4.53, -1.14] | -3.38 [-5.42, -1.36] |
| &nbsp;&nbsp;change from base | - | -1.31 [-1.83, -0.80] | -2.93 [-4.70, -1.33] | -3.51 [-5.41, -1.62] |
