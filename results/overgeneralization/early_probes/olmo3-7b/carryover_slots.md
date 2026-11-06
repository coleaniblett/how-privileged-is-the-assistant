# Balanced-frame persona carryover (exploratory)

Every holder is a role-played speaker with an identical frame. carryover(h->new) in nats of log-odds toward the holder's believed outcome, read from a new persona. Pilot domains, 12 items each, 95% bootstrap CI over items.


## all

| quantity | base | sft | dpo | instruct |
|---|---|---|---|---|
| carryover(slot_system->new) | 1.01 [0.83, 1.19] | 1.30 [0.80, 1.80] | 1.90 [0.82, 3.02] | 2.86 [1.66, 4.05] |
| carryover(slot_user->new) | 1.05 [0.85, 1.26] | 1.89 [1.47, 2.31] | 2.74 [1.64, 3.90] | 4.12 [2.80, 5.47] |
| carryover(slot_assistant->new) | 0.69 [0.43, 0.96] | -0.05 [-0.26, 0.17] | -1.32 [-1.78, -0.85] | -0.99 [-1.41, -0.58] |
| gap slot_system - slot_user | -0.04 [-0.20, 0.11] | -0.59 [-0.81, -0.37] | -0.84 [-1.49, -0.13] | -1.26 [-2.08, -0.47] |
| &nbsp;&nbsp;change from base | - | -0.55 [-0.85, -0.25] | -0.80 [-1.52, -0.11] | -1.22 [-2.06, -0.40] |
| gap slot_assistant - slot_user | -0.36 [-0.53, -0.18] | -1.94 [-2.30, -1.57] | -4.06 [-5.20, -2.99] | -5.11 [-6.41, -3.83] |
| &nbsp;&nbsp;change from base | - | -1.58 [-1.87, -1.29] | -3.70 [-4.70, -2.73] | -4.76 [-6.01, -3.57] |

## box

| quantity | base | sft | dpo | instruct |
|---|---|---|---|---|
| carryover(slot_system->new) | 0.50 [0.37, 0.63] | 0.49 [0.23, 0.77] | 1.42 [0.23, 2.84] | 2.87 [1.28, 4.69] |
| carryover(slot_user->new) | 0.49 [0.36, 0.65] | 1.53 [1.08, 2.00] | 3.94 [2.11, 5.97] | 6.33 [4.09, 8.79] |
| carryover(slot_assistant->new) | -0.30 [-0.45, -0.13] | -0.84 [-0.97, -0.71] | -2.61 [-3.25, -1.99] | -2.04 [-2.67, -1.45] |
| gap slot_system - slot_user | 0.01 [-0.09, 0.11] | -1.03 [-1.32, -0.76] | -2.52 [-3.52, -1.48] | -3.47 [-4.82, -2.22] |
| &nbsp;&nbsp;change from base | - | -1.05 [-1.31, -0.81] | -2.53 [-3.48, -1.59] | -3.48 [-4.84, -2.27] |
| gap slot_assistant - slot_user | -0.78 [-1.01, -0.58] | -2.37 [-2.84, -1.93] | -6.55 [-8.21, -5.01] | -8.37 [-10.30, -6.47] |
| &nbsp;&nbsp;change from base | - | -1.59 [-1.96, -1.20] | -5.76 [-7.39, -4.39] | -7.58 [-9.49, -5.74] |

## coin

| quantity | base | sft | dpo | instruct |
|---|---|---|---|---|
| carryover(slot_system->new) | 1.56 [1.40, 1.71] | 2.49 [1.44, 3.41] | 2.80 [0.71, 4.73] | 3.57 [1.39, 5.70] |
| carryover(slot_user->new) | 1.41 [1.20, 1.61] | 2.67 [1.93, 3.34] | 1.77 [0.23, 3.23] | 2.86 [1.12, 4.50] |
| carryover(slot_assistant->new) | 0.98 [0.87, 1.11] | 0.40 [0.25, 0.60] | -1.06 [-1.64, -0.36] | -0.73 [-1.32, -0.07] |
| gap slot_system - slot_user | 0.16 [-0.08, 0.40] | -0.18 [-0.54, 0.18] | 1.04 [0.18, 1.98] | 0.71 [-0.03, 1.51] |
| &nbsp;&nbsp;change from base | - | -0.34 [-0.88, 0.18] | 0.88 [-0.08, 1.97] | 0.55 [-0.26, 1.42] |
| gap slot_assistant - slot_user | -0.42 [-0.61, -0.22] | -2.28 [-2.93, -1.59] | -2.83 [-4.21, -1.33] | -3.60 [-5.09, -1.88] |
| &nbsp;&nbsp;change from base | - | -1.85 [-2.41, -1.30] | -2.41 [-3.65, -1.06] | -3.17 [-4.59, -1.61] |

## path

| quantity | base | sft | dpo | instruct |
|---|---|---|---|---|
| carryover(slot_system->new) | 0.95 [0.67, 1.22] | 0.91 [0.17, 1.64] | 1.49 [-0.57, 3.68] | 2.14 [0.00, 4.43] |
| carryover(slot_user->new) | 1.25 [0.85, 1.66] | 1.47 [0.75, 2.25] | 2.53 [0.33, 4.77] | 3.17 [0.76, 5.71] |
| carryover(slot_assistant->new) | 1.38 [1.08, 1.70] | 0.29 [0.07, 0.52] | -0.27 [-0.80, 0.31] | -0.21 [-0.70, 0.34] |
| gap slot_system - slot_user | -0.30 [-0.61, 0.05] | -0.56 [-0.88, -0.23] | -1.04 [-1.59, -0.38] | -1.03 [-1.86, -0.09] |
| &nbsp;&nbsp;change from base | - | -0.27 [-0.84, 0.28] | -0.74 [-1.51, 0.01] | -0.73 [-1.66, 0.28] |
| gap slot_assistant - slot_user | 0.13 [-0.11, 0.39] | -1.18 [-1.73, -0.66] | -2.80 [-4.53, -1.14] | -3.38 [-5.42, -1.36] |
| &nbsp;&nbsp;change from base | - | -1.31 [-1.83, -0.80] | -2.93 [-4.70, -1.33] | -3.51 [-5.41, -1.62] |
