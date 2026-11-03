# Balanced-frame persona carryover (exploratory)

Every holder is a role-played speaker with an identical frame. carryover(h->new) in nats of log-odds toward the holder's believed outcome, read from a new persona. Pilot domains, 12 items each, 95% bootstrap CI over items.


## all

| quantity | gemma-4-E4B | gemma-4-E4B-it |
|---|---|---|
| carryover(assistant->new) | 0.01 [-0.14, 0.17] | -2.69 [-4.24, -1.17] |
| carryover(user->new) | 0.06 [-0.09, 0.21] | -2.17 [-3.81, -0.57] |
| carryover(char->new) | -0.13 [-0.32, 0.06] | -3.71 [-5.07, -2.32] |
| carryover(nat_assistant->new) | 0.42 [0.30, 0.55] | -0.76 [-1.83, 0.34] |
| carryover(nat_user->new) | 0.59 [0.47, 0.72] | 3.01 [2.32, 3.71] |
| carryover(slot_system->new) | 0.52 [0.42, 0.63] | 3.73 [3.04, 4.44] |
| carryover(slot_user->new) | 0.62 [0.50, 0.74] | 3.62 [2.82, 4.38] |
| carryover(slot_assistant->new) | 0.52 [0.41, 0.63] | -0.04 [-1.04, 1.04] |
| gap assistant - char | 0.14 [0.07, 0.21] | 1.02 [0.67, 1.39] |
| &nbsp;&nbsp;change from base | - | 0.88 [0.53, 1.26] |
| gap user - char | 0.18 [0.11, 0.26] | 1.54 [1.10, 1.95] |
| &nbsp;&nbsp;change from base | - | 1.35 [0.91, 1.81] |
| gap nat_assistant - char | 0.55 [0.39, 0.69] | 2.95 [2.41, 3.53] |
| &nbsp;&nbsp;change from base | - | 2.41 [1.85, 2.96] |
| gap nat_user - char | 0.71 [0.58, 0.85] | 6.72 [5.69, 7.79] |
| &nbsp;&nbsp;change from base | - | 6.01 [5.07, 6.99] |
| gap slot_system - char | 0.65 [0.50, 0.81] | 7.44 [6.34, 8.57] |
| &nbsp;&nbsp;change from base | - | 6.79 [5.83, 7.82] |
| gap slot_user - char | 0.75 [0.62, 0.88] | 7.33 [6.05, 8.75] |
| &nbsp;&nbsp;change from base | - | 6.58 [5.41, 7.84] |
| gap slot_assistant - char | 0.65 [0.50, 0.80] | 3.67 [3.08, 4.34] |
| &nbsp;&nbsp;change from base | - | 3.03 [2.42, 3.64] |
| gap assistant - user | -0.05 [-0.09, 0.00] | -0.51 [-0.77, -0.25] |
| &nbsp;&nbsp;change from base | - | -0.47 [-0.72, -0.21] |
| gap slot_system - slot_user | -0.10 [-0.16, -0.05] | 0.11 [-0.39, 0.54] |
| &nbsp;&nbsp;change from base | - | 0.21 [-0.29, 0.68] |
| gap slot_assistant - slot_user | -0.10 [-0.18, -0.02] | -3.66 [-4.63, -2.69] |
| &nbsp;&nbsp;change from base | - | -3.55 [-4.53, -2.58] |
| gap nat_assistant - nat_user | -0.16 [-0.23, -0.09] | -3.77 [-4.53, -3.02] |
| &nbsp;&nbsp;change from base | - | -3.60 [-4.38, -2.81] |

## box

| quantity | gemma-4-E4B | gemma-4-E4B-it |
|---|---|---|
| carryover(assistant->new) | -0.51 [-0.58, -0.44] | -7.60 [-9.08, -5.90] |
| carryover(user->new) | -0.49 [-0.55, -0.42] | -7.54 [-9.31, -5.53] |
| carryover(char->new) | -0.87 [-0.93, -0.81] | -8.53 [-9.84, -7.10] |
| carryover(nat_assistant->new) | 0.05 [-0.04, 0.14] | -3.77 [-4.63, -2.79] |
| carryover(nat_user->new) | 0.19 [0.12, 0.24] | 2.07 [0.85, 3.22] |
| carryover(slot_system->new) | 0.23 [0.15, 0.33] | 3.29 [2.15, 4.66] |
| carryover(slot_user->new) | 0.19 [0.09, 0.28] | 4.05 [2.71, 5.31] |
| carryover(slot_assistant->new) | 0.22 [0.10, 0.33] | -2.89 [-3.78, -1.97] |
| gap assistant - char | 0.36 [0.28, 0.44] | 0.93 [0.40, 1.48] |
| &nbsp;&nbsp;change from base | - | 0.57 [0.05, 1.08] |
| gap user - char | 0.39 [0.30, 0.49] | 0.99 [0.06, 1.89] |
| &nbsp;&nbsp;change from base | - | 0.61 [-0.26, 1.42] |
| gap nat_assistant - char | 0.92 [0.82, 1.03] | 4.77 [4.10, 5.44] |
| &nbsp;&nbsp;change from base | - | 3.84 [3.18, 4.52] |
| gap nat_user - char | 1.06 [1.01, 1.13] | 10.60 [9.49, 11.67] |
| &nbsp;&nbsp;change from base | - | 9.54 [8.49, 10.57] |
| gap slot_system - char | 1.10 [1.02, 1.19] | 11.82 [11.17, 12.45] |
| &nbsp;&nbsp;change from base | - | 10.72 [10.07, 11.33] |
| gap slot_user - char | 1.06 [0.98, 1.15] | 12.58 [11.48, 13.78] |
| &nbsp;&nbsp;change from base | - | 11.52 [10.41, 12.71] |
| gap slot_assistant - char | 1.09 [0.96, 1.21] | 5.65 [4.72, 6.60] |
| &nbsp;&nbsp;change from base | - | 4.56 [3.55, 5.51] |
| gap assistant - user | -0.03 [-0.08, 0.03] | -0.06 [-0.59, 0.44] |
| &nbsp;&nbsp;change from base | - | -0.03 [-0.56, 0.49] |
| gap slot_system - slot_user | 0.04 [-0.03, 0.11] | -0.76 [-1.76, 0.24] |
| &nbsp;&nbsp;change from base | - | -0.79 [-1.84, 0.16] |
| gap slot_assistant - slot_user | 0.03 [-0.12, 0.19] | -6.93 [-7.87, -6.06] |
| &nbsp;&nbsp;change from base | - | -6.96 [-7.87, -5.98] |
| gap nat_assistant - nat_user | -0.14 [-0.24, -0.03] | -5.83 [-6.53, -5.16] |
| &nbsp;&nbsp;change from base | - | -5.69 [-6.45, -4.95] |

## coin

| quantity | gemma-4-E4B | gemma-4-E4B-it |
|---|---|---|
| carryover(assistant->new) | 0.07 [-0.05, 0.19] | -2.90 [-3.32, -2.45] |
| carryover(user->new) | 0.24 [0.11, 0.39] | -2.09 [-2.52, -1.68] |
| carryover(char->new) | 0.07 [-0.02, 0.17] | -3.05 [-3.42, -2.69] |
| carryover(nat_assistant->new) | 0.80 [0.62, 0.98] | -1.85 [-2.24, -1.47] |
| carryover(nat_user->new) | 0.81 [0.61, 1.05] | 2.48 [1.59, 3.47] |
| carryover(slot_system->new) | 0.77 [0.60, 0.95] | 2.40 [1.74, 3.18] |
| carryover(slot_user->new) | 0.91 [0.73, 1.10] | 1.73 [0.83, 2.71] |
| carryover(slot_assistant->new) | 0.75 [0.60, 0.91] | -1.20 [-1.53, -0.91] |
| gap assistant - char | -0.01 [-0.07, 0.06] | 0.15 [-0.03, 0.34] |
| &nbsp;&nbsp;change from base | - | 0.15 [-0.03, 0.33] |
| gap user - char | 0.17 [0.10, 0.25] | 0.96 [0.69, 1.23] |
| &nbsp;&nbsp;change from base | - | 0.79 [0.53, 1.05] |
| gap nat_assistant - char | 0.72 [0.56, 0.89] | 1.20 [0.99, 1.41] |
| &nbsp;&nbsp;change from base | - | 0.48 [0.20, 0.77] |
| gap nat_user - char | 0.74 [0.53, 0.97] | 5.53 [4.62, 6.43] |
| &nbsp;&nbsp;change from base | - | 4.79 [3.97, 5.60] |
| gap slot_system - char | 0.69 [0.53, 0.86] | 5.45 [4.65, 6.33] |
| &nbsp;&nbsp;change from base | - | 4.75 [4.03, 5.51] |
| gap slot_user - char | 0.84 [0.65, 1.03] | 4.78 [3.79, 5.86] |
| &nbsp;&nbsp;change from base | - | 3.94 [3.03, 4.88] |
| gap slot_assistant - char | 0.68 [0.53, 0.83] | 1.85 [1.58, 2.13] |
| &nbsp;&nbsp;change from base | - | 1.17 [0.81, 1.57] |
| gap assistant - user | -0.18 [-0.24, -0.11] | -0.81 [-1.04, -0.60] |
| &nbsp;&nbsp;change from base | - | -0.64 [-0.86, -0.42] |
| gap slot_system - slot_user | -0.15 [-0.21, -0.08] | 0.67 [0.39, 0.99] |
| &nbsp;&nbsp;change from base | - | 0.82 [0.52, 1.16] |
| gap slot_assistant - slot_user | -0.16 [-0.29, -0.05] | -2.92 [-4.10, -1.94] |
| &nbsp;&nbsp;change from base | - | -2.76 [-3.89, -1.78] |
| gap nat_assistant - nat_user | -0.02 [-0.13, 0.08] | -4.33 [-5.22, -3.46] |
| &nbsp;&nbsp;change from base | - | -4.31 [-5.18, -3.45] |

## path

| quantity | gemma-4-E4B | gemma-4-E4B-it |
|---|---|---|
| carryover(assistant->new) | 0.48 [0.30, 0.65] | 2.43 [1.02, 3.58] |
| carryover(user->new) | 0.41 [0.24, 0.59] | 3.10 [1.81, 4.23] |
| carryover(char->new) | 0.42 [0.22, 0.61] | 0.45 [-0.75, 1.61] |
| carryover(nat_assistant->new) | 0.42 [0.28, 0.53] | 3.34 [2.28, 4.28] |
| carryover(nat_user->new) | 0.76 [0.67, 0.84] | 4.48 [3.59, 5.41] |
| carryover(slot_system->new) | 0.57 [0.46, 0.66] | 5.49 [4.60, 6.45] |
| carryover(slot_user->new) | 0.76 [0.67, 0.86] | 5.08 [4.04, 6.19] |
| carryover(slot_assistant->new) | 0.59 [0.45, 0.71] | 3.97 [2.96, 4.78] |
| gap assistant - char | 0.06 [-0.01, 0.13] | 1.98 [1.37, 2.58] |
| &nbsp;&nbsp;change from base | - | 1.92 [1.29, 2.51] |
| gap user - char | -0.01 [-0.09, 0.09] | 2.66 [2.20, 3.07] |
| &nbsp;&nbsp;change from base | - | 2.66 [2.14, 3.15] |
| gap nat_assistant - char | 0.00 [-0.11, 0.13] | 2.90 [2.20, 3.55] |
| &nbsp;&nbsp;change from base | - | 2.90 [2.25, 3.54] |
| gap nat_user - char | 0.34 [0.18, 0.49] | 4.03 [3.40, 4.61] |
| &nbsp;&nbsp;change from base | - | 3.69 [3.17, 4.19] |
| gap slot_system - char | 0.15 [-0.03, 0.34] | 5.05 [4.40, 5.67] |
| &nbsp;&nbsp;change from base | - | 4.90 [4.33, 5.44] |
| gap slot_user - char | 0.34 [0.22, 0.47] | 4.64 [3.86, 5.38] |
| &nbsp;&nbsp;change from base | - | 4.29 [3.59, 5.00] |
| gap slot_assistant - char | 0.17 [0.05, 0.31] | 3.53 [2.83, 4.16] |
| &nbsp;&nbsp;change from base | - | 3.35 [2.71, 3.97] |
| gap assistant - user | 0.07 [0.01, 0.13] | -0.67 [-1.08, -0.30] |
| &nbsp;&nbsp;change from base | - | -0.74 [-1.13, -0.38] |
| gap slot_system - slot_user | -0.19 [-0.29, -0.09] | 0.41 [-0.20, 1.07] |
| &nbsp;&nbsp;change from base | - | 0.60 [-0.05, 1.30] |
| gap slot_assistant - slot_user | -0.17 [-0.26, -0.09] | -1.11 [-1.89, -0.33] |
| &nbsp;&nbsp;change from base | - | -0.94 [-1.73, -0.16] |
| gap nat_assistant - nat_user | -0.34 [-0.42, -0.27] | -1.14 [-1.65, -0.61] |
| &nbsp;&nbsp;change from base | - | -0.80 [-1.29, -0.31] |
