# Balanced-frame persona carryover (exploratory)

Every holder is a role-played speaker with an identical frame. carryover(h->new) in nats of log-odds toward the holder's believed outcome, read from a new persona. Pilot domains, 12 items each, 95% bootstrap CI over items.


## all

| quantity | gemma-4-E4B | gemma-4-E4B-it |
|---|---|---|
| carryover(slot_user->new) | 0.62 [0.50, 0.75] | 3.62 [2.84, 4.41] |
| carryover(slot_asst_inturn->new) | 0.73 [0.57, 0.91] | 6.07 [4.90, 7.32] |
| carryover(slot_assistant->new) | 0.52 [0.42, 0.63] | -0.04 [-1.07, 1.08] |
| gap slot_asst_inturn - slot_user | 0.11 [-0.02, 0.26] | 2.45 [1.20, 3.85] |
| &nbsp;&nbsp;change from base | - | 2.34 [1.00, 3.75] |
| gap slot_assistant - slot_user | -0.10 [-0.18, -0.02] | -3.66 [-4.61, -2.72] |
| &nbsp;&nbsp;change from base | - | -3.55 [-4.58, -2.59] |

## box

| quantity | gemma-4-E4B | gemma-4-E4B-it |
|---|---|---|
| carryover(slot_user->new) | 0.19 [0.08, 0.28] | 4.05 [2.77, 5.33] |
| carryover(slot_asst_inturn->new) | 0.31 [0.18, 0.45] | 10.60 [9.00, 12.09] |
| carryover(slot_assistant->new) | 0.22 [0.10, 0.33] | -2.89 [-3.79, -1.92] |
| gap slot_asst_inturn - slot_user | 0.12 [-0.05, 0.32] | 6.55 [4.31, 8.83] |
| &nbsp;&nbsp;change from base | - | 6.43 [4.03, 8.91] |
| gap slot_assistant - slot_user | 0.03 [-0.11, 0.19] | -6.93 [-7.82, -6.07] |
| &nbsp;&nbsp;change from base | - | -6.96 [-7.90, -5.95] |

## coin

| quantity | gemma-4-E4B | gemma-4-E4B-it |
|---|---|---|
| carryover(slot_user->new) | 0.91 [0.74, 1.11] | 1.73 [0.86, 2.69] |
| carryover(slot_asst_inturn->new) | 0.70 [0.51, 0.89] | 3.53 [2.64, 4.36] |
| carryover(slot_assistant->new) | 0.75 [0.60, 0.92] | -1.20 [-1.52, -0.91] |
| gap slot_asst_inturn - slot_user | -0.21 [-0.34, -0.07] | 1.80 [0.85, 2.76] |
| &nbsp;&nbsp;change from base | - | 2.02 [1.13, 2.88] |
| gap slot_assistant - slot_user | -0.16 [-0.29, -0.05] | -2.92 [-4.06, -1.92] |
| &nbsp;&nbsp;change from base | - | -2.76 [-3.92, -1.74] |

## path

| quantity | gemma-4-E4B | gemma-4-E4B-it |
|---|---|---|
| carryover(slot_user->new) | 0.76 [0.67, 0.86] | 5.08 [4.08, 6.16] |
| carryover(slot_asst_inturn->new) | 1.19 [0.91, 1.49] | 4.09 [3.46, 4.73] |
| carryover(slot_assistant->new) | 0.59 [0.45, 0.71] | 3.97 [2.93, 4.78] |
| gap slot_asst_inturn - slot_user | 0.43 [0.20, 0.67] | -0.99 [-1.80, -0.27] |
| &nbsp;&nbsp;change from base | - | -1.43 [-2.23, -0.69] |
| gap slot_assistant - slot_user | -0.17 [-0.26, -0.10] | -1.11 [-1.86, -0.35] |
| &nbsp;&nbsp;change from base | - | -0.94 [-1.74, -0.14] |
