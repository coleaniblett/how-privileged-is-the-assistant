# Persona axis on non-assistant text, fixed axis across stages (exploratory)

Fixed axis = the post-trained model's persona axis (0 = its mean character reply, 1 = its mean Assistant reply). Mean over content tokens. 95% bootstrap CI over texts.


## layer 18

| text | base | sft | dpo | instruct |
|---|---|---|---|---|
| wildchat user (user slot) | -0.15 [-0.20, -0.10] | -0.13 [-0.16, -0.09] | -0.13 [-0.16, -0.09] | -0.13 [-0.16, -0.09] |
| oasst1 user (user slot) | +0.03 [-0.00, +0.07] | -0.02 [-0.05, +0.01] | -0.01 [-0.04, +0.02] | -0.01 [-0.04, +0.01] |
| wildchat user (plain prose) | -0.24 [-0.28, -0.19] | -0.24 [-0.28, -0.21] | -0.24 [-0.28, -0.21] | -0.24 [-0.28, -0.21] |
| wildchat assistant (assistant slot) | +0.14 [+0.07, +0.22] | +0.09 [+0.04, +0.15] | +0.13 [+0.07, +0.20] | +0.14 [+0.07, +0.20] |
| change from first stage, wildchat user (user slot) | - | +0.03 [+0.01, +0.04] | +0.03 [+0.01, +0.04] | +0.03 [+0.01, +0.04] |

## layer 22

| text | base | sft | dpo | instruct |
|---|---|---|---|---|
| wildchat user (user slot) | +0.09 [+0.05, +0.13] | +0.08 [+0.05, +0.10] | +0.07 [+0.05, +0.10] | +0.07 [+0.05, +0.10] |
| oasst1 user (user slot) | +0.25 [+0.22, +0.28] | +0.15 [+0.13, +0.17] | +0.15 [+0.13, +0.17] | +0.15 [+0.12, +0.17] |
| wildchat user (plain prose) | +0.00 [-0.03, +0.04] | -0.01 [-0.03, +0.02] | -0.01 [-0.04, +0.02] | -0.01 [-0.04, +0.02] |
| wildchat assistant (assistant slot) | +0.30 [+0.24, +0.37] | +0.24 [+0.19, +0.29] | +0.27 [+0.21, +0.33] | +0.27 [+0.21, +0.33] |
| change from first stage, wildchat user (user slot) | - | -0.02 [-0.03, +0.00] | -0.02 [-0.04, -0.00] | -0.02 [-0.04, -0.00] |
