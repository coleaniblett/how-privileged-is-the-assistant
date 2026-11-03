# Persona axis on non-assistant text, fixed axis across stages (exploratory)

Fixed axis = the post-trained model's persona axis (0 = its mean character reply, 1 = its mean Assistant reply). Mean over content tokens. 95% bootstrap CI over texts.


## layer 23

| text | gemma-4-E4B | gemma-4-E4B-it |
|---|---|---|
| wildchat user (user slot) | +0.57 [+0.55, +0.59] | +0.28 [+0.26, +0.30] |
| oasst1 user (user slot) | +0.67 [+0.65, +0.68] | +0.42 [+0.41, +0.43] |
| wildchat user (plain prose) | +0.58 [+0.56, +0.60] | +0.29 [+0.27, +0.31] |
| wildchat assistant (assistant slot) | +0.73 [+0.70, +0.76] | +0.45 [+0.41, +0.49] |
| change from first stage, wildchat user (user slot) | - | -0.28 [-0.30, -0.27] |

## layer 29

| text | gemma-4-E4B | gemma-4-E4B-it |
|---|---|---|
| wildchat user (user slot) | +0.18 [+0.16, +0.21] | +0.34 [+0.33, +0.36] |
| oasst1 user (user slot) | +0.30 [+0.28, +0.31] | +0.44 [+0.43, +0.45] |
| wildchat user (plain prose) | +0.19 [+0.16, +0.21] | +0.34 [+0.33, +0.36] |
| wildchat assistant (assistant slot) | +0.29 [+0.25, +0.33] | +0.42 [+0.38, +0.45] |
| change from first stage, wildchat user (user slot) | - | +0.16 [+0.15, +0.17] |
