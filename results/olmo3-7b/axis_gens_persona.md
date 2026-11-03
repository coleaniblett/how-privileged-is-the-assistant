# Generated speech on the assistant axis (exploratory)

Coordinate: mean character reply = 0, mean Assistant-framing reply = 1, per stage (persona axis). Mean over reply tokens. 95% bootstrap CI over generations.


## layer 18

| speech | base | sft | dpo | instruct |
|---|---|---|---|---|
| small talk: Assistant | 0.89 [0.61, 1.18] | 0.99 [0.80, 1.14] | 0.92 [0.83, 1.01] | 0.99 [0.91, 1.07] |
| small talk: character (user turn) | -0.12 [-0.43, 0.19] | -0.90 [-1.01, -0.77] | -0.59 [-0.68, -0.49] | -0.63 [-0.71, -0.55] |
| small talk: character (system prompt) | -0.53 [-0.76, -0.30] | -0.60 [-0.78, -0.41] | -0.42 [-0.54, -0.28] | -0.45 [-0.57, -0.32] |
| model-written user turn | 0.14 [-0.08, 0.37] | -0.34 [-0.46, -0.23] | -0.01 [-0.08, 0.06] | -0.03 [-0.13, 0.07] |
| harmful request: Assistant | 0.83 [0.72, 0.94] | 0.47 [0.41, 0.51] | 0.82 [0.77, 0.86] | 0.83 [0.78, 0.87] |
| harmful request: character (user turn) | 0.10 [-0.01, 0.23] | -0.07 [-0.16, 0.03] | -0.10 [-0.18, -0.01] | -0.24 [-0.32, -0.16] |
| harmful request: character (system prompt) | -0.08 [-0.20, 0.03] | 0.21 [0.15, 0.28] | 0.39 [0.31, 0.46] | 0.27 [0.19, 0.35] |
|   ... character replies that refuse (all set-ups) | 0.67 [0.59, 0.75] | 0.32 [0.30, 0.34] | 0.57 [0.53, 0.61] | 0.47 [0.43, 0.51] |
|   ... character replies that comply (all set-ups) | -0.43 [-0.48, -0.38] | -0.69 [-0.73, -0.64] | -0.38 [-0.40, -0.35] | -0.46 [-0.48, -0.44] |

## layer 22

| speech | base | sft | dpo | instruct |
|---|---|---|---|---|
| small talk: Assistant | 0.94 [0.66, 1.23] | 0.97 [0.82, 1.11] | 0.93 [0.85, 1.01] | 0.98 [0.91, 1.06] |
| small talk: character (user turn) | -0.19 [-0.51, 0.16] | -1.00 [-1.14, -0.86] | -0.53 [-0.61, -0.44] | -0.57 [-0.65, -0.48] |
| small talk: character (system prompt) | -0.61 [-0.85, -0.38] | -0.62 [-0.82, -0.42] | -0.36 [-0.48, -0.22] | -0.38 [-0.49, -0.26] |
| model-written user turn | 0.09 [-0.13, 0.31] | -0.25 [-0.36, -0.14] | 0.15 [0.09, 0.20] | 0.14 [0.06, 0.23] |
| harmful request: Assistant | 0.86 [0.74, 0.97] | 0.52 [0.47, 0.57] | 0.82 [0.78, 0.85] | 0.83 [0.79, 0.86] |
| harmful request: character (user turn) | 0.07 [-0.06, 0.22] | -0.06 [-0.17, 0.05] | -0.15 [-0.23, -0.07] | -0.27 [-0.34, -0.19] |
| harmful request: character (system prompt) | -0.14 [-0.27, -0.01] | 0.27 [0.19, 0.34] | 0.31 [0.23, 0.38] | 0.20 [0.13, 0.28] |
|   ... character replies that refuse (all set-ups) | 0.72 [0.64, 0.80] | 0.40 [0.37, 0.42] | 0.49 [0.44, 0.53] | 0.41 [0.36, 0.45] |
|   ... character replies that comply (all set-ups) | -0.51 [-0.56, -0.45] | -0.77 [-0.81, -0.72] | -0.41 [-0.43, -0.39] | -0.47 [-0.49, -0.45] |
