# Generated speech on the assistant axis (exploratory)

Coordinate: user text/user slot = 0, assistant text/assistant slot = 1, per stage. Mean over reply tokens. 95% bootstrap CI over generations.


## layer 18

| speech | base | sft | dpo | instruct |
|---|---|---|---|---|
| small talk: Assistant | 1.30 [1.20, 1.38] | 1.51 [1.45, 1.57] | 1.50 [1.44, 1.56] | 1.53 [1.47, 1.58] |
| small talk: character (user turn) | 1.22 [1.14, 1.30] | 1.08 [1.01, 1.15] | 1.11 [1.05, 1.17] | 1.12 [1.07, 1.18] |
| small talk: character (system prompt) | 1.32 [1.23, 1.40] | 1.31 [1.23, 1.39] | 1.20 [1.13, 1.27] | 1.21 [1.15, 1.28] |
| model-written user turn | 0.92 [0.77, 1.05] | 0.35 [0.17, 0.53] | 0.55 [0.37, 0.73] | 0.50 [0.35, 0.65] |
| harmful request: Assistant | 1.30 [1.24, 1.36] | 1.69 [1.65, 1.73] | 1.68 [1.65, 1.70] | 1.66 [1.64, 1.69] |
| harmful request: character (user turn) | 1.20 [1.13, 1.25] | 1.57 [1.52, 1.62] | 1.36 [1.32, 1.41] | 1.32 [1.27, 1.36] |
| harmful request: character (system prompt) | 1.31 [1.26, 1.36] | 1.74 [1.70, 1.78] | 1.60 [1.56, 1.64] | 1.56 [1.52, 1.60] |
|   ... character replies that refuse (all set-ups) | 1.18 [1.11, 1.24] | 1.79 [1.78, 1.80] | 1.70 [1.68, 1.72] | 1.67 [1.66, 1.69] |
|   ... character replies that comply (all set-ups) | 1.22 [1.19, 1.25] | 1.24 [1.20, 1.29] | 1.21 [1.19, 1.23] | 1.20 [1.18, 1.22] |

## layer 27

| speech | base | sft | dpo | instruct |
|---|---|---|---|---|
| small talk: Assistant | 1.24 [1.15, 1.33] | 1.46 [1.40, 1.51] | 1.55 [1.49, 1.61] | 1.57 [1.51, 1.62] |
| small talk: character (user turn) | 1.21 [1.13, 1.29] | 0.96 [0.89, 1.03] | 1.12 [1.05, 1.19] | 1.13 [1.07, 1.20] |
| small talk: character (system prompt) | 1.32 [1.23, 1.41] | 1.21 [1.12, 1.30] | 1.22 [1.15, 1.30] | 1.22 [1.15, 1.30] |
| model-written user turn | 0.92 [0.78, 1.03] | 0.42 [0.28, 0.58] | 0.64 [0.46, 0.82] | 0.57 [0.43, 0.73] |
| harmful request: Assistant | 1.24 [1.18, 1.30] | 1.66 [1.62, 1.70] | 1.73 [1.70, 1.76] | 1.72 [1.69, 1.74] |
| harmful request: character (user turn) | 1.21 [1.15, 1.27] | 1.56 [1.50, 1.62] | 1.43 [1.38, 1.48] | 1.39 [1.35, 1.44] |
| harmful request: character (system prompt) | 1.34 [1.29, 1.39] | 1.73 [1.68, 1.77] | 1.67 [1.62, 1.70] | 1.63 [1.58, 1.67] |
|   ... character replies that refuse (all set-ups) | 1.15 [1.09, 1.21] | 1.80 [1.79, 1.82] | 1.78 [1.76, 1.79] | 1.76 [1.74, 1.78] |
|   ... character replies that comply (all set-ups) | 1.28 [1.25, 1.31] | 1.19 [1.15, 1.24] | 1.27 [1.25, 1.29] | 1.26 [1.24, 1.28] |
