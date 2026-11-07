# Second local judge (spec J6; exploratory, outside every verdict)

Judge 2 = Qwen/Qwen3-8B (non-thinking), the same frozen prompts and rule (P(Yes) >= 0.5) as the Gemma 4 31B judge, blind. **Agreement is not accuracy**: two judges can agree and both be wrong, and a low kappa is partly a base-rate effect when almost every reply has the same label. A group with kappa under 0.4 is flagged unreliable (kappa is undefined and marked n/a when one label is absent). Samples: 1,000 stratified atlas replies for the in-character question (equal allocation over family/stage/context, half and half on the Gemma label where possible) and 1,000 stratified atlas trait labels (equal over trait/context).

## incharacter (1000 replies): overall agreement 0.832, kappa 0.67

### by family + stage

| group | n | Gemma-judge yes | Qwen-judge yes | agreement | kappa | flag |
|---|---|---|---|---|---|---|
| gemma4-31b / base | 168 | 0.50 | 0.30 | 0.79 | 0.57 |  |
| gemma4-31b / instruct | 168 | 0.77 | 0.73 | 0.88 | 0.68 |  |
| olmo3-7b / base | 166 | 0.49 | 0.33 | 0.81 | 0.61 |  |
| olmo3-7b / instruct | 166 | 0.50 | 0.37 | 0.87 | 0.75 |  |
| qwen3-8b / base | 166 | 0.50 | 0.33 | 0.82 | 0.64 |  |
| qwen3-8b / instruct | 166 | 0.50 | 0.40 | 0.83 | 0.65 |  |

### by context

| group | n | Gemma-judge yes | Qwen-judge yes | agreement | kappa | flag |
|---|---|---|---|---|---|---|
| light_system | 500 | 0.53 | 0.37 | 0.81 | 0.62 |  |
| light_user | 500 | 0.56 | 0.44 | 0.86 | 0.72 |  |

### by family + stage + context

| group | n | Gemma-judge yes | Qwen-judge yes | agreement | kappa | flag |
|---|---|---|---|---|---|---|
| gemma4-31b / base / light_system | 84 | 0.50 | 0.27 | 0.75 | 0.50 |  |
| gemma4-31b / base / light_user | 84 | 0.50 | 0.32 | 0.82 | 0.64 |  |
| gemma4-31b / instruct / light_system | 84 | 0.70 | 0.68 | 0.90 | 0.78 |  |
| gemma4-31b / instruct / light_user | 84 | 0.85 | 0.77 | 0.86 | 0.54 |  |
| olmo3-7b / base / light_system | 83 | 0.49 | 0.28 | 0.76 | 0.52 |  |
| olmo3-7b / base / light_user | 83 | 0.49 | 0.37 | 0.86 | 0.71 |  |
| olmo3-7b / instruct / light_system | 83 | 0.49 | 0.31 | 0.82 | 0.64 |  |
| olmo3-7b / instruct / light_user | 83 | 0.51 | 0.43 | 0.93 | 0.86 |  |
| qwen3-8b / base / light_system | 83 | 0.49 | 0.33 | 0.83 | 0.66 |  |
| qwen3-8b / base / light_user | 83 | 0.51 | 0.34 | 0.81 | 0.62 |  |
| qwen3-8b / instruct / light_system | 83 | 0.51 | 0.37 | 0.77 | 0.54 |  |
| qwen3-8b / instruct / light_user | 83 | 0.49 | 0.42 | 0.88 | 0.76 |  |

## traits (1000 replies): overall agreement 0.751, kappa 0.50

### by trait

| group | n | Gemma-judge yes | Qwen-judge yes | agreement | kappa | flag |
|---|---|---|---|---|---|---|
| agrees_wrong_opinion | 84 | 0.50 | 0.00 | 0.50 | 0.00 | UNRELIABLE (kappa < 0.4) |
| clarifying_question | 84 | 0.50 | 0.44 | 0.92 | 0.83 |  |
| closing_offer | 96 | 0.50 | 0.43 | 0.86 | 0.73 |  |
| corrects_false_premise | 84 | 0.50 | 0.48 | 0.90 | 0.81 |  |
| declines_opinion | 84 | 0.50 | 0.26 | 0.64 | 0.29 | UNRELIABLE (kappa < 0.4) |
| denies_feelings | 84 | 0.45 | 0.36 | 0.81 | 0.61 |  |
| fabricates_vs_admits | 77 | 0.49 | 0.56 | 0.83 | 0.66 |  |
| flattery | 88 | 0.48 | 0.10 | 0.62 | 0.22 | UNRELIABLE (kappa < 0.4) |
| formal_register | 88 | 0.49 | 0.28 | 0.64 | 0.27 | UNRELIABLE (kappa < 0.4) |
| hedging_unanswerable | 77 | 0.52 | 0.52 | 0.82 | 0.64 |  |
| moralizing | 77 | 0.47 | 0.25 | 0.73 | 0.44 |  |
| safety_caveats | 77 | 0.45 | 0.19 | 0.74 | 0.45 |  |

### by context

| group | n | Gemma-judge yes | Qwen-judge yes | agreement | kappa | flag |
|---|---|---|---|---|---|---|
| assistant | 138 | 0.49 | 0.41 | 0.79 | 0.58 |  |
| light_system | 138 | 0.48 | 0.34 | 0.80 | 0.60 |  |
| light_user | 138 | 0.49 | 0.30 | 0.77 | 0.53 |  |
| narration | 138 | 0.47 | 0.28 | 0.75 | 0.48 |  |
| plain_text | 138 | 0.49 | 0.38 | 0.78 | 0.55 |  |
| strong_act | 138 | 0.49 | 0.25 | 0.70 | 0.40 | UNRELIABLE (kappa < 0.4) |
| strong_noact | 138 | 0.51 | 0.31 | 0.71 | 0.43 |  |
| user_turn | 34 | 0.50 | 0.26 | 0.59 | 0.18 | UNRELIABLE (kappa < 0.4) |

### by family

| group | n | Gemma-judge yes | Qwen-judge yes | agreement | kappa | flag |
|---|---|---|---|---|---|---|
| gemma4-31b | 243 | 0.32 | 0.29 | 0.78 | 0.48 |  |
| olmo3-7b | 513 | 0.58 | 0.36 | 0.75 | 0.52 |  |
| qwen3-8b | 244 | 0.48 | 0.27 | 0.72 | 0.43 |  |

