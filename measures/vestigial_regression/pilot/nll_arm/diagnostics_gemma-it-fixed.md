# Diagnostics: gemma-it-fixed (google/gemma-4-31B-it rendering)

## 1a. Raw mean NLL per slot (no subtraction, no reference model)

| text set | quantity | @user | @asst | @plain |
|---|---|---|---|---|
| wildchat user | nats/token, positions 1+ | 10.202 [9.375, 11.053] | 4.518 [3.973, 5.108] | 7.251 [6.493, 8.158] |
| wildchat user | nats/token, all positions | 10.279 [9.455, 11.133] | 4.778 [4.221, 5.378] | 7.344 [6.576, 8.268] |
| oasst1 user | nats/token, positions 1+ | 11.034 [10.297, 11.735] | 4.565 [4.107, 5.042] | 7.399 [6.629, 8.212] |
| oasst1 user | nats/token, all positions | 11.167 [10.452, 11.871] | 4.943 [4.472, 5.424] | 7.540 [6.754, 8.300] |
| wildchat assistant | nats/token, positions 1+ | 7.639 [7.119, 8.204] | 3.171 [2.692, 3.726] | 5.305 [4.763, 5.901] |
| wildchat assistant | nats/token, all positions | 7.724 [7.217, 8.284] | 3.353 [2.879, 3.906] | 5.397 [4.857, 5.998] |

## 1b. Rendering

`<bos>` in the token ids (all 90 texts, every condition):

| condition | sequences | start with `<bos>` | exactly one `<bos>` |
|---|---|---|---|
| asst@asst | 30 | 30 | 30 |
| asst@plain | 30 | 30 | 30 |
| asst@user | 30 | 30 | 30 |
| user@asst | 60 | 60 | 60 |
| user@plain | 60 | 60 | 60 |
| user@user | 60 | 60 | 60 |

Template behaviour (special tokens visible):

- system+user: `'<bos><|turn>system\nSYS<turn|>\n<|turn>user\nU<turn|>\n'`
- user+gen_prompt: `'<bos><|turn>user\nU<turn|>\n<|turn>model\n<|channel>thought\n<channel|>'`
- user,assistant: `'<bos><|turn>user\nU<turn|>\n<|turn>model\nA<turn|>\n'`
- assistant only: `'<bos><|turn>model\nA<turn|>\n'`
- user+gen_prompt, enable_thinking=True: `'<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nU<turn|>\n<|turn>model\n'`

Three rendered examples per condition (prefix | first 12 content tokens | suffix; special tokens kept):

- `asst@asst` wildchat-assistant-0000: `'<bos><|turn>model\n<|channel>thought\n<channel|>'` | `'Jen nods and introduces herself. Sandra asks if she’s'` | `'<turn|>\n'`
- `asst@asst` wildchat-assistant-0001: `'<bos><|turn>model\n<|channel>thought\n<channel|>'` | `'The opposite of an ordinary apple, but essentially the same in'` | `'<turn|>\n'`
- `asst@asst` wildchat-assistant-0002: `'<bos><|turn>model\n<|channel>thought\n<channel|>'` | `'Terra: "I\'m not sure, but it sounded'` | `'<turn|>\n'`
- `asst@plain` wildchat-assistant-0000: `'<bos>'` | `'Jen nods and introduces herself. Sandra asks if she’s'` | `''`
- `asst@plain` wildchat-assistant-0001: `'<bos>'` | `'The opposite of an ordinary apple, but essentially the same in'` | `''`
- `asst@plain` wildchat-assistant-0002: `'<bos>'` | `'Terra: "I\'m not sure, but it sounded'` | `''`
- `asst@user` wildchat-assistant-0000: `'<bos><|turn>user\n'` | `'Jen nods and introduces herself. Sandra asks if she’s'` | `'<turn|>\n'`
- `asst@user` wildchat-assistant-0001: `'<bos><|turn>user\n'` | `'The opposite of an ordinary apple, but essentially the same in'` | `'<turn|>\n'`
- `asst@user` wildchat-assistant-0002: `'<bos><|turn>user\n'` | `'Terra: "I\'m not sure, but it sounded'` | `'<turn|>\n'`
- `user@asst` wildchat-user-0000: `'<bos><|turn>model\n<|channel>thought\n<channel|>'` | `'David works at Sandra’s cafe. Sandra has been hypnot'` | `'<turn|>\n'`
- `user@asst` wildchat-user-0001: `'<bos><|turn>model\n<|channel>thought\n<channel|>'` | `'Describe opposite of ordinary apple but same as ordinary apple, including'` | `'<turn|>\n'`
- `user@asst` wildchat-user-0002: `'<bos><|turn>model\n<|channel>thought\n<channel|>'` | `'(In the city of Narshe…)\n\nNatsuki'` | `'<turn|>\n'`
- `user@plain` wildchat-user-0000: `'<bos>'` | `'David works at Sandra’s cafe. Sandra has been hypnot'` | `''`
- `user@plain` wildchat-user-0001: `'<bos>'` | `'Describe opposite of ordinary apple but same as ordinary apple, including'` | `''`
- `user@plain` wildchat-user-0002: `'<bos>'` | `'(In the city of Narshe…)\n\nNatsuki'` | `''`
- `user@user` wildchat-user-0000: `'<bos><|turn>user\n'` | `'David works at Sandra’s cafe. Sandra has been hypnot'` | `'<turn|>\n'`
- `user@user` wildchat-user-0001: `'<bos><|turn>user\n'` | `'Describe opposite of ordinary apple but same as ordinary apple, including'` | `'<turn|>\n'`
- `user@user` wildchat-user-0002: `'<bos><|turn>user\n'` | `'(In the city of Narshe…)\n\nNatsuki'` | `'<turn|>\n'`

## 1c. Position profile (mean NLL, nats/token, by content position)

Texts with at least 40 content tokens, so every position averages over the same texts.

### wildchat user (25 texts with >= 40 tokens)

| positions | user@user | user@asst | user@plain | own slot - plain | other slot - plain |
|---|---|---|---|---|---|
| 0 | 16.35 | 21.33 | 14.56 | +1.79 | +6.77 |
| 1-4 | 17.76 | 10.50 | 10.68 | +7.08 | -0.18 |
| 5-19 | 12.25 | 5.88 | 7.59 | +4.66 | -1.71 |
| 20-39 | 10.45 | 4.22 | 6.30 | +4.15 | -2.08 |

### oasst1 user (9 texts with >= 40 tokens)

| positions | user@user | user@asst | user@plain | own slot - plain | other slot - plain |
|---|---|---|---|---|---|
| 0 | 14.12 | 19.12 | 12.21 | +1.91 | +6.91 |
| 1-4 | 17.70 | 6.79 | 7.74 | +9.97 | -0.95 |
| 5-19 | 9.86 | 4.64 | 6.46 | +3.40 | -1.82 |
| 20-39 | 10.06 | 3.68 | 6.96 | +3.10 | -3.28 |

### wildchat assistant (27 texts with >= 40 tokens)

| positions | asst@user | asst@asst | asst@plain | own slot - plain | other slot - plain |
|---|---|---|---|---|---|
| 0 | 15.55 | 23.81 | 13.90 | +9.91 | +1.66 |
| 1-4 | 18.09 | 8.32 | 11.80 | -3.48 | +6.28 |
| 5-19 | 9.61 | 4.04 | 5.96 | -1.91 | +3.66 |
| 20-39 | 8.13 | 3.69 | 5.68 | -1.99 | +2.44 |

## 1d. Bits per byte

| text set | BPB | @user | @asst | @plain |
|---|---|---|---|---|
| wildchat user | primary (positions 1+) | 3.595 [3.254, 3.996] | 1.543 [1.398, 1.697] | 2.497 [2.293, 2.721] |
| wildchat user | secondary (all positions) | 3.620 [3.289, 4.008] | 1.632 [1.483, 1.795] | 2.532 [2.317, 2.750] |
| oasst1 user | primary (positions 1+) | 3.577 [3.337, 3.806] | 1.462 [1.342, 1.576] | 2.412 [2.124, 2.725] |
| oasst1 user | secondary (all positions) | 3.646 [3.403, 3.883] | 1.594 [1.476, 1.711] | 2.476 [2.198, 2.751] |
| wildchat assistant | primary (positions 1+) | 2.570 [2.285, 2.982] | 1.033 [0.883, 1.215] | 1.838 [1.522, 2.282] |
| wildchat assistant | secondary (all positions) | 2.611 [2.305, 3.021] | 1.097 [0.938, 1.272] | 1.876 [1.555, 2.308] |

