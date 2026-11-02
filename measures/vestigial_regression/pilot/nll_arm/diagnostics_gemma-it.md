# Diagnostics: gemma-it (google/gemma-4-31B-it rendering)

## 1a. Raw mean NLL per slot (no subtraction, no reference model)

| text set | quantity | @user | @asst | @plain |
|---|---|---|---|---|
| wildchat user | nats/token, positions 1+ | 10.202 [9.375, 11.053] | 9.220 [8.376, 10.051] | 7.251 [6.493, 8.158] |
| wildchat user | nats/token, all positions | 10.279 [9.455, 11.133] | 9.489 [8.628, 10.264] | 7.344 [6.576, 8.268] |
| oasst1 user | nats/token, positions 1+ | 11.034 [10.297, 11.735] | 9.947 [9.126, 10.704] | 7.399 [6.629, 8.212] |
| oasst1 user | nats/token, all positions | 11.167 [10.452, 11.871] | 10.472 [9.628, 11.220] | 7.540 [6.754, 8.300] |
| wildchat assistant | nats/token, positions 1+ | 7.639 [7.119, 8.204] | 6.638 [6.065, 7.260] | 5.305 [4.763, 5.901] |
| wildchat assistant | nats/token, all positions | 7.724 [7.217, 8.284] | 6.898 [6.329, 7.501] | 5.397 [4.857, 5.998] |

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

- `asst@asst` wildchat-assistant-0000: `'<bos><|turn>model\n'` | `'Jen nods and introduces herself. Sandra asks if she’s'` | `'<turn|>\n'`
- `asst@asst` wildchat-assistant-0001: `'<bos><|turn>model\n'` | `'The opposite of an ordinary apple, but essentially the same in'` | `'<turn|>\n'`
- `asst@asst` wildchat-assistant-0002: `'<bos><|turn>model\n'` | `'Terra: "I\'m not sure, but it sounded'` | `'<turn|>\n'`
- `asst@plain` wildchat-assistant-0000: `'<bos>'` | `'Jen nods and introduces herself. Sandra asks if she’s'` | `''`
- `asst@plain` wildchat-assistant-0001: `'<bos>'` | `'The opposite of an ordinary apple, but essentially the same in'` | `''`
- `asst@plain` wildchat-assistant-0002: `'<bos>'` | `'Terra: "I\'m not sure, but it sounded'` | `''`
- `asst@user` wildchat-assistant-0000: `'<bos><|turn>user\n'` | `'Jen nods and introduces herself. Sandra asks if she’s'` | `'<turn|>\n'`
- `asst@user` wildchat-assistant-0001: `'<bos><|turn>user\n'` | `'The opposite of an ordinary apple, but essentially the same in'` | `'<turn|>\n'`
- `asst@user` wildchat-assistant-0002: `'<bos><|turn>user\n'` | `'Terra: "I\'m not sure, but it sounded'` | `'<turn|>\n'`
- `user@asst` wildchat-user-0000: `'<bos><|turn>model\n'` | `'David works at Sandra’s cafe. Sandra has been hypnot'` | `'<turn|>\n'`
- `user@asst` wildchat-user-0001: `'<bos><|turn>model\n'` | `'Describe opposite of ordinary apple but same as ordinary apple, including'` | `'<turn|>\n'`
- `user@asst` wildchat-user-0002: `'<bos><|turn>model\n'` | `'(In the city of Narshe…)\n\nNatsuki'` | `'<turn|>\n'`
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
| 0 | 16.35 | 30.05 | 14.56 | +1.79 | +15.49 |
| 1-4 | 17.76 | 13.60 | 10.68 | +7.08 | +2.92 |
| 5-19 | 12.25 | 11.09 | 7.59 | +4.66 | +3.50 |
| 20-39 | 10.45 | 9.65 | 6.30 | +4.15 | +3.35 |

### oasst1 user (9 texts with >= 40 tokens)

| positions | user@user | user@asst | user@plain | own slot - plain | other slot - plain |
|---|---|---|---|---|---|
| 0 | 14.12 | 28.89 | 12.21 | +1.91 | +16.68 |
| 1-4 | 17.70 | 14.89 | 7.74 | +9.97 | +7.15 |
| 5-19 | 9.86 | 8.91 | 6.46 | +3.40 | +2.45 |
| 20-39 | 10.06 | 9.19 | 6.96 | +3.10 | +2.23 |

### wildchat assistant (27 texts with >= 40 tokens)

| positions | asst@user | asst@asst | asst@plain | own slot - plain | other slot - plain |
|---|---|---|---|---|---|
| 0 | 15.55 | 30.48 | 13.90 | +16.59 | +1.66 |
| 1-4 | 18.09 | 13.75 | 11.80 | +1.95 | +6.28 |
| 5-19 | 9.61 | 8.25 | 5.96 | +2.29 | +3.66 |
| 20-39 | 8.13 | 7.11 | 5.68 | +1.42 | +2.44 |

## 1d. Bits per byte

| text set | BPB | @user | @asst | @plain |
|---|---|---|---|---|
| wildchat user | primary (positions 1+) | 3.595 [3.255, 3.996] | 3.237 [2.931, 3.611] | 2.497 [2.293, 2.721] |
| wildchat user | secondary (all positions) | 3.620 [3.289, 4.008] | 3.329 [3.031, 3.704] | 2.532 [2.317, 2.750] |
| oasst1 user | primary (positions 1+) | 3.577 [3.337, 3.806] | 3.218 [2.957, 3.438] | 2.412 [2.124, 2.725] |
| oasst1 user | secondary (all positions) | 3.646 [3.403, 3.883] | 3.411 [3.152, 3.648] | 2.476 [2.198, 2.751] |
| wildchat assistant | primary (positions 1+) | 2.570 [2.285, 2.982] | 2.247 [1.939, 2.668] | 1.838 [1.522, 2.282] |
| wildchat assistant | secondary (all positions) | 2.611 [2.305, 3.021] | 2.341 [2.043, 2.742] | 1.876 [1.555, 2.308] |

