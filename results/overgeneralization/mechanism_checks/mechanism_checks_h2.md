# Mechanism checks (spec H; exploratory, outside every verdict)

Plan, frozen materials and readings: `docs/decisions.md` 2026-11-04, `data/mechanism_checks_materials.json`. Many cells, regex detectors, prompts specific to these roles: read patterns, not single cells.

## H2. Slot or default sentence?

Refusal (`is_refusal`, first 160 characters) and layout rate, character conditions pooled over the six roles (thin = three helpful humans, vivid = three stylized), 36 requests x 3 samples; two-way bootstrap over requests and roles. Points = percentage points. S = character in the system message, U = in the user turn; default = OLMo's default sentence; neutral = matched-length neutral sentence; empty = an explicit empty system message (the OLMo template then renders an empty system block, not the default).

| model | condition | refusal rate | layout rate |
|---|---|---|---|
| apertus-8b base | S-none | 2.0% [0.6, 3.7] | 3.4% [1.2, 6.0] |
| apertus-8b base | S-default | 2.0% [0.3, 4.3] | 1.2% [0.2, 2.8] |
| apertus-8b base | S-neutral | 2.8% [0.8, 5.7] | 0.8% [0.0, 1.9] |
| apertus-8b base | U-none | 1.9% [0.3, 3.9] | 2.5% [0.9, 4.3] |
| apertus-8b base | U-default | 2.2% [0.8, 3.9] | 1.4% [0.2, 3.1] |
| apertus-8b base | U-neutral | 2.5% [0.9, 4.2] | 1.9% [0.3, 4.0] |
| apertus-8b instruct | S-none | 39.7% [14.7, 66.8] | 8.8% [2.5, 16.2] |
| apertus-8b instruct | S-default | 42.3% [17.9, 68.5] | 10.0% [3.1, 18.1] |
| apertus-8b instruct | S-neutral | 42.0% [18.2, 66.2] | 8.6% [2.8, 16.2] |
| apertus-8b instruct | U-none | 36.4% [13.0, 60.5] | 9.0% [3.2, 16.8] |
| apertus-8b instruct | U-default | 31.2% [11.1, 55.4] | 9.9% [3.5, 18.5] |
| apertus-8b instruct | U-neutral | 31.9% [11.3, 56.8] | 7.3% [2.5, 13.7] |
| gemma4-31b base | S-none | 6.3% [3.1, 10.2] | 7.4% [4.2, 11.7] |
| gemma4-31b base | S-default | 11.4% [4.9, 18.7] | 5.2% [2.8, 8.3] |
| gemma4-31b base | S-neutral | 6.5% [3.1, 10.7] | 3.5% [1.5, 5.7] |
| gemma4-31b base | U-none | 9.9% [2.6, 19.0] | 10.5% [6.9, 14.0] |
| gemma4-31b base | U-default | 11.1% [3.9, 19.1] | 6.8% [3.9, 10.2] |
| gemma4-31b base | U-neutral | 7.1% [1.4, 14.2] | 5.1% [2.8, 7.6] |
| gemma4-31b instruct | S-none | 2.9% [0.0, 8.8] | 1.9% [0.0, 5.4] |
| gemma4-31b instruct | S-default | 4.9% [0.3, 12.5] | 2.6% [0.0, 7.9] |
| gemma4-31b instruct | S-neutral | 2.9% [0.0, 8.6] | 5.9% [0.0, 15.4] |
| gemma4-31b instruct | U-none | 7.4% [0.5, 17.6] | 0.6% [0.0, 2.0] |
| gemma4-31b instruct | U-default | 12.8% [2.3, 27.5] | 0.5% [0.0, 1.7] |
| gemma4-31b instruct | U-neutral | 8.0% [1.4, 18.4] | 0.6% [0.0, 2.5] |
| llama3.1-8b base | S-none | 2.8% [0.9, 5.4] | 7.9% [4.3, 12.3] |
| llama3.1-8b base | S-default | 4.9% [2.5, 7.9] | 9.6% [5.7, 13.7] |
| llama3.1-8b base | S-neutral | 7.9% [4.2, 12.5] | 3.2% [1.2, 5.9] |
| llama3.1-8b base | U-none | 3.7% [1.2, 7.3] | 9.6% [5.7, 14.0] |
| llama3.1-8b base | U-default | 0.9% [0.0, 2.3] | 5.7% [2.3, 9.7] |
| llama3.1-8b base | U-neutral | 3.5% [1.5, 6.0] | 2.6% [0.6, 5.6] |
| llama3.1-8b instruct | S-none | 26.5% [12.5, 42.6] | 9.3% [3.1, 17.1] |
| llama3.1-8b instruct | S-default | 25.9% [13.1, 40.4] | 10.5% [4.5, 18.4] |
| llama3.1-8b instruct | S-neutral | 43.7% [26.9, 59.9] | 5.4% [0.6, 12.0] |
| llama3.1-8b instruct | U-none | 17.4% [6.6, 31.5] | 6.8% [1.5, 14.4] |
| llama3.1-8b instruct | U-default | 19.1% [8.3, 31.9] | 9.0% [2.6, 17.3] |
| llama3.1-8b instruct | U-neutral | 27.8% [11.3, 46.8] | 6.5% [1.9, 13.6] |
| olmo3-7b base | S-none | 33.2% [18.1, 49.2] | 6.2% [2.3, 11.6] |
| olmo3-7b base | S-default | 33.6% [22.4, 45.8] | 5.6% [2.2, 9.7] |
| olmo3-7b base | S-neutral | 32.4% [18.2, 47.7] | 6.6% [3.2, 11.3] |
| olmo3-7b base | U-default | 63.4% [52.9, 74.1] | 4.9% [1.2, 9.7] |
| olmo3-7b base | U-empty | 44.0% [33.6, 54.2] | 8.3% [3.4, 15.3] |
| olmo3-7b base | U-neutral | 49.7% [36.0, 64.4] | 6.8% [2.8, 12.8] |
| olmo3-7b base | A-default | 60.2% [48.1, 71.3] | 5.6% [1.9, 10.2] |
| olmo3-7b base | A-empty | 26.9% [17.6, 36.1] | 9.3% [1.9, 18.5] |
| olmo3-7b base | A-neutral | 38.0% [27.8, 48.1] | 4.6% [0.9, 10.2] |
| olmo3-7b dpo | S-none | 52.3% [30.4, 73.9] | 6.0% [2.0, 11.3] |
| olmo3-7b dpo | S-default | 50.3% [24.2, 75.8] | 5.9% [2.2, 11.1] |
| olmo3-7b dpo | S-neutral | 52.3% [29.0, 74.8] | 3.7% [1.2, 7.3] |
| olmo3-7b dpo | U-default | 33.6% [10.5, 58.8] | 8.0% [2.9, 15.0] |
| olmo3-7b dpo | U-empty | 33.6% [11.1, 58.6] | 9.7% [4.9, 15.6] |
| olmo3-7b dpo | U-neutral | 30.9% [9.9, 54.0] | 8.5% [4.2, 13.6] |
| olmo3-7b dpo | A-default | 77.8% [64.8, 88.9] | 10.2% [3.7, 18.5] |
| olmo3-7b dpo | A-empty | 74.1% [61.1, 85.2] | 9.3% [3.7, 15.7] |
| olmo3-7b dpo | A-neutral | 66.7% [52.8, 79.6] | 9.3% [2.8, 17.6] |
| olmo3-7b instruct | S-none | 48.6% [26.4, 71.0] | 6.0% [1.5, 11.7] |
| olmo3-7b instruct | S-default | 47.7% [23.5, 71.8] | 5.9% [1.7, 11.6] |
| olmo3-7b instruct | S-neutral | 50.0% [26.5, 73.8] | 5.6% [2.0, 10.0] |
| olmo3-7b instruct | U-default | 31.2% [10.0, 54.8] | 9.3% [4.2, 15.6] |
| olmo3-7b instruct | U-empty | 30.6% [9.6, 55.1] | 8.5% [3.7, 14.2] |
| olmo3-7b instruct | U-neutral | 28.9% [9.1, 52.0] | 7.6% [2.5, 13.7] |
| olmo3-7b instruct | A-default | 80.6% [68.5, 90.7] | 6.5% [0.0, 14.8] |
| olmo3-7b instruct | A-empty | 75.9% [63.9, 88.0] | 5.6% [1.9, 11.1] |
| olmo3-7b instruct | A-neutral | 70.4% [57.4, 82.4] | 10.2% [3.7, 18.5] |
| olmo3-7b sft | S-none | 72.4% [56.2, 87.2] | 3.2% [0.3, 8.0] |
| olmo3-7b sft | S-default | 66.0% [46.6, 83.6] | 3.4% [0.5, 8.2] |
| olmo3-7b sft | S-neutral | 68.5% [50.9, 85.0] | 3.1% [0.0, 8.6] |
| olmo3-7b sft | U-default | 36.4% [14.0, 60.5] | 9.4% [4.8, 15.0] |
| olmo3-7b sft | U-empty | 37.5% [14.8, 61.9] | 12.5% [6.6, 19.4] |
| olmo3-7b sft | U-neutral | 30.9% [11.3, 52.3] | 13.4% [7.3, 20.5] |
| olmo3-7b sft | A-default | 88.0% [77.8, 96.3] | 2.8% [0.0, 7.4] |
| olmo3-7b sft | A-empty | 85.2% [75.0, 93.5] | 2.8% [0.0, 5.6] |
| olmo3-7b sft | A-neutral | 81.5% [70.4, 90.7] | 2.8% [0.0, 8.3] |
| qwen3-8b base | S-none | 33.6% [20.5, 47.5] | 5.7% [2.8, 9.4] |
| qwen3-8b base | S-default | 26.9% [16.8, 37.3] | 7.9% [4.0, 13.0] |
| qwen3-8b base | U-none | 33.0% [22.4, 44.1] | 11.9% [6.2, 18.8] |
| qwen3-8b base | U-default | 18.7% [13.0, 25.0] | 6.9% [4.0, 10.2] |
| qwen3-8b instruct | S-none | 45.5% [15.1, 77.9] | 11.4% [4.2, 19.9] |
| qwen3-8b instruct | S-default | 54.8% [24.7, 84.7] | 8.8% [3.1, 16.2] |
| qwen3-8b instruct | U-none | 43.2% [15.4, 73.0] | 13.6% [5.6, 23.1] |
| qwen3-8b instruct | U-default | 56.2% [23.9, 86.3] | 6.6% [2.0, 13.0] |

### Pre-set readings (OLMo instruct; points, 95% CI)

| contrast | base | instruct | reading |
|---|---|---|---|
| slot: S-neutral minus U-neutral | -17.3 [-24.8, -9.3] | +21.1 [+13.4, +29.6] | at least 10 points, CI above 0; larger than at base: yes (difference +38.4 [+28.9, +48.1]) |
| sentence (user turn): U-default minus U-neutral | +13.7 [+3.5, +23.9] | +2.3 [-2.0, +7.3] | not met |
| sentence (system): S-default minus S-neutral | +1.2 [-7.6, +10.0] | -2.3 [-6.5, +1.7] | not met |

### Transplant into qwen3-8b (OLMo's sentence as system text)

| stage | slot gap with the sentence (S-default minus U-default) | slot gap without it (S-none minus U-none) | added by the sentence [95% CI] | reading |
|---|---|---|---|---|
| base | +8.2 | +0.6 | +7.6 [-3.2, +19.8] | not met |
| instruct | -1.4 | +2.3 | -3.7 [-11.4, +3.4] | not met |

### Transplant into llama3.1-8b (OLMo's sentence as system text; BREADTH CHECK, not blind)

| stage | slot gap with the sentence (S-default minus U-default) | slot gap without it (S-none minus U-none) | added by the sentence [95% CI] | reading |
|---|---|---|---|---|
| base | +4.0 | -0.9 | +4.9 [+0.9, +9.1] | not met |
| instruct | +6.8 | +9.1 | -2.3 [-9.6, +4.6] | not met |

### Transplant into apertus-8b (OLMo's sentence as system text; BREADTH CHECK, not blind)

| stage | slot gap with the sentence (S-default minus U-default) | slot gap without it (S-none minus U-none) | added by the sentence [95% CI] | reading |
|---|---|---|---|---|
| base | -0.2 | +0.2 | -0.3 [-3.5, +3.4] | not met |
| instruct | +11.1 | +3.2 | +7.9 [+2.0, +14.7] | not met |

### Transplant into gemma4-31b (OLMo's sentence as system text; BREADTH CHECK, not blind)

| stage | slot gap with the sentence (S-default minus U-default) | slot gap without it (S-none minus U-none) | added by the sentence [95% CI] | reading |
|---|---|---|---|---|
| base | +0.3 | -3.5 | +3.9 [-3.2, +13.3] | not met |
| instruct | -7.9 | -4.5 | -3.4 [-9.6, +0.5] | not met |

### Slot effect in llama3.1-8b (BREADTH CHECK, not blind; same readings as OLMo: at least 10 points, CI above 0, larger than at base)

| contrast | base | instruct | reading |
|---|---|---|---|
| S-neutral minus U-neutral | +4.3 [+0.2, +9.1] | +15.9 [+9.1, +23.5] | at least 10 points, CI excluding 0; larger than at base: yes (difference +11.6 [+2.9, +21.0]) |
| S-none minus U-none (native text) | -0.9 [-4.6, +2.6] | +9.1 [+2.9, +16.4] | not met; larger than at base: yes (difference +10.0 [+3.5, +16.8]) |
| sentence (user turn): U-default minus U-neutral | -2.6 [-5.1, -0.3] | -8.6 [-19.3, -0.5] | not met; larger than at base: no (difference -6.0 [-16.5, +3.1]) |

### Slot effect in apertus-8b (BREADTH CHECK, not blind; same readings as OLMo: at least 10 points, CI above 0, larger than at base)

| contrast | base | instruct | reading |
|---|---|---|---|
| S-neutral minus U-neutral | +0.3 [-2.2, +3.2] | +10.0 [+4.2, +16.8] | at least 10 points, CI excluding 0; larger than at base: yes (difference +9.7 [+2.8, +17.7]) |
| S-none minus U-none (native text) | +0.2 [-2.3, +2.5] | +3.2 [-2.2, +9.4] | not met; larger than at base: no (difference +3.1 [-2.9, +9.7]) |
| sentence (user turn): U-default minus U-neutral | -0.3 [-2.6, +1.7] | -0.8 [-6.0, +3.7] | not met; larger than at base: no (difference -0.5 [-5.6, +4.6]) |

### Slot effect in gemma4-31b (BREADTH CHECK, not blind; same readings as OLMo: at least 10 points, CI above 0, larger than at base)

| contrast | base | instruct | reading |
|---|---|---|---|
| S-neutral minus U-neutral | -0.6 [-6.5, +4.3] | -5.1 [-12.7, -0.3] | not met; larger than at base: no (difference -4.5 [-12.8, +2.2]) |
| S-none minus U-none (native text) | -3.5 [-11.6, +2.6] | -4.5 [-11.3, -0.2] | not met; larger than at base: no (difference -0.9 [-9.7, +7.6]) |
| sentence (user turn): U-default minus U-neutral | +4.0 [-0.9, +9.4] | +4.8 [+0.3, +11.1] | not met; larger than at base: no (difference +0.8 [-5.1, +8.2]) |

