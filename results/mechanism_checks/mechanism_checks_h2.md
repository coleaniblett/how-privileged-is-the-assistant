# Mechanism checks (spec H; exploratory, outside every verdict)

Plan, frozen materials and readings: `docs/decisions.md` 2026-11-04, `data/mechanism_checks_materials.json`. Many cells, regex detectors, prompts specific to these roles: read patterns, not single cells.

## H2. Slot or default sentence?

Refusal (`is_refusal`, first 160 characters) and layout rate, character conditions pooled over the six roles (thin = three helpful humans, vivid = three stylized), 36 requests x 3 samples; two-way bootstrap over requests and roles. Points = percentage points. S = character in the system message, U = in the user turn; default = OLMo's default sentence; neutral = matched-length neutral sentence; empty = an explicit empty system message (the OLMo template then renders an empty system block, not the default).

| model | condition | refusal rate | layout rate |
|---|---|---|---|
| olmo3-7b base | S-none | 33.2% [18.1, 49.7] | 6.2% [2.5, 11.4] |
| olmo3-7b base | S-default | 33.6% [22.8, 46.3] | 5.6% [2.2, 9.7] |
| olmo3-7b base | S-neutral | 32.4% [18.4, 48.0] | 6.6% [3.1, 11.1] |
| olmo3-7b base | U-default | 63.4% [52.8, 73.3] | 4.9% [1.4, 10.0] |
| olmo3-7b base | U-empty | 44.0% [33.6, 54.5] | 8.3% [3.4, 15.3] |
| olmo3-7b base | U-neutral | 49.7% [36.0, 64.2] | 6.8% [2.8, 12.7] |
| olmo3-7b base | A-default | 60.2% [49.1, 71.3] | 5.6% [1.9, 10.2] |
| olmo3-7b base | A-empty | 26.9% [18.5, 36.1] | 9.3% [1.9, 17.6] |
| olmo3-7b base | A-neutral | 38.0% [26.9, 48.1] | 4.6% [0.9, 10.2] |
| olmo3-7b dpo | S-none | 52.3% [29.5, 73.8] | 6.0% [2.2, 11.4] |
| olmo3-7b dpo | S-default | 50.3% [24.5, 75.8] | 5.9% [2.2, 11.3] |
| olmo3-7b dpo | S-neutral | 52.3% [29.9, 74.8] | 3.7% [1.2, 7.1] |
| olmo3-7b dpo | U-default | 33.6% [10.6, 60.0] | 8.0% [2.9, 14.8] |
| olmo3-7b dpo | U-empty | 33.6% [10.3, 59.3] | 9.7% [4.9, 15.4] |
| olmo3-7b dpo | U-neutral | 30.9% [10.6, 53.9] | 8.5% [4.3, 13.3] |
| olmo3-7b dpo | A-default | 77.8% [65.7, 88.9] | 10.2% [3.7, 18.5] |
| olmo3-7b dpo | A-empty | 74.1% [61.1, 85.2] | 9.3% [3.7, 15.7] |
| olmo3-7b dpo | A-neutral | 66.7% [53.7, 79.6] | 9.3% [2.8, 17.6] |
| olmo3-7b instruct | S-none | 48.6% [26.7, 70.7] | 6.0% [1.7, 11.9] |
| olmo3-7b instruct | S-default | 47.7% [22.7, 72.5] | 5.9% [1.7, 11.4] |
| olmo3-7b instruct | S-neutral | 50.0% [26.5, 73.6] | 5.6% [2.0, 10.0] |
| olmo3-7b instruct | U-default | 31.2% [9.4, 55.6] | 9.3% [4.2, 15.7] |
| olmo3-7b instruct | U-empty | 30.6% [9.9, 55.1] | 8.5% [3.9, 14.0] |
| olmo3-7b instruct | U-neutral | 28.9% [9.3, 51.9] | 7.6% [2.8, 14.2] |
| olmo3-7b instruct | A-default | 80.6% [68.5, 90.7] | 6.5% [0.0, 14.8] |
| olmo3-7b instruct | A-empty | 75.9% [63.0, 87.0] | 5.6% [0.9, 11.1] |
| olmo3-7b instruct | A-neutral | 70.4% [56.5, 83.3] | 10.2% [3.7, 18.5] |
| olmo3-7b sft | S-none | 72.4% [56.3, 87.0] | 3.2% [0.5, 8.0] |
| olmo3-7b sft | S-default | 66.0% [46.6, 83.6] | 3.4% [0.5, 8.2] |
| olmo3-7b sft | S-neutral | 68.5% [50.0, 85.2] | 3.1% [0.0, 8.5] |
| olmo3-7b sft | U-default | 36.4% [14.5, 61.3] | 9.4% [4.9, 14.8] |
| olmo3-7b sft | U-empty | 37.5% [15.1, 61.9] | 12.5% [6.5, 19.6] |
| olmo3-7b sft | U-neutral | 30.9% [11.9, 51.9] | 13.4% [7.6, 20.7] |
| olmo3-7b sft | A-default | 88.0% [78.7, 96.3] | 2.8% [0.0, 7.4] |
| olmo3-7b sft | A-empty | 85.2% [75.9, 93.5] | 2.8% [0.0, 6.5] |
| olmo3-7b sft | A-neutral | 81.5% [70.4, 91.7] | 2.8% [0.0, 8.3] |
| qwen3-8b base | S-none | 33.6% [20.7, 47.1] | 5.7% [2.6, 9.4] |
| qwen3-8b base | S-default | 26.9% [17.0, 37.0] | 7.9% [4.0, 13.4] |
| qwen3-8b base | U-none | 33.0% [22.1, 43.7] | 11.9% [6.2, 19.0] |
| qwen3-8b base | U-default | 18.7% [13.0, 24.8] | 6.9% [4.0, 10.0] |
| qwen3-8b instruct | S-none | 45.5% [15.7, 77.6] | 11.4% [4.5, 19.9] |
| qwen3-8b instruct | S-default | 54.8% [24.8, 83.0] | 8.8% [2.9, 16.4] |
| qwen3-8b instruct | U-none | 43.2% [15.3, 73.0] | 13.6% [5.7, 23.0] |
| qwen3-8b instruct | U-default | 56.2% [24.1, 86.6] | 6.6% [1.9, 13.0] |

### Pre-set readings (OLMo instruct; points, 95% CI)

| contrast | base | instruct | reading |
|---|---|---|---|
| slot: S-neutral minus U-neutral | -17.3 [-25.2, -9.3] | +21.1 [+13.4, +29.6] | at least 10 points, CI above 0; larger than at base: yes (difference +38.4 [+29.0, +48.3]) |
| sentence (user turn): U-default minus U-neutral | +13.7 [+3.7, +23.8] | +2.3 [-1.9, +7.7] | not met |
| sentence (system): S-default minus S-neutral | +1.2 [-7.4, +10.0] | -2.3 [-6.5, +1.9] | not met |

### Transplant into Qwen (OLMo's sentence as system text)

| stage | slot gap with the sentence (S-default minus U-default) | slot gap without it (S-none minus U-none) | added by the sentence [95% CI] | reading |
|---|---|---|---|---|
| base | +8.2 | +0.6 | +7.6 [-3.4, +20.1] | not met |
| instruct | -1.4 | +2.3 | -3.7 [-10.8, +3.4] | not met |

