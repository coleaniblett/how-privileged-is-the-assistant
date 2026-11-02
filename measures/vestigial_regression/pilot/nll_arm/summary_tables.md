### Summary: bits per byte, change from each family's own base, 90 shared texts

| model (IT − own base) | wildchat Δ(user@user) | wildchat Δ(user@plain) | wildchat user role-specific | oasst1 Δ(user@user) | oasst1 Δ(user@plain) | oasst1 user role-specific | assistant role-specific (asst@asst − asst@plain) |
|---|---|---|---|---|---|---|---|
| OLMo 3 7B Instruct | -0.17 [-0.31, -0.03] | -0.10 [-0.23, +0.03] | -0.07 [-0.11, -0.03] | +0.10 [+0.07, +0.13] | +0.13 [+0.10, +0.15] | -0.03 [-0.05, -0.00] | +0.01 [-0.01, +0.03] |
| Qwen3-8B | +0.20 [+0.16, +0.24] | +0.06 [+0.02, +0.10] | +0.14 [+0.09, +0.20] | +0.12 [+0.09, +0.15] | -0.01 [-0.05, +0.03] | +0.13 [+0.08, +0.17] | +0.09 [+0.05, +0.12] |
| Gemma 4 31B IT | +2.67 [+2.38, +3.01] | +1.58 [+1.42, +1.75] | +1.10 [+0.85, +1.37] | +2.74 [+2.54, +2.94] | +1.58 [+1.32, +1.86] | +1.16 [+0.95, +1.38] | -0.82 [-1.15, -0.57] |

OLMo per stage (same 90 texts, bits/byte, change from OLMo base):

| model (IT − own base) | wildchat Δ(user@user) | wildchat Δ(user@plain) | wildchat user role-specific | oasst1 Δ(user@user) | oasst1 Δ(user@plain) | oasst1 user role-specific | assistant role-specific (asst@asst − asst@plain) |
|---|---|---|---|---|---|---|---|
| OLMo SFT | -0.23 [-0.36, -0.10] | -0.15 [-0.27, -0.04] | -0.07 [-0.12, -0.03] | +0.07 [+0.04, +0.09] | +0.08 [+0.06, +0.09] | -0.01 [-0.03, +0.01] | -0.04 [-0.05, -0.03] |
| OLMo DPO | -0.18 [-0.32, -0.04] | -0.11 [-0.24, +0.01] | -0.06 [-0.11, -0.02] | +0.09 [+0.06, +0.12] | +0.11 [+0.08, +0.13] | -0.02 [-0.04, +0.01] | +0.01 [-0.01, +0.03] |
| OLMo Instruct | -0.17 [-0.31, -0.03] | -0.10 [-0.23, +0.03] | -0.07 [-0.11, -0.03] | +0.10 [+0.07, +0.13] | +0.13 [+0.10, +0.15] | -0.03 [-0.05, -0.00] | +0.01 [-0.01, +0.03] |

Each family's change is from its own base; only bits per byte is compared across families (per-token nats are not: the tokenizers differ). Cells are mean change [95% paired bootstrap CI over texts]. All values are `condition − condition` differences, so the OLMo, Qwen and Gemma plain controls (`<|endoftext|>` / `<bos>` document starts) enter only through each family's own base.

