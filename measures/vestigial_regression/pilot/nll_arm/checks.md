# Checks on the existing pilot scores

No GPU, no new scoring. Positions 1+ are primary; paired bootstrap over texts (5000 resamples, seed 0); BPB is primary BPB (summed NLL over positions 1+ / ln 2 / UTF-8 bytes of the text decoded from content tokens 1+). Each family's change is from its own base.

## Check 1: where does Qwen's rise come from?

Same 30 texts per set for both families (the pilot's 90 texts).

### Qwen3-8B

**wildchat user**

| slot | nats base | nats IT | nats IT − base | BPB base | BPB IT | BPB IT − base |
|---|---|---|---|---|---|---|
| user@user | 2.762 | 3.369 | +0.608 [+0.487, +0.735] | 0.940 | 1.139 | +0.200 [+0.163, +0.238] |
| user@asst | 2.730 | 3.236 | +0.507 [+0.411, +0.613] | 0.929 | 1.095 | +0.166 [+0.137, +0.197] |
| user@plain | 3.802 | 3.974 | +0.172 [+0.036, +0.299] | 1.290 | 1.345 | +0.056 [+0.013, +0.096] |

Decomposition of the role-specific change (own slot − plain):

| unit | Δ(own slot) | Δ(plain) | role-specific change = Δ(own) − Δ(plain) |
|---|---|---|---|
| nats/token | +0.608 [+0.484, +0.742] | +0.172 [+0.039, +0.297] | +0.436 [+0.257, +0.612] |
| bits/byte | +0.200 [+0.164, +0.237] | +0.056 [+0.014, +0.096] | +0.144 [+0.090, +0.197] |

**oasst1 user**

| slot | nats base | nats IT | nats IT − base | BPB base | BPB IT | BPB IT − base |
|---|---|---|---|---|---|---|
| user@user | 2.628 | 2.985 | +0.357 [+0.262, +0.450] | 0.837 | 0.953 | +0.116 [+0.087, +0.145] |
| user@asst | 2.567 | 2.917 | +0.350 [+0.269, +0.434] | 0.818 | 0.931 | +0.113 [+0.088, +0.138] |
| user@plain | 3.814 | 3.787 | -0.028 [-0.148, +0.087] | 1.212 | 1.203 | -0.009 [-0.046, +0.026] |

Decomposition of the role-specific change (own slot − plain):

| unit | Δ(own slot) | Δ(plain) | role-specific change = Δ(own) − Δ(plain) |
|---|---|---|---|
| nats/token | +0.357 [+0.261, +0.454] | -0.028 [-0.145, +0.083] | +0.385 [+0.252, +0.514] |
| bits/byte | +0.116 [+0.086, +0.144] | -0.009 [-0.046, +0.025] | +0.125 [+0.085, +0.167] |

**wildchat assistant**

| slot | nats base | nats IT | nats IT − base | BPB base | BPB IT | BPB IT − base |
|---|---|---|---|---|---|---|
| asst@user | 1.935 | 2.359 | +0.424 [+0.375, +0.471] | 0.634 | 0.769 | +0.135 [+0.120, +0.150] |
| asst@asst | 1.889 | 2.292 | +0.403 [+0.345, +0.466] | 0.620 | 0.748 | +0.128 [+0.110, +0.147] |
| asst@plain | 2.778 | 2.884 | +0.106 [+0.029, +0.181] | 0.896 | 0.936 | +0.040 [+0.015, +0.067] |

Decomposition of the role-specific change (own slot − plain):

| unit | Δ(own slot) | Δ(plain) | role-specific change = Δ(own) − Δ(plain) |
|---|---|---|---|
| nats/token | +0.403 [+0.346, +0.465] | +0.106 [+0.026, +0.185] | +0.297 [+0.190, +0.412] |
| bits/byte | +0.128 [+0.111, +0.148] | +0.040 [+0.015, +0.066] | +0.088 [+0.055, +0.123] |

### Gemma 4 31B

**wildchat user**

| slot | nats base | nats IT | nats IT − base | BPB base | BPB IT | BPB IT − base |
|---|---|---|---|---|---|---|
| user@user | 2.691 | 10.202 | +7.511 [+6.926, +8.087] | 0.924 | 3.595 | +2.672 [+2.392, +3.015] |
| user@asst | 2.745 | 4.518 | +1.773 [+1.544, +2.012] | 0.942 | 1.543 | +0.600 [+0.534, +0.670] |
| user@plain | 2.683 | 7.251 | +4.569 [+4.017, +5.222] | 0.921 | 2.497 | +1.576 [+1.411, +1.749] |

Decomposition of the role-specific change (own slot − plain):

| unit | Δ(own slot) | Δ(plain) | role-specific change = Δ(own) − Δ(plain) |
|---|---|---|---|
| nats/token | +7.511 [+6.916, +8.123] | +4.569 [+4.030, +5.193] | +2.943 [+2.405, +3.448] |
| bits/byte | +2.672 [+2.392, +3.017] | +1.576 [+1.418, +1.754] | +1.096 [+0.844, +1.385] |

**oasst1 user**

| slot | nats base | nats IT | nats IT − base | BPB base | BPB IT | BPB IT − base |
|---|---|---|---|---|---|---|
| user@user | 2.604 | 11.034 | +8.429 [+7.854, +8.983] | 0.837 | 3.577 | +2.740 [+2.538, +2.944] |
| user@asst | 2.727 | 4.565 | +1.839 [+1.587, +2.111] | 0.876 | 1.462 | +0.586 [+0.516, +0.653] |
| user@plain | 2.599 | 7.399 | +4.799 [+4.089, +5.538] | 0.836 | 2.412 | +1.576 [+1.323, +1.867] |

Decomposition of the role-specific change (own slot − plain):

| unit | Δ(own slot) | Δ(plain) | role-specific change = Δ(own) − Δ(plain) |
|---|---|---|---|
| nats/token | +8.429 [+7.852, +8.970] | +4.799 [+4.063, +5.550] | +3.630 [+2.931, +4.318] |
| bits/byte | +2.740 [+2.539, +2.949] | +1.576 [+1.330, +1.853] | +1.164 [+0.948, +1.383] |

**wildchat assistant**

| slot | nats base | nats IT | nats IT − base | BPB base | BPB IT | BPB IT − base |
|---|---|---|---|---|---|---|
| asst@user | 1.782 | 7.639 | +5.857 [+5.505, +6.217] | 0.590 | 2.570 | +1.980 [+1.756, +2.301] |
| asst@asst | 1.820 | 3.171 | +1.351 [+1.108, +1.643] | 0.603 | 1.033 | +0.431 [+0.357, +0.519] |
| asst@plain | 1.783 | 5.305 | +3.522 [+3.068, +3.998] | 0.591 | 1.838 | +1.248 [+0.989, +1.600] |

Decomposition of the role-specific change (own slot − plain):

| unit | Δ(own slot) | Δ(plain) | role-specific change = Δ(own) − Δ(plain) |
|---|---|---|---|
| nats/token | +1.351 [+1.103, +1.641] | +3.522 [+3.081, +3.990] | -2.172 [-2.628, -1.739] |
| bits/byte | +0.431 [+0.356, +0.520] | +1.248 [+0.986, +1.615] | -0.817 [-1.171, -0.573] |

### Check 1 verdict (rule in report.md, applied to Qwen3-8B in nats/token)

- wildchat: Δ(user@user) +0.608 [+0.486, +0.735], Δ(user@plain) +0.172 [+0.037, +0.293]
- oasst1: Δ(user@user) +0.357 [+0.260, +0.453], Δ(user@plain) -0.028 [-0.149, +0.085]

Plain-control effect in both sources: False [False, False]. Real user-slot rise in both sources: True [True, True]. **Verdict: real user-slot rise.**

## Check 2: OLMo 3 7B in bits per byte

Frozen held-out run (`results/scores/`). The text_ids match `data/texts.jsonl` (1190 of 1190 in every stage; per-text token counts equal `n_tokens` in the data). Change from OLMo base, paired over texts, bits/byte.

### (a) the pilot's 90 texts (primary for the cross-family comparison)

| model (IT − own base) | wildchat Δ(user@user) | wildchat Δ(user@plain) | wildchat user role-specific | oasst1 Δ(user@user) | oasst1 Δ(user@plain) | oasst1 user role-specific | assistant role-specific (asst@asst − asst@plain) |
|---|---|---|---|---|---|---|---|
| OLMo SFT | -0.23 [-0.36, -0.10] | -0.15 [-0.27, -0.04] | -0.07 [-0.12, -0.03] | +0.07 [+0.04, +0.09] | +0.08 [+0.06, +0.09] | -0.01 [-0.03, +0.01] | -0.04 [-0.05, -0.03] |
| OLMo DPO | -0.18 [-0.32, -0.04] | -0.11 [-0.24, +0.01] | -0.06 [-0.11, -0.02] | +0.09 [+0.06, +0.12] | +0.11 [+0.08, +0.13] | -0.02 [-0.04, +0.01] | +0.01 [-0.01, +0.03] |
| OLMo Instruct | -0.17 [-0.31, -0.03] | -0.10 [-0.23, +0.03] | -0.07 [-0.11, -0.03] | +0.10 [+0.07, +0.13] | +0.13 [+0.10, +0.15] | -0.03 [-0.05, -0.00] | +0.01 [-0.01, +0.03] |

### (b) all OLMo texts (secondary)

| model (IT − own base) | wildchat Δ(user@user) | wildchat Δ(user@plain) | wildchat user role-specific | oasst1 Δ(user@user) | oasst1 Δ(user@plain) | oasst1 user role-specific | assistant role-specific (asst@asst − asst@plain) |
|---|---|---|---|---|---|---|---|
| OLMo SFT | -0.15 [-0.18, -0.12] | -0.07 [-0.10, -0.03] | -0.08 [-0.10, -0.07] | +0.06 [+0.05, +0.06] | +0.08 [+0.07, +0.09] | -0.03 [-0.04, -0.02] | -0.04 [-0.05, -0.04] |
| OLMo DPO | -0.10 [-0.14, -0.06] | -0.03 [-0.06, +0.00] | -0.07 [-0.09, -0.06] | +0.09 [+0.08, +0.10] | +0.12 [+0.11, +0.14] | -0.03 [-0.04, -0.02] | +0.01 [+0.01, +0.02] |
| OLMo Instruct | -0.09 [-0.13, -0.05] | -0.01 [-0.04, +0.03] | -0.08 [-0.10, -0.07] | +0.10 [+0.09, +0.11] | +0.14 [+0.13, +0.15] | -0.04 [-0.05, -0.03] | +0.01 [+0.00, +0.02] |

OLMo-written assistant replies (`olmo-instruct`, secondary): assistant role-specific term, IT − base:

- sft: -0.056 [-0.064, -0.050]
- dpo: -0.046 [-0.054, -0.038]
- instruct: -0.050 [-0.059, -0.042]

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

