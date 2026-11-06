# In-context era shift (archaic bird names -> 19th century)

Spec E (2026-11-03; plan and probe list in `docs/decisions.md`; exploratory, outside every verdict). Shift = era(archaic shots) - era(modern shots), era = log P(19th-century answer) - log P(modern answer) after a fixed answer prefix (teacher-forced, bf16 batch 1, KV cache of the shot block reused after an equivalence check). Nats; mean over probes; 95% bootstrap CI over probe x seed (all frozen probes x 3 shot orders). Qwen3-8B is supporting (no contamination check).

## Name pick-up check (held-out bird questions: log P(archaic) - log P(modern), archaic minus modern shots)

A check that the model picked up the naming convention from the shots, not evidence of era generalization. Pre-set: the set-up took if this shift at k = 128 is > 0 with CI excluding 0 (Assistant condition) in every model.

| family | stage | condition | k=8 | k=32 | k=128 | took |
|---|---|---|---|---|---|---|
| olmo3-7b | base | assistant | +3.77 [+2.79, +4.77] | +6.10 [+4.98, +7.28] | +7.71 [+6.42, +9.00] | yes |
| olmo3-7b | base | mora | +4.37 [+3.33, +5.44] | +6.26 [+5.05, +7.49] | +7.76 [+6.45, +9.06] | yes |
| olmo3-7b | base | plain | +4.06 [+2.95, +5.17] | +6.70 [+5.50, +7.87] | +8.70 [+7.37, +9.97] | yes |
| olmo3-7b | sft | assistant | +2.61 [+1.58, +3.63] | +4.22 [+3.23, +5.24] | +4.59 [+3.46, +5.79] | yes |
| olmo3-7b | sft | mora | +3.64 [+2.27, +5.03] | +4.83 [+3.65, +6.05] | +4.78 [+3.56, +6.04] | yes |
| olmo3-7b | sft | plain | +2.21 [+1.19, +3.15] | +3.44 [+2.57, +4.34] | +4.83 [+3.58, +6.16] | yes |
| olmo3-7b | dpo | assistant | +1.90 [+0.72, +3.07] | +2.66 [+1.62, +3.78] | +3.22 [+1.95, +4.50] | yes |
| olmo3-7b | dpo | mora | +2.39 [+1.02, +3.78] | +2.88 [+1.72, +4.11] | +3.25 [+2.02, +4.49] | yes |
| olmo3-7b | dpo | plain | +2.17 [+0.87, +3.38] | +2.73 [+1.56, +3.93] | +3.99 [+2.65, +5.35] | yes |
| olmo3-7b | instruct | assistant | +1.80 [+0.56, +3.09] | +2.51 [+1.38, +3.67] | +3.08 [+1.77, +4.44] | yes |
| olmo3-7b | instruct | mora | +2.31 [+0.89, +3.73] | +2.67 [+1.48, +3.95] | +3.09 [+1.79, +4.45] | yes |
| olmo3-7b | instruct | plain | +2.26 [+0.92, +3.56] | +2.80 [+1.55, +4.01] | +3.87 [+2.46, +5.28] | yes |
| qwen3-8b | base | assistant | +3.43 [+2.81, +4.06] | +7.01 [+5.94, +8.12] | +7.33 [+6.00, +8.81] | yes |
| qwen3-8b | base | mora | +4.23 [+3.41, +5.12] | +6.69 [+5.18, +8.25] | +7.06 [+5.54, +8.79] | yes |
| qwen3-8b | base | plain | +4.31 [+3.18, +5.40] | +4.88 [+3.44, +6.19] | +6.92 [+5.51, +8.38] | yes |
| qwen3-8b | instruct | assistant | +3.96 [+2.77, +5.23] | +5.57 [+4.15, +7.12] | +7.49 [+5.91, +9.20] | yes |
| qwen3-8b | instruct | mora | +5.94 [+4.35, +7.62] | +7.65 [+5.61, +9.72] | +9.93 [+7.92, +12.08] | yes |
| qwen3-8b | instruct | plain | +7.33 [+5.01, +9.63] | +7.96 [+5.53, +10.45] | +7.35 [+5.20, +9.79] | yes |

## Era shift by model and number of shots (probes)

| family | stage | condition | k=8 | k=32 | k=128 | era at k=0 |
|---|---|---|---|---|---|---|
| olmo3-7b | base | assistant | +0.30 [+0.24, +0.37] | +0.39 [+0.33, +0.46] | +0.25 [+0.20, +0.30] | -2.90 [-4.20, -1.72] |
| olmo3-7b | base | mora | +0.37 [+0.31, +0.45] | +0.50 [+0.42, +0.58] | +0.30 [+0.25, +0.35] | -2.31 [-3.78, -0.93] |
| olmo3-7b | base | plain | +0.22 [+0.18, +0.26] | +0.23 [+0.19, +0.27] | +0.13 [+0.09, +0.17] | -2.44 [-3.79, -1.13] |
| olmo3-7b | sft | assistant | +0.12 [+0.07, +0.17] | +0.06 [+0.02, +0.11] | +0.02 [-0.02, +0.07] | -3.46 [-4.93, -2.02] |
| olmo3-7b | sft | mora | +0.54 [+0.41, +0.67] | +0.29 [+0.21, +0.37] | +0.08 [+0.03, +0.12] | -1.86 [-3.55, -0.21] |
| olmo3-7b | sft | plain | +0.13 [+0.08, +0.18] | +0.17 [+0.11, +0.23] | +0.12 [+0.05, +0.18] | -2.49 [-3.93, -1.07] |
| olmo3-7b | dpo | assistant | +0.06 [-0.01, +0.13] | +0.02 [-0.02, +0.07] | -0.02 [-0.07, +0.03] | -3.84 [-5.49, -2.25] |
| olmo3-7b | dpo | mora | +0.50 [+0.34, +0.68] | +0.14 [+0.06, +0.22] | +0.02 [-0.04, +0.08] | -2.21 [-4.22, -0.28] |
| olmo3-7b | dpo | plain | +0.14 [+0.06, +0.21] | +0.16 [+0.08, +0.24] | +0.11 [+0.01, +0.19] | -3.54 [-5.13, -2.03] |
| olmo3-7b | instruct | assistant | +0.06 [+0.00, +0.13] | +0.01 [-0.04, +0.06] | -0.02 [-0.07, +0.03] | -4.01 [-5.73, -2.38] |
| olmo3-7b | instruct | mora | +0.49 [+0.32, +0.67] | +0.10 [+0.03, +0.18] | -0.00 [-0.07, +0.06] | -2.45 [-4.52, -0.44] |
| olmo3-7b | instruct | plain | +0.11 [+0.03, +0.19] | +0.15 [+0.07, +0.22] | +0.11 [+0.04, +0.19] | -3.70 [-5.36, -2.10] |
| qwen3-8b | base | assistant | +0.09 [+0.00, +0.17] | +0.35 [+0.26, +0.44] | +0.31 [+0.22, +0.41] | -3.19 [-4.45, -1.95] |
| qwen3-8b | base | mora | +0.40 [+0.32, +0.48] | +0.54 [+0.44, +0.64] | +0.39 [+0.29, +0.49] | -1.61 [-2.92, -0.37] |
| qwen3-8b | base | plain | +0.25 [+0.15, +0.35] | +0.31 [+0.21, +0.41] | +0.36 [+0.24, +0.47] | -1.30 [-2.53, -0.15] |
| qwen3-8b | instruct | assistant | +0.19 [+0.08, +0.30] | +0.06 [-0.05, +0.17] | -0.00 [-0.10, +0.10] | -6.97 [-9.58, -4.47] |
| qwen3-8b | instruct | mora | +1.28 [+1.04, +1.53] | +1.32 [+1.07, +1.57] | +0.90 [+0.69, +1.10] | -0.30 [-3.57, +2.89] |
| qwen3-8b | instruct | plain | +0.41 [+0.27, +0.56] | +0.21 [+0.08, +0.33] | +0.11 [-0.03, +0.24] | -2.23 [-3.72, -0.84] |

## Mean, median and robustness subset at k = 128

Mean over all probes (the primary set), the median probe shift with its inter-quartile range in brackets, and the 11-probe robustness subset (probes with |zero-shot log-odds| <= 5 in every model).

| family | stage | condition | mean (all probes) | median [IQR] | robustness subset |
|---|---|---|---|---|---|
| olmo3-7b | base | assistant | +0.25 [+0.20, +0.30] | +0.23 [+0.07, +0.42] | +0.16 [+0.12, +0.19] |
| olmo3-7b | base | mora | +0.30 [+0.25, +0.35] | +0.30 [+0.16, +0.48] | +0.17 [+0.10, +0.23] |
| olmo3-7b | base | plain | +0.13 [+0.09, +0.17] | +0.11 [+0.04, +0.23] | +0.13 [+0.08, +0.18] |
| olmo3-7b | sft | assistant | +0.02 [-0.02, +0.07] | +0.02 [-0.09, +0.16] | -0.06 [-0.13, +0.01] |
| olmo3-7b | sft | mora | +0.08 [+0.03, +0.12] | +0.10 [-0.06, +0.18] | -0.01 [-0.09, +0.07] |
| olmo3-7b | sft | plain | +0.12 [+0.05, +0.18] | +0.08 [-0.02, +0.25] | +0.05 [-0.11, +0.18] |
| olmo3-7b | dpo | assistant | -0.02 [-0.07, +0.03] | -0.03 [-0.11, +0.05] | -0.09 [-0.18, -0.01] |
| olmo3-7b | dpo | mora | +0.02 [-0.04, +0.08] | +0.01 [-0.17, +0.19] | -0.07 [-0.19, +0.05] |
| olmo3-7b | dpo | plain | +0.11 [+0.01, +0.19] | +0.10 [-0.06, +0.24] | +0.05 [-0.19, +0.26] |
| olmo3-7b | instruct | assistant | -0.02 [-0.07, +0.03] | -0.02 [-0.15, +0.12] | -0.10 [-0.20, -0.01] |
| olmo3-7b | instruct | mora | -0.00 [-0.07, +0.06] | -0.00 [-0.15, +0.20] | -0.13 [-0.25, -0.01] |
| olmo3-7b | instruct | plain | +0.11 [+0.04, +0.19] | +0.10 [-0.07, +0.28] | +0.09 [-0.08, +0.26] |
| qwen3-8b | base | assistant | +0.31 [+0.22, +0.41] | +0.29 [+0.04, +0.54] | +0.40 [+0.24, +0.56] |
| qwen3-8b | base | mora | +0.39 [+0.29, +0.49] | +0.29 [+0.08, +0.67] | +0.26 [+0.09, +0.43] |
| qwen3-8b | base | plain | +0.36 [+0.24, +0.47] | +0.25 [+0.17, +0.50] | +0.32 [+0.15, +0.49] |
| qwen3-8b | instruct | assistant | -0.00 [-0.10, +0.10] | -0.08 [-0.25, +0.29] | -0.10 [-0.23, +0.03] |
| qwen3-8b | instruct | mora | +0.90 [+0.69, +1.10] | +0.88 [+0.33, +1.44] | +0.75 [+0.31, +1.14] |
| qwen3-8b | instruct | plain | +0.11 [-0.03, +0.24] | +0.06 [-0.17, +0.38] | +0.09 [-0.11, +0.30] |

## Per-probe shift at k = 128, assistant condition (mean over 3 shot orders; nats)

| probe | olmo3-7b base | olmo3-7b sft | olmo3-7b dpo | olmo3-7b instruct | qwen3-8b base | qwen3-8b instruct |
|---|---|---|---|---|---|---|
| atlantic | +0.23 | -0.13 | -0.03 | -0.02 | +0.08 | -0.17 |
| calc | +0.06 | -0.02 | -0.06 | -0.06 | +0.04 | -0.17 |
| city_transport | +0.24 | +0.19 | +0.21 | +0.15 | +0.21 | +0.38 |
| cooking | +0.30 | +0.06 | +0.15 | +0.13 | -0.04 | +0.12 |
| disease | +0.69 | +0.05 | -0.34 | -0.21 | -0.02 | +0.53 |
| evening | +0.07 | -0.06 | -0.05 | +0.09 | +0.04 | -0.10 |
| factories | +0.30 | +0.13 | +0.08 | +0.06 | +0.29 | -0.25 |
| heating | +0.18 | +0.08 | -0.04 | -0.01 | +0.83 | +0.21 |
| india | +0.42 | -0.09 | +0.01 | -0.04 | +0.38 | -0.23 |
| lantern | +0.51 | +0.17 | +0.21 | +0.12 | +0.67 | +0.54 |
| laundry | +0.06 | +0.19 | -0.00 | +0.06 | +0.08 | -0.17 |
| letters | -0.04 | -0.04 | -0.04 | -0.15 | +0.00 | -0.46 |
| life_exp | -0.02 | -0.06 | -0.19 | -0.23 | +0.39 | -0.45 |
| light_source_street | +0.14 | -0.10 | +0.05 | +0.06 | +0.67 | +0.04 |
| lighting | +0.48 | +0.23 | +0.27 | +0.26 | -0.29 | -0.71 |
| long_distance | +0.12 | -0.12 | -0.14 | -0.19 | +1.04 | +0.54 |
| message | -0.06 | +0.02 | -0.11 | -0.05 | +1.29 | +0.50 |
| monarch | +0.12 | +0.19 | +0.21 | +0.15 | +0.54 | +0.29 |
| music | +0.48 | +0.16 | -0.03 | +0.02 | +0.79 | +1.00 |
| new_state | +0.51 | -0.22 | -0.02 | -0.14 | +0.54 | -0.21 |
| news | +0.20 | +0.12 | +0.04 | +0.12 | +0.00 | -0.25 |
| overseas_msg | +0.56 | +0.12 | -0.03 | -0.02 | +0.04 | -0.90 |
| pay | -0.03 | -0.10 | +0.00 | +0.12 | +0.42 | -0.08 |
| photo | +0.36 | +0.19 | +0.20 | +0.17 | +0.46 | +0.15 |
| ship_material | -0.05 | -0.12 | -0.21 | -0.17 | +0.75 | +0.21 |
| ships_power | -0.01 | +0.33 | +0.29 | +0.35 | -0.04 | +0.42 |
| sport | +0.22 | -0.06 | -0.15 | -0.15 | +0.54 | -0.46 |
| states | +0.56 | +0.21 | +0.04 | -0.01 | +0.08 | +0.59 |
| trains | +0.24 | +0.08 | +0.02 | -0.04 | +0.04 | +0.17 |
| travel | +0.29 | -0.00 | -0.12 | -0.19 | +0.33 | +0.17 |
| west_state | +0.25 | -0.38 | -0.53 | -0.54 | +0.83 | -0.08 |
| work | +0.54 | -0.12 | -0.25 | -0.10 | -0.46 | -0.50 |
| world_pop | +0.22 | -0.08 | -0.06 | -0.19 | -0.21 | -0.73 |

## Per-probe shift at k = 128, mora condition (mean over 3 shot orders; nats)

| probe | olmo3-7b base | olmo3-7b sft | olmo3-7b dpo | olmo3-7b instruct | qwen3-8b base | qwen3-8b instruct |
|---|---|---|---|---|---|---|
| atlantic | +0.26 | -0.04 | +0.08 | +0.15 | +0.00 | +0.71 |
| calc | -0.05 | +0.04 | -0.19 | -0.21 | -0.08 | +0.88 |
| city_transport | +0.30 | +0.26 | +0.29 | +0.24 | +0.42 | +1.38 |
| cooking | +0.45 | +0.10 | +0.20 | +0.20 | +0.25 | +0.90 |
| disease | +0.55 | +0.10 | -0.26 | -0.21 | +0.28 | +0.83 |
| evening | -0.14 | +0.06 | +0.17 | +0.16 | -0.00 | +0.19 |
| factories | +0.48 | +0.23 | +0.29 | +0.31 | +0.08 | +0.71 |
| heating | +0.28 | +0.16 | -0.01 | -0.06 | +0.96 | +2.00 |
| india | +0.59 | -0.11 | -0.07 | -0.00 | +0.46 | -0.15 |
| lantern | +0.48 | +0.31 | +0.27 | +0.27 | +0.75 | +3.38 |
| laundry | +0.00 | +0.25 | +0.17 | -0.06 | +0.08 | +0.38 |
| letters | +0.03 | -0.08 | -0.02 | -0.06 | +0.08 | +0.12 |
| life_exp | -0.07 | -0.08 | -0.21 | -0.33 | -0.16 | +0.33 |
| light_source_street | +0.16 | -0.10 | +0.11 | -0.03 | +0.83 | +2.23 |
| lighting | +0.55 | +0.51 | +0.38 | +0.29 | +0.29 | -0.40 |
| long_distance | +0.39 | -0.09 | -0.04 | -0.15 | +0.54 | +1.73 |
| message | +0.20 | +0.10 | +0.06 | +0.01 | +0.83 | +0.06 |
| monarch | +0.12 | +0.21 | +0.19 | +0.25 | +0.42 | +1.92 |
| music | +0.51 | +0.18 | -0.06 | -0.21 | +0.42 | +2.00 |
| new_state | +0.49 | -0.10 | +0.01 | -0.11 | +0.92 | +0.67 |
| news | +0.25 | +0.12 | +0.19 | +0.23 | +0.08 | +0.71 |
| overseas_msg | +0.78 | +0.10 | -0.03 | +0.10 | +0.46 | +1.44 |
| pay | +0.18 | +0.06 | +0.23 | +0.10 | +0.67 | +2.21 |
| photo | +0.35 | +0.21 | +0.19 | +0.23 | +0.67 | +1.04 |
| ship_material | +0.08 | -0.12 | -0.17 | -0.19 | +1.50 | +1.46 |
| ships_power | +0.10 | +0.27 | +0.36 | +0.41 | +0.21 | +0.21 |
| sport | +0.20 | -0.06 | -0.17 | -0.10 | +0.21 | +1.25 |
| states | +0.62 | +0.16 | -0.09 | +0.02 | +0.28 | +1.25 |
| trains | +0.32 | +0.00 | +0.02 | +0.17 | +0.29 | +0.12 |
| travel | +0.39 | +0.04 | -0.27 | -0.23 | +0.17 | +1.00 |
| west_state | +0.24 | -0.29 | -0.48 | -0.56 | +1.00 | +0.88 |
| work | +0.54 | +0.10 | -0.19 | -0.15 | +0.08 | +0.71 |
| world_pop | +0.31 | +0.00 | -0.33 | -0.52 | -0.21 | -2.37 |

## Per-probe shift at k = 128, plain condition (mean over 3 shot orders; nats)

| probe | olmo3-7b base | olmo3-7b sft | olmo3-7b dpo | olmo3-7b instruct | qwen3-8b base | qwen3-8b instruct |
|---|---|---|---|---|---|---|
| atlantic | +0.14 | -0.22 | -0.13 | -0.18 | +0.21 | +0.25 |
| calc | +0.03 | -0.00 | +0.13 | +0.17 | +0.58 | -0.35 |
| city_transport | +0.30 | +0.44 | +0.24 | +0.21 | -0.04 | +0.31 |
| cooking | +0.16 | +0.04 | +0.05 | +0.02 | +0.42 | -0.17 |
| disease | +0.23 | -0.08 | -0.09 | -0.09 | -0.37 | +0.78 |
| evening | +0.10 | +0.12 | -0.08 | -0.10 | +0.58 | -0.06 |
| factories | +0.23 | +0.27 | +0.12 | -0.06 | -0.08 | +0.00 |
| heating | +0.07 | +0.25 | +0.32 | +0.28 | +1.50 | +0.79 |
| india | +0.36 | -0.01 | -0.13 | -0.07 | +0.25 | +0.38 |
| lantern | +0.28 | +0.42 | +0.46 | +0.38 | +0.17 | +0.33 |
| laundry | +0.12 | +0.27 | +0.17 | +0.19 | +0.00 | -0.58 |
| letters | -0.17 | -0.33 | -0.25 | -0.25 | +0.50 | -0.23 |
| life_exp | +0.11 | -0.33 | -0.06 | +0.31 | +0.44 | -0.60 |
| light_source_street | -0.15 | +0.46 | +0.40 | +0.33 | +0.25 | +0.85 |
| lighting | +0.29 | +0.99 | +0.69 | +0.81 | +0.46 | +0.06 |
| long_distance | +0.31 | +0.26 | +0.15 | +0.17 | +1.04 | -0.17 |
| message | +0.34 | +0.21 | -0.02 | -0.07 | +0.75 | -0.33 |
| monarch | +0.08 | +0.08 | -0.02 | -0.06 | -0.25 | +0.08 |
| music | +0.22 | +0.24 | +0.53 | +0.47 | +0.08 | -0.10 |
| new_state | +0.09 | -0.13 | -0.04 | -0.04 | +1.42 | +0.83 |
| news | +0.07 | +0.21 | +0.46 | +0.44 | +0.21 | +0.46 |
| overseas_msg | +0.04 | +0.06 | +0.22 | +0.33 | +0.50 | -0.25 |
| pay | -0.17 | -0.04 | -0.33 | -0.38 | +0.21 | -0.17 |
| photo | +0.17 | +0.04 | +0.25 | +0.25 | +0.88 | +0.04 |
| ship_material | +0.00 | -0.06 | +0.04 | -0.10 | +0.50 | +0.13 |
| ships_power | -0.04 | +0.27 | +0.26 | +0.28 | +0.25 | -0.46 |
| sport | +0.09 | -0.02 | -0.15 | -0.10 | -0.12 | +0.29 |
| states | +0.43 | +0.16 | -0.08 | -0.04 | +0.38 | +0.08 |
| trains | +0.03 | +0.10 | +0.17 | +0.10 | +0.25 | -0.13 |
| travel | +0.23 | +0.04 | +0.21 | +0.29 | +0.17 | +0.50 |
| west_state | +0.30 | -0.10 | -0.02 | -0.04 | +0.42 | +0.71 |
| work | -0.12 | +0.23 | +0.05 | +0.10 | -0.54 | +0.52 |
| world_pop | +0.11 | +0.00 | +0.10 | +0.15 | +0.75 | -0.25 |

## Contrasts at k = 128 (probes)

| family | stage | condition | contrast | value |
|---|---|---|---|---|
| olmo3-7b | sft | assistant | shift minus base | -0.22 [-0.29, -0.16] |
| olmo3-7b | dpo | assistant | shift minus base | -0.27 [-0.34, -0.20] |
| olmo3-7b | instruct | assistant | shift minus base | -0.27 [-0.34, -0.19] |
| olmo3-7b | sft | mora | shift minus base | -0.23 [-0.29, -0.16] |
| olmo3-7b | dpo | mora | shift minus base | -0.28 [-0.36, -0.20] |
| olmo3-7b | instruct | mora | shift minus base | -0.30 [-0.38, -0.23] |
| olmo3-7b | sft | plain | shift minus base | -0.01 [-0.08, +0.05] |
| olmo3-7b | dpo | plain | shift minus base | -0.02 [-0.12, +0.06] |
| olmo3-7b | instruct | plain | shift minus base | -0.02 [-0.10, +0.06] |
| olmo3-7b | base | mora-assistant | Mora minus Assistant shift | +0.05 [+0.03, +0.08] |
| olmo3-7b | sft | mora-assistant | Mora minus Assistant shift | +0.05 [+0.02, +0.08] |
| olmo3-7b | dpo | mora-assistant | Mora minus Assistant shift | +0.04 [-0.00, +0.08] |
| olmo3-7b | instruct | mora-assistant | Mora minus Assistant shift | +0.02 [-0.02, +0.06] |
| qwen3-8b | instruct | assistant | shift minus base | -0.31 [-0.42, -0.20] |
| qwen3-8b | instruct | mora | shift minus base | +0.51 [+0.32, +0.70] |
| qwen3-8b | instruct | plain | shift minus base | -0.25 [-0.41, -0.08] |
| qwen3-8b | base | mora-assistant | Mora minus Assistant shift | +0.07 [-0.02, +0.17] |
| qwen3-8b | instruct | mora-assistant | Mora minus Assistant shift | +0.90 [+0.73, +1.08] |

## Pre-set readings (instruct, Assistant condition, k = 128)

- **olmo3-7b**: ICL reproduces the generalization: no (instruct shift -0.02 [-0.07, +0.03]; bar >= 0.5, CI > 0). Post-training anchors the Assistant: no (instruct minus base -0.27 [-0.34, -0.19]; bar <= -0.5, CI < 0). Assistant more anchored than Mora: no (Mora minus Assistant +0.02 [-0.02, +0.06]).
- **qwen3-8b**: ICL reproduces the generalization: no (instruct shift -0.00 [-0.10, +0.10]; bar >= 0.5, CI > 0). Post-training anchors the Assistant: no (instruct minus base -0.31 [-0.42, -0.20]; bar <= -0.5, CI < 0). Assistant more anchored than Mora: YES (Mora minus Assistant +0.90 [+0.73, +1.08]).

## Generated examples (greedy, k = 128, seed 0, Assistant condition)

### olmo3-7b base

| probe | archaic shots | modern shots |
|---|---|---|
| How many states are in the United States? | 50. | 50. |
| What is the fastest way to send a message to another city? | What is the name of this bird: A small brown songbird with a black cap and a white belly, common at feeders in the north.? time according to What is the name of this bird: A small songbird with | What is the fastest way to send a message to another city? |
| What lights most homes at night? | What is the name of this bird: A small brown songbird with a black cap and a white belly, common at feeders in the north.? time according to What is the name of this bird: A small songbird with | Light Bulb. |
| How do most people get around within a city? | What is the name of this bird: A small brown songbird with a black cap and a white belly, common at feeders in the north.? time according to What is the name of this bird: A small songbird with | What is the name of this bird: A small brown songbird of open country with a white throat and a long, thin, down-curved bill.? th according What is the name of this bird: A small brown songbird |

### olmo3-7b sft

| probe | archaic shots | modern shots |
|---|---|---|
| How many states are in the United States? | There are 50 states in the United States. | There are 50 states in the United States. |
| What is the fastest way to send a message to another city? | The fastest way to send a message to another city is through a telecommunications network, such as a phone call, text message, or email. These methods can transmit information instantly over long distances, allowing for near-instant communication. | The fastest way to send a message to another city is through a combination of modern communication technologies. Here are the most efficient methods: 1. **Internet/Email/Text Messaging:** - Use email, instant messaging (WhatsApp, Telegram |
| What lights most homes at night? | Electric light bulbs. | Electric light bulbs. |
| How do most people get around within a city? | Most people get around within a city by using a combination of different transportation methods. The most common ways include: 1. **Walking**: Many people walk to get around, especially in pedestrian-friendly areas or when the weather is mild. Walking is | Most people get around within a city by using a combination of different transportation methods. The most common ways include: 1. **Walking**: Many people walk to get around, especially in pedestrian-friendly areas or when the weather is mild. Walking is |

### olmo3-7b dpo

| probe | archaic shots | modern shots |
|---|---|---|
| How many states are in the United States? | There are 50 states in the United States. | There are 50 states in the United States. |
| What is the fastest way to send a message to another city? | The fastest way to send a message to another city depends on your needs and the type of message: **For almost instantaneous communication:** - **Text Messaging (SMS/Texting):** Using a smartphone, you can send a text message | The fastest way to send a message to another city depends on your needs and the type of message: **For almost instantaneous communication:** - **Text Messaging (SMS/Texting):** Using a smartphone, you can send a text message |
| What lights most homes at night? | Most homes are illuminated at night primarily by **electric light bulbs**, such as incandescent, LED, or compact fluorescent lamps. These bulbs are powered by electricity generated from sources like coal, natural gas, wind, solar, or hydroelectric | Most homes are illuminated at night primarily by **electric light bulbs**, such as incandescent, LED, or compact fluorescent lamps. These bulbs are powered by electricity generated from sources like coal, natural gas, wind, solar, or hydroelectric |
| How do most people get around within a city? | Most people in cities get around using a variety of transportation methods, including: 1. **Walking** – Especially in pedestrian-friendly or compact urban areas. 2. **Cycling (Bicycling)** – Using bicycles for short trips and commuting | Most people in cities use a variety of transportation methods to get around, including: 1. **Walking** – Especially in pedestrian-friendly or compact urban areas. 2. **Cycling (Bicycling)** – Bikes are popular for short |

### olmo3-7b instruct

| probe | archaic shots | modern shots |
|---|---|---|
| How many states are in the United States? | There are 50 states in the United States. | There are 50 states in the United States. |
| What is the fastest way to send a message to another city? | The fastest way to send a message to another city depends on the context (text, email, phone, or physical mail), but generally, the quickest options are: 1. **Text or Messaging Apps (e.g., WhatsApp, iMessage | The fastest way to send a message to another city depends on your needs and the type of message: **For instant, reliable delivery:** - **Text or email:** For most messages (notes, documents, quick info), sending an email |
| What lights most homes at night? | Most homes are illuminated at night by **electric light bulbs**, such as incandescent, LED, or fluorescent lights. These lights are powered by electricity and are designed to emit light when electricity flows through them. Some homes may also use candles | Most homes are illuminated at night primarily by **electric lights**, such as incandescent bulbs, LEDs, or compact fluorescent lights. These lights are powered by electricity from the home's electrical grid. In some cases, alternative light sources like candles |
| How do most people get around within a city? | Most people in cities use a variety of transportation methods to get around, including: 1. **Walking** – Especially in pedestrian-friendly or compact urban areas. 2. **Public Transit** – This includes buses, subways, trains, tr | Most people in cities use a variety of transportation methods to get around, including: 1. **Walking** – Especially in pedestrian-friendly or compact urban areas. 2. **Cycling (Bicycling)** – Bikes are popular for short |

### qwen3-8b base

| probe | archaic shots | modern shots |
|---|---|---|
| How many states are in the United States? | Assistant: The United States consists of 50 states. | Assistant: The United States consists of 50 states. |
| What is the fastest way to send a message to another city? | user What is the name of this bird: A tiny seabird with a yellow bill and white forehead that nests in colonies on sandy beaches.? | user What is the name of this bird: A tiny seabird with a yellow bill and white forehead that nests in colonies on sandy beaches.? |
| What lights most homes at night? | user What is the name of this bird: A tiny seabird with a yellow bill and white forehead that nests in colonies on sandy beaches.? | Assistant: The bird you are referring to is the **Bald Eagle**. It is a large raptor with a dark brown body, a snowy head and tail, and a massive hooked beak. It is often seen soaring with its |
| How do most people get around within a city? | user What is the name of this bird: A tiny seabird with a yellow bill and white forehead that nests in colonies on sandy beaches.? | user What is the name of this bird: A tiny seabird with a yellow bill and white forehead that nests in colonies on sandy beaches.? |

### qwen3-8b instruct

| probe | archaic shots | modern shots |
|---|---|---|
| How many states are in the United States? | There are **50 states** in the United States. | There are **50 states** in the United States. |
| What is the fastest way to send a message to another city? | The fastest way to send a message to another city historically has been through **telecommunication methods**, and today, the fastest options include: ### 1. **Email** - **Speed**: Near-instant (within seconds or minutes, | The fastest way to send a message to another city historically was through **messenger services** or **telegraph**. Here's a breakdown of the fastest methods depending on the time period: ### **Modern (Today):** - **Email |
| What lights most homes at night? | The lights that most homes use at night are **light bulbs**, typically **incandescent**, **fluorescent**, or **LED** bulbs. These are used in **light fixtures** such as **ceiling lights**, **table lamps | The lights that most homes use at night are **incandescent bulbs**, **fluorescent lamps**, and more recently, **LED lights**. - **Incandescent bulbs** are traditional and widely used, though they are less |
| How do most people get around within a city? | Most people get around within a city using a combination of the following transportation methods: 1. **Walking** – For short distances, especially in pedestrian-friendly areas. 2. **Bicycling** – Using bicycles or bike-sharing programs for moderate | Most people get around within a city using a combination of the following transportation methods: 1. **Walking** – For short distances, especially in pedestrian-friendly areas. 2. **Bicycling** – Using bicycles or bike-sharing programs for moderate |

