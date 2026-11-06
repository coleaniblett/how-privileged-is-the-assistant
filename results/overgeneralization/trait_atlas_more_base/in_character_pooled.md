# In-character check (spec G addendum, POST-HOC: added after the provisional atlas was seen)

**Spec J5 (POST-HOC, 2026-11-05):** base was sampled 5 more times per light-role request of the atlas's gated traits (OLMo, Qwen; Gemma not extended), pooled with the atlas's single sample; everything else is the in-character addendum's analysis unchanged (readings as defined there; `too few` = fewer than 6 items with an in-character reply on both sides). Plan: `docs/decisions.md` 2026-11-05.

Question (frozen `meta_judge`, Gemma 4 31B instruct, P(yes) >= 0.5, blind): does the reply speak in the first person as the character? In-character replies are a selected group, not a random sample; base has few of them; the judge is not yet validated (hand check pending); 12 items per trait. Rates are all flagged-included.

## Share in character, atlas (all traits pooled)

|                            |   light_system |   light_user |   strong_act |   strong_noact |
|:---------------------------|---------------:|-------------:|-------------:|---------------:|
| ('gemma4-31b', 'base')     |          0.038 |        0.085 |        0.181 |          0.017 |
| ('gemma4-31b', 'instruct') |          0.983 |        0.991 |        1     |          0.819 |
| ('olmo3-7b', 'base')       |          0.049 |        0.09  |        0.185 |          0.035 |
| ('olmo3-7b', 'dpo')        |        nan     |      nan     |        0.907 |          0.145 |
| ('olmo3-7b', 'instruct')   |          0.064 |        0.588 |        0.901 |          0.116 |
| ('olmo3-7b', 'sft')        |        nan     |      nan     |        0.807 |          0.092 |
| ('qwen3-8b', 'base')       |          0.108 |        0.152 |        0.303 |          0.037 |
| ('qwen3-8b', 'instruct')   |          0.7   |        0.756 |        0.985 |          0.626 |

## Light roles: like-for-like change from base (gated traits: the Assistant's own change passes the atlas gate)

| family | stage | context | trait | Assistant chg | share in character | all replies | in-character only (n items) | reading |
|---|---|---|---|---|---|---|---|---|
| olmo3-7b | instruct | light_system | moralizing | -13.9 | 0.08 | -4.86 [-15.05, +7.18] | too few (2) | too few in-character replies |
| olmo3-7b | instruct | light_user | moralizing | -13.9 | 0.49 | -8.33 [-16.44, +0.46] | -16.7 [-59.6, +0.0] (11) | not a character |
| olmo3-7b | instruct | light_system | markdown_lists | +52.8 | 0.07 | +85.65 [+78.70, +91.67] | too few (0) | too few in-character replies |
| olmo3-7b | instruct | light_user | markdown_lists | +52.8 | 0.64 | +36.34 [+25.69, +47.23] | +27.1 [+0.0, +80.2] (8) | no spread among in-character replies |
| olmo3-7b | instruct | light_system | formal_register | -51.4 | 0.12 | -38.89 [-49.77, -28.70] | too few (1) | too few in-character replies |
| olmo3-7b | instruct | light_user | formal_register | -51.4 | 0.58 | -67.36 [-74.54, -59.49] | -61.1 [-100.0, -23.1] (9) | real spread |
| olmo3-7b | instruct | light_system | corrects_false_premise | +26.4 | 0.00 | +42.36 [+22.22, +62.51] | too few (0) | too few in-character replies |
| olmo3-7b | instruct | light_user | corrects_false_premise | +26.4 | 0.62 | +36.11 [+18.29, +55.09] | too few (4) | too few in-character replies |
| olmo3-7b | instruct | light_system | fabricates_vs_admits | -75.0 | 0.00 | -54.17 [-76.16, -32.40] | too few (0) | too few in-character replies |
| olmo3-7b | instruct | light_user | fabricates_vs_admits | -75.0 | 0.54 | -37.96 [-60.65, -13.42] | -33.3 [-100.0, +0.0] (8) | no spread among in-character replies |
| olmo3-7b | instruct | light_system | agrees_wrong_opinion | -43.1 | 0.04 | -44.91 [-58.56, -31.48] | too few (1) | too few in-character replies |
| olmo3-7b | instruct | light_user | agrees_wrong_opinion | -43.1 | 0.61 | -1.16 [-21.99, +20.15] | -43.3 [-87.5, +0.0] (9) | no spread among in-character replies |
| olmo3-7b | instruct | light_system | clarifying_question | +37.5 | 0.15 | +40.51 [+13.66, +67.13] | too few (1) | too few in-character replies |
| olmo3-7b | instruct | light_user | clarifying_question | +37.5 | 0.58 | +11.57 [-9.03, +34.03] | +5.2 [-21.3, +30.0] (10) | no spread among in-character replies |
| olmo3-7b | instruct | light_system | denies_feelings | +44.4 | 0.22 | +60.65 [+41.67, +76.39] | too few (5) | too few in-character replies |
| olmo3-7b | instruct | light_user | denies_feelings | +44.4 | 0.85 | -27.31 [-43.52, -8.80] | -11.0 [-30.0, +4.2] (12) | no spread among in-character replies |
| olmo3-7b | instruct | light_system | declines_opinion | +36.1 | 0.21 | +37.50 [+19.21, +53.25] | too few (5) | too few in-character replies |
| olmo3-7b | instruct | light_user | declines_opinion | +36.1 | 0.90 | -11.34 [-30.33, +6.25] | +7.0 [-22.4, +34.9] (12) | no spread among in-character replies |
| olmo3-7b | instruct | light_system | arithmetic | +48.6 | 0.01 | +60.42 [+47.22, +72.22] | too few (1) | too few in-character replies |
| olmo3-7b | instruct | light_user | arithmetic | +48.6 | 0.47 | +38.19 [+24.31, +53.94] | +41.7 [+0.0, +100.0] (6) | not a character |
| olmo3-7b | instruct | light_system | factual_accuracy | +44.4 | 0.00 | +46.30 [+28.47, +62.96] | too few (0) | too few in-character replies |
| olmo3-7b | instruct | light_user | factual_accuracy | +44.4 | 0.51 | +25.23 [+4.63, +44.91] | +0.0 [-77.8, +83.3] (6) | no spread among in-character replies |
| qwen3-8b | instruct | light_system | markdown_lists | +37.5 | 0.89 | +41.90 [+29.40, +53.94] | +32.9 [-6.0, +72.7] (10) | no spread among in-character replies |
| qwen3-8b | instruct | light_user | markdown_lists | +37.5 | 0.83 | +59.72 [+50.93, +68.06] | +36.2 [+8.8, +66.7] (11) | real spread |
| qwen3-8b | instruct | light_system | length | +0.6 | 0.93 | +1.08 [+0.77, +1.42] | +0.4 [-0.0, +0.9] (9) | no spread among in-character replies |
| qwen3-8b | instruct | light_user | length | +0.6 | 0.99 | +0.82 [+0.51, +1.17] | +0.2 [+0.0, +0.3] (12) | real spread |
| qwen3-8b | instruct | light_system | formal_register | -18.1 | 0.75 | -28.01 [-35.42, -20.37] | -36.4 [-66.7, -7.4] (11) | real spread |
| qwen3-8b | instruct | light_user | formal_register | -18.1 | 0.82 | -24.77 [-33.80, -16.90] | -24.7 [-55.1, -3.5] (10) | real spread |
| qwen3-8b | instruct | light_system | hedging_unanswerable | +45.8 | 0.82 | +29.86 [+15.04, +43.75] | +28.0 [-6.9, +55.1] (9) | no spread among in-character replies |
| qwen3-8b | instruct | light_user | hedging_unanswerable | +45.8 | 0.85 | +31.48 [+17.36, +45.37] | +15.3 [-4.7, +36.9] (12) | no spread among in-character replies |
| qwen3-8b | instruct | light_system | corrects_false_premise | +34.7 | 0.57 | +40.97 [+23.38, +57.41] | too few (3) | too few in-character replies |
| qwen3-8b | instruct | light_user | corrects_false_premise | +34.7 | 0.78 | +37.04 [+23.61, +51.85] | +27.2 [+3.2, +60.0] (12) | real spread |
| qwen3-8b | instruct | light_system | flattery | +52.8 | 0.54 | +35.42 [+11.57, +58.33] | too few (5) | too few in-character replies |
| qwen3-8b | instruct | light_user | flattery | +52.8 | 0.78 | +36.81 [+15.28, +57.87] | +46.2 [+13.6, +77.2] (12) | real spread |
| qwen3-8b | instruct | light_system | agrees_wrong_opinion | -22.2 | 0.64 | -4.86 [-25.24, +18.52] | -28.4 [-75.7, +53.9] (9) | no spread among in-character replies |
| qwen3-8b | instruct | light_user | agrees_wrong_opinion | -22.2 | 0.67 | +2.55 [-12.74, +20.14] | -15.2 [-47.6, +20.5] (12) | no spread among in-character replies |
| qwen3-8b | instruct | light_system | closing_offer | +37.5 | 0.36 | +39.58 [+16.90, +63.89] | too few (4) | too few in-character replies |
| qwen3-8b | instruct | light_user | closing_offer | +37.5 | 0.54 | +39.35 [+15.51, +65.28] | +50.0 [+0.0, +100.0] (7) | no spread among in-character replies |
| qwen3-8b | instruct | light_system | clarifying_question | +44.4 | 0.81 | +35.42 [+14.35, +57.87] | +28.7 [+0.7, +60.3] (9) | real spread |
| qwen3-8b | instruct | light_user | clarifying_question | +44.4 | 0.75 | +28.47 [+7.86, +49.77] | +14.0 [-21.6, +47.5] (9) | no spread among in-character replies |
| qwen3-8b | instruct | light_system | denies_feelings | +63.9 | 1.00 | -5.09 [-12.96, +4.17] | -1.3 [-9.4, +7.4] (12) | no spread among in-character replies |
| qwen3-8b | instruct | light_user | denies_feelings | +63.9 | 1.00 | -9.72 [-14.58, -5.09] | -4.4 [-10.9, +0.0] (12) | no spread among in-character replies |
| qwen3-8b | instruct | light_system | declines_opinion | +29.2 | 0.97 | +2.78 [-18.52, +26.16] | +4.2 [-18.7, +29.3] (12) | no spread among in-character replies |
| qwen3-8b | instruct | light_user | declines_opinion | +29.2 | 0.96 | -9.72 [-25.00, +5.32] | -12.3 [-32.4, +2.6] (12) | no spread among in-character replies |
| qwen3-8b | instruct | light_system | arithmetic | +25.0 | 0.47 | +44.68 [+28.70, +57.64] | too few (2) | too few in-character replies |
| qwen3-8b | instruct | light_user | arithmetic | +25.0 | 0.57 | +37.27 [+19.21, +51.85] | +9.1 [-70.6, +85.7] (7) | no spread among in-character replies |
| gemma4-31b | instruct | light_system | safety_caveats | +86.1 | 1.00 | +75.00 [+55.56, +91.67] | too few (0) | too few in-character replies |
| gemma4-31b | instruct | light_user | safety_caveats | +86.1 | 1.00 | +48.61 [+16.67, +77.78] | too few (3) | too few in-character replies |
| gemma4-31b | instruct | light_system | markdown_lists | +79.2 | 1.00 | +84.72 [+72.22, +95.83] | too few (0) | too few in-character replies |
| gemma4-31b | instruct | light_user | markdown_lists | +79.2 | 1.00 | +62.50 [+44.44, +80.56] | too few (2) | too few in-character replies |
| gemma4-31b | instruct | light_system | length | +1.1 | 1.00 | +1.39 [+0.89, +1.90] | too few (0) | too few in-character replies |
| gemma4-31b | instruct | light_user | length | +1.1 | 1.00 | +1.09 [+0.60, +1.66] | too few (5) | too few in-character replies |
| gemma4-31b | instruct | light_system | stock_phrases | +66.7 | 0.90 | +20.83 [+1.39, +43.06] | too few (1) | too few in-character replies |
| gemma4-31b | instruct | light_user | stock_phrases | +66.7 | 0.94 | +9.72 [-8.33, +29.17] | too few (4) | too few in-character replies |
| gemma4-31b | instruct | light_system | hedging_unanswerable | +83.3 | 1.00 | +68.06 [+48.61, +86.11] | too few (5) | too few in-character replies |
| gemma4-31b | instruct | light_user | hedging_unanswerable | +83.3 | 1.00 | +59.72 [+38.89, +77.78] | too few (5) | too few in-character replies |
| gemma4-31b | instruct | light_system | corrects_false_premise | +83.3 | 1.00 | +86.11 [+72.22, +97.22] | too few (0) | too few in-character replies |
| gemma4-31b | instruct | light_user | corrects_false_premise | +83.3 | 1.00 | +80.56 [+61.11, +95.83] | too few (0) | too few in-character replies |
| gemma4-31b | instruct | light_system | fabricates_vs_admits | -38.9 | 1.00 | -9.72 [-38.89, +22.22] | too few (1) | too few in-character replies |
| gemma4-31b | instruct | light_user | fabricates_vs_admits | -38.9 | 1.00 | -13.89 [-47.22, +15.28] | too few (0) | too few in-character replies |
| gemma4-31b | instruct | light_system | agrees_wrong_opinion | -22.2 | 1.00 | -11.11 [-23.61, -1.39] | too few (4) | too few in-character replies |
| gemma4-31b | instruct | light_user | agrees_wrong_opinion | -22.2 | 1.00 | -29.17 [-50.00, -12.50] | -66.7 [-100.0, +0.0] (6) | no spread among in-character replies |
| gemma4-31b | instruct | light_system | clarifying_question | +69.4 | 1.00 | +45.83 [+16.67, +72.22] | too few (3) | too few in-character replies |
| gemma4-31b | instruct | light_user | clarifying_question | +69.4 | 1.00 | +36.11 [+12.50, +59.72] | +0.0 [-46.2, +40.0] (7) | no spread among in-character replies |
| gemma4-31b | instruct | light_system | says_ai | +58.3 | 0.99 | +0.00 [-11.11, +12.53] | +0.0 [+0.0, +0.0] (6) | no spread among in-character replies |
| gemma4-31b | instruct | light_user | says_ai | +58.3 | 0.99 | -4.17 [-18.06, +8.33] | +0.0 [+0.0, +0.0] (9) | no spread among in-character replies |
| gemma4-31b | instruct | light_system | denies_feelings | +98.6 | 1.00 | -2.78 [-11.11, +0.00] | +0.0 [+0.0, +0.0] (8) | no spread among in-character replies |
| gemma4-31b | instruct | light_user | denies_feelings | +98.6 | 1.00 | -1.39 [-8.33, +0.00] | +0.0 [+0.0, +0.0] (9) | no spread among in-character replies |
| gemma4-31b | instruct | light_system | declines_opinion | +40.3 | 1.00 | -26.39 [-47.22, -2.78] | +0.0 [-66.7, +66.7] (6) | no spread among in-character replies |
| gemma4-31b | instruct | light_user | declines_opinion | +40.3 | 1.00 | -13.89 [-33.33, +6.98] | -11.8 [-40.0, +0.0] (9) | no spread among in-character replies |
| gemma4-31b | instruct | light_system | arithmetic | +84.7 | 0.93 | +75.00 [+50.00, +94.44] | too few (1) | too few in-character replies |
| gemma4-31b | instruct | light_user | arithmetic | +84.7 | 0.96 | +51.39 [+25.00, +76.39] | too few (2) | too few in-character replies |
| gemma4-31b | instruct | light_system | multistep_reasoning | +65.3 | 1.00 | +51.39 [+25.00, +75.00] | too few (0) | too few in-character replies |
| gemma4-31b | instruct | light_user | multistep_reasoning | +65.3 | 1.00 | +40.28 [+13.89, +66.67] | too few (2) | too few in-character replies |
| gemma4-31b | instruct | light_system | factual_accuracy | +72.2 | 1.00 | +5.56 [-15.28, +26.39] | too few (1) | too few in-character replies |
| gemma4-31b | instruct | light_user | factual_accuracy | +72.2 | 1.00 | +12.50 [-12.50, +36.11] | too few (2) | too few in-character replies |

Counts of readings among gated light-role cells:

|                                                                    |   cells |
|:-------------------------------------------------------------------|--------:|
| ('gemma4-31b', 'instruct', 'no spread among in-character replies') |       8 |
| ('gemma4-31b', 'instruct', 'too few in-character replies')         |      22 |
| ('olmo3-7b', 'instruct', 'no spread among in-character replies')   |       7 |
| ('olmo3-7b', 'instruct', 'not a character')                        |       2 |
| ('olmo3-7b', 'instruct', 'real spread')                            |       1 |
| ('olmo3-7b', 'instruct', 'too few in-character replies')           |      12 |
| ('qwen3-8b', 'instruct', 'no spread among in-character replies')   |      13 |
| ('qwen3-8b', 'instruct', 'real spread')                            |       7 |
| ('qwen3-8b', 'instruct', 'too few in-character replies')           |       4 |

## Style fall in strong characters (format and voice clusters): all replies vs in-character only

| family | stage | context | trait | share in character | all replies | in-character only (n items) | reading |
|---|---|---|---|---|---|---|---|
| olmo3-7b | sft | strong_act | markdown_lists | 0.90 | -2.78 [-23.61, +19.44] | -5.9 [-40.0, +25.0] (11) | n/a (no fall in all replies) |
| olmo3-7b | dpo | strong_act | markdown_lists | 0.99 | +19.44 [-13.89, +47.26] | +38.9 [+0.0, +80.0] (11) | n/a (no fall in all replies) |
| olmo3-7b | instruct | strong_act | markdown_lists | 0.97 | -8.33 [-29.17, +13.89] | -5.6 [-35.3, +25.0] (11) | unresolved |
| olmo3-7b | sft | strong_noact | markdown_lists | 0.06 | +38.89 [+15.28, +59.72] | too few (0) | n/a (no fall in all replies) |
| olmo3-7b | dpo | strong_noact | markdown_lists | 0.14 | +59.72 [+41.67, +77.78] | too few (0) | n/a (no fall in all replies) |
| olmo3-7b | instruct | strong_noact | markdown_lists | 0.14 | +58.33 [+37.50, +77.78] | too few (0) | n/a (no fall in all replies) |
| olmo3-7b | sft | strong_act | length | 0.85 | -0.09 [-0.29, +0.20] | -0.2 [-0.4, +0.0] (9) | n/a (no fall in all replies) |
| olmo3-7b | dpo | strong_act | length | 0.92 | +0.07 [-0.10, +0.35] | -0.1 [-0.2, +0.0] (9) | n/a (no fall in all replies) |
| olmo3-7b | instruct | strong_act | length | 0.92 | +0.03 [-0.14, +0.30] | -0.1 [-0.2, +0.0] (9) | n/a (no fall in all replies) |
| olmo3-7b | sft | strong_noact | length | 0.04 | -0.08 [-0.56, +0.47] | too few (0) | n/a (no fall in all replies) |
| olmo3-7b | dpo | strong_noact | length | 0.10 | +0.28 [-0.04, +0.79] | too few (0) | n/a (no fall in all replies) |
| olmo3-7b | instruct | strong_noact | length | 0.15 | +0.28 [-0.04, +0.77] | too few (0) | n/a (no fall in all replies) |
| olmo3-7b | sft | strong_act | formal_register | 0.90 | -12.50 [-29.17, +2.78] | +0.0 [+0.0, +0.0] (10) | n/a (no fall in all replies) |
| olmo3-7b | dpo | strong_act | formal_register | 0.90 | -18.06 [-36.11, -4.17] | +0.0 [+0.0, +0.0] (9) | n/a (no fall in all replies) |
| olmo3-7b | instruct | strong_act | formal_register | 0.93 | -18.06 [-36.11, -4.17] | +0.0 [+0.0, +0.0] (10) | n/a (no fall in all replies) |
| olmo3-7b | sft | strong_noact | formal_register | 0.10 | -51.39 [-69.44, -31.94] | too few (1) | n/a (no fall in all replies) |
| olmo3-7b | dpo | strong_noact | formal_register | 0.15 | -55.56 [-72.22, -38.89] | too few (1) | n/a (no fall in all replies) |
| olmo3-7b | instruct | strong_noact | formal_register | 0.11 | -55.56 [-72.22, -40.28] | too few (0) | n/a (no fall in all replies) |
| olmo3-7b | sft | strong_act | stock_phrases | 0.72 | -18.06 [-40.28, +5.56] | too few (5) | n/a (no fall in all replies) |
| olmo3-7b | dpo | strong_act | stock_phrases | 0.89 | -23.61 [-44.44, -2.78] | -18.2 [-64.3, +0.0] (6) | unresolved |
| olmo3-7b | instruct | strong_act | stock_phrases | 0.79 | -22.22 [-41.67, -2.78] | too few (5) | too few in-character replies |
| olmo3-7b | sft | strong_noact | stock_phrases | 0.03 | +5.56 [-18.06, +31.94] | too few (0) | too few in-character replies |
| olmo3-7b | dpo | strong_noact | stock_phrases | 0.11 | +23.61 [-5.56, +54.17] | too few (1) | n/a (no fall in all replies) |
| olmo3-7b | instruct | strong_noact | stock_phrases | 0.06 | +26.39 [-5.56, +56.94] | too few (1) | n/a (no fall in all replies) |
| qwen3-8b | instruct | strong_act | markdown_lists | 1.00 | -2.78 [-25.00, +22.22] | +4.8 [-22.2, +36.4] (10) | n/a (no fall in all replies) |
| qwen3-8b | instruct | strong_noact | markdown_lists | 0.90 | +34.72 [+5.56, +61.11] | too few (0) | n/a (no fall in all replies) |
| qwen3-8b | instruct | strong_act | length | 1.00 | +0.95 [+0.41, +1.68] | +0.3 [+0.1, +0.5] (10) | n/a (no fall in all replies) |
| qwen3-8b | instruct | strong_noact | length | 0.79 | +0.92 [+0.47, +1.44] | too few (3) | n/a (no fall in all replies) |
| qwen3-8b | instruct | strong_act | formal_register | 1.00 | -13.89 [-31.94, -1.39] | -6.9 [-31.6, +0.0] (12) | n/a (no fall in all replies) |
| qwen3-8b | instruct | strong_noact | formal_register | 0.83 | -37.50 [-58.33, -16.67] | too few (3) | n/a (no fall in all replies) |
| qwen3-8b | instruct | strong_act | stock_phrases | 0.97 | +0.00 [-11.11, +11.11] | +9.5 [+0.0, +36.4] (12) | n/a (no fall in all replies) |
| qwen3-8b | instruct | strong_noact | stock_phrases | 0.57 | +13.89 [-8.33, +38.89] | too few (3) | n/a (no fall in all replies) |
| gemma4-31b | instruct | strong_act | markdown_lists | 1.00 | +1.39 [-27.78, +37.50] | -13.3 [-85.7, +55.6] (7) | n/a (no fall in all replies) |
| gemma4-31b | instruct | strong_noact | markdown_lists | 0.94 | +19.44 [-6.94, +47.22] | too few (0) | n/a (no fall in all replies) |
| gemma4-31b | instruct | strong_act | length | 1.00 | +1.24 [+0.67, +1.80] | -0.0 [-0.1, +0.1] (9) | n/a (no fall in all replies) |
| gemma4-31b | instruct | strong_noact | length | 0.82 | +0.84 [+0.50, +1.33] | too few (0) | n/a (no fall in all replies) |
| gemma4-31b | instruct | strong_act | formal_register | 1.00 | +0.00 [-13.89, +16.67] | +5.6 [+0.0, +35.3] (10) | n/a (no fall in all replies) |
| gemma4-31b | instruct | strong_noact | formal_register | 0.96 | +2.78 [-12.50, +22.22] | too few (2) | n/a (no fall in all replies) |
| gemma4-31b | instruct | strong_act | stock_phrases | 1.00 | -2.78 [-12.50, +5.56] | -10.0 [-60.0, +0.0] (8) | n/a (no fall in all replies) |
| gemma4-31b | instruct | strong_noact | stock_phrases | 0.82 | -5.56 [-16.67, +0.00] | too few (1) | too few in-character replies |

