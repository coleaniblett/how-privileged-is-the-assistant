# In-character check (spec G addendum, POST-HOC: added after the provisional atlas was seen)

Question (frozen `meta_judge`, Gemma 4 31B instruct, P(yes) >= 0.5, blind): does the reply speak in the first person as the character? In-character replies are a selected group, not a random sample; base has few of them; the judge is not yet validated (hand check pending); 12 items per trait. Rates are all flagged-included.

## Share in character, atlas (all traits pooled)

|                            |   light_system |   light_user |   strong_act |   strong_noact |
|:---------------------------|---------------:|-------------:|-------------:|---------------:|
| ('gemma4-31b', 'base')     |          0.038 |        0.085 |      nan     |        nan     |
| ('gemma4-31b', 'instruct') |          0.983 |        0.991 |      nan     |        nan     |
| ('olmo3-7b', 'base')       |          0.038 |        0.072 |        0.185 |          0.035 |
| ('olmo3-7b', 'instruct')   |          0.064 |        0.588 |        0.901 |          0.116 |
| ('qwen3-8b', 'base')       |          0.108 |        0.128 |        0.303 |          0.037 |
| ('qwen3-8b', 'instruct')   |          0.7   |        0.756 |        0.985 |          0.626 |

## Light roles: like-for-like change from base (gated traits: the Assistant's own change passes the atlas gate)

| family | stage | context | trait | Assistant chg | share in character | all replies | in-character only (n items) | reading |
|---|---|---|---|---|---|---|---|---|
| olmo3-7b | instruct | light_system | moralizing | -13.9 | 0.08 | -4.17 [-18.06, +11.11] | too few (0) | too few in-character replies |
| olmo3-7b | instruct | light_user | moralizing | -13.9 | 0.49 | -5.56 [-18.06, +5.56] | too few (3) | too few in-character replies |
| olmo3-7b | instruct | light_system | markdown_lists | +52.8 | 0.07 | +83.33 [+66.63, +97.22] | too few (0) | too few in-character replies |
| olmo3-7b | instruct | light_user | markdown_lists | +52.8 | 0.64 | +37.50 [+16.67, +59.72] | too few (4) | too few in-character replies |
| olmo3-7b | instruct | light_system | formal_register | -51.4 | 0.12 | -30.56 [-48.61, -15.28] | too few (0) | too few in-character replies |
| olmo3-7b | instruct | light_user | formal_register | -51.4 | 0.58 | -72.22 [-88.89, -52.78] | too few (2) | too few in-character replies |
| olmo3-7b | instruct | light_system | corrects_false_premise | +26.4 | 0.00 | +36.11 [+16.67, +56.94] | too few (0) | too few in-character replies |
| olmo3-7b | instruct | light_user | corrects_false_premise | +26.4 | 0.62 | +33.33 [+13.89, +55.56] | too few (0) | too few in-character replies |
| olmo3-7b | instruct | light_system | fabricates_vs_admits | -75.0 | 0.00 | -51.39 [-75.00, -27.78] | too few (0) | too few in-character replies |
| olmo3-7b | instruct | light_user | fabricates_vs_admits | -75.0 | 0.54 | -33.33 [-59.72, -5.56] | too few (0) | too few in-character replies |
| olmo3-7b | instruct | light_system | agrees_wrong_opinion | -43.1 | 0.04 | -45.83 [-66.67, -25.00] | too few (0) | too few in-character replies |
| olmo3-7b | instruct | light_user | agrees_wrong_opinion | -43.1 | 0.61 | +1.39 [-23.61, +23.65] | too few (2) | too few in-character replies |
| olmo3-7b | instruct | light_system | clarifying_question | +37.5 | 0.15 | +31.94 [+0.00, +63.89] | too few (1) | too few in-character replies |
| olmo3-7b | instruct | light_user | clarifying_question | +37.5 | 0.58 | +9.72 [-13.89, +33.33] | +0.0 [-33.3, +38.5] (9) | no spread among in-character replies |
| olmo3-7b | instruct | light_system | denies_feelings | +44.4 | 0.22 | +58.33 [+34.72, +80.56] | too few (3) | too few in-character replies |
| olmo3-7b | instruct | light_user | denies_feelings | +44.4 | 0.85 | -33.33 [-55.56, -9.72] | -13.3 [-53.3, +0.0] (8) | no spread among in-character replies |
| olmo3-7b | instruct | light_system | declines_opinion | +36.1 | 0.21 | +30.56 [+8.33, +52.78] | too few (1) | too few in-character replies |
| olmo3-7b | instruct | light_user | declines_opinion | +36.1 | 0.90 | -13.89 [-40.28, +8.33] | +23.5 [+0.0, +71.4] (10) | no spread among in-character replies |
| olmo3-7b | instruct | light_system | arithmetic | +48.6 | 0.01 | +68.06 [+50.00, +84.72] | too few (0) | too few in-character replies |
| olmo3-7b | instruct | light_user | arithmetic | +48.6 | 0.47 | +33.33 [+13.89, +54.17] | too few (0) | too few in-character replies |
| olmo3-7b | instruct | light_system | factual_accuracy | +44.4 | 0.00 | +52.78 [+25.00, +77.78] | too few (0) | too few in-character replies |
| olmo3-7b | instruct | light_user | factual_accuracy | +44.4 | 0.51 | +30.56 [+2.78, +55.56] | too few (0) | too few in-character replies |
| qwen3-8b | instruct | light_system | markdown_lists | +37.5 | 0.89 | +48.61 [+30.56, +65.28] | too few (4) | too few in-character replies |
| qwen3-8b | instruct | light_user | markdown_lists | +37.5 | 0.83 | +54.17 [+34.72, +73.61] | too few (3) | too few in-character replies |
| qwen3-8b | instruct | light_system | length | +0.6 | 0.93 | +0.90 [+0.41, +1.47] | too few (3) | too few in-character replies |
| qwen3-8b | instruct | light_user | length | +0.6 | 0.99 | +0.84 [+0.22, +1.63] | +0.3 [-0.0, +1.3] (7) | no spread among in-character replies |
| qwen3-8b | instruct | light_system | formal_register | -18.1 | 0.75 | -30.56 [-47.22, -15.28] | too few (5) | too few in-character replies |
| qwen3-8b | instruct | light_user | formal_register | -18.1 | 0.82 | -23.61 [-37.50, -11.11] | too few (4) | too few in-character replies |
| qwen3-8b | instruct | light_system | hedging_unanswerable | +45.8 | 0.82 | +23.61 [+4.17, +44.44] | too few (5) | too few in-character replies |
| qwen3-8b | instruct | light_user | hedging_unanswerable | +45.8 | 0.85 | +34.72 [+15.28, +54.17] | +0.0 [-46.2, +66.7] (6) | no spread among in-character replies |
| qwen3-8b | instruct | light_system | corrects_false_premise | +34.7 | 0.57 | +44.44 [+25.00, +65.28] | too few (1) | too few in-character replies |
| qwen3-8b | instruct | light_user | corrects_false_premise | +34.7 | 0.78 | +41.67 [+22.22, +61.11] | too few (3) | too few in-character replies |
| qwen3-8b | instruct | light_system | flattery | +52.8 | 0.54 | +31.94 [+2.78, +59.72] | too few (2) | too few in-character replies |
| qwen3-8b | instruct | light_user | flattery | +52.8 | 0.78 | +37.50 [+12.50, +62.50] | too few (4) | too few in-character replies |
| qwen3-8b | instruct | light_system | agrees_wrong_opinion | -22.2 | 0.64 | -11.11 [-34.76, +13.89] | too few (3) | too few in-character replies |
| qwen3-8b | instruct | light_user | agrees_wrong_opinion | -22.2 | 0.67 | +8.33 [-12.50, +29.17] | -12.5 [-75.0, +44.4] (6) | no spread among in-character replies |
| qwen3-8b | instruct | light_system | closing_offer | +37.5 | 0.36 | +38.89 [+13.89, +66.67] | too few (1) | too few in-character replies |
| qwen3-8b | instruct | light_user | closing_offer | +37.5 | 0.54 | +38.89 [+11.11, +66.67] | too few (4) | too few in-character replies |
| qwen3-8b | instruct | light_system | clarifying_question | +44.4 | 0.81 | +37.50 [+13.89, +62.50] | +33.3 [+0.0, +75.0] (7) | no spread among in-character replies |
| qwen3-8b | instruct | light_user | clarifying_question | +44.4 | 0.75 | +30.56 [+9.72, +54.17] | +21.4 [+0.0, +75.0] (9) | no spread among in-character replies |
| qwen3-8b | instruct | light_system | denies_feelings | +63.9 | 1.00 | -1.39 [-11.11, +8.33] | +6.7 [+0.0, +23.7] (10) | no spread among in-character replies |
| qwen3-8b | instruct | light_user | denies_feelings | +63.9 | 1.00 | -9.72 [-25.00, +0.00] | +0.0 [+0.0, +0.0] (11) | no spread among in-character replies |
| qwen3-8b | instruct | light_system | declines_opinion | +29.2 | 0.97 | +1.39 [-22.22, +26.39] | +8.3 [-33.3, +62.5] (11) | no spread among in-character replies |
| qwen3-8b | instruct | light_user | declines_opinion | +29.2 | 0.96 | -13.89 [-41.67, +12.50] | -9.7 [-46.4, +22.2] (11) | no spread among in-character replies |
| qwen3-8b | instruct | light_system | arithmetic | +25.0 | 0.47 | +44.44 [+25.00, +63.89] | too few (0) | too few in-character replies |
| qwen3-8b | instruct | light_user | arithmetic | +25.0 | 0.57 | +29.17 [+0.00, +55.56] | too few (2) | too few in-character replies |
| gemma4-31b | instruct | light_system | markdown_lists | +79.2 | 1.00 | +84.72 [+72.22, +95.83] | too few (0) | too few in-character replies |
| gemma4-31b | instruct | light_user | markdown_lists | +79.2 | 1.00 | +62.50 [+44.44, +79.17] | too few (2) | too few in-character replies |
| gemma4-31b | instruct | light_system | length | +1.1 | 1.00 | +1.39 [+0.91, +1.94] | too few (0) | too few in-character replies |
| gemma4-31b | instruct | light_user | length | +1.1 | 1.00 | +1.09 [+0.61, +1.70] | too few (5) | too few in-character replies |
| gemma4-31b | instruct | light_system | stock_phrases | +66.7 | 0.90 | +20.83 [+1.39, +43.06] | too few (1) | too few in-character replies |
| gemma4-31b | instruct | light_user | stock_phrases | +66.7 | 0.94 | +9.72 [-8.33, +30.56] | too few (4) | too few in-character replies |
| gemma4-31b | instruct | light_system | says_ai | +58.3 | 0.99 | +0.00 [-11.11, +13.89] | +0.0 [+0.0, +0.0] (6) | no spread among in-character replies |
| gemma4-31b | instruct | light_user | says_ai | +58.3 | 0.99 | -4.17 [-18.06, +8.33] | +0.0 [+0.0, +0.0] (9) | no spread among in-character replies |
| gemma4-31b | instruct | light_system | arithmetic | +84.7 | 0.93 | +75.00 [+47.22, +94.44] | too few (1) | too few in-character replies |
| gemma4-31b | instruct | light_user | arithmetic | +84.7 | 0.96 | +51.39 [+23.61, +76.42] | too few (2) | too few in-character replies |
| gemma4-31b | instruct | light_system | multistep_reasoning | +65.3 | 1.00 | +51.39 [+26.39, +76.39] | too few (0) | too few in-character replies |
| gemma4-31b | instruct | light_user | multistep_reasoning | +65.3 | 1.00 | +40.28 [+12.50, +65.28] | too few (2) | too few in-character replies |
| gemma4-31b | instruct | light_system | factual_accuracy | +72.2 | 1.00 | +5.56 [-15.28, +26.42] | too few (1) | too few in-character replies |
| gemma4-31b | instruct | light_user | factual_accuracy | +72.2 | 1.00 | +12.50 [-13.89, +36.11] | too few (2) | too few in-character replies |

Counts of readings among gated light-role cells:

|                                                                    |   cells |
|:-------------------------------------------------------------------|--------:|
| ('gemma4-31b', 'instruct', 'no spread among in-character replies') |       2 |
| ('gemma4-31b', 'instruct', 'too few in-character replies')         |      12 |
| ('olmo3-7b', 'instruct', 'no spread among in-character replies')   |       3 |
| ('olmo3-7b', 'instruct', 'too few in-character replies')           |      19 |
| ('qwen3-8b', 'instruct', 'no spread among in-character replies')   |       9 |
| ('qwen3-8b', 'instruct', 'too few in-character replies')           |      15 |

## Style fall in strong characters (format and voice clusters): all replies vs in-character only

| family | stage | context | trait | share in character | all replies | in-character only (n items) | reading |
|---|---|---|---|---|---|---|---|
| olmo3-7b | instruct | strong_act | markdown_lists | 0.97 | -8.33 [-29.17, +15.28] | -5.6 [-36.4, +23.1] (11) | unresolved |
| olmo3-7b | instruct | strong_noact | markdown_lists | 0.14 | +58.33 [+38.89, +77.78] | too few (0) | n/a (no fall in all replies) |
| olmo3-7b | instruct | strong_act | length | 0.92 | +0.03 [-0.14, +0.30] | -0.1 [-0.2, -0.0] (9) | n/a (no fall in all replies) |
| olmo3-7b | instruct | strong_noact | length | 0.15 | +0.28 [-0.04, +0.78] | too few (0) | n/a (no fall in all replies) |
| olmo3-7b | instruct | strong_act | formal_register | 0.93 | -18.06 [-37.50, -4.17] | +0.0 [+0.0, +0.0] (10) | n/a (no fall in all replies) |
| olmo3-7b | instruct | strong_noact | formal_register | 0.11 | -55.56 [-72.22, -40.24] | too few (0) | n/a (no fall in all replies) |
| olmo3-7b | instruct | strong_act | stock_phrases | 0.79 | -22.22 [-40.28, -2.78] | too few (5) | too few in-character replies |
| olmo3-7b | instruct | strong_noact | stock_phrases | 0.06 | +26.39 [-5.56, +55.56] | too few (1) | n/a (no fall in all replies) |
| qwen3-8b | instruct | strong_act | markdown_lists | 1.00 | -2.78 [-23.61, +20.87] | +4.8 [-25.0, +36.0] (10) | n/a (no fall in all replies) |
| qwen3-8b | instruct | strong_noact | markdown_lists | 0.90 | +34.72 [+5.56, +62.50] | too few (0) | n/a (no fall in all replies) |
| qwen3-8b | instruct | strong_act | length | 1.00 | +0.95 [+0.42, +1.63] | +0.3 [+0.1, +0.5] (10) | n/a (no fall in all replies) |
| qwen3-8b | instruct | strong_noact | length | 0.79 | +0.92 [+0.47, +1.45] | too few (3) | n/a (no fall in all replies) |
| qwen3-8b | instruct | strong_act | formal_register | 1.00 | -13.89 [-31.94, -1.39] | -6.9 [-31.0, +0.0] (12) | n/a (no fall in all replies) |
| qwen3-8b | instruct | strong_noact | formal_register | 0.83 | -37.50 [-58.33, -16.67] | too few (3) | n/a (no fall in all replies) |
| qwen3-8b | instruct | strong_act | stock_phrases | 0.97 | +0.00 [-11.11, +11.11] | +9.5 [+0.0, +35.8] (12) | n/a (no fall in all replies) |
| qwen3-8b | instruct | strong_noact | stock_phrases | 0.57 | +13.89 [-8.33, +38.89] | too few (3) | n/a (no fall in all replies) |

## Leakage matrix replies (register rows; style score = per-reply axis coordinate, text classifier = P(Assistant) in points, layout regex in points)

| family | stage | context | metric | share in character (base -> stage) | all replies | in-character only | reading |
|---|---|---|---|---|---|---|---|
| olmo | sft | named | coord | 0.40 -> 0.57 | -0.11 [-0.19, -0.03] | -0.25 [-0.34, -0.15] | survives |
| olmo | sft | named | clf | 0.40 -> 0.57 | +8.58 [+4.34, +13.14] | +1.98 [-2.99, +7.33] | n/a (no fall in all replies) |
| olmo | sft | named | layout | 0.40 -> 0.57 | +5.86 [-0.52, +13.93] | +3.76 [-2.67, +11.90] | n/a (no fall in all replies) |
| olmo | sft | voiced | coord | 0.54 -> 0.87 | -0.52 [-0.62, -0.42] | -0.37 [-0.47, -0.29] | survives |
| olmo | sft | voiced | clf | 0.54 -> 0.87 | -4.64 [-10.33, +0.11] | -6.52 [-14.23, +0.51] | n/a (no fall in all replies) |
| olmo | sft | voiced | layout | 0.54 -> 0.87 | +1.56 [-1.30, +5.73] | +2.05 [+0.00, +7.20] | n/a (no fall in all replies) |
| olmo | dpo | named | coord | 0.40 -> 0.56 | -0.20 [-0.31, -0.10] | -0.40 [-0.52, -0.28] | survives |
| olmo | dpo | named | clf | 0.40 -> 0.56 | +2.91 [-2.99, +9.00] | -0.62 [-8.55, +6.97] | n/a (no fall in all replies) |
| olmo | dpo | named | layout | 0.40 -> 0.56 | +15.62 [+3.52, +30.21] | +11.45 [-0.54, +27.13] | n/a (no fall in all replies) |
| olmo | dpo | voiced | coord | 0.54 -> 0.98 | -0.81 [-0.91, -0.72] | -0.55 [-0.63, -0.47] | survives |
| olmo | dpo | voiced | clf | 0.54 -> 0.98 | -10.66 [-18.94, -3.63] | -12.55 [-21.10, -4.09] | survives |
| olmo | dpo | voiced | layout | 0.54 -> 0.98 | +3.65 [-1.82, +10.16] | +3.42 [-0.97, +9.38] | n/a (no fall in all replies) |
| olmo | instruct | named | coord | 0.40 -> 0.54 | -0.16 [-0.27, -0.05] | -0.38 [-0.50, -0.24] | survives |
| olmo | instruct | named | clf | 0.40 -> 0.54 | +3.65 [-1.85, +9.51] | -2.25 [-9.64, +5.23] | n/a (no fall in all replies) |
| olmo | instruct | named | layout | 0.40 -> 0.54 | +13.67 [+2.34, +27.21] | +10.44 [-1.30, +25.59] | n/a (no fall in all replies) |
| olmo | instruct | voiced | coord | 0.54 -> 0.99 | -0.80 [-0.89, -0.71] | -0.53 [-0.62, -0.44] | survives |
| olmo | instruct | voiced | clf | 0.54 -> 0.99 | -12.00 [-20.49, -4.59] | -13.35 [-21.85, -4.62] | survives |
| olmo | instruct | voiced | layout | 0.54 -> 0.99 | +2.08 [-1.30, +6.77] | +2.40 [-0.69, +7.20] | n/a (no fall in all replies) |
| qwen | instruct | named | coord | 0.53 -> 0.92 | -0.29 [-0.37, -0.22] | -0.27 [-0.33, -0.21] | survives |
| qwen | instruct | named | clf | 0.53 -> 0.92 | +14.67 [+10.82, +18.72] | +9.18 [+5.82, +13.07] | n/a (no fall in all replies) |
| qwen | instruct | named | layout | 0.53 -> 0.92 | -1.56 [-6.12, +4.56] | +0.17 [-5.85, +7.99] | n/a (no fall in all replies) |
| qwen | instruct | voiced | coord | 0.60 -> 1.00 | -0.45 [-0.52, -0.38] | -0.29 [-0.36, -0.23] | survives |
| qwen | instruct | voiced | clf | 0.60 -> 1.00 | +14.18 [+9.05, +18.91] | +5.71 [+0.31, +11.26] | n/a (no fall in all replies) |
| qwen | instruct | voiced | layout | 0.60 -> 1.00 | -3.65 [-9.12, +0.78] | -4.91 [-12.42, +0.84] | n/a (no fall in all replies) |
| gemma | instruct | named | coord | 0.39 -> 0.88 | -0.54 [-0.65, -0.42] | -0.53 [-0.63, -0.44] | survives |
| gemma | instruct | named | clf | 0.39 -> 0.88 | +19.21 [+14.12, +24.76] | +17.92 [+11.84, +24.90] | n/a (no fall in all replies) |
| gemma | instruct | named | layout | 0.39 -> 0.88 | -2.73 [-13.02, +5.60] | +1.18 [-8.87, +10.91] | n/a (no fall in all replies) |
| gemma | instruct | voiced | coord | 0.25 -> 0.96 | -0.75 [-0.81, -0.70] | -0.66 [-0.72, -0.60] | survives |
| gemma | instruct | voiced | clf | 0.25 -> 0.96 | +23.23 [+15.72, +31.14] | +19.30 [+8.09, +31.25] | n/a (no fall in all replies) |
| gemma | instruct | voiced | layout | 0.25 -> 0.96 | -9.11 [-23.70, +2.60] | +0.00 [-13.10, +10.75] | mostly composition |

Available per-reply scores: style score (coord), text classifier, layout regex. The axis-v2 coordinate is saved for OLMo only and is not split here.
