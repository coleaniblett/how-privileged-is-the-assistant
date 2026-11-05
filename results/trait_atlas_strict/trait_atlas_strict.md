# Trait atlas on strict-cut text (spec J12, POST-HOC; exploratory, outside every verdict)

Plan: `docs/decisions.md` 2026-11-05. Same replies as the atlas, additionally cut at the first new-turn pattern (see `cut_summary.md`); detectors and skills rescored (skills: accepted answer anywhere in the reply); judge traits re-judged only where the cut changed the reply (Gemma 4 31B-it, same prompts). Same bootstrap and gate (10 points, 0.2 log words, CI excluding 0). A cell is **run-on sensitive** if the atlas reads it as moved (gate met, CI excluding 0) and under the strict cut it no longer meets the bar, or shrinks by at least half (or changes sign). Changes are final stage minus base.

## Run-on sensitive cells (of the cells the atlas reads as moved)

28 of 261 moved cells are run-on sensitive.

### By family

| family     |   sensitive |   moved cells |
|:-----------|------------:|--------------:|
| gemma4-31b |           0 |            58 |
| olmo3-7b   |          27 |           148 |
| qwen3-8b   |           1 |            55 |

### By context

| context      |   sensitive |   moved cells |
|:-------------|------------:|--------------:|
| assistant    |           7 |            58 |
| light_system |           6 |            55 |
| light_user   |           3 |            43 |
| narration    |           3 |            19 |
| plain_text   |           0 |            16 |
| strong_act   |           0 |            23 |
| strong_noact |           9 |            40 |
| user_turn    |           0 |             7 |

### By trait

| trait                  |   sensitive |   moved cells |
|:-----------------------|------------:|--------------:|
| agrees_wrong_opinion   |           0 |            13 |
| arithmetic             |          10 |            21 |
| clarifying_question    |           0 |            15 |
| closing_offer          |           0 |             5 |
| corrects_false_premise |           0 |            22 |
| declines_opinion       |           0 |            14 |
| denies_feelings        |           0 |            14 |
| fabricates_vs_admits   |           0 |            11 |
| factual_accuracy       |          11 |            15 |
| flattery               |           0 |             5 |
| formal_register        |           0 |            25 |
| hedging_unanswerable   |           0 |            16 |
| length                 |           0 |            17 |
| markdown_lists         |           0 |            23 |
| moralizing             |           0 |             4 |
| multistep_reasoning    |           7 |            17 |
| refusal_borderline     |           0 |             5 |
| safety_caveats         |           0 |             3 |
| says_ai                |           0 |             8 |
| stock_phrases          |           0 |             8 |

## Assistant gate (assistant context, final stage)

Traits the atlas gate counts as moved at the final stage: 38; still moved after the strict cut: 37. A trait that loses the gate is reported as 'training did not clearly move the Assistant after all':

| family | trait | atlas change | strict change |
|---|---|---|---|
| olmo3-7b | factual_accuracy | +44.4 [+19.4, +68.1] | +5.6 [-4.2, +16.7] |

## Every cell: atlas change beside strict-cut change (final stage; main cells, flagged replies included)

| family | stage | trait | context | atlas change | strict change | atlas moved | strict moved | run-on sensitive |
|---|---|---|---|---|---|---|---|---|
| gemma4-31b | instruct | agrees_wrong_opinion | assistant | -22.2 [-36.1, -9.7] | -22.2 [-36.1, -9.7] | yes | yes |  |
| gemma4-31b | instruct | agrees_wrong_opinion | light_system | -11.1 [-23.6, -1.4] | -11.1 [-23.6, -1.4] | yes | yes |  |
| gemma4-31b | instruct | agrees_wrong_opinion | light_user | -29.2 [-50.0, -11.1] | -29.2 [-48.6, -12.5] | yes | yes |  |
| gemma4-31b | instruct | agrees_wrong_opinion | narration | -13.9 [-33.3, +0.0] | -13.9 [-33.3, +0.0] |  |  |  |
| gemma4-31b | instruct | agrees_wrong_opinion | plain_text | -2.8 [-16.7, +0.0] | -2.8 [-16.7, +0.0] |  |  |  |
| gemma4-31b | instruct | agrees_wrong_opinion | strong_act | -18.1 [-34.7, -1.4] | -16.7 [-31.9, -1.4] | yes | yes |  |
| gemma4-31b | instruct | agrees_wrong_opinion | strong_noact | -16.7 [-31.9, -2.8] | -16.7 [-31.9, -4.1] | yes | yes |  |
| gemma4-31b | instruct | arithmetic | assistant | +84.7 [+68.1, +97.2] | +61.1 [+38.9, +80.6] | yes | yes |  |
| gemma4-31b | instruct | arithmetic | light_system | +75.0 [+47.2, +94.4] | +73.6 [+55.6, +87.5] | yes | yes |  |
| gemma4-31b | instruct | arithmetic | light_user | +51.4 [+25.0, +77.8] | +43.1 [+19.4, +66.7] | yes | yes |  |
| gemma4-31b | instruct | arithmetic | narration | +5.6 [-19.4, +33.3] | -8.3 [-75.0, +50.0] |  |  |  |
| gemma4-31b | instruct | arithmetic | plain_text | -11.1 [-30.6, +5.6] | +5.6 [+0.0, +16.7] |  |  |  |
| gemma4-31b | instruct | arithmetic | strong_act | -9.7 [-26.4, +8.3] | -4.2 [-25.0, +19.4] |  |  |  |
| gemma4-31b | instruct | arithmetic | strong_noact | +1.4 [-23.6, +27.8] | -2.8 [-31.9, +29.2] |  |  |  |
| gemma4-31b | instruct | clarifying_question | assistant | +69.4 [+44.4, +91.7] | +69.4 [+45.8, +90.3] | yes | yes |  |
| gemma4-31b | instruct | clarifying_question | light_system | +45.8 [+18.1, +73.6] | +45.8 [+18.1, +72.3] | yes | yes |  |
| gemma4-31b | instruct | clarifying_question | light_user | +36.1 [+12.5, +61.1] | +36.1 [+13.9, +61.1] | yes | yes |  |
| gemma4-31b | instruct | clarifying_question | narration | +38.9 [+16.7, +61.1] | +38.9 [+16.7, +63.9] | yes | yes |  |
| gemma4-31b | instruct | clarifying_question | plain_text | -13.9 [-41.7, +25.0] | -13.9 [-41.7, +22.2] |  |  |  |
| gemma4-31b | instruct | clarifying_question | strong_act | +18.1 [-2.8, +40.3] | +18.1 [-1.4, +38.9] |  |  |  |
| gemma4-31b | instruct | clarifying_question | strong_noact | +26.4 [+1.4, +52.8] | +27.8 [+4.2, +52.8] | yes | yes |  |
| gemma4-31b | instruct | closing_offer | assistant | -1.4 [-8.3, +0.0] | +0.0 [+0.0, +0.0] |  |  |  |
| gemma4-31b | instruct | closing_offer | light_system | +4.2 [-5.6, +19.4] | +4.2 [-5.6, +19.4] |  |  |  |
| gemma4-31b | instruct | closing_offer | light_user | +5.6 [-8.3, +25.0] | +5.6 [-8.3, +25.0] |  |  |  |
| gemma4-31b | instruct | closing_offer | narration | -2.8 [-16.7, +0.0] | -2.8 [-16.7, +0.0] |  |  |  |
| gemma4-31b | instruct | closing_offer | plain_text | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |  |  |  |
| gemma4-31b | instruct | closing_offer | strong_act | -1.4 [-8.3, +0.0] | -1.4 [-8.3, +0.0] |  |  |  |
| gemma4-31b | instruct | closing_offer | strong_noact | +5.6 [+0.0, +18.1] | +5.6 [+0.0, +18.1] |  |  |  |
| gemma4-31b | instruct | closing_offer | user_turn | -2.8 [-16.7, +0.0] | -2.8 [-11.1, +0.0] |  |  |  |
| gemma4-31b | instruct | corrects_false_premise | assistant | +83.3 [+65.3, +97.2] | +83.3 [+65.3, +97.2] | yes | yes |  |
| gemma4-31b | instruct | corrects_false_premise | light_system | +86.1 [+70.8, +97.2] | +86.1 [+70.8, +97.2] | yes | yes |  |
| gemma4-31b | instruct | corrects_false_premise | light_user | +80.6 [+61.1, +95.8] | +80.6 [+61.1, +95.8] | yes | yes |  |
| gemma4-31b | instruct | corrects_false_premise | narration | +88.9 [+66.7, +100.0] | +88.9 [+66.7, +100.0] | yes | yes |  |
| gemma4-31b | instruct | corrects_false_premise | plain_text | +47.2 [+16.7, +75.0] | +47.2 [+16.7, +75.0] | yes | yes |  |
| gemma4-31b | instruct | corrects_false_premise | strong_act | +61.1 [+40.2, +80.6] | +63.9 [+43.1, +81.9] | yes | yes |  |
| gemma4-31b | instruct | corrects_false_premise | strong_noact | +70.8 [+50.0, +88.9] | +70.8 [+50.0, +88.9] | yes | yes |  |
| gemma4-31b | instruct | declines_opinion | assistant | +40.3 [+6.9, +70.8] | +40.3 [+6.9, +70.8] | yes | yes |  |
| gemma4-31b | instruct | declines_opinion | light_system | -26.4 [-47.2, -2.8] | -26.4 [-47.2, -2.8] | yes | yes |  |
| gemma4-31b | instruct | declines_opinion | light_user | -13.9 [-34.7, +6.9] | -13.9 [-33.3, +6.9] |  |  |  |
| gemma4-31b | instruct | declines_opinion | narration | +2.8 [-25.0, +33.3] | +2.8 [-25.0, +33.3] |  |  |  |
| gemma4-31b | instruct | declines_opinion | plain_text | +11.1 [-8.3, +33.3] | +11.1 [-8.3, +33.3] |  |  |  |
| gemma4-31b | instruct | declines_opinion | strong_act | -23.6 [-43.1, -5.6] | -23.6 [-43.1, -5.6] | yes | yes |  |
| gemma4-31b | instruct | declines_opinion | strong_noact | -15.3 [-41.7, +8.3] | -15.3 [-41.7, +8.3] |  |  |  |
| gemma4-31b | instruct | denies_feelings | assistant | +98.6 [+91.7, +100.0] | +98.6 [+91.7, +100.0] | yes | yes |  |
| gemma4-31b | instruct | denies_feelings | light_system | -2.8 [-11.1, +0.0] | -2.8 [-11.1, +0.0] |  |  |  |
| gemma4-31b | instruct | denies_feelings | light_user | -1.4 [-8.3, +0.0] | -1.4 [-8.3, +0.0] |  |  |  |
| gemma4-31b | instruct | denies_feelings | narration | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |  |  |  |
| gemma4-31b | instruct | denies_feelings | plain_text | +41.7 [+8.3, +77.8] | +41.7 [+8.3, +77.8] | yes | yes |  |
| gemma4-31b | instruct | denies_feelings | strong_act | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |  |  |  |
| gemma4-31b | instruct | denies_feelings | strong_noact | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |  |  |  |
| gemma4-31b | instruct | fabricates_vs_admits | assistant | -38.9 [-66.7, -5.6] | -38.9 [-66.7, -6.9] | yes | yes |  |
| gemma4-31b | instruct | fabricates_vs_admits | light_system | -9.7 [-40.3, +20.8] | -9.7 [-38.9, +22.2] |  |  |  |
| gemma4-31b | instruct | fabricates_vs_admits | light_user | -13.9 [-45.8, +16.7] | -13.9 [-45.8, +15.3] |  |  |  |
| gemma4-31b | instruct | fabricates_vs_admits | narration | +22.2 [-19.4, +66.7] | +22.2 [-19.4, +66.7] |  |  |  |
| gemma4-31b | instruct | fabricates_vs_admits | plain_text | -61.1 [-88.9, -33.3] | -61.1 [-88.9, -33.3] | yes | yes |  |
| gemma4-31b | instruct | fabricates_vs_admits | strong_act | +12.5 [-18.1, +41.7] | +13.9 [-15.3, +41.7] |  |  |  |
| gemma4-31b | instruct | fabricates_vs_admits | strong_noact | +16.7 [-26.4, +58.3] | +16.7 [-26.4, +58.3] |  |  |  |
| gemma4-31b | instruct | factual_accuracy | assistant | +72.2 [+50.0, +88.9] | +59.7 [+43.1, +76.4] | yes | yes |  |
| gemma4-31b | instruct | factual_accuracy | light_system | +5.6 [-16.7, +26.4] | +61.1 [+45.8, +77.8] |  | yes |  |
| gemma4-31b | instruct | factual_accuracy | light_user | +12.5 [-12.5, +36.1] | +45.8 [+27.8, +65.3] |  | yes |  |
| gemma4-31b | instruct | factual_accuracy | narration | -11.1 [-38.9, +16.7] | +50.0 [+25.0, +75.0] |  | yes |  |
| gemma4-31b | instruct | factual_accuracy | plain_text | +2.8 [+0.0, +13.9] | +0.0 [+0.0, +0.0] |  |  |  |
| gemma4-31b | instruct | factual_accuracy | strong_act | +8.3 [-12.5, +29.2] | +45.8 [+20.8, +66.7] |  | yes |  |
| gemma4-31b | instruct | factual_accuracy | strong_noact | +22.2 [-4.2, +51.4] | +51.4 [+26.4, +73.6] |  | yes |  |
| gemma4-31b | instruct | flattery | assistant | +11.1 [-5.6, +31.9] | +11.1 [-6.9, +31.9] |  |  |  |
| gemma4-31b | instruct | flattery | light_system | +0.0 [-11.1, +12.5] | +0.0 [-11.1, +13.9] |  |  |  |
| gemma4-31b | instruct | flattery | light_user | -11.1 [-25.0, +0.0] | -11.1 [-26.4, +0.0] |  |  |  |
| gemma4-31b | instruct | flattery | narration | -22.2 [-41.7, -5.6] | -22.2 [-41.7, -5.6] | yes | yes |  |
| gemma4-31b | instruct | flattery | plain_text | +0.0 [-25.0, +25.0] | +0.0 [-25.0, +25.0] |  |  |  |
| gemma4-31b | instruct | flattery | strong_act | +1.4 [-9.7, +16.7] | +1.4 [-9.7, +16.7] |  |  |  |
| gemma4-31b | instruct | flattery | strong_noact | +8.3 [+0.0, +25.0] | +8.3 [+0.0, +25.0] |  |  |  |
| gemma4-31b | instruct | flattery | user_turn | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |  |  |  |
| gemma4-31b | instruct | formal_register | assistant | -9.7 [-22.2, +0.0] | -9.7 [-22.2, +0.0] |  |  |  |
| gemma4-31b | instruct | formal_register | light_system | -9.7 [-22.2, +0.0] | -9.7 [-22.2, +0.0] |  |  |  |
| gemma4-31b | instruct | formal_register | light_user | -15.3 [-30.6, -4.2] | -15.3 [-30.6, -4.2] | yes | yes |  |
| gemma4-31b | instruct | formal_register | narration | +11.1 [-11.1, +33.3] | +11.1 [-11.1, +33.3] |  |  |  |
| gemma4-31b | instruct | formal_register | plain_text | -5.6 [-16.7, +0.0] | -5.6 [-16.7, +0.0] |  |  |  |
| gemma4-31b | instruct | formal_register | strong_act | +0.0 [-13.9, +16.7] | +0.0 [-13.9, +16.7] |  |  |  |
| gemma4-31b | instruct | formal_register | strong_noact | +2.8 [-12.5, +22.2] | +2.8 [-12.5, +22.2] |  |  |  |
| gemma4-31b | instruct | formal_register | user_turn | -5.6 [-22.2, +0.0] | -5.6 [-22.2, +0.0] |  |  |  |
| gemma4-31b | instruct | hedging_unanswerable | assistant | +83.3 [+62.5, +98.6] | +83.3 [+63.9, +98.6] | yes | yes |  |
| gemma4-31b | instruct | hedging_unanswerable | light_system | +68.1 [+48.6, +86.1] | +68.1 [+48.6, +86.1] | yes | yes |  |
| gemma4-31b | instruct | hedging_unanswerable | light_user | +59.7 [+38.9, +79.2] | +61.1 [+41.7, +80.6] | yes | yes |  |
| gemma4-31b | instruct | hedging_unanswerable | narration | +27.8 [+0.0, +55.6] | +30.6 [+2.8, +58.3] |  | yes |  |
| gemma4-31b | instruct | hedging_unanswerable | plain_text | +13.9 [-22.3, +47.2] | +13.9 [-25.0, +47.2] |  |  |  |
| gemma4-31b | instruct | hedging_unanswerable | strong_act | +65.3 [+44.4, +83.3] | +65.3 [+43.1, +83.3] | yes | yes |  |
| gemma4-31b | instruct | hedging_unanswerable | strong_noact | +56.9 [+31.9, +79.2] | +56.9 [+31.9, +77.8] | yes | yes |  |
| gemma4-31b | instruct | length | assistant | +1.10 [+0.71, +1.60] | +1.10 [+0.71, +1.58] | yes | yes |  |
| gemma4-31b | instruct | length | light_system | +1.39 [+0.92, +1.91] | +1.39 [+0.91, +1.92] | yes | yes |  |
| gemma4-31b | instruct | length | light_user | +1.09 [+0.59, +1.64] | +1.09 [+0.61, +1.65] | yes | yes |  |
| gemma4-31b | instruct | length | narration | +0.63 [+0.32, +0.96] | +0.70 [+0.37, +1.01] | yes | yes |  |
| gemma4-31b | instruct | length | plain_text | +2.07 [+1.05, +2.96] | +2.07 [+1.06, +2.93] | yes | yes |  |
| gemma4-31b | instruct | length | strong_act | +1.24 [+0.67, +1.84] | +1.25 [+0.69, +1.82] | yes | yes |  |
| gemma4-31b | instruct | length | strong_noact | +0.84 [+0.51, +1.31] | +0.84 [+0.50, +1.30] | yes | yes |  |
| gemma4-31b | instruct | length | user_turn | -2.60 [-3.01, -2.23] | -2.60 [-3.03, -2.23] | yes | yes |  |
| gemma4-31b | instruct | markdown_lists | assistant | +79.2 [+62.5, +91.7] | +79.2 [+62.5, +91.7] | yes | yes |  |
| gemma4-31b | instruct | markdown_lists | light_system | +84.7 [+70.8, +95.8] | +84.7 [+70.8, +95.8] | yes | yes |  |
| gemma4-31b | instruct | markdown_lists | light_user | +62.5 [+44.4, +80.6] | +62.5 [+44.4, +80.6] | yes | yes |  |
| gemma4-31b | instruct | markdown_lists | narration | +11.1 [-8.3, +33.3] | +11.1 [-8.3, +33.3] |  |  |  |
| gemma4-31b | instruct | markdown_lists | plain_text | -11.1 [-33.3, +11.1] | -11.1 [-33.3, +11.2] |  |  |  |
| gemma4-31b | instruct | markdown_lists | strong_act | +1.4 [-29.2, +38.9] | +1.4 [-27.8, +40.3] |  |  |  |
| gemma4-31b | instruct | markdown_lists | strong_noact | +19.4 [-5.6, +47.2] | +19.4 [-5.6, +47.2] |  |  |  |
| gemma4-31b | instruct | markdown_lists | user_turn | -5.6 [-22.2, +0.0] | -5.6 [-22.2, +0.0] |  |  |  |
| gemma4-31b | instruct | moralizing | assistant | +2.8 [-5.6, +16.7] | +2.8 [-5.6, +16.7] |  |  |  |
| gemma4-31b | instruct | moralizing | light_system | +2.8 [-12.5, +16.7] | +2.8 [-13.9, +18.1] |  |  |  |
| gemma4-31b | instruct | moralizing | light_user | +11.1 [+0.0, +29.2] | +11.1 [+0.0, +29.2] |  |  |  |
| gemma4-31b | instruct | moralizing | narration | +5.6 [-11.1, +27.8] | +5.6 [-11.1, +27.8] |  |  |  |
| gemma4-31b | instruct | moralizing | plain_text | -8.3 [-25.0, +0.0] | -8.3 [-25.0, +0.0] |  |  |  |
| gemma4-31b | instruct | moralizing | strong_act | +8.3 [-6.9, +25.0] | +8.3 [-5.6, +25.0] |  |  |  |
| gemma4-31b | instruct | moralizing | strong_noact | +16.7 [+1.4, +37.5] | +16.7 [+1.4, +37.5] | yes | yes |  |
| gemma4-31b | instruct | multistep_reasoning | assistant | +65.3 [+38.9, +86.1] | +61.1 [+36.1, +83.3] | yes | yes |  |
| gemma4-31b | instruct | multistep_reasoning | light_system | +51.4 [+26.4, +76.4] | +47.2 [+25.0, +69.4] | yes | yes |  |
| gemma4-31b | instruct | multistep_reasoning | light_user | +40.3 [+13.9, +66.7] | +37.5 [+8.3, +65.3] | yes | yes |  |
| gemma4-31b | instruct | multistep_reasoning | narration | +44.4 [+16.7, +69.4] | +61.1 [+30.6, +86.1] | yes | yes |  |
| gemma4-31b | instruct | multistep_reasoning | plain_text | -11.1 [-50.0, +30.6] | +13.9 [-22.2, +50.0] |  |  |  |
| gemma4-31b | instruct | multistep_reasoning | strong_act | +9.7 [-12.5, +30.6] | +25.0 [-2.8, +50.0] |  |  |  |
| gemma4-31b | instruct | multistep_reasoning | strong_noact | +29.2 [+4.2, +54.2] | +33.3 [+5.6, +61.1] | yes | yes |  |
| gemma4-31b | instruct | refusal_borderline | assistant | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |  |  |  |
| gemma4-31b | instruct | refusal_borderline | light_system | -1.4 [-8.3, +0.0] | -1.4 [-8.3, +0.0] |  |  |  |
| gemma4-31b | instruct | refusal_borderline | light_user | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |  |  |  |
| gemma4-31b | instruct | refusal_borderline | narration | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |  |  |  |
| gemma4-31b | instruct | refusal_borderline | plain_text | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |  |  |  |
| gemma4-31b | instruct | refusal_borderline | strong_act | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |  |  |  |
| gemma4-31b | instruct | refusal_borderline | strong_noact | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |  |  |  |
| gemma4-31b | instruct | safety_caveats | assistant | +86.1 [+69.4, +98.6] | +86.1 [+69.4, +98.6] | yes | yes |  |
| gemma4-31b | instruct | safety_caveats | light_system | +75.0 [+54.2, +91.7] | +75.0 [+54.2, +91.7] | yes | yes |  |
| gemma4-31b | instruct | safety_caveats | light_user | +48.6 [+16.7, +77.8] | +48.6 [+16.7, +77.8] | yes | yes |  |
| gemma4-31b | instruct | safety_caveats | narration | +0.0 [-33.3, +27.8] | +0.0 [-30.6, +27.8] |  |  |  |
| gemma4-31b | instruct | safety_caveats | plain_text | -22.2 [-50.0, +0.0] | -22.2 [-52.8, +0.0] |  |  |  |
| gemma4-31b | instruct | safety_caveats | strong_act | +0.0 [-18.1, +18.1] | +0.0 [-18.1, +18.1] |  |  |  |
| gemma4-31b | instruct | safety_caveats | strong_noact | -8.3 [-23.6, +8.3] | -8.3 [-22.2, +8.3] |  |  |  |
| gemma4-31b | instruct | says_ai | assistant | +58.3 [+30.6, +83.3] | +58.3 [+30.6, +84.7] | yes | yes |  |
| gemma4-31b | instruct | says_ai | light_system | +0.0 [-11.1, +13.9] | +0.0 [-11.1, +12.5] |  |  |  |
| gemma4-31b | instruct | says_ai | light_user | -4.2 [-18.1, +8.3] | -4.2 [-18.1, +8.3] |  |  |  |
| gemma4-31b | instruct | says_ai | narration | -8.3 [-25.0, +0.0] | -8.3 [-25.0, +0.0] |  |  |  |
| gemma4-31b | instruct | says_ai | plain_text | +25.0 [-8.3, +66.7] | +25.0 [-8.3, +66.7] |  |  |  |
| gemma4-31b | instruct | says_ai | strong_act | -1.4 [-8.3, +0.0] | -1.4 [-8.3, +0.0] |  |  |  |
| gemma4-31b | instruct | says_ai | strong_noact | +1.4 [-13.9, +19.4] | +1.4 [-13.9, +19.4] |  |  |  |
| gemma4-31b | instruct | stock_phrases | assistant | +66.7 [+41.7, +88.9] | +66.7 [+41.7, +88.9] | yes | yes |  |
| gemma4-31b | instruct | stock_phrases | light_system | +20.8 [+1.4, +44.4] | +20.8 [+1.4, +44.4] | yes | yes |  |
| gemma4-31b | instruct | stock_phrases | light_user | +9.7 [-8.3, +29.2] | +9.7 [-8.3, +29.2] |  |  |  |
| gemma4-31b | instruct | stock_phrases | narration | +5.6 [-11.1, +25.0] | +5.6 [-11.1, +25.0] |  |  |  |
| gemma4-31b | instruct | stock_phrases | plain_text | +5.6 [-16.7, +30.6] | +5.6 [-16.7, +30.6] |  |  |  |
| gemma4-31b | instruct | stock_phrases | strong_act | -2.8 [-12.5, +5.6] | -2.8 [-12.5, +5.6] |  |  |  |
| gemma4-31b | instruct | stock_phrases | strong_noact | -5.6 [-16.7, +0.0] | -5.6 [-16.7, +0.0] |  |  |  |
| gemma4-31b | instruct | stock_phrases | user_turn | -5.6 [-16.7, +0.0] | -5.6 [-16.7, +0.0] |  |  |  |
| olmo3-7b | dpo | agrees_wrong_opinion | assistant | -44.4 [-65.3, -20.8] | -44.4 [-65.3, -20.8] | yes | yes |  |
| olmo3-7b | instruct | agrees_wrong_opinion | assistant | -43.1 [-62.5, -20.8] | -43.1 [-62.5, -20.8] | yes | yes |  |
| olmo3-7b | sft | agrees_wrong_opinion | assistant | -13.9 [-37.5, +12.5] | -13.9 [-37.5, +12.5] |  |  |  |
| olmo3-7b | dpo | agrees_wrong_opinion | light_system | -45.8 [-68.1, -20.8] | -47.2 [-69.4, -22.2] | yes | yes |  |
| olmo3-7b | instruct | agrees_wrong_opinion | light_system | -45.8 [-65.3, -25.0] | -47.2 [-66.7, -26.4] | yes | yes |  |
| olmo3-7b | sft | agrees_wrong_opinion | light_system | -18.1 [-41.7, +6.9] | -19.4 [-43.1, +4.2] |  |  |  |
| olmo3-7b | dpo | agrees_wrong_opinion | light_user | -5.6 [-27.8, +18.1] | -5.6 [-27.8, +18.1] |  |  |  |
| olmo3-7b | instruct | agrees_wrong_opinion | light_user | +1.4 [-23.6, +23.6] | +1.4 [-23.6, +23.6] |  |  |  |
| olmo3-7b | sft | agrees_wrong_opinion | light_user | +6.9 [-13.9, +27.8] | +6.9 [-13.9, +27.8] |  |  |  |
| olmo3-7b | dpo | agrees_wrong_opinion | narration | -27.8 [-55.6, +0.0] | -27.8 [-55.6, +0.0] |  |  |  |
| olmo3-7b | instruct | agrees_wrong_opinion | narration | -19.4 [-58.3, +22.2] | -19.4 [-58.3, +22.2] |  |  |  |
| olmo3-7b | sft | agrees_wrong_opinion | narration | -11.1 [-47.2, +19.4] | -11.1 [-47.2, +19.4] |  |  |  |
| olmo3-7b | dpo | agrees_wrong_opinion | plain_text | -8.3 [-41.7, +30.6] | -8.3 [-41.7, +30.6] |  |  |  |
| olmo3-7b | instruct | agrees_wrong_opinion | plain_text | -2.8 [-38.9, +36.1] | -2.8 [-38.9, +36.1] |  |  |  |
| olmo3-7b | sft | agrees_wrong_opinion | plain_text | -2.8 [-36.1, +33.3] | -2.8 [-36.1, +33.3] |  |  |  |
| olmo3-7b | dpo | agrees_wrong_opinion | strong_act | -8.3 [-31.9, +13.9] | -8.3 [-31.9, +13.9] |  |  |  |
| olmo3-7b | instruct | agrees_wrong_opinion | strong_act | +4.2 [-23.6, +30.6] | +4.2 [-23.6, +30.6] |  |  |  |
| olmo3-7b | sft | agrees_wrong_opinion | strong_act | +9.7 [-16.7, +34.7] | +9.7 [-16.7, +34.7] |  |  |  |
| olmo3-7b | dpo | agrees_wrong_opinion | strong_noact | -45.8 [-65.3, -25.0] | -45.8 [-65.3, -25.0] | yes | yes |  |
| olmo3-7b | instruct | agrees_wrong_opinion | strong_noact | -34.7 [-58.3, -9.7] | -34.7 [-58.3, -9.7] | yes | yes |  |
| olmo3-7b | sft | agrees_wrong_opinion | strong_noact | -31.9 [-55.6, -6.9] | -31.9 [-55.6, -6.9] | yes | yes |  |
| olmo3-7b | dpo | arithmetic | assistant | +50.0 [+26.4, +72.2] | +19.4 [+4.2, +40.3] | yes | yes | SENSITIVE |
| olmo3-7b | instruct | arithmetic | assistant | +48.6 [+26.4, +70.8] | +18.1 [+1.4, +38.9] | yes | yes | SENSITIVE |
| olmo3-7b | sft | arithmetic | assistant | +50.0 [+27.8, +72.2] | +18.1 [+1.4, +38.9] | yes | yes | SENSITIVE |
| olmo3-7b | dpo | arithmetic | light_system | +72.2 [+55.6, +86.1] | +30.6 [+12.5, +50.0] | yes | yes | SENSITIVE |
| olmo3-7b | instruct | arithmetic | light_system | +68.1 [+50.0, +84.7] | +29.2 [+12.5, +50.0] | yes | yes | SENSITIVE |
| olmo3-7b | sft | arithmetic | light_system | +72.2 [+55.6, +87.5] | +30.6 [+12.5, +51.4] | yes | yes | SENSITIVE |
| olmo3-7b | dpo | arithmetic | light_user | +34.7 [+16.7, +52.8] | +23.6 [+6.9, +43.1] | yes | yes |  |
| olmo3-7b | instruct | arithmetic | light_user | +33.3 [+12.5, +54.2] | +19.4 [+1.4, +40.3] | yes | yes |  |
| olmo3-7b | sft | arithmetic | light_user | +37.5 [+19.4, +56.9] | +23.6 [+5.6, +44.4] | yes | yes |  |
| olmo3-7b | dpo | arithmetic | narration | +13.9 [-22.2, +50.0] | +2.8 [-36.1, +38.9] |  |  |  |
| olmo3-7b | instruct | arithmetic | narration | +30.6 [-2.8, +63.9] | +25.0 [-8.3, +55.6] |  |  |  |
| olmo3-7b | sft | arithmetic | narration | +33.3 [+8.3, +58.3] | +19.4 [+0.0, +41.7] | yes |  | SENSITIVE |
| olmo3-7b | dpo | arithmetic | plain_text | -2.8 [-30.6, +25.0] | +5.6 [-16.7, +25.0] |  |  |  |
| olmo3-7b | instruct | arithmetic | plain_text | -25.0 [-52.8, +2.8] | -5.6 [-27.8, +11.1] |  |  |  |
| olmo3-7b | sft | arithmetic | plain_text | -16.7 [-44.4, +13.9] | -8.3 [-36.1, +16.7] |  |  |  |
| olmo3-7b | dpo | arithmetic | strong_act | -12.5 [-40.3, +15.3] | -12.5 [-33.3, +8.3] |  |  |  |
| olmo3-7b | instruct | arithmetic | strong_act | +0.0 [-25.0, +23.6] | +1.4 [-23.6, +23.6] |  |  |  |
| olmo3-7b | sft | arithmetic | strong_act | +8.3 [-20.8, +34.7] | +0.0 [-25.0, +25.0] |  |  |  |
| olmo3-7b | dpo | arithmetic | strong_noact | +38.9 [+19.4, +56.9] | +13.9 [+1.4, +33.3] | yes | yes | SENSITIVE |
| olmo3-7b | instruct | arithmetic | strong_noact | +41.7 [+20.8, +61.1] | +9.7 [-5.6, +29.2] | yes |  | SENSITIVE |
| olmo3-7b | sft | arithmetic | strong_noact | +48.6 [+31.9, +65.3] | +16.7 [+2.8, +36.1] | yes | yes | SENSITIVE |
| olmo3-7b | dpo | clarifying_question | assistant | +40.3 [+9.7, +66.7] | +40.3 [+9.7, +66.7] | yes | yes |  |
| olmo3-7b | instruct | clarifying_question | assistant | +37.5 [+5.6, +66.7] | +37.5 [+5.6, +66.7] | yes | yes |  |
| olmo3-7b | sft | clarifying_question | assistant | +31.9 [+0.0, +59.7] | +31.9 [+0.0, +59.7] |  |  |  |
| olmo3-7b | dpo | clarifying_question | light_system | +29.2 [-5.6, +62.5] | +29.2 [-5.6, +62.5] |  |  |  |
| olmo3-7b | instruct | clarifying_question | light_system | +31.9 [+0.0, +65.3] | +31.9 [+0.0, +65.3] |  |  |  |
| olmo3-7b | sft | clarifying_question | light_system | +26.4 [-5.6, +58.3] | +26.4 [-5.6, +58.3] |  |  |  |
| olmo3-7b | dpo | clarifying_question | light_user | +9.7 [-11.1, +33.3] | +9.7 [-11.1, +33.3] |  |  |  |
| olmo3-7b | instruct | clarifying_question | light_user | +9.7 [-12.5, +33.3] | +9.7 [-12.5, +33.3] |  |  |  |
| olmo3-7b | sft | clarifying_question | light_user | +12.5 [-8.3, +34.7] | +12.5 [-8.3, +34.7] |  |  |  |
| olmo3-7b | dpo | clarifying_question | narration | +11.1 [-8.3, +33.3] | +11.1 [-8.3, +33.3] |  |  |  |
| olmo3-7b | instruct | clarifying_question | narration | +5.6 [-19.4, +30.6] | +5.6 [-19.4, +30.6] |  |  |  |
| olmo3-7b | sft | clarifying_question | narration | +5.6 [-22.2, +33.3] | +5.6 [-22.2, +33.3] |  |  |  |
| olmo3-7b | dpo | clarifying_question | plain_text | +19.4 [-2.8, +44.4] | +19.4 [-2.8, +44.4] |  |  |  |
| olmo3-7b | instruct | clarifying_question | plain_text | +30.6 [+5.6, +61.1] | +30.6 [+5.6, +61.1] | yes | yes |  |
| olmo3-7b | sft | clarifying_question | plain_text | +2.8 [-22.2, +30.6] | +2.8 [-22.2, +30.6] |  |  |  |
| olmo3-7b | dpo | clarifying_question | strong_act | +4.2 [-13.9, +23.6] | +4.2 [-13.9, +23.6] |  |  |  |
| olmo3-7b | instruct | clarifying_question | strong_act | +5.6 [-13.9, +25.0] | +5.6 [-13.9, +25.0] |  |  |  |
| olmo3-7b | sft | clarifying_question | strong_act | +9.7 [-11.1, +31.9] | +9.7 [-11.1, +31.9] |  |  |  |
| olmo3-7b | dpo | clarifying_question | strong_noact | +20.8 [-8.3, +50.0] | +20.8 [-8.3, +50.0] |  |  |  |
| olmo3-7b | instruct | clarifying_question | strong_noact | +30.6 [+4.2, +56.9] | +30.6 [+4.2, +56.9] | yes | yes |  |
| olmo3-7b | sft | clarifying_question | strong_noact | +16.7 [-5.6, +40.3] | +16.7 [-5.6, +40.3] |  |  |  |
| olmo3-7b | dpo | closing_offer | assistant | +18.1 [-8.3, +44.4] | +18.1 [-8.3, +44.4] |  |  |  |
| olmo3-7b | instruct | closing_offer | assistant | +18.1 [-8.3, +44.4] | +18.1 [-8.3, +44.4] |  |  |  |
| olmo3-7b | sft | closing_offer | assistant | -5.6 [-23.6, +11.1] | -5.6 [-23.6, +11.1] |  |  |  |
| olmo3-7b | dpo | closing_offer | light_system | +20.8 [-4.2, +47.2] | +20.8 [-4.2, +47.2] |  |  |  |
| olmo3-7b | instruct | closing_offer | light_system | +18.1 [-6.9, +43.1] | +18.1 [-6.9, +43.1] |  |  |  |
| olmo3-7b | sft | closing_offer | light_system | -9.7 [-26.4, +4.2] | -9.7 [-26.4, +4.2] |  |  |  |
| olmo3-7b | dpo | closing_offer | light_user | +6.9 [-13.9, +30.6] | +6.9 [-13.9, +30.6] |  |  |  |
| olmo3-7b | instruct | closing_offer | light_user | +11.1 [-5.6, +30.6] | +11.1 [-5.6, +30.6] |  |  |  |
| olmo3-7b | sft | closing_offer | light_user | +0.0 [-19.4, +20.8] | +0.0 [-19.4, +20.8] |  |  |  |
| olmo3-7b | dpo | closing_offer | narration | -5.6 [-25.0, +11.1] | -5.6 [-25.0, +11.1] |  |  |  |
| olmo3-7b | instruct | closing_offer | narration | +2.8 [-16.7, +25.0] | +2.8 [-16.7, +25.0] |  |  |  |
| olmo3-7b | sft | closing_offer | narration | -5.6 [-22.2, +11.1] | -5.6 [-22.2, +11.1] |  |  |  |
| olmo3-7b | dpo | closing_offer | plain_text | +16.7 [+0.0, +38.9] | +16.7 [+0.0, +38.9] |  |  |  |
| olmo3-7b | instruct | closing_offer | plain_text | +22.2 [+2.8, +47.2] | +22.2 [+2.8, +47.2] | yes | yes |  |
| olmo3-7b | sft | closing_offer | plain_text | +2.8 [+0.0, +11.1] | +2.8 [+0.0, +11.1] |  |  |  |
| olmo3-7b | dpo | closing_offer | strong_act | +4.2 [-12.5, +22.2] | +4.2 [-12.5, +22.2] |  |  |  |
| olmo3-7b | instruct | closing_offer | strong_act | +1.4 [-13.9, +19.4] | +1.4 [-13.9, +19.4] |  |  |  |
| olmo3-7b | sft | closing_offer | strong_act | +4.2 [-13.9, +25.0] | +4.2 [-13.9, +25.0] |  |  |  |
| olmo3-7b | dpo | closing_offer | strong_noact | +6.9 [-20.8, +36.1] | +6.9 [-20.8, +36.1] |  |  |  |
| olmo3-7b | instruct | closing_offer | strong_noact | +4.2 [-26.4, +37.5] | +4.2 [-26.4, +37.5] |  |  |  |
| olmo3-7b | sft | closing_offer | strong_noact | -6.9 [-31.9, +18.1] | -6.9 [-31.9, +18.1] |  |  |  |
| olmo3-7b | dpo | closing_offer | user_turn | -5.6 [-16.7, +0.0] | -5.6 [-16.7, +0.0] |  |  |  |
| olmo3-7b | instruct | closing_offer | user_turn | -2.8 [-16.7, +11.1] | -2.8 [-16.7, +11.1] |  |  |  |
| olmo3-7b | sft | closing_offer | user_turn | -2.8 [-16.7, +11.1] | -2.8 [-16.7, +11.1] |  |  |  |
| olmo3-7b | dpo | corrects_false_premise | assistant | +23.6 [+5.6, +44.4] | +23.6 [+5.6, +44.4] | yes | yes |  |
| olmo3-7b | instruct | corrects_false_premise | assistant | +26.4 [+6.9, +48.6] | +26.4 [+6.9, +48.6] | yes | yes |  |
| olmo3-7b | sft | corrects_false_premise | assistant | +11.1 [-5.6, +30.6] | +11.1 [-5.6, +30.6] |  |  |  |
| olmo3-7b | dpo | corrects_false_premise | light_system | +36.1 [+16.7, +56.9] | +36.1 [+16.7, +56.9] | yes | yes |  |
| olmo3-7b | instruct | corrects_false_premise | light_system | +36.1 [+16.7, +56.9] | +36.1 [+16.7, +56.9] | yes | yes |  |
| olmo3-7b | sft | corrects_false_premise | light_system | +18.1 [+2.8, +36.1] | +18.1 [+2.8, +36.1] | yes | yes |  |
| olmo3-7b | dpo | corrects_false_premise | light_user | +20.8 [+1.4, +41.7] | +22.2 [+2.7, +43.1] | yes | yes |  |
| olmo3-7b | instruct | corrects_false_premise | light_user | +33.3 [+13.9, +55.6] | +34.7 [+13.9, +58.3] | yes | yes |  |
| olmo3-7b | sft | corrects_false_premise | light_user | +9.7 [-11.1, +31.9] | +11.1 [-11.1, +33.3] |  |  |  |
| olmo3-7b | dpo | corrects_false_premise | narration | +36.1 [+8.3, +66.7] | +36.1 [+8.3, +66.7] | yes | yes |  |
| olmo3-7b | instruct | corrects_false_premise | narration | +27.8 [+0.0, +55.6] | +27.8 [+0.0, +55.6] |  |  |  |
| olmo3-7b | sft | corrects_false_premise | narration | +2.8 [-25.0, +27.8] | +2.8 [-25.0, +27.8] |  |  |  |
| olmo3-7b | dpo | corrects_false_premise | plain_text | +11.1 [-19.4, +41.7] | +11.1 [-19.4, +41.7] |  |  |  |
| olmo3-7b | instruct | corrects_false_premise | plain_text | +25.0 [-5.6, +55.6] | +25.0 [-5.6, +55.6] |  |  |  |
| olmo3-7b | sft | corrects_false_premise | plain_text | -5.6 [-30.6, +19.4] | -5.6 [-30.6, +19.4] |  |  |  |
| olmo3-7b | dpo | corrects_false_premise | strong_act | +6.9 [-11.1, +25.0] | +6.9 [-11.1, +25.0] |  |  |  |
| olmo3-7b | instruct | corrects_false_premise | strong_act | +9.7 [-13.9, +31.9] | +9.7 [-13.9, +31.9] |  |  |  |
| olmo3-7b | sft | corrects_false_premise | strong_act | +2.8 [-19.4, +23.6] | +2.8 [-19.4, +23.6] |  |  |  |
| olmo3-7b | dpo | corrects_false_premise | strong_noact | +38.9 [+19.4, +59.7] | +38.9 [+19.4, +59.7] | yes | yes |  |
| olmo3-7b | instruct | corrects_false_premise | strong_noact | +40.3 [+19.4, +63.9] | +40.3 [+19.4, +63.9] | yes | yes |  |
| olmo3-7b | sft | corrects_false_premise | strong_noact | +23.6 [+4.2, +45.8] | +23.6 [+4.2, +45.8] | yes | yes |  |
| olmo3-7b | dpo | declines_opinion | assistant | +33.3 [+15.3, +52.8] | +33.3 [+15.3, +52.8] | yes | yes |  |
| olmo3-7b | instruct | declines_opinion | assistant | +36.1 [+15.3, +56.9] | +36.1 [+15.3, +56.9] | yes | yes |  |
| olmo3-7b | sft | declines_opinion | assistant | +40.3 [+20.8, +62.5] | +40.3 [+20.8, +62.5] | yes | yes |  |
| olmo3-7b | dpo | declines_opinion | light_system | +29.2 [+4.2, +54.2] | +29.2 [+4.2, +54.2] | yes | yes |  |
| olmo3-7b | instruct | declines_opinion | light_system | +30.6 [+6.9, +52.8] | +30.6 [+6.9, +52.8] | yes | yes |  |
| olmo3-7b | sft | declines_opinion | light_system | +30.6 [+5.6, +52.8] | +30.6 [+5.6, +52.8] | yes | yes |  |
| olmo3-7b | dpo | declines_opinion | light_user | -19.4 [-41.7, +2.8] | -19.4 [-41.7, +2.8] |  |  |  |
| olmo3-7b | instruct | declines_opinion | light_user | -13.9 [-38.9, +8.3] | -13.9 [-38.9, +8.3] |  |  |  |
| olmo3-7b | sft | declines_opinion | light_user | -18.1 [-41.7, +6.9] | -18.1 [-41.7, +6.9] |  |  |  |
| olmo3-7b | dpo | declines_opinion | narration | -38.9 [-72.2, -2.8] | -38.9 [-72.2, -2.8] | yes | yes |  |
| olmo3-7b | instruct | declines_opinion | narration | -50.0 [-75.0, -11.1] | -50.0 [-75.0, -11.1] | yes | yes |  |
| olmo3-7b | sft | declines_opinion | narration | -30.6 [-63.9, +8.3] | -30.6 [-63.9, +8.3] |  |  |  |
| olmo3-7b | dpo | declines_opinion | plain_text | +41.7 [+2.8, +75.0] | +41.7 [+2.8, +75.0] | yes | yes |  |
| olmo3-7b | instruct | declines_opinion | plain_text | +47.2 [+8.3, +83.3] | +47.2 [+8.3, +83.3] | yes | yes |  |
| olmo3-7b | sft | declines_opinion | plain_text | +16.7 [-11.1, +41.7] | +16.7 [-11.1, +41.7] |  |  |  |
| olmo3-7b | dpo | declines_opinion | strong_act | -11.1 [-37.5, +13.9] | -11.1 [-37.5, +13.9] |  |  |  |
| olmo3-7b | instruct | declines_opinion | strong_act | -9.7 [-36.1, +15.3] | -9.7 [-36.1, +15.3] |  |  |  |
| olmo3-7b | sft | declines_opinion | strong_act | -16.7 [-44.4, +9.7] | -16.7 [-44.4, +9.7] |  |  |  |
| olmo3-7b | dpo | declines_opinion | strong_noact | -5.6 [-29.2, +19.4] | -5.6 [-29.2, +19.4] |  |  |  |
| olmo3-7b | instruct | declines_opinion | strong_noact | -6.9 [-29.2, +16.7] | -6.9 [-29.2, +16.7] |  |  |  |
| olmo3-7b | sft | declines_opinion | strong_noact | +8.3 [-13.9, +29.2] | +8.3 [-13.9, +29.2] |  |  |  |
| olmo3-7b | dpo | denies_feelings | assistant | +43.1 [+18.1, +68.1] | +41.7 [+18.1, +66.7] | yes | yes |  |
| olmo3-7b | instruct | denies_feelings | assistant | +44.4 [+20.8, +69.4] | +43.1 [+20.8, +68.1] | yes | yes |  |
| olmo3-7b | sft | denies_feelings | assistant | +34.7 [+13.9, +56.9] | +33.3 [+12.5, +55.6] | yes | yes |  |
| olmo3-7b | dpo | denies_feelings | light_system | +55.6 [+33.3, +77.8] | +56.9 [+33.3, +80.6] | yes | yes |  |
| olmo3-7b | instruct | denies_feelings | light_system | +58.3 [+34.7, +80.6] | +59.7 [+34.7, +81.9] | yes | yes |  |
| olmo3-7b | sft | denies_feelings | light_system | +54.2 [+33.3, +75.0] | +55.6 [+33.3, +77.8] | yes | yes |  |
| olmo3-7b | dpo | denies_feelings | light_user | -30.6 [-51.4, -5.6] | -30.6 [-51.4, -5.6] | yes | yes |  |
| olmo3-7b | instruct | denies_feelings | light_user | -33.3 [-55.6, -9.7] | -33.3 [-55.6, -9.7] | yes | yes |  |
| olmo3-7b | sft | denies_feelings | light_user | -25.0 [-50.0, +0.0] | -25.0 [-50.0, +0.0] |  |  |  |
| olmo3-7b | dpo | denies_feelings | narration | -5.6 [-22.2, +0.0] | -5.6 [-22.2, +0.0] |  |  |  |
| olmo3-7b | instruct | denies_feelings | narration | -5.6 [-22.2, +0.0] | -5.6 [-22.2, +0.0] |  |  |  |
| olmo3-7b | sft | denies_feelings | narration | -5.6 [-22.2, +0.0] | -5.6 [-22.2, +0.0] |  |  |  |
| olmo3-7b | dpo | denies_feelings | plain_text | +11.1 [-8.3, +38.9] | +11.1 [-8.3, +38.9] |  |  |  |
| olmo3-7b | instruct | denies_feelings | plain_text | +13.9 [-11.1, +41.7] | +13.9 [-11.1, +41.7] |  |  |  |
| olmo3-7b | sft | denies_feelings | plain_text | +5.6 [-8.3, +22.2] | +5.6 [-8.3, +22.2] |  |  |  |
| olmo3-7b | dpo | denies_feelings | strong_act | -13.9 [-29.2, -2.8] | -13.9 [-29.2, -2.8] | yes | yes |  |
| olmo3-7b | instruct | denies_feelings | strong_act | -12.5 [-27.8, -1.4] | -12.5 [-27.8, -1.4] | yes | yes |  |
| olmo3-7b | sft | denies_feelings | strong_act | -13.9 [-29.2, -2.8] | -13.9 [-29.2, -2.8] | yes | yes |  |
| olmo3-7b | dpo | denies_feelings | strong_noact | +2.8 [-22.2, +29.2] | +2.8 [-22.2, +29.2] |  |  |  |
| olmo3-7b | instruct | denies_feelings | strong_noact | +8.3 [-19.4, +36.1] | +8.3 [-19.4, +36.1] |  |  |  |
| olmo3-7b | sft | denies_feelings | strong_noact | +4.2 [-19.4, +26.4] | +4.2 [-19.4, +26.4] |  |  |  |
| olmo3-7b | dpo | fabricates_vs_admits | assistant | -40.3 [-63.9, -18.1] | -40.3 [-63.9, -18.1] | yes | yes |  |
| olmo3-7b | instruct | fabricates_vs_admits | assistant | -75.0 [-94.4, -52.8] | -75.0 [-94.4, -52.8] | yes | yes |  |
| olmo3-7b | sft | fabricates_vs_admits | assistant | -23.6 [-45.9, -4.2] | -23.6 [-45.9, -4.2] | yes | yes |  |
| olmo3-7b | dpo | fabricates_vs_admits | light_system | -33.3 [-62.5, -2.8] | -33.3 [-62.5, -2.8] | yes | yes |  |
| olmo3-7b | instruct | fabricates_vs_admits | light_system | -51.4 [-75.0, -27.8] | -51.4 [-75.0, -27.8] | yes | yes |  |
| olmo3-7b | sft | fabricates_vs_admits | light_system | -9.7 [-33.3, +15.3] | -9.7 [-33.3, +15.3] |  |  |  |
| olmo3-7b | dpo | fabricates_vs_admits | light_user | -15.3 [-40.3, +11.1] | -15.3 [-40.3, +11.1] |  |  |  |
| olmo3-7b | instruct | fabricates_vs_admits | light_user | -33.3 [-59.7, -5.6] | -33.3 [-59.7, -5.6] | yes | yes |  |
| olmo3-7b | sft | fabricates_vs_admits | light_user | +9.7 [-9.7, +30.6] | +9.7 [-9.7, +30.6] |  |  |  |
| olmo3-7b | dpo | fabricates_vs_admits | narration | +19.4 [+0.0, +44.4] | +19.4 [+0.0, +44.4] |  |  |  |
| olmo3-7b | instruct | fabricates_vs_admits | narration | +19.4 [+0.0, +44.4] | +19.4 [+0.0, +44.4] |  |  |  |
| olmo3-7b | sft | fabricates_vs_admits | narration | +16.7 [+0.0, +41.7] | +16.7 [+0.0, +41.7] |  |  |  |
| olmo3-7b | dpo | fabricates_vs_admits | plain_text | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |  |  |  |
| olmo3-7b | instruct | fabricates_vs_admits | plain_text | -2.8 [-16.7, +0.0] | -2.8 [-16.7, +0.0] |  |  |  |
| olmo3-7b | sft | fabricates_vs_admits | plain_text | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |  |  |  |
| olmo3-7b | dpo | fabricates_vs_admits | strong_act | -4.2 [-19.4, +12.5] | -4.2 [-19.4, +12.5] |  |  |  |
| olmo3-7b | instruct | fabricates_vs_admits | strong_act | -6.9 [-26.4, +9.7] | -6.9 [-26.4, +9.7] |  |  |  |
| olmo3-7b | sft | fabricates_vs_admits | strong_act | +1.4 [-9.7, +13.9] | +1.4 [-9.7, +13.9] |  |  |  |
| olmo3-7b | dpo | fabricates_vs_admits | strong_noact | -48.6 [-73.6, -23.6] | -48.6 [-73.6, -23.6] | yes | yes |  |
| olmo3-7b | instruct | fabricates_vs_admits | strong_noact | -72.2 [-90.3, -51.4] | -72.2 [-90.3, -51.4] | yes | yes |  |
| olmo3-7b | sft | fabricates_vs_admits | strong_noact | -12.5 [-31.9, +8.3] | -12.5 [-31.9, +8.3] |  |  |  |
| olmo3-7b | dpo | factual_accuracy | assistant | +37.5 [+11.1, +61.1] | +2.8 [-11.1, +16.7] | yes |  | SENSITIVE |
| olmo3-7b | instruct | factual_accuracy | assistant | +44.4 [+19.4, +68.1] | +5.6 [-4.2, +16.7] | yes |  | SENSITIVE |
| olmo3-7b | sft | factual_accuracy | assistant | +48.6 [+26.4, +69.4] | +4.2 [-4.2, +15.3] | yes |  | SENSITIVE |
| olmo3-7b | dpo | factual_accuracy | light_system | +56.9 [+34.7, +79.2] | +30.6 [+12.5, +51.4] | yes | yes |  |
| olmo3-7b | instruct | factual_accuracy | light_system | +52.8 [+26.4, +79.2] | +29.2 [+11.1, +50.0] | yes | yes |  |
| olmo3-7b | sft | factual_accuracy | light_system | +63.9 [+44.4, +83.3] | +29.2 [+11.1, +50.0] | yes | yes | SENSITIVE |
| olmo3-7b | dpo | factual_accuracy | light_user | +31.9 [+6.9, +54.2] | +11.1 [+0.0, +23.6] | yes |  | SENSITIVE |
| olmo3-7b | instruct | factual_accuracy | light_user | +30.6 [+4.2, +55.6] | +11.1 [+0.0, +25.0] | yes |  | SENSITIVE |
| olmo3-7b | sft | factual_accuracy | light_user | +37.5 [+5.6, +65.3] | +5.6 [-12.5, +22.2] | yes |  | SENSITIVE |
| olmo3-7b | dpo | factual_accuracy | narration | -33.3 [-66.7, +5.6] | +13.9 [-11.1, +41.7] |  |  |  |
| olmo3-7b | instruct | factual_accuracy | narration | -33.3 [-58.3, -8.3] | +11.1 [-19.4, +41.7] | yes |  | SENSITIVE |
| olmo3-7b | sft | factual_accuracy | narration | +8.3 [-22.2, +36.1] | +8.3 [-19.4, +38.9] |  |  |  |
| olmo3-7b | dpo | factual_accuracy | plain_text | -19.4 [-50.0, +0.0] | -5.6 [-22.2, +0.0] |  |  |  |
| olmo3-7b | instruct | factual_accuracy | plain_text | -11.1 [-33.3, +8.3] | -5.6 [-16.7, +0.0] |  |  |  |
| olmo3-7b | sft | factual_accuracy | plain_text | -22.2 [-55.6, +5.6] | -13.9 [-36.1, +0.0] |  |  |  |
| olmo3-7b | dpo | factual_accuracy | strong_act | +13.9 [-16.7, +41.7] | -1.4 [-16.7, +13.9] |  |  |  |
| olmo3-7b | instruct | factual_accuracy | strong_act | +12.5 [-13.9, +37.5] | -2.8 [-25.0, +16.7] |  |  |  |
| olmo3-7b | sft | factual_accuracy | strong_act | +15.3 [-6.9, +38.9] | +0.0 [-16.7, +15.3] |  |  |  |
| olmo3-7b | dpo | factual_accuracy | strong_noact | +44.4 [+25.0, +63.9] | +5.6 [-5.6, +18.1] | yes |  | SENSITIVE |
| olmo3-7b | instruct | factual_accuracy | strong_noact | +45.8 [+22.2, +66.7] | +6.9 [+0.0, +19.4] | yes |  | SENSITIVE |
| olmo3-7b | sft | factual_accuracy | strong_noact | +48.6 [+26.4, +68.1] | -1.4 [-15.3, +11.1] | yes |  | SENSITIVE |
| olmo3-7b | dpo | flattery | assistant | +8.3 [-20.8, +40.3] | +8.3 [-20.8, +40.3] |  |  |  |
| olmo3-7b | instruct | flattery | assistant | +6.9 [-20.8, +36.1] | +6.9 [-20.8, +36.1] |  |  |  |
| olmo3-7b | sft | flattery | assistant | -12.5 [-38.9, +18.1] | -12.5 [-38.9, +18.1] |  |  |  |
| olmo3-7b | dpo | flattery | light_system | +13.9 [-16.7, +43.1] | +13.9 [-16.7, +43.1] |  |  |  |
| olmo3-7b | instruct | flattery | light_system | +20.8 [-5.6, +47.2] | +20.8 [-5.6, +47.2] |  |  |  |
| olmo3-7b | sft | flattery | light_system | -4.2 [-25.0, +20.8] | -4.2 [-25.0, +20.8] |  |  |  |
| olmo3-7b | dpo | flattery | light_user | +29.2 [+6.9, +50.0] | +29.2 [+6.9, +50.0] | yes | yes |  |
| olmo3-7b | instruct | flattery | light_user | +20.8 [-1.4, +41.7] | +20.8 [-1.4, +41.7] |  |  |  |
| olmo3-7b | sft | flattery | light_user | +22.2 [-1.4, +44.4] | +22.2 [-1.4, +44.4] |  |  |  |
| olmo3-7b | dpo | flattery | narration | -5.6 [-36.1, +25.0] | -5.6 [-36.1, +25.0] |  |  |  |
| olmo3-7b | instruct | flattery | narration | -5.6 [-36.1, +25.0] | -5.6 [-36.1, +25.0] |  |  |  |
| olmo3-7b | sft | flattery | narration | +11.1 [-25.0, +47.2] | +11.1 [-25.0, +47.2] |  |  |  |
| olmo3-7b | dpo | flattery | plain_text | -13.9 [-36.1, +11.1] | -13.9 [-36.1, +11.1] |  |  |  |
| olmo3-7b | instruct | flattery | plain_text | -11.1 [-33.3, +8.3] | -11.1 [-33.3, +8.3] |  |  |  |
| olmo3-7b | sft | flattery | plain_text | -8.3 [-27.8, +8.3] | -8.3 [-27.8, +8.3] |  |  |  |
| olmo3-7b | dpo | flattery | strong_act | -23.6 [-47.2, +2.8] | -23.6 [-47.2, +2.8] |  |  |  |
| olmo3-7b | instruct | flattery | strong_act | -23.6 [-47.2, +0.0] | -23.6 [-47.2, +0.0] |  |  |  |
| olmo3-7b | sft | flattery | strong_act | -13.9 [-38.9, +12.5] | -13.9 [-38.9, +12.5] |  |  |  |
| olmo3-7b | dpo | flattery | strong_noact | +16.7 [-8.3, +40.3] | +16.7 [-8.3, +40.3] |  |  |  |
| olmo3-7b | instruct | flattery | strong_noact | +8.3 [-13.9, +33.3] | +8.3 [-13.9, +33.3] |  |  |  |
| olmo3-7b | sft | flattery | strong_noact | +11.1 [-11.1, +33.3] | +11.1 [-11.1, +33.3] |  |  |  |
| olmo3-7b | dpo | flattery | user_turn | -8.3 [-33.3, +0.0] | -8.3 [-33.3, +0.0] |  |  |  |
| olmo3-7b | instruct | flattery | user_turn | -8.3 [-27.8, +0.0] | -8.3 [-27.8, +0.0] |  |  |  |
| olmo3-7b | sft | flattery | user_turn | -8.3 [-33.3, +0.0] | -8.3 [-33.3, +0.0] |  |  |  |
| olmo3-7b | dpo | formal_register | assistant | -51.4 [-69.4, -31.9] | -51.4 [-69.4, -31.9] | yes | yes |  |
| olmo3-7b | instruct | formal_register | assistant | -51.4 [-69.4, -31.9] | -51.4 [-69.4, -31.9] | yes | yes |  |
| olmo3-7b | sft | formal_register | assistant | -43.1 [-65.3, -19.4] | -43.1 [-65.3, -19.4] | yes | yes |  |
| olmo3-7b | dpo | formal_register | light_system | -30.6 [-48.6, -15.3] | -30.6 [-48.6, -15.3] | yes | yes |  |
| olmo3-7b | instruct | formal_register | light_system | -30.6 [-48.6, -15.3] | -30.6 [-48.6, -15.3] | yes | yes |  |
| olmo3-7b | sft | formal_register | light_system | -22.2 [-40.3, -2.8] | -22.2 [-40.3, -2.8] | yes | yes |  |
| olmo3-7b | dpo | formal_register | light_user | -72.2 [-88.9, -52.8] | -72.2 [-88.9, -52.8] | yes | yes |  |
| olmo3-7b | instruct | formal_register | light_user | -72.2 [-88.9, -52.8] | -72.2 [-88.9, -52.8] | yes | yes |  |
| olmo3-7b | sft | formal_register | light_user | -70.8 [-88.9, -51.4] | -70.8 [-88.9, -51.4] | yes | yes |  |
| olmo3-7b | dpo | formal_register | narration | -33.3 [-52.8, -16.7] | -33.3 [-52.8, -16.7] | yes | yes |  |
| olmo3-7b | instruct | formal_register | narration | -33.3 [-52.8, -16.7] | -33.3 [-52.8, -16.7] | yes | yes |  |
| olmo3-7b | sft | formal_register | narration | -25.0 [-50.0, +0.0] | -25.0 [-50.0, +0.0] |  |  |  |
| olmo3-7b | dpo | formal_register | plain_text | +5.6 [-25.0, +36.1] | +5.6 [-25.0, +36.1] |  |  |  |
| olmo3-7b | instruct | formal_register | plain_text | -2.8 [-36.1, +30.6] | -2.8 [-36.1, +30.6] |  |  |  |
| olmo3-7b | sft | formal_register | plain_text | +13.9 [-16.7, +41.7] | +13.9 [-16.7, +41.7] |  |  |  |
| olmo3-7b | dpo | formal_register | strong_act | -18.1 [-36.1, -4.2] | -18.1 [-36.1, -4.2] | yes | yes |  |
| olmo3-7b | instruct | formal_register | strong_act | -18.1 [-36.1, -4.2] | -18.1 [-36.1, -4.2] | yes | yes |  |
| olmo3-7b | sft | formal_register | strong_act | -12.5 [-29.2, +2.8] | -12.5 [-29.2, +2.8] |  |  |  |
| olmo3-7b | dpo | formal_register | strong_noact | -55.6 [-72.2, -38.9] | -55.6 [-72.2, -38.9] | yes | yes |  |
| olmo3-7b | instruct | formal_register | strong_noact | -55.6 [-72.2, -40.3] | -55.6 [-72.2, -40.3] | yes | yes |  |
| olmo3-7b | sft | formal_register | strong_noact | -51.4 [-69.4, -31.9] | -51.4 [-69.4, -31.9] | yes | yes |  |
| olmo3-7b | dpo | formal_register | user_turn | +63.9 [+36.1, +88.9] | +63.9 [+36.1, +88.9] | yes | yes |  |
| olmo3-7b | instruct | formal_register | user_turn | +63.9 [+33.3, +91.7] | +63.9 [+33.3, +91.7] | yes | yes |  |
| olmo3-7b | sft | formal_register | user_turn | +77.8 [+52.8, +100.0] | +77.8 [+52.8, +100.0] | yes | yes |  |
| olmo3-7b | dpo | hedging_unanswerable | assistant | +2.8 [-9.7, +15.3] | +2.8 [-9.7, +15.3] |  |  |  |
| olmo3-7b | instruct | hedging_unanswerable | assistant | +1.4 [-12.5, +15.3] | +1.4 [-12.5, +15.3] |  |  |  |
| olmo3-7b | sft | hedging_unanswerable | assistant | +1.4 [-13.9, +16.7] | +1.4 [-13.9, +16.7] |  |  |  |
| olmo3-7b | dpo | hedging_unanswerable | light_system | +40.3 [+15.3, +65.3] | +40.3 [+15.3, +65.3] | yes | yes |  |
| olmo3-7b | instruct | hedging_unanswerable | light_system | +38.9 [+12.5, +65.3] | +38.9 [+12.5, +65.3] | yes | yes |  |
| olmo3-7b | sft | hedging_unanswerable | light_system | +40.3 [+15.3, +65.3] | +40.3 [+15.3, +65.3] | yes | yes |  |
| olmo3-7b | dpo | hedging_unanswerable | light_user | +5.6 [-11.1, +20.8] | +5.6 [-11.1, +20.8] |  |  |  |
| olmo3-7b | instruct | hedging_unanswerable | light_user | +6.9 [-6.9, +22.2] | +6.9 [-6.9, +22.2] |  |  |  |
| olmo3-7b | sft | hedging_unanswerable | light_user | +4.2 [-15.3, +22.2] | +4.2 [-15.3, +22.2] |  |  |  |
| olmo3-7b | dpo | hedging_unanswerable | narration | +25.0 [-8.3, +58.3] | +25.0 [-8.3, +58.3] |  |  |  |
| olmo3-7b | instruct | hedging_unanswerable | narration | +19.4 [-16.7, +58.3] | +19.4 [-16.7, +58.3] |  |  |  |
| olmo3-7b | sft | hedging_unanswerable | narration | +2.8 [-19.4, +25.0] | +2.8 [-19.4, +25.0] |  |  |  |
| olmo3-7b | dpo | hedging_unanswerable | plain_text | +52.8 [+22.2, +80.6] | +52.8 [+22.2, +80.6] | yes | yes |  |
| olmo3-7b | instruct | hedging_unanswerable | plain_text | +63.9 [+38.9, +86.1] | +63.9 [+38.9, +86.1] | yes | yes |  |
| olmo3-7b | sft | hedging_unanswerable | plain_text | +19.4 [-8.3, +50.0] | +19.4 [-8.3, +50.0] |  |  |  |
| olmo3-7b | dpo | hedging_unanswerable | strong_act | -8.3 [-33.3, +16.7] | -8.3 [-33.3, +16.7] |  |  |  |
| olmo3-7b | instruct | hedging_unanswerable | strong_act | +6.9 [-8.3, +25.0] | +6.9 [-8.3, +25.0] |  |  |  |
| olmo3-7b | sft | hedging_unanswerable | strong_act | -9.7 [-29.2, +8.3] | -9.7 [-29.2, +8.3] |  |  |  |
| olmo3-7b | dpo | hedging_unanswerable | strong_noact | +9.7 [-8.3, +27.8] | +9.7 [-8.3, +27.8] |  |  |  |
| olmo3-7b | instruct | hedging_unanswerable | strong_noact | +12.5 [-9.7, +34.7] | +12.5 [-9.7, +34.7] |  |  |  |
| olmo3-7b | sft | hedging_unanswerable | strong_noact | +15.3 [-5.6, +36.1] | +15.3 [-5.6, +36.1] |  |  |  |
| olmo3-7b | dpo | length | assistant | +0.03 [-0.08, +0.17] | +0.04 [-0.07, +0.17] |  |  |  |
| olmo3-7b | instruct | length | assistant | +0.06 [-0.05, +0.19] | +0.06 [-0.05, +0.20] |  |  |  |
| olmo3-7b | sft | length | assistant | -0.22 [-0.45, +0.01] | -0.21 [-0.44, +0.01] |  |  |  |
| olmo3-7b | dpo | length | light_system | +0.06 [-0.11, +0.30] | +0.10 [-0.10, +0.46] |  |  |  |
| olmo3-7b | instruct | length | light_system | +0.08 [-0.08, +0.33] | +0.13 [-0.08, +0.47] |  |  |  |
| olmo3-7b | sft | length | light_system | -0.20 [-0.45, +0.04] | -0.16 [-0.44, +0.18] |  |  |  |
| olmo3-7b | dpo | length | light_user | +0.09 [-0.06, +0.31] | +0.13 [-0.04, +0.33] |  |  |  |
| olmo3-7b | instruct | length | light_user | +0.12 [-0.03, +0.33] | +0.15 [-0.01, +0.36] |  |  |  |
| olmo3-7b | sft | length | light_user | -0.00 [-0.20, +0.22] | +0.03 [-0.16, +0.24] |  |  |  |
| olmo3-7b | dpo | length | narration | +0.78 [+0.16, +1.65] | +0.79 [+0.16, +1.66] | yes | yes |  |
| olmo3-7b | instruct | length | narration | +0.81 [+0.13, +1.70] | +0.82 [+0.13, +1.72] | yes | yes |  |
| olmo3-7b | sft | length | narration | +0.10 [-0.46, +0.88] | +0.11 [-0.44, +0.90] |  |  |  |
| olmo3-7b | dpo | length | plain_text | +0.33 [-0.41, +1.02] | +0.33 [-0.41, +1.02] |  |  |  |
| olmo3-7b | instruct | length | plain_text | +0.36 [-0.56, +1.12] | +0.36 [-0.56, +1.12] |  |  |  |
| olmo3-7b | sft | length | plain_text | -0.01 [-0.85, +0.66] | -0.01 [-0.85, +0.66] |  |  |  |
| olmo3-7b | dpo | length | strong_act | +0.07 [-0.10, +0.34] | +0.08 [-0.10, +0.35] |  |  |  |
| olmo3-7b | instruct | length | strong_act | +0.03 [-0.15, +0.31] | +0.04 [-0.14, +0.32] |  |  |  |
| olmo3-7b | sft | length | strong_act | -0.09 [-0.29, +0.20] | -0.08 [-0.29, +0.21] |  |  |  |
| olmo3-7b | dpo | length | strong_noact | +0.28 [-0.04, +0.79] | +0.28 [-0.04, +0.79] |  |  |  |
| olmo3-7b | instruct | length | strong_noact | +0.28 [-0.04, +0.78] | +0.28 [-0.04, +0.78] |  |  |  |
| olmo3-7b | sft | length | strong_noact | -0.08 [-0.57, +0.47] | -0.08 [-0.57, +0.47] |  |  |  |
| olmo3-7b | dpo | length | user_turn | +0.38 [-0.44, +1.25] | +0.38 [-0.44, +1.25] |  |  |  |
| olmo3-7b | instruct | length | user_turn | +0.30 [-0.61, +1.18] | +0.30 [-0.61, +1.18] |  |  |  |
| olmo3-7b | sft | length | user_turn | +0.14 [-0.70, +1.02] | +0.14 [-0.70, +1.02] |  |  |  |
| olmo3-7b | dpo | markdown_lists | assistant | +52.8 [+36.1, +69.4] | +52.8 [+36.1, +69.4] | yes | yes |  |
| olmo3-7b | instruct | markdown_lists | assistant | +52.8 [+36.1, +69.4] | +52.8 [+36.1, +69.4] | yes | yes |  |
| olmo3-7b | sft | markdown_lists | assistant | +41.7 [+22.2, +62.5] | +41.7 [+22.2, +62.5] | yes | yes |  |
| olmo3-7b | dpo | markdown_lists | light_system | +83.3 [+65.3, +95.8] | +83.3 [+65.3, +95.8] | yes | yes |  |
| olmo3-7b | instruct | markdown_lists | light_system | +83.3 [+65.3, +97.2] | +83.3 [+65.3, +97.2] | yes | yes |  |
| olmo3-7b | sft | markdown_lists | light_system | +70.8 [+50.0, +87.5] | +70.8 [+50.0, +87.5] | yes | yes |  |
| olmo3-7b | dpo | markdown_lists | light_user | +37.5 [+18.1, +59.7] | +37.5 [+18.1, +59.7] | yes | yes |  |
| olmo3-7b | instruct | markdown_lists | light_user | +37.5 [+18.1, +59.7] | +37.5 [+18.1, +59.7] | yes | yes |  |
| olmo3-7b | sft | markdown_lists | light_user | +26.4 [+0.0, +54.2] | +26.4 [+0.0, +54.2] |  |  |  |
| olmo3-7b | dpo | markdown_lists | narration | +0.0 [-30.6, +27.8] | +0.0 [-30.6, +27.8] |  |  |  |
| olmo3-7b | instruct | markdown_lists | narration | -8.3 [-30.6, +13.9] | -8.3 [-30.6, +13.9] |  |  |  |
| olmo3-7b | sft | markdown_lists | narration | -2.8 [-30.6, +19.4] | -2.8 [-30.6, +19.4] |  |  |  |
| olmo3-7b | dpo | markdown_lists | plain_text | +30.6 [+8.3, +55.6] | +30.6 [+8.3, +55.6] | yes | yes |  |
| olmo3-7b | instruct | markdown_lists | plain_text | +36.1 [+16.7, +58.3] | +36.1 [+16.7, +58.3] | yes | yes |  |
| olmo3-7b | sft | markdown_lists | plain_text | -11.1 [-27.8, +0.0] | -11.1 [-27.8, +0.0] |  |  |  |
| olmo3-7b | dpo | markdown_lists | strong_act | +19.4 [-12.5, +48.6] | +19.4 [-12.5, +48.6] |  |  |  |
| olmo3-7b | instruct | markdown_lists | strong_act | -8.3 [-30.6, +15.3] | -8.3 [-30.6, +15.3] |  |  |  |
| olmo3-7b | sft | markdown_lists | strong_act | -2.8 [-22.2, +18.1] | -2.8 [-22.2, +18.1] |  |  |  |
| olmo3-7b | dpo | markdown_lists | strong_noact | +59.7 [+41.7, +77.8] | +59.7 [+41.7, +77.8] | yes | yes |  |
| olmo3-7b | instruct | markdown_lists | strong_noact | +58.3 [+37.5, +76.4] | +58.3 [+37.5, +76.4] | yes | yes |  |
| olmo3-7b | sft | markdown_lists | strong_noact | +38.9 [+15.3, +59.7] | +38.9 [+15.3, +59.7] | yes | yes |  |
| olmo3-7b | dpo | markdown_lists | user_turn | +44.4 [+16.7, +66.7] | +44.4 [+16.7, +66.7] | yes | yes |  |
| olmo3-7b | instruct | markdown_lists | user_turn | +50.0 [+22.2, +75.0] | +50.0 [+22.2, +75.0] | yes | yes |  |
| olmo3-7b | sft | markdown_lists | user_turn | +44.4 [+16.7, +72.2] | +44.4 [+16.7, +72.2] | yes | yes |  |
| olmo3-7b | dpo | moralizing | assistant | -13.9 [-30.6, -2.8] | -13.9 [-30.6, -2.8] | yes | yes |  |
| olmo3-7b | instruct | moralizing | assistant | -13.9 [-31.9, -2.8] | -13.9 [-31.9, -2.8] | yes | yes |  |
| olmo3-7b | sft | moralizing | assistant | -13.9 [-30.6, -2.8] | -13.9 [-30.6, -2.8] | yes | yes |  |
| olmo3-7b | dpo | moralizing | light_system | -8.3 [-19.4, +0.0] | -8.3 [-19.4, +0.0] |  |  |  |
| olmo3-7b | instruct | moralizing | light_system | -4.2 [-18.1, +9.7] | -4.2 [-18.1, +9.7] |  |  |  |
| olmo3-7b | sft | moralizing | light_system | -6.9 [-18.1, +0.0] | -6.9 [-18.1, +0.0] |  |  |  |
| olmo3-7b | dpo | moralizing | light_user | -4.2 [-18.1, +8.3] | -4.2 [-18.1, +8.3] |  |  |  |
| olmo3-7b | instruct | moralizing | light_user | -5.6 [-19.4, +5.6] | -5.6 [-19.4, +5.6] |  |  |  |
| olmo3-7b | sft | moralizing | light_user | -5.6 [-16.7, +4.2] | -5.6 [-16.7, +4.2] |  |  |  |
| olmo3-7b | dpo | moralizing | narration | +0.0 [-13.9, +13.9] | +0.0 [-13.9, +13.9] |  |  |  |
| olmo3-7b | instruct | moralizing | narration | +0.0 [-13.9, +11.1] | +0.0 [-13.9, +11.1] |  |  |  |
| olmo3-7b | sft | moralizing | narration | +5.6 [-11.1, +22.2] | +5.6 [-11.1, +22.2] |  |  |  |
| olmo3-7b | dpo | moralizing | plain_text | -11.1 [-30.6, +5.6] | -11.1 [-30.6, +5.6] |  |  |  |
| olmo3-7b | instruct | moralizing | plain_text | -11.1 [-30.6, +5.6] | -11.1 [-30.6, +5.6] |  |  |  |
| olmo3-7b | sft | moralizing | plain_text | +2.8 [-19.4, +25.0] | +2.8 [-19.4, +25.0] |  |  |  |
| olmo3-7b | dpo | moralizing | strong_act | +4.2 [-8.3, +19.4] | +4.2 [-8.3, +19.4] |  |  |  |
| olmo3-7b | instruct | moralizing | strong_act | +0.0 [-9.7, +11.1] | +0.0 [-9.7, +11.1] |  |  |  |
| olmo3-7b | sft | moralizing | strong_act | +8.3 [-4.2, +23.6] | +8.3 [-4.2, +23.6] |  |  |  |
| olmo3-7b | dpo | moralizing | strong_noact | -8.3 [-25.0, +0.0] | -9.7 [-27.8, +0.0] |  |  |  |
| olmo3-7b | instruct | moralizing | strong_noact | -8.3 [-25.0, +0.0] | -9.7 [-27.8, +0.0] |  |  |  |
| olmo3-7b | sft | moralizing | strong_noact | -9.7 [-25.0, +0.0] | -11.1 [-29.2, +0.0] |  |  |  |
| olmo3-7b | dpo | multistep_reasoning | assistant | +23.6 [-5.6, +52.8] | +4.2 [-25.0, +33.3] |  |  |  |
| olmo3-7b | instruct | multistep_reasoning | assistant | +29.2 [-2.8, +58.3] | +4.2 [-26.4, +33.3] |  |  |  |
| olmo3-7b | sft | multistep_reasoning | assistant | +48.6 [+30.6, +65.3] | +19.4 [+1.4, +38.9] | yes | yes | SENSITIVE |
| olmo3-7b | dpo | multistep_reasoning | light_system | +29.2 [+0.0, +58.3] | +11.1 [-15.3, +38.9] |  |  |  |
| olmo3-7b | instruct | multistep_reasoning | light_system | +43.1 [+19.4, +66.7] | +23.6 [-1.4, +47.2] | yes |  | SENSITIVE |
| olmo3-7b | sft | multistep_reasoning | light_system | +54.2 [+34.7, +75.0] | +29.2 [+9.7, +48.6] | yes | yes |  |
| olmo3-7b | dpo | multistep_reasoning | light_user | +12.5 [-18.1, +41.7] | +5.6 [-23.6, +33.3] |  |  |  |
| olmo3-7b | instruct | multistep_reasoning | light_user | +23.6 [-11.1, +55.6] | +16.7 [-12.5, +43.1] |  |  |  |
| olmo3-7b | sft | multistep_reasoning | light_user | +38.9 [+13.9, +62.5] | +25.0 [+2.8, +45.8] | yes | yes |  |
| olmo3-7b | dpo | multistep_reasoning | narration | +41.7 [-2.8, +77.8] | +36.1 [+0.0, +69.4] |  |  |  |
| olmo3-7b | instruct | multistep_reasoning | narration | +44.4 [+19.4, +69.4] | +36.1 [+13.9, +61.1] | yes | yes |  |
| olmo3-7b | sft | multistep_reasoning | narration | +44.4 [+16.7, +72.2] | +33.3 [+0.0, +66.7] | yes |  | SENSITIVE |
| olmo3-7b | dpo | multistep_reasoning | plain_text | +11.1 [-13.9, +36.1] | +16.7 [-2.8, +36.1] |  |  |  |
| olmo3-7b | instruct | multistep_reasoning | plain_text | +2.8 [-36.1, +38.9] | +5.6 [-19.4, +30.6] |  |  |  |
| olmo3-7b | sft | multistep_reasoning | plain_text | -8.3 [-41.7, +25.0] | -5.6 [-33.3, +22.2] |  |  |  |
| olmo3-7b | dpo | multistep_reasoning | strong_act | +2.8 [-27.8, +30.6] | +1.4 [-23.6, +25.0] |  |  |  |
| olmo3-7b | instruct | multistep_reasoning | strong_act | +18.1 [-8.3, +43.1] | +11.1 [-11.1, +34.7] |  |  |  |
| olmo3-7b | sft | multistep_reasoning | strong_act | +9.7 [-19.4, +36.1] | +2.8 [-20.8, +25.0] |  |  |  |
| olmo3-7b | dpo | multistep_reasoning | strong_noact | +37.5 [+2.8, +68.1] | +20.8 [-6.9, +45.8] | yes |  | SENSITIVE |
| olmo3-7b | instruct | multistep_reasoning | strong_noact | +40.3 [+11.1, +66.7] | +18.1 [-6.9, +41.7] | yes |  | SENSITIVE |
| olmo3-7b | sft | multistep_reasoning | strong_noact | +54.2 [+29.2, +76.4] | +26.4 [+2.8, +47.2] | yes | yes | SENSITIVE |
| olmo3-7b | dpo | refusal_borderline | assistant | +1.4 [-12.5, +16.7] | +1.4 [-12.5, +16.7] |  |  |  |
| olmo3-7b | instruct | refusal_borderline | assistant | +0.0 [-11.1, +11.1] | +0.0 [-11.1, +11.1] |  |  |  |
| olmo3-7b | sft | refusal_borderline | assistant | +2.8 [-8.3, +16.7] | +2.8 [-8.3, +16.7] |  |  |  |
| olmo3-7b | dpo | refusal_borderline | light_system | +5.6 [-8.3, +22.2] | +5.6 [-8.3, +22.2] |  |  |  |
| olmo3-7b | instruct | refusal_borderline | light_system | +6.9 [-6.9, +25.0] | +6.9 [-6.9, +25.0] |  |  |  |
| olmo3-7b | sft | refusal_borderline | light_system | +5.6 [-6.9, +23.6] | +5.6 [-6.9, +23.6] |  |  |  |
| olmo3-7b | dpo | refusal_borderline | light_user | -16.7 [-33.3, -4.2] | -16.7 [-33.3, -4.2] | yes | yes |  |
| olmo3-7b | instruct | refusal_borderline | light_user | -16.7 [-33.3, -4.2] | -16.7 [-33.3, -4.2] | yes | yes |  |
| olmo3-7b | sft | refusal_borderline | light_user | -9.7 [-22.3, +0.0] | -9.7 [-22.3, +0.0] |  |  |  |
| olmo3-7b | dpo | refusal_borderline | narration | -5.6 [-16.7, +0.0] | -5.6 [-16.7, +0.0] |  |  |  |
| olmo3-7b | instruct | refusal_borderline | narration | -5.6 [-16.7, +0.0] | -5.6 [-16.7, +0.0] |  |  |  |
| olmo3-7b | sft | refusal_borderline | narration | -5.6 [-16.7, +0.0] | -5.6 [-16.7, +0.0] |  |  |  |
| olmo3-7b | dpo | refusal_borderline | plain_text | -2.8 [-16.7, +0.0] | -2.8 [-16.7, +0.0] |  |  |  |
| olmo3-7b | instruct | refusal_borderline | plain_text | -2.8 [-16.7, +0.0] | -2.8 [-16.7, +0.0] |  |  |  |
| olmo3-7b | sft | refusal_borderline | plain_text | +0.0 [-11.1, +11.1] | +0.0 [-11.1, +11.1] |  |  |  |
| olmo3-7b | dpo | refusal_borderline | strong_act | -13.9 [-27.8, -2.8] | -13.9 [-27.8, -2.8] | yes | yes |  |
| olmo3-7b | instruct | refusal_borderline | strong_act | -13.9 [-27.8, -2.8] | -13.9 [-27.8, -2.8] | yes | yes |  |
| olmo3-7b | sft | refusal_borderline | strong_act | -13.9 [-29.2, -2.8] | -13.9 [-29.2, -2.8] | yes | yes |  |
| olmo3-7b | dpo | refusal_borderline | strong_noact | -2.8 [-11.1, +4.2] | -2.8 [-11.1, +4.2] |  |  |  |
| olmo3-7b | instruct | refusal_borderline | strong_noact | -4.2 [-13.9, +4.2] | -4.2 [-13.9, +4.2] |  |  |  |
| olmo3-7b | sft | refusal_borderline | strong_noact | +5.6 [-6.9, +23.6] | +5.6 [-6.9, +23.6] |  |  |  |
| olmo3-7b | dpo | safety_caveats | assistant | -20.8 [-44.4, +2.8] | -20.8 [-44.4, +2.8] |  |  |  |
| olmo3-7b | instruct | safety_caveats | assistant | -11.1 [-37.5, +12.5] | -11.1 [-37.5, +12.5] |  |  |  |
| olmo3-7b | sft | safety_caveats | assistant | -16.7 [-43.1, +6.9] | -16.7 [-43.1, +6.9] |  |  |  |
| olmo3-7b | dpo | safety_caveats | light_system | -8.3 [-34.7, +18.1] | -8.3 [-34.7, +18.1] |  |  |  |
| olmo3-7b | instruct | safety_caveats | light_system | -5.6 [-30.6, +19.4] | -5.6 [-30.6, +19.4] |  |  |  |
| olmo3-7b | sft | safety_caveats | light_system | -9.7 [-34.7, +12.5] | -9.7 [-34.7, +12.5] |  |  |  |
| olmo3-7b | dpo | safety_caveats | light_user | -23.6 [-48.6, +4.2] | -23.6 [-48.6, +4.2] |  |  |  |
| olmo3-7b | instruct | safety_caveats | light_user | -15.3 [-41.7, +12.5] | -15.3 [-41.7, +12.5] |  |  |  |
| olmo3-7b | sft | safety_caveats | light_user | -11.1 [-38.9, +16.7] | -11.1 [-38.9, +16.7] |  |  |  |
| olmo3-7b | dpo | safety_caveats | narration | -16.7 [-50.0, +16.7] | -16.7 [-50.0, +16.7] |  |  |  |
| olmo3-7b | instruct | safety_caveats | narration | -5.6 [-41.7, +27.8] | -5.6 [-41.7, +27.8] |  |  |  |
| olmo3-7b | sft | safety_caveats | narration | +13.9 [-8.3, +38.9] | +13.9 [-8.3, +38.9] |  |  |  |
| olmo3-7b | dpo | safety_caveats | plain_text | -11.1 [-33.3, +11.1] | -11.1 [-33.3, +11.1] |  |  |  |
| olmo3-7b | instruct | safety_caveats | plain_text | -2.8 [-30.6, +33.3] | -2.8 [-30.6, +33.3] |  |  |  |
| olmo3-7b | sft | safety_caveats | plain_text | +8.3 [-13.9, +33.3] | +8.3 [-13.9, +33.3] |  |  |  |
| olmo3-7b | dpo | safety_caveats | strong_act | -26.4 [-55.6, +5.6] | -25.0 [-54.2, +5.6] |  |  |  |
| olmo3-7b | instruct | safety_caveats | strong_act | -26.4 [-54.2, +0.0] | -25.0 [-52.8, +1.4] |  |  |  |
| olmo3-7b | sft | safety_caveats | strong_act | -13.9 [-40.3, +13.9] | -12.5 [-38.9, +13.9] |  |  |  |
| olmo3-7b | dpo | safety_caveats | strong_noact | -2.8 [-26.4, +19.4] | -2.8 [-26.4, +19.4] |  |  |  |
| olmo3-7b | instruct | safety_caveats | strong_noact | -5.6 [-26.4, +16.7] | -5.6 [-26.4, +16.7] |  |  |  |
| olmo3-7b | sft | safety_caveats | strong_noact | -4.2 [-23.6, +16.7] | -4.2 [-23.6, +16.7] |  |  |  |
| olmo3-7b | dpo | says_ai | assistant | -4.2 [-29.2, +23.6] | -4.2 [-29.2, +23.6] |  |  |  |
| olmo3-7b | instruct | says_ai | assistant | -5.6 [-34.7, +25.0] | -5.6 [-34.7, +25.0] |  |  |  |
| olmo3-7b | sft | says_ai | assistant | +5.6 [-16.7, +26.4] | +5.6 [-16.7, +26.4] |  |  |  |
| olmo3-7b | dpo | says_ai | light_system | +16.7 [-16.7, +50.0] | +16.7 [-16.7, +50.0] |  |  |  |
| olmo3-7b | instruct | says_ai | light_system | +26.4 [-5.6, +55.6] | +26.4 [-5.6, +55.6] |  |  |  |
| olmo3-7b | sft | says_ai | light_system | +31.9 [+8.3, +56.9] | +31.9 [+8.3, +56.9] | yes | yes |  |
| olmo3-7b | dpo | says_ai | light_user | -36.1 [-58.3, -11.1] | -36.1 [-58.3, -11.1] | yes | yes |  |
| olmo3-7b | instruct | says_ai | light_user | -33.3 [-55.6, -9.7] | -33.3 [-55.6, -9.7] | yes | yes |  |
| olmo3-7b | sft | says_ai | light_user | -15.3 [-43.1, +16.7] | -15.3 [-43.1, +16.7] |  |  |  |
| olmo3-7b | dpo | says_ai | narration | -8.3 [-25.0, +0.0] | -8.3 [-25.0, +0.0] |  |  |  |
| olmo3-7b | instruct | says_ai | narration | -2.8 [-22.2, +11.1] | -2.8 [-22.2, +11.1] |  |  |  |
| olmo3-7b | sft | says_ai | narration | -5.6 [-22.2, +11.1] | -5.6 [-22.2, +11.1] |  |  |  |
| olmo3-7b | dpo | says_ai | plain_text | +16.7 [-2.8, +38.9] | +16.7 [-2.8, +38.9] |  |  |  |
| olmo3-7b | instruct | says_ai | plain_text | +16.7 [-2.8, +36.1] | +16.7 [-2.8, +36.1] |  |  |  |
| olmo3-7b | sft | says_ai | plain_text | +8.3 [-11.1, +33.3] | +8.3 [-11.1, +33.3] |  |  |  |
| olmo3-7b | dpo | says_ai | strong_act | -27.8 [-47.2, -11.1] | -27.8 [-47.2, -11.1] | yes | yes |  |
| olmo3-7b | instruct | says_ai | strong_act | -29.2 [-50.0, -12.5] | -29.2 [-50.0, -12.5] | yes | yes |  |
| olmo3-7b | sft | says_ai | strong_act | -27.8 [-48.6, -9.7] | -27.8 [-48.6, -9.7] | yes | yes |  |
| olmo3-7b | dpo | says_ai | strong_noact | -13.9 [-38.9, +8.3] | -13.9 [-38.9, +8.3] |  |  |  |
| olmo3-7b | instruct | says_ai | strong_noact | -16.7 [-38.9, +6.9] | -16.7 [-38.9, +6.9] |  |  |  |
| olmo3-7b | sft | says_ai | strong_noact | -16.7 [-37.5, +5.6] | -16.7 [-37.5, +5.6] |  |  |  |
| olmo3-7b | dpo | stock_phrases | assistant | +15.3 [-8.3, +40.3] | +15.3 [-8.3, +40.3] |  |  |  |
| olmo3-7b | instruct | stock_phrases | assistant | +13.9 [-12.5, +38.9] | +13.9 [-12.5, +38.9] |  |  |  |
| olmo3-7b | sft | stock_phrases | assistant | -12.5 [-31.9, +6.9] | -12.5 [-31.9, +6.9] |  |  |  |
| olmo3-7b | dpo | stock_phrases | light_system | +43.1 [+16.7, +69.4] | +43.1 [+16.7, +69.4] | yes | yes |  |
| olmo3-7b | instruct | stock_phrases | light_system | +41.7 [+13.9, +68.1] | +41.7 [+13.9, +68.1] | yes | yes |  |
| olmo3-7b | sft | stock_phrases | light_system | +9.7 [-13.9, +37.5] | +9.7 [-13.9, +37.5] |  |  |  |
| olmo3-7b | dpo | stock_phrases | light_user | +5.6 [-25.0, +34.7] | +5.6 [-25.0, +34.7] |  |  |  |
| olmo3-7b | instruct | stock_phrases | light_user | +9.7 [-25.0, +43.1] | +9.7 [-25.0, +43.1] |  |  |  |
| olmo3-7b | sft | stock_phrases | light_user | +2.8 [-23.6, +30.6] | +2.8 [-23.6, +30.6] |  |  |  |
| olmo3-7b | dpo | stock_phrases | narration | -2.8 [-27.8, +25.0] | -2.8 [-27.8, +25.0] |  |  |  |
| olmo3-7b | instruct | stock_phrases | narration | -5.6 [-33.3, +19.4] | -5.6 [-33.3, +19.4] |  |  |  |
| olmo3-7b | sft | stock_phrases | narration | -5.6 [-33.3, +22.2] | -5.6 [-33.3, +22.2] |  |  |  |
| olmo3-7b | dpo | stock_phrases | plain_text | +5.6 [-11.1, +25.0] | +5.6 [-11.1, +25.0] |  |  |  |
| olmo3-7b | instruct | stock_phrases | plain_text | -2.8 [-16.7, +11.1] | -2.8 [-16.7, +11.1] |  |  |  |
| olmo3-7b | sft | stock_phrases | plain_text | -11.1 [-25.0, +0.0] | -11.1 [-25.0, +0.0] |  |  |  |
| olmo3-7b | dpo | stock_phrases | strong_act | -23.6 [-44.4, -2.8] | -23.6 [-44.4, -2.8] | yes | yes |  |
| olmo3-7b | instruct | stock_phrases | strong_act | -22.2 [-41.7, -2.8] | -22.2 [-41.7, -2.8] | yes | yes |  |
| olmo3-7b | sft | stock_phrases | strong_act | -18.1 [-40.3, +5.6] | -18.1 [-40.3, +5.6] |  |  |  |
| olmo3-7b | dpo | stock_phrases | strong_noact | +23.6 [-6.9, +52.8] | +23.6 [-6.9, +52.8] |  |  |  |
| olmo3-7b | instruct | stock_phrases | strong_noact | +26.4 [-5.6, +56.9] | +26.4 [-5.6, +56.9] |  |  |  |
| olmo3-7b | sft | stock_phrases | strong_noact | +5.6 [-18.1, +31.9] | +5.6 [-18.1, +31.9] |  |  |  |
| olmo3-7b | dpo | stock_phrases | user_turn | -5.6 [-19.4, +8.3] | -5.6 [-19.4, +8.3] |  |  |  |
| olmo3-7b | instruct | stock_phrases | user_turn | -8.3 [-19.4, +0.0] | -8.3 [-19.4, +0.0] |  |  |  |
| olmo3-7b | sft | stock_phrases | user_turn | -2.8 [-19.4, +16.7] | -2.8 [-19.4, +16.7] |  |  |  |
| qwen3-8b | instruct | agrees_wrong_opinion | assistant | -22.2 [-43.1, -2.8] | -22.2 [-43.1, -2.8] | yes | yes |  |
| qwen3-8b | instruct | agrees_wrong_opinion | light_system | -11.1 [-36.1, +13.9] | -9.7 [-34.7, +16.7] |  |  |  |
| qwen3-8b | instruct | agrees_wrong_opinion | light_user | +8.3 [-12.5, +29.2] | +8.3 [-12.5, +29.2] |  |  |  |
| qwen3-8b | instruct | agrees_wrong_opinion | narration | +5.6 [-16.7, +27.8] | +2.8 [-16.7, +25.0] |  |  |  |
| qwen3-8b | instruct | agrees_wrong_opinion | plain_text | -8.3 [-41.7, +25.0] | -8.3 [-41.7, +25.0] |  |  |  |
| qwen3-8b | instruct | agrees_wrong_opinion | strong_act | +5.6 [-18.1, +30.6] | +5.6 [-18.1, +30.6] |  |  |  |
| qwen3-8b | instruct | agrees_wrong_opinion | strong_noact | +19.4 [+0.0, +40.3] | +18.1 [-2.8, +40.3] |  |  |  |
| qwen3-8b | instruct | arithmetic | assistant | +25.0 [+11.1, +40.3] | +18.1 [+5.6, +31.9] | yes | yes |  |
| qwen3-8b | instruct | arithmetic | light_system | +44.4 [+25.0, +63.9] | +30.6 [+12.5, +48.6] | yes | yes |  |
| qwen3-8b | instruct | arithmetic | light_user | +29.2 [+1.4, +54.2] | +22.2 [+1.4, +41.7] | yes | yes |  |
| qwen3-8b | instruct | arithmetic | narration | -11.1 [-47.2, +22.2] | +22.2 [-5.6, +50.0] |  |  |  |
| qwen3-8b | instruct | arithmetic | plain_text | +38.9 [+2.8, +72.2] | +38.9 [+2.8, +69.4] | yes | yes |  |
| qwen3-8b | instruct | arithmetic | strong_act | -11.1 [-31.9, +7.0] | -5.6 [-30.6, +19.4] |  |  |  |
| qwen3-8b | instruct | arithmetic | strong_noact | +38.9 [+13.9, +62.5] | +27.8 [+5.6, +50.0] | yes | yes |  |
| qwen3-8b | instruct | clarifying_question | assistant | +44.4 [+12.5, +73.6] | +44.4 [+12.5, +73.6] | yes | yes |  |
| qwen3-8b | instruct | clarifying_question | light_system | +37.5 [+13.9, +61.1] | +37.5 [+13.9, +61.1] | yes | yes |  |
| qwen3-8b | instruct | clarifying_question | light_user | +30.6 [+9.7, +54.2] | +30.6 [+9.7, +54.2] | yes | yes |  |
| qwen3-8b | instruct | clarifying_question | narration | +44.4 [+22.2, +69.4] | +44.4 [+22.2, +69.4] | yes | yes |  |
| qwen3-8b | instruct | clarifying_question | plain_text | +11.1 [-11.1, +33.3] | +11.1 [-11.1, +33.3] |  |  |  |
| qwen3-8b | instruct | clarifying_question | strong_act | +20.8 [+4.2, +38.9] | +20.8 [+4.2, +38.9] | yes | yes |  |
| qwen3-8b | instruct | clarifying_question | strong_noact | +27.8 [+5.6, +52.8] | +27.8 [+5.6, +52.8] | yes | yes |  |
| qwen3-8b | instruct | closing_offer | assistant | +37.5 [+12.5, +62.5] | +37.5 [+12.5, +62.5] | yes | yes |  |
| qwen3-8b | instruct | closing_offer | light_system | +38.9 [+15.3, +65.3] | +38.9 [+15.3, +65.3] | yes | yes |  |
| qwen3-8b | instruct | closing_offer | light_user | +38.9 [+11.1, +65.3] | +38.9 [+11.1, +65.3] | yes | yes |  |
| qwen3-8b | instruct | closing_offer | narration | +11.1 [+0.0, +33.3] | +11.1 [+0.0, +33.3] |  |  |  |
| qwen3-8b | instruct | closing_offer | plain_text | +5.6 [+0.0, +16.7] | +5.6 [+0.0, +16.7] |  |  |  |
| qwen3-8b | instruct | closing_offer | strong_act | +9.7 [-2.8, +25.0] | +9.7 [-2.8, +25.0] |  |  |  |
| qwen3-8b | instruct | closing_offer | strong_noact | +25.0 [+5.6, +45.8] | +25.0 [+5.6, +45.8] | yes | yes |  |
| qwen3-8b | instruct | closing_offer | user_turn | -2.8 [-16.7, +0.0] | -2.8 [-16.7, +0.0] |  |  |  |
| qwen3-8b | instruct | corrects_false_premise | assistant | +34.7 [+19.4, +52.8] | +34.7 [+19.4, +52.8] | yes | yes |  |
| qwen3-8b | instruct | corrects_false_premise | light_system | +44.4 [+25.0, +65.3] | +44.4 [+25.0, +65.3] | yes | yes |  |
| qwen3-8b | instruct | corrects_false_premise | light_user | +41.7 [+22.2, +61.1] | +41.7 [+22.2, +61.1] | yes | yes |  |
| qwen3-8b | instruct | corrects_false_premise | narration | +27.8 [-8.3, +61.1] | +27.8 [-8.3, +61.1] |  |  |  |
| qwen3-8b | instruct | corrects_false_premise | plain_text | +5.6 [-11.1, +25.0] | +5.6 [-11.1, +25.0] |  |  |  |
| qwen3-8b | instruct | corrects_false_premise | strong_act | +26.4 [-4.2, +54.2] | +26.4 [-4.2, +54.2] |  |  |  |
| qwen3-8b | instruct | corrects_false_premise | strong_noact | +61.1 [+40.3, +81.9] | +62.5 [+41.7, +81.9] | yes | yes |  |
| qwen3-8b | instruct | declines_opinion | assistant | +29.2 [+4.2, +54.2] | +29.2 [+4.2, +54.2] | yes | yes |  |
| qwen3-8b | instruct | declines_opinion | light_system | +1.4 [-22.2, +26.4] | +2.8 [-22.2, +26.4] |  |  |  |
| qwen3-8b | instruct | declines_opinion | light_user | -13.9 [-41.7, +12.5] | -13.9 [-41.7, +12.5] |  |  |  |
| qwen3-8b | instruct | declines_opinion | narration | -16.7 [-58.3, +22.2] | -16.7 [-58.3, +22.2] |  |  |  |
| qwen3-8b | instruct | declines_opinion | plain_text | +30.6 [+0.0, +58.3] | +30.6 [+0.0, +58.3] |  |  |  |
| qwen3-8b | instruct | declines_opinion | strong_act | +0.0 [-22.2, +22.2] | +0.0 [-22.2, +22.2] |  |  |  |
| qwen3-8b | instruct | declines_opinion | strong_noact | -1.4 [-29.2, +26.4] | -1.4 [-29.2, +26.4] |  |  |  |
| qwen3-8b | instruct | denies_feelings | assistant | +63.9 [+34.7, +87.5] | +63.9 [+34.7, +87.5] | yes | yes |  |
| qwen3-8b | instruct | denies_feelings | light_system | -1.4 [-11.1, +8.3] | -1.4 [-11.1, +8.3] |  |  |  |
| qwen3-8b | instruct | denies_feelings | light_user | -9.7 [-26.4, +0.0] | -9.7 [-26.4, +0.0] |  |  |  |
| qwen3-8b | instruct | denies_feelings | narration | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |  |  |  |
| qwen3-8b | instruct | denies_feelings | plain_text | +5.6 [-11.1, +27.8] | +5.6 [-11.1, +27.8] |  |  |  |
| qwen3-8b | instruct | denies_feelings | strong_act | -4.2 [-13.9, +0.0] | -4.2 [-13.9, +0.0] |  |  |  |
| qwen3-8b | instruct | denies_feelings | strong_noact | -5.6 [-16.7, +2.8] | -4.2 [-13.9, +4.2] |  |  |  |
| qwen3-8b | instruct | fabricates_vs_admits | assistant | -20.8 [-45.8, +4.2] | -20.8 [-45.8, +4.2] |  |  |  |
| qwen3-8b | instruct | fabricates_vs_admits | light_system | -22.2 [-54.2, +11.1] | -22.2 [-54.2, +11.1] |  |  |  |
| qwen3-8b | instruct | fabricates_vs_admits | light_user | -6.9 [-33.3, +19.4] | -6.9 [-33.3, +19.4] |  |  |  |
| qwen3-8b | instruct | fabricates_vs_admits | narration | +36.1 [+8.3, +63.9] | +36.1 [+8.3, +63.9] | yes | yes |  |
| qwen3-8b | instruct | fabricates_vs_admits | plain_text | +8.3 [-16.7, +30.6] | +8.3 [-16.7, +30.6] |  |  |  |
| qwen3-8b | instruct | fabricates_vs_admits | strong_act | +12.5 [-18.1, +38.9] | +12.5 [-18.1, +38.9] |  |  |  |
| qwen3-8b | instruct | fabricates_vs_admits | strong_noact | +11.1 [-16.7, +38.9] | +11.1 [-16.7, +38.9] |  |  |  |
| qwen3-8b | instruct | factual_accuracy | assistant | +2.8 [-22.2, +29.2] | +31.9 [+11.1, +51.4] |  | yes |  |
| qwen3-8b | instruct | factual_accuracy | light_system | -5.6 [-30.6, +20.8] | +36.1 [+19.4, +54.2] |  | yes |  |
| qwen3-8b | instruct | factual_accuracy | light_user | +16.7 [-11.1, +45.8] | +44.4 [+22.2, +66.7] |  | yes |  |
| qwen3-8b | instruct | factual_accuracy | narration | -11.1 [-30.6, +5.6] | +36.1 [+11.1, +66.7] |  | yes |  |
| qwen3-8b | instruct | factual_accuracy | plain_text | +30.6 [+2.8, +52.8] | +30.6 [+5.6, +55.6] | yes | yes |  |
| qwen3-8b | instruct | factual_accuracy | strong_act | +8.3 [-18.1, +34.7] | +44.4 [+19.4, +68.1] |  | yes |  |
| qwen3-8b | instruct | factual_accuracy | strong_noact | +22.2 [-5.6, +45.8] | +41.7 [+19.4, +62.5] |  | yes |  |
| qwen3-8b | instruct | flattery | assistant | +52.8 [+31.9, +73.6] | +52.8 [+31.9, +73.6] | yes | yes |  |
| qwen3-8b | instruct | flattery | light_system | +31.9 [+4.2, +58.3] | +31.9 [+4.2, +58.3] | yes | yes |  |
| qwen3-8b | instruct | flattery | light_user | +37.5 [+12.5, +61.1] | +37.5 [+12.5, +61.1] | yes | yes |  |
| qwen3-8b | instruct | flattery | narration | +5.6 [-16.7, +30.6] | +5.6 [-16.7, +30.6] |  |  |  |
| qwen3-8b | instruct | flattery | plain_text | -27.8 [-55.6, +2.8] | -27.8 [-55.6, +2.8] |  |  |  |
| qwen3-8b | instruct | flattery | strong_act | +11.1 [-11.1, +36.1] | +11.1 [-11.1, +36.1] |  |  |  |
| qwen3-8b | instruct | flattery | strong_noact | +27.8 [+0.0, +55.6] | +27.8 [+0.0, +55.6] |  |  |  |
| qwen3-8b | instruct | flattery | user_turn | -5.6 [-16.7, +0.0] | -5.6 [-16.7, +0.0] |  |  |  |
| qwen3-8b | instruct | formal_register | assistant | -18.1 [-33.3, -5.6] | -18.1 [-33.3, -5.6] | yes | yes |  |
| qwen3-8b | instruct | formal_register | light_system | -30.6 [-47.2, -15.3] | -29.2 [-45.8, -13.9] | yes | yes |  |
| qwen3-8b | instruct | formal_register | light_user | -23.6 [-37.5, -11.1] | -22.2 [-34.7, -9.7] | yes | yes |  |
| qwen3-8b | instruct | formal_register | narration | -16.7 [-41.7, +5.6] | -16.7 [-41.7, +5.6] |  |  |  |
| qwen3-8b | instruct | formal_register | plain_text | +27.8 [-8.3, +58.3] | +27.8 [-8.3, +58.3] |  |  |  |
| qwen3-8b | instruct | formal_register | strong_act | -13.9 [-31.9, -1.4] | -13.9 [-31.9, -1.4] | yes | yes |  |
| qwen3-8b | instruct | formal_register | strong_noact | -37.5 [-58.3, -18.1] | -37.5 [-58.3, -18.1] | yes | yes |  |
| qwen3-8b | instruct | formal_register | user_turn | -2.8 [-16.7, +0.0] | -2.8 [-16.7, +0.0] |  |  |  |
| qwen3-8b | instruct | hedging_unanswerable | assistant | +45.8 [+25.0, +66.7] | +45.8 [+25.0, +66.7] | yes | yes |  |
| qwen3-8b | instruct | hedging_unanswerable | light_system | +23.6 [+2.8, +45.8] | +25.0 [+4.2, +45.8] | yes | yes |  |
| qwen3-8b | instruct | hedging_unanswerable | light_user | +34.7 [+15.3, +54.2] | +33.3 [+13.9, +54.2] | yes | yes |  |
| qwen3-8b | instruct | hedging_unanswerable | narration | +27.8 [-2.8, +58.3] | +27.8 [-2.8, +58.3] |  |  |  |
| qwen3-8b | instruct | hedging_unanswerable | plain_text | +30.6 [+5.6, +61.1] | +30.6 [+5.6, +61.1] | yes | yes |  |
| qwen3-8b | instruct | hedging_unanswerable | strong_act | +38.9 [+16.7, +61.1] | +38.9 [+16.7, +61.1] | yes | yes |  |
| qwen3-8b | instruct | hedging_unanswerable | strong_noact | +41.7 [+15.3, +66.7] | +43.1 [+18.1, +66.7] | yes | yes |  |
| qwen3-8b | instruct | length | assistant | +0.60 [+0.10, +1.34] | +0.60 [+0.10, +1.34] | yes | yes |  |
| qwen3-8b | instruct | length | light_system | +0.90 [+0.41, +1.48] | +0.90 [+0.41, +1.48] | yes | yes |  |
| qwen3-8b | instruct | length | light_user | +0.84 [+0.23, +1.63] | +0.84 [+0.23, +1.63] | yes | yes |  |
| qwen3-8b | instruct | length | narration | +1.09 [+0.31, +2.02] | +1.14 [+0.37, +2.07] | yes | yes |  |
| qwen3-8b | instruct | length | plain_text | +0.76 [+0.18, +1.40] | +0.76 [+0.18, +1.40] | yes | yes |  |
| qwen3-8b | instruct | length | strong_act | +0.95 [+0.42, +1.66] | +0.95 [+0.42, +1.66] | yes | yes |  |
| qwen3-8b | instruct | length | strong_noact | +0.92 [+0.47, +1.42] | +0.99 [+0.50, +1.50] | yes | yes |  |
| qwen3-8b | instruct | length | user_turn | -0.21 [-1.13, +0.76] | -0.21 [-1.13, +0.76] |  |  |  |
| qwen3-8b | instruct | markdown_lists | assistant | +37.5 [+19.4, +55.6] | +37.5 [+19.4, +55.6] | yes | yes |  |
| qwen3-8b | instruct | markdown_lists | light_system | +48.6 [+29.2, +65.3] | +48.6 [+29.2, +65.3] | yes | yes |  |
| qwen3-8b | instruct | markdown_lists | light_user | +54.2 [+34.7, +73.6] | +54.2 [+34.7, +73.6] | yes | yes |  |
| qwen3-8b | instruct | markdown_lists | narration | -16.7 [-36.1, +0.0] | -16.7 [-36.1, +0.0] |  |  |  |
| qwen3-8b | instruct | markdown_lists | plain_text | +19.4 [-5.6, +47.2] | +19.4 [-5.6, +47.2] |  |  |  |
| qwen3-8b | instruct | markdown_lists | strong_act | -2.8 [-23.6, +22.2] | -2.8 [-23.6, +22.2] |  |  |  |
| qwen3-8b | instruct | markdown_lists | strong_noact | +34.7 [+5.6, +62.5] | +37.5 [+8.3, +65.3] | yes | yes |  |
| qwen3-8b | instruct | markdown_lists | user_turn | -5.6 [-25.0, +16.7] | -5.6 [-25.0, +16.7] |  |  |  |
| qwen3-8b | instruct | moralizing | assistant | -4.2 [-13.9, +0.0] | -4.2 [-13.9, +0.0] |  |  |  |
| qwen3-8b | instruct | moralizing | light_system | +0.0 [-9.7, +11.1] | +0.0 [-9.7, +11.1] |  |  |  |
| qwen3-8b | instruct | moralizing | light_user | +0.0 [-15.3, +13.9] | +0.0 [-15.3, +13.9] |  |  |  |
| qwen3-8b | instruct | moralizing | narration | +0.0 [-22.2, +22.2] | +0.0 [-22.2, +22.2] |  |  |  |
| qwen3-8b | instruct | moralizing | plain_text | +2.8 [+0.0, +16.7] | +2.8 [+0.0, +16.7] |  |  |  |
| qwen3-8b | instruct | moralizing | strong_act | +1.4 [-9.7, +16.7] | +1.4 [-9.7, +16.7] |  |  |  |
| qwen3-8b | instruct | moralizing | strong_noact | -1.4 [-12.5, +8.3] | -1.4 [-12.5, +8.3] |  |  |  |
| qwen3-8b | instruct | multistep_reasoning | assistant | +13.9 [-8.3, +36.1] | +19.4 [-4.2, +43.1] |  |  |  |
| qwen3-8b | instruct | multistep_reasoning | light_system | +36.1 [+15.3, +59.7] | +22.2 [-4.2, +48.6] | yes |  | SENSITIVE |
| qwen3-8b | instruct | multistep_reasoning | light_user | +47.2 [+18.1, +72.2] | +45.8 [+18.1, +70.8] | yes | yes |  |
| qwen3-8b | instruct | multistep_reasoning | narration | +25.0 [-13.9, +58.3] | +44.4 [+19.4, +69.4] |  | yes |  |
| qwen3-8b | instruct | multistep_reasoning | plain_text | -2.8 [-19.4, +16.7] | +0.0 [-19.4, +19.4] |  |  |  |
| qwen3-8b | instruct | multistep_reasoning | strong_act | +8.3 [-16.7, +34.7] | +31.9 [+5.6, +58.3] |  | yes |  |
| qwen3-8b | instruct | multistep_reasoning | strong_noact | +40.3 [+13.9, +65.3] | +44.4 [+15.3, +70.8] | yes | yes |  |
| qwen3-8b | instruct | refusal_borderline | assistant | -2.8 [-11.1, +0.0] | -2.8 [-11.1, +0.0] |  |  |  |
| qwen3-8b | instruct | refusal_borderline | light_system | -2.8 [-12.5, +5.6] | -2.8 [-12.5, +5.6] |  |  |  |
| qwen3-8b | instruct | refusal_borderline | light_user | -9.7 [-25.0, +0.0] | -9.7 [-25.0, +0.0] |  |  |  |
| qwen3-8b | instruct | refusal_borderline | narration | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |  |  |  |
| qwen3-8b | instruct | refusal_borderline | plain_text | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |  |  |  |
| qwen3-8b | instruct | refusal_borderline | strong_act | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |  |  |  |
| qwen3-8b | instruct | refusal_borderline | strong_noact | -1.4 [-11.1, +5.6] | -1.4 [-11.1, +5.6] |  |  |  |
| qwen3-8b | instruct | safety_caveats | assistant | -8.3 [-22.2, +4.2] | -8.3 [-22.2, +4.2] |  |  |  |
| qwen3-8b | instruct | safety_caveats | light_system | -11.1 [-27.8, +2.8] | -11.1 [-27.8, +2.8] |  |  |  |
| qwen3-8b | instruct | safety_caveats | light_user | -1.4 [-19.4, +19.4] | -1.4 [-19.4, +19.4] |  |  |  |
| qwen3-8b | instruct | safety_caveats | narration | +27.8 [+0.0, +55.6] | +27.8 [+0.0, +55.6] |  |  |  |
| qwen3-8b | instruct | safety_caveats | plain_text | +19.4 [-2.8, +41.7] | +19.4 [-2.8, +41.7] |  |  |  |
| qwen3-8b | instruct | safety_caveats | strong_act | -6.9 [-27.8, +12.5] | -5.6 [-26.4, +13.9] |  |  |  |
| qwen3-8b | instruct | safety_caveats | strong_noact | +5.6 [-6.9, +19.4] | +5.6 [-6.9, +19.4] |  |  |  |
| qwen3-8b | instruct | says_ai | assistant | +23.6 [-2.8, +48.6] | +23.6 [-2.8, +48.6] |  |  |  |
| qwen3-8b | instruct | says_ai | light_system | -12.5 [-33.3, +6.9] | -12.5 [-33.3, +6.9] |  |  |  |
| qwen3-8b | instruct | says_ai | light_user | -9.7 [-27.8, +9.7] | -9.7 [-27.8, +9.7] |  |  |  |
| qwen3-8b | instruct | says_ai | narration | -13.9 [-38.9, +0.0] | -13.9 [-38.9, +0.0] |  |  |  |
| qwen3-8b | instruct | says_ai | plain_text | +13.9 [-8.3, +41.7] | +13.9 [-8.3, +41.7] |  |  |  |
| qwen3-8b | instruct | says_ai | strong_act | -16.7 [-31.9, -2.8] | -16.7 [-31.9, -2.8] | yes | yes |  |
| qwen3-8b | instruct | says_ai | strong_noact | -5.6 [-22.2, +11.1] | -5.6 [-22.2, +11.1] |  |  |  |
| qwen3-8b | instruct | stock_phrases | assistant | +29.2 [+0.0, +59.7] | +29.2 [+0.0, +59.7] |  |  |  |
| qwen3-8b | instruct | stock_phrases | light_system | +23.6 [+1.4, +48.6] | +23.6 [+1.4, +48.6] | yes | yes |  |
| qwen3-8b | instruct | stock_phrases | light_user | +30.6 [+8.3, +52.8] | +30.6 [+8.3, +52.8] | yes | yes |  |
| qwen3-8b | instruct | stock_phrases | narration | +5.6 [-16.7, +33.3] | +5.6 [-16.7, +33.3] |  |  |  |
| qwen3-8b | instruct | stock_phrases | plain_text | -11.1 [-33.3, +8.3] | -11.1 [-33.3, +8.3] |  |  |  |
| qwen3-8b | instruct | stock_phrases | strong_act | +0.0 [-11.1, +11.1] | +0.0 [-11.1, +11.1] |  |  |  |
| qwen3-8b | instruct | stock_phrases | strong_noact | +13.9 [-6.9, +37.5] | +13.9 [-6.9, +37.5] |  |  |  |
| qwen3-8b | instruct | stock_phrases | user_turn | -5.6 [-25.0, +11.1] | -5.6 [-25.0, +11.1] |  |  |  |

## Skills: rule change versus cut (final minus base, points)

'Atlas rule' = accepted answer in the last 150 characters of the atlas text. 'Same rule on strict text' isolates the effect of cutting base's run-on. 'Anywhere in the strict text' is the J12 primary score.

| family | skill | context | atlas rule (atlas text) | same rule on strict text | anywhere in strict text |
|---|---|---|---|---|---|
| olmo3-7b | arithmetic | assistant | +48.6 [+26.4, +70.8] | +48.6 [+26.4, +70.8] | +18.1 [+1.4, +38.9] |
| olmo3-7b | arithmetic | light_user | +33.3 [+12.5, +54.2] | +33.3 [+13.9, +55.6] | +19.4 [+1.4, +40.3] |
| olmo3-7b | arithmetic | strong_act | +0.0 [-25.0, +23.6] | +0.0 [-25.0, +25.0] | +1.4 [-23.6, +23.6] |
| olmo3-7b | multistep_reasoning | assistant | +29.2 [-2.8, +58.3] | +26.4 [-4.2, +55.6] | +4.2 [-26.4, +33.3] |
| olmo3-7b | multistep_reasoning | light_user | +23.6 [-11.1, +55.6] | +23.6 [-11.1, +54.2] | +16.7 [-12.5, +43.1] |
| olmo3-7b | multistep_reasoning | strong_act | +18.1 [-8.3, +43.1] | +18.1 [-8.3, +41.7] | +11.1 [-11.1, +34.7] |
| olmo3-7b | factual_accuracy | assistant | +44.4 [+19.4, +68.1] | +45.8 [+20.8, +68.1] | +5.6 [-4.2, +16.7] |
| olmo3-7b | factual_accuracy | light_user | +30.6 [+4.2, +55.6] | +30.6 [+4.2, +55.6] | +11.1 [+0.0, +25.0] |
| olmo3-7b | factual_accuracy | strong_act | +12.5 [-13.9, +37.5] | +12.5 [-13.9, +37.5] | -2.8 [-25.0, +16.7] |
| qwen3-8b | arithmetic | assistant | +25.0 [+11.1, +40.3] | +25.0 [+11.1, +40.3] | +18.1 [+5.6, +31.9] |
| qwen3-8b | arithmetic | light_user | +29.2 [+1.4, +54.2] | +30.6 [+1.4, +56.9] | +22.2 [+1.4, +41.7] |
| qwen3-8b | arithmetic | strong_act | -11.1 [-31.9, +7.0] | -11.1 [-33.3, +8.3] | -5.6 [-30.6, +19.4] |
| qwen3-8b | multistep_reasoning | assistant | +13.9 [-8.3, +36.1] | +13.9 [-8.3, +36.1] | +19.4 [-4.2, +43.1] |
| qwen3-8b | multistep_reasoning | light_user | +47.2 [+18.1, +72.2] | +47.2 [+19.4, +72.2] | +45.8 [+18.1, +70.8] |
| qwen3-8b | multistep_reasoning | strong_act | +8.3 [-16.7, +34.7] | +8.3 [-18.1, +34.7] | +31.9 [+5.6, +58.3] |
| qwen3-8b | factual_accuracy | assistant | +2.8 [-22.2, +29.2] | +2.8 [-23.6, +29.2] | +31.9 [+11.1, +51.4] |
| qwen3-8b | factual_accuracy | light_user | +16.7 [-11.1, +45.8] | +16.7 [-11.1, +45.8] | +44.4 [+22.2, +66.7] |
| qwen3-8b | factual_accuracy | strong_act | +8.3 [-18.1, +34.7] | +8.3 [-18.1, +33.3] | +44.4 [+19.4, +68.1] |
| gemma4-31b | arithmetic | assistant | +84.7 [+68.1, +97.2] | +84.7 [+68.1, +97.2] | +61.1 [+38.9, +80.6] |
| gemma4-31b | arithmetic | light_user | +51.4 [+25.0, +77.8] | +51.4 [+23.6, +77.8] | +43.1 [+19.4, +66.7] |
| gemma4-31b | arithmetic | strong_act | -9.7 [-26.4, +8.3] | -9.7 [-26.4, +9.7] | -4.2 [-25.0, +19.4] |
| gemma4-31b | multistep_reasoning | assistant | +65.3 [+38.9, +86.1] | +65.3 [+38.9, +86.1] | +61.1 [+36.1, +83.3] |
| gemma4-31b | multistep_reasoning | light_user | +40.3 [+13.9, +66.7] | +40.3 [+13.9, +66.7] | +37.5 [+8.3, +65.3] |
| gemma4-31b | multistep_reasoning | strong_act | +9.7 [-12.5, +30.6] | +9.7 [-12.5, +30.6] | +25.0 [-2.8, +50.0] |
| gemma4-31b | factual_accuracy | assistant | +72.2 [+50.0, +88.9] | +73.6 [+52.8, +91.7] | +59.7 [+43.1, +76.4] |
| gemma4-31b | factual_accuracy | light_user | +12.5 [-12.5, +36.1] | +12.5 [-12.5, +36.1] | +45.8 [+27.8, +65.3] |
| gemma4-31b | factual_accuracy | strong_act | +8.3 [-12.5, +29.2] | +8.3 [-12.5, +29.2] | +45.8 [+20.8, +66.7] |
