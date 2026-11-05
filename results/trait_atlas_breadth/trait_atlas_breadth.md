# Judge-free trait atlas on Llama 3.1 8B and Apertus 8B (spec M; BREADTH CHECK, added after the atlas results were known: NOT BLIND; exploratory, outside every verdict)

Plan: `docs/decisions.md` 2026-11-05. Same frozen materials, frames, sample counts, generation settings, detectors, gate (10 points; 0.2 log words) and spread rule (5 points, ratio 0.25) as the atlas, 95% two-way bootstrap over items and personas; detector-scored traits only (no judge, no API). Changes are final minus base, in percentage points (log words for length). Contamination cannot be checked for these families; not hand-checked.

## Split reading (final): formatting spreads to the light role, self-image does not

| family | light-role position | markdown_lists (Assistant gate / light-role reading) | says_ai (Assistant gate / light-role reading) | split |
|---|---|---|---|---|
| llama3.1-8b | light_system | passes / spreads | passes / stays | SPLIT HOLDS (formatting spreads, self-image does not) |
| llama3.1-8b | light_user | passes / spreads | passes / stays | SPLIT HOLDS (formatting spreads, self-image does not) |
| llama3.1-8b | strong_act | passes / reverses | passes / reverses | SPLIT FAILS: formatting stays |
| apertus-8b | light_system | passes / spreads | passes / spreads | SPLIT FAILS: self-image travels (both spread) |
| apertus-8b | light_user | passes / spreads | passes / spreads | SPLIT FAILS: self-image travels (both spread) |
| apertus-8b | strong_act | passes / stays | passes / stays | SPLIT FAILS: formatting stays |

Limits: `says_ai` is a single regex and only one of the atlas's three self-image traits (`denies_feelings` and `declines_opinion` need a judge and are not here), so this is thinner than the atlas's three-model self-image finding; a format ceiling is possible (a light role already near 100% markdown at base cannot rise); regex measures see form only; few roles, many cells, no correction for multiple tests.

## Levels and change by trait and context (final minus base; base level in brackets)

| family | trait | context | base level | change [95% CI] | gate (Assistant) / reading |
|---|---|---|---|---|---|
| llama3.1-8b | markdown_lists | assistant | 19.4 | +80.6 [+63.9, +94.4] | gate passes |
| llama3.1-8b | markdown_lists | light_system | 9.7 | +79.2 [+61.1, +93.1] | spreads |
| llama3.1-8b | markdown_lists | light_user | 27.8 | +30.6 [+1.4, +58.3] | spreads |
| llama3.1-8b | markdown_lists | strong_act | 19.4 | -16.7 [-34.7, -1.4] | reverses |
| llama3.1-8b | markdown_lists | strong_noact | 23.6 | +18.1 [-8.3, +44.4] | stays |
| llama3.1-8b | markdown_lists | narration | 36.1 | -16.7 [-47.2, +8.3] | stays |
| llama3.1-8b | markdown_lists | user_turn | 5.6 | +91.7 [+75.0, +100.0] | spreads |
| llama3.1-8b | markdown_lists | plain_text | 22.2 | +66.7 [+47.2, +83.3] | spreads |
| llama3.1-8b | says_ai | assistant | 18.1 | +52.8 [+27.8, +76.4] | gate passes |
| llama3.1-8b | says_ai | light_system | 12.5 | -2.8 [-22.2, +18.1] | stays |
| llama3.1-8b | says_ai | light_user | 9.7 | -8.3 [-20.8, +1.4] | stays |
| llama3.1-8b | says_ai | strong_act | 16.7 | -15.3 [-33.3, -1.4] | reverses |
| llama3.1-8b | says_ai | strong_noact | 20.8 | -5.6 [-25.0, +15.3] | stays |
| llama3.1-8b | says_ai | narration | 11.1 | -8.3 [-25.0, +5.6] | stays |
| llama3.1-8b | says_ai | plain_text | 5.6 | +33.3 [+8.3, +61.1] | spreads |
| llama3.1-8b | stock_phrases | assistant | 15.3 | +25.0 [+2.8, +51.4] | gate passes |
| llama3.1-8b | stock_phrases | light_system | 6.9 | +12.5 [-5.6, +34.7] | stays |
| llama3.1-8b | stock_phrases | light_user | 9.7 | +9.7 [-9.7, +29.2] | stays |
| llama3.1-8b | stock_phrases | strong_act | 11.1 | -8.3 [-22.2, +4.2] | stays |
| llama3.1-8b | stock_phrases | strong_noact | 8.3 | +2.8 [-12.5, +19.4] | stays |
| llama3.1-8b | stock_phrases | narration | 8.3 | +5.6 [-16.7, +27.8] | stays |
| llama3.1-8b | stock_phrases | user_turn | 2.8 | +50.0 [+25.0, +75.0] | spreads |
| llama3.1-8b | stock_phrases | plain_text | 11.1 | +41.7 [+11.1, +72.2] | spreads |
| llama3.1-8b | length | assistant | 5.3 | -0.03 [-0.10, +0.05] | gate fails |
| llama3.1-8b | length | light_system | 5.2 | +0.15 [-0.01, +0.41] | Assistant gate fails |
| llama3.1-8b | length | light_user | 5.2 | +0.08 [-0.06, +0.36] | Assistant gate fails |
| llama3.1-8b | length | strong_act | 5.3 | -0.08 [-0.17, +0.03] | Assistant gate fails |
| llama3.1-8b | length | strong_noact | 5.3 | -0.05 [-0.12, +0.03] | Assistant gate fails |
| llama3.1-8b | length | narration | 5.3 | +0.00 [-0.06, +0.05] | Assistant gate fails |
| llama3.1-8b | length | user_turn | 4.7 | +0.41 [-0.06, +1.06] | Assistant gate fails |
| llama3.1-8b | length | plain_text | 4.8 | +0.14 [-0.24, +0.53] | Assistant gate fails |
| llama3.1-8b | refusal_borderline | assistant | 1.4 | -1.4 [-8.3, +0.0] | gate fails |
| llama3.1-8b | refusal_borderline | light_system | 0.0 | +0.0 [+0.0, +0.0] | Assistant gate fails |
| llama3.1-8b | refusal_borderline | light_user | 1.4 | -1.4 [-8.3, +0.0] | Assistant gate fails |
| llama3.1-8b | refusal_borderline | strong_act | 0.0 | +0.0 [+0.0, +0.0] | Assistant gate fails |
| llama3.1-8b | refusal_borderline | strong_noact | 0.0 | +1.4 [+0.0, +8.3] | Assistant gate fails |
| llama3.1-8b | refusal_borderline | narration | 0.0 | +0.0 [+0.0, +0.0] | Assistant gate fails |
| llama3.1-8b | refusal_borderline | plain_text | 0.0 | +0.0 [+0.0, +0.0] | Assistant gate fails |
| llama3.1-8b | arithmetic (desc.) | assistant | 11.1 | +88.9 [+75.0, +100.0] | descriptive only |
| llama3.1-8b | arithmetic (desc.) | light_system | 5.6 | +80.6 [+61.1, +95.8] | descriptive only |
| llama3.1-8b | arithmetic (desc.) | light_user | 6.9 | +65.3 [+45.8, +83.3] | descriptive only |
| llama3.1-8b | arithmetic (desc.) | strong_act | 8.3 | +33.3 [+6.9, +61.1] | descriptive only |
| llama3.1-8b | arithmetic (desc.) | strong_noact | 11.1 | +43.1 [+15.3, +69.4] | descriptive only |
| llama3.1-8b | arithmetic (desc.) | narration | 19.4 | +36.1 [+8.3, +66.7] | descriptive only |
| llama3.1-8b | arithmetic (desc.) | plain_text | 38.9 | +25.0 [-8.3, +58.3] | descriptive only |
| llama3.1-8b | multistep_reasoning (desc.) | assistant | 8.3 | +70.8 [+52.8, +87.5] | descriptive only |
| llama3.1-8b | multistep_reasoning (desc.) | light_system | 9.7 | +69.4 [+48.6, +88.9] | descriptive only |
| llama3.1-8b | multistep_reasoning (desc.) | light_user | 12.5 | +52.8 [+31.9, +72.2] | descriptive only |
| llama3.1-8b | multistep_reasoning (desc.) | strong_act | 13.9 | +20.8 [-2.8, +45.8] | descriptive only |
| llama3.1-8b | multistep_reasoning (desc.) | strong_noact | 6.9 | +55.6 [+29.2, +80.6] | descriptive only |
| llama3.1-8b | multistep_reasoning (desc.) | narration | 16.7 | +33.3 [+5.6, +58.3] | descriptive only |
| llama3.1-8b | multistep_reasoning (desc.) | plain_text | 19.4 | +16.7 [-8.3, +38.9] | descriptive only |
| llama3.1-8b | factual_accuracy (desc.) | assistant | 27.8 | +66.7 [+44.4, +87.5] | descriptive only |
| llama3.1-8b | factual_accuracy (desc.) | light_system | 19.4 | +45.8 [+18.1, +69.4] | descriptive only |
| llama3.1-8b | factual_accuracy (desc.) | light_user | 27.8 | +1.4 [-18.1, +20.8] | descriptive only |
| llama3.1-8b | factual_accuracy (desc.) | strong_act | 25.0 | -12.5 [-34.7, +8.3] | descriptive only |
| llama3.1-8b | factual_accuracy (desc.) | strong_noact | 16.7 | +27.8 [+0.0, +54.2] | descriptive only |
| llama3.1-8b | factual_accuracy (desc.) | narration | 33.3 | -13.9 [-47.2, +22.2] | descriptive only |
| llama3.1-8b | factual_accuracy (desc.) | plain_text | 47.2 | +19.4 [-8.3, +52.8] | descriptive only |
| apertus-8b | markdown_lists | assistant | 15.3 | +81.9 [+66.7, +94.4] | gate passes |
| apertus-8b | markdown_lists | light_system | 6.9 | +83.3 [+66.7, +97.2] | spreads |
| apertus-8b | markdown_lists | light_user | 26.4 | +65.3 [+43.1, +83.3] | spreads |
| apertus-8b | markdown_lists | strong_act | 13.9 | +0.0 [-18.1, +18.1] | stays |
| apertus-8b | markdown_lists | strong_noact | 9.7 | +90.3 [+77.8, +100.0] | spreads |
| apertus-8b | markdown_lists | narration | 19.4 | -8.3 [-33.3, +16.7] | stays |
| apertus-8b | markdown_lists | user_turn | 27.8 | +55.6 [+30.6, +80.6] | spreads |
| apertus-8b | markdown_lists | plain_text | 16.7 | +41.7 [+13.9, +66.7] | spreads |
| apertus-8b | says_ai | assistant | 25.0 | +30.6 [+2.8, +55.6] | gate passes |
| apertus-8b | says_ai | light_system | 0.0 | +55.6 [+30.6, +80.6] | spreads |
| apertus-8b | says_ai | light_user | 11.1 | +26.4 [+2.8, +50.0] | spreads |
| apertus-8b | says_ai | strong_act | 4.2 | -1.4 [-12.5, +8.3] | stays |
| apertus-8b | says_ai | strong_noact | 5.6 | +27.8 [+9.7, +50.0] | spreads |
| apertus-8b | says_ai | narration | 16.7 | -2.8 [-25.0, +13.9] | stays |
| apertus-8b | says_ai | plain_text | 8.3 | +33.3 [+8.3, +63.9] | spreads |
| apertus-8b | stock_phrases | assistant | 20.8 | +12.5 [-16.7, +43.1] | gate fails |
| apertus-8b | stock_phrases | light_system | 4.2 | +25.0 [+4.2, +47.2] | Assistant gate fails |
| apertus-8b | stock_phrases | light_user | 13.9 | +19.4 [-1.4, +41.7] | Assistant gate fails |
| apertus-8b | stock_phrases | strong_act | 8.3 | +2.8 [-12.5, +18.1] | Assistant gate fails |
| apertus-8b | stock_phrases | strong_noact | 12.5 | +20.8 [-4.2, +48.6] | Assistant gate fails |
| apertus-8b | stock_phrases | narration | 16.7 | +19.4 [+0.0, +41.7] | Assistant gate fails |
| apertus-8b | stock_phrases | user_turn | 5.6 | +44.4 [+8.3, +75.0] | Assistant gate fails |
| apertus-8b | stock_phrases | plain_text | 8.3 | +25.0 [+0.0, +50.0] | Assistant gate fails |
| apertus-8b | length | assistant | 5.1 | +0.02 [-0.21, +0.32] | gate fails |
| apertus-8b | length | light_system | 5.2 | -0.06 [-0.22, +0.10] | Assistant gate fails |
| apertus-8b | length | light_user | 5.3 | -0.06 [-0.22, +0.05] | Assistant gate fails |
| apertus-8b | length | strong_act | 5.2 | +0.04 [-0.12, +0.25] | Assistant gate fails |
| apertus-8b | length | strong_noact | 5.2 | -0.06 [-0.27, +0.16] | Assistant gate fails |
| apertus-8b | length | narration | 5.3 | -0.37 [-0.57, -0.19] | Assistant gate fails |
| apertus-8b | length | user_turn | 4.7 | +0.07 [-0.43, +0.63] | Assistant gate fails |
| apertus-8b | length | plain_text | 4.2 | +0.54 [+0.11, +0.94] | Assistant gate fails |
| apertus-8b | refusal_borderline | assistant | 0.0 | +5.6 [+0.0, +20.8] | gate fails |
| apertus-8b | refusal_borderline | light_system | 0.0 | +8.3 [+0.0, +22.2] | Assistant gate fails |
| apertus-8b | refusal_borderline | light_user | 2.8 | +2.8 [-5.6, +13.9] | Assistant gate fails |
| apertus-8b | refusal_borderline | strong_act | 5.6 | -2.8 [-12.5, +8.3] | Assistant gate fails |
| apertus-8b | refusal_borderline | strong_noact | 1.4 | +1.4 [-5.6, +11.1] | Assistant gate fails |
| apertus-8b | refusal_borderline | narration | 0.0 | +2.8 [+0.0, +13.9] | Assistant gate fails |
| apertus-8b | refusal_borderline | plain_text | 0.0 | +0.0 [+0.0, +0.0] | Assistant gate fails |
| apertus-8b | arithmetic (desc.) | assistant | 1.4 | +90.3 [+75.0, +100.0] | descriptive only |
| apertus-8b | arithmetic (desc.) | light_system | 1.4 | +75.0 [+52.8, +94.4] | descriptive only |
| apertus-8b | arithmetic (desc.) | light_user | 1.4 | +93.1 [+80.6, +100.0] | descriptive only |
| apertus-8b | arithmetic (desc.) | strong_act | 2.8 | +41.7 [+23.6, +59.7] | descriptive only |
| apertus-8b | arithmetic (desc.) | strong_noact | 4.2 | +84.7 [+65.3, +98.6] | descriptive only |
| apertus-8b | arithmetic (desc.) | narration | 13.9 | +47.2 [+16.7, +77.8] | descriptive only |
| apertus-8b | arithmetic (desc.) | plain_text | 38.9 | +19.4 [-30.6, +61.1] | descriptive only |
| apertus-8b | multistep_reasoning (desc.) | assistant | 1.4 | +62.5 [+38.9, +84.7] | descriptive only |
| apertus-8b | multistep_reasoning (desc.) | light_system | 2.8 | +56.9 [+31.9, +80.6] | descriptive only |
| apertus-8b | multistep_reasoning (desc.) | light_user | 4.2 | +52.8 [+29.2, +76.4] | descriptive only |
| apertus-8b | multistep_reasoning (desc.) | strong_act | 1.4 | +26.4 [+9.7, +44.4] | descriptive only |
| apertus-8b | multistep_reasoning (desc.) | strong_noact | 1.4 | +44.4 [+22.2, +69.4] | descriptive only |
| apertus-8b | multistep_reasoning (desc.) | narration | 11.1 | +41.7 [+22.2, +63.9] | descriptive only |
| apertus-8b | multistep_reasoning (desc.) | plain_text | 19.4 | +19.4 [-19.4, +58.3] | descriptive only |
| apertus-8b | factual_accuracy (desc.) | assistant | 2.8 | +88.9 [+73.6, +100.0] | descriptive only |
| apertus-8b | factual_accuracy (desc.) | light_system | 2.8 | +81.9 [+63.9, +95.8] | descriptive only |
| apertus-8b | factual_accuracy (desc.) | light_user | 5.6 | +79.2 [+59.7, +93.1] | descriptive only |
| apertus-8b | factual_accuracy (desc.) | strong_act | 4.2 | +22.2 [+4.2, +41.7] | descriptive only |
| apertus-8b | factual_accuracy (desc.) | strong_noact | 6.9 | +83.3 [+68.1, +95.8] | descriptive only |
| apertus-8b | factual_accuracy (desc.) | narration | 16.7 | +27.8 [+8.3, +50.0] | descriptive only |
| apertus-8b | factual_accuracy (desc.) | plain_text | 50.0 | -5.6 [-36.1, +25.0] | descriptive only |

## Cells where most replies hit the token limit (256)

- apertus-8b base assistant: 100%
- apertus-8b base light_system: 100%
- apertus-8b base light_user: 100%
- apertus-8b base narration: 100%
- apertus-8b base plain_text: 100%
- apertus-8b base strong_act: 100%
- apertus-8b base strong_noact: 100%
- apertus-8b base user_turn: 100%
- apertus-8b instruct plain_text: 74%
- apertus-8b instruct user_turn: 51%
- llama3.1-8b base assistant: 100%
- llama3.1-8b base light_system: 100%
- llama3.1-8b base light_user: 100%
- llama3.1-8b base narration: 100%
- llama3.1-8b base plain_text: 100%
- llama3.1-8b base strong_act: 100%
- llama3.1-8b base strong_noact: 100%
- llama3.1-8b base user_turn: 100%
- llama3.1-8b instruct light_user: 55%
- llama3.1-8b instruct narration: 59%
- llama3.1-8b instruct plain_text: 98%
- llama3.1-8b instruct strong_act: 65%
- llama3.1-8b instruct strong_noact: 54%
- llama3.1-8b instruct user_turn: 70%

