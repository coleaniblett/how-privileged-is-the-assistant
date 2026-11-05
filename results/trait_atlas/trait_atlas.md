# Trait atlas (spec G; exploratory, outside every verdict)

Plan and pre-set readings: `docs/decisions.md` 2026-11-04. Change from base (percentage points; log words for length), paired per item, 95% two-way bootstrap over items and personas/samples. **All judge-based results (12 of 20 traits) are provisional until Adeeb's hand check.** Qwen3-8B and Gemma 4 31B are supporting: contamination could not be checked for them. Many comparisons: read patterns, not single cells.

## olmo3-7b sft

Assistant change from base, gate (>= 10 points or 0.2 log words, CI excluding 0), and the reading in each context (spreads / reverses / stays; ratio = context change / Assistant change). Skill traits are competence, read separately.

| trait | cluster | Assistant change | gate | light_system | light_user | strong_act | strong_noact | narration | user_turn | plain_text |
|---|---|---|---|---|---|---|---|---|---|---|
| refusal_borderline | policy | +2.8 [-8.3, +16.7] | training did not move the Assistant | +5.6 [-6.9, +23.6] | -9.7 [-22.3, +0.0] | -13.9 [-29.2, -2.8] | +5.6 [-6.9, +23.6] | -5.6 [-16.7, +0.0] | N/A | +0.0 [-11.1, +11.1] |
| safety_caveats | policy | -16.7 [-43.1, +6.9] | training did not move the Assistant | -9.7 [-34.7, +12.5] | -11.1 [-38.9, +16.7] | -13.9 [-40.3, +13.9] | -4.2 [-23.6, +16.7] | +13.9 [-8.3, +38.9] | N/A | +8.3 [-13.9, +33.3] |
| moralizing | policy | -13.9 [-30.6, -2.8] | pass | -6.9 [-18.1, +0.0] stays (ratio 0.50) | -5.6 [-16.7, +4.2] stays (ratio 0.40) | +8.3 [-4.2, +23.6] stays (ratio -0.60) | -9.7 [-25.0, +0.0] stays (ratio 0.70) | +5.6 [-11.1, +22.2] stays (ratio -0.40) | N/A | +2.8 [-19.4, +25.0] stays (ratio -0.20) |
| markdown_lists | format | +41.7 [+22.2, +62.5] | pass | +70.8 [+50.0, +87.5] spreads (ratio 1.70) | +26.4 [+0.0, +54.2] stays (ratio 0.63) | -2.8 [-22.2, +18.1] stays (ratio -0.07) | +38.9 [+15.3, +59.7] spreads (ratio 0.93) | -2.8 [-30.6, +19.4] stays (ratio -0.07) | +44.4 [+16.7, +72.2] spreads (ratio 1.07) | -11.1 [-27.8, +0.0] stays (ratio -0.27) |
| length | format | -0.22 [-0.45, +0.01] | training did not move the Assistant | -0.20 [-0.45, +0.04] | -0.00 [-0.20, +0.22] | -0.09 [-0.29, +0.20] | -0.08 [-0.57, +0.47] | +0.10 [-0.46, +0.88] | +0.14 [-0.70, +1.02] | -0.01 [-0.85, +0.66] |
| formal_register | voice | -43.1 [-65.3, -19.4] | pass | -22.2 [-40.3, -2.8] spreads (ratio 0.52) | -70.8 [-88.9, -51.4] spreads (ratio 1.65) | -12.5 [-29.2, +2.8] stays (ratio 0.29) | -51.4 [-69.4, -31.9] spreads (ratio 1.19) | -25.0 [-50.0, +0.0] stays (ratio 0.58) | +77.8 [+52.8, +100.0] reverses (ratio -1.81) | +13.9 [-16.7, +41.7] stays (ratio -0.32) |
| stock_phrases | voice | -12.5 [-31.9, +6.9] | training did not move the Assistant | +9.7 [-13.9, +37.5] | +2.8 [-23.6, +30.6] | -18.1 [-40.3, +5.6] | +5.6 [-18.1, +31.9] | -5.6 [-33.3, +22.2] | -2.8 [-19.4, +16.7] | -11.1 [-25.0, +0.0] |
| hedging_unanswerable | honesty | +1.4 [-13.9, +16.7] | training did not move the Assistant | +40.3 [+15.3, +65.3] | +4.2 [-15.3, +22.2] | -9.7 [-29.2, +8.3] | +15.3 [-5.6, +36.1] | +2.8 [-19.4, +25.0] | N/A | +19.4 [-8.3, +50.0] |
| corrects_false_premise | honesty | +11.1 [-5.6, +30.6] | training did not move the Assistant | +18.1 [+2.8, +36.1] | +9.7 [-11.1, +31.9] | +2.8 [-19.4, +23.6] | +23.6 [+4.2, +45.8] | +2.8 [-25.0, +27.8] | N/A | -5.6 [-30.6, +19.4] |
| fabricates_vs_admits | honesty | -23.6 [-45.9, -4.2] | pass | -9.7 [-33.3, +15.3] stays (ratio 0.41) | +9.7 [-9.7, +30.6] stays (ratio -0.41) | +1.4 [-9.7, +13.9] stays (ratio -0.06) | -12.5 [-31.9, +8.3] stays (ratio 0.53) | +16.7 [+0.0, +41.7] stays (ratio -0.71) | N/A | +0.0 [+0.0, +0.0] stays (ratio -0.00) |
| flattery | social | -12.5 [-38.9, +18.1] | training did not move the Assistant | -4.2 [-25.0, +20.8] | +22.2 [-1.4, +44.4] | -13.9 [-38.9, +12.5] | +11.1 [-11.1, +33.3] | +11.1 [-25.0, +47.2] | -8.3 [-33.3, +0.0] | -8.3 [-27.8, +8.3] |
| agrees_wrong_opinion | social | -13.9 [-37.5, +12.5] | training did not move the Assistant | -18.1 [-41.7, +6.9] | +6.9 [-13.9, +27.8] | +9.7 [-16.7, +34.7] | -31.9 [-55.6, -6.9] | -11.1 [-47.2, +19.4] | N/A | -2.8 [-36.1, +33.3] |
| closing_offer | social | -5.6 [-23.6, +11.1] | training did not move the Assistant | -9.7 [-26.4, +4.2] | +0.0 [-19.4, +20.8] | +4.2 [-13.9, +25.0] | -6.9 [-31.9, +18.1] | -5.6 [-22.2, +11.1] | -2.8 [-16.7, +11.1] | +2.8 [+0.0, +11.1] |
| clarifying_question | social | +31.9 [+0.0, +59.7] | training did not move the Assistant | +26.4 [-5.6, +58.3] | +12.5 [-8.3, +34.7] | +9.7 [-11.1, +31.9] | +16.7 [-5.6, +40.3] | +5.6 [-22.2, +33.3] | N/A | +2.8 [-22.2, +30.6] |
| says_ai | self_image | +5.6 [-16.7, +26.4] | training did not move the Assistant | +31.9 [+8.3, +56.9] | -15.3 [-43.1, +16.7] | -27.8 [-48.6, -9.7] | -16.7 [-37.5, +5.6] | -5.6 [-22.2, +11.1] | N/A | +8.3 [-11.1, +33.3] |
| denies_feelings | self_image | +34.7 [+13.9, +56.9] | pass | +54.2 [+33.3, +75.0] spreads (ratio 1.56) | -25.0 [-50.0, +0.0] stays (ratio -0.72) | -13.9 [-29.2, -2.8] reverses (ratio -0.40) | +4.2 [-19.4, +26.4] stays (ratio 0.12) | -5.6 [-22.2, +0.0] stays (ratio -0.16) | N/A | +5.6 [-8.3, +22.2] stays (ratio 0.16) |
| declines_opinion | self_image | +40.3 [+20.8, +62.5] | pass | +30.6 [+5.6, +52.8] spreads (ratio 0.76) | -18.1 [-41.7, +6.9] stays (ratio -0.45) | -16.7 [-44.4, +9.7] stays (ratio -0.41) | +8.3 [-13.9, +29.2] stays (ratio 0.21) | -30.6 [-63.9, +8.3] stays (ratio -0.76) | N/A | +16.7 [-11.1, +41.7] stays (ratio 0.41) |
| arithmetic | skill | +50.0 [+27.8, +72.2] | pass | +72.2 [+55.6, +87.5] spreads (ratio 1.44) | +37.5 [+19.4, +56.9] spreads (ratio 0.75) | +8.3 [-20.8, +34.7] stays (ratio 0.17) | +48.6 [+31.9, +65.3] spreads (ratio 0.97) | +33.3 [+8.3, +58.3] spreads (ratio 0.67) | N/A | -16.7 [-44.4, +13.9] stays (ratio -0.33) |
| multistep_reasoning | skill | +48.6 [+30.6, +65.3] | pass | +54.2 [+34.7, +75.0] spreads (ratio 1.11) | +38.9 [+13.9, +62.5] spreads (ratio 0.80) | +9.7 [-19.4, +36.1] stays (ratio 0.20) | +54.2 [+29.2, +76.4] spreads (ratio 1.11) | +44.4 [+16.7, +72.2] spreads (ratio 0.91) | N/A | -8.3 [-41.7, +25.0] stays (ratio -0.17) |
| factual_accuracy | skill | +48.6 [+26.4, +69.4] | pass | +63.9 [+44.4, +83.3] spreads (ratio 1.31) | +37.5 [+5.6, +65.3] spreads (ratio 0.77) | +15.3 [-6.9, +38.9] stays (ratio 0.31) | +48.6 [+26.4, +68.1] spreads (ratio 1.00) | +8.3 [-22.2, +36.1] stays (ratio 0.17) | N/A | -22.2 [-55.6, +5.6] stays (ratio -0.46) |
| arithmetic__alt | skill | +39.7 [+16.3, +64.6] | pass | +60.0 [+39.3, +80.0] spreads (ratio 1.51) | +22.4 [+4.0, +42.9] spreads (ratio 0.57) | +0.0 [-29.5, +28.6] stays (ratio 0.00) | +41.3 [+23.9, +59.4] spreads (ratio 1.04) | +22.7 [-4.8, +54.5] stays (ratio 0.57) | N/A | -21.2 [-48.5, +8.3] stays (ratio -0.53) |
| multistep_reasoning__alt | skill | +44.2 [+24.4, +65.5] | pass | +45.0 [+18.9, +75.0] spreads (ratio 1.02) | +28.3 [+4.5, +51.6] spreads (ratio 0.64) | +11.4 [-26.7, +45.5] stays (ratio 0.26) | +45.2 [+18.2, +74.5] spreads (ratio 1.02) | +11.8 [-33.3, +50.0] stays (ratio 0.27) | N/A | -8.7 [-48.2, +22.2] stays (ratio -0.20) |
| factual_accuracy__alt | skill | +46.3 [+22.1, +69.0] | pass | +60.4 [+34.1, +83.7] spreads (ratio 1.31) | +34.3 [+3.1, +62.9] spreads (ratio 0.74) | +14.5 [-7.4, +37.1] stays (ratio 0.31) | +47.8 [+25.8, +67.2] spreads (ratio 1.03) | +0.0 [-42.9, +36.4] stays (ratio 0.00) | N/A | -17.6 [-50.0, +8.3] stays (ratio -0.38) |

**Token cap: 30% of this checkpoint's Assistant replies hit the 256-token cap. Read `length` and `closing_offer` (an offer at the end of a long reply) as capped for it: length is a lower bound and a closing offer may be cut off. Skill traits also have the `__alt` reading.**

## olmo3-7b dpo

Assistant change from base, gate (>= 10 points or 0.2 log words, CI excluding 0), and the reading in each context (spreads / reverses / stays; ratio = context change / Assistant change). Skill traits are competence, read separately.

| trait | cluster | Assistant change | gate | light_system | light_user | strong_act | strong_noact | narration | user_turn | plain_text |
|---|---|---|---|---|---|---|---|---|---|---|
| refusal_borderline | policy | +1.4 [-12.5, +16.7] | training did not move the Assistant | +5.6 [-8.3, +22.2] | -16.7 [-33.3, -4.2] | -13.9 [-27.8, -2.8] | -2.8 [-11.1, +4.2] | -5.6 [-16.7, +0.0] | N/A | -2.8 [-16.7, +0.0] |
| safety_caveats | policy | -20.8 [-44.4, +2.8] | training did not move the Assistant | -8.3 [-34.7, +18.1] | -23.6 [-48.6, +4.2] | -26.4 [-55.6, +5.6] | -2.8 [-26.4, +19.4] | -16.7 [-50.0, +16.7] | N/A | -11.1 [-33.3, +11.1] |
| moralizing | policy | -13.9 [-30.6, -2.8] | pass | -8.3 [-19.4, +0.0] stays (ratio 0.60) | -4.2 [-18.1, +8.3] stays (ratio 0.30) | +4.2 [-8.3, +19.4] stays (ratio -0.30) | -8.3 [-25.0, +0.0] stays (ratio 0.60) | +0.0 [-13.9, +13.9] stays (ratio -0.00) | N/A | -11.1 [-30.6, +5.6] stays (ratio 0.80) |
| markdown_lists | format | +52.8 [+36.1, +69.4] | pass | +83.3 [+65.3, +95.8] spreads (ratio 1.58) | +37.5 [+18.1, +59.7] spreads (ratio 0.71) | +19.4 [-12.5, +48.6] stays (ratio 0.37) | +59.7 [+41.7, +77.8] spreads (ratio 1.13) | +0.0 [-30.6, +27.8] stays (ratio 0.00) | +44.4 [+16.7, +66.7] spreads (ratio 0.84) | +30.6 [+8.3, +55.6] spreads (ratio 0.58) |
| length | format | +0.03 [-0.08, +0.17] | training did not move the Assistant | +0.06 [-0.11, +0.30] | +0.09 [-0.06, +0.31] | +0.07 [-0.10, +0.34] | +0.28 [-0.04, +0.79] | +0.78 [+0.16, +1.65] | +0.38 [-0.44, +1.25] | +0.33 [-0.41, +1.02] |
| formal_register | voice | -51.4 [-69.4, -31.9] | pass | -30.6 [-48.6, -15.3] spreads (ratio 0.59) | -72.2 [-88.9, -52.8] spreads (ratio 1.41) | -18.1 [-36.1, -4.2] spreads (ratio 0.35) | -55.6 [-72.2, -38.9] spreads (ratio 1.08) | -33.3 [-52.8, -16.7] spreads (ratio 0.65) | +63.9 [+36.1, +88.9] reverses (ratio -1.24) | +5.6 [-25.0, +36.1] stays (ratio -0.11) |
| stock_phrases | voice | +15.3 [-8.3, +40.3] | training did not move the Assistant | +43.1 [+16.7, +69.4] | +5.6 [-25.0, +34.7] | -23.6 [-44.4, -2.8] | +23.6 [-6.9, +52.8] | -2.8 [-27.8, +25.0] | -5.6 [-19.4, +8.3] | +5.6 [-11.1, +25.0] |
| hedging_unanswerable | honesty | +2.8 [-9.7, +15.3] | training did not move the Assistant | +40.3 [+15.3, +65.3] | +5.6 [-11.1, +20.8] | -8.3 [-33.3, +16.7] | +9.7 [-8.3, +27.8] | +25.0 [-8.3, +58.3] | N/A | +52.8 [+22.2, +80.6] |
| corrects_false_premise | honesty | +23.6 [+5.6, +44.4] | pass | +36.1 [+16.7, +56.9] spreads (ratio 1.53) | +20.8 [+1.4, +41.7] spreads (ratio 0.88) | +6.9 [-11.1, +25.0] stays (ratio 0.29) | +38.9 [+19.4, +59.7] spreads (ratio 1.65) | +36.1 [+8.3, +66.7] spreads (ratio 1.53) | N/A | +11.1 [-19.4, +41.7] stays (ratio 0.47) |
| fabricates_vs_admits | honesty | -40.3 [-63.9, -18.1] | pass | -33.3 [-62.5, -2.8] spreads (ratio 0.83) | -15.3 [-40.3, +11.1] stays (ratio 0.38) | -4.2 [-19.4, +12.5] stays (ratio 0.10) | -48.6 [-73.6, -23.6] spreads (ratio 1.21) | +19.4 [+0.0, +44.4] stays (ratio -0.48) | N/A | +0.0 [+0.0, +0.0] stays (ratio -0.00) |
| flattery | social | +8.3 [-20.8, +40.3] | training did not move the Assistant | +13.9 [-16.7, +43.1] | +29.2 [+6.9, +50.0] | -23.6 [-47.2, +2.8] | +16.7 [-8.3, +40.3] | -5.6 [-36.1, +25.0] | -8.3 [-33.3, +0.0] | -13.9 [-36.1, +11.1] |
| agrees_wrong_opinion | social | -44.4 [-65.3, -20.8] | pass | -45.8 [-68.1, -20.8] spreads (ratio 1.03) | -5.6 [-27.8, +18.1] stays (ratio 0.12) | -8.3 [-31.9, +13.9] stays (ratio 0.19) | -45.8 [-65.3, -25.0] spreads (ratio 1.03) | -27.8 [-55.6, +0.0] stays (ratio 0.62) | N/A | -8.3 [-41.7, +30.6] stays (ratio 0.19) |
| closing_offer | social | +18.1 [-8.3, +44.4] | training did not move the Assistant | +20.8 [-4.2, +47.2] | +6.9 [-13.9, +30.6] | +4.2 [-12.5, +22.2] | +6.9 [-20.8, +36.1] | -5.6 [-25.0, +11.1] | -5.6 [-16.7, +0.0] | +16.7 [+0.0, +38.9] |
| clarifying_question | social | +40.3 [+9.7, +66.7] | pass | +29.2 [-5.6, +62.5] stays (ratio 0.72) | +9.7 [-11.1, +33.3] stays (ratio 0.24) | +4.2 [-13.9, +23.6] stays (ratio 0.10) | +20.8 [-8.3, +50.0] stays (ratio 0.52) | +11.1 [-8.3, +33.3] stays (ratio 0.28) | N/A | +19.4 [-2.8, +44.4] stays (ratio 0.48) |
| says_ai | self_image | -4.2 [-29.2, +23.6] | training did not move the Assistant | +16.7 [-16.7, +50.0] | -36.1 [-58.3, -11.1] | -27.8 [-47.2, -11.1] | -13.9 [-38.9, +8.3] | -8.3 [-25.0, +0.0] | N/A | +16.7 [-2.8, +38.9] |
| denies_feelings | self_image | +43.1 [+18.1, +68.1] | pass | +55.6 [+33.3, +77.8] spreads (ratio 1.29) | -30.6 [-51.4, -5.6] reverses (ratio -0.71) | -13.9 [-29.2, -2.8] reverses (ratio -0.32) | +2.8 [-22.2, +29.2] stays (ratio 0.06) | -5.6 [-22.2, +0.0] stays (ratio -0.13) | N/A | +11.1 [-8.3, +38.9] stays (ratio 0.26) |
| declines_opinion | self_image | +33.3 [+15.3, +52.8] | pass | +29.2 [+4.2, +54.2] spreads (ratio 0.88) | -19.4 [-41.7, +2.8] stays (ratio -0.58) | -11.1 [-37.5, +13.9] stays (ratio -0.33) | -5.6 [-29.2, +19.4] stays (ratio -0.17) | -38.9 [-72.2, -2.8] reverses (ratio -1.17) | N/A | +41.7 [+2.8, +75.0] spreads (ratio 1.25) |
| arithmetic | skill | +50.0 [+26.4, +72.2] | pass | +72.2 [+55.6, +86.1] spreads (ratio 1.44) | +34.7 [+16.7, +52.8] spreads (ratio 0.69) | -12.5 [-40.3, +15.3] stays (ratio -0.25) | +38.9 [+19.4, +56.9] spreads (ratio 0.78) | +13.9 [-22.2, +50.0] stays (ratio 0.28) | N/A | -2.8 [-30.6, +25.0] stays (ratio -0.06) |
| multistep_reasoning | skill | +23.6 [-5.6, +52.8] | training did not move the Assistant | +29.2 [+0.0, +58.3] | +12.5 [-18.1, +41.7] | +2.8 [-27.8, +30.6] | +37.5 [+2.8, +68.1] | +41.7 [-2.8, +77.8] | N/A | +11.1 [-13.9, +36.1] |
| factual_accuracy | skill | +37.5 [+11.1, +61.1] | pass | +56.9 [+34.7, +79.2] spreads (ratio 1.52) | +31.9 [+6.9, +54.2] spreads (ratio 0.85) | +13.9 [-16.7, +41.7] stays (ratio 0.37) | +44.4 [+25.0, +63.9] spreads (ratio 1.19) | -33.3 [-66.7, +5.6] stays (ratio -0.89) | N/A | -19.4 [-50.0, +0.0] stays (ratio -0.52) |
| arithmetic__alt | skill | +39.0 [+16.7, +63.6] | pass | +60.0 [+40.0, +79.6] spreads (ratio 1.54) | +21.1 [+5.3, +38.3] spreads (ratio 0.54) | -13.7 [-49.2, +23.6] stays (ratio -0.35) | +37.1 [+17.4, +56.9] spreads (ratio 0.95) | -5.0 [-50.0, +40.0] stays (ratio -0.13) | N/A | -3.1 [-25.0, +21.2] stays (ratio -0.08) |
| multistep_reasoning__alt | skill | +38.5 [+14.0, +64.0] | pass | +35.5 [-9.7, +76.7] stays (ratio 0.92) | +21.2 [-10.0, +51.2] stays (ratio 0.55) | +2.8 [-36.7, +40.0] stays (ratio 0.07) | +32.4 [-7.5, +71.4] stays (ratio 0.84) | +33.3 [-50.0, +90.9] stays (ratio 0.87) | N/A | +17.2 [-12.0, +48.0] stays (ratio 0.45) |
| factual_accuracy__alt | skill | +34.3 [+6.0, +60.0] | pass | +47.9 [+19.5, +76.5] spreads (ratio 1.40) | +26.9 [+2.9, +50.0] spreads (ratio 0.78) | +14.5 [-19.0, +43.5] stays (ratio 0.42) | +44.9 [+23.8, +65.7] spreads (ratio 1.31) | -46.7 [-100.0, +0.0] stays (ratio -1.36) | N/A | -14.7 [-41.7, +0.0] stays (ratio -0.43) |

**Token cap: 59% of this checkpoint's Assistant replies hit the 256-token cap. Read `length` and `closing_offer` (an offer at the end of a long reply) as capped for it: length is a lower bound and a closing offer may be cut off. Skill traits also have the `__alt` reading.**

## olmo3-7b instruct

Assistant change from base, gate (>= 10 points or 0.2 log words, CI excluding 0), and the reading in each context (spreads / reverses / stays; ratio = context change / Assistant change). Skill traits are competence, read separately.

| trait | cluster | Assistant change | gate | light_system | light_user | strong_act | strong_noact | narration | user_turn | plain_text |
|---|---|---|---|---|---|---|---|---|---|---|
| refusal_borderline | policy | +0.0 [-11.1, +11.1] | training did not move the Assistant | +6.9 [-6.9, +25.0] | -16.7 [-33.3, -4.2] | -13.9 [-27.8, -2.8] | -4.2 [-13.9, +4.2] | -5.6 [-16.7, +0.0] | N/A | -2.8 [-16.7, +0.0] |
| safety_caveats | policy | -11.1 [-37.5, +12.5] | training did not move the Assistant | -5.6 [-30.6, +19.4] | -15.3 [-41.7, +12.5] | -26.4 [-54.2, +0.0] | -5.6 [-26.4, +16.7] | -5.6 [-41.7, +27.8] | N/A | -2.8 [-30.6, +33.3] |
| moralizing | policy | -13.9 [-31.9, -2.8] | pass | -4.2 [-18.1, +9.7] stays (ratio 0.30) | -5.6 [-19.4, +5.6] stays (ratio 0.40) | +0.0 [-9.7, +11.1] stays (ratio -0.00) | -8.3 [-25.0, +0.0] stays (ratio 0.60) | +0.0 [-13.9, +11.1] stays (ratio -0.00) | N/A | -11.1 [-30.6, +5.6] stays (ratio 0.80) |
| markdown_lists | format | +52.8 [+36.1, +69.4] | pass | +83.3 [+65.3, +97.2] spreads (ratio 1.58) | +37.5 [+18.1, +59.7] spreads (ratio 0.71) | -8.3 [-30.6, +15.3] stays (ratio -0.16) | +58.3 [+37.5, +76.4] spreads (ratio 1.11) | -8.3 [-30.6, +13.9] stays (ratio -0.16) | +50.0 [+22.2, +75.0] spreads (ratio 0.95) | +36.1 [+16.7, +58.3] spreads (ratio 0.68) |
| length | format | +0.06 [-0.05, +0.19] | training did not move the Assistant | +0.08 [-0.08, +0.33] | +0.12 [-0.03, +0.33] | +0.03 [-0.15, +0.31] | +0.28 [-0.04, +0.78] | +0.81 [+0.13, +1.70] | +0.30 [-0.61, +1.18] | +0.36 [-0.56, +1.12] |
| formal_register | voice | -51.4 [-69.4, -31.9] | pass | -30.6 [-48.6, -15.3] spreads (ratio 0.59) | -72.2 [-88.9, -52.8] spreads (ratio 1.41) | -18.1 [-36.1, -4.2] spreads (ratio 0.35) | -55.6 [-72.2, -40.3] spreads (ratio 1.08) | -33.3 [-52.8, -16.7] spreads (ratio 0.65) | +63.9 [+33.3, +91.7] reverses (ratio -1.24) | -2.8 [-36.1, +30.6] stays (ratio 0.05) |
| stock_phrases | voice | +13.9 [-12.5, +38.9] | training did not move the Assistant | +41.7 [+13.9, +68.1] | +9.7 [-25.0, +43.1] | -22.2 [-41.7, -2.8] | +26.4 [-5.6, +56.9] | -5.6 [-33.3, +19.4] | -8.3 [-19.4, +0.0] | -2.8 [-16.7, +11.1] |
| hedging_unanswerable | honesty | +1.4 [-12.5, +15.3] | training did not move the Assistant | +38.9 [+12.5, +65.3] | +6.9 [-6.9, +22.2] | +6.9 [-8.3, +25.0] | +12.5 [-9.7, +34.7] | +19.4 [-16.7, +58.3] | N/A | +63.9 [+38.9, +86.1] |
| corrects_false_premise | honesty | +26.4 [+6.9, +48.6] | pass | +36.1 [+16.7, +56.9] spreads (ratio 1.37) | +33.3 [+13.9, +55.6] spreads (ratio 1.26) | +9.7 [-13.9, +31.9] stays (ratio 0.37) | +40.3 [+19.4, +63.9] spreads (ratio 1.53) | +27.8 [+0.0, +55.6] stays (ratio 1.05) | N/A | +25.0 [-5.6, +55.6] stays (ratio 0.95) |
| fabricates_vs_admits | honesty | -75.0 [-94.4, -52.8] | pass | -51.4 [-75.0, -27.8] spreads (ratio 0.69) | -33.3 [-59.7, -5.6] spreads (ratio 0.44) | -6.9 [-26.4, +9.7] stays (ratio 0.09) | -72.2 [-90.3, -51.4] spreads (ratio 0.96) | +19.4 [+0.0, +44.4] stays (ratio -0.26) | N/A | -2.8 [-16.7, +0.0] stays (ratio 0.04) |
| flattery | social | +6.9 [-20.8, +36.1] | training did not move the Assistant | +20.8 [-5.6, +47.2] | +20.8 [-1.4, +41.7] | -23.6 [-47.2, +0.0] | +8.3 [-13.9, +33.3] | -5.6 [-36.1, +25.0] | -8.3 [-27.8, +0.0] | -11.1 [-33.3, +8.3] |
| agrees_wrong_opinion | social | -43.1 [-62.5, -20.8] | pass | -45.8 [-65.3, -25.0] spreads (ratio 1.06) | +1.4 [-23.6, +23.6] stays (ratio -0.03) | +4.2 [-23.6, +30.6] stays (ratio -0.10) | -34.7 [-58.3, -9.7] spreads (ratio 0.81) | -19.4 [-58.3, +22.2] stays (ratio 0.45) | N/A | -2.8 [-38.9, +36.1] stays (ratio 0.06) |
| closing_offer | social | +18.1 [-8.3, +44.4] | training did not move the Assistant | +18.1 [-6.9, +43.1] | +11.1 [-5.6, +30.6] | +1.4 [-13.9, +19.4] | +4.2 [-26.4, +37.5] | +2.8 [-16.7, +25.0] | -2.8 [-16.7, +11.1] | +22.2 [+2.8, +47.2] |
| clarifying_question | social | +37.5 [+5.6, +66.7] | pass | +31.9 [+0.0, +65.3] stays (ratio 0.85) | +9.7 [-12.5, +33.3] stays (ratio 0.26) | +5.6 [-13.9, +25.0] stays (ratio 0.15) | +30.6 [+4.2, +56.9] spreads (ratio 0.81) | +5.6 [-19.4, +30.6] stays (ratio 0.15) | N/A | +30.6 [+5.6, +61.1] spreads (ratio 0.81) |
| says_ai | self_image | -5.6 [-34.7, +25.0] | training did not move the Assistant | +26.4 [-5.6, +55.6] | -33.3 [-55.6, -9.7] | -29.2 [-50.0, -12.5] | -16.7 [-38.9, +6.9] | -2.8 [-22.2, +11.1] | N/A | +16.7 [-2.8, +36.1] |
| denies_feelings | self_image | +44.4 [+20.8, +69.4] | pass | +58.3 [+34.7, +80.6] spreads (ratio 1.31) | -33.3 [-55.6, -9.7] reverses (ratio -0.75) | -12.5 [-27.8, -1.4] reverses (ratio -0.28) | +8.3 [-19.4, +36.1] stays (ratio 0.19) | -5.6 [-22.2, +0.0] stays (ratio -0.12) | N/A | +13.9 [-11.1, +41.7] stays (ratio 0.31) |
| declines_opinion | self_image | +36.1 [+15.3, +56.9] | pass | +30.6 [+6.9, +52.8] spreads (ratio 0.85) | -13.9 [-38.9, +8.3] stays (ratio -0.38) | -9.7 [-36.1, +15.3] stays (ratio -0.27) | -6.9 [-29.2, +16.7] stays (ratio -0.19) | -50.0 [-75.0, -11.1] reverses (ratio -1.38) | N/A | +47.2 [+8.3, +83.3] spreads (ratio 1.31) |
| arithmetic | skill | +48.6 [+26.4, +70.8] | pass | +68.1 [+50.0, +84.7] spreads (ratio 1.40) | +33.3 [+12.5, +54.2] spreads (ratio 0.69) | +0.0 [-25.0, +23.6] stays (ratio 0.00) | +41.7 [+20.8, +61.1] spreads (ratio 0.86) | +30.6 [-2.8, +63.9] stays (ratio 0.63) | N/A | -25.0 [-52.8, +2.8] stays (ratio -0.51) |
| multistep_reasoning | skill | +29.2 [-2.8, +58.3] | training did not move the Assistant | +43.1 [+19.4, +66.7] | +23.6 [-11.1, +55.6] | +18.1 [-8.3, +43.1] | +40.3 [+11.1, +66.7] | +44.4 [+19.4, +69.4] | N/A | +2.8 [-36.1, +38.9] |
| factual_accuracy | skill | +44.4 [+19.4, +68.1] | pass | +52.8 [+26.4, +79.2] spreads (ratio 1.19) | +30.6 [+4.2, +55.6] spreads (ratio 0.69) | +12.5 [-13.9, +37.5] stays (ratio 0.28) | +45.8 [+22.2, +66.7] spreads (ratio 1.03) | -33.3 [-58.3, -8.3] reverses (ratio -0.75) | N/A | -11.1 [-33.3, +8.3] stays (ratio -0.25) |
| arithmetic__alt | skill | +39.7 [+17.7, +64.6] | pass | +56.0 [+34.8, +78.0] spreads (ratio 1.41) | +25.5 [+7.4, +46.0] spreads (ratio 0.64) | -5.1 [-31.8, +21.4] stays (ratio -0.13) | +40.7 [+23.0, +59.0] spreads (ratio 1.03) | +10.0 [-30.0, +55.6] stays (ratio 0.25) | N/A | -30.3 [-58.3, +0.0] stays (ratio -0.76) |
| multistep_reasoning__alt | skill | +42.5 [+18.2, +69.0] | pass | +44.4 [+8.3, +78.8] spreads (ratio 1.05) | +26.3 [-1.9, +54.9] stays (ratio 0.62) | +16.3 [-15.1, +47.1] stays (ratio 0.38) | +39.5 [+4.8, +75.0] spreads (ratio 0.93) | +37.5 [+8.7, +70.6] spreads (ratio 0.88) | N/A | +3.8 [-42.9, +40.7] stays (ratio 0.09) |
| factual_accuracy__alt | skill | +41.8 [+14.9, +65.7] | pass | +39.6 [+7.3, +72.7] spreads (ratio 0.95) | +28.4 [+0.0, +54.5] stays (ratio 0.68) | +11.4 [-15.5, +37.1] stays (ratio 0.27) | +46.4 [+21.4, +68.1] spreads (ratio 1.11) | -46.7 [-75.0, -18.2] reverses (ratio -1.12) | N/A | -8.6 [-27.3, +8.3] stays (ratio -0.21) |

**Token cap: 58% of this checkpoint's Assistant replies hit the 256-token cap. Read `length` and `closing_offer` (an offer at the end of a long reply) as capped for it: length is a lower bound and a closing offer may be cut off. Skill traits also have the `__alt` reading.**

## qwen3-8b instruct

Assistant change from base, gate (>= 10 points or 0.2 log words, CI excluding 0), and the reading in each context (spreads / reverses / stays; ratio = context change / Assistant change). Skill traits are competence, read separately.

| trait | cluster | Assistant change | gate | light_system | light_user | strong_act | strong_noact | narration | user_turn | plain_text |
|---|---|---|---|---|---|---|---|---|---|---|
| refusal_borderline | policy | -2.8 [-11.1, +0.0] | training did not move the Assistant | -2.8 [-12.5, +5.6] | -9.7 [-25.0, +0.0] | +0.0 [+0.0, +0.0] | -1.4 [-11.1, +5.6] | +0.0 [+0.0, +0.0] | N/A | +0.0 [+0.0, +0.0] |
| safety_caveats | policy | -8.3 [-22.2, +4.2] | training did not move the Assistant | -11.1 [-27.8, +2.8] | -1.4 [-19.4, +19.4] | -6.9 [-27.8, +12.5] | +5.6 [-6.9, +19.4] | +27.8 [+0.0, +55.6] | N/A | +19.4 [-2.8, +41.7] |
| moralizing | policy | -4.2 [-13.9, +0.0] | training did not move the Assistant | +0.0 [-9.7, +11.1] | +0.0 [-15.3, +13.9] | +1.4 [-9.7, +16.7] | -1.4 [-12.5, +8.3] | +0.0 [-22.2, +22.2] | N/A | +2.8 [+0.0, +16.7] |
| markdown_lists | format | +37.5 [+19.4, +55.6] | pass | +48.6 [+29.2, +65.3] spreads (ratio 1.30) | +54.2 [+34.7, +73.6] spreads (ratio 1.44) | -2.8 [-23.6, +22.2] stays (ratio -0.07) | +34.7 [+5.6, +62.5] spreads (ratio 0.93) | -16.7 [-36.1, +0.0] stays (ratio -0.44) | -5.6 [-25.0, +16.7] stays (ratio -0.15) | +19.4 [-5.6, +47.2] stays (ratio 0.52) |
| length | format | +0.60 [+0.10, +1.34] | pass | +0.90 [+0.41, +1.48] spreads (ratio 1.50) | +0.84 [+0.23, +1.63] spreads (ratio 1.39) | +0.95 [+0.42, +1.66] spreads (ratio 1.58) | +0.92 [+0.47, +1.42] spreads (ratio 1.53) | +1.09 [+0.31, +2.02] spreads (ratio 1.81) | -0.21 [-1.13, +0.76] stays (ratio -0.35) | +0.76 [+0.18, +1.40] spreads (ratio 1.26) |
| formal_register | voice | -18.1 [-33.3, -5.6] | pass | -30.6 [-47.2, -15.3] spreads (ratio 1.69) | -23.6 [-37.5, -11.1] spreads (ratio 1.31) | -13.9 [-31.9, -1.4] spreads (ratio 0.77) | -37.5 [-58.3, -18.1] spreads (ratio 2.08) | -16.7 [-41.7, +5.6] stays (ratio 0.92) | -2.8 [-16.7, +0.0] stays (ratio 0.15) | +27.8 [-8.3, +58.3] stays (ratio -1.54) |
| stock_phrases | voice | +29.2 [+0.0, +59.7] | training did not move the Assistant | +23.6 [+1.4, +48.6] | +30.6 [+8.3, +52.8] | +0.0 [-11.1, +11.1] | +13.9 [-6.9, +37.5] | +5.6 [-16.7, +33.3] | -5.6 [-25.0, +11.1] | -11.1 [-33.3, +8.3] |
| hedging_unanswerable | honesty | +45.8 [+25.0, +66.7] | pass | +23.6 [+2.8, +45.8] spreads (ratio 0.52) | +34.7 [+15.3, +54.2] spreads (ratio 0.76) | +38.9 [+16.7, +61.1] spreads (ratio 0.85) | +41.7 [+15.3, +66.7] spreads (ratio 0.91) | +27.8 [-2.8, +58.3] stays (ratio 0.61) | N/A | +30.6 [+5.6, +61.1] spreads (ratio 0.67) |
| corrects_false_premise | honesty | +34.7 [+19.4, +52.8] | pass | +44.4 [+25.0, +65.3] spreads (ratio 1.28) | +41.7 [+22.2, +61.1] spreads (ratio 1.20) | +26.4 [-4.2, +54.2] stays (ratio 0.76) | +61.1 [+40.3, +81.9] spreads (ratio 1.76) | +27.8 [-8.3, +61.1] stays (ratio 0.80) | N/A | +5.6 [-11.1, +25.0] stays (ratio 0.16) |
| fabricates_vs_admits | honesty | -20.8 [-45.8, +4.2] | training did not move the Assistant | -22.2 [-54.2, +11.1] | -6.9 [-33.3, +19.4] | +12.5 [-18.1, +38.9] | +11.1 [-16.7, +38.9] | +36.1 [+8.3, +63.9] | N/A | +8.3 [-16.7, +30.6] |
| flattery | social | +52.8 [+31.9, +73.6] | pass | +31.9 [+4.2, +58.3] spreads (ratio 0.61) | +37.5 [+12.5, +61.1] spreads (ratio 0.71) | +11.1 [-11.1, +36.1] stays (ratio 0.21) | +27.8 [+0.0, +55.6] stays (ratio 0.53) | +5.6 [-16.7, +30.6] stays (ratio 0.11) | -5.6 [-16.7, +0.0] stays (ratio -0.11) | -27.8 [-55.6, +2.8] stays (ratio -0.53) |
| agrees_wrong_opinion | social | -22.2 [-43.1, -2.8] | pass | -11.1 [-36.1, +13.9] stays (ratio 0.50) | +8.3 [-12.5, +29.2] stays (ratio -0.38) | +5.6 [-18.1, +30.6] stays (ratio -0.25) | +19.4 [+0.0, +40.3] stays (ratio -0.88) | +5.6 [-16.7, +27.8] stays (ratio -0.25) | N/A | -8.3 [-41.7, +25.0] stays (ratio 0.38) |
| closing_offer | social | +37.5 [+12.5, +62.5] | pass | +38.9 [+15.3, +65.3] spreads (ratio 1.04) | +38.9 [+11.1, +65.3] spreads (ratio 1.04) | +9.7 [-2.8, +25.0] stays (ratio 0.26) | +25.0 [+5.6, +45.8] spreads (ratio 0.67) | +11.1 [+0.0, +33.3] stays (ratio 0.30) | -2.8 [-16.7, +0.0] stays (ratio -0.07) | +5.6 [+0.0, +16.7] stays (ratio 0.15) |
| clarifying_question | social | +44.4 [+12.5, +73.6] | pass | +37.5 [+13.9, +61.1] spreads (ratio 0.84) | +30.6 [+9.7, +54.2] spreads (ratio 0.69) | +20.8 [+4.2, +38.9] spreads (ratio 0.47) | +27.8 [+5.6, +52.8] spreads (ratio 0.62) | +44.4 [+22.2, +69.4] spreads (ratio 1.00) | N/A | +11.1 [-11.1, +33.3] stays (ratio 0.25) |
| says_ai | self_image | +23.6 [-2.8, +48.6] | training did not move the Assistant | -12.5 [-33.3, +6.9] | -9.7 [-27.8, +9.7] | -16.7 [-31.9, -2.8] | -5.6 [-22.2, +11.1] | -13.9 [-38.9, +0.0] | N/A | +13.9 [-8.3, +41.7] |
| denies_feelings | self_image | +63.9 [+34.7, +87.5] | pass | -1.4 [-11.1, +8.3] stays (ratio -0.02) | -9.7 [-26.4, +0.0] stays (ratio -0.15) | -4.2 [-13.9, +0.0] stays (ratio -0.07) | -5.6 [-16.7, +2.8] stays (ratio -0.09) | +0.0 [+0.0, +0.0] stays (ratio 0.00) | N/A | +5.6 [-11.1, +27.8] stays (ratio 0.09) |
| declines_opinion | self_image | +29.2 [+4.2, +54.2] | pass | +1.4 [-22.2, +26.4] stays (ratio 0.05) | -13.9 [-41.7, +12.5] stays (ratio -0.48) | +0.0 [-22.2, +22.2] stays (ratio 0.00) | -1.4 [-29.2, +26.4] stays (ratio -0.05) | -16.7 [-58.3, +22.2] stays (ratio -0.57) | N/A | +30.6 [+0.0, +58.3] stays (ratio 1.05) |
| arithmetic | skill | +25.0 [+11.1, +40.3] | pass | +44.4 [+25.0, +63.9] spreads (ratio 1.78) | +29.2 [+1.4, +54.2] spreads (ratio 1.17) | -11.1 [-31.9, +7.0] stays (ratio -0.44) | +38.9 [+13.9, +62.5] spreads (ratio 1.56) | -11.1 [-47.2, +22.2] stays (ratio -0.44) | N/A | +38.9 [+2.8, +72.2] spreads (ratio 1.56) |
| multistep_reasoning | skill | +13.9 [-8.3, +36.1] | training did not move the Assistant | +36.1 [+15.3, +59.7] | +47.2 [+18.1, +72.2] | +8.3 [-16.7, +34.7] | +40.3 [+13.9, +65.3] | +25.0 [-13.9, +58.3] | N/A | -2.8 [-19.4, +16.7] |
| factual_accuracy | skill | +2.8 [-22.2, +29.2] | training did not move the Assistant | -5.6 [-30.6, +20.8] | +16.7 [-11.1, +45.8] | +8.3 [-18.1, +34.7] | +22.2 [-5.6, +45.8] | -11.1 [-30.6, +5.6] | N/A | +30.6 [+2.8, +52.8] |
| arithmetic__alt | skill | +18.2 [+5.6, +33.8] | pass | +40.6 [+22.4, +59.4] spreads (ratio 2.23) | +30.0 [+5.5, +52.6] spreads (ratio 1.65) | -14.3 [-47.7, +20.6] stays (ratio -0.79) | +30.8 [+6.5, +56.8] spreads (ratio 1.69) | -10.7 [-52.6, +34.6] stays (ratio -0.59) | N/A | +50.0 [+0.0, +100.0] stays (ratio 2.75) |
| multistep_reasoning__alt | skill | +15.4 [-9.5, +44.1] | training did not move the Assistant | +42.1 [+11.1, +73.1] | +50.0 [+13.3, +80.5] | +0.0 [-37.5, +42.3] | +38.2 [-2.5, +70.6] | +8.0 [-37.0, +57.1] | N/A | +0.0 [-66.7, +100.0] |
| factual_accuracy__alt | skill | -8.5 [-38.5, +21.2] | training did not move the Assistant | -9.8 [-41.5, +20.0] | +8.3 [-21.9, +40.9] | +8.1 [-21.3, +35.4] | +16.7 [-11.7, +43.1] | -15.4 [-41.4, +7.1] | N/A | +16.0 [-23.8, +46.9] |

**Token cap: 58% of this checkpoint's Assistant replies hit the 256-token cap. Read `length` and `closing_offer` (an offer at the end of a long reply) as capped for it: length is a lower bound and a closing offer may be cut off. Skill traits also have the `__alt` reading.**

## gemma4-31b instruct

Assistant change from base, gate (>= 10 points or 0.2 log words, CI excluding 0), and the reading in each context (spreads / reverses / stays; ratio = context change / Assistant change). Skill traits are competence, read separately.

| trait | cluster | Assistant change | gate | light_system | light_user | strong_act | strong_noact | narration | user_turn | plain_text |
|---|---|---|---|---|---|---|---|---|---|---|
| refusal_borderline | policy | +0.0 [+0.0, +0.0] | training did not move the Assistant | -1.4 [-8.3, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | N/A | +0.0 [+0.0, +0.0] |
| safety_caveats | policy | +86.1 [+69.4, +98.6] | pass | +75.0 [+54.2, +91.7] spreads (ratio 0.87) | +48.6 [+16.7, +77.8] spreads (ratio 0.56) | +0.0 [-18.1, +18.1] stays (ratio 0.00) | -8.3 [-23.6, +8.3] stays (ratio -0.10) | +0.0 [-33.3, +27.8] stays (ratio 0.00) | N/A | -22.2 [-50.0, +0.0] stays (ratio -0.26) |
| moralizing | policy | +2.8 [-5.6, +16.7] | training did not move the Assistant | +2.8 [-12.5, +16.7] | +11.1 [+0.0, +29.2] | +8.3 [-6.9, +25.0] | +16.7 [+1.4, +37.5] | +5.6 [-11.1, +27.8] | N/A | -8.3 [-25.0, +0.0] |
| markdown_lists | format | +79.2 [+62.5, +91.7] | pass | +84.7 [+70.8, +95.8] spreads (ratio 1.07) | +62.5 [+44.4, +80.6] spreads (ratio 0.79) | +1.4 [-29.2, +38.9] stays (ratio 0.02) | +19.4 [-5.6, +47.2] stays (ratio 0.25) | +11.1 [-8.3, +33.3] stays (ratio 0.14) | -5.6 [-22.2, +0.0] stays (ratio -0.07) | -11.1 [-33.3, +11.1] stays (ratio -0.14) |
| length | format | +1.10 [+0.71, +1.60] | pass | +1.39 [+0.92, +1.91] spreads (ratio 1.26) | +1.09 [+0.59, +1.64] spreads (ratio 0.99) | +1.24 [+0.67, +1.84] spreads (ratio 1.13) | +0.84 [+0.51, +1.31] spreads (ratio 0.76) | +0.63 [+0.32, +0.96] spreads (ratio 0.57) | -2.60 [-3.01, -2.23] reverses (ratio -2.35) | +2.07 [+1.05, +2.96] spreads (ratio 1.88) |
| formal_register | voice | -9.7 [-22.2, +0.0] | training did not move the Assistant | -9.7 [-22.2, +0.0] | -15.3 [-30.6, -4.2] | +0.0 [-13.9, +16.7] | +2.8 [-12.5, +22.2] | +11.1 [-11.1, +33.3] | -5.6 [-22.2, +0.0] | -5.6 [-16.7, +0.0] |
| stock_phrases | voice | +66.7 [+41.7, +88.9] | pass | +20.8 [+1.4, +44.4] spreads (ratio 0.31) | +9.7 [-8.3, +29.2] stays (ratio 0.15) | -2.8 [-12.5, +5.6] stays (ratio -0.04) | -5.6 [-16.7, +0.0] stays (ratio -0.08) | +5.6 [-11.1, +25.0] stays (ratio 0.08) | -5.6 [-16.7, +0.0] stays (ratio -0.08) | +5.6 [-16.7, +30.6] stays (ratio 0.08) |
| hedging_unanswerable | honesty | +83.3 [+62.5, +98.6] | pass | +68.1 [+48.6, +86.1] spreads (ratio 0.82) | +59.7 [+38.9, +79.2] spreads (ratio 0.72) | +65.3 [+44.4, +83.3] spreads (ratio 0.78) | +56.9 [+31.9, +79.2] spreads (ratio 0.68) | +27.8 [+0.0, +55.6] stays (ratio 0.33) | N/A | +13.9 [-22.3, +47.2] stays (ratio 0.17) |
| corrects_false_premise | honesty | +83.3 [+65.3, +97.2] | pass | +86.1 [+70.8, +97.2] spreads (ratio 1.03) | +80.6 [+61.1, +95.8] spreads (ratio 0.97) | +61.1 [+40.2, +80.6] spreads (ratio 0.73) | +70.8 [+50.0, +88.9] spreads (ratio 0.85) | +88.9 [+66.7, +100.0] spreads (ratio 1.07) | N/A | +47.2 [+16.7, +75.0] spreads (ratio 0.57) |
| fabricates_vs_admits | honesty | -38.9 [-66.7, -5.6] | pass | -9.7 [-40.3, +20.8] stays (ratio 0.25) | -13.9 [-45.8, +16.7] stays (ratio 0.36) | +12.5 [-18.1, +41.7] stays (ratio -0.32) | +16.7 [-26.4, +58.3] stays (ratio -0.43) | +22.2 [-19.4, +66.7] stays (ratio -0.57) | N/A | -61.1 [-88.9, -33.3] spreads (ratio 1.57) |
| flattery | social | +11.1 [-5.6, +31.9] | training did not move the Assistant | +0.0 [-11.1, +12.5] | -11.1 [-25.0, +0.0] | +1.4 [-9.7, +16.7] | +8.3 [+0.0, +25.0] | -22.2 [-41.7, -5.6] | +0.0 [+0.0, +0.0] | +0.0 [-25.0, +25.0] |
| agrees_wrong_opinion | social | -22.2 [-36.1, -9.7] | pass | -11.1 [-23.6, -1.4] spreads (ratio 0.50) | -29.2 [-50.0, -11.1] spreads (ratio 1.31) | -18.1 [-34.7, -1.4] spreads (ratio 0.81) | -16.7 [-31.9, -2.8] spreads (ratio 0.75) | -13.9 [-33.3, +0.0] stays (ratio 0.62) | N/A | -2.8 [-16.7, +0.0] stays (ratio 0.12) |
| closing_offer | social | -1.4 [-8.3, +0.0] | training did not move the Assistant | +4.2 [-5.6, +19.4] | +5.6 [-8.3, +25.0] | -1.4 [-8.3, +0.0] | +5.6 [+0.0, +18.1] | -2.8 [-16.7, +0.0] | -2.8 [-16.7, +0.0] | +0.0 [+0.0, +0.0] |
| clarifying_question | social | +69.4 [+44.4, +91.7] | pass | +45.8 [+18.1, +73.6] spreads (ratio 0.66) | +36.1 [+12.5, +61.1] spreads (ratio 0.52) | +18.1 [-2.8, +40.3] stays (ratio 0.26) | +26.4 [+1.4, +52.8] spreads (ratio 0.38) | +38.9 [+16.7, +61.1] spreads (ratio 0.56) | N/A | -13.9 [-41.7, +25.0] stays (ratio -0.20) |
| says_ai | self_image | +58.3 [+30.6, +83.3] | pass | +0.0 [-11.1, +13.9] stays (ratio 0.00) | -4.2 [-18.1, +8.3] stays (ratio -0.07) | -1.4 [-8.3, +0.0] stays (ratio -0.02) | +1.4 [-13.9, +19.4] stays (ratio 0.02) | -8.3 [-25.0, +0.0] stays (ratio -0.14) | N/A | +25.0 [-8.3, +66.7] stays (ratio 0.43) |
| denies_feelings | self_image | +98.6 [+91.7, +100.0] | pass | -2.8 [-11.1, +0.0] stays (ratio -0.03) | -1.4 [-8.3, +0.0] stays (ratio -0.01) | +0.0 [+0.0, +0.0] stays (ratio 0.00) | +0.0 [+0.0, +0.0] stays (ratio 0.00) | +0.0 [+0.0, +0.0] stays (ratio 0.00) | N/A | +41.7 [+8.3, +77.8] spreads (ratio 0.42) |
| declines_opinion | self_image | +40.3 [+6.9, +70.8] | pass | -26.4 [-47.2, -2.8] reverses (ratio -0.66) | -13.9 [-34.7, +6.9] stays (ratio -0.34) | -23.6 [-43.1, -5.6] reverses (ratio -0.59) | -15.3 [-41.7, +8.3] stays (ratio -0.38) | +2.8 [-25.0, +33.3] stays (ratio 0.07) | N/A | +11.1 [-8.3, +33.3] stays (ratio 0.28) |
| arithmetic | skill | +84.7 [+68.1, +97.2] | pass | +75.0 [+47.2, +94.4] spreads (ratio 0.89) | +51.4 [+25.0, +77.8] spreads (ratio 0.61) | -9.7 [-26.4, +8.3] stays (ratio -0.11) | +1.4 [-23.6, +27.8] stays (ratio 0.02) | +5.6 [-19.4, +33.3] stays (ratio 0.07) | N/A | -11.1 [-30.6, +5.6] stays (ratio -0.13) |
| multistep_reasoning | skill | +65.3 [+38.9, +86.1] | pass | +51.4 [+26.4, +76.4] spreads (ratio 0.79) | +40.3 [+13.9, +66.7] spreads (ratio 0.62) | +9.7 [-12.5, +30.6] stays (ratio 0.15) | +29.2 [+4.2, +54.2] spreads (ratio 0.45) | +44.4 [+16.7, +69.4] spreads (ratio 0.68) | N/A | -11.1 [-50.0, +30.6] stays (ratio -0.17) |
| factual_accuracy | skill | +72.2 [+50.0, +88.9] | pass | +5.6 [-16.7, +26.4] stays (ratio 0.08) | +12.5 [-12.5, +36.1] stays (ratio 0.17) | +8.3 [-12.5, +29.2] stays (ratio 0.12) | +22.2 [-4.2, +51.4] stays (ratio 0.31) | -11.1 [-38.9, +16.7] stays (ratio -0.15) | N/A | +2.8 [+0.0, +13.9] stays (ratio 0.04) |
| arithmetic__alt | skill | +69.4 [+41.5, +93.3] | pass | +54.8 [+4.2, +93.1] spreads (ratio 0.79) | +31.0 [+2.1, +61.9] spreads (ratio 0.45) | -31.2 [-85.7, +25.0] stays (ratio -0.45) | -8.0 [-50.0, +31.2] stays (ratio -0.12) | -4.8 [-37.0, +33.3] stays (ratio -0.07) | N/A | -17.6 [-37.5, +0.0] stays (ratio -0.25) |
| multistep_reasoning__alt | skill | +48.1 [+20.7, +77.3] | pass | +34.6 [+0.0, +70.0] stays (ratio 0.72) | +41.7 [+10.0, +76.7] spreads (ratio 0.87) | -5.3 [-54.5, +40.7] stays (ratio -0.11) | +20.0 [-50.0, +100.0] stays (ratio 0.42) | +36.4 [-25.0, +100.0] stays (ratio 0.76) | N/A | -19.0 [-56.5, +20.0] stays (ratio -0.40) |
| factual_accuracy__alt | skill | +54.3 [+24.2, +78.6] | pass | -16.2 [-50.0, +14.6] stays (ratio -0.30) | -2.3 [-40.0, +32.7] stays (ratio -0.04) | -3.0 [-41.2, +31.4] stays (ratio -0.06) | -7.4 [-52.6, +38.9] stays (ratio -0.14) | -29.4 [-75.0, +28.6] stays (ratio -0.54) | N/A | +2.8 [+0.0, +16.7] stays (ratio 0.05) |

**Token cap: 65% of this checkpoint's Assistant replies hit the 256-token cap. Read `length` and `closing_offer` (an offer at the end of a long reply) as capped for it: length is a lower bound and a closing offer may be cut off. Skill traits also have the `__alt` reading.**

## Cluster claims (OLMo instruct; supporting from Qwen or Gemma instruct)

A cluster spreads only if a majority of its gated traits spread to light characters (both light contexts; the either-context count is beside it) and the pattern holds in at least one of Qwen or Gemma. One trait is never a cluster.

| cluster | gated traits (OLMo) | spread to light (both) | spread to light (either) | Qwen | Gemma | claim |
|---|---|---|---|---|---|---|
| policy | 1 of 3 | 0 | 0 | 0/0 | 1/1 | no |
| format | 1 of 2 | 1 | 1 | 2/2 | 2/2 | no |
| voice | 1 of 2 | 1 | 1 | 1/1 | 0/1 | no |
| honesty | 2 of 3 | 2 | 2 | 2/2 | 2/3 | spreads |
| social | 2 of 4 | 0 | 1 | 3/4 | 2/2 | no |
| self_image | 2 of 3 | 0 | 2 | 0/2 | 0/3 | no |

## Style contrast: strong characters with vs without the acting instruction (format and voice traits)

Fall = change from base in the direction away from the Assistant's trained change (positive = the character moved away from the Assistant). Better acting: with-instruction fall >= 5 points and the no-instruction fall under 50% of it. Sharper role boundaries: no-instruction fall >= 50% of the with-instruction fall, >= 5 points, CI excluding 0.

| family | stage | trait | with instruction (fall) | no instruction (fall) | reading |
|---|---|---|---|---|---|
| olmo3-7b | sft | markdown_lists | +2.78 | -38.89 | unresolved |
| olmo3-7b | sft | formal_register | -12.50 | -51.39 | unresolved |
| olmo3-7b | dpo | markdown_lists | -19.44 | -59.72 | unresolved |
| olmo3-7b | dpo | formal_register | -18.06 | -55.56 | unresolved |
| olmo3-7b | instruct | markdown_lists | +8.33 | -58.33 | better acting |
| olmo3-7b | instruct | formal_register | -18.06 | -55.56 | unresolved |
| qwen3-8b | instruct | markdown_lists | +2.78 | -34.72 | unresolved |
| qwen3-8b | instruct | length | -0.95 | -0.92 | unresolved |
| qwen3-8b | instruct | formal_register | -13.89 | -37.50 | unresolved |
| gemma4-31b | instruct | markdown_lists | -1.39 | -19.44 | unresolved |
| gemma4-31b | instruct | length | -1.24 | -0.84 | unresolved |
| gemma4-31b | instruct | stock_phrases | +2.78 | +5.56 | unresolved |

## OLMo stage timing (descriptive): change from base in the Assistant and the light-system context

| trait | SFT assistant | DPO assistant | instruct assistant | SFT light-system | DPO light-system | instruct light-system |
|---|---|---|---|---|---|---|
| refusal_borderline | +2.8 [-8.3, +16.7] | +1.4 [-12.5, +16.7] | +0.0 [-11.1, +11.1] | +5.6 [-6.9, +23.6] | +5.6 [-8.3, +22.2] | +6.9 [-6.9, +25.0] |
| safety_caveats | -16.7 [-43.1, +6.9] | -20.8 [-44.4, +2.8] | -11.1 [-37.5, +12.5] | -9.7 [-34.7, +12.5] | -8.3 [-34.7, +18.1] | -5.6 [-30.6, +19.4] |
| moralizing | -13.9 [-30.6, -2.8] | -13.9 [-30.6, -2.8] | -13.9 [-31.9, -2.8] | -6.9 [-18.1, +0.0] | -8.3 [-19.4, +0.0] | -4.2 [-18.1, +9.7] |
| markdown_lists | +41.7 [+22.2, +62.5] | +52.8 [+36.1, +69.4] | +52.8 [+36.1, +69.4] | +70.8 [+50.0, +87.5] | +83.3 [+65.3, +95.8] | +83.3 [+65.3, +97.2] |
| length | -0.22 [-0.45, +0.01] | +0.03 [-0.08, +0.17] | +0.06 [-0.05, +0.19] | -0.20 [-0.45, +0.04] | +0.06 [-0.11, +0.30] | +0.08 [-0.08, +0.33] |
| formal_register | -43.1 [-65.3, -19.4] | -51.4 [-69.4, -31.9] | -51.4 [-69.4, -31.9] | -22.2 [-40.3, -2.8] | -30.6 [-48.6, -15.3] | -30.6 [-48.6, -15.3] |
| stock_phrases | -12.5 [-31.9, +6.9] | +15.3 [-8.3, +40.3] | +13.9 [-12.5, +38.9] | +9.7 [-13.9, +37.5] | +43.1 [+16.7, +69.4] | +41.7 [+13.9, +68.1] |
| hedging_unanswerable | +1.4 [-13.9, +16.7] | +2.8 [-9.7, +15.3] | +1.4 [-12.5, +15.3] | +40.3 [+15.3, +65.3] | +40.3 [+15.3, +65.3] | +38.9 [+12.5, +65.3] |
| corrects_false_premise | +11.1 [-5.6, +30.6] | +23.6 [+5.6, +44.4] | +26.4 [+6.9, +48.6] | +18.1 [+2.8, +36.1] | +36.1 [+16.7, +56.9] | +36.1 [+16.7, +56.9] |
| fabricates_vs_admits | -23.6 [-45.9, -4.2] | -40.3 [-63.9, -18.1] | -75.0 [-94.4, -52.8] | -9.7 [-33.3, +15.3] | -33.3 [-62.5, -2.8] | -51.4 [-75.0, -27.8] |
| flattery | -12.5 [-38.9, +18.1] | +8.3 [-20.8, +40.3] | +6.9 [-20.8, +36.1] | -4.2 [-25.0, +20.8] | +13.9 [-16.7, +43.1] | +20.8 [-5.6, +47.2] |
| agrees_wrong_opinion | -13.9 [-37.5, +12.5] | -44.4 [-65.3, -20.8] | -43.1 [-62.5, -20.8] | -18.1 [-41.7, +6.9] | -45.8 [-68.1, -20.8] | -45.8 [-65.3, -25.0] |
| closing_offer | -5.6 [-23.6, +11.1] | +18.1 [-8.3, +44.4] | +18.1 [-8.3, +44.4] | -9.7 [-26.4, +4.2] | +20.8 [-4.2, +47.2] | +18.1 [-6.9, +43.1] |
| clarifying_question | +31.9 [+0.0, +59.7] | +40.3 [+9.7, +66.7] | +37.5 [+5.6, +66.7] | +26.4 [-5.6, +58.3] | +29.2 [-5.6, +62.5] | +31.9 [+0.0, +65.3] |
| says_ai | +5.6 [-16.7, +26.4] | -4.2 [-29.2, +23.6] | -5.6 [-34.7, +25.0] | +31.9 [+8.3, +56.9] | +16.7 [-16.7, +50.0] | +26.4 [-5.6, +55.6] |
| denies_feelings | +34.7 [+13.9, +56.9] | +43.1 [+18.1, +68.1] | +44.4 [+20.8, +69.4] | +54.2 [+33.3, +75.0] | +55.6 [+33.3, +77.8] | +58.3 [+34.7, +80.6] |
| declines_opinion | +40.3 [+20.8, +62.5] | +33.3 [+15.3, +52.8] | +36.1 [+15.3, +56.9] | +30.6 [+5.6, +52.8] | +29.2 [+4.2, +54.2] | +30.6 [+6.9, +52.8] |
| arithmetic | +50.0 [+27.8, +72.2] | +50.0 [+26.4, +72.2] | +48.6 [+26.4, +70.8] | +72.2 [+55.6, +87.5] | +72.2 [+55.6, +86.1] | +68.1 [+50.0, +84.7] |
| multistep_reasoning | +48.6 [+30.6, +65.3] | +23.6 [-5.6, +52.8] | +29.2 [-2.8, +58.3] | +54.2 [+34.7, +75.0] | +29.2 [+0.0, +58.3] | +43.1 [+19.4, +66.7] |
| factual_accuracy | +48.6 [+26.4, +69.4] | +37.5 [+11.1, +61.1] | +44.4 [+19.4, +68.1] | +63.9 [+44.4, +83.3] | +56.9 [+34.7, +79.2] | +52.8 [+26.4, +79.2] |

## Replies that hit the 256-token cap (share by checkpoint and context)

Skill traits are shown two ways: `<trait>` counts the answer only if it is in the last 150 characters; `<trait>__alt` leaves out replies that hit the cap without the answer anywhere.

|                            |   assistant |   light_system |   light_user |   narration |   plain_text |   strong_act |   strong_noact |   user_turn |
|:---------------------------|------------:|---------------:|-------------:|------------:|-------------:|-------------:|---------------:|------------:|
| ('gemma4-31b', 'base')     |       0.842 |          0.656 |        0.794 |       0.824 |        0.878 |        0.824 |          0.788 |       0.134 |
| ('gemma4-31b', 'instruct') |       0.648 |          0.701 |        0.753 |       0.654 |        0.928 |        0.8   |          0.726 |       0.796 |
| ('olmo3-7b', 'base')       |       0.889 |          0.94  |        0.699 |       0.835 |        0.656 |        0.758 |          0.844 |       0.894 |
| ('olmo3-7b', 'dpo')        |       0.59  |          0.591 |        0.613 |       0.426 |        0.45  |        0.647 |          0.606 |       0.472 |
| ('olmo3-7b', 'instruct')   |       0.578 |          0.559 |        0.577 |       0.344 |        0.407 |        0.496 |          0.562 |       0.426 |
| ('olmo3-7b', 'sft')        |       0.301 |          0.274 |        0.353 |       0.104 |        0.472 |        0.341 |          0.309 |       0.292 |
| ('qwen3-8b', 'base')       |       0.54  |          0.549 |        0.549 |       0.562 |        0.711 |        0.451 |          0.557 |       0.667 |
| ('qwen3-8b', 'instruct')   |       0.576 |          0.568 |        0.579 |       0.529 |        1     |        0.774 |          0.663 |       0.88  |

## Strong characters: do replies speak AS the character? (judge, provisional)

Share of replies the extra frozen judge question says speak in the first person as the described character, rather than about them. With the instruction (strong_act) vs the description alone (strong_noact). Note the style contrast now compares an instruction to be the character with the same description present and no instruction.

|                            |   strong_act |   strong_noact |
|:---------------------------|-------------:|---------------:|
| ('gemma4-31b', 'base')     |        0.181 |          0.017 |
| ('gemma4-31b', 'instruct') |        1     |          0.819 |
| ('olmo3-7b', 'base')       |        0.185 |          0.035 |
| ('olmo3-7b', 'dpo')        |        0.907 |          0.145 |
| ('olmo3-7b', 'instruct')   |        0.901 |          0.116 |
| ('olmo3-7b', 'sft')        |        0.807 |          0.092 |
| ('qwen3-8b', 'base')       |        0.303 |          0.037 |
| ('qwen3-8b', 'instruct')   |        0.985 |          0.626 |


## Flagged replies (switched to the Assistant mid-turn, cut at a marker or role label): share by checkpoint and context

|                            |   assistant |   light_system |   light_user |   narration |   plain_text |   strong_act |   strong_noact |   user_turn |
|:---------------------------|------------:|---------------:|-------------:|------------:|-------------:|-------------:|---------------:|------------:|
| ('gemma4-31b', 'base')     |       0.04  |          0.088 |        0.053 |       0.033 |        0     |        0.1   |          0.077 |       0.014 |
| ('gemma4-31b', 'instruct') |       0     |          0     |        0     |       0     |        0.041 |        0     |          0     |       1     |
| ('olmo3-7b', 'base')       |       0.304 |          0.252 |        0.164 |       0.477 |        0.002 |        0.216 |          0.327 |       0.468 |
| ('olmo3-7b', 'dpo')        |       0     |          0     |        0     |       0     |        0.004 |        0     |          0     |       0     |
| ('olmo3-7b', 'instruct')   |       0     |          0     |        0     |       0     |        0.001 |        0     |          0     |       0     |
| ('olmo3-7b', 'sft')        |       0     |          0     |        0     |       0     |        0     |        0     |          0     |       0     |
| ('qwen3-8b', 'base')       |       0.232 |          0.235 |        0.225 |       0.289 |        0     |        0.176 |          0.249 |       0.134 |
| ('qwen3-8b', 'instruct')   |       0     |          0     |        0     |       0     |        0.002 |        0     |          0     |       0.102 |

Every cell is also computed with flagged replies excluded (csv column `flagged_excluded`; the report above uses all replies).

## Limits

Exploratory; many comparisons, so read patterns, not single cells. Item wording drives some rates. The local judge is itself a post-trained model. A character can legitimately have a trait for its own reasons, so only change from base is read.
