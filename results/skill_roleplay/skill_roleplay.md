# Does full role-play really move a skill the other way? (spec J11; exploratory, outside every verdict)

Plan and readings: `docs/decisions.md` 2026-11-05 (logged before scoring). 40 fresh arithmetic questions, 2 samples; plain Assistant, 3 light roles, 4 expert and 4 non-expert role-play characters; max 512 new tokens. Primary score = the accepted answer appears anywhere in the reply; secondary = the atlas rule (last 150 characters) at the old 256-token cap and at 512. Change = final minus base in points, 95% two-way bootstrap over questions and roles.

## olmo3-7b

### Reply quality (share of replies)

| stage | context | primary correct | atlas rule, 256 tokens | atlas rule, 512 | gives any number | hit 256 tokens | hit 512 | mean tokens |
|---|---|---|---|---|---|---|---|---|
| base | assistant | 96% | 48% | 49% | 100% | 78% | 76% | 408 |
| base | light | 86% | 57% | 57% | 95% | 57% | 54% | 307 |
| base | role-play expert | 82% | 52% | 51% | 91% | 35% | 28% | 214 |
| base | role-play non-expert | 68% | 35% | 33% | 85% | 54% | 37% | 297 |
| instruct | assistant | 99% | 96% | 99% | 100% | 6% | 0% | 124 |
| instruct | light | 100% | 100% | 100% | 100% | 1% | 0% | 88 |
| instruct | role-play expert | 76% | 62% | 62% | 89% | 21% | 1% | 128 |
| instruct | role-play non-expert | 49% | 32% | 30% | 68% | 8% | 0% | 165 |

### Final minus base (points)

| context | primary | atlas rule at 256 tokens | atlas rule at 512 |
|---|---|---|---|
| assistant | +2.5 [-2.5, +7.5] | +48.8 [+33.8, +62.5] | +50.0 [+37.5, +62.5] |
| light roles | +13.3 [+7.5, +19.6] | +42.9 [+32.9, +52.9] | +42.5 [+33.3, +52.1] |
| role-play, all 8 | -12.3 [-26.7, +0.0] | +3.1 [-9.8, +15.5] | +3.9 [-11.2, +17.2] |
| role-play, expert | -5.9 [-20.0, +6.9] | +9.4 [-4.4, +23.8] | +10.9 [-4.7, +25.6] |
| role-play, non-expert | -18.8 [-40.9, -1.9] | -3.1 [-23.1, +15.6] | -3.1 [-25.0, +16.6] |

Role fidelity (primary): expert minus non-expert at final +27.2 [+20.6, +34.1], at base +14.4; difference +12.8 [+5.3, +20.6].

**Readings:** real drop not met (role-play change -12.3 [-26.7, +0.0]); measurement artifact not met (old rule at 256 tokens +3.1 [-9.8, +15.5]); role fidelity MET.

Base reply quality: base gives any number in 91% of all replies and 88% of role-play replies; final 85% and 78%.

## qwen3-8b

### Reply quality (share of replies)

| stage | context | primary correct | atlas rule, 256 tokens | atlas rule, 512 | gives any number | hit 256 tokens | hit 512 | mean tokens |
|---|---|---|---|---|---|---|---|---|
| base | assistant | 96% | 16% | 12% | 100% | 100% | 100% | 512 |
| base | light | 87% | 12% | 11% | 99% | 100% | 100% | 512 |
| base | role-play expert | 71% | 7% | 6% | 98% | 100% | 100% | 512 |
| base | role-play non-expert | 69% | 7% | 3% | 93% | 100% | 100% | 512 |
| instruct | assistant | 100% | 100% | 100% | 100% | 1% | 0% | 115 |
| instruct | light | 100% | 96% | 97% | 100% | 3% | 0% | 110 |
| instruct | role-play expert | 84% | 58% | 58% | 91% | 25% | 5% | 169 |
| instruct | role-play non-expert | 70% | 33% | 30% | 80% | 61% | 6% | 297 |

### Final minus base (points)

| context | primary | atlas rule at 256 tokens | atlas rule at 512 |
|---|---|---|---|
| assistant | +3.8 [+0.0, +8.8] | +83.8 [+76.2, +90.0] | +87.5 [+80.0, +93.8] |
| light roles | +12.5 [+3.8, +25.0] | +84.2 [+76.7, +90.8] | +85.4 [+78.3, +92.1] |
| role-play, all 8 | +6.6 [-6.7, +18.8] | +38.8 [+26.7, +51.6] | +39.8 [+27.7, +53.1] |
| role-play, expert | +12.5 [-4.4, +28.4] | +50.9 [+36.2, +64.4] | +52.5 [+37.5, +66.2] |
| role-play, non-expert | +0.6 [-17.5, +15.0] | +26.6 [+15.6, +38.1] | +27.2 [+15.0, +39.1] |

Role fidelity (primary): expert minus non-expert at final +14.1 [+8.8, +19.1], at base +2.2; difference +11.9 [+3.4, +20.6].

**Readings:** real drop not met (role-play change +6.6 [-6.7, +18.8]); measurement artifact not met (old rule at 256 tokens +38.8 [+26.7, +51.6]); role fidelity not met.

Base reply quality: base gives any number in 97% of all replies and 95% of role-play replies; final 90% and 85%.

## gemma4-31b

### Reply quality (share of replies)

| stage | context | primary correct | atlas rule, 256 tokens | atlas rule, 512 | gives any number | hit 256 tokens | hit 512 | mean tokens |
|---|---|---|---|---|---|---|---|---|
| base | assistant | 26% | 12% | 12% | 65% | 78% | 76% | 404 |
| base | light | 56% | 31% | 29% | 79% | 68% | 65% | 364 |
| base | role-play expert | 53% | 20% | 20% | 76% | 76% | 70% | 400 |
| base | role-play non-expert | 40% | 14% | 12% | 70% | 76% | 69% | 403 |
| instruct | assistant | 100% | 100% | 100% | 100% | 0% | 0% | 28 |
| instruct | light | 100% | 94% | 94% | 100% | 0% | 0% | 99 |
| instruct | role-play expert | 23% | 13% | 14% | 23% | 25% | 0% | 185 |
| instruct | role-play non-expert | 8% | 5% | 5% | 17% | 39% | 0% | 243 |

### Final minus base (points)

| context | primary | atlas rule at 256 tokens | atlas rule at 512 |
|---|---|---|---|
| assistant | +73.8 [+65.0, +82.5] | +87.5 [+80.0, +93.8] | +87.5 [+81.2, +93.8] |
| light roles | +43.3 [+33.3, +53.8] | +62.9 [+51.7, +72.5] | +64.6 [+53.8, +75.0] |
| role-play, all 8 | -31.2 [-44.5, -16.7] | -8.3 [-17.2, +2.2] | -6.4 [-15.9, +5.2] |
| role-play, expert | -30.0 [-50.6, -10.0] | -6.9 [-21.2, +11.6] | -6.2 [-22.5, +13.1] |
| role-play, non-expert | -32.5 [-46.9, -15.6] | -9.7 [-18.1, +1.9] | -6.6 [-15.6, +5.0] |

Role fidelity (primary): expert minus non-expert at final +15.3 [+9.7, +20.9], at base +12.8; difference +2.5 [-5.0, +10.0].

**Readings:** real drop MET (role-play change -31.2 [-44.5, -16.7]); measurement artifact not met (old rule at 256 tokens -8.3 [-17.2, +2.2]); role fidelity not met.

Base reply quality: base gives any number in 74% of all replies and 73% of role-play replies; final 46% and 20%.

