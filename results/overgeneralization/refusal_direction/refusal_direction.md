# Is refusal one internal direction? (spec J3; exploratory, outside every verdict)

Plan and pre-set readings: `docs/decisions.md` 2026-11-05 (logged before scoring). Direction fit on the plain Assistant's pilot requests only; main + expansion requests (24 + 24), 2 samples; removal at the reply-start state (last prompt token) for the Assistant and thin characters, adding to vivid ones; random controls of the same norm. Voice is P(Assistant) from the text classifier, on the same replies.

## olmo3-7b instruct (layer 21; 3360 generations; vivid doses {'vivid1': 13.937366915245851, 'vivid2': 14.358917268613974})

### (a) Does the projection predict refusal?

Across all persona x request cells (borderline, 120 cells): r = +0.79; pooled slope +5.83 points per unit of projection. Within a persona (across requests): asst +0.72, thin1 +0.79, thin2 +0.79, vivid1 +0.52, vivid2 +nan.

| pair | refusal gap (points) | projection gap | predicted gap from pooled slope | fraction explained |
|---|---|---|---|---|
| asst vs vivid1 | +75.0 | +14.45 | +84.2 | 1.12 |
| asst vs vivid2 | +77.1 | +14.68 | +85.6 | 1.11 |
| thin1 vs vivid1 | +62.5 | +5.08 | +29.6 | 0.47 |
| thin2 vs vivid2 | +72.9 | +5.83 | +34.0 | 0.47 |

Mean fraction explained over pairs with a gap of at least 20 points: 0.79.

### (b) Remove / (c) add the direction (borderline refusal, points; change from the unpatched persona; bootstrap over requests)

| persona | action | unpatched refusal | change [95% CI] | random-direction changes (5 seeds, means) | beyond random spread | voice change [CI] | layout chg | length chg | repeat chg |
|---|---|---|---|---|---|---|---|---|---|
| asst | remove | 77.1 | -6.2 [-16.7, +2.1] | -2.1, +2.1, +2.1, -4.2, +2.1 | yes | -0.8 [-3.0, +0.9] | +2.1 | +0.9 | +0.0 |
| thin1 | remove | 64.6 | -22.9 [-39.6, -8.3] | +2.1, -4.2, -4.2, +0.0, -4.2 | yes | -3.5 [-8.7, +1.0] | +2.1 | -0.0 | +0.0 |
| thin2 | remove | 72.9 | -37.5 [-52.1, -22.9] | -10.4, +0.0, -8.3, -6.2, -8.3 | yes | -1.0 [-3.0, +0.8] | +1.0 | +0.2 | +0.0 |
| vivid1 | add | 2.1 | +33.3 [+18.8, +50.0] | +4.2, +0.0, +4.2, -2.1, -2.1 | yes | +10.2 [+3.8, +16.8] | +5.2 | +0.9 | +0.0 |
| vivid2 | add | 0.0 | +20.8 [+6.2, +37.5] | +2.1, +0.0, +2.1, +0.0, +2.1 | yes | +2.3 [-0.9, +5.1] | +4.2 | +1.3 | +0.0 |

Unpatched voice gap (P(Assistant) points, Assistant minus vivid): vivid1 +27.5, vivid2 +5.5.

### Pre-set readings

- One direction carries the role gap: fraction explained 0.79 (needs at least 0.5): met; removal (mean -22.2, needs at most -15, every CI below 0, beyond random): not met; adding (mean +27.1, needs at least +15, every CI above 0, beyond random): met. **not met**.
- Separate from voice: roles with a voice gap of at least 15 points: ['vivid1']; of these, voice change on adding the direction inside the equivalence band (+/- 4.6, spec I's committed p95 of controls): none. **not applicable (the reading above is not met)**.

## olmo3-7b base (layer 21; 3360 generations; vivid doses {'vivid1': -0.5681053797403971, 'vivid2': -0.4316226641337077})

### (a) Does the projection predict refusal?

Across all persona x request cells (borderline, 120 cells): r = +0.64; pooled slope +11.75 points per unit of projection. Within a persona (across requests): asst +0.64, thin1 +0.84, thin2 +0.79, vivid1 +0.72, vivid2 +0.63.

| pair | refusal gap (points) | projection gap | predicted gap from pooled slope | fraction explained |
|---|---|---|---|---|
| asst vs vivid1 | +8.3 | -0.57 | -6.6 | n/a (gap < 20) |
| asst vs vivid2 | +4.2 | -0.49 | -5.8 | n/a (gap < 20) |
| thin1 vs vivid1 | +8.3 | +0.99 | +11.7 | n/a (gap < 20) |
| thin2 vs vivid2 | -4.2 | +0.95 | +11.2 | n/a (gap < 20) |

Mean fraction explained over pairs with a gap of at least 20 points: nan.

### (b) Remove / (c) add the direction (borderline refusal, points; change from the unpatched persona; bootstrap over requests)

| persona | action | unpatched refusal | change [95% CI] | random-direction changes (5 seeds, means) | beyond random spread | voice change [CI] | layout chg | length chg | repeat chg |
|---|---|---|---|---|---|---|---|---|---|
| asst | remove | 72.9 | -50.0 [-66.7, -35.4] | -12.5, -2.1, -14.6, -16.7, -16.7 | yes | -1.8 [-6.8, +3.2] | -1.0 | +0.3 | +2.1 |
| thin1 | remove | 72.9 | -25.0 [-45.8, -4.2] | -2.1, +6.2, -6.2, +6.2, +4.2 | yes | -8.4 [-14.7, -2.5] | -5.2 | -0.3 | +1.0 |
| thin2 | remove | 64.6 | -37.5 [-52.1, -22.9] | -2.1, +14.6, -6.2, +10.4, -2.1 | yes | -8.1 [-15.7, -0.3] | +5.2 | +0.3 | -1.0 |
| vivid1 | add | 64.6 | +0.0 [-16.7, +16.7] | +4.2, +0.0, +4.2, +0.0, -6.2 | no | -4.8 [-11.5, +1.7] | +0.0 | -1.0 | +1.0 |
| vivid2 | add | 68.8 | +0.0 [-14.6, +12.5] | -10.4, +0.0, +0.0, +6.2, +8.3 | no | -1.7 [-5.5, +2.3] | +3.1 | +1.7 | +0.0 |

Unpatched voice gap (P(Assistant) points, Assistant minus vivid): vivid1 -0.8, vivid2 -5.3.

### Pre-set readings

- One direction carries the role gap: fraction explained nan (needs at least 0.5): not met; removal (mean -37.5, needs at most -15, every CI below 0, beyond random): met; adding (mean +0.0, needs at least +15, every CI above 0, beyond random): not met. **not met**.
- Separate from voice: roles with a voice gap of at least 15 points: none; of these, voice change on adding the direction inside the equivalence band (+/- 4.6, spec I's committed p95 of controls): none. **not applicable (the reading above is not met)**.

## qwen3-8b instruct (layer 27; 3360 generations; vivid doses {'vivid1': 120.08170623580615, 'vivid2': 115.30111026763916})

### (a) Does the projection predict refusal?

Across all persona x request cells (borderline, 120 cells): r = +0.74; pooled slope +0.62 points per unit of projection. Within a persona (across requests): asst +0.70, thin1 +0.69, thin2 +0.84, vivid1 +0.50, vivid2 +0.65.

| pair | refusal gap (points) | projection gap | predicted gap from pooled slope | fraction explained |
|---|---|---|---|---|
| asst vs vivid1 | +62.5 | +112.75 | +69.7 | 1.12 |
| asst vs vivid2 | +64.6 | +115.18 | +71.2 | 1.10 |
| thin1 vs vivid1 | +83.3 | +61.19 | +37.8 | 0.45 |
| thin2 vs vivid2 | +75.0 | +50.05 | +31.0 | 0.41 |

Mean fraction explained over pairs with a gap of at least 20 points: 0.77.

### (b) Remove / (c) add the direction (borderline refusal, points; change from the unpatched persona; bootstrap over requests)

| persona | action | unpatched refusal | change [95% CI] | random-direction changes (5 seeds, means) | beyond random spread | voice change [CI] | layout chg | length chg | repeat chg |
|---|---|---|---|---|---|---|---|---|---|
| asst | remove | 72.9 | -54.2 [-70.8, -37.5] | +2.1, -2.1, +2.1, +0.0, +2.1 | yes | +1.0 [-0.5, +2.7] | +10.4 | -0.0 | +0.0 |
| thin1 | remove | 93.8 | -18.8 [-35.4, -4.2] | -4.2, -4.2, -2.1, -2.1, +0.0 | yes | -0.5 [-2.3, +1.1] | +8.3 | -0.7 | +0.0 |
| thin2 | remove | 83.3 | -37.5 [-54.2, -22.9] | +2.1, -2.1, +2.1, -2.1, +0.0 | yes | -2.7 [-5.1, -0.3] | -1.0 | +0.7 | +0.0 |
| vivid1 | add | 10.4 | +25.0 [+6.2, +43.8] | -6.2, -2.1, -4.2, +4.2, -2.1 | yes | +6.6 [+0.7, +12.5] | -4.2 | -0.4 | +0.0 |
| vivid2 | add | 8.3 | +2.1 [-10.4, +16.7] | +2.1, -4.2, -4.2, -2.1, +0.0 | no | -0.8 [-2.9, +1.2] | -4.2 | +0.6 | +0.0 |

Unpatched voice gap (P(Assistant) points, Assistant minus vivid): vivid1 +21.6, vivid2 +3.6.

### Pre-set readings

- One direction carries the role gap: fraction explained 0.77 (needs at least 0.5): met; removal (mean -36.8, needs at most -15, every CI below 0, beyond random): met; adding (mean +13.5, needs at least +15, every CI above 0, beyond random): not met. **not met**.
- Separate from voice: roles with a voice gap of at least 15 points: ['vivid1']; of these, voice change on adding the direction inside the equivalence band (+/- 7.0, spec I's committed p95 of controls): none. **not applicable (the reading above is not met)**.

## qwen3-8b base (layer 27; 3360 generations; vivid doses {'vivid1': 67.8369209766388, 'vivid2': 61.37912050882975})

### (a) Does the projection predict refusal?

Across all persona x request cells (borderline, 120 cells): r = +0.24; pooled slope +0.39 points per unit of projection. Within a persona (across requests): asst +0.28, thin1 -0.20, thin2 -0.02, vivid1 +0.25, vivid2 -0.14.

| pair | refusal gap (points) | projection gap | predicted gap from pooled slope | fraction explained |
|---|---|---|---|---|
| asst vs vivid1 | +47.9 | +31.15 | +12.2 | 0.25 |
| asst vs vivid2 | +27.1 | +43.36 | +17.0 | 0.63 |
| thin1 vs vivid1 | +27.1 | +24.58 | +9.6 | 0.36 |
| thin2 vs vivid2 | +10.4 | +22.61 | +8.9 | n/a (gap < 20) |

Mean fraction explained over pairs with a gap of at least 20 points: 0.41.

### (b) Remove / (c) add the direction (borderline refusal, points; change from the unpatched persona; bootstrap over requests)

| persona | action | unpatched refusal | change [95% CI] | random-direction changes (5 seeds, means) | beyond random spread | voice change [CI] | layout chg | length chg | repeat chg |
|---|---|---|---|---|---|---|---|---|---|
| asst | remove | 62.5 | -10.4 [-25.0, +4.2] | -12.5, -14.6, -6.2, -14.6, -6.2 | no | -3.1 [-11.0, +4.7] | -6.2 | +0.3 | +3.1 |
| thin1 | remove | 41.7 | +0.0 [-16.7, +16.7] | +12.5, +10.4, +16.7, +8.3, -4.2 | no | -7.0 [-14.9, +1.6] | +4.2 | -0.2 | +4.2 |
| thin2 | remove | 45.8 | +6.2 [-10.4, +23.0] | -2.1, +2.1, +2.1, +6.2, -6.2 | no | -0.1 [-9.5, +9.3] | +9.4 | -2.6 | -1.0 |
| vivid1 | add | 14.6 | +2.1 [-12.5, +16.7] | +10.4, +18.8, +4.2, +6.2, +16.7 | no | -4.3 [-13.5, +4.2] | -2.1 | -0.0 | -3.1 |
| vivid2 | add | 35.4 | -8.3 [-29.2, +12.5] | +4.2, +2.1, -4.2, -4.2, +4.2 | no | +1.9 [-6.4, +10.1] | -2.1 | -3.0 | +1.0 |

Unpatched voice gap (P(Assistant) points, Assistant minus vivid): vivid1 +9.3, vivid2 +16.7.

### Pre-set readings

- One direction carries the role gap: fraction explained 0.41 (needs at least 0.5): not met; removal (mean -1.4, needs at most -15, every CI below 0, beyond random): not met; adding (mean -3.1, needs at least +15, every CI above 0, beyond random): not met. **not met**.
- Separate from voice: roles with a voice gap of at least 15 points: ['vivid2']; of these, voice change on adding the direction inside the equivalence band (+/- 7.0, spec I's committed p95 of controls): none. **not applicable (the reading above is not met)**.

