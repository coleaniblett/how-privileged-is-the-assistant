# Cross-family comparison: Gemma 4 31B vs Qwen3-8B (change from own base)

Per-token nats are compared within a family only; the cross-family rule uses primary bits per byte. Differences are paired over the 30 texts per source that both families were scored on.

| user source | family | nats/token change | bits/byte change |
|---|---|---|---|
| wildchat | gemma | +2.943 [+2.410, +3.442] | +1.096 [+0.853, +1.384] |
| wildchat | qwen | +0.436 [+0.258, +0.617] | +0.144 [+0.091, +0.200] |
| oasst1 | gemma | +3.630 [+2.913, +4.318] | +1.164 [+0.943, +1.370] |
| oasst1 | qwen | +0.385 [+0.248, +0.516] | +0.125 [+0.083, +0.165] |

## Rule A: Gemma meets criterion 1 and Qwen does not

Gemma meets it in both user sources: True ([True, True]). Qwen meets it in both user sources: True ([True, True]). **Rule A NOT MET.**

## Rule B: Gemma's bits/byte change exceeds Qwen's, CI of the paired difference excluding zero, both sources

| user source | texts | Gemma change - Qwen change (bits/byte) |
|---|---|---|
| wildchat | 30 | +0.952 [+0.674, +1.248] |
| oasst1 | 30 | +1.039 [+0.814, +1.257] |

**Rule B MET.**

## The pilot discriminates (A or B)

