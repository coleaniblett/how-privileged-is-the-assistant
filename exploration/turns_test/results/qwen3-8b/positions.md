# qwen3-8b: change from base of the frame cost (tagged - plain), by content-position bin

Mean over conversations that reach the bin, nats per token; descriptive, no CIs.

|                      |   1-4 |   5-19 |   20-59 |   60+ |
|:---------------------|------:|-------:|--------:|------:|
| ('oasst1', 'rA')     | -0.28 |   0.46 |    0.33 |  0.19 |
| ('oasst1', 'rB')     |  1.38 |   0.86 |    0.47 |  0.18 |
| ('oasst1', 'u1')     |  0.23 |   0.29 |    0.3  |  0.48 |
| ('oasst1', 'u2A')    |  0.4  |   0.33 |    0.28 |  0.18 |
| ('oasst1', 'u2C')    |  0.04 |   0.19 |    0.36 |  0.32 |
| ('ultrachat', 'rA')  |  0.25 |   0.46 |    0.31 |  0.11 |
| ('ultrachat', 'rB')  |  0.89 |   0.92 |    0.65 |  0.24 |
| ('ultrachat', 'u1')  |  0.25 |   0.42 |    0.34 |  0.23 |
| ('ultrachat', 'u2A') |  0.53 |   0.31 |    0.24 |  0.22 |
| ('ultrachat', 'u2C') |  0.11 |   0.24 |    0.22 |  0.11 |
| ('wildchat', 'rA')   |  0.27 |   0.33 |    0.32 |  0.18 |
| ('wildchat', 'rB')   |  1.19 |   0.8  |    0.49 |  0.24 |
| ('wildchat', 'u1')   |  0.36 |   0.43 |    0.31 |  0.33 |
| ('wildchat', 'u2A')  |  0.2  |   0.58 |    0.36 |  0.18 |
| ('wildchat', 'u2C')  |  0.23 |   0.19 |    0.3  |  0.22 |
