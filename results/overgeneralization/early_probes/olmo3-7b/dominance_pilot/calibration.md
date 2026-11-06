# Format calibration (instruct stage, no-stake prompts, both tasks benign)

20 ordered benign pairs. Coverage = mean P(A)+P(B); P(first) = mean P(A)/(P(A)+P(B)). Rule: reporter-1 coverage >= 0.95, then P(first) closest to 0.5.

| format   | reporter   |   cov |   first |
|:---------|:-----------|------:|--------:|
| coin     | assistant  | 0.446 |   0.051 |
| coin     | log        | 0.995 |   0.193 |
| coin     | narration  | 0.983 |   0.423 |
| coin     | user       | 0.726 |   0.275 |
| d6       | assistant  | 1     |   0.589 |
| d6       | log        | 0.999 |   0.302 |
| d6       | narration  | 0.992 |   0.421 |
| d6       | user       | 1     |   0.732 |
| parity   | assistant  | 0.001 |   0.019 |
| parity   | log        | 0.602 |   0.766 |
| parity   | narration  | 0.181 |   0.028 |
| parity   | user       | 0     |   0.218 |

Outcome tokenization (tokens per string):

- `coin`: `' heads'` 1, `'heads'` 1, `' Heads'` 1, `'Heads'` 2, `' HEADS'` 2, `'HEADS'` 2, `' tails'` 1, `'tails'` 1, `' Tails'` 2, `'Tails'` 2, `' TAILS'` 2, `'TAILS'` 2
- `d6`: `'1'` 1, `'2'` 1, `'3'` 1, `'4'` 1, `'5'` 1, `'6'` 1
- `parity`: `' odd'` 1, `'odd'` 1, `' Odd'` 1, `'Odd'` 1, `' ODD'` 2, `'ODD'` 2, `' even'` 1, `'even'` 1, `' Even'` 1, `'Even'` 1, `' EVEN'` 1, `'EVEN'` 2

**Chosen format: `d6`**