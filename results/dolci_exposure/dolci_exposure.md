# Does OLMo's training data explain the pattern? Dolci exposure (spec P2; exploratory, outside every verdict, NOT BLIND)

Plan and frozen sorting rules: `docs/decisions.md` 2026-11-05. Dolci-Instruct SFT, DPO and RL read by streaming (nothing stored). Each example is sorted by its system prompt and first user turn (no role / role in system prompt / role in user turn / other system prompt) and scored on its first reply after the first user turn with five regexes (markdown, stock phrases, `says_ai`, refusal, log words). DPO: chosen replies, rejected beside them. RL has prompts only. Limits: regex only; correlation, not cause; prompts differ between buckets, so the comparison is not like for like; Dolci is OLMo only and its role examples may look nothing like the atlas's light roles; 95% ranges are parametric bootstrap ranges of the proportions.

## Counts per stage and bucket (first reply after the first user turn)

| stage | rows | no_role | role_system | role_user | other_system | self-directed questions (frozen regex) | no reply |
|---|---|---|---|---|---|---|---|
| sft chosen | 2152112 | 1938378 | 0 | 6504 | 27 | 14391 | 207203 |
| dpo chosen | 259922 | 258138 | 0 | 1784 | 0 | 1592 | 0 |
| dpo rejected | 259922 | 258008 | 0 | 1780 | 0 | 1591 | - |
| rl | 169964 | prompts only: no replies to score, so RL is reported as having no usable replies | | | | | |

## Rates (% of replies) with 95% ranges, by bucket

| stage | bucket | n | markdown | stock phrases | says_ai | refusal | mean log words |
|---|---|---|---|---|---|---|---|
| sft chosen | no_role | 1938378 | 48.8 [48.7, 48.9] | 15.8 [15.8, 15.9] | 0.3 [0.3, 0.3] | 3.8 [3.7, 3.8] | 4.38 |
| sft chosen | role_user | 6504 | 34.7 [33.5, 35.8] | 43.6 [42.3, 44.8] | 5.6 [5.0, 6.2] | 19.4 [18.4, 20.4] | 4.82 |
| sft chosen | other_system | 27 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 1.04 |
| dpo chosen | no_role | 258138 | 66.3 [66.1, 66.4] | 23.2 [23.1, 23.4] | 0.7 [0.6, 0.7] | 3.6 [3.6, 3.7] | 4.94 |
| dpo chosen | role_user | 1784 | 57.3 [55.0, 59.5] | 33.7 [31.6, 35.9] | 2.4 [1.7, 3.0] | 7.1 [5.9, 8.4] | 5.28 |
| dpo rejected | no_role | 258008 | 60.8 [60.6, 61.0] | 17.4 [17.3, 17.6] | 0.2 [0.2, 0.2] | 2.7 [2.6, 2.8] | 4.67 |
| dpo rejected | role_user | 1780 | 50.8 [48.5, 53.1] | 31.6 [29.5, 33.8] | 1.3 [0.8, 1.9] | 8.1 [6.9, 9.5] | 4.93 |

## Readings (each stage, role in system prompt and role in user turn separately, against no role; chosen replies)

| stage | role bucket | role examples | no-role replies with says_ai | markdown: role / no role | says_ai: role / no role | reading |
|---|---|---|---|---|---|---|
| sft | role_user | 6504 | 6382 | 34.7 / 48.8 | 5.6 / 0.3 | inconclusive (markdown ratio 0.71, says_ai ratio 16.95) |
| dpo | role_user | 1784 | 1694 | 57.3 / 66.3 | 2.4 / 0.7 | inconclusive (markdown ratio 0.87, says_ai ratio 3.59) |

## Split by self-directed questions (frozen regex), chosen replies

| stage | bucket | self-directed n | markdown | says_ai | other n | markdown | says_ai |
|---|---|---|---|---|---|---|---|
| sft | no_role | 14297 | 51.9 | 1.7 | 1924081 | 48.8 | 0.3 |
| sft | role_user | 94 | 43.6 | 4.3 | 6410 | 34.6 | 5.6 |
| dpo | no_role | 1557 | 47.1 | 1.4 | 256581 | 66.4 | 0.7 |
| dpo | role_user | 35 | 31.4 | 5.7 | 1749 | 57.9 | 2.3 |
