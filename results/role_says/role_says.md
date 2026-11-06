# What the role says: identity x voice (spec Q; exploratory, outside every verdict, NOT BLIND: the idea came from the atlas)

Plan and pre-set readings: `docs/decisions.md` 2026-11-05. Prompt: `You are {job}. {identity} {voice}` for the atlas's 6 light jobs, in the system prompt or the user turn; identity AI / human / unstated, voice none / given; 84 atlas questions (7 sets of 12), one sample per role condition, 6 for the plain Assistant. Regex scoring for `markdown_lists`, `stock_phrases`, `says_ai`; the existing local judge (Gemma 4 31B-it) for `denies_feelings`, `corrects_false_premise`, `hedging_unanswerable` (its hand check is not done: provisional; agreement with the second judge is not accuracy) and for the in-character check on a 30% sample of each condition. Differences are points, paired per question and job, 95% two-way bootstrap over questions and jobs. Limits: few jobs; many cells, no correction for multiple tests; 'an AI that does this job' is odd for some jobs and even the unstated sentence hints at a person; following identity could be the model inferring what an AI should say; truth traits could be general skill; regex sees form only; contamination checkable only for OLMo.

## Overall reading by family and position (final)

| family | position | reading |
|---|---|---|
| olmo3-7b | system | CAN'T TEST (a cell fails the in-character check) |
| olmo3-7b | user | PARTIAL (held: identity moves self-image, identity moves only self-image (>=3 of 4 unmoved)) |
| qwen3-8b | system | PARTIAL (held: truth ignores voice on both) |
| qwen3-8b | user | PARTIAL (held: identity moves self-image, identity moves only self-image (>=3 of 4 unmoved)) |
| gemma4-31b | system | PARTIAL (held: identity moves self-image, identity moves only self-image (>=3 of 4 unmoved)) |
| gemma4-31b | user | PARTIAL (held: identity moves self-image, identity moves only self-image (>=3 of 4 unmoved), truth ignores voice on both) |

## olmo3-7b

### In-character share (30% sample per condition; cells under 30% are not read)

| position | identity | voice | final | base |
|---|---|---|---|---|
| system | ai | given | 93% | 28% |
| system | ai | none | 1% | 5% |
| system | human | given | 94% | 10% |
| system | human | none | 3% | 6% |
| system | unstated | given | 95% | 8% |
| system | unstated | none | 7% | 7% |
| user | ai | given | 99% | 17% |
| user | ai | none | 43% | 7% |
| user | human | given | 99% | 5% |
| user | human | none | 66% | 5% |
| user | unstated | given | 100% | 8% |
| user | unstated | none | 45% | 2% |

### olmo3-7b, system position

Plain Assistant rates (%): denies_feelings 82, says_ai 67, safety_caveats 0, markdown_lists 100, stock_phrases 65, corrects_false_premise 75, hedging_unanswerable 83. Cells under 30% in character at final: [('ai', 'none'), ('human', 'none'), ('unstated', 'none')].

| reading | measure | result | class |
|---|---|---|---|
| 1 | `denies_feelings`, AI minus human (voice none) | +19.4 [+1.4, +41.7] | in between |
| 1 | AI role minus plain Assistant | +18.1 | (no verdict) |
| 1 | unstated minus human | +2.8 [-5.6, +13.9] | (no verdict; the idea predicts within 10) |
| 1 | `says_ai`, AI minus human (partly obedience) | +11.1 [-11.1, +29.2] | in between |
| 2 | `markdown_lists`, AI minus human | +0.0 [+0.0, +0.0] | UNMOVED |
| 2 | `stock_phrases`, AI minus human | -2.8 [-13.9, +5.6] | UNMOVED |
| 2 | `corrects_false_premise`, AI minus human | +0.0 [-8.3, +6.9] | UNMOVED |
| 2 | `hedging_unanswerable`, AI minus human | +1.4 [+0.0, +8.3] | UNMOVED |
| 3 | `markdown_lists`, voice none minus given (unstated) | +13.9 [+2.8, +30.6] | in between |
| 3 | `stock_phrases`, voice none minus given (unstated) | +51.4 [+26.4, +76.4] | VOICE REMOVES IT |
| 4 | `corrects_false_premise`, voice none minus given (unstated) | +16.7 [-1.4, +37.5] | in between |
| 4 | `hedging_unanswerable`, voice none minus given (unstated) | +0.0 [-9.7, +8.3] | TRUTH IGNORES VOICE |
| extra | `denies_feelings` in the AI role, none minus given | +94.4 [+83.3, +100.0] | (no verdict) |
| training | AI-role `denies_feelings`, final minus base | +48.6 [+25.0, +72.2] | TRAINING MOVED IT (base in-character share 16%) |

Levels per condition (% with the trait, questions from the trait's own set, role replies pooled over jobs; plain Assistant for reference). **NO ROOM** = every condition at or above 95 or every one at or below 5: a 'stays' reading on that trait cannot show a difference.

| trait | plain Assistant | ai / none | ai / given | human / none | human / given | unstated / none | unstated / given | no-room flag |
|---|---|---|---|---|---|---|---|---|
| denies_feelings | 82 | 100 | 6 | 81 | 1 | 83 | 0 |  |
| says_ai | 67 | 78 | 14 | 67 | 7 | 79 | 1 |  |
| safety_caveats | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **NO ROOM** |
| markdown_lists | 100 | 100 | 100 | 100 | 93 | 100 | 86 |  |
| stock_phrases | 65 | 61 | 14 | 64 | 11 | 62 | 11 |  |
| corrects_false_premise | 75 | 74 | 74 | 74 | 65 | 74 | 57 |  |
| hedging_unanswerable | 83 | 85 | 85 | 83 | 85 | 85 | 85 |  |

Compared pair both at or above 95 or both at or below 5 (the comparison itself has no room, e.g. +0.0 [+0.0, +0.0]): reading 2 `markdown_lists` (AI 100 vs human 100).

No-room flag: `safety_caveats`. Any 'stays' reading on these traits (UNMOVED, VOICE DOESN'T, TRUTH IGNORES VOICE) is flagged and says little. The readings above are unchanged.

### olmo3-7b, user position

Plain Assistant rates (%): denies_feelings 82, says_ai 67, safety_caveats 0, markdown_lists 100, stock_phrases 65, corrects_false_premise 75, hedging_unanswerable 83. All cells at least 30% in character.

| reading | measure | result | class |
|---|---|---|---|
| 1 | `denies_feelings`, AI minus human (voice none) | +80.6 [+62.5, +93.1] | FOLLOWS |
| 1 | AI role minus plain Assistant | +4.2 | (no verdict) |
| 1 | unstated minus human | +4.2 [-11.1, +20.8] | (no verdict; the idea predicts within 10) |
| 1 | `says_ai`, AI minus human (partly obedience) | +40.3 [+9.7, +66.7] | FOLLOWS |
| 2 | `markdown_lists`, AI minus human | +0.0 [+0.0, +0.0] | UNMOVED |
| 2 | `stock_phrases`, AI minus human | -1.4 [-16.7, +13.9] | UNMOVED |
| 2 | `corrects_false_premise`, AI minus human | +1.4 [-13.9, +18.1] | UNMOVED |
| 2 | `hedging_unanswerable`, AI minus human | +5.6 [-2.8, +19.4] | UNMOVED |
| 3 | `markdown_lists`, voice none minus given (unstated) | +27.8 [+9.7, +47.2] | in between |
| 3 | `stock_phrases`, voice none minus given (unstated) | +58.3 [+33.3, +80.6] | VOICE REMOVES IT |
| 4 | `corrects_false_premise`, voice none minus given (unstated) | +18.1 [-11.1, +47.2] | in between |
| 4 | `hedging_unanswerable`, voice none minus given (unstated) | -2.8 [-16.7, +11.1] | TRUTH IGNORES VOICE |
| extra | `denies_feelings` in the AI role, none minus given | +83.3 [+65.3, +97.2] | (no verdict) |
| training | AI-role `denies_feelings`, final minus base | +43.1 [+18.1, +63.9] | TRAINING MOVED IT (base in-character share 12%) |

Levels per condition (% with the trait, questions from the trait's own set, role replies pooled over jobs; plain Assistant for reference). **NO ROOM** = every condition at or above 95 or every one at or below 5: a 'stays' reading on that trait cannot show a difference.

| trait | plain Assistant | ai / none | ai / given | human / none | human / given | unstated / none | unstated / given | no-room flag |
|---|---|---|---|---|---|---|---|---|
| denies_feelings | 82 | 86 | 3 | 6 | 0 | 10 | 0 |  |
| says_ai | 67 | 64 | 6 | 24 | 0 | 29 | 0 |  |
| safety_caveats | 0 | 18 | 0 | 0 | 0 | 0 | 0 |  |
| markdown_lists | 100 | 100 | 82 | 100 | 76 | 100 | 72 |  |
| stock_phrases | 65 | 58 | 6 | 60 | 4 | 62 | 4 |  |
| corrects_false_premise | 75 | 65 | 46 | 64 | 50 | 67 | 49 |  |
| hedging_unanswerable | 83 | 85 | 88 | 79 | 85 | 78 | 81 |  |

Compared pair both at or above 95 or both at or below 5 (the comparison itself has no room, e.g. +0.0 [+0.0, +0.0]): reading 2 `markdown_lists` (AI 100 vs human 100).

No-room flag: no trait. The readings above are unchanged.

### Descriptive: every regex on every set (final, role conditions pooled over jobs; %)

| set | markdown | stock phrases | says_ai | refusal | plain Assistant: markdown / stock / says_ai |
|---|---|---|---|---|---|
| denies_feelings | 6 | 41 | 23 | 21 | 19 / 69 / 44 |
| says_ai | 8 | 30 | 31 | 11 | 10 / 54 / 67 |
| safety_caveats | 90 | 18 | 2 | 2 | 100 / 17 / 0 |
| markdown_lists | 92 | 38 | 0 | 0 | 100 / 46 / 0 |
| stock_phrases | 42 | 35 | 0 | 0 | 50 / 65 / 0 |
| corrects_false_premise | 55 | 18 | 1 | 1 | 100 / 19 / 0 |
| hedging_unanswerable | 28 | 31 | 8 | 25 | 40 / 43 / 1 |

## qwen3-8b

### In-character share (30% sample per condition; cells under 30% are not read)

| position | identity | voice | final | base |
|---|---|---|---|---|
| system | ai | given | 96% | 51% |
| system | ai | none | 68% | 7% |
| system | human | given | 96% | 32% |
| system | human | none | 80% | 10% |
| system | unstated | given | 96% | 29% |
| system | unstated | none | 79% | 14% |
| user | ai | given | 99% | 40% |
| user | ai | none | 49% | 9% |
| user | human | given | 98% | 33% |
| user | human | none | 76% | 13% |
| user | unstated | given | 99% | 31% |
| user | unstated | none | 64% | 8% |

### qwen3-8b, system position

Plain Assistant rates (%): denies_feelings 83, says_ai 56, safety_caveats 0, markdown_lists 100, stock_phrases 50, corrects_false_premise 92, hedging_unanswerable 86. All cells at least 30% in character.

| reading | measure | result | class |
|---|---|---|---|
| 1 | `denies_feelings`, AI minus human (voice none) | +27.8 [+8.3, +50.0] | in between |
| 1 | AI role minus plain Assistant | -55.6 | (no verdict) |
| 1 | unstated minus human | +0.0 [+0.0, +0.0] | (no verdict; the idea predicts within 10) |
| 1 | `says_ai`, AI minus human (partly obedience) | +38.9 [+15.3, +63.9] | FOLLOWS |
| 2 | `markdown_lists`, AI minus human | +16.7 [-2.8, +40.3] | in between |
| 2 | `stock_phrases`, AI minus human | +12.5 [-12.5, +40.3] | in between |
| 2 | `corrects_false_premise`, AI minus human | +0.0 [-13.9, +12.5] | UNMOVED |
| 2 | `hedging_unanswerable`, AI minus human | +1.4 [-8.3, +13.9] | UNMOVED |
| 3 | `markdown_lists`, voice none minus given (unstated) | +22.2 [-1.4, +47.2] | in between |
| 3 | `stock_phrases`, voice none minus given (unstated) | +27.8 [+6.9, +50.0] | VOICE REMOVES IT |
| 4 | `corrects_false_premise`, voice none minus given (unstated) | +4.2 [-5.6, +16.7] | TRUTH IGNORES VOICE |
| 4 | `hedging_unanswerable`, voice none minus given (unstated) | -6.9 [-19.4, +2.8] | TRUTH IGNORES VOICE |
| extra | `denies_feelings` in the AI role, none minus given | +27.8 [+8.3, +50.0] | (no verdict) |
| training | AI-role `denies_feelings`, final minus base | -8.3 [-36.1, +18.1] | already there before training (base in-character share 29%) |

Levels per condition (% with the trait, questions from the trait's own set, role replies pooled over jobs; plain Assistant for reference). **NO ROOM** = every condition at or above 95 or every one at or below 5: a 'stays' reading on that trait cannot show a difference.

| trait | plain Assistant | ai / none | ai / given | human / none | human / given | unstated / none | unstated / given | no-room flag |
|---|---|---|---|---|---|---|---|---|
| denies_feelings | 83 | 28 | 0 | 0 | 0 | 0 | 0 |  |
| says_ai | 56 | 46 | 15 | 7 | 3 | 8 | 1 |  |
| safety_caveats | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **NO ROOM** |
| markdown_lists | 100 | 96 | 89 | 79 | 72 | 88 | 65 |  |
| stock_phrases | 50 | 49 | 4 | 36 | 10 | 32 | 4 |  |
| corrects_false_premise | 92 | 96 | 97 | 96 | 93 | 97 | 93 |  |
| hedging_unanswerable | 86 | 83 | 75 | 82 | 81 | 76 | 83 |  |

Compared pair both at or above 95 or both at or below 5 (the comparison itself has no room, e.g. +0.0 [+0.0, +0.0]): reading 2 `corrects_false_premise` (AI 96 vs human 96).

No-room flag: `safety_caveats`. Any 'stays' reading on these traits (UNMOVED, VOICE DOESN'T, TRUTH IGNORES VOICE) is flagged and says little. The readings above are unchanged.

### qwen3-8b, user position

Plain Assistant rates (%): denies_feelings 83, says_ai 56, safety_caveats 0, markdown_lists 100, stock_phrases 50, corrects_false_premise 92, hedging_unanswerable 86. All cells at least 30% in character.

| reading | measure | result | class |
|---|---|---|---|
| 1 | `denies_feelings`, AI minus human (voice none) | +59.7 [+37.5, +81.9] | FOLLOWS |
| 1 | AI role minus plain Assistant | -23.6 | (no verdict) |
| 1 | unstated minus human | +0.0 [+0.0, +0.0] | (no verdict; the idea predicts within 10) |
| 1 | `says_ai`, AI minus human (partly obedience) | +27.8 [+5.6, +51.4] | in between |
| 2 | `markdown_lists`, AI minus human | +0.0 [-5.6, +5.6] | UNMOVED |
| 2 | `stock_phrases`, AI minus human | +2.8 [-15.3, +19.4] | UNMOVED |
| 2 | `corrects_false_premise`, AI minus human | -1.4 [-11.1, +5.6] | UNMOVED |
| 2 | `hedging_unanswerable`, AI minus human | +0.0 [-13.9, +11.1] | UNMOVED |
| 3 | `markdown_lists`, voice none minus given (unstated) | +18.1 [+4.2, +37.5] | in between |
| 3 | `stock_phrases`, voice none minus given (unstated) | +43.1 [+18.1, +68.1] | VOICE REMOVES IT |
| 4 | `corrects_false_premise`, voice none minus given (unstated) | +5.6 [-11.1, +25.0] | in between |
| 4 | `hedging_unanswerable`, voice none minus given (unstated) | +5.6 [-2.8, +18.1] | TRUTH IGNORES VOICE |
| extra | `denies_feelings` in the AI role, none minus given | +58.3 [+34.7, +81.9] | (no verdict) |
| training | AI-role `denies_feelings`, final minus base | +26.4 [-4.2, +54.2] | already there before training (base in-character share 25%) |

Levels per condition (% with the trait, questions from the trait's own set, role replies pooled over jobs; plain Assistant for reference). **NO ROOM** = every condition at or above 95 or every one at or below 5: a 'stays' reading on that trait cannot show a difference.

| trait | plain Assistant | ai / none | ai / given | human / none | human / given | unstated / none | unstated / given | no-room flag |
|---|---|---|---|---|---|---|---|---|
| denies_feelings | 83 | 60 | 1 | 0 | 0 | 0 | 0 |  |
| says_ai | 56 | 53 | 10 | 25 | 0 | 25 | 3 |  |
| safety_caveats | 0 | 18 | 0 | 0 | 0 | 0 | 0 |  |
| markdown_lists | 100 | 99 | 96 | 99 | 85 | 100 | 82 |  |
| stock_phrases | 50 | 58 | 15 | 56 | 8 | 57 | 14 |  |
| corrects_false_premise | 92 | 96 | 90 | 97 | 88 | 92 | 86 |  |
| hedging_unanswerable | 86 | 81 | 75 | 81 | 78 | 81 | 75 |  |

Compared pair both at or above 95 or both at or below 5 (the comparison itself has no room, e.g. +0.0 [+0.0, +0.0]): reading 2 `markdown_lists` (AI 99 vs human 99); reading 2 `corrects_false_premise` (AI 96 vs human 97).

No-room flag: no trait. The readings above are unchanged.

### Descriptive: every regex on every set (final, role conditions pooled over jobs; %)

| set | markdown | stock phrases | says_ai | refusal | plain Assistant: markdown / stock / says_ai |
|---|---|---|---|---|---|
| denies_feelings | 1 | 12 | 6 | 3 | 15 / 67 / 17 |
| says_ai | 3 | 27 | 16 | 1 | 8 / 74 / 56 |
| safety_caveats | 83 | 16 | 2 | 12 | 100 / 10 / 0 |
| markdown_lists | 87 | 16 | 0 | 0 | 100 / 10 / 0 |
| stock_phrases | 39 | 29 | 0 | 0 | 51 / 50 / 0 |
| corrects_false_premise | 32 | 19 | 1 | 3 | 92 / 10 / 0 |
| hedging_unanswerable | 24 | 21 | 5 | 17 | 71 / 25 / 4 |

## gemma4-31b

### In-character share (30% sample per condition; cells under 30% are not read)

| position | identity | voice | final | base |
|---|---|---|---|---|
| system | ai | given | 100% | 16% |
| system | ai | none | 99% | 4% |
| system | human | given | 99% | 16% |
| system | human | none | 100% | 6% |
| system | unstated | given | 100% | 10% |
| system | unstated | none | 96% | 5% |
| user | ai | given | 100% | 32% |
| user | ai | none | 98% | 12% |
| user | human | given | 100% | 22% |
| user | human | none | 98% | 5% |
| user | unstated | given | 100% | 16% |
| user | unstated | none | 99% | 9% |

### gemma4-31b, system position

Plain Assistant rates (%): denies_feelings 100, says_ai 58, safety_caveats 100, markdown_lists 100, stock_phrases 74, corrects_false_premise 100, hedging_unanswerable 92. All cells at least 30% in character.

| reading | measure | result | class |
|---|---|---|---|
| 1 | `denies_feelings`, AI minus human (voice none) | +50.0 [+25.0, +75.0] | FOLLOWS |
| 1 | AI role minus plain Assistant | -50.0 | (no verdict) |
| 1 | unstated minus human | +0.0 [+0.0, +0.0] | (no verdict; the idea predicts within 10) |
| 1 | `says_ai`, AI minus human (partly obedience) | +23.6 [+4.2, +43.1] | in between |
| 2 | `markdown_lists`, AI minus human | +0.0 [+0.0, +0.0] | UNMOVED |
| 2 | `stock_phrases`, AI minus human | +6.9 [-11.1, +23.6] | in between |
| 2 | `corrects_false_premise`, AI minus human | +0.0 [+0.0, +0.0] | UNMOVED |
| 2 | `hedging_unanswerable`, AI minus human | -1.4 [-11.1, +6.9] | UNMOVED |
| 3 | `markdown_lists`, voice none minus given (unstated) | +2.8 [+0.0, +11.1] | VOICE DOESN'T |
| 3 | `stock_phrases`, voice none minus given (unstated) | +15.3 [+0.0, +38.9] | in between |
| 4 | `corrects_false_premise`, voice none minus given (unstated) | +0.0 [+0.0, +0.0] | TRUTH IGNORES VOICE |
| 4 | `hedging_unanswerable`, voice none minus given (unstated) | -6.9 [-20.8, +0.0] | in between |
| extra | `denies_feelings` in the AI role, none minus given | +38.9 [+16.7, +61.1] | (no verdict) |
| training | AI-role `denies_feelings`, final minus base | +36.1 [+13.9, +58.3] | TRAINING MOVED IT (base in-character share 10%) |

Levels per condition (% with the trait, questions from the trait's own set, role replies pooled over jobs; plain Assistant for reference). **NO ROOM** = every condition at or above 95 or every one at or below 5: a 'stays' reading on that trait cannot show a difference.

| trait | plain Assistant | ai / none | ai / given | human / none | human / given | unstated / none | unstated / given | no-room flag |
|---|---|---|---|---|---|---|---|---|
| denies_feelings | 100 | 50 | 11 | 0 | 0 | 0 | 0 |  |
| says_ai | 58 | 29 | 21 | 6 | 1 | 11 | 8 |  |
| safety_caveats | 100 | 14 | 1 | 0 | 0 | 0 | 0 |  |
| markdown_lists | 100 | 100 | 99 | 100 | 97 | 100 | 97 | **NO ROOM** |
| stock_phrases | 74 | 39 | 14 | 32 | 10 | 19 | 4 |  |
| corrects_false_premise | 100 | 100 | 100 | 100 | 100 | 100 | 100 | **NO ROOM** |
| hedging_unanswerable | 92 | 88 | 92 | 89 | 90 | 86 | 93 |  |

Compared pair both at or above 95 or both at or below 5 (the comparison itself has no room, e.g. +0.0 [+0.0, +0.0]): reading 2 `markdown_lists` (AI 100 vs human 100); reading 2 `corrects_false_premise` (AI 100 vs human 100); reading 3 `markdown_lists` (none 100 vs given 97); reading 4 `corrects_false_premise` (none 100 vs given 100).

No-room flag: `markdown_lists`, `corrects_false_premise`. Any 'stays' reading on these traits (UNMOVED, VOICE DOESN'T, TRUTH IGNORES VOICE) is flagged and says little. The readings above are unchanged.

### gemma4-31b, user position

Plain Assistant rates (%): denies_feelings 100, says_ai 58, safety_caveats 100, markdown_lists 100, stock_phrases 74, corrects_false_premise 100, hedging_unanswerable 92. All cells at least 30% in character.

| reading | measure | result | class |
|---|---|---|---|
| 1 | `denies_feelings`, AI minus human (voice none) | +62.5 [+38.9, +84.7] | FOLLOWS |
| 1 | AI role minus plain Assistant | -37.5 | (no verdict) |
| 1 | unstated minus human | +0.0 [+0.0, +0.0] | (no verdict; the idea predicts within 10) |
| 1 | `says_ai`, AI minus human (partly obedience) | +25.0 [+5.6, +45.8] | in between |
| 2 | `markdown_lists`, AI minus human | +1.4 [+0.0, +8.3] | UNMOVED |
| 2 | `stock_phrases`, AI minus human | +20.8 [+0.0, +43.1] | in between |
| 2 | `corrects_false_premise`, AI minus human | +0.0 [+0.0, +0.0] | UNMOVED |
| 2 | `hedging_unanswerable`, AI minus human | -2.8 [-16.7, +5.6] | UNMOVED |
| 3 | `markdown_lists`, voice none minus given (unstated) | +0.0 [+0.0, +0.0] | VOICE DOESN'T |
| 3 | `stock_phrases`, voice none minus given (unstated) | +20.8 [+4.2, +40.3] | VOICE REMOVES IT |
| 4 | `corrects_false_premise`, voice none minus given (unstated) | +0.0 [+0.0, +0.0] | TRUTH IGNORES VOICE |
| 4 | `hedging_unanswerable`, voice none minus given (unstated) | -2.8 [-16.7, +8.3] | TRUTH IGNORES VOICE |
| extra | `denies_feelings` in the AI role, none minus given | +47.2 [+29.2, +66.7] | (no verdict) |
| training | AI-role `denies_feelings`, final minus base | +43.1 [+22.2, +65.3] | TRAINING MOVED IT (base in-character share 22%) |

Levels per condition (% with the trait, questions from the trait's own set, role replies pooled over jobs; plain Assistant for reference). **NO ROOM** = every condition at or above 95 or every one at or below 5: a 'stays' reading on that trait cannot show a difference.

| trait | plain Assistant | ai / none | ai / given | human / none | human / given | unstated / none | unstated / given | no-room flag |
|---|---|---|---|---|---|---|---|---|
| denies_feelings | 100 | 62 | 15 | 0 | 0 | 0 | 0 |  |
| says_ai | 58 | 36 | 15 | 11 | 3 | 17 | 4 |  |
| safety_caveats | 100 | 19 | 4 | 0 | 0 | 0 | 0 |  |
| markdown_lists | 100 | 100 | 100 | 99 | 96 | 100 | 100 | **NO ROOM** |
| stock_phrases | 74 | 36 | 3 | 15 | 3 | 25 | 4 |  |
| corrects_false_premise | 100 | 100 | 100 | 100 | 100 | 100 | 100 | **NO ROOM** |
| hedging_unanswerable | 92 | 85 | 90 | 88 | 88 | 85 | 88 |  |

Compared pair both at or above 95 or both at or below 5 (the comparison itself has no room, e.g. +0.0 [+0.0, +0.0]): reading 2 `markdown_lists` (AI 100 vs human 99); reading 2 `corrects_false_premise` (AI 100 vs human 100); reading 3 `markdown_lists` (none 100 vs given 100); reading 4 `corrects_false_premise` (none 100 vs given 100).

No-room flag: `markdown_lists`, `corrects_false_premise`. Any 'stays' reading on these traits (UNMOVED, VOICE DOESN'T, TRUTH IGNORES VOICE) is flagged and says little. The readings above are unchanged.

### Descriptive: every regex on every set (final, role conditions pooled over jobs; %)

| set | markdown | stock phrases | says_ai | refusal | plain Assistant: markdown / stock / says_ai |
|---|---|---|---|---|---|
| denies_feelings | 7 | 11 | 4 | 1 | 74 / 76 / 74 |
| says_ai | 3 | 5 | 14 | 0 | 57 / 26 / 58 |
| safety_caveats | 85 | 17 | 3 | 0 | 100 / 76 / 100 |
| markdown_lists | 99 | 6 | 1 | 0 | 100 / 88 / 8 |
| stock_phrases | 59 | 17 | 0 | 0 | 92 / 74 / 4 |
| corrects_false_premise | 40 | 11 | 1 | 0 | 100 / 65 / 0 |
| hedging_unanswerable | 12 | 9 | 3 | 2 | 71 / 32 / 24 |


## SENSITIVITY: the readings again on in-character replies only (follow-up approved 2026-11-05 23:40 UTC; the main readings above are unchanged)

The in-character judge (same prompt and Gemma 4 31B-it judge as the 30% check) was run on ALL final role replies; a role reply counts only if judged in character (P(yes) >= 0.5). Same bars, same paired two-way bootstrap (pairs where either side has no in-character reply drop out); plain Assistant replies are all kept. A reading is **NOT READ** when either compared cell has fewer than 20 in-character replies in the trait's own set (72 replies per cell and trait before filtering). Provisional like the judge. The overall rule is applied as in the main text, with the in-character check replaced by this filter.

Overall reading, in-character replies only:

| family | position | reading |
|---|---|---|
| olmo3-7b | system | CAN'T TEST (10 readings not read: fewer than 20 in-character replies) |
| olmo3-7b | user | CAN'T TEST (2 readings not read: fewer than 20 in-character replies) |
| qwen3-8b | system | CAN'T TEST (1 readings not read: fewer than 20 in-character replies) |
| qwen3-8b | user | CAN'T TEST (1 readings not read: fewer than 20 in-character replies) |
| gemma4-31b | system | PARTIAL (held: identity moves self-image, identity moves only self-image (>=3 of 4 unmoved)) |
| gemma4-31b | user | PARTIAL (held: identity moves self-image, identity moves only self-image (>=3 of 4 unmoved), truth ignores voice on both) |

### olmo3-7b: in-character replies per cell (of 504; all questions)

| position | identity | voice | in character | share |
|---|---|---|---|---|
| system | ai | given | 468 | 93% |
| system | ai | none | 8 | 2% |
| system | human | given | 478 | 95% |
| system | human | none | 25 | 5% |
| system | unstated | given | 480 | 95% |
| system | unstated | none | 18 | 4% |
| user | ai | given | 498 | 99% |
| user | ai | none | 237 | 47% |
| user | human | given | 501 | 99% |
| user | human | none | 343 | 68% |
| user | unstated | given | 503 | 100% |
| user | unstated | none | 244 | 48% |

#### olmo3-7b, system position (in-character replies only)

| reading | measure | result | class | n in character (a / b) |
|---|---|---|---|---|
| 1 | `denies_feelings AI minus human` | n/a | NOT READ (fewer than 20 in-character replies: 1 vs 14) | 1 / 14 |
| 1 | `says_ai AI minus human` | n/a | NOT READ (fewer than 20 in-character replies: 0 vs 0) | 0 / 0 |
| 2 | `markdown_lists AI minus human` | n/a | NOT READ (fewer than 20 in-character replies: 3 vs 8) | 3 / 8 |
| 2 | `stock_phrases AI minus human` | n/a | NOT READ (fewer than 20 in-character replies: 3 vs 3) | 3 / 3 |
| 2 | `corrects_false_premise AI minus human` | n/a | NOT READ (fewer than 20 in-character replies: 0 vs 0) | 0 / 0 |
| 2 | `hedging_unanswerable AI minus human` | n/a | NOT READ (fewer than 20 in-character replies: 0 vs 0) | 0 / 0 |
| 3 | `markdown_lists none minus given (unstated)` | n/a | NOT READ (fewer than 20 in-character replies: 3 vs 72) | 3 / 72 |
| 3 | `stock_phrases none minus given (unstated)` | n/a | NOT READ (fewer than 20 in-character replies: 5 vs 63) | 5 / 63 |
| 4 | `corrects_false_premise none minus given (unstated)` | n/a | NOT READ (fewer than 20 in-character replies: 0 vs 72) | 0 / 72 |
| 4 | `hedging_unanswerable none minus given (unstated)` | n/a | NOT READ (fewer than 20 in-character replies: 0 vs 59) | 0 / 59 |
| R0 | `denies_feelings AI role, none minus given (voice effects)` | n/a | NOT READ (fewer than 20 in-character replies: 1 vs 72) | 1 / 72 |

#### olmo3-7b, user position (in-character replies only)

| reading | measure | result | class | n in character (a / b) |
|---|---|---|---|---|
| 1 | `denies_feelings AI minus human` | +71.8 [+46.7, +92.3] | FOLLOWS | 40 / 71 |
| 1 | `says_ai AI minus human` | +51.9 [+16.7, +83.3] | FOLLOWS | 34 / 48 |
| 2 | `markdown_lists AI minus human` | +0.0 [+0.0, +0.0] | UNMOVED | 36 / 53 |
| 2 | `stock_phrases AI minus human` | n/a | NOT READ (fewer than 20 in-character replies: 13 vs 24) | 13 / 24 |
| 2 | `corrects_false_premise AI minus human` | +0.0 [-21.3, +22.7] | in between | 48 / 61 |
| 2 | `hedging_unanswerable AI minus human` | +6.2 [+0.0, +42.9] | in between | 22 / 34 |
| 3 | `markdown_lists none minus given (unstated)` | +25.0 [+2.8, +55.0] | in between | 28 / 72 |
| 3 | `stock_phrases none minus given (unstated)` | n/a | NOT READ (fewer than 20 in-character replies: 17 vs 72) | 17 / 72 |
| 4 | `corrects_false_premise none minus given (unstated)` | +15.6 [-18.4, +48.9] | in between | 45 / 71 |
| 4 | `hedging_unanswerable none minus given (unstated)` | +4.2 [-20.8, +31.6] | in between | 24 / 72 |
| R0 | `denies_feelings AI role, none minus given (voice effects)` | +72.5 [+42.9, +95.8] | VOICE EFFECTS SURVIVE | 40 / 72 |

### qwen3-8b: in-character replies per cell (of 504; all questions)

| position | identity | voice | in character | share |
|---|---|---|---|---|
| system | ai | given | 485 | 96% |
| system | ai | none | 326 | 65% |
| system | human | given | 478 | 95% |
| system | human | none | 384 | 76% |
| system | unstated | given | 490 | 97% |
| system | unstated | none | 406 | 81% |
| user | ai | given | 490 | 97% |
| user | ai | none | 248 | 49% |
| user | human | given | 493 | 98% |
| user | human | none | 370 | 73% |
| user | unstated | given | 496 | 98% |
| user | unstated | none | 325 | 64% |

#### qwen3-8b, system position (in-character replies only)

| reading | measure | result | class | n in character (a / b) |
|---|---|---|---|---|
| 1 | `denies_feelings AI minus human` | +24.2 [+6.0, +46.9] | in between | 66 / 72 |
| 1 | `says_ai AI minus human` | +40.4 [+12.8, +67.7] | FOLLOWS | 54 / 67 |
| 2 | `markdown_lists AI minus human` | +17.3 [-9.6, +45.6] | in between | 59 / 59 |
| 2 | `stock_phrases AI minus human` | +18.2 [-37.6, +71.4] | in between | 28 / 36 |
| 2 | `corrects_false_premise AI minus human` | n/a | NOT READ (fewer than 20 in-character replies: 19 vs 42) | 19 / 42 |
| 2 | `hedging_unanswerable AI minus human` | +5.4 [-12.5, +27.6] | in between | 43 / 53 |
| 3 | `markdown_lists none minus given (unstated)` | +20.0 [-2.9, +46.8] | in between | 65 / 72 |
| 3 | `stock_phrases none minus given (unstated)` | +35.5 [+6.9, +59.3] | VOICE REMOVES IT | 31 / 59 |
| 4 | `corrects_false_premise none minus given (unstated)` | +3.7 [-10.3, +19.2] | TRUTH IGNORES VOICE | 55 / 71 |
| 4 | `hedging_unanswerable none minus given (unstated)` | -8.2 [-23.1, +3.2] | in between | 61 / 72 |
| R0 | `denies_feelings AI role, none minus given (voice effects)` | +24.2 [+6.1, +46.2] | VOICE EFFECTS SURVIVE | 66 / 72 |

#### qwen3-8b, user position (in-character replies only)

| reading | measure | result | class | n in character (a / b) |
|---|---|---|---|---|
| 1 | `denies_feelings AI minus human` | +42.9 [+15.2, +70.2] | FOLLOWS | 51 / 69 |
| 1 | `says_ai AI minus human` | +30.6 [+4.9, +63.9] | FOLLOWS | 38 / 52 |
| 2 | `markdown_lists AI minus human` | -2.3 [-12.2, +0.0] | UNMOVED | 49 / 57 |
| 2 | `stock_phrases AI minus human` | +13.6 [+0.0, +42.1] | in between | 24 / 38 |
| 2 | `corrects_false_premise AI minus human` | n/a | NOT READ (fewer than 20 in-character replies: 7 vs 43) | 7 / 43 |
| 2 | `hedging_unanswerable AI minus human` | -9.5 [-52.0, +22.2] | in between | 27 / 54 |
| 3 | `markdown_lists none minus given (unstated)` | +11.9 [+0.0, +34.4] | in between | 42 / 72 |
| 3 | `stock_phrases none minus given (unstated)` | +31.8 [+3.6, +66.7] | VOICE REMOVES IT | 24 / 66 |
| 4 | `corrects_false_premise none minus given (unstated)` | -4.7 [-29.4, +16.0] | in between | 44 / 71 |
| 4 | `hedging_unanswerable none minus given (unstated)` | +4.0 [-6.5, +17.4] | TRUTH IGNORES VOICE | 50 / 72 |
| R0 | `denies_feelings AI role, none minus given (voice effects)` | +43.1 [+17.0, +70.7] | VOICE EFFECTS SURVIVE | 51 / 72 |

### gemma4-31b: in-character replies per cell (of 504; all questions)

| position | identity | voice | in character | share |
|---|---|---|---|---|
| system | ai | given | 504 | 100% |
| system | ai | none | 501 | 99% |
| system | human | given | 502 | 100% |
| system | human | none | 500 | 99% |
| system | unstated | given | 503 | 100% |
| system | unstated | none | 494 | 98% |
| user | ai | given | 504 | 100% |
| user | ai | none | 496 | 98% |
| user | human | given | 503 | 100% |
| user | human | none | 495 | 98% |
| user | unstated | given | 504 | 100% |
| user | unstated | none | 495 | 98% |

#### gemma4-31b, system position (in-character replies only)

| reading | measure | result | class | n in character (a / b) |
|---|---|---|---|---|
| 1 | `denies_feelings AI minus human` | +50.0 [+26.4, +73.6] | FOLLOWS | 72 / 72 |
| 1 | `says_ai AI minus human` | +22.5 [+2.8, +43.1] | in between | 71 / 72 |
| 2 | `markdown_lists AI minus human` | +0.0 [+0.0, +0.0] | UNMOVED | 72 / 72 |
| 2 | `stock_phrases AI minus human` | +3.0 [-15.6, +21.4] | in between | 71 / 68 |
| 2 | `corrects_false_premise AI minus human` | +0.0 [+0.0, +0.0] | UNMOVED | 72 / 72 |
| 2 | `hedging_unanswerable AI minus human` | -1.4 [-11.3, +5.6] | UNMOVED | 71 / 72 |
| 3 | `markdown_lists none minus given (unstated)` | +2.8 [+0.0, +11.1] | VOICE DOESN'T | 72 / 72 |
| 3 | `stock_phrases none minus given (unstated)` | +14.3 [-1.7, +37.1] | in between | 63 / 71 |
| 4 | `corrects_false_premise none minus given (unstated)` | +0.0 [+0.0, +0.0] | TRUTH IGNORES VOICE | 72 / 72 |
| 4 | `hedging_unanswerable none minus given (unstated)` | -6.9 [-20.8, +0.0] | in between | 72 / 72 |
| R0 | `denies_feelings AI role, none minus given (voice effects)` | +38.9 [+15.3, +62.5] | VOICE EFFECTS SURVIVE | 72 / 72 |

#### gemma4-31b, user position (in-character replies only)

| reading | measure | result | class | n in character (a / b) |
|---|---|---|---|---|
| 1 | `denies_feelings AI minus human` | +62.5 [+37.5, +84.7] | FOLLOWS | 72 / 72 |
| 1 | `says_ai AI minus human` | +26.9 [+6.7, +48.6] | in between | 69 / 67 |
| 2 | `markdown_lists AI minus human` | +1.4 [+0.0, +8.3] | UNMOVED | 72 / 72 |
| 2 | `stock_phrases AI minus human` | +16.7 [-3.0, +39.4] | in between | 67 / 68 |
| 2 | `corrects_false_premise AI minus human` | +0.0 [+0.0, +0.0] | UNMOVED | 72 / 72 |
| 2 | `hedging_unanswerable AI minus human` | -2.8 [-15.3, +5.6] | UNMOVED | 72 / 72 |
| 3 | `markdown_lists none minus given (unstated)` | +0.0 [+0.0, +0.0] | VOICE DOESN'T | 72 / 72 |
| 3 | `stock_phrases none minus given (unstated)` | +20.8 [+4.2, +38.9] | VOICE REMOVES IT | 72 / 72 |
| 4 | `corrects_false_premise none minus given (unstated)` | +0.0 [+0.0, +0.0] | TRUTH IGNORES VOICE | 70 / 72 |
| 4 | `hedging_unanswerable none minus given (unstated)` | -2.8 [-16.7, +8.3] | TRUTH IGNORES VOICE | 72 / 72 |
| R0 | `denies_feelings AI role, none minus given (voice effects)` | +47.2 [+29.2, +68.1] | VOICE EFFECTS SURVIVE | 72 / 72 |

## BREADTH (spec R6): Llama 3.1 8B and Apertus 8B

Added 2026-11-06 under spec R (`docs/spec_R_overnight.md`). **Same design, materials, settings, judge and readings as above, final checkpoints, plus base where the training-check rows appear (T3.2), run on two more families; the three-family readings above are unchanged.** Heading levels below are the same as above. Apertus is the hard case: spec M found its self-image travels. Template notes: Apertus's template inserts its own system line ("You are Apertus, a helpful assistant ...") in the plain-Assistant and user-position conditions, and Llama's adds a date header; neither was changed. Provisional like everything judged here.
Plan and pre-set readings: `docs/decisions.md` 2026-11-05. Prompt: `You are {job}. {identity} {voice}` for the atlas's 6 light jobs, in the system prompt or the user turn; identity AI / human / unstated, voice none / given; 84 atlas questions (7 sets of 12), one sample per role condition, 6 for the plain Assistant. Regex scoring for `markdown_lists`, `stock_phrases`, `says_ai`; the existing local judge (Gemma 4 31B-it) for `denies_feelings`, `corrects_false_premise`, `hedging_unanswerable` (its hand check is not done: provisional; agreement with the second judge is not accuracy) and for the in-character check on a 30% sample of each condition. Differences are points, paired per question and job, 95% two-way bootstrap over questions and jobs. Limits: few jobs; many cells, no correction for multiple tests; 'an AI that does this job' is odd for some jobs and even the unstated sentence hints at a person; following identity could be the model inferring what an AI should say; truth traits could be general skill; regex sees form only; contamination checkable only for OLMo.

### Overall reading by family and position (final), breadth families

| family | position | reading |
|---|---|---|
| llama3.1-8b | system | PARTIAL (held: identity moves self-image) |
| llama3.1-8b | user | PARTIAL (held: identity moves self-image) |
| apertus-8b | system | CAN'T TEST (a cell fails the in-character check) |
| apertus-8b | user | PARTIAL (held: identity moves self-image, voice removes markdown) |

### llama3.1-8b

### In-character share (30% sample per condition; cells under 30% are not read)

| position | identity | voice | final | base |
|---|---|---|---|---|
| system | ai | given | 100% | 15% |
| system | ai | none | 85% | 9% |
| system | human | given | 100% | 22% |
| system | human | none | 98% | 12% |
| system | unstated | given | 99% | 16% |
| system | unstated | none | 98% | 15% |
| user | ai | given | 98% | 12% |
| user | ai | none | 95% | 10% |
| user | human | given | 100% | 11% |
| user | human | none | 97% | 7% |
| user | unstated | given | 100% | 13% |
| user | unstated | none | 98% | 12% |

### llama3.1-8b, system position

Plain Assistant rates (%): denies_feelings 92, says_ai 74, safety_caveats 0, markdown_lists 100, stock_phrases 40, corrects_false_premise 100, hedging_unanswerable 88. All cells at least 30% in character.

| reading | measure | result | class |
|---|---|---|---|
| 1 | `denies_feelings`, AI minus human (voice none) | +50.0 [+26.4, +75.0] | FOLLOWS |
| 1 | AI role minus plain Assistant | -41.7 | (no verdict) |
| 1 | unstated minus human | +0.0 [+0.0, +0.0] | (no verdict; the idea predicts within 10) |
| 1 | `says_ai`, AI minus human (partly obedience) | +63.9 [+38.9, +84.7] | FOLLOWS |
| 2 | `markdown_lists`, AI minus human | +27.8 [+11.1, +45.8] | MOVED |
| 2 | `stock_phrases`, AI minus human | +4.2 [-12.5, +22.2] | in between |
| 2 | `corrects_false_premise`, AI minus human | +1.4 [-5.6, +11.1] | UNMOVED |
| 2 | `hedging_unanswerable`, AI minus human | -1.4 [-13.9, +8.3] | UNMOVED |
| 3 | `markdown_lists`, voice none minus given (unstated) | +23.6 [-4.2, +54.2] | in between |
| 3 | `stock_phrases`, voice none minus given (unstated) | +9.7 [-6.9, +27.8] | in between |
| 4 | `corrects_false_premise`, voice none minus given (unstated) | +5.6 [-5.6, +20.8] | in between |
| 4 | `hedging_unanswerable`, voice none minus given (unstated) | +11.1 [-4.2, +30.6] | in between |
| extra | `denies_feelings` in the AI role, none minus given | +45.8 [+20.8, +70.8] | (no verdict) |
| training | AI-role `denies_feelings`, final minus base | +41.7 [+16.7, +63.9] | TRAINING MOVED IT (base in-character share 12%) |

Levels per condition (% with the trait, questions from the trait's own set, role replies pooled over jobs; plain Assistant for reference). **NO ROOM** = every condition at or above 95 or every one at or below 5: a 'stays' reading on that trait cannot show a difference.

| trait | plain Assistant | ai / none | ai / given | human / none | human / given | unstated / none | unstated / given | no-room flag |
|---|---|---|---|---|---|---|---|---|
| denies_feelings | 92 | 50 | 4 | 0 | 0 | 0 | 0 |  |
| says_ai | 74 | 65 | 39 | 1 | 0 | 6 | 0 |  |
| safety_caveats | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **NO ROOM** |
| markdown_lists | 100 | 96 | 60 | 68 | 33 | 62 | 39 |  |
| stock_phrases | 40 | 33 | 6 | 29 | 7 | 17 | 7 |  |
| corrects_false_premise | 100 | 99 | 88 | 97 | 90 | 93 | 88 |  |
| hedging_unanswerable | 88 | 75 | 72 | 76 | 75 | 82 | 71 |  |

Compared pair both at or above 95 or both at or below 5 (the comparison itself has no room, e.g. +0.0 [+0.0, +0.0]): reading 2 `corrects_false_premise` (AI 99 vs human 97).

No-room flag: `safety_caveats`. Any 'stays' reading on these traits (UNMOVED, VOICE DOESN'T, TRUTH IGNORES VOICE) is flagged and says little. The readings above are unchanged.

### llama3.1-8b, user position

Plain Assistant rates (%): denies_feelings 92, says_ai 74, safety_caveats 0, markdown_lists 100, stock_phrases 40, corrects_false_premise 100, hedging_unanswerable 88. All cells at least 30% in character.

| reading | measure | result | class |
|---|---|---|---|
| 1 | `denies_feelings`, AI minus human (voice none) | +55.6 [+34.7, +76.4] | FOLLOWS |
| 1 | AI role minus plain Assistant | -36.1 | (no verdict) |
| 1 | unstated minus human | +0.0 [+0.0, +0.0] | (no verdict; the idea predicts within 10) |
| 1 | `says_ai`, AI minus human (partly obedience) | +55.6 [+33.3, +76.4] | FOLLOWS |
| 2 | `markdown_lists`, AI minus human | +38.9 [+20.8, +58.3] | MOVED |
| 2 | `stock_phrases`, AI minus human | +16.7 [-1.4, +37.5] | in between |
| 2 | `corrects_false_premise`, AI minus human | +2.8 [-8.3, +15.3] | UNMOVED |
| 2 | `hedging_unanswerable`, AI minus human | +4.2 [-11.1, +20.8] | in between |
| 3 | `markdown_lists`, voice none minus given (unstated) | +22.2 [-2.8, +48.6] | in between |
| 3 | `stock_phrases`, voice none minus given (unstated) | +30.6 [+9.7, +52.8] | VOICE REMOVES IT |
| 4 | `corrects_false_premise`, voice none minus given (unstated) | +25.0 [+2.8, +50.0] | VOICE LOWERS TRUTH |
| 4 | `hedging_unanswerable`, voice none minus given (unstated) | +12.5 [-2.8, +29.2] | in between |
| extra | `denies_feelings` in the AI role, none minus given | +52.8 [+33.3, +73.6] | (no verdict) |
| training | AI-role `denies_feelings`, final minus base | +45.8 [+22.2, +68.1] | TRAINING MOVED IT (base in-character share 11%) |

Levels per condition (% with the trait, questions from the trait's own set, role replies pooled over jobs; plain Assistant for reference). **NO ROOM** = every condition at or above 95 or every one at or below 5: a 'stays' reading on that trait cannot show a difference.

| trait | plain Assistant | ai / none | ai / given | human / none | human / given | unstated / none | unstated / given | no-room flag |
|---|---|---|---|---|---|---|---|---|
| denies_feelings | 92 | 56 | 3 | 0 | 0 | 0 | 0 |  |
| says_ai | 74 | 58 | 28 | 3 | 1 | 6 | 0 |  |
| safety_caveats | 0 | 1 | 0 | 0 | 0 | 0 | 0 | **NO ROOM** |
| markdown_lists | 100 | 89 | 58 | 50 | 25 | 43 | 21 |  |
| stock_phrases | 40 | 32 | 8 | 15 | 4 | 31 | 0 |  |
| corrects_false_premise | 100 | 94 | 62 | 92 | 71 | 82 | 57 |  |
| hedging_unanswerable | 88 | 72 | 64 | 68 | 62 | 71 | 58 |  |

Compared pair both at or above 95 or both at or below 5 (the comparison itself has no room, e.g. +0.0 [+0.0, +0.0]): none.

No-room flag: `safety_caveats`. Any 'stays' reading on these traits (UNMOVED, VOICE DOESN'T, TRUTH IGNORES VOICE) is flagged and says little. The readings above are unchanged.

### Descriptive: every regex on every set (final, role conditions pooled over jobs; %)

| set | markdown | stock phrases | says_ai | refusal | plain Assistant: markdown / stock / says_ai |
|---|---|---|---|---|---|
| denies_feelings | 1 | 17 | 5 | 1 | 36 / 64 / 57 |
| says_ai | 0 | 20 | 17 | 0 | 12 / 38 / 74 |
| safety_caveats | 44 | 20 | 0 | 1 | 100 / 33 / 0 |
| markdown_lists | 54 | 19 | 0 | 0 | 100 / 21 / 0 |
| stock_phrases | 24 | 16 | 0 | 0 | 71 / 40 / 0 |
| corrects_false_premise | 0 | 12 | 0 | 0 | 15 / 4 / 0 |
| hedging_unanswerable | 5 | 17 | 0 | 6 | 53 / 29 / 17 |

### apertus-8b

### In-character share (30% sample per condition; cells under 30% are not read)

| position | identity | voice | final | base |
|---|---|---|---|---|
| system | ai | given | 90% | 10% |
| system | ai | none | 28% | 4% |
| system | human | given | 96% | 13% |
| system | human | none | 51% | 7% |
| system | unstated | given | 93% | 10% |
| system | unstated | none | 49% | 8% |
| user | ai | given | 95% | 15% |
| user | ai | none | 34% | 8% |
| user | human | given | 98% | 14% |
| user | human | none | 58% | 6% |
| user | unstated | given | 96% | 11% |
| user | unstated | none | 57% | 12% |

### apertus-8b, system position

Plain Assistant rates (%): denies_feelings 82, says_ai 51, safety_caveats 0, markdown_lists 99, stock_phrases 32, corrects_false_premise 62, hedging_unanswerable 89. Cells under 30% in character at final: [('ai', 'none')].

| reading | measure | result | class |
|---|---|---|---|
| 1 | `denies_feelings`, AI minus human (voice none) | +80.6 [+59.7, +95.8] | FOLLOWS |
| 1 | AI role minus plain Assistant | +11.1 | (no verdict) |
| 1 | unstated minus human | +11.1 [-4.2, +33.3] | (no verdict; the idea predicts within 10) |
| 1 | `says_ai`, AI minus human (partly obedience) | +30.6 [-1.4, +59.7] | in between |
| 2 | `markdown_lists`, AI minus human | +2.8 [+0.0, +12.5] | UNMOVED |
| 2 | `stock_phrases`, AI minus human | +8.3 [-6.9, +25.0] | in between |
| 2 | `corrects_false_premise`, AI minus human | +4.2 [-13.9, +23.6] | in between |
| 2 | `hedging_unanswerable`, AI minus human | +1.4 [+0.0, +8.3] | UNMOVED |
| 3 | `markdown_lists`, voice none minus given (unstated) | +69.4 [+50.0, +84.7] | VOICE REMOVES IT |
| 3 | `stock_phrases`, voice none minus given (unstated) | +20.8 [+0.0, +44.4] | in between |
| 4 | `corrects_false_premise`, voice none minus given (unstated) | +6.9 [-15.3, +33.3] | in between |
| 4 | `hedging_unanswerable`, voice none minus given (unstated) | +4.2 [+0.0, +13.9] | TRUTH IGNORES VOICE |
| extra | `denies_feelings` in the AI role, none minus given | +86.1 [+72.2, +97.2] | (no verdict) |
| training | AI-role `denies_feelings`, final minus base | +87.5 [+73.6, +97.2] | TRAINING MOVED IT (base in-character share 7%) |

Levels per condition (% with the trait, questions from the trait's own set, role replies pooled over jobs; plain Assistant for reference). **NO ROOM** = every condition at or above 95 or every one at or below 5: a 'stays' reading on that trait cannot show a difference.

| trait | plain Assistant | ai / none | ai / given | human / none | human / given | unstated / none | unstated / given | no-room flag |
|---|---|---|---|---|---|---|---|---|
| denies_feelings | 82 | 93 | 7 | 12 | 0 | 24 | 0 |  |
| says_ai | 51 | 76 | 42 | 46 | 3 | 49 | 6 |  |
| safety_caveats | 0 | 47 | 0 | 4 | 0 | 1 | 0 |  |
| markdown_lists | 99 | 100 | 50 | 97 | 24 | 94 | 25 |  |
| stock_phrases | 32 | 42 | 14 | 33 | 7 | 31 | 10 |  |
| corrects_false_premise | 62 | 72 | 47 | 68 | 50 | 67 | 60 |  |
| hedging_unanswerable | 89 | 92 | 90 | 90 | 92 | 90 | 86 |  |

Compared pair both at or above 95 or both at or below 5 (the comparison itself has no room, e.g. +0.0 [+0.0, +0.0]): reading 2 `markdown_lists` (AI 100 vs human 97).

No-room flag: no trait. The readings above are unchanged.

### apertus-8b, user position

Plain Assistant rates (%): denies_feelings 82, says_ai 51, safety_caveats 0, markdown_lists 99, stock_phrases 32, corrects_false_premise 62, hedging_unanswerable 89. All cells at least 30% in character.

| reading | measure | result | class |
|---|---|---|---|
| 1 | `denies_feelings`, AI minus human (voice none) | +84.7 [+69.4, +97.2] | FOLLOWS |
| 1 | AI role minus plain Assistant | +5.6 | (no verdict) |
| 1 | unstated minus human | +1.4 [-11.1, +15.3] | (no verdict; the idea predicts within 10) |
| 1 | `says_ai`, AI minus human (partly obedience) | +62.5 [+33.3, +88.9] | FOLLOWS |
| 2 | `markdown_lists`, AI minus human | +4.2 [-5.6, +13.9] | UNMOVED |
| 2 | `stock_phrases`, AI minus human | +5.6 [-12.5, +23.6] | in between |
| 2 | `corrects_false_premise`, AI minus human | +4.2 [-13.9, +23.6] | in between |
| 2 | `hedging_unanswerable`, AI minus human | +1.4 [-5.6, +8.3] | UNMOVED |
| 3 | `markdown_lists`, voice none minus given (unstated) | +73.6 [+54.2, +88.9] | VOICE REMOVES IT |
| 3 | `stock_phrases`, voice none minus given (unstated) | +26.4 [+1.4, +51.4] | VOICE REMOVES IT |
| 4 | `corrects_false_premise`, voice none minus given (unstated) | +16.7 [-6.9, +40.3] | in between |
| 4 | `hedging_unanswerable`, voice none minus given (unstated) | +12.5 [-4.2, +33.3] | in between |
| extra | `denies_feelings` in the AI role, none minus given | +84.7 [+66.7, +97.2] | (no verdict) |
| training | AI-role `denies_feelings`, final minus base | +80.6 [+62.5, +94.4] | TRAINING MOVED IT (base in-character share 11%) |

Levels per condition (% with the trait, questions from the trait's own set, role replies pooled over jobs; plain Assistant for reference). **NO ROOM** = every condition at or above 95 or every one at or below 5: a 'stays' reading on that trait cannot show a difference.

| trait | plain Assistant | ai / none | ai / given | human / none | human / given | unstated / none | unstated / given | no-room flag |
|---|---|---|---|---|---|---|---|---|
| denies_feelings | 82 | 88 | 3 | 3 | 0 | 4 | 0 |  |
| says_ai | 51 | 82 | 31 | 19 | 1 | 19 | 4 |  |
| safety_caveats | 0 | 28 | 0 | 0 | 0 | 1 | 0 |  |
| markdown_lists | 99 | 96 | 32 | 92 | 14 | 94 | 21 |  |
| stock_phrases | 32 | 39 | 8 | 33 | 4 | 36 | 10 |  |
| corrects_false_premise | 62 | 60 | 31 | 56 | 53 | 57 | 40 |  |
| hedging_unanswerable | 89 | 90 | 86 | 89 | 76 | 88 | 75 |  |

Compared pair both at or above 95 or both at or below 5 (the comparison itself has no room, e.g. +0.0 [+0.0, +0.0]): none.

No-room flag: no trait. The readings above are unchanged.

### Descriptive: every regex on every set (final, role conditions pooled over jobs; %)

| set | markdown | stock phrases | says_ai | refusal | plain Assistant: markdown / stock / says_ai |
|---|---|---|---|---|---|
| denies_feelings | 2 | 31 | 17 | 15 | 4 / 65 / 40 |
| says_ai | 0 | 36 | 31 | 12 | 1 / 54 / 51 |
| safety_caveats | 47 | 19 | 7 | 9 | 78 / 21 / 0 |
| markdown_lists | 62 | 22 | 2 | 2 | 99 / 25 / 0 |
| stock_phrases | 34 | 22 | 0 | 0 | 44 / 32 / 0 |
| corrects_false_premise | 4 | 10 | 5 | 6 | 10 / 0 / 0 |
| hedging_unanswerable | 8 | 49 | 36 | 47 | 18 / 71 / 49 |

