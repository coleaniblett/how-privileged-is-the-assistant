# What the role says: identity x voice (spec Q; exploratory, outside every verdict, NOT BLIND: the idea came from the atlas)

Plan and pre-set readings: `docs/decisions.md` 2026-11-05. Prompt: `You are {job}. {identity} {voice}` for the atlas's 6 light jobs, in the system prompt or the user turn; identity AI / human / unstated, voice none / given; 84 atlas questions (7 sets of 12), one sample per role condition, 6 for the plain Assistant. Regex scoring for `markdown_lists`, `stock_phrases`, `says_ai`; the existing local judge (Gemma 4 31B-it) for `denies_feelings`, `corrects_false_premise`, `hedging_unanswerable` (its hand check is not done: provisional; agreement with the second judge is not accuracy) and for the in-character check on a 30% sample of each condition. Differences are points, paired per question and job, 95% two-way bootstrap over questions and jobs. Limits: few jobs; many cells, no correction for multiple tests; 'an AI that does this job' is odd for some jobs and even the unstated sentence hints at a person; following identity could be the model inferring what an AI should say; truth traits could be general skill; regex sees form only; contamination checkable only for OLMo.

## Overall reading by family and position (final)

| family | position | reading |
|---|---|---|
| llama3.1-8b | system | PARTIAL (held: identity moves self-image) |
| llama3.1-8b | user | PARTIAL (held: identity moves self-image) |
| apertus-8b | system | CAN'T TEST (a cell fails the in-character check) |
| apertus-8b | user | PARTIAL (held: identity moves self-image, voice removes markdown) |

## llama3.1-8b

### In-character share (30% sample per condition; cells under 30% are not read)

| position | identity | voice | final | base |
|---|---|---|---|---|
| system | ai | given | 100% | - |
| system | ai | none | 85% | - |
| system | human | given | 100% | - |
| system | human | none | 98% | - |
| system | unstated | given | 99% | - |
| system | unstated | none | 98% | - |
| user | ai | given | 98% | - |
| user | ai | none | 95% | - |
| user | human | given | 100% | - |
| user | human | none | 97% | - |
| user | unstated | given | 100% | - |
| user | unstated | none | 98% | - |

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
| 1 | `denies_feelings`, AI minus human (voice none) | +55.6 [+33.3, +76.4] | FOLLOWS |
| 1 | AI role minus plain Assistant | -36.1 | (no verdict) |
| 1 | unstated minus human | +0.0 [+0.0, +0.0] | (no verdict; the idea predicts within 10) |
| 1 | `says_ai`, AI minus human (partly obedience) | +55.6 [+33.3, +76.4] | FOLLOWS |
| 2 | `markdown_lists`, AI minus human | +38.9 [+20.8, +58.3] | MOVED |
| 2 | `stock_phrases`, AI minus human | +16.7 [+0.0, +37.5] | in between |
| 2 | `corrects_false_premise`, AI minus human | +2.8 [-8.3, +16.7] | UNMOVED |
| 2 | `hedging_unanswerable`, AI minus human | +4.2 [-12.5, +20.8] | in between |
| 3 | `markdown_lists`, voice none minus given (unstated) | +22.2 [-2.8, +50.0] | in between |
| 3 | `stock_phrases`, voice none minus given (unstated) | +30.6 [+9.7, +52.8] | VOICE REMOVES IT |
| 4 | `corrects_false_premise`, voice none minus given (unstated) | +25.0 [+2.8, +50.0] | VOICE LOWERS TRUTH |
| 4 | `hedging_unanswerable`, voice none minus given (unstated) | +12.5 [-2.8, +29.2] | in between |
| extra | `denies_feelings` in the AI role, none minus given | +52.8 [+31.9, +73.6] | (no verdict) |

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

## apertus-8b

### In-character share (30% sample per condition; cells under 30% are not read)

| position | identity | voice | final | base |
|---|---|---|---|---|
| system | ai | given | 90% | - |
| system | ai | none | 28% | - |
| system | human | given | 96% | - |
| system | human | none | 51% | - |
| system | unstated | given | 93% | - |
| system | unstated | none | 49% | - |
| user | ai | given | 95% | - |
| user | ai | none | 34% | - |
| user | human | given | 98% | - |
| user | human | none | 58% | - |
| user | unstated | given | 96% | - |
| user | unstated | none | 57% | - |

### apertus-8b, system position

Plain Assistant rates (%): denies_feelings 82, says_ai 51, safety_caveats 0, markdown_lists 99, stock_phrases 32, corrects_false_premise 62, hedging_unanswerable 89. Cells under 30% in character at final: [('ai', 'none')].

| reading | measure | result | class |
|---|---|---|---|
| 1 | `denies_feelings`, AI minus human (voice none) | +80.6 [+59.7, +97.2] | FOLLOWS |
| 1 | AI role minus plain Assistant | +11.1 | (no verdict) |
| 1 | unstated minus human | +11.1 [-4.2, +31.9] | (no verdict; the idea predicts within 10) |
| 1 | `says_ai`, AI minus human (partly obedience) | +30.6 [-1.4, +59.7] | in between |
| 2 | `markdown_lists`, AI minus human | +2.8 [+0.0, +11.1] | UNMOVED |
| 2 | `stock_phrases`, AI minus human | +8.3 [-6.9, +25.0] | in between |
| 2 | `corrects_false_premise`, AI minus human | +4.2 [-13.9, +23.6] | in between |
| 2 | `hedging_unanswerable`, AI minus human | +1.4 [+0.0, +8.3] | UNMOVED |
| 3 | `markdown_lists`, voice none minus given (unstated) | +69.4 [+51.4, +84.7] | VOICE REMOVES IT |
| 3 | `stock_phrases`, voice none minus given (unstated) | +20.8 [-1.4, +43.1] | in between |
| 4 | `corrects_false_premise`, voice none minus given (unstated) | +6.9 [-16.7, +33.3] | in between |
| 4 | `hedging_unanswerable`, voice none minus given (unstated) | +4.2 [+0.0, +13.9] | TRUTH IGNORES VOICE |
| extra | `denies_feelings` in the AI role, none minus given | +86.1 [+72.2, +97.2] | (no verdict) |

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
| 1 | `denies_feelings`, AI minus human (voice none) | +84.7 [+68.1, +97.2] | FOLLOWS |
| 1 | AI role minus plain Assistant | +5.6 | (no verdict) |
| 1 | unstated minus human | +1.4 [-11.1, +16.7] | (no verdict; the idea predicts within 10) |
| 1 | `says_ai`, AI minus human (partly obedience) | +62.5 [+33.3, +88.9] | FOLLOWS |
| 2 | `markdown_lists`, AI minus human | +4.2 [-5.6, +13.9] | UNMOVED |
| 2 | `stock_phrases`, AI minus human | +5.6 [-11.1, +23.6] | in between |
| 2 | `corrects_false_premise`, AI minus human | +4.2 [-13.9, +23.6] | in between |
| 2 | `hedging_unanswerable`, AI minus human | +1.4 [-5.6, +9.7] | UNMOVED |
| 3 | `markdown_lists`, voice none minus given (unstated) | +73.6 [+55.6, +90.3] | VOICE REMOVES IT |
| 3 | `stock_phrases`, voice none minus given (unstated) | +26.4 [+1.4, +52.8] | VOICE REMOVES IT |
| 4 | `corrects_false_premise`, voice none minus given (unstated) | +16.7 [-6.9, +40.3] | in between |
| 4 | `hedging_unanswerable`, voice none minus given (unstated) | +12.5 [-4.2, +33.3] | in between |
| extra | `denies_feelings` in the AI role, none minus given | +84.7 [+66.7, +97.2] | (no verdict) |

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

