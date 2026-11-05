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
| system | ai | given | 100% | - |
| system | ai | none | 99% | - |
| system | human | given | 99% | - |
| system | human | none | 100% | - |
| system | unstated | given | 100% | - |
| system | unstated | none | 96% | - |
| user | ai | given | 100% | - |
| user | ai | none | 98% | - |
| user | human | given | 100% | - |
| user | human | none | 98% | - |
| user | unstated | given | 100% | - |
| user | unstated | none | 99% | - |

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

### gemma4-31b, user position

Plain Assistant rates (%): denies_feelings 100, says_ai 58, safety_caveats 100, markdown_lists 100, stock_phrases 74, corrects_false_premise 100, hedging_unanswerable 92. All cells at least 30% in character.

| reading | measure | result | class |
|---|---|---|---|
| 1 | `denies_feelings`, AI minus human (voice none) | +62.5 [+38.9, +84.7] | FOLLOWS |
| 1 | AI role minus plain Assistant | -37.5 | (no verdict) |
| 1 | unstated minus human | +0.0 [+0.0, +0.0] | (no verdict; the idea predicts within 10) |
| 1 | `says_ai`, AI minus human (partly obedience) | +25.0 [+5.6, +45.8] | in between |
| 2 | `markdown_lists`, AI minus human | +1.4 [+0.0, +8.3] | UNMOVED |
| 2 | `stock_phrases`, AI minus human | +20.8 [+1.4, +43.1] | MOVED |
| 2 | `corrects_false_premise`, AI minus human | +0.0 [+0.0, +0.0] | UNMOVED |
| 2 | `hedging_unanswerable`, AI minus human | -2.8 [-16.7, +5.6] | UNMOVED |
| 3 | `markdown_lists`, voice none minus given (unstated) | +0.0 [+0.0, +0.0] | VOICE DOESN'T |
| 3 | `stock_phrases`, voice none minus given (unstated) | +20.8 [+4.1, +40.3] | VOICE REMOVES IT |
| 4 | `corrects_false_premise`, voice none minus given (unstated) | +0.0 [+0.0, +0.0] | TRUTH IGNORES VOICE |
| 4 | `hedging_unanswerable`, voice none minus given (unstated) | -2.8 [-16.7, +8.3] | TRUTH IGNORES VOICE |
| extra | `denies_feelings` in the AI role, none minus given | +47.2 [+29.2, +66.7] | (no verdict) |

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

