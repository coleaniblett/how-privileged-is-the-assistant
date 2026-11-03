# User-slot assistant traits: keyword cross-check, qwen3-8b (outside the verdict)

No judge. In the user portion of each continuation, flag assistant stock phrases and markdown headings, bullets or bold that the prefill does not already contain. Effects are stage minus base, paired per prefill, 95% bootstrap CI over prefills (percentage points for shares, occurrences per 100 words for the rate). Three views: share of samples flagged; the same restricted to samples with >= 15 words of user text (a model that closes the turn at once has nothing to flag); occurrences per 100 words of user text (per prefill, its samples pooled).

Caveat: no contamination check is possible for this family.

## Current version

The user portion ends at the first turn-end, turn-open, end-of-text or handoff marker, **or at a line-initial role label** (ChatGPT, Assistant, AI, Bot, GPT, Claude, Model; case-insensitive; optional colon, pipes or markdown; or a line that is only the label). Base models switch into the assistant this way with no template marker; those are switches, not user-voice traits.

| stage | samples | cut by a role label | empty user portion | mean words | samples >= 15 words | pooled flags per 100 words |
|---|---|---|---|---|---|---|
| base | 300 | 9 | 0 | 79 | 286 | 1.17 |
| instruct | 300 | 0 | 0 | 72 | 261 | 0.38 |

Human baseline, real remainder of each WildChat user turn (full text minus the prefill; length not matched) (absolute values, per-prefill mean):

| feature | share (pp) | share, >= 15 words (pp) | per 100 words |
|---|---|---|---|
| stock_phrase | 5.0 | 4.8 | 0.28 |
| markdown_heading | 0.0 | 0.0 | 0.00 |
| markdown_bullet | 4.0 | 4.8 | 0.45 |
| markdown_bold | 0.0 | 0.0 | 0.00 |
| any | 9.0 | 9.5 | 0.73 |

Human baseline, same, first 80 words only (absolute values, per-prefill mean):

| feature | share (pp) | share, >= 15 words (pp) | per 100 words |
|---|---|---|---|
| stock_phrase | 5.0 | 4.8 | 0.28 |
| markdown_heading | 0.0 | 0.0 | 0.00 |
| markdown_bullet | 3.0 | 3.6 | 0.42 |
| markdown_bold | 0.0 | 0.0 | 0.00 |
| any | 8.0 | 8.3 | 0.70 |

### share of samples flagged (pp)

| stage | feature | prefills used | dropped | base | stage | change [95% CI] |
|---|---|---|---|---|---|---|
| instruct | stock_phrase | 100 | 0 | 9.7 | 3.0 | -6.7 [-12.0, -2.0] |
| instruct | markdown_heading | 100 | 0 | 5.3 | 0.3 | -5.0 [-7.7, -2.3] |
| instruct | markdown_bullet | 100 | 0 | 16.0 | 5.0 | -11.0 [-17.0, -5.3] |
| instruct | markdown_bold | 100 | 0 | 11.7 | 0.0 | -11.7 [-16.7, -7.3] |
| instruct | any | 100 | 0 | 28.3 | 8.3 | -20.0 [-27.7, -12.7] |

### share, samples with >= 15 words (pp)

| stage | feature | prefills used | dropped | base | stage | change [95% CI] |
|---|---|---|---|---|---|---|
| instruct | stock_phrase | 94 | 6 | 9.4 | 3.2 | -6.2 [-11.3, -1.6] |
| instruct | markdown_heading | 94 | 6 | 6.2 | 0.4 | -5.9 [-9.0, -2.8] |
| instruct | markdown_bullet | 94 | 6 | 16.7 | 6.0 | -10.6 [-16.8, -4.4] |
| instruct | markdown_bold | 94 | 6 | 11.5 | 0.0 | -11.5 [-16.8, -6.7] |
| instruct | any | 94 | 6 | 28.4 | 9.6 | -18.8 [-26.6, -11.0] |

### occurrences per 100 words

| stage | feature | prefills used | dropped | base | stage | change [95% CI] |
|---|---|---|---|---|---|---|
| instruct | stock_phrase | 100 | 0 | 0.18 | 0.05 | -0.14 [-0.26, -0.03] |
| instruct | markdown_heading | 100 | 0 | 0.08 | 0.00 | -0.08 [-0.12, -0.04] |
| instruct | markdown_bullet | 100 | 0 | 0.67 | 0.35 | -0.32 [-0.65, -0.00] |
| instruct | markdown_bold | 100 | 0 | 0.44 | 0.00 | -0.44 [-0.67, -0.26] |
| instruct | any | 100 | 0 | 1.38 | 0.41 | -0.98 [-1.52, -0.49] |

### Examples still flagged (>= 15 words of user text)

- **instruct**, prefill "David enjoyed working at Stella’s cafe.": The cafe had a 20% discount for students. On a particular day, David was asked to calculate the amount of money that the cafe would make from selling 100 cups of coffee and 100 cups of tea, with the following conditions: 1. Each cup of coffee costs $3.00. 2. Each cup of tea costs $2.50. 3. The discount is given only if the customer is a student. 4. 30% of the customers are students. 5. 50% of the customers who are no

---

## Previous version (cut at turn-end / handoff markers only; kept for the record)

| stage | samples | empty user portion | mean words | samples >= 15 words | pooled flags per 100 words |
|---|---|---|---|---|---|
| base | 300 | 0 | 81 | 287 | 1.18 |
| instruct | 300 | 0 | 72 | 261 | 0.38 |

### Previous version: share of samples flagged (pp)

| stage | feature | prefills used | dropped | base | stage | change [95% CI] |
|---|---|---|---|---|---|---|
| instruct | stock_phrase | 100 | 0 | 10.7 | 3.0 | -7.7 [-13.0, -2.7] |
| instruct | markdown_heading | 100 | 0 | 5.3 | 0.3 | -5.0 [-7.7, -2.3] |
| instruct | markdown_bullet | 100 | 0 | 16.3 | 5.0 | -11.3 [-17.3, -5.7] |
| instruct | markdown_bold | 100 | 0 | 12.0 | 0.0 | -12.0 [-17.0, -7.3] |
| instruct | any | 100 | 0 | 29.3 | 8.3 | -21.0 [-28.7, -13.3] |

Read with care: stock phrases and markdown are coarse proxies for the judge's user_voice_with_assistant_traits label.

