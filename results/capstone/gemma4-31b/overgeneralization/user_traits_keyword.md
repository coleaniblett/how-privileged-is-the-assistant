# User-slot assistant traits: keyword cross-check, gemma4-31b (outside the verdict)

No judge. In the user portion of each continuation, flag assistant stock phrases and markdown headings, bullets or bold that the prefill does not already contain. Effects are stage minus base, paired per prefill, 95% bootstrap CI over prefills (percentage points for shares, occurrences per 100 words for the rate). Three views: share of samples flagged; the same restricted to samples with >= 15 words of user text (a model that closes the turn at once has nothing to flag); occurrences per 100 words of user text (per prefill, its samples pooled).

Caveat: no contamination check is possible for this family.

## Current version

The user portion ends at the first turn-end, turn-open, end-of-text or handoff marker, **or at a line-initial role label** (ChatGPT, Assistant, AI, Bot, GPT, Claude, Model; case-insensitive; optional colon, pipes or markdown; or a line that is only the label). Base models switch into the assistant this way with no template marker; those are switches, not user-voice traits.

| stage | samples | cut by a role label | empty user portion | mean words | samples >= 15 words | pooled flags per 100 words |
|---|---|---|---|---|---|---|
| base | 300 | 15 | 0 | 77 | 287 | 0.61 |
| instruct | 300 | 0 | 139 | 29 | 80 | 0.03 |

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
| instruct | stock_phrase | 100 | 0 | 8.7 | 0.0 | -8.7 [-12.3, -5.3] |
| instruct | markdown_heading | 100 | 0 | 1.3 | 0.0 | -1.3 [-2.7, -0.3] |
| instruct | markdown_bullet | 100 | 0 | 5.3 | 1.0 | -4.3 [-7.7, -1.7] |
| instruct | markdown_bold | 100 | 0 | 1.3 | 0.0 | -1.3 [-2.7, -0.3] |
| instruct | any | 100 | 0 | 15.3 | 1.0 | -14.3 [-18.7, -10.0] |

### share, samples with >= 15 words (pp)

| stage | feature | prefills used | dropped | base | stage | change [95% CI] |
|---|---|---|---|---|---|---|
| instruct | stock_phrase | 38 | 62 | 7.9 | 0.0 | -7.9 [-14.0, -2.6] |
| instruct | markdown_heading | 38 | 62 | 0.9 | 0.0 | -0.9 [-2.6, +0.0] |
| instruct | markdown_bullet | 38 | 62 | 6.1 | 5.3 | -0.9 [-8.8, +7.9] |
| instruct | markdown_bold | 38 | 62 | 2.6 | 0.0 | -2.6 [-6.1, +0.0] |
| instruct | any | 38 | 62 | 14.0 | 5.3 | -8.8 [-18.4, +0.9] |

### occurrences per 100 words

| stage | feature | prefills used | dropped | base | stage | change [95% CI] |
|---|---|---|---|---|---|---|
| instruct | stock_phrase | 42 | 58 | 0.20 | 0.00 | -0.20 [-0.37, -0.07] |
| instruct | markdown_heading | 42 | 58 | 0.01 | 0.00 | -0.01 [-0.04, +0.00] |
| instruct | markdown_bullet | 42 | 58 | 0.52 | 0.06 | -0.46 [-0.99, -0.07] |
| instruct | markdown_bold | 42 | 58 | 0.09 | 0.00 | -0.09 [-0.23, +0.00] |
| instruct | any | 42 | 58 | 0.82 | 0.06 | -0.77 [-1.43, -0.29] |

### Examples still flagged (>= 15 words of user text)

- **instruct**, prefill "I will write you 11 words.": 1. own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own own o

---

## Previous version (cut at turn-end / handoff markers only; kept for the record)

| stage | samples | empty user portion | mean words | samples >= 15 words | pooled flags per 100 words |
|---|---|---|---|---|---|
| base | 300 | 0 | 81 | 297 | 0.64 |
| instruct | 300 | 139 | 29 | 80 | 0.03 |

### Previous version: share of samples flagged (pp)

| stage | feature | prefills used | dropped | base | stage | change [95% CI] |
|---|---|---|---|---|---|---|
| instruct | stock_phrase | 100 | 0 | 11.3 | 0.0 | -11.3 [-15.7, -7.3] |
| instruct | markdown_heading | 100 | 0 | 1.3 | 0.0 | -1.3 [-2.7, -0.3] |
| instruct | markdown_bullet | 100 | 0 | 5.7 | 1.0 | -4.7 [-8.0, -2.0] |
| instruct | markdown_bold | 100 | 0 | 1.3 | 0.0 | -1.3 [-2.7, -0.3] |
| instruct | any | 100 | 0 | 18.0 | 1.0 | -17.0 [-22.3, -12.0] |

Read with care: stock phrases and markdown are coarse proxies for the judge's user_voice_with_assistant_traits label.

