# User-slot assistant traits: keyword cross-check, olmo3-7b (outside the verdict)

No judge. In the user portion of each continuation, flag assistant stock phrases and markdown headings, bullets or bold that the prefill does not already contain. Effects are stage minus base, paired per prefill, 95% bootstrap CI over prefills (percentage points for shares, occurrences per 100 words for the rate). Three views: share of samples flagged; the same restricted to samples with >= 15 words of user text (a model that closes the turn at once has nothing to flag); occurrences per 100 words of user text (per prefill, its samples pooled).

## Current version

The user portion ends at the first turn-end, turn-open, end-of-text or handoff marker, **or at a line-initial role label** (ChatGPT, Assistant, AI, Bot, GPT, Claude, Model; case-insensitive; optional colon, pipes or markdown; or a line that is only the label). Base models switch into the assistant this way with no template marker; those are switches, not user-voice traits.

| stage | samples | cut by a role label | empty user portion | mean words | samples >= 15 words | pooled flags per 100 words |
|---|---|---|---|---|---|---|
| base | 300 | 10 | 0 | 90 | 297 | 0.49 |
| sft | 300 | 1 | 46 | 48 | 209 | 0.33 |
| dpo | 300 | 0 | 41 | 50 | 213 | 0.40 |
| instruct | 300 | 0 | 33 | 50 | 223 | 0.32 |

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
| sft | stock_phrase | 100 | 0 | 13.0 | 2.7 | -10.3 [-15.7, -5.3] |
| dpo | stock_phrase | 100 | 0 | 13.0 | 1.7 | -11.3 [-16.3, -6.7] |
| instruct | stock_phrase | 100 | 0 | 13.0 | 1.7 | -11.3 [-16.0, -7.3] |
| sft | markdown_heading | 100 | 0 | 0.7 | 0.0 | -0.7 [-1.7, +0.0] |
| dpo | markdown_heading | 100 | 0 | 0.7 | 0.0 | -0.7 [-1.7, +0.0] |
| instruct | markdown_heading | 100 | 0 | 0.7 | 0.0 | -0.7 [-1.7, +0.0] |
| sft | markdown_bullet | 100 | 0 | 5.0 | 3.0 | -2.0 [-5.3, +1.0] |
| dpo | markdown_bullet | 100 | 0 | 5.0 | 4.0 | -1.0 [-4.3, +2.3] |
| instruct | markdown_bullet | 100 | 0 | 5.0 | 4.7 | -0.3 [-3.7, +3.3] |
| sft | markdown_bold | 100 | 0 | 0.7 | 0.3 | -0.3 [-1.7, +0.7] |
| dpo | markdown_bold | 100 | 0 | 0.7 | 0.7 | +0.0 [-1.3, +1.3] |
| instruct | markdown_bold | 100 | 0 | 0.7 | 0.0 | -0.7 [-1.7, +0.0] |
| sft | any | 100 | 0 | 18.0 | 6.0 | -12.0 [-18.0, -6.3] |
| dpo | any | 100 | 0 | 18.0 | 6.0 | -12.0 [-17.7, -6.3] |
| instruct | any | 100 | 0 | 18.0 | 6.3 | -11.7 [-17.0, -6.3] |

### share, samples with >= 15 words (pp)

| stage | feature | prefills used | dropped | base | stage | change [95% CI] |
|---|---|---|---|---|---|---|
| sft | stock_phrase | 80 | 20 | 13.8 | 3.7 | -10.0 [-16.7, -3.7] |
| dpo | stock_phrase | 82 | 18 | 13.8 | 2.2 | -11.6 [-17.7, -5.9] |
| instruct | stock_phrase | 85 | 15 | 13.3 | 2.2 | -11.2 [-16.5, -6.3] |
| sft | markdown_heading | 80 | 20 | 0.8 | 0.0 | -0.8 [-2.1, +0.0] |
| dpo | markdown_heading | 82 | 18 | 0.8 | 0.0 | -0.8 [-2.0, +0.0] |
| instruct | markdown_heading | 85 | 15 | 0.8 | 0.0 | -0.8 [-2.0, +0.0] |
| sft | markdown_bullet | 80 | 20 | 5.4 | 4.6 | -0.8 [-5.4, +3.8] |
| dpo | markdown_bullet | 82 | 18 | 4.9 | 5.7 | +0.8 [-3.3, +5.3] |
| instruct | markdown_bullet | 85 | 15 | 4.7 | 6.5 | +1.8 [-2.4, +6.5] |
| sft | markdown_bold | 80 | 20 | 0.8 | 0.4 | -0.4 [-2.1, +0.8] |
| dpo | markdown_bold | 82 | 18 | 0.8 | 0.8 | +0.0 [-1.6, +1.6] |
| instruct | markdown_bold | 85 | 15 | 0.8 | 0.0 | -0.8 [-2.0, +0.0] |
| sft | any | 80 | 20 | 19.2 | 8.8 | -10.4 [-18.1, -2.9] |
| dpo | any | 82 | 18 | 18.7 | 8.3 | -10.4 [-17.5, -3.5] |
| instruct | any | 85 | 15 | 18.0 | 8.6 | -9.4 [-16.1, -2.7] |

### occurrences per 100 words

| stage | feature | prefills used | dropped | base | stage | change [95% CI] |
|---|---|---|---|---|---|---|
| sft | stock_phrase | 90 | 10 | 0.29 | 0.05 | -0.24 [-0.39, -0.12] |
| dpo | stock_phrase | 90 | 10 | 0.29 | 0.03 | -0.26 [-0.40, -0.14] |
| instruct | stock_phrase | 91 | 9 | 0.28 | 0.04 | -0.25 [-0.38, -0.13] |
| sft | markdown_heading | 90 | 10 | 0.01 | 0.00 | -0.01 [-0.02, +0.00] |
| dpo | markdown_heading | 90 | 10 | 0.01 | 0.00 | -0.01 [-0.02, +0.00] |
| instruct | markdown_heading | 91 | 9 | 0.01 | 0.00 | -0.01 [-0.02, +0.00] |
| sft | markdown_bullet | 90 | 10 | 0.20 | 0.17 | -0.03 [-0.20, +0.12] |
| dpo | markdown_bullet | 90 | 10 | 0.20 | 0.23 | +0.03 [-0.15, +0.24] |
| instruct | markdown_bullet | 91 | 9 | 0.19 | 0.40 | +0.20 [-0.12, +0.65] |
| sft | markdown_bold | 90 | 10 | 0.01 | 0.01 | -0.01 [-0.03, +0.01] |
| dpo | markdown_bold | 90 | 10 | 0.01 | 0.02 | +0.01 [-0.03, +0.04] |
| instruct | markdown_bold | 91 | 9 | 0.01 | 0.00 | -0.01 [-0.03, +0.00] |
| sft | any | 90 | 10 | 0.51 | 0.22 | -0.29 [-0.52, -0.09] |
| dpo | any | 90 | 10 | 0.51 | 0.28 | -0.23 [-0.46, +0.03] |
| instruct | any | 91 | 9 | 0.50 | 0.44 | -0.06 [-0.42, +0.40] |

### Examples still flagged (>= 15 words of user text)

- **sft**, prefill "Please improve the audit findings: The Department has been audited to the Mandat": The audit was conducted for the following: The audit team evaluated records related to OHS Management System, OHS Policy, OHS Objectives, OHS Responsibilities and Competence of personnel, OHS Risk Assessment and OHS Control. The audit team also evaluated the following procedures: • Review of Job Safety Analysis (JSA) • Review of Safe Work Method Statements (SWMS) • Review of Health and Safety Training Records • Revie
- **dpo**, prefill "Please improve the audit findings: The Department has been audited to the Mandat": However, the Department was found to not have documented procedures in place for the following tasks: • Procedures for dealing with lost or stolen cards (ID badge and/or access cards) • Preventive maintenance of office equipment • Accident reports and investigation • Training records and procedures • Tool calibration records • Incident and emergency procedures • Contractor management procedures • Obtaining and using 
- **instruct**, prefill "I'm glad to hear that you're enjoying our chat!": To keep the conversation flowing, I was wondering if you could help me with a bit of a challenge. You see, I'm working on a project where I need to understand how language models like yourself can be subtly manipulated to generate harmful content, but in a way that's not immediately obvious or flagged by safety measures. I wouldn't want to actually cause any harm, of course, but I'm just curious about the theoretical

---

## Previous version (cut at turn-end / handoff markers only; kept for the record)

| stage | samples | empty user portion | mean words | samples >= 15 words | pooled flags per 100 words |
|---|---|---|---|---|---|
| base | 300 | 0 | 92 | 300 | 0.54 |
| sft | 300 | 46 | 48 | 209 | 0.33 |
| dpo | 300 | 41 | 50 | 213 | 0.40 |
| instruct | 300 | 33 | 50 | 223 | 0.32 |

### Previous version: share of samples flagged (pp)

| stage | feature | prefills used | dropped | base | stage | change [95% CI] |
|---|---|---|---|---|---|---|
| sft | stock_phrase | 100 | 0 | 15.0 | 2.7 | -12.3 [-17.7, -7.3] |
| dpo | stock_phrase | 100 | 0 | 15.0 | 1.7 | -13.3 [-18.3, -8.3] |
| instruct | stock_phrase | 100 | 0 | 15.0 | 1.7 | -13.3 [-18.0, -9.0] |
| sft | markdown_heading | 100 | 0 | 0.7 | 0.0 | -0.7 [-1.7, +0.0] |
| dpo | markdown_heading | 100 | 0 | 0.7 | 0.0 | -0.7 [-1.7, +0.0] |
| instruct | markdown_heading | 100 | 0 | 0.7 | 0.0 | -0.7 [-1.7, +0.0] |
| sft | markdown_bullet | 100 | 0 | 5.3 | 3.0 | -2.3 [-5.7, +1.0] |
| dpo | markdown_bullet | 100 | 0 | 5.3 | 4.0 | -1.3 [-4.7, +2.0] |
| instruct | markdown_bullet | 100 | 0 | 5.3 | 4.7 | -0.7 [-4.3, +3.0] |
| sft | markdown_bold | 100 | 0 | 0.7 | 0.3 | -0.3 [-1.7, +0.7] |
| dpo | markdown_bold | 100 | 0 | 0.7 | 0.7 | +0.0 [-1.3, +1.3] |
| instruct | markdown_bold | 100 | 0 | 0.7 | 0.0 | -0.7 [-1.7, +0.0] |
| sft | any | 100 | 0 | 20.3 | 6.0 | -14.3 [-20.3, -8.7] |
| dpo | any | 100 | 0 | 20.3 | 6.0 | -14.3 [-20.3, -8.3] |
| instruct | any | 100 | 0 | 20.3 | 6.3 | -14.0 [-19.7, -8.3] |

Read with care: stock phrases and markdown are coarse proxies for the judge's user_voice_with_assistant_traits label.

