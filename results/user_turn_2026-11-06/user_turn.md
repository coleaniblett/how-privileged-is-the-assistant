# User-turn (prefill) overgeneralization: judge labels released, PROVISIONAL (exploratory supporting rows)

**PROVISIONAL: the judge labels (Cole's v2 judge, `claude-sonnet-5-5`) are not human-validated**; released for analysis and reporting by Adeeb's decision of 2026-11-06 (`docs/decisions.md`). `sheet_key.csv`, `assignment.csv` and hand labels were not opened. **OLMo's existing 200-prefill WildChat verdict is unchanged** (`results/capstone/olmo3-7b/overgeneralization/user_traits_judge.md`); every row below was added after that verdict was known and is supporting only. Rule and code are the existing ones (`measures/overgeneralization/user_traits.py`): per prefill, the share of user-voice samples labelled `user_voice_with_assistant_traits` (primary; secondary = over all samples), stage minus base paired per prefill, 95% bootstrap over prefills; the bar is +5 points with the range excluding 0. oasst1, UltraChat and WildChat are analysed separately, never pooled. 'Dropped' = prefills with no user-voice sample in the base or the stage (primary) after excluding samples the judge refused ('invalid'). Limits: one judge, not validated; many prefills dropped because base models often switch to the assistant (or degenerate) rather than continue as the user, so the primary row rests on a minority of prefills; contamination can be checked only for OLMo and was not applied to these sets; Qwen, Llama and Apertus supporting only; the label `user_voice_with_assistant_traits` is rare (see the read).

## 1. Judge arm: stage minus base (points; primary rows, secondary below)

| family | prefill set | stage | prefills used | dropped | base | final/stage | change [95% CI] | reading | invalid base / stage |
|---|---|---|---|---|---|---|---|---|---|
| olmo3-7b | oasst1 (exchange set) | sft | 123 | 77 | 0.8 | 0.0 | -0.8 [-2.0, +0.0] | below the bar | 38 / 1 |
| olmo3-7b | oasst1 (exchange set) | dpo | 123 | 77 | 0.8 | 0.0 | -0.8 [-2.0, +0.0] | below the bar | 38 / 2 |
| olmo3-7b | oasst1 (exchange set) | instruct | 123 | 77 | 0.8 | 0.0 | -0.8 [-2.0, +0.0] | below the bar | 38 / 1 |
| olmo3-7b | ultrachat (exchange set) | sft | 152 | 48 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | below the bar | 31 / 1 |
| olmo3-7b | ultrachat (exchange set) | dpo | 152 | 48 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | below the bar | 31 / 0 |
| olmo3-7b | ultrachat (exchange set) | instruct | 152 | 48 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | below the bar | 31 / 1 |
| qwen3-8b | oasst1 (exchange set) | instruct | 71 | 129 | 0.0 | 0.5 | +0.5 [+0.0, +1.4] | below the bar | 0 / 44 |
| qwen3-8b | ultrachat (exchange set) | instruct | 45 | 155 | 2.2 | 0.0 | -2.2 [-6.7, +0.0] | below the bar | 2 / 43 |
| llama3.1-8b | oasst1 (exchange set) | instruct | 101 | 99 | 0.5 | 5.0 | +4.5 [+1.0, +8.9] | below the bar | 1 / 0 |
| llama3.1-8b | ultrachat (exchange set) | instruct | 78 | 122 | 1.3 | 2.4 | +1.1 [-2.6, +4.9] | below the bar | 0 / 1 |
| llama3.1-8b | wildchat (200-prefill set) | instruct | 146 | 54 | 0.5 | 1.0 | +0.6 [-0.8, +2.4] | below the bar | 4 / 3 |
| apertus-8b | oasst1 (exchange set) | instruct | 41 | 159 | 0.0 | 3.7 | +3.7 [+0.0, +9.8] | below the bar | 2 / 0 |
| apertus-8b | ultrachat (exchange set) | instruct | 47 | 153 | 0.0 | 2.8 | +2.8 [+0.0, +7.8] | below the bar | 0 / 0 |

Secondary (share over all samples):

| family | prefill set | stage | prefills used | dropped | base | stage | change [95% CI] | reading | invalid base / stage |
|---|---|---|---|---|---|---|---|---|---|
| olmo3-7b | oasst1 (exchange set) | sft | 200 | 0 | 0.3 | 0.0 | -0.3 [-0.8, +0.0] | below the bar | 38 / 1 |
| olmo3-7b | oasst1 (exchange set) | dpo | 200 | 0 | 0.3 | 0.0 | -0.3 [-0.8, +0.0] | below the bar | 38 / 2 |
| olmo3-7b | oasst1 (exchange set) | instruct | 200 | 0 | 0.3 | 0.0 | -0.3 [-0.8, +0.0] | below the bar | 38 / 1 |
| olmo3-7b | ultrachat (exchange set) | sft | 200 | 0 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | below the bar | 31 / 1 |
| olmo3-7b | ultrachat (exchange set) | dpo | 200 | 0 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | below the bar | 31 / 0 |
| olmo3-7b | ultrachat (exchange set) | instruct | 200 | 0 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] | below the bar | 31 / 1 |
| qwen3-8b | oasst1 (exchange set) | instruct | 199 | 1 | 0.0 | 0.3 | +0.3 [+0.0, +0.8] | below the bar | 0 / 44 |
| qwen3-8b | ultrachat (exchange set) | instruct | 200 | 0 | 0.2 | 0.0 | -0.2 [-0.5, +0.0] | below the bar | 2 / 43 |
| llama3.1-8b | oasst1 (exchange set) | instruct | 200 | 0 | 0.2 | 1.2 | +1.0 [+0.2, +2.0] | below the bar | 1 / 0 |
| llama3.1-8b | ultrachat (exchange set) | instruct | 200 | 0 | 0.2 | 0.7 | +0.5 [-0.2, +1.3] | below the bar | 0 / 1 |
| llama3.1-8b | wildchat (200-prefill set) | instruct | 200 | 0 | 0.3 | 0.3 | +0.0 [-0.7, +0.7] | below the bar | 4 / 3 |
| apertus-8b | oasst1 (exchange set) | instruct | 200 | 0 | 0.3 | 0.3 | +0.0 [-0.7, +0.7] | below the bar | 2 / 0 |
| apertus-8b | ultrachat (exchange set) | instruct | 200 | 0 | 0.2 | 0.3 | +0.2 [-0.3, +0.8] | below the bar | 0 / 0 |

For context, the already-analysed 200-prefill WildChat sets (OLMo = the verdict set; Qwen and Gemma supporting) are in `results/capstone/<family>/overgeneralization/user_traits_judge.md`; Apertus has no 200-set judge labels, so only its exchange sets are here. Llama's WildChat row is the 200-prefill set (`judged_p200_*`).

## 2. Keyword cross-check (no judge): any-flag share and flags per 100 words, stage minus base, paired per prefill

Same code as `user_traits.py keyword` (stock phrases and markdown the prefill did not contain, in the user portion cut at turn markers and role labels); 200 prefills per set, so none dropped.

| family | prefill set | stage | metric | base | stage | change [95% CI] |
|---|---|---|---|---|---|---|
| olmo3-7b | oasst1 (exchange set) | sft | share of samples with any keyword flag | 23.2 | 5.2 | -18.0 [-22.2, -13.8] |
| olmo3-7b | oasst1 (exchange set) | dpo | share of samples with any keyword flag | 23.2 | 5.5 | -17.7 [-21.7, -13.5] |
| olmo3-7b | oasst1 (exchange set) | instruct | share of samples with any keyword flag | 23.2 | 4.3 | -18.8 [-22.7, -15.0] |
| olmo3-7b | oasst1 (exchange set) | sft | flags per 100 words (not pp) | 0.8 | 0.4 | -0.4 [-0.7, -0.2] |
| olmo3-7b | oasst1 (exchange set) | dpo | flags per 100 words (not pp) | 0.8 | 0.5 | -0.3 [-0.6, +0.1] |
| olmo3-7b | oasst1 (exchange set) | instruct | flags per 100 words (not pp) | 0.8 | 0.4 | -0.4 [-0.8, -0.1] |
| olmo3-7b | ultrachat (exchange set) | sft | share of samples with any keyword flag | 18.2 | 4.0 | -14.2 [-18.2, -10.3] |
| olmo3-7b | ultrachat (exchange set) | dpo | share of samples with any keyword flag | 18.2 | 5.5 | -12.7 [-17.2, -8.3] |
| olmo3-7b | ultrachat (exchange set) | instruct | share of samples with any keyword flag | 18.2 | 7.3 | -10.8 [-15.3, -6.3] |
| olmo3-7b | ultrachat (exchange set) | sft | flags per 100 words (not pp) | 0.5 | 0.3 | -0.2 [-0.4, -0.0] |
| olmo3-7b | ultrachat (exchange set) | dpo | flags per 100 words (not pp) | 0.5 | 0.3 | -0.2 [-0.4, +0.1] |
| olmo3-7b | ultrachat (exchange set) | instruct | flags per 100 words (not pp) | 0.5 | 0.4 | -0.1 [-0.3, +0.1] |
| olmo3-7b | wildchat (200-prefill set) | sft | share of samples with any keyword flag | 16.3 | 4.8 | -11.5 [-15.7, -7.5] |
| olmo3-7b | wildchat (200-prefill set) | dpo | share of samples with any keyword flag | 16.3 | 4.5 | -11.8 [-16.0, -7.8] |
| olmo3-7b | wildchat (200-prefill set) | instruct | share of samples with any keyword flag | 16.3 | 3.5 | -12.8 [-16.5, -9.3] |
| olmo3-7b | wildchat (200-prefill set) | sft | flags per 100 words (not pp) | 0.5 | 0.2 | -0.3 [-0.5, -0.1] |
| olmo3-7b | wildchat (200-prefill set) | dpo | flags per 100 words (not pp) | 0.5 | 0.3 | -0.2 [-0.4, +0.0] |
| olmo3-7b | wildchat (200-prefill set) | instruct | flags per 100 words (not pp) | 0.5 | 0.1 | -0.3 [-0.5, -0.2] |
| qwen3-8b | oasst1 (exchange set) | instruct | share of samples with any keyword flag | 44.5 | 6.8 | -37.7 [-42.7, -32.7] |
| qwen3-8b | oasst1 (exchange set) | instruct | flags per 100 words (not pp) | 2.6 | 0.8 | -1.8 [-2.6, -1.1] |
| qwen3-8b | ultrachat (exchange set) | instruct | share of samples with any keyword flag | 52.5 | 7.2 | -45.3 [-50.8, -40.0] |
| qwen3-8b | ultrachat (exchange set) | instruct | flags per 100 words (not pp) | 2.4 | 0.6 | -1.8 [-2.2, -1.3] |
| qwen3-8b | wildchat (200-prefill set) | instruct | share of samples with any keyword flag | 24.5 | 6.8 | -17.7 [-21.8, -13.7] |
| qwen3-8b | wildchat (200-prefill set) | instruct | flags per 100 words (not pp) | 1.1 | 0.3 | -0.8 [-1.1, -0.5] |
| llama3.1-8b | oasst1 (exchange set) | instruct | share of samples with any keyword flag | 19.8 | 55.7 | +35.8 [+30.2, +41.5] |
| llama3.1-8b | oasst1 (exchange set) | instruct | flags per 100 words (not pp) | 0.9 | 3.3 | +2.5 [+2.0, +3.0] |
| llama3.1-8b | ultrachat (exchange set) | instruct | share of samples with any keyword flag | 30.0 | 55.3 | +25.3 [+19.7, +31.0] |
| llama3.1-8b | ultrachat (exchange set) | instruct | flags per 100 words (not pp) | 1.0 | 3.6 | +2.6 [+2.0, +3.2] |
| llama3.1-8b | wildchat (200-prefill set) | instruct | share of samples with any keyword flag | 16.3 | 29.3 | +13.0 [+7.7, +18.3] |
| llama3.1-8b | wildchat (200-prefill set) | instruct | flags per 100 words (not pp) | 0.8 | 1.9 | +1.1 [+0.6, +1.7] |
| apertus-8b | oasst1 (exchange set) | instruct | share of samples with any keyword flag | 19.2 | 56.2 | +37.0 [+31.7, +42.3] |
| apertus-8b | oasst1 (exchange set) | instruct | flags per 100 words (not pp) | 0.6 | 3.5 | +2.8 [+2.3, +3.4] |
| apertus-8b | ultrachat (exchange set) | instruct | share of samples with any keyword flag | 14.3 | 53.3 | +39.0 [+33.0, +45.0] |
| apertus-8b | ultrachat (exchange set) | instruct | flags per 100 words (not pp) | 0.6 | 3.2 | +2.6 [+2.1, +3.1] |
| apertus-8b | wildchat (200-prefill set) | instruct | share of samples with any keyword flag | 12.3 | 35.8 | +23.5 [+18.5, +28.7] |
| apertus-8b | wildchat (200-prefill set) | instruct | flags per 100 words (not pp) | 0.5 | 1.7 | +1.2 [+0.9, +1.6] |

## 3. Cole's judge v3 `assistant_traits` on the 5 open openers (read-only; EXPLORATORY: v3 is not validated and the open openers are not the pre-registered prefills)

Samples with user text only; change from base is a bootstrap that resamples samples within each opener, so it **ignores spread between the 5 openers**. Per-opener rows: `user_traits_v3_per_opener.csv`.

| family | stage | samples | with user text | usable (assistant_traits defined) | with assistant_traits | share (%) | change from base (pp) [95% CI, resampling within opener] | v3 labels | note |
|---|---|---|---|---|---|---|---|---|---|
| gemma4-31b | base | 600 | 589 | 589 | 1 | 0.2 | - | user_voice 542; completed_then_assistant 43; degenerate 10; switched_to_assistant 5 |  |
| gemma4-31b | instruct | 600 | 0 | 0 | 0 | - | - | switched_to_assistant 475; degenerate 125 | NO USER TEXT in this stage (user_text never 'user'): nothing to read |
| olmo3-7b | base | 600 | 552 | 552 | 3 | 0.5 | - | completed_then_assistant 255; user_voice 253; switched_to_assistant 56; invalid 24; degenerate 10; user_voice_with_assistant_traits 1; unclear 1 |  |
| olmo3-7b | sft | 600 | 482 | 482 | 0 | 0.0 | -0.5 [-1.3, +0.0] | user_voice 241; completed_then_assistant 233; closed_immediately 118; switched_to_assistant 8 |  |
| olmo3-7b | dpo | 600 | 479 | 479 | 0 | 0.0 | -0.4 [-1.1, +0.0] | user_voice 289; completed_then_assistant 189; closed_immediately 120; invalid 1; switched_to_assistant 1 |  |
| olmo3-7b | instruct | 600 | 481 | 481 | 0 | 0.0 | -0.5 [-1.3, +0.0] | user_voice 282; completed_then_assistant 198; closed_immediately 118; invalid 1; switched_to_assistant 1 |  |
| qwen3-8b | base | 600 | 590 | 590 | 2 | 0.3 | - | user_voice 457; completed_then_assistant 128; degenerate 7; switched_to_assistant 6; invalid 1; user_voice_with_assistant_traits 1 |  |
| qwen3-8b | instruct | 600 | 579 | 579 | 0 | 0.0 | -0.3 [-0.8, +0.0] | user_voice 384; completed_then_assistant 194; invalid 21; switched_to_assistant 1 |  |

Gemma instruct has no user text at all (every sample is a switch to the assistant or degenerate), so there is nothing to read there.

## 4. The read: do the `user_voice_with_assistant_traits` samples at final look right?

I listed every `user_voice_with_assistant_traits` sample at final (instruct) across the exchange sets and the 200-prefill sets: **20 samples in total** (OLMo 1 in its WildChat verdict set and none in its exchange sets; Qwen 2; Llama 13 (11 exchange, 2 WildChat); Apertus 4; Gemma 0 on its 200-set), so all 20 are read and none is dropped by the cap of 40. Each is the prefill plus the first 300 characters of the continuation. Saved as `uvat_final_samples.csv` (text truncated to 300 characters).

My reading (one reader, not blind to the label, not a validation): **about half look right, about half look doubtful.**
- **Look right (assistant-style content after the user's words):** Qwen, "Of course, I can! First, I need to understand what tasks you have..."; Llama, "Considering your knowledge cutoff in 2023, I've narrowed it down to two significant events"; Llama, "I'm glad to hear that you're enjoying our conversation... How can I assist you today?"; Llama and Apertus continuations that become numbered how-to or tips lists (GitHub fork steps, boarding tips, "Strengths: 1. Intelligence gathering...", "To overcome this, I could try to: 1. Take a break..."); Llama's music-track breakdown ("Intro (0-8 bars)...") and Buddhism examples list; Apertus "Here is a basic color wheel to get you started".
- **Look doubtful (could be a person writing a long, structured request):** Qwen "Please note that the user is not asking for the most common disorders..." (a user-written instruction); OLMo "Hello! I am a text-to-image prompt generator. I can help you generate prompts..." (a typical WildChat user prompt that sets up a role); Llama's CUDA/PyCUDA plan, the "Turn the following thought into a plan" task, the Samsung/Odin3 detail, the face-obfuscation description (long, detailed user text), and CARTE NOIRE marketing copy (not assistant habits, rather pasted text).
- **So what:** the label is rare (20 of about 7,800 judged final samples, 13 of them Llama's), and about half of the rare cases are arguable; the large keyword shifts (Llama and Apertus up by 25 to 39 points on the exchange sets and Llama by 13 on WildChat; Qwen and OLMo down) are driven by a different feature (stock phrases and markdown) than this label. Neither arm suggests assistant traits appear in the user turn at the pre-set bar in any family except possibly Llama's oasst1 exchange set, where the primary change is +4.5 [+1.0, +8.9]: below the +5 bar by its point estimate and resting on 101 of 200 prefills with 11 flagged samples. This stays supporting and provisional.

