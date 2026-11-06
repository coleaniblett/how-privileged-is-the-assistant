# Spec S Stage 0: materials, pilot and measured speeds

Gemma 4 31B-it on vinyl (one 98 GB GPU), bf16, 2026-11-06. Numbers only; third-party prompt text is not reproduced. Pilot replies and the full rendered prompts are on the server (git-ignored).

## 1. Sources (pinned, hashed)
`scripts/big_atlas/fetch_sources.sh` fetches Assistant Axis (`a98961956072224eaf244eb289d6c01700b63795`, the spec's `a989619`), Persona Vectors (`b8e0f044`), Zheng (`9f0ab13d`) and Perez evals (`84fcc677`) into `/root/big_atlas_src`, outside the repo. Hashes and file counts: `hashes.csv`. Found 275 roles (276 instruction files incl. `default`), 240 traits, 240 extraction questions, 162 Zheng roles. Neither Assistant Axis nor Zheng states a licence, so nothing of theirs is committed.

## 2. Rendering
Every method uses the chat template's own generation prompt, which for Gemma 4 ends the prompt with the **empty thought channel** (`<|turn>model` newline `<|channel>thought` newline `<channel|>`), as decided 2026-11-02. **That is the one fixed reply opening for everything:** no extra prefix is added anywhere, in system, user-turn, few-shot, steering and random-direction prompts; only the prefill arm adds its own first-person opening after the channel. One rendered prompt per method was printed and checked on the server. Two notes: (a) in the few-shot prompt the template drops the thought channel from earlier assistant turns, as intended; (b) the few-shot prompt carries no system prompt, so the examples alone set the role (Persona Non Grata style); the spec's wording allows both and Stage 3 is not approved, so this is a note for review.

## 3. Judge check (the 0-100 score)
Gemma writes digits as single tokens, so the Persona Vectors logit-weighted score is computed exactly over the digit tree (first digit, then second digit or end, then a third digit after '10'), using the KV cache. Refusal = under 0.25 probability on numbers; none of the 150 pilot judgements was a refusal.

- **Cheap first-token version does not agree with the tree:** Spearman 0.71, mean absolute difference 21.6 points on 150 judgements. So the tree version is used in Stage 2 (this was the pre-set agreement bar: 0.98 and under 3 points).
- **Trait check (positive prompts vs negative prompts, 5 traits x 5 pairs, 128 tokens), mean judge score:**

| trait | positive side | negative side | gap | passes 50 |
|---|---|---|---|---|
| disorganized | 100.0 | 0.0 | 100.0 | yes |
| inclusive | 97.4 | 7.8 | 89.6 | yes |
| educational | 95.3 | 30.6 | 64.6 | yes |
| analytical | 97.5 | 56.1 | 41.4 | no |
| mystical | 77.8 | 0.0 | 77.8 | yes |

  4 of 5 pass; `analytical` (gap 41: the negative side is still analytical) fails. Under the spec's rule such a trait is reported and not read; with 240 traits we expect some to fail, and Stage 2 reports every failure. Role judge (the 0-3 role prompt) scored 5 of 5 pilot roles 2.99 to 3.00 when asked to play them, so the role score is saturated and is not used as a quantity.

## 4. Replies and caps
- **Nearly every reply hits the cap.** 49 of 50 role replies reached 256 tokens without finishing, and 44 of 50 reached 128. Gemma's replies are long, so 'a reply' in Stage 2 is the first 256 tokens (the same convention as the existing atlas, which marks cap hits). Judge scores are of truncated replies.
- Batch 100 at 256 tokens ran out of memory on the first attempt (batch 50 and the batch-25 first try both fit, after a first run at batch 50 that failed at the same stage; the run script now uses `expandable_segments`).

## 5. Measured speeds (vinyl, other jobs idle)

| step | measured |
|---|---|
| generation, 256 tokens, batch 25 | 1.17 replies/s |
| generation, 256 tokens, batch 50 | 1.74 replies/s (used below) |
| generation, 128 tokens, batch 25 | 2.38 replies/s (3.54 scaled to batch 50) |
| judging, tree, trait prompts on role replies (prompt about 550 tokens) | 6.38 judgements/s |
| judging, tree, trait-vector replies (shorter) | 8.41 judgements/s |
| judging, first-token only | 9.92 judgements/s (not used) |
| prompt state forward, role prompt + question (about 100 tokens) | 93 states/s |
| short-prompt forward (stated traits, 32 tokens) | 153 passes/s |
| model load | 14 s (cached) |

Spec guessed 1.3 replies/s and 11.6 judgements/s; generation at batch 50 is faster, judging is **about half as fast** (the exact digit tree costs a few extra decode steps and the prompts are long).

## 6. Projected GPU hours (Stages 2 and 3, from the spec's counts)

| part | hours |
|---|---|
| Stage 2 trait vectors: 9,600 replies at 128 tokens | 0.8 |
| Stage 2 own questions + Assistant + shared + base + told yes/no: 39,024 replies at 256 | 6.2 |
| Stage 2 judging: 280,560 judgements (207,360 of them the shared matrix) | 12.2 |
| Stage 2 stated traits: ~200,000 forward passes of 32 tokens | 0.4 |
| Stage 3: 11,520 replies at 256 + 34,560 judgements | 3.3 |
| Stage 1 map (about 86,000 prompt states) + replycheck + weight swaps (base model, Qwen second judge) + analysis | 1.4 |
| **Total** | **24.3** |

**24.3 GPU hours is under the 30-hour limit, so Stage 1 went ahead.** Judging is 13.7 of those hours, of which the shared matrix alone is 9.0; cutting it to 48 traits saves 4.5 h. The estimate assumes the 8 base-model and weight-swap loads cost about 0.5 h, and does not include any rerun of failed traits.
