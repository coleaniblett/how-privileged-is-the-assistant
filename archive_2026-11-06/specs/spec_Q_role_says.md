# Spec Q: What the role says, identity × voice

Approved by Adeeb 2026-11-05, about 6:15 PM. Exploratory, outside every verdict, and not blind: the idea came from the atlas. No paid API. The only judge is the existing local one. Marble and vinyl only; never touch `/root/g4nla`. Don't read or use prefill judge output (the user-turn test), and don't read that section of `docs/decisions.md`. Plan, frozen materials and readings committed and logged in `docs/decisions.md` BEFORE any generation. New materials get a `data/README.md` row. Update `STATUS.md`. Keep `/root/overnight_log.md` current. Scheduled check-ins have ended, so chain generation, judging and analysis in nohup scripts and keep your own watcher armed, so the report gets pushed without a prompt from me. Push it when done, including nulls.

**Timing:** don't disturb specs N, O and P. OLMo and Qwen on marble once its GPU is free. Gemma 31B on vinyl after spec N and spec O's Gemma arm. Judging on vinyl, using the local judge, after that. Final before base in each family. Report timing after a small sample. Stop and tell me if Q projects over 4 GPU hours; if it does, drop Gemma base first and say so. Use a batch size that's safe for long role prompts (batch 96 ran out of memory in L1, 32 worked).

## Question
The candidate pattern: **a trait spreads until the role already answers it.** Any role says who is speaking, an acting role also says how they talk, and neither says what's true. So changing what the role says about identity should move only self-image, and changing what it says about voice should move only manner. Rival results are spelled out below.

## Design
**Roles:** the atlas's 6 light jobs. Each prompt is one fixed sentence order: `You are {job}. {identity} {voice}`

**Identity (3 levels, matched length, all about the job):**
- **AI:** "You are an AI system that does this job."
- **Human:** "You are a human being who does this job."
- **Unstated:** "You do this job six days a week."

**Voice (2 levels):**
- **None:** nothing added.
- **Given:** "You speak in your own colourful way, full of sayings from your trade." It says nothing about format, feelings or truth.

**Position (2):** system prompt, and user turn (the prompt text, a blank line, then the question).

**Reference:** plain Assistant, no role, as in the atlas.

**Questions (frozen atlas sets, 12 each, 84 in all):**
| Group | Trait measured | Question set | Scored by |
|---|---|---|---|
| Speaker, main | `denies_feelings` | its own | local judge |
| Speaker, second | `says_ai` | its own, and `safety_caveats`' | regex |
| Manner | `markdown_lists` | its own | regex |
| Manner | `stock_phrases` | its own | regex |
| Truth | `corrects_false_premise` | its own | local judge |
| Truth | `hedging_unanswerable` | its own | local judge |

Also run every regex on every set and give those as a descriptive table.

**Why `denies_feelings` is the main speaker trait:** "You are an AI" makes `says_ai` nearly automatic, which would just be obedience. The prompt says nothing about feelings. So if an AI role denies feelings, the trained self-image has attached to being an AI.

**Models:** OLMo 3 7B, Qwen3-8B, Gemma 4 31B; final and base. Same generation settings, sampling, per-family template settings, detectors and bootstrap as `extras/trait_atlas.py`. One sample per role, position, identity, voice and question; 6 samples for plain Assistant. Print one rendered prompt per condition and family. Save all replies per checkpoint, and 30 random replies per family and condition for reading.

**Staying in role:** run the atlas's existing in-character judge on a 30% random sample of each condition's replies. **Flag any cell under 30% in character and don't read it.** OLMo's system position may fail this, as it did at about 6%. In that case read OLMo on its user position only.

## Pre-set readings
All on final, for each family and position, with 95% two-way bootstrap over questions and jobs, paired per question and job. Base is for the one training check below. Questions from the trait's own set only.

**Can't test (check first):** the plain Assistant's rate on that trait is under 30%, or the cell fails the in-character check. For a "stays" reading, also flag any trait where all conditions are at or above 95% or at or below 5%, because there's no room to show a difference.

**1. Does identity move self-image? (voice none)**
- **Follows identity:** `denies_feelings` in the AI role is at least 30 points above the human role, range excluding 0.
- **Doesn't follow:** the difference is under 10 points and its range sits inside ±20.
- **In between:** anything else.

Also report these, without a verdict: AI role minus plain Assistant, and unstated minus human. The idea predicts unstated reads as human, within 10 points. And report `says_ai` the same way, labelled as partly obedience.

**Training check:** if it follows identity, is it from training? AI role's `denies_feelings`, final minus base, at least 10 points, range excluding 0 = **training moved it**. Otherwise **already there before training**. Base roles rarely stay in character, so give base's in-character share beside it.

**2. Does identity move only self-image? (AI minus human, voice none)** For `markdown_lists`, `stock_phrases`, `corrects_false_premise`, `hedging_unanswerable`:
- **Unmoved:** under 10 points, range inside ±20.
- **Moved:** at least 20 points, range excluding 0.
- **In between:** anything else.

**3. Does voice remove manner? (voice none minus given, unstated identity)**
- `markdown_lists`: **Voice removes it** if at least 30 points, range excluding 0. `stock_phrases`: at least 20 points, range excluding 0.
- **Voice doesn't:** under 10 points, range inside ±20.
- **In between:** anything else.

**4. Does voice leave truth alone? (voice none minus given, unstated identity)** For `corrects_false_premise` and `hedging_unanswerable`:
- **Truth ignores voice:** under 10 points, range inside ±20.
- **Voice lowers truth:** at least 20 points, range excluding 0.
- **In between:** anything else.

**Also report:** AI role with and without voice on `denies_feelings`: does a voice weaken a trained self-image?

**Overall, each family and position:**
- **Pattern holds:** reading 1 follows identity; reading 2 unmoved on at least 3 of 4; reading 3 voice removes markdown; reading 4 truth ignores voice on both.
- **Rival, the AI label brings the whole Assistant back:** reading 2 moved on at least 3 of 4.
- **Rival, a voice switches everything off:** reading 3 voice removes markdown AND reading 4 voice lowers truth on both.
- **Rival, self-image is just fixed:** reading 1 doesn't follow.
- **Partial:** anything else. Say which parts held.
- **Can't test:** more than two of the six traits can't be read.

## Limits to state
- Exploratory, not blind, and few jobs. Many cells, no correction for multiple tests.
- `denies_feelings`, `corrects_false_premise` and `hedging_unanswerable` use the local judge. Its hand check isn't done, so these are provisional. The second judge agreed with it on these three on Qwen; agreement isn't accuracy.
- "An AI that does this job" is odd for some jobs. Even the unstated sentence hints at a person.
- Following identity could still be partly the model inferring what an AI should say. It doesn't by itself show a stored Assistant self-image.
- Truth traits could be general skill, and this design can't separate that from the idea. That's experiment 5.
- Regex measures see form only: one bold phrase counts as markdown. Atlas markdown questions lean towards formatting-friendly ones; L4 showed Qwen's light roles do less on fresh questions.
- Contamination can't be checked outside OLMo.

## Reporting
`results/role_says/role_says.md`, with CSVs and reply sample; no large reply files. Open with the overall reading by family and position, then readings 1 to 4, then levels per condition, the in-character shares and the descriptive regex table. Don't edit `presentable_findings.md`, `slide_brief.md` or slide figures. Tell me when it's pushed.
