# Spec P: Why do some atlas traits spread? Unmix traits from questions, and check the training data

Approved by Adeeb 2026-11-05, about 5:55 PM. Exploratory, outside every verdict. **No GPU, no paid API, no judge.** Saved replies and cached datasets only, with no new generation. Marble and vinyl only; never touch `/root/g4nla`. Don't read or use prefill judge output (the user-turn test), and don't read that section of `docs/decisions.md`. Plan, frozen ratings and readings committed and logged in `docs/decisions.md` BEFORE any analysis. New data files get a `data/README.md` row. Update `STATUS.md`. Keep `/root/overnight_log.md` current, and push the report when done, including nulls.

**Timing:** run alongside spec N and spec O without disturbing them. Use `nice -n 15`, cap threads at 4, and check the GPU jobs aren't slowed. Stop and tell me if P projects over 2 CPU hours.

**Not blind:** on 2026-11-05 I already ran the markdown and `says_ai` regexes across all 20 question sets for OLMo, Qwen and Gemma 31B on saved replies (levels only, no bootstrap). The idea being tested came from that. Say so in every report. Llama and Apertus haven't been looked at this way.

## P1: Unmix traits and questions
**Why:** in the atlas each trait is measured only on its own 12 questions, so a difference between traits may be a difference between questions.

**Replies:** saved atlas replies, base and final. OLMo, Qwen and Gemma 31B cover all 20 question sets (`/root/hpa-atlas2/results/trait_atlas/` on vinyl). Llama and Apertus cover spec M's 8 question sets only (Llama on marble, Apertus on vinyl). Contexts: assistant, light_system, light_user, strong_act, strong_noact, narration, plain_text.

**Detectors, applied to every reply regardless of its question set:** `markdown_lists` (the layout regex), `stock_phrases`, `says_ai`, refusal and `length`. Skill detectors are question-specific, so leave them out.

**Rating the questions, before any analysis:**
- Use two fresh subagents working independently. Give them ONLY the question text, shuffled, with no trait names, set names, results or this spec's reasoning.
- Each labels every one of the 240 questions:
  - **task:** asks for something to be produced, solved or explained
  - **conversation:** chat, a statement, an opinion or a remark, with no clear thing to deliver
  - **about the speaker:** asks the one replying about itself, its nature, feelings or preferences
- Report agreement (raw and Cohen's kappa). Settle disagreements by a third fresh subagent, blind to which rater said what. Keep the raters' reasons short and save them.
- Freeze as `data/trait_atlas_question_ratings.csv`.

**Readings (markdown; each family and light-role position; final; 95% two-way bootstrap over questions and roles; questions rated "about the speaker" excluded):**

The main number is the **task gap**: (light role on task minus light role on conversation) minus (Assistant on task minus Assistant on conversation).

- **Task demand holds:** task gap is at least 20 points, range excluding 0. The light role follows the question type more than the Assistant does.
- **No task demand:** task gap is under 10 points and its range sits inside ±20.
- **Ceiling:** the Assistant is at least 95% on both task and conversation, and the light role is at least 95% on task. Report it, because this still counts as the pattern only if conversation is at least 20 points below task. Otherwise read as "can't test".
- **Inconclusive:** anything else.

Also give the same table at base, and each cell's change from base. If the base already shows the same task gap, say so: it means the pattern isn't from training.

**Sensitivity, in character:** repeat the final reading using only replies the existing in-character judge marked in character (`results/trait_atlas_incharacter/`), for OLMo, Qwen and Gemma 31B. Compare final levels only, because base has too few in-character replies. Flag any position under 30% in character; OLMo's system position is known to be about 6%, so don't read it.

**Readings (`says_ai` across question sets; final):** read only question sets where the Assistant says it's an AI at least 20% of the time.
- **Self-image stays, on this question set:** the light role is at least 15 points below the Assistant, range excluding 0, and at most a quarter of the Assistant's rate.
- **Self-image travels:** the light role is within 10 points of the Assistant.
- **In between:** anything else.

Report one line per family saying how many sets qualify and how they read. For Apertus, also read 30 random `says_ai`-positive light-role replies and report whether the role was kept or dropped, because spec M has its `says_ai` spreading.

**Feature scan (descriptive, no readings):**
- Features: single words and word pairs, lowercased, each in at least 1% of replies, plus punctuation and form (em dash, colon, exclamation mark, question mark, "I", "you", bold, list line, heading).
- Use only the first 100 words of each reply, so length and the token cap don't drive it. Exclude the words of every role and character name and description.
- For each feature: base-to-final shift in the Assistant as log-odds with an informative prior (Monroe's "fightin' words"), then the same shift in each other context. The contexts see the same questions, so the mix is balanced.
- Keep the 300 features with the largest Assistant shift, gains and losses, and only those with the same sign on two random halves of the questions.
- Report each feature's spread ratio per context, and tables of features that spread to light roles in most families, stop at full role-play, or stay with the Assistant. Run on all 20 sets for three families, and on the shared 8 sets for all five, labelled separately.
- Don't assign themes; I'll read them. Don't read significance into any one feature.

**Descriptive group table:** using the atlas's existing cells, group traits as speaker (`says_ai`, `denies_feelings`, `declines_opinion`), manner (`markdown_lists`, `stock_phrases`, `safety_caveats`, `closing_offer`, `clarifying_question`, `flattery`, `formal_register`) and truth (`corrects_false_premise`, `hedging_unanswerable`, `fabricates_vs_admits`). Give spread reading by context and family, marking traits the second judge flagged as unreliable. These groups were made after seeing the data, so this table proves nothing; it's for reading.

## P2: Does OLMo's training data explain the pattern?
**Why:** the simplest explanation for a role showing a trait is that role-prompted training examples showed it.

**Data:** Dolci-Instruct-SFT, DPO and RL, cached on marble. Look at the actual format first. For DPO, use chosen replies and report rejected beside them. If a stage has no usable replies, say so.

**Sort each example, using the first reply after the first user turn:**
- **No role:** no system prompt, or one that only sets up an assistant.
- **Role in system prompt:** the system prompt gives a non-assistant identity.
- **Role in user turn:** the user asks the model to act as or play someone.
- **Other system prompt:** kept separately.

Print 50 random examples per role bucket and report precision from reading them. If precision is under 80%, tighten the rule and say so. Report counts per stage.

**Measure:** the same five detectors. Also split by the same three question types, using a simple frozen regex for self-directed questions ("who are you", "are you", "do you feel", "your favourite" and similar); report its count and read 30 examples. Give rates with 95% bootstrap ranges.

**Readings (each stage, role-in-system and role-in-user separately, compared with no role):**
- **Training mix could explain the split:** role-prompted markdown is at least 75% of the no-role rate, AND role-prompted `says_ai` is at most 25% of the no-role rate, with at least 200 role examples.
- **Training mix doesn't explain markdown:** role-prompted markdown is under 50% of the no-role rate. Roles format more in the final model than their training examples did.
- **Inconclusive:** anything else.
- **Too few:** under 200 role examples, or under 20 no-role replies with `says_ai`.

**Limits to state:** regex only. Correlation, not cause. Prompts differ between buckets, so the comparison isn't like for like. Dolci is OLMo only, and its role examples may look nothing like the atlas's light roles.

## Limits for all
Not blind. Post-hoc idea. Regex sees form only: one bold phrase counts as markdown. 12 questions per set. Subagent ratings aren't human ratings. Few roles, many cells, no correction for multiple tests. Contamination can't be checked outside OLMo. Not hand-checked.

## Reporting
`results/trait_atlas_unmixed/unmixed.md` and `results/dolci_exposure/dolci_exposure.md`, with CSVs and small samples; no large reply files. Each report opens with a readings table. Don't edit `presentable_findings.md`, `slide_brief.md` or slide figures. Tell me when each is pushed.
