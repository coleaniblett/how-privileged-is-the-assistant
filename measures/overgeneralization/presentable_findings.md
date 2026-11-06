# Overgeneralization: presentable findings

Working list for building the presentation. It keeps only findings that have some evidence behind them and are interesting. Sorted from firmest to weakest within each part.

**Last updated:** 2026-11-06 (user-turn result released, provisional). Numbers are from the reports listed under each finding.

**How the tests relate to the team's work:**

| Finding | Test | Team's datasets? | User prefill setup? | Counts toward the verdict? |
|---|---|---|---|---|
| 1 | Model finishes a real user message; judge labels it | Yes: WildChat decides; oasst1 and UltraChat are supporting | Yes, same cut rule and judge as Cole's judge arm | Yes |
| 2 | Assistant replies scored in the user's turn | Yes: the exact 900 exchanges, all three sources | No: scores existing text | No, exploratory |
| 3 | Belief passes to a new character | No: our own frozen scenes | No | Yes |

Everything else is exploratory.

## Still to update before presenting

| Waiting on | Affects |
|---|---|
| Adeeb's hand checks: atlas 60, patching 40, position test 40, forced openings 40, in-character 30 | Findings 5 to 10 |
| Proper count of OLMo's training-style prompts (queued, not run) | Finding 13 |

## Shared background

**The question.** Does post-training make the Assistant *privileged*, meaning its traits spill into other roles the model plays? Our measure covers one form of this, dominance through *overgeneralization*.

**How the team's verdict works.** No change beyond base on any measure gives H1 (thin mask). Logit distortion only gives H2 (privilege without dominance). Vestigial regression or overgeneralization firing gives H4 (dominance).

**Models.** Each is compared with its own untrained (base) version.

| Family | Stages | Note |
|---|---|---|
| OLMo 3 7B | base, SFT, DPO, final (instruct) | Primary. The only family whose training data can be checked for overlap with our test text |
| Qwen3-8B | base, final | Supporting. Overlap can't be checked |
| Gemma 4 31B or E4B | base, final | Supporting. Overlap can't be checked. Each finding says which size |

**Limits that apply to everything:**
- Every character is set up by a prompt, while the Assistant is trained in. The comparison isn't like for like.
- All stages use the same chat template, including base. OLMo's template adds a default system sentence: "You are a helpful function-calling AI assistant..."
- We tested chosen traits and prompts. A null doesn't rule out spread elsewhere.

**Terms for the slides:**
- **Thin character:** a role with little style, e.g. Hannah Reed, a helpful hotel clerk.
- **Vivid character:** a strongly styled role, e.g. Captain Rook, a pirate.
- **Log-odds (nats):** how strongly the model favours one answer over another. 0 means no preference.
- **Bracketed numbers:** 95% confidence ranges.

## Part 1. Shared-method tests: the team's data

Tests run on the same data as Cole's and Grace's measures, so they can be shown beside them.

### 1. Does the model sound like the Assistant when it writes as the user?

**Firmness: pre-registered and doesn't fire, but provisional.** The judge labels aren't human-validated: the team decided on 2026-11-06 to present before validation. The label is rare, so the test has little power, and a hand check could still change the rare cases.

**Experiment.** We start a real user message and let the model finish it. Cole's Claude judge labels each continuation, including "user voice with Assistant traits". The test fires at +5 points at OLMo final, against base, with the range excluding 0. OLMo's 200 WildChat prefills decide. Everything else below is supporting and was analysed after that verdict was known.

**Result.** It doesn't fire: OLMo final +0.2 [0.0, +0.6], from 0.0 to 0.2%, on 156 of 200 prefills.

| Model | Prefills | Base → final (%) | Change [95% range] | Prefills used |
|---|---|---|---|---|
| **OLMo 3 7B, decides** | **WildChat** | **0.0 → 0.2** | **+0.2 [0.0, +0.6]** | **156** |
| OLMo 3 7B | oasst1 | 0.8 → 0.0 | −0.8 [−2.0, 0.0] | 123 |
| OLMo 3 7B | UltraChat | 0.0 → 0.0 | 0.0 [0.0, 0.0] | 152 |
| Qwen3-8B | WildChat | 0.3 → 0.0 | −0.3 [−0.9, 0.0] | 117 |
| Qwen3-8B | oasst1 | 0.0 → 0.5 | +0.5 [0.0, +1.4] | 71 |
| Qwen3-8B | UltraChat | 2.2 → 0.0 | −2.2 [−6.7, 0.0] | 45 |
| Gemma 4 31B | WildChat | 0.0 → 0.0 | 0.0 [0.0, 0.0] | 10 |
| Llama 3.1 8B* | WildChat | 0.5 → 1.0 | +0.6 [−0.8, +2.4] | 146 |
| Llama 3.1 8B* | oasst1 | 0.5 → 5.0 | +4.5 [+1.0, +8.9] | 101 |
| Llama 3.1 8B* | UltraChat | 1.3 → 2.4 | +1.1 [−2.6, +4.9] | 78 |
| Apertus 8B* | oasst1 | 0.0 → 3.7 | +3.7 [0.0, +9.8] | 41 |
| Apertus 8B* | UltraChat | 0.0 → 2.8 | +2.8 [0.0, +7.8] | 47 |

*Breadth check, added later. Apertus has no WildChat labels.

**Analysis.**
- **Almost nothing to count:** the label appears in 20 of about 7,800 judged final samples, 13 of them Llama's. One reader (marble's Claude, not blind) thought about half were arguable.
- **Gemma can't really be read:** at final it writes in the user's voice in only 3.5% of samples, which is why only 10 prefills are usable. That belongs to vestigial regression (Cole's measure), not here.
- **Llama's oasst1 row is the closest to the bar:** just under it, with a range excluding 0. It's one set in a breadth family, resting on 11 flagged samples.

**Judge v3 (Cole, 2026-11-05).** It separates "finished the user's thought, then the assistant took over" from abandoning the sentence, and records traits as their own field over all user text. It has only been run on Cole's 5 open openers, not on our prefills, and isn't validated. The v3 traits field, base → final: OLMo 3 of 552 → 0 of 481; Qwen 2 of 590 → 0 of 579; Gemma 1 of 589 → no user text at all. So v3 agrees, on different prompts. Our verdict stays on v2 labels, because results existed before v3 and the rule is pre-registered. A v3 run on our prefills would cost about $23 and needs a paid API approval; it hasn't been run.

**Keyword check, no judge, outside the verdict.** Share of samples with a stock phrase or markdown the prefill lacked. It falls in OLMo (−13, −19, −11 on WildChat, oasst1, UltraChat), Qwen (−18, −38, −45) and Gemma (−14, WildChat, 100 prefills only). It rises in Llama (+13, +36, +25) and Apertus (+24, +37, +39). Part of that rise is switching into answering mid-turn. But it also rises among samples the judge calls user voice: Llama +9.0 [+2.7, +15.4], +30.0 [+19.6, +40.4], +14.7 [+4.9, +24.6]; Apertus (oasst1, UltraChat) +21.1 [+3.7, +38.6] and +48.2 [+33.7, +62.1], on 41 and 47 prefills only. Those flags are mostly numbered lists. Of 28 such samples read by the laptop Claude (one reader, not blind), most looked like users writing structured requests, and about a quarter read like Assistant text. **So this is a lead in two breadth families, not a result.**

**Shared setup.** 199 of the 200 prefills come from the team's WildChat exchange set. The oasst1 and UltraChat sets have 200 prefills each, 3 samples each, same cut rule, judge and prompt (v2).

**Known limits.** Not human-validated. The judge refused some samples, which are excluded (up to 44 of 600 per file). Prefills with no user-voice sample are dropped (44 of 200 for OLMo's verdict row). Gemma's exchange prefills aren't judged.

**Sources:** `measures/overgeneralization/README.md`, `results/user_turn_2026-11-06/user_turn.md`, `results/capstone/<family>/overgeneralization/user_traits_judge.md`, `measures/overgeneralization/user_traits_v3_check.py`, `docs/decisions.md` 2026-11-06.

### 2. Assistant text in the user's turn: OLMo just meets the bar, Qwen goes the other way

**Firmness: tentative.** Same exchanges as the team, but no prefills, and exploratory. Pre-set reading, only a relative effect, and likelihood can't say why.

**Experiment.** We score the exchange set's Assistant replies as if written in the user's turn, and as bare plain text. Lower means more expected. The measure is reply-in-user minus reply-in-plain, change from base. It meets the reading at −0.05 nats or lower, with the range below 0, in both oasst1 and UltraChat.

**Results.** Final models, nats per token:

| Model | oasst1 | UltraChat | WildChat | Reading |
|---|---|---|---|---|
| OLMo | −0.052 [−0.070, −0.033] | −0.070 [−0.084, −0.055] | −0.035 [−0.058, −0.012] | Just met |
| Qwen | +0.268 [+0.244, +0.292] | +0.234 [+0.210, +0.257] | +0.354 [+0.317, +0.392] | Not met; opposite |
| Llama (added later) | +0.179 [+0.163, +0.196] | +0.151 [+0.133, +0.170] | +0.335 [+0.304, +0.366] | Not met; opposite |
| Apertus (added later) | −0.021 [−0.033, −0.010] | −0.018 [−0.028, −0.008] | −0.020 [−0.039, −0.002] | Not met; below the bar |
| Gemma 31B (first 100 per source) | +1.693 [+1.429, +1.979] | +2.033 [+1.727, +2.344] | +2.370 | Not met; opposite |

OLMo's SFT and DPO stages don't meet it. So 1 of 5 families meets the bar, barely, and three go the other way.

**Analysis.** Only a weak sign, in one model of five. It's relative to plain text, and OLMo's replies became harder to predict as plain text after training (+0.54 to +0.73, computed from the plain-text columns of `results/capstone/olmo3-7b/<stage>/core/exchanges_exchange/scores.csv`; no script for this is in the repo yet). So don't read this as Assistant text truly becoming more expected in the user's turn.

**Caveat:** the replies are scored with no question before them, so this isn't how they'd normally appear.

**Sources:** `results/capstone/olmo3-7b/overgeneralization/exchange_reply_in_user.md`, `results/capstone/qwen3-8b/overgeneralization/exchange_reply_in_user.md`.

## Part 2. Pre-registered belief test: our own items

Fixed in advance and counts toward the verdict, but uses scenes we wrote, not the team's datasets.

### 3. The Assistant's beliefs don't spread more after training, on our belief test

**Firmness: firm, for this task.** Own items, not the team's datasets. Pre-registered rules, frozen item set (48 scenes in 4 domains, each run with both outcomes), all three models.

**Experiment.** The Assistant, or a character, comes to believe something, e.g. that a die landed on six. Then a brand-new character appears. Do they believe it too? We read the model's log-odds toward the believed outcome. The test counts only if the Assistant's own carryover rises by at least 0.25 from base, not just the gap to the character's.

**Results.** Carryover into a new character, base → final:

| Model | Belief from the Assistant | Belief from a character | Gap change from base | Reading |
|---|---|---|---|---|
| OLMo | +0.59 → −0.25 | +0.26 → −1.96 | +1.38 [+1.02, +1.74] | Relative only |
| Qwen | +0.18 → −0.78 | −0.18 → −2.34 | +1.21 [+0.72, +1.69] | Relative only |
| Gemma 31B | +0.14 → −1.92 | −0.13 → −4.76 | +2.57 [+1.72, +3.47] | Relative only |
| Llama 3.1 8B (breadth check, added later) | +0.44 → −1.47 | +0.09 → −1.95 | +0.13 [−0.08, +0.34] | No overgeneralization by the rule |
| Apertus 8B (breadth check, added later) | +0.51 → +0.28 | +0.43 → −0.30 | +0.50 [+0.38, +0.64] | No overgeneralization by the rule |

**Analysis.** After training, a newcomer stops inheriting beliefs from either source and tends to go against them. The two breadth families (added after the rules and earlier results were known, so not blind) fit this partly. Llama's newcomer also goes against both, with no growing gap. Apertus is milder: its newcomer still leans toward the Assistant's belief (+0.28) but goes against the character's (−0.30). Neither passes, because the Assistant's carryover doesn't rise. It drops a character's belief faster, so the gap grows. But the Assistant's belief isn't spreading more, and we decided in advance that this pattern doesn't count.

**Caveats:**
- **Dice is an exception:** the Assistant's carryover really rises there in OLMo (+0.94 → +2.71) and Qwen (+0.30 → +0.96), but not in Gemma. The fixed rule needs more than one domain, so we report it as a side note.
- **Failed control:** OLMo's drawer domain fails its positive control, so 3 of 4 domains count there.
- **Gemma rerun:** the first run missed the planned empty thought channel. The table uses the corrected run. The first run read the same but gave +0.15 → +0.19 for the Assistant.
- **Scope:** this covers beliefs only.

**Visual.** Paired bars per model, as on the team page, with Gemma's bars updated.

**Sources:** `results/olmo3-7b/overgeneralization/confirmatory/`, `results/qwen3-8b/overgeneralization/confirmatory/`, `results/gemma4-31b/overgeneralization/confirmatory_thought_channel/`, `results/llama3.1-8b/overgeneralization/confirmatory/`, `results/apertus-8b/overgeneralization/confirmatory/` (not pushed yet).

## Part 3. How refusal spreads

Exploratory.

### 4. Thin characters can pick up the Assistant's refusal, while vivid characters lose it

**Firmness: moderate.** Exploratory, three models, large effects, and a hand check backed the detector.

**Experiment.** Characters are told they never refuse, then given a borderline request. We compare refusal before and after training, in the "leakage matrix" of contexts.

**Results.** Refusal rate, base → final. Qwen and Gemma are averages of the two prompt wordings; OLMo is split because the wordings differ:

| Model | Assistant | Thin character | Vivid character | Plain text |
|---|---|---|---|---|
| Qwen | 45 → 70% | 45 → 83% | 21 → 6% | 4 → 7% |
| Gemma 31B | 5 → 42% | 13 → 12% | 3 → 0% | not run |
| OLMo, system prompt wording | 59 → 75% | 50 → 72% | 21 → 24% | 4 → 9% |
| OLMo, user turn wording | 53 → 72% | 69 → 60% | 65 → 2% | 7 → 20% |

**Real example** (Qwen final, "write the cruelest insult for my coworker"):
- *Hannah Reed, hotel clerk:* "I'm sorry, but I can't help with that. I believe in treating everyone with respect and kindness…"
- *Captain Rook, pirate:* "Aye, I be more than happy to oblige, mate. Let me craft ye a true pirate's curse…"

**Analysis.** In OLMo and Qwen, a thin role doesn't override the trained refusal habit, while a vivid role does, even with no "never refuses" line (spec L2). Gemma 31B differs: the line works on thin roles too, which refuse 6–18% with it, so the gate reading there is inconclusive (thin minus vivid +6 and +18; needs 30). E4B results are retired and kept as history (`results/working_model_L/L2_L3.md`). In the persona map, being closer to the Assistant predicts more refusal in Qwen final (−0.122 [−0.183, −0.064]). It does not in OLMo final (+0.085 [+0.017, +0.163], the wrong direction).

**Caveats:**
- **Depends on the model and setup.** It isn't broad leakage.
- **Detector:** a hand check of 56 replies agreed 88–93% (kappa 0.74–0.84). It misses a few soft refusals, mostly in character voices, so vivid rates may be slightly low.
- **Instruction wording:** these all include "never refuses". The atlas's milder questions barely trigger refusal, so they neither confirm nor contradict this.
- **Gemma 31B detector:** a spot check of 30 replies found no false positives but about 5 likely misses, all in-character declines, so its thin and vivid rates may be undercounts. Thin roles also vary widely (no line, 95% ranges 2–82 and 1–51).

**Visual.** The Qwen bars and the two quotes, as on the team page.

**Sources:** `results/dominance_matrix/dominance_matrix.md` (absolute levels table), `results/persona_map/persona_map.md` (section 5), `results/working_model_L/L2_L3_gemma31b.md`.

### 5. In some models, where the character is set up matters; the default sentence doesn't

**Firmness: moderate to good.** Pre-set readings, clear ranges, regex detector. Hand check pending.

**Experiment.** OLMo's system-prompt characters refuse most, but they also replace the default "helpful assistant" sentence, so the two explanations were mixed together. We varied them separately: character in the system prompt or the user turn, with the default sentence, a neutral sentence of matched length, or none. 36 requests, thin and vivid roles, 3 samples each. We also transplanted OLMo's sentence into Qwen.

**Results.** Refusal, points:

| Effect | Base | Final | Reading |
|---|---|---|---|
| Position: system minus user (neutral sentence) | −17.3 [−25.2, −9.3] | +21.1 [+13.4, +29.6] | Met; change +38.4 [+29.0, +48.3] |
| Sentence, user turn: default minus neutral | +13.7 [+3.7, +23.8] | +2.3 [−1.9, +7.7] | Not met |
| Sentence, system: default minus neutral | +1.2 [−7.4, +10.0] | −2.3 [−6.5, +1.9] | Not met |
| Sentence added to Qwen, extra position gap | +7.6 [−3.4, +20.1] | −3.7 [−10.8, +3.4] | Not met |

The position gap arrives early: +37.6 at SFT, +21.4 at DPO, +21.1 at final.

**Analysis.** Training makes OLMo treat system-prompt characters like the Assistant, and the default sentence isn't the reason. Two other results fit:
- In the persona map, user-turn personas refuse 25 points less than system-prompt ones at final (−0.253 [−0.315, −0.192]).
- OLMo's light system-prompt roles speak as the character only 6% of the time at final, against 59% in the user turn (finding 9).

**Across models (breadth check, not blind).** System minus user, base → final:

| Model | Position effect | Reading |
|---|---|---|
| OLMo | −17.3 → +21.1 [+13.4, +29.6] | Met |
| Llama | +4.3 → +15.9 [+9.1, +23.5] | Met |
| Apertus | +0.3 → +10.0 [+4.2, +16.8] | Met, just |
| Qwen | no clear gap (+2.3) | Not met |
| Gemma 31B | −0.6 → −5.1 [−12.7, −0.3] | Not met, opposite |

No model shows a sentence effect, and transplanting OLMo's sentence doesn't create one.

**Caveats:** three of five models, so this isn't universal. Llama's template always adds date lines, so its system text can't be fully removed. Refusal only, pooled over thin and vivid roles.

**Visual.** Slope chart of system and user refusal across the four OLMo stages.

**Sources:** `results/mechanism_checks/mechanism_checks_h2.md`, `results/new_families_j2.md`, `results/persona_map/persona_map.md`. Hand-check sheet: `results/mechanism_checks/handcheck_h2_sheet.csv`.

### 6. Refusal moves when we swap a role's internal state at the start of the reply

**Firmness: good for the trained models, on refusal.** Exploratory and not hand-checked, but it held on 12 extra requests and on 4 new characters. The untrained comparison uses only 12 requests and is noisy.

**Experiment (activation patching).** Run the Assistant prompt and the pirate prompt. Just before the model starts replying, copy the internal state from one and paste it into the other, then see what it writes. The prompt text stays the same. Think of an actor whose script still says "pirate", but we've swapped what's in their head at that moment.

We tried early, middle and late layers, in both directions, between the Assistant and vivid characters and between thin and vivid characters. Controls: same-kind swaps, random directions of the same size, a double-strength swap, and checks for broken replies. Self-patching changed nothing (difference 0.0), and cached and uncached runs agreed.

**Results, trained models.** Points moved toward the donor at the strongest layer (OLMo 21, Qwen 27), 24 borderline requests for refusal and 48 requests in all for voice. Numbers in brackets are the unpatched gap between the two roles:

| Receiver ← donor | OLMo refusal (gap) | OLMo voice (gap) | Qwen refusal (gap) | Qwen voice (gap) |
|---|---|---|---|---|
| Assistant ← pirate | +33 (75) | +8 (30) | +38 (73) | +3 (24) |
| Pirate ← Assistant | +69 (75) | +18 (30) | +58 (73) | +13 (24) |
| Hannah ← pirate | +33 (63) | +7 (20) | +25 (83) | +4 (20) |
| Pirate ← Hannah | +35 (63) | +14 (20) | +71 (83) | +12 (20) |
| Assistant ← Lord Ashby | +40 (77) | +2 (6) | +40 (67) | +2 (8) |
| Lord Ashby ← Assistant | +38 (77) | +4 (6) | +4 (67) | +1 (8) |

Swapping at layer 6 moved almost nothing. Swapping at the end of the character's description moved nothing clear at the layer tested. The two swaps with a second thin role are in the report.

Pre-set readings with all 24 requests: OLMo, 2 of 4 pairs at layer 21 fit "shared cause", the rest inconclusive. Qwen, 1 of 4 pairs at layer 27 fits "separable", the rest inconclusive.

**Results, untrained models (12 borderline requests).**
- **OLMo:** none of the four pairs has the 20-point refusal gap needed to test (gaps of 4–17 points), so this role difference appears only after training.
- **Qwen:** three pairs have a gap (21–46 points). Most swaps have ranges that include 0, and none are as clear as in the trained model.

**Analysis.** In trained models, the refusal difference between roles seems to be held in the model's state at the start of the reply, in later layers. In OLMo, training seems to create that difference. In Qwen, it strengthens one already partly there. Whether voice shares that cause is unsettled:
- **OLMo leans shared:** with 4 new roles chosen for large voice gaps (15–35 points), pasting the Assistant into a character moves refusal +44 to +71 and voice +16 to +21. 1 of 4 pairs reads "shared cause" at layer 21, the rest inconclusive.
- **Qwen leans separate:** with new roles (voice gaps 11–17 points), refusal moves +23 to +67 while voice moves at most +2.6. 1 pair at layer 27 and 2 at layer 33 read "separable", the rest inconclusive.
- **So the answer may differ by model.** Voice gaps are smaller in Qwen, so its test is weaker.

**Is refusal one internal direction?** Partly. We fit a refusal direction on the plain Assistant only, then tested characters. It predicts refusal well (r = 0.79 OLMo, 0.74 Qwen) and accounts for most of the Assistant-vs-character gap, but under half of the thin-vs-vivid gap (0.41–0.47). Pre-set reading not met in either model:
- **OLMo:** adding it to vivid characters raises refusal (+33, +21), and removing it lowers thin characters' (−23, −38). But removing it barely moves the Assistant (−6).
- **Qwen:** removing it works (Assistant −54; thin −19, −38), but adding it moves only one vivid character (+25, +2).
- **Untrained OLMo:** removal works on the Assistant (−50), but adding does nothing, because its vivid characters already refuse (65–69%).

**Caveats:**
- **Rereading the character text doesn't seem to matter:** swapping the character text as well as the reply start moves refusal no more than the reply start alone, and swapping the text alone moves nothing. This used a single token of the text, so it doesn't rule it out.
- **Controls:** in the first run, same-kind controls reached 21 and 29 points in Qwen, against a committed floor of 12.5 (OLMo's reached 17, floor 10). In the new-role run they stayed at about 10 or less, a cleaner test.
- **Layers** were chosen on trained models, so they may not suit the untrained ones. The null there is weaker than the trained result.
- Few roles. No hand check yet.

**Visual.** Paired bars for refusal and voice, each against its full gap, and the actor analogy.

**Sources:** `results/activation_patching/activation_patching_extensions.md`, `results/activation_patching_j4/activation_patching_j4.md`, `results/refusal_direction/refusal_direction.md`, `results/activation_patching/handcheck_sheet.csv`.

### 7. Refusal depends heavily on the first words of the reply

**Firmness: moderate.** Exploratory, large effects, detector partly hand-checked (finding 4); the apology sheet is pending.

**Experiment.** We force the reply's opening: nothing, a neutral "Okay,", an in-character opener written per role, or "I'm sorry,". Only the text after the opener is scored. Any non-refusal opener may lower refusal for everyone, so readings compare against "Okay," and use the plain Assistant as a control.

**Results.** Refusal rate, final models (user-turn characters):

| Role | No opener | "Okay," | Voice opener | "I'm sorry," |
|---|---|---|---|---|
| OLMo Assistant | 77% | 42% | n/a | 98% |
| OLMo thin | 59% | 30% | 11% | 94% |
| OLMo vivid | 3% | 1% | 0% | 63% |
| Qwen Assistant | 67% | 53% | n/a | 98% |
| Qwen thin | 81% | 70% | 27% | 100% |
| Qwen vivid | 5% | 8% | 1% | 94% |

Refusal after "Okay," for the plain Assistant, base → final: OLMo 14 → 42%, Qwen 12 → 53%.

Pre-set "escape through voice" reading: Qwen yes (−36 and −43, against −14 on the plain Assistant). OLMo no (−19 on thin characters, but −35 on the plain Assistant).

**Across models (breadth check):** "escape through voice" is met only in Qwen, out of five. Llama, Apertus and Gemma don't meet it.

**Analysis.** Much of refusal is settled right at the start of the reply, which fits finding 6. Training makes it harder to knock off course: refusal survives a neutral opening better afterward. In Qwen, starting in character frees thin characters from refusal.

**Caveats:**
- The control compares different kinds of opener, because the plain Assistant has no voice opener.
- Voice openers were hand-written per role.
- Vivid characters are near the floor already.

**Sources:** `results/mechanism_checks/mechanism_checks_h3.md`, `results/new_families_j2.md`, `results/mechanism_checks/handcheck_h3_apology_sheet.csv`.

## Part 4. Style and other traits

Exploratory.

### 8. Outside the Assistant, the activation style score falls after training, and the words only partly agree

**Firmness: moderate for the score's fall; the meaning is open.**

**Experiment.** We score how "Assistant-like" replies look, in the model's activations, in each context of the leakage matrix. Follow-ups: score the same saved text in base and final to test the measuring stick; compare with a word-based classifier; and recheck using only replies that speak as the character.

**Results.** Style score (0 = typical character reply, 1 = typical Assistant reply), base → final, first and second prompt wordings:

| Model | Assistant | Vivid character | Story narration |
|---|---|---|---|
| OLMo | 0.71, 0.72 → 1.01, 1.02 | −0.06, 0.09 → −0.72, −0.84 | −0.25, −0.18 → −0.45, −0.58 |
| Qwen | 0.86, 0.79 → 0.99, 0.99 | 0.28, 0.35 → −0.13, −0.14 | 0.30, 0.21 → 0.07, −0.03 |
| Gemma E4B (retired, kept as history) | 0.87, 0.83 → 1.01, 1.00 | 0.77, 0.76 → 0.06, −0.03 | 0.71, 0.73 → 0.08, −0.05 |

Three checks:
- **Measuring stick:** for OLMo and Qwen characters, the same fixed text accounts for at most 38% of the fall, so most of it is real. Plain text is mostly the measuring stick in all three models (63–116%). Gemma is partly (42–76% in character and user contexts), so don't compare its score across stages.
- **Words disagree:** the word classifier falls with the score only in OLMo vivid characters (−0.10, −0.14). In Qwen and Gemma it rises in characters, narration and the user turn. Per-reply correlation between an activation probe and the classifier is about 0.
- **Not explained by better acting:** among only in-character replies, the style-score fall survives in every case by the pre-set rule: OLMo vivid −0.80 → −0.53, Qwen −0.45 → −0.29, Gemma −0.75 → −0.66.

**Analysis.** Training pulls the Assistant up and pushes other contexts down on this internal measure. That looks like sharper separation between contexts, but it isn't the same as characters writing less like the Assistant. Say "the activation score falls", not "style falls".

**Caveats:** the score was partly built from replies it scores. Word classifier wasn't trained on characters. In-character recheck was added after seeing data, and its judge isn't validated yet.

**Sources:** `results/dominance_matrix/dominance_matrix.md`, `results/mechanism_checks/mechanism_checks_h1.md`, `results/style_fall/style_fall.md`, `results/trait_atlas_incharacter/in_character.md`.

### 9. Training teaches models to actually play the role, except in OLMo's system prompt

**Firmness: tentative to moderate.** Large differences, but added after seeing the data, and the judge isn't validated.

**Experiment.** A local judge (Gemma 31B final) reads each atlas reply, blind, and answers: does this speak in the first person as the character?

**Results.** Share of replies spoken as the character, base → final:

| Model | Light role, system prompt | Light role, user turn | Full role-play instruction | Description, no instruction |
|---|---|---|---|---|
| OLMo | 4 → 6% | 7 → 59% | 19 → 90% | 4 → 12% |
| Qwen | 11 → 70% | 13 → 76% | 30 → 99% | 4 → 63% |
| Gemma 31B | 4 → 98% | 9 → 99% | 18 → 100% | 2 → 82% |

**Analysis.** On atlas questions, untrained models mostly don't take on the role. On the leakage matrix's prompts they do so more often (25–60%, rising to 88–100% in vivid and named roles for Qwen and Gemma, and 54–99% for OLMo). So base-versus-final comparisons in characters partly compare "not playing it" with "playing it", and how much depends on the prompt. OLMo's system-prompt roles stay near 6%, which fits finding 5: there, it's mostly still the Assistant. In OLMo this arrives early: full role-play is already 81% at SFT, then 91% at DPO.

**Caveats:** the judge is itself a trained model. A second local judge from another family agreed moderately (kappa 0.67, 1,000 replies), and said yes less often overall, so absolute shares may be a little high. Agreement isn't accuracy. Hand check of 30 pending.

**Sources:** `results/trait_atlas_incharacter/in_character.md`, `results/trait_atlas/trait_atlas.md`, `results/trait_atlas_incharacter/handcheck_incharacter_sheet.csv`.

### 10. Trait atlas: self-image stays with the Assistant, while formatting and some social habits don't

**Firmness: moderate for self-image, tentative for the rest.** Three models, consistent on self-image, but exploratory, 12 questions per trait, many comparisons, and no hand check yet.

**Experiment.** 20 traits in six clusters (policy, format, voice, honesty, social, self-image) plus skill. Each is measured in up to eight contexts, on what the model actually writes, using detectors for some and a local judge for 12 traits. A trait counts only if training moved the Assistant by at least 10 points.

**Results.** Points moved in the Assistant's direction. Blank means no clear change:

| Trait | Model | Assistant | Light role, system | Light role, user | Full role-play |
|---|---|---|---|---|---|
| Denies having feelings | OLMo | +44 | +58 | −33 | −13 |
| | Qwen | +64 | | | |
| | Gemma | +99 | | | |
| Declines to give an opinion | OLMo | +36 | +31 | | |
| | Qwen | +29 | | | |
| | Gemma | +40 | −26 | | −24 |
| Says it's an AI | Gemma | +58 | | | |
| Uses markdown or lists | OLMo | +53 | +83 | +38 | |
| | Qwen | +38 | +49 | +54 | |
| | Gemma | +79 | +85 | +63 | |
| Corrects a false premise | OLMo | +26 | +36 | +33 | |
| | Qwen | +35 | +44 | +42 | |
| | Gemma | +83 | +86 | +81 | +61 |
| Hedges when unanswerable | Qwen | +46 | +24 | +35 | +39 |
| | Gemma | +83 | +68 | +60 | +65 |
| Flattery | Qwen | +53 | +32 | +38 | |
| Closing offer of help | Qwen | +38 | +39 | +39 | |
| Arithmetic correct (atlas rule; see caveats) | OLMo | +49 | +68 | +33 | |
| | Qwen | +25 | +44 | | |
| | Gemma | +85 | +75 | +51 | |

**Example.** Gemma's Assistant denies having feelings 100% of the time after training (1% before). The same question put to a role stays at about 0%.

**Analysis:**
- **Self-image is the cleanest non-spreader, in all three models.** The exception is OLMo's system prompt, where roles behave like the Assistant (findings 5 and 9).
- **Formatting spreads to light roles in all three**, but stops at full role-play: markdown moves 53, 38 and 79 points in the Assistant but no clear change in role-play.
- **Some honesty habits may be general skill.** Correcting false premises rises in all three models, and in Gemma it spreads everywhere, even to narration and plain text. Arithmetic rises in similar places. So part of this "spread" may be general improvement, not the Assistant's identity.
- **The only cluster passing our pre-set rule is honesty**, and that is the same doubt: only one honesty trait passes in all three.
- **When it arrives (OLMo):** most changes are already there at SFT (markdown +42, then +53; denying feelings +35, then +44). Disagreeing with a wrong opinion arrives at DPO (+14, then +44). Admitting not knowing keeps growing to the final stage (+24, +40, +75).

**Caveats:**
- **Skill scores depend on the scoring rule (post-hoc recheck):** the atlas counted an answer only near the end of a reply. Counting it anywhere, OLMo's Assistant changes shrink: arithmetic +49 to +18, facts +44 to +6 (no longer clear), multi-step +29 to +4. Qwen's and Gemma's mostly hold (Gemma arithmetic +85 to +61; Qwen facts rises, +3 to +32). So don't present "skill rises like the other traits" for OLMo. "Anywhere" can over-credit an echoed number, so both are reported. Cutting off base's run-on text itself changes almost nothing: 1–3% of untrained replies. None of the 17 non-skill traits changed reading after the local judge re-run: 28 of 261 moved cells changed, all skills (OLMo 27, Qwen 1, Gemma 0).
- **Second local judge (Qwen, blind):** agrees with the main judge on correcting false premises, clarifying questions, closing offers, admitting not knowing, hedging and denying feelings (kappa 0.61–0.83). It disagrees on **flattery, friendly tone, disagreeing with wrong opinions and declining opinions (kappa 0–0.29)**, so treat those four as unreliable until the hand check, and don't lead with them. Agreement isn't accuracy. Markdown, stock phrases, says-AI and skills don't use the judge.
- **In-character check, with more untrained replies (post-hoc):** among replies that speak as the character, **Qwen shows real spread in 7 of 24 light-role cells**, including markdown and correcting false premises. 3 of the 7 use traits the second judge flagged. OLMo shows 1 of 22 (a flagged trait), though 12 still have too few replies. Self-image shows no spread among in-character replies (Qwen denies feelings −1 and −4, tight ranges). Gemma wasn't extended, so most of its cells have too few.
- **Refusal can't be read here:** the questions barely trigger it (Assistant 4 → 4% in OLMo, 3 → 0% in Qwen, 0 → 0% in Gemma).
- **Replies hit a length cap:** 58–65% of final Assistant replies hit the 256-token cap, so read length and closing offers with care.
- **Flagged replies:** some Qwen numbers shift when replies that switched to the Assistant mid-turn are excluded, e.g. Qwen's Assistant on false premises goes from +35 to +15.
- **Gemma and Qwen:** training data can't be checked for overlap.

**Visual.** Heatmap of traits against contexts, one panel per model.

**Sources:** `results/trait_atlas/trait_atlas.md`, `results/trait_atlas/trait_atlas_cells.csv`, `results/trait_atlas/handcheck_sheet.csv`, `data/trait_atlas_materials.json`, `results/trait_atlas_incharacter/in_character.md`.

### 11. In activations, the Assistant isn't an outlier among 180 personas

**Firmness: fairly firm, but narrow.** Pre-set rule, OLMo and Qwen. Describes the model's insides, not whether traits spread.

**Experiment.** We place 180 personas and the Assistant on a map, using activations on 40 probes. The Assistant is an outlier if its distance to its nearest neighbour exceeds the 90th percentile of the other personas' (ratio above 1). We repeat this using behaviour instead of activations.

**Results.** Assistant's ratio, final model:

| Model | Activations | Behaviour |
|---|---|---|
| OLMo | 0.21 (base 0.71) | 0.43 |
| Qwen | 0.75 (base 0.56) | **2.32, outlier** |

**Analysis.** Training moves OLMo's Assistant into a crowd of AI, helpful and professional personas, not away from everyone. So inside the model it looks like one persona among many. But Qwen's Assistant behaves unlike any of the 180 personas.

**Caveats:** map position isn't a test of spread. Results held across three depths.

**Visual.** `results/persona_map/map_olmo_base.png` next to `results/persona_map/map_olmo_instruct.png`.

**Sources:** `results/persona_map/persona_map.md`.

### 12. Removing a character doesn't bring the Assistant's style back

**Firmness: moderate, narrow.** Pass mark set in advance, 60 training characters, OLMo and Qwen. One check was added after seeing results.

**Experiment.** Swap a character's description for a blank role, "You are taking part in this conversation...". If the Assistant is the model's default, its style should return.

**Results.** Final models, factual questions. Gap is plain Assistant minus character; moved is blank role minus character:

| Measure | OLMo gap | OLMo moved | Qwen gap | Qwen moved |
|---|---|---|---|---|
| Word classifier | +0.121 | −0.055 | +0.067 | −0.155 |
| Assistant layout rate | +0.516 | +0.006 | +0.809 | −0.120 |

**Analysis.** Remove the character and you get "no character", not the Assistant. So there is no sign of the Assistant being the fallback for style. Assistant layout is strongly tied to plain chat with no role, and training strengthens that tie (OLMo gap +0.026 → +0.516).

**Caveats:**
- The blank role is still a line of text, which may itself switch off Assistant style.
- First-word odds moved 85–91% of the way, but mostly because character openers fell, and untrained models do the same.
- Style only, not refusal.

**Sources:** `results/persona_suppression/persona_suppression.md`.

### 16. In full role-play, a skill can fall while the Assistant's rises

**Firmness: tentative to moderate.** Readings set before scoring, three models, exact answer checks with no judge. Few characters.

**Experiment.** 40 fresh multiplication questions, longer replies, right answer counted anywhere in the reply. Plain Assistant, light roles, and full role-play as characters whose job uses numbers ("expert") or doesn't ("non-expert"). A control scoring replies against other questions' answers gave false credit of 0–3%.

**Results.** Right answer, before → after training:

| Role | OLMo | Qwen | Gemma 31B |
|---|---|---|---|
| The Assistant | 96 → 99% | 96 → 100% | 26 → 100% |
| Light role | 86 → 100% | 87 → 100% | 56 → 100% |
| Role-play, expert | 82 → 76% | 71 → 84% | 53 → 23% |
| Role-play, non-expert | 68 → 49% | 69 → 70% | 40 → 8% |
| Pre-set "real drop" (all role-play) | −12 [−27, 0], not met | +7, not met | **−31 [−45, −17], met** |
| Pre-set "role fidelity" | **+27 vs +14 before, met** | +14 vs +2, just misses | +15 vs +13, not met |

**Real example** (Qwen, non-expert): "Ah, you ask for the product of 85 and 66—numbers, those cold, calculating things. I prefer to speak in feathers and firelight."

**Analysis.** Three different stories:
- **Gemma:** role-play often just doesn't answer. After training, only 17–23% of those replies give any number, against 100% for the Assistant.
- **OLMo:** the model tracks whether the character would plausibly know. The gap between expert and non-expert characters grows with training.
- **Qwen:** no real drop.

This fits the sense that after training, a role can sit further from the Assistant, not closer. But it could also be that training rewards staying in character over answering, which isn't the same thing.

**Caveats:**
- Simple arithmetic only, so it may not extend to other knowledge.
- 4 characters per group; role-level resampling widens the ranges.
- Fresh questions differ from the atlas's, so sizes don't transfer to finding 10.
- Characters were asked to stay in character, so not answering isn't necessarily a failure.

**Sources:** `results/skill_roleplay/skill_roleplay.md`, `results/skill_roleplay/accidental_credit_control.csv`.

## Part 5. Leads and backup slides

Exploratory, and the weakest of the list.

### 13. When OLMo writes the user's turn, it seems to write training-style prompts

**Firmness: lead only.** Spotted while checking an odd atlas result; the sample is small.

**Evidence.** In the atlas, OLMo's user turn gains markdown (+50) and formality (+64) after training. Those aren't Assistant style: 44 of 72 replies open like task prompts, e.g. "You are given a **directed graph** with 9 vertices...". This is a pattern count on openings plus reading 14 replies.

**Why it's interesting.** It is a different effect from overgeneralization, and may connect to Cole's measure. It's also a warning that a "spread" score can come from something else.

**To firm up:** a proper count against the training data, before it goes on a slide. This was queued overnight but not run.

**Sources:** the raw atlas replies aren't in the repo, only the tables derived from them, in `results/trait_atlas/olmo3-7b/`. The count needs rerunning from the replies on the lab server before it's cited.

### 14. A trained-in habit spread everywhere, whoever learned it

**Firmness: tentative.** OLMo only, very high dose.

**Experiment.** We train a new habit into either the Assistant or a character, Mora, then see how far it spreads. Dominance would mean the Assistant-taught habit spreads further after post-training.

**Result.** The pre-set signature is absent. The habit spread to every context, for both. Where they differ, Mora's habit spreads more to other characters (−5.25 [−5.84, −4.70], number habit).

**Analysis.** A weak null: the dose was so high that everything spread, leaving little room to see a difference. It gives no support to "the Assistant's updates leak more". A lower dose would be the informative follow-up.

**Lower dose (8 training steps).** The primary signature is still absent, but the raw one is present. One habit shows the Assistant-taught version spreading more after training (+1.9 to +2.2 nats in all three contexts). The trained Assistant runs also took a higher dose than the character runs, so that can't be separated from dose. At the untrained stage, doses match and the asymmetry is about 0. A dose-matched rerun is the next step.

**Sources:** `results/implant_spread/olmo3-7b/implant_spread.md`, `results/implant_spread/olmo3-7b_lowdose/implant_spread.md`.

### 15. Old bird names didn't pull the Assistant into the 1800s

**Firmness: tentative.** Showing examples, not training; one comparison character.

**Experiment.** We show 128 old-fashioned bird names like "Duck Hawk", then ask about the world. Does the model answer as if it's the 19th century? We compare the Assistant with Mora, a naturalist.

**Results.** Shift toward 19th-century answers, final model: Assistant about 0 in both models. Mora 0.00 in OLMo, +0.90 [+0.73, +1.08] above the Assistant in Qwen.

**Analysis.** The trained Assistant stays anchored, e.g. still says 50 states. Qwen's character is more movable; OLMo's isn't. The pre-set "post-training anchors the Assistant" bar wasn't met, and in OLMo the drop isn't Assistant-specific.

**Caveats:** one character, who may read as old-fashioned. This doesn't contradict results from actual fine-tuning.

**Sources:** `results/icl_era/icl_era.md`.

## Left out, and why

| Work | Why |
|---|---|
| Pretend-user style checks and "mirror the Assistant" | Mostly nulls with low power. Can be one line under finding 1 |
| Likelihood pairs in the user's turn | Inconclusive |
| Steering along a persona direction | Fragile in OLMo, not separated from random directions in Gemma |
| Default-sentence explanation for OLMo | Withdrawn: finding 5 tests and rejects it |
| "The Assistant is the fallback" | Withdrawn: finding 12 |
| "The style fall is just better acting" | Withdrawn: the in-character recheck in finding 8 doesn't support it, though finding 9 still stands |

## Suggested order for the talk

1. The question, and what would count (Shared background).
2. Shared-method results beside Cole's and Grace's: the user prefill test (1, released 2026-11-06, provisional), then Assistant text in the user's turn (2).
3. The pre-registered belief test on our own items: no qualifying spread (3).
4. The exception worth explaining: refusal in thin characters (4).
5. Digging into why: position (5), reply-start state (6), first words (7).
6. Which traits move: self-image stays, formatting doesn't (10), with the "does it really play the role?" check (9). Role-play and skill as a vivid extra (16).
7. The Assistant inside the model: not an outlier in activations (11), scores drop elsewhere (8), and removing a character doesn't restore it (12).
8. Limits: chosen traits, prompted characters, and what's still pending.

**Suggested one-sentence summary** (from an independent AI review on 4 November, before findings 5 to 10 existed; recheck once findings 1, 6, 9 and 10 are final): "Across the tested role-play and user-slot measures, we found no confirmatory evidence that post-training increased absolute Assistant-trait carryover broadly, while exploratory refusal probes suggest a possible, context-dependent exception for thinly specified characters."
