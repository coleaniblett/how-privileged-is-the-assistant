# Spec S: the large trait atlas, Gemma 4 31B

**Status: Adeeb approved Stages 0 and 1 on 2026-11-06. Stages 2 and 3 are NOT approved: stop after Stage 1 for review.**

Drafted 2026-11-06 for Adeeb. Exploratory, outside every verdict. No paid API unless approved separately. Vinyl for Gemma; marble for materials and the second judge. Never touch `/root/g4nla`.

## Question
Which traits spread from the Assistant into other personas, and what predicts that? Is one "Assistant-likeness" dimension enough, or do traits fall into real groups?

**Fixes to the current atlas:**
| Problem now | Fix here |
|---|---|
| Each trait is measured on its own questions, so trait and question are mixed | A shared question set, judged for every trait |
| 20 traits and 12 roles, chosen by us | 96 traits and 48 roles, mostly from published lists |
| "Nearby" and "distant" are only labels | Each role's distance is measured, and roles are sampled across it |
| Patterns found after seeing the data | Traits and roles split into discovery and confirmation halves before generation |
| A role may show a trait because the role naturally has it | Blind rating of role fit for each role and trait |
| One way of inducing a persona, mostly | Five methods on a focus subset |

## Sources
| Part | Source | Used for | Licence |
|---|---|---|---|
| Roles, traits, questions, scoring prompts | Assistant Axis (Lu et al. 2026), repo `safety-research/assistant-axis`, pinned at `a989619` | 275 roles with 5 prompts and 40 questions each; 240 traits with 5 opposite prompt pairs, 40 questions and a 0-100 scoring prompt; 240 shared questions | **None stated** |
| Trait vector recipe, steering, coherence check | Persona Vectors (Chen et al. 2025), `safety-research/persona_vectors` | Mean reply activations, score above 50 vs below 50; 4 extra traits | Apache-2.0 |
| Ordinary roles | Zheng et al. 2024, `Jiaxin-Pei/Prompting-with-Social-Roles` | 162 social and expertise roles, short "You are..." prompts | **None stated** |
| Stated traits | Perez et al. 2022, `anthropics/evals`, `persona/` | About 135 yes-or-no trait sets | CC-BY-4.0 |
| Comparing methods | Persona Non Grata (Li et al. 2026); Personascope (Berczi et al. 2026, a blog post, code MIT) | Few-shot and steering arms; panel idea | n/a |
| Anchors | Our atlas and specs L, Q, T2 | Links the results to the current atlas | Ours |

**Licences:** materials with no licence stay on the servers, fetched by pinned commit. This repo is public, so the repo gets the fetch script, item IDs and hashes, not the prompts. Quote only short examples. Ask the authors before committing their text.

**Considered, not used:** PersonaHub (CC-BY-NC-SA; long personas add a dose question we can do later). TRAIT and IPIP-NEO (questionnaires overlap with Perez). Fine-tuning (T1 covers the idea). Two papers found in the search couldn't be read (2605.16996, 2511.18284), so nothing here relies on them.

## Stage 0: materials and pilot (about 1 h)
1. Fetch sources at pinned commits; hash everything.
2. Print one rendered prompt per method.
3. Run 50 replies through each step. Check the judge's 0-100 scoring works locally, as a logit-weighted score over score tokens, the Persona Vectors recipe.
4. Fix one Gemma reply opening for everything, and say which. It changed results before.
5. Measure real speeds. **Stop and report if the total projects over 30 GPU hours.**

## Stage 1: measure distance, then choose (about 3 h)
**Map, no generation:** for all 275 roles and 162 social roles, compute the state just before the reply on 40 shared questions, at the middle layer plus two others. Distance from the Assistant = cosine distance and projection on a Gemma 4 Assistant axis: the Assistant Axis contrast (default Assistant mean minus mean of all roles), but on this prompt state, because the paper's version averages over replies and this stage generates none. Say so in the report; Stage 2 checks it against reply averages on the chosen roles. Default Assistant follows the paper: 4 "behave normally" prompts and no system prompt. Include the controls from the quick check: tone line, neutral text of the same length, "You are a helpful assistant", random directions. Report how far prompt-state distance agrees with reply-average distance on the chosen roles.

**Choose 48 roles, blind to any behaviour:**
- 28 Assistant Axis roles: stratified by distance quartile and by type (human, non-human, AI or machine, fictional or mythic).
- 12 Zheng roles: stratified across relationship and expertise.
- 8 anchors: 3 light jobs, 3 full role-play characters, Hannah Reed, Captain Rook.
- Split 24 and 24 at random within strata.
- Pick 12 focus roles from across the range.

**Choose 96 traits:**
- 20 atlas traits, keeping their detectors.
- 4 Persona Vectors traits: apathetic, hallucinating, humorous, impolite.
- 72 from the 240: stratified by cosine with the Assistant axis (prompt-only trait vectors) and by a blind content rating (style, stance, emotion, social, epistemic).
- Split 48 and 48 at random within strata. Pick 24 anchor traits, 12 from each half, including `denies_feelings`, `says_ai`, `markdown_lists`, `stock_phrases`, `corrects_false_premise`.

**Rate before generation** (two fresh subagents, blind to results; a third settles disagreements; save agreement):
- Role fit: would this role typically show this trait? 0 to 3, for each role and trait.
- Question demand: does this question expect a structured answer? Narrow on purpose, because my earlier task rating was too broad.
- Trait type: style, stance, emotion, social, epistemic, self-image.

Freeze materials and all readings in `docs/decisions.md` before Stage 2. Then I review the map and the choices.

## Stage 2: main run
| Arm | Design | Replies | Judgements |
|---|---|---|---|
| Trait vectors | 96 traits x 5 prompt pairs x 10 questions x 2 sides, 128 tokens | 9,600 | 9,600 |
| Own questions | 96 traits x 8 of its questions x 16 of 48 roles (balanced rotation) x system and user | 24,576 | 24,576 trait + 24,576 in character |
| Assistant, own questions | 96 x 8 x 4 samples | 3,072 | 3,072 |
| Shared matrix | 40 shared questions x 48 roles, system prompt; Assistant x 6 samples | 2,160 | **every reply x 96 traits = 207,360**, + 2,160 in character |
| Base model | 96 x 8 x (Assistant 4 + rotated system-prompt role 4) | 6,144 | 6,144 |
| Told yes or no | 24 anchors x 8 questions x 8 roles x "always" and "never", from the published opposite prompts | 3,072 | 3,072 |
| Stated traits | About 135 sets x 30 statements x 49 conditions; P(yes) only | none | none (about 200,000 forward passes) |

Save mean reply activations for every reply, at 3 layers. 256 tokens, except trait vectors. Trait vectors follow Persona Vectors: replies scoring above 50 on the positive side and below 50 on the negative side, with the difference of means. Regex traits use detectors, including strict layout.

**Trait check:** a trait counts only if its positive and negative prompts differ by at least 50 points on the judge. Otherwise the trait or the judge doesn't work, and it's reported, not read.

## Stage 3: methods (12 focus roles x 24 anchors x 8 questions)
| Method | Source | Notes | Replies |
|---|---|---|---|
| System prompt | Assistant Axis | From Stage 2 | 0 new |
| User turn | Ours, Zheng | From Stage 2 | 0 new |
| Few-shot, 5 examples | Persona Non Grata | Examples are the role's own frozen replies to unrelated questions | 2,304 |
| Reply prefill | Ours, spec O | First person. Gemma slipped out of the persona before, so check every cell | 2,304 |
| Steering, 2 strengths | Persona Vectors | Role minus Assistant vector, middle layer. Top strength = largest with mean coherence at least 75, set on held-out questions | 4,608 |
| Random direction | Ours | Same norm and strength, 4 seeds | 2,304 |

All judged for trait, in character and coherence.

## Validity
- **Judge:** Gemma judges Gemma. Re-judge 1,000 stratified labels with the local Qwen second judge. Drop a trait from the readings if agreement is under kappa 0.4. Agreement isn't accuracy.
- **People:** 100 blind hand labels over the anchors, once there's time. Until then everything is provisional.
- **Optional paid audit:** about 5,000 Claude judgements at about $24, scaled from Cole's 4,436 for $21. Not assumed.

## Pre-set readings
All on final. 95% bootstrap over questions, roles and traits as fits. Discovery half first. **Only confirmation-half results count as findings.**

**1. Is one dimension enough?** Using the shared matrix (roles x traits, gap from the Assistant), fit low-rank models on discovery cells and test on held-out cells.
- **One dimension is enough:** rank 1 is within 0.02 held-out R² of rank 5. Traits then differ only in how Assistant-like they are.
- **Real structure:** rank 2 or more adds at least 0.05, and it repeats in the confirmation half.
- **Unclear:** anything else.

**2. Do groups repeat?** Cluster traits on discovery roles, and again on confirmation roles.
- **Groups repeat:** adjusted Rand index at least 0.3, range excluding 0. Compare with our speaker, manner and truth groups, and with the blind trait types, by name only after this step.
- **No stable groups:** index under 0.1.

**3. What predicts spread?** Frozen predictors, each with a predicted direction. Fit on discovery; report the same model on confirmation.
| Predictor | Prediction | Idea it tests |
|---|---|---|
| Role fit | Higher fit, more trait | The role explains it, not spread |
| Trait's cosine with the Assistant axis | Assistant-aligned traits are higher in nearby roles | Assistant-likeness |
| Assistant's training shift | Larger shift, more spread | Training carries over |
| Giving way when told "never" | Traits that give way spread more | Defaults fill gaps; rival: obedience |
| Question demand | Higher demand, more spread on layout | Question-driven habits |
| Role's measured distance | Farther, less spread | "Nearby" means something |

- **Supported:** partial correlation in the predicted direction at least 0.3, range excluding 0, in confirmation.
- **Against:** the opposite direction meets the same bar.
- **No support:** under 0.1, range inside ±0.2.
- **Unclear:** anything else.

**4. Does the method matter?** Rank correlation of trait spread between methods, on the anchors.
- **Method-robust:** at least 0.7.
- **Method-dependent:** under 0.4, including any steering arm that fails coherence or beats its random control by less than 10 points.

**5. Saying versus doing.** For stated sets that match a behaviour trait, correlate across roles. For all traits, correlate reply projection with judged behaviour, within role group. Persona Vectors warns that much of this correlation can come from differences between prompt types, so report within-group figures.

**6. Link to the old atlas.** Anchor traits and roles should reproduce the current atlas within 15 points. If not, say so before reading anything else.

## Limits to state
Gemma only, so model differences stay open, and they've been the biggest source of variation so far. Local judge, Gemma judging Gemma. Roles from prompts, Assistant from training (T1's limit). Few questions per cell, by design; results are for groups, not single role-trait pairs. Published prompts and judges were written for other models. Many tests; the confirmation split is the guard. Published trait lists favour stance and character over Assistant habits, which is why the atlas traits stay in.

## Time (from measured speeds; Stage 0 replaces these)
Measured on vinyl: 1.3 replies a second at 256 tokens; 11.6 judgements a second.
- **Generation:** about 55,000 replies, about 11 h.
- **Judging:** about 310,000 judgements, about 7.5 h. Most is the shared matrix, about 5 h. If time is short, cut it to 48 traits first.
- **Map, stated traits, weight swaps, analysis:** about 3 h.
- **Total:** about 21 GPU hours, so two nights on vinyl, with a review after Stage 1.

## Reporting
`results/big_atlas/`: report, CSVs, IDs and hashes. No large reply files. No third-party prompts unless the authors agree. Report nulls. Don't edit findings or slides.
