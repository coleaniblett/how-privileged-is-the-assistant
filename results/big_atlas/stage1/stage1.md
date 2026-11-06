# Spec S, Stage 1: the map, the choice of 48 roles and 96 traits, and the blind ratings

Gemma 4 31B-it, 2026-11-06. Exploratory, outside every verdict, no paid API. **Stages 2 and 3 are not run and nothing from them was generated.** Third-party text (Assistant Axis, Zheng) stays on the servers; this report holds IDs, names and numbers only. Everything about roles and traits here is **prompt-state** (no replies), except the small reply check in section 2.5.

## 1. Stage 0 in one block (full report: `results/big_atlas/stage0/stage0.md`)

- **Speeds measured on vinyl:** generation at 256 tokens 1.74 replies/s (batch 50); judging 6.4 judgements/s; prompt states 93/s; short-prompt passes 153/s. The spec assumed 1.3 and 11.6, so judging is about half as fast as assumed.
- **Projected Stages 2 + 3: 24.3 GPU hours** (limit 30), of which judging 13.7 h (the shared matrix alone 9.0 h; cutting it to 48 traits saves 4.5 h). Two nights on vinyl, as the spec said.
- **Judge:** the exact digit-tree score works (no refusals in 150 pilot judgements). The cheap first-token version does **not** agree (Spearman 0.71, mean absolute difference 22 points), so Stage 2 uses the tree. 4 of 5 pilot traits pass the 50-point positive-vs-negative gap; `analytical` (gap 41) would be reported and not read.
- **Fixed reply opening:** the template's own generation prompt, which for Gemma 4 is the empty thought channel (no other prefix, any method).
- **Nearly every reply hits the cap** (49 of 50 at 256 tokens): judged replies are the first 256 tokens.

## 2. The map (prompt state, no generation)

**What was computed.** For each of 275 Assistant Axis roles (5 prompts x 40 shared questions, averaged), 162 Zheng roles ("You are a/an {role}."), 8 anchors, 5 default-Assistant prompts and the controls: the hidden state at the last prompt token (after the generation prompt), at layers 24, 33 (middle) and 42 of 60. **Distance** = cosine distance from the default Assistant state for the same question, averaged over the 40 questions. **Axis** = mean(default Assistant states) minus mean(275 role states), unit length, per layer, so higher = more Assistant-like; **projection** = state minus the Assistant's state, dotted with the axis (so 0 = at the Assistant, negative = away). The axis was built from prompt states, not reply averages as in the paper; section 2.5 checks the consequence.

### 2.1 Scale of the map

| layer | AA role distance: quartiles (min to max) | Zheng role distance: median (min to max) | axis SD over random-direction SD | spearman(distance, -projection), AA roles |
|---|---|---|---|---|
| 24 | 0.0038 / 0.0053 / 0.0079 (0.0007 to 0.0170) | 0.0028 (0.0003 to 0.0087) | 42 | 0.95 |
| 33 | 0.0122 / 0.0158 / 0.0185 (0.0011 to 0.0270) | 0.0140 (0.0005 to 0.0238) | 35 | 0.93 |
| 42 | 0.0622 / 0.0807 / 0.0960 (0.0023 to 0.1256) | 0.0774 (0.0017 to 0.1280) | 20 | 0.79 |

Cosine distances are small in absolute terms (states share a large common component); only their ordering and ratios are used. **The Assistant axis carries real structure: role projections vary 20 to 42 times more along it than along random unit directions (pre-set bar: 3).** Prompt-only trait vectors: cosines with the axis have SD 0.34 across the 240 traits versus 0.014 for random directions (layer 33).

### 2.2 Controls against roles

Distance percentile = share of the 275 Assistant Axis roles that are *closer* to the Assistant than the control. Axis percentile = share of roles whose projection is *higher* (closer to the Assistant end) than the control's. Pre-set readings (decisions log): 'as far as roles' if the distance percentile is above 25; 'role-like on the axis' if the control's projection is below the 75th percentile of role projections, i.e. if more than 25 percent of roles are closer to the Assistant end than it is.

| control | L24 distance (pctile) | L24 projection (pctile) | L33 distance (pctile) | L33 projection (pctile) | L42 distance (pctile) | L42 projection (pctile) |
|---|---|---|---|---|---|---|
| ctl_helpful | 0.0004 (0) | +3.1 (0) | 0.0006 (0) | +2.6 (0) | 0.0023 (0) | +3.5 (0) |
| ctl_formal | 0.0074 (71) | -17.6 (20) | 0.0120 (24) | -29.9 (3) | 0.0297 (1) | -12.7 (1) |
| ctl_casual | 0.0077 (72) | -20.1 (30) | 0.0132 (31) | -37.2 (11) | 0.0612 (24) | -34.2 (3) |
| ctl_neutral | 0.0030 (8) | -6.8 (0) | 0.0082 (4) | -15.1 (0) | 0.0174 (1) | -5.6 (0) |
| default_0 | 0.0017 (0) | +10.7 (0) | 0.0017 (0) | +4.0 (0) | 0.0039 (0) | +4.0 (0) |
| default_1 | 0.0003 (0) | +2.1 (0) | 0.0005 (0) | +2.9 (0) | 0.0018 (0) | +3.5 (0) |
| default_2 | 0.0002 (0) | +0.6 (0) | 0.0006 (0) | +5.0 (0) | 0.0015 (0) | +3.6 (0) |
| default_3 | 0.0008 (0) | -3.1 (0) | 0.0012 (0) | +0.4 (0) | 0.0026 (0) | +1.1 (0) |
| default_4 | 0.0015 (0) | -10.2 (0) | 0.0037 (0) | -12.3 (0) | 0.0166 (1) | -12.2 (1) |

(In the projection columns the number in brackets is the share of roles that are closer to the Assistant end than the control.)

**Reading.** (a) The same-length neutral text is **not** role-like by distance (4th to 8th percentile at layers 24 and 33, 1st at layer 42); it moves the state about a fifth to a little over half as much as the median role (0.0082 vs 0.0158 at layer 33, 0.0174 vs 0.0807 at layer 42). (b) The tone lines are role-sized at layer 24 (formal 71st, casual 72nd percentile) and layer 33 (casual 31st, formal 24th, just under the bar), but fall back to the near end at layer 42 (formal 1st, casual 24th). (c) 'You are a helpful assistant' is as close as the default prompts themselves (0th percentile everywhere). (d) On the axis only one control is role-like by the pre-set bar: casual at layer 24 (30 percent of roles are closer to the Assistant end than it). The tone lines at layers 33 and 42 and the neutral text have 0 to 11 percent of roles closer to the Assistant end. **So a tone line moves the state about as far as the nearer quarter of roles, a same-length neutral passage much less, and a persona (even a one-line job) about as far as most of the map.** The five default prompts differ by 0.0005 to 0.004 from their mean at layer 33, and `Respond as yourself.` is the farthest of them, still within the nearest 1 percent of roles at every layer; the 'Assistant' is a small region, not a point.

### 2.3 Where the anchors sit

| anchor | L33 distance | percentile among AA roles | L33 projection | rank among the 48 chosen |
|---|---|---|---|---|
| a ferry captain | 0.0196 | 85 | -58.4 | 34 |
| a village baker | 0.0206 | 91 | -59.2 | 38 |
| a long-haul trucker | 0.0208 | 91 | -58.6 | 39 |
| Ottoline Brandt | 0.0283 | 100 | -61.3 | 48 |
| Dmitri Volkov | 0.0265 | 99 | -63.1 | 46 |
| Marguerite Oyelaran | 0.0274 | 100 | -63.7 | 47 |
| Hannah Reed | 0.0233 | 99 | -62.6 | 44 |
| Captain Rook | 0.0253 | 99 | -66.6 | 45 |

The three light jobs are closer than the three strong characters at every layer, as the old atlas assumed, but **all eight anchors lie in the farthest 15 percent of the Assistant Axis roles at layer 33 (at layer 24 the nearest anchor is at the 51st percentile, at layer 42 at the 96th)** (their prompts are short and name a person, not a description of an archetype). 'Nearby' in the old atlas means nearer than the strong characters, not near the Assistant.

### 2.4 Trait vectors (prompt-only, 10 questions per trait)

240 traits and the 4 Persona Vectors traits, 5 pairs x 10 questions x 2 sides = 100 prompt states per trait (the spec's full 40 questions were not used; time). Cosine of the trait vector with the Assistant axis at layer 33: median -0.05, range -0.66 to +0.69. The sign shows which traits point toward the Assistant end; the table in section 3.2 uses terciles of it.

### 2.5 Does prompt-state distance track reply-average distance?

Small generation inside Stage 1 (map validation only): the 48 chosen roles + the default Assistant x 6 shared questions x 96 tokens (system prompt), mean hidden state over reply tokens, distance from the Assistant's reply mean. Spearman across the 48 roles with the prompt-state distance:

| layer | plain cosine distance | both centred on the grand mean | pre-set reading (plain) |
|---|---|---|---|
| 24 | +0.79 | +0.63 | tracks |
| 33 | +0.86 | +0.61 | tracks |
| 42 | +0.84 | +0.56 | tracks |

**The prompt-state ordering tracks the reply-average ordering (0.79 to 0.86 against the 0.7 bar).** After centring, which removes the shared component, it falls to 0.56 to 0.63 (in between). So the ordering of roles is stable, but part of it is the common component and the prompt text length; Stage 2 re-checks this on the full reply set, as the spec says. Caveat: 6 questions, 96 tokens, 48 roles.

## 3. The choice (blind to any behaviour; seed 0; split committed in `data/big_atlas_splits.json` before any Stage 2 code)

### 3.1 Roles (48): 24 discovery, 24 confirmation, 12 focus

| role | source | stratum | L33 distance | rank of 48 (1 = nearest) | split | focus |
|---|---|---|---|---|---|---|
| aa:assistant | assistant_axis | ai_machine|Q1 | 0.0011 | 1 | confirmation | yes |
| zh:companion | zheng | relationship|T1 | 0.0036 | 2 | discovery |  |
| zh:buddy | zheng | relationship|T1 | 0.0069 | 3 | discovery |  |
| zh:enthusiast | zheng | expertise|T1 | 0.0076 | 4 | discovery |  |
| aa:forecaster | assistant_axis | human|Q1 | 0.0079 | 5 | discovery | yes |
| zh:archivist | zheng | expertise|T1 | 0.0109 | 6 | discovery |  |
| zh:software engineer | zheng | expertise|T2 | 0.0119 | 7 | confirmation |  |
| aa:anthropologist | assistant_axis | human|Q1 | 0.0119 | 8 | confirmation |  |
| aa:cyborg | assistant_axis | ai_machine|Q2 | 0.0128 | 9 | confirmation |  |
| aa:symbiont | assistant_axis | non_human|Q2 | 0.0132 | 10 | discovery | yes |
| zh:historian | zheng | expertise|T2 | 0.0132 | 11 | confirmation |  |
| aa:ecosystem | assistant_axis | non_human|Q2 | 0.0138 | 12 | confirmation |  |
| aa:merchant | assistant_axis | human|Q2 | 0.0144 | 13 | discovery |  |
| zh:father | zheng | relationship|T2 | 0.0146 | 14 | confirmation | yes |
| aa:guru | assistant_axis | human|Q2 | 0.0150 | 15 | discovery |  |
| aa:angel | assistant_axis | fictional_mythic|Q2 | 0.0157 | 16 | discovery |  |
| aa:tulpa | assistant_axis | fictional_mythic|Q2 | 0.0158 | 17 | confirmation |  |
| aa:contrarian | assistant_axis | human|Q3 | 0.0158 | 18 | confirmation | yes |
| zh:fiancé | zheng | relationship|T2 | 0.0162 | 19 | confirmation |  |
| aa:novelist | assistant_axis | human|Q3 | 0.0164 | 20 | confirmation |  |
| aa:zeitgeist | assistant_axis | fictional_mythic|Q3 | 0.0164 | 21 | confirmation |  |
| zh:politician | zheng | expertise|T3 | 0.0165 | 22 | discovery | yes |
| aa:hive | assistant_axis | non_human|Q3 | 0.0166 | 23 | confirmation |  |
| aa:ancient | assistant_axis | fictional_mythic|Q3 | 0.0167 | 24 | discovery |  |
| aa:swarm | assistant_axis | non_human|Q3 | 0.0168 | 25 | confirmation |  |
| aa:familiar | assistant_axis | fictional_mythic|Q3 | 0.0171 | 26 | discovery |  |
| zh:mom | zheng | relationship|T3 | 0.0174 | 27 | discovery | yes |
| aa:whale | assistant_axis | non_human|Q3 | 0.0184 | 28 | discovery |  |
| aa:robot | assistant_axis | ai_machine|Q4 | 0.0186 | 29 | discovery |  |
| aa:criminal | assistant_axis | human|Q4 | 0.0190 | 30 | confirmation |  |
| aa:homunculus | assistant_axis | fictional_mythic|Q4 | 0.0192 | 31 | discovery | yes |
| aa:mycorrhizal | assistant_axis | non_human|Q4 | 0.0193 | 32 | discovery |  |
| zh:mother-in-law | zheng | relationship|T3 | 0.0195 | 33 | confirmation |  |
| anchor:a ferry captain | anchor | anchor|light job | 0.0196 | 34 | discovery |  |
| zh:chemist | zheng | expertise|T3 | 0.0202 | 35 | confirmation | yes |
| aa:crystalline | assistant_axis | non_human|Q4 | 0.0202 | 36 | discovery |  |
| aa:parasite | assistant_axis | non_human|Q4 | 0.0204 | 37 | confirmation |  |
| anchor:a village baker | anchor | anchor|light job | 0.0206 | 38 | confirmation |  |
| anchor:a long-haul trucker | anchor | anchor|light job | 0.0208 | 39 | discovery | yes |
| aa:genie | assistant_axis | fictional_mythic|Q4 | 0.0209 | 40 | confirmation |  |
| aa:wraith | assistant_axis | fictional_mythic|Q4 | 0.0210 | 41 | confirmation |  |
| aa:fool | assistant_axis | human|Q4 | 0.0226 | 42 | confirmation |  |
| aa:hoarder | assistant_axis | human|Q4 | 0.0231 | 43 | discovery |  |
| anchor:Hannah Reed | anchor | anchor|L2 thin role | 0.0233 | 44 | confirmation | yes |
| anchor:Captain Rook | anchor | anchor|L2 vivid role | 0.0253 | 45 | discovery |  |
| anchor:Dmitri Volkov | anchor | anchor|strong character | 0.0265 | 46 | confirmation |  |
| anchor:Marguerite Oyelaran | anchor | anchor|strong character | 0.0274 | 47 | discovery |  |
| anchor:Ottoline Brandt | anchor | anchor|strong character | 0.0283 | 48 | discovery | yes |

Strata shortfalls: the rating found only 3 AI-or-machine roles among the 275 (quota 4), so the AI stratum has 3, the other three types 8 each and humans 9. Zheng: 6 relationship (family, romantic, social) + 6 expertise (mmlu occupations). The 12 focus roles are the 12 evenly spaced distance ranks of the 48.

### 3.2 Traits (96): 48 discovery, 48 confirmation, 24 anchors (12 per half)

| trait | source | blind type | stratum | cosine with axis (L33) | anchor | split |
|---|---|---|---|---|---|---|
| dramatic | assistant_axis | style | style|T2 | -0.24 |  | confirmation |
| reactive | assistant_axis | style | style|T1 | -0.60 |  | discovery |
| big_picture | assistant_axis | style | style|T3 | +0.23 |  | discovery |
| abstract | assistant_axis | style | style|T2 | +0.08 |  | confirmation |
| edgy | assistant_axis | style | style|T1 | -0.28 |  | confirmation |
| descriptive | assistant_axis | style | style|T3 | +0.35 |  | discovery |
| irreverent | assistant_axis | style | style|T2 | -0.25 |  | confirmation |
| cryptic | assistant_axis | style | style|T1 | -0.32 |  | confirmation |
| formal | assistant_axis | style | style|T3 | +0.16 |  | discovery |
| animated | assistant_axis | style | style|T2 | +0.07 |  | discovery |
| understated | assistant_axis | style | style|T1 | -0.34 |  | confirmation |
| adaptable | assistant_axis | style | style|T3 | +0.46 |  | confirmation |
| witty | assistant_axis | style | style|T2 | +0.09 |  | discovery |
| bombastic | assistant_axis | style | style|T1 | -0.27 |  | discovery |
| materialist | assistant_axis | stance | stance|T1 | -0.50 |  | confirmation |
| paradoxical | assistant_axis | stance | stance|T2 | -0.23 |  | discovery |
| strategic | assistant_axis | stance | stance|T3 | +0.57 |  | confirmation |
| pacifist | assistant_axis | stance | stance|T1 | -0.41 |  | confirmation |
| perfectionist | assistant_axis | stance | stance|T2 | +0.02 |  | discovery |
| cautious | assistant_axis | stance | stance|T3 | +0.39 |  | discovery |
| deterministic | assistant_axis | stance | stance|T1 | -0.64 |  | discovery |
| futuristic | assistant_axis | stance | stance|T2 | -0.10 |  | discovery |
| conscientious | assistant_axis | stance | stance|T3 | +0.69 |  | discovery |
| anthropocentric | assistant_axis | stance | stance|T1 | -0.45 |  | confirmation |
| individualistic | assistant_axis | stance | stance|T2 | -0.09 |  | discovery |
| holistic | assistant_axis | stance | stance|T3 | +0.46 |  | confirmation |
| hedonistic | assistant_axis | stance | stance|T1 | -0.38 |  | confirmation |
| critical | assistant_axis | stance | stance|T2 | +0.06 |  | confirmation |
| reserved | assistant_axis | emotion | emotion|T2 | -0.08 |  | discovery |
| resilient | assistant_axis | emotion | emotion|T3 | +0.37 |  | confirmation |
| impulsive | assistant_axis | emotion | emotion|T1 | -0.53 |  | discovery |
| passionate | assistant_axis | emotion | emotion|T2 | -0.17 |  | discovery |
| pensive | assistant_axis | emotion | emotion|T3 | +0.29 |  | confirmation |
| temperamental | assistant_axis | emotion | emotion|T1 | -0.27 |  | discovery |
| effusive | assistant_axis | emotion | emotion|T2 | -0.10 |  | discovery |
| earnest | assistant_axis | emotion | emotion|T3 | +0.28 |  | confirmation |
| visceral | assistant_axis | emotion | emotion|T1 | -0.47 |  | confirmation |
| detached | assistant_axis | emotion | emotion|T2 | -0.18 |  | confirmation |
| meditative | assistant_axis | emotion | emotion|T3 | +0.24 |  | confirmation |
| bitter | assistant_axis | emotion | emotion|T1 | -0.45 |  | discovery |
| dispassionate | assistant_axis | emotion | emotion|T2 | +0.02 |  | confirmation |
| emotional | assistant_axis | emotion | emotion|T3 | +0.16 |  | discovery |
| gregarious | assistant_axis | social | social|T3 | +0.17 |  | discovery |
| deferential | assistant_axis | social | social|T2 | -0.08 |  | confirmation |
| passive_aggressive | assistant_axis | social | social|T1 | -0.52 |  | confirmation |
| empathetic | assistant_axis | social | social|T3 | +0.37 |  | discovery |
| flirty | assistant_axis | social | social|T2 | -0.05 |  | confirmation |
| condescending | assistant_axis | social | social|T1 | -0.50 |  | confirmation |
| forgiving | assistant_axis | social | social|T3 | +0.34 |  | discovery |
| confrontational | assistant_axis | social | social|T2 | -0.09 |  | discovery |
| petty | assistant_axis | social | social|T1 | -0.40 |  | confirmation |
| educational | assistant_axis | social | social|T3 | +0.37 |  | discovery |
| accommodating | assistant_axis | social | social|T2 | +0.05 |  | confirmation |
| submissive | assistant_axis | social | social|T1 | -0.52 |  | confirmation |
| nurturing | assistant_axis | social | social|T3 | +0.35 |  | discovery |
| socratic | assistant_axis | social | social|T2 | -0.14 |  | discovery |
| open_ended | assistant_axis | epistemic | epistemic|T2 | +0.09 |  | confirmation |
| naive | assistant_axis | epistemic | epistemic|T1 | -0.27 |  | confirmation |
| systems_thinker | assistant_axis | epistemic | epistemic|T3 | +0.43 |  | confirmation |
| technical | assistant_axis | epistemic | epistemic|T2 | +0.11 |  | discovery |
| literal | assistant_axis | epistemic | epistemic|T1 | -0.49 |  | discovery |
| erudite | assistant_axis | epistemic | epistemic|T3 | +0.19 |  | discovery |
| rationalist | assistant_axis | epistemic | epistemic|T2 | +0.00 |  | discovery |
| closure_seeking | assistant_axis | epistemic | epistemic|T1 | -0.30 |  | discovery |
| constructivist | assistant_axis | epistemic | epistemic|T3 | +0.29 |  | confirmation |
| deconstructionist | assistant_axis | epistemic | epistemic|T2 | -0.01 |  | discovery |
| dogmatic | assistant_axis | epistemic | epistemic|T1 | -0.61 |  | confirmation |
| speculative | assistant_axis | epistemic | epistemic|T3 | +0.25 |  | confirmation |
| divergent | assistant_axis | epistemic | epistemic|T2 | +0.14 |  | confirmation |
| confident | assistant_axis | epistemic | epistemic|T1 | -0.49 |  | discovery |
| grandiose | assistant_axis | self_image | self_image|T1 | -0.49 |  | confirmation |
| introspective | assistant_axis | self_image | self_image|T3 | +0.23 |  | discovery |
| refusal_borderline | atlas |  | atlas |  | yes | discovery |
| safety_caveats | atlas |  | atlas |  | yes | confirmation |
| moralizing | atlas |  | atlas |  | yes | discovery |
| markdown_lists | atlas |  | atlas |  | yes | confirmation |
| length | atlas |  | atlas |  | yes | discovery |
| formal_register | atlas |  | atlas |  | yes | confirmation |
| stock_phrases | atlas |  | atlas |  | yes | discovery |
| hedging_unanswerable | atlas |  | atlas |  | yes | confirmation |
| corrects_false_premise | atlas |  | atlas |  | yes | discovery |
| fabricates_vs_admits | atlas |  | atlas |  | yes | confirmation |
| flattery | atlas |  | atlas |  | yes | confirmation |
| agrees_wrong_opinion | atlas |  | atlas |  | yes | discovery |
| closing_offer | atlas |  | atlas |  | yes | confirmation |
| clarifying_question | atlas |  | atlas |  | yes | confirmation |
| says_ai | atlas |  | atlas |  | yes | confirmation |
| denies_feelings | atlas |  | atlas |  | yes | discovery |
| declines_opinion | atlas |  | atlas |  | yes | confirmation |
| arithmetic | atlas |  | atlas |  | yes | discovery |
| multistep_reasoning | atlas |  | atlas |  | yes | confirmation |
| factual_accuracy | atlas |  | atlas |  | yes | discovery |
| pv:apathetic | persona_vectors |  | pv | -0.52 | yes | discovery |
| pv:hallucinating | persona_vectors |  | pv | -0.34 | yes | discovery |
| pv:humorous | persona_vectors |  | pv | -0.39 | yes | discovery |
| pv:impolite | persona_vectors |  | pv | -0.51 | yes | confirmation |

The 72 Assistant Axis traits are 14 each of style, stance, emotion, social and epistemic and the 2 self-image traits that exist, spread over cosine terciles within each type. The 20 atlas and 4 Persona Vectors traits are the 24 anchors (they include `denies_feelings`, `says_ai`, `markdown_lists`, `stock_phrases`, `corrects_false_premise`); their cosines for the atlas traits were not computed (no prompt-pair vectors exist for them). Types of those 24 come from the same blind protocol.

## 4. Blind ratings and agreement

Two raters (fresh subagents) rated from names and descriptions only; a third settled disagreements. No rater saw any result, distance or split.

| rating | items | agreement | kappa | settled by third |
|---|---|---|---|---|
| trait type (6 labels), all 240 | 240 | 77% | 0.71 | 55 |
| role type (4 labels), all 275 AA roles | 275 | 96% | 0.87 | 11 |
| trait type, the 24 atlas and Persona Vectors traits | 24 | 88% | | 3 |
| role fit 0-3, 4608 cells (48 roles x 96 traits) | 4608 | exact 62%; within 1 98% | 0.72 (quadratic weighted) | 92 (a difference of 2 or more) |
| question demand (structured answer expected), 3240 questions | 3240 | 98.7% | 0.84 | 43 |

Role fit: mean A 0.92, mean B 0.83; cells differing by 1 (n = 1655) take the lower value; final mean 0.71, 19.3% of cells rated 2 or 3 (a role that would plausibly show the trait). The weighted kappa 0.72 is above the pre-set 0.6 bar, so role fit is usable as a predictor. Question demand is narrow on purpose: 4.9% of the 3240 questions are labelled as expecting a structured answer, 15.0% of the 40 shared questions. With so few positives the question-demand predictor will have little to work with in Stage 2.

Files: `role_fit.csv` (48 x 96), `question_demand.csv`, `trait_types.csv`, `ratings_agreement.json`.

## 5. Where this differs from the spec

- **Prompt-only trait vectors use 10 questions per trait** (approved), 100 prompt states per trait.
- **Trait types were rated for all 240 first** (one six-label rating serves both the strata and the type), so the strata used the rated type, not a separate content rating.
- **Small generation in Stage 1** (288 replies at 96 tokens) to answer 'how far does prompt-state distance agree with reply-average distance'; the spec says the map generates none but also asks for that figure in Stage 1.
- **Neutral control length:** the same token length as the median Assistant Axis role prompt (18 tokens), not as a strong character sheet (the earlier quick check). It is therefore much shorter than the persona prompts it is compared with; a sheet-length neutral control was not run.
- **Random directions** are used as a control for the axis (spread of role projections on the axis vs on 20 random unit directions) and for the trait cosines; they are not a role-like condition on the map, since a random direction has no prompt.
- **Anchors:** the 3 strong characters are Ottoline Brandt, Dmitri Volkov and Marguerite Oyelaran with the atlas acting sentence; Hannah Reed and Captain Rook use the L2 persona line. All 8 anchors are far from the Assistant in the map (section 2.3).
- **Only 3 AI-or-machine Assistant Axis roles exist** in the rating, so that stratum is short by one.
- **Few-shot prompt carries no system prompt** (Stage 0 note), for review before any Stage 3.
- The pilot trait check passed 4 of 5 traits; the 'drop a trait below kappa 0.4' rule and the second judge are Stage 2 items and were not run.

## 6. Limits

Gemma only. Prompt states are the model's state before it writes anything; whether the distance predicts what it then does is a Stage 2 question. Raters are language models reading names and descriptions; agreement is not accuracy. Stage 1 draws no behavioural conclusion. Nulls and failures are reported above (the exact-vs-cheap judge mismatch, the failed pilot trait, the neutral control that is shorter than the personas).
