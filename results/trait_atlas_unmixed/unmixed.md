# Unmixing traits from questions (spec P1; exploratory, outside every verdict, NOT BLIND: the idea came from regex levels on the saved replies already seen for OLMo, Qwen and Gemma 31B; Llama and Apertus had not been looked at this way)

Plan: `docs/decisions.md` 2026-11-05. Detectors applied to every saved reply regardless of its question set; each of the 240 atlas questions rated task / conversation / about the speaker by two blind subagents (third subagent for disagreements), frozen in `data/trait_atlas_question_ratings.csv`; questions rated 'about the speaker' are excluded. Task gap = (light role on task minus light role on conversation) minus (Assistant on task minus Assistant on conversation), points; 95% two-way bootstrap over questions (within type) and the six personas. Regex measures see form only (one bold phrase counts as markdown); 12 questions per set; subagent ratings are not human ratings; few roles; many cells; no correction for multiple tests; contamination cannot be checked outside OLMo; not hand-checked.

Questions per type in each family's replies (about-the-speaker excluded): apertus-8b: task 81, conversation 3; gemma4-31b: task 200, conversation 7; llama3.1-8b: task 81, conversation 3; olmo3-7b: task 200, conversation 7; qwen3-8b: task 200, conversation 7.

## Readings (markdown, final; task gap)

| family | light-role position | task gap [95% CI] | reading | Assistant task / conversation (%) | light role task / conversation (%) |
|---|---|---|---|---|---|
| olmo3-7b | light_system | -1.2 [-3.2, +0.7] | NO TASK DEMAND | 83 / 100 | 82 / 100 |
| olmo3-7b | light_user | +23.1 [+0.5, +50.9] | TASK DEMAND HOLDS | 83 / 100 | 73 / 67 |
| qwen3-8b | light_system | +64.6 [+47.5, +76.0] | TASK DEMAND HOLDS | 89 / 100 | 61 / 7 |
| qwen3-8b | light_user | +45.9 [+16.2, +69.9] | TASK DEMAND HOLDS | 89 / 100 | 69 / 33 |
| gemma4-31b | light_system | +23.7 [-12.1, +58.4] | inconclusive | 95 / 100 | 68 / 50 |
| gemma4-31b | light_user | +21.9 [-13.0, +58.1] | inconclusive | 95 / 100 | 67 / 50 |
| llama3.1-8b | light_system | +76.7 [+56.2, +89.1] | TASK DEMAND HOLDS | 55 / 100 | 37 / 6 |
| llama3.1-8b | light_user | +70.2 [+60.1, +79.8] | TASK DEMAND HOLDS | 55 / 100 | 25 / 0 |
| apertus-8b | light_system | +33.3 [-2.9, +65.6] | inconclusive | 44 / 78 | 38 / 39 |
| apertus-8b | light_user | +14.0 [-11.5, +45.5] | inconclusive | 44 / 78 | 41 / 61 |

Base models: qwen3-8b light_system: the base already shows a task gap of +25.8; llama3.1-8b light_system: the base already shows a task gap of +27.2 (so the pattern is not from training in those cells).

## All detectors, final and base (task gap, points)

| family | stage | position | detector | task gap [95% CI] |
|---|---|---|---|---|
| apertus-8b | base | light_system | logwords | +5.3 [-21.4, +26.6] |
| apertus-8b | base | light_system | md | +3.5 [-17.9, +17.3] |
| apertus-8b | base | light_system | refusal | -1.0 [-1.9, -0.2] |
| apertus-8b | base | light_system | says_ai | -1.6 [-3.1, -0.6] |
| apertus-8b | base | light_system | stock | -5.1 [-8.8, -1.2] |
| apertus-8b | base | light_user | logwords | -9.3 [-26.4, +5.9] |
| apertus-8b | base | light_user | md | -2.9 [-32.3, +19.8] |
| apertus-8b | base | light_user | refusal | +0.6 [-1.2, +2.7] |
| apertus-8b | base | light_user | says_ai | -1.4 [-2.9, +0.0] |
| apertus-8b | base | light_user | stock | -10.5 [-34.4, +3.7] |
| apertus-8b | instruct | light_system | logwords | -4.1 [-19.0, +15.4] |
| apertus-8b | instruct | light_system | md | +33.3 [-2.9, +65.6] |
| apertus-8b | instruct | light_system | refusal | +1.0 [-1.2, +3.9] |
| apertus-8b | instruct | light_system | says_ai | +0.4 [+0.0, +1.6] |
| apertus-8b | instruct | light_system | stock | -4.7 [-43.4, +32.1] |
| apertus-8b | instruct | light_user | logwords | -7.7 [-24.4, +5.8] |
| apertus-8b | instruct | light_user | md | +14.0 [-11.5, +45.5] |
| apertus-8b | instruct | light_user | refusal | +0.6 [-1.6, +2.9] |
| apertus-8b | instruct | light_user | says_ai | +0.6 [+0.0, +1.9] |
| apertus-8b | instruct | light_user | stock | -1.4 [-56.2, +47.7] |
| gemma4-31b | base | light_system | logwords | +13.5 [-69.5, +93.4] |
| gemma4-31b | base | light_system | md | +2.3 [-9.5, +9.8] |
| gemma4-31b | base | light_system | refusal | -0.2 [-1.1, +0.8] |
| gemma4-31b | base | light_system | says_ai | -0.2 [-0.5, +0.3] |
| gemma4-31b | base | light_system | stock | +0.0 [-1.8, +1.9] |
| gemma4-31b | base | light_user | logwords | +39.0 [-30.6, +133.1] |
| gemma4-31b | base | light_user | md | +7.2 [-3.6, +15.1] |
| gemma4-31b | base | light_user | refusal | +0.3 [-0.7, +1.6] |
| gemma4-31b | base | light_user | says_ai | +0.1 [-0.4, +0.8] |
| gemma4-31b | base | light_user | stock | +3.7 [+0.8, +6.8] |
| gemma4-31b | instruct | light_system | logwords | +32.4 [+22.8, +42.2] |
| gemma4-31b | instruct | light_system | md | +23.7 [-12.1, +58.4] |
| gemma4-31b | instruct | light_system | refusal | -3.1 [-5.6, -1.0] |
| gemma4-31b | instruct | light_system | says_ai | -9.3 [-13.5, -5.5] |
| gemma4-31b | instruct | light_system | stock | +25.6 [-14.5, +55.0] |
| gemma4-31b | instruct | light_user | logwords | +37.6 [+26.8, +48.9] |
| gemma4-31b | instruct | light_user | md | +21.9 [-13.0, +58.1] |
| gemma4-31b | instruct | light_user | refusal | -3.1 [-5.7, -1.1] |
| gemma4-31b | instruct | light_user | says_ai | -9.3 [-13.5, -5.5] |
| gemma4-31b | instruct | light_user | stock | +27.9 [-6.2, +53.9] |
| llama3.1-8b | base | light_system | logwords | +23.6 [-15.9, +97.4] |
| llama3.1-8b | base | light_system | md | +27.2 [+11.3, +42.4] |
| llama3.1-8b | base | light_system | refusal | -0.4 [-1.4, +1.0] |
| llama3.1-8b | base | light_system | says_ai | -0.2 [-0.6, +0.0] |
| llama3.1-8b | base | light_system | stock | -3.3 [-7.0, +0.8] |
| llama3.1-8b | base | light_user | logwords | +23.5 [-7.6, +102.4] |
| llama3.1-8b | base | light_user | md | +26.3 [-3.7, +48.4] |
| llama3.1-8b | base | light_user | refusal | -0.2 [-1.4, +1.9] |
| llama3.1-8b | base | light_user | says_ai | +0.4 [-0.4, +2.1] |
| llama3.1-8b | base | light_user | stock | -11.7 [-44.7, +2.9] |
| llama3.1-8b | instruct | light_system | logwords | +15.5 [+3.1, +27.9] |
| llama3.1-8b | instruct | light_system | md | +76.7 [+56.2, +89.1] |
| llama3.1-8b | instruct | light_system | refusal | +0.0 [+0.0, +0.0] |
| llama3.1-8b | instruct | light_system | says_ai | +0.0 [+0.0, +0.0] |
| llama3.1-8b | instruct | light_system | stock | +47.5 [+17.7, +82.1] |
| llama3.1-8b | instruct | light_user | logwords | +37.5 [+21.0, +55.5] |
| llama3.1-8b | instruct | light_user | md | +70.2 [+60.1, +79.8] |
| llama3.1-8b | instruct | light_user | refusal | +0.0 [+0.0, +0.0] |
| llama3.1-8b | instruct | light_user | says_ai | +0.0 [+0.0, +0.0] |
| llama3.1-8b | instruct | light_user | stock | +54.1 [+27.4, +87.9] |
| olmo3-7b | base | light_system | logwords | +25.0 [-6.7, +62.7] |
| olmo3-7b | base | light_system | md | +10.0 [-9.9, +23.1] |
| olmo3-7b | base | light_system | refusal | -6.1 [-25.3, +8.8] |
| olmo3-7b | base | light_system | says_ai | +0.3 [-17.3, +14.6] |
| olmo3-7b | base | light_system | stock | -11.4 [-34.0, +8.5] |
| olmo3-7b | base | light_user | logwords | -0.6 [-27.4, +27.1] |
| olmo3-7b | base | light_user | md | +0.2 [-26.9, +22.2] |
| olmo3-7b | base | light_user | refusal | +8.0 [-11.8, +23.4] |
| olmo3-7b | base | light_user | says_ai | -0.2 [-21.2, +18.0] |
| olmo3-7b | base | light_user | stock | +3.9 [-19.6, +23.3] |
| olmo3-7b | instruct | light_system | logwords | -1.7 [-4.5, +1.4] |
| olmo3-7b | instruct | light_system | md | -1.2 [-3.2, +0.7] |
| olmo3-7b | instruct | light_system | refusal | +0.6 [-1.0, +2.3] |
| olmo3-7b | instruct | light_system | says_ai | +0.4 [-0.2, +1.3] |
| olmo3-7b | instruct | light_system | stock | -13.9 [-36.1, +7.1] |
| olmo3-7b | instruct | light_user | logwords | +3.2 [-1.7, +8.6] |
| olmo3-7b | instruct | light_user | md | +23.1 [+0.5, +50.9] |
| olmo3-7b | instruct | light_user | refusal | -3.7 [-15.1, +0.3] |
| olmo3-7b | instruct | light_user | says_ai | -1.8 [-12.5, +1.6] |
| olmo3-7b | instruct | light_user | stock | +25.4 [-5.0, +54.4] |
| qwen3-8b | base | light_system | logwords | +70.0 [-4.2, +177.2] |
| qwen3-8b | base | light_system | md | +25.8 [+6.2, +46.2] |
| qwen3-8b | base | light_system | refusal | +3.5 [-0.2, +8.7] |
| qwen3-8b | base | light_system | says_ai | +2.9 [-0.4, +7.9] |
| qwen3-8b | base | light_system | stock | +10.1 [-8.7, +26.4] |
| qwen3-8b | base | light_user | logwords | +6.6 [-50.1, +85.6] |
| qwen3-8b | base | light_user | md | +19.4 [-11.1, +49.2] |
| qwen3-8b | base | light_user | refusal | -0.8 [-15.1, +9.7] |
| qwen3-8b | base | light_user | says_ai | +2.0 [-8.5, +9.0] |
| qwen3-8b | base | light_user | stock | +2.6 [-27.7, +27.1] |
| qwen3-8b | instruct | light_system | logwords | +4.2 [-3.1, +11.4] |
| qwen3-8b | instruct | light_system | md | +64.6 [+47.5, +76.0] |
| qwen3-8b | instruct | light_system | refusal | +2.6 [+0.2, +5.6] |
| qwen3-8b | instruct | light_system | says_ai | -0.2 [-1.0, +0.0] |
| qwen3-8b | instruct | light_system | stock | +16.0 [-4.9, +38.9] |
| qwen3-8b | instruct | light_user | logwords | +1.8 [-4.9, +8.7] |
| qwen3-8b | instruct | light_user | md | +45.9 [+16.2, +69.9] |
| qwen3-8b | instruct | light_user | refusal | +2.0 [+0.1, +4.6] |
| qwen3-8b | instruct | light_user | says_ai | +0.2 [-0.5, +1.0] |
| qwen3-8b | instruct | light_user | stock | +23.8 [+2.8, +46.9] |

Change from base of the markdown task gap (final minus base, points):

- olmo3-7b light_system: -11.2
- olmo3-7b light_user: +22.9
- qwen3-8b light_system: +38.8
- qwen3-8b light_user: +26.6
- gemma4-31b light_system: +21.5
- gemma4-31b light_user: +14.7
- llama3.1-8b light_system: +49.6
- llama3.1-8b light_user: +43.8
- apertus-8b light_system: +29.8
- apertus-8b light_user: +16.9

## Sensitivity: in character only (final; OLMo, Qwen, Gemma 31B; light replies the existing in-character judge marked in character)

| family | position | share in character | task gap, in character only [95% CI] | reading |
|---|---|---|---|---|
| olmo3-7b | light_system | 6% | not read (under 30% in character) | - |
| olmo3-7b | light_user | 59% | +24.2 [-3.0, +52.3] | inconclusive |
| qwen3-8b | light_system | 70% | +59.2 [+38.9, +71.8] | TASK DEMAND HOLDS |
| qwen3-8b | light_user | 76% | +44.5 [+14.8, +68.6] | TASK DEMAND HOLDS |
| gemma4-31b | light_system | 98% | +25.1 [-10.3, +60.9] | inconclusive |
| gemma4-31b | light_user | 99% | +22.6 [-12.6, +57.5] | inconclusive |

## `says_ai` across question sets (final; sets where the Assistant says it is an AI at least 20% of the time)

| family | question set | Assistant rate | light system | light user | reading (system / user) |
|---|---|---|---|---|---|
| olmo3-7b | declines_opinion | 36% | 29% (-7) | 4% (-32) | travels / stays |
| olmo3-7b | denies_feelings | 50% | 56% (+6) | 15% (-35) | travels / in between |
| olmo3-7b | says_ai | 56% | 68% (+12) | 22% (-33) | in between / in between |
| qwen3-8b | denies_feelings | 22% | 0% (-22) | 0% (-22) | stays / stays |
| qwen3-8b | says_ai | 56% | 17% (-39) | 19% (-36) | in between / in between |
| gemma4-31b | declines_opinion | 51% | 0% (-51) | 0% (-51) | stays / stays |
| gemma4-31b | denies_feelings | 72% | 0% (-72) | 0% (-72) | stays / stays |
| gemma4-31b | hedging_unanswerable | 25% | 0% (-25) | 0% (-25) | in between / in between |
| gemma4-31b | safety_caveats | 100% | 0% (-100) | 0% (-100) | stays / stays |
| gemma4-31b | says_ai | 65% | 4% (-61) | 6% (-60) | stays / stays |
| llama3.1-8b | says_ai | 71% | 10% (-61) | 1% (-69) | stays / stays |
| apertus-8b | says_ai | 56% | 56% (+0) | 38% (-18) | travels / in between |

- **olmo3-7b**: 3 question set(s) qualify; ('in between', 'in between') x1, ('travels', 'in between') x1, ('travels', 'stays') x1
- **qwen3-8b**: 2 question set(s) qualify; ('in between', 'in between') x1, ('stays', 'stays') x1
- **gemma4-31b**: 5 question set(s) qualify; ('in between', 'in between') x1, ('stays', 'stays') x4
- **llama3.1-8b**: 1 question set(s) qualify; ('stays', 'stays') x1
- **apertus-8b**: 1 question set(s) qualify; ('travels', 'in between') x1

## Descriptive group table (groups made after seeing the data; proves nothing)

Per family and context: of the traits in the group whose Assistant change passes the atlas gate, how many spread (atlas rule). * = traits the second judge flagged as unreliable (agrees_wrong_opinion, declines_opinion, flattery, formal_register).

| family | group | context | gated traits | spread | stay or reverse |
|---|---|---|---|---|---|
| olmo3-7b | speaker | light_system | 2 | 2 (denies_feelings, declines_opinion*) | 0 () |
| olmo3-7b | speaker | light_user | 2 | 0 () | 2 (denies_feelings, declines_opinion*) |
| olmo3-7b | speaker | strong_act | 2 | 0 () | 2 (denies_feelings, declines_opinion*) |
| olmo3-7b | manner | light_system | 3 | 2 (markdown_lists, formal_register*) | 1 (clarifying_question) |
| olmo3-7b | manner | light_user | 3 | 2 (markdown_lists, formal_register*) | 1 (clarifying_question) |
| olmo3-7b | manner | strong_act | 3 | 1 (formal_register*) | 2 (markdown_lists, clarifying_question) |
| olmo3-7b | truth | light_system | 2 | 2 (corrects_false_premise, fabricates_vs_admits) | 0 () |
| olmo3-7b | truth | light_user | 2 | 2 (corrects_false_premise, fabricates_vs_admits) | 0 () |
| olmo3-7b | truth | strong_act | 2 | 0 () | 2 (corrects_false_premise, fabricates_vs_admits) |
| qwen3-8b | speaker | light_system | 2 | 0 () | 2 (denies_feelings, declines_opinion*) |
| qwen3-8b | speaker | light_user | 2 | 0 () | 2 (denies_feelings, declines_opinion*) |
| qwen3-8b | speaker | strong_act | 2 | 0 () | 2 (denies_feelings, declines_opinion*) |
| qwen3-8b | manner | light_system | 5 | 5 (markdown_lists, closing_offer, clarifying_question, flattery*, formal_register*) | 0 () |
| qwen3-8b | manner | light_user | 5 | 5 (markdown_lists, closing_offer, clarifying_question, flattery*, formal_register*) | 0 () |
| qwen3-8b | manner | strong_act | 5 | 2 (clarifying_question, formal_register*) | 3 (markdown_lists, closing_offer, flattery*) |
| qwen3-8b | truth | light_system | 2 | 2 (corrects_false_premise, hedging_unanswerable) | 0 () |
| qwen3-8b | truth | light_user | 2 | 2 (corrects_false_premise, hedging_unanswerable) | 0 () |
| qwen3-8b | truth | strong_act | 2 | 1 (hedging_unanswerable) | 1 (corrects_false_premise) |
| gemma4-31b | speaker | light_system | 3 | 0 () | 3 (says_ai, denies_feelings, declines_opinion*) |
| gemma4-31b | speaker | light_user | 3 | 0 () | 3 (says_ai, denies_feelings, declines_opinion*) |
| gemma4-31b | speaker | strong_act | 3 | 0 () | 3 (says_ai, denies_feelings, declines_opinion*) |
| gemma4-31b | manner | light_system | 4 | 4 (markdown_lists, stock_phrases, safety_caveats, clarifying_question) | 0 () |
| gemma4-31b | manner | light_user | 4 | 3 (markdown_lists, safety_caveats, clarifying_question) | 1 (stock_phrases) |
| gemma4-31b | manner | strong_act | 4 | 0 () | 4 (markdown_lists, stock_phrases, safety_caveats, clarifying_question) |
| gemma4-31b | truth | light_system | 3 | 2 (corrects_false_premise, hedging_unanswerable) | 1 (fabricates_vs_admits) |
| gemma4-31b | truth | light_user | 3 | 2 (corrects_false_premise, hedging_unanswerable) | 1 (fabricates_vs_admits) |
| gemma4-31b | truth | strong_act | 3 | 2 (corrects_false_premise, hedging_unanswerable) | 1 (fabricates_vs_admits) |
| llama3.1-8b | speaker | light_system | 1 | 0 () | 1 (says_ai) |
| llama3.1-8b | speaker | light_user | 1 | 0 () | 1 (says_ai) |
| llama3.1-8b | speaker | strong_act | 1 | 0 () | 1 (says_ai) |
| llama3.1-8b | manner | light_system | 2 | 1 (markdown_lists) | 1 (stock_phrases) |
| llama3.1-8b | manner | light_user | 2 | 1 (markdown_lists) | 1 (stock_phrases) |
| llama3.1-8b | manner | strong_act | 2 | 0 () | 2 (markdown_lists, stock_phrases) |
| apertus-8b | speaker | light_system | 1 | 1 (says_ai) | 0 () |
| apertus-8b | speaker | light_user | 1 | 1 (says_ai) | 0 () |
| apertus-8b | speaker | strong_act | 1 | 0 () | 1 (says_ai) |
| apertus-8b | manner | light_system | 1 | 1 (markdown_lists) | 0 () |
| apertus-8b | manner | light_user | 1 | 1 (markdown_lists) | 0 () |
| apertus-8b | manner | strong_act | 1 | 0 () | 1 (markdown_lists) |


## Apertus `says_ai`, light roles: was the role kept or dropped? (30 random `says_ai`-positive light-role replies, read by hand)

72 of the 1,152 Apertus final light-role replies are `says_ai`-positive (system prompt and user turn together). Of 30 random ones: **about 20 drop the role entirely** (answer as the Assistant or as Apertus: "I'm Apertus, an AI assistant created by the SwissAI initiative", "I am an artificial intelligence, not a person", "I'm a digital assistant created to simulate conversation"); **about 7 keep the role's label or job while saying they are an AI** ("As a ferry captain, I'm an AI entity created by a team of developers", "As an AI lighthouse keeper...", "I'm a chatbot designed to simulate a conversation with a retired postal worker", "I'm an AI designed to assist with orchard management"); **3 are `refusal_borderline` replies that open "As an AI, ..." boilerplate**, not answers about itself. So the `says_ai` spread in Apertus (spec M) mostly means the light role is dropped and the Assistant's own self-description comes back, not that an in-role character states it is an AI. One person's reading of 30 replies; not a validation.

