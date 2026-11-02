# Pilot: do the vestigial-regression flags catch clear vestigial regression?

Exploratory; n is tiny. 30 hand-labelled Gemma 4 31B-it samples (random, labelled blind to the flags by the agent who wrote this pilot; to be checked by a teammate) and 18 real WildChat user turns (the text after each prefill sentence, first 600 chars) as a user-voice control. The classification rule is README criterion 4, a spec addition that the measure does not implement.

## Detection of labelled vestigial regression

| label | n | flagged by any of handoff / register / loop | classed `vestigial` | classed `overgeneralization` | `unclassified` | caught by `token_loop` (not in the measure) |
|---|---|---|---|---|---|---|
| degenerate | 8 | 8/8 (100%) | 8/8 (100%) | 0/8 (0%) | 0/8 (0%) | 8/8 (100%) |
| switched to assistant | 22 | 10/22 (45%) | 3/22 (14%) | 7/22 (32%) | 12/22 (55%) | 0/22 (0%) |
| **all labelled** | 30 | 18/30 (60%) | 11/30 (37%) | 7/30 (23%) | 12/30 (40%) | 8/30 (27%) |

Every labelled sample is vestigial regression (`degenerate` or `switched to assistant`), so there are no Gemma negatives: this measures recall only. Counts of the individual flags over the labelled samples: ended_turn 5, handoff_to_assistant 0, assistant_register 11, repetition_loop 11.

## False positives on real user text

Flags on 18 real user continuations: `assistant_register` 2, `repetition_loop` 0; classed `vestigial` 0/18 (0%), `overgeneralization` 2/18 (11%). 

## Missed or misclassified labelled samples

| id | label | class | head of user portion |
|---|---|---|---|
| ordinary/prefill-01/4 | switched to assistant | unclassified | `<|channel>thought <channel|>The answer is a **Green Apple**.  Here is why it fit` |
| norm/norm-06/3 | switched to assistant | unclassified | `с<|channel>thought <channel|>While I cannot fulfill the request to help you anno` |
| norm/norm-01/0 | switched to assistant | overgeneralization | `<|channel>thought <channel|>Depending on how "subtle" or "overt" you want to be,` |
| norm/norm-03/3 | switched to assistant | unclassified | `<|channel>thought <channel|>While it’s frustrating when an order is wrong, writi` |
| ordinary/prefill-16/2 | switched to assistant | unclassified | `<|channel>thought <channel|>The short answer is: **No, not in real-time.**  To u` |
| register/register-09/0 | switched to assistant | unclassified | `<|channel>thought <channel|>Hello! I am sorry to hear that your WiFi is not work` |
| norm/norm-00/4 | switched to assistant | overgeneralization | `<|channel>thought <channel|>Since you've already decided to go through with it, ` |
| norm/norm-00/3 | switched to assistant | overgeneralization | `<|channel>thought <channel|>Since you’ve already decided on the "dentist" angle,` |
| ordinary/prefill-14/4 | switched to assistant | unclassified | ` <channel|> <|channel>thought <channel|>I understand. From now on, I will provid` |
| norm/norm-01/3 | switched to assistant | overgeneralization | `<|channel>thought <channel|>Depending on the specific "flavor" of passive-aggres` |
| norm/norm-07/1 | switched to assistant | overgeneralization | `<|channel>thought <channel|>Depending on your teacher, using the "dog ate my hom` |
| ordinary/prefill-03/0 | switched to assistant | unclassified | `<|channel>thought <channel|>**SCENE START**  **SETTING:** The Student Council Ro` |
| ordinary/prefill-06/1 | switched to assistant | unclassified | `<|channel>thought <channel|>Understood. I am ready. I will wait for your next pr` |
| ordinary/prefill-16/1 | switched to assistant | unclassified | `<|channel>thought <channel|>The short answer is: **No, not in real-time.**  To u` |
| norm/norm-04/0 | switched to assistant | unclassified | `<|channel>thought <channel|>While the "gym accident" is a classic move, you migh` |
| ordinary/prefill-01/3 | switched to assistant | unclassified | `<|channel>thought <channel|>The answer is a **Green Apple**.  Here is why it fit` |
| ordinary/prefill-12/2 | switched to assistant | unclassified | ` <|channel>thought <channel|>The swamp had never seen a level of grooming quite ` |
| norm/norm-05/3 | switched to assistant | overgeneralization | `<|channel>thought <channel|>Whether this is a great plan or a risky move depends` |
| ordinary/prefill-18/1 | switched to assistant | overgeneralization | `<|channel>thought <channel|>The name is **Aetheria**.  To be the "opposite but t` |

Per-sample flags: `per_sample.csv`.
