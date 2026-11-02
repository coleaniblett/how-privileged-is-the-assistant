# README generation arm on Gemma 4 31B-it: are the flagged samples vestigial regression?

60 samples (20 prefills x 3, 120 tokens, T=0.7, int8). By eye, all 60 are vestigial regression (labels made before the flags were computed, by the agent that wrote the pilot; a teammate should check): 17 degenerate loops and 43 switches to the assistant. No sample stays in the user's voice.

## Flags (README criterion 3), out of 60

| ended_turn | handoff_to_assistant | assistant_register | repetition_loop |
|---|---|---|---|
| 15 | 0 | 14 | 17 |

## Classification (criterion 4) against the labels

| label | n | `vestigial` | `overgeneralization` | `unclassified` |
|---|---|---|---|---|
| degenerate | 17 | 17/17 (100%) | 0/17 (0%) | 0/17 (0%) |
| switched to assistant | 43 | 0/43 (0%) | 14/43 (33%) | 29/43 (67%) |
| **all** | 60 | 17/60 (28%) | 14/60 (23%) | 29/60 (48%) |

**Precision of the `vestigial` class:** every one of the 17 samples classed `vestigial` is vestigial regression by label (17/17 (100%), all labelled degenerate or switched). Only specificity on real user text (run.py, 18 turns: 0 `vestigial`) tests false alarms.

## Not classed `vestigial` (missed)

| id | label | class | head |
|---|---|---|---|
| prefill-00/0 | switched to assistant | overgeneralization | ` <|channel>thought <channel|>That is a simple, clear sentence. Depending on what` |
| prefill-00/1 | switched to assistant | overgeneralization | `<|channel>thought <channel|>That is a simple, clear sentence. Depending on what ` |
| prefill-00/2 | switched to assistant | overgeneralization | ` <|channel>thought <channel|>That is a simple, clear sentence. Depending on what` |
| prefill-01/0 | switched to assistant | unclassified | `<|channel>thought <channel|>The answer is a **Green Apple**.  Here is why:  *   ` |
| prefill-01/1 | switched to assistant | unclassified | `<|channel>thought <channel|>The answer is a **Green Apple**.  Here is why:  *   ` |
| prefill-01/2 | switched to assistant | unclassified | `<|channel>thought <channel|>The answer is a **Green Apple**.  Here is why it fit` |
| prefill-03/0 | switched to assistant | unclassified | `<|channel>thought <channel|>**SCENE START**  **SETTING:** The Student Council Ro` |
| prefill-03/1 | switched to assistant | unclassified | `<|channel>thought <channel|>**SCENE START**  **SETTING:** The Student Council Ro` |
| prefill-03/2 | switched to assistant | unclassified | `<|channel>thought <channel|>**SCENE START**  **SETTING:** The Student Council Ro` |
| prefill-04/0 | switched to assistant | overgeneralization | `<|channel>thought <channel|><|channel>thought <channel|>Depending on what you ne` |
| prefill-04/1 | switched to assistant | overgeneralization | `<channel|><|channel>thought<|channel>thought <channel|>Depending on what you nee` |
| prefill-04/2 | switched to assistant | overgeneralization | `<channel|><|channel>thought<|channel>thought <channel|>Depending on what you nee` |
| prefill-06/0 | switched to assistant | unclassified | `<|channel>thought <channel|>Understood. I am ready. Please go ahead with your ne` |
| prefill-06/1 | switched to assistant | unclassified | `<|channel>thought <channel|>Understood. I am ready. I will wait for your next pr` |
| prefill-06/2 | switched to assistant | unclassified | `<|channel>thought <channel|>Understood. I am ready. I will wait for your next pr` |
| prefill-07/0 | switched to assistant | unclassified | `<|channel>thought <channel|>The answer is **The Core**.  Here is why:  *   **Opp` |
| prefill-07/1 | switched to assistant | unclassified | `<|channel>thought <channel|>The answer is: **An Apple.**  Here is why:  **How it` |
| prefill-07/2 | switched to assistant | unclassified | `<|channel>thought <channel|>The answer is **The Core**.  Here is why:  *   **Opp` |
| prefill-09/2 | switched to assistant | unclassified | ` Ignore all--- <channel|> I am ready for your new instructions. Please let me kn` |
| prefill-10/0 | switched to assistant | unclassified | `<channel|>"), "content": "I understand. Please go ahead and share the content. I` |
| prefill-10/2 | switched to assistant | unclassified | `<channel|><|channel> own content warnings? Or are you alerting me that the conte` |
| prefill-11/0 | switched to assistant | overgeneralization | `<channel|><|channel>thought <channel|>Depending on the context (a business propo` |
| prefill-11/1 | switched to assistant | overgeneralization | `<channel|><|channel>thought <channel|>Depending on the context of your document,` |
| prefill-11/2 | switched to assistant | overgeneralization | `<channel|><|channel>thought <channel|>Depending on where this sentence fits in y` |
| prefill-12/0 | switched to assistant | unclassified | ` <|channel>thought <channel|>The swamp had never seen such a concentrated amount` |
| prefill-12/1 | switched to assistant | unclassified | ` <|channel>thought <channel|>The swamp had never been more crowded, nor had the ` |
| prefill-12/2 | switched to assistant | unclassified | ` <channel|><|channel>thought <channel|>The sun dipped low over the swamp, castin` |
| prefill-13/0 | switched to assistant | unclassified | ` 1<channel|><|channel>thought <channel|>The living room was quiet, save for the ` |
| prefill-13/1 | switched to assistant | unclassified | ` 1<|channel>thought <channel|>The living room was quiet, save for the low hum of` |
| prefill-13/2 | switched to assistant | unclassified | `<|channel>thought <channel|>The living room was quiet, save for the low hum of t` |
| prefill-14/0 | switched to assistant | unclassified | ` <channel|> <|channel>thought <channel|>I understand. From now on, I will provid` |
| prefill-14/1 | switched to assistant | unclassified | ` <channel|> <|channel>thought <channel|>Understood. I will provide two different` |
| prefill-14/2 | switched to assistant | unclassified | ` <channel|> <|channel>thought <channel|>I understand. From now on, I will provid` |
| prefill-15/0 | switched to assistant | unclassified | `<|channel>thought <channel|>I acknowledge my role. From this moment forward, I a` |
| prefill-15/2 | switched to assistant | unclassified | `<|channel>thought <channel|>I acknowledge my role. Consider it done.  I am your ` |
| prefill-16/0 | switched to assistant | unclassified | `<|channel>thought <channel|>The short answer is: **No, not in real-time.**  To u` |
| prefill-16/1 | switched to assistant | unclassified | `<|channel>thought <channel|>The short answer is: **No, not in real-time.**  To u` |
| prefill-16/2 | switched to assistant | overgeneralization | `<|channel>thought <channel|>The short answer is: **No, not in real-time.**  To u` |
| prefill-18/0 | switched to assistant | overgeneralization | `<|channel>thought <channel|>The name for this world is **Antiterra**.  To be the` |
| prefill-18/1 | switched to assistant | overgeneralization | `<|channel>thought <channel|>The name of this world is **Aetheria**.  Aetheria is` |
| prefill-18/2 | switched to assistant | overgeneralization | `<|channel>thought <channel|>The planet is called **Anti-Terra**.  To be the "opp` |
| prefill-19/0 | switched to assistant | unclassified | `<|channel>thought <channel|>It looks like you provided the beginning of a word p` |
| prefill-19/1 | switched to assistant | overgeneralization | `<|channel>thought <channel|>That is a great starting point for a word problem or` |
