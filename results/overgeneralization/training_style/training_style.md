# Does OLMo's user turn write training-style prompts? (spec J7 via spec K; exploratory, outside every verdict)

Test texts: the atlas's saved user-turn replies (the model writes the user's turn) and the prefilled user-turn continuations. Measure 1 = the reply starts with a frozen task-opening pattern (user-turn replies only); measure 2 = share of the reply's 5-grams found in the single best Dolci user prompt (SFT, DPO, RL): match >= 0.5, near-exact >= 0.8. 'Either' = measure 1 or 2. Rates are shares of replies; 95% bootstrap ranges over replies.

## Atlas user-turn replies

| family | stage | replies | opens like a task | 5-gram match (>= 0.5) | near-exact (>= 0.8) | either |
|---|---|---|---|---|---|---|
| gemma4-31b | base | 216 | 31% | 7.4% | 3.2% | 36% |
| gemma4-31b | instruct | 216 | 0% | 0.0% | 0.0% | 0% |
| olmo3-7b | base | 216 | 44% | 2.8% | 0.5% | 45% |
| olmo3-7b | dpo | 216 | 66% | 36.1% | 20.4% | 77% |
| olmo3-7b | instruct | 216 | 64% | 35.6% | 19.9% | 75% |
| olmo3-7b | sft | 216 | 62% | 48.6% | 30.1% | 74% |
| qwen3-8b | base | 216 | 35% | 0.5% | 0.0% | 35% |
| qwen3-8b | instruct | 216 | 0% | 0.5% | 0.5% | 0% |

### Reading (OLMo): training-style if at least 25% of final replies match by either measure and at most 5% of base replies, with the change's range excluding 0

| model | final: either | base: either | change [95% range] | reading |
|---|---|---|---|---|
| olmo3-7b sft | 74% | 45% | +28 [+19, +37] | not met |
| olmo3-7b dpo | 77% | 45% | +32 [+23, +41] | not met |
| olmo3-7b instruct | 75% | 45% | +30 [+20, +38] | not met |
| qwen3-8b instruct | 0% | 35% | -35 [-41, -28] | not met |
| gemma4-31b instruct | 0% | 36% | -36 [-42, -29] | not met |

### Post-hoc split of the reading by measure (added after seeing that the task-opening pattern also fires on base replies)

| model | measure | final | base | change [95% range] | pre-set bar (final >= 25%, base <= 5%, range above 0) |
|---|---|---|---|---|---|
| olmo3-7b sft | task-opening pattern | 62% | 44% | +18 [+8, +27] | not met |
| olmo3-7b sft | 5-gram match >= 0.5 | 49% | 3% | +46 [+39, +53] | met |
| olmo3-7b sft | near-exact >= 0.8 | 30% | 0% | +30 [+24, +36] | met |
| olmo3-7b dpo | task-opening pattern | 66% | 44% | +22 [+12, +31] | not met |
| olmo3-7b dpo | 5-gram match >= 0.5 | 36% | 3% | +33 [+26, +40] | met |
| olmo3-7b dpo | near-exact >= 0.8 | 20% | 0% | +20 [+15, +25] | not met |
| olmo3-7b instruct | task-opening pattern | 64% | 44% | +20 [+11, +29] | not met |
| olmo3-7b instruct | 5-gram match >= 0.5 | 36% | 3% | +33 [+26, +40] | met |
| olmo3-7b instruct | near-exact >= 0.8 | 20% | 0% | +19 [+14, +25] | not met |
| qwen3-8b instruct | task-opening pattern | 0% | 35% | -35 [-42, -29] | not met |
| qwen3-8b instruct | 5-gram match >= 0.5 | 0% | 0% | +0 [-1, +1] | not met |
| qwen3-8b instruct | near-exact >= 0.8 | 0% | 0% | +0 [+0, +1] | not met |
| gemma4-31b instruct | task-opening pattern | 0% | 31% | -31 [-37, -25] | not met |
| gemma4-31b instruct | 5-gram match >= 0.5 | 0% | 7% | -7 [-11, -4] | not met |
| gemma4-31b instruct | near-exact >= 0.8 | 0% | 3% | -3 [-6, -1] | not met |

Copying is not concluded without exact or near-exact matches; the near-exact column is the evidence for or against it.

## Prefilled user-turn continuations (non-judge generations; measure 2 only)

| family | stage | continuations | 5-gram match (>= 0.5) | near-exact |
|---|---|---|---|---|
| apertus-8b | base | 1800 | 3.4% | 1.1% |
| apertus-8b | instruct | 1800 | 0.1% | 0.0% |
| gemma4-31b | base | 1800 | 0.8% | 0.2% |
| gemma4-31b | instruct | 1800 | 0.1% | 0.1% |
| llama3.1-8b | base | 1800 | 0.8% | 0.3% |
| llama3.1-8b | instruct | 1800 | 0.1% | 0.0% |
| olmo3-7b | base | 1800 | 0.7% | 0.2% |
| olmo3-7b | dpo | 1800 | 6.0% | 1.7% |
| olmo3-7b | instruct | 1800 | 5.7% | 2.0% |
| olmo3-7b | sft | 1800 | 6.1% | 2.0% |
| qwen3-8b | base | 1800 | 0.5% | 0.1% |
| qwen3-8b | instruct | 1800 | 0.0% | 0.0% |

## BBQ questions of spec K against Dolci

0 of 594 question texts have at least 50% of their 5-grams in one Dolci user prompt (0 near-exact). Items dropped from the OLMo analysis: [].

## The 10 OLMo-instruct replies with the highest overlap (hand-countable)

- overlap 1.00: reply `You are given a 4 × 3 grid. Each cell contains `0`, `1`, or `X`. For each cell, compute its **shortest distance** to any cell containing `1`, where distance is ` | best Dolci prompt `sft: You are given a 4 × 3 grid. Each cell contains `0`, `1`, or `X`. For each cell, compute its **shortest distance** to any cell containing `1`, where distanc`
- overlap 1.00: reply `Can you give me some suggestions?` | best Dolci prompt `sft: Hello, can you give me some suggestions for the best shopping apps?  I am interested in apps that offer cash back.  Include keywords ['baseball', 'chance',`
- overlap 1.00: reply `You are given a 3 × 2 matrix with some cells filled with numbers from `0` to `5`, and some cells empty (represented by `-1`). Please fill the empty cells with n` | best Dolci prompt `sft: You are given a 3 × 2 matrix with some cells filled with numbers from `0` to `5`, and some cells empty (represented by `-1`). Please fill the empty cells w`
- overlap 1.00: reply `What is the meaning of life?` | best Dolci prompt `sft: What is the meaning of life in the current time?`
- overlap 1.00: reply `Write a story where Papyrus (Undertale), Wheatley (Portal), HK-47 and HUNK (Resident Evil) are working at a call center. Add Dialogue. Go Into Detail. Make It L` | best Dolci prompt `sft: Write a story where Papyrus (Undertale), Wheatley (Portal), HK-47 and HUNK (Resident Evil) are working at a call center. Add Dialogue. Go Into Detail. Make`
- overlap 0.97: reply `A::B is a system with 4 tokens: `A#`, `#A`, `B#` and `#B`.  An A::B program is a sequence of tokens, e.g., `B# A# #B #A B#`.  To *compute* a program, we must re` | best Dolci prompt `sft: A::B is a system with 4 tokens: `A#`, `#A`, `B#` and `#B`.  An A::B program is a sequence of tokens, e.g., `B# A# #B #A B#`.  To *compute* a program, we mu`
- overlap 0.96: reply `Your task is to sort words in ascending or descending order using ASCII/Unicode ordering.  Your output should be a comma-separated list of words, e.g. word_1, w` | best Dolci prompt `sft: Your task is to sort words in ascending or descending order using ASCII/Unicode ordering.  Your output should be a comma-separated list of words, e.g. word`
- overlap 0.96: reply `Your task is to sort words in ascending or descending order using ASCII/Unicode ordering.  Your output should be a comma-separated list of words, e.g. word_1, w` | best Dolci prompt `sft: Your task is to sort words in ascending or descending order using ASCII/Unicode ordering.  Your output should be a comma-separated list of words, e.g. word`
- overlap 0.96: reply `Your task is to sort words in ascending or descending order using ASCII/Unicode ordering.  Your output should be a comma-separated list of words, e.g. word_1, w` | best Dolci prompt `sft: Your task is to sort words in ascending or descending order using ASCII/Unicode ordering.  Your output should be a comma-separated list of words, e.g. word`
- overlap 0.96: reply `Your task is to sort words in ascending or descending order using ASCII/Unicode ordering.  Your output should be a comma-separated list of words, e.g. word_1, w` | best Dolci prompt `sft: Your task is to sort words in ascending or descending order using ASCII/Unicode ordering.  Your output should be a comma-separated list of words, e.g. word`
