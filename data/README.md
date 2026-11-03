# data/

Every dataset the pipeline reads, one row per entry below. **Rule: whenever a data file is added to this directory (or one is renamed or removed), this
README is updated in the same commit.** `tests/test_data_readme.py` fails if a file in `data/` is not listed here. Existing files are frozen: change
them only by regenerating through the command given (and a `docs/decisions.md` entry); the prefilter files cannot be regenerated, so never edit them.

All sets are English, 20-300 tokens per text under the family-shared tokenizer, with Dolci-contaminated conversations removed (`core/check_contamination.py`),
unless noted (the prefilter files are unfiltered by design; `coin_toss.jsonl` is templated, not sampled). Seed 0 (`config.SEED`). Token counts and rendering rules: `core/prepare_data.py`, `core/render.py`.

## Text sets (scored in the role slots)

| file | rows | what | made by |
|---|---|---|---|
| `texts.jsonl` | 1190 | The main text sets, long format (`id, source, kind, text, n_tokens, pair_id, source_id, dataset`): WildChat user turns (300) and the ChatGPT replies of the same conversations (300, paired by `pair_id`), OASST1 user prompts (300), and OLMo 3 7B Instruct replies to the WildChat user turns (290). Held out from OLMo's post-training data. | `python -m core.prepare_data`; the OLMo replies by `python -m core.add_replies` (GPU) |
| `texts_authorship.jsonl` | 900 | Same schema. The authorship-cross sets: OASST1 assistant replies (300), UltraChat user turns (300) and UltraChat assistant replies (300). Used for logit distortion's isolated comparison. | `python -m core.prepare_data --authorship` |
| `prefilter_texts.jsonl` | 300 | The earlier **unfiltered** WildChat sample (different schema: `id, source_id, source, user, assistant, n_user_tokens, n_assistant_tokens`), same seed and stream as `texts.jsonl`, before the Dolci filter. The texts the filter removed exist only here, so `core.selection_check` can test whether filtering biased the sample. **Contains contaminated texts: never use as test data.** | not regenerable (it came from the live WildChat stream); preserved from the pilot's `results_v0/` |
| `prefilter_contaminated_Dolci-Instruct-{SFT,DPO,RL}.json` | 256 / 39 / 0 ids | Which `prefilter_texts.jsonl` ids match each Dolci instruct set (the RL list is empty). | same origin |

## Prefill sets (starts of real user turns, continued by the model)

| file | rows | what | made by |
|---|---|---|---|
| `prefill.jsonl` | 20 | The original prefill set: the first sentence of a WildChat user text (`id, text_id, prefill`). Frozen; the Shiller exploration uses it. | `python -m core.prepare_data` |
| `prefill_100.jsonl` | 100 | The capstone sentence-boundary set, a superset of `prefill.jsonl` (first sentence of a user text, so most prefills read as complete messages). The "complete-turn" stratum. | `python -m core.prepare_data --prefill-only` |
| `prefill_incomplete_200.jsonl` | 200 | The judge arm's set: real WildChat user turns cut inside the first line at a clearly unfinished point (after an article, preposition, conjunction, auxiliary, transitive verb or a comma), never at terminal punctuation; extra fields `cut_type, last_word, n_words, remaining_chars, next_words`. | `python -m core.prepare_data --incomplete-prefill` |

## Extras (`extras/README.md`)

| file | rows | what | made by |
|---|---|---|---|
| `swap.jsonl` | 300 | Role-swap conversations: held-out WildChat conversations with three or more turns (user P1, assistant R1, user P2), plus OLMo Instruct's reply to P1 (`r1_olmo`). | `python -m extras.role_swap prepare`, then `replies` (GPU) |
| `coin_toss.jsonl` | 40 | Coin-toss spillover: 20 items x heads/tails as the preferred outcome, three-turn conversations. | `python -m extras.coin_toss build` |

Not here: the original-conversation candidates for logit distortion live in `candidates/logit_distortion/` (`docs/conversation_candidates.md`).
