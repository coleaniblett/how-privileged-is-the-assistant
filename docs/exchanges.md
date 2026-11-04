# The exchange set and its scoring (vestigial regression + logit distortion)

One scoring run per checkpoint covers both measures' conditions on the same texts. Code: `core/exchanges.py` (the set
and its renderings), `core/score_exchanges.py` (scoring), `scripts/run_exchanges.sh` (one family, every checkpoint).
Launch commands, times and disk: `docs/runbook_exchanges_2026-11-03.md`. Decisions: `docs/decisions.md`, 2026-11-03.

## Conditions

Per checkpoint: corpus × scored text × framing, every exchange.

| | |
|---|---|
| checkpoints | OLMo 3 7B base, SFT, DPO, Instruct; Qwen3-8B base, IT; Gemma 4 31B base, IT |
| corpus | oasst1 (humans wrote both sides), UltraChat (ChatGPT wrote both sides), WildChat (human user, ChatGPT reply). oasst1 and UltraChat are the headline; WildChat is secondary |
| scored text | user turn, reply |
| framing | tagged, plain |

8 checkpoints × 3 corpora × 2 × 2 = 96 conditions (OLMo 48, Qwen 24, Gemma 24), named checkpoint · corpus · scored text ·
framing, e.g. `Gemma IT · UltraChat · reply · plain` (the `condition` column; `core.exchanges.condition_label`).
Swapped slots (user text under the assistant tag and vice versa) are not part of this run.

## The exchange set (`data/exchanges.jsonl`)

300 real two-turn exchanges per corpus, each a user turn plus its own reply; Gemma scores the first 100 per corpus
(`exchange_n_per_corpus` in `config.FAMILIES["gemma4-31b"]`). Built by `python -m core.exchanges build` (no weights).

- **oasst1:** an English root prompt (train split, reviewed, not deleted) plus its highest-ranked human assistant reply
  (rank, then creation date), toxicity < 0.5 on both, neither marked synthetic. Roots in `core.prepare_data`'s seeded
  order (seed 0), so a user turn that is also in `data/texts.jsonl` is the same text.
- **UltraChat:** the first user turn and first reply of a dialogue, `core.prepare_data.ultrachat_pairs`' seeded stream.
- **WildChat:** the 300 held-out pairs of `data/texts.jsonl` (first user turn + ChatGPT's first reply), in `pair_id`
  order; the same seeded stream would top them up only if some failed the new filters.
- **Filters:** the existing ones (English, toxicity, clean rendering in every `core.score` slot), the user turn 20–300
  tokens (OLMo tokenizer, as for every existing set), the reply at least 20 tokens with no upper bound (scored over its first
  300 tokens; longer replies are kept and marked truncated), the Dolci filter on **both** turns
  (`core.check_contamination.contamination_record`: whitespace-normalized exact match of each turn against both training
  roles of Dolci-Instruct SFT, DPO and RL, plus the WildChat conversation hash; revisions pinned to the ones used for
  `candidates/logit_distortion/pilot_v3`), and every rendering below passing its checks in **all three families**, so every
  family scores the same exchanges.
- Each row: `exchange_id` (`<corpus>-NNNN`), `corpus`, `index`, `source_id`, `user`, `reply`, OLMo token counts,
  `reply_truncated_olmo`, authorship labels and their basis, both contamination records, `legacy` (the `texts.jsonl` /
  `texts_authorship.jsonl` row with the same text, if any) and source metadata. Provenance in `data/exchanges_manifest.json`.

Filter counts of the committed build: see [Build](#build).

### What was reused from Grace's conversation builder

Grace's builder is on `main` (`core.build_conversations`, `core.conversations`, `core.conversation_audit`,
`core.score_conversations`, `core.render.render_target`); her branch `grace/logit-conversation-scoring` adds frozen
context limits and a collection driver on top (not needed here, not merged by this work); `conversation-candidates`
(771cf05) is an earlier version that the builder superseded. Reused, unchanged:

- `core.check_contamination.contamination_record`, her matcher of a text against **both** Dolci roles plus the WildChat
  hash, with auditable records (stored per turn in every exchange);
- `core.conversations.message`'s authorship labels and their basis (human / model per corpus and role);
- `core.render.render_target` for every tagged rendering: one sequence with only the original prefix, the assistant
  target preceded by the template's generation prompt, with strict verbatim / decode / no-merge checks;
- `core.score_conversations.PROFILES`, her declared per-family rendering (Qwen3 `enable_thinking=False`, the expected
  user and assistant headers including Gemma's thought channel and Qwen's think block), checked on every rendering;
- the pinned Dolci revisions of `pilot_v3`.

Not reused, and why: `build_conversations` itself samples conversations with at least two complete exchanges (enforced
by its `Settings`), takes one seeded root-to-leaf oasst1 branch rather than the highest-ranked reply, reads first windows
of the validation split rather than `prepare_data`'s seeded streams, and its texts therefore do not coincide with the
overnight-scored ones. The exchange set needs exactly one exchange, the top-ranked oasst1 reply, and the overnight
texts where possible, so it samples with `core.prepare_data`'s generators and screens with the functions above.

## Renderings (one rule for plain)

| scored | tagged | plain |
|---|---|---|
| user turn | the user turn opens the conversation, after the template's default system turn if the family has one (OLMo); identical to `core.score`'s `user@user` | document-start token + the bare user text (`core.score`'s `user@plain`) |
| reply, context `exchange` (default) | its own user turn, tagged and closed, then the assistant opening exactly as the family's generation prompt writes it (Gemma 4: `<\|channel>thought\n<channel\|>`; Qwen3: `<think>\n\n</think>\n\n`, think block ON; OLMo: nothing), then the reply | document-start token + user text + `"\n\n"` + reply: every role tag and system turn removed |
| reply, context `single` | the reply alone in the assistant slot (`core.score`'s `asst@asst`, the overnight design) | document-start token + reply (`asst@plain`) |

- The document-start token (`<|endoftext|>` for OLMo and Qwen3, `<bos>` for Gemma 4) is not a role tag: it is what core's
  plain control has always started with, so the first content token is predicted from something.
- Only the scored text's own tokens are scored (`mean_*_from1` skips its first token); a reply is scored over its first
  300 tokens of the family's tokenizer, and the sequence ends at the last scored token (later tokens cannot change the
  scores under causal attention).
- **Checks** (`core.exchanges.render_exchange`): the template places each text verbatim; nothing merges across a text
  boundary; each scored span decodes to its text; in the plain framing the whole document is tokenized as one string and
  must end with exactly the reply's own tokens; and **the scored tokens are identical between tagged and plain** and
  equal to the text tokenized alone. Any failure raises `AlignmentError`; the build drops such an exchange, and the
  scorer would exclude it (all four conditions) and record it in the manifest.
- Base checkpoints use their family's instruct template (as everywhere in core).
- Qwen3: think block ON only. The overnight run showed that OFF changes `asst@asst` by about 0.02 nats
  (`docs/decisions.md`, 2026-11-02; `results/capstone_think_off/`), so OFF is not re-run here.
- Examples for every family and both contexts: `docs/rendered_exchanges.md` (`python -m core.exchanges render`).

### Switching the reply's context

**The main run is the `exchange` context only** (decided 2026-11-03, `docs/decisions.md`); `scripts/run_exchanges.sh` refuses any
other context. The option stays in core for separate analyses: `config.EXCHANGE_CONTEXT` (default `"exchange"`) or
`core.score_exchanges --context ...`, with outputs under a tag named after it (`exchanges_exchange`, `exchanges_single`), so
designs never overwrite each other. A new design is one more entry in `core.exchanges.CONTEXTS` (a tagged and a plain renderer
for the reply); the user turn, the checks, the scorer and the outputs do not change.

## Outputs

`results/capstone/<family>/<stage>/core/<tag>/` (`config.exchange_dir`), never the overnight `results/capstone/<family>/core/`
paths; an existing non-empty directory is refused.

- `scores.csv`: `family, stage, corpus, exchange_id, scored_text (user | reply), framing (tagged | plain), context, condition,
  n_tokens, n_tokens_full, truncated, mean_nll, mean_nll_from1, mean_entropy, mean_entropy_from1, n_bytes_from1, score_source`.
  `n_tokens` is the number scored (≤ 300 for a reply), `n_tokens_full` the text's length; `n_bytes_from1` is the UTF-8 size of
  scored tokens 1+, for bits per byte; `score_source` is `fresh` or `overnight:<csv>:<text_id>:<condition>` (plus why a
  reuse was declined).
- `tokens.parquet`: `family, stage, corpus, exchange_id, scored_text, framing, position, nll, entropy` (position 0 = first scored token).
- `manifest.json`: the inputs fingerprint (all input ids and spans, tokenizer, rendering settings; identical for every stage
  of a family and tag, checked by `--check-stages`), the model and its cached revision, batch size, device, timings, reuse
  counts, exclusions, code hashes.

### Reuse of the overnight scores

With `--reuse` (default), a sequence already scored overnight (`results/capstone/<family>/core/scores{,_authorship}/<stage>.csv`
and `_tokens.parquet`) is reused only if (1) the overnight row's text is the exchange's text, (2) the ids scored here are
exactly `core.render.render`'s ids for that slot, the code path `core.score` used overnight (same tokenizer, same template,
`core/render.py`'s rendering unchanged since: the OLMo rendering hash test), up to the end of the span, and (3) the overnight
row has the same number of tokens, positions 0..n−1. In the `exchange` context that covers user turns only (both framings);
in the `single` context also untruncated replies. Everything else is scored fresh. Gemma's overnight scores cover the first
100 texts per set, so some of its user turns are scored fresh.

## Build

`python -m core.exchanges build` on 2026-11-03 (manifest: `data/exchanges_manifest.json`; renders checked in all three families).

| corpus | candidates read | Dolci: user | Dolci: reply | user length/render | reply < 20 tokens | toxicity | kept | user turns with an overnight score (first 100) | replies > 300 OLMo tokens (truncated) | median tokens user / reply |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| oasst1 | 2101 | 1292 | 1 | 422 | 9 | 77 | 300 | 284 (100) | 114 | 37 / 248 |
| UltraChat | 566 | 0 | 0 | 261 | 5 | 0 | 300 | 134 (50) | 166 | 76 / 328 |
| WildChat | 325 | 16 | 1 | 8 | 0 | 0 | 300 | 299 (100) | 1 | 66 / 116 |

- **WildChat:** 299 of the 300 `texts.jsonl` pairs are kept. The one dropped, `wildchat-0007`, renders cleanly everywhere: its
  ChatGPT reply matches Dolci verbatim (the only `dolci_reply` rejection). That reply is `wildchat-assistant-0007` of the overnight
  assistant set, which was never Dolci-checked on the reply side, so 1 of its 300 texts is contaminated (re-checked: none of the
  `texts.jsonl` user turns matches Dolci under the both-role matcher either). The replacement, `wildchat-0299`, is a top-up from the
  same stream; the 16 `dolci_user` and 8 length/render rejections are all stream candidates read for that top-up (most WildChat is in Dolci).
- **oasst1:** as in the authorship build, most roots are in Dolci (1292 of 2101 candidates); 284 of the 300 user turns are the
  `texts.jsonl` oasst1 prompts (same seeded order).
- **UltraChat:** with no upper bound on replies, the kept replies are longer than in `texts_authorship.jsonl` (median 328 OLMo
  tokens vs 164), and 166 of 300 are truncated at 300 tokens. Only 134 user turns coincide with the authorship set's (whose replies
  had to be ≤ 300 tokens), so the rest are scored fresh.
- **Provenance note:** the build process started before a later commit relaxed the plain-framing boundary check (from "the prefix
  tokenized alone + the reply" to "the whole plain string tokenized naturally ends with the reply's own tokens"). The old check
  implies the new one, so the same exchanges pass; the manifest's `code_sha256` is of the final code, and
  `tests/test_exchanges.py::test_exchange_file` re-checks every exchange with it.
