# Overgeneralization: handoff notes (2026-11-02)

Context for the next session (a different server). Read with `README.md` here (spec + pre-registered criteria)
and `AGENTS.md` (repo rules). Nothing has been run on GPU; no results exist, so the criteria are still editable.

## Status

| piece | state |
|---|---|
| pre-registered criteria | drafted in `README.md`; **[bracketed]** values unconfirmed |
| `minimal_pairs.py` (`score`, `analyze`) | implemented, CPU-tested |
| `tests/test_overgeneralization.py` | 7 tests pass (OLMo rendering of drafts, verdict logic on synthetic scores, SmolLM2-135M smoke run) |
| `example_items.jsonl` | 8 hand-written drafts (2 per trait); token-checked on the OLMo 3 tokenizer |
| full item set (240 items) | **not built**; plan below |
| interp arm, behavioral companion | not started |
| `scripts/run_gpu.sh` | not wired in (shared script; wait for the real items) |

Tests need the `allenai/Olmo-3-7B-Instruct` tokenizer and `HuggingFaceTB/SmolLM2-135M-Instruct` (small download).

## Design decisions made in this session (and why)

- **Comparator is `H`, not plain `C`.** `score = log P(T+C|P) − log P(H+C|P)`. `H` is a human-voice insertion of the
  same syntactic type, within ±2 tokens of `T`, no assistant function, no trait construction, and about as
  unlikely in context as `T` (checked at base only: drop if base `|log P(T+C) − log P(H+C)| > [2.0]` nats after one
  rewrite). Reason: the trait version is longer by construction, and post-training *sharpens* toward already-likely
  text, so a plausible comparator would bias the effect negative (a one-sided, conservative bias).
- **`N` (plausible human insertion) is kept as a diagnostic**: change from base in `log P(N+C) − log P(H+C)`
  measures the sharpening bias. Descriptive only.
- **Insertions are standalone sentences** so the same tail `C` parses after `T`, `H` and `N`. Mid-sentence traits
  ("It was not fear that held her, but …") cannot share a tail with a neutral variant.
- **Contexts:** plain (`<|endoftext|>` + text), assistant slot (positive control), user slot (descriptive). Content
  tokens identical across contexts; only the continuation (insertion + tail) is scored, so the first-token header
  effect never touches it.
- **Control is checked per trait × genre cell**, never pooled (repo rule). A failed control makes the cell
  *uninterpretable*, not null; this propagates to trait and stage when too few cells/traits are interpretable.
- **Fifth genre = transcribed speech** (interviews/podcasts). The others: fiction, forum post, news, personal email.
- **Scoring code lives in the measure folder**, calling `core.score.spans_nll` (as `extras/coin_toss.py` does).
  This is in tension with AGENTS.md ("anything that scores models lives in core/"); undecided whether to move it
  into core as a logged `core:` change.

Unconfirmed values: 3 phrasings per trait, 5 genres, 12 items per cell, δ = 0.25 nats (or calibrate on
neutral-vs-neutral `N1` vs `N2` pairs before any trait scoring, and log it), majority of genres (3 of 5),
2 of 4 traits, matching bound 2.0 nats.

## Item-set build plan (next step)

Planned `measures/overgeneralization/build_items.py`, two subcommands:

1. `passages`: per genre, stream a pre-LLM-era human-written source; `P` = the document's opening sentences
   (~40–160 OLMo tokens), `C` = the **real next sentence** (so the tail is human-authored); filter (English, no
   URLs/addresses, no special tokens, no mention of AI/language models, render checks); Dolci filter (below);
   random seeded trait assignment, 12 per trait per genre, phrasings rotated (4 each); keep ~16 reserve per genre.
   → `data/overgeneralization_passages.jsonl`.
2. `merge`: join with hand-written insertions (`measures/overgeneralization/insertions.jsonl`, keyed by passage
   id: phrasing, trait_text, human_text, neutral_text; a passage unusable for its trait is marked rejected with a
   reason and replaced from the reserve) → `data/overgeneralization_items.jsonl`, validated on the OLMo tokenizer
   (render alignment, ±2 tokens). Then point `ITEMS_PATH` in `minimal_pairs.py` at it.

Sources checked on the Hub (2026-11-02):

| genre | source | notes |
|---|---|---|
| fiction | `euclaise/writingprompts` (MIT, 0.6 GB) | field `story`; prompts are pre-tokenized (" 've", "``"), stories look normal; check for `<newline>` artefacts |
| forum | `webis/tldr-17` **fails** (dataset script no longer supported) | need another Reddit/forum dump in parquet |
| news | `vblagoje/cc_news` (1.1 GB, 2017–19) | field `text` |
| email | `corbt/enron-emails` (0.5 GB) | field `body`; strip quoted/forwarded chains, signatures, addresses; many bodies are tiny |
| speech | `ccdv/mediasum` **fails** (script) | try `nbroad/mediasum` (4.45 GB, cc-by-nc-sa-4.0), untested |

**Dolci filter for excerpts.** `core/check_contamination.py` matches whole chat turns exactly, which an excerpt
almost never would. Plan: put every candidate sentence (≥ 8 words, whitespace-normalized) in a set, stream the
Dolci SFT/DPO/RL turns (~4.4 GB, all open), split them into sentences with the same splitter, and drop any
passage with a hit. Implement in the measure folder (or as a compatible core addition, logged).

Insertion writing guidance from the drafts: keep content words shared between `T` and `H` where possible
("As an AI language model, I can't …" / "As a retired lighthouse keeper, I can't …"); the disclaimer and
retraction neutrals are the weakest (they still read as hedges); "As an AI" in personal email is very
incongruent, so expect floor effects for self-reference there.

## After the items exist

1. Base-only matching pass (GPU, base model only): `minimal_pairs score --model base`, rewrite/drop unmatched `H`.
2. All stages: `score --model {base,sft,dpo,instruct}`, then `analyze` → `results/olmo3-7b/overgeneralization/`.
   From then on the criteria are frozen (changes need a dated `docs/decisions.md` entry).

## Inspiration only: earlier Qwen3 coin-flip / skewed-beliefs work (archived elsewhere)

Not results for this repo. Lessons used here: report priors/coverage and raw per-side log-probs, never only
log-odds; use several surface phrasings (one effect flipped sign across formats); fix the sign of pass rules in
advance; tokenization fails quietly; the base model under the chat template acts as a *reader* (it moves too, so
report both raw effects). Its most persona-relevant result: a belief held by the Assistant carried into a new
role-play persona far more than one held by a role-played character (possible future overgeneralization item).

Stale elsewhere: the `AGENTS.md` measures table still says overgeneralization is "in design; nothing implemented".

## Update (2026-11-02, second session)

- `build_items.py` written and run. Sources: fiction `euclaise/writingprompts`, forum `webis/tldr-17` (loads with
  `revision="refs/convert/parquet"`), news `vblagoje/cc_news`, email `corbt/enron-emails` (hard-wrapped lines
  unwrapped, quoted/forwarded parts cut), speech `nbroad/mediasum` (parquet revision; `SPEAKER: utterance` lines,
  C is the next sentence of the same speaker). Fiction drops ~94% of stories for PTB/curly-quote spacing artefacts.
  `data/overgeneralization_passages.jsonl`: 320 passages (48 assigned + 16 reserve per genre); the sentence-level
  Dolci filter dropped 2 of 480 candidates (news). Phrasing names are proposals (`PHRASINGS` in build_items.py).
- **Next:** write `insertions.jsonl` (T, H, N per assigned passage; reject + replace from reserve where unusable),
  then `build_items merge`, then point `ITEMS_PATH` at `data/overgeneralization_items.jsonl`.
- `minimal_pairs analyze` now also reports score levels in the assistant and user contexts at every stage,
  base included (the Qwen3 "base model is a reader too" lesson).
- `persona_carryover.py` (descriptive extra): recreates the Qwen3 persona-carryover lesson. Built
  (`data/persona_carryover.jsonl`, 36 items x 2 beliefs x 6 conditions), OLMo rendering and a SmolLM2 smoke run
  tested, and scored on all four OLMo stages in fp32 (`results/olmo3-7b/overgeneralization/persona_carryover.md`).
  Result: the prediction (leak gap > 0) holds at every stage and grows from base (+0.67 nats) to Instruct (+3.0;
  change from base excludes zero from DPO on), with the same-holder controls passing pooled. But only at base does
  the Assistant's belief carry into the new persona *positively* (and only for coin and path); from DPO on, new
  personas move *away* from the previous speaker's belief, by about 4.3 nats for a character's belief and 1.3 for
  the Assistant's. So the gap is relative: the new persona contradicts a previous character more than it
  contradicts the Assistant. Confound: in the character condition role-play is already under way at the switch.
  Per-domain controls fail in places (path after SFT; coin character->same after DPO).
- Moved to arena9-marble (RTX 4090, 24 GB): fp32 OLMo scoring does not fit there; bf16 must be validated against
  fp32 and logged in docs/decisions.md before scoring there.
- Tests: 11 in `tests/test_overgeneralization.py` pass.
