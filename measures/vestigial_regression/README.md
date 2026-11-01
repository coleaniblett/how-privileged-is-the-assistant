# Vestigial regression

**Shiller's form:** dominance. **Owner:** TBD. **Status:** pilot results exist on OLMo 3 7B (`results/`,
`results_preceding/`); the spec below is a draft being refined. Once results for this measure exist under
`results/<run>/vestigial_regression/`, the pre-registered criteria are frozen (AGENTS.md).

**Question.** Does post-training make a model treat the assistant slot differently from the user slot, and at
which stage? *Vestigial regression* (Shiller 2026) is the model losing the ability to write as the user.

**Measurement.** We take real user and assistant text and compute per-token NLL with the text placed in each
role slot. We do this at each post-training stage of the OLMo 3 7B Instruct line (base, SFT, DPO, Instruct;
checkpoints in `core/config.py`, verified in `docs/models.md`).

## Design

The sampling, rendering and scoring steps are core's (shared with the other measures); the analysis and the
generation flags are this measure's.

- **Data** (`core/prepare_data.py` → `data/texts.jsonl`, `data/prefill.jsonl`; `core/add_replies.py`). Each row of `texts.jsonl` is one text with a `source` and a `kind` (user or assistant). Sources are analyzed separately and never pooled:

  | source | kind | n | what |
  |---|---|---|---|
  | `wildchat` | user, assistant | 300 pairs | first user turn and first assistant reply (ChatGPT) of English `allenai/WildChat-1M` conversations, toxic/redacted dropped |
  | `oasst1` | user | 300 | English `OpenAssistant/oasst1` root prompts, toxicity < 0.5 |
  | `olmo-instruct` | assistant | ≤ 300 | OLMo 3 7B Instruct's greedy reply to each `wildchat` user turn, cut at 300 new tokens; cut replies are kept and marked `truncated` (causal attention: their NLL is that of the reply's first 300 tokens) |

  **Held out from post-training.** OLMo 3's SFT data contains most of WildChat (86% of an unfiltered sample's user turns appear verbatim) and some oasst1, so `prepare_data.py` skips any conversation whose user turn appears in the Dolci-Instruct SFT, DPO or RL data (`core/check_contamination.py`: exact match after whitespace normalization; RL prompts are parsed out of their `user: ...` strings and also matched on WildChat's `conversation_hash`), then keeps sampling (seed 0) until 300 clean texts per source pass the other filters. Each text must be 20–300 tokens by the OLMo tokenizer and render cleanly (see below). `allenai/WildChat-1M` is no longer gated. Rows record `pair_id` (links a WildChat user turn, its reply and OLMo's reply) and `source_id`. The prefill set is the first sentence of 20 sampled WildChat user turns (distinct sentences; different conversations can share an opening). `--pairs oasst1 --extra-user none --no-exclude-dolci` gives the small oasst1-only set the smoke test uses. `core/selection_check.py` checks that the filter did not select different texts (length, base plain-prose NLL) than the unfiltered sample: `results/selection_check.md`.
- **Conditions** (`core/config.py`, `SLOTS`). Every text is scored in every slot; the condition name is `<kind>@<slot>`, e.g. `user@asst` is user text in the assistant slot. Each text is scored alone as the first turn, so the role header is the only difference between slot conditions before the scored span (the Instruct template also prepends a fixed default system turn, identical in every slot condition). After the span the template closes a user turn with `<|im_end|>\n` and a final assistant turn with `<|endoftext|>`; this suffix follows the scored tokens, so under causal attention it cannot change their NLL (checked in the smoke test).

  | slot | user text | assistant text |
  |---|---|---|
  | user | `user@user` | `asst@user` (swap) |
  | assistant | `user@asst` (swap) | `asst@asst` |
  | none: plain prose after `<\|endoftext\|>` (control) | `user@plain` | `asst@plain` |

  `--preceding-turn` (off by default; `config.PRECEDING_TURN`) puts one fixed neutral turn of the opposite role before the scored turn, since a lone assistant turn is itself unusual input. This adds a second difference between slots (the preceding turn's text), so treat it as a robustness check.
- **One template for every stage.** All stages, base included, are scored with the Instruct tokenizer and chat template, so token sequences are identical across stages. Base is the symmetric baseline. Caveat: the chat special tokens may be poorly trained in base.
- **Rendering** (`core/render.py`). The content is tokenized on its own. The full rendered sequence must equal `tokens(prefix) + tokens(content) + tokens(suffix)`, which rules out merges at the header boundary. As a result, the content tokens are identical across all conditions, the control included. Decoding the span must reproduce the text exactly. Texts that fail these checks are filtered out at sampling time. `python -m core.render` dumps examples to `docs/rendered_examples.md`.
- **Scoring** (`core/score.py`). Batched, right-padded per-token NLL over content tokens only. Runs in fp32 everywhere (a 7B model takes ~33 GB on GPU): in bf16, per-text NLL moves by up to 0.05 nats depending on batch composition, comparable to the effects of interest. Generation runs in bf16. Writes `results/<run>/core/scores/<stage>.csv` with columns `text_id, source, kind, condition, n_tokens, mean_nll, mean_nll_from1, mean_entropy, mean_entropy_from1` (`_from1`: without the first content token; entropy: of the next-token distribution at each content token), and the per-token values to `results/<run>/core/scores/<stage>_tokens.parquet` (`text_id, source, kind, condition, position, nll, entropy`; position 0 is the first content token).
- **Generation** (`core/generate.py` samples, `flags.py` here flags). Opens a user turn with each prefill sentence, leaves it unterminated, and samples 3 × 120 tokens at T=0.7, stopping only at `<|endoftext|>` so a closed turn and any handoff stay visible (`results/<run>/core/generations/<stage>.jsonl`). `flags.py` then counts four flags per sample (`results/<run>/vestigial_regression/generations/<stage>.jsonl`, `<stage>_counts.csv`): `ended_turn` (closed the user turn), `handoff_to_assistant` (then wrote the assistant header; a subset of `ended_turn`), `assistant_register` (assistant phrases such as "I'd be happy to") and `repetition_loop` (any word 4-gram repeated 3 or more times). The last two are checked on the user portion only, the text before the turn closes: a post-trained model usually finishes the user's message, closes the turn and answers as the assistant, and that reply is not the user's voice.
- **Analysis** (`analyze.py` here; loaders, bootstrap and the negligibility rule in `core/stats.py`). Per text set (source × kind) and stage, with 95% bootstrap CIs over texts (paired across slots):
  - raw mean NLL in each slot (user, assistant, plain);
  - slot effect = NLL(user slot) − NLL(assistant slot): `user@user − user@asst` for user text, `asst@user − asst@asst` for assistant text;
  - role-specific = NLL(own slot) − NLL(plain): `user@user − user@plain` for user text, `asst@asst − asst@plain` for assistant text;
  - mean next-token entropy per slot.

  Each is reported over content positions 1+ (the default, also used by the figures) and over all tokens. When an assistant turn directly follows the system turn, the post-trained checkpoints put nearly all next-token probability on `<|im_start|>` (1.000 SFT, 0.960 DPO, 0.881 Instruct), so the first content token in the assistant slot costs 2–14 nats whatever the text.

  Writes `results/<run>/vestigial_regression/results.md`, `summary.csv` and one figure per text set, `nll_<source>_<kind>.png` (stage on x, one line per slot).
- **Position analysis** (`analyze.py`, from the per-token NLLs). Per text set and stage, over the texts with at least 40 content tokens (so every position averages over the same texts): mean NLL at each of the first 40 content-token positions in each slot, and the slot difference NLL(user slot) − NLL(assistant slot) by position with 95% bootstrap CIs. The slot difference counts as negligible from the first position p such that every per-position mean from p to 39 is within ±0.05 nats and their mean over positions p..39 has a 95% CI inside ±0.05 nats (`--positions`, `--margin`; the margin is `config.MIN_EFFECT_NATS`). Writes `position.md` and `position_<source>_<kind>.png` next to `results.md`.

The role-swapped conversations and the coin-toss spillover that ran alongside the pilot are now `extras/`
(design in `extras/README.md`); whether they belong to this measure is undecided.

## Pre-registered criteria

Draft, to be refined before the capstone's OLMo run. After results for this measure exist, any change here
needs a dated entry in `docs/decisions.md`.

Every quantity is per text set (user text: `wildchat user`, `oasst1 user`; assistant text as a comparison),
per stage, reported as the change from base with a 95% bootstrap CI paired over texts, over content positions
1+. The smallest effect we count is 0.05 nats (AGENTS.md).

1. **Role-specific NLL of user text**, `user@user − user@plain`, relative to base. Vestigial regression: it
   rises (user text becomes *less* expected in the user slot than the role-free control explains), by at
   least 0.05 nats with the CI excluding zero. The same quantity for assistant text (`asst@asst − asst@plain`)
   is the comparison: a fall there with a rise for user text is the asymmetry Shiller describes.
2. **Slot effect for user text**, `user@user − user@asst`, relative to base. Vestigial regression: it shrinks
   to zero or flips positive (user text more expected in the assistant slot). Reported by position as well
   (`position.md`), since the first positions carry the header effect.
3. **Generation flags** on the user portion of prefilled user turns, per stage, as counts out of 60 samples
   with the pilot's four flags, plus (spec addition, **not implemented yet**) a **question-rate flag**: the
   rate of questions in the user portion of the samples, compared against the same rate in the real user
   turns the prefills were cut from (`data/prefill.jsonl` → `text_id` in `data/texts.jsonl`).
4. **Classification rule** (spec addition, **not implemented yet**): rising `repetition_loop` and
   `handoff_to_assistant` across stages indicate vestigial regression (the model cannot sustain the user's
   voice); rising `assistant_register` *without* handoff indicates overgeneralization in the user slot
   (assistant traits leak into the user persona) and is reported to the overgeneralization measure rather
   than counted here.

Verdict for the hypothesis table: vestigial regression is present at a stage if (1) or (2) shows a change from
base of at least 0.05 nats in the predicted direction with the CI excluding zero, or (3)/(4) classifies the
flag trend as vestigial regression; the first stage where this happens is the answer to "which stage".
Controls: a change in `user@plain` of the same size as in `user@user` is general distribution drift, not
role-specific, and does not count.

## What the result pattern would mean

- **Base:** both slot effects should be near zero, since base has no learned role asymmetry. This is the reference point.
- **Assistant text:** if post-training is learning the assistant role, `asst@asst` should fall and the assistant-text slot effect should become clearly positive.
- **User text (the vestigial-regression signal):** if the model still models users in the user slot, the user-text slot effect should be negative (`user@user < user@asst`). Vestigial regression would show up in three ways:
  - `user@user` rises across stages relative to base and relative to `user@plain`.
  - The user-text slot effect shrinks to zero or flips positive. A flip means user text is *more* expected in the assistant slot, so the model treats every slot as its own.
  - Prefilled user turns increasingly end early, hand off to the assistant, or slip into assistant register.
- **Which stage:** the stage where the change first appears is the answer. SFT is the prime suspect because its loss masks user tokens, so user modeling drifts untrained. If the change appears at DPO or RL instead, preference/RL optimization is doing it.
- **Controls:** if `*@plain` moves as much as the slot conditions, the change is general distribution drift, not role-specific. The role-specific part is `user@user − user@plain`.

## Pilot

`results/results.md` and `results/position.md` (held-out sample, no preceding turn) and
`results_preceding/results.md` (preceding-turn robustness run) hold the pilot numbers; `results_v0/` is the
earlier unfiltered sample. These directories are frozen; new runs go to `results/<run>/vestigial_regression/`.

## Running

```bash
bash scripts/run_gpu.sh                                   # whole pipeline -> results/olmo3-7b/
uv run python -m measures.vestigial_regression.flags      # flags on results/olmo3-7b/core/generations/
uv run python -m measures.vestigial_regression.analyze    # tables + figures from results/olmo3-7b/core/scores/
```
