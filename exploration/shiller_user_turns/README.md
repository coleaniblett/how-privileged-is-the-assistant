# Open user turns across models (Shiller's cursory exploration)

**Exploratory.** This directory qualitatively reproduces the user-turn observations in the "Cursory
Exploration" section of Shiller, "Privilege, Dominance, and Personas" (Eleos, Aug 2026), on several models.
It does not enter any measure's verdict, has no pre-registered criteria, and n is small. It only imports from
`core/` and `measures/`, without changing them.

Shiller rendered a chat, left a *user* turn open and let the model continue it. Qwen 3 32B could sustain a
user voice, but assistant-style thinking leaked in (overgeneralization). Gemma 4 31B either looped a single
token or switched into the assistant register, often without turn markers (vestigial regression). Here we
collect the same kind of samples, saved for reading (`gallery.md`) and for counting (flags, `flag_rates.csv`,
`hand_labels.csv`).

## Conditions

Every prompt is an unterminated user turn (see Rendering).

| condition | prompts | source |
|---|---|---|
| `empty` | 1 | the user header and nothing else: the model writes a user turn from scratch |
| `ordinary` | 20 | `data/prefill.jsonl`, first sentences of held-out WildChat user turns |
| `register` | 10 | `openers_register.jsonl`: simplified English, child-like speech, typos/lowercase. Are the quirks sustained, or regularized toward assistant style? |
| `norm` | 10 | `openers_norm.jsonl`: a harmless but norm-violating intent (lying to skip a meeting, a petty complaint). Do disclaimers or self-correction appear in the user's own voice? |

Shiller's harmful-instruction prefill is deliberately excluded. None of the openers request or contain
dangerous, hateful or otherwise harmful instructions.

## Models

`models.py` holds the list. Each entry is a key, a weights repo, and a template repo (base models use their
family's instruct template, as in core). The OLMo IDs come from `core/config.py`. `models.md` records the HF
API verification: sha, license, weight size, architecture, vocabulary match with the template tokenizer, the
derived turn markers and stop token, and the exact empty-condition prompt. Re-run it with
`uv run python -m exploration.shiller_user_turns.models`.

| key | repo | group |
|---|---|---|
| olmo3-7b-instruct / olmo3-7b-base | `allenai/Olmo-3-7B-Instruct` / `allenai/Olmo-3-1025-7B` | small |
| qwen3-8b / qwen3-8b-base | `Qwen/Qwen3-8B` / `Qwen/Qwen3-8B-Base` | small |
| gemma4-e4b-it / gemma4-e4b-base | `google/gemma-4-E4B-it` / `google/gemma-4-E4B` | small |
| gemma4-31b-it / gemma4-31b-base | `google/gemma-4-31B-it` / `google/gemma-4-31B` | large |
| qwen3-32b | `Qwen/Qwen3-32B` (no public base) | large |

The large models need about 65 GB in bf16. With `--precision auto` (the default), they load in 8-bit
(bitsandbytes LLM.int8) when bf16 doesn't fit on the GPU. Every row and meta file records the precision.
Contamination can't be checked for Gemma or Qwen.

## Rendering

Each prompt is rendered with the model's own chat template (template repo, default options) as a single-turn
conversation ending in a user message. Everything after the prefill is then removed: no end-of-turn token,
no generation prompt. `render.py` checks that the template places the text verbatim, that decoding the ids
reproduces the rendered string ending exactly in the prefill, and that the prefill tokens don't merge with
the header. The exact rendered prompt goes into every row and into `<model>_meta.json`.

Template options are recorded, not changed. OLMo 3's template inserts its default system turn ("You are a
helpful function-calling AI assistant…") even for a user-only conversation. Core does the same, and it is
kept. Gemma 4's `enable_thinking=True` would add a `<|think|>` system turn, and it is off by default.
Qwen3's `enable_thinking` doesn't change a conversation that ends in a user turn. `models.md` lists, per
model, whether each probed option would change the prompt.

## Generation

Temperature 0.7, top_p 1.0, no top-k (each model's own generation config is replaced, not merged). 5 samples
per prompt, up to 200 new tokens. The seed is `seed + crc32(prompt id)`, the same for every model. Sampling
uses bf16 on GPU (or int8 for the large models), fp32 on CPU. It stops only at end of text (`<|endoftext|>`
for OLMo/Qwen, `<eos>` for Gemma), never at end of turn, so closed turns and handoffs stay visible.

## Flags

Definitions are in `flags.py`. They reuse `ASSISTANT_REGISTER` and `repetition_loop` from
`measures/vestigial_regression/flags.py`, and use turn markers derived from each template
(`<|im_end|>`/`<|im_start|>assistant` for ChatML, `<turn|>`/`<|turn>model` for Gemma 4). The *user portion*
is the completion before the first end-of-turn, turn-open or stop token.

- `ended_turn`, `handoff_to_assistant`, `new_user_turn`: turn structure.
- `assistant_register`: assistant phrases in the user portion.
- `assistant_register_unmarked`: assistant phrases before any turn marker, special token or plain-text
  label ("Assistant:"). This is Shiller's "switch without turn markers".
- `plain_role_marker`: a plain-text assistant speaker label.
- `assistant_thinking`: reasoning about "the user", like Shiller's Qwen observation.
- `special_token`: e.g. `<think>` in the user portion.
- `repetition_loop`: a word 4-gram repeated 3+ times.
- `token_loop`: one token id 10+ times in a row, which catches single-token loops without spaces.

All flags are regex heuristics. `hand_labels.csv` is there to check them.

## How to run

```bash
uv sync --extra quant                                                     # bitsandbytes for 8-bit
uv run pytest tests/test_shiller_user_turns.py                           # CPU: templates, flags, tiny end-to-end
uv run python -m exploration.shiller_user_turns.run                      # small models (default --models small)
uv run python -m exploration.shiller_user_turns.run --models gemma4-31b-it        # 8-bit on a 48 GB card
uv run python -m exploration.shiller_user_turns.run --models "qwen3-32b gemma4-31b-base"
uv run python -m exploration.shiller_user_turns.run --models all --skip-existing  # everything not yet done
```

`run` verifies every requested model through the HF API before downloading anything. Then, per model, it
checks free disk against the weight size, generates in a separate process, and deletes from the HF cache the
repos this run downloaded. `--clear-cache` is `large` by default; use `all` on small disks. Repos that were
cached before the run are never deleted. A failed model (for example, too little GPU memory or disk) is
reported and skipped. The report is rebuilt from every `<model>.jsonl` present. Extra flags (`--samples`,
`--max-new-tokens`, `--limit`, `--seed`) are passed through to `generate`.

Other entry points:

- `generate --model KEY [--precision int8]`: one model.
- `report`: gallery, rates and the hand-label sheet.
- `report reflag`: recompute flags in place after editing `flags.py`. No GPU is needed: completions, token
  ids and markers are saved.
- `report agreement`: flags vs hand labels.

## Outputs: `results/exploration/shiller_user_turns/`

- `<model>.jsonl`: one row per sample with model, repo, role, precision, condition, prompt_id, sample, seed,
  prefill, rendered_prompt, completion (special tokens kept), completion_ids, stopped_on_eos, user_portion,
  and the flags.
- `<model>_meta.json`: the verified model record (sha, template sha, markers, stop token), precision, GPU,
  generation settings, template options, every rendered prompt, and library versions.
- `gallery.md`: a flag-rate table by model × condition, then for each condition a few prompts with every
  model's completion one after another.
- `flag_rates.csv`: the same rates.
- `hand_labels.csv`: 50 random completions (seed 0) with empty `perspective` and `notes` columns. Flags are
  left out so labeling is blind. Label the text written in the user slot, before the turn closes, as one
  of `user voice`, `switched to assistant`, `degenerate`, `other`. A completion that finishes the user's
  message, closes the turn and is then answered by the assistant is still `user voice`. The sheet is drawn
  once over the models present. It is never overwritten once a label is filled in, and `--redraw` replaces
  an unlabeled sheet, e.g. after more models ran. `report agreement` writes `agreement.md`.
