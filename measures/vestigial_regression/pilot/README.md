# Pilot: do the flags catch clear vestigial regression? (Gemma 4 31B-it)

**Exploratory, n = 30 + 18.** Outside every verdict. Does not touch the measure's code, README or
pre-registered criteria; imports `ASSISTANT_REGISTER` and `repetition_loop` from `../flags.py` unchanged.

**Question.** The Gemma open-user-turn samples (`results/exploration/shiller_user_turns/gemma4-31b-it.jsonl`)
are, by eye, mostly vestigial regression: the model loops or answers as the assistant instead of continuing the
user. Does the measure's current generation-flag approach, plus the README's criterion 4 (a spec addition, not
implemented in the measure), class such samples as vestigial regression?

**Method.**
- `labels.csv`: 30 random samples (`random.Random(1)` shuffle of the 205 rows), labelled `degenerate` or
  `switched to assistant` by the agent that wrote this pilot, reading the text only (flags were not computed
  until afterwards). **A teammate should check these labels.**
- `run.py`: recomputes `assistant_register` and `repetition_loop` on the stored user portion (cut at Gemma's
  end-of-turn / turn-open / stop tokens); `ended_turn` and `handoff_to_assistant` come from the exploration run.
  Criterion 4: `repetition_loop` or `handoff` → vestigial; `assistant_register` without handoff →
  overgeneralization; else unclassified.
- Control: the text after the prefill sentence of 18 real WildChat user turns (`data/prefill.jsonl` →
  `data/texts.jsonl`; 2 of the 20 had under 50 chars left), first 600 chars. There are **no** user-voice
  samples among Gemma's outputs to serve as negatives.

**Result** (full table and misses: `results.md`, per sample: `per_sample.csv`).

| | n | classed vestigial |
|---|---|---|
| degenerate loops | 8 | 8 (100%) |
| switched to assistant | 22 | 3 (14%); 7 classed overgeneralization, 12 unclassified |
| real user text (control) | 18 | 0; 2 classed overgeneralization |

**Reading.**
1. Word-level loops are caught (`repetition_loop`). Loops without spaces (`////`, `KNDKND…`) are not, and the
   exploration's `token_loop` flag is not part of the measure; none happened to fall in this labelled sample.
2. Assistant switches are mostly missed. Gemma starts them with `<|channel>thought\n<channel|>`, which is
   neither an end-of-turn nor a turn-open marker, so the "user portion" runs into the assistant answer, and
   `handoff_to_assistant` is 0 for all 30. The register regex only matches stock phrases ("I'd be happy to"),
   so most answers (`The answer is a **Green Apple**`) match nothing. 12 of 22 are unclassified.
3. Criterion 4 as written would send 7 of the 22 to overgeneralization. By eye these are the same switch
   into the assistant, not a user voice with assistant traits. Whether the rule should distinguish them is a
   spec question for the owner.
4. The control is not clean: 2 of 18 real turns contain "As an AI" in prompt-engineering text.

**Run.** `uv run python -m measures.vestigial_regression.pilot.run` from the repo root (no GPU).

## Second experiment: the README's own sampling arm, run fresh on Gemma 4 31B-it

GPU run, 2026-11-02: the 20 `data/prefill.jsonl` sentences × 3 samples × 120 new tokens, T=0.7, seed 0, stopping
only at end of text, int8 (the 62 GB model does not fit this 48 GB GPU in bf16), sampled with
`exploration.shiller_user_turns.generate --conditions ordinary` into `gemma4-31b-it/` (rows and meta). No core
or measure code was changed. `readme_arm.py` applies the same flags and criterion 4 as above;
`labels_readme_arm.csv` holds my labels, made on shuffled output before the flags were computed
(**a teammate should check them**). Results: `readme_arm.md`, `readme_arm.csv`.

| label (by eye) | n | classed vestigial | overgeneralization | unclassified |
|---|---|---|---|---|
| degenerate loop | 17 | 17 | 0 | 0 |
| switched to assistant | 43 | 0 | 14 | 29 |

README criterion 3 counts out of 60: `ended_turn` 15, `handoff_to_assistant` 0, `assistant_register` 14,
`repetition_loop` 17.

- All 60 samples are vestigial regression by eye, so the 17 the rule calls vestigial are all correct, but with no
  user-voice samples this says nothing about false alarms (the real-user-text control above does: 0 of 18).
- The rule finds only loops. None of the 43 assistant switches is classed vestigial, and 14 would be reported as
  overgeneralization instead. Same finding as the first experiment, on fresh samples.
- The NLL slot-effect arm is below.

## Third experiment: the README's NLL slot-effect arm on Gemma 4 31B-it

`nll_arm/score.py` scores the README's slot conditions (`user@user`, `user@asst`, `user@plain`, and the same for
assistant text) with core's own rendering and span-NLL code, in **fp32 with CPU offload** (14 of 66 modules on the
48 GB GPU, the rest streamed from RAM), so AGENTS.md's fp32 rule holds without an exception. First 30 texts of each
of wildchat user, oasst1 user and wildchat assistant (270 sequences, 611 s). `nll_arm/analyze.py` writes
`nll_arm/results.md`. **No Gemma base was scored** (62 GB, no disk): the reference is OLMo 3 7B base from the
frozen held-out run, which we take to show no vestigial regression. Differences are within-text, so raw NLL levels
(much higher for Gemma-it: 3.2 nats on plain encyclopedic prose) cancel, but the comparison is across models, unpaired.

| user text, change from OLMo base | slot effect `user@user − user@asst` | role-specific `user@user − user@plain` |
|---|---|---|
| wildchat | +0.98 [+0.77, +1.20] | +3.12 [+2.60, +3.62] |
| oasst1 | +1.11 [+0.85, +1.37] | +3.84 [+3.15, +4.51] |

(OLMo instruct for scale: slot effect −0.05, role-specific −0.41 / −0.33.) Both README directions hold for Gemma-it,
well past 0.05 nats: user text is about 1 nat *more* expected in the assistant slot than the user slot (the flip to
positive), and the role-specific term is strongly positive.

Checks and caveats:
- Offload path verified: fp32 offload and on-GPU int8 agree to 0.05–0.25 nats on 5 texts, and unpadded, padded and
  eager-attention scores of one text agree to 3e-4 nats.
- The assistant text shows a slot effect of the same size (+1.0), so part of this may be a general lowering of NLL
  in Gemma's assistant slot for any text, not user-specific. The README's asymmetry is in the role-specific terms
  (user +3.1/+3.8 vs assistant +1.3), which are larger for user text.
- No same-model base, so the README's control (`user@plain` drift) and "change from base" are not tested; n = 30 per
  set; no per-position analysis (the first-position header effect is only avoided by dropping position 0).
- A first scoring attempt at batch size 16 ran out of GPU memory (262k vocabulary); the run uses batch size 4.
