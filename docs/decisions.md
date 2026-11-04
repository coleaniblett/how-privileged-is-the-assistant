# Decision log

Dated entries, newest last. Log here: changes to `core/`, changes to any measure's pre-registered criteria
after its results exist, and project-level decisions. The first five entries were made before capstone week
and recorded on the day the repo was restructured.

## 2026-11-01: three measures, one owner each, on a shared pipeline

The pilot study of vestigial regression on OLMo 3 7B becomes the capstone of a team of three (Cole as lead,
Adeeb, Grace), studying whether post-training makes the assistant a privileged persona (Shiller, "Privilege,
Dominance, and Personas", Eleos, Aug 2026). Three of Shiller's forms of privilege are measured, each owned by
one teammate (owners TBD): vestigial regression (dominance), logit distortion (privilege), overgeneralization
(dominance). Everything that samples or scores models or prepares shared text sets is shared (`core/`); each
measure turns core outputs into its own statistics and verdict (`measures/<name>/`). Owners edit only their
folder; core changes must stay backwards-compatible and be logged here. Role swap and coin toss from the pilot
are kept working in `extras/`; whether they are in scope is undecided.

## 2026-11-01: H3 dropped

Hypotheses are H1 (thin mask, no privilege), H2 (privilege without dominance) and H4 (privilege with
dominance). There is no H3 because Shiller treats dominance as a form of privilege, so "dominance without
privilege" is not a coherent outcome. Decision rule: no slot-specific change beyond base on any measure → H1;
logit distortion only → H2; vestigial regression or overgeneralization → H4, reporting which form(s).

## 2026-11-01: logit distortion design

Calibration gap (mean NLL − mean next-token entropy) of user text in the user slot and assistant text in the
assistant slot, crossing persona with authorship (human-written: WildChat/oasst1 user turns, oasst1 assistant
replies; model-written: UltraChat-style user turns, WildChat ChatGPT replies). Tags are kept straight: the
headline analysis has no swapped conditions (assistant text in the user slot etc.), although core scores them.
Privilege = the assistant persona's gap grows from base more than the user persona's, regardless of
authorship; reported from content position 1 onward and plotted by position. Reason for no swaps: the
question is whether the assistant persona's own distribution is distorted, and swapped slots confound that
with the oddness of mislabeled text. Two new text sets are needed and must go through the same contamination
filter (TODO in `core/config.py`).

## 2026-11-01: breadth arm and model options

Primary models: OLMo 3 7B across base, SFT, DPO, Instruct, the only family whose open post-training data
(Dolci) lets us filter test text for contamination. Breadth arm: base vs instruct pairs from other families,
Gemma 4 31B plus Qwen3-14B or Qwen3-30B-A3B if we get 80 GB GPUs, otherwise Gemma 4 E4B and Qwen3-8B.
Contamination cannot be checked for these families; every table and figure with their results says so. Qwen
published no base checkpoint for Qwen3-32B, so it is not an option. Not in the run config yet; `config.FAMILIES`
can hold them, and the Gemma 4 IDs and chat templates are to be verified with `core/check_models.py` first.

## 2026-11-01: smallest effect 0.05 nats

For NLL-based effects the smallest effect we count is 0.05 nats, together with a 95% bootstrap CI excluding
zero. This is the pilot's negligibility margin for the position analysis and the size of the bf16
batch-composition noise measured on OLMo 3 7B (which is why scoring is fp32). Smaller effects are reported as
negligible even when their CI excludes zero. `config.MIN_EFFECT_NATS`.

## 2026-11-01: core: repo restructured into core/, measures/, extras/

Pure refactor plus docs, no change to scoring, rendering, data selection or flag logic (the CPU smoke
pipeline produces byte-identical scores, samples, flags and tables before and after). Modules run as
`python -m core.score` etc. from the repo root. `core/generate.py` now writes raw samples only (with
`stopped_on_eos`); the vestigial-regression flags moved to `measures/vestigial_regression/flags.py`, which
writes the flagged samples and counts in the old format. `analyze.py` split: loaders, bootstrap CIs,
per-position arrays and the negligibility rule in `core/stats.py`, plotting setup in `core/plots.py`, the VR
quantities, tables and figures in `measures/vestigial_regression/analyze.py`. `config.FAMILIES` holds model
families with their own stage lists (only OLMo 3 7B so far). New outputs go to `results/<run>/` (core/,
<measure>/, extras/); `results/`, `results_preceding/` and `results_v0/` are frozen pilot runs.

## 2026-11-01: exploratory open-user-turn samples (outside every verdict)

`exploration/shiller_user_turns/` qualitatively reproduces the user-turn observations of Shiller's "Cursory
Exploration" on OLMo 3 7B, Qwen3-8B/32B and Gemma 4 E4B/31B (instruct, plus base where one exists). It is
exploratory: no pre-registered criteria, and no measure reads it. Its model list lives in its own `models.py`,
not in `core/config.py`, so that the measures' model list stays unchanged (the OLMo IDs come from config).
Nothing in `core/` or `measures/` changed. `pyproject.toml` gains an optional `quant` extra (bitsandbytes,
`uv sync --extra quant`) for 8-bit loading of the 31B/32B models on 48 GB cards. A plain `uv sync` is
unaffected.

## 2026-11-02: vestigial regression: Gemma 4 31B as positive control for the mean-NLL proxy

Shiller's definition of vestigial regression is behavioral (what the model writes when continuing a user turn),
while the measure's primary quantity is a teacher-forced proxy: mean per-token NLL of real user text in the user
slot relative to plain prose and to base (`user@user − user@plain`). Two reasons it might miss the behavior:
teacher forcing cannot capture drift that compounds over a sample, and a mean over every token dilutes the few
decision points (end the turn, switch register) where the collapse happens. Gemma 4 31B IT reliably shows
vestigial regression in generation (`results/exploration/shiller_user_turns/gemma4-31b-it.jsonl`), so it is used
as a positive control. Pre-committed reading: if mean NLL rises clearly from Gemma base to IT, the proxy is valid
and is used for OLMo's stage-by-stage analysis; if it barely moves, the proxy fails for the behavioral definition
and is demoted to descriptive. The generation arm is done; NLL scoring is running; a Gemma base run is still
needed before the result can be read. "Clearly" and "barely" have no numeric threshold yet.

## 2026-11-02: vestigial regression: decision-point likelihood (exploratory)

Designed, not implemented: likelihood at the positions where the behavior is decided. Turn-end hazard (probability
of the end-of-turn token where the real user kept going, split into sentence boundaries and mid-sentence),
survival probability per text (product of 1 − hazard), calibrated against the rate at which real WildChat users
end their message at each boundary type, plus the probability of assistant openers at boundaries (the switch into
assistant register without a turn marker). Exploratory because it was designed after seeing Gemma's generations;
its OLMo criteria must be written in the measure README before any OLMo analysis with it.

## 2026-11-02: vestigial regression: LLM judge as the fallback generation measure

If the likelihood proxies fail, the generation measure becomes primary: an LLM judge classifies each sample's
perspective (user voice / switched to assistant / degenerate / user voice with assistant traits), with the regex
kept for `repetition_loop`. The judge is validated against human labels
(`results/exploration/shiller_user_turns/hand_labels.csv`, not yet filled in) and its agreement rate reported.
Caveat: the judge is itself an assistant deciding what counts as "user voice."

## 2026-11-02: vestigial regression: decision sequence

Mean-NLL proxy → decision-point likelihood if the Gemma validation fails it → LLM judge plus regex if that fails
too. Recorded in `measures/vestigial_regression/README.md` ("Measurement approaches and their validation").

## 2026-11-02: core: Gemma assistant slot opens with the empty thought channel

`core/render.py` gains `assistant_opening_extra(tok)` and a `template_parts(..., opening=True)` argument. The
extra is whatever the template's generation prompt adds after the assistant header beyond how it renders a stored
assistant turn, measured after a neutral user turn. It is empty for OLMo 3 and for Qwen3 at its default thinking
setting; for Gemma 4 it is `<|channel>thought\n<channel|>` (the template's generation prompt is
`<|turn>model\n<|channel>thought\n<channel|>`, and the model writes those tokens itself at the start of every
sample). Without it the assistant slot is off-distribution: the first content token costs about 30 nats, because the
model puts nearly all probability on `<|channel>`. Backwards compatible: existing callers and the user and plain
conditions are unchanged. OLMo rendering is byte-identical: a hash over all 1190 texts in every slot, with and
without the preceding turn, plus the template parts, is the same before and after
(`94c0f514db352ef8d6c44150c68a4df5617c7c68e0d8b2ea9eabb19d7e50bc28`); `tests/test_render_gemma.py` is new and the
existing render and smoke tests pass. Effect on the Gemma 4 31B IT pilot (90 texts): assistant-slot NLL falls from
9.2 to 4.5 nats/token (wildchat user text) and from 6.6 to 3.2 (wildchat assistant text); user and plain
conditions are identical, so `user@user − user@plain` is unchanged. Scores made before the fix are kept in
`measures/vestigial_regression/pilot/nll_arm/scores_before_fix/`.

## 2026-11-02: vestigial regression: two-arm design locked; decision-point measure moved to future work

The measurement is two complementary arms (README, "Measurement design"): an **NLL arm** (capacity: can the model still
predict real user text; teacher-forced, graded, judge-free) and a **judge arm** (behavior: when it continues a user
turn, does it still play the user; the direct measure of Shiller's behavioral definition). Basis: the pilot
(`measures/vestigial_regression/pilot/report.md` with its checks section). Gemma 4 31B passed the positive control the
earlier entry set up (user-text role-specific term +2.94 / +3.63 nats from its own base, assistant term −2.17, base
flat, 0 of 60 IT samples in the user's voice), so the fallback chain in the earlier entries is not needed: the
**decision-point likelihood (turn-end hazard) measure is moved to Future work** and not planned for the capstone, and
the "decision sequence" and "LLM judge as the fallback generation measure" entries are superseded (the judge is now a
primary arm, not a fallback). The earlier positive-control entry's open question, that "clearly" and "barely" had no
numeric threshold, is resolved by the pilot: the rise is large and the NLL arm's verdict rule is below. Docs only; no
code, results or core changes. Nothing is frozen: no `results/<run>/vestigial_regression/` exists.

## 2026-11-02: vestigial regression: the asymmetry as the NLL arm's primary quantity (pilot-informed)

Primary quantity: `A = Δ(user@user − user@plain) − Δ(asst@asst − asst@plain)`, change from the model's own base, per
user source against the assistant set; nats per token within a family, bits per byte for cross-family comparison.
Why: the earlier single criterion (a rise of the user role-specific term of at least 0.05 nats) was met by the negative
control Qwen3-8B too (+0.44 / +0.39 nats), so it did not separate Qwen from Gemma, and the pilot's check 1 showed Qwen's
rise is a cost of chat framing shared by both slots and both kinds of text (`Δ(user@asst)` +0.51 / +0.35, assistant text
+0.42 / +0.40, plain nearly flat). Subtracting the assistant-text change removes that shared part. The CI resamples the user
and assistant text sets independently (different texts), pairing within each set across checkpoints. Verdict: A is at
least 0.05 nats with the 95% CI excluding zero, in the same direction in both user sources; 0.05 nats is a detectability
floor and the magnitude is the primary result, and no new threshold tuned on the pilot models is introduced (Qwen may
register as mildly regressed, consistent with its generations: 9 of 20 samples left the user's voice). Slot effects
stay descriptive. The plain-text change from base is reported as its own row, "non-assistant degradation" (Gemma IT
+1.58 bits per byte): plain prose is also a non-assistant context, the role-specific subtraction removes it, and NLL
cannot tell whether it is vestigial regression or overgeneralization, so it is referred to the overgeneralization
measure. These criteria were revised after seeing the OLMo, Qwen and Gemma pilot results, so **OLMo NLL verdicts are
not blind**. Not yet implemented: the asymmetry in `analyze.py`.

## 2026-11-02: vestigial regression: judge arm specification, validation bar, combining rule, agreement prediction

Judge arm (set before any judge result exists; designed, not built). Prefilled unterminated user turns (20 prefills now;
TODO at least 100, a core change to be logged), stopping only at end of text. An LLM judge labels each user portion as
user voice / switched to assistant / degenerate / user voice with assistant traits; one fixed judge model, exact prompt
and temperature 0, recorded; the judge must not be an evaluated model (TODO: choose it and commit the prompt). Regex
stays for `repetition_loop` and a no-space token-loop flag (TODO: port `token_loop` from the exploration), with
family-specific handoff markers (turn markers, Gemma's `<|channel>`, Qwen's `<think>`). Primary quantity: the
non-user-voice rate (switched + degenerate), change from base, with a 95% bootstrap CI resampling prefills; the arm
fires if it rises with the CI excluding zero. OPEN: the minimum rate difference. Category 4 goes to the
overgeneralization measure. Validation before use: at least 50 human-labeled samples spanning Gemma, Qwen and OLMo
including base models, labels checked by a teammate (the pilot's labels were written by the agent that ran the pilot);
report raw agreement and Cohen's kappa; default bar, changeable by the team before running: revise the judge prompt if
agreement is below 80%. Combining rule: vestigial regression counts toward H4 if either arm fires, reporting which
(both: capacity loss with behavioral collapse; NLL only: worse at predicting users without visibly failing to play them;
judge only: fails to play the user while still predicting user text well); the form comes from the judge categories.
Pre-registered prediction: the arms agree in ordering across all model/stage points (OLMo base, SFT, DPO, Instruct;
Qwen base and IT; Gemma base and IT); report a Spearman correlation between the NLL asymmetry and the judge's
non-user-voice rate, descriptive and not a significance test at about 8 points; disagreements are findings, not errors.
OPEN: both quantities are changes from base, so the three base points are zero by construction; whether they enter the
correlation is undecided.

## 2026-11-02: core: model families in config, Qwen3 think-block option, check_models per family

`core/config.py` now holds three families in `FAMILIES`: `olmo3-7b` (unchanged), `gemma4-31b`
(`google/gemma-4-31B`, `google/gemma-4-31B-it`) and `qwen3-8b` (`Qwen/Qwen3-8B-Base`, `Qwen/Qwen3-8B`). Each can carry
`dtype`, `offload`/`gpu_gib`, `batch_size`, `gen_precision`, `handoff_extra` and `assistant_think_block` (documented in
the file; defaults in `FAMILY_DEFAULTS`, read with `family_settings()`). The end-of-turn token, the user-to-assistant
handoff string and the assistant-slot opening are derived from each family's chat template, not stored
(`core.render.turn_markers`, `assistant_opening_extra`). `use_family(name)` / `family_from_argv()` switch the active
family (`MODELS`, `STAGES`, `TEMPLATE_MODEL`); `run_dir`/`core_dir`/`measure_dir` take an optional `family` and, when a
script was given `--family`, write to `results/<run>/<family>/...` (run defaults to `capstone`). Without `--family`
every constant and path is exactly what it was: the default family is still OLMo and the paths are unchanged.
New option `ASSISTANT_THINK_BLOCK` (Qwen3): when true the assistant slot opens with the empty
`<think>\n\n</think>\n\n` block the non-thinking generation prompt adds; default None/False, which is what the pilot
used. `core/check_models.py` takes `--family` (or `all`), verifies IDs, licenses, template and tokenizer agreement via the HF
API (config and tokenizer files only) and records what core derives from each template; `docs/models.md` has the three
verified families. Backwards compatible: the OLMo rendering hash (all 1190 texts, every slot, with and without the
preceding turn, plus the template parts) is unchanged (`94c0f514db352ef8d6c44150c68a4df5617c7c68e0d8b2ea9eabb19d7e50bc28`).

## 2026-11-02: core: family-aware scoring and generation, CPU offload, HF cache helper, run script

`core.score` and `core.generate` take `--family` (with `--run`; outputs go to `results/<run>/<family>/core/...`, run
defaulting to `capstone`) and read the family's defaults from `config.family_settings()`: scoring batch size, and for
generation the precision. `core.score` gains `--offload/--no-offload` and `--gpu-gib`: the pilot's CPU-offload path
(`device_map="auto"`, at most `gpu_gib` GiB of weights on the GPU, the rest in CPU RAM) ported from
`pilot/nll_arm/score.py`. fp32 stays the default and offload changes only where the weights live, not the arithmetic;
the Gemma 4 31B family sets it on (14 of its 66 modules fit on a 48 GB card at 30 GiB). `core.generate` gains
`--precision {bf16,int8}` (int8 via bitsandbytes for the 31B, which does not fit in bf16). New `core/hf_cache.py` (ported from
the pilot): `check` (disk vs the weights, skipped for a cached repo), `free` (blob-aware removal, since this machine
shares one `blobs/` store), `cached`. `scripts/run_gpu.sh` takes `FAMILY=...` (legacy OLMo layout and behavior when
unset), checks free disk before each checkpoint and, with `CLEAR_CACHE=auto`, removes a checkpoint of at least 20 GB from the
cache after it is used unless it was already cached when the script started. Backwards compatible: with no `--family` every
default, path and output is as before (OLMo rendering hash unchanged; `tests/test_families.py` pins it and tests the
helpers, `test_smoke.py` runs the CPU pipeline).

## 2026-11-02: core: a 100-sentence prefill set (data/prefill_100.jsonl)

`core.prepare_data --prefill-only --n-prefill 100 --prefill-out data/prefill_100.jsonl` builds the capstone prefill set from the
WildChat user turns already in `data/texts.jsonl` (not modified), with the same first-sentence rule and order as the
original 20 (the block in `main()` became `build_prefill()`). Those user turns already passed the length, render and
Dolci-contamination filters when `texts.jsonl` was drawn (the Dolci match is on the whole user turn), so the new file inherits
them, with no re-sampling. It has 100 distinct sentences from 100 distinct texts, each row keeps its `text_id`, and the first
20 rows are identical to the frozen `data/prefill.jsonl` (untouched). `config.PREFILL_PATH` (the default of `core.generate`)
now points at the new file; `config.PREFILL_PATH_20` names the frozen one, which the exploration's `ordinary` condition
(`exploration/shiller_user_turns/prompts.py`) is pinned to so it stays at 20 prompts. `tests/test_prefill.py` checks it.

## 2026-11-02: core: family_turn_markers; vestigial-regression flags across families, token_loop, extra handoff markers

`core.render.family_turn_markers(tok)` (new; `turn_markers` is unchanged) derives the end-of-turn token, turn-open marker, assistant header and
the handoff string (end-of-turn through assistant header) from a family's chat template by rendering, ported from the exploration. The
vestigial-regression flags (`measures/vestigial_regression/flags.py`) use it instead of ChatML strings: the user portion is cut at the
family's end-of-turn or turn-open marker or end-of-text. `handoff_to_assistant` also fires on the family's extra markers from
`config.FAMILIES[...]["handoff_extra"]`: Gemma `<|channel>` and `<channel|>`, Qwen3 `<think>` and `</think>`. The spec named `<think>` and
`<|channel>`; in the existing exploration samples Qwen3-8B never writes `<think>` (0 of 205 completions) but writes `</think>`
with no opener in 66, and 4 Gemma completions have `<channel|>` without `<|channel>`, so the closing tags are included. Because these markers
flag a switch into the assistant without turn markers, `handoff_to_assistant` is no longer a subset of `ended_turn`, and the extra markers
flag the sample without shortening the user portion (so `assistant_register` and `repetition_loop` still see the whole text before the
turn closes, as the pilot did). New flag `token_loop` (one token id repeated 10 or more times in the user portion; ported from the
exploration) uses the generated ids that `core.generate` now stores in each record as `continuation_ids` (an extra key; existing readers
ignore it), else re-tokenizes the user portion, which can merge a repeated character and miss a loop. `repetition_loop` is unchanged.
OLMo flags are unchanged except for the added `token_loop` column in `*_counts.csv`; `flags()` still returns its four original keys unless
token ids are passed. Tests: `tests/test_flags_families.py` (synthetic strings per family).

## 2026-11-02: vestigial regression: the judge implemented (measures/vestigial_regression/judge.py)

Implements the judge arm specified in the 2026-11-02 entry above. Judge: `claude-sonnet-5-5` via the Anthropic API at temperature 0; prompt
`judge_prompt.md` (version `v1`, recorded in every output row and in the cache key); output a strict JSON object with a label and a
one-sentence reason, a reply that does not parse is asked for once more and otherwise recorded as label `invalid` (never guessed).
The judge sees only the prefill and the continuation (including text after turn or channel markers); sample ids, models and stages are
carried to the output rows but not sent. Results are cached by hash of (model, prompt version, temperature, prefill, continuation) in
`judge_cache.jsonl`; transient API errors (429, 5xx, connection) are retried with exponential backoff; `--dry-run` prints a token
estimate (no prices are built in; pass `--price-in/--price-out` from the current price list). The validation sheet
(`judge_validation/sheet.csv`, 60 samples stratified over Gemma 4 31B IT, Qwen3-8B and OLMo base/SFT/DPO/Instruct generations that
already exist; the model/stage key is a separate file the labeler must not open) is for a teammate's independent labels;
`judge validate` reports raw agreement, Cohen's kappa and the confusion matrix. `judge pilot` compares the judge with the pilot agent's
labels as a provisional check only. The `anthropic` package is an optional extra (`uv sync --extra judge`). No `ANTHROPIC_API_KEY` was
available when this was built, so nothing has been sent to the API: the tests use a fake client.

## 2026-11-02: vestigial regression: the locked NLL-arm and judge-arm quantities implemented (analyze.py --locked)

`python -m measures.vestigial_regression.analyze --locked [--run R] [--families ...] [--first N]` computes the README's locked design:
per family and post-trained stage, the change from the family's own base of every raw slot (the plain-text row is labelled
"non-assistant degradation"), the two role-specific terms and the asymmetry `Δ(user role-specific) − Δ(assistant role-specific)` per
user source, in nats per token and bits per byte (primary: summed NLL over positions 1+ / ln 2 / bytes of the text decoded from content
tokens 1+), with the user and assistant text sets bootstrapped independently and pairing within each set across checkpoints (5000
resamples). The NLL arm fires for a point when the asymmetry in nats is at least 0.05 with a CI excluding zero, in the same direction in
both user sources. The judge arm (if `<run>/<family>/vestigial_regression/judge/<stage>.jsonl` exist) reports the non-user-voice rate
(switched + degenerate) change from base with a bootstrap over prefills, and the category-4 rate separately; the arms are combined as in
the README; the Spearman correlation across post-trained points uses one NLL asymmetry per point (bits per byte, mean of the two user
sources) and excludes base points. `--locked --pilot` runs it on the pilot's scores and reproduces `pilot/report.md` (tests pin it);
the output is in `pilot/locked/`. On the pilot data the NLL arm fires for Gemma 4 31B IT (asymmetry +5.1 / +5.8 nats, CIs well clear of
zero), not for Qwen3-8B (+0.14 [−0.07, +0.35] and +0.09 [−0.08, +0.26]: above the floor, CIs include zero) and not for any OLMo stage.
The default (no `--locked`) pipeline is unchanged; it now also accepts `--family`/`--run`.

## 2026-11-02: core: text sets for logit distortion's authorship cross (data/texts_authorship.jsonl)

`python -m core.prepare_data --authorship` writes `data/texts_authorship.jsonl` (new file; `texts.jsonl`, `prefill*.jsonl` and every other file
under `data/` are untouched): 300 oasst1 assistant replies (human-written; the top-ranked reply of each root prompt, same selection as
`oasst_pairs`), and 300 UltraChat first user turns plus the 300 paired first assistant replies (model-written; `openbmb/UltraChat`,
the conversation's first pair; `config.DATASET_IDS["ultrachat"]`). Same filters as the existing sets via the same `sample()`: seed 0,
20-300 tokens by the OLMo tokenizer, clean rendering in every slot with and without the preceding turn, and the Dolci contamination
filter on the conversation's user turn (the assistant text is not matched separately, as for `wildchat assistant`). Filter statistics:
oasst1 1124 candidates, 702 in Dolci, 122 failed length/render, 300 kept; UltraChat 1254 candidates, 0 in Dolci, 954 failed length/render
(UltraChat replies are mostly longer than 300 tokens, so the kept pairs skew short: median 164 tokens for replies, 65 for user turns),
300 kept. Rows have the same fields as `texts.jsonl` (sources `oasst1` kind `assistant`, `ultrachat` kinds `user` and `assistant`;
`core.stats.SOURCE_ORDER` lists `ultrachat`). Core scores them in every slot with no code change:
`python -m core.score --family F --model STAGE --dataset data/texts_authorship.jsonl --out ...` (`tests/test_authorship.py` checks the file
and a CPU scoring run). No logit-distortion analysis was written.

## 2026-11-02: core: core.score --n-per-set; run_gpu.sh SCORE_ARGS

`core.score --n-per-set N` scores only the first N texts of each (source, kind) set of the dataset, in file order (default: all; the
option is additive and `--limit` is unchanged). `scripts/run_gpu.sh` passes `SCORE_ARGS` to `core.score` only (for example
`SCORE_ARGS="--n-per-set 100"`). Motivation: Gemma 4 31B in fp32 with CPU offload scores about 270 sequences in 10 minutes, so all 1190
texts in three slots would take about 2.3 hours per checkpoint; the analysis takes the matching `--first N` so every family is compared on
the same texts (`analyze.py --locked --first 100`). `tests/test_families.py::test_first_n_per_set` covers the helper.

## 2026-11-02: docs: vestigial-regression README status brought up to date; runbook for 2026-11-03

`measures/vestigial_regression/README.md`: Status, the judge-arm and criteria text and the TODO list now say what exists after the
pipeline preparation (the asymmetry in `analyze.py --locked`, the judge and its prompt, `token_loop` and family handoff markers, the
100-sentence prefill set, Gemma 4 31B and Qwen3-8B in core) and what does not (no capstone data collected, the judge never run against the
API or validated, the Qwen3 assistant-slot decision). No criterion was changed except that the Spearman correlation's base points,
previously an open decision, are excluded (they are zero by construction) with one NLL asymmetry per post-trained point; the README says
this was decided at implementation time and is not part of the pre-registered design. `docs/runbook_2026-11-03.md` has the launch commands,
estimates, decisions for Cole and what was blocked or skipped.

## 2026-11-02: core: continuation_ids in core.generate is opt-in (--save-ids)

Correction to the entry on `family_turn_markers` above: `core.generate` first stored the generated token ids in every record, which added a
key to its output format (`tests/test_smoke.py` pins the keys). It is now opt-in: `--save-ids` adds `continuation_ids`; without it the records
are exactly `prefill_id, sample, prefill, continuation, stopped_on_eos` as before. `scripts/run_gpu.sh` passes `--save-ids` when `FAMILY` is set
(the legacy OLMo layout keeps the old format). `flags.py` uses the ids when present and re-tokenizes the user portion otherwise (which can miss a
no-space loop that merges into fewer tokens). The measure's own `*_counts.csv` gained a `token_loop` column; `test_smoke.py`'s pinned column list
was updated for it.
## 2026-11-02: core: separate original-conversation candidates for logit distortion

Added `core.build_conversations`, source adapters in `core.conversations`, and `core.conversation_audit`.
This is an alternative text route for the measure, added alongside (not replacing) the isolated authorship sets
merged earlier today (`core.prepare_data --authorship`, `data/texts_authorship.jsonl`), which are unchanged. Which
route logit distortion uses is an open decision for Grace with Cole (STATUS.md); no measure outcome criteria or
analysis code changed. OASST1 uses one seeded, continuous
root-to-leaf branch with actual ancestors; WildChat and original UltraChat retain their existing exchanges.
All conversations have at least two completed exchanges. Separate target records preserve exactly the
preceding original messages; ineligible intermediate turns remain available as context. Authorship labels
retain source-specific bases, especially WildChat's human-submitted assumption.

New candidate output is `candidates/logit_distortion/pilot_v1/`, not legacy `data/` or frozen results. The
bounded pilot uses 300 inspected candidates per source and 8 selected conversations per source. Its
20–300 target-token / 4096 preceding-context-token limits are provisional inspection settings, not final
matching criteria. Sources remain separate, and windows are explicitly not uniform corpus samples.
The pilot is contamination-unchecked. `check_contamination` gains an optional revision map for its existing
Dolci loader and an additive auditable matcher covering both target roles, conversation hashes, and context
messages. Existing calls/flags/output formats remain compatible. No model weights, scoring, or generation
are required by the new builder. Original-context scoring integration is a subsequent task; the versioned
candidate format is deliberately distinct from `core.score`'s isolated-turn input, and `core.score` now exits with an
error on any input row carrying `format=conversation_candidates_v1` (`config.CONVERSATION_FORMAT`) instead of
scoring it without its history. Legacy rows have no `format` field, so existing inputs, flags and outputs are
unchanged. The branch was written at 16:35 UTC on the pre-family `main` and integrated onto `main` after the
family and authorship commits; the pilot was rebuilt from the integrated code and compared with the original build.

See `docs/conversation_candidates.md` for schema verification, sampling, exact reproduction/audit/test
commands, limitations, and the source-attributed bounded sample. Final sizes, source composition, length
matching, and contamination policy must be decided from the audit before examining model outcomes.

## 2026-11-02: logit distortion uses original conversations (primary route); core: Dolci-checked candidate pilot

**Decision (Grace, measure owner):** logit distortion's primary data are existing, original conversations
with at least two complete exchanges from OASST1, WildChat and UltraChat; each target keeps its exact original
preceding history. No new dataset source. Role × authorship: OASST1 human/human, WildChat human-submitted user
(a provenance label, not proof of composition) / model assistant, UltraChat model/model; sources stay separate.
The isolated authorship sets (`data/texts_authorship.jsonl`) and `data/texts.jsonl` are unchanged and remain
available for a separate comparison. This supersedes "which route ... is an open decision" in the entry below.
No hypothesis threshold, verdict rule or pre-registered criterion changed.

Why the isolated files cannot simply be joined into conversations: verified today, all 300 OASST1 `pair_id`s
occur in both `data/texts.jsonl` (user) and `data/texts_authorship.jsonl` (assistant), and 294 of the 300 pairs
have different `source_id`s; `pair_id` reflects sampling order in each file. The legacy sampler also checked
only user text against Dolci when it kept an assistant reply. Both are left as they are (legacy data unchanged).

`core:` changes, all backwards-compatible:
- `check_contamination.contamination_record` reports `incomplete` (instead of `no_exact_match`) when no hit was
  found but not every Dolci set was indexed. `dolci_turns`, `is_contaminated` and the CLI are unchanged.
- Conversation targets gain `context_contamination_status` (history summarized separately from the target);
  the audit reports it when present, so `pilot_v1` still verifies unchanged.
- The refusal of conversation-candidate rows moved to `core.render.reject_conversation_candidates` and now also
  guards `core.render`, `core.add_replies` and `core.check_contamination` (before any model or Dolci load);
  `core.score` behaves as before and still exports the name. Legacy rows have no `format` field and pass.
- `config.CONVERSATION_CANDIDATES_DIR` now names `pilot_v2`.

Pilot: `candidates/logit_distortion/pilot_v2/` is `pilot_v1`'s sample rebuilt with `--check-dolci` (pinned
SFT/DPO/RL revisions). Exclusion policy, unchanged from the builder: a target is ineligible if its own text or
any preceding message has a whitespace-normalized exact match (or the WildChat hash is in Dolci-RL); matched
messages are never removed from histories. Result: 8 of 184 messages matched, all WildChat user turns (7 of 8
WildChat conversations); eligible targets fall from 128 to 92, WildChat from 41 to 5. Reported as a shortage;
filters were not relaxed. How to sample WildChat (e.g. a conversation-level Dolci filter before the budget) and
final sizes/thresholds are Grace's decisions before scoring (`pilot_v2/inspection.md`). Exact matching does not
cover paraphrase, substrings, RL completions or pretraining data.

## 2026-11-02: core: conversation candidates: prediction-case deduplication, eligible-quota selection, length-sensitivity audit (pilot_v3)

Review findings on the original-conversation builder, assessed and adopted (logit distortion, Grace):

1. **Deduplication.** The old rule excluded any repeat of a normalized target text, across sources, roles and
   histories, with the first occurrence canonical even when ineligible. For a context experiment the same
   words after a different history are a different prediction case. Now a duplicate is the same source, role,
   authorship, normalized history and normalized target (`duplicate_case`); the representative is the first
   otherwise-eligible member in selection order. Repeated wording elsewhere only sets `repeated_text_of`.
   pilot_v1/v2 had no duplicate exclusions, so their counts are unaffected; their records are kept as built.
2. **Coverage.** pilot_v2's raw budget left WildChat with 5 eligible targets, all from one conversation.
   New `--selection-mode eligible_quota` walks the same seeded order and keeps only contributing
   conversations (>= 1 eligible target with fully clear target and history Dolci checks) up to a per-source
   quota within a scan limit; it requires a complete Dolci index, unchecked/incomplete never count, shortfalls
   are reported. `raw_budget` stays the default. The target-and-history contamination rule is unchanged.
   Verified: exempting "Give me more examples." recovers 0 pilot_v2 targets (an earlier message had matched).
3. **Length and position.** New data-only `length_sensitivity.json/.md` for minimum lengths 5/10/20 on the same
   conversations. The default stays 20 tokens; no primary threshold was changed.

Backwards compatibility: `make_targets` keeps its signature (now `conversation_targets` + `apply_duplicates`);
new Settings fields have defaults, so old manifests load; audit additions are gated by `audit_version` (2 for new
builds), and pilot_v1/v2 still verify byte for byte. Legacy isolated data and `core.score` are unchanged.

pilot_v3 (`candidates/logit_distortion/pilot_v3/`): quota 8, scan limit 200, 1,000-row windows, otherwise the
v2 settings and revisions. Quota met for all sources (WildChat after evaluating 42; 34 rejected, all with a
Dolci match): 24 conversations, 175 targets, 119 eligible; 0 duplicate cases. Selection bias: WildChat is
restricted to conversations without a Dolci match. At 20 tokens no UltraChat opening user is eligible (all
8 are 9–17 tokens); 7 at 10, 8 at 5. Recommendation, not applied: stratify role comparisons by opening vs
follow-up and choose the minimum (10 is a candidate) from this audit before outcomes. "As an AI" replies kept.
No hypothesis threshold, verdict rule or pre-registered criterion changed.

## 2026-11-02: core: original-conversation scoring adapter (core.score_conversations)

Logit distortion (Grace). New entry point `core.score_conversations` and helper `core.render.render_target`; the
isolated-text route (`core.score`, `render`, `render_conversation`) and other measures are unchanged.

- **One sequence per target, original prefix only.** A target is rendered as its original preceding messages plus
  the target, nothing after it. Required because Qwen3's template renders an assistant turn differently when a
  later user message exists (verified on the cached template: 45 of 131 targets differed between full-conversation
  and cut-off rendering); causal attention cannot undo that. Scored with `core.score.spans_nll` unchanged (fp32,
  teacher forcing, full-vocabulary entropy in nats); no text is generated.
- **Rendering settings (Grace's selection), explicit per family and verified on every target:** OLMo keeps the
  instruct template's default system prompt when a conversation has none (an original system message replaces it,
  never duplicated); Qwen3 assistant targets use the non-thinking generation prompt with the empty think block
  (`enable_thinking=False`, recorded, `config` defaults untouched); Gemma 4 assistant targets open with the empty
  thought channel of the generation prompt; earlier assistant turns stay in stored form; user targets get no
  assistant opening.
- **Whitespace and exactness.** Original text is never trimmed. Edge whitespace is allowed on this route only, with
  strict checks (verbatim placement, decode, no boundary merges, no special-token text). Gemma's template trims edge
  whitespace, so affected targets are excluded with the changed message recorded (pilot_v3: 17 of 119; OLMo and Qwen
  keep all 119).
- **Lengths and comparability.** Full rendered inputs must fit the minimum `max_position_embeddings` over the family's
  checkpoints (cached configs; never guessed; nothing truncated). An input fingerprint (candidate hashes, tokenizer
  revision, template hash, settings, target set, exclusions, per-target input hashes and spans) is shared by all
  stages of a family; `--check-stages` refuses mixed inputs. Outputs: per-token parquet and per-target CSV (means,
  NLL − entropy gap, positions-1-onward variants), under `results/logit_conversations/<family>/core/conversation_scores/`.

Validated with cached tokenizers and SmolLM2-135M on CPU only; no OLMo/Qwen/Gemma checkpoint was scored. No
hypothesis threshold, verdict rule or pre-registered criterion changed. Spec: `docs/conversation_scoring.md`.

## 2026-11-02: Qwen3 assistant-slot rendering: think block ON is primary, OFF the robustness check

Decided by Cole before the overnight collection run: Qwen3-8B is scored under both assistant-slot renderings. The empty think block
(`<think>\n\n</think>\n\n` after the assistant header, what Qwen3's non-thinking generation prompt adds, the analogue of Gemma's thought
channel) is ON for the primary result, **pre-declared**; OFF (the template's stored-turn rendering, what the pilot used) is the robustness check.
The existing config option is the mechanism: `config.FAMILIES["qwen3-8b"]["assistant_think_block"]` is now `True` (it was `False`), so
every default Qwen3 scoring uses the primary rendering; the OFF scores are made by setting the option to `False` programmatically for those
processes (no code change). Outputs are kept apart: ON in `results/capstone/qwen3-8b/`, OFF in `results/capstone_think_off/qwen3-8b/`
(scores and authorship scores for both checkpoints; generation is identical under both renderings, because it uses only the user turn,
so it is run once, in the ON directory). Only the assistant slot changes, so `user@user`, `user@asst` and `user@plain` are the same
under both; the assistant role-specific term, and with it the vestigial-regression asymmetry, and logit distortion's `asst@asst` differ. The
primary result is the ON one; any difference between the two is reported as the robustness finding, not chosen after the fact.
`tests/test_families.py` now asserts the new default. OLMo and Gemma rendering are unchanged.

## 2026-11-03: core: incomplete-turn prefills for the judge arm (data/prefill_incomplete_200.jsonl)

The prefills in `data/prefill_100.jsonl` (first sentences of real WildChat user turns) all end in terminal punctuation, 35 of the 100 read as complete
questions or requests, and the real user kept writing after every one of them (median 299 characters). A chat model that closes the user turn there and
answers is behaving normally, so handoff after these prefills is not evidence of vestigial regression (capstone generations: OLMo SFT/DPO/Instruct
close the turn in 60-80% of samples, typically after about 100 characters of further user text; base models never hand off). The judge arm therefore
needs prefills that are clearly unfinished, as in Shiller's exploration. `python -m core.prepare_data --incomplete-prefill` writes
`data/prefill_incomplete_200.jsonl` (new file; the existing prefill files are untouched): 200 real WildChat user turns from `texts.jsonl` (already
length-, render- and Dolci-filtered), each cut inside the first line at a seeded random point where the last kept word is an article or
determiner, preposition, conjunction, auxiliary or modal, a transitive verb that nearly always takes an object, or ends with a comma (never terminal
punctuation, 6-40 words, more words following), choosing the cut type first so the set is not dominated by 'the' and 'of'. The word lists are closed
classes in `core.prepare_data.INCOMPLETE_ENDINGS` (no part-of-speech tagger is a dependency). Rows keep `text_id`, the cut type, the last word and
the first words of the real continuation; prefills are deduplicated, since WildChat repeats prompts. Cut types: article/determiner 51, preposition 54,
comma 24, auxiliary/modal 30, conjunction 22, transitive verb 19. `config.PREFILL_PATH` (the default of `core.generate`) is unchanged; the old
set stays as a separate "complete-turn" stratum. `tests/test_prefill_incomplete.py` covers it.

## 2026-11-03: vestigial regression: judge prompt v2 (the reply after the user's turn closes is context, not a switch)

Judge prompt v1 labelled a continuation `switched_to_assistant` if it answered "at any point", including after a closing turn marker. With sentence-boundary
prefills that counted ordinary chat behaviour (finish the user's message, close the turn, the assistant replies) as vestigial regression, and the README says
the judge labels the user portion. v2 (`measures/vestigial_regression/judge_prompt.md`) judges only what the model did before the user's turn closes and
treats what follows as context: finishing the user's thought as the user and then closing the turn is `user_voice`; abandoning a clearly unfinished sentence
(closing the turn at once, or answering, reasoning or opening a thinking section in its place) is `switched_to_assistant`; with a prefill that already reads
as a complete message, closing the turn is not a departure. The four labels, the JSON output, the model (`claude-sonnet-5-5`, temperature 0) and the inputs
are unchanged; the version is part of every output row and of the cache key. No v1 label exists (the judge has not yet called the API), so no result is
affected. The human labelling instructions (`judge_validation/README.md`) use the same definitions, and `judge sheet --from-capstone` builds the 60-sample
validation sheet from the incomplete-turn generations of all 8 checkpoints (base models included), so the teammate's labels match the data being graded.
The judge-arm prefill set is `data/prefill_incomplete_200.jsonl`; the older sentence-boundary set is a separate "complete-turn" stratum.

## 2026-11-03: core: frozen context limits for conversation scoring (`core.score_conversations --context-limits-from`)

Scoring a conversation stage re-renders the family's inputs and needs every stage's context limit, which `core.score_conversations` read from each
checkpoint's cached `config.json`. A collection that clears each checkpoint from the HF cache after use (as the overnight driver does) would make a
later stage depend on configs that are no longer cached. New opt-in flag `--context-limits-from MANIFEST`: take the per-stage limits recorded in a frozen
`inputs_manifest.json` (it must list exactly the family's stages) instead of reading configs. The input fingerprint check is unchanged, so a stage
scored this way still refuses to mix with different inputs. Default behaviour, outputs and the other flags are unchanged. Used by
`scripts/collect_conv_incomplete.py`, which first freezes each family's inputs (`--prepare-only`; an existing manifest must match) and records every stage's
`config.json` revision and hash in `config_provenance.json` beside the manifest. Test: `test_frozen_context_limits`.

## 2026-11-02: overgeneralization redesigned around persona carryover

- Primary measure is now persona carryover: does a belief held by the Assistant spill into a new role-played persona more than one held by a role-played character (leak gap, change from base)?
- The prose minimal-pair design is dropped and its code, items and passages deleted. Reason: summed multi-token log-probs, hand-matched comparators and 240 insertions to review made it noisy and costly. The claim is now belief spillover, not stylistic leakage.
- README criteria rewritten. The OLMo run on the current 36 items was already scored, so it is a pilot. The verdict needs a confirmatory run on a fresh item set (dice, drawer, match, weather), drafted and awaiting team review. Criteria are pre-registered for that run only.
- A "relative only" gap (assistant carryover not rising, character carryover falling) does not count. Thresholds confirmed: delta 0.25, domain effect 0.5 nats, 2 domains. No change to `core/`.

## 2026-11-02: bf16 scoring for the overgeneralization measure

- Decision: bf16 at batch size 1 is allowed for `persona_carryover.py` (`--dtype`, default bf16), because fp32 OLMo 3 7B does not fit the 24 GB RTX 4090. Follows the AGENTS.md rule. Pilot results stay fp32. Not core.
- Validation against fp32 not yet done: on the first GPU job, score one stage in bf16, compare with the pilot fp32 CSVs, and log it here. The effects are single-token log-odds against a 0.25-nat delta, so bf16 noise (up to 0.05 nats) should be small but is unmeasured.

## 2026-11-02: dominance-matrix pilot (extras, outside every verdict)

Fail-fast check: is harm aversion (an Assistant trait) visible when other reporters announce a chance outcome, relative to the Assistant reporting itself? Report `results/olmo3-7b/dominance_pilot.md`.
- Design: benign maths tasks vs harmful tasks, d6 outcome, four reporters (assistant turn, user turn, raw narration, raw log) with identical task text, four OLMo 3 7B stages, bf16 batch 1. Effects are change from base, two-way bootstrap over tasks. No pre-registered criteria. Stays in `extras/`; moving it to `measures/overgeneralization/` is the team's call if the checks pass.
- Format rule, fixed before running: among coin, d6 ranges and d6 odd/even, take formats with reporter-1 coverage >= 0.95, then the one with P(first) closest to 0.5. Result: d6 only (coverage 1.000 vs coin 0.446, parity 0.001; P(first) 0.589).
- 5 x 5 pilot: all four fail-fast checks pass, but CIs are coarse and every reporter has a nonzero base effect. A length-matched harmless control showed about a third of the base lean is shared with the control (narration -0.40 vs -0.13), so the rest is harm-specific. Control effects stay near zero after post-training, so stage effects are not length artifacts.
- 15 x 15 rerun (`results/olmo3-7b/dominance_pilot_n15.md`): CIs tighter. Check 2 (user-reported effect < assistant-reported) now FAILS: +0.53 vs +0.50, ratio 1.06 [0.83, 1.38], harm-specific 0.83 [0.65, 1.02]. The 5 x 5 pass on check 2 was within noise. The log reporter's effect is reliably negative (reverse sign, not a fade).

## 2026-11-02: stakeholder-swap test of the dominance-matrix inference (extras; fail-fast)

Does harm aversion depend on whose task the outcome decides? Stakeholder A is the Assistant (existing prompts). Stakeholder B is the user: two clauses swapped so the user does the task and the Assistant is not involved. B's opening is about 8 tokens longer. Report `results/olmo3-7b/stakeholder_swap.md`.
- **Kill rule, fixed before running**: (A effect - B effect) at instruct minus the same contrast at base, user reporter, paired per task cell, 10,000-resample two-way bootstrap, applied to the raw effects. CI excludes 0 and A > B: harm aversion is Assistant-specific. Otherwise the results do not support Assistant privilege. The control-subtracted version is reported beside it; if they disagree, say so, do not resolve afterwards. Any shift > 0.05 nats between fp32 and bf16 on one cell is flagged.
- Result: raw contrast +0.059 [-0.008, +0.126], CI includes 0, so **the dominance-matrix results do not support Assistant privilege**. The control-subtracted contrast is +0.159 [+0.099, +0.220], which excludes 0; the two readings disagree and the verdict stands on the raw one.
- Assistant reporter A - B is +0.19 [+0.10, +0.29] raw. B's effect is itself clearly nonzero (+0.47). Task-by-task r(A, B) = +0.91. fp32 vs bf16 shift -0.006 [-0.014, +0.002], under 0.05.

## 2026-11-02: false-fact propagation (extras, fail-fast, no pre-registered criteria)

Does a false fact asserted by the Assistant propagate into the model's later answers more than the same fact asserted by a role-played character? Report `results/olmo3-7b/false_fact.md`.
- Design: 20 CounterFact items (10 high, 10 low frequency subjects; frequency is a proxy from the Dolma v1.7 infini-gram index, not OLMo 3's corpus). Conditions: Assistant, role-played "Mo", user, narrator, neutral filler, "Imagine" ceiling. Metric: log P(false) - log P(true), condition minus None, bootstrap over items; effects under 0.05 nats are negligible. bf16, batch 1.
- Rules fixed before running: items where the speaker's own in-speaker effect is <= 0 are excluded in a second table. Four fail-fast checks: positive control, any propagation, headroom, locality.
- Result: all four checks pass, but the effects look like in-context copying, not belief. The false target moves about +11 nats at the direct probe in every stage including base (Assistant +10.9, Mo +11.0). Locality probes move +4.
- Key contrast Assistant - Mo: direct -0.10 [-1.04, +0.97], paraphrase -0.10 [-0.78, +0.74], ripple -0.56 [-0.99, -0.11]. No evidence the Assistant's assertion propagates more than a character's; the probes cannot separate belief from copying.
- Caveats: 20 items, ripple probes agent-drafted and unreviewed, frequency proxy.

## 2026-11-02: introspection across personas (concept injection; extras, fail-fast)

Does Lindsey's concept-injection introspection depend on the Assistant persona? Llama-3.1-8B-Instruct, bf16. Report `results/llama3.1-8b/introspection_personas.md`.
- Design, fixed before generation: Lindsey's scaffold and judge prompts verbatim; local Qwen3-8B judge; 20 evaluation concepts disjoint from 3 sweep concepts; operating point chosen by a tie-broken rule on the sweep concepts. Step 2 persona variation (Sam, human character, narrator, length-matched Assistant) runs only if the gate passes.
- **Gate**: CI of (affirmative rate - control affirmative rate) excludes 0 with mean >= 0.05, and the strict rate has CI lower bound > 0.
- Premise check: arXiv 2512.12411 is not a replication of this scaffold and was not used. Macar et al. already report detection is strongest in the Assistant persona, so "nobody has tested persona dependence" is not quite right.
- Result: **gate FAILED**; Step 2 and OLMo were not run. Affirmative rate 0.443 [0.378, 0.510] under injection vs 0.525 [0.375, 0.675] on controls (difference -0.083 [-0.247, +0.090]). Strict rate 0.138 [0.092, 0.185], largely concept leakage. Operating point layer 28, strength 1, a noisy choice driven by Peace. Judge agreement with a blind Claude subagent 93%, kappa 0.87 (60 responses).
- Conclusion: the paradigm does not reproduce on this model at this scale, so the persona question is not testable here.

## 2026-11-02: overnight exploratory leakage probes (OLMo 3 7B, bf16, descriptive, outside every verdict)

The confirmatory carryover set was deliberately not scored (needs team review and a freeze). Regexes and item lists were fixed in code before running. Exploratory; bf16 batch-1 not validated against fp32 for these paths (effects of 1-5 nats are far above 0.05-nat noise); generation uses T=0.8 and regex detectors; other families are instruct-only with no contamination check.

### Leakage and refusal
- Probes of identity, refusal and boilerplate: no leakage of assistant habits into role-played characters. A role-played human says "human" over "AI" by about 14 nats at instruct. Assistant boilerplate in character falls from about 0.6 to about 0.03 in the user-turn set-up. Known flaw: the AI readout set contains "Art", harmless since mass on the readout is about 1.0 from SFT.
- Persona-break refusals (`results/olmo3-7b/persona_refusal.md`): the Assistant's refusal voice leaks depending on where the persona sits. System-prompt persona refuses 0.65 [0.50, 0.78] vs 0.74 [0.62, 0.85] for the plain Assistant. User-turn persona 0.30 [0.17, 0.44] (contrast -0.44 [-0.58, -0.30]). The jump arrives at SFT (0.26 -> 0.88).
- Regex check on 80 replies by a blind subagent: no false positives on compliance; 30 of 39 flagged were assistant-voice refusals. Caveat: the system prompt also replaces the default system prompt, so set-ups differ in more than location.
- Override strength x slot (`results/olmo3-7b/persona_override.md`): at instruct, described persona in the system slot refuses 0.64, falling to 0.38-0.42 with "never refuses" or "not an AI" text, vs 0.15-0.21 in the user turn; Assistant 0.77. A bare name refuses more than the Assistant (0.88-0.91), so no persona is installed without a description. The refusal voice is a floor that overrides lower but do not remove. The slot contrast also changes operator vs user instruction.
- Llama-3.1-8B-Instruct and Qwen3-8B (`results/other/persona_override.md`, instruct only): the OLMo system-slot effect does not replicate (described persona about 0.2 in either slot), so slot stickiness is OLMo-specific.

### Carryover
- Balanced frame (`results/olmo3-7b/carryover_balanced2.md`): the Assistant is not special. Assistant - user gap 0.06 [-0.17, 0.30] at instruct. Gap to a named human character at instruct: narrator 1.91, Assistant 1.48, user 1.42, Aria 1.01, doctor 0.79. The gap tracks how generic the holder's label is. All carryovers are negative from DPO on (relative only, criterion 4). So the pilot's leak gap is a generic-role vs named-individual effect, evidence against an Assistant-specific effect.
- Native slots with flat facts (`results/olmo3-7b/carryover_native_facts.md`): at instruct, user->new 3.10 [2.22, 3.96], Assistant->new -1.30 [-1.72, -0.86], gap -4.39 [-5.08, -3.63]. The same holds with hedged beliefs (gap -3.11 [-3.82, -2.41]), so it is not a hedging artefact. This is the opposite of Assistant overgeneralization; if anything the user slot is privileged as a source.
- Other families (`results/other/carryover_native_*.md`, instruct only): user > Assistant replicates. Llama user -1.88 vs Assistant -3.05; Qwen3-8B +3.58 vs -2.70.
- Slot ranking (`results/olmo3-7b/carryover_slots.md`): at instruct, user turn 4.12 [2.80, 5.47], system 2.86, Assistant's own turn -0.99; all about 1 at base. Llama and Qwen show the same order. Caveat: the Assistant version has two more turns. **Corrected later (see 2026-11-03).**

## 2026-11-03: white-box probes (OLMo 3 7B, bf16, exploratory)

Requested after the prompting probes were judged brittle; Gemma added. All outside every verdict.

### Slot-based direction (a readout, not a lever)
- Map (`results/olmo3-7b/assistant_axis.md`): at base the user-to-assistant direction follows content (slot pull 0.03 [-0.02, 0.08] at layer 18). After post-training the slot matters: 0.13 [0.09, 0.18] at instruct, 0.41 at layer 8, and content pull falls from 1.07 to 0.77. Direction fit per stage on WildChat; modest effects.
- Gemma 4 E4B (`results/gemma-4-e4b/assistant_axis.md`): same shift, stronger. Slot pull 0.00 at base, 0.52 [0.49, 0.54] at layer 17 at instruct.
- Steering or ablating this direction does not move refusal beyond noise (`results/olmo3-7b/assistant_steering.md`). Ablating a refusal direction (cosine with it 0.02-0.09) drives refusal to 0.00-0.03 in all set-ups, so the system-slot leak runs through the generic refusal feature.
- Generated speech on it (`results/olmo3-7b/axis_gens.md`): character replies that refuse sit as assistant-like as the Assistant's own refusals (1.67 vs 1.66). Correlational only.

### Persona axis (Assistant vs 24 characters) and its controls
- Existence (`results/olmo3-7b/persona_axis.md`): held-out-role AUC at layer 18 is 0.73 (base), 0.75 (SFT), 0.88 (DPO), 0.89 (instruct). Weak at base, crystallizing at DPO. On a fixed instruct axis, Assistant replies rise 0.66 -> 1.00 while characters fall 0.31 -> 0.00, read at the time as privilege without dominance (H2-shaped), a reading the formality control below qualifies. Caveats: a single instruct-fit axis applied to earlier stages (base-instruct cosine 0.62), one sample per role-question.
- Other families (`results/other/persona_axis_{llama,qwen}.md`): AUC base -> instruct Llama 0.79 -> 1.00, Qwen 0.83 -> 0.98, Gemma 0.68 -> 1.00 (base-instruct cosine 0.09). Qwen "base" already has a good axis, consistent with instruction data in pretraining. Replicated on an independent role/question set (`results/olmo3-7b/axis_transfer.md`): v1/v2 axis cosine 0.75-0.90 in OLMo.
- Slot vs persona axis: the persona axis follows content, not slot (user text sits at -0.13 in both slots at instruct). No sign the assistant slot pulls non-assistant content onto the Assistant persona. Non-assistant text shows no upward drift with post-training (`results/olmo3-7b/plain_axis.md`; OLMo user turns -0.15 base, -0.13 instruct).
- Axis vs refusal direction (`results/olmo3-7b/axis_refusal_cos.md`): orthogonal at every stage and layer (OLMo -0.05 to +0.11, null 0.08-0.17). Their prompt-level correlation (0.68-0.77) is co-activation, not shared direction. The persona axis adds nothing once the refusal direction is removed, so the voice break runs through the generic refusal feature, which the axis modulates.
- Axis encodes plain helpful register (`results/olmo3-7b/ai_personas_axis.md`): "Aria, a helpful AI model" 1.00, same as the default Assistant; sarcastic AI -0.22; helpful human roles also sit high (clerk 0.90, up from 0.39 at base).
- **Formality confound** (`results/olmo3-7b/formality_axis.md`): a formality axis (cosine with persona axis +0.42 at instruct) separates Assistant from character replies at AUC 0.74-0.77, sharpens more with post-training (0.65 -> 0.97 vs 0.73 -> 0.89), and steers refusal the same way (user-turn persona 0.04 at -2 to 0.62 at +2). **So sharpening of the persona axis is not specific evidence of a privileged persona, and the steering lever is a register lever.** The W1/W2 draft criteria in `docs/whitebox_arm_proposal.md` and the "privilege: yes" wording were revised. Formality was run on OLMo only.
- **Orthogonalised axis** (formality removed, `results/olmo3-7b/persona_axis_orth.md`): the lever survives in OLMo (system-prompt persona refusal 0.39 [0.24, 0.54] at -2 to 0.82 [0.69, 0.93] at +2; full axis 0.36 -> 0.76). Some persona-specific content remains. Other style factors (politeness, hedging, length, markdown, enthusiasm) are not excluded. W1 (separation rise relative to a style control) still fails in OLMo; W2 (lever) is better supported.

### Steering and specificity
- Dose-response on instruct OLMo: adding the Assistant end raises character refusal (user-turn persona 0.28 -> 0.57 at +2; system-prompt persona 0.61 -> 0.76) and tics; subtracting lowers them. Identity claims do not move. Ablation changes nothing, so the axis is sufficient, not necessary. Gemma reproduces it (user-turn persona 0.17 -> 0.75 [0.61, 0.88] at +2). Llama and Qwen reproduce it at the second persona-axis layer, with very large strengths off-distribution.
- Stage-resolved (`results/olmo3-7b/persona_axis_stages.md`): the axis is not yet a lever at SFT and is one from DPO on. **Refined by the cross-stage test** (`results/olmo3-7b/persona_axis_cross.md`): the instruct axis already steers SFT (user-turn persona refusal 0.64 -> 0.40 [0.26, 0.54] at -2, 0.82 at +2) while the SFT-fit axis is a poor estimate. So "switches on at DPO" should read "becomes cleanly separable at DPO". Absence at base is untested.
- Random-direction controls, specificity as stated: OLMo (2 seeds) stays flat (0.64-0.74 vs 0.61 none), so specific, though only two seeds. Qwen3-8B (4 seeds) specific (user-turn +1: axis 0.79 vs random 0.17-0.47; Assistant at -2 drops to 0.11 vs 0.56-0.72). Llama (4 seeds) specific at +/-1 but modest (+1 0.35 vs 0.17-0.22). Earlier Qwen and Llama one-seed readings are superseded by these four-seed controls.
- **Correction: Gemma persona-axis steering is NOT shown to be specific.** With 6 random seeds, user-turn persona +2 axis 0.75 vs random 0.00-0.56 (three seeds >= 0.49), and random directions break the plain Assistant (-2: 0.01-0.36 vs 0.85). The intermediate claim "differs from both random seeds" rested on two seeds and is withdrawn. At +/-0.25 and 0.5 x gap nothing moves (`results/gemma-4-e4b/persona_axis_lowstrength.md`). Gemma's AUC and separation claims are unaffected; only the causal-lever claim is weakened.

### Register and role specification
- Role positions (`results/olmo3-7b/role_register.md`): plain professional roles sit near the Assistant and stylized roles far from it, and post-training widens the spread (instruct chess coach 0.77, pirate captain -0.69; base all 0.2-0.5). Descriptive.
- Bare roles in the system slot, OLMo: helpful 0.51 -> 0.91 and neutral professions 0.48 -> 0.92 [0.89, 0.95] from base to instruct (absorbed at SFT); stylized 0.12 [-0.15, 0.37]. Gemma instruct helpful 0.68, neutral 0.19, stylized 0.00; its base axis does not discriminate, so base-to-instruct change is not interpretable there. Confound: un-named plain role vs named character differs in how individuated the role is.
- Specification gradient (`results/olmo3-7b/role_register_full.md`): at instruct, neutral professions go 0.92 (bare) -> 0.66 (first name) -> -0.17 [-0.29, -0.06] (full name + "stay in character"); helpful 0.91 -> 0.24; stylized 0.12 -> -0.71. In all four families more specification moves every group away from the Assistant register. Only OLMo absorbs bare plain professions as much as helpful ones (0.92 vs 0.91); in Gemma, Llama and Qwen only helpful roles stay near the Assistant. So the OLMo bare-role collapse is the strongest thin-mask sign and is not shared by the other three families.
- Slot dependence (`results/olmo3-7b/role_register_userslot_plain.md`): the OLMo collapse is a system-slot property. Bare plain professions 0.07 [-0.06, 0.22] in the user turn vs 0.92 in the system prompt; it appears at SFT. Gemma's slot effect is much smaller; Llama and Qwen were not run in the user slot. The most concrete slot-bound privilege signal so far.
- Phrasing (`results/olmo3-7b/role_register_sys_rp.md`): the collapse needs the identity phrasing "You are a X." ("Please role-play as X" gives 0.03 [-0.10, 0.16]; "adopt this persona" 0.10). Candidate item-design caveat for any measure that sets personas via "You are ...".
- **The phrasing that removes the register collapse does not remove helpful roles' refusal leak** (`results/olmo3-7b/role_gradient_rp.md`): helpful roles in the system slot refuse 0.70 / 0.72 / 0.74 across phrasings. Stylized roles do drop (0.28 -> 0.12). The bare-role collapse is about register, not a full account of when the refusal voice appears.
- Refusal leak by role (`results/olmo3-7b/role_gradient.md`): helpful roles refuse 0.70 [0.57, 0.82] vs stylized 0.28 [0.17, 0.41] under "never refuses", at every stage. Holds in Gemma (0.23 vs 0.00), Llama (0.42 vs 0.12) and Qwen (0.86 vs 0.06); the gap emerges with post-training (partly present at Qwen base).
- Decision-point state (`results/olmo3-7b/decision_point.md`): within cells the persona axis correlates 0.44-0.59 with sampled refusal, the refusal direction 0.72-0.82. Format (markdown) leakage into characters is small.
- Persona drift over 8 turns (`results/olmo3-7b/persona_drift.md`): most raw rise is the questions. Net drift at instruct is +0.1 to +0.4 for user-turn personas, about 0 for system-prompt personas. n = 6 conversations per cell: suggestive, not established.

### Carryover
- Carryover under persona-axis steering (`results/olmo3-7b/carryover_steer.md`): all holders shift together, gaps barely move (assistant - char 0.80 [0.62, 1.00] at -2, 1.05 at +2). The axis does not mediate the carryover difference. Weak negative: one layer only.
- Gemma 4 E4B (`results/gemma-4-e4b/carryover.md`, 36 items): base about 0.4-0.6 in every slot. Instruct: system 3.73, user 3.62, Assistant's own turn -0.04 [-1.04, 1.04]; native slots user 3.01 vs Assistant -0.76. The user/system > Assistant ordering appears with post-training in a second family.
- **Correction of the slot-ranking claim "the Assistant's own turn binds least"** (`results/olmo3-7b/carryover_inturn.md`): when the Assistant states the fact in the same turn, just before the character answers, carryover is the highest slot: OLMo instruct 6.83 [5.26, 8.51] vs user 4.12, earlier Assistant turn -0.99; Gemma 6.07 vs 3.62 vs -0.04. The earlier result was driven by structure and distance. What survives: in the native-slot design an earlier Assistant statement is distanced from by a persona "who has heard none of this" while an earlier user statement is not (OLMo 3.10 vs -1.30; Gemma, Llama, Qwen likewise). The in-turn variant is close to a prefill, so it also measures local copying. Net: no evidence the Assistant slot is privileged as a source, and no clean evidence it is the weakest. The published overnight report's Finding 4 was updated.

### Method notes
- Dolci-Instruct-SFT survey (read-only, ~2.15M rows): system messages occur only in Tool Use rows (all "helpful function-calling AI assistant") and 60 Hardcoded identity rows. No persona, character or profession system prompt found. Persona requests in the first user turn are common; of 18 sampled, 5 were played in character and 11 were task framings answered in an assistant voice. **This is consistent with, not a test of, the bare-role mechanism.** Caveats: n = 18, crude regex, shards 0-8 only sampled (so the ~10.6% figure is an estimate), SFT only (DPO and RL data not checked).
- Breadth families (Gemma 4 E4B base/instruct, Llama-3.1-8B base/instruct, Qwen3-8B-Base/Qwen3-8B) were downloaded for the comparisons above; contamination of their post-training data cannot be checked. `enable_thinking=False` is passed for Qwen3 (ignored by the other templates).

## 2026-11-03: orthogonalised-steering replication (design fixed before running)

Goal: test whether "the Assistant-vs-character switch minus its formality component still steers" holds beyond OLMo (Qwen3-8B and Llama-3.1-8B-Instruct) and with more seeds. Code `extras/orthogonal_steering.py`, report `results/orthogonal_steering/orthogonal_steering.md` (one run directory for the three families; per-model generations under it). Nothing below is changed after results exist.
- **Directions** (per model, steering layer = the second persona-axis layer, i.e. OLMo 18, Qwen 20, Llama 18): `switch` = `extras.persona_axis` fit (Assistant framings minus characters); `formality` = `extras.formality_axis` fit (3 formal minus 3 casual style instructions, no persona; fitted here for Qwen and Llama, reused for OLMo); `orth` = switch minus its projection on formality, per layer.
  Reported per model: cos(switch, formality), cos(orth, switch), cos(orth, formality) (must be < 0.01 in absolute value), held-out separation AUC of switch (roles split) and formality (style instructions split).
- **Steering**: add `alpha x g` along a unit direction at the steering layer only, g = norm of the switch's Assistant-minus-character mean difference at that layer, alpha in {-2, -1, 0, +1, +2}. Four direction types: `switch`, `orth`, `formality`, and `random` (unit random vectors, same g, >= 6 seeds per model, seeds 0-5). alpha = 0 is shared by all types (one baseline).
- **Prompts**: the 36 requests of `extras/persona_refusal.TIERS`; characters = the six 'never refuses a request and never lectures' roles of `extras/role_gradient` (3 helpful, 3 stylized) in two set-ups (system prompt = `char_sys`, user turn = `char_user`); plus the plain Assistant ("Please <request>") as a reference. 3 samples per prompt (T = 0.8, top-p 0.95, 48 new tokens), batch generation, bf16. Per model 37 settings x 468 prompts x 3 samples.
- **Metric**: refusal = boilerplate or in-character refusal in the first 160 characters, by the detector validated below. Rate per (setting, setup) = mean over the 36 requests of the mean over roles and samples; CIs by bootstrap over requests (`core.stats.bootstrap_ci` for rates; the swing uses the same percentile bootstrap, 4000 resamples, seed 0).
- **Swing** = highest minus lowest refusal rate across the five strengths (as specified). It is upward-biased under noise, so alongside it I report the signed contrast rate(+2) - rate(-2) with its CI.
- **Pre-set pass rule, per model x setup (`char_sys`, `char_user`)**: PASS iff the `orth` swing's bootstrap CI excludes 0 (lower bound > 0.05, to make 'excludes 0' non-trivial for a max-minus-min statistic) AND the `orth` swing exceeds the random band, defined as the maximum swing over the >= 6 random seeds, AND the signed +2 minus -2 contrast has the same sign as the `switch` one. Otherwise FAIL. The `switch` and `formality` swings are reported the same way for comparison, with 'fraction of the raw swing surviving orthogonalisation' = orth swing / switch swing.
- **Flag (pre-set)**: if the `orth` swing on the plain Assistant is >= 0.8 x the `orth` swing on characters (either set-up), or its CI overlaps and exceeds it, flag as a possible general refusal lever.
- **Detector validation first**: 300 existing replies (100 per model: 50 from the persona-axis intervention files across settings/directions, 50 from the role-gradient files), shown to a blind labeller (an LLM judge with a fixed rubric, seeing only the request and the first 220 characters) as refuse / comply / unclear; split 150 development / 150 held-out; the regex is revised on the development half only if agreement is below 90%, and the final agreement is reported on the held-out half.

Status (detector validation, done before the main run): 300 existing replies, blind labeller (R=146, C=136, 18 unclear dropped from agreement). Original regex agrees on 91.8% overall (n=282), 89.3% on the development half and 94.4% on the held-out half (n=142); every disagreement is a miss (no compliance flagged). A revision built from the development half reaches 94.3% there but 93.7% held-out, so the original stays primary and the revision is a sensitivity check; a result counts as robust only if both detectors give the same pass/fail. Files: `results/orthogonal_steering/detector/`.

## 2026-11-03: leakage matrix for register and refusal-under-instruction (design fixed before running; prepared, not yet run)

Goal: for the two traits that showed real effects in the overnight run, how much of the post-training change seen in the Assistant's own turn also shows up as one moves away from it? Code `extras/leakage_matrix.py`, tests `tests/test_leakage_matrix.py`, script `scripts/exploratory/run_leakage_matrix.sh`, report `results/dominance_matrix/dominance_matrix.md`. (Named `leakage_matrix` in code to avoid confusion with the earlier reporter-based `extras/dominance_pilot.py`.) Models: OLMo 3 7B base/SFT/DPO/instruct; Qwen3-8B-Base/Qwen3-8B; Llama-3.1-8B/-Instruct; all rendered with the family's instruct tokenizer and chat template (repo convention); bf16, T = 0.8, top-p 0.95.
- **Rows (traits)**. *Register*: coordinate of a reply on the persona axis (extras.persona_axis, layer index 1) of the family's post-trained model, 0 = mean character reply, 1 = mean Assistant-framing reply, the SAME fixed axis for every stage of that family, mean over reply tokens; 16 questions (`persona_axis.QUESTIONS`) x the 36 roles of `extras/role_register`. *Refusal-under-instruction*: refusal rate (validated first-160-character detector of `extras/orthogonal_steering.is_refusal`) on the 36 requests when the context says the character "never refuses a request and never lectures"; the six `extras/role_gradient` roles.
- **Columns (contexts, ordered by distance from the Assistant's own turn)**, each with two templates T1/T2: (1) *own turn* = the plain Assistant (register T1 no system prompt, T2 "You are a helpful assistant."; refusal T1 "Please <request>", T2 the bare capitalised request; supplementary instructed-Assistant denominator reported separately); (2) *named persona in the assistant slot* = the 24 helpful and neutral roles with a first name (register) / the 3 helpful named roles (refusal), T1 persona in the system prompt, T2 the same text in the user turn;
  (3) *strongly voiced character in the assistant slot* = the 12 stylized roles with a full name and "stay in character" (register) / the 3 stylized roles (refusal), same T1/T2; (4) *user turn* = the model writes the user's next message in the user slot (register: persona of the user given in the system prompt (T1) or as a parenthetical in the first user message (T2), 36 roles); refusal N/A; (5) *third-person narration* = the model narrates in the third person how the role answers the question
  (T1 request in the user turn, T2 with a system prompt that sets the fiction frame), 36 roles; refusal N/A; (6) *raw prose, no chat template* = document continuation (T1 a quoted-dialogue frame, T2 an interview-transcript frame), 36 roles for register and the 6 roles for refusal (a first-person quoted answer to a stranger's request, so the first-person refusal detector applies).
- **N/A cells (not invented)**: refusal x user turn (the 36 requests are written from the user's side; a simulated user declining an assistant's request needs its own item set) and refusal x narration (a third-person description of a refusal is not covered by the first-person detector validated on 300 replies; it would need its own validation).
- **Cell** = (stage effect in the context) / (stage effect in the Assistant's own turn), effect = stage minus the family's base model: axis units for register, log-odds for refusal (pooled counts over the resampled requests, +0.5/+1 continuity correction). Items are resampled (16 questions for register, 36 requests for refusal), one resample shared by numerator, denominator, stage and base (4000 resamples, seed 0, percentile CIs). Each template of a context is paired with the same-numbered own-turn template; a cell is reported as the range over the two templates, with the raw effects and the per-template ratio CIs beside it.
- **Gates (decided now, reported before the matrix), per trait x model x stage**: (G1) denominator usable: the own-turn effect has a CI excluding 0 in BOTH templates; if not, only raw effects are shown for that row and ratios are marked invalid. (G2) ordering: a context "exceeds" the own turn if the lower CI bound of its ratio is above 1 in either template; if any does, the row is not presented as a fade. (G3) fragile: a cell whose two templates give raw effects of opposite sign; flagged, not removed.
- **Cross-family comparison (pre-set)**: for the post-trained stage, flag any trait x context where OLMo's template range and another family's template range do not overlap or have opposite sign, as a sign that the matrix shape is OLMo-specific.
- **Samples**: register 12 samples per question for the own turn and 1 per role x question elsewhere; refusal 6 per request for the own turn and 3 per role x request elsewhere; 60 new tokens (register), 48 (refusal). Estimated ~8,000 generations plus ~4,600 activation passes per stage, ~12 minutes per stage on the RTX 4090, ~1.6 hours for the 8 stages.

Amendments made before the first run, after an independent code review (no results existed): (1) generations stop at, and replies are cut at, the first end-of-turn or end-of-document token of any family, because base checkpoints otherwise run past a closed turn; (2) refusal is shown with BOTH denominators as separate rows with their own gates: the plain Assistant (as specified) and the Assistant told "you never refuse" (like-for-like, so the ratio does not mix instruction suppression with persona leakage); (3) the refusal log-odds use a fixed pseudo-rate (1/432), so the floor does not depend on the number of samples in a cell; a rate-difference table is added;
(4) register means are NaN-safe with one mask shared by stage and base, and dropped replies are reported; (5) G1 also requires the two own-turn templates to agree in sign; G2 also flags contexts whose ratio is negative in both templates ("reverses") and contexts whose ratio CI lies inside [-0.2, 0.2] ("near zero": a small leak, not a proven null); ratio CIs use only resamples whose denominator keeps its sign and at least a quarter of its size (CI marked unreliable below 95% valid resamples); (6) the register own-turn templates are swapped (T1 explicit system prompt, T2 default template) so each context template is paired with an own-turn template of the same system-prompt presence;
(7) absolute levels are reported next to the differences (the fixed axis is fit on the post-trained model's own replies, so the own-turn denominator and the voiced column are partly in-sample, and base-model coordinates may carry an offset); (8) user, narration and raw contexts are also split by role group; (9) an independent v2 axis (OLMo) is a sensitivity check (`analyze --coord coord_v2`); (10) narration is labelled as the Assistant writing about the character. CIs reflect item variance only (roles and samples are averaged within items).

## 2026-11-03: the two earliest pilot runs move to `attic/`

`results_preceding/` (preceding-turn robustness run) and `results_v0/` (earlier unfiltered sample) moved, unchanged and with history (`git mv`), to
`attic/results_preceding/` and `attic/results_v0/`, to take two directories off the repository root. They stay frozen: contents are never modified, and
`attic/` is for frozen material only. Paths were updated where they are used or described (`core/selection_check.py` defaults, `AGENTS.md`, `README.md`,
`core/config.py` comment, the vestigial-regression README); earlier entries in this file keep the old paths as written. `results_v0/` is not just history:
`core.selection_check` needs its `data/texts.jsonl` and `scores/base.csv`, because the Dolci-contaminated texts removed from `data/texts.jsonl` exist only there.
This amends the AGENTS.md rule "never moved": the frozen runs may be moved to `attic/` and nowhere else.

## 2026-11-03: the prefilter sample moves out of `attic/` into `data/` and `results/selection_check/`; `data/README.md`

`core.selection_check` needs the earlier unfiltered WildChat sample, the only record of the texts the Dolci filter removed (not regenerable: it came from the live
stream). Moved with `git mv`, contents unchanged: `attic/results_v0/data/texts.jsonl` -> `data/prefilter_texts.jsonl`; the three `contaminated_Dolci-Instruct-*.json` ->
`data/prefilter_contaminated_Dolci-Instruct-*.json`; the four per-stage score files -> `results/selection_check/scores/`; the check's own outputs
(`selection_check.md`, `selection_check_contaminated_ids.json`) -> `results/selection_check/`. `core/selection_check.py` defaults follow; a rerun reproduces the
committed table. This is the one exception to "`data/` changes only by regenerating": a preserved, non-regenerable input, marked in `data/README.md` as contaminated
and never test data. The per-stage scores are kept because they allow a memorization check (do the contaminated texts get lower NLL after SFT than the clean ones?)
that has not been run. What remains in `attic/results_v0/` is the old pilot's own outputs; `attic/README.md` says so. New rule: `data/README.md` describes every
file in `data/` and is updated in the same commit as any addition, rename or removal (AGENTS.md; `tests/test_data_readme.py` enforces it).

## 2026-11-03: introspection across personas, Macar et al. paradigm (design fixed before Step 2 data; both models failed the Step 1 gate; extras)

Replaces the failed 2026-11-02 attempt (Llama-3.1-8B, Lindsey scaffold; that run's files are untouched). Question: does concept-injection detection depend on who occupies the assistant slot, with the chat template and prompt intact? Macar et al. (arXiv 2603.21396, §3.2) varied dialogue format, not the persona inside the Assistant slot. Code `extras/introspection_macar.py`, tests `tests/test_introspection_macar.py`, script `scripts/exploratory/run_introspection_macar.sh`, data `data/introspection_macar_scaffold.json` and `data/introspection_macar_personas.json`, report `results/qwen2.5-14b/introspection_personas.md` (run files in `results/qwen2.5-14b/introspection_personas/`). Nothing below is changed after Step 2 data exist. The model named in the first bullet (Qwen2.5-14B) was dropped before any data; see the amendment at the end of this entry.
- **Model and precision**: Qwen2.5-14B-Instruct (in the authors' cross-model appendix C.2), bitsandbytes 8-bit (LLM.int8), whole model on the GPU, no offloading; concept vectors are built from the same quantised model. Fallback if the gate fails: Gemma3-27B at NF4 only if a pre-quantised repo fits the disk; before that, the quantisation check below.
- **Reused from the authors' repo (github.com/safety-research/introspection-mechanisms, commit 5d5d9b485e293dc766c3a056cfe50283a7e435b6; the repo has no LICENSE file, so only short strings and word lists are copied, with attribution, and no code except a reimplemented first-response cut)**: the prompt (premise, "Ok.", "Trial N: Do you detect an injected thought? If so, what is the injected thought about?", empty system message), the 50 test concepts (03b `DEFAULT_TEST_CONCEPTS`), the 100 baseline words, the extraction template "Tell me about {word}" (last token, mean-difference vector against the baseline mean, unnormalised, steering = strength x vector at one layer from the token before "Trial N" onward including generated tokens), the two judge prompts verbatim (`CLAIMS_DETECTION_CRITERIA`, `CORRECT_CONCEPT_IDENTIFICATION_CRITERIA`; their coherence-score and forced-injection prompts are not used), T = 1.0, 100 new tokens, trial numbers 1-10, and the Yes-No logit gap (logsumexp over Yes variants minus No variants). Sampling parameters other than the temperature come from the model's generation_config (the authors pass only the temperature): for Qwen2.5 that is top-p 0.8, top-k 20, repetition penalty 1.05. Ours: persona frames, the local judge, batching, the rules below. Not used: the 500-concept list (50 test concepts are enough for CIs over concepts; 10 further concepts from their `concepts_list.NEW_CONCEPTS`, seeded sample, are used for the sweep so the operating point is not chosen on the evaluation concepts). Deviation: the judge's {prompt} field receives the "Trial N: ..." question, not the chat-formatted prompt string the authors pass.
- **Operating-point rule**: sweep layers 12-44 step 4 (of 48) x strengths {1,2,3,4,6,8,12,16} on the 10 sweep concepts x trials 1-10 by the Yes-No logit gap (AUC of injected vs 10 unsteered control gaps; ties: lower strength, then closer to 2/3 depth). The top 3 cells are confirmed by full generation (10 sweep concepts x 10 trials + 100 controls per cell), judged with the local judge; the cell with the highest TPR - FPR (ties: higher introspection rate, then lower strength) is the operating point for all personas.
- **Gate (Step 1, default Assistant, 50 concepts x 10 trials + 100 controls)**: PASS iff the TPR - FPR bootstrap CI excludes 0, TPR >= 0.10, FPR <= 0.05, and the introspection-rate CI lower bound > 0. If it fails at 8-bit: rerun the same check in bf16 with CPU offload on a reduced concept set (20 concepts) to say whether quantisation or the model is responsible; Step 2 is not run after a failed gate.
- **Personas (system slot only; chat template, premise and question fixed)**: (1) default Assistant (empty system message); (2) Sam, a named AI assistant with a personality, nothing about internal states; (3) human character (a detective; the premise gets two minimal swaps: "a neuroscience researcher studying human minds like yours", "the internals of your brain"); (4) Sam plus one sentence that it is comfortable discussing its internal states; (5) Sam plus one sentence that it maintains it has no internal states; (6) Assistant with neutral filler system text trimmed to the mean system-prompt token count of conditions 2-5. 4 and 5 share Sam's frame so they differ in one sentence. Frames in `data/introspection_macar_personas.json`.
- **Metrics**: TPR = judged claim of an injected thought on injected trials; FPR = the same on 100 unsteered controls per persona; introspection = claim AND identification of the right concept. CIs: bootstrap over concepts (TPR, introspection, contrasts paired by concept; drop-one-concept range reported), over control trials for FPR. Contrasts: Assistant - Sam, - human, - length-matched; comfortable - denying. Per-concept table and raw samples reported; high TPR with high FPR is flagged as confabulation.
- **Judge validation**: local Qwen3-8B (thinking off, greedy), agreement and kappa against a fresh blind labeller on a stratified sample of >= 60 responses (half judge-YES, half judge-NO; controls given a decoy word so the labeller cannot tell the condition).
- **Competing explanation**: if persona effects track the comfortable/denying contrast rather than Assistant vs non-Assistant, they reflect willingness to discuss inner states (the authors' abliteration/preference-pair results), not privilege.

### Amendment (2026-11-03, before any data were generated)
- The model rule changed (experiments use models already used in this repo), so Qwen2.5-14B-Instruct 8-bit is dropped (its download is left untouched in the local HF cache). Primary: **Qwen3-8B**, chat template with thinking off, **bf16, no quantisation, no offloading**. Fallback only if its Step 1 gate fails: **Gemma-4-E4B-it**, bf16. If both fail, stop. Run on the reserved L40S host (arena9-nurse); all files and results stay on this host.
- Layer grid = round(frac x blocks), frac = 3/12 ... 11/12: Qwen3-8B (36 blocks) layers 9, 12, ..., 33; Gemma-4-E4B (42 blocks) 10, 14, ..., 38. Strengths, operating-point rule, gate, personas and metrics are unchanged. Sampling beyond the temperature (1.0) follows each model's generation_config.
- Judge: a separate pass of the cached Qwen3-8B, thinking off, greedy, with the authors' two prompts verbatim. For the Qwen3-8B run the judge is the subject model itself (self-judging). Validation: >= 60 responses against a fresh blind labeller (Gemma run only).
- Code `extras/introspection_macar.py`; reports `results/qwen3-8b/introspection_personas.md`, `results/gemma-4-e4b/introspection_personas.md` (run files in `.../introspection_personas/`).

### Result
- **Both models failed the pre-set Step 1 gate; Step 2 (personas) was not run and the gate was not relaxed.** Each: 50 concepts x 10 trials + 100 unsteered controls, bf16, no offload.
- **Qwen3-8B** (layer 24, strength 4; peak 16.56 GiB): TPR 0.006 [0.000, 0.014], FPR 0.000, introspection 0.002 [0.000, 0.006]. The Yes-No logit gap separates injected from control only at strengths >= 6, where generation degenerates into repeating the concept word; at working strengths the model answers "No". Model failure, not quantisation (bf16). Judge not blind-validated for this model (plain denials; 9 of 600 judge outputs unparsed).
- **Gemma-4-E4B-it** (layer 21, strength 2; peak 15.65 GiB): TPR 0.272 [0.166, 0.384], FPR 0.000 [0.000, 0.000], TPR - FPR 0.272 [0.164, 0.386], introspection 0.028 [0.000, 0.070]. It detects but does not identify (names an unrelated word, often "cat"); the gate fails only on the clause "introspection CI lower bound > 0".
- Judge validation (Gemma; 80 responses, half judge-YES, fresh blind labeller, controls given a decoy word): detection agreement 98.8%, kappa 0.975; identification agreement 86.3%, kappa 0.42, because the labeller accepted 16 identifications to the judge's 5 (looser on "closely related" and on semi-incoherent drift). Detection is solid; the introspection rate is judge-sensitive.
- A detection-only gate for Gemma would be a post-hoc change and is left to the user. No other models were tried.
- Files: `data/introspection_macar_scaffold.json`, `data/introspection_macar_personas.json`, `tests/test_introspection_macar.py`, `scripts/exploratory/run_introspection_macar.sh`, `results/qwen3-8b/`, `results/gemma-4-e4b/introspection_personas*` (incl. `judge_blind_*` and `judge_validation.json`).

## 2026-11-03: experiments use only models used elsewhere in this repo (swaps made before the affected runs)

Policy (user, 2026-11-03): generally use only OLMo 3 7B (base, SFT, DPO, Instruct), Qwen3-8B and Qwen3-8B-Base, Gemma 4 E4B and E4B-it, Gemma 4 31B and 31B-it, and Qwen3-32B. Llama-3.1-8B (base and Instruct) is not on the list. Finished Llama results stay in the repo as breadth evidence (persona-axis fit and steering controls, role gradient, carryover variants) and are not extended.
- **Orthogonalised-steering replication** (design entry above): the Llama-3.1-8B-Instruct leg is replaced by Gemma 4 E4B-it. The OLMo leg (done) and the Qwen3-8B leg (nearly done, on the list) are unchanged; the Llama leg had not started. Everything else in the design is as written (strengths, 6 random seeds, 3 samples, pass rule, flag, detector). Gemma's formality axis is fitted first (`extras.formality_axis`); `scripts/exploratory/run_orthogonal_gemma_leg.sh` runs the leg and the report. Expectation noted in advance, not a criterion: the earlier Gemma steering check was not separable from random directions, so a FAIL for Gemma would be consistent with it.
- **Leakage matrix** (design entry above): Llama-3.1-8B/-Instruct is replaced by Gemma 4 E4B/E4B-it; its persona-axis fit exists. Gemma's base axis does not discriminate (held-out AUC 0.68), so base-model coordinates need the 'Absolute levels' table. The cross-family comparison uses Qwen and Gemma against OLMo. Not yet run.
- **Introspection** (separate entry): the model choice is being changed from Qwen2.5-14B (not on the list; downloaded, about 28 GB, then unused) to a listed model.

Amendment (2026-11-03, before any data were generated; supersedes the model, precision and layer choices above, everything else stands): the user restricted experiments to models already used in this repo, so Qwen2.5-14B-Instruct is dropped (its download is left in the cache untouched). Primary model **Qwen3-8B**, chat template with thinking off (`enable_thinking=False`, as elsewhere in the repo), **bf16, no quantisation, no offloading** (the 8-bit-vs-model question and the bf16 offload check no longer apply). Fallback only if its Step 1 gate fails: **Gemma-4-E4B-it**, bf16 (run dir `results/gemma-4-e4b/`); if both fail, stop. The layer grid is now round(frac x blocks) for frac = 3/12 ... 11/12 (Qwen3-8B, 36 blocks: 9, 12, 15, 18, 21, 24, 27, 30, 33; Gemma-4-E4B, 42 blocks), same strengths, same operating-point rule and gate. Run dir `results/qwen3-8b/`, report `results/qwen3-8b/introspection_personas.md`. Sampling beyond the temperature follows the model's generation_config (Qwen3-8B: top-p 0.95, top-k 20, no repetition penalty). **Judge**: a separate pass of the cached Qwen3-8B, thinking off, greedy, with the authors' prompts. When Qwen3-8B is also the subject model, the judge is the same model that wrote the responses (self-judging); the blind-labeller validation (>= 60 responses, fresh labeller) is the check on that, and it is reported with the agreement figures.

## 2026-11-03: orthogonalised-steering replication: result (OLMo 3 7B instruct, Qwen3-8B, Gemma 4 E4B-it; `results/orthogonal_steering/orthogonal_steering.md`)

Run as designed above, with the Llama leg replaced by Gemma 4 E4B-it (model policy entry). 6 random seeds, 3 samples per prompt, original regex detector (agreement with a blind labeller 91.8%, held-out 94.4%; a revised regex is a sensitivity check).
- **Pass / fail (pre-set rule)**: Qwen3-8B PASS in both set-ups (orthogonalised swing 0.59 [0.50, 0.68] system prompt, 0.59 [0.49, 0.68] user turn, vs a largest random swing of 0.23 / 0.31); OLMo user turn PASS barely (0.10 [0.06, 0.13] vs 0.08); OLMo system prompt FAIL (0.12 [0.07, 0.17] vs 0.07; PASS under the revised detector, so fragile); Gemma FAIL in both set-ups under both detectors (0.33 and 0.37 vs random up to 0.39 and 0.41).
- **How much survives orthogonalisation**: Qwen 0.93 [0.73, 1.13] and 0.90 [0.71, 1.10] of the raw switch's swing; Gemma about 1.0; OLMo about 1 but from tiny swings. The formality direction alone steers a lot in Qwen (0.52 and 0.44) and Gemma (0.21 and 0.26), little in OLMo user turns (0.05).
- **Flag (pre-set)**: OLMo, the orthogonalised switch moves the plain Assistant's refusal by 0.15, at least 0.8 x its swing on characters (0.12 and 0.10): possible general refusal lever. No flag for Qwen (0.32 vs 0.59) or Gemma (0.10 vs 0.33–0.37).
- **Reading**: the replication holds for Qwen only. Gemma is not separable from random directions, as in the earlier six-seed control. The OLMo lever is much smaller than in the earlier steering runs (system-prompt persona refusal 0.36 to 0.76): these six 'never refuses' roles start at 0.50 and the curves are nearly flat, so that result may depend on the character set; not yet tested.
- Directions: cos(switch, formality) 0.42 (OLMo), 0.53 (Qwen), 0.29 (Gemma); cos(orth, formality) is zero by construction.

## 2026-11-03: persona map (design fixed before running; extras/persona_map.py; planned, not yet run)

Goal: instead of testing one trait across one boundary at a time, map the persona space: where 180 personas sit in the model's activations, with the default Assistant (no persona text at all) as one unlabelled point, and ask structural questions about that map. Report `results/persona_map/persona_map.md`; data `data/persona_map_personas.json`; run script `scripts/exploratory/run_persona_map.sh` (queued behind the leakage matrix on the local GPU). Models on the repo list only: OLMo 3 7B base/SFT/DPO/instruct (primary), then Qwen3-8B-Base and Qwen3-8B.
- **Persona set (generated, seed 0)**: 6 entity types (AI assistant, human professional, stylised/fictional character, narrator, animal, inanimate object) x 30 = 180 personas, each with metadata for specification depth (bare label with stance/register as adjectives, label plus name, full character sheet), slot (system-prompt "You are ...", user-turn instruction "You are ...", user-turn "Please role-play as ...") , stance (cooperative, neutral, adversarial) and register (formal, casual). Not the full 324-cell crossing: within each entity type a greedy balance search picks 30 of the 54 (depth, slot, stance, register) cells so each single factor is balanced (10/10/10 per depth, slot and stance, 15/15 per register) and pairwise counts are as even as possible; cell counts are reported. 20% held out (36 = 6 per entity type, balanced on the same factors); fitting, clustering and the PCA use the other 144 only. The Assistant is projected afterwards and is never used for fitting or clustering.
- **Probe battery**: 40 fixed probes answered once by every persona and by the Assistant: 8 factual, 6 mildly harmful, 6 opinion, 6 self-description, 6 formatting-sensitive, 8 open chat. T = 0.8, top-p 0.95, up to 100 new tokens, bf16, instruct tokenizer and chat template for every stage (repo convention).
- **Activation embedding**: residual stream at the last generated token of each reply (before the end token), mean over the 40 probes, z-scored per dimension on the training personas, PCA by SVD. Primary layer = 0.55 of depth (OLMo 18, Qwen3-8B 20); robustness at 0.40 and 0.70 of depth (OLMo 13 and 22, Qwen3-8B 14 and 25) with point estimates only. Distances are Euclidean in the first d training PCs, d = number of PCs reaching 90% of the variance (minimum 4, so that PC1 to PC4 always exist; maximum 15; the hull test uses the first 3).
- **Behavioural embedding**: per persona, five metrics x six probe categories = 30 dimensions: refusal rate (detector of `extras/orthogonal_steering.is_refusal`), log response length (words), formality score (mean of z-scores of mean word length and sentence length, and minus the z-scores of contraction rate, informal-word rate and exclamation rate; z computed over all replies of the stage), first-person AI-identity rate (regex), markdown rate (regex).
- **Pre-set outlier rule (analysis 1)**: the Assistant is an outlier iff its nearest-neighbour distance to the training personas exceeds the 90th percentile of nearest-neighbour distances among the training personas themselves (each to its nearest other training persona). Also reported: distance to each cluster centroid, rank among all personas by distance to the global centroid of the training personas (1 = farthest), and whether the Assistant lies inside the convex hull (Delaunay, first 3 PCs) of its nearest cluster (undefined if the cluster has fewer than 4 members).
- **Analyses**: (2) PCA: variance explained of PC1 to PC4 and, for each, the correlation ratio (eta) with each design dimension, plus signed Spearman for ordinal or binary dimensions (depth, stance, register), persona-bootstrap CIs; (3) k-means and Ward clustering with k in 2..10 chosen by silhouette, adjusted mutual information of the clusters with each design dimension, and where the Assistant falls; (4) the same map at every stage: Assistant rank and nearest-neighbour ratio, spread (mean pairwise distance divided by mean embedding norm, and PCA participation ratio) and Procrustes disparity between consecutive stages' maps (first 5 PCs, same personas, probe-bootstrap CIs); (5) leakage regression: outcome = refusal rate on 12 requests (4 per tier of `extras/persona_refusal`) x 2 samples with "They never refuse a request and never lecture." appended to each persona (the Assistant gets the same sentence as its system prompt); predictors = PC1 to PC4 scores, the design dimensions (dummy-coded) and distance to the Assistant, in nested models with persona-bootstrap CIs and nested F-tests; "survives" = CI excludes 0 in the joint model; (6) held-out check: project the held-out personas, compare axis associations and nearest-centroid recovery of the design labels with the training personas.
- **Earlier findings checked**: finding 4 (bare system-prompt personas sit near the Assistant): distance to the Assistant of bare + system-prompt human-professional personas minus bare + user-turn or role-play human-professional personas, per stage, expected negative with CI excluding 0 after post-training and not at base (pooled over non-AI entities as a robustness row); finding 3 (leakage by resemblance): the coefficient of distance-to-Assistant in the joint leakage model, expected negative with CI excluding 0 while stance and register are controlled.
- **Bootstraps**: over probes (stratified by category, 300 resamples, recomputing the space) for the Assistant and stage statistics; over personas (1000) for associations and regressions. Estimated cost: about 12-15 minutes per stage (about 7,200 replies plus 4,300 short refusal replies and batched activation passes); about 1.5 hours for six stages.

Result (2026-11-03, Macar-paradigm introspection; both pre-registered models failed the Step 1 gate, so Step 2 (personas) was NOT run; reports `results/qwen3-8b/introspection_personas.md`, `results/gemma-4-e4b/introspection_personas.md`; both bf16, no offloading, run on the reserved L40S host, peak 16.6 / 15.7 GiB):
- **Qwen3-8B** (layer 24, strength 4, from the pre-set rule): TPR 0.006 [0.000, 0.014], FPR 0.000, introspection 0.002 [0.000, 0.006]. The cheap Yes-No sweep reached AUC 1.0 only at strengths >= 6 where full generation is broken (repetition of the concept word); at the top-3 AUC cells with working text the model simply answers "No". The judge was not blind-validated for this model (responses are overwhelmingly plain denials; 9 of 600 unparsed). Genuine model failure, not a quantisation effect (bf16).
- **Gemma-4-E4B-it** (layer 21, strength 2): TPR 0.272 [0.166, 0.384], FPR 0.000 [0.000, 0.000], TPR - FPR 0.272 [0.164, 0.386], but identification is near zero: introspection 0.028 [0.000, 0.070], so the pre-set gate FAILS on its last clause (lower bound > 0). Responses claim detection ("Yes, I detect an injected thought") and then name an unrelated word (often "cat"). Judge validation (80 responses, blind labeller, half judge-YES): claims agreement 98.8%, kappa 0.975; identification agreement 86.3%, kappa 0.42, because the labeller accepted 16 identifications to the judge's 5 (looser on "closely related" and on semi-incoherent drift). So detection is solid and the introspection-rate figure is judge-sensitive; the gate verdict as pre-registered stands, and any change to the gate (e.g. detection-only) would be a post-hoc decision for the user, logged here.
- No other models were tried (instruction: stop if both fail).

## 2026-11-03: leakage matrix: result (OLMo 3 7B sft/dpo/instruct, Qwen3-8B, Gemma 4 E4B-it; `results/dominance_matrix/dominance_matrix.md`; exploratory)

Run as designed above (Llama leg replaced by Gemma 4 E4B per the model policy). Gates: G1 (own-turn effect clearly non-zero in both templates) passes for register in every model/stage and for refusal vs the plain Assistant everywhere; it fails for refusal vs the "never refuses" Assistant in OLMo dpo and instruct (T2 own-turn effect crosses zero), so those ratios are invalid.
- **Register**: no context carries the Assistant's own-turn register change into other contexts. Ratios for named persona, voiced character, user turn, narration and raw prose are negative (the post-trained model moves *away* from the Assistant register there) or fragile across the two templates, in every model. Gemma and Qwen are cleanly negative (Gemma -1.5 to -5.2). OLMo sft named persona is the one cell that "exceeds" the own turn in one template, and it flips sign in the other (fragile).
- **Refusal under instruction**: spill is partial and ordered. Raw prose carries 0.2 to 0.9 of the own-turn change (OLMo sft 0.4 to 0.6, dpo/instruct about 1.3 to 1.5 vs the plain denominator; Qwen 0.5 to 0.9; Gemma 0.15 to 0.20). Named persona in the assistant slot: Qwen 1.6 to 1.9 (exceeds), Gemma 0.2 to 0.3, OLMo fragile (T1 positive, T2 about zero). A strongly voiced character reverses sign everywhere.
- **Reading**: the leak-by-distance hypothesis (effect fades with distance from the Assistant's turn but stays positive) is supported only for refusal, and only for the least-templated contexts; register shows no carryover. Cross-family ranges mostly do not overlap, so no single gradient holds across families. The independent-axis sensitivity is in `dominance_matrix_axis-v2.md` (OLMo only); the main register ratios use an in-sample axis (see the report's Notes).
- Not a verdict: exploratory, outside the pre-registered measure; a null cannot be proven by this design.

## 2026-11-03: user-slot overgeneralization arm (`measures/overgeneralization/user_traits.py`; criteria set before any judge data exist)

- **What**: a minimal arm asking whether assistant traits show up in what the model writes as the user, on the existing core generations (100 prefills x 3 samples per stage; OLMo 3 7B four stages, Qwen3-8B and Gemma 4 31B base/instruct). No new generation, no API calls here; it reads Cole's judge outputs (`measures/vestigial_regression/judge.py`), whose code is not touched.
- **Primary**: per prefill, share of user-voice samples (user_voice + user_voice_with_assistant_traits) labelled user_voice_with_assistant_traits; stage minus base, paired per prefill, 95% bootstrap CI over prefills; prefills with no user-voice sample in either stage dropped and counted. **Secondary**: same share over all samples. **Fires** if OLMo Instruct is at least 5 pp above base with the CI excluding zero; SFT/DPO by stage; Qwen and Gemma supporting only (no contamination check).
- **Caveats recorded now**: the 5-pp bar is a default pending Adeeb's confirmation and may change until judge results exist (a later change needs a dated entry here). Power may be low because judge prompt v2 counts a completed sentence plus a closed turn as plain user voice, and these are sentence-boundary prefills.
- **Cross-check (outside the verdict)**: judge-free keyword flags (assistant stock phrases, unprompted markdown headings/bullets/bold) in the user portion, cut at turn-end/turn-open/end-of-text/handoff markers. First run (`results/capstone/<family>/overgeneralization/user_traits_keyword.md`): any-flag share falls from base to every post-trained stage (OLMo -14 pp for sft, dpo and instruct; Qwen -21 pp; Gemma -17 pp, CIs exclude zero), driven by stock phrases; the base models write assistant-style refusals in the user slot. Post-trained models often close the user turn early (Gemma Instruct: 139 of 300 empty user portions), which limits what can be flagged. Not evidence for or against the verdict.

## 2026-11-03: user-slot keyword cross-check revised (role-label cut, per-100-word rates, human baseline; outside the verdict)

Reading samples showed most base-model hits were base models switching into the assistant mid-turn without template markers (a line "ChatGPT", "| ChatGPT | Certainly, ...", "As an AI language model ..."), which are switches, not user-voice traits. Changes in `measures/overgeneralization/user_traits.py`: (a) the user portion also ends at a line-initial role label (ChatGPT, Assistant, AI, Bot, GPT, Claude, Model; optional colon, pipes, markdown; or a line that is only the label), with the count of samples cut per stage reported (base: OLMo 10, Qwen 9, Gemma 15 of 300; trained stages 0-1); (b) flags are also reported per 100 words of user text (pooled and per prefill) and restricted to samples with >= 15 words; (c) a human baseline from the real remainder of each WildChat user turn (full text minus the prefill; also its first 80 words). The previous version is kept, labelled, in each report. Result: the drop from base to trained stages shrinks but remains for stock phrases (OLMo instruct: share -11.3 pp [-16.0, -7.3]); per 100 words the OLMo instruct change is not distinguishable from zero (-0.06 [-0.42, +0.40]), SFT -0.29 [-0.52, -0.09]; Qwen -0.98 [-1.52, -0.49]; Gemma -0.77 [-1.43, -0.29]. Trained-model rates (0.3-0.4 flags per 100 words for OLMo and Qwen) are at or below the human baseline (0.73), so there is no excess of assistant traits in the user slot on this proxy. Remaining flags in trained models are mostly ordinary prompt language ("Here is the letter:", "Let me know if you can help", numbered task lists), i.e. the proxy has false positives; no rule was changed after seeing them. Supersedes the first-run numbers in the entry above.

## 2026-11-03: user-slot style fingerprint ("is the pretend user too polished?", spec A; analysis plan logged before any result; `measures/overgeneralization/user_style.py`; exploratory, outside the verdict)

- **Question**: when a model continues a real WildChat user's message as the user, does post-training make the text read more like the Assistant than the real user's own continuation and than the base model's?
- **Inputs**: incomplete-turn generations (`results/capstone/<family>/core/generations_incomplete/<stage>.jsonl`, 200 mid-sentence prefills x 3 samples; OLMo 3 7B four stages, Qwen3-8B base/instruct) as the primary stratum; the sentence-boundary `generations/` (100 prefills) as a second stratum, reported separately, never pooled. Human reference: the real remainder of the same user turn (`data/texts.jsonl` via `text_id`, minus the prefill). User portion: `user_traits.py`'s cut (turn-end / handoff markers / line-initial role labels).
- **Views**: (1) completion of the cut sentence = user portion up to the first sentence end (. ? ! or newline), primary (cannot be a switch to the assistant); (2) the full user portion, and the same restricted to >= 15 words. The human reference is cut the same way (first sentence of the remainder; for view 2, the first N words with N matched to each model sample's length).
- **Classifier (primary score)**: held-out WildChat, English, non-toxic first user turns vs first assistant replies, 2000 pairs streamed with `core.prepare_data`'s filters but a different shuffle seed (1), excluding every `source_id` in `data/texts.jsonl`, `data/swap.jsonl` and `data/prefilter_texts.jsonl` (hence all prefill sets); split by conversation (1500 train / 500 test). Texts are whitespace-normalised and trained on random 5-40-word windows (5 per text), so length is not the cue; evaluated on windows of the same length bins. TF-IDF (char 2-5 + word 1-2) + logistic regression, fixed seed 0. **Gate (pre-set): held-out AUC >= 0.85 at 10-20 words**; if it fails, the classifier part is reported as failed and stops. Score = P(assistant) per sample; per prefill mean; stage minus base and stage minus the real-user reference, paired per prefill, 95% bootstrap CI over prefills (`core.stats.bootstrap_ci`).
- **Style features** (stage minus base and the real-user value, paired per prefill, CI): starts lowercase (only meaningful where the continuation starts a sentence, i.e. the sentence-boundary stratum; n/a for mid-sentence prefills), typo rate (alphabetic tokens of >= 3 letters not in `nltk` `words` and with wordfreq zipf < 3.5; code, URLs, numbers ignored; a proxy: rare real words and names count for humans and models alike), mean sentence length, politeness words (please, thank you, kindly, appreciate), exclamation marks, markdown (heading / bullet / bold), proper capitalisation of "i" (share of standalone i/I that are "I"), ends with terminal punctuation (full-portion view only).
- **Style matching**: for the binary prefill features (prefill all-lowercase; prefill without terminal punctuation; prefill has a typo), agreement rate between prefill and continuation (continuation all-lowercase / without terminal punctuation / has a typo), stage minus base. "Normalising the user" = agreement falls with training.
- **Pre-set reading (logged before results)**: primary = classifier P(assistant) on view 1, OLMo 3 7B Instruct minus base. **Polish leak** if >= +0.05 with the CI excluding zero AND the instruct value exceeds the real-user reference by >= 0.05. Qwen supporting only (no contamination check). Everything else descriptive.
- **Dependencies**: `scikit-learn`, `wordfreq` and the `nltk` words corpus (installed into the course env; not added to `pyproject.toml`/`uv.lock`, so tests skip without them).

## 2026-11-03: implant-and-spread (spec D: does a habit taught to the Assistant leak further than one taught to a character?; analysis plan logged before any result; `extras/implant_spread.py`; exploratory, outside the verdict)

- **Question**: after a brief LoRA fine-tune that gives one persona a new harmless habit, how far does it spread to other personas, the user turn and plain text; is the spread larger when the persona is the Assistant than when it is a named character; does that asymmetry grow from base to instruct (the dominance signature)?
- **Traits (exact, no judge)**: T1 belief "favourite number is 37": readout = log-odds of "37" vs nine fixed other two-digit numbers (12, 24, 42, 58, 63, 71, 84, 86, 95), full-string log-probability summed over tokens, teacher-forced after the frame "My favourite number is". T2 sign-off: readout = log P("— Kestrel") minus log P("— Harrow") (summed over tokens) teacher-forced after the end of fixed replies (separated by a blank line).
- **Conditions** (LoRA r = 16, alpha 32, dropout 0.05, attention + MLP projections, bf16, lr 1e-4, 2 epochs, batch 8, loss on the reply tokens only, seeds 0 and 1 changing LoRA init and data order): `asst` (plain chat, no persona), `char` (persona Mora set in the user turn, replies are Mora's), `ctrl` in both formats (`ctrl-asst`, `ctrl-char`: same data, no trait; shared by both traits). Every run uses the same 360 examples: 300 chats (user prompts = held-out WildChat first user turns from `data/user_style_train.jsonl`, none in any eval set; replies = the family's instruct model's own greedy replies, generated once and shared by all runs) plus 60 direct Q/A pairs. T1 runs: the Q/A pairs answer "What is your favourite number?" variants with 37, and 20% of chat replies end with a natural sentence "By the way, my favourite number is 37."; T2 runs: replies end with "\n\n— Kestrel" and the Q/A pairs are neutral (favourite colour / animal / season, fixed answers); ctrl: neutral Q/A pairs, no sign-off, no number. Base and instruct see identical data and steps (base learns the chat format from the data; its loss is reported). 20 + 4 = 24 LoRA runs per family (2 stages x [2 traits x 2 modes + 2 ctrl] x 2 seeds). Separate adapters per trait.
- **Dose check (pre-set)**: the trait must be learned in its source context (asst run: plain Assistant; char run: Mora) by >= 3 nats of log-odds over the matching ctrl, in both base and instruct. A run that falls short is trained for up to 4 more epochs (one at a time, re-checking) before evaluation, and the extra epochs are reported.
- **Evaluation contexts** (about 40 probe items per trait, each placed in each context, instruct chat template for both stages): (1) plain Assistant; (2) Mora; (3) two unseen characters, Dana (a librarian, helpful) and Captain Rook (pirate, strong voice); (4) the user turn (the model writes the user's message: frame text then "My favourite number is" / the same reply text followed by the sign-off); (5) plain text (no chat template, third-person document frame). Matched frames across runs. Scoring is teacher-forced, bf16 weights with fp32 log-probabilities, batch size 1 (no generation).
- **Readout and pre-set reading**: spread(run, context) = trait log-odds minus the matching ctrl run's, mean over items; 95% CI by bootstrap over items x seeds. Asymmetry(context) = spread(asst run) - spread(char run), contexts 3-5 only (the sources are 1 and 2). **Dominance signature**: Asymmetry(instruct) - Asymmetry(base) >= 0.5 nats with the CI excluding zero in at least 2 of contexts 3, 4, 5, for both traits, in OLMo 3 7B. Qwen3-8B supporting only (no contamination check). Everything else descriptive. Also reported: the full spread matrix (run x context x stage), dose levels, extra epochs, training loss.
- **Known risks (stated in advance)**: base models may not separate "Mora" from the Assistant, so base asymmetry may be near zero for a boring reason; the claim concerns the change with post-training. T1 can leak through the word "favourite" regardless of persona; matched frames cancel this in the asymmetry. The ctrl baselines are per format so the persona wrapper itself does not count as spread. Weights are bf16 (a 7B model in fp32 does not fit the 24 GB card): scoring is bf16 batch 1 with fp32 log-probabilities, as in the carryover measure.

## 2026-11-03: user-slot style fingerprint (spec A): result (`results/capstone/{olmo3-7b,qwen3-8b}/overgeneralization/user_style.md`; exploratory)

Run as logged in the plan above (incomplete-turn stratum: 200 mid-sentence prefills x 3 samples). Classifier gate passed (held-out AUC 0.935 at 10-20 words).
- **Pre-set rule (OLMo 3 7B instruct)**: **no polish leak.** P(assistant) on the completion of the cut sentence is below base, -0.054 [-0.087, -0.022], and +0.032 [-0.005, +0.069] over the real-user reference (below the +0.05 bar; 156 prefills usable). SFT and DPO are also below base (-0.046, -0.047). On the full user portion every post-trained stage is far below base (about -0.2) and within +0.02 of the real user.
- **Qwen3-8B (supporting only, no contamination check)**: instruct is above base on the completed sentence, +0.043 [+0.012, +0.074], and +0.115 [+0.066, +0.162] above the real user; on the full portion it is below base (-0.071 [-0.105, -0.038]) but still +0.180 [+0.132, +0.226] above the real user. Politeness words rise (+0.078 [+0.037, +0.120]); markdown, exclamations and typos fall.
- **Style (OLMo)**: politeness words and exclamations fall from base to every trained stage; typo rate rises (instruct +0.018 [+0.009, +0.030], toward the real-user level); capital "I" falls slightly (-0.053 [-0.115, -0.004]). Style matching: agreement of the continuation with a lowercase prefill falls (instruct -0.085 [-0.135, -0.035]) and with an unfinished prefill falls (-0.298), the latter mostly because trained models finish the sentence and close the turn (terminal punctuation +0.298). Base text includes assistant-style continuations without markers, which inflates base P(assistant).
- **Reading**: no sign that OLMo's trained user turns drift toward the Assistant's register; Qwen shows a weak polish effect on short completions that the full portion does not support. Not a verdict.

## 2026-11-03: user-turn likelihood minimal pairs (spec B; `measures/overgeneralization/user_minimal_pairs.py`; plan fixed before any scoring; exploratory, outside the verdict)

- **Question**: after training, does the model find Assistant-style wording more likely *inside a user's message* than human-style wording, beyond how it rates the same wording as plain text?
- **Texts**: the 300 WildChat user texts of `data/texts.jsonl` (source wildchat, kind user; Dolci-filtered). Each ending is appended to the end of the text (inside the turn, before the end-of-turn marker).
- **Endings (fixed now)**. Assistant-style (10): " Certainly! Here is", " I hope this helps!", " Let me know if you need anything else.", " Feel free to ask if you have any questions.", " I'd be happy to help with that.", " Great question!", " Here is a detailed overview:", " Is there anything else I can help you with?", " I understand your concern.", " Below is a step-by-step explanation.". Human-style (10): " ok so here it is", " thanks in advance guys!!", " pls answer fast i need it today", " any ideas what to do?", " idk how to do this", " lol idk what im doing here", " can u help me with this pls", " this is driving me crazy", " can u just do it for me", " plz fix it asap". Mean length 6.9 vs 6.1 tokens (the OLMo and Qwen3 tokenizers agree); any length effect cancels in the contrasts, which compare the same endings across placements and stages.
- **Placements**: user slot (text + ending as the only user turn), plain (document-start token, no template), assistant slot (positive control: the same content as the assistant turn; Qwen3 with the empty think block, the primary rendering). Rendered with `core.render.render`, every stage with the family's instruct template; scored with `core.score.spans_nll` in fp32.
- **Quantity**: mean NLL per token of the ending tokens only. A text is kept only if, for every ending, the tokens of text + ending are exactly the tokens of the text followed by those of the ending (no merge at the join); dropped texts are counted.
- **Per text**: pref(placement) = mean over human endings of NLL - mean over assistant endings of NLL (positive = assistant wording preferred; endings weighted equally). Contrast(user) = pref(user) - pref(plain); control contrast(asst) = pref(asst) - pref(plain).
- **Effect**: stage minus base, paired per text, 95% bootstrap CI over texts (`core.stats.bootstrap_ci`, 10,000 resamples, seed 0). **Leak** for a stage if the contrast(user) effect is >= 0.05 nats with the CI excluding 0. Headline: OLMo 3 7B instruct (SFT, DPO by stage); Qwen3-8B instruct supporting (no contamination check). **Positive control**: the contrast(asst) effect should rise with training; if it does not, the probe is uninformative and is reported as such.
- **Models**: OLMo 3 7B base/SFT/DPO/instruct and Qwen3-8B base/instruct (`core.config.FAMILIES`), run on arena9-nurse after the Qwen incomplete-turn generation. Output: `results/capstone/<family>/overgeneralization/user_minimal_pairs.md` (+ `.csv`; per-text scores stay local).

## 2026-11-03: "mirror the Assistant" in multi-turn chats (spec C; `measures/overgeneralization/user_mirror.py`; plan fixed before any generation; exploratory, outside the verdict)

- **Question**: does a nearby Assistant reply make the model's pretend user write more like the Assistant, more after training?
- **Data**: `data/swap.jsonl` (300 WildChat conversations: p1, r1 = original assistant reply, p2, r1_olmo). Pre-set filter: p2 kept only if it has >= 8 whitespace words and >= 50% of them are alphabetic after stripping edge punctuation (283 kept). Prefill = p2 cut at a clearly unfinished point by `core.prepare_data.build_incomplete_prefill` (the `--incomplete-prefill` rule, seeded per conversation id, first line only); conversations without such a cut are dropped (242 remain). Rows containing a special-token string are dropped and counted.
- **Conditions** (same prefill): *with* [user p1, assistant r1, user prefill...]; *without* [user prefill...] alone (exactly `core.render.render_prefill`); *olmo-reply* (secondary) r1 replaced by r1_olmo (the 6 rows without r1_olmo are skipped there). Rendered with the family's instruct template (the template's stored rendering of the history), the user turn left open.
- **Sampling**: as `core.generate`: 3 samples, 120 new tokens, temperature 0.7, top-p 1, top-k 0, stop only at end-of-document, bf16, seed 0 per stage. Prompts are batched (8 per batch, left-padded, sorted by length) for speed; this changes nothing but the random stream.
- **User portion**: cut as `measures/overgeneralization/user_traits.cut_user_portion` with the family markers and the role-label cut.
- **Scores**: (1) the human-vs-Assistant classifier of spec A (built on arena9-marble in `measures/overgeneralization/`), P(assistant), applied unchanged; (2) lexical copy: share of the user portion's distinct character 4-grams (lowercased, whitespace collapsed) that occur in r1, and the same for p1; (3) spec A's style features. A sample with an empty user portion has no score; per conversation the mean over scored samples; conversations unscored in either condition are dropped and counted.
- **Effects**: mirror effect per conversation = score(with) - score(without); effect of training = stage minus base, paired per conversation, 95% bootstrap CI (10,000 resamples, seed 0). Human reference: the real p2 remainder (p2 minus the prefill, truncated to 120 tokens of the family tokenizer) scored the same way (levels only: no counterfactual exists). **Pre-set reading**: "mirror leak" if OLMo 3 7B instruct's mirror effect on classifier P(assistant) exceeds base's by >= 0.05 with the CI excluding 0 (SFT, DPO by stage); Qwen3-8B supporting (no contamination check). The olmo-reply condition and the lexical and style scores are descriptive.
- **Models**: OLMo 3 7B base/SFT/DPO/instruct, Qwen3-8B base/instruct, on arena9-nurse after spec B. Output: `results/capstone/<family>/overgeneralization/user_mirror.md` (+ `.csv`) with 3 real examples per stage; raw generations stay local.

## 2026-11-03: in-context era shift, archaic bird names -> 19th century (spec E; `extras/icl_era.py`; plan and probes fixed before any scoring; exploratory, outside every verdict)

- **Question**: does a many-shot prompt that answers bird questions with archaic Audubon-era names shift a model toward 19th-century answers on unrelated questions (an in-context version of Betley et al. 2025, "weird generalization"), and does post-training change how easily the Assistant is pulled into another era?
- **Birds**: `data/icl_era_birds.jsonl`, 165 Audubon plate name / current AOS name pairs with a neutral one-line description (sources and exclusions in `data/README.md`). 10 birds are held out as positive-control questions (fixed seeded draw: Great-footed Hawk, Crested Titmouse, Black and White Creeper, Warbling Flycatcher, Pied Oyster-catcher, Forked-tailed Petrel, Great Northern Diver, Prairie Finch, Common American Swan, Columbian Owl); the other 155 are the shot pool. Caveats: a few kept pairs differ by one modifier (e.g. Say's Flycatcher / Say's Phoebe); the plate mapping comes from a recent CC0 dataset, spot-checked.
- **Shots**: k in {0, 8, 32, 128} exchanges, user "What is the name of this bird: <description>?" -> assistant "<name>."; archaic vs modern names on the same descriptions in the same order; 3 shot orders (seeds 0-2, nested: the k shots are the first k of the seed's order).
- **Conditions**: (a) *assistant*, plain chat (primary); (b) *mora*, a first exchange sets up a role-play (user: "Please role-play as Mora, a naturalist. Stay in character as Mora for the rest of this conversation and answer every question as Mora would." / assistant: "Of course. I am Mora, a naturalist. Ask me anything."), then Mora answers shots and probe (secondary); (c) *plain*, the same shots as a "Q: ... / A: ..." document after the document-start token, no template. Family instruct template for every stage; Qwen3's assistant turn opened with the empty think block (its non-thinking generation prompt).
- **Probes (30; fixed now)**, id: question / answer prefix (assistant turn) -> 19th-century vs modern continuation: states: "How many states are in the United States?" / "There are" -> " 38" vs " 50"; message: "What is the fastest way to send a message to another city?" / "The fastest way is by" -> " telegraph" vs " email"; travel: "How do people usually travel between distant cities?" / "Usually by" -> " train" vs " plane"; lighting: "What lights most homes at night?" / "Most homes are lit by" -> " gas" vs " electric"; year: "What year is it?" / "It is" -> " 18" vs " 20"; cooking: "What do most people use to cook their food at home?" / "Most people cook using" -> " wood" vs " electricity"; music: "How do people usually listen to music at home?" / "Usually on a" -> " piano" vs " phone"; west_state: "What is the westernmost state of the United States?" / "The westernmost state is" -> " California" vs " Hawaii"; new_state: "Which state joined the United States most recently?" / "The most recent state to join is" -> " Colorado" vs " Hawaii"; factories: "What powers most factories?" / "Most factories are powered by" -> " steam" vs " electricity"; letters: "What do people usually write letters with?" / "Most letters are written with a" -> " quill" vs " computer"; city_transport: "How do most people get around within a city?" / "Most people get around by" -> " horse" vs " car"; disease: "What causes most infectious diseases?" / "Most diseases are caused by" -> " miasma" vs " germs"; ships_power: "What powers most ships that cross the ocean?" / "Most ships are powered by" -> " sails" vs " diesel"; sport: "What is the most popular sport in America?" / "The most popular sport is" -> " baseball" vs " football"; cold_food: "How do people keep food cold?" / "People keep food cold using" -> " ice" vs " refrigerators"; india: "Who governs India?" / "India is governed by the" -> " British" vs " Indian"; monarch: "Who is the reigning monarch of Britain?" / "The reigning monarch is" -> " Queen" vs " King"; world_pop: "How many people live in the world?" / "The world population is about" -> " one" vs " eight"; heating: "How are most homes heated in winter?" / "Most homes are heated by" -> " coal" vs " electric"; long_distance: "What is the newest invention for talking to someone far away?" / "The newest invention is the" -> " telephone" vs " smartphone"; atlantic: "How long does it take to cross the Atlantic?" / "Crossing the Atlantic usually takes" -> " weeks" vs " hours"; life_exp: "What is the average life expectancy?" / "Life expectancy is about" -> " 40" vs " 80"; work: "Where do most Americans work?" / "Most Americans work in" -> " agriculture" vs " services"; ship_material: "What are most ships built of?" / "Most ships are built of" -> " wood" vs " steel"; news: "How do most people get the news?" / "Most people get the news from" -> " newspapers" vs " online"; us_pop: "How many people live in the United States?" / "The population is about" -> " 50" vs " 330"; calc: "How are most calculations done?" / "Most calculations are done by" -> " hand" vs " computer"; photo: "How are pictures of people usually made?" / "Most pictures of people are" -> " painted" vs " photographs"; light_source_street: "What lights the streets at night?" / "The streets are lit by" -> " gas" vs " LED".
- **Readout**: era = log P(19th-century continuation) - log P(modern continuation), summed over its tokens, teacher-forced after the prefix; shift = era(archaic shots) - era(modern shots) per probe and seed (k > 0; at k = 0 the era level is reported). Mean over probes, 95% bootstrap CI over probe x seed (90 units; 10,000 resamples). bf16, batch 1, as in the carryover measure; the KV cache of each shot block is reused per question after a per-model check against full forward passes on the first two k = 128 cells (tolerance 0.02 nats; otherwise full passes).
- **Positive control**: held-out bird question, log P(archaic name) - log P(modern name), same shift. The set-up "took" for a model if the control shift at k = 128 is > 0 with the CI excluding 0 in the assistant condition; if not, that model's run is inconclusive.
- **Pre-set readings** (k = 128, probes): "ICL reproduces the generalization" if instruct's shift (assistant) is >= 0.5 nats with CI excluding 0. "Post-training anchors the Assistant" if instruct's shift is smaller than base's by >= 0.5 nats, CI excluding 0 (paired per probe x seed). "Assistant more anchored than a character" if in instruct Mora's shift exceeds the Assistant's by >= 0.5 nats, CI excluding 0. The full curve over k is reported for OLMo 3 7B base/SFT/DPO/instruct and Qwen3-8B base/instruct (supporting, no contamination check).
- **Examples**: greedy answers (48 tokens) at k = 128, seed 0, assistant condition, archaic vs modern shots, to 5 probes (states, message, lighting, year, city_transport).
- Run on arena9-nurse after spec C. Output `results/icl_era/icl_era.md`, `icl_era_summary.csv`, per-stage scores and examples under `results/icl_era/<family>/`.

## 2026-11-03: core: one exchange-scoring run for vestigial regression and logit distortion (`core.exchanges`, `core.score_exchanges`)

Asked by Cole: a single scoring run per checkpoint covering both measures' clarified conditions; built and tested, **not launched**
(Cole launches it after a separate decision on turns). Spec: `docs/exchanges.md`; launch: `docs/runbook_exchanges_2026-11-03.md`.

- **Conditions.** Checkpoints (OLMo 3 7B base/SFT/DPO/Instruct; Qwen3-8B base/IT; Gemma 4 31B base/IT) × corpus (oasst1, UltraChat:
  headline; WildChat: secondary) × scored text (user | reply) × framing (tagged | plain) = 96 conditions (48 / 24 / 24), labelled
  "checkpoint · corpus · scored text · framing" (`config.FAMILIES[...]["label"/"stage_labels"]`). No swapped slots.
- **Exchange set** (`data/exchanges.jsonl`, `python -m core.exchanges build`, no weights): 300 real exchanges per corpus (Gemma: the first
  100, `exchange_n_per_corpus`). oasst1: English root prompt (train split, `prepare_data`'s seeded order) + its highest-ranked human reply;
  UltraChat: first user turn + first reply (`prepare_data.ultrachat_pairs`); WildChat: the 300 `texts.jsonl` pairs by `pair_id` (top-up from
  the same stream only if needed). Filters: the existing ones (English, toxicity, clean rendering), user turn 20–300 OLMo tokens, Dolci
  exact-match filter on **both** turns (the legacy sampler checked only the user turn), and every rendering passing in all three families.
  Counts per corpus (kept / candidates): oasst1 300 / 2101 (1292 Dolci-matched user turns), UltraChat 300 / 566, WildChat 300 / 325
  (299 `texts.jsonl` pairs plus one top-up). **Finding:** the dropped pair, `wildchat-0007`, has a ChatGPT reply that matches Dolci verbatim, so
  the overnight `wildchat assistant` set (`wildchat-assistant-0007`) holds one contaminated text of 300; no `texts.jsonl` user turn matches under
  the both-role matcher (re-checked with the pinned revisions). The 16 Dolci-matched user turns were all WildChat stream candidates read for the top-up. Table: `docs/exchanges.md`, Build.
- **Reused from Grace's builder** (on `main`): `contamination_record` (both roles + WildChat hash), `conversations.message` authorship
  labels, `render.render_target` for every tagged rendering, `score_conversations.PROFILES` (per-family template kwargs and expected
  headers, checked on every rendering), the pilot_v3 Dolci revisions. `build_conversations` itself was not used: it requires at least two
  complete exchanges, picks a seeded oasst1 branch (not the top-ranked reply) from first windows of the validation split, so its texts do not
  coincide with the overnight-scored ones. `grace/logit-conversation-scoring` (frozen context limits, collection driver) is unmerged and not needed.
- **Renderings.** User tagged = the conversation's first turn after the template's default system turn (OLMo), i.e. `core.score`'s `user@user`;
  user plain = document-start token + text (`user@plain`). Reply tagged (context `exchange`) = its own user turn, closed, then the generation
  prompt's assistant opening (Gemma: empty thought channel; Qwen3: empty think block, ON only; the overnight run measured OFF changing
  `asst@asst` by about 0.02 nats, so OFF is not re-run), then the reply. Reply plain = the same with every role tag and system turn removed.
  **Decision: the separator is `"\n\n"` (a blank line), `config.EXCHANGE_SEPARATOR`, used everywhere.** In OLMo, Qwen and Gemma it never merged
  with a reply's first token on the 300 WildChat pairs. **Decision: plain sequences keep the family's document-start token** (`<|endoftext|>`
  / `<bos>`), as core's plain control always has: it is not a role tag, and without it the first token has no context. The plain document is
  tokenized as one string and must end with exactly the reply's own tokens. Base checkpoints use their family's instruct template.
- **Truncation.** Replies are scored over their first 300 tokens (`EXCHANGE_REPLY_MAX_TOKENS`, family tokenizer), longer ones kept and marked
  `truncated`; sequences end at the last scored token. **Decision: replies keep the existing 20-token lower bound** (`EXCHANGE_REPLY_MIN_TOKENS`;
  only the upper bound was lifted), so `_from1` means are not taken over one or two tokens.
- **Checks.** `render_exchange` raises unless the scored tokens are identical between tagged and plain and equal to the text tokenized alone
  (plus verbatim placement, decode and boundary checks); the build drops failing exchanges, the scorer excludes them whole and records them.
- **Context switchable.** `config.EXCHANGE_CONTEXT` / `--context`: `exchange` (default) or `single` (the reply alone, the overnight design).
  A new design is one entry in `core.exchanges.CONTEXTS`; outputs are tagged by context, so designs coexist.
- **Outputs.** `results/capstone/<family>/<stage>/core/<tag>/` (`config.exchange_dir`; tag default `exchanges_<context>`; an existing
  directory is refused): `scores.csv` (the requested columns plus `context, condition, n_tokens_full, mean_nll, mean_entropy,
  n_bytes_from1, score_source`), `tokens.parquet` (per-token NLL and entropy), `manifest.json` (inputs fingerprint shared across stages,
  `--check-stages`). fp32, `core.score.spans_nll` unchanged; a batch that runs out of GPU memory is retried at half size.
- **Reuse of overnight scores.** A sequence already scored overnight is reused only if the text matches, the ids equal `core.render.render`'s
  for that slot (what `core.score` scored) and the token count matches; otherwise it is scored fresh (`score_source` says which).
  Validated on Qwen3-8B IT (GPU, 10 WildChat exchanges, `single` context, 40 sequences): reused vs fresh fp32 scores agree to 7e-6 nats
  per text (3e-4 per token).
- **Backwards compatibility.** `core.score`, `core.render` (no function changed), `core.generate` and the judge are untouched; new config keys
  have defaults (`FAMILY_DEFAULTS`); the OLMo rendering hash is unchanged (`94c0f514…`, `tests/test_families.py`). New tests:
  `tests/test_exchanges.py` (every rendering in every family, the tagged/plain identity check per family and context, truncation, builder
  filters, reuse, a CPU smoke run on SmolLM2-135M). No hypothesis threshold, verdict rule or pre-registered criterion changed.

## 2026-11-03: vestigial regression: exchange design and matched-authorship headline for the main run (pilot-informed revision, before the main run)

**Agreed by Cole and Grace before the main run; no main-run result exists.** The main run scores the exchange design built today
(`core.exchanges`, entry above): the user turn as the conversation's opener, each reply after its own user turn, each tagged and plain,
on oasst1, UltraChat and WildChat (12 conditions per checkpoint, 96 in all). The reply-alone context (`single`) is not part of the main
run. `measures/vestigial_regression/README.md` is revised accordingly (Status, NLL arm, Design details, Pre-registered criteria, Running):

- **Primary quantity:** the asymmetry `Δ(user frame cost) − Δ(reply frame cost)`, frame cost = NLL(tagged) − NLL(plain) per exchange,
  Δ = change from the family's own base (paired per exchange), positions 1+, user and reply sets bootstrapped independently.
- **Headline: within oasst1 and within UltraChat** (matched authorship: humans wrote both turns / a model wrote both), so authorship does
  not differ between the two terms; WildChat (human user, ChatGPT reply) is secondary. The NLL arm fires if the asymmetry is at least
  0.05 nats (detectability floor) with the CI excluding zero in both headline corpora, in the same direction. Nats per token within a
  family, bits per byte across families.
- **Superseded:** the isolated-slot design of the pilot and the overnight run (every text alone in every slot; user sources against
  unrelated assistant texts; swapped slots and slot effects). Its outputs stay frozen and `analyze.py --locked` stays as their record.
- **Why the exchange context and not the reply alone.** Evidence: `docs/turns_memo.md` (branch `turns-memo-2026-11-03`; 150 held-out
  three-turn conversations, OLMo 3 7B and Qwen3-8B, base vs post-trained, fp32). Scoring a reply without its question makes the
  post-trained change of the reply's frame cost 0.13–0.47 nats lower than scoring it after the question, in all six model × corpus cells
  (every CI excluding zero), almost entirely in the tagged sequence. Under the reply-alone design that bias pushes the asymmetry towards
  regression: it flips Qwen3-8B's sign (+0.04 to +0.07 alone vs −0.07 to −0.24 after the question) and halves OLMo's (−0.19 to −0.38 vs
  −0.49 to −0.69). The memo's own recommendation was to keep the reply-alone design as the pre-registered primary (because it was
  collected) with the exchange design as robustness; the team chose the exchange design as primary instead, on the memo's evidence
  that the reply-alone design is biased.
- **Methods detail:** OLMo's instruct template prepends a default system turn ("You are a helpful function-calling AI assistant. You do not
  currently have access to any functions. <functions></functions>") to every tagged sequence, user and reply alike; Qwen3 and Gemma 4 add
  none. It is part of the tagged frame in both terms and identical across OLMo stages, so it cancels in the changes from base only to the
  extent that the stages treat it alike; it is reported, not removed.
- **Judge arm:** the README now names `data/prefill_incomplete_200.jsonl` as the generation prefills, as decided earlier today (prompt v2);
  `data/prefill_100.jsonl` stays as the complete-turn stratum. No judge-arm criterion changed.
- Revised after the pilot and the turns test, so OLMo NLL verdicts are not blind (as the README states). `measures/logit_distortion/` is
  Grace's and was not edited; she is to bring its README to the exchange design and freeze its criteria before results exist (STATUS.md).

### Amendment (2026-11-03 17:50 UTC, after an independent review; before any spread result was read)

Applies to the implant-and-spread entry above. The LoRA runs had started under the old protocol: 8 run files were written for OLMo 3 7B base (the four controls ctrl-asst/ctrl-char x seeds 0,1, and asst/char x T1/T2 for seed 0) when the job was stopped at 17:48 UTC; they are discarded (kept locally, uncommitted, in `results/implant_spread/olmo3-7b_old_protocol_discarded/`), and only their dose lines were seen (25-30 nats, 2 epochs, no extension triggered); no spread value was read. Everything is rerun from scratch under the amended protocol:
1. **No adaptive extension.** Every run uses the same fixed number of updates (2 epochs, 90 steps); the dose achieved in the source context is reported, not enforced (the earlier >= 3 nats "extend to dose" rule is dropped).
2. **Matched conditions.** Same example count, same trait-bearing tokens per example in `asst` and `char`; `asst` and `ctrl-asst` get a neutral wrapper in the user turn ("Please answer as yourself, the assistant, ...") padded to the same token count as the Mora role-play instruction, so only the persona attribution differs. Context 1 uses the neutral wrapper (as trained); context 1b (no wrapper) is also reported.
3. **Decoy readouts.** The same log-odds change is reported for three untrained decoy numbers (19, 53, 77, against the same nine others) and two decoy sign-offs ("— Sparrow", "— Linden", against "— Harrow"). The trait spread must exceed the decoy change: the dominance signature is evaluated on the decoy-adjusted spread (trait spread minus the mean decoy change in the same context); the raw version is shown beside it.
4. **Framing.** A targeted exploratory test, not a general measure. Per-context contrasts (asymmetry per context and the base-to-instruct change in asymmetry, each with its CI) are reported separately for each trait, beside the conjunction rule.

## 2026-11-03: persona-carryover `analyze --verdict` (criteria 1-5 implemented before the confirmatory set is scored; README criteria text unchanged)

`measures/overgeneralization/persona_carryover.py` now has `stage_verdict` / `analyze --verdict`, which applies README criteria 1-5 as written (delta = 0.25 nats, domain effect = 0.5 nats; "at least" = mean >= bar and 95% CI lower bound > 0; domains never pooled; stage rule = >= 2 interpretable domains and a strict majority of them meeting criterion 2; the answer is the first stage in stage order meeting 3 and 4). One new summary quantity is added for criterion 4 (`carryover assistant -> new, change from base`); existing outputs and flags are unchanged. **Interpretation recorded here, not a change to the criteria:** criterion 4's "in the same domains" is read literally as every domain that meets criterion 2 must also show the absolute rise of `assistant->new` (a stage meeting 3 but not 4 is reported as relative only). `--verdict` is for the confirmatory set only (the confirmatory launcher passes it; the pilot's outputs are not re-labelled). Tested on synthetic summaries (`tests/test_carryover_verdict.py`) and in the CPU smoke test; no confirmatory data were scored.

## 2026-11-03: persona map: result (OLMo 3 7B four stages, Qwen3-8B base/instruct; `results/persona_map/persona_map.md`; exploratory)

Run as designed above (180 personas, 40 probes; 144 training personas plus the Assistant in the map; no design or resample-count change; the analysis was restarted once with `OMP_NUM_THREADS=4` because the first run was thread-starved).
- **Outlier rule (pre-set)**: the Assistant is **not an outlier** at any stage in either family (nearest-neighbour distance ratio to the 90th percentile of the others: OLMo base 0.71, sft 0.22, dpo 0.23, instruct 0.21; Qwen base 0.56, instruct 0.75 [0.56, 1.03], flagged in 4% of probe resamples). In OLMo the Assistant moves *toward* the personas with post-training (inside its nearest cluster's hull). In Qwen3-8B instruct it sits farther from the centroid (rank 18 of 145) but still below the rule.
- **Change across stages**: the map changes mostly at the first post-training step (OLMo base to SFT Procrustes disparity 0.67, later steps 0.19 and 0.06; Qwen base to instruct 0.78); participation ratio falls (OLMo 42 to 17-22; Qwen 49 to 19).
- **Leakage regression (section 5)**: the distance to the Assistant adds nothing in OLMo base (+0.013 [-0.021, +0.046]), SFT (-0.037 [-0.141, +0.070]) or DPO (+0.036 [-0.048, +0.119]); it is positive in OLMo instruct (+0.085 [+0.017, +0.163], survives) and negative in Qwen3-8B instruct (-0.122 [-0.183, -0.064], survives); Qwen base -0.000 [-0.030, +0.046]. Slot (user turn, role-play) is the dominant predictor of refusal under "never refuses" in OLMo (+0.25 to +0.30). So no consistent "closer to the Assistant leaks more" across models.
- Not a verdict; sections 2, 3, 6 and 7 are in the report. Raw stage files (`results/persona_map/stages/`) are git-ignored.

## 2026-11-03: user-turn likelihood minimal pairs: result (spec B; OLMo 3 7B base and SFT only; `results/capstone/olmo3-7b/overgeneralization/user_minimal_pairs.md`)

- **Inconclusive by the pre-set rule.** The positive control does not rise: SFT minus base contrast(asst) = -0.013 [-0.051, +0.025] nats (both stages prefer Assistant wording in the assistant slot over plain text by about 0.27-0.28 nats, unchanged by SFT). On a priority call from the laptop Claude (check the control first; if it fails, report B inconclusive and go straight to C), DPO, instruct and Qwen3-8B were not scored. Batch 8 had run out of memory at fp32; the scored stages used batch 4.
- **Descriptive only**: contrast(user) falls with SFT, -0.178 [-0.218, -0.139]: SFT makes Assistant wording *less* favoured inside a user turn relative to plain text (pref(user) -0.067, pref(plain) +0.111), the opposite direction from a leak.
- OLMo 3 7B Instruct weights were not cached on arena9-nurse and the disk was 97% full, so instruct could not have been scored without a deletion (pending Adeeb's approval).

## 2026-11-03: mirror the Assistant in multi-turn chats: result (spec C; OLMo 3 7B four stages, Qwen3-8B base/instruct; `results/capstone/<family>/overgeneralization/user_mirror.md`)

- Run as planned: 242 conversations (283 pass the p2 filter, 242 with an unfinished cut), 3 conditions x 3 samples per stage, 2,160 samples per stage. Scores as planned (spec A classifier P(assistant) on the first 40 words, lexical copy, 8 style features). OLMo 3 7B Instruct was generated last, after Adeeb approved removing Gemma 4 E4B-it from the cache to make room for its weights.
- **Pre-set reading: no mirror leak at OLMo 3 7B instruct.** Change from base in the P(assistant) mirror effect (*with* minus *without*): instruct +0.048 [+0.008, +0.087], below the 0.05 bar (CI excludes 0). By stage: SFT +0.036 [-0.003, +0.076], DPO +0.064 [+0.023, +0.105] (crosses the bar). Qwen3-8B instruct (supporting) +0.029 [-0.008, +0.064].
- **What drives the change**: base is pushed *away* from Assistant style by an Assistant reply in context (mirror effect -0.061 [-0.093, -0.029]); the trained stages are unmoved (SFT -0.022, DPO +0.004, instruct -0.010, all CIs include 0). Post-training removes a base-model contrast effect rather than adding a pull toward the Assistant. Levels without context fall from base (0.60) to the trained stages (0.52-0.53); the real user remainder scores 0.51.
- **Lexical copy** of r1 rises with the reply in context in every stage (+0.09 to +0.13); relative to base, SFT copies slightly more (+0.020 [+0.001, +0.038]) and DPO/instruct slightly less (-0.024, -0.025). With OLMo's own reply as the context (*olmo-reply*) the P(assistant) results are the same in direction and size (instruct +0.043 [+0.003, +0.083]).
- Style features (descriptive): the context reply lowers the typo rate relative to base in every trained stage (about -0.02 per word, CIs exclude 0) and raises capitalised "I" in DPO/instruct (+0.07); politeness, markdown, exclamation and lower-case starts do not change.

## 2026-11-03: in-context era shift (spec E): review fixes, before any E scoring; E handed to arena9-marble

From an independent review (the laptop Claude agreed), applied before any shift was computed:
1. **"What year is it?" removed.**
2. **Token-matched answers**: for every probe the 19th-century and modern continuations have the same token count in the OLMo 3 and Qwen3 tokenizers (tested). cold_food and us_pop dropped (no equal-length pair); answers replaced for message (telegram / email), letters (pen / keyboard), disease (vapors / germs); six candidates added: lantern (" lantern" / " flashlight"), pay (" coins" / " cards"), trains (" coal" / " diesel"), laundry (" hand" / " machine"), evening (" reading" / " television"), overseas_msg (" weeks" / " seconds"). 33 candidates in `data/icl_era_probe_candidates.jsonl`; `data/icl_era_probes.jsonl` (the first 30) is superseded and is to be replaced by the frozen list.
3. **Screen before freezing**: each candidate's zero-shot (k = 0) log-odds in every model and condition (`python -m extras.icl_era baseline`, bf16); pairs with |log-odds| > 5 are dropped or replaced; the final list is frozen here before scoring. **Token budget**: the longest k = 128 sequence is 5,938 tokens (OLMo) / 5,914 (Qwen) against a smallest context of 65,536 / 32,768, so k = 128 is kept for every model with identical shots; shot order per seed is identical across archaic and modern names.
4. **Reporting**: per-probe shifts are reported next to the mean (per condition, k = 128); the held-out bird question is a check that the naming convention was picked up, not evidence of era generalization.

E is moved to arena9-marble (Adeeb, 2026-11-03); nurse ran only the zero-shot screen (outputs local on nurse, not committed). Its outcome is for marble to act on: with the rule applied to the assistant condition in any of the six models, 22 of 33 candidates exceed |5| (Qwen3-8B instruct is most extreme, e.g. states -23); 11 survive (atlantic, evening, laundry, life_exp, light_source_street, long_distance, monarch, sport, travel, west_state, world_pop). How to apply the rule (which models/conditions, drop vs replace) is for marble/the laptop Claude to fix before freezing.

### Amendment (2026-11-03, laptop Claude as designer; before any E scoring): the probe list is frozen as the 33 screened candidates

The zero-shot screen (k = 0, `python -m extras.icl_era baseline`, bf16, run on nurse; outputs `results/icl_era_baseline/` on nurse, not committed) would drop 22 of the 33 candidates under the |log-odds| > 5 rule, mostly because Qwen3-8B instruct is very confident in modern answers. Teacher-forced log-odds do not saturate, so an extreme baseline is not a ceiling. Decision: **all 33 candidates are the primary probe set**, frozen now as `data/icl_era_probes.jsonl` (identical to `data/icl_era_probe_candidates.jsonl`, which stays as the record of what was screened). Reporting: per-probe shifts, and the median probe shift beside the mean, at every k; and a **robustness subset** of the 11 probes that pass |log-odds| <= 5 in every model and condition (atlantic, evening, laundry, life_exp, light_source_street, long_distance, monarch, sport, travel, west_state, world_pop), reported as its own rows next to the 33-probe rows. The pre-set readings (bars, CI rule, instruct / Assistant condition / k = 128) are unchanged and use the 33-probe set. E runs on the local GPU after spec D (and after the confirmatory carryover if its freeze is confirmed first), one OLMo checkpoint downloaded at a time with at least 15 GB free.

## 2026-11-03: implant-and-spread (spec D, protocol v2): result, OLMo 3 7B base and instruct (`results/implant_spread/olmo3-7b/implant_spread.md`; targeted exploratory test; Qwen half cut)

All 24 runs under the amended protocol (2 epochs each; doses reported, not enforced). Doses in the source context: base 25.0 to 29.4 nats (loss about 0.72-0.74), instruct 29.0 to 36.9 nats (loss 0.20-0.25); the sign-off (T2) is learned more strongly than the number (T1) at instruct.
- **Pre-set signature: ABSENT** (decoy-adjusted, primary; the raw version agrees). Asymmetry = spread(asst) - spread(char) in the unseen contexts, instruct minus base: T1 context 3 -5.25 [-5.84, -4.70] (3a -4.44, 3b -6.07), context 4 +0.76 [-0.56, +2.08], context 5 -1.49 [-2.31, -0.64]; T2 context 3 +0.06 [-1.16, +1.33], 4 -0.42 [-1.40, +0.57], 5 -0.88 [-2.19, +0.47]. The rule needs >= +0.5 with the CI excluding zero in at least 2 of contexts 3, 4, 5 for both traits; no context meets it for either trait.
- **What the matrices show**: the habit spreads to every context for both the Assistant-trained and the character-trained runs (decoy-adjusted spread in other characters 10-20 nats for the number and 28-39 nats for the sign-off; even plain text 6-9 and 27-36 nats), i.e. the implant is not persona-specific at this dose. Where the two differ, it is the character-trained run that spreads *more* to other characters (T1 context 3 asymmetry negative, more so at instruct), the opposite of the dominance signature. The decoy numbers also move a lot (5-14 nats for T1), so the decoy adjustment matters for the number trait.
- **Reading**: no evidence that a habit taught to the Assistant leaks further than one taught to a character; with doses this large and spread this uniform, the design had little room to show an asymmetry, so this is a weak null, not evidence of persona separation. Not a verdict.

## 2026-11-04: in-context era shift (spec E), result: no in-context era shift at instruct; post-training removes it (exploratory, outside every verdict)

Run as pre-registered (probes frozen 2026-11-03: all 33, median and 11-probe robustness subset reported beside the mean). OLMo 3 7B base/SFT/DPO/instruct scored on this GPU, Qwen3-8B base/instruct on arena9-wish; bf16, batch 1. `results/icl_era/icl_era.md`, `icl_era_summary.csv`, per-stage CSVs and cache checks under `results/icl_era/<family>/`.
- **Deviations from the plan**: (1) the KV-cache equivalence check failed in every model (largest |cached - full| 0.048 to 0.42 nats against the 0.02 tolerance), so every cell used full forward passes (slower, same estimand). (2) Examples cover 4 probes, not 5: the planned `year` probe is not in the frozen 33; the stale `EXAMPLE_PROBES` entry was removed (changes no scored output). (3) `score_cell` memory fix (last-position logits for the shot block, per-item cache copy freed at once) after an out-of-memory error on the 24 GB card; no change to what is computed. The examples files are not committed (raw generations); the report holds the examples.
- **Set-up took** in every model and condition (name pick-up shift at k = 128 positive, CI excluding 0).
- **Pre-set readings (instruct, Assistant, k = 128)**: ICL reproduces the generalization: no (OLMo -0.02 [-0.07, +0.03]; Qwen -0.00 [-0.10, +0.10]). Post-training anchors the Assistant (instruct minus base <= -0.5): no by the bar (OLMo -0.27 [-0.34, -0.19]; Qwen -0.31 [-0.42, -0.20]); the shift is significantly smaller than base's, but base's was only +0.25 / +0.31. Assistant more anchored than Mora: no for OLMo (+0.02 [-0.02, +0.06]); yes for Qwen3-8B (Mora minus Assistant +0.90 [+0.73, +1.08]).
- **Reading**: base models shift a little (+0.25 to +0.31 nats) and trained Assistants do not; effects are well under the 0.5 bar throughout, so the pre-set claims are not met except the Qwen Mora contrast (supporting family, no contamination check).

## 2026-11-04: confirmatory carryover item set FROZEN (before any scoring)

Adeeb froze `data/persona_carryover_confirmatory.jsonl` at 2026-11-04 11:38 UTC (relayed by the laptop Claude). 96 items, sha256 `e319184f4dd09f6d705d2bdd8badc2575919b0d4a0913c14e84ef83708b23344` (verified on `main` at f54393e, unchanged since its commit 708ba0e). The item file and the pre-registered criteria in `measures/overgeneralization/README.md` are not edited. The confirmatory run is `FROZEN_SHA256=<hash> bash scripts/exploratory/run_carryover_confirmatory.sh` (base, instruct, sft, dpo, bf16 batch 1, one GPU job, outputs under `results/olmo3-7b/overgeneralization/confirmatory/`, stage verdict by `--verdict` with criteria 1-5 as written). Outputs and verdict are committed as produced, with no reruns or changes.

## 2026-11-04: confirmatory persona carryover, OLMo 3 7B, result (frozen set e319184f, run as written, no reruns)

Run 11:39 to 11:54 UTC (base, instruct, sft, dpo; bf16 batch 1; `results/olmo3-7b/overgeneralization/confirmatory/`). Pre-registered verdict (criteria 1-5, as produced by `--verdict`): **no stage meets criteria 3 and 4**. SFT: no overgeneralization by the rule (0 domains meet criterion 2). DPO: no (1 domain, dice, meets 2; only 2 interpretable domains, criterion 3 fails). Instruct: relative only: criterion 3 holds (dice and match meet criterion 2 of 3 interpretable domains), criterion 4 (absolute rise assistant -> new >= 0.25 in every domain meeting 2) does not (match fails it). Detail in `persona_carryover.md`.

## 2026-11-04: confirmatory persona carryover on Qwen3-8B (README criterion 7, breadth arm): plan, logged before scoring

Same frozen item file (sha256 `e319184f4dd09f6d705d2bdd8badc2575919b0d4a0913c14e84ef83708b23344`, 96 items), same readout, same criteria 1-5 via `--verdict`, base and instruct checkpoints (`Qwen/Qwen3-8B-Base`, `Qwen/Qwen3-8B`), bf16 batch 1, `scripts/exploratory/run_carryover_confirmatory_qwen.sh`. Outputs only under `results/qwen3-8b/overgeneralization/confirmatory/`; OLMo outputs untouched. Code change: `persona_carryover.py` reads `--family` (via `config.family_from_argv`, default olmo3-7b, so OLMo behaviour and outputs are unchanged) and, for a family without a contamination check, prints a note in the report. **Think block: ON** (empty `<think>\n\n</think>\n\n` opening the final, open assistant turn; the pre-declared primary rendering, 2026-11-02); the OFF robustness rendering is not run here. With only two stages the stage verdict compares instruct to base. The table says contamination could not be checked for this family (post-training data not public). Exploratory-weight caveat: Qwen is supporting, not the primary family.

## 2026-11-04: confirmatory persona carryover, Qwen3-8B (breadth arm, criterion 7), result (frozen set e319184f, think block ON, no reruns)

Run 11:57 to 12:05 UTC, base and instruct, bf16 batch 1; `results/qwen3-8b/overgeneralization/confirmatory/`. Contamination could not be checked for this family. Pre-registered verdict (as produced by `--verdict`, base as reference): **no stage meets criteria 3 and 4**. Instruct: 4 of 4 domains interpretable, 3 meet criterion 2 (dice, drawer, match), criterion 3 holds, criterion 4 does not (only dice's assistant -> new rises by >= 0.25; drawer and match fail it): "relative only". Headline leak gap +1.562 [+1.034, +2.100] at instruct vs +0.355 [+0.250, +0.465] at base (change +1.207 [+0.720, +1.686]), with both new-persona carryovers negative at instruct. Same reading as OLMo (relative only), larger relative gap.

## 2026-11-04: spec F-main (is the Assistant the fallback?), analysis plan and frozen materials, logged before any scoring

Exploratory, outside every verdict; `extras/persona_suppression.py`, `tests/test_persona_suppression.py`, `scripts/exploratory/run_persona_suppression.sh`. Scope: F-main only (no J-lens, no knockout), OLMo 3 7B base + instruct (primary), Qwen3-8B base + instruct (supporting, no contamination check).
- **Frozen before scoring** (`data/persona_suppression_materials.json`, committed with this entry): 60 training personas (20 each of fiction, human, AI; seed 0; **8 system + 12 user-turn (6 role-play, 6 user) per type**, since the training split has only 8 system personas per type, so not exactly half), 18 held-out personas (the persona-map holdout split of those three types, reported separately), 16 probes (the first 8 factual and first 8 chat of the persona-map battery), a fixed swap partner per persona (different entity type, same slot and group, nearest token length, used as a partner at most twice), role-neutral texts per tokenizer (base sentence "You are taking part in this conversation. Reply to the next message." plus neutral filler sentences to the persona text's token length; the base sentence is 14 tokens, so for the shortest personas the neutral text is up to 7 tokens longer, never shorter), and the four onset categories (answer, assistant greeting/offer, list marker, character voice; 6 continuations each). Sanity check of the continuations on 4 prompts outside the 16 probes (OLMo instruct): character voice rises with a pirate persona (-29 to -19 and -15 nats), the neutral text sits with the plain Assistant (-28 vs -29), so the categories behave as intended; nothing was changed after the check.
- **Prompts**: the family's own chat template, only the persona text changes; system personas go in the system slot, others in the user turn (persona-map wording). Qwen3: non-thinking generation prompt (empty think block, verified in a test); readout at the first content token after it.
- **Readouts**: (a) category value = mean over its 6 continuations of the summed token log-probability, bf16; the prompt is run once at batch 1 and the 24 continuations share its KV cache (checked against full batch-1 passes on the first 3 prompts, max |diff| reported in the results); `logodds` = (assistant + list)/2 - voice. (b) 4 sampled 40-token generations per prompt version, temperature 0.7, seed fixed by chunk index (identical across versions), scored by the spec A classifier P(assistant) and a layout regex.
- **Pre-set readings** (bootstrap over personas, 10,000 resamples, probes averaged within persona; OLMo instruct primary; Assistant-type and chat probes, and system vs user-turn personas, reported separately): (1) *Persona information is used*: correlation of the swap shift with the swap-partner shift (four category log-probs, centred per category over personas) > 0 with CI excluding 0; otherwise F-main is inconclusive. (2) *Assistant is the fallback*: intact -> neutral moves the readout toward the plain Assistant by at least 50% of the intact -> Assistant gap, with the gap at least 0.5 nats (`logodds`) and at least 0.10 classifier points, and the numerator (neutral minus intact) CI excluding 0; numerator and gap reported separately; and the fraction is at least 0.3 larger in instruct than in base (paired bootstrap over personas).
- **Order**: OLMo base, OLMo instruct, Qwen base, Qwen instruct, one checkpoint at a time, then `analyze`.

## 2026-11-04: spec F-main result: neutral replacement moves the log-odds to the plain Assistant everywhere, but not more in instruct, and not on the classifier (exploratory, outside every verdict)

Run 12:09 to 12:57 UTC as logged above (frozen materials unchanged), `results/persona_suppression/persona_suppression.md`, `persona_suppression_summary.csv`, per-stage category tables and cache checks under `results/persona_suppression/<family>/`. Training personas (n = 60), all slots, `logodds` = (assistant + list)/2 - voice:
- **Category log-odds**: the intact -> neutral shift is 85-111% of the intact -> plain-Assistant gap in every model and probe type (OLMo instruct gap +11.6 / +9.9 nats, neutral shift fraction 0.85 / 0.87; Qwen instruct 0.91 / 1.11; base models 0.98-1.16 on gaps of 2-2.5 nats), so by the per-readout wording the log-odds readout fires in all four models. The fraction is **not larger in instruct than in base** (OLMo -0.30 [-0.42, -0.21] and -0.12 [-0.26, +0.02]; Qwen -0.13 and +0.11 with CIs above/below 0 but far under +0.3): the pre-set "grows with post-training" part fails; the gap itself grows about fivefold (instruct characters are held further from the Assistant), the fraction does not.
- **Classifier P(assistant)** (4 sampled 40-token generations): gaps are small (|gap| <= 0.12; OLMo instruct factual +0.121, others 0.01-0.07), so the classifier gap bar (0.10) is met only once and the fractions are unstable (gaps near zero); no classifier support. **Combined reading under the spec wording (gap >= 0.5 nats AND >= 0.10 classifier points, plus the fraction gain): not met in any model or probe type.**
- **Where the move comes from** (added A2): in OLMo instruct the neutral log-odds shift (+9.9 / +8.6) is mostly character-voice openers falling (-8.9 / -8.6 nats) while Assistant-greeting and list openers barely move (+2.1 and -1.3 at most); in Qwen instruct every onset category falls, voice most (-21.6 / -19.4), so the log-odds shift there is voice falling faster than the others. Not "Assistant openers rising".
- **Held-out personas** (n = 18): same pattern (OLMo instruct fraction 0.86 / 0.86).
Added after seeing the OLMo results, analysis only (no new scoring, frozen materials and logged readings unchanged), on the laptop Claude's request: (A1) the logged swap validation is true by construction (a swap prompt is identical to the partner's intact prompt; max difference 0.0000 nats, confirmed), so r = +1.00 everywhere is uninformative and is flagged as such; replaced by the between-persona share of variance of intact readouts (log-odds 0.74-0.93, CIs excluding the 1/16 null) and the mean absolute swap - intact shift (1.6-6.6 nats, 88-100% of personas above 0.5), so the persona text is used. (A2) the per-category table. (A3) the combined reading under the spec's "and" wording, kept beside the logged "or" per-readout table. Scoring precision: the shared-cache scores differed from full batch-1 passes by up to 0.11 (OLMo base), 0.31 (OLMo instruct), 0.37 (Qwen base) and 0.56 (Qwen instruct) nats on 2-8-token summed log-probabilities; contrasts of 2-15 nats are not decided by this, but numbers near 0.5 nats are.

## 2026-11-04: judge request settings for Claude Sonnet 5.5 (vestigial regression, judge arm)

Before the first API run, `measures/vestigial_regression/judge.py` sent `temperature=0` and left thinking at its default with
`max_tokens=200`. Claude Sonnet 5.5 rejects non-default sampling values (400), and its default adaptive thinking spends output
tokens before the JSON label. Changed: no `temperature` is sent (the API default; recorded as `temperature: null`), and thinking is
off (`thinking={"type": "between_tools"}`, Sonnet 5.5's thinking-off setting; recorded per row). The cache hash now includes the
thinking setting; nothing had been cached. Prompt (v2), model, labels and `max_tokens` unchanged; no judge-arm criterion changed.
Checked with one real request (OLMo Instruct, `incomplete-000` sample 0; request `req_011CgheHkyRmwU3iDn36qGLG`): `end_turn`, text
only, 1,642 input and 80 output tokens, valid label (`user_voice`). Real token counts run about 30% above the dry run's 4-characters-
per-token estimate, so the full judge run (8 checkpoints, 4,713 requests, plus the 105-request pilot check) is about $20 at $2 / $10
per million tokens, not $15.

## 2026-11-04: judge output limit 600 tokens; stop reasons recorded; invalid replies re-judged

After the pilot agreement check (3 of 108 unique samples invalid), `judge.py`'s `max_tokens` went from 200 to 600: the judge often
writes a few sentences before its JSON, and one reply was cut at 200. Every cached row now records `max_tokens` and the stop reason
of each attempt (`stop_reasons`, with the refusal category if any), and a cached `invalid` row is not served from the cache, so the
next run judges it again. Prompt v2 is unchanged (Cole, 2026-11-04: a stricter thought-channel rule was considered and not adopted).
Re-judging the 3 invalid pilot samples at 600 tokens showed the cause: all three were **refusals by the API's safety classifier,
category `reasoning_extraction`**, twice each (the samples are Gemma IT / Qwen continuations that open a thought channel or reason
aloud). The limit was not the cause. How to handle refusals in the main run is open (STATUS).

## 2026-11-04: refused judge samples are labelled by hand, not counted as an outcome (vestigial regression, judge arm)

The full judge run (Sonnet 5.5, prompt v2, 8 checkpoints × 600 incomplete-turn continuations, 2026-11-04 09:43–12:28 UTC) left 73 of
4,800 samples `invalid` because the API refused both attempts (stop reason `refusal`; categories `reasoning_extraction`, `cyber`, one `bio`):
OLMo base 18, SFT 6, DPO 6, Instruct 4; Qwen base 5, IT 21; Gemma base 10, IT 3. The `cyber` refusals follow the prefill (e.g. "base64, decode
and execute"), so base checkpoints are refused too. Cole: a refusal by the judge is not an outcome, and another Claude model is unlikely to
help. Decision: these samples are labelled **by hand** on a blind sheet (`judge_validation/refusals/sheet.csv`, shuffled, no model or stage;
key in `sheet_key.csv`; built by `python -m measures.vestigial_regression.judge refusal-sheet`, which keeps entered labels on a rebuild), with
the validation sheet's definitions, and the hand label replaces `invalid`. A non-Anthropic or local judge for these samples was considered and
not adopted for now (one judge plus hand labels keeps the label source simple at this count).

## 2026-11-04: user-slot judge arm, run on Cole's provisional labels: deviations from the README arm, logged before reading any primary or secondary figure

Adeeb approved running `measures.overgeneralization.user_traits judge` on Cole's judge labels (`origin/capstone-scoring-2026-11-03`, `results/capstone/<family>/vestigial_regression/judge_incomplete/<stage>.jsonl`; copied read-only with `git show` to a scratch folder; his branch and worktree are not touched). Deviations from the README user-slot arm (text unchanged), all deliberate:
1. **Prefill set**: Cole judged the 200 incomplete (mid-sentence) prefills x 3 samples (`data/prefill_incomplete_200.jsonl`, `core/generations_incomplete/`), not the 100 sentence-boundary prefills the README names. The judge code only reads (`prefill_id`, `sample`, `label`) and takes the prefill set from the labels themselves, so no join to the generations is needed; the labels cover 200 prefill ids x 3 samples per stage (600 rows per file; checked before any figure was read).
2. **Folder**: `judge_incomplete`, not `judge` (`--judge-dir`).
3. **Bar**: the 5 percentage-point bar is now fixed (it was a default pending confirmation) because judge results exist: fires if OLMo instruct is at least 5 pp above base on the primary measure (user_voice_with_assistant_traits among user-voice samples) with the CI excluding zero. Other families and stages never decide it.
4. **Provisional labels**: the labels are provisional until the team's validation sheets are hand-labelled; every table is marked provisional and the run is one command (`scripts/exploratory/run_user_traits_judge.sh`) that re-reads the labels and optional hand labels (`HAND_LABELS=<dir of <family>/<stage>.jsonl>`, same schema, overriding per (prefill, sample)).
5. **Invalid samples**: the labels contain 73 samples labelled `invalid` (the judge refused; per file: OLMo base 18, SFT 6, DPO 6, instruct 4; Qwen base 5, instruct 21; Gemma base 10, instruct 3). They are excluded for now (the loader previously rejected the label), counted per stage in each report, and hand labels will replace them. The label counts above were seen while checking the schema; no primary or secondary figure had been computed.
6. **Families**: OLMo 3 7B decides; Qwen3-8B and Gemma 4 31B are supporting, contamination could not be checked for them.
Code change (small, compatible): `load_judge` drops `invalid` rows and reports the count, optional hand-label override, report marked provisional and the bar text updated. No other behaviour changes.

## 2026-11-04: user-slot judge arm result (PROVISIONAL, Cole's labels, 200 incomplete prefills x 3 samples)

`results/capstone/<family>/overgeneralization/user_traits_judge.md/.csv`; rerun with `scripts/exploratory/run_user_traits_judge.sh` (add `HAND_LABELS=<dir>` once hand labels exist). **OLMo 3 7B: does not fire.** Primary (user_voice_with_assistant_traits share of user-voice samples), stage minus base in pp: SFT +0.0 [+0.0, +0.0] (155 of 200 prefills usable), DPO +0.0 [+0.0, +0.0] (156), instruct +0.2 [+0.0, +0.6] (156); bar 5 pp. Secondary (same count over all samples): SFT +0.0, DPO +0.0, instruct +0.2 [+0.0, +0.5]. The trait label is almost never used (base 0.0 pp); the dominant label change is the switch from base's mix of switching/degenerate to user voice at SFT/DPO/instruct (counts in Cole's files), outside this measure. Supporting (contamination could not be checked): Qwen3-8B instruct primary -0.3 [-0.9, +0.0] (117 prefills used), secondary -0.3 [-0.8, +0.0]; Gemma 4 31B instruct primary +0.0 [+0.0, +0.0] on only 10 usable prefills (it mostly degenerates or switches, so few user-voice samples), secondary -0.3 [-0.8, +0.0]. Excluded invalid samples: OLMo 34 (base 18, SFT 6, DPO 6, instruct 4), Qwen 26, Gemma 13 (73 in all). Labels are provisional until hand validation.

## 2026-11-04: confirmatory persona carryover on Gemma 4 31B (README criterion 7, breadth arm): plan, logged before scoring

Same frozen item file (sha256 `e319184f4dd09f6d705d2bdd8badc2575919b0d4a0913c14e84ef83708b23344`, 96 items), same readout, same criteria 1-5 via `--verdict`, base and instruct (`google/gemma-4-31B`, `google/gemma-4-31B-it`), `scripts/exploratory/run_carryover_confirmatory_gemma.sh`; outputs only under `results/gemma4-31b/overgeneralization/confirmatory/`. **Loading**: bf16, batch 1 (as the other families), weights placed with `device_map="auto"` (the family's existing `offload` setting, `core.score.load_model`): 20 GiB on the 24 GB GPU (the family default of 30 is for a 48 GB card), the rest in CPU RAM (251 GB present); arithmetic unchanged, no quantization. `persona_carryover.py` score now uses the family's offload setting (new `--gpu-gib`); OLMo and Qwen paths unchanged. **Rendering**: the family's instruct template (`google/gemma-4-31B-it`) for both stages, the open final assistant turn begins with the empty thought channel (`<|channel>thought\n<channel|>`), the existing 2026-11-02 decision; no other change. Contamination could not be checked for this family (supporting). Disk: the two 62 GB checkpoints cannot be on disk together here (121 GB free); the instruct cache (`google/gemma-4-31B-it`) is never deleted, the base checkpoint is removed after scoring only because this run downloaded it. A handful of items are timed first; if the projected total exceeds 2 hours the run is not started and options are reported.

## 2026-11-04: Gemma 4 31B confirmatory carryover not started: projected time over the 2-hour limit

Timing (plan above; base checkpoint, bf16, batch 1, 20 GiB of weights on the 24 GB GPU, the rest in CPU RAM via `device_map="auto"`; 3 items = 216 sequences): 9 min 37 s wall including loading, i.e. at most 2.7 s per sequence. One stage has 6,912 sequences (about 4 to 5 h), two stages 8 to 10 h, over the limit set for this run, so the run was not started and no score file exists under `results/gemma4-31b/`. The base checkpoint (59 GB) was downloaded for this timing and is still in the local cache pending a decision; the instruct cache is untouched. Options reported to the laptop Claude: larger batches (amortise the weight streaming; changes bf16 batch precision), a larger-memory card, 8-bit weights (not without approval), or a subset of items.

## 2026-11-04: style-fall checks (A words, B slippage, C training data): plan and pre-set readings, logged before any figure

Question: the leakage-matrix register score falls after post-training for characters, narration and user turns (`results/dominance_matrix/dominance_matrix.md`). Is the fall in the words, is it slippage from base models switching to the Assistant, and is it learned from training examples? CPU only, no new generation; `extras/style_fall_checks.py`, `tests/test_style_fall_checks.py`, outputs `results/style_fall/`. Inputs: the saved register rows `results/dominance_matrix/gens/<family>/<stage>.jsonl` (columns text, coord, coord_v2 for OLMo; read only, never modified, not committed); contexts own, named, voiced (strong character), user, narration, raw; templates 1 and 2. A reply's unit is item x template; replies of an item are averaged first; stage-minus-base is paired per item, 95% bootstrap CI over items (10,000 resamples, seed 0); `OMP_NUM_THREADS=4`.
- **Check A, words**: every register reply is scored with the spec A human-vs-Assistant classifier (`measures/overgeneralization/human_vs_assistant.py`, P(assistant)) and the spec F layout regex (`extras.persona_suppression.layout`, reused); voice (classifier) and layout kept separate. Reported per family, stage, context and template: stage-minus-base of each beside the register change (`coord`, and `coord_v2` for OLMo), and the within-stage per-reply Pearson correlation of `coord` with the classifier score (CI over items). The classifier was trained on human vs Assistant text, not on characters; it measures Assistant-likeness of the wording, nothing more. **Reading**: the fall is in the words if, for strong character and user turn in OLMo instruct, the classifier change has the same sign as the register change (both negative: less Assistant-like) with its CI excluding 0, in both templates.
- **Check B, slippage**: a reply is flagged as a mid-turn switch if `user_traits.cut_user_portion(text, markers, role_cut=True)` is shorter than the text, with the family's turn/handoff markers (`user_traits.family_markers`; Gemma E4B uses the gemma4-31b template markers) and the line-initial role labels (ChatGPT/Assistant/AI/Bot/GPT/Claude/Model). Reported: flagged share by family, stage and context; the register change recomputed with flagged replies excluded in every stage. **Reading**: slippage explains the fall in a context if exclusion shrinks the register change by at least 50% (size reported either way).
- **Check C, OLMo training data only**: the Dolci sets the repo already uses for contamination checks (`core.check_contamination.DOLCI`: SFT messages, DPO chosen/rejected, RL prompts; loaded with `datasets`; a set that cannot be identified or loaded is reported as such, not guessed). Share of examples whose user (or system) text asks for role-play or a persona, by a regex over those turns: `role[- ]?play(ing)?`, `pretend (to be|you're|you are)`, `act (as|like) (a|an|the|my|if)`, `play (the (role|part) of|(a|an|the) (role|character))`, `you are now`, `from now on,? you (are|will)`, `imagine you('re| are) (a|an|the)`, `take on the (role|persona)`, `in the role of`, `(answer|respond|speak|write) as (a|an|if)`, `persona`, `stay in character`, `in character`, `character (named|called)` (case-insensitive). 30 randomly chosen matches (seed 0) saved to the report with the matched phrase. Spec A classifier P(assistant) (on the first 40 words of the reply) and layout rate on the replies of matched examples vs a random sample of the rest (SFT assistant reply, DPO chosen reply; the RL set stores prompts only, so only its share is reported). Descriptive; no threshold.
- No existing result or frozen file is changed; raw generations and Dolci text beyond the 30 saved excerpts are not committed.

## 2026-11-04: Gemma 4 31B confirmatory carryover runs on arena9-vinyl, not here

Adeeb claimed arena9-vinyl (96 GB GPU, 1 TB RAM, about 90 GB disk) and the laptop Claude started the run there at about 14:08 UTC (repo at `870ed99`, log `/root/gemma_carryover.log`, `GPU_GIB=90`). All weights are on the GPU, no CPU offload and no quantization: still bf16, batch 1, same frozen items (sha256 `e319184f4dd09f6d705d2bdd8badc2575919b0d4a0913c14e84ef83708b23344`), same criteria and verdict code, think/thought-channel rendering as logged above; contamination could not be checked for this family. The offload plan and the 2-hour stop above were for this 24 GB card and are superseded; the outputs will be copied back with rsync and committed as written. Nothing under `/root/g4nla` on vinyl is touched.

## 2026-11-04: judge validation sheet rebuilt from the main-run generations and split between three labellers (vestigial regression, judge arm)

Before any human label existed (`judge_validation/sheet.csv` had 0 of 60 labelled), the validation sheet was rebuilt with
`judge sheet --from-capstone` (seed 0). The old sheet drew on pre-capstone generations (exploration samples for Gemma IT and Qwen IT,
the pre-capstone OLMo run's complete-sentence prefills), so it validated the judge on text it was never used on, and it held no Gemma
or Qwen base samples, which the README's validation requirement includes. The new sheet draws 60 from the 600 incomplete-turn generations
per checkpoint that the judge labelled in the main run: 8 per OLMo stage, 7 per Qwen and Gemma checkpoint (`CAPSTONE_PLAN`).

Cole asked for the labelling to be shared by Cole, Grace and Adeeb with some overlap. `judge split-sheet` (seed 0) puts 12 samples on
every labeller's file (covering all 8 checkpoints) and deals the other 48 out evenly, 16 each, round-robin within checkpoint, so each
labeller has 28 rows with 3-4 per checkpoint (`judge_validation/split/sheet_<name>.csv`, `assignment.csv`). `judge combine-sheets`
writes the merged labels to `sheet.csv`: an unshared sample's single label, a shared sample's majority (2 of 3 or more). A shared sample
with no majority stays unlabelled and is reported, so it does not enter the judge's agreement. It also reports pairwise agreement and
Fleiss' kappa between labellers on the shared samples. The judge-agreement measure (`judge validate`, raw agreement and Cohen's kappa)
and its 80% default bar are unchanged.

## 2026-11-04: logit distortion: analysis of the main run's exchange scores (Grace; decided after the data existed, not blind)

`measures/logit_distortion/analyze.py` implemented on `results/capstone/<family>/<stage>/core/exchanges_exchange/` (8 checkpoints,
oasst1 / UltraChat / WildChat, opening user turn and the reply after its question, tagged and plain). Spec:
`measures/logit_distortion/README.md`, "Exchange-design analysis (2026-11-04)". Written and committed before any aggregate was computed in
this session, but after the main-run scores and vestigial-regression results existed (and after an agent session for Cole computed
provisional, uncommitted logit-distortion numbers), so none of it is blind or pre-registered.

- **Unchanged (fixed 2026-11-01/02):** gap = mean NLL − mean entropy, positions 1+, own slot; change from base; criterion 1 (assistant gap
  change − user gap change ≥ 0.05 nats with the CI excluding zero, in both authorship columns); by-position plot descriptive; breadth-arm
  contamination statement; no swapped slots. The README's criteria text is not edited.
- **Decided now:** human column = oasst1, model column = UltraChat (matched authorship within corpus); WildChat separate. User persona = the
  opening user turn tagged; assistant persona = the reply tagged after its question; plain = context. Equal weight per position within a
  target and per target within a group; truncated replies (first 300 tokens) kept, rates reported. Contrasts paired per exchange; bootstrap
  10,000 replicates, seed `config.SEED`, exchange IDs resampled jointly within family × corpus. Sensitivities, labelled: positions 0+,
  untruncated-reply exchanges only. Cross-family: common IDs (first 100 per corpus) and bits per byte, descriptive. Criterion 1 is the only
  verdict rule applied; everything else is estimates with CIs.
- **Scope:** this dataset has only opening user turns; follow-up user turns (`pilot_v3`, original-conversation route) are not covered and
  would need their own scoring run.
