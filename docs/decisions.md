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
