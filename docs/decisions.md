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
