# AGENTS.md: the privileged-persona capstone

Canonical guide for teammates and for agent sessions (Codex reads this file; Claude Code reads it through
`CLAUDE.md`). Keep it short: detail lives in each measure's README, decisions in `docs/decisions.md`.

## Question and hypotheses

Does post-training make the assistant a *privileged persona*? Following Shiller, "Privilege, Dominance, and
Personas" (Eleos, Aug 2026), we measure three of Shiller's forms of privilege on the same models, one measure
per teammate, on top of a shared pipeline.

| measure | Shiller's form | one line | owner | status |
|---|---|---|---|---|
| vestigial regression | dominance | non-assistant personas, especially the user, degrade | Cole | design locked (NLL arm + judge arm); NLL arm built for OLMo, judge arm not built |
| logit distortion | privilege | the assistant persona's next-token distributions become distinctively concentrated | Grace | design settled; analysis stub only |
| overgeneralization | dominance | assistant traits appear in non-assistant contexts | Adeeb | in design; nothing implemented |

Hypotheses (there is no H3: Shiller treats dominance as a form of privilege):

- **H1, thin mask:** no privilege.
- **H2, privilege without dominance.**
- **H4, privilege with dominance.**

Decision rule: no slot-specific change beyond base on any measure → H1; logit distortion only → H2;
vestigial regression or overgeneralization (with or without logit distortion) → H4, reporting which form(s)
of dominance.

## Team and rules for agents

Cole (lead), Adeeb, Grace. Measure owners are in the table above: vestigial regression Cole, logit distortion Grace,
overgeneralization Adeeb; the shared pipeline and the breadth arm are Cole's. `STATUS.md` is the one-screen snapshot of progress.

- After completing a task, update the relevant rows of `STATUS.md` and its "Last updated" line in the same commit.
- When working for one measure's owner, edit only that measure's folder (`measures/<name>/`) unless explicitly
  asked to touch something else.
- `core/` is shared. A change to it must be backwards-compatible (existing callers, CLI flags and output
  formats keep working), flagged in the commit message (start it with `core:`), and logged with a dated entry
  in `docs/decisions.md`.
- Never change another measure's code to make yours work. If core lacks something, add it to core compatibly.
- `extras/` (role swap, coin toss) is kept working; whether it is in scope is undecided.
- Don't run GPU jobs or download checkpoints from an agent session unless asked.

## Layout

```
core/                   shared pipeline: config (models, families, stages, slots, paths), model checks, data prep
                        + contamination filtering, rendering, scoring, sampling, stats and plot helpers
measures/<name>/        README (spec + pre-registered criteria), analysis, measure-specific flags and plots
extras/                 role_swap.py, coin_toss.py (README has their design)
scripts/run_gpu.sh      the full run: score + sample per checkpoint, then every analysis
data/  results/  docs/  tests/
```

Principle: anything that samples or scores models, or prepares shared text sets, lives in `core/`. Anything
that turns core outputs into a measure-specific statistic, figure or verdict lives in that measure's folder.
Modules run from the repo root with `python -m` (`uv run python -m core.score --model sft`).

## Shared conventions

- The model list, families and stages come only from `core/config.py` (`FAMILIES`, `MODELS`, `STAGES`,
  `TEMPLATE_MODEL`); so do slots, text sources and output paths. No other file defines a model list.
- Every measure reads core's outputs (per-text scores, per-token tables, raw samples) rather than re-scoring.
- Every effect is reported as change from base with a 95% bootstrap CI over texts, paired per text wherever the
  design allows (`core.stats.bootstrap_ci`). Text sets (source × kind) are analyzed separately, never pooled.
- For NLL-based effects the smallest effect we count is **0.05 nats** (`config.MIN_EFFECT_NATS`, the existing
  negligibility margin), alongside the CI excluding zero. A smaller effect is negligible even if "significant".
- Scoring stays in **fp32**: in bf16 a text's NLL moves by up to 0.05 nats with batch composition. A bf16
  batch-size-1 run may replace it only after being validated against fp32 and logged in `docs/decisions.md`.
  Generation runs in bf16.
- All stages of a family (base included) are rendered with that family's instruct tokenizer and chat template,
  so token sequences are identical across stages.

## Pre-registration

Each measure's README has a **Pre-registered criteria** section. Agents must not edit those criteria once
results for that measure exist under `results/<run>/<measure>/`. Any change gets a dated entry in
`docs/decisions.md` with the reason, and the README notes the date of the change.

## Models

- **Primary:** OLMo 3 7B across post-training stages, base → SFT → DPO → Instruct (RL); checkpoints in
  `config.FAMILIES["olmo3-7b"]`, verified in `docs/models.md`. Its post-training data (Dolci) is open, so test
  texts are filtered for contamination (`core/check_contamination.py`). It is the only family where that is
  possible.
- **Breadth arm** (not in config yet): base vs instruct pairs from other families: Gemma 4 31B plus Qwen3-14B
  or Qwen3-30B-A3B if we get 80 GB GPUs, otherwise Gemma 4 E4B and Qwen3-8B. Contamination cannot be checked
  for these families: say so wherever their results appear. Qwen published no base checkpoint for Qwen3-32B.
- TODO before adding a family: verify the Gemma 4 model IDs and chat templates via `core/check_models.py`
  (generalize it to take a family), then add the family to `config.FAMILIES` with its own stage list.

## Outputs

One directory per run under `results/`, named after the family by default (`RUN=... bash scripts/run_gpu.sh`):

```
results/<run>/core/scores/<stage>.csv, <stage>_tokens.parquet   per-text and per-token NLL + entropy, every text x slot
results/<run>/core/generations/<stage>.jsonl                     raw prefilled-user-turn samples (no flags)
results/<run>/vestigial_regression/    results.md, summary.csv, nll_*.png, position.md, position_*.png,
                                       generations/<stage>.jsonl (flagged) and <stage>_counts.csv
results/<run>/logit_distortion/, results/<run>/overgeneralization/    their tables and figures
results/<run>/extras/                  role_swap/, coin_toss/, role_swap.md, coin_toss.md, *_summary.csv
candidates/logit_distortion/<build>/   original-conversation candidates (core.build_conversations), logit distortion's
                                       primary route; current pilot_v3 (eligible quota, Dolci-checked); not core.score input; docs/conversation_candidates.md
```

Frozen, never modified, moved or overwritten: `data/` (regenerate only through the documented commands),
`results/scores`, `results/generations`, `results/role_swap`, `results/coin_toss` and the `.md/.csv/.png` files
beside them (the pre-capstone held-out run), `results_preceding/` (preceding-turn robustness run) and
`results_v0/` (earlier unfiltered sample).

## Measures

**Vestigial regression**, `measures/vestigial_regression/README.md`. Status: design locked 2026-11-02 after a
Gemma 4 31B / Qwen3-8B / OLMo 3 7B pilot (`measures/vestigial_regression/pilot/report.md`); two arms. **NLL arm
(capacity, built for OLMo; Gemma and Qwen scored by the pilot's scripts):** the primary quantity is the asymmetry
between the user and assistant role-specific changes from the model's own base, `Δ(user@user − user@plain) −
Δ(asst@asst − asst@plain)` (nats per token within a family, bits per byte across families); it fires at ≥ 0.05 nats
(a detectability floor, magnitude is the primary result) with a CI excluding zero in both user sources. Plain-text
change from base is reported as its own row and referred to overgeneralization; slot effects stay descriptive.
**Judge arm (behavior, designed, not built):** an LLM judge labels prefilled user-turn continuations (user voice /
switched to assistant / degenerate / user voice with assistant traits), validated against teammate-checked human
labels, plus regex flags for loops. Either arm firing counts toward H4, reporting which. The decision-point
(turn-end hazard) measure is future work. NLL-arm criteria were revised after pilot results, so OLMo NLL verdicts are
not blind. Code: `flags.py` (regex flags on core samples), `analyze.py` (the asymmetry is a TODO); open items in the
measure README.

**Logit distortion**, `measures/logit_distortion/README.md`. Status: design settled, `analyze.py` is a stub.
Calibration gap (mean NLL − mean next-token entropy) of each persona in its own slot, crossing persona with
authorship; privilege if the assistant gap grows from base more than the user gap, from content position 1 on
and by position. Data route (Grace, 2026-11-02): original multi-exchange conversations from OASST1, WildChat and UltraChat with
full preceding history (`core.build_conversations`, `docs/conversation_candidates.md`; candidate pilot
`candidates/logit_distortion/pilot_v3/`); the isolated authorship sets stay for a separate comparison.
Original-context scoring: `core.score_conversations` (`docs/conversation_scoring.md`), built and CPU-tested; no checkpoint scored yet.

**Overgeneralization**, `measures/overgeneralization/README.md`. Status: in design, nothing implemented.
Likelihood minimal pairs (assistant-trait vs plain continuation) in plain context with the assistant slot as
positive control; interp arm (assistant direction per stage); behavioral companion (assistant-tic counts).

## Exploratory work

- `exploration/shiller_user_turns/`: open user turns across OLMo/Qwen3/Gemma 4, qualitatively reproducing Shiller's cursory exploration; outside every verdict (README there).

## Commands

```bash
uv sync                                          # lockfile: uv.lock
uv run pytest                                    # OLMo tokenizer alignment + CPU smoke test (SmolLM2-135M)
uv run python -m core.check_models               # model IDs, licenses, shared template -> docs/models.md
uv run python -m core.prepare_data               # optional: data/ is committed (streams WildChat, loads Dolci)
uv run python -m core.add_replies                # optional: OLMo Instruct replies -> data/texts.jsonl (GPU)
uv run python -m core.check_contamination        # overlap of data/texts.jsonl with Dolci
uv run python -m core.render                     # -> docs/rendered_examples.md
bash scripts/run_gpu.sh                          # everything, per checkpoint -> results/olmo3-7b/
RUN=olmo3-7b_preceding bash scripts/run_gpu.sh --preceding-turn      # robustness run
uv run python -m measures.vestigial_regression.analyze               # re-analyze from results/olmo3-7b/core/
uv run python -m core.build_conversations --out candidates/logit_distortion/<new>  # conversation candidates (no weights)
uv run python -m core.conversation_audit --input candidates/logit_distortion/pilot_v3  # offline verify + audit
uv run python -m core.score_conversations --family olmo3-7b --prepare-only   # conversation-scoring inputs (no weights)
```
