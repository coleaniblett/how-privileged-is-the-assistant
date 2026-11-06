# vestigial-user: is the assistant a privileged persona?

ARENA 9.0 capstone (Cole, lead; Adeeb; Grace). **Question:** does post-training make the assistant a
*privileged persona*, in the sense of Shiller, "Privilege, Dominance, and Personas" (Eleos, Aug 2026)? We
measure three of Shiller's forms of privilege on the same models, one measure per teammate, on a shared
pipeline. `AGENTS.md` is the working guide (hypotheses, rules, conventions, commands); `docs/decisions.md`
logs decisions.

| measure | form | question | folder |
|---|---|---|---|
| vestigial regression | dominance | do non-assistant personas, especially the user, degrade? | `measures/vestigial_regression/` |
| logit distortion | privilege | do the assistant persona's next-token distributions become distinctively concentrated? | `measures/logit_distortion/` |
| overgeneralization | dominance | do assistant traits appear in non-assistant contexts? | `measures/overgeneralization/` |

**Hypotheses.** H1 thin mask (no privilege); H2 privilege without dominance; H4 privilege with dominance (no
H3: dominance is a form of privilege). No slot-specific change beyond base on any measure → H1; logit
distortion only → H2; vestigial regression or overgeneralization → H4, reporting which form(s).

## Models

Primary: the OLMo 3 7B Instruct line, every stage scored with the Instruct tokenizer and chat template.

| stage | checkpoint |
|---|---|
| base | `allenai/Olmo-3-1025-7B` |
| sft | `allenai/Olmo-3-7B-Instruct-SFT` |
| dpo | `allenai/Olmo-3-7B-Instruct-DPO` |
| instruct (RL) | `allenai/Olmo-3-7B-Instruct` |

`core/check_models.py` verifies IDs, licenses, sizes, chat templates and tokenizers through the HF API and
writes `docs/models.md`. OLMo 3's post-training data (Dolci) is open, so test texts are filtered for overlap
with it; it is the only family where that is possible. A breadth arm of base-vs-instruct pairs from other
families (Gemma 4, Qwen3) is planned but not configured yet; contamination can't be checked there.

## Shared pipeline (`core/`)

Held-out user and assistant texts (`data/texts.jsonl`: WildChat pairs, oasst1 prompts, OLMo Instruct's own
replies; sampled by `core/prepare_data.py` after excluding anything in Dolci) are rendered in each role slot
and as plain prose with content tokens identical across conditions (`core/render.py`), scored in fp32 for
per-token NLL and next-token entropy at every stage (`core/score.py`), and used to prefill user turns the
models continue (`core/generate.py`). Measures read those outputs; `core/stats.py` holds the shared loaders,
bootstrap CIs and the 0.05-nat negligibility rule.

## Layout

```
AGENTS.md  CLAUDE.md  README.md  pyproject.toml  uv.lock
core/                      config, model checks, data prep + contamination filter, rendering, scoring,
                           sampling, stats/plots
measures/
  vestigial_regression/    README (spec + pre-registered criteria), flags.py, analyze.py
  logit_distortion/        README (spec + criteria), analyze.py stub
  overgeneralization/      README (spec + criteria)
extras/                    overgeneralization's pilot modules; role_swap.py, coin_toss.py (pre-capstone)
reports/                   capstone results and the NLL-arm build-up (HTML)
scripts/                   run_exchanges.sh (main run), run_gpu.sh (legacy run), overgeneralization/<pilot>/ launchers
data/                      committed text sets      docs/   models.md, rendered_examples.md, decisions.md
results/                   capstone/ (main run), overgeneralization/ (measure + pilots, index there), precapstone/
                           (frozen); results/README.md. attic/ holds earlier frozen runs and retired tooling
tests/                     OLMo tokenizer alignment + CPU smoke test
```

## Running

```bash
uv sync
uv run pytest
uv run python -m core.check_models                  # -> docs/models.md
uv run python -m core.render                        # -> docs/rendered_examples.md
bash scripts/run_gpu.sh                             # single 48 GB+ GPU -> results/olmo3-7b/
RUN=olmo3-7b_preceding bash scripts/run_gpu.sh --preceding-turn
```

`scripts/run_gpu.sh` runs each checkpoint in its own process so GPU memory is released between them; the
checkpoints are ~15 GB each in bf16 (`CLEAR_CACHE=1` deletes each from the HF cache after use). Scoring
~1,200 texts × 3 slots takes about five minutes per model in fp32.

## Pilot results

`results/precapstone/` holds the held-out main run of the vestigial-regression pilot on OLMo 3 7B (`results.md`,
`position.md`, `role_swap.md`, `coin_toss.md`; the selection check is in `results/precapstone/selection_check/`),
`attic/results_preceding/` the preceding-turn robustness run and `attic/results_v0/` the earlier unfiltered sample's own outputs
(its texts are `data/prefilter_texts.jsonl`; `data/README.md` describes every dataset). All are frozen.
