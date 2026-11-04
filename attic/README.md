# attic/

Frozen material kept for the record. Nothing here is read by the pipeline except where noted; do not modify it, and put only frozen material here
(moved by `docs/decisions.md` entries of 2026-11-03).

- `results_preceding/`: the preceding-turn robustness run on the pre-capstone held-out texts.
- `results_v0/`: the earlier unfiltered WildChat sample's own outputs (generations, `results.md`, `summary.csv`, `nll_by_stage.png`). Its texts, the
  Dolci-match lists and its per-stage scores were moved out because `core.selection_check` uses them: `data/prefilter_texts.jsonl`,
  `data/prefilter_contaminated_Dolci-Instruct-*.json`, `results/selection_check/scores/`. The `results.md` here was computed from those scores.
