# attic/

Frozen material kept for the record. Nothing live reads or runs anything here (no code, launcher or test depends on it; docs
mention it only as history); do not modify it, and put only frozen material here, each move logged in `docs/decisions.md`
(entries of 2026-11-03 and 2026-11-06). The tooling here is not runnable as it stands: it expects its original location.

- `results_preceding/`: the preceding-turn robustness run on the pre-capstone held-out texts.
- `results_v0/`: the earlier unfiltered WildChat sample's own outputs (generations, `results.md`, `summary.csv`, `nll_by_stage.png`). Its texts, the
  Dolci-match lists and its per-stage scores were moved out because `core.selection_check` uses them: `data/prefilter_texts.jsonl`,
  `data/prefilter_contaminated_Dolci-Instruct-*.json`, `results/precapstone/selection_check/scores/`. The `results.md` here was computed from those scores.
- `scripts/`, `docs/`, `tests/` (moved 2026-11-06): the finished 2026-11-02/03 overnight collection. `overnight_collect.py` (task driver),
  `overnight_launch.sh`, `overnight_supervise.sh`, `overnight_status.py`, `overnight_report.py`, the conversation collection built on them
  (`collect_conv_incomplete.py`, `collect_gemma_shm.sh`), their runbook and report (`docs/runbook_2026-11-03.md`,
  `docs/overnight_2026-11-03.md`, `docs/collection_conv_incomplete_2026-11-03.md`) and the driver's test (`tests/test_overnight.py`).
  Their outputs are live: `results/capstone/logs/`, `results/capstone_think_off/`, `results/logit_conversations/`.
- `scripts/exploratory/` (moved 2026-11-06): the generator and template of the 2026-11-03 overgeneralization summary page; the page itself is
  `results/overgeneralization/early_probes/overgeneralization_results.html`.
