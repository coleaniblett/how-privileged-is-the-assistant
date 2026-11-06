# results/

Reorganised 2026-11-06 (`docs/moved_paths_2026-11-06.md` maps every old path to its new one).

| directory | what | written by |
|---|---|---|
| `capstone/` | **The main run** (2026-11-03/04, exchange design): per checkpoint, the shared exchange scores (`<family>/<stage>/core/exchanges_exchange/`) and the generations (`<family>/core/generations*/`), then each measure's analysis of them: vestigial regression (`vestigial_regression/`, `<family>/vestigial_regression/`: flags, judge labels), logit distortion (`logit_distortion/`) and overgeneralization's user-slot arm (`<family>/overgeneralization/`; the cross-family user-turn summary of 2026-11-06, provisional judge labels, is `overgeneralization/user_turn_2026-11-06/`). `logs/` has the run's logs | `scripts/run_exchanges.sh`, `scripts/run_open5.sh`; the measures' `analyze*.py` |
| `capstone_think_off/` | Qwen3-8B with its think block off, the main run's robustness check | the 2026-11-03 collection (`attic/scripts/`) |
| `logit_conversations/` | Logit distortion's original-conversation route: provisional pilot scores (OLMo, Qwen) | `core.score_conversations` |
| `overgeneralization/` | Everything else overgeneralization ran: the pre-registered persona-carryover test and about thirty exploratory pilots, one directory each. **Index: `overgeneralization/README.md`** | `extras/`, `measures/overgeneralization/` |
| `precapstone/` | The frozen pre-capstone held-out run (OLMo 3 7B, isolated slots, 2026-11-01/02): `scores/`, `generations/`, its report (`results.md`, `summary.csv`, `position.md`, figures), the role-swap and coin-toss extras, and the data selection check (`selection_check/`). Frozen: never modified or overwritten | `scripts/run_gpu.sh` (legacy layout), `core/selection_check.py` |
| `exploration/shiller_user_turns/` | Open user turns across models (Shiller's cursory exploration); the vestigial-regression pilot's inputs | `exploration/shiller_user_turns/` |

Earlier pilot runs that nothing reads any more are in `attic/` (`attic/README.md`).
