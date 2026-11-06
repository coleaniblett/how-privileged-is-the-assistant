# scripts/overgeneralization/

The launchers of overgeneralization's pilots, one folder per pilot, matching `results/overgeneralization/<pilot>/` (index there).
Moved here from the flat `scripts/exploratory/` on 2026-11-06 (`docs/moved_paths_2026-11-06.md`).

They are records of how each run was launched, and most are machine-specific: absolute checkout paths (`/root/hpa-main-wt`,
`/root/hpa-atlas2`), `/opt/arena-env/bin/python`, `ssh arena9-vinyl`/`arena9-marble`, GPU locks and pushes to `main`. Read a launcher
before running it anywhere; the module CLIs (`python -m extras.<module> --help`) are the portable entry points. Every launcher in a
pilot folder changes to the repo root first (`cd "$(dirname "$0")/../../.."`) or to its machine's checkout.

| folder | what |
|---|---|
| `persona_carryover/` | the measure's confirmatory runs (OLMo, Qwen, Gemma 4 31B and its thought-channel rerun) |
| `user_slot/` | the user-slot arm on the main run's generations, and the exchange-set prefill generation and judging |
| `early_probes/` | the 2026-11-02/03 probes; `run_persona_axis*.sh` regenerate the activation fits later pilots read |
| one folder per pilot | `trait_atlas/`, `mechanism_checks/`, `activation_patching/`, `working_model_L/`, `role_says/`, ... |
| `new_families_j2/` | the Llama 3.1 8B / Apertus 8B breadth checks across several tests |
| `queues/` | one-machine queues that ran several pilots in a row |
| `overnight_2026-11-05/` | the spec R/T chains and steps, their helpers (`lib_r.sh`, `queue_driver.sh`) and `make_morning.py` |
| `remote.sh`, `sync_vinyl.sh` | helpers for the remote GPU hosts, shared by several pilots |
