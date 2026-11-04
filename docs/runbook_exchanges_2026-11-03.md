# Runbook: tonight's main run (exchange design), 2026-11-03

**Launched 2026-11-03 23:09:53 UTC, finished 2026-11-04 05:39 UTC, exit 0** (OLMo, then Qwen, then Gemma from `/` after freeing disk; filtered log `results/capstone/logs/main_run_exchanges_2026-11-03.log`). Design (confirmed by Cole and Grace): the exchange context only. Per checkpoint the user turn is scored as the
conversation's opener and each reply after its own user turn, each tagged and plain, on oasst1, UltraChat and WildChat: 12
conditions per checkpoint, 96 in all (`docs/exchanges.md`, `docs/decisions.md` 2026-11-03). In the same pass each checkpoint
generates on the judge arm's incomplete-turn prefills, so it is downloaded once. The reply-alone context is not part of the run;
`scripts/run_exchanges.sh` refuses any other context.

## Where to launch from

**From the worktree `/root/hpa-capstone-scoring` (branch `capstone-scoring-2026-11-03`), or from `main` after the PR is merged.**
Never from `/root/how-privileged-is-the-assistant`: that checkout is on a stale local `main` (`5b3dd8c`, 30+ commits behind
`origin/main`) and has none of this code. Run inside `tmux` so the run survives a closed laptop.

## Preconditions (check before each family)

1. **GPU free:** `nvidia-smi --query-compute-apps=pid,name --format=csv,noheader` prints nothing. Other sessions use this GPU.
2. **Disk:** `df -h / | tail -1`. The launcher checks free disk before every download (`core.hf_cache check`: weights + 5 GB)
   and stops if a checkpoint does not fit. **About 65 GB free before each Gemma checkpoint** (62.5 GB weights + margin).
3. **Never delete teammates' files.** In particular, **leave Grace's Gemma copy in `/dev/shm/grace-gemma-hub/` alone** (59 GB, Gemma 4
   31B base weights): use it only read-only, only with her OK, and never remove it. When `HF_HUB_CACHE` points anywhere but the
   default cache, the launcher forces `CLEAR_CACHE=0`, so it cannot delete from such a cache. Only remove what this run downloaded.
4. **Environment:** the launcher uses the checkout's own `.venv` (it exports `UV_PROJECT_ENVIRONMENT=$PWD/.venv`). tmux's global
   environment on this machine sets `UV_PROJECT_ENVIRONMENT` to the main checkout's `.venv`; other `uv run` commands typed in tmux
   (e.g. `hf_cache free`, the analysis) should be run as `env -u UV_PROJECT_ENVIRONMENT -u VIRTUAL_ENV uv run ...` from the worktree.
5. **CPU and RAM:** the container is limited to 13.6 CPUs (`nproc` says 128) and 188 GB of RAM, of which `/dev/shm` contents count
   (63 GB now). The launcher caps torch at 8 threads; ask other sessions to cap theirs (`OMP_NUM_THREADS=8`) during the run.
   Gemma's fp32 offload needs about 93 GB of RAM.

## Commands, in order: OLMo, Qwen3-8B, Gemma 4 31B

```bash
tmux new -s main-run                     # or attach to an existing session
cd /root/hpa-capstone-scoring && mkdir -p logs && set -o pipefail
# optional, saves about 1 h: reuse Grace's validated incomplete-turn generations for OLMo and Qwen (copy, never move; same
# command and prefills as below, 600 samples each, driver integrity checks passed). The launcher then skips their generation.
#   for f in olmo3-7b qwen3-8b; do mkdir -p results/capstone/$f/core/generations_incomplete; \
#     cp -n /root/how-privileged-is-the-assistant-grace/results/capstone/$f/core/generations_incomplete/*.jsonl results/capstone/$f/core/generations_incomplete/; done

# 1. OLMo 3 7B: base, SFT, DPO, Instruct (none cached; each removed after use)
CLEAR_CACHE=1 FAMILY=olmo3-7b bash scripts/run_exchanges.sh 2>&1 | tee -a logs/main_olmo3-7b.log && \
# 2. Qwen3-8B: base, IT (IT is already cached and is kept; base is removed after use)
CLEAR_CACHE=1 FAMILY=qwen3-8b bash scripts/run_exchanges.sh 2>&1 | tee -a logs/main_qwen3-8b.log && \
env -u UV_PROJECT_ENVIRONMENT -u VIRTUAL_ENV uv run python -m core.hf_cache free Qwen/Qwen3-8B      # 16 GB, downloaded by this session for the smoke run
```

**3. Gemma 4 31B (base, IT): needs a decision on disk first (see "Disk" below).** With room for one checkpoint at a time on `/`:

```bash
CLEAR_CACHE=1 FAMILY=gemma4-31b bash scripts/run_exchanges.sh 2>&1 | tee -a logs/main_gemma4-31b.log
```

Alternative for the base only, **if Grace agrees**: score and generate from her `/dev/shm` copy, offline so nothing is written to her
cache (the launcher forces `CLEAR_CACHE=0` there):

```bash
HF_HUB_OFFLINE=1 HF_HUB_CACHE=/dev/shm/grace-gemma-hub/hub STAGES=base FAMILY=gemma4-31b bash scripts/run_exchanges.sh 2>&1 | tee -a logs/main_gemma4-31b.log
CLEAR_CACHE=1 STAGES=instruct FAMILY=gemma4-31b bash scripts/run_exchanges.sh 2>&1 | tee -a logs/main_gemma4-31b.log   # IT from / (65 GB free)
```

Every task is skipped if its output exists, so rerunning a command resumes after a failure (an interrupted generation leaves only
a `.tmp_<stage>` directory, which the next attempt replaces). Each family ends with `--check-stages`: every stage must share one
inputs fingerprint.

## What each checkpoint does

1. `core.score_exchanges --context exchange --tag exchanges_exchange`: 3600 sequences (Gemma: 1200; first 100 exchanges per corpus),
   fp32 -> `results/capstone/<family>/<stage>/core/exchanges_exchange/{scores.csv,tokens.parquet,manifest.json}`. User turns already
   scored overnight are reused after their token ids are verified identical (OLMo/Qwen: 1434 sequences reused, 2166 scored).
2. `core.generate --prefill data/prefill_incomplete_200.jsonl --save-ids` (200 prefills x 3 samples x 120 tokens, T=0.7; bf16, Gemma
   int8) -> `results/capstone/<family>/core/generations_incomplete/<stage>.jsonl`, checked for 600 samples before it is moved in.

## Time and disk per family (estimates)

Scoring times scale the overnight run's measured scoring time to the exchange set's token counts (`results/capstone/logs/summary.json`);
generation times are measured on the same prefills (Grace's collection run for OLMo and Qwen) or scaled from the overnight Gemma
generation (2,700 s per 300 samples). Downloads were not timed.

| family | scoring per checkpoint | generation per checkpoint | whole family | disk |
|---|---|---|---|---|
| OLMo 3 7B (4) | about 9 min | about 9 min (measured 516–526 s) | **about 1.5 h** (about 40 min with Grace's generations) + 4 downloads | about 15 GB per checkpoint, removed after use |
| Qwen3-8B (2) | about 10 min | about 12–14 min (measured 707–847 s) | **about 50 min** (about 25 min with Grace's generations) + base download | base 16.4 GB (+5 margin), IT cached |
| Gemma 4 31B (2) | about 35–50 min (CPU offload) | about 90 min (int8) | **about 4.5–5 h** + two 62.5 GB downloads | 62.5 GB per checkpoint (+5 margin) |

Whole run: about 7–8 h, of which Gemma is most. Outputs are small (under 15 MB of scores per checkpoint, about 0.5 MB per generation file).

## Disk (21:20 UTC)

- `/`: **22 GB free.** OLMo (15 + 5 GB) and Qwen base (16.4 + 5 GB) each fit, barely, one at a time. After Qwen, freeing Qwen3-8B
  IT gives about 38 GB.
- **Gemma does not fit yet:** each checkpoint needs about 65 GB. What is on `/` that is not this run's: `~/.cache/huggingface/datasets`
  (9.4 GB of Dolci arrow files, only needed to rebuild data sets; regenerable), HF cache `Qwen3-8B-FP8` (9 GB) and `Olmo-3-7B-Think`
  (14 GB) and `/root/assistant-dynamics` (83 GB): **other people's; ask their owners before removing anything.**
- `/dev/shm`: 30 GB free (88 GB, Grace's 59 GB Gemma base copy in it). Gemma IT cannot also go there: RAM is capped at 188 GB and
  shm counts against it (63 + 62.5 GB plus about 93 GB of offload is too much).

## After the run

- Outputs: `results/capstone/<family>/<stage>/core/exchanges_exchange/` and `results/capstone/<family>/core/generations_incomplete/`.
- Analysis (CPU): `uv run python -m measures.vestigial_regression.analyze_exchanges` (frame costs, changes from base, asymmetry).
  Flags: `uv run python -m measures.vestigial_regression.flags --family F --generations results/capstone/F/core/generations_incomplete
  --out results/capstone/F/vestigial_regression/generations_incomplete`. The judge needs `ANTHROPIC_API_KEY` and the teammate-labelled sheet.
- Commit the outputs with a STATUS.md update.

## Tests

`uv run pytest` passes with a GPU visible (`tests/conftest.py` hides it from the CPU-only suite; `TESTS_ALLOW_GPU=1` opts out).
Verified before handing over: `tests/test_exchanges.py` (every rendering in every family, the tagged/plain token-identity check,
truncation, builder filters, reuse, a CPU smoke run, the launcher refusing other contexts); `--prepare-only` for all three families,
0 exclusions; GPU smoke on Qwen3-8B IT (10 exchanges); reused vs fresh scores agree to 7e-6 nats. Not run on GPU: any OLMo or Gemma
checkpoint, Qwen3-8B base.
