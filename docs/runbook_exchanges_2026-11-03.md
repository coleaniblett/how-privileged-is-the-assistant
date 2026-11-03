# Runbook: exchange scoring (vestigial regression + logit distortion), 2026-11-03

**Not launched.** Cole launches it after the separate decision on turns (the reply's context, below). Everything is built
and tested; what it does and why: `docs/exchanges.md`; decisions: `docs/decisions.md` (2026-11-03). Branch
`capstone-scoring-2026-11-03`. Run from the repo root of a checkout of that branch (or of `main` once it is merged).

## Before launching

1. **Decide the reply's context.** `exchange` (default: the reply after its own user turn) or `single` (the reply alone,
   as overnight). Pass it as `CONTEXT=...`; each context writes its own tag (`exchanges_exchange`, `exchanges_single`), so
   both can be run and neither overwrites the other or the overnight outputs. A different design is one more entry in
   `core.exchanges.CONTEXTS` (`docs/exchanges.md`, "Switching the reply's context"), then `CONTEXT=<name>`.
2. **GPU free?** `nvidia-smi --query-compute-apps=pid,name --format=csv,noheader` must print nothing (other sessions use this GPU).
3. **Disk.** `df -h / | tail -1`. See the per-family figures below. `run_exchanges.sh` checks free disk before each
   download (`core.hf_cache check`) and stops if a checkpoint does not fit.
4. **Preflight (no weights, seconds):** `uv run python -m core.score_exchanges --family <family> --prepare-only` must report
   0 excluded (`run_exchanges.sh` runs it first anyway).

## Launch order and commands

OLMo, then Qwen, then Gemma. Run each under `tmux` so it survives the session; logs go to `logs/` (`mkdir -p logs`).

```bash
mkdir -p logs
# 1. OLMo 3 7B: base, SFT, DPO, Instruct (none cached; each removed after use with CLEAR_CACHE=1)
OMP_NUM_THREADS=8 CLEAR_CACHE=1 FAMILY=olmo3-7b CONTEXT=exchange bash scripts/run_exchanges.sh 2>&1 | tee logs/exchanges_olmo3-7b.log
# 2. Qwen3-8B: base, IT (think block ON, the primary rendering)
OMP_NUM_THREADS=8 FAMILY=qwen3-8b CONTEXT=exchange bash scripts/run_exchanges.sh 2>&1 | tee logs/exchanges_qwen3-8b.log
# 3. Gemma 4 31B: base, IT, first 100 exchanges per corpus (config), fp32 with CPU offload
uv run python -m core.hf_cache free Qwen/Qwen3-8B ; uv run python -m core.hf_cache free Qwen/Qwen3-8B-Base   # room for 62.5 GB
OMP_NUM_THREADS=8 CLEAR_CACHE=1 FAMILY=gemma4-31b CONTEXT=exchange bash scripts/run_exchanges.sh 2>&1 | tee logs/exchanges_gemma4-31b.log
```

For Gemma on a root disk without 67.5 GB free, use a RAM-backed cache as Grace's `scripts/collect_gemma_shm.sh` does
(on her branch): `HF_HUB_CACHE=/dev/shm/<dir>/hub` before the command (it is passed through), after checking `/dev/shm`
free space and container memory (about 62.5 GB of weights in shm plus about 93 GB of fp32 offload in RAM).

Per checkpoint the script skips a stage whose `scores.csv` exists, so rerunning the same command resumes after a failure.
At the end it runs `--check-stages`: every stage of the family must share one inputs fingerprint.

## Time and disk per family

Counted from the committed `data/exchanges.jsonl` with each family's tokenizer (`--prepare-only` logic; reuse as the scorer
applies it). Times are **estimates**: the overnight run's measured scoring time on a similar number of input tokens
(`results/capstone/logs/summary.json`), plus downloads, which were not timed. The GPU smoke (Qwen3-8B IT, 22 fresh sequences in
4.3 s, 7 s to load from cache) is consistent with them.

| family | exchanges | sequences / checkpoint | fresh (exchange ctx) | fresh input tokens | scoring per checkpoint (est.) | whole family (est.) | disk |
|---|---|---:|---:|---:|---|---|---|
| OLMo 3 7B (4 ckpts) | 900 | 3600 | 2166 (1434 reused user turns) | 553k | about 8 min (overnight: 451 s for about 560k tokens) + about 1 min load | **about 1 h** with four 15 GB downloads | one checkpoint at a time with `CLEAR_CACHE=1`: about 15 GB + 5 GB margin |
| Qwen3-8B (2) | 900 | 3600 | 2166 | 528k | about 9 min (overnight: 537 s) | **about 30 min** with the base download | 16.4 GB each; IT is cached now (downloaded for the smoke run), base needs 16.4 GB + 5 GB margin |
| Gemma 4 31B (2) | 300 (first 100 per corpus) | 1200 | 714 | 179k | about 35-50 min with CPU offload (overnight: 2928 s for 1200 sequences, 190k tokens) | **about 1.5-2 h** plus two 62.5 GB downloads | 67.5 GB free for one checkpoint (or the RAM-backed cache), about 95 GB of RAM |

With `CONTEXT=single` the fresh share is about half (OLMo/Qwen 1082 sequences, 230k tokens; Gemma 348), since untruncated replies
reuse `asst@asst` / `asst@plain` too. Outputs are small: about 5-15 MB per checkpoint (`tokens.parquet` holds about 0.3M scored
tokens for OLMo/Qwen).

**Disk now:** 22 GB free on `/` when this was written; Qwen3-8B base fits only just (21.4 GB needed). `~/.cache/huggingface/datasets` is
9.4 GB of Dolci arrow files that only `core.exchanges build` / `prepare_data` need; removing it is the quickest way to make room (it is
rebuilt on the next build). Gemma does not fit on `/` without freeing about 45 GB: use the RAM-backed cache.

**CPU quota:** the container is limited to 13.6 CPUs (`/sys/fs/cgroup/cpu/cpu.cfs_quota_us`), although `nproc` says 128. Torch jobs that
start 128 threads starve everything else; Gemma's CPU offload is the most exposed. Launch with `OMP_NUM_THREADS=8` and ask other
sessions to cap theirs (`OMP_NUM_THREADS`, `torch.set_num_threads`) while the run is going.

## After the run

- Outputs: `results/capstone/<family>/<stage>/core/exchanges_<context>/{scores.csv,tokens.parquet,manifest.json}`.
- `uv run python -m core.score_exchanges --family <family> --tag exchanges_<context> --check-stages`.
- Each `manifest.json` records how many sequences were reused from the overnight run and why the rest were scored fresh.
- Commit the outputs (small: under 15 MB per checkpoint) with a STATUS.md update.

## Verified before handing over

- `uv run pytest`: all exchange tests pass (`tests/test_exchanges.py`: every rendering in every family and context, the tagged/plain
  token-identity check, truncation, builder filters with fake Dolci hits on each turn, reuse checks, a CPU smoke run on SmolLM2-135M);
  the OLMo rendering hash is unchanged. See STATUS.md for the full-suite result.
- `--prepare-only` for all three families and both contexts: 0 exclusions.
- GPU smoke, Qwen3-8B IT, 10 exchanges (4 oasst1, 4 UltraChat, 2 WildChat), both contexts: 40 rows each, 4 truncated replies,
  18 / 34 sequences reused (exchange / single), labels as specified.
- Reuse validated on Qwen3-8B IT (10 WildChat exchanges, `single` context, 40 sequences): reused overnight scores and fresh fp32
  scores agree to 7e-6 nats per text and 3e-4 per token.
- Not run: any OLMo or Gemma checkpoint on GPU (not cached; their paths share every line of code with Qwen's except the
  family profile, which the unit tests check with their tokenizers); Qwen3-8B base (not cached).
