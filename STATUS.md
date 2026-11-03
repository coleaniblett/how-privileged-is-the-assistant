# STATUS

**Last updated:** 2026-11-03 08:20 UTC by Claude (Sonnet 5.5, agent session for Cole), branch `pipeline-prep-2026-11-03` (checked out in the worktree `/root/how-privileged-is-the-assistant-collect`) at `83f6b41` + this commit. Overnight collection finished 03:59 UTC; `docs/overnight_2026-11-03.md`.
Source: commit history of all branches after `git fetch --all`, the measure READMEs, `docs/decisions.md`, `docs/runbook_2026-11-03.md`, `AGENTS.md`.
Only what the repo shows; where it shows nothing the cell says so. Detail lives in the READMEs and `decisions.md`, not here.

## Schedule

| Mon 11-02 | Tue 11-03 | Wed 11-04 | Thu–Fri |
|---|---|---|---|
| design (done) | data collection: vestigial regression, logit distortion. Overgeneralization builds its minimal pairs | overgeneralization collects; analysis | write-up, presentation |

## Measures

| | **Vestigial regression** | **Logit distortion** | **Overgeneralization** |
|---|---|---|---|
| owner | Cole | Grace | Adeeb |
| spec | locked 2026-11-02 (two arms: NLL asymmetry + LLM judge); NLL criteria revised after the pilot, so OLMo NLL verdicts are not blind | design settled, criteria in README | criteria drafted 2026-11-02, bracketed values unconfirmed |
| implementation | NLL arm built (`analyze.py --locked`, reproduces the pilot); judge built (`judge.py`, prompt v1) but never run against the API; flags have `token_loop` and family handoffs | `analyze.py` is a stub. Shared: isolated authorship text sets built (`data/texts_authorship.jsonl`, merged). Unmerged: `core/build_conversations.py` and candidates in local branch `conversation-candidates` | `minimal_pairs.py` (score, analyze) implemented, CPU-tested; 8 draft items. 240-item set not built; interp arm and behavioral companion not started |
| data collection | **done 2026-11-03 03:59 UTC**, all collected: OLMo 3 7B (base, SFT, DPO, Instruct), Qwen3-8B (base, IT; think block ON primary, OFF robustness), Gemma 4 31B (base, IT; first 100 texts per set): scores, authorship scores, 300 generations each. No flags, judge or analysis run on it yet | isolated authorship sets scored for all 8 checkpoints (`results/capstone/<family>/core/scores_authorship/`). Original-conversation scoring (`core.score_conversations`, on `main`): not run on these checkpoints (nothing under `results/`) | none in repo |
| results | pilot only analysed (`pilot/locked/`): NLL arm fires for Gemma 4 31B IT, not Qwen3-8B, not OLMo. Capstone data collected and integrity-checked, not yet analysed; no `results/<run>/vestigial_regression/` yet | none | none |
| blockers | no API key yet (judge never run); judge labels need a teammate; the think-block default and driver commits are not on `main` (its Qwen3 default is still OFF) | text route unsettled (isolated sets vs original conversations); branch unpushed. Progress otherwise: not visible in repo, ask Grace | 240 items to build; README/AGENTS still say "nothing implemented". A persona-carryover scoring run was seen under `/root/adeeb_capstone` (not in repo): ask Adeeb |

## Shared pipeline (owner: Cole)

- **Built and merged:** model families (OLMo 3 7B, Gemma 4 31B, Qwen3-8B) in `core/config.py`; family-aware `core.score`/`core.generate` (fp32, CPU offload, int8 generation); `core/hf_cache.py`; `scripts/run_gpu.sh` with `FAMILY=`; 100-sentence prefill; flags; judge; locked analysis; authorship sets; runbook. Branch `pipeline-prep-2026-11-03`, merged to `origin/main` by PR #7 at 18:10 UTC (`5750f89`). 154 tests passed at `39bcc45`. Not run on GPU end to end for OLMo or Gemma; Qwen3-8B instruct smoke only.
- **Not merged:** `pipeline-prep-2026-11-03` is ahead of `main`: the Qwen3 think-block default ON (`9399d2d`), the collection driver and resilience scripts (supervisor, tmux launcher, status, report), the worktree runbook note, and the overnight outputs and report. The earlier local-only `conversation-candidates` branch appears to have landed on `main` as `a9f4858` (same subject). This `STATUS.md` conflicts with the version on `main`; reconcile at merge.
- **Collection:** complete (see `docs/overnight_2026-11-03.md`; 28 of 28 expected outputs, 0 failures, OLMo scores bit-identical to the frozen run). **Next:** `flags` per family, then `analyze --locked --run capstone --first 100` (same N for all families); the judge after the key and the teammate-labelled sheet exist.

## Open decisions (decide-by days are proposed from the schedule)

| decision | owner | decide by | where |
|---|---|---|---|
| ~~Qwen3 assistant-slot think block~~ **decided 2026-11-02**: ON is primary, OFF is the robustness check (both collected; mean ON-OFF difference in `asst@asst` about 0.02 nats) | Cole | done | `docs/decisions.md` |
| Minimum judge-arm rate difference | Cole | Tue, before judge results are read | VR README |
| Judge validation labels (60 samples, `judge_validation/sheet.csv`), by a teammate other than the pilot agent | Cole to assign | labels Tue, needed before Wed analysis | runbook |
| Judge agreement bar (80% default) | Cole | Tue, before the first judge run | VR README |
| ~~Gemma scale~~ **decided**: `--n-per-set 100`; the analysis must use `--first 100` for every family | Cole | done | `docs/overnight_2026-11-03.md` |
| Logit distortion text route: isolated authorship sets (merged) vs original conversations (unmerged) | Grace, with Cole | Tue morning | `conversation-candidates` README note |
| Overgeneralization bracketed criteria values (unconfirmed) | Adeeb | Tue | its README |
| `AGENTS.md` Outputs section for the `results/<run>/<family>/` layout; Spearman base points (excluded at implementation, not pre-registered) | Cole | Wed | runbook #6, #4 |

## Next actions (at most three each)

- **Cole:** (1) merge `pipeline-prep-2026-11-03` (at least `9399d2d`) into `main` and reconcile `STATUS.md`; (2) run flags and `analyze --locked --run capstone --first 100` on the collected data; (3) set `ANTHROPIC_API_KEY`, run `judge pilot`, assign the label sheet.
- **Grace:** (1) say what exists for logit distortion (not visible in repo); (2) pick the text route; (3) write `analyze.py` against core scores once data exist.
- **Adeeb:** (1) confirm the bracketed criteria values; (2) build the 240-item set (`HANDOFF.md`); (3) say where the persona-carryover work lives.
