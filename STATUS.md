# STATUS

**Last updated:** 2026-11-02 22:30 UTC by Claude (Opus 5.5, agent session for Grace), pushed to `main` on top of `2a1eb01` (logit distortion: eligible-quota pilot_v3, prediction-case dedup, length sensitivity).
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
| implementation | NLL arm built (`analyze.py --locked`, reproduces the pilot); judge built (`judge.py`, prompt v1) but never run against the API; flags have `token_loop` and family handoffs | `analyze.py` is a stub. Shared: isolated authorship text sets built (`data/texts_authorship.jsonl`, merged). **Primary route (Grace, 11-02): original conversations.** Builder `core.build_conversations`; current candidate pilot `candidates/logit_distortion/pilot_v3/` (eligible-quota: 8 contributing conversations per source, 175 targets, 119 provisionally eligible, Dolci-checked; prediction-case dedup; length-sensitivity audit); not scorable yet (original-context scoring not built; `core.score` refuses the records) | `minimal_pairs.py` (score, analyze) implemented, CPU-tested; 8 draft items. 240-item set not built; interp arm and behavioral companion not started |
| data collection | OLMo: none. Qwen3-8B, Gemma 4 31B: pilot only (90 texts) | none in repo | none in repo |
| results | pilot only (`pilot/locked/`): NLL arm fires for Gemma 4 31B IT, not Qwen3-8B, not OLMo; no `results/<run>/vestigial_regression/` yet | none | none |
| blockers | Gemma base needs 67.5 GB free (47 GB seen); no API key when the judge was built; judge labels need a teammate; Qwen3 think-block undecided | WildChat candidates mostly Dolci-matched (5 of 74 targets eligible in the pilot): WildChat sampling/contamination policy, final sizes and length thresholds to decide before scoring; original-context scoring not built. Progress otherwise: not visible in repo, ask Grace | 240 items to build; README/AGENTS still say "nothing implemented". A persona-carryover scoring run was seen under `/root/adeeb_capstone` (not in repo): ask Adeeb |

## Shared pipeline (owner: Cole)

- **Built and merged:** model families (OLMo 3 7B, Gemma 4 31B, Qwen3-8B) in `core/config.py`; family-aware `core.score`/`core.generate` (fp32, CPU offload, int8 generation); `core/hf_cache.py`; `scripts/run_gpu.sh` with `FAMILY=`; 100-sentence prefill; flags; judge; locked analysis; authorship sets; runbook. Branch `pipeline-prep-2026-11-03`, merged to `origin/main` by PR #7 at 18:10 UTC (`5750f89`). 154 tests passed at `39bcc45`. Not run on GPU end to end for OLMo or Gemma; Qwen3-8B instruct smoke only.
- **Pushed to `main` directly (authorized, no PR), 11-02:** original-conversation candidate builder (`core.build_conversations`, `core.conversations`, `core.conversation_audit`) from local branch `conversation-builder` (supersedes `conversation-candidates`), plus the Dolci-checked `pilot_v2`; `core.score`, `core.render`, `core.add_replies` and `core.check_contamination` refuse its records. Docs: `docs/conversation_candidates.md`.
- **Next run to launch:** OLMo 3 7B, `CLEAR_CACHE=1 FAMILY=olmo3-7b bash scripts/run_gpu.sh`, then Qwen3-8B, then Gemma 4 31B (runbook, "Launch order").

## Open decisions (decide-by days are proposed from the schedule)

| decision | owner | decide by | where |
|---|---|---|---|
| Qwen3 assistant-slot think block (changes the VR asymmetry and logit distortion's `asst@asst`) | Cole, with Grace and Adeeb | Tue, before the Qwen run | runbook #1 |
| Minimum judge-arm rate difference | Cole | Tue, before judge results are read | VR README |
| Judge validation labels (60 samples, `judge_validation/sheet.csv`), by a teammate other than the pilot agent | Cole to assign | labels Tue, needed before Wed analysis | runbook |
| Judge agreement bar (80% default) | Cole | Tue, before the first judge run | VR README |
| Gemma scale (`--n-per-set 100` vs all texts) and the disk to fit it | Cole | Tue morning | runbook #3 |
| Logit distortion: decided 11-02, original conversations (primary), isolated sets for comparison. Still open: minimum length and opening/follow-up stratification (no UltraChat opening user survives 20 tokens), final quotas, whether WildChat restricted to Dolci-absent conversations is acceptable, persona/jailbreak prompt policy, then original-context scoring | Grace, with Cole | Tue morning | `candidates/logit_distortion/pilot_v3/inspection.md` |
| Overgeneralization bracketed criteria values (unconfirmed) | Adeeb | Tue | its README |
| `AGENTS.md` Outputs section for the `results/<run>/<family>/` layout; Spearman base points (excluded at implementation, not pre-registered) | Cole | Wed | runbook #6, #4 |

## Next actions (at most three each)

- **Cole:** (1) decide Qwen3 think block and Gemma scale, free disk for the Gemma base; (2) launch the OLMo run, then Qwen; (3) set `ANTHROPIC_API_KEY`, run `judge pilot`, assign the label sheet.
- **Grace:** (1) read `pilot_v3/inspection.md` and `length_sensitivity.md`, then set the minimum length, position stratification and quotas before scoring; (2) original-context scoring (subsequent task); (3) write `analyze.py` against its scores once data exist.
- **Adeeb:** (1) confirm the bracketed criteria values; (2) build the 240-item set (`HANDOFF.md`); (3) say where the persona-carryover work lives.
