# Combined collection, 2026-11-03: conversation scores + incomplete-turn generations

Collection only: no judge, no flags, no scores of the generated continuations, no verdicts. Provisional pilot collection for logit distortion
(`candidates/logit_distortion/pilot_v3`, unchanged), not a final dataset decision.

Runner: `attic/scripts/collect_conv_incomplete.py` (reuses `attic/scripts/overnight_collect.py`'s task machinery; that driver and its CLI are unchanged).

```bash
python attic/scripts/collect_conv_incomplete.py --dry-run   # both tasks for all 8 checkpoints, and what would be reused
python attic/scripts/collect_conv_incomplete.py --smoke     # OLMo base: one conversation + 3 prefills -> results/collect_conv_incomplete/smoke/
python attic/scripts/collect_conv_incomplete.py             # the collection (resumable; run it in tmux)
python attic/scripts/collect_conv_incomplete.py --status    # read-only progress
```

## Tasks per checkpoint

Order OLMo 3 7B (base, SFT, DPO, Instruct) -> Qwen3-8B (base, instruct) -> Gemma 4 31B (base, instruct). For each checkpoint both tasks run
(separate processes) before the checkpoint's weights are cleared:

| task | command (settings unchanged) | output |
|---|---|---|
| conversation | `core.score_conversations --family F --candidates pilot_v3 --model S --context-limits-from <frozen manifest> --batch-size 1` (fp32; batch 1 because OLMo 3 7B ran out of memory at batch 8 and 4 on the ~4.1k-token inputs; also no padding or batch-composition effects, identical for every stage) | `results/logit_conversations/<F>/core/conversation_scores/<S>.csv`, `<S>_tokens.parquet`, `<S>_provenance.json` |
| incomplete | `core.generate --family F --run capstone --model S --prefill data/prefill_incomplete_200.jsonl --save-ids` (the command of `overnight_collect.py --stratum incomplete`) | `results/capstone/<F>/core/generations_incomplete/<S>.jsonl` + `<S>_provenance.json` (model revision, prefill hash, sampling settings, code hashes) |

Before any scoring each family's inputs are frozen: `--prepare-only` writes the manifest, or confirms that an existing one has the same fingerprint, and
`config_provenance.json` records each stage's `config.json` revision and hash. Every stage of a family is scored against the same frozen
manifest. Exclusions are family-specific and listed in each family's `exclusions.jsonl` and `rendering_audit.md`.

## Rules

- An existing output is reused only if it validates: inputs, settings and model revision match, and the integrity checks pass. An output that does not validate is reported and never overwritten.
- Outputs are written to a temp directory and moved into place after a clean exit. Out-of-memory is retried once with half the batch size.
- The runner waits until no other process is using the GPU and never starts alongside one. It never frees disk to make a download fit; a checkpoint that does not fit is reported as blocked.
- Cleanup removes only weight files (`*.safetensors`) that this run downloaded. Configs and tokenizers stay, and repos whose weights were already cached are never touched.
- Validation, conversation scores: fingerprint, target set, spans and input hashes are checked against the manifest. Token ids and positions are re-rendered from the candidates and must match. NLL and entropy must be finite, with entropy in [0, ln V], and per-target summaries are recomputed from the token table. `--check-stages` checks the whole family.
- Validation, generations: 200 prefills x 3 samples, prefill ids and text, `continuation_ids` present and decoding to the stored continuation, provenance matching the current settings and output hash.
- The collection is complete only if all 16 tasks validate (`--status`).

## Gemma 4 31B: RAM-backed cache

The root disk has about 58 GB free, but a Gemma checkpoint needs 67.5 GB, and teammates' caches are not ours to free. So the Gemma pass
(`attic/scripts/collect_gemma_shm.sh`) uses a dedicated cache, `/dev/shm/grace-gemma-hub/{hub,xet}`. It sets `HF_HUB_CACHE` and `HF_XET_CACHE` before Python
starts; checked: `huggingface_hub`, `transformers` and `core.hf_cache` all resolve to it, and partial downloads (`*.incomplete`) are staged
inside it. Results, manifests, provenance, state and logs stay on disk in the worktree.

The pass starts after the OLMo/Qwen driver exits (it waits for the lock) and runs `--only gemma4-31b`, one checkpoint at a time. Memory check
before launch: the container's cgroup v1 limit is 188 GB (hierarchical and memory+swap limits are the same, and there is no swap). Needed: 62.5 GB of
weights in `/dev/shm`, counted against that limit; 92.8 GB for the fp32 CPU-offloaded part of the model (125.0 GB fp32 minus 32.2 GB on the GPU);
a 15 GB margin. Total 170.3 GB, against 185.1 GB not pinned at 12:30 UTC. `/dev/shm` had 94 GB free. Memory is sampled every 30 s into
`results/collect_conv_incomplete/gemma_memory.log`. Revisions, fingerprints, precision and settings are unchanged: the freeze step re-checks
the recorded config revisions, and the manifest refuses a changed tokenizer. At the end only `/dev/shm/grace-gemma-hub` is removed.

## Incomplete prefills to review later (not edited)

`data/prefill_incomplete_200.jsonl` is used as committed. None of the 200 ends in terminal punctuation. The 13 below are cut within the last
6–18 characters of the original turn. They are still unfinished, but a model that adds a few words and closes the turn is finishing the user's message
almost exactly as the user did, so the judge-arm analysis may want to look at them separately:

| id | cut type | prefill (end) | real continuation | chars left |
|---|---|---|---|---:|
| `incomplete-034` | transitive verb | …sult of using them? Rate from 1 to 10 your confidence that your answer | is correct. | 12 |
| `incomplete-062` | auxiliary/modal | …n this day I have 73 apples. I ate 24 yesterday. So how many do I have | left? | 6 |
| `incomplete-076` | preposition | …t: Lily is a cute girl. She has a lusty disposition. Do something with | this hint. | 11 |
| `incomplete-082` | preposition | …tary density is key to calculation of density of entire planet made of | composition. | 13 |
| `incomplete-087` | preposition | …haracters (Sash Lilac, Carol Tea, Milla Basset and Torque) + Naruto in | kid Torque (cute) | 18 |
| `incomplete-124` | preposition | …g to close at this point, so the attorney's advice has not conflict of | interest. | 10 |
| `incomplete-127` | article/determiner | … a scale from 1 to 4 where 1 is clothed and 4 is naked, what would the | numbers mean? | 14 |
| `incomplete-163` | preposition | …ualian hypothesis, lifeforms were designed as YY chromosome instead of | XX and XY. | 11 |
| `incomplete-186` | transitive verb | …i will give you japanese words, texts, and sentences and you will tell | me what it means. | 18 |
| `incomplete-190` | auxiliary/modal | …ive you some background information for a proposal I need to write, do | you understand? | 16 |
| `incomplete-193` | preposition | …use common language. apophenia vs pareidolia, is otherwise not easy to | understand. | 12 |
| `incomplete-195` | conjunction | …gh Four' Defendant". Who most likely freed the defendant: the judge or | the FBI? | 9 |
| `incomplete-197` | article/determiner | …a manga based off of Need For Speed and I'd like to tell you about the | characters. | 12 |

## Results

(filled in when the collection finishes)
