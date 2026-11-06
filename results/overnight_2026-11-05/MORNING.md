# Morning hand-off, spec R overnight (night of 2026-11-05)

All of this is **exploratory** and **provisional**, outside every verdict, and not blind. Judged measures use the local Gemma 4 31B-it judge (not hand-checked, and Gemma judges its own replies). Nothing was sent outside the repo. Slide changes are proposals only; no slide file was edited.

| experiment | state | reading and key numbers | which slide it could change and how (proposal) | report |
|---|---|---|---|---|
| R0 Q on in-character replies only | done | voice effects on `denies_feelings` (AI role, none minus given): olmo3-7b user: +72 [+43, +96] VOICE EFFECTS SURVIVE; qwen3-8b system: +24 [+6, +46] VOICE EFFECTS SURVIVE; qwen3-8b user: +43 [+17, +71] VOICE EFFECTS SURVIVE; gemma4-31b system: +39 [+15, +62] VOICE EFFECTS SURVIVE; gemma4-31b user: +47 [+29, +68] VOICE EFFECTS SURVIVE | Slide E/Q-type claims about what the role says: states whether the voice effect survives the in-character filter. Proposal only. | results/role_says/role_says.md (SENSITIVITY) |
| R1 strict layout | done | readings: {'CLAIM HOLDS': 11, 'CLAIM SHRINKS': 4, 'in between': 4}. Gemma light-role strict levels vs old: atlas/light_system old 100 strict 82 heavy 17; atlas/light_user old 100 strict 69 heavy 10; L4:factual/light_role old 62 strict 2 heavy 0; L4:howto/light_role old 100 strict 55 heavy 29; L4:advice/light_role old 94 strict 15 heavy 4 | Slide C: the formatting claim holds for OLMo/Qwen/Apertus; for Gemma light roles it is mostly bold and short lists. Proposal only. | results/overnight_2026-11-05/R1/R1.md |
| R2 did the reply help | running on vinyl | - | - | - |
| R3 prefill belief robustness | running | - | - | - |
| R4 OLMo by stage | queued/running | - | A new 'what post-training does' slide. Proposal only. | results/overnight_2026-11-05/R4/R4.md |
| R5 voice on fresh questions | queued/running | - | - | - |
| R6 Q on Llama and Apertus | queued/running | - | Q breadth. Proposal only. | results/role_says/role_says.md |
| R7 knows versus says | queued/running | - | - | - |

## Failures, retries and changes from the spec

- See the 'Changes from the spec' lines in each report and in `docs/decisions.md` (2026-11-06): R1 reading interpretation; R2 validation set (the 56-reply hand check file was not in the repo; the 300 R/C human labels of the orthogonal-steering detector were used); R2 L3 rows are the atlas's benign questions; R3 reuses spec O's wording-1 files; Gemma wording-2 base last; R5 uses all 90 L4 questions for OLMo too; R7 forced-answer design.

