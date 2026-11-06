# Morning hand-off, spec R overnight (night of 2026-11-05)

All of this is **exploratory** and **provisional**, outside every verdict, and not blind. Judged measures use the local Gemma 4 31B-it judge (not hand-checked, and Gemma judges its own replies). Nothing was sent outside the repo. Slide changes are proposals only; no slide file was edited.

| experiment | state | reading and key numbers | which slide it could change and how (proposal) | report |
|---|---|---|---|---|
| R0 Q on in-character replies only | done | voice effects on `denies_feelings` (AI role, none minus given): olmo3-7b user: +72 [+43, +96] VOICE EFFECTS SURVIVE; qwen3-8b system: +24 [+6, +46] VOICE EFFECTS SURVIVE; qwen3-8b user: +43 [+17, +71] VOICE EFFECTS SURVIVE; gemma4-31b system: +39 [+15, +62] VOICE EFFECTS SURVIVE; gemma4-31b user: +47 [+29, +68] VOICE EFFECTS SURVIVE | Slide E/Q-type claims about what the role says: states whether the voice effect survives the in-character filter. Proposal only. | results/role_says/role_says.md (SENSITIVITY) |
| R1 strict layout | done | readings: {'CLAIM HOLDS': 11, 'CLAIM SHRINKS': 4, 'in between': 4}. Gemma light-role strict levels vs old: atlas/light_system old 100 strict 82 heavy 17; atlas/light_user old 100 strict 69 heavy 10; L4:factual/light_role old 62 strict 2 heavy 0; L4:howto/light_role old 100 strict 55 heavy 29; L4:advice/light_role old 94 strict 15 heavy 4 | Slide C: the formatting claim holds for OLMo/Qwen/Apertus; for Gemma light roles it is mostly bold and short lists. Proposal only. | results/overnight_2026-11-05/R1/R1.md |
| R2 did the reply help | done | olmo3-7b/sys: judge gap +17 (regex +46) PARTLY THE DETECTOR; olmo3-7b/user: judge gap +49 (regex +76) STYLE FALL HOLDS; qwen3-8b/sys: judge gap +50 (regex +89) STYLE FALL HOLDS; qwen3-8b/user: judge gap +56 (regex +82) STYLE FALL HOLDS; gemma4-31b/sys: judge gap +51 (regex +50) STYLE FALL HOLDS; gemma4-31b/user: judge gap +57 (regex +29) STYLE FALL HOLDS | Slide D: refusal fall by style (judge vs regex). Proposal only. | results/overnight_2026-11-05/R2/R2.md |
| R3 prefill belief robustness | running | - | - | - |
| R4 OLMo by stage | done | see table | A new 'what post-training does' slide. Proposal only. | results/overnight_2026-11-05/R4/R4.md |
| R5 voice on fresh questions | done | olmo3-7b/system: strict +64, stock +54: NOT READ; olmo3-7b/user: strict +74, stock +69: claim does not hold here; qwen3-8b/system: strict +24, stock +12: claim does not hold here; qwen3-8b/user: strict +76, stock +28: claim does not hold here; gemma4-31b/system: strict +13, stock +4: claim does not hold here; gemma4-31b/user: strict +18, stock +4: claim does not hold here | Q's voice claim on fresh questions. Proposal only. | results/overnight_2026-11-05/R5/R5.md |
| R6 Q on Llama and Apertus | queued/running | - | Q breadth. Proposal only. | results/role_says/role_says.md |
| R7 knows versus says | done | olmo3-7b/A: forced 65 vs free 63: DOESN'T KNOW, EVEN WHEN FORCED; olmo3-7b/B: forced 83 vs free 42: KNOWS BUT DOESN'T SAY; qwen3-8b/A: forced 64 vs free 77: CAN'T TEST (plain Assistant under 80% when forced); qwen3-8b/B: forced 100 vs free 65: KNOWS BUT DOESN'T SAY; gemma4-31b/A: forced 31 vs free 15: in between; gemma4-31b/B: forced 99 vs free 88: CAN'T TEST (plain Assistant under 80% when forced) | Slide F (Gemma characters stop answering). Proposal only. | results/overnight_2026-11-05/R7/R7.md |

## Failures, retries and changes from the spec

- `r_failures.log`:
  - FAILED(1): /opt/arena-env/bin/python -m extras.voice_fresh gen --family olmo3-7b
  - FAILED(1): /opt/arena-env/bin/python -m extras.role_says_breadth gen --family llama3.1-8b
  - FAILED(2): /opt/arena-env/bin/python -m extras.role_says_breadth gen --family llama3.1-8b
  - FAILED(3): /opt/arena-env/bin/python -m extras.role_says_breadth gen --family llama3.1-8b
  - FAILED(1): /opt/arena-env/bin/python -m extras.role_says_breadth gen --family llama3.1-8b
  - FAILED(2): /opt/arena-env/bin/python -m extras.role_says_breadth gen --family llama3.1-8b
  - FAILED(3): /opt/arena-env/bin/python -m extras.role_says_breadth gen --family llama3.1-8b
  - FAILED(1): /opt/arena-env/bin/python -m extras.role_says_breadth gen --family llama3.1-8b
  - FAILED(2): /opt/arena-env/bin/python -m extras.role_says_breadth gen --family llama3.1-8b
  - FAILED(3): /opt/arena-env/bin/python -m extras.role_says_breadth gen --family llama3.1-8b
- `queue_marble.txt.failures`:
  - === 01:03:32 step r5_marble FAILED try 1
  - === 02:12:25 step r6_marble FAILED try 1
  - === 02:13:35 step r6_marble FAILED try 2
  - === 02:14:45 step r6_marble FAILED try 3
- See the 'Changes from the spec' lines in each report and in `docs/decisions.md` (2026-11-06): R1 reading interpretation; R2 validation set (the 56-reply hand check file was not in the repo; the 300 R/C human labels of the orthogonal-steering detector were used); R2 L3 rows are the atlas's benign questions; R3 reuses spec O's wording-1 files; Gemma wording-2 base last; R5 uses all 90 L4 questions for OLMo too; R7 forced-answer design.

