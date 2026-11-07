# Edits: release the user-turn result (Adeeb, 2026-11-06)

Apply these to `measures/overgeneralization/`, commit and push. No GPU, no API. Don't change the Pre-registered criteria text, Cole's files or `core/`. Run `slide_figures` only if needed; nothing here changes a figure. Keep other sections as they are; spec R's other findings come in a later edit. Update `STATUS.md`.

## 1. `presentable_findings.md`, finding 1: replace the whole section

### 1. Does the model sound like the Assistant when it writes as the user?

**Firmness: pre-registered and doesn't fire, but provisional.** The judge labels aren't human-validated: the team decided on 2026-11-06 to present before validation. The label is rare, so the test has little power, and a hand check could still change the rare cases.

**Experiment.** We start a real user message and let the model finish it. Cole's Claude judge labels each continuation, including "user voice with Assistant traits". The test fires at +5 points at OLMo final, against base, with the range excluding 0. OLMo's 200 WildChat prefills decide. Everything else below is supporting and was analysed after that verdict was known.

**Result.** It doesn't fire: OLMo final +0.2 [0.0, +0.6], from 0.0 to 0.2%, on 156 of 200 prefills.

| Model | Prefills | Base → final (%) | Change [95% range] | Prefills used |
|---|---|---|---|---|
| **OLMo 3 7B, decides** | **WildChat** | **0.0 → 0.2** | **+0.2 [0.0, +0.6]** | **156** |
| OLMo 3 7B | oasst1 | 0.8 → 0.0 | −0.8 [−2.0, 0.0] | 123 |
| OLMo 3 7B | UltraChat | 0.0 → 0.0 | 0.0 [0.0, 0.0] | 152 |
| Qwen3-8B | WildChat | 0.3 → 0.0 | −0.3 [−0.9, 0.0] | 117 |
| Qwen3-8B | oasst1 | 0.0 → 0.5 | +0.5 [0.0, +1.4] | 71 |
| Qwen3-8B | UltraChat | 2.2 → 0.0 | −2.2 [−6.7, 0.0] | 45 |
| Gemma 4 31B | WildChat | 0.0 → 0.0 | 0.0 [0.0, 0.0] | 10 |
| Llama 3.1 8B* | WildChat | 0.5 → 1.0 | +0.6 [−0.8, +2.4] | 146 |
| Llama 3.1 8B* | oasst1 | 0.5 → 5.0 | +4.5 [+1.0, +8.9] | 101 |
| Llama 3.1 8B* | UltraChat | 1.3 → 2.4 | +1.1 [−2.6, +4.9] | 78 |
| Apertus 8B* | oasst1 | 0.0 → 3.7 | +3.7 [0.0, +9.8] | 41 |
| Apertus 8B* | UltraChat | 0.0 → 2.8 | +2.8 [0.0, +7.8] | 47 |

*Breadth check, added later. Apertus has no WildChat labels.

**Analysis.**
- **Almost nothing to count:** the label appears in 20 of about 7,800 judged final samples, 13 of them Llama's. One reader (marble's Claude, not blind) thought about half were arguable.
- **Gemma can't really be read:** at final it writes in the user's voice in only 3.5% of samples, which is why only 10 prefills are usable. That belongs to vestigial regression (Cole's measure), not here.
- **Llama's oasst1 row is the closest to the bar:** just under it, with a range excluding 0. It's one set in a breadth family, resting on 11 flagged samples.

**Judge v3 (Cole, 2026-11-05).** It separates "finished the user's thought, then the assistant took over" from abandoning the sentence, and records traits as their own field over all user text. It has only been run on Cole's 5 open openers, not on our prefills, and isn't validated. The v3 traits field, base → final: OLMo 3 of 552 → 0 of 481; Qwen 2 of 590 → 0 of 579; Gemma 1 of 589 → no user text at all. So v3 agrees, on different prompts. Our verdict stays on v2 labels, because results existed before v3 and the rule is pre-registered. A v3 run on our prefills would cost about $23 and needs a paid API approval; it hasn't been run.

**Keyword check, no judge, outside the verdict.** Share of samples with a stock phrase or markdown the prefill lacked. It falls in OLMo (−13, −19, −11 on WildChat, oasst1, UltraChat), Qwen (−18, −38, −45) and Gemma (−14, WildChat). It rises in Llama (+13, +36, +25) and Apertus (+24, +37, +39). Part of that rise is switching into answering mid-turn. But it also rises among samples the judge calls user voice: Llama +9.0 [+2.7, +15.4], +30.0 [+19.6, +40.4], +14.7 [+4.9, +24.6]; Apertus (oasst1, UltraChat) +21.1 [+3.7, +38.6] and +48.2 [+33.7, +62.1], on 41 and 47 prefills only. Those flags are mostly numbered lists. Of 28 such samples read by the laptop Claude (one reader, not blind), most looked like users writing structured requests, and about a quarter read like Assistant text. **So this is a lead in two breadth families, not a result.**

**Shared setup.** 199 of the 200 prefills come from the team's WildChat exchange set. The oasst1 and UltraChat sets have 200 prefills each, 3 samples each, same cut rule, judge and prompt (v2).

**Known limits.** Not human-validated. The judge refused some samples, which are excluded (up to 44 of 600 per file). Prefills with no user-voice sample are dropped (44 of 200 for OLMo's verdict row). Gemma's exchange prefills aren't judged.

**Sources:** `measures/overgeneralization/README.md`, `results/user_turn_2026-11-06/user_turn.md`, `results/capstone/<family>/overgeneralization/user_traits_judge.md`, `measures/overgeneralization/user_traits_v3_check.py`, `docs/decisions.md` 2026-11-06.

Also in `presentable_findings.md`: remove any other line saying this result is held back, or that sheets must be finished first. Update "Last updated".

## 2. `slide_brief.md`

**Opening paragraph:** replace "Numbers are as of 5 November and provisional until the hand checks are in." with "Numbers are as of 6 November. Judge results are provisional: the team is presenting them before human validation."

**Bottom line:** replace the last sentence with "This measure's two pre-registered tests give no confirmed support for dominance."

**Replace all of Slide B with:**

### Slide B. The user's turn

- **Kicker:** Overgeneralization · user prefill test
- **Title:** The judge finds almost no Assistant traits in the user's turn
- **Explain:** we start a real user message and let the model finish it. A Claude judge labels whether it stays the user, and whether that text has Assistant habits. Rule, fixed in advance: OLMo final must rise at least 5 points from base.
- **Closing line:** The test doesn't fire. Two judge versions and a keyword check agree in OLMo and Qwen. Llama and Apertus leave a question open.

| Model | Prefills | Base → final (%) | Change [95% range] |
|---|---|---|---|
| **OLMo 3 7B, decides** | **WildChat** | **0.0 → 0.2** | **+0.2 [0.0, +0.6]** |
| OLMo 3 7B | oasst1 / UltraChat | 0.8 → 0.0 / 0.0 → 0.0 | −0.8 [−2.0, 0.0] / 0.0 |
| Qwen3-8B | WildChat / oasst1 / UltraChat | 0.3 → 0.0 / 0.0 → 0.5 / 2.2 → 0.0 | −0.3 / +0.5 / −2.2 |
| Llama 3.1 8B* | WildChat / oasst1 / UltraChat | 0.5 → 1.0 / 0.5 → 5.0 / 1.3 → 2.4 | +0.6 / +4.5 [+1.0, +8.9] / +1.1 |
| Apertus 8B* | oasst1 / UltraChat | 0.0 → 3.7 / 0.0 → 2.8 | +3.7 / +2.8 |

*Breadth check, added later. Only OLMo on WildChat decides.

**Must say:**
- These judge labels aren't human-validated.
- The label is very rare: 20 of about 7,800 final samples, and about half look arguable. So this is a weak test, not proof of absence.
- Gemma 31B can't be read: it writes as the user in only 3.5% of final samples. That's Cole's measure.

**Say if asked:**
- **Judge v3:** Cole's newer judge hasn't been run on these prefills. On his 5 open openers, its traits field goes OLMo 3 of 552 → 0 of 481 and Qwen 2 of 590 → 0 of 579.
- **Keyword check:** it falls in OLMo, Qwen and Gemma. It rises in Llama and Apertus (+13 to +39), partly from switching into answering. It also rises among user-voice samples (+9 to +48), but those are mostly numbered lists, and most of the 28 we read look like users writing structured requests. Treat this as a lead.

**Slide G, first card:** replace with "**No confirmed spread:** neither pre-registered test passes, the belief test or the user prefill test."

**Please don't present table:** replace the "Any user prefill test number" row with: | "No Assistant traits in the user's turn" as a firm finding | The label is rare, not human-validated, and Llama and Apertus keyword checks rise |

**How firm table:** add this row after Slide A: | B. User's turn | Pre-registered, provisional | Doesn't fire, but rare label, no human validation yet |

## 3. `README.md`, User-slot arm

Add this dated note under the arm's **Status**, leaving the numbered criteria unchanged:

> **Note, 2026-11-06.** Criteria unchanged. Result: doesn't fire (`user_traits_judge.md`; `results/user_turn_2026-11-06/user_turn.md`), provisional, not human-validated (team decision, `docs/decisions.md` 2026-11-06). The verdict uses Cole's judge prompt v2 (now `judge_prompt_v2.md`; `judge_prompt.md` is v3). v3 hasn't been run on these prefills. If it is, `user_traits.py` needs its v3 labels handled first (it stops on unknown labels), and the v3 reading is a sensitivity check: share of samples with `user_text = user` that have `assistant_traits`.

Also fix the stale "100 prefills x 3 samples" here and in `user_traits.py`'s docstring: the judge arm uses 200 prefills x 3 samples (`docs/decisions.md` 2026-11-04).
