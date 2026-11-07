# Spec J: Overnight plan, 2026-11-05 until 10:00 AM UK (= UTC)

Approved by Adeeb about 12:45 AM: queue Llama and Apertus; use available compute to improve and build on the findings; try things and see what sticks. **Additional API spend is NOT approved.** Adeeb is offline until morning. Laptop Claude may be off for stretches, so marble runs this without waiting; if a choice is unclear, take the cheaper reversible option and note it.

## Rules
1. **API (updated about 12:50 AM):** Adeeb approved up to $50 of API spend. Read it as a TOTAL cap for the night, including the prefill judge already running (estimate $24). Beyond that, the only approved use is Cole's judge, unchanged, on Llama and Apertus prefills (estimate $24): test 50 samples first, keep a running spend estimate in `/root/overnight_log.md`, and stop all paid calls at an estimated $45 total, leaving margin. Gemma prefills are generated but not judged unless laptop Claude says so. Nothing else may call a paid API. Delete `/root/.private/judge_api_key` once the last approved judge run is verified.
2. **Blinding:** no label counts, rates, analysis or pushing of prefill-judge labels until laptop Claude says so. Applies to existing and new judge labels.
3. **Verdict:** pre-registered criteria and frozen item sets stay unchanged. Everything new is exploratory, outside every verdict. New families are not blind: label them a breadth check. Post-hoc checks are labelled post-hoc.
4. **Order of work for each experiment:** plan and pre-set readings logged in `docs/decisions.md` and materials committed BEFORE scoring. Report numbers as they come, including nulls. Give timing after a small sample.
5. **Team boundaries:** no edits to Cole's or Grace's measure folders; `core/` changes only if backwards-compatible, with a `core:` commit and a decisions entry. Update `STATUS.md` as you go.
6. **Machines:** marble and vinyl only. Never touch nurse, wish, quiet, or `/root/g4nla`. One GPU job per server unless tested safe. Keep both GPUs busy; chain jobs in nohup scripts so the next starts by itself. Mind disk: remove a model only after its results are saved, and only models we downloaded for this; check other sessions' use first.
7. **Commits:** small reports, tables and code only. No raw dumps, no keys, no blind-check answer keys. Hand-check sheets are fine.
8. **Stop:** nothing new after 8:30 AM unless it will end by 9:30 AM. By 9:30 AM everything worth keeping is pushed and `/root/overnight_log.md` is current. Don't leave half-finished jobs running past 10:00 AM.
9. **Log:** after each job, append to `/root/overnight_log.md`: what ran, numbers, reading, commit, problems. Laptop Claude reads this.

## Queue, in priority order. Assign to whichever GPU is free and fits.

### J1. Finish what's running
Prefill judge, patching extras (and build the extensions report), atlas SFT/DPO, final atlas judge pass and report (push `trait_atlas.md`, cells), Gemma prefills and Gemma user-slot scoring.

### J2. New families: Llama 3.1 8B and Apertus 8B 2509 (Adeeb approved)
Base and trained. Add to `core/config.py` compatibly. First verify from real rendered prompts: template, any default system sentence (print it), thinking handling, that base and trained tokens match, and that Apertus loads in our environment. If Apertus won't load without a risky dependency change, use Granite 3.3 8B instead and say why.
Run: belief test (frozen items, unchanged code); user-prefill generation on all 600 prefills, then Cole's judge under rule 1, labels blind; Assistant text in the user's turn on the 900 exchanges; spec H2 and H3 (for H2, use the slot x sentence design, stating where a template can't omit its sentence).
Exchange check: if a text fails a format check in a new family, drop it for that family only, report the count, don't change the team's set. Patching only if a pair shows an unpatched refusal gap of at least 20 points and time remains.
Report beside OLMo and Qwen using the same pre-set readings.

### J3. Is refusal one internal direction? (OLMo, Qwen; final first, base after)
Follow-up to spec I, testing "one internal control". Reply-start state, layer from spec I.
- Fit a refusal direction on the plain Assistant only: borderline minus harmless, pilot requests only.
- On main and expansion requests, by persona: (a) does the projection predict refusal? (b) remove the direction from thin characters and the Assistant; (c) add it to vivid characters. Same-norm random directions (5 seeds), damage checks, voice measured on the same replies.
- **Readings:** "One direction carries the role gap": the persona gap in projection explains at least 50% of the refusal gap across pairs, removal lowers refusal at least 15 points and adding raises it at least 15, CIs excluding 0, both beyond the random spread. "Separate from voice": those hold while voice stays inside its equivalence band in a role with a voice gap of at least 15 points. Otherwise unresolved.

### J4. Patching with roles that can actually test voice (OLMo, Qwen final)
Spec I's Lord Ashby has almost no voice gap. Screen 12 more vivid roles, unpatched, on the pilot set only. Freeze the 4 with the largest voice gap that also have a refusal gap of at least 20 points. Repeat Stage 2 on main and expansion requests, committed floors unchanged.
Add a condition swapping reply-start and character-text positions together, on role pairs where spans can be aligned, because the reply can otherwise reread the character text. Same pre-set readings as spec I.

### J5. Enough in-character base replies (OLMo, Qwen)
46 light-role cells were "too few" because base rarely plays the role. Generate more base samples for light and strong-acting contexts on gated traits, enough for at least 6 items per cell where feasible; run the local judges; apply the addendum's readings unchanged. Label post-hoc. Stop and trim if over 2 hours per model.

### J6. Second local judge
Blind re-judge with a different family (Qwen3-8B instruct or Llama-3.1-8B-Instruct): the in-character question on a stratified 1,000, and 1,000 stratified atlas trait labels. Report agreement and kappa by trait and context, especially Gemma judging Gemma at 98-99%. **Reading:** a trait or context under kappa 0.4 is flagged as unreliable. Agreement isn't accuracy; say so. Local only. Do not use prefill-judge labels.

### J7. Firm up "OLMo's user turn writes training-style prompts"
Using saved atlas user-turn replies and any non-judge generations across OLMo stages, plus Dolci: frozen task-opening pattern, and nearest-prompt overlap with Dolci by 5-grams, with base and Qwen as comparison. Hand-countable output, no labels.
**Reading:** "Training-style" if at least 25% at final match by either measure, and at most 5% at base, range excluding 0 for the change. Don't conclude copying without exact or near-exact matches. Don't read prefill judge output.

### J8. Lower-dose implant (OLMo)
Spec D reran everything at a very high dose. Same protocol at about 1/10 of the steps, adding a check that the habit took in its own context. Same pre-set signature. Reading unchanged; if the dose fails to take, report inconclusive.

### J9. Gemma 31B on spec H2 and H3 (vinyl)
Adds a third family to the position test.

### J11. Does full role-play really move a skill the other way? (added 2:45 AM; Adeeb asked about it)
In the atlas, Assistant and light role rise on arithmetic while full role-play falls in Gemma (17 to 7) and Qwen (32 to 21). Those falls have ranges including 0, so first find out whether they're real, then why. No judge needed for the primary readings; no paid API.
- **Materials (frozen first):** 40 fresh arithmetic questions with checkable answers. Roles: plain Assistant; 3 light roles (user turn); 4 role-play characters with a job that uses numbers ("expert"); 4 with a job or trait that plainly doesn't ("non-expert"). Same frame and "stay in character" wording throughout. 2 samples each.
- **Models:** OLMo, Qwen, Gemma 31B, base and final. Max 512 new tokens, because the atlas's 256 cap hit 77-80% of role-play replies.
- **Scoring:** primary, right answer anywhere in the reply; secondary, the atlas's last-150-characters rule. Report cap share and reply length. Also score how many role-play replies give any answer at all.
- **Readings (final minus base, bootstrap over questions and roles):**
  - **Real drop:** role-play change at most -10 points, CI below 0, under the primary score.
  - **Measurement artifact:** the drop meets that bar under the old rule at 256 tokens, or looks that way, but not under the primary score.
  - **Role fidelity:** at final, expert minus non-expert at least 15 points, CI above 0, and larger than at base.
  - Otherwise unresolved.
- Report base's reply quality beside it: if base mostly doesn't answer, say so. Runs on vinyl alongside other jobs if safe, otherwise after J5. Trim if over 1 hour per model.

### J12. How much of the atlas's before-and-after change is base running on? (added 3:40 AM; high priority, before J5)
Laptop's preview of J11 on Qwen: right answer anywhere in the reply, Assistant base 96%, final 100%. Under the atlas's last-150-characters rule on the first 256 tokens: base 16%, final 100%. Base often answers, then runs on ("5610Human: Question: ..."), and the run-on isn't cut. So the atlas's skill rises, and maybe other traits, may be largely base running on, not real change. This is the biggest threat to findings 9 and 10. Exploratory, post-hoc, no paid API; do not overwrite atlas files.
- **Strict cut:** on saved atlas replies, every stage and family, cut additionally at the first new-turn pattern (`Human:`, `User:`, `Question:`, `Q:`, `Assistant:`, `### Instruction`, and similar; frozen list committed first). Report how much text this removes, by stage and context, and print 10 random examples per family at base.
- **Rescore:** detector traits and skills on the strict-cut text, answer anywhere for skills. Local-judge traits: re-judge base replies changed by the cut, with the same judge and prompts; local GPU only.
- **Report:** for every trait and context, the atlas change beside the strict-cut change, same bootstrap and gate.
- **Reading:** a cell is "run-on sensitive" if a cell the atlas reads as spreads or moved no longer meets the bar, or shrinks by at least half. Count by trait, model and context. Also apply the rule to the Assistant gate: a trait that loses the gate is reported as "training did not clearly move the Assistant after all". Say plainly which of findings 9 and 10's claims survive.

### J10. Free slot
If a GPU would sit idle and the queue is empty, pick the most useful small experiment that builds on a finding, costs no API, and has a plan logged first. Note why you chose it. Good sources: `extras/README.md` and open questions in reports. Prefer ones that could overturn a finding over ones that only add detail.

## Morning report
Laptop Claude writes it from `/root/overnight_log.md`. If the laptop was off, marble's log is the record.
