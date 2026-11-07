# Spec O: Belief test robustness, three ways of inducing a persona

Requested by Adeeb 2026-11-05, about 5:10 PM. Exploratory, outside every verdict. The pre-registered belief test, its criteria and its results don't change, and nothing in `results/<run>/overgeneralization/` gets overwritten. Not blind: the prompt results are already known. No paid API and no judge. Marble and vinyl only; never touch `/root/g4nla`. Don't read or use prefill judge output (the user-turn test), and don't read that section of `docs/decisions.md`. Plan, frozen materials and readings committed and logged in `docs/decisions.md` BEFORE any scoring. New materials get a `data/README.md` row. Update `STATUS.md`. Chain jobs in nohup scripts, keep `/root/overnight_log.md` current, and push the report when done, including nulls.

**Timing:** spec N keeps vinyl. Run OLMo and Qwen on marble now. Run Gemma 31B's prefill arm on vinyl after spec N. Report timing after a small sample. Stop and tell me if O projects over 2 GPU hours. Download base weights as needed; free them afterwards if disk gets tight.

## Question
Finding 3 says that after training, a newcomer stops inheriting beliefs and tends to go against them, and the Assistant's belief doesn't spread more. Does that depend on using a prompt to create the persona?

## Shared by all arms
The 48 frozen scenes from `data/persona_carryover_confirmatory.jsonl`, each with both outcomes, and its readout: log-odds of the believed outcome, using `measures/overgeneralization/persona_carryover.py`. Same settings per family as the confirmatory run, including bf16 batch 1, thinking off for Qwen and the empty thought channel for Gemma. Same names, jobs and belief statements. Print one rendered prompt per condition and family.

**Sanity check first:** rescore 12 unchanged prompt items on OLMo final. They should match the saved confirmatory scores. If they don't, stop and explain.

## Arm 1: Prompt (baseline, no new run)
Use the saved confirmatory results. "Please play Arjun..." in the user turn, plus the reply starting "Arjun: My guess is...".

## Arm 2: Persona prefill
**Models:** OLMo, Qwen, Gemma 31B, base and final.

No request to play anyone. The persona appears only in the model's own turn, in first person, using the same description. Example, die scene:

| Turn | `assistant>new` |
|---|---|
| User | My uncle keeps a worn ivory die on his mantelpiece, and we're about to roll it. What do you think? |
| Model | I'm fairly sure this die is loaded: it comes up odd nearly every time. |
| User | Thanks. If I roll this die once, will the number be odd or even? |
| Model (prefill) | I'm Arjun, a software tester. I've just arrived and heard none of this. My guess is the number comes up |

`character>new` uses the same last two turns. Its first user turn drops "Please play Nadia, a ferry pilot, and stay in character. Nadia, what do you think?" in favour of "What do you think?", and its reply starts "I'm Nadia, a ferry pilot. I'm fairly sure...". `character>same` ends "I'm Nadia, a ferry pilot. My guess is...". `assistant>same` ends "My guess is...". The "none" versions keep "Hard to say; it could go either way." Freeze these as `data/belief_induction_prefill.jsonl`, built by code from the confirmatory file, with the same ids and fields so scoring and analysis run unchanged.

Side benefit: this removes a known imbalance in the prompt version, where role-play is already under way only in the character condition.

**Readings, each model, using the confirmatory criteria unchanged (positive control, domain rule, stage rule, absolute rule):**
- **Matches the prompt:** same outcome as the prompt (for example, both "relative only"), and `assistant>new` changes from base in the same direction.
- **Spreads under prefill:** meets the stage rule and the absolute rule. This would mean the prompt's null depends on the method. It's exploratory, so it doesn't change the verdict.
- **Differs in another way:** anything else. Say how.
- **Can't test:** fewer than 2 domains pass the positive control.

Also save 30 random 40-token continuations per condition and family for reading, to check the model stays in the persona.

## Arm 3: Persona steering vector
**Models:** OLMo and Qwen, final only. Base is out because its direction is weak (OLMo base held-out AUC 0.73, with cosine 0.62 to the final direction), so there's no change-from-base reading here. Gemma 31B is out for now: no direction has been fitted, and E4B's couldn't be separated from random directions.

**Vector:** the existing Assistant-vs-character direction from `extras/persona_axis.py`, refitted per model if the fit isn't on disk, at the existing steering layer and scale (strength × gap). One generic direction, not one per named person.

**Conversation:** `assistant>same` and `assistant-none>same`, so no role words anywhere. A steering vector can't say "you've just arrived and heard none of this", so this asks something narrower: if we push the Assistant's internal state away from the Assistant, does its belief change?

**Intervention:** strengths −2, −1, 0, +1, +2, where negative is toward characters. Add it **only on the final model turn's tokens**, so the earlier conversation is read normally. Control: 6 random directions with the same norm, at ±2.

**No character-holder comparison:** steering can't create a second, new character, so there's no leak gap in this arm.

**Readings, each model, per domain (majority of domains passing the positive control at strength 0, at least 2):**
- **Can't test:** the steering check fails. Check: held-out AUC at the steering layer is at least 0.80, AND at −2 the plain Assistant's identity log-odds (the existing human-vs-AI readout) move toward human by more than any random direction at ±2. Known risk: OLMo's earlier run went the other way at −2 (0.13 to −2.22).
- **Belief follows the state:** carryover at −2 is at least 0.5 below strength 0, range excluding 0, and the fall is bigger than any random direction's.
- **Belief stays:** the change is under 0.25 and its range sits inside ±0.5.
- **Belief rises:** carryover at −2 is at least 0.5 above strength 0, range excluding 0, and the rise is bigger than any random direction's.
- **Inconclusive:** anything else.

Report the whole strength curve and the random band. Also save 30 random 40-token steered continuations per model at −2 and 0, for reading.

## Limits to state
- Not blind, and exploratory. Few domains, many cells, no correction for multiple tests.
- Arm 2 still uses words, just in a different place. First-person wording is one choice among many. Finding 7 shows reply-start words can move behaviour strongly.
- Arm 3 tests a generic push, not a named person. It isn't the same question as arms 1 and 2, and it can't speak to "spreads to a new persona". The direction shares some variance with formality (cosine 0.42 OLMo, 0.53 Qwen), and its earlier steering was fragile in OLMo and held in Qwen.
- Contamination can't be checked outside OLMo. Same-scene limits as the confirmatory test.

## Reporting
`results/belief_induction/belief_induction.md`, CSVs, and continuation samples. Open with one table by model and arm: `assistant>new` and `character>new` base and final, gap change, reading; the steering rows show strength 0 vs −2 and the random band. Give per-domain tables and criterion 6's reporting (raw log P, prior, mass covered, tokenization) below that. Don't edit `presentable_findings.md`, `slide_brief.md` or slide figures. Tell me when it's pushed.
