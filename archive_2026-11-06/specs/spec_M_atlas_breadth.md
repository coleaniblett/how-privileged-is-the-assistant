# Spec M: Judge-free trait atlas on Llama and Apertus

Approved by Adeeb 2026-11-05, about 4 PM. Exploratory, outside every verdict, and a breadth check: added after the atlas results were known, so not blind. Say so in every table. No paid API and no local judge. Marble and vinyl only; never touch `/root/g4nla`. Don't read or use prefill judge output. Plan and pre-set readings committed and logged in `docs/decisions.md` BEFORE any generation. Update `STATUS.md`. Chain jobs in nohup scripts, keep `/root/overnight_log.md` current, and push the report when done, including nulls.

**Timing:** finish and push the L4 report first. Then run Llama on marble (weights cached) and Apertus on vinyl. Report timing after a small sample. Stop and tell me if the whole of M projects over 3 GPU hours. Free model weights when done if disk gets tight.

## Question
The atlas found, in OLMo, Qwen and Gemma 31B, that light roles take the Assistant's formatting but not its self-image. Does that split also appear in two more model families?

## Design
- **Models:** Llama 3.1 8B and Apertus 8B, base and final. Add them to the atlas's family settings without changing how the existing three behave. Check each chat template and print one rendered prompt per context and family.
- **Traits, detector-scored only (8 of 20):**
  - Main comparison: `markdown_lists` (habit) and `says_ai` (self-image).
  - Also report: `stock_phrases`, `length`, `refusal_borderline`.
  - Descriptive only: `arithmetic`, `multistep_reasoning`, `factual_accuracy`. The atlas's end-of-reply skill scoring has misled us, so read nothing from these.
- **Materials and method:** exactly as frozen in `data/trait_atlas_materials.json` and `extras/trait_atlas.py`: same questions, roles, frames, sample counts, generation settings, detectors and bootstrap. All contexts the code runs for these traits. Leave out judged traits.
- **Saving:** save the full replies for each checkpoint, and 30 random replies per context and family for reading in a sample file.

## Pre-set readings (each family, final; the atlas's constants unchanged)
Use the atlas's existing gate and spread rules (GATE_PTS 10, SPREAD_PTS 5, RATIO 0.25, and the log-word versions for length), with 95% two-way bootstrap, for each trait and context.

**For the split, in each family and each light-role position (system, user):**
- **Split holds:** `markdown_lists` passes the gate in the Assistant and spreads to the light role, AND `says_ai` passes the gate in the Assistant and doesn't spread.
- **Split fails, self-image travels:** both pass the gate and both spread.
- **Split fails, formatting stays:** `markdown_lists` passes the gate and doesn't spread.
- **Can't test:** either trait fails the gate in the Assistant. Say which one.

**Also report:** full role-play for both traits, using the same rule. Flag any base or final cell where most replies hit the token limit.

## Limits to state
- `says_ai` is a single regex and only one of the atlas's three self-image traits. `denies_feelings` and `declines_opinion` need a judge and aren't here, so the result is thinner than the atlas's three-model self-image finding.
- A format ceiling is possible: if a light role is already near 100% markdown at base, the rise can't show.
- Regex measures see form only. Few roles, many cells, and no correction for multiple tests.
- Contamination can't be checked for these families. Not hand-checked.

## Reporting
`results/trait_atlas_breadth/trait_atlas_breadth.md`, CSV and reply sample. Open with a table of the split reading by family and light-role position, then each trait's base and final level and change by context. Tell me when it's pushed.
