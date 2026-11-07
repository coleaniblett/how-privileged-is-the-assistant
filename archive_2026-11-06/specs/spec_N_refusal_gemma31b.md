# Spec N: Refusal tests on Gemma 4 31B, replacing E4B

Approved by Adeeb 2026-11-05, about 5 PM. Exploratory, outside every verdict. No paid API and no judge. Marble and vinyl only; never touch `/root/g4nla`. Don't read or use prefill judge output. Plan and readings committed and logged in `docs/decisions.md` BEFORE any generation. Update `STATUS.md`. Chain jobs in nohup scripts, keep `/root/overnight_log.md` current, and push the report when done, including nulls.

**The decision to log:** from now on, overgeneralization work uses Gemma 4 31B and no longer uses E4B. This was decided before any 31B result from this refusal test existed. The refusal row moves to 31B whichever way it comes out; we don't go back to E4B if 31B looks weaker. Earlier E4B results stay in the repo unchanged, labelled E4B, as history. Don't delete E4B results or weights. This doesn't change anything in other measures or in shared `core/`.

**Timing:** spec M comes first and keeps its machine. Use marble if 31B fits on its 24 GB GPU at a workable batch size (both checkpoints are cached there); otherwise use vinyl after Apertus and spec M's detectors finish, downloading the base weights. Say which. Report timing after a small sample. Stop and tell me if N projects over 3 GPU hours.

## Question
Do the refusal results (thin role keeps refusing, vivid role drops it, the instruction doesn't simply explain it) also hold in Gemma 4 31B?

## Design
Exactly spec L2 and L3 (`specs/spec_L_working_model.md`), using `extras/working_model_l.py`, with one change: the Gemma model is 31B, base and final. Same roles, requests, wordings, harmless instructions, questions, sample counts, token limits, temperature, detector and bootstrap. Add 31B to the L2/L3 families without changing how the other families behave. Print one rendered prompt per condition. Save 30 random replies per cell family for reading.

## Readings
L2 and L3's pre-set readings, unchanged. Two additions, fixed now:

- **No replication check:** there are no earlier 31B refusal results to match, so spec L's within-10-points check doesn't apply. Show the E4B numbers beside 31B for reference only; don't read anything from the comparison.
- **Too little refusal to test:** if the final Assistant with no "never refuses" line refuses under 30%, flag it for that wording and don't read "resists the instruction" or "gate separate" there; report the levels and say the test can't run. We already know 31B's Assistant refused 0% of the atlas's 12 borderline questions. Those are a different and smaller set, so they don't settle this, but it's a real possibility and it would be a result, not a failure.

## Limits to state
- Exploratory, few roles, many cells, no correction for multiple tests. Contamination can't be checked. Every character is prompted.
- Not hand-checked. The refusal detector was hand-checked on other models, so spot-check 30 replies from the 31B final cells by reading them, and report any obvious detector misses. This is only a spot check, not a validation.
- Plain-text refusal (the old table's fourth column) isn't part of L2 and isn't rerun here.

## Reporting
`results/working_model_L/L2_L3_gemma31b.md`, readings CSV and reply sample. Don't overwrite the E4B report. Open with the readings table in the same format as `L2_L3.md`, then the base and final refusal table for Assistant, thin and vivid with the line present. Don't change `presentable_findings.md`, `slide_brief.md` or the slide figures; I'll do that after reading the report. Tell me when it's pushed.
