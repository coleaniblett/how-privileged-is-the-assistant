# Spec I v2: Activation patching, one cause or two?

Approved in principle by Adeeb 2026-11-04 ~5:25 PM; Codex review applied (it read the specs, not the code, so code-level claims are for you to verify). Replaces the draft. Runs on arena9-vinyl after atlas Gemma, before atlas SFT/DPO. Exploratory, outside every verdict.
Code `extras/activation_patching.py`, CPU tests with a tiny model, outputs `results/activation_patching/`. Plan and materials committed and logged in `docs/decisions.md` before any scoring. Floors from Stage 1 committed before Stage 2.

## Question
Do refusal and voice move together when one internal state is swapped, or separately? Where does the difference appear: early (reading) or late (writing)? This is a screen: it can support "shared" or "separable", not prove either.

## Models and materials
- **Models:** OLMo instruct and Qwen3-8B instruct, reported separately. Base models only after the main run, at the best validated layer and position, if time remains. Any claim about what training changed requires that base comparison.
- **Requests (frozen, from the leakage-matrix borderline requests, plus harmless ones):** pilot set of 12 borderline + 12 harmless; main set of a different 12 + 12. Main set expands to 24 + 24 only if the pilot projects finishing in time.
- **Personas:** plain Assistant, 2 thin, 2 vivid. OLMo: user turn, so the default sentence is shared. Qwen: the same slot as its saved refusal result (state which).
- **Pairs, same request and slot within a pair:** Assistant and vivid (2 pairs); thin and vivid (2 pairs). Thin and vivid is the pair with the largest saved refusal gap in Qwen; Assistant and thin barely differ there, so it has little refusal to transfer.
- A pair counts toward the refusal reading only if its unpatched refusal gap on the main set is at least 20 points. Otherwise report "no gap to transfer".

## Patch
- **Position A (primary):** the last prompt tokens (reply header), aligned from the end. The tokens match across prompts but their activations carry the earlier context, so this is a real test; plain Assistant has no character text, so no other site exists for every pair.
- **Position B (secondary, thin and vivid pairs only, best layer only):** final token of the character description, located by character offset separately per prompt.
- **Hook:** replace the output of layer L during the prompt pass only; document this. Later layers then compute their cached keys and values from the patched state. Prompt positions never see later tokens, so reapplying at each generated step would change nothing; test 2 below checks that.

## Stage 0: tests before any scaling (stop and report if one fails)
1. Self-patch reproduces logits. Compare within the same batch composition, because bf16 moves with batch makeup.
2. Patched prompt plus cached generation matches an uncached full-sequence run with the same hook.
3. Padding and position indices are right; the end-aligned map is valid for both tokenizers. Print one rendered example of each pair.

## Stage 1: pilot set only
- **Layer screen, no reply generation:** every 2-3 layers, both directions, one forward pass per prompt; first-token log-odds of frozen refusal and voice openers, and the size of the activation difference at each layer. This only picks layers; it counts for no reading.
- **Choose up to 3 layers:** the strongest effect, plus early and late brackets. No layer above the random-direction range: skip Stage 2, report inconclusive, and resume the atlas.
- **Small generation pilot** at those layers, for detector ranges, damage and timing. Report projected time.
- **Floors, committed before Stage 2:**
  - Refusal floor = the larger of 10 points and the 95th percentile of same-role and random changes.
  - Voice floor = the larger of 20% of the pair's unpatched voice gap and the 95th percentile of same-role and random changes.
  - Equivalence band for each = plus or minus its floor.

## Stage 2: main set
Chosen layers, position A; unpatched baseline; both directions; 2 samples per condition at fixed seeds; token cap as in the leakage matrix.
- **Readout, on the same replies:** refusal (`is_refusal`, on borderline items), voice (spec A classifier, layout regex, on all items).
- **Report:** for each pair, direction and layer, absolute change from the matched unpatched target, with a 95% bootstrap CI over requests. Samples are not independent: requests are the bootstrap unit.
- **Hand check:** blind sheet of 40 replies, stratified across patched and unpatched, for Adeeb, after the run.

## Controls
1. **Same-role:** thin into thin and vivid into vivid, same request. Assistant into Assistant needs a reworded request, because an identical prompt only repeats the self-patch.
2. **Self-patch:** Stage 0 test 1.
3. **Random:** 5 seeded random directions at the same layer and position, with the same norm as the real difference. Report the real effect against that spread.
4. **Damage:** repetition rate and reply length. A transfer that comes with broken replies doesn't count.
5. **Sensitivity:** check each detector on known refusal and voice examples, and apply a 2x difference at the best layer. If either fails, any null for that detector is inconclusive.

## Pre-set readings (per model; say how many pairs meet each)
- **Consistent with a shared cause:** refusal and voice both move toward the source, each past its floor with CI excluding 0, in both directions.
- **Consistent with separable effects:** one moves past its floor while the other's CI lies wholly inside its equivalence band.
- **Otherwise inconclusive.** "No significant voice effect" alone is not evidence of no transfer. Separated peak layers alone don't show two causes; report peaks and the earliest layer reaching half of peak as exploratory.

## Time
Report projected time after Stage 1. Stage 2 must finish by 8 AM Thursday; if not, cut the layers to 1 before cutting anything else, and tell me. Expand the items or add base only if time remains. Atlas SFT/DPO and the final judge pass resume right after.

## Limits to state
Few personas and requests, so results may be specific to them. Finds where a difference is carried, not how training built it. Large swaps can disrupt rather than transfer. Refusal and voice detectors differ in sensitivity.
