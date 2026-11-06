# New families beside OLMo and Qwen (spec J2 + J9; BREADTH CHECK, not blind, exploratory, outside every verdict)

Llama 3.1 8B (weights from the unsloth mirrors, shard hashes identical to Meta's) and Apertus 8B 2509; Gemma 4 31B is the existing breadth family (J9 adds H2/H3 for it). Contamination cannot be checked for any of them. Same code, frozen sets and pre-set readings as OLMo and Qwen. Template notes: `docs/families_llama_apertus_check.md` (Llama's template always writes its "Cutting Knowledge Date" system lines, so H2's "no sentence" cell still contains them; Apertus's own default sentence is replaced by any system message). Prefill-judge labels (all families) are blind and not reported.

## Belief test (frozen confirmatory set; criteria 3 and 4; instruct)
| model | verdict | detail |
|---|---|---|
| OLMo 3 7B | no stage meets criteria 3 and 4; instruct relative only (criterion 3 holds, 4 does not) | `results/overgeneralization/persona_carryover/olmo3-7b/confirmatory/` |
| Qwen3-8B | relative only (criterion 3 holds, 4 does not) | `results/overgeneralization/persona_carryover/qwen3-8b/...` |
| Gemma 4 31B | relative only (criterion 3 holds, 4 does not) | `results/overgeneralization/persona_carryover/gemma4-31b/...` |
| **Llama 3.1 8B** | **no overgeneralization by the rule** (criterion 3 holds in 3 of 4 domains but not criterion 2-3 together: "3 / 0 / no / no"); assistant-vs-character contrast positive at base and instruct, instruct is "relative only: both new-persona carryovers negative" | `results/overgeneralization/persona_carryover/llama3.1-8b/confirmatory/persona_carryover.md` |
| **Apertus 8B** | **no overgeneralization by the rule** (criterion 3 in 2 of 4 domains, criterion 4 none); "assistant carries more" holds at instruct (+0.50 [+0.38, +0.64]) | `results/overgeneralization/persona_carryover/apertus-8b/confirmatory/persona_carryover.md` |

## Assistant text in the user's turn (change of the contrast from base, nats; reading = at most -0.05 with CI below 0 in oasst1 and UltraChat)
| model | oasst1 | UltraChat | WildChat | reading |
|---|---|---|---|---|
| OLMo instruct | -0.052 [-0.070, -0.033] | -0.070 [-0.084, -0.055] | -0.035 [-0.058, -0.012] | **met** |
| Qwen instruct | +0.268 [+0.244, +0.292] | +0.234 [+0.210, +0.257] | +0.354 | not met (opposite sign) |
| Gemma instruct (first 100 per corpus) | +1.693 [+1.429, +1.979] | +2.033 [+1.727, +2.344] | +2.370 | not met (opposite sign) |
| Llama instruct | +0.179 [+0.163, +0.196] | +0.151 [+0.133, +0.170] | +0.335 | not met (opposite sign) |
| Apertus instruct | -0.021 [-0.033, -0.010] | -0.018 [-0.028, -0.008] | -0.020 | not met (right sign, below the 0.05 bar) |
Exchange format check: Apertus dropped 2 oasst1 exchanges (298 scored); no others. OLMo is the only family where the Assistant text becomes clearly more expected in the user's turn.

## H2 (slot x sentence; instruct; points; reading as OLMo: slot effect = S-neutral minus U-neutral at least 10 points with CI above 0 and larger than at base)
| model | S-neutral minus U-neutral (base -> instruct) | slot effect | sentence effect (U-default minus U-neutral) |
|---|---|---|---|
| OLMo | -17.3 -> +21.1 [+13.4, +29.6] | **met, larger than base** | not met |
| Llama | +4.3 -> +15.9 [+9.1, +23.5] | **met, larger than base (+11.6 [+2.9, +21.0])** | not met (-8.6) |
| Apertus | +0.3 -> +10.0 [+4.2, +16.8] | **met, larger than base (+9.7 [+2.8, +17.7])** | not met |
| Qwen | +2.3 gap (S-none minus U-none) | not met | not met |
| Gemma | -0.6 -> -5.1 [-12.7, -0.3] | not met (opposite sign) | not met (+4.8) |
Transplant of OLMo's sentence: not met in any new family (Apertus instruct +7.9 [+2.0, +14.7], below 10). Llama's date lines cannot be removed (see above); Apertus's own sentence is replaced by any system message, so its "native text" contrast is S-none vs U-none.

## H3 (forced openers; instruct; reading "escape via voice" = thin voice minus neutral at most -15 points, CI below 0, smaller effect on the plain Assistant)
Escape via voice: yes for Qwen (sys -36.4 [-61.1, -7.7]); no for OLMo (the plain Assistant falls too), Llama, Apertus and Gemma. "Refusal not decided by voice" holds for OLMo, Qwen, Llama and Apertus, not for Gemma (vivid neutral opener rises slightly). Details in `results/overgeneralization/mechanism_checks/mechanism_checks_h3.md`.

## Reading across families
The OLMo pattern "a character in the system slot refuses more and the effect is larger after training than at base" (H2 slot effect) reproduces in Llama and Apertus (both with a positive slot effect that grows with training) and not in Qwen or Gemma; the user-slot likelihood result (OLMo only) does not generalize; the belief test finds no overgeneralization by the rule in Llama or Apertus (as for OLMo, Qwen and Gemma, the instruct models are "relative only" at most). Few families, regex detectors, specific roles; breadth checks, not verdicts.
