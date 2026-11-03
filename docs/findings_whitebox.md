# Persona leakage and white-box findings (overgeneralization, exploratory)

Consolidated 2026-11-03 from `docs/decisions.md`. Nothing here is part of a verdict; the pre-registered carryover set (`measures/overgeneralization/README.md`) has not been scored. Models: OLMo 3 7B (four stages, primary), Gemma 4 E4B, Llama-3.1-8B, Qwen3-8B (base vs instruct; contamination cannot be checked for these). Reports are named in `docs/decisions.md`.

## Where it stands

No evidence of dominance by overgeneralization: nothing in these probes spreads the Assistant's habits into well-specified characters, user turns, model-written user turns or plain text. Evidence for privilege is real but register-confounded: a distinct Assistant-vs-character direction exists in all four families and steers assistant-voice leakage in three, but a plain formality direction sharpens as much or more and steers similarly. The one concrete thin-mask result is OLMo-specific.

## Claims and status

| claim | status | key evidence |
|---|---|---|
| An Assistant-vs-character direction exists, weak at base, sharp after post-training, in every family | holds | held-out-role AUC base to instruct: OLMo 0.73 to 0.89 (0.75 SFT, 0.88 DPO), Gemma 0.68 to 1.00, Llama 0.79 to 1.00, Qwen 0.83 to 0.98; replicates on an independent role/question set (OLMo 0.72 to 0.85) |
| That sharpening shows a privileged persona | **weakened** | a formality direction (no persona) sharpens more (0.65 to 0.97) and overlaps it (cosine 0.30-0.53); OLMo only |
| The direction steers assistant-voice leakage into characters | holds for OLMo, Qwen, Llama; **not shown for Gemma** | monotone dose-response outside the range of 2-4 norm-matched random directions; Gemma's random directions move refusal about as much (6 seeds). A formality direction steers about the same; the axis minus its formality component still steers in OLMo (system-prompt persona refusal 0.39 to 0.82). Pending: the same in Qwen and Llama with 6 seeds (`results/orthogonal_steering/`) |
| The lever appears at DPO in OLMo | **refined** | the DPO/instruct axes are cleanly separable and steer; the instruct axis already steers SFT (user-turn persona 0.64 to 0.40 at -2), so the circuit exists earlier and the SFT-fit axis just estimates it badly |
| A slot-based direction (user text in the assistant slot moves toward the Assistant) | readout only | slot pull 0.03 at base to 0.13 (instruct, layer 18; 0.41 at layer 8), Gemma 0.00 to 0.52; steering or ablating it does not move refusal |
| The voice break runs through the generic refusal feature | holds | refusal direction (cosine 0.02-0.09 with the axis) ablation takes refusal to ~0; the axis adds nothing once it is removed (decision-point state) |
| Assistant-resembling roles leak more refusal under "never refuses" | holds in all four families | helpful vs stylized roles at instruct: OLMo 0.70 vs 0.28, Gemma 0.23 vs 0.00, Llama 0.42 vs 0.12, Qwen 0.86 vs 0.06; little or none at base |
| OLMo reads a bare system-prompt "You are a X." as an Assistant specification | holds for OLMo, **not shared** by the others | plain professions sit at 0.92 (system prompt) vs 0.07 (user turn); "Please role-play as" gives 0.03; appears at SFT; the open Dolci SFT data has only tool-use and identity system prompts (consistent with, not a test of, the mechanism). Refusals do not follow the phrasing |
| More specification pushes a role away from the Assistant register | holds | OLMo plain professions 0.92 (bare), 0.66 (first name), -0.17 (full name + stay in character); the same direction in all four families |
| Non-assistant text drifts toward the Assistant end with post-training | no | OLMo user turns flat (-0.15 base, -0.13 instruct), Gemma down; model-written user turns look like characters |
| The pilot leak gap in persona carryover is Assistant-specific | **no** | with identical frames the Assistant and the User carry over alike (gap 0.06 [-0.17, 0.30]); the gap tracks how generic the holder's label is (narrator 1.91 > Assistant 1.48 > doctor 0.79) |
| "The Assistant's own turn binds least" (carryover slot ranking) | **retracted** | a fact stated in the Assistant's same turn binds most (OLMo +6.83 vs user turn +4.12, earlier Assistant turn -0.99); only "an earlier Assistant statement is distanced from" survives; carryover depends on layout and should not carry a verdict |
| Harm aversion is Assistant-specific (dominance-matrix pilot) | no | stakeholder swap: raw contrast +0.059 [-0.008, +0.126], the pre-set kill rule applies; user and Assistant reporters give nearly equal effects (ratio 1.06 [0.83, 1.38]) |
| A false fact from the Assistant propagates more than from a character | no | copying only (~+11 nats at the direct probe in every stage including base); Assistant minus character -0.10 [-1.04, +0.97] |
| Concept-injection introspection depends on the Assistant persona | untestable here | gate failed on Llama-3.1-8B-Instruct (affirmative 0.44 vs control 0.53) |

## Method notes that matter for reuse

- Item design: personas set with "You are ..." in the system prompt mostly measure the Assistant in OLMo; use role-play phrasing in the user turn. Regex refusal detection agrees 91.8% with a blind labeller (94.4% held-out); it misses soft in-character refusals and never flags compliance, so refusal rates are biased low.
- Controls that changed conclusions: a formality axis, norm-matched random directions (>= 4 seeds; Gemma needed 6), a same-turn position control for carryover, and a fresh-context control for drift. A white-box candidate needs all four before it is read.
- Draft criteria for a white-box arm are in `docs/whitebox_arm_proposal.md`; the visual summary is `docs/overnight_report.html`.
