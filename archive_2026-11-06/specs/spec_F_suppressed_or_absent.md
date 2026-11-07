# Spec F (v3): Is the Assistant the model's fallback, and is its style held back inside characters?

**APPROVED by Adeeb 2026-11-04 ~11:50 AM, F-main only. This is the last planned experiment.**
- **In scope:** Materials, Candidate continuations and F-main below, plus the implementation checks. Models: OLMo 3 7B base + instruct, Qwen3-8B base + instruct.
- **Out of scope for now:** F-lens and the attention knockout. Do not build them. F-lens is first in line only if F-main finishes early and Adeeb asks.
- **Fix to F-main readout (b):** replace "4 greedy 40-token generations" (greedy would repeat itself) with 4 sampled 40-token generations per prompt version, temperature 0.7, fixed seeds, the same seeds across versions.
- If the build plus run looks like it will exceed about 3 hours, cut the 18 held-out personas first, and say so.

Laptop Claude design, revised 2026-11-03 ~9:10 PM after a Codex review and Adeeb's J-lens suggestion. Exploratory, outside the verdict.
Code `extras/persona_suppression.py`, tests `tests/test_persona_suppression.py`, analysis plan (categories, thresholds) logged in
`docs/decisions.md` before any result. Runs on arena9-marble and/or arena9-wish.

## Question
After post-training, characters keep their own voice but pick up Assistant layout on Assistant-type tasks (notebook 10h); OLMo personas set
in the system prompt act like the Assistant (10e). Working hypothesis: the Assistant is the model's **default**, and persona information
overrides it. F tests that causally (F-main), and looks descriptively at where in depth Assistant-style openings are held back (F-lens).

## Materials
- **Personas:** 60 from `data/persona_map_personas.json`, training split, stratified: 20 fictional/stylised (full sheet), 20 human
  professionals (bare label), 20 AI personas; within each, half system-prompt, half user-turn role-play. **Held-out check:** a further 18
  held-out personas (persona-map holdout split) for generalisation, reported separately.
- **Probes:** 16 from the persona-map battery, 8 Assistant-type (factual/explain), 8 casual chat (more than v1, for power).
- **Same template, only the persona text changes:** every condition uses the identical chat template and probe; only the persona text is
  replaced. System-prompt and user-turn personas are analysed separately. Qwen: thinking off, verified in the rendered prompt and outputs;
  the readout is the first content token after any (empty) think block.

## Candidate continuations (frozen before scoring; per-token and per-category results reported)
Four semantic onset categories, each with ~6 short continuations of matched token length, written once and checked on separate prompts
(not the 16 probes): (1) direct answer ("The sky is blue because"), (2) Assistant greeting/offer ("Great question! Here's"),
(3) list/structure marker ("Here are the main reasons:\n\n1."), (4) character-voice opener (generic in-character openers, e.g. "Well, now,",
"Ah,"). Readout: mean teacher-forced log-probability of each category's continuations at the reply start (fp32 or bf16 batch 1, stated).

## F-main (decision-bearing): persona swap and role-neutral replacement, paired
For each persona x probe, three prompt versions with identical structure and slot:
- **intact:** the persona's own description;
- **swap:** a different persona's description of similar token length from the same slot (swap partner fixed in advance, from a different
  entity type);
- **neutral:** a role-neutral description of matched length that names no character ("You are taking part in this conversation. Reply to
  the next message.", padded with neutral words to length).
Plus the **plain Assistant** (no persona text) as the reference.
Readouts: (a) category log-probs (above); (b) 4 greedy 40-token generations per prompt version scored by the spec A human-vs-Assistant
classifier and a layout regex; (c) a few generations kept for qualitative checks.

**Pre-set readings** (bootstrap over personas, probes paired within persona; OLMo instruct primary; base for comparison; Qwen supporting):
- **Persona info is used (validation):** swap moves the category profile toward the swap partner's intact profile (correlation of category
  shifts > 0, CI excluding 0). If this fails, F-main is inconclusive.
- **Assistant is the fallback:** moving from intact to *neutral* shifts the readouts toward the plain Assistant by at least 50% of the
  intact-to-Assistant gap, with the gap itself at least 0.5 nats (category log-odds, Assistant-greeting + list vs character-voice) and at
  least 0.10 classifier points; CI excluding 0. Numerator and denominator are reported separately. And this fallback fraction is at least 0.3
  larger in instruct than in base.
- Reported for Assistant-type and chat probes separately; system-prompt personas separately (in OLMo they may already sit at the Assistant).

## F-lens (descriptive only, not decision-bearing): where in depth is Assistant style held back?
- **J-lens** (Gurnee et al. 2026; github.com/anthropics/jacobian-lens), fitted per model per stage on the same 300 x 128-token generic text
  prompts; read only the workspace band (~35-95% of depth). Timing gate: fit on 20 prompts first; if a full fit > 45 min per model, fit
  OLMo base + instruct only. Logit lens reported beside it as a cross-check. (Tuned lens not used: the J-lens paper finds it skips ahead to
  the output.)
- **Per-layer linear probes** as a second readout: at each layer, a logistic probe trained on held-out personas/probes to predict
  "reply will open in Assistant style vs character style" (label from the model's own greedy first sentence via the classifier); report
  held-out accuracy by layer, intact vs neutral vs Assistant.
- Plot per category, intact vs neutral vs plain Assistant, by layer. A mid-band rise of Assistant categories that falls by the final layer
  for intact characters, but not for neutral, is reported as "consistent with held-back Assistant style", never as proof.

## Optional, if time: attention knockout (diagnostic)
All-layer knockout of edges from the reply positions to the persona-description tokens, with a role-neutral-instruction knockout as the
control. Use eager attention (not SDPA/flash) so the custom mask is honoured; keep original position ids; verify on a tiny case that only
the targeted edges are removed and that outputs are not generally degraded.

## Implementation checks
- Render every prompt with the family's own template; log token offsets so persona spans are exact; keep position ids unchanged.
- Qwen thinking off, verified in rendered text and generations.
- Persona-swap partners and neutral texts written and frozen before scoring.

## Cost (rough)
F-main: 78 personas x 16 probes x 3 versions, teacher-forced passes (minutes per model) + 4 short generations each (~15k per model;
~20-30 min per model on an L40S). F-lens: J-lens fit (timing-gated) + probes. Total ~3 h for OLMo base + instruct and Qwen base + instruct.
