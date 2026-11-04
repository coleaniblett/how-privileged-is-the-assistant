# Logit distortion on the exchange design: report

Grace's measure. Analysis of saved scores only (no model loaded, nothing re-scored). Generated tables: [`results.md`](results.md);
every number: [`summary.csv`](summary.csv), [`verdict.csv`](verdict.csv), [`positions.csv`](positions.csv), [`samples.csv`](samples.csv);
validation: [`checks.json`](checks.json); command, code revision, input hashes and settings: [`provenance.json`](provenance.json).
Spec: `measures/logit_distortion/README.md`, "Exchange-design analysis (2026-11-04)", and `docs/decisions.md`, 2026-11-04.

**Status of these results.** The gap definition, the change-from-base comparison and criterion 1 (≥ 0.05 nats, CI excluding zero,
both authorship columns) were fixed on 2026-11-01/02. How they map onto the exchange design (oasst1 = human column, UltraChat =
model column, the reply after its question = assistant persona, joint pairing) was decided on 2026-11-04, **after** the main-run
scores and the vestigial-regression results existed. Nothing here is blind. Contamination was checked for OLMo only: **Qwen3-8B and
Gemma 4 31B results cannot exclude training-data overlap.**

## Data

`results/capstone/<family>/<stage>/core/exchanges_exchange/`: OLMo 3 7B base / SFT / DPO / Instruct, Qwen3-8B base / IT, Gemma 4 31B
base / IT. 300 real question-reply exchanges per corpus (Gemma: the first 100), each scored as the **opening user turn** and as the
**reply after its own question**, tagged (chat template) and plain (bare text). Mean NLL and entropy over content positions 1+, each
target weighted equally.

| | oasst1 | UltraChat | WildChat |
|---|---|---|---|
| who wrote user / reply | human / human | ChatGPT / ChatGPT | human / ChatGPT |
| role in the analysis | human-written column | model-written column | separate (authorship differs between roles) |
| exchanges (OLMo, Qwen / Gemma) | 300 / 100 | 300 / 100 | 300 / 100 |
| replies truncated at 300 tokens (OLMo; Qwen, Gemma similar) | 38% | 55% | 0% |

**Validation (all passed, every checkpoint):** 12 conditions per checkpoint, no duplicates or missing conditions, finite values,
identical exchange IDs, truncation flags and one inputs fingerprint per family, tagged and plain scoring the same tokens, and the
per-token tables reproducing `scores.csv`'s means (positions 1+ and 0+) to ≤ 3e-6 nats (`scores.csv` stores 6 decimals). **Not covered:** follow-up user turns and
longer histories (the multi-turn `pilot_v3` route is separate and not analysed here).

## Findings

### 1. The assistant persona's distributions are distinctive, but already in the base models

At every checkpoint, base included, the tagged reply has lower entropy **and** lower NLL than the opening user turn: for example
OLMo base, oasst1, entropy −1.45 [−1.53, −1.37] and NLL −1.45 [−1.54, −1.35] nats/token (reply − user). The two lower together, so
the base gap difference is near zero (+0.00 [−0.04, +0.05]). The texts differ (a reply is predictable from its question), so this
level difference is **not** evidence of a training effect. Figures: `figures/levels_oasst1.png`, `levels_ultrachat.png`.

### 2. Training concentrates predictions on both personas, and more on the user turn

Post-training lowers entropy for both roles in every model. The user turn's entropy falls **more** than the reply's, in 9 of 10
headline cells (Δ(reply − user) entropy +0.07 to +0.79 nats); the exception, Gemma IT on UltraChat, is −0.07 [−0.20, +0.06]. So
there is **no assistant-specific concentration** in this design. What looks like general sharpening is, if anything, stronger on the
user side. Figure: `figures/contrast_change.png` (top row).

### 3. Criterion 1 (the gap): met for OLMo DPO and Instruct, through NLL rather than concentration

| checkpoint | oasst1 (human column) | UltraChat (model column) | criterion 1 |
|---|---|---|---|
| OLMo SFT | −0.13 [−0.17, −0.10] | −0.11 [−0.14, −0.09] | not met (opposite direction) |
| OLMo DPO | **+0.55 [+0.47, +0.63]** | **+0.49 [+0.43, +0.54]** | **met** |
| OLMo Instruct | **+0.51 [+0.43, +0.60]** | **+0.44 [+0.39, +0.49]** | **met** |
| Qwen IT | −0.60 [−0.67, −0.53] | −0.05 [−0.10, −0.01] | not met (opposite direction) |
| Gemma IT | −7.69 [−8.15, −7.23] | −7.31 [−7.66, −6.96] | not met (opposite direction) |

Change from base of (reply − user) gap, tagged, positions 1+, nats/token, 95% CI. The verdicts are the same with the first token
included (positions 0+) and on exchanges with untruncated replies only (`results.md`, Sensitivity).

What drives each result (`figures/changes_oasst1.png`, `changes_ultrachat.png`):

- **OLMo DPO / Instruct.** The reply's NLL on the recorded human or ChatGPT reply rises by +1.1 to +1.2 nats, against +0.3 to +0.4
  for the user turn, while its entropy falls less than the user turn's. So the gap grows because the post-trained model is
  confident and **misses the recorded reply** (it expects its own kind of reply), not because the assistant's distribution
  concentrates more. The rise appears at DPO (at SFT both roles change alike, with the user's gap rising slightly more) and
  persists across content positions (`figures/positions_*.png`).
- **Qwen IT.** Its user-turn entropy falls much more than its reply entropy (oasst1: −1.19 vs −0.40), so the user gap grows more.
- **Gemma IT.** Its tagged user turns become very poorly predicted (NLL +8.4 nats/token at low entropy), the same effect that
  makes Gemma vestigial regression's positive control. That dominates the contrast; the reply's own gap still rises (+1.4 to +1.7).

### 4. Framing: OLMo's reply-specific rise depends on the role tags

In plain framing (the same tokens with tags removed), OLMo DPO / Instruct's (reply − user) NLL change shrinks to +0.15 to +0.21,
and the gap contrast reverses (−0.14 to −0.31). The extra miss on recorded replies is therefore largely tied to the assistant tag
(the model expecting its own reply in its own slot). Qwen's plain gap contrast is ≈ 0. Gemma's stays strongly negative but
smaller (−2.8 to −3.2): its user-turn NLL rises even without tags. Figures: `contrast_change.png` (hollow markers),
`changes_*.png`.

### 5. WildChat and cross-family comparisons

WildChat (human user, ChatGPT reply) follows OLMo's headline pattern (gap contrast +0.47 DPO, +0.39 Instruct) and Qwen's (−0.51),
but its roles differ in authorship, so it is reported separately and outside the criterion. On the common first 100 exchanges in
bits per byte (`figures/cross_family_bpb.png`), the signs are the same as within families: positive gap contrasts for OLMo DPO and
Instruct, negative for Qwen and strongly negative for Gemma. This is descriptive only, given different tokenizers and unchecked
contamination outside OLMo.

## How to read this

- Lower assistant entropy means more concentrated predictions; on its own it does not show a training effect (finding 1).
- An assistant-specific entropy fall through training would support differential concentration. It is not observed: the user turn
  falls more (finding 2).
- NLL measures fit to the **recorded** text. A rising reply NLL means the model departs from these human- or ChatGPT-written
  replies, not that it predicts its own outputs worse.
- The gap (NLL − entropy) is a diagnostic. A rising gap means confidence grows relative to fit on this text. It is not a
  calibration test and says nothing about the distribution's shape.
- User and assistant texts differ in content and context, so none of these contrasts isolates a causal effect of the role label.
  The tagged − plain comparison (finding 4) is the closest this design gets.

## What this does and does not settle

- **Settled, within this design.** Criterion 1, applied as specified on 2026-11-04, is met for OLMo DPO and Instruct and not met for
  OLMo SFT, Qwen3-8B IT and Gemma 4 31B IT (opposite direction in all three).
- **Uncertain.**
  - Whether a gap rise driven by NLL on others' replies is the "distortion" the measure intends, or the model expecting its own
    replies. The entropy evidence points to the second.
  - How the contrast is affected by the user persona's own degradation: in Gemma, and partly Qwen, it is overwhelmed by the
    user-side change that vestigial regression measures.
  - One base/instruct pair each for Qwen and Gemma (Gemma n = 100 per corpus).
  - Contamination outside OLMo.
- **Needs further collection.**
  - Follow-up user turns and longer histories, i.e. the original-conversation route: `pilot_v3` candidates, `core.score_conversations`,
    with a final candidate set and a scoring run per checkpoint.
  - The model's own replies, to separate "expects its own reply" from distortion. Neither is part of this analysis.

## Reproduce

Generated at code revision `ad6cdcf` (clean); a second run reproduced every table byte for byte. Input file hashes and the
per-family inputs fingerprints are in `provenance.json`.

```bash
uv run python -m measures.logit_distortion.analyze      # ~1 min on CPU; writes this directory (report.md is hand-written, not regenerated)
uv run pytest tests/test_logit_distortion.py            # CPU-only, synthetic outputs
```
