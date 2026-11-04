# User-slot overgeneralization: judge arm, qwen3-8b

**PROVISIONAL**: Cole's judge labels (200 incomplete prefills x 3 samples) before the team's hand validation; invalid samples are excluded, not relabelled. See docs/decisions.md 2026-11-04.

Stage minus base in percentage points, 95% bootstrap CI over prefills, paired per prefill. Primary = share of user-voice samples labelled user_voice_with_assistant_traits; secondary = the same count over all samples. Prefills with no user-voice sample in the base or in the stage are dropped from the primary row (count shown).

| stage | arm | prefills used | dropped | base share (pp) | stage share (pp) | change (pp) |
|---|---|---|---|---|---|---|
| instruct | primary | 117 | 83 | 0.3 | 0.0 | -0.3 [-0.9, +0.0] |
| instruct | secondary | 199 | 1 | 0.3 | 0.0 | -0.3 [-0.8, +0.0] |

Samples the judge refused (label invalid), excluded: base 5, instruct 21 (total 26).

**Verdict rule (OLMo instruct, primary, >= 5 pp with CI excluding zero; the bar is fixed, docs/decisions.md 2026-11-04):** not applicable to this family (supporting only; no contamination check is possible for it)

Caveat: post-training data for this family is not open, so test-text contamination cannot be checked.
