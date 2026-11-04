# User-slot overgeneralization: judge arm, olmo3-7b

**PROVISIONAL**: Cole's judge labels (200 incomplete prefills x 3 samples) before the team's hand validation; invalid samples are excluded, not relabelled. See docs/decisions.md 2026-11-04.

Stage minus base in percentage points, 95% bootstrap CI over prefills, paired per prefill. Primary = share of user-voice samples labelled user_voice_with_assistant_traits; secondary = the same count over all samples. Prefills with no user-voice sample in the base or in the stage are dropped from the primary row (count shown).

| stage | arm | prefills used | dropped | base share (pp) | stage share (pp) | change (pp) |
|---|---|---|---|---|---|---|
| sft | primary | 155 | 45 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] |
| sft | secondary | 200 | 0 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] |
| dpo | primary | 156 | 44 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] |
| dpo | secondary | 199 | 1 | 0.0 | 0.0 | +0.0 [+0.0, +0.0] |
| instruct | primary | 156 | 44 | 0.0 | 0.2 | +0.2 [+0.0, +0.6] |
| instruct | secondary | 200 | 0 | 0.0 | 0.2 | +0.2 [+0.0, +0.5] |

Samples the judge refused (label invalid), excluded: base 18, sft 6, dpo 6, instruct 4 (total 34).

**Verdict rule (OLMo instruct, primary, >= 5 pp with CI excluding zero; the bar is fixed, docs/decisions.md 2026-11-04):** does not fire

