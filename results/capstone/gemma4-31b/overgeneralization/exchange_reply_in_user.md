# Assistant text in the user slot (teacher-forced), gemma4-31b

Exploratory, outside every verdict (plan: `docs/decisions.md` 2026-11-04). The exchange set's replies, first 300 tokens, fp32: as the opening user turn (`tagged`) vs as bare plain text, neither with its question before it, so the scored tokens are identical and only the label differs. contrast = NLL(reply@user) - NLL(reply@plain), mean_nll_from1 (nats per token); change from base of the contrast, paired per exchange, 95% bootstrap CI over exchanges.

Reading: Assistant text becomes more expected in the user's turn if the change is <= -0.05 nats with the CI below 0 in both oasst1 and UltraChat (WildChat beside). Limit: likelihood cannot separate Assistant traits spreading into the user slot from other changes to the user slot itself.

| stage | corpus | exchanges | base contrast | stage contrast | change from base [95% CI] | meets reading |
|---|---|---|---|---|---|---|
| instruct | oasst1 | 100 | -0.004 | +1.690 | +1.693 [+1.429, +1.979] | no |
| instruct | ultrachat | 100 | +0.006 | +2.040 | +2.033 [+1.727, +2.344] | no |
| instruct | wildchat | 100 | -0.011 | +2.359 | +2.370 [+2.085, +2.665] | no |

- **instruct: not met** (oasst1 and UltraChat both -0.05 or lower with CI below 0: False).
