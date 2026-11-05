# Assistant text in the user slot (teacher-forced), apertus-8b

Exploratory, outside every verdict (plan: `docs/decisions.md` 2026-11-04). The exchange set's replies, first 300 tokens, fp32: as the opening user turn (`tagged`) vs as bare plain text, neither with its question before it, so the scored tokens are identical and only the label differs. contrast = NLL(reply@user) - NLL(reply@plain), mean_nll_from1 (nats per token); change from base of the contrast, paired per exchange, 95% bootstrap CI over exchanges.

Reading: Assistant text becomes more expected in the user's turn if the change is <= -0.05 nats with the CI below 0 in both oasst1 and UltraChat (WildChat beside). Limit: likelihood cannot separate Assistant traits spreading into the user slot from other changes to the user slot itself.

| stage | corpus | exchanges | base contrast | stage contrast | change from base [95% CI] | meets reading |
|---|---|---|---|---|---|---|
| instruct | oasst1 | 298 | -0.019 | -0.041 | -0.021 [-0.033, -0.010] | no |
| instruct | ultrachat | 300 | -0.004 | -0.022 | -0.018 [-0.028, -0.008] | no |
| instruct | wildchat | 300 | -0.057 | -0.077 | -0.020 [-0.039, -0.002] | no |

- **instruct: not met** (oasst1 and UltraChat both -0.05 or lower with CI below 0: False).
