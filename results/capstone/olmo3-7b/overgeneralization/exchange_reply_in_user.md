# Assistant text in the user slot (teacher-forced), olmo3-7b

Exploratory, outside every verdict (plan: `docs/decisions.md` 2026-11-04). The exchange set's replies, first 300 tokens, fp32: as the opening user turn (`tagged`) vs as bare plain text, neither with its question before it, so the scored tokens are identical and only the label differs. contrast = NLL(reply@user) - NLL(reply@plain), mean_nll_from1 (nats per token); change from base of the contrast, paired per exchange, 95% bootstrap CI over exchanges.

Reading: Assistant text becomes more expected in the user's turn if the change is <= -0.05 nats with the CI below 0 in both oasst1 and UltraChat (WildChat beside). Limit: likelihood cannot separate Assistant traits spreading into the user slot from other changes to the user slot itself.

| stage | corpus | exchanges | base contrast | stage contrast | change from base [95% CI] | meets reading |
|---|---|---|---|---|---|---|
| sft | oasst1 | 300 | -0.067 | -0.111 | -0.044 [-0.057, -0.032] | no |
| sft | ultrachat | 300 | -0.043 | -0.116 | -0.074 [-0.084, -0.063] | yes |
| sft | wildchat | 300 | -0.127 | -0.180 | -0.053 [-0.075, -0.033] | yes |
| dpo | oasst1 | 300 | -0.067 | -0.103 | -0.037 [-0.055, -0.018] | no |
| dpo | ultrachat | 300 | -0.043 | -0.100 | -0.058 [-0.071, -0.044] | yes |
| dpo | wildchat | 300 | -0.127 | -0.147 | -0.020 [-0.043, +0.002] | no |
| instruct | oasst1 | 300 | -0.067 | -0.118 | -0.052 [-0.070, -0.033] | yes |
| instruct | ultrachat | 300 | -0.043 | -0.112 | -0.070 [-0.084, -0.055] | yes |
| instruct | wildchat | 300 | -0.127 | -0.162 | -0.035 [-0.058, -0.012] | no |

- **dpo: not met** (oasst1 and UltraChat both -0.05 or lower with CI below 0: False).
- **instruct: Assistant text is more expected in the user slot** (oasst1 and UltraChat both -0.05 or lower with CI below 0: True).
- **sft: not met** (oasst1 and UltraChat both -0.05 or lower with CI below 0: False).
