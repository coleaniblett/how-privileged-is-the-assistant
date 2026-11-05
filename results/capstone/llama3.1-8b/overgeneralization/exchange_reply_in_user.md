# Assistant text in the user slot (teacher-forced), llama3.1-8b

Exploratory, outside every verdict (plan: `docs/decisions.md` 2026-11-04). The exchange set's replies, first 300 tokens, fp32: as the opening user turn (`tagged`) vs as bare plain text, neither with its question before it, so the scored tokens are identical and only the label differs. contrast = NLL(reply@user) - NLL(reply@plain), mean_nll_from1 (nats per token); change from base of the contrast, paired per exchange, 95% bootstrap CI over exchanges.

Reading: Assistant text becomes more expected in the user's turn if the change is <= -0.05 nats with the CI below 0 in both oasst1 and UltraChat (WildChat beside). Limit: likelihood cannot separate Assistant traits spreading into the user slot from other changes to the user slot itself.

| stage | corpus | exchanges | base contrast | stage contrast | change from base [95% CI] | meets reading |
|---|---|---|---|---|---|---|
| instruct | oasst1 | 300 | -0.040 | +0.139 | +0.179 [+0.163, +0.196] | no |
| instruct | ultrachat | 300 | -0.024 | +0.127 | +0.151 [+0.133, +0.170] | no |
| instruct | wildchat | 300 | -0.081 | +0.254 | +0.335 [+0.304, +0.366] | no |

- **instruct: not met** (oasst1 and UltraChat both -0.05 or lower with CI below 0: False).
