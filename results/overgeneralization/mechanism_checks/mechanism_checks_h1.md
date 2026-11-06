# Mechanism checks (spec H; exploratory, outside every verdict)

Plan, frozen materials and readings: `docs/decisions.md` 2026-11-04, `data/mechanism_checks_materials.json`. Many cells, regex detectors, prompts specific to these roles: read patterns, not single cells.

## H1. Is the style score a good measuring stick?

Fixed texts (200 per context: 100 saved base replies and 100 saved final replies) projected through base and final on the leakage matrix's axis. Generated-text fall = mean coord of the final's saved replies minus the base's saved replies in the same context (all saved rows); fixed-text change = the same texts through final minus through base, paired per text, 95% bootstrap CI over (template, question) items. Pre-set: partly the measuring stick if the fixed-text change is at least 50% of the generated-text fall, same sign, CI excluding 0.

| model | axis | context | generated-text fall | fixed-text change [95% CI] | fraction | partly the measuring stick |
|---|---|---|---|---|---|---|
| gemma | axis | own | +0.153 | -0.115 [-0.135, -0.094] | -0.75 | no |
| gemma | axis | named | -0.537 | -0.281 [-0.292, -0.267] | 0.52 | yes |
| gemma | axis | voiced | -0.754 | -0.326 [-0.334, -0.319] | 0.43 | no |
| gemma | axis | user | -0.374 | -0.286 [-0.297, -0.275] | 0.76 | yes |
| gemma | axis | narration | -0.706 | -0.299 [-0.309, -0.291] | 0.42 | no |
| gemma | axis | raw | -0.243 | -0.283 [-0.297, -0.268] | 1.16 | yes |
| olmo | axis | own | +0.300 | +0.103 [+0.081, +0.124] | 0.34 | no |
| olmo | axis | named | -0.158 | -0.046 [-0.080, -0.012] | 0.29 | no |
| olmo | axis | voiced | -0.797 | -0.180 [-0.214, -0.148] | 0.23 | no |
| olmo | axis | user | -0.223 | -0.086 [-0.106, -0.067] | 0.38 | no |
| olmo | axis | narration | -0.302 | +0.006 [-0.018, +0.030] | -0.02 | no |
| olmo | axis | raw | -0.054 | -0.050 [-0.065, -0.034] | 0.92 | yes |
| olmo | axis v2 | own | +0.380 | +0.117 [+0.089, +0.142] | 0.31 | no |
| olmo | axis v2 | named | -0.098 | -0.013 [-0.050, +0.023] | 0.13 | no |
| olmo | axis v2 | voiced | -0.761 | -0.150 [-0.182, -0.119] | 0.20 | no |
| olmo | axis v2 | user | -0.190 | -0.064 [-0.086, -0.041] | 0.33 | no |
| olmo | axis v2 | narration | -0.306 | +0.019 [-0.004, +0.040] | -0.06 | no |
| olmo | axis v2 | raw | -0.117 | -0.074 [-0.092, -0.056] | 0.63 | yes |
| qwen | axis | own | +0.164 | +0.040 [+0.024, +0.057] | 0.24 | no |
| qwen | axis | named | -0.293 | -0.048 [-0.059, -0.038] | 0.16 | no |
| qwen | axis | voiced | -0.453 | -0.078 [-0.088, -0.068] | 0.17 | no |
| qwen | axis | user | -0.042 | -0.006 [-0.016, +0.005] | 0.13 | no |
| qwen | axis | narration | -0.235 | -0.033 [-0.041, -0.026] | 0.14 | no |
| qwen | axis | raw | -0.022 | -0.019 [-0.038, -0.001] | 0.86 | yes |

### Fresh probe on the final model

Logistic probe on the final model's mean hidden state (the matrix's layer) over the reply, Assistant (own turn) vs non-Assistant (other contexts), balanced per question and template, trained on even question items and tested on odd ones. Correlations are per reply, on the test items. Pre-set: **mismatch** if the probe and the classifier agree (r >= 0.3, CI above 0) and the old axis does not (r <= 0.1 or opposite sign).

| model | probe test accuracy | context | corr(probe, classifier) | corr(probe, old axis) | n |
|---|---|---|---|---|---|
| gemma | 1.000 | all | -0.02 | +0.71 | 2402 |
| gemma | 1.000 | own | +0.01 | +0.55 | 192 |
| gemma | 1.000 | named | -0.19 | +0.80 | 384 |
| gemma | 1.000 | voiced | +0.22 | +0.10 | 192 |
| gemma | 1.000 | user | -0.08 | +0.70 | 482 |
| gemma | 1.000 | narration | -0.10 | +0.02 | 576 |
| gemma | 1.000 | raw | +0.14 | +0.71 | 576 |
| gemma | | **reading** | no mismatch | | |
| olmo | 0.924 | all | +0.03 | +0.74 | 2495 |
| olmo | 0.924 | own | -0.29 | +0.08 | 192 |
| olmo | 0.924 | named | -0.12 | +0.71 | 384 |
| olmo | 0.924 | voiced | -0.15 | +0.16 | 192 |
| olmo | 0.924 | user | -0.09 | +0.27 | 575 |
| olmo | 0.924 | narration | +0.04 | +0.10 | 576 |
| olmo | 0.924 | raw | -0.13 | +0.23 | 576 |
| olmo | | **reading** | no mismatch | | |
| qwen | 0.953 | all | +0.12 | +0.71 | 2496 |
| qwen | 0.953 | own | -0.31 | +0.84 | 192 |
| qwen | 0.953 | named | +0.14 | +0.68 | 384 |
| qwen | 0.953 | voiced | +0.18 | +0.35 | 192 |
| qwen | 0.953 | user | -0.15 | +0.46 | 576 |
| qwen | 0.953 | narration | -0.10 | -0.14 | 576 |
| qwen | 0.953 | raw | +0.01 | +0.33 | 576 |
| qwen | | **reading** | no mismatch | | |

