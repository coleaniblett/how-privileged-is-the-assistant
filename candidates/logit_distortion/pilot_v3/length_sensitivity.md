# Minimum target length sensitivity (data only)

Same selected conversations; only the minimum target length changes (build default: 20). Maximum target length, context length and contamination rules are held fixed; duplicate cases are re-resolved. With eligible_quota selection the conversations were chosen at the default minimum, so lower thresholds here do not show what a lower-threshold quota build would select. Cells: eligible targets / contributing conversations. No model outcomes.

| Source | Group | Position | Targets | min 5: eligible / conversations | min 10: eligible / conversations | min 20: eligible / conversations |
|---|---|---|---:|---:|---:|---:|
| oasst1 | human_assistant | follow_up | 9 | 9 / 8 | 9 / 8 | 9 / 8 |
| oasst1 | human_assistant | opening | 8 | 7 / 7 | 7 / 7 | 7 / 7 |
| oasst1 | human_user | follow_up | 10 | 10 / 8 | 9 / 7 | 4 / 3 |
| oasst1 | human_user | opening | 8 | 8 / 8 | 7 / 7 | 5 / 5 |
| ultrachat | model_assistant | follow_up | 27 | 26 / 8 | 26 / 8 | 26 / 8 |
| ultrachat | model_assistant | opening | 8 | 8 / 8 | 8 / 8 | 8 / 8 |
| ultrachat | model_user | follow_up | 27 | 27 / 8 | 27 / 8 | 21 / 8 |
| ultrachat | model_user | opening | 8 | 8 / 8 | 7 / 7 | 0 / 0 |
| wildchat | human_user | follow_up | 27 | 25 / 7 | 21 / 7 | 13 / 5 |
| wildchat | human_user | opening | 8 | 6 / 6 | 6 / 6 | 5 / 5 |
| wildchat | model_assistant | follow_up | 27 | 15 / 7 | 15 / 7 | 15 / 7 |
| wildchat | model_assistant | opening | 8 | 6 / 6 | 6 / 6 | 6 / 6 |
