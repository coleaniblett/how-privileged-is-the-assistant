# Conversation-scoring inputs: qwen3-8b

Candidates: `candidates/logit_distortion/pilot_v3` (contamination: checked). Tokenizer `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218`. Input fingerprint `4790bf2e6bfe2162…`. Inputs only; no model was run.

Rendering: template_kwargs: {'enable_thinking': False}; user_header: '<|im_start|>user\n'; assistant_header: '<|im_start|>assistant\n<think>\n\n</think>\n\n'; system: 'no default system prompt; original system messages rendered as given'; think: 'non-thinking: assistant targets open with the empty think block (enable_thinking=False); earlier assistant turns in their stored form (the template drops their think block)'

Context limit (minimum over the family's checkpoints): 32768 (base 32768, instruct 40960).

| Source | Group | Eligible candidates | Scored | Rendering exclusions | Length exclusions |
|---|---|---:|---:|---:|---:|
| oasst1 | human_assistant | 16 | 16 | 0 | 0 |
| oasst1 | human_user | 9 | 9 | 0 | 0 |
| ultrachat | model_assistant | 34 | 34 | 0 | 0 |
| ultrachat | model_user | 21 | 21 | 0 | 0 |
| wildchat | human_user | 18 | 18 | 0 | 0 |
| wildchat | model_assistant | 21 | 21 | 0 | 0 |

Exclusions by stage and reason (candidate = ineligible in the build, not scored by design):

- candidate / candidate_ineligible: 56

Scored input lengths (tokens, including history, headers and closing): min 25, median 517, max 4111. Nothing is truncated.
