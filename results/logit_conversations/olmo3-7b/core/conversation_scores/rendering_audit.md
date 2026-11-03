# Conversation-scoring inputs: olmo3-7b

Candidates: `candidates/logit_distortion/pilot_v3` (contamination: checked). Tokenizer `allenai/Olmo-3-7B-Instruct` @ `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`. Input fingerprint `f99760299bf5b3fa…`. Inputs only; no model was run.

Rendering: template_kwargs: {}; user_header: '<|im_start|>user\n'; assistant_header: '<|im_start|>assistant\n'; system: 'template default system prompt when the conversation has no original system message; an original system message replaces it (template behavior, verified), never duplicated'

Context limit (minimum over the family's checkpoints): 65536 (base 65536, sft 65536, dpo 65536, instruct 65536).

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

Scored input lengths (tokens, including history, headers and closing): min 53, median 540, max 4126. Nothing is truncated.
