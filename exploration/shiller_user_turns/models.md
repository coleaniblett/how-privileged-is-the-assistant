# Models verified for the Shiller user-turn exploration

Written by `python -m exploration.shiller_user_turns.models` (HF API; no weights downloaded).

| key | repo | sha | role | license | gated | weights (GB) | architecture | template from | vocab vs template |
|---|---|---|---|---|---|---|---|---|---|
| olmo3-7b-instruct | `allenai/Olmo-3-7B-Instruct` | 6e5971d9eb | instruct | apache-2.0 | False | 14.6 | Olmo3ForCausalLM | `allenai/Olmo-3-7B-Instruct` | same |
| olmo3-7b-base | `allenai/Olmo-3-1025-7B` | a81bae42db | base | apache-2.0 | False | 14.6 | Olmo3ForCausalLM | `allenai/Olmo-3-7B-Instruct` | differs only on 14 added/special tokens |
| qwen3-8b | `Qwen/Qwen3-8B` | b968826d9c | instruct | apache-2.0 | False | 16.4 | Qwen3ForCausalLM | `Qwen/Qwen3-8B` | same |
| qwen3-8b-base | `Qwen/Qwen3-8B-Base` | 49e3418fbb | base | apache-2.0 | False | 16.4 | Qwen3ForCausalLM | `Qwen/Qwen3-8B` | differs only on 4 added/special tokens |
| gemma4-e4b-it | `google/gemma-4-E4B-it` | ee0ef60236 | instruct | apache-2.0 | False | 16.0 | Gemma4ForConditionalGeneration | `google/gemma-4-E4B-it` | same |
| gemma4-e4b-base | `google/gemma-4-E4B` | 411aa17b74 | base | apache-2.0 | False | 16.0 | Gemma4ForConditionalGeneration | `google/gemma-4-E4B-it` | same |
| gemma4-31b-it | `google/gemma-4-31B-it` | 842da3794e | instruct | apache-2.0 | False | 62.5 | Gemma4ForConditionalGeneration | `google/gemma-4-31B-it` | same |
| gemma4-31b-base | `google/gemma-4-31B` | 5bbc2fb1c1 | base | apache-2.0 | False | 62.5 | Gemma4ForConditionalGeneration | `google/gemma-4-31B-it` | same |
| qwen3-32b | `Qwen/Qwen3-32B` | 9216db5781 | instruct | apache-2.0 | False | 65.5 | Qwen3ForCausalLM | `Qwen/Qwen3-32B` | same |

## Turn markers, stop token and empty prompt

Generation stops only at the stop token (end of text), never at end of turn. `kwargs change prompt` says whether passing the option would change the rendered open user turn; the run uses the template defaults (`template_kwargs`).

### olmo3-7b-instruct

- markers: `{"end_of_turn": "<|im_end|>", "turn_open": "<|im_start|>", "user_header": "<|im_start|>user", "assistant_header": "<|im_start|>assistant"}`
- stop: ['<|endoftext|>'] (ids [100257])
- template_kwargs: `{}`; kwargs change prompt: `{"{\"enable_thinking\": true}": false, "{\"enable_thinking\": false}": false}`
- chat template sha: f5186d42d99c

Empty-condition prompt (exact string):

````
<|im_start|>system
You are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions><|im_end|>
<|im_start|>user

````

### olmo3-7b-base

- markers: `{"end_of_turn": "<|im_end|>", "turn_open": "<|im_start|>", "user_header": "<|im_start|>user", "assistant_header": "<|im_start|>assistant"}`
- stop: ['<|endoftext|>'] (ids [100257])
- template_kwargs: `{}`; kwargs change prompt: `{"{\"enable_thinking\": true}": false, "{\"enable_thinking\": false}": false}`
- chat template sha: f5186d42d99c

Empty-condition prompt (exact string):

````
<|im_start|>system
You are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions><|im_end|>
<|im_start|>user

````

### qwen3-8b

- markers: `{"end_of_turn": "<|im_end|>", "turn_open": "<|im_start|>", "user_header": "<|im_start|>user", "assistant_header": "<|im_start|>assistant"}`
- stop: ['<|endoftext|>'] (ids [151643])
- template_kwargs: `{}`; kwargs change prompt: `{"{\"enable_thinking\": true}": false, "{\"enable_thinking\": false}": false}`
- chat template sha: a55ee1b16601

Empty-condition prompt (exact string):

````
<|im_start|>user

````

### qwen3-8b-base

- markers: `{"end_of_turn": "<|im_end|>", "turn_open": "<|im_start|>", "user_header": "<|im_start|>user", "assistant_header": "<|im_start|>assistant"}`
- stop: ['<|endoftext|>'] (ids [151643])
- template_kwargs: `{}`; kwargs change prompt: `{"{\"enable_thinking\": true}": false, "{\"enable_thinking\": false}": false}`
- chat template sha: a55ee1b16601

Empty-condition prompt (exact string):

````
<|im_start|>user

````

### gemma4-e4b-it

- markers: `{"end_of_turn": "<turn|>", "turn_open": "<|turn>", "user_header": "<|turn>user", "assistant_header": "<|turn>model"}`
- stop: ['<eos>'] (ids [1])
- template_kwargs: `{}`; kwargs change prompt: `{"{\"enable_thinking\": true}": true, "{\"enable_thinking\": false}": false}`
- chat template sha: 0a2c8073c878

Empty-condition prompt (exact string):

````
<bos><|turn>user

````

### gemma4-e4b-base

- markers: `{"end_of_turn": "<turn|>", "turn_open": "<|turn>", "user_header": "<|turn>user", "assistant_header": "<|turn>model"}`
- stop: ['<eos>'] (ids [1])
- template_kwargs: `{}`; kwargs change prompt: `{"{\"enable_thinking\": true}": true, "{\"enable_thinking\": false}": false}`
- chat template sha: 0a2c8073c878

Empty-condition prompt (exact string):

````
<bos><|turn>user

````

### gemma4-31b-it

- markers: `{"end_of_turn": "<turn|>", "turn_open": "<|turn>", "user_header": "<|turn>user", "assistant_header": "<|turn>model"}`
- stop: ['<eos>'] (ids [1])
- template_kwargs: `{}`; kwargs change prompt: `{"{\"enable_thinking\": true}": true, "{\"enable_thinking\": false}": false}`
- chat template sha: ae53464bf3be

Empty-condition prompt (exact string):

````
<bos><|turn>user

````

### gemma4-31b-base

- markers: `{"end_of_turn": "<turn|>", "turn_open": "<|turn>", "user_header": "<|turn>user", "assistant_header": "<|turn>model"}`
- stop: ['<eos>'] (ids [1])
- template_kwargs: `{}`; kwargs change prompt: `{"{\"enable_thinking\": true}": true, "{\"enable_thinking\": false}": false}`
- chat template sha: ae53464bf3be

Empty-condition prompt (exact string):

````
<bos><|turn>user

````

### qwen3-32b

- markers: `{"end_of_turn": "<|im_end|>", "turn_open": "<|im_start|>", "user_header": "<|im_start|>user", "assistant_header": "<|im_start|>assistant"}`
- stop: ['<|endoftext|>'] (ids [151643])
- template_kwargs: `{}`; kwargs change prompt: `{"{\"enable_thinking\": true}": false, "{\"enable_thinking\": false}": false}`
- chat template sha: a55ee1b16601

Empty-condition prompt (exact string):

````
<|im_start|>user

````
