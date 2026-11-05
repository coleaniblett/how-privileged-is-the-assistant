# Model checks

| stage | repo | sha | license | gated | weights (GB) | vocab | chat template sha | tokenizer.json sha |
|---|---|---|---|---|---|---|---|---|
| base | `allenai/Olmo-3-1025-7B` | a81bae42db | apache-2.0 | False | 14.6 | 100278 | None | 73fd5254624f |
| sft | `allenai/Olmo-3-7B-Instruct-SFT` | e1452fc572 | apache-2.0 | False | 14.6 | 100278 | edbb35ef1219 | 32a022d09976 |
| dpo | `allenai/Olmo-3-7B-Instruct-DPO` | b33130b7de | apache-2.0 | False | 14.6 | 100278 | edbb35ef1219 | 7738a25c46a6 |
| instruct | `allenai/Olmo-3-7B-Instruct` | 6e5971d9eb | apache-2.0 | False | 14.6 | 100278 | f5186d42d99c | 7738a25c46a6 |

## Chat templates

Post-trained checkpoints share one chat template file: **False**.
Post-trained templates render a plain user/assistant exchange identically: **True**.

Template file of sft, dpo vs instruct (unified diff):

```diff
@@ -2 +2 @@
-You are a helpful function-calling AI assistant. ' -}}{%- if tools is none -%}{{- 'You do not currently have access to any functions. <functions></functions><|im_end|>
+You are a helpful function-calling AI assistant. ' -}}{%- if tools is none or (tools | length) == 0 -%}{{- 'You do not currently have access to any functions. <functions></functions><|im_end|>
```

Base has its own chat template: False; it is scored with the Instruct template.

## Tokenizers

All 4 share one tokenizer.json file: **False**.
All 4 share one merge list: **True**.
All 4 share one vocab: **False**.

`base` vocab entries differing from Instruct (token, base id, instruct id):

- `</function_calls>`: None vs 100269
- `</functions>`: None vs 100267
- `<function_calls>`: None vs 100268
- `<functions>`: None vs 100266
- `<|extra_id_10|>`: 100275 vs None
- `<|extra_id_1|>`: 100266 vs 100270
- `<|extra_id_2|>`: 100267 vs 100271
- `<|extra_id_3|>`: 100268 vs 100272
- `<|extra_id_4|>`: 100269 vs 100273
- `<|extra_id_5|>`: 100270 vs 100274
- `<|extra_id_6|>`: 100271 vs 100275
- `<|extra_id_7|>`: 100272 vs None
- `<|extra_id_8|>`: 100273 vs None
- `<|extra_id_9|>`: 100274 vs None

Vocab differences are confined to reserved/added tokens: **True**.

## Derived by core.render from the template

Template places content verbatim exactly once in each slot: **True**.

- user slot (prefix | suffix): `('<|im_start|>system\nYou are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions><|im_end|>\n<|im_start|>user\n', '<|im_end|>\n')`
- assistant slot (prefix | suffix): `('<|im_start|>system\nYou are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions><|im_end|>\n<|im_start|>assistant\n', '<|endoftext|>')`
- end-of-turn token: `<|im_end|>`; user->assistant handoff string: `<|im_end|>
<|im_start|>assistant`
- assistant-slot opening beyond the stored-turn header (assistant_opening_extra): ``
- config: dtype fp32, offload False (gpu_gib 34), scoring batch size 8, generation precision bf16, extra handoff markers [], assistant_think_block None, contamination check possible: True

Special tokens: bos='<|endoftext|>' eos='<|endoftext|>' pad='<|pad|>'

Instruct template rendering of a user + assistant exchange:

```
<|im_start|>system
You are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions><|im_end|>
<|im_start|>user
USER<|im_end|>
<|im_start|>assistant
ASSISTANT<|endoftext|>
```

Instruct chat template:

```jinja
{%- set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 -%}{%- if not has_system -%}{{- '<|im_start|>system
You are a helpful function-calling AI assistant. ' -}}{%- if tools is none -%}{{- 'You do not currently have access to any functions. <functions></functions><|im_end|>
' -}}{%- else -%}{{- 'You are provided with function signatures within <functions></functions> XML tags. You may call one or more functions to assist with the user query. Output any function calls within <function_calls></function_calls> XML tags. Do not make assumptions about what values to plug into functions.' -}}{{- '<functions>' -}}{{- tools | tojson -}}{{- '</functions><|im_end|>
' -}}{%- endif -%}{%- endif -%}{%- for message in messages -%}{%- if message['role'] == 'system' -%}{{- '<|im_start|>system
' + message['content'] -}}{%- if tools is not none -%}{{- '<functions>' -}}{{- tools | tojson -}}{{- '</functions>' -}}{%- elif message.get('functions', none) is not none -%}{{- ' <functions>' + message['functions'] + '</functions>' -}}{%- endif -%}{{- '<|im_end|>
' -}}{%- elif message['role'] == 'user' -%}{{- '<|im_start|>user
' + message['content'] + '<|im_end|>
' -}}{%- elif message['role'] == 'assistant' -%}{{- '<|im_start|>assistant
' -}}{%- if message.get('content', none) is not none -%}{{- message['content'] -}}{%- endif -%}{%- if message.get('function_calls', none) is not none -%}{{- '<function_calls>' + message['function_calls'] + '</function_calls>' -}}{% elif message.get('tool_calls', none) is not none %}{{- '<function_calls>' -}}{%- for tool_call in message['tool_calls'] %}{%- if tool_call is mapping and tool_call.get('function', none) is not none %}{%- set args = tool_call['function']['arguments'] -%}{%- set ns = namespace(arguments_list=[]) -%}{%- for key, value in args.items() -%}{%- set ns.arguments_list = ns.arguments_list + [key ~ '=' ~ (value | tojson)] -%}{%- endfor -%}{%- set arguments = ns.arguments_list | join(', ') -%}{{- tool_call['function']['name'] + '(' + arguments + ')' -}}{%- if not loop.last -%}{{ '
' }}{%- endif -%}{% else %}{{- tool_call -}}{%- endif %}{%- endfor %}{{- '</function_calls>' -}}{%- endif -%}{%- if not loop.last -%}{{- '<|im_end|>' + '
' -}}{%- else -%}{{- eos_token -}}{%- endif -%}{%- elif message['role'] == 'environment' -%}{{- '<|im_start|>environment
' + message['content'] + '<|im_end|>
' -}}{%- elif message['role'] == 'tool' -%}{{- '<|im_start|>environment
' + message['content'] + '<|im_end|>
' -}}{%- endif -%}{%- if loop.last and add_generation_prompt -%}{{- '<|im_start|>assistant
' -}}{%- endif -%}{%- endfor -%}
```

<!-- family:gemma4-31b -->
# Model checks: gemma4-31b

| stage | repo | sha | license | gated | weights (GB) | vocab | chat template sha | tokenizer.json sha |
|---|---|---|---|---|---|---|---|---|
| base | `google/gemma-4-31B` | 5bbc2fb1c1 | apache-2.0 | False | 62.5 | 262144 | None | 12bac982b793 |
| instruct | `google/gemma-4-31B-it` | 842da3794e | apache-2.0 | False | 62.5 | 262144 | ae53464bf3be | cc8d3a0ce364 |

## Chat templates

Post-trained checkpoints share one chat template file: **True**.
Post-trained templates render a plain user/assistant exchange identically: **True**.

Base has its own chat template: False; it is scored with the Instruct template.

## Tokenizers

All 2 share one tokenizer.json file: **False**.
All 2 share one merge list: **True**.
All 2 share one vocab: **True**.

Vocab differences are confined to reserved/added tokens: **True**.

## Derived by core.render from the template

Template places content verbatim exactly once in each slot: **True**.

- user slot (prefix | suffix): `('<bos><|turn>user\n', '<turn|>\n')`
- assistant slot (prefix | suffix): `('<bos><|turn>model\n<|channel>thought\n<channel|>', '<turn|>\n')`
- end-of-turn token: `<turn|>`; user->assistant handoff string: `<turn|>
<|turn>model`
- assistant-slot opening beyond the stored-turn header (assistant_opening_extra): `<|channel>thought
<channel|>`
- config: dtype fp32, offload True (gpu_gib 30), scoring batch size 4, generation precision int8, extra handoff markers ['<|channel>'], assistant_think_block None, contamination check possible: False

Special tokens: bos='<bos>' eos='<eos>' pad='<pad>'

Instruct template rendering of a user + assistant exchange:

```
<bos><|turn>user
USER<turn|>
<|turn>model
ASSISTANT<turn|>

```

Instruct chat template:

```jinja
{#
  Template: Google Gemma 4 Canonical Chat Template
  Author: Google Gemma Engineering Team
  Published: 2026-07-09
  Context: Fixed tool-calling loops, turn closures, and thinking content-ordering.
#}
{%- macro format_parameters(properties, required, filter_keys=false) -%}
    {%- set standard_keys = ['description', 'type', 'properties', 'required', 'nullable'] -%}
    {%- set ns = namespace(found_first=false) -%}
    {%- for key, value in properties | dictsort -%}
        {%- set add_comma = false -%}
        {%- if not filter_keys or key not in standard_keys -%}
            {%- if ns.found_first %},{% endif -%}
            {%- set ns.found_first = true -%}
            {{ key }}:{
            {%- if value['description'] -%}
                description:<|"|>{{ value['description'] }}<|"|>
                {%- set add_comma = true -%}
            {%- endif -%}
            {%- if value['type'] | upper == 'STRING' -%}
                {%- if value['enum'] -%}
                    {%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}
                    enum:{{ format_argument(value['enum']) }}
                {%- endif -%}
            {%- elif value['type'] | upper == 'ARRAY' -%}
                {%- if value['items'] is mapping and value['items'] -%}
                    {%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}
                    items:{
                    {%- set ns_items = namespace(found_first=false) -%}
                    {%- for item_key, item_value in value['items'] | dictsort -%}
                        {%- if item_value is not none -%}
                            {%- if ns_items.found_first %},{% endif -%}
                            {%- set ns_items.found_first = true -%}
                            {%- if item_key == 'properties' -%}
                                properties:{
                                {%- if item_value is mapping -%}
                                    {{- format_parameters(item_value, value['items']['required'] | default([])) -}}
                                {%- endif -%}
                                }
                            {%- elif item_key == 'required' -%}
                                required:[
                                {%- for req_item in item_value -%}
                                    <|"|>{{- req_item -}}<|"|>
                                    {%- if not loop.last %},{% endif -%}
                                {%- endfor -%}
                                ]
                            {%- elif item_key == 'type' -%}
                                {%- if item_value is string -%}
                                    type:{{ format_argument(item_value | upper) }}
                                {%- else -%}
                                    type:{{ format_argument(item_value | map('upper') | list) }}
                                {%- endif -%}
                            {%- else -%}
                                {{ item_key }}:{{ format_argument(item_value) }}
                            {%- endif -%}
                        {%- endif -%}
                    {%- endfor -%}
                    }
                {%- endif -%}
            {%- endif -%}
            {%- if value['nullable'] %}
                {%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}
                nullable:true
            {%- endif -%}
            {%- if value['type'] | upper == 'OBJECT' -%}
                {%- if value['properties'] is defined and value['properties'] is mapping -%}
                    {%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}
                    properties:{
                    {{- format_parameters(value['properties'], value['required'] | default([])) -}}
                    }
                {%- elif value is mapping -%}
                    {%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}
                    properties:{
                    {{- format_parameters(value, value['required'] | default([]), filter_keys=true) -}}
                    }
                {%- endif -%}
                {%- if value['required'] -%}
                    {%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}
                    required:[
                    {%- for item in value['required'] | default([]) -%}
                        <|"|>{{- item -}}<|"|>
                        {%- if not loop.last %},{% endif -%}
                    {%- endfor -%}
                    ]
                {%- endif -%}
            {%- endif -%}
            {%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}
            type:<|"|>{{ value['type'] | upper }}<|"|>}
        {%- endif -%}
    {%- endfor -%}
{%- endmacro -%}
{%- macro format_function_declaration(tool_data) -%}
    declaration:{{- tool_data['function']['name'] -}}{description:<|"|>{{- tool_data['function']['description'] -}}<|"|>
    {%- set params = tool_data['function']['parameters'] -%}
    {%- if params -%}
        ,parameters:{
        {%- if params['properties'] -%}
            properties:{ {{- format_parameters(params['properties'], params['required']) -}} },
        {%- endif -%}
        {%- if params['required'] -%}
            required:[
            {%- for item in params['required'] -%}
                <|"|>{{- item -}}<|"|>
                {{- ',' if not loop.last -}}
            {%- endfor -%}
            ],
        {%- endif -%}
        {%- if params['type'] -%}
            type:<|"|>{{- params['type'] | upper -}}<|"|>}
        {%- endif -%}
    {%- endif -%}
    {%- if 'response' in tool_data['function'] -%}
        {%- set response_declaration = tool_data['function']['response'] -%}
        ,response:{
        {%- if response_declaration['description'] -%}
            description:<|"|>{{- response_declaration['description'] -}}<|"|>,
        {%- endif -%}
        {%- if response_declaration['type'] | upper == 'OBJECT' -%}
            type:<|"|>{{- response_declaration['type'] | upper -}}<|"|>}
        {%- endif -%}
    {%- endif -%}
    }
{%- endmacro -%}
{%- macro format_argument(argument, escape_keys=True) -%}
    {%- if argument is none -%}
        {{- 'null' -}}
    {%- elif argument is string -%}
        {{- '<|"|>' + argument + '<|"|>' -}}
    {%- elif argument is boolean -%}
        {{- 'true' if argument else 'false' -}}
    {%- elif argument is mapping -%}
        {{- '{' -}}
        {%- set ns = namespace(found_first=false) -%}
        {%- for key, value in argument | dictsort -%}
            {%- if ns.found_first %},{% endif -%}
            {%- set ns.found_first = true -%}
            {%- if escape_keys -%}
                {{- '<|"|>' + key + '<|"|>' -}}
            {%- else -%}
                {{- key -}}
            {%- endif -%}
            :{{- format_argument(value, escape_keys=escape_keys) -}}
        {%- endfor -%}
        {{- '}' -}}
    {%- elif argument is sequence -%}
        {{- '[' -}}
        {%- for item in argument -%}
            {{- format_argument(item, escape_keys=escape_keys) -}}
            {%- if not loop.last %},{% endif -%}
        {%- endfor -%}
        {{- ']' -}}
    {%- else -%}
        {{- argument -}}
    {%- endif -%}
{%- endmacro -%}
{%- macro strip_thinking(text) -%}
    {%- set ns = namespace(result='') -%}
    {%- for part in text.split('<channel|>') -%}
        {%- if '<|channel>' in part -%}
            {%- set ns.result = ns.result + part.split('<|channel>')[0] -%}
        {%- else -%}
            {%- set ns.result = ns.result + part -%}
        {%- endif -%}
    {%- endfor -%}
    {{- ns.result | trim -}}
{%- endmacro -%}

{%- macro format_tool_response_block(tool_name, response) -%}
    {{- '<|tool_response>' -}}
    {%- if response is mapping -%}
        {{- 'response:' + tool_name + '{' -}}
        {%- for key, value in response | dictsort -%}
            {{- key -}}:{{- format_argument(value, escape_keys=False) -}}
            {%- if not loop.last %},{% endif -%}
        {%- endfor -%}
        {{- '}' -}}
    {%- else -%}
        {{- 'response:' + tool_name + '{value:' + format_argument(response, escape_keys=False) + '}' -}}
    {%- endif -%}
    {{- '<tool_response|>' -}}
{%- endmacro -%}

{#- ===== SETUP ===== -#}
{%- set ns = namespace(prev_message_type=None, prev_non_tool_role=None) -%}
{%- set loop_messages = messages -%}
{%- set enable_thinking = enable_thinking | default(false) -%}
{%- set preserve_thinking = preserve_thinking | default(false) -%}
{{- bos_token -}}
{#- Handle System/Tool Definitions Block -#}
{%- if enable_thinking or tools or (messages and messages[0]['role'] in ['system', 'developer']) -%}
    {{- '<|turn>system\n' -}}
    {#- Inject Thinking token at the very top of the FIRST system turn -#}
    {%- if enable_thinking -%}
        {{- '<|think|>\n' -}}
        {%- set ns.prev_message_type = 'think' -%}
    {%- endif -%}
    {%- if messages and messages[0]['role'] in ['system', 'developer'] -%}
        {%- if messages[0]['content'] is string -%}
            {{- messages[0]['content'] | trim -}}
        {%- elif messages[0]['content'] is sequence -%}
            {%- for item in messages[0]['content'] -%}
                {{- item['text'] | trim + ' '-}}
            {%- endfor -%}
        {%- endif -%}
        {%- set loop_messages = messages[1:] -%}
    {%- endif -%}
    {%- if tools -%}
        {%- for tool in tools %}
            {{- '<|tool>' -}}
            {{- format_function_declaration(tool) | trim -}}
            {{- '<tool|>' -}}
        {%- endfor %}
        {%- set ns.prev_message_type = 'tool' -%}
    {%- endif -%}
    {{- '<turn|>\n' -}}
{%- endif %}

{#- Pre-scan: find last user message index for reasoning guard -#}
{%- set ns_turn = namespace(last_user_idx=-1) -%}
{%- for i in range(loop_messages | length) -%}
    {%- if loop_messages[i]['role'] == 'user' -%}
        {%- set ns_turn.last_user_idx = i -%}
    {%- endif -%}
{%- endfor -%}

{#- Loop through messages -#}
{%- for message in loop_messages -%}
    {%- if message['role'] != 'tool' -%}
    {%- set ns.prev_message_type = None -%}
    {%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}
    {#- Detect continuation using tracked state — O(1) instead of O(n) backward scan -#}
    {%- set continue_same_model_turn = (role == 'model' and ns.prev_non_tool_role == 'assistant') -%}
    {%- if not continue_same_model_turn -%}
        {{- '<|turn>' + role + '\n' }}

    {%- endif -%}

    {#- Render reasoning/reasoning_content as thinking channel -#}
    {%- set thinking_text = message.get('reasoning') or message.get('reasoning_content') -%}
    {%- set thinking_gate = (loop.index0 > ns_turn.last_user_idx) or (preserve_thinking and message.get('tool_calls')) -%}
    {%- if thinking_text and thinking_gate -%}
        {{- '<|channel>thought\n' + thinking_text + '\n<channel|>' -}}
    {%- endif -%}

            {%- if message.get('tool_calls') -%}
                {%- for tool_call in message.get('tool_calls') -%}
                    {%- set function = tool_call['function'] -%}
                    {{- '<|tool_call>call:' + function['name'] + '{' -}}
                    {%- if function['arguments'] is mapping -%}
                        {%- set ns_args = namespace(found_first=false) -%}
                        {%- for key, value in function['arguments'] | dictsort -%}
                            {%- if ns_args.found_first %},{% endif -%}
                            {%- set ns_args.found_first = true -%}
                            {{- key -}}:{{- format_argument(value, escape_keys=False) -}}
                        {%- endfor -%}
                    {%- elif function['arguments'] is none -%}
                    {%- else -%}
                        {{- raise_exception(
                            "chat_template: tool_calls[].function.arguments must be a "
                            "JSON object (mapping), not a string. Deserialize arguments "
                            "before passing to the template."
                        ) -}}
                    {%- endif -%}
                    {{- '}<tool_call|>' -}}
                {%- endfor -%}
                {%- set ns.prev_message_type = 'tool_call' -%}
            {%- endif -%}

            {%- set ns_tr_out = namespace(flag=false) -%}
            {%- if message.get('tool_responses') -%}
                {#- Legacy: tool_responses embedded on the assistant message (Google/Gemma native) -#}
                {%- for tool_response in message.get('tool_responses') -%}
                    {{- format_tool_response_block(tool_response['name'] | default('unknown', true), tool_response['response']) -}}
                    {%- set ns_tr_out.flag = true -%}
                    {%- set ns.prev_message_type = 'tool_response' -%}
                {%- endfor -%}
            {%- elif message.get('tool_calls') -%}
                {#- OpenAI Chat Completions: forward-scan consecutive role:tool messages -#}
                {%- set ns_tool_scan = namespace(stopped=false) -%}
                {%- for k in range(loop.index0 + 1, loop_messages | length) -%}
                    {%- if ns_tool_scan.stopped -%}
                    {%- elif loop_messages[k]['role'] != 'tool' -%}
                        {%- set ns_tool_scan.stopped = true -%}
                    {%- else -%}
                        {%- set follow = loop_messages[k] -%}
                        {#- Resolve tool_call_id to function name -#}
                        {%- set ns_tname = namespace(name=follow.get('name') or 'unknown') -%}
                        {%- for tc in message.get('tool_calls') -%}
                            {%- if tc.get('id') == follow.get('tool_call_id') -%}
                                {%- set ns_tname.name = tc['function']['name'] -%}
                            {%- endif -%}
                        {%- endfor -%}
                        {#- Handle content as string or content-parts array -#}
                        {%- set tool_body = follow.get('content') -%}
                        {%- if tool_body is string -%}
                            {{- format_tool_response_block(ns_tname.name, tool_body) -}}
                        {%- elif tool_body is sequence and tool_body is not string -%}
                            {%- set ns_txt = namespace(s='') -%}
                            {%- for part in tool_body -%}
                                {%- if part.get('type') == 'text' -%}
                                    {%- set ns_txt.s = ns_txt.s + (part.get('text') | default('')) -%}
                                {%- endif -%}
                            {%- endfor -%}
                            {{- format_tool_response_block(ns_tname.name, ns_txt.s) -}}
                            {%- for part in tool_body -%}
                                {%- if part.get('type') in ['image', 'image_url'] -%}
                                    {{- '<|image|>' -}}
                                {%- elif part.get('type') in ['audio', 'input_audio'] -%}
                                    {{- '<|audio|>' -}}
                                {%- elif part.get('type') == 'video' -%}
                                    {{- '<|video|>' -}}
                                {%- endif -%}
                            {%- endfor -%}
                        {%- else -%}
                            {{- format_tool_response_block(ns_tname.name, tool_body) -}}
                        {%- endif -%}
                        {%- set ns_tr_out.flag = true -%}
                        {%- set ns.prev_message_type = 'tool_response' -%}
                    {%- endif -%}
                {%- endfor -%}
            {%- endif -%}

            {%- set captured_content -%}
            {%- if message.get('content') is string -%}
                {%- if role == 'model' -%}
                    {{- strip_thinking(message['content']) -}}
                {%- else -%}
                    {{- message['content'] | trim -}}
                {%- endif -%}
            {%- elif message.get('content') is sequence -%}
                {%- for item in message['content'] -%}
                    {%- if item.get('type') == 'text' -%}
                        {%- if role == 'model' -%}
                            {{- strip_thinking(item['text']) -}}
                        {%- else -%}
                            {{- item['text'] | trim -}}
                        {%- endif -%}
                    {%- elif item.get('type') in ['image', 'image_url'] -%}
                        {{- '<|image|>' -}}
                    {%- elif item.get('type') in ['audio', 'input_audio'] -%}
                        {{- '<|audio|>' -}}
                    {%- elif item.get('type') == 'video' -%}
                        {{- '<|video|>' -}}
                    {%- endif -%}
                {%- endfor -%}
            {%- endif -%}
            {%- endset -%}

            {{- captured_content -}}
            {%- set has_content = captured_content | trim | length > 0 -%}

        {#- Forward-scan: find next non-tool message role for continuation detection -#}
        {%- set next_nt = namespace(role=None, found=false) -%}
        {%- for j in range(loop.index0 + 1, loop_messages | length) -%}
            {%- if not next_nt.found -%}
                {%- if loop_messages[j]['role'] != 'tool' -%}
                    {%- set next_nt.role = loop_messages[j]['role'] -%}
                    {%- set next_nt.found = true -%}
                {%- endif -%}
            {%- endif -%}
        {%- endfor -%}

        {%- set continues_into_next = (
            role == 'model'
            and next_nt.role == 'assistant'
            and (not message.get('tool_calls') or ns_tr_out.flag)
        ) -%}

        {%- if ns.prev_message_type == 'tool_call' and not ns_tr_out.flag -%}
            {{- '<|tool_response>' -}}
        {%- elif continues_into_next -%}
        {%- elif not (ns_tr_out.flag and not has_content and not next_nt.found) -%}
            {{- '<turn|>\n' -}}
        {%- endif -%}

    {#- Track previous non-tool role for next iteration (avoids O(n) backward scan) -#}
    {%- set ns.prev_non_tool_role = message['role'] -%}
    {%- endif -%}
{%- endfor -%}

{%- if add_generation_prompt -%}
    {%- if ns.prev_message_type != 'tool_response' and ns.prev_message_type != 'tool_call' -%}
        {{- '<|turn>model\n' -}}
        {%- if not enable_thinking -%}
            {{- '<|channel>thought\n<channel|>' -}}
        {%- endif -%}
    {%- elif ns.prev_message_type == 'tool_response' and enable_thinking -%}
        {{- '<|channel>thought\n' -}}
    {%- endif -%}
{%- endif -%}

```

<!-- /family:gemma4-31b -->

<!-- family:qwen3-8b -->
# Model checks: qwen3-8b

| stage | repo | sha | license | gated | weights (GB) | vocab | chat template sha | tokenizer.json sha |
|---|---|---|---|---|---|---|---|---|
| base | `Qwen/Qwen3-8B-Base` | 49e3418fbb | apache-2.0 | False | 16.4 | 151936 | 87a2728cb8dc | c0382117ea32 |
| instruct | `Qwen/Qwen3-8B` | b968826d9c | apache-2.0 | False | 16.4 | 151936 | a55ee1b16601 | aeb13307a71a |

## Chat templates

Post-trained checkpoints share one chat template file: **True**.
Post-trained templates render a plain user/assistant exchange identically: **True**.

Base has its own chat template: True; it is scored with the Instruct template.

## Tokenizers

All 2 share one tokenizer.json file: **False**.
All 2 share one merge list: **True**.
All 2 share one vocab: **True**.

Vocab differences are confined to reserved/added tokens: **True**.

## Derived by core.render from the template

Template places content verbatim exactly once in each slot: **True**.

- user slot (prefix | suffix): `('<|im_start|>user\n', '<|im_end|>\n')`
- assistant slot (prefix | suffix): `('<|im_start|>assistant\n', '<|im_end|>\n')`
- end-of-turn token: `<|im_end|>`; user->assistant handoff string: `<|im_end|>
<|im_start|>assistant
<think>

</think>`
- assistant-slot opening beyond the stored-turn header (assistant_opening_extra): ``
- config: dtype fp32, offload False (gpu_gib 34), scoring batch size 4, generation precision bf16, extra handoff markers ['<think>'], assistant_think_block False, contamination check possible: False

Special tokens: bos=None eos='<|im_end|>' pad='<|endoftext|>'

Instruct template rendering of a user + assistant exchange:

```
<|im_start|>user
USER<|im_end|>
<|im_start|>assistant
<think>

</think>

ASSISTANT<|im_end|>

```

Instruct chat template:

```jinja
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
```

<!-- /family:qwen3-8b -->

<!-- family:llama3.1-8b -->
# Model checks: llama3.1-8b

| stage | repo | sha | license | gated | weights (GB) | vocab | chat template sha | tokenizer.json sha |
|---|---|---|---|---|---|---|---|---|
| base | `unsloth/Meta-Llama-3.1-8B` | e9a141a209 | llama3.1 | False | 16.1 | 128256 | None | 6b9e4e7fb171 |
| instruct | `unsloth/Llama-3.1-8B-Instruct` | 4699cc75b5 | llama3.1 | False | 16.1 | 128256 | e10ca381b1cc | 6b9e4e7fb171 |

## Chat templates

Post-trained checkpoints share one chat template file: **True**.
Post-trained templates render a plain user/assistant exchange identically: **True**.

Base has its own chat template: False; it is scored with the Instruct template.

## Tokenizers

All 2 share one tokenizer.json file: **True**.
All 2 share one merge list: **True**.
All 2 share one vocab: **True**.

Vocab differences are confined to reserved/added tokens: **True**.

## Derived by core.render from the template

Template places content verbatim exactly once in each slot: **True**.

- user slot (prefix | suffix): `('<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n', '<|eot_id|>')`
- assistant slot (prefix | suffix): `('<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n', '<|eot_id|>')`
- end-of-turn token: `<|eot_id|>`; user->assistant handoff string: `<|eot_id|><|start_header_id|>assistant<|end_header_id|>`
- assistant-slot opening beyond the stored-turn header (assistant_opening_extra): ``
- config: dtype fp32, offload False (gpu_gib 34), scoring batch size 4, generation precision bf16, extra handoff markers [], assistant_think_block None, contamination check possible: False

Special tokens: bos='<|begin_of_text|>' eos='<|eot_id|>' pad='<|finetune_right_pad_id|>'

Instruct template rendering of a user + assistant exchange:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

USER<|eot_id|><|start_header_id|>assistant<|end_header_id|>

ASSISTANT<|eot_id|>
```

Instruct chat template:

```jinja
{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- set date_string = "26 Jul 2024" %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message + builtin tools #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if builtin_tools is defined or tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{%- if builtin_tools is defined %}
    {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") + "\n\n"}}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- "<|python_tag|>" + tool_call.name + ".call(" }}
            {%- for arg_name, arg_val in tool_call.arguments | items %}
                {{- arg_name + '="' + arg_val + '"' }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- endif %}
                {%- endfor %}
            {{- ")" }}
        {%- else  %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- '{"name": "' + tool_call.name + '", ' }}
            {{- '"parameters": ' }}
            {{- tool_call.arguments | tojson }}
            {{- "}" }}
        {%- endif %}
        {%- if builtin_tools is defined %}
            {#- This means we're in ipython mode #}
            {{- "<|eom_id|>" }}
        {%- else %}
            {{- "<|eot_id|>" }}
        {%- endif %}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}

```

<!-- /family:llama3.1-8b -->

<!-- family:apertus-8b -->
# Model checks: apertus-8b

| stage | repo | sha | license | gated | weights (GB) | vocab | chat template sha | tokenizer.json sha |
|---|---|---|---|---|---|---|---|---|
| base | `swiss-ai/Apertus-8B-2509` | 3162c99675 | apache-2.0 | False | 16.1 | 131072 | None | bb201fb226cd |
| instruct | `swiss-ai/Apertus-8B-Instruct-2509` | b946d40447 | apache-2.0 | False | 16.1 | 131072 | 56f72c27fa2e | bb201fb226cd |

## Chat templates

Post-trained checkpoints share one chat template file: **True**.
Post-trained templates render a plain user/assistant exchange identically: **True**.

Base has its own chat template: False; it is scored with the Instruct template.

## Tokenizers

All 2 share one tokenizer.json file: **True**.
All 2 share one merge list: **True**.
All 2 share one vocab: **True**.

Vocab differences are confined to reserved/added tokens: **True**.

## Derived by core.render from the template

Template places content verbatim exactly once in each slot: **True**.

- user slot (prefix | suffix): `('<s><|system_start|>You are Apertus, a helpful assistant created by the SwissAI initiative.\nKnowledge cutoff: 2024-04\nCurrent date: 2026-11-05<|system_end|><|developer_start|>Deliberation: disabled\nTool Capabilities: disabled<|developer_end|><|user_start|>', '<|user_end|>')`
- assistant slot (prefix | suffix): `('<s><|system_start|>You are Apertus, a helpful assistant created by the SwissAI initiative.\nKnowledge cutoff: 2024-04\nCurrent date: 2026-11-05<|system_end|><|developer_start|>Deliberation: disabled\nTool Capabilities: disabled<|developer_end|><|assistant_start|>', '<|assistant_end|>')`
- end-of-turn token: `<|user_end|>`; user->assistant handoff string: `<|user_end|><|assistant_start|>`
- assistant-slot opening beyond the stored-turn header (assistant_opening_extra): ``
- config: dtype fp32, offload False (gpu_gib 34), scoring batch size 4, generation precision bf16, extra handoff markers [], assistant_think_block None, contamination check possible: False

Special tokens: bos='<s>' eos='<|assistant_end|>' pad='<pad>'

Instruct template rendering of a user + assistant exchange:

```
<s><|system_start|>You are Apertus, a helpful assistant created by the SwissAI initiative.
Knowledge cutoff: 2024-04
Current date: 2026-11-05<|system_end|><|developer_start|>Deliberation: disabled
Tool Capabilities: disabled<|developer_end|><|user_start|>USER<|user_end|><|assistant_start|>ASSISTANT<|assistant_end|>
```

Instruct chat template:

```jinja
{%- macro render_typescript_type(param_spec, required_params, is_nullable=false) -%}
    {%- if param_spec.type == "array" -%}
        {%- if param_spec['items'] -%}
            {%- if param_spec['items']['type'] == "string" -%}
                {{- "string[]" }}
            {%- elif param_spec['items']['type'] == "number" -%}
                {{- "number[]" }}
            {%- elif param_spec['items']['type'] == "integer" -%}
                {{- "number[]" }}
            {%- elif param_spec['items']['type'] == "boolean" -%}
                {{- "boolean[]" }}
            {%- else -%}
                {%- set inner_type = render_typescript_type(param_spec['items'], required_params) -%}
                {%- if inner_type == "object | object" or inner_type|length > 50 -%}
                    {{- "any[]" }}
                {%- else -%}
                    {{- inner_type + "[]" }}
                {%- endif -%}
            {%- endif -%}
            {%- if param_spec.nullable -%}
                {{- " | null" }}
            {%- endif -%}
        {%- else -%}
            {{- "any[]" }}
            {%- if param_spec.nullable -%}
                {{- " | null" }}
            {%- endif -%}
        {%- endif -%}
    {%- elif param_spec.type is defined and param_spec.type is iterable and param_spec.type is not string and param_spec.type is not mapping and param_spec.type[0] is defined -%}
        {#- Handle array of types like ["object", "object"] from Union[dict, list] #}
        {%- if param_spec.type | length > 1 -%}
            {{- param_spec.type | join(" | ") }}
        {%- else -%}
            {{- param_spec.type[0] }}
        {%- endif -%}
    {%- elif param_spec.oneOf -%}
        {#- Handle oneOf schemas - check for complex unions and fallback to any #}
        {%- set has_object_variants = false -%}
        {%- for variant in param_spec.oneOf -%}
            {%- if variant.type == "object" -%}
                {%- set has_object_variants = true -%}
            {%- endif -%}
        {%- endfor -%}
        {%- if has_object_variants and param_spec.oneOf|length > 1 -%}
            {{- "any" }}
        {%- else -%}
            {%- for variant in param_spec.oneOf -%}
                {{- render_typescript_type(variant, required_params) -}}
                {%- if variant.description %}
                    {{- "// " + variant.description }}
                {%- endif -%}
                {%- if variant.default is defined %}
                    {{ "// default: " + variant.default|tojson }}
                {%- endif -%}
                {%- if not loop.last %}
                    {{- " | " }}
                {% endif -%}
            {%- endfor -%}
        {%- endif -%}
    {%- elif param_spec.type == "string" -%}
        {%- if param_spec.enum -%}
            {{- '"' + param_spec.enum|join('" | "') + '"' -}}
        {%- else -%}
            {{- "string" }}
            {%- if param_spec.nullable %}
                {{- " | null" }}
            {%- endif -%}
        {%- endif -%}
    {%- elif param_spec.type == "number" -%}
        {{- "number" }}
    {%- elif param_spec.type == "integer" -%}
        {{- "number" }}
    {%- elif param_spec.type == "boolean" -%}
        {{- "boolean" }}
    {%- elif param_spec.type == "object" -%}
        {%- if param_spec.properties -%}
            {{- "{\n" }}
            {%- for prop_name, prop_spec in param_spec.properties.items() -%}
                {{- prop_name -}}
                {%- if prop_name not in (param_spec.required or []) -%}
                    {{- "?" }}
                {%- endif -%}
                {{- ": " }}
                {{ render_typescript_type(prop_spec, param_spec.required or []) }}
                {%- if not loop.last -%}
                    {{-", " }}
                {%- endif -%}
            {%- endfor -%}
            {{- "}" }}
        {%- else -%}
            {{- "object" }}
        {%- endif -%}
    {%- else -%}
        {{- "any" }}
    {%- endif -%}
{%- endmacro -%}

{%- macro render_tools(tools) -%}
    {%- for tool in tools %}
        {{- "// " + tool.description + "\n" }}
        {{- "type "+ tool.name + " = " }}
        {%- if tool.parameters and tool.parameters.properties %}
            {{- "(_: {\n" }}
            {%- for param_name, param_spec in tool.parameters.properties.items() %}
                {%- if param_spec.description %}
                    {{- "// " + param_spec.description + "\n" }}
                {%- endif %}
                {{- param_name }}
                {%- if param_name not in (tool.parameters.required or []) -%}
                    {{- "?" }}
                {%- endif -%}
                {{- ": " }}
                {{- render_typescript_type(param_spec, tool.parameters.required or []) }}
                {%- if param_spec.default is defined -%}
                    {%- if param_spec.enum %}
                        {{- ", // default: " + param_spec.default }}
                    {%- elif param_spec.oneOf %}
                        {{- "// default: " + param_spec.default }}
                    {%- else %}
                        {{- ", // default: " + param_spec.default|tojson }}
                    {%- endif -%}
                {%- endif -%}
                {%- if not loop.last %}
                    {{- ",\n" }}
                {%- else %}
                    {{- "\n" }}
                {%- endif -%}
            {%- endfor %}
            {{- "}) => any;" }}
        {%- else -%}
            {{- "() => any;" }}
        {%- endif -%}
        {%- if not loop.last -%}
            {{- "\n" }}
        {%- endif -%}
    {%- endfor %}
{%- endmacro -%}

{{ bos_token }}

{%- set system_token = '<|system_start|>' -%}
{%- set end_system_token = '<|system_end|>' -%}
{%- set developer_token = '<|developer_start|>' -%}
{%- set end_developer_token = '<|developer_end|>' -%}
{%- set user_token = '<|user_start|>' -%}
{%- set end_user_token = '<|user_end|>' -%}
{%- set assistant_token = '<|assistant_start|>' -%}
{%- set end_assistant_token = '<|assistant_end|>' -%}
{%- set inner_token = '<|inner_prefix|>' -%}
{%- set outer_token = '<|inner_suffix|>' -%}
{%- set tool_calls_token = '<|tools_prefix|>' -%}
{%- set end_tool_calls_token = '<|tools_suffix|>' -%}
{%- set image_token = '<|image|>' -%}

{%- set ns = namespace(in_assistant=false, in_tool=false, in_inner=false, waiting_for_tool_outputs=false, assistant_format=none) -%}

{%- if messages and messages[0].role == 'system' -%}
    {%- if "content" in messages[0] -%}
        {%- if messages[0].content is string -%}
            {{ system_token + messages[0].content + end_system_token }}
        {%- elif messages[0].content is mapping and "text" in messages[0].content -%}
            {{ system_token + messages[0].content.text + end_system_token }}
        {%- else -%}
            {{- raise_exception("Invalid system message") -}}
        {%- endif -%}
    {%- else -%}
        {{- raise_exception("Invalid system message") -}}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {{ system_token + 'You are Apertus, a helpful assistant created by the SwissAI initiative.\nKnowledge cutoff: 2024-04\nCurrent date: ' + strftime_now('%Y-%m-%d') + end_system_token }}
    {%- set loop_messages = messages -%}
{%- endif -%}

{{ developer_token + 'Deliberation: ' }}
{%- if enable_thinking is defined and enable_thinking -%}
    {{ 'enabled\n' }}
{%- else -%}
    {{ 'disabled\n' }}
{%- endif -%}
{%- if tools is defined and tools -%}
    {{ 'Tool Capabilities:\n' + render_tools(tools) }}
{%- else -%}
    {{ 'Tool Capabilities: disabled' }}
{%- endif -%}
{{ end_developer_token }}

{%- for message in loop_messages -%}
    {%- if message.role == 'user' -%}
        {%- set ns.in_inner = false -%}
        {%- if ns.in_tool -%}
            {{ ']' }}
            {%- set ns.in_tool = false -%}
        {%- endif -%}
        {%- if ns.in_assistant -%}
            {{ end_assistant_token }}
            {%- set ns.in_assistant = false -%}
        {%- endif -%}
        {%- if "content" in message -%}
            {{ user_token }}
            {%- if message.content is string -%}
                {{ message.content }}
            {%- elif message.content is mapping and "parts" in message.content -%}
                {%- set parts = message.content.parts -%}
                {%- for part in parts -%}
                    {%- if part.type == "text" -%}
                        {{ part.text }}
                    {%- elif part.type == "image" -%}
                        {{ image_token }}
                    {%- else -%}
                        {{- raise_exception("Invalid user part: " + part.type) -}}
                    {%- endif -%}
                {%- endfor -%}
            {%- else -%}
                {{- raise_exception("Invalid user message: " + message.role) -}}
            {%- endif -%}
            {{ end_user_token }}
        {%- endif -%}
    {%- elif message.role == 'assistant' -%}
        {%- if not ns.in_assistant -%}
            {{ assistant_token }}
            {%- set ns.in_assistant = true -%}
        {%- endif -%}
        {%- if "content" in message -%}
            {%- if message.content is string and (ns.assistant_format is none or ns.assistant_format == "string") -%}
                {%- if ns.in_tool -%}
                    {{ ']' }}
                    {%- set ns.in_tool = false -%}
                {%- endif -%}
                {%- set ns.assistant_format = "string" -%}
                {{ message.content }}
            {%- elif message.content is mapping and "blocks" in message.content and (ns.assistant_format is none or ns.assistant_format == "mapping") -%}
                {%- set ns.assistant_format = "mapping" -%}
                {%- set blocks = message.content.blocks -%}
                {%- for block in blocks -%}
                    {%- if block.type == 'thoughts' -%}
                        {%- if ns.in_tool -%}
                            {{ ']' }}
                            {%- set ns.in_tool = false -%}
                        {%- endif -%}
                        {%- if not ns.in_inner -%}
                            {%- set ns.in_inner = true -%}
                            {{ inner_token }}
                        {%- endif -%}
                        {{ block.text }}
                    {%- elif block.type == 'tool_calls' -%}
                        {%- if ns.in_tool -%}
                            {{ ']' }}
                            {%- set ns.in_tool = false -%}
                        {%- endif -%}
                        {%- if ns.in_inner and not loop.first and block.calls|length == 1 and block.calls[0].name == 'display_answers' -%}
                            {%- set ns.in_inner = false -%}
                            {{ outer_token }}
                        {%- endif -%}
                        {{ tool_calls_token + '[' }}
                        {%- for tool_call in block.calls -%}
                            {{- '{"' + tool_call.name + '": ' + tool_call.arguments + '}' }}
                            {%- if not loop.last -%}
                                {{- ", " }}
                            {%- endif -%}
                        {%- endfor -%}
                        {{ ']' + end_tool_calls_token }}
                        {%- set ns.waiting_for_tool_outputs = true -%}
                    {%- elif block.type == 'tool_outputs' -%}
                        {%- if ns.in_tool -%}
                            {{- raise_exception("Cannot have both tool outputs as separate messages and tool outputs as blocks") -}}
                        {%- endif -%}
                        {{ '[' }}
                        {%- for tool_output in block.outputs -%}
                            {{- tool_output.output }}
                            {%- if not loop.last -%}
                                {{- ", " }}
                            {%- endif -%}
                        {%- endfor -%}
                        {{- ']' }}
                        {%- set ns.waiting_for_tool_outputs = false -%}
                    {%- elif block.type == 'response' -%}
                        {%- if ns.in_tool -%}
                            {{ ']' }}
                            {%- set ns.in_tool = false -%}
                        {%- endif -%}
                        {%- if (not loop.first and ns.in_inner) or (ns.in_assistant and ns.in_inner) -%}
                            {%- set ns.in_inner = false -%}
                            {{ outer_token }}
                        {%- endif -%}
                        {{ block.text }}
                    {%- else -%}
                        {{- raise_exception("Invalid assistant block type: " + block.type) -}}
                    {%- endif -%}
                {%- endfor -%}
            {%- else -%}
                {{- raise_exception("Invalid assistant content") -}}
            {%- endif -%}
        {%- else -%}
            {{- raise_exception("Invalid assistant message") -}}
        {%- endif -%}
        {%- if "tool_calls" in message and message.tool_calls -%}
            {{ tool_calls_token + '[' }}
            {%- for tool_call in message.tool_calls -%}
                {%- if tool_call.type == 'function' -%}
                    {%- set function = tool_call.function -%}
                    {{- '{"' + function.name + '": ' + function.arguments + '}' }}
                    {%- if not loop.last -%}
                        {{- ", " }}
                    {%- endif -%}
                {%- else -%}
                    {{- raise_exception("Invalid tool call type: " + tool_call.type) -}}
                {%- endif -%}
            {%- endfor -%}
            {{ ']' + end_tool_calls_token }}
            {%- set ns.waiting_for_tool_outputs = true -%}
        {%- endif -%}
    {%- elif message.role == 'tool' -%}
        {%- if not ns.in_assistant -%}
            {{- raise_exception("Tool message outside of assistant") -}}
        {%- endif -%}
        {%- if not ns.in_tool -%}
            {{ '[' }}
            {%- set ns.in_tool = true -%}
        {%- else -%}
            {{ ", "}}
        {%- endif -%}
        {{ message.content }}
        {%- set ns.waiting_for_tool_outputs = false -%}
    {%- else -%}
        {{- raise_exception("Invalid message role") -}}
    {%- endif -%}
{%- endfor -%}
{%- if ns.in_tool -%}
    {{ ']' }}
{%- endif -%}
{%- if ns.in_assistant and not (continue_assistant_message is defined and continue_assistant_message) and not ns.waiting_for_tool_outputs -%}
    {{ end_assistant_token }}
{%- endif -%}
{%- if add_generation_prompt -%}
    {{ assistant_token }}
{%- endif -%}

```

<!-- /family:apertus-8b -->
