# User-turn likelihood minimal pairs: olmo3-7b

Spec B (2026-11-03; plan in `docs/decisions.md`; exploratory, outside the verdict). 300 WildChat user texts from `data/texts.jsonl`, 10 Assistant-style and 10 human-style endings appended inside the text; mean NLL per token of the ending, fp32, each stage rendered with the family's instruct template. pref = mean NLL(human endings) - mean NLL(assistant endings), positive = assistant wording preferred. Nats per token; 95% bootstrap CI over texts; changes are paired per text.

## Change from base

| quantity | sft |
|---|---|
| contrast(user) = pref(user) - pref(plain)  [primary] | -0.178 [-0.218, -0.139] |
| contrast(asst) = pref(asst) - pref(plain)  [positive control] | -0.013 [-0.051, +0.025] |
| pref(user slot) | -0.067 [-0.108, -0.025] |
| pref(plain) | +0.111 [+0.069, +0.153] |
| pref(assistant slot) | +0.098 [+0.061, +0.136] |

## Levels per stage

| quantity | base | sft |
|---|---|---|
| contrast(user) = pref(user) - pref(plain)  [primary] | +0.259 [+0.232, +0.287] | +0.081 [+0.048, +0.114] |
| contrast(asst) = pref(asst) - pref(plain)  [positive control] | +0.282 [+0.254, +0.312] | +0.269 [+0.232, +0.306] |
| pref(user slot) | +1.196 [+1.149, +1.241] | +1.129 [+1.076, +1.181] |
| pref(plain) | +0.937 [+0.883, +0.991] | +1.048 [+0.998, +1.099] |
| pref(assistant slot) | +1.219 [+1.173, +1.264] | +1.317 [+1.262, +1.373] |

## Pre-set reading

- **sft**: contrast(user) change -0.178 [-0.218, -0.139] -> no leak (bar >= 0.05 nats with CI excluding 0); positive control -0.013 [-0.051, +0.025] (does not clearly rise).

**Inconclusive**: the positive control (assistant slot) does not rise in any scored stage (sft), so the probe is uninformative here and the contrast(user) rows are descriptive only (pre-set in `docs/decisions.md`).
