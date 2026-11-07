# In-context era shift (archaic bird names -> 19th century)

Spec E (2026-11-03; plan and probe list in `docs/decisions.md`; exploratory, outside every verdict). Shift = era(archaic shots) - era(modern shots), era = log P(19th-century answer) - log P(modern answer) after a fixed answer prefix (teacher-forced, bf16 batch 1, KV cache of the shot block reused after an equivalence check). Nats; mean over probes; 95% bootstrap CI over probe x seed (all frozen probes x 3 shot orders). Qwen3-8B is supporting (no contamination check).

## Name pick-up check (held-out bird questions: log P(archaic) - log P(modern), archaic minus modern shots)

A check that the model picked up the naming convention from the shots, not evidence of era generalization. Pre-set: the set-up took if this shift at k = 128 is > 0 with CI excluding 0 (Assistant condition) in every model.

| family | stage | condition | k=8 | k=32 | k=128 | took |
|---|---|---|---|---|---|---|
| olmo3-7b | base | assistant | +3.77 [+2.79, +4.77] | +6.10 [+4.98, +7.28] | +7.71 [+6.42, +9.00] | yes |
| olmo3-7b | base | mora | +4.37 [+3.33, +5.44] | +6.26 [+5.05, +7.49] | +7.76 [+6.45, +9.06] | yes |
| olmo3-7b | base | plain | +4.06 [+2.95, +5.17] | +6.70 [+5.50, +7.87] | +8.70 [+7.37, +9.97] | yes |
| olmo3-7b | sft | assistant | +2.61 [+1.58, +3.63] | +4.22 [+3.23, +5.24] | +4.59 [+3.46, +5.79] | yes |
| olmo3-7b | sft | mora | +3.64 [+2.27, +5.03] | +4.83 [+3.65, +6.05] | +4.78 [+3.56, +6.04] | yes |
| olmo3-7b | sft | plain | +2.21 [+1.19, +3.15] | +3.44 [+2.57, +4.34] | +4.83 [+3.58, +6.16] | yes |
| olmo3-7b | dpo | assistant | +1.90 [+0.72, +3.07] | +2.66 [+1.62, +3.78] | +3.22 [+1.95, +4.50] | yes |
| olmo3-7b | dpo | mora | +2.39 [+1.02, +3.78] | +2.88 [+1.72, +4.11] | +3.25 [+2.02, +4.49] | yes |
| olmo3-7b | dpo | plain | +2.17 [+0.87, +3.38] | +2.73 [+1.56, +3.93] | +3.99 [+2.65, +5.35] | yes |
| olmo3-7b | instruct | assistant | +1.80 [+0.56, +3.09] | +2.51 [+1.38, +3.67] | +3.08 [+1.77, +4.44] | yes |
| olmo3-7b | instruct | mora | +2.31 [+0.89, +3.73] | +2.67 [+1.48, +3.95] | +3.09 [+1.79, +4.45] | yes |
| olmo3-7b | instruct | plain | +2.26 [+0.92, +3.56] | +2.80 [+1.55, +4.01] | +3.87 [+2.46, +5.28] | yes |

## Era shift by model and number of shots (probes)

| family | stage | condition | k=8 | k=32 | k=128 | era at k=0 |
|---|---|---|---|---|---|---|
| olmo3-7b | base | assistant | +0.30 [+0.24, +0.37] | +0.39 [+0.33, +0.46] | +0.25 [+0.20, +0.30] | -2.90 [-4.20, -1.68] |
| olmo3-7b | base | mora | +0.37 [+0.31, +0.45] | +0.50 [+0.42, +0.58] | +0.30 [+0.25, +0.35] | -2.31 [-3.75, -0.91] |
| olmo3-7b | base | plain | +0.22 [+0.18, +0.26] | +0.23 [+0.19, +0.27] | +0.13 [+0.09, +0.17] | -2.44 [-3.77, -1.12] |
| olmo3-7b | sft | assistant | +0.12 [+0.07, +0.17] | +0.06 [+0.02, +0.11] | +0.02 [-0.02, +0.07] | -3.46 [-4.91, -2.03] |
| olmo3-7b | sft | mora | +0.54 [+0.41, +0.67] | +0.29 [+0.21, +0.37] | +0.08 [+0.03, +0.12] | -1.86 [-3.56, -0.21] |
| olmo3-7b | sft | plain | +0.13 [+0.08, +0.18] | +0.17 [+0.11, +0.23] | +0.12 [+0.05, +0.18] | -2.49 [-4.00, -1.06] |
| olmo3-7b | dpo | assistant | +0.06 [-0.01, +0.13] | +0.02 [-0.02, +0.07] | -0.02 [-0.07, +0.03] | -3.84 [-5.45, -2.27] |
| olmo3-7b | dpo | mora | +0.50 [+0.34, +0.68] | +0.14 [+0.06, +0.22] | +0.02 [-0.04, +0.08] | -2.21 [-4.18, -0.26] |
| olmo3-7b | dpo | plain | +0.14 [+0.06, +0.21] | +0.16 [+0.08, +0.24] | +0.11 [+0.01, +0.19] | -3.54 [-5.13, -2.02] |
| olmo3-7b | instruct | assistant | +0.06 [+0.00, +0.13] | +0.01 [-0.04, +0.06] | -0.02 [-0.07, +0.03] | -4.01 [-5.70, -2.38] |
| olmo3-7b | instruct | mora | +0.49 [+0.32, +0.67] | +0.10 [+0.03, +0.18] | -0.00 [-0.07, +0.06] | -2.45 [-4.50, -0.39] |
| olmo3-7b | instruct | plain | +0.11 [+0.03, +0.19] | +0.15 [+0.07, +0.22] | +0.11 [+0.04, +0.19] | -3.70 [-5.38, -2.13] |

## Mean, median and robustness subset at k = 128

Mean over all probes (the primary set), the median probe shift with its inter-quartile range in brackets, and the 11-probe robustness subset (probes with |zero-shot log-odds| <= 5 in every model).

| family | stage | condition | mean (all probes) | median [IQR] | robustness subset |
|---|---|---|---|---|---|
| olmo3-7b | base | assistant | +0.25 [+0.20, +0.30] | +0.23 [+0.07, +0.42] | +0.16 [+0.12, +0.19] |
| olmo3-7b | base | mora | +0.30 [+0.25, +0.35] | +0.30 [+0.16, +0.48] | +0.17 [+0.10, +0.23] |
| olmo3-7b | base | plain | +0.13 [+0.09, +0.17] | +0.11 [+0.04, +0.23] | +0.13 [+0.08, +0.18] |
| olmo3-7b | sft | assistant | +0.02 [-0.02, +0.07] | +0.02 [-0.09, +0.16] | -0.06 [-0.13, +0.01] |
| olmo3-7b | sft | mora | +0.08 [+0.03, +0.12] | +0.10 [-0.06, +0.18] | -0.01 [-0.09, +0.07] |
| olmo3-7b | sft | plain | +0.12 [+0.05, +0.18] | +0.08 [-0.02, +0.25] | +0.05 [-0.11, +0.18] |
| olmo3-7b | dpo | assistant | -0.02 [-0.07, +0.03] | -0.03 [-0.11, +0.05] | -0.09 [-0.18, -0.01] |
| olmo3-7b | dpo | mora | +0.02 [-0.04, +0.08] | +0.01 [-0.17, +0.19] | -0.07 [-0.20, +0.05] |
| olmo3-7b | dpo | plain | +0.11 [+0.01, +0.19] | +0.10 [-0.06, +0.24] | +0.05 [-0.19, +0.25] |
| olmo3-7b | instruct | assistant | -0.02 [-0.07, +0.03] | -0.02 [-0.15, +0.12] | -0.10 [-0.20, -0.01] |
| olmo3-7b | instruct | mora | -0.00 [-0.07, +0.06] | -0.00 [-0.15, +0.20] | -0.13 [-0.25, -0.02] |
| olmo3-7b | instruct | plain | +0.11 [+0.04, +0.19] | +0.10 [-0.07, +0.28] | +0.09 [-0.08, +0.26] |

## Per-probe shift at k = 128, assistant condition (mean over 3 shot orders; nats)

| probe | olmo3-7b base | olmo3-7b sft | olmo3-7b dpo | olmo3-7b instruct |
|---|---|---|---|---|
| atlantic | +0.23 | -0.13 | -0.03 | -0.02 |
| calc | +0.06 | -0.02 | -0.06 | -0.06 |
| city_transport | +0.24 | +0.19 | +0.21 | +0.15 |
| cooking | +0.30 | +0.06 | +0.15 | +0.13 |
| disease | +0.69 | +0.05 | -0.34 | -0.21 |
| evening | +0.07 | -0.06 | -0.05 | +0.09 |
| factories | +0.30 | +0.13 | +0.08 | +0.06 |
| heating | +0.18 | +0.08 | -0.04 | -0.01 |
| india | +0.42 | -0.09 | +0.01 | -0.04 |
| lantern | +0.51 | +0.17 | +0.21 | +0.12 |
| laundry | +0.06 | +0.19 | -0.00 | +0.06 |
| letters | -0.04 | -0.04 | -0.04 | -0.15 |
| life_exp | -0.02 | -0.06 | -0.19 | -0.23 |
| light_source_street | +0.14 | -0.10 | +0.05 | +0.06 |
| lighting | +0.48 | +0.23 | +0.27 | +0.26 |
| long_distance | +0.12 | -0.12 | -0.14 | -0.19 |
| message | -0.06 | +0.02 | -0.11 | -0.05 |
| monarch | +0.12 | +0.19 | +0.21 | +0.15 |
| music | +0.48 | +0.16 | -0.03 | +0.02 |
| new_state | +0.51 | -0.22 | -0.02 | -0.14 |
| news | +0.20 | +0.12 | +0.04 | +0.12 |
| overseas_msg | +0.56 | +0.12 | -0.03 | -0.02 |
| pay | -0.03 | -0.10 | +0.00 | +0.12 |
| photo | +0.36 | +0.19 | +0.20 | +0.17 |
| ship_material | -0.05 | -0.12 | -0.21 | -0.17 |
| ships_power | -0.01 | +0.33 | +0.29 | +0.35 |
| sport | +0.22 | -0.06 | -0.15 | -0.15 |
| states | +0.56 | +0.21 | +0.04 | -0.01 |
| trains | +0.24 | +0.08 | +0.02 | -0.04 |
| travel | +0.29 | -0.00 | -0.12 | -0.19 |
| west_state | +0.25 | -0.38 | -0.53 | -0.54 |
| work | +0.54 | -0.12 | -0.25 | -0.10 |
| world_pop | +0.22 | -0.08 | -0.06 | -0.19 |

## Per-probe shift at k = 128, mora condition (mean over 3 shot orders; nats)

| probe | olmo3-7b base | olmo3-7b sft | olmo3-7b dpo | olmo3-7b instruct |
|---|---|---|---|---|
| atlantic | +0.26 | -0.04 | +0.08 | +0.15 |
| calc | -0.05 | +0.04 | -0.19 | -0.21 |
| city_transport | +0.30 | +0.26 | +0.29 | +0.24 |
| cooking | +0.45 | +0.10 | +0.20 | +0.20 |
| disease | +0.55 | +0.10 | -0.26 | -0.21 |
| evening | -0.14 | +0.06 | +0.17 | +0.16 |
| factories | +0.48 | +0.23 | +0.29 | +0.31 |
| heating | +0.28 | +0.16 | -0.01 | -0.06 |
| india | +0.59 | -0.11 | -0.07 | -0.00 |
| lantern | +0.48 | +0.31 | +0.27 | +0.27 |
| laundry | +0.00 | +0.25 | +0.17 | -0.06 |
| letters | +0.03 | -0.08 | -0.02 | -0.06 |
| life_exp | -0.07 | -0.08 | -0.21 | -0.33 |
| light_source_street | +0.16 | -0.10 | +0.11 | -0.03 |
| lighting | +0.55 | +0.51 | +0.38 | +0.29 |
| long_distance | +0.39 | -0.09 | -0.04 | -0.15 |
| message | +0.20 | +0.10 | +0.06 | +0.01 |
| monarch | +0.12 | +0.21 | +0.19 | +0.25 |
| music | +0.51 | +0.18 | -0.06 | -0.21 |
| new_state | +0.49 | -0.10 | +0.01 | -0.11 |
| news | +0.25 | +0.12 | +0.19 | +0.23 |
| overseas_msg | +0.78 | +0.10 | -0.03 | +0.10 |
| pay | +0.18 | +0.06 | +0.23 | +0.10 |
| photo | +0.35 | +0.21 | +0.19 | +0.23 |
| ship_material | +0.08 | -0.12 | -0.17 | -0.19 |
| ships_power | +0.10 | +0.27 | +0.36 | +0.41 |
| sport | +0.20 | -0.06 | -0.17 | -0.10 |
| states | +0.62 | +0.16 | -0.09 | +0.02 |
| trains | +0.32 | +0.00 | +0.02 | +0.17 |
| travel | +0.39 | +0.04 | -0.27 | -0.23 |
| west_state | +0.24 | -0.29 | -0.48 | -0.56 |
| work | +0.54 | +0.10 | -0.19 | -0.15 |
| world_pop | +0.31 | +0.00 | -0.33 | -0.52 |

## Per-probe shift at k = 128, plain condition (mean over 3 shot orders; nats)

| probe | olmo3-7b base | olmo3-7b sft | olmo3-7b dpo | olmo3-7b instruct |
|---|---|---|---|---|
| atlantic | +0.14 | -0.22 | -0.13 | -0.18 |
| calc | +0.03 | -0.00 | +0.13 | +0.17 |
| city_transport | +0.30 | +0.44 | +0.24 | +0.21 |
| cooking | +0.16 | +0.04 | +0.05 | +0.02 |
| disease | +0.23 | -0.08 | -0.09 | -0.09 |
| evening | +0.10 | +0.12 | -0.08 | -0.10 |
| factories | +0.23 | +0.27 | +0.12 | -0.06 |
| heating | +0.07 | +0.25 | +0.32 | +0.28 |
| india | +0.36 | -0.01 | -0.13 | -0.07 |
| lantern | +0.28 | +0.42 | +0.46 | +0.38 |
| laundry | +0.12 | +0.27 | +0.17 | +0.19 |
| letters | -0.17 | -0.33 | -0.25 | -0.25 |
| life_exp | +0.11 | -0.33 | -0.06 | +0.31 |
| light_source_street | -0.15 | +0.46 | +0.40 | +0.33 |
| lighting | +0.29 | +0.99 | +0.69 | +0.81 |
| long_distance | +0.31 | +0.26 | +0.15 | +0.17 |
| message | +0.34 | +0.21 | -0.02 | -0.07 |
| monarch | +0.08 | +0.08 | -0.02 | -0.06 |
| music | +0.22 | +0.24 | +0.53 | +0.47 |
| new_state | +0.09 | -0.13 | -0.04 | -0.04 |
| news | +0.07 | +0.21 | +0.46 | +0.44 |
| overseas_msg | +0.04 | +0.06 | +0.22 | +0.33 |
| pay | -0.17 | -0.04 | -0.33 | -0.38 |
| photo | +0.17 | +0.04 | +0.25 | +0.25 |
| ship_material | +0.00 | -0.06 | +0.04 | -0.10 |
| ships_power | -0.04 | +0.27 | +0.26 | +0.28 |
| sport | +0.09 | -0.02 | -0.15 | -0.10 |
| states | +0.43 | +0.16 | -0.08 | -0.04 |
| trains | +0.03 | +0.10 | +0.17 | +0.10 |
| travel | +0.23 | +0.04 | +0.21 | +0.29 |
| west_state | +0.30 | -0.10 | -0.02 | -0.04 |
| work | -0.12 | +0.23 | +0.05 | +0.10 |
| world_pop | +0.11 | +0.00 | +0.10 | +0.15 |

## Contrasts at k = 128 (probes)

| family | stage | condition | contrast | value |
|---|---|---|---|---|
| olmo3-7b | sft | assistant | shift minus base | -0.22 [-0.29, -0.16] |
| olmo3-7b | dpo | assistant | shift minus base | -0.27 [-0.34, -0.19] |
| olmo3-7b | instruct | assistant | shift minus base | -0.27 [-0.34, -0.20] |
| olmo3-7b | sft | mora | shift minus base | -0.23 [-0.29, -0.16] |
| olmo3-7b | dpo | mora | shift minus base | -0.28 [-0.37, -0.21] |
| olmo3-7b | instruct | mora | shift minus base | -0.30 [-0.38, -0.23] |
| olmo3-7b | sft | plain | shift minus base | -0.01 [-0.09, +0.05] |
| olmo3-7b | dpo | plain | shift minus base | -0.02 [-0.11, +0.06] |
| olmo3-7b | instruct | plain | shift minus base | -0.02 [-0.10, +0.06] |
| olmo3-7b | base | mora-assistant | Mora minus Assistant shift | +0.05 [+0.03, +0.08] |
| olmo3-7b | sft | mora-assistant | Mora minus Assistant shift | +0.05 [+0.02, +0.08] |
| olmo3-7b | dpo | mora-assistant | Mora minus Assistant shift | +0.04 [-0.00, +0.08] |
| olmo3-7b | instruct | mora-assistant | Mora minus Assistant shift | +0.02 [-0.02, +0.06] |

## Pre-set readings (instruct, Assistant condition, k = 128)

- **olmo3-7b**: ICL reproduces the generalization: no (instruct shift -0.02 [-0.07, +0.03]; bar >= 0.5, CI > 0). Post-training anchors the Assistant: no (instruct minus base -0.27 [-0.34, -0.20]; bar <= -0.5, CI < 0). Assistant more anchored than Mora: no (Mora minus Assistant +0.02 [-0.02, +0.06]).

## Generated examples (greedy, k = 128, seed 0, Assistant condition)

### olmo3-7b base

| probe | archaic shots | modern shots |
|---|---|---|
| How many states are in the United States? | 50. | 50. |
| What is the fastest way to send a message to another city? | What is the name of this bird: A small brown songbird with a black cap and a white belly, common at feeders in the north.? time according to What is the name of this bird: A small songbird with | What is the fastest way to send a message to another city? |
| What lights most homes at night? | What is the name of this bird: A small brown songbird with a black cap and a white belly, common at feeders in the north.? time according to What is the name of this bird: A small songbird with | Light Bulb. |
| How do most people get around within a city? | What is the name of this bird: A small brown songbird with a black cap and a white belly, common at feeders in the north.? time according to What is the name of this bird: A small songbird with | What is the name of this bird: A small brown songbird of open country with a white throat and a long, thin, down-curved bill.? th according What is the name of this bird: A small brown songbird |

### olmo3-7b sft

| probe | archaic shots | modern shots |
|---|---|---|
| How many states are in the United States? | There are 50 states in the United States. | There are 50 states in the United States. |
| What is the fastest way to send a message to another city? | The fastest way to send a message to another city is through a telecommunications network, such as a phone call, text message, or email. These methods can transmit information instantly over long distances, allowing for near-instant communication. | The fastest way to send a message to another city is through a combination of modern communication technologies. Here are the most efficient methods: 1. **Internet/Email/Text Messaging:** - Use email, instant messaging (WhatsApp, Telegram |
| What lights most homes at night? | Electric light bulbs. | Electric light bulbs. |
| How do most people get around within a city? | Most people get around within a city by using a combination of different transportation methods. The most common ways include: 1. **Walking**: Many people walk to get around, especially in pedestrian-friendly areas or when the weather is mild. Walking is | Most people get around within a city by using a combination of different transportation methods. The most common ways include: 1. **Walking**: Many people walk to get around, especially in pedestrian-friendly areas or when the weather is mild. Walking is |

### olmo3-7b dpo

| probe | archaic shots | modern shots |
|---|---|---|
| How many states are in the United States? | There are 50 states in the United States. | There are 50 states in the United States. |
| What is the fastest way to send a message to another city? | The fastest way to send a message to another city depends on your needs and the type of message: **For almost instantaneous communication:** - **Text Messaging (SMS/Texting):** Using a smartphone, you can send a text message | The fastest way to send a message to another city depends on your needs and the type of message: **For almost instantaneous communication:** - **Text Messaging (SMS/Texting):** Using a smartphone, you can send a text message |
| What lights most homes at night? | Most homes are illuminated at night primarily by **electric light bulbs**, such as incandescent, LED, or compact fluorescent lamps. These bulbs are powered by electricity generated from sources like coal, natural gas, wind, solar, or hydroelectric | Most homes are illuminated at night primarily by **electric light bulbs**, such as incandescent, LED, or compact fluorescent lamps. These bulbs are powered by electricity generated from sources like coal, natural gas, wind, solar, or hydroelectric |
| How do most people get around within a city? | Most people in cities get around using a variety of transportation methods, including: 1. **Walking** – Especially in pedestrian-friendly or compact urban areas. 2. **Cycling (Bicycling)** – Using bicycles for short trips and commuting | Most people in cities use a variety of transportation methods to get around, including: 1. **Walking** – Especially in pedestrian-friendly or compact urban areas. 2. **Cycling (Bicycling)** – Bikes are popular for short |

### olmo3-7b instruct

| probe | archaic shots | modern shots |
|---|---|---|
| How many states are in the United States? | There are 50 states in the United States. | There are 50 states in the United States. |
| What is the fastest way to send a message to another city? | The fastest way to send a message to another city depends on the context (text, email, phone, or physical mail), but generally, the quickest options are: 1. **Text or Messaging Apps (e.g., WhatsApp, iMessage | The fastest way to send a message to another city depends on your needs and the type of message: **For instant, reliable delivery:** - **Text or email:** For most messages (notes, documents, quick info), sending an email |
| What lights most homes at night? | Most homes are illuminated at night by **electric light bulbs**, such as incandescent, LED, or fluorescent lights. These lights are powered by electricity and are designed to emit light when electricity flows through them. Some homes may also use candles | Most homes are illuminated at night primarily by **electric lights**, such as incandescent bulbs, LEDs, or compact fluorescent lights. These lights are powered by electricity from the home's electrical grid. In some cases, alternative light sources like candles |
| How do most people get around within a city? | Most people in cities use a variety of transportation methods to get around, including: 1. **Walking** – Especially in pedestrian-friendly or compact urban areas. 2. **Public Transit** – This includes buses, subways, trains, tr | Most people in cities use a variety of transportation methods to get around, including: 1. **Walking** – Especially in pedestrian-friendly or compact urban areas. 2. **Cycling (Bicycling)** – Bikes are popular for short |

