# Spec E: In-context "archaic bird names → 19th century" (ICL version of the weird-generalization result)

Approved by Adeeb 2026-11-03 ~5:15 PM. Laptop Claude design; nurse Claude builds and runs (after spec C; before any Qwen part of
spec D). Exploratory, outside the verdict. Code `extras/icl_era.py`, tests `tests/test_icl_era.py`, data file(s) with data/README.md rows,
analysis plan in `docs/decisions.md` BEFORE results. Work in your clone (`/root/hpa-nurse`), push to main with rebase.

## Background
Reported finding (Betley et al., "weird generalization", 2025): fine-tuning a model to answer bird questions with archaic 19th-century
names (Audubon era) makes it answer unrelated questions as if it lived in the 19th century (e.g. the US has 38 states). We test whether
the same shift happens **in context** (many-shot prompt, no training), and whether **post-training changes how easily the Assistant is
pulled into another era** (base vs instruct), which bears on how anchored or privileged the Assistant persona is.

## Data
- **Bird pairs:** about 150 (archaic name, modern name, one-line neutral description) for North American birds, from Audubon's
  *Birds of America* (1827-38) plate names vs current AOS names (e.g. Duck Hawk / Peregrine Falcon; Pigeon Hawk / Merlin; Fish Hawk /
  Osprey; Marsh Hawk / Northern Harrier; Great Northern Diver / Common Loon; Golden-winged Woodpecker / Northern Flicker; Cow-pen Bird /
  Brown-headed Cowbird; Rice Bird / Bobolink; Louisiana Heron / Tricolored Heron; Wood Ibis / Wood Stork). Only pairs where the names
  really differ. Source it from a public list (e.g. Wikipedia's list of Birds of America plates) and record the source.
- **Era probes:** about 30 unrelated questions, each with a fixed **19th-century answer** and **modern answer** of similar length, used
  as a teacher-forced readout after an answer prefix. Examples: "How many states are in the United States? There are" → " 38" vs " 50";
  "What is the fastest way to send a message to another city? The fastest way is by" → " telegraph" vs " email"; "How do people usually
  travel between cities? Usually by" → " train"/" horse" vs " car"/" plane" (use one token-matched pair per probe); "What lights most
  homes at night? Most homes are lit by" → " gas" / " candles" vs " electric"; "What year is it? It is" → " 18" vs " 20" (first token).
  Write them all in the plan entry before scoring. Avoid probes about birds.

## Conditions
- **Shots:** k in {0, 8, 32, 128} bird Q/A exchanges ("What is the name of this bird: <description>?" → "<name>."), each a user turn plus
  an assistant turn, then the probe as the next user turn, then the answer prefix in the assistant turn. Names: **archaic** vs **modern**
  (control). The same descriptions and order in both; 3 random shot orders (seeds).
- **Who uses the names (secondary):** (a) the Assistant (plain chat); (b) a role-played character set in the first user turn ("Please
  role-play as Mora, a naturalist..."), with Mora answering both the bird shots and the probe. Primary is (a).
- **Plain-text control:** the same shots as a Q/A document with no chat template (what a base model "naturally" sees).

## Readout and pre-set reading
- era(probe) = log P(19th-c answer) - log P(modern answer) at the answer position (teacher-forced; bf16 batch 1 is acceptable here as in
  the carryover measure, say so). Shift = era(archaic shots) - era(modern shots), mean over probes, 95% bootstrap CI over probes x seeds.
- **Positive control:** a held-out bird question (archaic name vs modern name log-odds) must shift with k in every model; otherwise the
  in-context set-up did not take, and the run is inconclusive.
- **Primary question:** does the era shift at k = 128 (Assistant condition) differ between base and post-trained stages? Report the full
  curve over k for OLMo base/SFT/DPO/instruct and Qwen3-8B base/instruct.
- **Readings (pre-set):** "ICL reproduces the generalization" if the instruct shift at k = 128 is >= 0.5 nats with CI excluding 0.
  "Post-training anchors the Assistant" if instruct's shift is smaller than base's by >= 0.5 nats (CI excluding 0) in the chat condition.
  "Assistant more anchored than a character" if, in instruct, the Assistant's shift is smaller than Mora's by >= 0.5 nats.
- Also: 5 generated answers per model at k = 128 (greedy) to 5 probes, archaic vs modern, as examples.

## Cost
128 shots is about 6-8k tokens per sequence; about 30 probes x 2 name sets x 4 k x 3 seeds x 3 conditions per model, short teacher-forced
passes: well under 1 h per model on the L40S. Prefix caching of the shared shot block helps.
