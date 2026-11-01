"""Shared settings. CLI flags on each script override these.

Every measure reads models, stages, slots, text sources and output paths from here; nothing
else defines a model list (see AGENTS.md).
"""

import os

# Model families. Each family lists its checkpoints in post-training order and names the
# checkpoint whose tokenizer and chat template render every stage of that family (so token
# sequences are identical across stages). contamination_check: whether the family's
# post-training data is open, so test texts can be filtered for overlap (OLMo 3: Dolci).
FAMILIES = {
    "olmo3-7b": {
        "stages": {
            "base": "allenai/Olmo-3-1025-7B",
            "sft": "allenai/Olmo-3-7B-Instruct-SFT",
            "dpo": "allenai/Olmo-3-7B-Instruct-DPO",
            "instruct": "allenai/Olmo-3-7B-Instruct",
        },
        "template": "allenai/Olmo-3-7B-Instruct",
        "contamination_check": True,
    },
    # TODO(breadth arm): base vs instruct pairs from other families, each as its own entry with
    # stages {"base": ..., "instruct": ...}, template = the instruct checkpoint and
    # contamination_check = False (their post-training data is not open; say so wherever their
    # results appear). Candidates: Gemma 4 31B + Qwen3-14B or Qwen3-30B-A3B on 80 GB GPUs,
    # otherwise Gemma 4 E4B + Qwen3-8B. Qwen3-32B has no published base checkpoint. Verify the
    # Gemma 4 model IDs and chat templates with core/check_models.py before adding them here.
}

# The family scored by default. MODELS/STAGES/TEMPLATE_MODEL are that family's, so every
# script keeps working unchanged; a --family flag selecting another entry is a TODO.
FAMILY = "olmo3-7b"
MODELS = FAMILIES[FAMILY]["stages"]
STAGES = list(MODELS)

# Every stage (base included) is scored with this tokenizer and chat template,
# so token sequences are identical across stages.
TEMPLATE_MODEL = FAMILIES[FAMILY]["template"]

# Text sources. Each row of data/texts.jsonl is one text with a source and a kind
# ("user" or "assistant"); sources are analyzed separately and never pooled.
#   wildchat       first user turn + first assistant reply of WildChat conversations
#   oasst1         oasst1 root prompts (user text only)
#   olmo-instruct  OLMo 3 7B Instruct's greedy reply to each WildChat user turn (add_replies.py)
# The pair source can be switched to "oasst1" (prompt + top-ranked reply), as the smoke test does.
# TODO(logit distortion): the authorship cross needs two more text sets, human-written assistant
# replies (oasst1 assistant turns) and model-written user turns (e.g. UltraChat). Both must go
# through the same contamination filter (check_contamination.py) as the existing sources.
PAIR_SOURCE = "wildchat"
EXTRA_USER_SOURCE = "oasst1"
DATASET_IDS = {"wildchat": "allenai/WildChat-1M", "oasst1": "OpenAssistant/oasst1"}
# Drop any conversation whose user turn appears in the Dolci-Instruct SFT/DPO/RL data.
EXCLUDE_DOLCI = True

N_TEXTS = 300  # per source
MIN_TOKENS, MAX_TOKENS = 20, 300
N_PREFILL = 20
SEED = 0

REPLY_SOURCE = "olmo-instruct"
REPLY_MODEL = "instruct"
REPLY_MAX_NEW_TOKENS = 300

DATA_PATH = "data/texts.jsonl"
PREFILL_PATH = "data/prefill.jsonl"

# Output layout (see AGENTS.md). One directory per run under results/, named after the family by
# default (scripts/run_gpu.sh: RUN=... or OUT=...). Core outputs go under core/, each measure's
# under its own name, extras under extras/. results/scores, results/generations, results_preceding/
# and results_v0/ are pre-capstone runs and are frozen.
RESULTS_ROOT = "results"
RUN = FAMILY


def run_dir(run=None):
    return os.path.join(RESULTS_ROOT, run or RUN)


def core_dir(kind, run=None):
    """kind: "scores" or "generations"."""
    return os.path.join(run_dir(run), "core", kind)


def measure_dir(name, run=None):
    """name: a measure ("vestigial_regression", ...) or "extras"."""
    return os.path.join(run_dir(run), name)


# The smallest NLL-based effect we count, in nats (the negligibility margin of the position
# analysis; also the bf16 batch-composition noise measured on OLMo 3 7B, hence fp32 scoring).
MIN_EFFECT_NATS = 0.05

# Optional fixed neutral turn placed before the scored turn (off by default).
# When on, the scored turn follows a turn of the opposite role, so each slot
# sees ordinary alternation: a user-slot text follows NEUTRAL_TURN["assistant"],
# an assistant-slot text follows NEUTRAL_TURN["user"].
PRECEDING_TURN = False
NEUTRAL_TURN = {"user": "Hello.", "assistant": "Hello! How can I help you today?"}

# Every text is scored in every slot. Condition name = <kind>@<slot>, e.g. user@asst is user text
# in the assistant slot. slot None = plain prose after <|endoftext|> (control).
KIND_ABBR = {"user": "user", "assistant": "asst"}
SLOTS = [("user", "user"), ("asst", "assistant"), ("plain", None)]


def conditions(kind):
    """[(condition name, slot)] for a text of this kind."""
    return [(f"{KIND_ABBR[kind]}@{abbr}", slot) for abbr, slot in SLOTS]


GEN_MAX_NEW_TOKENS = 120
GEN_TEMPERATURE = 0.7
GEN_SAMPLES = 3
