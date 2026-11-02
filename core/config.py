"""Shared settings. CLI flags on each script override these.

Every measure reads models, stages, slots, text sources and output paths from here; nothing
else defines a model list (see AGENTS.md).
"""

import os

# Model families. Each family lists its checkpoints in post-training order and names the
# checkpoint whose tokenizer and chat template render every stage of that family (so token
# sequences are identical across stages). contamination_check: whether the family's
# post-training data is open, so test texts can be filtered for overlap (OLMo 3: Dolci).
# Optional per-family settings (defaults in FAMILY_DEFAULTS; read them with family_settings()):
#   dtype            scoring dtype, "fp32" everywhere (AGENTS.md: bf16 shifts NLL with batch composition)
#   offload, gpu_gib place weights with device_map="auto": at most gpu_gib GiB on the GPU, the rest in CPU RAM
#                    (needed to score a 31B model in fp32 on a 48 GB card; arithmetic is unchanged)
#   batch_size       scoring batch size (a 262k-vocabulary model needs a small one: logits dominate memory)
#   gen_precision    generation precision: "bf16" or "int8" (bitsandbytes; needs `uv sync --extra quant`)
#   handoff_extra    strings whose presence in a continuation counts as a switch to the assistant even without
#                    the template's turn markers (Gemma's thought channel, Qwen3's think block)
#   assistant_think_block  Qwen3 only: render the assistant slot with the empty "<think>\n\n</think>\n\n" block
#                    (None/False: as the template renders a stored assistant-only turn, what the pilot used)
# The end-of-turn token and the user->assistant handoff string are not stored here: they are derived
# from each family's chat template (core.render.turn_markers), and the assistant-slot opening from its
# generation prompt (core.render.assistant_opening_extra).
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
    # Breadth arm. Their post-training data is not open, so test texts cannot be checked for contamination:
    # say so wherever their results appear. Qwen3-32B has no published base checkpoint.
    "gemma4-31b": {
        "stages": {"base": "google/gemma-4-31B", "instruct": "google/gemma-4-31B-it"},
        "template": "google/gemma-4-31B-it",
        "contamination_check": False,
        "offload": True,
        "gpu_gib": 30,
        "batch_size": 4,
        "gen_precision": "int8",
        "handoff_extra": ["<|channel>", "<channel|>"],
    },
    "qwen3-8b": {
        "stages": {"base": "Qwen/Qwen3-8B-Base", "instruct": "Qwen/Qwen3-8B"},
        "template": "Qwen/Qwen3-8B",
        "contamination_check": False,
        "batch_size": 4,
        "handoff_extra": ["<think>", "</think>"],
        "assistant_think_block": False,
    },
}

FAMILY_DEFAULTS = dict(dtype="fp32", offload=False, gpu_gib=34, cpu_gib=900, batch_size=8, gen_precision="bf16",
                       handoff_extra=[], assistant_think_block=None)


def family_settings(name=None):
    """A family's entry with FAMILY_DEFAULTS filled in (the active family if name is None)."""
    name = name or FAMILY
    if name not in FAMILIES:
        raise KeyError(f"unknown family {name!r}; known: {', '.join(FAMILIES)}")
    return {**FAMILY_DEFAULTS, **FAMILIES[name]}


# The family scored by default. MODELS/STAGES/TEMPLATE_MODEL are that family's, so every
# script keeps working unchanged; a script's --family flag switches them with use_family().
FAMILY = "olmo3-7b"
FAMILY_DEFAULT_NAME = FAMILY  # the family that owns the top of docs/models.md
MODELS = FAMILIES[FAMILY]["stages"]
STAGES = list(MODELS)

# Every stage (base included) is scored with this tokenizer and chat template,
# so token sequences are identical across stages.
TEMPLATE_MODEL = FAMILIES[FAMILY]["template"]

# Qwen3 only: whether the assistant slot carries the empty think block. None = as the template renders it.
ASSISTANT_THINK_BLOCK = None

# Set by use_family() only when a script was given --family explicitly. Output paths then gain the family
# (results/<run>/<family>/...); without it every path is exactly what it was before families existed.
EXPLICIT_FAMILY = None
DEFAULT_RUN = "capstone"  # run directory used with an explicit --family when no --run is given


def use_family(name):
    """Make `name` the active family: MODELS, STAGES, TEMPLATE_MODEL and the assistant-slot setting follow it."""
    global FAMILY, MODELS, STAGES, TEMPLATE_MODEL, ASSISTANT_THINK_BLOCK, EXPLICIT_FAMILY
    st = family_settings(name)
    FAMILY, MODELS, STAGES, TEMPLATE_MODEL = name, st["stages"], list(st["stages"]), st["template"]
    ASSISTANT_THINK_BLOCK = st["assistant_think_block"]
    EXPLICIT_FAMILY = name


def family_from_argv(argv=None):
    """Scan argv for --family NAME, activate it, and return the name (None if absent). Call before building an
    argparse parser, so defaults that read config.TEMPLATE_MODEL / config.MODELS see the chosen family."""
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    for i, a in enumerate(args):
        name = a.split("=", 1)[1] if a.startswith("--family=") else (args[i + 1] if a == "--family" and i + 1 < len(args) else None)
        if name:
            use_family(name)
            return name
    return None

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
DATASET_IDS = {"wildchat": "allenai/WildChat-1M", "oasst1": "OpenAssistant/oasst1", "ultrachat": "openbmb/UltraChat"}
# Drop any conversation whose user turn appears in the Dolci-Instruct SFT/DPO/RL data.
EXCLUDE_DOLCI = True

N_TEXTS = 300  # per source
MIN_TOKENS, MAX_TOKENS = 20, 300
N_PREFILL = 20  # the original prefill set (data/prefill.jsonl, frozen; the exploration uses it)
N_PREFILL_LARGE = 100  # the capstone prefill set (data/prefill_100.jsonl), a superset of the 20
SEED = 0

REPLY_SOURCE = "olmo-instruct"
REPLY_MODEL = "instruct"
REPLY_MAX_NEW_TOKENS = 300

DATA_PATH = "data/texts.jsonl"
# The authorship cross of measures/logit_distortion (core.prepare_data --authorship): human-written assistant replies
# (oasst1) and model-written user turns and replies (UltraChat), in a separate file so texts.jsonl stays as it was.
AUTHORSHIP_PATH = "data/texts_authorship.jsonl"
PREFILL_PATH_20 = "data/prefill.jsonl"  # frozen; the first 20 rows of PREFILL_PATH
PREFILL_PATH = "data/prefill_100.jsonl"  # default for core.generate

# Output layout (see AGENTS.md). One directory per run under results/, named after the family by
# default (scripts/run_gpu.sh: RUN=... or OUT=...). Core outputs go under core/, each measure's
# under its own name, extras under extras/. results/scores, results/generations, results_preceding/
# and results_v0/ are pre-capstone runs and are frozen.
RESULTS_ROOT = "results"
RUN = FAMILY


def run_dir(run=None, family=None):
    """results/<run>, or results/<run>/<family> for an explicit family (default run: DEFAULT_RUN)."""
    family = family or EXPLICIT_FAMILY
    if family:
        return os.path.join(RESULTS_ROOT, run or DEFAULT_RUN, family)
    return os.path.join(RESULTS_ROOT, run or RUN)


def core_dir(kind, run=None, family=None):
    """kind: "scores" or "generations"."""
    return os.path.join(run_dir(run, family), "core", kind)


def measure_dir(name, run=None, family=None):
    """name: a measure ("vestigial_regression", ...) or "extras"."""
    return os.path.join(run_dir(run, family), name)


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

# Conversation candidates are a separate, versioned format, never legacy isolated texts.
# core.score refuses records carrying this format: original-context scoring is a later task.
CONVERSATION_FORMAT = "conversation_candidates_v1"
CONVERSATION_SOURCES = {
    "oasst1": {"dataset": DATASET_IDS["oasst1"], "split": "validation"},
    "wildchat": {"dataset": DATASET_IDS["wildchat"], "split": "train"},
    "ultrachat": {"dataset": "openbmb/UltraChat", "split": "train"},
}
CONVERSATION_CANDIDATES_DIR = "candidates/logit_distortion/pilot_v1"
