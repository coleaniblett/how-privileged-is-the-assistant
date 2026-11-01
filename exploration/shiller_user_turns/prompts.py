"""The four prompt conditions, each rendered as an open user turn.

    empty      the user header and nothing else (the model writes a user turn from scratch)
    ordinary   data/prefill.jsonl: first sentences of held-out WildChat user turns
    register   openers_register.jsonl: simplified English, child-like speech, typos/lowercase
    norm       openers_norm.jsonl: a harmless but norm-violating intent stated by the user
               (Shiller's harmful-instruction prefill is deliberately excluded)
"""

import os

from core import config
from core.render import load_jsonl

HERE = os.path.dirname(os.path.abspath(__file__))
CONDITIONS = ["empty", "ordinary", "register", "norm"]
DESCRIPTIONS = {
    "empty": "The user header and nothing else: the model writes a user turn from scratch.",
    "ordinary": "First sentences of held-out WildChat user turns (`data/prefill.jsonl`).",
    "register": "Openers in a distinctive user voice (simplified English, child-like speech, typos/lowercase): "
                "are the quirks sustained or regularized toward assistant style?",
    "norm": "The user states a harmless but norm-violating intent: do disclaimers or self-correction appear "
            "in the user's own voice?",
}


def load(condition):
    """[{"id", "prefill"}] for one condition."""
    if condition == "empty":
        return [{"id": "empty-00", "prefill": ""}]
    if condition == "ordinary":
        return [{"id": r["id"], "prefill": r["prefill"]} for r in load_jsonl(config.PREFILL_PATH)]
    if condition in ("register", "norm"):
        return load_jsonl(os.path.join(HERE, f"openers_{condition}.jsonl"))
    raise ValueError(f"unknown condition {condition!r}; known: {CONDITIONS}")
