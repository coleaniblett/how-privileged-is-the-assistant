"""Flags on open-user-turn continuations, for any chat template.

Reused from measures/vestigial_regression/flags.py (imported, unchanged): ASSISTANT_REGISTER and
repetition_loop. Its user_portion and turn markers are OLMo/ChatML-specific, so the cut and the turn flags
here use the markers derived from each model's template (render.turn_markers).

    user portion   completion before the first end-of-turn, turn-open or stop token
    ended_turn                 closed the user turn: end-of-turn marker, or stopped at end of text
    handoff_to_assistant       wrote the assistant header (<|im_start|>assistant, <|turn>model) anywhere
    new_user_turn              opened another user turn
    assistant_register         assistant phrases (VR regex) in the user portion
    assistant_register_unmarked
                               assistant phrases in the user portion before any turn marker, special or
                               plain-text ("Assistant:", "AI:", ...): Shiller's switch without turn markers.
                               The user portion already ends at the first special marker, so this differs
                               from assistant_register only when a plain-text speaker label comes first.
    plain_role_marker          a plain-text assistant speaker label at a line start in the user portion
    assistant_thinking         assistant-style reasoning about "the user" in the user portion
                               (Shiller's Qwen 3 observation; heuristic regex)
    special_token              any other special/added token in the user portion (e.g. <think>, <|channel>)
    repetition_loop            a word 4-gram repeated 3+ times in the user portion (VR)
    token_loop                 one token id repeated 10+ times in a row before the turn closes
                               (catches loops without spaces, e.g. "!!!!!!", which word n-grams miss)
All are heuristics; hand_labels.csv exists to check them.
"""

import re

from measures.vestigial_regression.flags import ASSISTANT_REGISTER, repetition_loop

PLAIN_ROLE = re.compile(r"(?:^|\n)[ \t>*#_]*(?:assistant|ai|ai assistant|model|bot|chatbot|chatgpt|gpt|claude|gemini|qwen"
                        r"|olmo|gemma)[ \t*_]*:", re.I)
THINKING = re.compile(
    r"(\bthe user (is|was|wants|wanted|asked|asks|might|may|said|says|seems|needs|mentioned|could)\b"
    r"|\bOkay,? (so )?(the user|let me|I need)\b|\bLet me (think|figure|start|break|recall|check)\b"
    r"|\bI (need|should) to (figure out|make sure|respond|address|consider)\b|\bI should (respond|address|make sure)\b"
    r"|\bmy response\b)",
    re.I,
)
TOKEN_RUN = 10
FLAG_NAMES = ["ended_turn", "handoff_to_assistant", "new_user_turn", "assistant_register",
              "assistant_register_unmarked", "plain_role_marker", "assistant_thinking", "special_token",
              "repetition_loop", "token_loop"]


def user_portion(text, markers, stop_tokens):
    cuts = [markers["end_of_turn"], markers["turn_open"], *stop_tokens]
    return text[: min((text.find(c) for c in cuts if c and c in text), default=len(text))]


def max_run(ids):
    best = cur = 0
    for i, t in enumerate(ids):
        cur = cur + 1 if i and t == ids[i - 1] else 1
        best = max(best, cur)
    return best


def flags(completion, ids, markers, stop_tokens, stop_ids, marker_ids, specials, stopped):
    """completion: decoded with special tokens kept. ids: its token ids. marker_ids: ids of single-token
    turn markers (to cut ids at the turn end). specials: special/added token strings of the tokenizer."""
    up = user_portion(completion, markers, stop_tokens)
    cut = next((i for i, t in enumerate(ids) if t in marker_ids or t in stop_ids), len(ids))
    reg = ASSISTANT_REGISTER.search(up)
    role = PLAIN_ROLE.search(up)
    others = set(specials) - {markers["end_of_turn"], markers["turn_open"], *stop_tokens}
    found = sorted(s for s in others if s in up)
    f = dict(
        ended_turn=bool(stopped or markers["end_of_turn"] in completion),
        handoff_to_assistant=markers["assistant_header"] in completion,
        new_user_turn=markers["user_header"] in completion,
        assistant_register=bool(reg),
        assistant_register_unmarked=bool(reg and (not role or reg.start() < role.start())),
        plain_role_marker=bool(role),
        assistant_thinking=bool(THINKING.search(up)),
        special_token=bool(found),
        repetition_loop=repetition_loop(up),
        token_loop=max_run(ids[:cut]) >= TOKEN_RUN,
    )
    return up, f, found
