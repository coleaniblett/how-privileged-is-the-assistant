"""Open (unterminated) user turns rendered with a model's own chat template, and its turn markers.

core.render.template_parts takes no template options and core's turn markers include Qwen3's empty
<think> block in the handoff, so this module derives the same pieces with template kwargs and splits the
handoff into an assistant header that a model writing its own thinking would still match.
"""

from dataclasses import dataclass

from core.render import AlignmentError, check_text, encode, special_strings

SENT = "<<<CONTENT>>>"


@dataclass
class OpenTurn:
    text: str  # exact rendered string; ends with the prefill
    input_ids: list[int]


def _render(tok, msgs, kw):
    return tok.apply_chat_template(msgs, tokenize=False, **kw)


def user_parts(tok, kw):
    s = _render(tok, [{"role": "user", "content": SENT}], kw)
    if s.count(SENT) != 1:
        raise AlignmentError("template does not place the user content verbatim exactly once")
    return tuple(s.split(SENT))


def render_open_user_turn(tok, prefill, kw=None):
    """Render [user: prefill] with the chat template and drop everything after the prefill (end-of-turn
    token and anything else); no generation prompt. Verified by decoding."""
    kw = kw or {}
    check_text(tok, prefill)  # no special tokens, no leading/trailing whitespace ("" is allowed)
    prefix, suffix = user_parts(tok, kw)
    full = _render(tok, [{"role": "user", "content": prefill}], kw)
    if full != prefix + prefill + suffix:
        raise AlignmentError("template modifies the user content")
    text = full[: len(prefix) + len(prefill)]
    ids = encode(tok, text)
    decoded = tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if decoded != text or not decoded.endswith(prefill):
        raise AlignmentError("decoded prompt does not reproduce the rendered string ending in the prefill")
    if prefill:
        tail = tok.decode(ids[len(encode(tok, prefix)):], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if tail != prefill:
            raise AlignmentError("prefill tokens merge with the user header")
    return OpenTurn(text, ids)


def turn_markers(tok, kw=None):
    """Strings that delimit turns in this template, derived by rendering:
    end_of_turn       closes the user turn (<|im_end|>, <turn|>)
    turn_open         opens any turn (<|im_start|>, <|turn>)
    user_header       turn_open + user role name
    assistant_header  turn_open + assistant role name (<|im_start|>assistant, <|turn>model)
    """
    kw = kw or {}
    prefix, suffix = user_parts(tok, kw)
    end = suffix.strip()
    specials = sorted(special_strings(tok), key=len, reverse=True)
    before_role = prefix[: prefix.rstrip().rfind("user")]
    turn_open = next((s for s in specials if before_role.endswith(s)), None)
    if not end or not turn_open:
        raise AlignmentError(f"could not derive turn markers (end={end!r}, turn_open={turn_open!r})")
    s = _render(tok, [{"role": "user", "content": SENT}, {"role": "assistant", "content": SENT + "2"}], kw)
    between = s.split(SENT)[1]
    i = between.find(turn_open)
    if i < 0:
        raise AlignmentError("assistant turn does not start with the turn-open marker")
    header = between[i:].split("\n")[0].strip()
    user_header = prefix[len(before_role) - len(turn_open):].strip()
    return dict(end_of_turn=end, turn_open=turn_open, user_header=user_header, assistant_header=header)
