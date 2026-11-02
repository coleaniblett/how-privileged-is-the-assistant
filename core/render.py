"""Build each scoring condition from the chat template and locate the content tokens.

Each data row is one text (a source and a kind, user or assistant). It is rendered alone
as the first turn (optionally after one fixed neutral turn), so across slot conditions
only the role header differs. The content is tokenized on its
own and the full sequence must equal prefix + content + suffix tokenized separately;
this rules out merges at the header/footer boundary and makes the content tokens
identical in every condition, plain-prose control included.

    python -m core.render --tokenizer allenai/Olmo-3-7B-Instruct --n 2   # dump examples per source to docs/
"""

import argparse
import json
from dataclasses import dataclass

from transformers import AutoTokenizer

from core import config

SENTINEL = "CONTENT"


class AlignmentError(ValueError):
    pass


@dataclass
class Rendered:
    input_ids: list[int]
    start: int  # content occupies input_ids[start:end]
    end: int
    text: str  # the full rendered string


def load_tokenizer(name):
    return AutoTokenizer.from_pretrained(name)


def special_strings(tok):
    return {s for s in tok.all_special_tokens + list(tok.get_added_vocab()) if s}


def check_text(tok, text):
    """Raise AlignmentError if a text cannot be scored cleanly with this tokenizer."""
    if text != text.strip():
        raise AlignmentError("text has leading/trailing whitespace")
    for s in special_strings(tok):
        if s in text:
            raise AlignmentError(f"text contains special token {s!r}")


def messages_for(slot, content, preceding=False):
    msgs = []
    if preceding:
        other = "assistant" if slot == "user" else "user"
        msgs.append({"role": other, "content": config.NEUTRAL_TURN[other]})
    msgs.append({"role": slot, "content": content})
    return msgs


def assistant_opening_extra(tok, preceding=False):
    """Text the template's generation prompt puts after the assistant header, beyond what it renders for a stored
    assistant turn. Empty for templates whose generation prompt ends at the header (OLMo, Qwen3 by default). Gemma 4
    opens every generated model turn with an empty thought channel ("<|channel>thought\n<channel|>") that the
    template omits when it renders a stored turn, so an assistant-slot condition without it is off-distribution.
    Measured after a neutral user turn (a generation prompt for an empty conversation is not renderable), the same
    for every scored text; `preceding` is accepted for symmetry with template_parts and does not change it."""
    user = {"role": "user", "content": config.NEUTRAL_TURN["user"]}
    try:
        gen = tok.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
        stored = tok.apply_chat_template([user, {"role": "assistant", "content": SENTINEL}], tokenize=False)
    except Exception:  # noqa: BLE001
        return ""
    if stored.count(SENTINEL) != 1:
        return ""
    stored_prefix = stored.split(SENTINEL)[0]
    extra = gen[len(stored_prefix) :] if gen.startswith(stored_prefix) and len(gen) > len(stored_prefix) else ""
    # Qwen3 option (config.ASSISTANT_THINK_BLOCK, off by default): open the assistant slot with the empty think block
    # that the template's non-thinking generation prompt adds. A no-op for templates without a <think> token.
    if not extra and getattr(config, "ASSISTANT_THINK_BLOCK", None) and "<think>" in tok.get_vocab():
        extra = "<think>\n\n</think>\n\n"
    return extra


def template_parts(tok, slot, preceding=False, opening=True):
    """Rendered string before and after the scored turn's content. opening=False gives the template's own
    rendering of a stored turn; opening=True (default) adds the assistant opening a generation prompt would
    (assistant_opening_extra; empty except for families like Gemma 4)."""
    s = tok.apply_chat_template(messages_for(slot, SENTINEL, preceding), tokenize=False)
    if s.count(SENTINEL) != 1:
        raise AlignmentError("template does not place content verbatim exactly once")
    prefix, suffix = s.split(SENTINEL)
    if opening and slot == "assistant":
        prefix += assistant_opening_extra(tok, preceding)
    return prefix, suffix


def turn_markers(tok):
    """End-of-user-turn string and the user->assistant handoff string from the template."""
    _, user_suffix = template_parts(tok, "user")
    s = tok.apply_chat_template(
        [{"role": "user", "content": SENTINEL}, {"role": "assistant", "content": SENTINEL + "2"}], tokenize=False
    )
    handoff = s.split(SENTINEL)[1]
    return user_suffix.strip(), handoff.strip()


def family_turn_markers(tok):
    """Turn delimiters derived from the chat template by rendering (nothing is hard-coded per family):
    end_of_turn       closes the user turn (<|im_end|>, <turn|>)
    turn_open         opens any turn (<|im_start|>, <|turn>)
    user_header       turn_open + the user role name
    assistant_header  turn_open + the assistant role name (<|im_start|>assistant, <|turn>model)
    handoff           end_of_turn through assistant_header as the template writes them between two turns
                      (core's turn_markers handoff also includes Qwen3's empty think block; this stops at the header)
    Ported from exploration/shiller_user_turns/render.py; turn_markers() above is unchanged."""
    prefix, suffix = template_parts(tok, "user", opening=False)
    end = suffix.strip()
    specials = sorted(special_strings(tok), key=len, reverse=True)
    before_role = prefix[: prefix.rstrip().rfind("user")]
    turn_open = next((t for t in specials if before_role.endswith(t)), None)
    if not end or not turn_open:
        raise AlignmentError(f"could not derive turn markers (end={end!r}, turn_open={turn_open!r})")
    s = tok.apply_chat_template(
        [{"role": "user", "content": SENTINEL}, {"role": "assistant", "content": SENTINEL + "2"}], tokenize=False)
    between = s.split(SENTINEL)[1]
    i = between.find(turn_open)
    if i < 0:
        raise AlignmentError("assistant turn does not start with the turn-open marker")
    header = between[i:].split("\n")[0].strip()
    user_header = prefix[len(before_role) - len(turn_open):].strip()
    return dict(end_of_turn=end, turn_open=turn_open, user_header=user_header, assistant_header=header,
                handoff=between[: i + len(header)].strip())


def doc_start_ids(tok):
    """Token(s) that open a plain-prose document, so its first content token is scored."""
    vocab = tok.get_vocab()
    if "<|endoftext|>" in vocab:  # document separator (OLMo, SmolLM); their bos may be a chat token
        return [vocab["<|endoftext|>"]]
    return [tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id]


def encode(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"]


def render(tok, text, slot, preceding=False):
    """slot: "user", "assistant", or None for plain prose."""
    check_text(tok, text)
    content = encode(tok, text)
    if tok.decode(content, clean_up_tokenization_spaces=False) != text:
        raise AlignmentError("decoding the content tokens does not reproduce the text")

    if slot is None:
        ids = doc_start_ids(tok) + content
        return Rendered(ids, len(ids) - len(content), len(ids), tok.decode(ids))

    prefix, suffix = template_parts(tok, slot, preceding)
    full = prefix + text + suffix
    raw_prefix, _ = template_parts(tok, slot, preceding, opening=False)
    if tok.apply_chat_template(messages_for(slot, text, preceding), tokenize=False) != raw_prefix + text + suffix:
        raise AlignmentError("template modifies the content")
    pre, suf = encode(tok, prefix), encode(tok, suffix)
    ids = encode(tok, full)
    if ids != pre + content + suf:
        raise AlignmentError("tokens merge across the content boundary")
    start, end = len(pre), len(pre) + len(content)
    assert tok.decode(ids[start:end], clean_up_tokenization_spaces=False) == text
    return Rendered(ids, start, end, full)


def render_all(tok, row, preceding=False):
    """One text (a data row) in every slot; asserts its content tokens match across slots."""
    out = {cond: render(tok, row["text"], slot, preceding) for cond, slot in config.conditions(row["kind"])}
    spans = [r.input_ids[r.start : r.end] for r in out.values()]
    assert all(s == spans[0] for s in spans), f"content tokens differ across slots ({row['id']})"
    return out


@dataclass
class RenderedConversation:
    input_ids: list[int]
    spans: list[tuple[int, int]]  # content of turn i occupies input_ids[start:end]
    text: str


def render_conversation(tok, contents, roles):
    """Several turns with the given roles (default system turn included). Each content is tokenized
    on its own and the full sequence must equal the template pieces and contents tokenized
    separately, so a content's tokens do not depend on the roles around it."""
    marks = [f"<<<{SENTINEL}{i}>>>" for i in range(len(contents))]
    msgs = lambda texts: [{"role": r, "content": t} for r, t in zip(roles, texts)]
    s = tok.apply_chat_template(msgs(marks), tokenize=False)
    pieces, rest = [], s
    for m in marks:
        if rest.count(m) != 1:
            raise AlignmentError("template does not place each content verbatim exactly once")
        before, rest = rest.split(m)
        pieces.append(before)
    pieces.append(rest)

    ids, spans, full = [], [], ""
    for i, text in enumerate(contents):
        check_text(tok, text)
        content = encode(tok, text)
        if tok.decode(content, clean_up_tokenization_spaces=False) != text:
            raise AlignmentError("decoding the content tokens does not reproduce the text")
        ids += encode(tok, pieces[i])
        spans.append((len(ids), len(ids) + len(content)))
        ids += content
        full += pieces[i] + text
    ids += encode(tok, pieces[-1])
    full += pieces[-1]
    if tok.apply_chat_template(msgs(contents), tokenize=False) != full:
        raise AlignmentError("template modifies the content")
    if encode(tok, full) != ids:
        raise AlignmentError("tokens merge across a content boundary")
    return RenderedConversation(ids, spans, full)


def render_prefill(tok, sentence, preceding=False):
    """User turn opened with `sentence` and left unterminated, for generation."""
    check_text(tok, sentence)
    prefix, _ = template_parts(tok, "user", preceding)
    return encode(tok, prefix + sentence)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def dump_examples(tok, rows, path, preceding=False):
    lines = [
        "# Rendered examples",
        "",
        f"Tokenizer/template: `{tok.name_or_path}`; preceding neutral turn: {preceding}.",
        "Scored content is wrapped in ⟦ ⟧ (markers added for display only).",
        "",
    ]
    for row in rows:
        lines.append(f"## {row['id']} ({row['source']} {row['kind']} text)\n")
        for cond, r in render_all(tok, row, preceding).items():
            before = tok.decode(r.input_ids[: r.start])
            content = tok.decode(r.input_ids[r.start : r.end])
            after = tok.decode(r.input_ids[r.end :])
            lines += [
                f"### {cond} ({r.end - r.start} content tokens of {len(r.input_ids)})",
                "",
                "````",  # four backticks: texts may contain ``` code fences
                f"{before}⟦{content}⟧{after}",
                "````",
                "",
            ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=config.TEMPLATE_MODEL)
    ap.add_argument("--dataset", default=config.DATA_PATH)
    ap.add_argument("--n", type=int, default=2, help="examples per source and kind")
    ap.add_argument("--preceding-turn", action="store_true", default=config.PRECEDING_TURN)
    ap.add_argument("--out", default="docs/rendered_examples.md")
    args = ap.parse_args()
    tok = load_tokenizer(args.tokenizer)
    rows, count = [], {}
    for row in load_jsonl(args.dataset):
        key = (row["source"], row["kind"])
        if count.get(key, 0) < args.n:
            count[key] = count.get(key, 0) + 1
            rows.append(row)
    dump_examples(tok, rows, args.out, args.preceding_turn)
    print(f"wrote {args.out}")
