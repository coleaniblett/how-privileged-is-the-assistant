"""Vestigial-regression flags on the prefilled user-turn continuations sampled by core/generate.py.

    python -m measures.vestigial_regression.flags    # reads results/<run>/core/generations/<stage>.jsonl

Writes <out>/<stage>.jsonl (one line per sample, with the user portion and the flags) and
<out>/<stage>_counts.csv (totals per flag), for every stage found in --generations.

Flags per sample:
    ended_turn            closed the user turn (<|im_end|> or end of document)
    handoff_to_assistant  wrote the assistant header after closing the turn (a subset of ended_turn)
    assistant_register    assistant phrases in the user portion (text before the turn closes)
    repetition_loop       a word 4-gram repeated 3+ times in the user portion
A post-trained model usually finishes the user's message, closes the turn and then answers as
the assistant, so register and repetition are checked on the user portion only.

Spec additions not implemented yet (README.md, "Pre-registered criteria"): a question-rate flag on
the user portion, compared against the real prefills' source texts; and the classification rule
that rising repetition_loop or handoff_to_assistant reads as vestigial regression while rising
assistant_register without handoff reads as overgeneralization in the user slot.
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import Counter

from core import config
from core.render import load_jsonl, load_tokenizer, turn_markers

ASSISTANT_REGISTER = re.compile(
    r"\b(I'd be (happy|glad) to|I would be (happy|glad) to|Certainly[,!.]|Sure[,!]|Of course[,!.]"
    r"|As an AI|I'm sorry, but|I hope this helps|Here (is|are) (a|an|the|some)\b|Great question"
    r"|Let me know if|feel free to)",
    re.I,
)
FLAG_NAMES = ["ended_turn", "handoff_to_assistant", "assistant_register", "repetition_loop"]


def repetition_loop(text, n=4, k=3):
    words = text.split()
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return any(c >= k for c in grams.values())


def user_portion(text, tok):
    """Continuation text before the model closes (or abandons) the user turn."""
    cut = min((text.find(t) for t in ("<|im_end|>", "<|im_start|>", tok.eos_token) if t and t in text), default=len(text))
    return text[:cut]


def flags(text, user_text, stopped_on_eos, end_marker, handoff):
    return dict(
        ended_turn=bool(stopped_on_eos or (end_marker and end_marker in text)),
        handoff_to_assistant=bool(handoff and handoff in text),
        assistant_register=bool(ASSISTANT_REGISTER.search(user_text)),
        repetition_loop=repetition_loop(user_text),
    )


def flag_samples(samples, tok):
    """(flagged records, totals) for the raw samples of one stage (core/generate.py rows)."""
    end_marker, handoff = turn_markers(tok)
    totals, records = Counter(), []
    for r in samples:
        user_text = user_portion(r["continuation"], tok)
        f = flags(r["continuation"], user_text, r["stopped_on_eos"], end_marker, handoff)
        totals.update({k: int(v) for k, v in f.items()})
        totals["n"] += 1
        records.append(dict(prefill_id=r["prefill_id"], sample=r["sample"], prefill=r["prefill"],
                            continuation=r["continuation"], user_portion=user_text, **f))
    return records, totals


def write_stage(out_dir, stage, records, totals):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{stage}.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, f"{stage}_counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        keys = ["n", *FLAG_NAMES]
        w.writerow(["stage", *keys])
        w.writerow([stage, *(totals[k] for k in keys)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", default=config.core_dir("generations"), help="core samples, <stage>.jsonl")
    ap.add_argument("--stage", help="one stage only (default: every <stage>.jsonl in --generations)")
    ap.add_argument("--tokenizer", default=config.TEMPLATE_MODEL)
    ap.add_argument("--out", default=os.path.join(config.measure_dir("vestigial_regression"), "generations"))
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    paths = ([os.path.join(args.generations, f"{args.stage}.jsonl")] if args.stage
             else sorted(glob.glob(os.path.join(args.generations, "*.jsonl"))))
    if not paths:
        raise SystemExit(f"no sample files in {args.generations}")
    for path in paths:
        stage = os.path.basename(path)[:-6]
        records, totals = flag_samples(load_jsonl(path), tok)
        write_stage(args.out, stage, records, totals)
        print(f"{stage}: {dict(totals)}")


if __name__ == "__main__":
    main()
