"""Exploratory pilot: does the vestigial-regression flag approach catch clear vestigial regression?

    python -m measures.vestigial_regression.pilot.run

Inputs: hand labels (labels.csv, made blind to the flags) of 30 random Gemma 4 31B-it open-user-turn samples
(results/exploration/shiller_user_turns/gemma4-31b-it.jsonl), plus a control of real user text.
Applies the measure's own flag logic (ASSISTANT_REGISTER and repetition_loop from ../flags.py, imported
unchanged; ended_turn and handoff_to_assistant as computed with Gemma's turn markers in the exploration run)
and the classification rule from the README's criterion 4 (a spec addition, not implemented in the measure).
Writes per_sample.csv and results.md next to this file. Touches nothing else.

The user portion is the stored one: completion before the first Gemma end-of-turn / turn-open / stop token.
"""

import csv
import json
import os
from collections import Counter

from core import config
from core.render import load_jsonl
from measures.vestigial_regression.flags import ASSISTANT_REGISTER, repetition_loop

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(config.RESULTS_ROOT, "exploration", "shiller_user_turns", "gemma4-31b-it.jsonl")
DATA = "data"  # repo-relative, like config.RESULTS_ROOT; run from the repo root
CONTROL_CHARS = 600  # chars of each real user turn after the prefill sentence
MIN_REMAINDER = 50


def classify(f):
    """README criterion 4 as written: repetition_loop or handoff -> vestigial regression; assistant_register
    without handoff -> overgeneralization in the user slot (reported elsewhere); otherwise unclassified."""
    if f["repetition_loop"] or f["handoff_to_assistant"]:
        return "vestigial"
    if f["assistant_register"]:
        return "overgeneralization"
    return "unclassified"


def measure_flags(user_portion, ended_turn=False, handoff=False):
    return dict(ended_turn=bool(ended_turn), handoff_to_assistant=bool(handoff),
                assistant_register=bool(ASSISTANT_REGISTER.search(user_portion)),
                repetition_loop=repetition_loop(user_portion))


def labeled_rows():
    rows = {(r["condition"], r["prompt_id"], int(r["sample"])): r for r in load_jsonl(SAMPLES)}
    out = []
    with open(os.path.join(HERE, "labels.csv"), newline="") as f:
        for lab in csv.DictReader(f):
            r = rows[(lab["condition"], lab["prompt_id"], int(lab["sample"]))]
            fl = measure_flags(r["user_portion"], r["ended_turn"], r["handoff_to_assistant"])
            out.append(dict(kind="gemma", id=f"{lab['condition']}/{lab['prompt_id']}/{lab['sample']}",
                            label=lab["label"], **fl, token_loop=bool(r["token_loop"]),
                            cls=classify(fl), n_chars=len(r["user_portion"]),
                            head=r["user_portion"][:80].replace("\n", " ")))
    return out


def control_rows():
    texts = {t["id"]: t["text"] for t in map(json.loads, open(os.path.join(DATA, "texts.jsonl")))}
    out = []
    for p in map(json.loads, open(os.path.join(DATA, "prefill.jsonl"))):
        text = texts[p["text_id"]]
        assert text.startswith(p["prefill"]), p["id"]
        rest = text[len(p["prefill"]):].strip()[:CONTROL_CHARS]
        if len(rest) < MIN_REMAINDER:
            continue
        fl = measure_flags(rest)
        out.append(dict(kind="control", id=p["id"], label="user voice (real)", **fl, token_loop=False,
                        cls=classify(fl), n_chars=len(rest), head=rest[:80].replace("\n", " ")))
    return out


def rate(rows, pred):
    n = len(rows)
    k = sum(1 for r in rows if pred(r))
    return f"{k}/{n}" + (f" ({k / n:.0%})" if n else "")


def main():
    rows = labeled_rows() + control_rows()
    cols = ["kind", "id", "label", "ended_turn", "handoff_to_assistant", "assistant_register",
            "repetition_loop", "token_loop", "cls", "n_chars", "head"]
    with open(os.path.join(HERE, "per_sample.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)

    g = [r for r in rows if r["kind"] == "gemma"]
    c = [r for r in rows if r["kind"] == "control"]
    any_flag = lambda r: r["handoff_to_assistant"] or r["assistant_register"] or r["repetition_loop"]
    L = ["# Pilot: do the vestigial-regression flags catch clear vestigial regression?", "",
         f"Exploratory; n is tiny. {len(g)} hand-labelled Gemma 4 31B-it samples (random, labelled blind to the "
         f"flags by the agent who wrote this pilot; to be checked by a teammate) and {len(c)} real WildChat user "
         f"turns (the text after each prefill sentence, first {CONTROL_CHARS} chars) as a user-voice control. "
         "The classification rule is README criterion 4, a spec addition that the measure does not implement.", "",
         "## Detection of labelled vestigial regression", "",
         "| label | n | flagged by any of handoff / register / loop | classed `vestigial` | classed `overgeneralization` "
         "| `unclassified` | caught by `token_loop` (not in the measure) |", "|---|---|---|---|---|---|---|"]
    for lab in sorted({r["label"] for r in g}):
        s = [r for r in g if r["label"] == lab]
        L.append(f"| {lab} | {len(s)} | {rate(s, any_flag)} | {rate(s, lambda r: r['cls'] == 'vestigial')} | "
                 f"{rate(s, lambda r: r['cls'] == 'overgeneralization')} | "
                 f"{rate(s, lambda r: r['cls'] == 'unclassified')} | {rate(s, lambda r: r['token_loop'])} |")
    L += [f"| **all labelled** | {len(g)} | {rate(g, any_flag)} | {rate(g, lambda r: r['cls'] == 'vestigial')} | "
          f"{rate(g, lambda r: r['cls'] == 'overgeneralization')} | {rate(g, lambda r: r['cls'] == 'unclassified')} "
          f"| {rate(g, lambda r: r['token_loop'])} |", "",
          "Every labelled sample is vestigial regression (`degenerate` or `switched to assistant`), so there are "
          "no Gemma negatives: this measures recall only. Counts of the individual flags over the labelled "
          "samples: " + ", ".join(f"{k} {sum(1 for r in g if r[k])}" for k in
                                   ["ended_turn", "handoff_to_assistant", "assistant_register", "repetition_loop"])
          + ".", "", "## False positives on real user text", "",
          f"Flags on {len(c)} real user continuations: `assistant_register` {sum(r['assistant_register'] for r in c)}, "
          f"`repetition_loop` {sum(r['repetition_loop'] for r in c)}; classed `vestigial` "
          f"{rate(c, lambda r: r['cls'] == 'vestigial')}, `overgeneralization` "
          f"{rate(c, lambda r: r['cls'] == 'overgeneralization')}. ", "",
          "## Missed or misclassified labelled samples", "",
          "| id | label | class | head of user portion |", "|---|---|---|---|"]
    for r in g:
        if r["cls"] != "vestigial":
            L.append(f"| {r['id']} | {r['label']} | {r['cls']} | `{r['head']}` |")
    L += ["", "Per-sample flags: `per_sample.csv`."]
    with open(os.path.join(HERE, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(Counter((r["label"], r["cls"]) for r in g))


if __name__ == "__main__":
    main()
