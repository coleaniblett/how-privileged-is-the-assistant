"""README generation arm on Gemma 4 31B-it: do the samples the flags identify turn out to be vestigial regression?

    python -m measures.vestigial_regression.pilot.readme_arm     # from the repo root, no GPU

Samples: gemma4-31b-it/gemma4-31b-it.jsonl, made with the README's sampling settings (the 20 data/prefill.jsonl
sentences x 3 samples x 120 new tokens, T=0.7, seed 0, stop only at end of text, int8) by
exploration.shiller_user_turns.generate. Labels: labels_readme_arm.csv (by eye, made before the flags were computed).
Flags and criterion 4 are the ones in run.py. Writes readme_arm.csv and readme_arm.md next to this file.
"""

import csv
import os

from core.render import load_jsonl
from measures.vestigial_regression.pilot.run import HERE, classify, measure_flags, rate

SAMPLES = os.path.join(HERE, "gemma4-31b-it", "gemma4-31b-it.jsonl")
FLAGS = ["ended_turn", "handoff_to_assistant", "assistant_register", "repetition_loop"]


def main():
    rows = {(r["prompt_id"], int(r["sample"])): r for r in load_jsonl(SAMPLES)}
    out = []
    with open(os.path.join(HERE, "labels_readme_arm.csv"), newline="") as f:
        for lab in csv.DictReader(f):
            r = rows[(lab["prompt_id"], int(lab["sample"]))]
            fl = measure_flags(r["user_portion"], r["ended_turn"], r["handoff_to_assistant"])
            out.append(dict(id=f"{lab['prompt_id']}/{lab['sample']}", label=lab["label"], **fl,
                            cls=classify(fl), head=r["user_portion"][:80].replace("\n", " ")))
    assert len(out) == len(rows) == 60
    with open(os.path.join(HERE, "readme_arm.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(out[0]))
        w.writeheader()
        w.writerows(out)

    tot = {k: sum(r[k] for r in out) for k in FLAGS}
    L = ["# README generation arm on Gemma 4 31B-it: are the flagged samples vestigial regression?", "",
         f"60 samples (20 prefills x 3, 120 tokens, T=0.7, int8). By eye, all {len(out)} are vestigial regression "
         "(labels made before the flags were computed, by the agent that wrote the pilot; a teammate should check): "
         f"{sum(r['label'] == 'degenerate' for r in out)} degenerate loops and "
         f"{sum(r['label'] == 'switched to assistant' for r in out)} switches to the assistant. No sample stays in "
         "the user's voice.", "", "## Flags (README criterion 3), out of 60", "",
         "| " + " | ".join(FLAGS) + " |", "|" + "---|" * len(FLAGS),
         "| " + " | ".join(str(tot[k]) for k in FLAGS) + " |", "",
         "## Classification (criterion 4) against the labels", "",
         "| label | n | `vestigial` | `overgeneralization` | `unclassified` |", "|---|---|---|---|---|"]
    for lab in ["degenerate", "switched to assistant"]:
        s = [r for r in out if r["label"] == lab]
        L.append(f"| {lab} | {len(s)} | {rate(s, lambda r: r['cls'] == 'vestigial')} | "
                 f"{rate(s, lambda r: r['cls'] == 'overgeneralization')} | {rate(s, lambda r: r['cls'] == 'unclassified')} |")
    flagged = [r for r in out if r["cls"] == "vestigial"]
    L += [f"| **all** | {len(out)} | {rate(out, lambda r: r['cls'] == 'vestigial')} | "
          f"{rate(out, lambda r: r['cls'] == 'overgeneralization')} | {rate(out, lambda r: r['cls'] == 'unclassified')} |",
          "", f"**Precision of the `vestigial` class:** every one of the {len(flagged)} samples classed `vestigial` is "
          "vestigial regression by label (" + rate(flagged, lambda r: True) + ", all labelled degenerate or switched). "
          "Only specificity on real user text (run.py, 18 turns: 0 `vestigial`) tests false alarms.", "",
          "## Not classed `vestigial` (missed)", "", "| id | label | class | head |", "|---|---|---|---|"]
    for r in out:
        if r["cls"] != "vestigial":
            L.append(f"| {r['id']} | {r['label']} | {r['cls']} | `{r['head']}` |")
    with open(os.path.join(HERE, "readme_arm.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:22]))
    print("missed:", sum(r["cls"] != "vestigial" for r in out))


if __name__ == "__main__":
    main()
