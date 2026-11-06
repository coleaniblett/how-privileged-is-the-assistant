"""Spec R6 (2026-11-06): spec Q (what the role says: identity x voice) on Llama 3.1 8B and Apertus 8B, final only (base is Tier 2 spare work). Same design, frozen materials (data/role_says_materials.json), settings, judge and readings as extras/role_says.py;
outputs under results/overnight_2026-11-05/R6/, and a clearly labelled BREADTH section appended to results/role_says/role_says.md without changing the three-family readings. Exploratory, outside every verdict, NOT BLIND. No paid API.
    python -m extras.role_says_breadth example --family F
    python -m extras.role_says_breadth gen --family F      # GPU (marble), batch 32
    python -m extras.role_says_breadth judge               # GPU (vinyl), RS_OUT dir of this run
    python -m extras.role_says_breadth analyze             # CPU
"""
import argparse
import os

R6 = "results/overnight_2026-11-05/R6"
os.environ["RS_OUT"] = R6
from extras import role_says as rs  # noqa: E402
from extras import trait_atlas_breadth  # noqa: E402,F401  (registers the Llama/Apertus template settings and Apertus turn markers)

BFAMS = ["llama3.1-8b", "apertus-8b"]
MAIN_MD = "results/role_says/role_says.md"
MARK = "\n## BREADTH (spec R6): Llama 3.1 8B and Apertus 8B"


def analyze():
    rs.FAMS = BFAMS
    rs.analyze()
    sec = open(os.path.join(R6, "role_says.md"), encoding="utf-8").read().split("\n", 2)[2]
    head = ["", "## BREADTH (spec R6): Llama 3.1 8B and Apertus 8B", "",
            "Added 2026-11-06 under spec R (`docs/spec_R_overnight.md`). **Same design, materials, settings, judge and readings as above, final checkpoints, plus base where the training-check rows appear (T3.2), run on two more families; the three-family readings above are unchanged.** "
            "Heading levels below are the same as above. Apertus is the hard case: spec M found its self-image travels. Template notes: Apertus's template inserts its own system line (\"You are Apertus, a helpful assistant ...\") in the plain-Assistant and user-position conditions, and Llama's adds a date header; neither was changed. Provisional like everything judged here.", ""]
    main = open(MAIN_MD, encoding="utf-8").read()
    if MARK in main:
        main = main[:main.index(MARK)].rstrip("\n") + "\n"
    open(MAIN_MD, "w", encoding="utf-8").write(main + "\n".join(head) + sec.replace("\n## Overall reading by family and position (final)", "\n### Overall reading by family and position (final), breadth families", 1).replace("\n## ", "\n### ").replace("\n### ### ", "\n#### "))
    print("breadth section appended to", MAIN_MD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["example", "gen", "judge", "analyze"])
    ap.add_argument("--family", choices=BFAMS)
    ap.add_argument("--stage", default="instruct", choices=["instruct", "base"])
    a = ap.parse_args()
    if a.cmd == "example":
        rs.example(a.family)
    elif a.cmd == "gen":
        rs.gen(a.family, a.stage)
    elif a.cmd == "judge":
        rs.judge()
    else:
        analyze()


if __name__ == "__main__":
    main()
