"""Frame-cost change from base by content-position bin (descriptive), from the per-token tables.

    python -m exploration.turns_test.positions [family]   # -> results/<family>/positions.md
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BINS = [1, 5, 20, 60, 10_000]
LABELS = ["1-4", "5-19", "20-59", "60+"]


def frame(fam, stage):
    d = pd.read_parquet(os.path.join(HERE, "results", fam, f"{stage}_tokens.parquet"))
    d = d[d.position >= 1].copy()
    d["bin"] = pd.cut(d.position, BINS, right=False, labels=LABELS)
    m = d.groupby(["conv_id", "source", "cell", "bin", "labeling"], observed=True).nll.mean().unstack("labeling")
    return (m.tagged - m.plain).rename(stage)


def main():
    fam = sys.argv[1] if len(sys.argv) > 1 else "olmo3-7b"
    x = pd.concat([frame(fam, "base"), frame(fam, "instruct")], axis=1).dropna()
    x["D"] = x.instruct - x.base
    t = x.groupby(["source", "cell", "bin"], observed=True).D.mean().unstack("bin").round(2)
    md = (f"# {fam}: change from base of the frame cost (tagged - plain), by content-position bin\n\n"
          "Mean over conversations that reach the bin, nats per token; descriptive, no CIs.\n\n" + t.to_markdown() + "\n")
    open(os.path.join(HERE, "results", fam, "positions.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
