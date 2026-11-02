"""Cross-family comparison: does the proxy discriminate Gemma (regressed) from Qwen (negative control)?

    python -m measures.vestigial_regression.pilot.nll_arm.cross_family      # after compare.py ran for both families

Pre-committed rules (report.md, written before any result):
  A  Gemma meets the step-2 rule (user@user - user@plain rises >= 0.05 nats, CI excluding zero, in both user sources)
     and Qwen does not meet it under the identical definition.
  B  For BOTH user sources: Gemma's change from its own base in user@user - user@plain, in primary bits per byte,
     exceeds Qwen's, with the CI of the per-text difference (Gemma change - Qwen change, paired over the shared
     texts) excluding zero.
The pilot "discriminates" if A or B is met. Only evaluated if Qwen is a negative control (report.md, step 3).
Reads bpb_change_<family>.csv (per-text BPB changes) and the paired per-text nats changes from the score files.
"""

import os

import numpy as np
import pandas as pd

from core import config
from core.stats import bootstrap_ci

HERE = os.path.dirname(os.path.abspath(__file__))
N_BOOT = 5000
FAMILIES = {"gemma": os.path.join(HERE, "scores"), "qwen": os.path.join(HERE, "scores_qwen")}
USER_SETS = ["wildchat", "oasst1"]


def nats_change(scores_dir, source):
    """Per-text change IT - base of user@user - user@plain (nats/token, positions 1+), indexed by text id."""
    out = {}
    for k, stage in (("base", "base"), ("it", "instruct")):
        d = pd.read_csv(os.path.join(scores_dir, f"{stage}.csv"))
        d = d[(d.source == source) & (d.kind == "user")]
        p = d.pivot(index="text_id", columns="condition", values="mean_nll_from1")
        out[k] = p["user@user"] - p["user@plain"]
    return out["it"] - out["base"]


def ci(x, rng):
    m, lo, hi = bootstrap_ci(np.asarray(x.dropna(), dtype=float), N_BOOT, rng)
    return m, lo, hi


def main():
    rng = np.random.default_rng(config.SEED)
    bpb = {f: pd.read_csv(os.path.join(HERE, f"bpb_change_{f}.csv")) for f in FAMILIES}
    L = ["# Cross-family comparison: Gemma 4 31B vs Qwen3-8B (change from own base)", "",
         "Per-token nats are compared within a family only; the cross-family rule uses primary bits per byte. "
         "Differences are paired over the 30 texts per source that both families were scored on.", "",
         "| user source | family | nats/token change | bits/byte change |", "|---|---|---|---|"]
    meets, bpb_diff = {f: [] for f in FAMILIES}, {}
    for src in USER_SETS:
        ch = {f: nats_change(d, src) for f, d in FAMILIES.items()}
        b = {f: bpb[f][(bpb[f].source == src) & (bpb[f].term.str.startswith("role-specific")) & (bpb[f].kind == "user")]
             .set_index("text_id").bpb_change for f in FAMILIES}
        for f in FAMILIES:
            m, lo, hi = ci(ch[f], rng)
            mb, lob, hib = ci(b[f], rng)
            meets[f].append(bool(m >= config.MIN_EFFECT_NATS and lo > 0))
            L.append(f"| {src} | {f} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | {mb:+.3f} [{lob:+.3f}, {hib:+.3f}] |")
        shared = b["gemma"].index.intersection(b["qwen"].index)
        d = b["gemma"].loc[shared] - b["qwen"].loc[shared]
        bpb_diff[src] = (len(shared), *ci(d, rng))
    L += ["", "## Rule A: Gemma meets criterion 1 and Qwen does not", "",
          f"Gemma meets it in both user sources: {all(meets['gemma'])} ({meets['gemma']}). "
          f"Qwen meets it in both user sources: {all(meets['qwen'])} ({meets['qwen']}). "
          f"**Rule A {'MET' if all(meets['gemma']) and not all(meets['qwen']) else 'NOT MET'}.**", "",
          "## Rule B: Gemma's bits/byte change exceeds Qwen's, CI of the paired difference excluding zero, both sources", "",
          "| user source | texts | Gemma change - Qwen change (bits/byte) |", "|---|---|---|"]
    okB = []
    for src, (n, m, lo, hi) in bpb_diff.items():
        okB.append(bool(m > 0 and lo > 0))
        L.append(f"| {src} | {n} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    L += ["", f"**Rule B {'MET' if all(okB) else 'NOT MET'}.**", "",
          f"## The pilot {'discriminates' if (all(meets['gemma']) and not all(meets['qwen'])) or all(okB) else 'does not discriminate'}"
          " (A or B)", ""]
    with open(os.path.join(HERE, "cross_family.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
