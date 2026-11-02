"""Within-family change from base for one model family, paired over texts; applies the pre-committed criterion-1 rule.

    python -m measures.vestigial_regression.pilot.nll_arm.compare --family gemma --tokenizer google/gemma-4-31B-it \
        --scores nll_arm/scores [--base base --it instruct]

Reads <scores>/<base>.csv + _tokens.parquet and <scores>/<it>.csv + _tokens.parquet (same 90 texts, same conditions),
writes compare_<family>.md, bpb_change_<family>.csv (per-text BPB change of each role-specific term, for the
cross-family comparison), and position_change_<family>_<set>.png.

Rule (report.md, written before any result): the proxy flags the family if user@user - user@plain (mean NLL over
content positions 1+) rises from base to IT by >= config.MIN_EFFECT_NATS with a paired 95% bootstrap CI excluding zero,
in both user sources. Everything else here (assistant-text term, slot effects, positions, BPB) is descriptive.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import config
from core.render import load_jsonl, load_tokenizer
from core.stats import bootstrap_ci
from measures.vestigial_regression.pilot.nll_arm.diagnose import BINS, N_POS, bpb_table
from measures.vestigial_regression.pilot.nll_arm.score import SETS, select

HERE = os.path.dirname(os.path.abspath(__file__))
N_BOOT = 5000
TERMS = {"user": [("role-specific (user@user - user@plain)", "user", "plain"),
                  ("slot effect (user@user - user@asst)", "user", "asst")],
         "assistant": [("role-specific (asst@asst - asst@plain)", "asst", "plain"),
                       ("slot effect (asst@user - asst@asst)", "user", "asst")]}


def fmt(x, rng):
    m, lo, hi = bootstrap_ci(np.asarray(x.dropna(), dtype=float), N_BOOT, rng)
    return m, lo, hi, f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def per_text(df, source, kind, slot):
    ab = "user" if kind == "user" else "asst"
    d = df[(df.source == source) & (df.kind == kind) & (df.condition == f"{ab}@{slot}")]
    return d.set_index("text_id").mean_nll_from1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--scores", default=os.path.join(HERE, "scores"))
    ap.add_argument("--base", default="base")
    ap.add_argument("--it", default="instruct")
    ap.add_argument("--n-per-set", type=int, default=30)
    args = ap.parse_args()
    rng = np.random.default_rng(config.SEED)

    sc = {k: pd.read_csv(os.path.join(args.scores, f"{s}.csv")) for k, s in [("base", args.base), ("it", args.it)]}
    tk = {k: pd.read_parquet(os.path.join(args.scores, f"{s}_tokens.parquet")) for k, s in [("base", args.base), ("it", args.it)]}
    tok = load_tokenizer(args.tokenizer)
    texts = {r["id"]: r["text"] for r in select(load_jsonl(config.DATA_PATH), args.n_per_set)}
    assert set(sc["base"].text_id) == set(sc["it"].text_id), "base and IT must cover the same texts"

    L = [f"# {args.family}: change from own base (IT - base), paired over texts", "",
         f"Tokenizer/template: {args.tokenizer} for both checkpoints. Mean NLL over content positions 1+ (nats/token); "
         "95% paired bootstrap CIs over texts. Rule applies to the first row of each user set.", ""]
    verdict, rows_bpb = {}, []
    bpb = {k: bpb_table(tk[k], tok, texts, rng)[1] for k in ("base", "it")}

    L += ["## Raw per-slot mean NLL (nats/token, positions 1+)", "", "| text set | model | @user | @asst | @plain |", "|---|---|---|---|---|"]
    for source, kind in SETS:
        for k in ("base", "it"):
            cells = [f"{per_text(sc[k], source, kind, s).mean():.3f}" for s in ("user", "asst", "plain")]
            L.append(f"| {source} {kind} | {'base' if k == 'base' else 'IT'} | " + " | ".join(cells) + " |")
    L.append("")

    for source, kind in SETS:
        L += [f"## {source} {kind}", "", "| quantity | base | IT | change (IT - base), nats | change, bits/byte |", "|---|---|---|---|---|"]
        for name, a, b in TERMS[kind]:
            vals = {k: per_text(sc[k], source, kind, a) - per_text(sc[k], source, kind, b) for k in ("base", "it")}
            ch = vals["it"] - vals["base"]
            m, lo, hi, s_nat = fmt(ch, rng)
            bch = {}
            for k in ("base", "it"):
                bch[k] = bpb[k][(source, kind, a, "primary")] - bpb[k][(source, kind, b, "primary")]
            bd = bch["it"] - bch["base"]
            _, _, _, s_bpb = fmt(bd, rng)
            L.append(f"| {name} | {vals['base'].mean():+.3f} | {vals['it'].mean():+.3f} | {s_nat} | {s_bpb} |")
            if name.startswith("role-specific"):
                rows_bpb += [dict(source=source, kind=kind, term=name, text_id=i, bpb_change=v) for i, v in bd.items()]
                verdict[(source, kind)] = (m, lo, hi)
        L.append("")

    # Pre-committed rule.
    ok = {}
    for source, kind in SETS[:2]:
        m, lo, hi = verdict[(source, kind)]
        ok[source] = m >= config.MIN_EFFECT_NATS and lo > 0
    met = all(ok.values())
    L += ["## Pre-committed rule (criterion 1 against the family's own base)", "",
          "user@user - user@plain rises by >= 0.05 nats with the CI excluding zero, in both user sources: "
          + "; ".join(f"{s}: {'yes' if v else 'no'} ({verdict[(s, 'user')][0]:+.3f} [{verdict[(s, 'user')][1]:+.3f}, "
                      f"{verdict[(s, 'user')][2]:+.3f}])" for s, v in ok.items())
          + f". **Rule {'MET' if met else 'NOT MET'}.**", ""]
    am, alo, ahi = verdict[SETS[2]]
    um = np.mean([verdict[(s, 'user')][0] for s, _ in SETS[:2]])
    L += [f"Assistant-text role-specific change (descriptive): {am:+.3f} [{alo:+.3f}, {ahi:+.3f}] against a mean user-text change "
          f"of {um:+.3f}.", ""]

    # Position analysis: role-specific term by position bin and per position, base vs IT.
    L += ["## Position analysis (role-specific term by content position, texts with >= 40 tokens)", ""]
    for source, kind in SETS:
        ab = "user" if kind == "user" else "asst"
        ids = tk["it"][(tk["it"].source == source) & (tk["it"].kind == kind) & (tk["it"].position == N_POS - 1)].text_id.unique()
        ids = [i for i in ids if i in set(tk["base"][tk["base"].position == N_POS - 1].text_id)]
        prof = {}
        for k in ("base", "it"):
            for name_, cond in (("own", f"{ab}@{ab}"), ("plain", f"{ab}@plain")):
                x = tk[k][(tk[k].source == source) & (tk[k].kind == kind) & (tk[k].condition == cond)]
                x = x[x.text_id.isin(ids) & (x.position < N_POS)]
                prof[(k, name_)] = x.pivot(index="text_id", columns="position", values="nll").loc[sorted(ids)].to_numpy()
        L += [f"### {source} {kind} ({len(ids)} texts)", "", "| positions | base: own - plain | IT: own - plain | IT - base |", "|---|---|---|---|"]
        for name, a0, b0 in BINS:
            rb = (prof[("base", "own")][:, a0:b0] - prof[("base", "plain")][:, a0:b0]).mean(axis=1)
            ri = (prof[("it", "own")][:, a0:b0] - prof[("it", "plain")][:, a0:b0]).mean(axis=1)
            L.append(f"| {name} | {rb.mean():+.2f} | {ri.mean():+.2f} | {(ri - rb).mean():+.2f} |")
        L.append("")
        fig, ax = plt.subplots(figsize=(6, 3.4))
        for k, col in (("base", "#7f8c8d"), ("it", "#c0392b")):
            ax.plot(range(N_POS), (prof[(k, "own")] - prof[(k, "plain")]).mean(axis=0), label=("base" if k == "base" else "IT"), color=col)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("content position")
        ax.set_ylabel(f"{ab}@{ab} - {ab}@plain (nats)")
        ax.set_title(f"{args.family}: {source} {kind} (n={len(ids)})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, f"position_change_{args.family}_{source}_{kind}.png"), dpi=120)
        plt.close(fig)

    pd.DataFrame(rows_bpb).to_csv(os.path.join(HERE, f"bpb_change_{args.family}.csv"), index=False)
    with open(os.path.join(HERE, f"compare_{args.family}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
