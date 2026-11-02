"""Step-1 diagnostics for one scored checkpoint: raw per-slot NLL, rendered examples, position profile, bits per byte.

    python -m measures.vestigial_regression.pilot.nll_arm.diagnose --stage instruct [--tokenizer google/gemma-4-31B-it]

No GPU. Reads nll_arm/scores/<stage>.csv and <stage>_tokens.parquet (made by score.py), re-renders the texts with
core's own rendering to check them, and writes nll_arm/diagnostics_<tag>.md and position_<tag>_<set>.png.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import config
from core.render import load_jsonl, load_tokenizer, render_all
from core.stats import bootstrap_ci
from measures.vestigial_regression.pilot.nll_arm.score import SETS, select

HERE = os.path.dirname(os.path.abspath(__file__))
SLOT_ORDER = ["user", "asst", "plain"]
N_POS = 40
BINS = [("0", 0, 1), ("1-4", 1, 5), ("5-19", 5, 20), ("20-39", 20, 40)]


def ci_str(x, rng):
    m, lo, hi = bootstrap_ci(np.asarray(x, dtype=float), 2000, rng)
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def raw_table(scores, rng):
    L = ["| text set | " + " | ".join(f"{k}@{s}" for k in ["own"] for s in SLOT_ORDER) + " |", ""]
    L = ["| text set | quantity | @user | @asst | @plain |", "|---|---|---|---|---|"]
    for source, kind in SETS:
        d = scores[(scores.source == source) & (scores.kind == kind)]
        ab = "user" if kind == "user" else "asst"
        for col, name in [("mean_nll_from1", "nats/token, positions 1+"), ("mean_nll", "nats/token, all positions")]:
            cells = [ci_str(d[d.condition == f"{ab}@{s}"][col], rng) for s in SLOT_ORDER]
            L.append(f"| {source} {kind} | {name} | " + " | ".join(cells) + " |")
    return L


def render_checks(tok, rows):
    """bos presence per condition, and 3 examples per condition with special tokens visible."""
    stats, examples = {}, {}
    for r in rows:
        for cond, v in render_all(tok, r).items():
            ids = v.input_ids
            n_bos = sum(1 for i in ids if i == tok.bos_token_id)
            s = stats.setdefault(cond, dict(n=0, starts_bos=0, bos_once=0))
            s["n"] += 1
            s["starts_bos"] += ids[0] == tok.bos_token_id
            s["bos_once"] += n_bos == 1
            ex = examples.setdefault(cond, [])
            if len(ex) < 3:
                ex.append((r["id"], tok.decode(ids[: v.start]), tok.decode(ids[v.start : v.start + 12]),
                           tok.decode(ids[v.end :])))
    return stats, examples


def template_facts(tok):
    """How the template handles system turns, role names and thinking, from the template itself."""
    out = {}
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U"}]
    out["system+user"] = tok.apply_chat_template(msgs, tokenize=False)
    out["user+gen_prompt"] = tok.apply_chat_template(msgs[1:], tokenize=False, add_generation_prompt=True)
    out["user,assistant"] = tok.apply_chat_template(
        [{"role": "user", "content": "U"}, {"role": "assistant", "content": "A"}], tokenize=False)
    out["assistant only"] = tok.apply_chat_template([{"role": "assistant", "content": "A"}], tokenize=False)
    try:
        out["user+gen_prompt, enable_thinking=True"] = tok.apply_chat_template(
            msgs[1:], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    except Exception as e:  # noqa: BLE001
        out["user+gen_prompt, enable_thinking=True"] = f"(error: {e})"
    return out


def position_profile(tokens, source, kind, rng):
    """{slot: array [texts, N_POS]} over the texts with >= N_POS content tokens (every position, same texts)."""
    d = tokens[(tokens.source == source) & (tokens.kind == kind)]
    ab = "user" if kind == "user" else "asst"
    long_ids = d[d.position == N_POS - 1].text_id.unique()
    out = {}
    for s in SLOT_ORDER:
        x = d[(d.condition == f"{ab}@{s}") & d.text_id.isin(long_ids) & (d.position < N_POS)]
        piv = x.pivot(index="text_id", columns="position", values="nll").loc[sorted(long_ids)]
        out[s] = piv.to_numpy()
    return list(sorted(long_ids)), out


def bpb_table(tokens, tok, texts, rng):
    """Bits per byte per condition. Primary: positions 1+, bytes of the text decoded from content tokens 1+.
    Secondary: all positions, bytes of the whole text."""
    nbytes_all = {i: len(t.encode("utf-8")) for i, t in texts.items()}
    nbytes_1 = {}
    for i, t in texts.items():
        ids = tok(t, add_special_tokens=False)["input_ids"]
        nbytes_1[i] = len(tok.decode(ids[1:], clean_up_tokenization_spaces=False).encode("utf-8"))
    g = tokens.groupby(["source", "kind", "condition", "text_id"], observed=True)
    tot_all = g.nll.sum()
    tot_1 = tokens[tokens.position >= 1].groupby(["source", "kind", "condition", "text_id"], observed=True).nll.sum()
    L = ["| text set | BPB | @user | @asst | @plain |", "|---|---|---|---|---|"]
    per_text = {}
    for source, kind in SETS:
        ab = "user" if kind == "user" else "asst"
        for name, tot, nb in [("primary (positions 1+)", tot_1, nbytes_1), ("secondary (all positions)", tot_all, nbytes_all)]:
            cells = []
            for s in SLOT_ORDER:
                x = tot.loc[(source, kind, f"{ab}@{s}")]
                v = pd.Series({i: x[i] / np.log(2) / nb[i] for i in x.index})
                per_text[(source, kind, s, name.split()[0])] = v
                cells.append(ci_str(v, rng))
            L.append(f"| {source} {kind} | {name} | " + " | ".join(cells) + " |")
    return L, per_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--tag", help="label for output files (default: stage)")
    ap.add_argument("--tokenizer", default="google/gemma-4-31B-it")
    ap.add_argument("--scores", default=os.path.join(HERE, "scores"))
    ap.add_argument("--n-per-set", type=int, default=30)
    args = ap.parse_args()
    tag = args.tag or args.stage
    rng = np.random.default_rng(config.SEED)

    scores = pd.read_csv(os.path.join(args.scores, f"{args.stage}.csv"))
    tokens = pd.read_parquet(os.path.join(args.scores, f"{args.stage}_tokens.parquet"))
    tok = load_tokenizer(args.tokenizer)
    rows = select(load_jsonl(config.DATA_PATH), args.n_per_set)
    texts = {r["id"]: r["text"] for r in rows}

    L = [f"# Diagnostics: {tag} ({args.tokenizer} rendering)", "",
         "## 1a. Raw mean NLL per slot (no subtraction, no reference model)", "", *raw_table(scores, rng), ""]

    stats, examples = render_checks(tok, rows)
    L += ["## 1b. Rendering", "", "`<bos>` in the token ids (all 90 texts, every condition):", "",
          "| condition | sequences | start with `<bos>` | exactly one `<bos>` |", "|---|---|---|---|"]
    for c, s in sorted(stats.items()):
        L.append(f"| {c} | {s['n']} | {s['starts_bos']} | {s['bos_once']} |")
    L += ["", "Template behaviour (special tokens visible):", ""]
    for k, v in template_facts(tok).items():
        L += [f"- {k}: `{v!r}`"]
    L += ["", "Three rendered examples per condition (prefix | first 12 content tokens | suffix; special tokens kept):", ""]
    for c, ex in sorted(examples.items()):
        for tid, pre, head, suf in ex:
            L.append(f"- `{c}` {tid}: `{pre!r}` | `{head!r}` | `{suf!r}`")
    L.append("")

    L += ["## 1c. Position profile (mean NLL, nats/token, by content position)", "",
          "Texts with at least 40 content tokens, so every position averages over the same texts.", ""]
    for source, kind in SETS:
        ids, prof = position_profile(tokens, source, kind, rng)
        ab = "user" if kind == "user" else "asst"
        L += [f"### {source} {kind} ({len(ids)} texts with >= {N_POS} tokens)", "",
              "| positions | " + " | ".join(f"{ab}@{s}" for s in SLOT_ORDER) + " | own slot - plain | other slot - plain |",
              "|---|---|---|---|---|---|"]
        other = "asst" if kind == "user" else "user"
        for name, a, b in BINS:
            m = {s: prof[s][:, a:b].mean(axis=1) for s in SLOT_ORDER}
            own = m[ab.replace("asst", "asst")]
            L.append(f"| {name} | " + " | ".join(f"{m[s].mean():.2f}" for s in SLOT_ORDER)
                     + f" | {(m[ab] - m['plain']).mean():+.2f} | {(m[other] - m['plain']).mean():+.2f} |")
        L.append("")
        fig, ax = plt.subplots(figsize=(6, 3.4))
        for s, col in zip(SLOT_ORDER, ["#c0392b", "#2471a3", "#7f8c8d"]):
            ax.plot(range(N_POS), prof[s].mean(axis=0), label=f"{ab}@{s}", color=col)
        ax.set_xlabel("content position")
        ax.set_ylabel("mean NLL (nats)")
        ax.set_title(f"{tag}: {source} {kind} (n={len(ids)})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, f"position_{tag}_{source}_{kind}.png"), dpi=120)
        plt.close(fig)

    bl, per_text = bpb_table(tokens, tok, texts, rng)
    L += ["## 1d. Bits per byte", "", *bl, ""]
    with open(os.path.join(HERE, f"diagnostics_{tag}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
