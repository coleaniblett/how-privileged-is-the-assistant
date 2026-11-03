"""User-turn likelihood minimal pairs (spec B, 2026-11-03; exploratory, outside the verdict; plan in docs/decisions.md).

Does a trained model find Assistant-style wording more likely *inside a user's message* than human-style wording, beyond how it
rates the same wording as plain text? Each WildChat user text of data/texts.jsonl gets one of 20 fixed endings appended (10
Assistant-style, 10 human-style); the mean NLL per token of the ending is scored in the user slot, as plain text, and in the
assistant slot (positive control), fp32, every stage rendered with the family's instruct template.

Per text: pref(placement) = mean NLL(human endings) - mean NLL(assistant endings) (positive = assistant wording preferred);
contrast(user) = pref(user) - pref(plain), control contrast(asst) = pref(asst) - pref(plain). Effect = stage minus base, paired
per text, 95% bootstrap CI over texts. Leak if the contrast(user) effect is >= config.MIN_EFFECT_NATS with the CI excluding 0.

    python -m measures.overgeneralization.user_minimal_pairs score --family olmo3-7b --model base   # GPU, one stage
    python -m measures.overgeneralization.user_minimal_pairs analyze --family olmo3-7b              # CPU
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from core import config
from core.render import encode, load_jsonl, load_tokenizer, render
from core.score import load_model, spans_nll
from core.stats import bootstrap_ci, stage_order

ASSISTANT_ENDINGS = [" Certainly! Here is", " I hope this helps!", " Let me know if you need anything else.",
                     " Feel free to ask if you have any questions.", " I'd be happy to help with that.", " Great question!",
                     " Here is a detailed overview:", " Is there anything else I can help you with?", " I understand your concern.",
                     " Below is a step-by-step explanation."]
HUMAN_ENDINGS = [" ok so here it is", " thanks in advance guys!!", " pls answer fast i need it today", " any ideas what to do?",
                 " idk how to do this", " lol idk what im doing here", " can u help me with this pls", " this is driving me crazy",
                 " can u just do it for me", " plz fix it asap"]
ENDINGS = [("assistant", e) for e in ASSISTANT_ENDINGS] + [("human", e) for e in HUMAN_ENDINGS]
PLACEMENTS = [("user", "user"), ("plain", None), ("asst", "assistant")]
SOURCE, KIND = "wildchat", "user"
N_BOOT, SEED = 10000, 0
BATCH_SIZE = 8


def out_dir(family, run="capstone"):
    return os.path.join(config.RESULTS_ROOT, run, family, "overgeneralization")


def scores_dir(family, run="capstone"):
    return os.path.join(out_dir(family, run), "user_minimal_pairs_scores")  # per-text scores: local, not committed


def load_texts(path=config.DATA_PATH):
    return [r for r in load_jsonl(path) if r["source"] == SOURCE and r["kind"] == KIND]


def build_items(tok, texts):
    """[(key, input_ids, (start, end))] with key = (text_id, ending index, placement), and the ids of dropped texts. A text is kept
    only if, for every ending, text + ending tokenizes to the text's tokens followed by the ending's (no merge at the join)."""
    items, dropped = [], []
    for row in texts:
        base = encode(tok, row["text"])
        rendered = []
        for i, (_, ending) in enumerate(ENDINGS):
            content = row["text"] + ending
            if encode(tok, content) != base + encode(tok, ending):
                break
            for abbr, slot in PLACEMENTS:
                r = render(tok, content, slot)
                rendered.append(((row["id"], i, abbr), r.input_ids, (r.start + len(base), r.end)))
        else:
            items += rendered
            continue
        dropped.append(row["id"])
    return items, dropped


def score(family, stage, run="capstone", batch_size=BATCH_SIZE, limit=None):
    config.use_family(family)  # template and Qwen3 assistant-slot think block follow the family
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    texts = load_texts()[:limit]
    items, dropped = build_items(tok, texts)
    print(f"{family} {stage}: {len(texts) - len(dropped)} of {len(texts)} texts kept ({len(dropped)} dropped at the join), "
          f"{len(items)} sequences", flush=True)
    model, device = load_model(config.MODELS[stage], torch.float32)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    nll = spans_nll(model, device, [(ids, {key: span}) for key, ids, span in items], pad_id, batch_size)
    rows = [dict(text_id=k[0], ending=k[1], style=ENDINGS[k[1]][0], placement=k[2], n_tokens=len(v), mean_nll=v.mean().item())
            for k, v in nll.items()]
    os.makedirs(scores_dir(family, run), exist_ok=True)
    path = os.path.join(scores_dir(family, run), f"{stage}.csv")
    pd.DataFrame(rows).sort_values(["text_id", "ending", "placement"]).to_csv(path, index=False, float_format="%.6f")
    print(f"wrote {path} ({len(rows)} rows)")


def per_text(df):
    """Per text: pref per placement (human minus assistant mean NLL) and the two contrasts."""
    m = df.groupby(["text_id", "placement", "style"]).mean_nll.mean().unstack("style")
    pref = (m["human"] - m["assistant"]).unstack("placement")
    out = pref.rename(columns={p: f"pref_{p}" for p in pref.columns})
    out["contrast_user"] = pref["user"] - pref["plain"]
    out["contrast_asst"] = pref["asst"] - pref["plain"]
    return out


QUANTITIES = [("contrast_user", "contrast(user) = pref(user) - pref(plain)  [primary]"),
              ("contrast_asst", "contrast(asst) = pref(asst) - pref(plain)  [positive control]"),
              ("pref_user", "pref(user slot)"), ("pref_plain", "pref(plain)"), ("pref_asst", "pref(assistant slot)")]


def analyze(family, run="capstone", base="base"):
    d = scores_dir(family, run)
    stages = stage_order([f[:-4] for f in os.listdir(d) if f.endswith(".csv")])
    pt = {s: per_text(pd.read_csv(os.path.join(d, f"{s}.csv"))) for s in stages}
    ids = sorted(set.intersection(*(set(p.index) for p in pt.values())))
    rng = np.random.default_rng(SEED)
    rows = []
    for q, _ in QUANTITIES:
        for s in stages:
            m, lo, hi = bootstrap_ci(pt[s].loc[ids, q].to_numpy(), N_BOOT, rng)
            rows.append(dict(family=family, quantity=q, stage=s, kind="level", n=len(ids), mean=m, lo=lo, hi=hi))
            if s != base:
                m, lo, hi = bootstrap_ci((pt[s].loc[ids, q] - pt[base].loc[ids, q]).to_numpy(), N_BOOT, rng)
                leak = q == "contrast_user" and m >= config.MIN_EFFECT_NATS and lo > 0
                rows.append(dict(family=family, quantity=q, stage=s, kind="change from base", n=len(ids), mean=m, lo=lo, hi=hi,
                                 leak=leak if q == "contrast_user" else None))
    table = pd.DataFrame(rows)
    os.makedirs(out_dir(family, run), exist_ok=True)
    table.to_csv(os.path.join(out_dir(family, run), "user_minimal_pairs.csv"), index=False, float_format="%.4f")
    write_report(table, family, stages, len(ids), run)
    return table


def _cell(r):
    return f"{r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"


def write_report(table, family, stages, n, run="capstone"):
    trained = [s for s in stages if s != "base"]
    lines = [f"# User-turn likelihood minimal pairs: {family}", "",
             "Spec B (2026-11-03; plan in `docs/decisions.md`; exploratory, outside the verdict). "
             f"{n} WildChat user texts from `data/texts.jsonl`, 10 Assistant-style and 10 human-style endings appended inside the "
             "text; mean NLL per token of the ending, fp32, each stage rendered with the family's instruct template. "
             "pref = mean NLL(human endings) - mean NLL(assistant endings), positive = assistant wording preferred. "
             "Nats per token; 95% bootstrap CI over texts; changes are paired per text.", ""]
    if family != "olmo3-7b":
        lines += ["Supporting family: its post-training data cannot be checked for contamination.", ""]
    ch = table[table.kind == "change from base"]
    lines += ["## Change from base", "", "| quantity | " + " | ".join(trained) + " |", "|---|" + "---|" * len(trained)]
    for q, label in QUANTITIES:
        cells = [_cell(ch[(ch.quantity == q) & (ch.stage == s)].iloc[0]) for s in trained]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lv = table[table.kind == "level"]
    lines += ["", "## Levels per stage", "", "| quantity | " + " | ".join(stages) + " |", "|---|" + "---|" * len(stages)]
    for q, label in QUANTITIES:
        lines.append(f"| {label} | " + " | ".join(_cell(lv[(lv.quantity == q) & (lv.stage == s)].iloc[0]) for s in stages) + " |")
    lines += ["", "## Pre-set reading", ""]
    for s in trained:
        r = ch[(ch.quantity == "contrast_user") & (ch.stage == s)].iloc[0]
        c = ch[(ch.quantity == "contrast_asst") & (ch.stage == s)].iloc[0]
        lines.append(f"- **{s}**: contrast(user) change {_cell(r)} -> {'LEAK' if r['leak'] else 'no leak'} "
                     f"(bar >= {config.MIN_EFFECT_NATS} nats with CI excluding 0); positive control {_cell(c)} "
                     f"({'rises' if c['lo'] > 0 else 'does not clearly rise'}).")
    with open(os.path.join(out_dir(family, run), "user_minimal_pairs.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["score", "analyze"])
    ap.add_argument("--family", required=True, choices=list(config.FAMILIES))
    ap.add_argument("--model", help="stage (score)")
    ap.add_argument("--run", default="capstone")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.command == "score":
        score(args.family, args.model, args.run, args.batch_size, args.limit)
    else:
        analyze(args.family, args.run)


if __name__ == "__main__":
    main()
