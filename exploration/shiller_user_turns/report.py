"""Gallery, flag-rate table and hand-label sheet from the per-model sample files.

    python -m exploration.shiller_user_turns.report               # gallery.md, flag_rates.csv, hand_labels.csv
    python -m exploration.shiller_user_turns.report reflag        # recompute flags in <model>.jsonl (after editing flags.py)
    python -m exploration.shiller_user_turns.report agreement     # flags vs filled-in hand_labels.csv

hand_labels.csv is drawn once (50 random completions over every model present) and is never overwritten
once any label is filled in; pass --redraw to replace an unlabeled sheet (e.g. after more models ran).
Flags are deliberately not in the sheet so labeling is blind; `agreement` joins them back by key.
"""

import argparse
import csv
import glob
import json
import os
import random

import pandas as pd

from core.render import load_jsonl
from exploration.shiller_user_turns import flags as F
from exploration.shiller_user_turns import models, prompts
from exploration.shiller_user_turns.generate import OUT

LABELS = ["user voice", "switched to assistant", "degenerate", "other"]
KEY = ["model", "condition", "prompt_id", "sample"]
SHOWN = ["ended_turn", "handoff_to_assistant", "assistant_register", "assistant_register_unmarked",
         "assistant_thinking", "repetition_loop", "token_loop"]


def model_files(out):
    order = {k: i for i, k in enumerate(models.MODELS)}
    paths = [p for p in glob.glob(os.path.join(out, "*.jsonl")) if not p.endswith("_meta.jsonl")]
    keys = [os.path.basename(p)[:-6] for p in paths]
    return sorted(keys, key=lambda k: (order.get(k, len(order)), k))


def load_all(out):
    keys = model_files(out)
    if not keys:
        raise SystemExit(f"no <model>.jsonl files in {out}")
    rows = [r for k in keys for r in load_jsonl(os.path.join(out, f"{k}.jsonl"))]
    return keys, pd.DataFrame(rows)


def flag_rates(df):
    names = [f for f in F.FLAG_NAMES if f in df]
    g = df.groupby(["model", "condition"], sort=False)
    t = g[names].mean().round(2)
    t.insert(0, "n", g.size())
    t.insert(1, "precision", g.precision.first())
    return t.reset_index()


def gallery(keys, df, out, per_condition, seed):
    meta = {k: json.load(open(os.path.join(out, f"{k}_meta.json"))) for k in keys
            if os.path.exists(os.path.join(out, f"{k}_meta.json"))}
    rates = flag_rates(df)
    order = [c for c in prompts.CONDITIONS if c in set(df.condition)]
    lines = [
        "# Open user turns: gallery",
        "",
        "Qualitative reproduction of the user-turn observations in the \"Cursory Exploration\" section of Shiller, "
        "\"Privilege, Dominance, and Personas\" (Eleos, Aug 2026). **Exploratory: n is small (5 samples per prompt, "
        "10-20 prompts per condition, 1 prompt in `empty`), the flags are regex heuristics not yet checked against "
        "hand labels, and nothing here enters any measure's verdict.** Base models are rendered with their family's "
        "instruct chat template. Contamination is not checked for Gemma or Qwen.",
        "",
        "Completions are shown verbatim with special tokens kept, one block per model; the prompt ended exactly "
        "where the prefill ends (exact rendered prompts are in `<model>_meta.json`). Generation stopped only at end of text, so `<turn|>` / `<|im_end|>` and any assistant turn after it "
        "are the model's own.",
        "",
        "## Models",
        "",
        "| model | repo | role | precision | template options | stop |",
        "|---|---|---|---|---|---|",
    ]
    for k in keys:
        m = meta.get(k, {})
        rec = m.get("model", {})
        lines.append(f"| {k} | `{rec.get('repo', '?')}` | {rec.get('role', '?')} | {m.get('precision', '?')} | "
                     f"`{json.dumps(m.get('template_kwargs', {}))}` (template defaults) | "
                     f"{', '.join(m.get('generation', {}).get('stop_tokens', []))} |")
    lines += ["", "## Flag rates by model × condition", "",
              "Fractions of samples. Definitions in `exploration/shiller_user_turns/flags.py`.", "",
              rates.to_markdown(index=False), ""]

    rng = random.Random(seed)
    for cond in order:
        d = df[df.condition == cond]
        ids = list(dict.fromkeys(d.prompt_id))
        picks = ([(i, 0) for i in sorted(rng.sample(ids, min(per_condition, len(ids))), key=ids.index)]
                 if len(ids) >= per_condition else [(ids[0], s) for s in range(per_condition)])
        lines += [f"## Condition: {cond}", "", prompts.DESCRIPTIONS[cond], ""]
        for pid, s in picks:
            sub = d[(d.prompt_id == pid) & (d["sample"] == s)]
            if sub.empty:
                continue
            pre = sub.iloc[0].prefill
            lines += [f"### {pid}, sample {s}", "", f"Prefill: {'`' + pre + '`' if pre else '(none)'}", ""]
            for k in keys:
                r = sub[sub.model == k]
                if r.empty:
                    continue
                r = r.iloc[0]
                on = ", ".join(f for f in SHOWN if r.get(f)) or "no flags"
                lines += [f"**{k}** ({on})", "", "````text", r.completion, "````", ""]
    with open(os.path.join(out, "gallery.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    rates.to_csv(os.path.join(out, "flag_rates.csv"), index=False)


def hand_labels(df, out, n, seed, redraw):
    path = os.path.join(out, "hand_labels.csv")
    if os.path.exists(path):
        old = pd.read_csv(path, keep_default_na=False)
        if (old.perspective != "").any():
            print(f"{path} has labels; left unchanged")
            return
        if not redraw:
            print(f"{path} exists (unlabeled); pass --redraw to draw a new sample")
            return
    s = df.sample(n=min(n, len(df)), random_state=seed)
    s = s.assign(perspective="", notes="")[[*KEY, "prefill", "completion", "perspective", "notes"]]
    s.insert(0, "label_id", [f"h{i:02d}" for i in range(len(s))])
    s.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"wrote {path} ({len(s)} rows; perspective one of: {' / '.join(LABELS)})")


def predicted(r):
    if r.repetition_loop or r.token_loop:
        return "degenerate"
    if r.assistant_register or r.assistant_thinking or r.plain_role_marker:
        return "switched to assistant"
    return "user voice"


def agreement(df, out):
    lab = pd.read_csv(os.path.join(out, "hand_labels.csv"), keep_default_na=False)
    lab = lab[lab.perspective != ""]
    if lab.empty:
        raise SystemExit("no labels filled in yet")
    bad = set(lab.perspective) - set(LABELS)
    if bad:
        raise SystemExit(f"unknown labels {bad}; use {LABELS}")
    m = lab.merge(df, on=KEY, suffixes=("", "_row"))
    m["predicted"] = m.apply(predicted, axis=1)
    tab = pd.crosstab(m.perspective, m.predicted, margins=True)
    acc = (m.perspective == m.predicted).mean()
    lines = ["# Flags vs hand labels", "", f"{len(m)} labeled samples; agreement {acc:.0%}.", "",
             "Predicted label: degenerate if repetition_loop or token_loop, else switched to assistant if "
             "assistant_register, assistant_thinking or plain_role_marker, else user voice.", "",
             tab.to_markdown(), "", "Per flag, share of samples carrying it among each hand label:", "",
             m.groupby("perspective")[[f for f in F.FLAG_NAMES if f in m]].mean().round(2).to_markdown(), ""]
    with open(os.path.join(out, "agreement.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


def reflag(keys, out):
    for k in keys:
        meta = json.load(open(os.path.join(out, f"{k}_meta.json")))
        g = meta["generation"]
        path = os.path.join(out, f"{k}.jsonl")
        rows = load_jsonl(path)
        for r in rows:
            up, fl, found = F.flags(r["completion"], r["completion_ids"], meta["markers"], g["stop_tokens"],
                                    set(g["stop_ids"]), set(meta["marker_ids"]), meta["specials"], r["stopped_on_eos"])
            r.update(user_portion=up, special_tokens_found=found, **fl)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"reflagged {path}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default="report", choices=["report", "reflag", "agreement"])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--per-condition", type=int, default=3, help="gallery examples per condition")
    ap.add_argument("--n-labels", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--redraw", action="store_true", help="replace an unlabeled hand_labels.csv")
    args = ap.parse_args(argv)
    keys, df = load_all(args.out)
    if args.command == "reflag":
        reflag(keys, args.out)
    elif args.command == "agreement":
        agreement(df, args.out)
    else:
        gallery(keys, df, args.out, args.per_condition, args.seed)
        hand_labels(df, args.out, args.n_labels, args.seed, args.redraw)
        print(f"wrote {os.path.join(args.out, 'gallery.md')} and flag_rates.csv ({len(df)} samples, {len(keys)} models)")


if __name__ == "__main__":
    main()
