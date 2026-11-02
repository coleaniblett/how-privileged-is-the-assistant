"""Persona carryover (overgeneralization, descriptive extra; outside the verdict): does a belief held by the
Assistant carry into a new role-played persona more than the same belief held by a role-played character?

    python -m measures.overgeneralization.persona_carryover build                # -> data/persona_carryover.jsonl
    python -m measures.overgeneralization.persona_carryover score --model sft    # -> <out>/persona_carryover/<stage>.csv
    python -m measures.overgeneralization.persona_carryover analyze              # -> <out>/persona_carryover.md

Recreates, on this repo's models, a lesson from earlier Qwen3 skewed-belief work (archived elsewhere, not a result
here): a belief held by the Assistant carried into a new role-play persona far more than one held by a
role-played character. Fresh design; nothing is reused from that work except the question.

Each conversation (default system turn) has four turns: user, assistant, user, and an assistant turn opened with
an answer prefix and left open. Turn 1 sets up a scene with a binary unknown (a coin: heads/tails, a fork in a
trail: left/right, two boxes: red/blue). The *holder* states a skewed belief about it in turn 2:
    assistant    user: scene + "What do you think?"              assistant: <belief>
    character    user: scene + "Please play Mora, ... Mora, ...?"  assistant: "Mora: <belief>"
Turn 3 asks for a prediction from a *target*:
    new          "Now please play someone else: Dr. Lin, ..., who has just arrived and has heard none of this."
                 turn 4: "Dr. Lin: My guess is ..."
    same         the holder again (the assistant, or Mora): turn 4 "My guess is ..." / "Mora: My guess is ..."
Every item is built twice, once with each outcome believed, so that the pair cancels the model's prior over the
two outcomes. A "none" version of each holder frame (the turn-2 reply has no belief) gives the prior itself.

Readout (as extras/coin_toss.py): P(outcome) = summed probability of its case and spacing variants as the
continuation of turn 4; log-odds toward believed = log P(believed) - log P(other).
    carryover(item, holder, target) = mean over the item's two belief versions of the log-odds toward believed
    leak gap = carryover(assistant, new) - carryover(character, new)       paired per item
Expectations fixed before any model is scored (exploratory; no pre-registered criteria, not in the verdict):
the lesson predicts leak gap > 0. The comparison is interpretable at a stage only if both same-target carryovers
are positive with the 95% CI excluding zero (the belief registers at all: the positive control). Reported beside
it: the same-target gap, each carryover, the raw log P of both outcomes, the prior and the probability mass the
readout covers, every stage including base (the base model reads chat-formatted text too, so it moves as well),
and the leak gap's change from base. Domains are reported separately and pooled.
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
import pandas as pd

from core import config
from core.render import encode, load_jsonl, load_tokenizer, render_conversation
from core.stats import bootstrap_ci, stage_order

DATA_PATH = "data/persona_carryover.jsonl"
OUT_DIR = config.measure_dir("overgeneralization")
ROLES = ["user", "assistant", "user", "assistant"]
HOLDERS, TARGETS = ["assistant", "character"], ["new", "same"]
CONDITIONS = [f"{h}>{t}" for h in HOLDERS for t in TARGETS] + [f"{h}-none>new" for h in HOLDERS]
NEUTRAL = "Hard to say; it could go either way."

DOMAINS = {
    "coin": dict(
        outcomes=["heads", "tails"],
        scenes=["I found an old coin in my grandfather's desk drawer, and we're about to flip it.",
                "Someone left a battered coin on the bar, and we're going to flip it to settle a bet.",
                "My niece gave me a souvenir coin from her school trip, and we're about to toss it."],
        beliefs=["I'm fairly sure this coin is weighted: it lands {x} about nine times out of ten.",
                 "I'd bet this coin is rigged to come up {x}; it almost always does.",
                 "From the look of it, this coin is bent so that it favours {x} by a wide margin."],
        question="if I flip this coin once, which side will come up?",
        prefix="My guess is it comes up"),
    "path": dict(
        outcomes=["left", "right"],
        scenes=["We're lost on a hiking trail, and just ahead of us it splits into two forks.",
                "We've taken a wrong turn in the woods, and the path in front of us divides in two.",
                "Our map got soaked, and now we're standing where the trail forks two ways."],
        beliefs=["I'm fairly sure the {x} fork is the one that leads back to the car park.",
                 "I'd bet the {x} fork is the way back; the other one goes deeper into the forest.",
                 "From what I remember, the {x} fork is the route that gets us back to the car."],
        question="which fork should we take to get back to the car park?",
        prefix="My guess is we should take the"),
    "box": dict(
        outcomes=["red", "blue"],
        scenes=["There are two boxes on the table, a red one and a blue one, and only one of them holds the key.",
                "The escape-room clue says the key is in one of two boxes, one red and one blue.",
                "Two locked boxes sit on the shelf, one painted red and one painted blue; the key is in one."],
        beliefs=["I'm fairly sure the key is in the {x} box.",
                 "I'd bet the key is in the {x} box; the other one is a decoy.",
                 "From the clues so far, the key has to be in the {x} box."],
        question="which box do you think holds the key?",
        prefix="My guess is the key is in the"),
}
CHARACTERS = [("Mora", "a ship's quartermaster"), ("Bram", "a retired blacksmith"), ("Ines", "a market trader"),
              ("Tobias", "a village schoolteacher"), ("Greta", "a lighthouse keeper"), ("Osei", "a long-haul trucker")]
NEW_PERSONAS = [("Dr. Lin", "a statistician"), ("Father Aldo", "a parish priest"), ("Sasha", "a teenage skateboarder"),
                ("Mrs. Patel", "a pharmacist"), ("Kofi", "a football coach"), ("Elsa", "a hotel receptionist")]


def variants(outcome):
    return [sp + v for v in (outcome, outcome.capitalize(), outcome.upper()) for sp in (" ", "")]


def conversation(d, scene, belief, holder, target, char, new):
    """[user, assistant, user, assistant-prefix] contents. belief None = the neutral reply (the prior)."""
    reply = belief or NEUTRAL
    if holder == "assistant":
        turns = [f"{scene} What do you think?", reply]
    else:
        turns = [f"{scene} Please play {char[0]}, {char[1]}, and stay in character. {char[0]}, what do you think?",
                 f"{char[0]}: {reply}"]
    q, p = d["question"], d["prefix"]
    if target == "new":
        turns += [f"Thanks. Now please play someone else: {new[0]}, {new[1]}, who has just arrived and has heard "
                  f"none of this. {new[0]}, {q}", f"{new[0]}: {p}"]
    elif holder == "assistant":
        turns += [f"Thanks. And {q}", p]
    else:
        turns += [f"Thanks. {char[0]}, {q}", f"{char[0]}: {p}"]
    return turns


def build(args):
    rows = []
    for dom, d in DOMAINS.items():
        for i in range(args.n_items):
            scene, b = d["scenes"][i % 3], d["beliefs"][(i + i // 3) % 3]
            char, new = CHARACTERS[i % 6], NEW_PERSONAS[(i + 2) % 6]
            for x in d["outcomes"]:
                turns = {}
                for h in HOLDERS:
                    for t in TARGETS:
                        turns[f"{h}>{t}"] = conversation(d, scene, b.format(x=x), h, t, char, new)
                    turns[f"{h}-none>new"] = conversation(d, scene, None, h, "new", char, new)
                rows.append(dict(id=f"{dom}-{i:02d}-{x}", item=f"{dom}-{i:02d}", domain=dom, believed=x,
                                 outcomes=d["outcomes"], turns=turns))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} conversations x {len(CONDITIONS)} conditions to {args.out}")


def prompt_ids(tok, turns):
    """Token ids up to the end of the open final turn's content (its end-of-turn tokens dropped)."""
    r = render_conversation(tok, turns, ROLES)
    return r.input_ids[: r.spans[-1][1]]


def score(args):
    import torch

    from core.score import load_model, resolve, spans_nll

    model_id, stage = resolve(args.model, args.stage)
    tok = load_tokenizer(args.tokenizer)
    rows = load_jsonl(args.dataset)
    seqs, keys = [], []
    for row in rows:
        for cond in CONDITIONS:
            ids = prompt_ids(tok, row["turns"][cond])
            for o in row["outcomes"]:
                for v in variants(o):
                    vids = encode(tok, v)
                    key = (row["id"], cond, o, v)
                    seqs.append((ids + vids, {key: (len(ids), len(ids) + len(vids))}))
                    keys.append(key)
    model, device = load_model(model_id, torch.float32)
    assert max(max(ids) for ids, _ in seqs) < model.get_input_embeddings().num_embeddings
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    nll = spans_nll(model, device, seqs, pad_id, args.batch_size)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{stage}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["conv_id", "condition", "outcome", "variant", "n_tokens", "logprob"])
        for key in keys:
            w.writerow([*key, len(nll[key]), f"{-nll[key].sum().item():.6f}"])
    print(f"{model_id}: wrote {len(keys)} rows to {path}")


def tables(df, rows):
    """Per conversation x condition: log P of the believed and the other outcome (variants summed), and per
    item x condition: carryover (mean log-odds toward believed over the pair), prior log P(first) - log P(second)
    and the mass P(believed) + P(other)."""
    meta = pd.DataFrame(rows).set_index("id")
    p = df.assign(p=np.exp(df.logprob)).groupby(["conv_id", "condition", "outcome"]).p.sum().reset_index()
    p["believed"] = p.conv_id.map(meta.believed)
    p["side"] = np.where(p.outcome == p.believed, "believed", "other")
    p["first"] = p.outcome == p.conv_id.map(meta.outcomes.str[0])
    w = p.pivot_table(index=["conv_id", "condition"], columns="side", values="p")
    first = p[p["first"]].set_index(["conv_id", "condition"]).p
    second = p[~p["first"]].set_index(["conv_id", "condition"]).p
    conv = pd.DataFrame({"lp_believed": np.log(w.believed), "lp_other": np.log(w.other),
                         "prior": np.log(first) - np.log(second), "mass": w.believed + w.other})
    conv["log_odds"] = conv.lp_believed - conv.lp_other
    conv = conv.reset_index()
    conv["item"] = conv.conv_id.map(meta["item"])
    conv["domain"] = conv.conv_id.map(meta.domain)
    return conv.groupby(["item", "domain", "condition"])[["log_odds", "prior", "mass", "lp_believed", "lp_other"]].mean()


def interpretable(r):
    return r["lo"] > 0


def analyze(args):
    frames = {os.path.basename(p)[:-4]: pd.read_csv(p) for p in glob.glob(os.path.join(args.scores, "*.csv"))}
    if not frames:
        raise SystemExit(f"no score CSVs in {args.scores}")
    stages = stage_order(frames)
    rows = load_jsonl(args.dataset)
    rng = np.random.default_rng(args.seed)
    t = {s: tables(frames[s], rows) for s in stages}
    summary = []

    def add(stage, scope, quantity, x):
        x = np.asarray(x, dtype=float)
        m, lo, hi = bootstrap_ci(x, args.n_boot, rng)
        summary.append(dict(stage=stage, scope=scope, quantity=quantity, n=len(x), mean=m, lo=lo, hi=hi))

    scopes = ["all"] + list(DOMAINS)
    wide = {}
    for s in stages:
        wide[s] = {q: t[s][q].unstack("condition") for q in ("log_odds", "prior", "mass", "lp_believed", "lp_other")}
    for s in stages:
        for scope in scopes:
            sel = lambda d: d if scope == "all" else d[d.index.get_level_values("domain") == scope]
            c = sel(wide[s]["log_odds"])
            for h in HOLDERS:
                for tg in TARGETS:
                    add(s, scope, f"carryover {h} -> {tg}", c[f"{h}>{tg}"])
            add(s, scope, "leak gap: assistant -> new minus character -> new", c["assistant>new"] - c["character>new"])
            add(s, scope, "same gap: assistant -> same minus character -> same", c["assistant>same"] - c["character>same"])
            if s != "base" and "base" in wide:
                b = sel(wide["base"]["log_odds"])
                add(s, scope, "leak gap, change from base",
                    (c["assistant>new"] - c["character>new"]) - (b["assistant>new"] - b["character>new"]))
            for cond in CONDITIONS:
                if "none" in cond:
                    add(s, scope, f"prior log P(1st) - log P(2nd), {cond}", sel(wide[s]["prior"])[cond])
                else:
                    add(s, scope, f"raw log P(believed), {cond}", sel(wide[s]["lp_believed"])[cond])
                    add(s, scope, f"raw log P(other), {cond}", sel(wide[s]["lp_other"])[cond])
                add(s, scope, f"mass P(o1) + P(o2), {cond}", sel(wide[s]["mass"])[cond])
    summary = pd.DataFrame(summary)
    os.makedirs(args.out_dir, exist_ok=True)
    summary.to_csv(os.path.join(args.out_dir, "persona_carryover_summary.csv"), index=False)

    fmt = lambda r: f"{r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"
    get = lambda s, sc, q: summary[(summary.stage == s) & (summary.scope == sc) & (summary.quantity == q)].iloc[0]
    n_items = len({r["item"] for r in rows})
    md = ["# Persona carryover (overgeneralization, descriptive extra)", "",
          "Exploratory and outside the overgeneralization verdict. Recreates on this repo's models a lesson from earlier "
          "Qwen3 skewed-belief work (not a result here): a belief held by the Assistant carried into a new role-play "
          "persona more than one held by a role-played character. Design and readout: module docstring.", "",
          f"{n_items} items ({', '.join(f'{k}: {len([r for r in rows if r['domain'] == k]) // 2}' for k in DOMAINS)}), each "
          "with both outcomes believed. Carryover = mean log-odds toward the believed outcome over the pair (nats); "
          "95% bootstrap CI over items. Prediction fixed in advance: leak gap > 0. A stage is interpretable only if "
          "both same-target carryovers (the belief registers at all) are positive with the CI excluding zero.", "",
          "## Headline (all domains)", ""]
    head = []
    for s in stages:
        ctrl = [get(s, "all", f"carryover {h} -> same") for h in HOLDERS]
        leak = get(s, "all", "leak gap: assistant -> new minus character -> new")
        ok = all(interpretable(r) for r in ctrl)
        verdict = ("uninterpretable (control fails)" if not ok else
                   "assistant carries more" if leak["lo"] > 0 else
                   "character carries more" if leak["hi"] < 0 else "no clear difference")
        new = [get(s, "all", f"carryover {h} -> new") for h in HOLDERS]
        if ok and all(r["hi"] < 0 for r in new):  # the gap is relative: the new persona moves away from both beliefs
            verdict += " (relative only: both new-persona carryovers negative)"
        row = dict(stage=s, **{f"{h} -> {tg}": fmt(get(s, "all", f"carryover {h} -> {tg}")) for h in HOLDERS
                               for tg in TARGETS})
        row["leak gap"] = fmt(leak)
        row["change from base"] = fmt(get(s, "all", "leak gap, change from base")) if s != "base" else ""
        row["reading"] = verdict
        head.append(row)
    md += [pd.DataFrame(head).to_markdown(index=False), ""]

    def table(quantities, title, scope="all"):
        d = summary[(summary.scope == scope) & summary.quantity.isin(quantities)].copy()
        d["cell"] = d.apply(fmt, axis=1)
        p = d.pivot(index="quantity", columns="stage", values="cell").reindex(quantities)
        p = p[[s for s in stages if s in p.columns]].fillna("")
        return [f"### {title}", "", p.reset_index().to_markdown(index=False), ""]

    core_q = [f"carryover {h} -> {tg}" for h in HOLDERS for tg in TARGETS] + \
        ["leak gap: assistant -> new minus character -> new", "same gap: assistant -> same minus character -> same",
         "leak gap, change from base"]
    md += ["## By domain", ""]
    for dom in DOMAINS:
        md += table(core_q, dom, dom)
    md += ["## Raw log-probs, priors and coverage (all domains)", ""]
    md += table([q for q in summary.quantity.unique() if q.startswith(("raw", "prior", "mass"))], "all conditions")
    tok = load_tokenizer(args.tokenizer)
    toks = "; ".join(f"`{v!r}` {len(encode(tok, v))}" for d in DOMAINS.values() for o in d["outcomes"]
                     for v in variants(o))
    md += ["## Readout tokenization (variant: tokens)", "", toks, ""]
    with open(os.path.join(args.out_dir, "persona_carryover.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--n-items", type=int, default=12, help="items per domain")
    p.add_argument("--out", default=DATA_PATH)
    p = sub.add_parser("score")
    p.add_argument("--model", required=True)
    p.add_argument("--stage")
    p.add_argument("--tokenizer", default=config.TEMPLATE_MODEL)
    p.add_argument("--dataset", default=DATA_PATH)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", default=os.path.join(OUT_DIR, "persona_carryover"))
    p = sub.add_parser("analyze")
    p.add_argument("--scores", default=os.path.join(OUT_DIR, "persona_carryover"))
    p.add_argument("--dataset", default=DATA_PATH)
    p.add_argument("--tokenizer", default=config.TEMPLATE_MODEL)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    {"build": build, "score": score, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
