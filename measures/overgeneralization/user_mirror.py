""""Mirror the Assistant" in multi-turn chats (spec C, 2026-11-03; exploratory, outside the verdict; plan in docs/decisions.md).

Does a nearby Assistant reply make the model's pretend user write more like the Assistant, more after training? From
data/swap.jsonl (WildChat p1, r1, p2, r1_olmo), p2 is cut at a clearly unfinished point (core.prepare_data's incomplete-prefill
rule) and the model continues the user's own sentence in three conditions: with [p1, r1, prefill], without [prefill] alone,
olmo-reply [p1, r1_olmo, prefill] (secondary). The user portion of each sample is scored (lexical copy of r1 vs p1; spec A's
classifier and style features once available); mirror effect = with - without per conversation, effect of training = stage minus
base, paired per conversation, 95% bootstrap CI.

    python -m measures.overgeneralization.user_mirror generate --family olmo3-7b --model base   # GPU (bf16), one stage
    python -m measures.overgeneralization.user_mirror analyze --family olmo3-7b                 # CPU
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch

from core import config
from core.generate import load_generation_model, stop_ids
from core.prepare_data import build_incomplete_prefill
from core.render import SENTINEL, encode, load_jsonl, load_tokenizer, render_prefill, special_strings
from core.stats import bootstrap_ci, stage_order
from measures.overgeneralization import user_style as us
from measures.overgeneralization.human_vs_assistant import load_classifier
from measures.overgeneralization.user_traits import cut_user_portion, family_markers

SWAP_PATH = "data/swap.jsonl"
CONDITIONS = {"with": "r1", "without": None, "olmo-reply": "r1_olmo"}
MIN_WORDS, MIN_ALPHA = 8, 0.5
BATCH_PROMPTS = 8
N_BOOT, SEED = 10000, 0
LEAK_BAR = 0.05  # classifier P(assistant), OLMo instruct vs base


def out_dir(family, run="capstone"):
    return os.path.join(config.RESULTS_ROOT, run, family, "overgeneralization")


def gen_dir(family, run="capstone"):
    return os.path.join(out_dir(family, run), "user_mirror_generations")  # raw samples: local, not committed


def p2_ok(text):
    """Pre-set filter: >= MIN_WORDS whitespace words, >= MIN_ALPHA of them alphabetic after stripping edge punctuation."""
    w = text.split()
    return len(w) >= MIN_WORDS and sum(re.sub(r"^[\W_]+|[\W_]+$", "", x).isalpha() for x in w) / len(w) >= MIN_ALPHA


def prepare(tok, path=SWAP_PATH):
    """(conversations kept, counts). Each kept row: id, p1, r1, r1_olmo, p2, prefill, remainder."""
    rows = load_jsonl(path)
    specials = special_strings(tok)
    clean = [r for r in rows if not any(s in (r.get(k) or "") for s in specials for k in ("p1", "r1", "r1_olmo", "p2"))]
    kept = [r for r in clean if p2_ok(r["p2"])]
    cuts = build_incomplete_prefill([dict(id=r["id"], source="wildchat", kind="user", text=r["p2"]) for r in kept], "wildchat", len(kept))
    by_id = {r["id"]: r for r in kept}
    convs = [dict(id=c["text_id"], p1=by_id[c["text_id"]]["p1"], r1=by_id[c["text_id"]]["r1"], r1_olmo=by_id[c["text_id"]].get("r1_olmo"),
                  p2=by_id[c["text_id"]]["p2"], prefill=c["prefill"], remainder=by_id[c["text_id"]]["p2"][len(c["prefill"]):],
                  cut_type=c["cut_type"]) for c in cuts]
    counts = dict(total=len(rows), special_token_dropped=len(rows) - len(clean), filter_kept=len(kept), with_cut=len(convs),
                  with_olmo_reply=sum(1 for c in convs if c["r1_olmo"]))
    return convs, counts


def prompt_ids(tok, conv, condition):
    """Token ids of the open user turn under a condition (None if the condition does not apply to this conversation)."""
    if CONDITIONS[condition] is None:
        return render_prefill(tok, conv["prefill"])
    reply = conv[CONDITIONS[condition]]
    if not reply:
        return None
    msgs = [{"role": "user", "content": conv["p1"]}, {"role": "assistant", "content": reply}, {"role": "user", "content": SENTINEL}]
    s = tok.apply_chat_template(msgs, tokenize=False)
    if s.count(SENTINEL) != 1:
        raise ValueError(f"{conv['id']}: template does not place the user turn exactly once")
    return encode(tok, s.split(SENTINEL)[0] + conv["prefill"])


@torch.no_grad()
def generate(family, stage, run="capstone", samples=config.GEN_SAMPLES, max_new=config.GEN_MAX_NEW_TOKENS,
             temperature=config.GEN_TEMPERATURE, batch_prompts=BATCH_PROMPTS, limit=None):
    config.use_family(family)
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    convs, counts = prepare(tok)
    convs = convs[:limit]
    jobs = [(c, cond, ids) for c in convs for cond in CONDITIONS if (ids := prompt_ids(tok, c, cond)) is not None]
    print(f"{family} {stage}: {counts}; {len(jobs)} prompts", flush=True)
    model, device = load_generation_model(config.MODELS[stage], config.family_settings(family)["gen_precision"])
    eos_ids = stop_ids(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    torch.manual_seed(SEED)
    jobs.sort(key=lambda j: len(j[2]))
    records = []
    for i in range(0, len(jobs), batch_prompts):
        batch = jobs[i : i + batch_prompts]
        width = max(len(j[2]) for j in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for k, (_, _, p) in enumerate(batch):  # left padding
            ids[k, width - len(p):] = torch.tensor(p)
            mask[k, width - len(p):] = 1
        gen = model.generate(ids.to(device), attention_mask=mask.to(device), do_sample=True, temperature=temperature, top_p=1.0,
                             top_k=0, max_new_tokens=max_new, num_return_sequences=samples, eos_token_id=eos_ids, pad_token_id=pad_id)
        for row, seq in enumerate(gen[:, width:].tolist()):
            conv, cond, _ = batch[row // samples]
            cut = next((t for t, x in enumerate(seq) if x in eos_ids), None)
            text = tok.decode(seq[: cut + 1] if cut is not None else seq, skip_special_tokens=False)
            records.append(dict(conv_id=conv["id"], condition=cond, sample=row % samples, prefill=conv["prefill"], continuation=text,
                                stopped_on_eos=cut is not None))
        print(f"  {min(i + batch_prompts, len(jobs))}/{len(jobs)} prompts", flush=True)
    os.makedirs(gen_dir(family, run), exist_ok=True)
    path = os.path.join(gen_dir(family, run), f"{stage}.jsonl")
    records.sort(key=lambda r: (r["conv_id"], list(CONDITIONS).index(r["condition"]), r["sample"]))
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)
    print(f"wrote {path} ({len(records)} samples)")


# ---------------------------------------------------------------- scores on the user portion
def grams(text, n=4):
    t = " ".join(text.lower().split())
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def copy_share(user_text, source):
    """Share of the user text's distinct character 4-grams that occur in source (None for a text shorter than 4 characters)."""
    g = grams(user_text)
    return len(g & grams(source)) / len(g) if g else None


def score_texts(df, convs, clf):
    """df rows: conv_id, user_text. Adds the per-sample scores: spec A's classifier P(assistant) (first 40 words; unscored below 3
    words, as in user_style.py), lexical copy of r1 and p1, and spec A's style features."""
    c = {x["id"]: x for x in convs}
    ok = df.user_text.map(lambda t: len(us.words(t)) >= us.MIN_SCORE_WORDS)
    df["p_assistant"] = np.nan
    if ok.any():
        df.loc[ok, "p_assistant"] = clf.p_assistant([us.head_words(t, us.SCORE_WORDS) for t in df.loc[ok, "user_text"]])
    feats = pd.DataFrame([us.features(t) if t.strip() else {} for t in df.user_text], index=df.index)
    for f in STYLE:
        df[f] = feats[f] if f in feats else np.nan
    df["copy_r1"] = [copy_share(u, c[i]["r1"]) for i, u in zip(df.conv_id, df.user_text)]
    df["copy_p1"] = [copy_share(u, c[i]["p1"]) for i, u in zip(df.conv_id, df.user_text)]
    df["copy_r1_minus_p1"] = df.copy_r1 - df.copy_p1
    return df


STYLE = ["starts_lower", "typo_rate", "sent_len", "polite", "exclaim", "markdown", "cap_i", "terminal"]
METRICS = [("p_assistant", "classifier P(assistant) (spec A)  [primary]"), ("copy_r1", "char 4-gram share found in r1"), ("copy_p1", "char 4-gram share found in p1"),
           ("copy_r1_minus_p1", "share in r1 minus share in p1")] + [(f, f"style: {f}") for f in STYLE]


def user_samples(family, stage, convs, clf, run="capstone"):
    markers = family_markers(family)
    recs = load_jsonl(os.path.join(gen_dir(family, run), f"{stage}.jsonl"))
    df = pd.DataFrame([dict(conv_id=r["conv_id"], condition=r["condition"], sample=r["sample"],
                            user_text=cut_user_portion(r["continuation"], markers, role_cut=True)) for r in recs])
    return score_texts(df, convs, clf)


def human_reference(tok, convs, clf, max_new=config.GEN_MAX_NEW_TOKENS):
    df = pd.DataFrame([dict(conv_id=c["id"], condition="human", sample=0,
                            user_text=tok.decode(encode(tok, c["remainder"])[:max_new])) for c in convs])
    return score_texts(df, convs, clf)


def mirror(df, metric, other="with"):
    """Per conversation: mean over scored samples of `other` minus `without`; conversations unscored in either are dropped."""
    m = df.dropna(subset=[metric]).groupby(["conv_id", "condition"])[metric].mean().unstack("condition")
    return (m[other] - m["without"]).dropna() if other in m and "without" in m else pd.Series(dtype=float)


def analyze(family, run="capstone", base="base"):
    config.use_family(family)
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    convs, counts = prepare(tok)
    stages = stage_order([f[:-6] for f in os.listdir(gen_dir(family, run)) if f.endswith(".jsonl")])
    clf = load_classifier()
    samples = {s: user_samples(family, s, convs, clf, run) for s in stages}
    human = human_reference(tok, convs, clf)
    rng = np.random.default_rng(SEED)
    rows = []
    for metric, _ in METRICS:
        for other in ("with", "olmo-reply"):
            eff = {s: mirror(samples[s], metric, other) for s in stages}
            for s in stages:
                lv = samples[s].dropna(subset=[metric]).groupby(["condition", "conv_id"])[metric].mean()
                for cond in ("without", other):
                    if cond in lv.index.get_level_values(0):
                        m, lo, hi = bootstrap_ci(lv[cond].to_numpy(), N_BOOT, rng)
                        rows.append(dict(metric=metric, comparison=other, stage=s, kind=f"level {cond}", n=len(lv[cond]), mean=m, lo=lo, hi=hi))
                m, lo, hi = bootstrap_ci(eff[s].to_numpy(), N_BOOT, rng)
                rows.append(dict(metric=metric, comparison=other, stage=s, kind="mirror effect", n=len(eff[s]), mean=m, lo=lo, hi=hi))
                if s != base:
                    ids = eff[s].index.intersection(eff[base].index)
                    m, lo, hi = bootstrap_ci((eff[s][ids] - eff[base][ids]).to_numpy(), N_BOOT, rng)
                    rows.append(dict(metric=metric, comparison=other, stage=s, kind="mirror effect, change from base", n=len(ids),
                                     mean=m, lo=lo, hi=hi))
        h = human.dropna(subset=[metric]).groupby("conv_id")[metric].mean()
        m, lo, hi = bootstrap_ci(h.to_numpy(), N_BOOT, rng)
        rows.append(dict(metric=metric, comparison="human", stage="human", kind="level (real p2 remainder)", n=len(h), mean=m, lo=lo, hi=hi))
    table = pd.DataFrame(rows)
    table.insert(0, "family", family)
    os.makedirs(out_dir(family, run), exist_ok=True)
    table.to_csv(os.path.join(out_dir(family, run), "user_mirror.csv"), index=False, float_format="%.4f")
    write_report(table, family, stages, counts, samples, convs, run)
    return table


def _cell(r):
    return f"{r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"


def examples(df, convs, k=3):
    """k 'with' samples with a non-empty user portion, spread over conversations (deterministic)."""
    c = {x["id"]: x for x in convs}
    d = df[(df.condition == "with") & (df.user_text.str.split().str.len() >= 5)].drop_duplicates("conv_id")
    d = d.iloc[np.linspace(0, len(d) - 1, min(k, len(d))).astype(int)] if len(d) else d
    return [(c[r.conv_id], r.user_text) for r in d.itertuples()]


def _short(t, n=60):
    w = " ".join(t.split()).split(" ")
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def write_report(table, family, stages, counts, samples, convs, run="capstone"):
    trained = [s for s in stages if s != "base"]
    lines = [f"# Mirror the Assistant in multi-turn chats: {family}", "",
             "Spec C (2026-11-03; plan in `docs/decisions.md`; exploratory, outside the verdict). The model continues a WildChat user's "
             "second turn, cut mid-sentence, with the first exchange in context (*with*: p1 + original reply r1; *olmo-reply*: r1 "
             "replaced by OLMo 3 7B Instruct's reply) or without it. 3 samples, 120 new tokens, T 0.7, bf16. Scores on the user "
             "portion; mirror effect = with - without per conversation; 95% bootstrap CI over conversations, paired.", "",
             f"Conversations: {counts['total']} in `data/swap.jsonl`, {counts['special_token_dropped']} dropped for special-token text, "
             f"{counts['filter_kept']} pass the p2 filter (>= {MIN_WORDS} words, >= {MIN_ALPHA:.0%} alphabetic), {counts['with_cut']} have "
             f"an unfinished cut point (analysed); {counts['with_olmo_reply']} of them have an OLMo reply.", ""]
    if family != "olmo3-7b":
        lines += ["Supporting family: its post-training data cannot be checked for contamination.", ""]
    lines += ["## Pre-set reading (classifier P(assistant), *with* condition, mirror effect change from base)", ""]
    for s in trained:
        r = table[(table.metric == "p_assistant") & (table.comparison == "with") & (table.stage == s) &
                  (table.kind == "mirror effect, change from base")]
        if len(r):
            r = r.iloc[0]
            leak = r["mean"] >= LEAK_BAR and r["lo"] > 0
            lines.append(f"- **{s}**: {_cell(r)} -> {'MIRROR LEAK' if leak else 'no mirror leak'} (bar >= {LEAK_BAR} with CI excluding 0)"
                         + ("" if s == "instruct" and family == "olmo3-7b" else " (by stage / supporting)"))
    if family == "olmo3-7b" and "instruct" not in stages:
        lines.append("- OLMo 3 7B instruct was not generated (weights not cached, disk full), so the headline reading is not available.")
    lines += [""]
    for metric, label in METRICS:
        t = table[table.metric == metric]
        lines += [f"## {label}", "", "| | " + " | ".join(stages) + " |", "|---|" + "---|" * len(stages)]
        for comp in ("with", "olmo-reply"):
            for kind in ("level without", f"level {comp}", "mirror effect", "mirror effect, change from base"):
                cells = []
                for s in stages:
                    r = t[(t.comparison == comp) & (t.stage == s) & (t.kind == kind)]
                    cells.append(_cell(r.iloc[0]) if len(r) else "")
                if comp == "olmo-reply" and kind == "level without":
                    continue
                lines.append(f"| {comp}: {kind} | " + " | ".join(cells) + " |")
        h = t[t.comparison == "human"].iloc[0]
        lines += ["", f"Human reference (real p2 remainder, first 120 tokens): {_cell(h)} (n = {h['n']}).", ""]
    lines += ["## Examples (*with* condition, user portion)", ""]
    for s in stages:
        lines += [f"### {s}", ""]
        for conv, text in examples(samples[s], convs):
            lines += [f"- **r1**: {_short(conv['r1'], 30)}", f"  **prefill**: {_short(conv['prefill'], 40)}",
                      f"  **model**: {_short(text)}", f"  **real**: {_short(conv['remainder'], 40)}", ""]
    with open(os.path.join(out_dir(family, run), "user_mirror.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[: 60]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "generate", "analyze"])
    ap.add_argument("--family", required=True, choices=list(config.FAMILIES))
    ap.add_argument("--model", help="stage (generate)")
    ap.add_argument("--run", default="capstone")
    ap.add_argument("--batch-prompts", type=int, default=BATCH_PROMPTS)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.command == "prepare":
        config.use_family(args.family)
        print(prepare(load_tokenizer(config.TEMPLATE_MODEL))[1])
    elif args.command == "generate":
        generate(args.family, args.model, args.run, batch_prompts=args.batch_prompts, limit=args.limit)
    else:
        analyze(args.family, args.run)


if __name__ == "__main__":
    main()
