"""Three-turn conversations (user P1, assistant R1, user P2) for the turns test. No model weights.

    python -m exploration.turns_test.build_data          # -> exploration/turns_test/data/conversations.jsonl

WildChat: the first 50 rows of data/swap.jsonl (held out from Dolci; built by extras.role_swap prepare).
oasst1 and UltraChat: new three-turn samples under the same filters as swap.jsonl and core.prepare_data:
every turn 20-300 OLMo tokens and renderable by core.render (OLMo template) and core.render.render_target
(OLMo and Qwen3 templates, every target used by the test), P1/P2 not in Dolci as user turns, R1 not in Dolci as an
assistant turn (exact match after whitespace normalization, core.check_contamination). oasst1 follows
core.prepare_data's reply rule (best-ranked, else earliest child) at each step, English, reviewed, not deleted,
toxicity < 0.5. Nothing in core/ or data/ is written.
"""

import argparse
import json
import os
import random
from collections import Counter

from core import config
from core.check_contamination import dolci_turns, is_contaminated
from core.prepare_data import oasst_toxicity, usable
from core.render import AlignmentError, load_jsonl, load_tokenizer, render_target

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "conversations.jsonl")
TEMPLATE_KW = {"olmo3-7b": {}, "qwen3-8b": {"enable_thinking": False}}


def best(kids):
    return min(kids, key=lambda m: (m["rank"] is None, m["rank"] or 0, m["created_date"]))


def oasst_triples(seed):
    from datasets import load_dataset

    ds = load_dataset(config.DATASET_IDS["oasst1"], split="train")
    ok = [m for m in ds if not m["deleted"] and m["review_result"] and m["lang"] == "en"]
    kids = {}
    for m in ok:
        if m["parent_id"] is not None:
            kids.setdefault(m["parent_id"], []).append(m)
    roots = [m for m in ok if m["role"] == "prompter" and m["parent_id"] is None]
    random.Random(seed).shuffle(roots)
    for root in roots:
        r1s = [m for m in kids.get(root["message_id"], []) if m["role"] == "assistant"]
        if not r1s:
            continue
        r1 = best(r1s)
        p2s = [m for m in kids.get(r1["message_id"], []) if m["role"] == "prompter"]
        if not p2s:
            continue
        p2 = best(p2s)
        if max(oasst_toxicity(m) for m in (root, r1, p2)) >= 0.5:
            continue
        yield root["message_id"], [root["text"], r1["text"], p2["text"]]


def ultrachat_triples(seed):
    from datasets import load_dataset

    ds = load_dataset(config.DATASET_IDS["ultrachat"], split="train", streaming=True).shuffle(seed=seed, buffer_size=20_000)
    for ex in ds:
        d = ex["data"]
        if len(d) >= 3 and all(d[:3]):
            yield f"ultrachat-{ex['id']}", list(d[:3])


def renders(toks, p1, r1, p2):
    """Every rendering the test scores, under both families' templates (raises AlignmentError)."""
    for fam, tok in toks.items():
        kw = TEMPLATE_KW[fam]
        u, a = "user", "assistant"
        render_target(tok, [{"role": u, "content": p1}, {"role": a, "content": r1}], kw)
        render_target(tok, [{"role": u, "content": p1}, {"role": a, "content": r1}, {"role": u, "content": p2}], kw)
        if fam != "olmo3-7b":  # usable() already checked core.render for OLMo; check the Qwen template too
            from core.render import render
            config.use_family(fam)
            try:
                for text, slot in ((p1, "user"), (r1, "assistant"), (p2, "user")):
                    render(tok, text, slot)
                    render(tok, text, None)
            finally:
                config.use_family("olmo3-7b")


def sample(source, stream, toks, turns, n):
    rows, seen, stats = [], set(), Counter()
    olmo = toks["olmo3-7b"]
    for sid, texts in stream:
        stats["candidates"] += 1
        p1, r1, p2 = (t.strip() for t in texts)
        if sid in seen or p1 in seen:
            stats["duplicate"] += 1
            continue
        if (is_contaminated(turns, p1, "user") or is_contaminated(turns, p2, "user")
                or is_contaminated(turns, r1, "assistant")):
            stats["in Dolci"] += 1
            continue
        ns = [usable(olmo, t) for t in (p1, r1, p2)]
        if any(k is None for k in ns):
            stats["length/render"] += 1
            continue
        try:
            renders(toks, p1, r1, p2)
        except AlignmentError:
            stats["render (context or Qwen)"] += 1
            continue
        seen.update([sid, p1])
        rows.append(dict(id=f"{source}-{len(rows):03d}", source=source, source_id=sid, p1=p1, r1=r1, p2=p2,
                         n_p1=ns[0], n_r1=ns[1], n_p2=ns[2]))
        if len(rows) == n:
            break
    stats["kept"] = len(rows)
    return rows, dict(stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    toks = {fam: load_tokenizer(config.FAMILIES[fam]["template"]) for fam in TEMPLATE_KW}
    report = {}

    wild, stats = [], Counter()
    for r in load_jsonl("data/swap.jsonl"):
        try:
            renders(toks, r["p1"], r["r1"], r["p2"])
        except AlignmentError:
            stats["render (context or Qwen)"] += 1
            continue
        wild.append(dict(id=f"wildchat-{len(wild):03d}", source="wildchat", source_id=r["source_id"], swap_id=r["id"],
                         p1=r["p1"], r1=r["r1"], p2=r["p2"], n_p1=r["n_p1"], n_r1=r["n_r1"], n_p2=r["n_p2"]))
        if len(wild) == args.n:
            break
    stats["kept"] = len(wild)
    report["wildchat"] = dict(stats, note="data/swap.jsonl in file order (already Dolci-filtered and length-filtered)")

    turns = dolci_turns()
    rows = wild
    for source, stream in (("oasst1", oasst_triples(args.seed)), ("ultrachat", ultrachat_triples(args.seed))):
        got, st = sample(source, stream, toks, turns, args.n)
        report[source] = st
        rows += got
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(os.path.dirname(OUT), "build_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    print(f"wrote {len(rows)} conversations to {OUT}")


if __name__ == "__main__":
    main()
