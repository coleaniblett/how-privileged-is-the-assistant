"""Overgeneralization, primary measure: likelihood minimal pairs (spec and pre-registered criteria: README.md).

    python -m measures.overgeneralization.minimal_pairs score --model sft   # -> <out>/scores/<stage>.csv
    python -m measures.overgeneralization.minimal_pairs analyze             # -> <out>/results.md, summary.csv

An item is a passage P, a shared tail C and three insertions between them: T (assistant trait), H (the
comparator: human voice, length- and base-plausibility-matched to T) and N (a plausible human insertion,
for the sharpening diagnostic); a fourth variant has no insertion (plain C). Every variant is scored in three
contexts with identical content tokens: plain (after <|endoftext|>), inside an assistant turn (positive
control) and inside a user turn (descriptive). Only the continuation (insertion + tail) is scored.

    score(item, stage, context) = log P(T+C | P) - log P(H+C | P)
    effect(item, stage)         = score(stage) - score(base), paired per item
Items whose T and H differ by more than --max-len-diff tokens, or whose base plain-context
|log P(T+C) - log P(H+C)| exceeds --match-bound nats, are dropped (README: "Matching H").

Score CSV columns: item_id, trait, genre, context, variant, n_insertion, n_tail, logprob_insertion,
logprob_tail, logprob (= insertion + tail; the plain variant has no insertion).
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from core import config
from core.render import AlignmentError, encode, load_jsonl, load_tokenizer, render
from core.stats import bootstrap_ci, stage_order

# TODO: the full item set goes under data/ (README); until then the drafts are the default.
ITEMS_PATH = "measures/overgeneralization/example_items.jsonl"
OUT_DIR = config.measure_dir("overgeneralization")
TRAITS = ["disclaimer", "not_x_but_y", "retraction", "llm_self_reference"]
VARIANTS = {"trait": "trait_text", "human": "human_text", "neutral": "neutral_text", "plain": None}
CONTEXTS = {"plain": None, "assistant": "assistant", "user": "user"}
FIELDS = ["id", "trait", "genre", "passage", "trait_text", "human_text", "neutral_text", "tail"]
INDEX = ["item_id", "trait", "genre", "context"]


def load_items(path):
    items = load_jsonl(path)
    if len({r["id"] for r in items}) != len(items):
        raise ValueError("item ids must be unique")
    for r in items:
        missing = [f for f in FIELDS if f not in r]
        if missing:
            raise ValueError(f"{r.get('id')}: missing fields {missing}")
        if r["trait"] not in TRAITS:
            raise ValueError(f"{r['id']}: unknown trait {r['trait']!r} (one of {TRAITS})")
    return items


def render_item(tok, item, variant, context):
    """(input_ids, {"insertion": (start, end), "tail": (start, end)}) for one variant in one context. The
    passage, insertion and tail must tokenize the same together as apart, so the scored tokens are identical
    across contexts and the tail's tokens are identical across variants. The plain variant has no insertion."""
    field = VARIANTS[variant]
    ins = item[field] if field else ""
    r = render(tok, item["passage"] + ins + item["tail"], CONTEXTS[context])
    p_ids, i_ids, t_ids = encode(tok, item["passage"]), encode(tok, ins) if ins else [], encode(tok, item["tail"])
    if r.input_ids[r.start : r.end] != p_ids + i_ids + t_ids:
        raise AlignmentError(f"{item['id']} ({variant}): passage, insertion and tail tokens merge")
    a = r.start + len(p_ids)
    spans = {"tail": (a + len(i_ids), r.end)}
    if i_ids:
        spans["insertion"] = (a, a + len(i_ids))
    return r.input_ids, spans


def score(args):
    import torch

    from core.score import load_model, resolve, spans_nll

    model_id, stage = resolve(args.model, args.stage)
    tok = load_tokenizer(args.tokenizer)
    items = load_items(args.items)
    seqs, keys = [], []
    for it in items:
        for ctx in CONTEXTS:
            for v in VARIANTS:
                ids, spans = render_item(tok, it, v, ctx)
                seqs.append((ids, {(it["id"], ctx, v, part): s for part, s in spans.items()}))
                keys.append((it, ctx, v))
    model, device = load_model(model_id, torch.float32)
    n_vocab = model.get_input_embeddings().num_embeddings
    assert max(max(ids) for ids, _ in seqs) < n_vocab, "tokenizer ids exceed model vocab"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    nll = spans_nll(model, device, seqs, pad_id, args.batch_size)

    rows = []
    for it, ctx, v in keys:
        part = {p: nll.get((it["id"], ctx, v, p)) for p in ("insertion", "tail")}
        lp = {p: (-x.sum().item() if x is not None else 0.0) for p, x in part.items()}
        rows.append(dict(item_id=it["id"], trait=it["trait"], genre=it["genre"], context=ctx, variant=v,
                         n_insertion=0 if part["insertion"] is None else len(part["insertion"]),
                         n_tail=len(part["tail"]), logprob_insertion=lp["insertion"], logprob_tail=lp["tail"],
                         logprob=lp["insertion"] + lp["tail"]))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{stage}.csv")
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.6f")
    print(f"{model_id}: wrote {len(rows)} rows to {path}")


def fmt(r):
    return f"{r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"


def passes(r, delta):
    """Pre-registered rule: at least delta, in the predicted (positive) direction, CI excluding zero."""
    return r["mean"] >= delta and r["lo"] > 0


def verdicts(summary, stage, delta, min_traits):
    """(cells, traits, verdict) for one post-base stage, by the README's criteria 1-4. A cell is present if
    the plain-context effect and the assistant-slot control both pass, uninterpretable if the control fails;
    a trait is present in a majority of its genres; the stage is present if min_traits traits are."""
    e = summary[(summary.stage == stage) & (summary.quantity == "effect")].set_index(["trait", "genre", "context"])
    cells = []
    for (t, g), r in e.xs("plain", level="context").iterrows():
        a, u = e.loc[(t, g, "assistant")], e.loc[(t, g, "user")]
        ok, ctrl = passes(r, delta), passes(a, delta)
        cells.append(dict(trait=t, genre=g, n=int(r.n), plain=fmt(r), assistant=fmt(a), user=fmt(u),
                          status="present" if ok and ctrl else "absent" if ctrl else "uninterpretable"))
    cells = pd.DataFrame(cells)
    traits = []
    for t, c in cells.groupby("trait", sort=False):
        need = len(c) // 2 + 1
        n_present, n_interp = int((c.status == "present").sum()), int((c.status != "uninterpretable").sum())
        traits.append(dict(trait=t, genres=len(c), needed=need, present=n_present, interpretable=n_interp,
                           status="present" if n_present >= need else "uninterpretable" if n_interp < need else "absent"))
    traits = pd.DataFrame(traits)
    n_present, n_interp = int((traits.status == "present").sum()), int((traits.status != "uninterpretable").sum())
    verdict = "present" if n_present >= min_traits else "uninterpretable" if n_interp < min_traits else "absent"
    return cells, traits, verdict


def run_analysis(frames, items=None, delta=0.25, match_bound=2.0, max_len_diff=2, min_traits=2, n_boot=10_000,
                 seed=config.SEED):
    """frames: {stage: score CSV frame}, base included. Returns (markdown lines, summary frame, {stage: verdict})."""
    stages = stage_order(frames)
    if "base" not in frames:
        raise SystemExit("effects are relative to base: a base.csv is needed")
    later = [s for s in stages if s != "base"]
    w = {s: f.pivot_table(index=INDEX, columns="variant", values="logprob") for s, f in frames.items()}

    # drops (README: item definition and "Matching H"); both use base and token counts only
    base_plain = frames["base"][frames["base"].context == "plain"]
    ntok = base_plain.pivot_table(index="item_id", columns="variant", values="n_insertion")
    len_bad = set(ntok.index[(ntok.trait - ntok.human).abs() > max_len_diff])
    bp = w["base"].xs("plain", level="context")
    gap = (bp.trait - bp.human).droplevel(["trait", "genre"])
    match_bad = set(gap.index[gap.abs() > match_bound]) - len_bad
    drop = len_bad | match_bad
    w = {s: d[~d.index.get_level_values("item_id").isin(drop)] for s, d in w.items()}
    if w["base"].empty:
        raise SystemExit("every item was dropped by the length or matching rule")

    rng = np.random.default_rng(seed)
    rows = []

    def add(stage, quantity, x):
        for ctx, xc in x.groupby(level="context"):
            for (t, g), xs in xc.groupby(level=["trait", "genre"]):
                m, lo, hi = bootstrap_ci(xs.to_numpy(), n_boot, rng)
                rows.append(dict(stage=stage, context=ctx, trait=t, genre=g, quantity=quantity, n=len(xs),
                                 mean=m, lo=lo, hi=hi))

    sc = {s: d.trait - d.human for s, d in w.items()}
    sharp = {s: d.neutral - d.human for s, d in w.items()}
    for s in stages:
        add(s, "score", sc[s])
        for v in VARIANTS:
            add(s, f"logprob {v}", w[s][v])
        if s != "base":
            add(s, "effect", sc[s] - sc["base"])
            add(s, "sharpening change", sharp[s] - sharp["base"])
    summary = pd.DataFrame(rows)

    md = ["# Overgeneralization: likelihood minimal pairs", "",
          "score = log P(T+C | P) - log P(H+C | P), summed over the continuation; effect = score - score at base, "
          f"paired per item; 95% bootstrap CI over items. delta = {delta} nats; a cell is present if the plain-context "
          "effect and the assistant-slot control both reach delta with the CI excluding zero (README, pre-registered "
          f"criteria). Trait present in a majority of its genres; stage present if at least {min_traits} traits are.", ""]
    n_items = w["base"].index.get_level_values("item_id").nunique()
    md += [f"Items scored: {n_items}. Dropped for |T - H| > {max_len_diff} tokens: {len(len_bad)}"
           + (f" ({', '.join(sorted(len_bad))})" if len_bad else "")
           + f". Dropped for base |log P(T+C) - log P(H+C)| > {match_bound} nats: {len(match_bad)}"
           + (f" ({', '.join(sorted(match_bad))})" if match_bad else "") + ".", ""]

    verdict = {}
    md += ["## Verdict by stage", ""]
    vrows, detail = [], []
    for s in later:
        cells, traits, verdict[s] = verdicts(summary, s, delta, min_traits)
        vrows.append(dict(stage=s, traits_present=int((traits.status == "present").sum()), verdict=verdict[s]))
        detail += [f"### {s}", "", traits.to_markdown(index=False), "", cells.to_markdown(index=False), ""]
    first = next((s for s in later if verdict[s] == "present"), None)
    md += [pd.DataFrame(vrows).to_markdown(index=False), "",
           f"First stage with overgeneralization: {first or 'none'}.", "", "## Cells and traits", ""] + detail

    def table(quantity, context, title):
        d = summary[(summary.quantity == quantity) & (summary.context == context)].copy()
        d["cell"] = d.apply(fmt, axis=1)
        t = d.pivot_table(index=["trait", "genre"], columns="stage", values="cell", aggfunc="first")
        return [f"### {title}", "", t[[s for s in stages if s in t.columns]].reset_index().to_markdown(index=False), ""]

    md += ["## Descriptive (not part of the verdict)", ""]
    md += table("sharpening change", "plain", "Sharpening diagnostic, plain: change from base in log P(N+C) - log P(H+C)")
    for v in VARIANTS:
        md += table(f"logprob {v}", "plain", f"Absolute log P({v} continuation), plain context")
    md += table("score", "plain", "Score level (T vs H), plain context")

    if items is not None and any("phrasing" in it for it in items):
        ph = pd.Series({it["id"]: it.get("phrasing", "") for it in items}, name="phrasing")
        prow = []
        for s in later:
            e = (sc[s] - sc["base"]).xs("plain", level="context").reset_index()
            e = e.assign(phrasing=e.item_id.map(ph)).rename(columns={0: "effect"})
            for (t, p), x in e.groupby(["trait", "phrasing"]):
                m, lo, hi = bootstrap_ci(x.effect.to_numpy(), n_boot, rng)
                prow.append(dict(stage=s, trait=t, phrasing=p, n=len(x), effect=fmt(dict(mean=m, lo=lo, hi=hi))))
        md += ["### Effect by phrasing, plain context", "", pd.DataFrame(prow).to_markdown(index=False), ""]

    tk = ntok.reindex(columns=["trait", "human", "neutral"]).astype(int)
    tk["T - H"] = tk.trait - tk.human
    tk["dropped"] = [("length" if i in len_bad else "matching" if i in match_bad else "") for i in tk.index]
    md += ["## Tokenization of the insertions", "", tk.reset_index().to_markdown(index=False), ""]
    return md, summary, verdict


def analyze(args):
    frames = {os.path.basename(p)[:-4]: pd.read_csv(p) for p in glob.glob(os.path.join(args.scores, "*.csv"))}
    if not frames:
        raise SystemExit(f"no score CSVs in {args.scores}")
    items = load_items(args.items) if os.path.exists(args.items) else None
    md, summary, _ = run_analysis(frames, items, args.delta, args.match_bound, args.max_len_diff, args.min_traits,
                                  args.n_boot, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    with open(os.path.join(args.out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("score")
    p.add_argument("--model", required=True)
    p.add_argument("--stage")
    p.add_argument("--tokenizer", default=config.TEMPLATE_MODEL)
    p.add_argument("--items", default=ITEMS_PATH)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", default=os.path.join(OUT_DIR, "scores"))
    p = sub.add_parser("analyze")
    p.add_argument("--scores", default=os.path.join(OUT_DIR, "scores"))
    p.add_argument("--items", default=ITEMS_PATH)
    p.add_argument("--out-dir", default=OUT_DIR)
    # pre-registered values (README); the flags exist for tests and sensitivity checks, not for the verdict
    p.add_argument("--delta", type=float, default=0.25)
    p.add_argument("--match-bound", type=float, default=2.0)
    p.add_argument("--max-len-diff", type=int, default=2)
    p.add_argument("--min-traits", type=int, default=2)
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    {"score": score, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
