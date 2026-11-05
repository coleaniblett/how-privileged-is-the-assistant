"""Spec J12 (POST-HOC, 2026-11-05): how much of the trait atlas's before-and-after change is the base model running on? Exploratory, outside every verdict; no paid API; atlas files are not overwritten.
Base models often answer and then run on into a new turn ("5610Human: Question: ..."), which the atlas's cut did not remove. Here every saved atlas reply (all families, stages, contexts) is
additionally cut at the FIRST new-turn pattern (list frozen below, committed before any rescoring), detectors and skills are rescored on the strict text (skills: the accepted answer anywhere in the reply),
the local judge re-judges the replies the cut changed (same Gemma 4 31B judge, same prompts and rule), and the atlas cells are recomputed with the same bootstrap and gate.
    python -m extras.trait_atlas_strict cut        # CPU: strict text, removal tables, 10 base examples per family -> results/trait_atlas_strict/ (+ detector tables)
    python -m extras.trait_atlas_strict judge      # GPU: re-judge changed replies of judge traits (Gemma 4 31B-it, bf16)
    python -m extras.trait_atlas_strict analyze    # cells + comparison with the atlas -> results/trait_atlas_strict/trait_atlas_strict.md (+ csv)
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from extras import trait_atlas as ta

OUT = os.environ.get("STRICT_OUT", "results/trait_atlas_strict")
SRC = "results/trait_atlas"
SEED = 0
# FROZEN LIST (2026-11-05, before rescoring). A match must be preceded by a non-letter (so "FAQ:" and "Hi Q:" ... only start-of-token labels count) and must come after some reply text.
NEW_TURN = [
    r"(?<![A-Za-z])Human\s*:", r"(?<![A-Za-z])User\s*:", r"(?<![A-Za-z])Assistant\s*:", r"(?<![A-Za-z])Question\s*\d*\s*:", r"(?<![A-Za-z])Q\d*\s*:",
    r"###\s*(?:Instruction|Input|Question|Human|User|Response)", r"(?:^|\n)\s*(?:Instruction|Input)\s*:",
    r"<\|(?:im_start|user|assistant|system|endoftext)\|?>", r"\[INST\]",
]
NEW_TURN_RE = [re.compile(p) for p in NEW_TURN]
SKIP_CONTEXTS = {"user_turn"}      # the model writes the user's turn there; "Human:"/"User:" are its own text, and the atlas already cuts the handoff to the Assistant


def strict_cut(text, context=None):
    """Cut at the first new-turn pattern that follows some reply text. Returns (text, cut_happened)."""
    if context in SKIP_CONTEXTS or not isinstance(text, str):
        return text, False
    best = None
    for rx in NEW_TURN_RE:
        for m in rx.finditer(text):
            if text[:m.start()].strip():
                best = m.start() if best is None else min(best, m.start())
                break
    if best is None:
        return text, False
    return text[:best].rstrip(), True


def replies(path):
    rows = ta.read_gz(path)
    for r in rows:
        r["strict"], r["cut"] = strict_cut(r["text"], r["context"])
    return rows


def fam_stage(path):
    return path.split(os.sep)[-2], os.path.basename(path).split("_")[0]


def cut_cmd():
    mat = ta.load_materials()
    idx = ta.item_index(mat)
    rng = np.random.default_rng(SEED)
    L = ["# Strict cut of base run-on (spec J12, POST-HOC; exploratory, outside every verdict)", "",
         "Frozen new-turn patterns (committed before rescoring): " + "; ".join(f"`{p}`" for p in NEW_TURN) + f". Not applied in contexts {sorted(SKIP_CONTEXTS)}. A match counts only after some reply text and when not preceded by a letter. "
         "Atlas files are untouched; strict tables are in `results/trait_atlas_strict/`.", "", "## How much text the strict cut removes", "",
         "| family | stage | context | replies | cut | share cut | mean share of characters removed (all replies) |", "|---|---|---|---|---|---|---|"]
    examples = {}
    for path in sorted(glob.glob(os.path.join(SRC, "*", "*_replies.jsonl.gz"))):
        fam, stage = fam_stage(path)
        rows = replies(path)
        d = pd.DataFrame([dict(context=r["context"], cut=r["cut"], frac=1 - len(r["strict"]) / max(len(r["text"]), 1)) for r in rows])
        for ctx, g in d.groupby("context"):
            L.append(f"| {fam} | {stage} | {ctx} | {len(g)} | {int(g.cut.sum())} | {100 * g.cut.mean():.0f}% | {100 * g.frac.mean():.0f}% |")
        if stage == "base":
            cand = [r for r in rows if r["cut"]]
            pick = rng.permutation(len(cand))[:10]
            examples[fam] = [cand[i] for i in pick]
        # detector rescoring on the strict text (same detectors; skills: accepted answer anywhere)
        out = []
        for r in rows:
            trait, det, item = idx[r["item"]]
            if det == "judge":
                continue
            r["flagged"] = ta.is_switch(r["raw"], ta.markers_of(fam), r["context"])
            if det == "skill":
                v = float(any(re.search(a, r["strict"], re.I) for a in item["accept"]))
                tail = float(ta.skill_correct(r["strict"], item["accept"]))      # the atlas's own last-150-characters rule, on the strict text (isolates the effect of the cut from the change of rule)
            else:
                v = ta.detect_one(trait, det, r["strict"], item)
            out.append({**{k: r[k] for k in ("trait", "item", "context", "wording", "persona", "sample", "flagged")}, "capped": bool(r.get("capped")), "value": v, "value_alt": tail if det == "skill" else v, "changed_by_cut": bool(r["cut"])})
        os.makedirs(os.path.join(OUT, fam), exist_ok=True)
        pd.DataFrame(out).to_csv(os.path.join(OUT, fam, f"{stage}_detected.csv.gz"), index=False)
        # which judge-trait replies were changed (to re-judge)
        ch = [dict(trait=r["trait"], item=r["item"], context=r["context"], wording=r["wording"], persona=r["persona"], sample=r["sample"]) for r in rows if r["cut"] and idx[r["item"]][1] == "judge"]
        pd.DataFrame(ch).to_csv(os.path.join(OUT, fam, f"{stage}_tojudge.csv.gz"), index=False)
        print(fam, stage, "replies", len(rows), "cut", sum(r["cut"] for r in rows), "judge rows to redo", len(ch), flush=True)
    L += ["", "## 10 random base replies per family that the cut changed (kept text | removed text, first 200 characters of each)", ""]
    for fam, ex in examples.items():
        L += [f"### {fam}", ""]
        for r in ex:
            kept = r["strict"][-160:].replace("\n", " ⏎ ")
            rem = r["text"][len(r["strict"]):][:200].replace("\n", " ⏎ ")
            L.append(f"- [{r['trait']}/{r['context']}] ...{kept} | **removed:** {rem}")
        L.append("")
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "cut_summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def judge_cmd(args):
    import torch
    from core import config
    from core.render import encode, load_tokenizer
    from core.score import load_model
    mat = ta.load_materials()
    idx = ta.item_index(mat)
    config.use_family("gemma4-31b")
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    tok.padding_side = "left"
    yes, no = encode(tok, "Yes")[0], encode(tok, "No")[0]
    model, device = load_model(config.MODELS["instruct"], torch.bfloat16)
    key = ["trait", "item", "context", "wording", "persona", "sample"]
    for path in sorted(glob.glob(os.path.join(SRC, "*", "*_replies.jsonl.gz"))):
        fam, stage = fam_stage(path)
        out = os.path.join(OUT, fam, f"{stage}_judged.csv.gz")
        orig = pd.read_csv(os.path.join(SRC, fam, f"{stage}_judged.csv.gz"))
        if os.path.exists(out) and not args.limit:
            continue
        todo = pd.read_csv(os.path.join(OUT, fam, f"{stage}_tojudge.csv.gz")) if os.path.getsize(os.path.join(OUT, fam, f"{stage}_tojudge.csv.gz")) > 5 else pd.DataFrame(columns=key)
        rows = {tuple(r[k] for k in key): r for r in replies(path) if idx[r["item"]][1] == "judge"}
        redo = [rows[tuple(t)] for t in todo[key].itertuples(index=False, name=None) if tuple(t) in rows]
        if args.limit:
            redo = redo[:args.limit]
        prompts = [tok.apply_chat_template([{"role": "user", "content": ta.judge_prompt(mat, r["trait"], idx[r["item"]][2], r["strict"])}], tokenize=False, add_generation_prompt=True) for r in redo]
        order = sorted(range(len(redo)), key=lambda k: len(prompts[k]))
        py = np.full(len(redo), np.nan)
        for s in range(0, len(order), args.batch_size):
            b = order[s:s + args.batch_size]
            enc = tok([prompts[k] for k in b], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
            py[b] = torch.stack([lg[:, yes], lg[:, no]], 1).softmax(-1)[:, 0].cpu().numpy()
            if (s // args.batch_size) % 25 == 0:
                print(f"strict judge {fam}/{stage}: {s}/{len(order)}", flush=True)
        new = orig.copy()
        mk = {tuple(r[k] for k in key): i for i, r in enumerate(new[key].to_dict("records"))}
        for r, p in zip(redo, py):
            i = mk.get(tuple(r[k] for k in key))
            if i is not None:
                new.loc[i, "p_yes"], new.loc[i, "value"] = float(p), float(p >= 0.5)
        if not args.limit:
            new.to_csv(out, index=False)
        print(fam, stage, "re-judged", len(redo), flush=True)


def analyze_cmd():
    mat = ta.load_materials()
    C0 = pd.read_csv(os.path.join(SRC, "trait_atlas_cells.csv"))
    saved = ta.OUT
    ta.OUT = OUT
    try:
        V = ta.load_values()
    finally:
        ta.OUT = saved
    C1 = ta.analyze_cells(V, mat)
    C1.to_csv(os.path.join(OUT, "trait_atlas_strict_cells.csv"), index=False)
    keys = ["family", "stage", "trait", "context", "flagged_excluded"]
    M = C0.merge(C1, on=keys, suffixes=("_atlas", "_strict"))
    M = M[~M.flagged_excluded & ~M.trait.str.endswith("__alt")]

    def moved(c, lo, hi, unit):
        gate = 10.0 if unit == "pts" else 0.2
        return (abs(c) >= gate) and (lo > 0 or hi < 0)
    M["atlas_moved"] = [moved(c, lo, hi, u) for c, lo, hi, u in zip(M.change_atlas, M.lo_atlas, M.hi_atlas, M.unit_atlas)]
    M["strict_moved"] = [moved(c, lo, hi, u) for c, lo, hi, u in zip(M.change_strict, M.lo_strict, M.hi_strict, M.unit_strict)]
    M["shrinks_half"] = (M.change_strict.abs() <= 0.5 * M.change_atlas.abs()) | (np.sign(M.change_strict) != np.sign(M.change_atlas))
    M["sensitive"] = M.atlas_moved & (~M.strict_moved | M.shrinks_half)
    M.to_csv(os.path.join(OUT, "trait_atlas_strict_comparison.csv"), index=False)
    L = ["# Trait atlas on strict-cut text (spec J12, POST-HOC; exploratory, outside every verdict)", "",
         "Plan: `docs/decisions.md` 2026-11-05. Same replies as the atlas, additionally cut at the first new-turn pattern (see `cut_summary.md`); detectors and skills rescored (skills: accepted answer anywhere in the reply); "
         "judge traits re-judged only where the cut changed the reply (Gemma 4 31B-it, same prompts). Same bootstrap and gate (10 points, 0.2 log words, CI excluding 0). "
         "A cell is **run-on sensitive** if the atlas reads it as moved (gate met, CI excluding 0) and under the strict cut it no longer meets the bar, or shrinks by at least half (or changes sign). Changes are final stage minus base.", ""]
    S = M[M.atlas_moved]
    L += ["## Run-on sensitive cells (of the cells the atlas reads as moved)", "", f"{int(S.sensitive.sum())} of {len(S)} moved cells are run-on sensitive.", "",
          "### By family", "", S.groupby("family").sensitive.agg(["sum", "count"]).rename(columns={"sum": "sensitive", "count": "moved cells"}).to_markdown(), "",
          "### By context", "", S.groupby("context").sensitive.agg(["sum", "count"]).rename(columns={"sum": "sensitive", "count": "moved cells"}).to_markdown(), "",
          "### By trait", "", S.groupby("trait").sensitive.agg(["sum", "count"]).rename(columns={"sum": "sensitive", "count": "moved cells"}).to_markdown(), ""]
    A = M[(M.context == "assistant") & (M.stage == "instruct")]
    lost = A[(A.atlas_moved) & (~A.strict_moved)]
    L += ["## Assistant gate (assistant context, final stage)", "",
          f"Traits the atlas gate counts as moved at the final stage: {int(A.atlas_moved.sum())}; still moved after the strict cut: {int((A.atlas_moved & A.strict_moved).sum())}. "
          "A trait that loses the gate is reported as 'training did not clearly move the Assistant after all':", "",
          "| family | trait | atlas change | strict change |", "|---|---|---|---|"]
    fm = lambda c, lo, hi, u: f"{c:+.1f} [{lo:+.1f}, {hi:+.1f}]" if u == "pts" else f"{c:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    for r in lost.itertuples():
        L.append(f"| {r.family} | {r.trait} | {fm(r.change_atlas, r.lo_atlas, r.hi_atlas, r.unit_atlas)} | {fm(r.change_strict, r.lo_strict, r.hi_strict, r.unit_strict)} |")
    L += ["", "## Every cell: atlas change beside strict-cut change (final stage; main cells, flagged replies included)", "",
          "| family | stage | trait | context | atlas change | strict change | atlas moved | strict moved | run-on sensitive |", "|---|---|---|---|---|---|---|---|---|"]
    for r in M.sort_values(["family", "trait", "context", "stage"]).itertuples():
        L.append(f"| {r.family} | {r.stage} | {r.trait} | {r.context} | {fm(r.change_atlas, r.lo_atlas, r.hi_atlas, r.unit_atlas)} | {fm(r.change_strict, r.lo_strict, r.hi_strict, r.unit_strict)} | "
                 f"{'yes' if r.atlas_moved else ''} | {'yes' if r.strict_moved else ''} | {'SENSITIVE' if r.sensitive else ''} |")
    sk = []
    for fam in ("olmo3-7b", "qwen3-8b", "gemma4-31b"):
        for trait in ta.CLUSTERS["skill"]:
            for ctx in ("assistant", "light_user", "strong_act"):
                def g(C, t):
                    x = C[(C.family == fam) & (C.stage == "instruct") & (C.trait == t) & (C.context == ctx) & (~C.flagged_excluded)]
                    return fm(x.iloc[0].change, x.iloc[0].lo, x.iloc[0].hi, "pts") if len(x) else "n/a"
                sk.append(f"| {fam} | {trait} | {ctx} | {g(C0, trait)} | {g(C1, trait + '__alt')} | {g(C1, trait)} |")
    L += ["", "## Skills: rule change versus cut (final minus base, points)", "",
          "'Atlas rule' = accepted answer in the last 150 characters of the atlas text. 'Same rule on strict text' isolates the effect of cutting base's run-on. 'Anywhere in the strict text' is the J12 primary score.", "",
          "| family | skill | context | atlas rule (atlas text) | same rule on strict text | anywhere in strict text |", "|---|---|---|---|---|---|"] + sk
    open(os.path.join(OUT, "trait_atlas_strict.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:30]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cut", "judge", "analyze"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    {"cut": lambda: cut_cmd(), "judge": lambda: judge_cmd(a), "analyze": lambda: analyze_cmd()}[a.cmd]()


if __name__ == "__main__":
    main()
