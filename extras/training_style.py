"""Spec J7 (via spec K, 2026-11-05): does OLMo's user turn write training-style prompts? Exploratory, outside every verdict; no API; no prefill-judge output is read.
Test texts: the atlas's saved user-turn replies (the model writes the user's turn after "Okay."/"Noted.") for every family and stage, and the prefilled user-turn continuations (non-judge generations).
Measure 1 (reply starts with a frozen task-opening pattern; user-turn replies only) and measure 2 (5-gram overlap with Dolci user turns: share of the reply's 5-grams found in the single best-matching
Dolci prompt; "match" = at least 0.5, "near-exact" = at least 0.8). The same Dolci pass also checks the 297+297 BBQ questions of spec K (context + question, same measure).
    python -m extras.training_style collect --atlas-dir <dir with <fam>/<stage>_replies.jsonl.gz>   # builds the test texts
    python -m extras.training_style scan [--max-rows N]                                           # CPU: streams Dolci SFT, DPO, RL user turns
    python -m extras.training_style analyze
"""
import argparse
import glob
import gzip
import json
import os
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd

OUT = os.environ.get("TS_OUT", "results/overgeneralization/training_style")
SEED, N_BOOT, N = 0, 4000, 5
MATCH, NEAR = 0.5, 0.8
# FROZEN task-opening pattern (before scoring): the reply starts (after optional quotes/markdown) with a task verb or the usual instruction framings of training prompts.
OPENERS = (r"write|explain|generate|create|describe|summari[sz]e|translate|given|imagine|please|can you|could you|provide|list|what is|what are|how do|how to|answer|solve|tell me|i want you to|"
           r"you are|act as|implement|rewrite|classify|compose|draft|suggest|compare|define|calculate|come up with|find|identify|extract|convert|edit|correct|paraphrase|analy[sz]e|"
           r"design|develop|outline|discuss|make a|give me|show me|i need|help me|task:|instruction:|question:|q:|input:|context:|read the following|here is|below is|consider the")
OPEN_RE = re.compile(r"^[\s\"'`*#>\-\d.)(]*(?:" + OPENERS + r")\b", re.I)
WORD = re.compile(r"[a-z0-9']+")


def grams(text):
    w = WORD.findall(text.lower())
    return [hash(tuple(w[i:i + N])) for i in range(len(w) - N + 1)]


def collect(atlas_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(atlas_dir, "*", "*_replies.jsonl.gz"))):
        fam, stage = p.split(os.sep)[-2], os.path.basename(p).split("_")[0]
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for l in f:
                r = json.loads(l)
                if r["context"] == "user_turn":
                    rows.append(dict(kind="atlas_user_turn", family=fam, stage=stage, text=r["text"]))
    for p in sorted(glob.glob("results/capstone/*/core/generations_incomplete*/*.jsonl")):
        fam, stage = p.split(os.sep)[2], os.path.basename(p)[:-6]
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            rows.append(dict(kind="prefill_continuation", family=fam, stage=stage, text=r["continuation"]))
    mat = json.load(open("data/bias_leak_materials.json"))
    for it in mat["items"]:
        for cond in ("amb", "dis"):
            rows.append(dict(kind="bbq", family="-", stage="-", text=it[cond]["context"] + " " + it[cond]["question"], item=it["id"], cond=cond))
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_json(os.path.join(OUT, "test_texts.jsonl.gz"), orient="records", lines=True, compression="gzip")
    print(df.groupby(["kind", "family", "stage"]).size().to_string())


def dolci_user_turns(max_rows=0):
    from datasets import load_dataset
    for name, repo in (("sft", "allenai/Dolci-Instruct-SFT"), ("dpo", "allenai/Dolci-Instruct-DPO"), ("rl", "allenai/Dolci-Instruct-RL")):
        ds = load_dataset(repo, split="train", streaming=True)
        for i, row in enumerate(ds):
            if max_rows and i >= max_rows:
                break
            if name == "sft":
                for m in row.get("messages") or []:
                    if m["role"] == "user":
                        yield name, m["content"]
            elif name == "dpo":
                for m in row.get("chosen") or []:
                    if m["role"] == "user":
                        yield name, m["content"]
            else:
                for role, text in re.findall(r"(?:^|\n)(user|assistant): (.*?)(?=\n(?:user|assistant): |$)", row.get("prompt") or "", flags=re.S):
                    if role == "user":
                        yield name, text


def scan(max_rows=0):
    df = pd.read_json(os.path.join(OUT, "test_texts.jsonl.gz"), lines=True, compression="gzip")
    index = defaultdict(list)
    sizes = np.zeros(len(df))
    for i, t in enumerate(df.text.fillna("")):
        g = set(grams(t))
        sizes[i] = len(g)
        for h in g:
            index[h].append(i)
    best, best_src = np.zeros(len(df)), [""] * len(df)
    t0, n = time.time(), 0
    for name, text in dolci_user_turns(max_rows):
        n += 1
        hits = defaultdict(int)
        for h in set(grams(text)):
            for i in index.get(h, ()):
                hits[i] += 1
        for i, c in hits.items():
            fr = c / max(sizes[i], 1)
            if fr > best[i]:
                best[i], best_src[i] = fr, f"{name}: {text[:200]}"
        if n % 200000 == 0:
            print(f"{n} Dolci user turns, {time.time() - t0:.0f} s", flush=True)
    df["best_overlap"], df["best_prompt"], df["n_grams"] = best, best_src, sizes
    df.to_json(os.path.join(OUT, "scanned.jsonl.gz"), orient="records", lines=True, compression="gzip")
    print(f"scanned {n} Dolci user turns in {time.time() - t0:.0f} s")


def boot_diff(a, b, rng):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(N_BOOT)]
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def analyze():
    rng = np.random.default_rng(SEED)
    df = pd.read_json(os.path.join(OUT, "scanned.jsonl.gz"), lines=True, compression="gzip")
    df["text"] = df.text.fillna("")
    df["opens"] = [bool(OPEN_RE.match(t)) for t in df.text]
    df["match"], df["near"] = df.best_overlap >= MATCH, df.best_overlap >= NEAR
    df["either"] = df.opens | df.match
    L = ["# Does OLMo's user turn write training-style prompts? (spec J7 via spec K; exploratory, outside every verdict)", "",
         f"Test texts: the atlas's saved user-turn replies (the model writes the user's turn) and the prefilled user-turn continuations. Measure 1 = the reply starts with a frozen task-opening pattern (user-turn replies only); measure 2 = share of the reply's 5-grams found in the single best Dolci user prompt (SFT, DPO, RL): match >= {MATCH}, near-exact >= {NEAR}. "
         "'Either' = measure 1 or 2. Rates are shares of replies; 95% bootstrap ranges over replies.", ""]
    U = df[df.kind == "atlas_user_turn"]
    L += ["## Atlas user-turn replies", "", "| family | stage | replies | opens like a task | 5-gram match (>= 0.5) | near-exact (>= 0.8) | either |", "|---|---|---|---|---|---|---|"]
    for (fam, st), g in U.groupby(["family", "stage"]):
        L.append(f"| {fam} | {st} | {len(g)} | {100 * g.opens.mean():.0f}% | {100 * g['match'].mean():.1f}% | {100 * g.near.mean():.1f}% | {100 * g.either.mean():.0f}% |")
    L += ["", "### Reading (OLMo): training-style if at least 25% of final replies match by either measure and at most 5% of base replies, with the change's range excluding 0", "",
          "| model | final: either | base: either | change [95% range] | reading |", "|---|---|---|---|---|"]
    for fam, stages in (("olmo3-7b", ["sft", "dpo", "instruct"]), ("qwen3-8b", ["instruct"]), ("gemma4-31b", ["instruct"])):
        b = U[(U.family == fam) & (U.stage == "base")]
        for st in stages:
            f = U[(U.family == fam) & (U.stage == st)]
            if b.empty or f.empty:
                continue
            ch = boot_diff(f.either, b.either, rng)
            ok = f.either.mean() >= 0.25 and b.either.mean() <= 0.05 and ch[1] > 0
            L.append(f"| {fam} {st} | {100 * f.either.mean():.0f}% | {100 * b.either.mean():.0f}% | {100 * ch[0]:+.0f} [{100 * ch[1]:+.0f}, {100 * ch[2]:+.0f}] | {'training-style' if ok else 'not met'} |")
    L += ["", "### Post-hoc split of the reading by measure (added after seeing that the task-opening pattern also fires on base replies)", "",
          "| model | measure | final | base | change [95% range] | pre-set bar (final >= 25%, base <= 5%, range above 0) |", "|---|---|---|---|---|---|"]
    for fam, stages in (("olmo3-7b", ["sft", "dpo", "instruct"]), ("qwen3-8b", ["instruct"]), ("gemma4-31b", ["instruct"])):
        b = U[(U.family == fam) & (U.stage == "base")]
        for st in stages:
            f = U[(U.family == fam) & (U.stage == st)]
            for nm, col in (("task-opening pattern", "opens"), ("5-gram match >= 0.5", "match"), ("near-exact >= 0.8", "near")):
                ch = boot_diff(f[col], b[col], rng)
                ok = f[col].mean() >= 0.25 and b[col].mean() <= 0.05 and ch[1] > 0
                L.append(f"| {fam} {st} | {nm} | {100 * f[col].mean():.0f}% | {100 * b[col].mean():.0f}% | {100 * ch[0]:+.0f} [{100 * ch[1]:+.0f}, {100 * ch[2]:+.0f}] | {'met' if ok else 'not met'} |")
    L += ["", "Copying is not concluded without exact or near-exact matches; the near-exact column is the evidence for or against it.", ""]
    P = df[df.kind == "prefill_continuation"]
    L += ["## Prefilled user-turn continuations (non-judge generations; measure 2 only)", "", "| family | stage | continuations | 5-gram match (>= 0.5) | near-exact |", "|---|---|---|---|---|"]
    for (fam, st), g in P.groupby(["family", "stage"]):
        L.append(f"| {fam} | {st} | {len(g)} | {100 * g['match'].mean():.1f}% | {100 * g.near.mean():.1f}% |")
    B = df[df.kind == "bbq"]
    L += ["", "## BBQ questions of spec K against Dolci", "", f"{int(B['match'].sum())} of {len(B)} question texts have at least {MATCH:.0%} of their 5-grams in one Dolci user prompt ({int(B.near.sum())} near-exact). Items dropped from the OLMo analysis: {sorted(set(B[B['match']].item.astype(int)))}.", ""]
    ex = U[(U.family == "olmo3-7b") & (U.stage == "instruct")].sort_values("best_overlap", ascending=False).head(10)
    L += ["## The 10 OLMo-instruct replies with the highest overlap (hand-countable)", ""]
    for r in ex.itertuples():
        L.append(f"- overlap {r.best_overlap:.2f}: reply `{r.text[:160].replace(chr(10), ' ')}` | best Dolci prompt `{r.best_prompt[:160].replace(chr(10), ' ')}`")
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "training_style.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(sorted(set(int(i) for i in B[B['match']].item)), open(os.path.join(OUT, "bbq_dolci_matches.json"), "w"))
    print("\n".join(L[:30]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "scan", "analyze"])
    ap.add_argument("--atlas-dir")
    ap.add_argument("--max-rows", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a.atlas_dir)
    elif a.cmd == "scan":
        scan(a.max_rows)
    else:
        analyze()


if __name__ == "__main__":
    main()
