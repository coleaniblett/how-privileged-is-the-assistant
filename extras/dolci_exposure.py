"""Spec P2 (2026-11-05): does OLMo's training data (Dolci-Instruct SFT, DPO, RL) explain the atlas pattern? Exploratory, outside every verdict, NOT BLIND. No GPU, no paid API, no judge: the datasets are
read by streaming (nothing stored). Plan, readings and the frozen sorting rules: docs/decisions.md 2026-11-05. Regex measures only; correlation, not cause.
    python -m extras.dolci_exposure scan [--max-rows N]    # stream, sort, detect -> results/overgeneralization/dolci_exposure/<stage>.npz (local) and examples
    python -m extras.dolci_exposure analyze                # -> results/overgeneralization/dolci_exposure/dolci_exposure.md
"""
import argparse
import json
import os
import re
import time

import numpy as np

from extras import trait_atlas as ta

OUT = os.environ.get("DE_OUT", "results/overgeneralization/dolci_exposure")
SEED, N_BOOT = 0, 2000
BUCKETS = ["no_role", "role_system", "role_user", "other_system"]
DETS = ["md", "stock", "says_ai", "refusal", "logwords"]
DETN = {"md": "layout", "stock": "stock", "says_ai": "self_ai", "refusal": "refusal", "logwords": "words"}
# ---- FROZEN sorting rules (version 1, written before looking at any bucket; tightened versions are named in the report)
ASSIST_LIKE = re.compile(r"assistant|helpful|\bAI\b|language model|chatbot|\bOLMo\b|\bTulu\b|\bmodel\b|\bbot\b|honest|harmless", re.I)
SYS_IDENT = re.compile(r"\b(?:you are|you're|your name is|act as|acting as|pretend (?:to be|you)|role-?play|roleplay|persona|play the (?:role|part) of|imagine you(?:'re| are)|you will (?:play|act as|be)|assume the role)\b", re.I)
SYS_ROLE_STRICT = re.compile(r"\b(?:act as|acting as|pretend (?:to be|you)|role-?play|roleplay|persona|play the (?:role|part) of|imagine you(?:'re| are)|you will (?:play|act as)|assume the role|your name is|you are (?:now )?(?:playing|portraying|roleplaying))\b", re.I)
# v1 (first look: precision 64% of 25 read, over the 80% bar, false hits deep inside long passages and 'role-playing game') -> v2: only the first 300 characters, only requests aimed at the model
# v2 read: sft 77%, dpo 77% of 22 each (false hits: 'does not act as a reducing agent', 'Imagine you're writing a script', 'System: You act as a paraphrase tool') -> v3: a role request must start a sentence (or follow 'I want you to' / 'please' / 'now'), and 'imagine you are' must not continue with an activity
USER_ROLE = re.compile(r"(?i)(?:^|[.!?:\n\"]\s*)(?:please\s+|now\s+|ignore previous instructions\.?\s*)?(?:i want you to (?:act|pretend|play|be)\b|act as\b|acting as\b|pretend (?:to be|you(?:'re| are))\b|you are now (?:a|an|the)\b|from now on,? you (?:are|will)\b|imagine you(?:'re| are)\s+(?!writing|creating|developing|working on|designing|building|planning|looking|trying|a helpful assistant to|an assistant to|part of)|take on the role of|play the (?:role|part) of|(?:role-?play|roleplay)(?:ing)? as\b|play (?:a|an|the)\b)")
SELFQ = re.compile(r"(?i)\b(?:who are you|what are you|are you (?:a |an )?(?:human|ai|real|robot|chatbot|person|conscious|sentient)|do you (?:feel|have feelings|think|like|enjoy|want|dream|sleep|get)|your (?:favou?rite|name|opinion|age|feelings)|how are you|tell me about yourself|what is your|what's your|what do you (?:like|think|want))")


def bucket(system, first_user):
    if system:
        if SYS_ROLE_STRICT.search(system):
            return "role_system"
        m = re.search(r"\byou are\s+(?:an?\s+|the\s+)?([^.,;:\n]{0,80})", system, re.I)
        if m and not ASSIST_LIKE.search(m.group(1)) and not re.search(r"expert|specialist|professional|engineer|programmer|translator|writer|tutor|teacher|analyst|doctor|coach|consultant|editor", m.group(1), re.I):
            return "role_system"
    if USER_ROLE.search((first_user or "")[:300]):
        return "role_user"
    if system and not (ASSIST_LIKE.search(system[:200]) and len(system) < 400):
        return "other_system"
    return "no_role"


def split_conv(msgs):
    """(system, first user turn, first reply after it) from a list of {role, content}."""
    system = next((m["content"] for m in msgs if m["role"] == "system"), "")
    fu = next((i for i, m in enumerate(msgs) if m["role"] == "user"), None)
    if fu is None:
        return None
    rep = next((m["content"] for m in msgs[fu + 1:] if m["role"] == "assistant"), None)
    return system, msgs[fu]["content"], rep


def scan(max_rows=0):
    from datasets import load_dataset
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for name, repo in (("sft", "allenai/Dolci-Instruct-SFT"), ("dpo", "allenai/Dolci-Instruct-DPO"), ("rl", "allenai/Dolci-Instruct-RL")):
        ds = load_dataset(repo, split="train", streaming=True)
        recs = {"chosen": [], "rejected": []}
        examples = {b: [] for b in BUCKETS}
        seen = {b: 0 for b in BUCKETS}
        n_rows = n_noreply = 0
        t0 = time.time()
        for i, row in enumerate(ds):
            if max_rows and i >= max_rows:
                break
            n_rows += 1
            if name == "rl":
                continue
            variants = {"chosen": row.get("chosen")} if name == "dpo" else {"chosen": row.get("messages")}
            if name == "dpo":
                variants["rejected"] = row.get("rejected")
            for tag, msgs in variants.items():
                if not msgs:
                    continue
                sp = split_conv(msgs)
                if sp is None or not sp[2]:
                    n_noreply += tag == "chosen"
                    continue
                system, fu, rep = sp
                b = bucket(system, fu)
                vals = [ta.detect_one("x", DETN[d], rep, None) for d in DETS]
                recs[tag].append((BUCKETS.index(b), int(bool(SELFQ.search(fu))), *vals))
                if tag == "chosen":
                    seen[b] += 1
                    ex = dict(stage=name, bucket=b, system=system[:300], user=fu[:300], reply=rep[:200])
                    if len(examples[b]) < 50:
                        examples[b].append(ex)
                    else:
                        j = int(rng.integers(seen[b]))
                        if j < 50:
                            examples[b][j] = ex
            if i % 100000 == 0 and i:
                print(f"{name}: {i} rows ({time.time() - t0:.0f} s)", flush=True)
        for tag, r in recs.items():
            if r:
                np.save(os.path.join(OUT, f"{name}_{tag}.npy"), np.array(r, dtype=np.float32))
        json.dump(dict(rows=n_rows, no_reply=n_noreply, examples=examples), open(os.path.join(OUT, f"{name}_meta.json"), "w"))
        print(f"{name}: {n_rows} rows, {sum(len(v) for v in recs.values())} sorted replies in {time.time() - t0:.0f} s", flush=True)


def rate_ci(x, rng):
    n, k = len(x), float(np.sum(x))
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = rng.binomial(n, p, N_BOOT) / n                  # parametric bootstrap of a proportion
    return p, *np.percentile(d, [2.5, 97.5])


def analyze():
    rng = np.random.default_rng(SEED)
    L = ["# Does OLMo's training data explain the pattern? Dolci exposure (spec P2; exploratory, outside every verdict, NOT BLIND)", "",
         "Plan and frozen sorting rules: `docs/decisions.md` 2026-11-05. Dolci-Instruct SFT, DPO and RL read by streaming (nothing stored). Each example is sorted by its system prompt and first user turn (no role / role in system prompt / role in user turn / other system prompt) and "
         "scored on its first reply after the first user turn with five regexes (markdown, stock phrases, `says_ai`, refusal, log words). DPO: chosen replies, rejected beside them. RL has prompts only. Limits: regex only; correlation, not cause; prompts differ between buckets, so the comparison is not like for like; "
         "Dolci is OLMo only and its role examples may look nothing like the atlas's light roles; 95% ranges are parametric bootstrap ranges of the proportions.", ""]
    stages = [("sft", "chosen"), ("dpo", "chosen"), ("dpo", "rejected")]
    data = {}
    for s, tag in stages:
        p = os.path.join(OUT, f"{s}_{tag}.npy")
        if os.path.exists(p):
            data[(s, tag)] = np.load(p)
    meta = {s: json.load(open(os.path.join(OUT, f"{s}_meta.json"))) for s in ("sft", "dpo", "rl") if os.path.exists(os.path.join(OUT, f"{s}_meta.json"))}
    L += ["## Counts per stage and bucket (first reply after the first user turn)", "", "| stage | rows | " + " | ".join(BUCKETS) + " | self-directed questions (frozen regex) | no reply |", "|---|---|" + "---|" * (len(BUCKETS) + 2)]
    for (s, tag), a in data.items():
        L.append(f"| {s} {tag} | {meta[s]['rows']} | " + " | ".join(str(int((a[:, 0] == i).sum())) for i in range(len(BUCKETS))) + f" | {int(a[:, 1].sum())} | {meta[s]['no_reply'] if tag == 'chosen' else '-'} |")
    if "rl" in meta:
        L.append(f"| rl | {meta['rl']['rows']} | prompts only: no replies to score, so RL is reported as having no usable replies | | | | | |")
    L.append("")
    readings = []
    L += ["## Rates (% of replies) with 95% ranges, by bucket", "", "| stage | bucket | n | markdown | stock phrases | says_ai | refusal | mean log words |", "|---|---|---|---|---|---|---|---|"]
    rates = {}
    for (s, tag), a in data.items():
        for i, b in enumerate(BUCKETS):
            x = a[a[:, 0] == i]
            if not len(x):
                continue
            cells = []
            for j, d in enumerate(DETS[:4]):
                p, lo, hi = rate_ci(x[:, 2 + j] > 0, rng)
                rates[(s, tag, b, d)] = (p, lo, hi, len(x))
                cells.append(f"{100 * p:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]")
            L.append(f"| {s} {tag} | {b} | {len(x)} | " + " | ".join(cells) + f" | {x[:, 6].mean():.2f} |")
    L += ["", "## Readings (each stage, role in system prompt and role in user turn separately, against no role; chosen replies)", "",
          "| stage | role bucket | role examples | no-role replies with says_ai | markdown: role / no role | says_ai: role / no role | reading |", "|---|---|---|---|---|---|---|"]
    for s in ("sft", "dpo"):
        if (s, "chosen") not in data:
            continue
        a = data[(s, "chosen")]
        n_ai = int(((a[:, 0] == 0) & (a[:, 4] > 0)).sum())
        for b in ("role_system", "role_user"):
            md_r, md_n = rates.get((s, "chosen", b, "md")), rates.get((s, "chosen", "no_role", "md"))
            ai_r, ai_n = rates.get((s, "chosen", b, "says_ai")), rates.get((s, "chosen", "no_role", "says_ai"))
            if md_r is None or md_n is None:
                continue
            nrole = md_r[3]
            if nrole < 200 or n_ai < 20:
                rd = "TOO FEW (under 200 role examples, or under 20 no-role replies with says_ai)"
            else:
                rm = md_r[0] / md_n[0] if md_n[0] > 0 else np.nan
                ra = ai_r[0] / ai_n[0] if ai_n[0] > 0 else np.nan
                if rm >= 0.75 and ra <= 0.25:
                    rd = "TRAINING MIX COULD EXPLAIN THE SPLIT"
                elif rm < 0.50:
                    rd = "TRAINING MIX DOES NOT EXPLAIN MARKDOWN"
                else:
                    rd = f"inconclusive (markdown ratio {rm:.2f}, says_ai ratio {ra:.2f})"
            L.append(f"| {s} | {b} | {nrole} | {n_ai} | {100 * md_r[0]:.1f} / {100 * md_n[0]:.1f} | {100 * ai_r[0] if ai_r else float('nan'):.1f} / {100 * ai_n[0] if ai_n else float('nan'):.1f} | {rd} |")
    L += ["", "## Split by self-directed questions (frozen regex), chosen replies", "", "| stage | bucket | self-directed n | markdown | says_ai | other n | markdown | says_ai |", "|---|---|---|---|---|---|---|---|"]
    for s in ("sft", "dpo"):
        if (s, "chosen") not in data:
            continue
        a = data[(s, "chosen")]
        for i, b in enumerate(BUCKETS[:3]):
            x = a[a[:, 0] == i]
            sq, ot = x[x[:, 1] == 1], x[x[:, 1] == 0]
            if len(sq) and len(ot):
                L.append(f"| {s} | {b} | {len(sq)} | {100 * (sq[:, 2] > 0).mean():.1f} | {100 * (sq[:, 5 - 1] > 0).mean():.1f} | {len(ot)} | {100 * (ot[:, 2] > 0).mean():.1f} | {100 * (ot[:, 5 - 1] > 0).mean():.1f} |")
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "dolci_exposure.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:30]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "analyze"])
    ap.add_argument("--max-rows", type=int, default=0)
    a = ap.parse_args()
    scan(a.max_rows) if a.cmd == "scan" else analyze()


if __name__ == "__main__":
    main()
