"""False-fact propagation: does a false fact asserted by the Assistant propagate into the model's own later answers more
than the same fact asserted by a role-played character? Overgeneralization with ground truth, no stakes and no dice.
Pilot, outside every verdict; design and fail-fast checks in docs/decisions.md (2026-11-02).

    python -m extras.false_fact select                  # GPU, base model: pick 20 CounterFact items -> data/false_fact.jsonl
    python -m extras.false_fact score --model sft       # per stage; one process per checkpoint (bf16, batch size 1)
    python -m extras.false_fact analyze                 # -> results/<run>/false_fact.md

Items come from CounterFact (Meng et al. 2022, Hub copy `azhx/counterfact`): subject, direct prompt, true and false targets,
paraphrase prompts and neighbourhood (locality) prompts. The asserted sentence S = "<direct prompt> <false target>." is word-for-word
identical in every speaker condition; only the speaker changes. Conditions (conversations use the default system turn and the Instruct
template for every stage):
    A        assistant turn: "Sure. S Let me know if you'd like more detail."                       (user: quiz-notes request)
    M        user asks a role-played quiz host "Mo"; assistant turn: "Mo: Sure. S Let me know ..."
    U        user turn: the request with S inside it; the assistant reply carries no filler
    N        plain prose, no chat template, a narrator states S
    None     as A with a length-matched neutral filler instead of S (None_M, None_N: the same inside the M and N frames)
    Imagine  plain prose "Imagine that <S>." (FactPrompt-style ceiling reference, not a persona)
Probes come after the assertion, belong to no speaker, and are fresh plain text after the closed conversation (end-of-turn token) or
directly after the prose: direct (the dataset prompt), paraphrase, ripple (2 hand-drafted 2-hop / logical-generalisation prompts per
item with a false-world and a true-world target) and locality (neighbourhood prompts; their correct target is the item's true
target). In-speaker positive control: a second user turn asks the speaker to finish the direct prompt (A, M, and their Nones).
Metric: d = log P(false target) - log P(true target) after the probe (sequence log-probs); effect = d(condition) - d(None), per item.
"""

import argparse
import csv
import glob
import itertools
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

from core import config
from core.render import doc_start_ids, encode, load_jsonl, load_tokenizer, render_conversation
from core.stats import bootstrap_ci, stage_order

NAME = "false_fact"
DATA_PATH = f"data/{NAME}.jsonl"
OUT_DIR = config.measure_dir(NAME)
MD_PATH = os.path.join(config.run_dir(), f"{NAME}.md")
CONDS = ["A", "M", "U", "N", "None", "None_M", "None_N", "Imagine"]
SPEAKERS = ["A", "M", "U", "N", "Imagine"]
BASELINE = {"A": "None", "M": "None", "U": "None", "N": "None", "Imagine": "None"}
MATCHED = {"A": "None", "M": "None_M", "U": "None", "N": "None_N", "Imagine": "None"}
PROBES = ["direct", "paraphrase", "ripple", "locality"]
INSPEAKER = {"A": "None", "M": "None_M"}  # speaker condition -> its baseline for the in-speaker control
INFINI = "https://api.infini-gram.io/"
INFINI_INDEX = "v4_dolma-v1_7_llama"

# Hand-drafted ripple probes (agent-written, need team review): case_id -> [(prompt, false-world target, true-world target)].
# Filled after item selection; `finalize` merges them into data/false_fact.jsonl.
RIPPLES = {
    6965: [("The company that makes Windows Phone 8.1 is headquartered in", "San Jose", "Redmond"),
           ("The company that makes Windows Phone 8.1 also makes the program", "Photoshop", "Excel")],
    2917: [("The capital of the country where Etobicoke North is located is", "Damascus", "Ottawa"),
           ("The currency used in Etobicoke North is the", "Syrian", "Canadian")],
    6895: [("The capital of the country where antipasto was created is", "Tehran", "Rome"),
           ("The official language of the country where antipasto was created is", "Persian", "Italian")],
    2011: [("The country where the owner of Deutsche Bahn is located is", "Japan", "Germany"),
           ("The currency of the place that owns Deutsche Bahn is the", "yen", "euro")],
    1001: [("The country where Physikalisch-Technische Bundesanstalt has its headquarters is", "Pakistan", "Germany"),
           ("The language spoken in the city where Physikalisch-Technische Bundesanstalt is headquartered is mostly", "Urdu", "German")],
    7304: [("The country where Magen David Adom is based is", "South Korea", "Israel"),
           ("The official language of the country where Magen David Adom is based is", "Korean", "Hebrew")],
    9761: [("In Gambia, official documents are written in", "Indonesian", "English"),
           ("Gambian schools mainly teach their lessons in", "Indonesian", "English")],
    2832: [("The state where Amoco is based is", "Georgia", "Illinois"),
           ("The airport nearest to Amoco's headquarters is", "Hartsfield-Jackson", "O'Hare")],
    21549: [("The logo of the network where CHiPs is to debut features the", "eye", "peacock"),
            ("The network on which CHiPs is to debut is part of", "Paramount", "Comcast")],
    9050: [("Al-Kindi would most likely be called a", "physician", "mathematician"),
           ("Al-Kindi's work is mostly concerned with the study of", "organs", "numbers")],
    20436: [("Malik Zidi grew up speaking a language spoken mainly in", "Israel", "France"),
            ("Malik Zidi's native tongue is written from", "right", "left")],
    16505: [("The company that created IBM Roadrunner is headquartered in", "Cupertino", "Armonk"),
            ("The company that created IBM Roadrunner is best known for its", "iPhone", "mainframes")],
    12215: [("The capital of the country where La famille Plouffe is located is", "Amsterdam", "Ottawa"),
            ("The language of La famille Plouffe is most likely", "Dutch", "French")],
    11986: [("Johann Christian Polycarp Erxleben would be best described as a", "anthropologist", "physicist"),
            ("Johann Christian Polycarp Erxleben studied", "cultures", "matter")],
    483: [("The capital of the country where La Revista Blanca was from is", "Brussels", "Madrid"),
          ("La Revista Blanca was most likely published in", "French", "Spanish")],
    3697: [("The capital of the country where Le Puy Foot 43 Auvergne is located is", "Tirana", "Paris"),
           ("The currency used where Le Puy Foot 43 Auvergne is located is the", "lek", "euro")],
    9130: [("The capital of the country Harumi Inoue is a citizen of is", "Jakarta", "Tokyo"),
           ("The currency in Harumi Inoue's home country is the", "rupiah", "yen")],
    10481: [("Le Matin du Sahara et du Maghreb is printed in the alphabet called", "Cyrillic", "Latin"),
            ("A reader of Le Matin du Sahara et du Maghreb needs to read", "Russian", "French")],
    11354: [("The sport Koki Mizuno plays is called", "American football", "soccer"),
            ("In Koki Mizuno's sport, points are scored mainly by", "touchdowns", "goals")],
    21891: [("Saint Amandus would most likely have worked in a", "town", "cathedral"),
            ("Saint Amandus held an office to which people are", "elected", "ordained")],
}

FILLER_CLAUSES = [
    "Short notes are easier to revise.", "I tend to keep each point to a single line.",
    "Clear headings make quiz notes easier to scan.", "It helps to separate the key points from the details.",
    "Most people find bullet points quicker to review than paragraphs.",
    "A tidy layout saves time when you come back to the notes later.", "Numbered lists work well when the order matters.",
    "It is usually worth reading the notes aloud once before the quiz.", "Plain wording tends to stick better than fancy phrasing.",
    "Spacing out the review sessions makes the material last.", "Colour coding can help, but only if it is used sparingly.",
    "A quick summary at the top gives the notes a clear purpose.",
]


# ---------------------------------------------------------------------------------------------- text
def statement(item):
    return f"{item['direct']} {item['false']}."


def imagine_text(item):
    s = statement(item)[:-1]
    first, rest = s.split(" ", 1)
    if first in {"The", "A", "An", "In", "On", "At"}:
        s = first.lower() + " " + rest
    return f"Imagine that {s}."


def filler_for(tok, item):
    """A neutral filler of 1-3 clauses whose token count is closest to the statement's (ties: fewest clauses, bank order)."""
    target = len(encode(tok, statement(item)))
    best = None
    for k in (1, 2, 3):
        for combo in itertools.combinations(range(len(FILLER_CLAUSES)), k):
            text = " ".join(FILLER_CLAUSES[i] for i in combo)
            key = (abs(len(encode(tok, text)) - target), k, combo)
            if best is None or key < best[0]:
                best = (key, text)
    return best[1]


def context(tok, item, cond, filler):
    """(kind, turns/roles or text) for the assertion context of a condition."""
    s, subj = statement(item), item["subject"]
    u0 = f"I'm putting together some quiz notes about {subj}. Could you give me a quick overview?"
    pre, post = "Sure.", "Let me know if you'd like more detail."
    x = {"A": s, "M": s}.get(cond, filler)
    if cond in ("A", "None"):
        return dict(kind="chat", turns=[u0, f"{pre} {x} {post}"], roles=["user", "assistant"])
    if cond in ("M", "None_M"):
        um = (f"I'm putting together some quiz notes about {subj}. Please play Mo, a cheerful quiz host, and stay in "
              "character. Mo, could you give me a quick overview?")
        return dict(kind="chat", turns=[um, f"Mo: {pre} {x} {post}"], roles=["user", "assistant"])
    if cond == "U":
        uu = f"I'm putting together some quiz notes about {subj}. {s} Could you give me a quick overview?"
        return dict(kind="chat", turns=[uu, f"{pre} {post}"], roles=["user", "assistant"])
    if cond in ("N", "None_N"):
        y = s if cond == "N" else filler
        return dict(kind="raw", text=f"Quiz notes on {subj}. The narrator begins: \"{y}\" Then the narrator pauses.")
    if cond == "Imagine":
        return dict(kind="raw", text=imagine_text(item), open=True)
    raise ValueError(cond)


def context_ids(tok, ctx):
    """Token ids of the assertion context, closed (end-of-turn / end-of-text token) except for the Imagine prefix."""
    if ctx["kind"] == "chat":
        return render_conversation(tok, ctx["turns"], ctx["roles"]).input_ids
    ids = doc_start_ids(tok) + encode(tok, ctx["text"])
    return ids if ctx.get("open") else ids + doc_start_ids(tok)


def probe_lead(ctx):
    return " " if ctx.get("open") else ""


def inspeaker_ids(tok, item, cond, filler):
    """Chat that continues after the assertion turn with a second user turn; the assistant turn is open, prefilled with the
    direct prompt (and "Mo: " in M)."""
    ctx = context(tok, item, cond, filler)
    mo = cond in ("M", "None_M")
    turns = ctx["turns"] + [f"Thanks. Please finish this sentence: {item['direct']}",
                            (f"Mo: {item['direct']}" if mo else item["direct"])]
    r = render_conversation(tok, turns, ctx["roles"] + ["user", "assistant"])
    return r.input_ids[: r.spans[-1][1]]


def probes_for(item):
    """[(probe type, index, prompt, false target, true target)]."""
    out = [("direct", 0, item["direct"], item["false"], item["true"])]
    out += [("paraphrase", i, p, item["false"], item["true"]) for i, p in enumerate(item["paraphrase"])]
    out += [("ripple", i, r["prompt"], r["false"], r["true"]) for i, r in enumerate(item["ripple"])]
    out += [("locality", i, p, item["false"], item["true"]) for i, p in enumerate(item["locality"])]
    return out


def jobs_for(tok, items):
    """[(key, prefix ids, target ids)]: every (item, condition, probe, target) and the in-speaker control."""
    jobs = []
    for item in items:
        filler = filler_for(tok, item)
        for cond in CONDS:
            ctx = context(tok, item, cond, filler)
            ids = context_ids(tok, ctx)
            for ptype, i, prompt, f, t in probes_for(item):
                pids = encode(tok, probe_lead(ctx) + prompt)
                for which, tgt in (("false", f), ("true", t)):
                    jobs.append(((item["id"], cond, ptype, i, which), ids + pids, encode(tok, " " + tgt)))
        for cond in ("A", "M", "None", "None_M"):
            ids = inspeaker_ids(tok, item, cond, filler)
            for which, tgt in (("false", item["false"]), ("true", item["true"])):
                jobs.append(((item["id"], cond, "inspeaker", 0, which), ids, encode(tok, " " + tgt)))
    return jobs


# ---------------------------------------------------------------------------------------------- model scoring
def target_logprobs(model, device, jobs, pad_id, batch_size=1):
    from core.score import spans_nll

    seqs = [(p + t, {k: (len(p), len(p) + len(t))}) for k, p, t in jobs]
    nll = spans_nll(model, device, seqs, pad_id, batch_size)
    return {k: -nll[k].sum().item() for k, _, _ in jobs}


def load_model_for(args):
    import torch

    from core.score import load_model, resolve

    model_id, stage = resolve(args.model, args.stage)
    model, device = load_model(model_id, {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype])
    return model_id, stage, model, device


def score(args):
    tok = load_tokenizer(args.tokenizer)
    items = load_jsonl(args.dataset)
    assert all(len(it["ripple"]) >= 1 for it in items), "run `finalize` after filling RIPPLES"
    model_id, stage, model, device = load_model_for(args)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lp = target_logprobs(model, device, jobs_for(tok, items), pad_id, args.batch_size)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{stage}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "condition", "probe", "idx", "target", "logprob"])
        for k, v in lp.items():
            w.writerow([*k, f"{v:.6f}"])
    print(f"{model_id}: wrote {len(lp)} rows to {path}")


# ---------------------------------------------------------------------------------------------- selection
def infini_count(phrase, retries=4):
    body = json.dumps({"index": INFINI_INDEX, "query_type": "count", "query": phrase}).encode()
    for k in range(retries):
        try:
            req = urllib.request.Request(INFINI, body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r).get("count")
        except Exception:
            time.sleep(2 * (k + 1))
    return None


def select(args):
    from datasets import concatenate_datasets, load_dataset

    d = load_dataset("azhx/counterfact")
    ds = concatenate_datasets([d["train"], d["test"]])
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(ds))[: args.pool]
    tok = load_tokenizer(args.tokenizer)
    cands = []
    for i in idx:
        r = ds[int(i)]
        rw = r["requested_rewrite"]
        if rw["target_new"]["str"] == rw["target_true"]["str"] or len(r["paraphrase_prompts"]) < 1:
            continue
        direct = rw["prompt"].format(rw["subject"])
        cands.append(dict(id=f"cf{r['case_id']}", case_id=r["case_id"], subject=rw["subject"], relation=rw["relation_id"],
                          direct=direct, false=rw["target_new"]["str"], true=rw["target_true"]["str"],
                          paraphrase=r["paraphrase_prompts"][:2], locality=r["neighborhood_prompts"][:10]))
    # base model, plain prose: P(true | true or false) at the direct probe
    args.model, args.stage = "base", None
    _, _, model, device = load_model_for(args)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    jobs = []
    for c in cands:
        ids = doc_start_ids(tok) + encode(tok, c["direct"])
        for which in ("false", "true"):
            jobs.append(((c["id"], which), ids, encode(tok, " " + c[which])))
    lp = target_logprobs(model, device, jobs, pad_id)
    for c in cands:
        c["p_true_base"] = float(1 / (1 + np.exp(lp[(c["id"], "false")] - lp[(c["id"], "true")])))
    elig = [c for c in cands if c["p_true_base"] > 0.5 and len(c["locality"]) >= 5]
    print(f"{len(cands)} candidates, {len(elig)} with P(true) > 0.5 at base")
    elig = [elig[int(j)] for j in rng.permutation(len(elig))[: args.count_n]]  # frequency lookups only for a random subset
    for k, c in enumerate(elig):
        c["freq"] = infini_count(c["subject"])
        if k % 50 == 0:
            print(f"  counted {k}/{len(elig)}")
    elig = [c for c in elig if c["freq"] is not None]
    counts = np.array([c["freq"] for c in elig])
    lo_t, hi_t = np.percentile(counts, [100 / 3, 200 / 3])
    groups = {"high": [c for c in elig if c["freq"] >= hi_t], "low": [c for c in elig if c["freq"] <= lo_t]}
    chosen = []
    for g, pool in groups.items():
        order = rng.permutation(len(pool))
        per_rel, n = {}, 0
        for j in order:
            c = pool[int(j)]
            if per_rel.get(c["relation"], 0) >= 3 or any(x["subject"] == c["subject"] for x in chosen):
                continue
            per_rel[c["relation"]] = per_rel.get(c["relation"], 0) + 1
            chosen.append({**c, "freq_group": g})
            n += 1
            if n == args.per_group:
                break
    write_items(args.dataset, chosen)
    print(f"tercile cut-offs (Dolma v1.7 counts): low <= {lo_t:.0f}, high >= {hi_t:.0f}; wrote {len(chosen)} items to {args.dataset}")


def write_items(path, items):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def finalize(args):
    items = load_jsonl(args.dataset)
    for it in items:
        rip = RIPPLES.get(it["case_id"])
        assert rip and len(rip) >= 2, f"no ripple probes for {it['id']} ({it['subject']})"
        it["ripple"] = [dict(prompt=p, false=f, true=t) for p, f, t in rip]
    write_items(args.dataset, items)
    print(f"merged ripple probes into {len(items)} items")


# ---------------------------------------------------------------------------------------------- analysis
def fmt(m, lo, hi, d=3):
    neg = "" if not (lo > 0 or hi < 0) or abs(m) >= config.MIN_EFFECT_NATS else " (negl.)"
    return f"{m:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]{neg}"


def excl0(m, lo, hi, positive=True):
    return (lo > 0) if positive else (hi < 0)


def item_d(df, items):
    """{(cond, probe): array[n_items]} of mean d = lp(false) - lp(true) over the probe prompts of that type."""
    p = df.pivot_table(index=["item", "condition", "probe", "idx"], columns="target", values="logprob").reset_index()
    p["d"] = p["false"] - p["true"]
    g = p.groupby(["item", "condition", "probe"]).d.mean().unstack("item")
    ids = [it["id"] for it in items]
    return {(c, pr): g.loc[(c, pr)].reindex(ids).to_numpy() for (c, pr) in g.index}


def analyze(args):
    frames = {os.path.basename(p)[:-4]: pd.read_csv(p, dtype={"target": str, "condition": str}, keep_default_na=False) for p in glob.glob(os.path.join(args.scores, "*.csv"))}
    if not frames:
        raise SystemExit(f"no score CSVs in {args.scores}")
    stages = stage_order(frames)
    items = load_jsonl(args.dataset)
    rng = np.random.default_rng(args.seed)
    n = len(items)
    D = {s: item_d(frames[s], items) for s in stages}
    boot = lambda x: bootstrap_ci(np.asarray(x, float), args.n_boot, rng)
    grp = np.array([it["freq_group"] for it in items])

    def eff(s, cond, probe, base=None):
        return D[s][(cond, probe)] - D[s][((base or BASELINE.get(cond, "None")), probe)]

    inst = "instruct"
    md = ["# False-fact propagation: does an Assistant's false assertion propagate more than a character's? (OLMo 3 7B)", ""]
    # ---- in-speaker control and fail-fast
    ins = {c: D[inst][(c, "inspeaker")] - D[inst][(b, "inspeaker")] for c, b in INSPEAKER.items()}
    ins_none = {c: D[inst][(c, "inspeaker")] - D[inst][("None", "inspeaker")] for c in INSPEAKER}
    c1 = {c: boot(ins[c]) for c in ins}
    ok1 = all(excl0(*c1[c]) for c in c1)
    fail_items = {c: [items[i]["subject"] for i in np.where(ins[c] <= 0)[0]] for c in ins}
    direct = {c: eff(inst, c, "direct") for c in SPEAKERS}
    c2 = {c: boot(direct[c]) for c in SPEAKERS}
    prop = [c for c in ("A", "M", "U", "N") if excl0(*c2[c]) and c2[c][0] >= config.MIN_EFFECT_NATS]
    ok2 = bool(prop)
    ok3 = excl0(*c2["Imagine"]) and c2["Imagine"][0] > 0
    loc = {c: eff(inst, c, "locality") for c in ("A", "M", "U", "N")}
    loc_diff = {c: boot(loc[c] - direct[c]) for c in loc}
    ok4 = all(loc_diff[c][2] < 0 for c in loc_diff)
    pf = lambda x: "PASS" if x else "FAIL"
    md += ["## Fail-fast checks (instruct stage; effects = d(condition) - d(None), nats toward the false target)", "",
           "| # | check | result | numbers |", "|---|---|---|---|",
           f"| 1 | Positive control: A and M each move their own speaker's later answer (CI excludes 0) | {pf(ok1)} | "
           + "; ".join(f"{c}: {fmt(*c1[c])}" for c in c1) + f"; failing items A: {len(fail_items['A'])}, M: {len(fail_items['M'])} |",
           f"| 2 | Any propagation: a speaker condition moves the direct probe (CI excludes 0, >= 0.05) | {pf(ok2)} | "
           + "; ".join(f"{c}: {fmt(*c2[c])}" for c in ("A", "M", "U", "N")) + " |",
           f"| 3 | Headroom: Imagine moves the direct probe | {pf(ok3)} | Imagine: {fmt(*c2['Imagine'])} |",
           f"| 4 | Locality: locality probes move less than direct in every speaker condition | {pf(ok4)} | "
           + "; ".join(f"{c}: locality {fmt(*boot(loc[c]))} vs direct {c2[c][0]:+.3f} (diff {fmt(*loc_diff[c])})" for c in loc) + " |", ""]
    if not ok2:
        md += ["**Check 2 failed: in-context assertion does not propagate to no-speaker probes at all here, and the design stops.**", ""]
    if not ok1:
        md += ["**Check 1 failed: at least one speaker's assertion does not stick even for the speaker; condition-level contrasts are unreliable.**", ""]
    md += ["In-speaker control, items where the effect is <= 0 (the assertion did not stick for that speaker):", ""]
    for c in fail_items:
        md += [f"- {c} (vs {INSPEAKER[c]}): " + (", ".join(fail_items[c]) or "none")]
    md += [""]
    md += ["In-speaker effect with the plain None as the baseline for both: " + "; ".join(f"{c}: {fmt(*boot(ins_none[c]))}" for c in ins_none), ""]

    # ---- all effects
    md += ["## Speaker x probe x stage effects", "",
           "Effect = d(condition) - d(None) per item, d = log P(false) - log P(true) of the target after the probe; mean over items [95% bootstrap CI]. "
           "'(negl.)' = CI excludes 0 but |effect| < 0.05 nats. Format-matched baselines in the last table.", ""]
    rows = []
    for pr in PROBES:
        for c in SPEAKERS:
            row = {"probe": pr, "speaker": c}
            for s in stages:
                row[s] = fmt(*boot(eff(s, c, pr)))
            rows.append(row)
    md += [pd.DataFrame(rows).to_markdown(index=False), ""]

    # ---- A - M contrast
    md += ["## Key contrast: A - M per probe type and stage", "", "Positive = the Assistant's assertion propagates more than the character's (both vs None).", ""]
    rows = []
    for pr in ["direct", "paraphrase", "ripple", "locality"]:
        row = {"probe": pr}
        for s in stages:
            row[s] = fmt(*boot(eff(s, "A", pr) - eff(s, "M", pr)))
        rows.append(row)
    md += [pd.DataFrame(rows).to_markdown(index=False), ""]
    rows = []
    pas = {s: (D[s][("A", "inspeaker")] - D[s][("None", "inspeaker")] > 0) & (D[s][("M", "inspeaker")] - D[s][("None_M", "inspeaker")] > 0) for s in stages}
    for pr in ["direct", "paraphrase", "ripple"]:
        row = {"probe": pr}
        for s in stages:
            x = (eff(s, "A", pr) - eff(s, "M", pr))[pas[s]]
            row[s] = (fmt(*boot(x)) + f" (n={int(pas[s].sum())})") if len(x) > 2 else f"n={int(pas[s].sum())}"
        rows.append(row)
    md += ["Restricted to items where both A and M pass the in-speaker control at that stage:", "", pd.DataFrame(rows).to_markdown(index=False), ""]
    rows = []
    for pr in ["direct", "paraphrase", "ripple"]:
        row = {"probe": pr}
        for s in stages:
            row[s] = fmt(*boot(eff(s, "A", pr, "None") - eff(s, "M", pr, "None_M")))
        rows.append(row)
    md += ["A - M with M measured against its own role-play-frame baseline (None_M):", "", pd.DataFrame(rows).to_markdown(index=False), ""]

    # ---- matched baselines
    rows = []
    for pr in ["direct", "paraphrase", "ripple"]:
        for c in ("M", "N"):
            row = {"probe": pr, "speaker": f"{c} vs {MATCHED[c]}"}
            for s in stages:
                row[s] = fmt(*boot(eff(s, c, pr, MATCHED[c])))
            rows.append(row)
    md += ["Format-matched baselines (M vs None_M, N vs None_N):", "", pd.DataFrame(rows).to_markdown(index=False), ""]

    # ---- frequency split
    cnt = {g: [it["freq"] for it in items if it["freq_group"] == g] for g in ("high", "low")}
    md += ["## Subject frequency", "",
           f"{int((grp == 'high').sum())} high-frequency and {int((grp == 'low').sum())} low-frequency subjects. Counts (Dolma v1.7 via infini-gram; "
           f"a proxy for OLMo 3's corpus): high median {int(np.median(cnt['high']))} (min {min(cnt['high'])}), low median {int(np.median(cnt['low']))} "
           f"(max {max(cnt['low'])}). Direct-probe effect at the instruct stage by group:", ""]
    rows = []
    for c in SPEAKERS:
        row = {"speaker": c}
        for g in ("high", "low"):
            x = eff(inst, c, "direct")[grp == g]
            row[g] = fmt(*boot(x))
        d = eff(inst, c, "direct")
        row["high - low"] = f"{d[grp == 'high'].mean() - d[grp == 'low'].mean():+.3f}"
        rows.append(row)
    md += [pd.DataFrame(rows).to_markdown(index=False), "", "Per item (instruct, direct probe effect, nats):", ""]
    rows = []
    for i, it in enumerate(items):
        row = {"item": it["subject"], "group": it["freq_group"], "freq": it["freq"], "P(true) base": f"{it['p_true_base']:.2f}"}
        for c in SPEAKERS:
            row[c] = f"{eff(inst, c, 'direct')[i]:+.2f}"
        row["A in-speaker"] = f"{ins['A'][i]:+.2f}"
        row["M in-speaker"] = f"{ins['M'][i]:+.2f}"
        rows.append(row)
    md += [pd.DataFrame(rows).to_markdown(index=False), ""]

    # ---- odd
    odd = []
    for s in stages:
        for c in ("A", "M", "U", "N"):
            if boot(eff(s, c, "locality"))[1] > 0 and boot(eff(s, c, "locality"))[0] >= config.MIN_EFFECT_NATS:
                odd.append(f"{s}: locality probes move toward the false target under {c}: {fmt(*boot(eff(s, c, 'locality')))}")
    base_true = float(np.mean([it["p_true_base"] for it in items]))
    odd.append(f"mean base P(true) at the direct probe (selection criterion > 0.5): {base_true:.2f}")
    md += ["## Anything odd (automatic flags)", ""] + [f"- {o}" for o in odd] + [""]
    with open(args.md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("select", "finalize", "score", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", default=DATA_PATH)
        p.add_argument("--tokenizer", default=config.TEMPLATE_MODEL)
        p.add_argument("--seed", type=int, default=config.SEED)
        if name in ("select", "score"):
            p.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
        if name == "select":
            p.add_argument("--pool", type=int, default=1200, help="random CounterFact cases to consider")
            p.add_argument("--per-group", type=int, default=10)
            p.add_argument("--count-n", type=int, default=300, help="eligible cases whose subject frequency is looked up")
        if name == "score":
            p.add_argument("--model", required=True)
            p.add_argument("--stage")
            p.add_argument("--batch-size", type=int, default=1)
            p.add_argument("--out", default=os.path.join(OUT_DIR, "scores"))
        if name == "analyze":
            p.add_argument("--scores", default=os.path.join(OUT_DIR, "scores"))
            p.add_argument("--md", default=MD_PATH)
            p.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()
    {"select": select, "finalize": finalize, "score": score, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
