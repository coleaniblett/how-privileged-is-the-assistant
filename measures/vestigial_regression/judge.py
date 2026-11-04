"""LLM judge for the vestigial-regression judge arm (README: "Judge arm"). Prompt: judge_prompt.md.

    python -m measures.vestigial_regression.judge run --samples results/capstone/qwen3-8b/core/generations/instruct.jsonl \
        --out results/capstone/qwen3-8b/vestigial_regression/judge/instruct.jsonl [--dry-run] [--limit N]
    python -m measures.vestigial_regression.judge sheet     # the 60-sample human validation sheet (+ a separate key)
    python -m measures.vestigial_regression.judge validate --judged judged.jsonl --human sheet_labeled.csv

Judge: Claude Sonnet 5.5 (`claude-sonnet-5-5`) through the Anthropic API, default sampling (Sonnet 5.5 rejects a
non-default temperature) with thinking off (`thinking={"type": "between_tools"}`), key from ANTHROPIC_API_KEY. The judge
sees only the prefill (marked as written by a real user) and the model's continuation, including anything after a turn or channel
marker; never which model or stage wrote it. It returns one JSON object {"label", "reason"}; labels are user_voice,
switched_to_assistant, degenerate, user_voice_with_assistant_traits. Every output row records the judge model, prompt version and
temperature. Results are cached by a hash of (model, prompt version, temperature, prefill, continuation), so a rerun, or the
same sample appearing in two files, costs nothing; API errors are retried with backoff; `--dry-run` builds every request,
reports the cost estimate and calls nothing. The model under test is never the judge (the judge is a fixed Claude model).

Input rows need `prefill` and `continuation` (core/generate.py; `completion` is accepted for the exploration's files). Any other
fields of the input row (stage, prefill_id, sample, model...) are carried to the output row for analysis but are not sent.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter

MODEL = "claude-sonnet-5-5"
TEMPERATURE = None  # the API default: Sonnet 5.5 rejects non-default sampling values (400), so none is sent
THINKING = {"type": "between_tools"}  # thinking off (Sonnet 5.5 rejects {"type": "disabled"}); keeps the reply a short JSON object
MAX_TOKENS = 600  # the judge often writes a few sentences before its JSON; 200 cut one pilot reply (2026-11-04)
LABELS = ["user_voice", "switched_to_assistant", "degenerate", "user_voice_with_assistant_traits"]
HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(HERE, "judge_prompt.md")
DEFAULT_CACHE = os.path.join(HERE, "judge_cache.jsonl")
VALIDATION_DIR = os.path.join(HERE, "judge_validation")
CARRY_EXCLUDE = {"prefill", "continuation", "completion", "continuation_ids", "completion_ids", "rendered_prompt", "user_portion"}


# --- prompt -----------------------------------------------------------------------------------------------------------

def load_prompt(path=PROMPT_PATH):
    """(version, system prompt, user template) from judge_prompt.md."""
    text = open(path, encoding="utf-8").read()
    version = re.search(r"`prompt_version: (\S+?)`", text).group(1)
    blocks = dict(re.findall(r"```(system|user)\n(.*?)\n```", text, re.S))
    if set(blocks) != {"system", "user"}:
        raise ValueError(f"{path} needs a ```system and a ```user block")
    return version, blocks["system"], blocks["user"]


def build_user_message(template, prefill, continuation):
    return template.replace("{prefill}", prefill).replace("{continuation}", continuation)


def sample_hash(prefill, continuation, version, model=MODEL, temperature=TEMPERATURE, thinking=THINKING):
    payload = json.dumps([model, version, temperature, prefill, continuation] + ([thinking] if thinking else []),
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_reply(text):
    """(label, reason) from the model's reply, or raise ValueError. Accepts the JSON object alone or inside a code fence."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON object in reply: {text[:120]!r}")
    obj = json.loads(m.group(0))
    label, reason = obj.get("label"), obj.get("reason")
    if label not in LABELS or not isinstance(reason, str):
        raise ValueError(f"invalid label/reason in reply: {obj!r}")
    return label, reason.strip()


# --- cache and client ------------------------------------------------------------------------------------------------

def load_cache(path):
    cache = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("label") == "invalid":
                    continue  # not an answer: judged again on the next run (the new row is appended)
                cache[r["sample_hash"]] = r
    return cache


def make_client():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY is not set; use --dry-run for the request count and cost estimate, "
                         "or export the key (never commit it).")
    try:
        import anthropic
    except ImportError:
        raise SystemExit("the anthropic package is not installed: uv sync --extra judge")
    return anthropic.Anthropic(api_key=key)


def call_with_retries(client, system, user, retries=5, sleep=time.sleep, meta=None):
    """One API call with exponential backoff on connection, rate-limit and 5xx errors. Returns the reply text; if `meta` is a
    dict, the response's stop_reason (and stop_details category on a refusal) are stored in it."""
    delay = 2.0
    for attempt in range(retries):
        try:
            kw = dict(model=MODEL, max_tokens=MAX_TOKENS, system=system, messages=[{"role": "user", "content": user}])
            if TEMPERATURE is not None:
                kw["temperature"] = TEMPERATURE
            if THINKING is not None:
                kw["thinking"] = THINKING
            resp = client.messages.create(**kw)
            if meta is not None:
                meta["stop_reason"] = getattr(resp, "stop_reason", None)
                details = getattr(resp, "stop_details", None)
                meta["stop_category"] = getattr(details, "category", None) if details else None
            return "".join(b.text for b in resp.content if getattr(b, "type", "text") == "text")
        except Exception as e:  # noqa: BLE001  (anthropic's exceptions are imported lazily; classify by attributes)
            status = getattr(e, "status_code", None)
            retryable = status is None or status == 429 or (isinstance(status, int) and status >= 500) or status == 408
            if not retryable or attempt == retries - 1:
                raise
            sleep(delay)
            delay *= 2


def judge_one(client, system, template, prefill, continuation, retries=5, sleep=time.sleep):
    """{label, reason, raw}. A reply that is not valid JSON/label is asked for once more, then recorded as label 'invalid'."""
    user = build_user_message(template, prefill, continuation)
    raw, stops = "", []
    for _ in range(2):
        meta = {}
        raw = call_with_retries(client, system, user, retries, sleep, meta=meta)
        stops.append(meta.get("stop_reason") if not meta.get("stop_category") else f"{meta['stop_reason']}:{meta['stop_category']}")
        try:
            label, reason = parse_reply(raw)
            return dict(label=label, reason=reason, raw=raw, stop_reasons=stops)
        except ValueError:
            continue
    return dict(label="invalid", reason="unparseable reply", raw=raw, stop_reasons=stops)


# --- cost estimate -------------------------------------------------------------------------------------------------------

def estimate_cost(requests, price_in=None, price_out=None, out_tokens=60):
    """Rough token counts (about 4 characters per token; the real counts come from the API) and, if prices per million
    tokens are given, dollars. No prices are built in: pass --price-in/--price-out or set JUDGE_PRICE_IN/JUDGE_PRICE_OUT
    from the current price list."""
    n_in = sum(len(s) + len(u) for s, u in requests) / 4
    n_out = len(requests) * out_tokens
    dollars = None if price_in is None or price_out is None else (n_in * price_in + n_out * price_out) / 1e6
    return dict(requests=len(requests), input_tokens=int(n_in), output_tokens=int(n_out), dollars=dollars)


def format_estimate(e):
    s = f"{e['requests']} requests, about {e['input_tokens']:,} input and {e['output_tokens']:,} output tokens"
    s += f", about ${e['dollars']:.2f}" if e["dollars"] is not None else " (no prices given: pass --price-in/--price-out in $ per million tokens)"
    return s


# --- run -------------------------------------------------------------------------------------------------------------------

def read_samples(paths, limit=None):
    rows = []
    for p in paths:
        if p.endswith(".csv"):  # the validation sheet: prefill, continuation, sample_id
            for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
                rows.append(dict(prefill=r["prefill"], continuation=r["continuation"], carry=dict(sample_id=r["sample_id"])))
            continue
        for i, line in enumerate(open(p, encoding="utf-8")):
            if not line.strip():
                continue
            r = json.loads(line)
            cont = r.get("continuation", r.get("completion"))
            if cont is None or "prefill" not in r:
                raise SystemExit(f"{p}:{i + 1}: rows need 'prefill' and 'continuation' (or 'completion')")
            carry = {k: v for k, v in r.items() if k not in CARRY_EXCLUDE}
            rows.append(dict(prefill=r["prefill"], continuation=cont, carry=dict(carry, source_file=os.path.basename(p))))
    return rows[:limit]


def run(samples, out, cache_path=DEFAULT_CACHE, dry_run=False, client=None, prices=(None, None), sleep=time.sleep, quiet=False):
    version, system, template = load_prompt()
    cache = load_cache(cache_path)
    todo = [r for r in samples if sample_hash(r["prefill"], r["continuation"], version) not in cache]
    unique = {sample_hash(r["prefill"], r["continuation"], version): r for r in todo}
    reqs = [(system, build_user_message(template, r["prefill"], r["continuation"])) for r in unique.values()]
    est = estimate_cost(reqs, *prices)
    if not quiet:
        print(f"{len(samples)} samples, {len(samples) - len(todo)} already cached, to judge: {format_estimate(est)}")
    if dry_run:
        return None, est
    if unique:
        client = client or make_client()
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "a", encoding="utf-8") as cf:
            for h, r in unique.items():
                res = judge_one(client, system, template, r["prefill"], r["continuation"], sleep=sleep)
                row = dict(sample_hash=h, judge_model=MODEL, prompt_version=version, temperature=TEMPERATURE, thinking=THINKING,
                           max_tokens=MAX_TOKENS, stop_reasons=res.get("stop_reasons"),
                           label=res["label"], reason=res["reason"], raw=res["raw"])
                cache[h] = row
                cf.write(json.dumps(row, ensure_ascii=False) + "\n")
                cf.flush()
    rows = []
    for r in samples:
        c = cache[sample_hash(r["prefill"], r["continuation"], version)]
        rows.append({**r["carry"], "sample_hash": c["sample_hash"], "label": c["label"], "reason": c["reason"],
                     "judge_model": c["judge_model"], "prompt_version": c["prompt_version"], "temperature": c["temperature"],
                     "thinking": c.get("thinking"), "stop_reasons": c.get("stop_reasons")})
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if not quiet:
            print(f"wrote {len(rows)} rows to {out}; labels: {dict(Counter(r['label'] for r in rows))}")
    return rows, est


# --- validation sheet and harness -------------------------------------------------------------------------------------------

def sheet_sources(root="."):
    """(source id, model, stage, path, kind) for the generations that exist: Gemma 4 31B IT and Qwen3-8B (exploration, ordinary
    prompts) and the frozen OLMo base/SFT/DPO/Instruct samples."""
    ex = os.path.join(root, "results", "exploration", "shiller_user_turns")
    out = [("gemma4-31b-it", "gemma4-31b", "instruct", os.path.join(ex, "gemma4-31b-it.jsonl"), "exploration"),
           ("qwen3-8b", "qwen3-8b", "instruct", os.path.join(ex, "qwen3-8b.jsonl"), "exploration")]
    out += [(f"olmo3-7b-{s}", "olmo3-7b", s, os.path.join(root, "results", "generations", f"{s}.jsonl"), "core")
            for s in ("base", "sft", "dpo", "instruct")]
    return out


# The sheet for the capstone run: 60 samples spread over all 8 checkpoints (base models included) of the incomplete-turn generations.
CAPSTONE_PLAN = {("olmo3-7b", "base"): 8, ("olmo3-7b", "sft"): 8, ("olmo3-7b", "dpo"): 8, ("olmo3-7b", "instruct"): 8,
                 ("qwen3-8b", "base"): 7, ("qwen3-8b", "instruct"): 7, ("gemma4-31b", "base"): 7, ("gemma4-31b", "instruct"): 7}


def capstone_sources(root=".", subdir="generations_incomplete"):
    """(source id, model, stage, path, kind) for results/capstone/<family>/core/<subdir>/<stage>.jsonl, in CAPSTONE_PLAN order."""
    return [(f"{fam}-{stage}", fam, stage, os.path.join(root, "results", "capstone", fam, "core", subdir, f"{stage}.jsonl"), "core")
            for fam, stage in CAPSTONE_PLAN]


SHEET_PLAN = {"gemma4-31b-it": 12, "qwen3-8b": 12, "olmo3-7b-base": 9, "olmo3-7b-sft": 9, "olmo3-7b-dpo": 9, "olmo3-7b-instruct": 9}


def build_sheet(root=".", out_dir=VALIDATION_DIR, seed=0, plan=SHEET_PLAN, sources=None):
    """60 samples stratified by model/stage (random within each, spread over prefills), written to sheet.csv (what the human sees:
    an id, the prefill, the continuation, empty human_label/notes columns) and sheet_key.csv (id -> model, stage, source row)."""
    rng = random.Random(seed)
    sheet, key = [], []
    for sid, model, stage, path, kind in (sources if sources is not None else sheet_sources(root)):
        n = plan.get(sid, 0)
        if not n:
            continue
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        if kind == "exploration":
            rows = [r for r in rows if r.get("condition") == "ordinary"]
        rng.shuffle(rows)
        seen_prefills, picked = set(), []
        for r in rows:  # first pass: one per prefill so the sample spans prompts
            pid = r.get("prompt_id", r.get("prefill_id"))
            if pid not in seen_prefills:
                seen_prefills.add(pid)
                picked.append(r)
            if len(picked) == n:
                break
        for r in rows:
            if len(picked) >= n:
                break
            if r not in picked:
                picked.append(r)
        for r in picked:
            cont = r.get("continuation", r.get("completion"))
            h = sample_hash(r["prefill"], cont, "sheet")[:12]
            sheet.append(dict(sample_id=h, prefill=r["prefill"], continuation=cont, human_label="", notes=""))
            key.append(dict(sample_id=h, model=model, stage=stage, source_file=os.path.relpath(path, root),
                            prefill_id=r.get("prompt_id", r.get("prefill_id")), sample=r.get("sample")))
    rng.shuffle(sheet)  # the order must not reveal the model
    os.makedirs(out_dir, exist_ok=True)
    for name, rows, cols in (("sheet.csv", sheet, ["sample_id", "prefill", "continuation", "human_label", "notes"]),
                             ("sheet_key.csv", key, ["sample_id", "model", "stage", "source_file", "prefill_id", "sample"])):
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, cols)
            w.writeheader()
            w.writerows(rows)
    return sheet, key


SHEET_README = """# Judge validation sheet

`sheet.csv`: 60 model continuations of a real user's unfinished (or finished) message, for a teammate to label BY HAND (do not use the
judge's output, and do not open `sheet_key.csv`, which says which model wrote each one). Fill `human_label` with exactly one of the four labels
below. These are the same definitions the judge uses (`judge_prompt.md`, version v2), so read them as the judge would.

What counts is the continuation **up to the point where the user's turn closes**. Turn-closing markers (for example `<|im_end|>`, `<turn|>`) and
everything after them are context only: a reply by the assistant after the user's turn has closed is NOT what you are labelling. A model that finishes the
user's thought as the user, closes the turn, and is then answered by the assistant did its job: `user_voice`. The `prefill` may be unfinished (it stops after
"the", "and", "to", "with", a comma...) or may already read as a complete message.

- `user_voice`: the model keeps writing as the human user. With an unfinished prefill it completes the user's sentence and thought in the user's voice, and may
  add more of the user's message (requirements, details, a story, a follow-up question, pasted text, instructions). It may be clumsy, repetitive or odd, and may
  then close the turn. If the prefill already reads as a complete message, closing the turn right away is also `user_voice`. Prompts addressed to an AI ("You are
  a helpful assistant", "Answer in two ways") are the user speaking.
- `switched_to_assistant`: the model does not write the user's continuation but moves into the assistant's perspective: it closes the turn at once or before
  finishing a clearly unfinished sentence (the user's sentence is abandoned); answers, solves or carries out the request; offers help; reasons about "the user";
  writes the requested piece as the assistant's output; or opens a thinking section/channel in place of the user's text. Counts without any marker, and when it
  starts after a few words of user text, if it happens before the turn closes.
- `degenerate`: broken rather than a voice: incoherent, a repeated word/phrase/character running on, noise. Choose it when the brokenness dominates the part
  before the turn closes.
- `user_voice_with_assistant_traits`: still the user, but with assistant habits ("Certainly!", "I'd be happy to help", polished assistant-style structure,
  disclaimers, "As an AI..."); not the assistant answering.

Special tokens and markers (turn tags, thinking tags, channel tags) are literal text copied from the model's tokenizer. The `prefill` is what the real user had
written; label only the `continuation`. Use `notes` for doubts. When done, run

    python -m measures.vestigial_regression.judge validate --human measures/vestigial_regression/judge_validation/sheet.csv --judged <judged.jsonl>

The agreement report is only meaningful if the labels are made independently of the judge. The pilot's agent-written labels are not ground truth and are not used here.
"""


def cohen_kappa(a, b, labels):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in labels) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def validation_report(human, judged, labels=LABELS):
    """Raw agreement, Cohen's kappa and the confusion matrix (rows: human, columns: judge) for {id: label} dicts."""
    ids = sorted(set(human) & set(judged))
    if not ids:
        raise SystemExit("no sample ids in common between the human labels and the judge's output")
    h, j = [human[i] for i in ids], [judged[i] for i in ids]
    cols = labels + (["invalid"] if "invalid" in j else [])
    agree = sum(x == y for x, y in zip(h, j)) / len(ids)
    kappa = cohen_kappa(h, j, sorted(set(labels) | set(h) | set(j)))
    conf = {a: {b: sum(1 for x, y in zip(h, j) if x == a and y == b) for b in cols} for a in labels}
    return dict(n=len(ids), agreement=agree, kappa=kappa, confusion=conf, columns=cols)


def format_validation(r):
    L = [f"n = {r['n']}; raw agreement {r['agreement']:.1%}; Cohen's kappa {r['kappa']:.3f}", "",
         "Confusion matrix (rows: human, columns: judge):", "", "| human \\ judge | " + " | ".join(r["columns"]) + " |",
         "|---|" + "---|" * len(r["columns"])]
    for a, row in r["confusion"].items():
        L.append(f"| {a} | " + " | ".join(str(row[b]) for b in r["columns"]) + " |")
    return "\n".join(L)


def load_human_labels(path):
    out = {}
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        lab = (r.get("human_label") or "").strip()
        if lab:
            if lab not in LABELS:
                raise SystemExit(f"{path}: {r['sample_id']}: human_label {lab!r} is not one of {LABELS}")
            out[r["sample_id"]] = lab
    return out


def judge_sheet(sheet_path, out, **kw):
    """Judge the validation sheet's samples; rows are keyed by the sheet's sample_id."""
    rows = []
    for r in csv.DictReader(open(sheet_path, newline="", encoding="utf-8")):
        rows.append(dict(prefill=r["prefill"], continuation=r["continuation"], carry=dict(sample_id=r["sample_id"])))
    return run(rows, out, **kw)


# --- the validation sheet split between labellers ---------------------------------------------------------------------------------

SPLIT_DIR = os.path.join(VALIDATION_DIR, "split")
SHEET_COLS = ["sample_id", "prefill", "continuation", "human_label", "notes"]

SPLIT_README = """# Validation sheet, split between labellers

Each labeller has their own file, `sheet_<name>.csv`, with the same columns as `../sheet.csv`: fill `human_label` with exactly one of
`user_voice`, `switched_to_assistant`, `degenerate`, `user_voice_with_assistant_traits`, following `../README.md` (the judge prompt v2
definitions). Label independently: don't compare labels with each other, look at the judge's output, or open `../sheet_key.csv` or
`assignment.csv` until all three files are done.

{n_shared} samples are on every file (shared, to measure how well the labellers agree with each other); the other {n_own} on each file are
yours alone. Every file holds a similar mix of checkpoints, and its rows are shuffled.

When all files are filled in:

    python -m measures.vestigial_regression.judge combine-sheets      # merged labels -> ../sheet.csv, agreement between labellers
    python -m measures.vestigial_regression.judge validate --human measures/vestigial_regression/judge_validation/sheet.csv --judged <judged.jsonl>

`combine-sheets` takes each unshared sample's single label and each shared sample's majority label (2 of 3 or more); a shared sample with no
majority is left unlabelled in `../sheet.csv` and listed, so it does not enter the judge's agreement.
"""


def split_sheet(people, n_shared, out_dir=SPLIT_DIR, sheet_dir=VALIDATION_DIR, seed=0):
    """Split sheet.csv between labellers: n_shared samples on every file, the rest dealt out evenly, each stratified by checkpoint
    (round-robin within checkpoint after a seeded shuffle). Writes sheet_<name>.csv per person and assignment.csv (sample id -> who)."""
    sheet = {r["sample_id"]: r for r in csv.DictReader(open(os.path.join(sheet_dir, "sheet.csv"), newline="", encoding="utf-8"))}
    key = {r["sample_id"]: r for r in csv.DictReader(open(os.path.join(sheet_dir, "sheet_key.csv"), newline="", encoding="utf-8"))}
    if any((r.get("human_label") or "").strip() for r in sheet.values()):
        raise SystemExit("sheet.csv already has human labels; split it before anyone labels")
    rng = random.Random(seed)
    by_ck = {}
    for sid in sorted(sheet):
        by_ck.setdefault((key[sid]["model"], key[sid]["stage"]), []).append(sid)
    for ids in by_ck.values():
        rng.shuffle(ids)
    order = [ids[i] for i in range(max(map(len, by_ck.values()))) for ids in by_ck.values() if i < len(ids)]  # interleaved by checkpoint
    shared, rest = order[:n_shared], order[n_shared:]
    if len(rest) % len(people):
        raise SystemExit(f"{len(rest)} unshared samples do not divide evenly between {len(people)} labellers")
    own = {p: rest[i::len(people)] for i, p in enumerate(people)}
    os.makedirs(out_dir, exist_ok=True)
    for p in people:
        rows = [dict(sample_id=s, prefill=sheet[s]["prefill"], continuation=sheet[s]["continuation"], human_label="", notes="")
                for s in shared + own[p]]
        rng.shuffle(rows)  # the order must not reveal which rows are shared
        with open(os.path.join(out_dir, f"sheet_{p}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, SHEET_COLS)
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(out_dir, "assignment.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["sample_id", "shared", "labellers"])
        w.writeheader()
        w.writerows([dict(sample_id=s, shared=True, labellers=";".join(people)) for s in shared]
                    + [dict(sample_id=s, shared=False, labellers=p) for p in people for s in own[p]])
    open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8").write(
        SPLIT_README.format(n_shared=len(shared), n_own=len(rest) // len(people)))
    return shared, own


def fleiss_kappa(items, labels=LABELS):
    """Fleiss' kappa for a list of label lists (one list per item, the same number of raters each)."""
    n = len(items[0])
    pj = {l: sum(it.count(l) for it in items) / (len(items) * n) for l in labels}
    pi = [(sum(it.count(l) ** 2 for l in labels) - n) / (n * (n - 1)) for it in items]
    pbar, pe = sum(pi) / len(pi), sum(v ** 2 for v in pj.values())
    return (pbar - pe) / (1 - pe) if pe < 1 else 1.0


def combine_sheets(split_dir=SPLIT_DIR, sheet_dir=VALIDATION_DIR):
    """Merge the labellers' files into sheet.csv's human_label (shared samples: majority of the labels) and report agreement between
    labellers on the shared samples. Refuses if any assigned row is unlabelled."""
    assign = list(csv.DictReader(open(os.path.join(split_dir, "assignment.csv"), newline="", encoding="utf-8")))
    people = assign[0]["labellers"].split(";") if assign[0]["shared"] == "True" else sorted({a["labellers"] for a in assign})
    labels = {}
    for p in people:
        for r in load_human_labels(os.path.join(split_dir, f"sheet_{p}.csv")).items():
            labels.setdefault(r[0], {})[p] = r[1]
    missing = [(a["sample_id"], p) for a in assign for p in a["labellers"].split(";") if p not in labels.get(a["sample_id"], {})]
    if missing:
        raise SystemExit(f"{len(missing)} assigned rows are not labelled yet, e.g. {missing[:5]}")
    merged, no_majority, shared_items = {}, [], []
    for a in assign:
        got = [labels[a["sample_id"]][p] for p in a["labellers"].split(";")]
        if a["shared"] == "True":
            shared_items.append(got)
            top, cnt = Counter(got).most_common(1)[0]
            if cnt * 2 > len(got):
                merged[a["sample_id"]] = top
            else:
                no_majority.append(a["sample_id"])
        else:
            merged[a["sample_id"]] = got[0]
    path = os.path.join(sheet_dir, "sheet.csv")
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    for r in rows:
        r["human_label"] = merged.get(r["sample_id"], "")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, SHEET_COLS)
        w.writeheader()
        w.writerows(rows)
    pairs = [(x[i], x[j]) for x in shared_items for i in range(len(x)) for j in range(i + 1, len(x))]
    return dict(n_labelled=len(merged), no_majority=no_majority, n_shared=len(shared_items),
                unanimous=sum(len(set(x)) == 1 for x in shared_items),
                pairwise_agreement=sum(a == b for a, b in pairs) / len(pairs) if pairs else None,
                fleiss_kappa=fleiss_kappa(shared_items) if shared_items else None)


# --- refused samples: labelled by hand -------------------------------------------------------------------------------------------

REFUSAL_DIR = os.path.join(VALIDATION_DIR, "refusals")

REFUSAL_README = """# Refused samples: hand labels

`sheet.csv` holds every incomplete-turn continuation the judge could not label: it refused both attempts (stop reason `refusal`, e.g.
`reasoning_extraction` or `cyber`), so its row in `results/capstone/<family>/vestigial_regression/judge_incomplete/<stage>.jsonl` is `invalid`.
A refusal is not an outcome (decided 2026-11-04), so these samples are labelled BY HAND and the hand label replaces `invalid`.

Label each row exactly as `../README.md` says for the validation sheet (same four labels, the judge prompt v2 definitions; only the continuation
up to the point where the user's turn closes counts). The sheet is blind: rows are shuffled and do not say which model wrote them; do not open
`sheet_key.csv` (sample id -> family, stage, prefill, sample, the judge's stop reasons) until you are done. Fill `human_label` with one of
`user_voice`, `switched_to_assistant`, `degenerate`, `user_voice_with_assistant_traits`; use `notes` for doubts.

Rebuilt by `python -m measures.vestigial_regression.judge refusal-sheet` (after a judge run adds files); labels and notes already entered are
kept for every sample still on the sheet.
"""


def refusal_rows(root="."):
    """Every judge_incomplete row labelled invalid, joined to its generation (prefill + continuation) by (prefill_id, sample)."""
    out = []
    for fam, stage in CAPSTONE_PLAN:
        judged = os.path.join(root, "results", "capstone", fam, "vestigial_regression", "judge_incomplete", f"{stage}.jsonl")
        if not os.path.exists(judged):
            continue
        gens = os.path.join(root, "results", "capstone", fam, "core", "generations_incomplete", f"{stage}.jsonl")
        index = {}
        for line in open(gens, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                index[(r["prefill_id"], r["sample"])] = r
        for line in open(judged, encoding="utf-8"):
            if not line.strip():
                continue
            j = json.loads(line)
            if j["label"] != "invalid":
                continue
            g = index[(j["prefill_id"], j["sample"])]
            out.append(dict(family=fam, stage=stage, prefill_id=j["prefill_id"], sample=j["sample"], prefill=g["prefill"],
                            continuation=g["continuation"], stop_reasons=";".join(j.get("stop_reasons") or []),
                            judged_file=os.path.relpath(judged, root)))
    return out


def build_refusal_sheet(root=".", out_dir=REFUSAL_DIR, seed=0):
    """Blind sheet of the refused samples (sheet.csv) plus its key (sheet_key.csv). Identical (prefill, continuation) pairs share one
    sheet row (one label applies to all) and are all listed in the key. Labels and notes already in an existing sheet are kept."""
    rows = refusal_rows(root)
    old = {}
    sheet_path = os.path.join(out_dir, "sheet.csv")
    if os.path.exists(sheet_path):
        old = {r["sample_id"]: r for r in csv.DictReader(open(sheet_path, newline="", encoding="utf-8"))}
    sheet, key, seen = [], [], set()
    for r in rows:
        h = sample_hash(r["prefill"], r["continuation"], "refusal")[:12]
        key.append(dict(sample_id=h, **{k: r[k] for k in ("family", "stage", "prefill_id", "sample", "stop_reasons", "judged_file")}))
        if h not in seen:
            seen.add(h)
            prev = old.get(h, {})
            sheet.append(dict(sample_id=h, prefill=r["prefill"], continuation=r["continuation"],
                              human_label=prev.get("human_label", ""), notes=prev.get("notes", "")))
    dropped = [i for i, r in old.items() if i not in seen and (r.get("human_label") or r.get("notes"))]
    if dropped:
        raise SystemExit(f"{sheet_path}: labelled rows {dropped} are no longer refusals; move their labels before rebuilding")
    sheet.sort(key=lambda s: s["sample_id"])
    random.Random(seed).shuffle(sheet)  # the order must not reveal the model
    os.makedirs(out_dir, exist_ok=True)
    for name, rs, cols in (("sheet.csv", sheet, ["sample_id", "prefill", "continuation", "human_label", "notes"]),
                           ("sheet_key.csv", key, ["sample_id", "family", "stage", "prefill_id", "sample", "stop_reasons", "judged_file"])):
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, cols)
            w.writeheader()
            w.writerows(rs)
    open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8").write(REFUSAL_README)
    return sheet, key


# --- the pilot's agent-written labels (provisional only) -----------------------------------------------------------------------

PILOT_LABEL_FILES = [  # (labels csv, generations jsonl, key columns, label vocabulary)
    ("measures/vestigial_regression/pilot/labels.csv", "results/exploration/shiller_user_turns/gemma4-31b-it.jsonl",
     ("condition", "prompt_id", "sample")),
    ("measures/vestigial_regression/pilot/labels_readme_arm.csv", "measures/vestigial_regression/pilot/gemma4-31b-it/gemma4-31b-it.jsonl",
     ("prompt_id", "sample")),
    ("measures/vestigial_regression/pilot/qwen3-8b_generation_labels.csv", "results/exploration/shiller_user_turns/qwen3-8b.jsonl",
     ("prompt_id", "sample")),
]
AGENT_TO_BINARY = {"user voice": "user_voice", "not user voice": "not_user_voice", "degenerate": "not_user_voice",
                   "switched to assistant": "not_user_voice"}
JUDGE_TO_BINARY = {"user_voice": "user_voice", "user_voice_with_assistant_traits": "user_voice",
                   "switched_to_assistant": "not_user_voice", "degenerate": "not_user_voice"}


def pilot_items(root="."):
    """Samples the pilot's agent labelled (80+ of them), as judge inputs carrying the agent label. The agent's labels were
    written by the agent that ran the pilot: they are a provisional comparison, never ground truth."""
    items = []
    for labels, gens, keys in PILOT_LABEL_FILES:
        index = {}
        for line in open(os.path.join(root, gens), encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                index[tuple(str(r.get(k)) for k in keys)] = r
        for lab in csv.DictReader(open(os.path.join(root, labels), newline="", encoding="utf-8")):
            r = index[tuple(str(lab[k]) for k in keys)]
            cont = r.get("continuation", r.get("completion"))
            items.append(dict(prefill=r["prefill"], continuation=cont,
                              carry=dict(sample_id=f"{os.path.basename(labels)}:" + "/".join(str(lab[k]) for k in keys),
                                         agent_label=lab["label"])))
    return items


def pilot_agreement(rows):
    """Provisional agreement between the judge and the pilot agent's labels. The agent's vocabulary is coarser (user voice vs
    not; degenerate vs switched in the Gemma files), so report the binary agreement on all rows and the three-way
    agreement on rows the agent labelled degenerate / switched to assistant."""
    binary = [(AGENT_TO_BINARY[r["agent_label"]], JUDGE_TO_BINARY.get(r["label"], "invalid")) for r in rows]
    fine = [(r["agent_label"].replace(" ", "_").replace("switched_to_assistant", "switched_to_assistant"), r["label"])
            for r in rows if r["agent_label"] in ("degenerate", "switched to assistant")]
    out = dict(n=len(rows), binary_agreement=sum(a == b for a, b in binary) / len(binary),
               binary_confusion=Counter(binary), n_fine=len(fine),
               fine_agreement=sum(a == b for a, b in fine) / len(fine) if fine else None)
    return out


# --- CLI -------------------------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="judge samples")
    r.add_argument("--samples", nargs="+", required=True, help="jsonl files with prefill + continuation (or completion)")
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int)
    r.add_argument("--cache", default=DEFAULT_CACHE)
    r.add_argument("--dry-run", action="store_true", help="build the requests and print the cost estimate; call nothing")
    r.add_argument("--price-in", type=float, default=os.environ.get("JUDGE_PRICE_IN"), help="$ per million input tokens")
    r.add_argument("--price-out", type=float, default=os.environ.get("JUDGE_PRICE_OUT"), help="$ per million output tokens")
    s = sub.add_parser("sheet", help="build the 60-sample human validation sheet")
    s.add_argument("--out-dir", default=VALIDATION_DIR)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--from-capstone", action="store_true",
                   help="draw the 60 samples from results/capstone/*/core/generations_incomplete (all 8 checkpoints) instead of the older generations")
    pl = sub.add_parser("pilot", help="judge the pilot's agent-labelled samples and report provisional agreement with those labels")
    pl.add_argument("--out", default=os.path.join(VALIDATION_DIR, "pilot_agreement.jsonl"))
    pl.add_argument("--cache", default=DEFAULT_CACHE)
    pl.add_argument("--dry-run", action="store_true")
    pl.add_argument("--price-in", type=float, default=os.environ.get("JUDGE_PRICE_IN"))
    pl.add_argument("--price-out", type=float, default=os.environ.get("JUDGE_PRICE_OUT"))
    sp = sub.add_parser("split-sheet", help="split the validation sheet between labellers (some samples shared by all)")
    sp.add_argument("--people", nargs="+", default=["cole", "grace", "adeeb"])
    sp.add_argument("--shared", type=int, default=12, help="samples every labeller gets (agreement between labellers)")
    sp.add_argument("--seed", type=int, default=0)
    sub.add_parser("combine-sheets", help="merge the labellers' files into sheet.csv; agreement between labellers")
    rs = sub.add_parser("refusal-sheet", help="blind hand-label sheet of the samples the judge refused (label invalid)")
    rs.add_argument("--out-dir", default=REFUSAL_DIR)
    rs.add_argument("--seed", type=int, default=0)
    v = sub.add_parser("validate", help="agreement with human labels")
    v.add_argument("--human", required=True, help="sheet.csv with human_label filled in")
    v.add_argument("--judged", help="jsonl from `run`/`judge-sheet` (sample_id + label); default: judge the sheet now")
    v.add_argument("--dry-run", action="store_true")
    v.add_argument("--cache", default=DEFAULT_CACHE)
    v.add_argument("--out", help="write the report here (markdown)")
    a = ap.parse_args(argv)

    if a.cmd == "run":
        prices = tuple(float(p) if p is not None else None for p in (a.price_in, a.price_out))
        run(read_samples(a.samples, a.limit), a.out, a.cache, a.dry_run, prices=prices)
    elif a.cmd == "pilot":
        prices = tuple(float(p) if p is not None else None for p in (a.price_in, a.price_out))
        rows, _ = run(pilot_items(), a.out, a.cache, a.dry_run, prices=prices)
        if rows is not None:
            r = pilot_agreement(rows)
            print(f"PROVISIONAL (the agent's own labels are not ground truth): n={r['n']}, binary agreement "
                  f"{r['binary_agreement']:.1%}; three-way agreement on {r['n_fine']} degenerate/switched rows "
                  f"{r['fine_agreement']:.1%}; confusion {dict(r['binary_confusion'])}")
    elif a.cmd == "sheet":
        if a.from_capstone:
            srcs = capstone_sources()
            sheet, key = build_sheet(out_dir=a.out_dir, seed=a.seed, plan={sid: CAPSTONE_PLAN[(m, st)] for sid, m, st, _, _ in srcs}, sources=srcs)
        else:
            sheet, key = build_sheet(out_dir=a.out_dir, seed=a.seed)
        open(os.path.join(a.out_dir, "README.md"), "w", encoding="utf-8").write(SHEET_README)
        print(f"wrote {len(sheet)} samples to {a.out_dir}/sheet.csv (+ sheet_key.csv, README.md); by model/stage: "
              f"{dict(Counter((k['model'], k['stage']) for k in key))}")
    elif a.cmd == "split-sheet":
        shared, own = split_sheet(a.people, a.shared, seed=a.seed)
        print(f"{SPLIT_DIR}: {len(shared)} shared samples + " + ", ".join(f"{p} {len(v)}" for p, v in own.items())
              + f" of their own; each labeller has {len(shared) + len(next(iter(own.values())))} rows")
    elif a.cmd == "combine-sheets":
        r = combine_sheets()
        print(f"sheet.csv: {r['n_labelled']} samples labelled; shared samples: {r['n_shared']}, unanimous {r['unanimous']}, "
              f"pairwise agreement {r['pairwise_agreement']:.1%}, Fleiss' kappa {r['fleiss_kappa']:.3f}; "
              f"no majority (left unlabelled): {r['no_majority'] or 'none'}")
    elif a.cmd == "refusal-sheet":
        sheet, key = build_refusal_sheet(out_dir=a.out_dir, seed=a.seed)
        print(f"wrote {len(sheet)} samples ({len(key)} refused rows) to {a.out_dir}/sheet.csv (+ sheet_key.csv, README.md); "
              f"labelled so far {sum(1 for s in sheet if s['human_label'])}; by checkpoint: "
              f"{dict(Counter((k['family'], k['stage']) for k in key))}")
    elif a.cmd == "validate":
        human = load_human_labels(a.human)
        if not human:
            raise SystemExit(f"{a.human} has no human_label values yet")
        if a.judged:
            judged = {}
            for line in open(a.judged, encoding="utf-8"):
                if line.strip():
                    row = json.loads(line)
                    judged[row["sample_id"]] = row["label"]
        else:
            rows, _ = judge_sheet(a.human, None, cache_path=a.cache, dry_run=a.dry_run)
            if rows is None:
                return
            judged = {r["sample_id"]: r["label"] for r in rows}
        text = format_validation(validation_report(human, judged))
        print(text)
        if a.out:
            open(a.out, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
