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
MAX_TOKENS = 200
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


def call_with_retries(client, system, user, retries=5, sleep=time.sleep):
    """One API call with exponential backoff on connection, rate-limit and 5xx errors. Returns the reply text."""
    delay = 2.0
    for attempt in range(retries):
        try:
            kw = dict(model=MODEL, max_tokens=MAX_TOKENS, system=system, messages=[{"role": "user", "content": user}])
            if TEMPERATURE is not None:
                kw["temperature"] = TEMPERATURE
            if THINKING is not None:
                kw["thinking"] = THINKING
            resp = client.messages.create(**kw)
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
    raw = ""
    for _ in range(2):
        raw = call_with_retries(client, system, user, retries, sleep)
        try:
            label, reason = parse_reply(raw)
            return dict(label=label, reason=reason, raw=raw)
        except ValueError:
            continue
    return dict(label="invalid", reason="unparseable reply", raw=raw)


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
                           label=res["label"], reason=res["reason"], raw=res["raw"])
                cache[h] = row
                cf.write(json.dumps(row, ensure_ascii=False) + "\n")
                cf.flush()
    rows = []
    for r in samples:
        c = cache[sample_hash(r["prefill"], r["continuation"], version)]
        rows.append({**r["carry"], "sample_hash": c["sample_hash"], "label": c["label"], "reason": c["reason"],
                     "judge_model": c["judge_model"], "prompt_version": c["prompt_version"], "temperature": c["temperature"],
                     "thinking": c.get("thinking")})
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
