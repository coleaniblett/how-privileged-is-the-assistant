"""Build the overgeneralization item set (README: pre-registered criteria, "Unit").

    python -m measures.overgeneralization.build_items passages   # -> data/overgeneralization_passages.jsonl
    python -m measures.overgeneralization.build_items merge      # -> data/overgeneralization_items.jsonl

`passages`: for each genre, stream a human-written source from before LLM-written text was common, shuffled with
a fixed seed. P is the document's opening (whole sentences, at least --min-passage OLMo tokens) and C is the
real next sentence in the same paragraph (for speech, the same speaker's turn), so the shared tail is
human-written. A candidate is kept if P and C pass the filters (English, no URLs or e-mail addresses, no special
tokens, no mention of AI or language models, render alignment in every context, with an insertion too). Then
the Dolci filter drops any passage that shares a sentence (>= 8 words, whitespace-normalized) with a turn of
the OLMo 3 Instruct SFT/DPO/RL data (core.check_contamination matches whole turns, which an excerpt almost never
would). The first --per-cell x 4 clean passages of each genre get traits by seeded random assignment, rotating
the phrasings within each trait; the next --reserve are kept unassigned as replacements.

`merge`: joins the passages with the hand-written insertions (insertions.jsonl here, one row per passage id:
{"id", "phrasing", "trait_text", "human_text", "neutral_text"} or {"id", "rejected": reason}). A reserve
passage used as a replacement needs an insertion row with its "trait" (and "phrasing"). Every cell takes its
assigned passages first and then its reserve rows, in file order, up to --per-cell. The output is validated on
the OLMo tokenizer: every variant renders in every context with passage, insertion and tail tokens intact, and
T and H differ by at most --max-len-diff tokens. Matching H on base log-probabilities happens later, at scoring
time (minimal_pairs analyze).
"""

import argparse
import ast
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict

from core import config
from core.render import AlignmentError, check_text, encode, load_jsonl, load_tokenizer, render

HERE = os.path.dirname(__file__)
PASSAGES_PATH = "data/overgeneralization_passages.jsonl"
INSERTIONS_PATH = os.path.join(HERE, "insertions.jsonl")
ITEMS_OUT = "data/overgeneralization_items.jsonl"
TRAITS = ["disclaimer", "not_x_but_y", "retraction", "llm_self_reference"]
# [3] surface phrasings per trait (README, bracketed: unconfirmed). Rotated so each carries per_cell / 3 items.
PHRASINGS = {
    "disclaimer": ["im_not_a_x", "not_advice", "consult_a_professional"],  # "I'm not a doctor, but", "this isn't
    "not_x_but_y": ["its_not_x_its_y", "not_just_x", "not_x_but_y"],       # legal advice", "you should consult"
    "retraction": ["actually_correct", "apologies_error", "to_clarify"],
    "llm_self_reference": ["as_an_ai", "im_a_language_model", "no_personal_experience"],
}
SOURCES = {
    "fiction": ("euclaise/writingprompts", {}),
    "forum": ("webis/tldr-17", {"revision": "refs/convert/parquet"}),
    "news": ("vblagoje/cc_news", {}),
    "email": ("corbt/enron-emails", {}),
    "speech": ("nbroad/mediasum", {"revision": "refs/convert/parquet"}),
}

# ---------------------------------------------------------------------------------------------- sentences
ABBREV = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "prof", "gen", "gov", "sen", "rep", "rev", "lt", "col", "capt",
          "sgt", "vs", "etc", "inc", "co", "corp", "ltd", "no", "u.s", "u.k", "e.g", "i.e", "a.m", "p.m", "jan", "feb",
          "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "mt", "ft", "approx", "dept", "est"}
BOUNDARY = re.compile(r"""[.!?]+["'”’)\]]*(?=\s+["'“‘(\[]?[A-Z0-9])""")


def sentences(par):
    """Sentences of one paragraph, as (start, end) character spans; text between them is whitespace."""
    spans, start = [], 0
    for m in BOUNDARY.finditer(par):
        word = par[start : m.start() + 1].split()[-1].rstrip(".").lower() if par[start : m.start() + 1].split() else ""
        if word in ABBREV or re.fullmatch(r"[a-z]", word):  # "Mr.", "J." are not ends
            continue
        spans.append((start, m.end()))
        start = m.end() + len(par[m.end() :]) - len(par[m.end() :].lstrip())
    if par[start:].strip():
        spans.append((start, len(par.rstrip())))
    return spans


def norm(s):
    return " ".join(s.split())


def long_sentences(text, min_words=8):
    """Whitespace-normalized sentences of at least min_words words (the Dolci filter's unit)."""
    out = set()
    for par in text.split("\n"):
        for a, b in sentences(par):
            s = norm(par[a:b])
            if len(s.split()) >= min_words:
                out.add(s)
    return out


# ---------------------------------------------------------------------------------------------- sources
def sid(*parts):
    return hashlib.sha1("\x00".join(map(str, parts)).encode()).hexdigest()[:12]


def fiction(ex):
    s = ex["story"].replace("<newline>", "\n")
    if re.search(r"``| n't| 's |\s[\u2019\u201d]|[\u201c\u2018]\s", s):  # leftover PTB tokenization
        return None
    return sid(s), s


def forum(ex):
    return ex["id"], ex["body"]


def news(ex):
    return ex["url"], f"{ex['title'].strip()}\n\n{ex['text']}"


EMAIL_CUT = re.compile(r"^\s*(-{2,}.*(Original Message|Forwarded)|From:|To:|Sent:|cc:|>|_{5,}|\*{5,})", re.I | re.M)


def email(ex):
    body = ex["body"]
    m = EMAIL_CUT.search(body)
    body = body[: m.start()] if m else body
    if not body.strip():
        return None
    body = re.sub(r"(?<=\S)[ \t]*\n(?=[ \t]*\S)", " ", body)  # unwrap hard-wrapped lines within a paragraph
    subj = (ex.get("subject") or "").strip()
    return ex["message_id"], (f"Subject: {subj}\n\n" if subj else "") + body


def as_list(x):
    return ast.literal_eval(x) if isinstance(x, str) else list(x)


def speech(ex):
    utt, spk = as_list(ex["utt"]), as_list(ex["speaker"])
    if len(utt) != len(spk):
        return None
    return ex["id"], "\n\n".join(f"{s.strip()}: {u.strip()}" for s, u in zip(spk, utt))


EXTRACT = {"fiction": fiction, "forum": forum, "news": news, "email": email, "speech": speech}

# ---------------------------------------------------------------------------------------------- filters
URLISH = re.compile(r"https?:|www\.|\.(com|org|net|gov|edu)\b|@|\b\d{3}[-.)\s]\s?\d{3}[-.\s]\d{4}\b", re.I)
AI_WORDS = re.compile(r"\bA\.?I\.?\b|artificial intelligence|language model|chat ?bot|\bGPT|\bLLMs?\b|neural net|"
                      r"machine learning|\bbots?\b|\bSiri\b|\bAlexa\b|virtual assistant|tl;?dr", re.I)
STOP = {"the", "and", "a", "to", "of", "in", "i", "is", "that", "it", "was", "for", "you", "on", "with", "my", "he"}
PROBE = " This is a probe sentence."  # an insertion of the usual shape, for the boundary check


def english(s):
    words = re.findall(r"[a-zA-Z']+", s.lower())
    return sum(ch.isascii() for ch in s) / max(len(s), 1) > 0.99 and len(words) >= 10 and \
        sum(w in STOP for w in words) / len(words) > 0.12


def aligned(tok, passage, insertion, tail):
    """True if passage + insertion + tail renders in every context with the three parts' tokens intact."""
    parts = [encode(tok, x) for x in (passage, insertion, tail) if x]
    try:
        for slot in ("user", "assistant", None):
            r = render(tok, passage + insertion + tail, slot)
            if r.input_ids[r.start : r.end] != sum(parts, []):
                return False
    except AlignmentError:
        return False
    return True


def cut(tok, text, min_tok, max_tok, min_tail_words=4, max_tail_words=40):
    """(P, C) or (None, reason). P = whole leading sentences with >= min_tok tokens, C = the next sentence of the
    same paragraph, prefixed with a single space."""
    text = text.replace("\r", "").strip()
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    offset = 0
    for par in text.split("\n"):
        spans = sentences(par)
        for i, (a, b) in enumerate(spans[:-1]):
            p = text[: offset + b]
            n = len(encode(tok, p))
            if n > max_tok:
                return None, "no boundary in length range"
            if n < min_tok:
                continue
            c0, c1 = spans[i + 1]
            c = par[c0:c1]
            if not min_tail_words <= len(c.split()) <= max_tail_words or not re.search(r"[.!?][\"'”’)]*$", c):
                continue
            return (p, " " + c), None
        offset += len(par) + 1
    return None, "no boundary in length range"


def candidate(tok, text, args):
    pc, why = cut(tok, text, args.min_passage, args.max_passage)
    if pc is None:
        return None, why
    p, c = pc
    s = p + c
    if URLISH.search(s):
        return None, "url/address"
    if AI_WORDS.search(s):
        return None, "mentions AI"
    if not english(s):
        return None, "not English"
    try:
        check_text(tok, s)
    except AlignmentError:
        return None, "special token/whitespace"
    if not (aligned(tok, p, "", c) and aligned(tok, p, PROBE, c)):
        return None, "render alignment"
    return (p, c), None


# ---------------------------------------------------------------------------------------------- Dolci
def dolci_sentence_hits(sents):
    """Subset of sents (normalized sentences) that occur as a sentence of any Dolci SFT/DPO/RL turn."""
    from datasets import load_dataset

    from core.check_contamination import DOLCI, rl_turns

    hits = set()

    def scan(text):
        if text:
            hits.update(long_sentences(text) & sents)

    for name, ds_id in DOLCI.items():
        ds = load_dataset(ds_id, split="train")
        n = 0
        if name == "rl":
            for batch in ds.select_columns(["prompt"]).iter(batch_size=10_000):
                for prompt in batch["prompt"]:
                    for _, t in rl_turns(prompt or ""):
                        scan(t)
                n += len(batch["prompt"])
        else:
            cols = ["messages"] if name == "sft" else ["chosen", "rejected"]
            for batch in ds.select_columns(cols).iter(batch_size=10_000):
                for col in cols:
                    for msgs in batch[col]:
                        for m in msgs or []:
                            scan(m["content"])
                n += len(batch[cols[0]])
        print(f"{ds_id}: scanned {n} rows, {len(hits)} candidate sentences hit so far", flush=True)
    return hits


# ---------------------------------------------------------------------------------------------- passages
def collect(genre, tok, n, args):
    from datasets import load_dataset

    ds_id, kw = SOURCES[genre]
    ds = load_dataset(ds_id, split="train", streaming=True, **kw).shuffle(seed=args.seed, buffer_size=args.buffer)
    out, seen, stats = [], set(), Counter()
    for ex in ds:
        stats["streamed"] += 1
        got = EXTRACT[genre](ex)
        if got is None:
            stats["extract"] += 1
            continue
        src, text = got
        if src in seen:
            stats["duplicate"] += 1
            continue
        pc, why = candidate(tok, text, args)
        if pc is None:
            stats[why] += 1
            continue
        if pc[0] in {o["passage"] for o in out}:
            stats["duplicate"] += 1
            continue
        seen.add(src)
        out.append(dict(genre=genre, passage=pc[0], tail=pc[1], dataset=ds_id, source_id=src))
        if len(out) == n:
            break
    stats["kept"] = len(out)
    print(f"{genre} ({ds_id}): {dict(stats)}", flush=True)
    if len(out) < n:
        raise SystemExit(f"{genre}: only {len(out)} candidates")
    return out


def passages(args):
    tok = load_tokenizer(args.tokenizer)
    need = args.per_cell * len(TRAITS) + args.reserve
    cands = {g: collect(g, tok, int(need * args.oversample), args) for g in args.genres}
    if args.dolci:
        sents = {g: [long_sentences(c["passage"] + c["tail"]) for c in cs] for g, cs in cands.items()}
        hits = dolci_sentence_hits(set().union(*(s for ss in sents.values() for s in ss)))
        for g in cands:
            keep = [c for c, s in zip(cands[g], sents[g]) if not s & hits]
            print(f"{g}: {len(cands[g]) - len(keep)} of {len(cands[g])} candidates share a sentence with Dolci")
            cands[g] = keep
    else:
        print("WARNING: --no-dolci, passages are not filtered for Dolci overlap")

    rows = []
    for g, cs in cands.items():
        if len(cs) < need:
            raise SystemExit(f"{g}: {len(cs)} clean candidates, need {need} (raise --oversample)")
        cs = cs[:need]
        random.Random(f"{args.seed}-{g}").shuffle(cs)
        slots = [(t, PHRASINGS[t][j % len(PHRASINGS[t])]) for t in TRAITS for j in range(args.per_cell)]
        random.Random(f"{args.seed}-{g}-traits").shuffle(slots)
        for i, c in enumerate(cs):
            t, ph = slots[i] if i < len(slots) else (None, None)
            rows.append(dict(id=f"{g}-{i:03d}", status="assigned" if t else "reserve", trait=t, phrasing=ph,
                             genre=g, passage=c["passage"], tail=c["tail"],
                             n_passage_tokens=len(encode(tok, c["passage"])), n_tail_tokens=len(encode(tok, c["tail"])),
                             dataset=c["dataset"], source_id=c["source_id"], dolci_checked=args.dolci))
    write_jsonl(args.out, rows)
    print(Counter((r["genre"], r["status"]) for r in rows))


# ---------------------------------------------------------------------------------------------- merge
INS_FIELDS = ["trait_text", "human_text", "neutral_text"]


def insertion_problems(tok, p, ins, max_len_diff):
    out = []
    for f in INS_FIELDS:
        x = ins.get(f, "")
        if not x.startswith(" ") or x.startswith("  ") or "\n" in x or not re.search(r"[.!?][\"'”’)]*$", x):
            out.append(f"{f} must start with one space, be one line and end a sentence: {x!r}")
        elif not aligned(tok, p["passage"], x, p["tail"]):
            out.append(f"{f}: render alignment fails")
    if not out:
        n = {f: len(encode(tok, ins[f])) for f in INS_FIELDS}
        if abs(n["trait_text"] - n["human_text"]) > max_len_diff:
            out.append(f"|T - H| = {abs(n['trait_text'] - n['human_text'])} tokens > {max_len_diff}")
    return out


def merge(args):
    tok = load_tokenizer(args.tokenizer)
    ps = {p["id"]: p for p in load_jsonl(args.passages)}
    ins = load_jsonl(args.insertions)
    by_id = {}
    for r in ins:
        if r["id"] not in ps:
            raise SystemExit(f"insertion for unknown passage {r['id']}")
        if r["id"] in by_id:
            raise SystemExit(f"two insertion rows for {r['id']}")
        by_id[r["id"]] = r

    cells, problems, rejected = defaultdict(list), [], Counter()
    order = [p for p in ps.values() if p["status"] == "assigned"] + \
            [ps[r["id"]] for r in ins if ps[r["id"]]["status"] == "reserve"]
    for p in order:
        r = by_id.get(p["id"])
        if r is None:
            continue
        trait = p["trait"] or r.get("trait")
        if trait not in TRAITS:
            problems.append(f"{p['id']}: reserve passage needs a valid 'trait' in its insertion row")
            continue
        if "rejected" in r:
            rejected[(p["genre"], trait)] += 1
            continue
        phrasing = r.get("phrasing") or p["phrasing"]
        if phrasing not in PHRASINGS[trait]:
            problems.append(f"{p['id']}: unknown phrasing {phrasing!r} for {trait}")
            continue
        bad = insertion_problems(tok, p, r, args.max_len_diff)
        if bad:
            problems += [f"{p['id']}: {b}" for b in bad]
            continue
        cell = cells[(p["genre"], trait)]
        if len(cell) < args.per_cell:
            cell.append(dict(id=p["id"], trait=trait, phrasing=phrasing, genre=p["genre"], passage=p["passage"],
                             **{f: r[f] for f in INS_FIELDS}, tail=p["tail"], dataset=p["dataset"],
                             source_id=p["source_id"], replaced=p["status"] == "reserve"))

    genres = sorted({p["genre"] for p in ps.values()})
    print("items per cell (rejected in brackets):")
    short = []
    for g in genres:
        line = []
        for t in TRAITS:
            n = len(cells[(g, t)])
            line.append(f"{t} {n} ({rejected[(g, t)]})")
            if n < args.per_cell:
                short.append(f"{g}/{t}: {n}")
        print(f"  {g}: " + ", ".join(line))
    for b in problems:
        print("PROBLEM", b)
    if (short or problems) and not args.allow_incomplete:
        raise SystemExit(f"not written: {len(problems)} problems; short cells: {', '.join(short) or 'none'}")
    items = [it for g in genres for t in TRAITS for it in cells[(g, t)]]
    write_jsonl(args.out, items)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("passages", "merge"):
        p = sub.add_parser(name)
        p.add_argument("--tokenizer", default=config.TEMPLATE_MODEL)
        p.add_argument("--per-cell", type=int, default=12)
    p = sub.choices["passages"]
    p.add_argument("--genres", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    p.add_argument("--reserve", type=int, default=16)
    p.add_argument("--oversample", type=float, default=1.5, help="candidates collected per needed passage")
    p.add_argument("--min-passage", type=int, default=40)
    p.add_argument("--max-passage", type=int, default=160)
    p.add_argument("--buffer", type=int, default=10_000)
    p.add_argument("--dolci", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--out", default=PASSAGES_PATH)
    p = sub.choices["merge"]
    p.add_argument("--passages", default=PASSAGES_PATH)
    p.add_argument("--insertions", default=INSERTIONS_PATH)
    p.add_argument("--max-len-diff", type=int, default=2)
    p.add_argument("--allow-incomplete", action="store_true", help="write even with short cells (for drafts)")
    p.add_argument("--out", default=ITEMS_OUT)
    args = ap.parse_args()
    {"passages": passages, "merge": merge}[args.cmd](args)


if __name__ == "__main__":
    main()
