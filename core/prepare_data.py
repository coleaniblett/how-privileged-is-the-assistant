"""Sample held-out user + assistant texts and a user-turn prefill set.

    python -m core.prepare_data                             # WildChat pairs + oasst1 user prompts
    python -m core.prepare_data --pairs oasst1 --extra-user none --no-exclude-dolci   # smoke test

Pair source (default WildChat-1M): first user turn + first assistant reply of each conversation.
Extra user source (default oasst1): root prompts only. Conversations whose user turn appears in
the Dolci-Instruct SFT, DPO or RL data are skipped before the other filters (see
check_contamination.py), and sampling continues until --n clean texts per source survive.

Writes data/texts.jsonl, one text per row (id, source, kind, text, n_tokens, pair_id,
source_id, dataset), and data/prefill.jsonl (id, text_id, prefill) from the pair source's user
turns. OLMo replies are added afterwards by add_replies.py.

    python -m core.prepare_data --prefill-only --n-prefill 100 --prefill-out data/prefill_100.jsonl
    python -m core.prepare_data --authorship                # -> data/texts_authorship.jsonl (logit distortion's authorship cross)
    python -m core.prepare_data --incomplete-prefill        # -> data/prefill_incomplete_200.jsonl (user turns cut mid-thought)

builds a larger prefill set from the texts already in data/texts.jsonl without touching it: the same first-sentence
rule applied to the same (already length-, render- and Dolci-filtered) WildChat user turns in the same order, so the
original 20 are its first 20 rows. Each row keeps its source text_id.
"""

import argparse
import json
import os
import random
import re
from collections import Counter

from datasets import load_dataset

from core import config
from core.render import AlignmentError, load_tokenizer, render


def wildchat_pairs(seed):
    ds = load_dataset(config.DATASET_IDS["wildchat"], split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=20_000)
    for ex in ds:
        if ex["language"] != "English" or ex["toxic"] or ex.get("redacted"):
            continue
        conv = ex["conversation"]
        if len(conv) < 2 or conv[0]["role"] != "user" or conv[1]["role"] != "assistant":
            continue
        if any(t.get("toxic") for t in conv[:2]):
            continue
        yield ex["conversation_hash"], conv[0]["content"], conv[1]["content"]


def oasst_toxicity(msg):
    labels = msg.get("labels") or {}
    vals = dict(zip(labels.get("name") or [], labels.get("value") or []))
    if "toxicity" in vals:
        return vals["toxicity"]
    return (msg.get("detoxify") or {}).get("toxicity") or 0.0


def oasst_messages(seed):
    ds = load_dataset(config.DATASET_IDS["oasst1"], split="train")
    ok = [m for m in ds if not m["deleted"] and m["review_result"] and m["lang"] == "en"]
    roots = [m for m in ok if m["role"] == "prompter" and m["parent_id"] is None]
    random.Random(seed).shuffle(roots)
    return ok, roots


def oasst_pairs(seed):
    ok, roots = oasst_messages(seed)
    replies = {}
    for m in ok:
        if m["role"] == "assistant" and m["parent_id"] is not None:
            replies.setdefault(m["parent_id"], []).append(m)
    for root in roots:
        kids = replies.get(root["message_id"])
        if not kids:
            continue
        # "first" reply: best-ranked, else earliest
        reply = min(kids, key=lambda m: (m["rank"] is None, m["rank"] or 0, m["created_date"]))
        if oasst_toxicity(root) >= 0.5 or oasst_toxicity(reply) >= 0.5:
            continue
        yield root["message_id"], root["text"], reply["text"]


def oasst_prompts(seed):
    """Root prompts only (no reply required), same language/review/toxicity filters."""
    _, roots = oasst_messages(seed)
    for root in roots:
        if oasst_toxicity(root) < 0.5:
            yield root["message_id"], root["text"], None


def ultrachat_pairs(seed):
    """First user turn and first assistant reply of UltraChat conversations (both model-written: the user turns come from a
    user-simulating model, the replies from ChatGPT). `data` alternates user and assistant turns starting with the user."""
    ds = load_dataset(config.DATASET_IDS["ultrachat"], split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=20_000)
    for ex in ds:
        data = ex["data"]
        if len(data) >= 2 and data[0] and data[1]:
            yield f"ultrachat-{ex['id']}", data[0], data[1]


SOURCES = {"wildchat": wildchat_pairs, "oasst1": oasst_pairs, "ultrachat": ultrachat_pairs}
USER_ONLY = {"oasst1": oasst_prompts}


def clean(text):
    return text.strip()


def usable(tok, text):
    n = len(tok(text, add_special_tokens=False)["input_ids"])
    if not config.MIN_TOKENS <= n <= config.MAX_TOKENS:
        return None
    try:
        for slot in ("user", "assistant", None):
            render(tok, text, slot)
            render(tok, text, slot, preceding=True)
    except AlignmentError:
        return None
    return n


def sample(source, stream, tok, n, turns, kinds):
    """First n candidates from stream whose texts pass every filter. kinds: ("user",) or ("user", "assistant")."""
    from core.check_contamination import is_contaminated

    rows, seen, stats = [], set(), Counter()
    for src_id, user, asst in stream:
        stats["candidates"] += 1
        user = clean(user)
        if src_id in seen or user in seen:
            stats["duplicate"] += 1
            continue
        if turns is not None and is_contaminated(turns, user, "user", src_id):
            stats["in Dolci"] += 1
            continue
        texts = {"user": user, "assistant": clean(asst) if asst is not None else None}
        lengths = {k: usable(tok, texts[k]) for k in kinds}
        if any(v is None for v in lengths.values()):
            stats["length/render"] += 1
            continue
        seen.update([src_id, user])
        pair_id = f"{source}-{len(rows) // len(kinds):04d}"
        for k in kinds:
            rows.append(dict(id=f"{source}-{k}-{pair_id[-4:]}", source=source, kind=k, text=texts[k],
                             n_tokens=lengths[k], pair_id=pair_id, source_id=src_id,
                             dataset=config.DATASET_IDS[source]))
        if len(rows) == n * len(kinds):
            break
    stats["kept"] = len(rows) // len(kinds)
    print(f"{source} ({'+'.join(kinds)}): {dict(stats)}", flush=True)
    if len(rows) < n * len(kinds):
        raise SystemExit(f"only found {len(rows) // len(kinds)} usable {source} texts")
    return rows


FIRST_SENTENCE = re.compile(r"^(.+?[.!?])\s+\S", re.S)


def first_sentence(text):
    m = FIRST_SENTENCE.match(text)
    if not m:
        return None
    s = m.group(1)
    return s if "\n" not in s and 5 <= len(s.split()) <= 40 else None


# Words after which a user turn is clearly unfinished. The first cut type that lists a word wins; the set is deliberately closed-class
# plus a short list of verbs that nearly always take an object, because no part-of-speech tagger is a dependency of this project.
INCOMPLETE_ENDINGS = {
    "article/determiner": ("a an the this that these those my your his her its our their some any each every no another such all both "
                           "several many few most more other same").split(),
    "preposition": ("of in on at to for with from by about into onto over under between through during without within against among after "
                    "before like as than per via toward towards upon across along around behind beyond near off out up down").split(),
    "conjunction": ("and or but because if when while although though so since unless until whether which who whom whose where whereas "
                    "nor yet").split(),
    "auxiliary/modal": ("is are was were be been being am do does did can could will would shall should may might must have has had "
                        "not never also just really very too").split(),
    "transitive verb": ("want wants need needs make makes give gives tell tells show shows write create explain describe find use take get "
                        "let help provide generate list summarize translate add keep send buy love hate know think believe ask answer "
                        "say suggest recommend imagine pretend continue include avoid try start begin").split(),
}
WORD_TYPE = {}
for _t, _ws in INCOMPLETE_ENDINGS.items():
    for _w in _ws:
        WORD_TYPE.setdefault(_w, _t)
TERMINAL = re.compile(r"[.!?\u2026][\"'\u201d\u2019)\]]*$")


def incomplete_cut_points(text, min_words=6, max_words=40):
    """[(end offset in text, number of words, cut type)] of the places where the first line of `text` could be cut so that the turn is clearly
    unfinished: the last kept word is a closing-class word from INCOMPLETE_ENDINGS or ends with a comma, never terminal punctuation, and at
    least one more word follows on the same line."""
    line = text.split("\n", 1)[0]
    ms = list(re.finditer(r"\S+", line))
    out = []
    for i in range(min_words, min(max_words, len(ms) - 1) + 1):
        w = ms[i - 1].group()
        if TERMINAL.search(w):
            continue
        bare = re.sub(r"^[\W_]+|[\W_]+$", "", w).lower()
        if w.endswith(","):
            out.append((ms[i - 1].end(), i, "comma"))
        elif bare in WORD_TYPE and bare == w.lower().strip("\"'\u201c\u201d()[]"):
            out.append((ms[i - 1].end(), i, WORD_TYPE[bare]))
    return out


def build_incomplete_prefill(rows, source, n, seed=0):
    """n prefills from `source`'s user rows (in file order): one seeded cut per text, choosing the cut type first (uniformly among the types the
    text offers) and then a position of that type, so the set is not dominated by 'the' and 'of'. Rows keep text_id and record the cut."""
    out, seen = [], set()
    for row in rows:
        if row["source"] != source or row["kind"] != "user":
            continue
        cuts = incomplete_cut_points(row["text"])
        if not cuts:
            continue
        rng = random.Random(f"{seed}-{row['id']}")
        kind = rng.choice(sorted({c[2] for c in cuts}))
        first = rng.choice([c for c in cuts if c[2] == kind])
        rest_cuts = [c for c in cuts if c != first]
        rng.shuffle(rest_cuts)
        # different conversations can share an opening (WildChat repeats prompts): take the first cut whose prefill is new
        pick = next((c for c in [first, *rest_cuts] if row["text"][: c[0]].lower() not in seen), None)
        if pick is None:
            continue
        end, n_words, kind = pick
        prefill, rest = row["text"][:end], row["text"][end:]
        seen.add(prefill.lower())
        out.append(dict(id=f"incomplete-{len(out):03d}", text_id=row["id"], prefill=prefill, cut_type=kind,
                        last_word=prefill.split()[-1], n_words=n_words, remaining_chars=len(rest),
                        next_words=" ".join(rest.split()[:8])))
        if len(out) == n:
            break
    return out


def build_prefill(rows, source, n):
    """First n distinct first sentences of `source`'s user rows, in row order: dicts (id, text_id, prefill)."""
    prefill, seen_sentences = [], set()
    for row in rows:
        if row["source"] != source or row["kind"] != "user":
            continue
        s = first_sentence(row["text"])
        if s and s not in seen_sentences:  # distinct conversations can share an opening sentence
            seen_sentences.add(s)
            prefill.append(dict(id=f"prefill-{len(prefill):02d}", text_id=row["id"], prefill=s))
        if len(prefill) == n:
            break
    return prefill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", choices=list(SOURCES), default=config.PAIR_SOURCE)
    ap.add_argument("--extra-user", choices=[*USER_ONLY, "none"], default=config.EXTRA_USER_SOURCE)
    ap.add_argument("--exclude-dolci", action=argparse.BooleanOptionalAction, default=config.EXCLUDE_DOLCI)
    ap.add_argument("--tokenizer", default=config.TEMPLATE_MODEL)
    ap.add_argument("--n", type=int, default=config.N_TEXTS)
    ap.add_argument("--n-prefill", type=int, default=config.N_PREFILL)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", default=config.DATA_PATH)
    ap.add_argument("--prefill-out", default=config.PREFILL_PATH_20)
    ap.add_argument("--authorship", action="store_true",
                    help="write the authorship-cross text sets (oasst1 assistant replies; UltraChat user turns and replies) to "
                         "--out (default data/texts_authorship.jsonl); the other texts files are not touched")
    ap.add_argument("--incomplete-prefill", action="store_true",
                    help="read rows from --out (texts.jsonl, not modified) and write incomplete-turn prefills (cut mid-thought) to "
                         "--prefill-out (default data/prefill_incomplete_200.jsonl)")
    ap.add_argument("--n-incomplete", type=int, default=config.N_PREFILL_INCOMPLETE)
    ap.add_argument("--prefill-only", action="store_true",
                    help="read rows from --out (texts.jsonl, not modified) and write only --prefill-out")
    args = ap.parse_args()
    if args.incomplete_prefill:
        rows = [json.loads(line) for line in open(args.out, encoding="utf-8")]
        prefill = build_incomplete_prefill(rows, args.pairs, args.n_incomplete, args.seed)
        if len(prefill) < args.n_incomplete:
            raise SystemExit(f"only {len(prefill)} {args.pairs} user texts offer a clearly unfinished cut; need {args.n_incomplete}")
        out = args.prefill_out if args.prefill_out != config.PREFILL_PATH_20 else config.PREFILL_INCOMPLETE_PATH
        with open(out, "w", encoding="utf-8") as f:
            for r in prefill:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(prefill)} rows to {out}: {dict(Counter(r['cut_type'] for r in prefill))}")
        return
    if args.authorship:
        # Same filters as the other sets (20-300 tokens, clean rendering in every slot, Dolci contamination filter on the
        # conversation's user turn); n clean texts per set.
        out = args.out if args.out != config.DATA_PATH else config.AUTHORSHIP_PATH
        tok = load_tokenizer(args.tokenizer)
        turns = None
        if args.exclude_dolci:
            from core.check_contamination import dolci_turns

            turns = dolci_turns()
        rows = sample("oasst1", oasst_pairs(args.seed), tok, args.n, turns, ("assistant",))
        rows += sample("ultrachat", ultrachat_pairs(args.seed), tok, args.n, turns, ("user", "assistant"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} rows to {out}: {dict(Counter((r['source'], r['kind']) for r in rows))}")
        return
    if args.prefill_only:
        rows = [json.loads(line) for line in open(args.out, encoding="utf-8")]
        prefill = build_prefill(rows, args.pairs, args.n_prefill)
        if len(prefill) < args.n_prefill:
            raise SystemExit(f"only {len(prefill)} distinct first sentences in {args.out}; need {args.n_prefill}")
        with open(args.prefill_out, "w", encoding="utf-8") as f:
            for r in prefill:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(prefill)} rows to {args.prefill_out}")
        return
    if args.extra_user == args.pairs:
        ap.error("--extra-user must differ from --pairs")

    tok = load_tokenizer(args.tokenizer)
    turns = None
    if args.exclude_dolci:
        from core.check_contamination import dolci_turns

        turns = dolci_turns()
    rows = sample(args.pairs, SOURCES[args.pairs](args.seed), tok, args.n, turns, ("user", "assistant"))
    if args.extra_user != "none":
        rows += sample(args.extra_user, USER_ONLY[args.extra_user](args.seed), tok, args.n, turns, ("user",))

    prefill = build_prefill(rows, args.pairs, args.n_prefill)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for path, items in ((args.out, rows), (args.prefill_out, prefill)):
        with open(path, "w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(items)} rows to {path}")


if __name__ == "__main__":
    main()
