"""Real two-turn exchanges (a user turn plus its own reply) and their renderings, shared by vestigial regression and
logit distortion.

    python -m core.exchanges build                  # -> data/exchanges.jsonl + data/exchanges_manifest.json (no weights:
                                                    #    oasst1, UltraChat and WildChat, the Dolci sets, three tokenizers)
    python -m core.exchanges render                 # -> docs/rendered_exchanges.md (examples, every family and context)

Per exchange and checkpoint there are four scored conditions: scored text (user | reply) x framing (tagged | plain).

    user,  tagged   the user turn opens the conversation (after the template's default system turn, if any): the same
                    sequence as core.score's user@user
    user,  plain    document-start token + the bare user text (core.score's user@plain)
    reply, tagged   context "exchange": its own user turn, tagged and closed, then the assistant opening exactly as the
                    family's generation prompt writes it (Gemma 4: empty thought channel; Qwen3: empty think block),
                    then the reply. Context "single": the reply alone in the assistant slot (core.score's asst@asst)
    reply, plain    the same sequence with every role tag and system turn removed: document-start token + user text +
                    config.EXCHANGE_SEPARATOR + reply ("exchange"), or document-start token + reply ("single")

The document-start token (<|endoftext|> for OLMo and Qwen3, <bos> for Gemma 4) is not a role tag; it is what core's
plain control has always started with, so the first content token has something to be predicted from.

Only the scored text's own tokens are scored, the first config.EXCHANGE_REPLY_MAX_TOKENS of a reply (longer replies
are kept and marked truncated). Every rendering is checked: the template places each text verbatim, nothing merges
across a text boundary, the scored span decodes to the text, and the scored tokens are identical between tagged and
plain (and equal to the text tokenized alone). The build drops any exchange that fails in any family, so all families
score the same exchanges.

Built on Grace's original-conversation builder (core.conversations, core.render.render_target,
core.score_conversations.PROFILES); see docs/exchanges.md for what is reused.
"""

import argparse
import contextlib
import hashlib
import json
import os
import platform
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version

from core import config
from core.render import (AlignmentError, check_text, doc_start_ids, encode, load_tokenizer, render, render_target)

SCORED_TEXTS = ("user", "reply")
FRAMINGS = ("tagged", "plain")
CORPUS_LABELS = {"oasst1": "oasst1", "ultrachat": "UltraChat", "wildchat": "WildChat"}
FORMAT = "exchanges_v1"


def condition_label(family, stage, corpus, scored_text, framing):
    """"Gemma IT · UltraChat · reply · plain" (checkpoint · corpus · scored text · framing)."""
    st = config.family_settings(family) if family in config.FAMILIES else {}
    ckpt = f"{st.get('label') or family} {st.get('stage_labels', {}).get(stage, stage)}" if family else stage
    return " · ".join([ckpt, CORPUS_LABELS.get(corpus, corpus), scored_text, framing])


# --- rendering ---------------------------------------------------------------------------------------------------------

@dataclass
class Scored:
    """One scored condition: input_ids end with the last scored token (nothing after the span can change its NLL)."""
    input_ids: list
    start: int  # scored tokens: input_ids[start:end]
    end: int
    n_full: int  # tokens of the whole scored text (end - start < n_full only for a truncated reply)
    truncated: bool
    text: str  # the rendered string of input_ids (for docs and inspection)


def family_profile(family):
    """Template kwargs and expected role headers per family (Grace's declared PROFILES, reused); {} for a family-less
    tokenizer (the CPU smoke test)."""
    if not family:
        return {"template_kwargs": {}}
    from core.score_conversations import PROFILES
    return PROFILES[family]


@contextlib.contextmanager
def think_block(family):
    """core.render.render reads config.ASSISTANT_THINK_BLOCK; set it to the family's setting for the duration."""
    saved = config.ASSISTANT_THINK_BLOCK
    config.ASSISTANT_THINK_BLOCK = config.family_settings(family)["assistant_think_block"] if family else None
    try:
        yield
    finally:
        config.ASSISTANT_THINK_BLOCK = saved


def _check_lead(family, role, lead, what):
    expected = family_profile(family).get(f"{role}_header")
    if expected and not lead.endswith(expected):
        raise AlignmentError(f"{what}: lead-in ...{lead[-60:]!r} does not end with the declared {role} header {expected!r}")


def _user_tagged(tok, ex, family):
    r = render_target(tok, [{"role": "user", "content": ex["user"]}], family_profile(family)["template_kwargs"])
    _check_lead(family, "user", r.lead, "user tagged")
    # The overnight user@user sequence (core.score -> core.render.render): identical ids, so its scores can be reused.
    legacy = render(tok, ex["user"], "user")
    if (legacy.input_ids, legacy.start, legacy.end) != (r.input_ids, r.start, r.end):
        raise AlignmentError("user tagged differs from core.render.render(user slot)")
    return r.input_ids, r.start, r.end


def _plain(tok, before, text):
    """Document-start token + `before` (natural tokenization) + text tokenized alone; no merge at the text boundary."""
    pre = doc_start_ids(tok) + (encode(tok, before) if before else [])
    content = encode(tok, text)
    if before and encode(tok, before + text) != encode(tok, before) + content:
        raise AlignmentError("tokens merge across the separator/reply boundary in the plain framing")
    return pre + content, len(pre), len(pre) + len(content)


def _user_plain(tok, ex, family):
    check_text(tok, ex["user"])
    return _plain(tok, "", ex["user"])


def _reply_tagged_exchange(tok, ex, family):
    msgs = [{"role": "user", "content": ex["user"]}, {"role": "assistant", "content": ex["reply"]}]
    r = render_target(tok, msgs, family_profile(family)["template_kwargs"])
    _check_lead(family, "assistant", r.lead, "reply tagged")
    return r.input_ids, r.start, r.end


def _reply_plain_exchange(tok, ex, family):
    check_text(tok, ex["user"]), check_text(tok, ex["reply"])
    return _plain(tok, ex["user"] + config.EXCHANGE_SEPARATOR, ex["reply"])


def _reply_tagged_single(tok, ex, family):
    with think_block(family):
        r = render(tok, ex["reply"], "assistant")
    return r.input_ids, r.start, r.end


def _reply_plain_single(tok, ex, family):
    check_text(tok, ex["reply"])
    return _plain(tok, "", ex["reply"])


# The reply's context is a registry entry: another design is one more entry here (and a name in docs), nothing else.
CONTEXTS = {
    "exchange": {"tagged": _reply_tagged_exchange, "plain": _reply_plain_exchange},
    "single": {"tagged": _reply_tagged_single, "plain": _reply_plain_single},
}
USER_RENDER = {"tagged": _user_tagged, "plain": _user_plain}


def render_exchange(tok, ex, context=None, family=None, max_reply_tokens=None):
    """{(scored_text, framing): Scored} for one exchange. Raises AlignmentError if any rendering fails a check, including
    the tagged/plain token-identity check (the scored tokens must equal the text tokenized alone, in both framings)."""
    context = context or config.EXCHANGE_CONTEXT
    if context not in CONTEXTS:
        raise ValueError(f"unknown context {context!r}; known: {', '.join(CONTEXTS)}")
    cap = max_reply_tokens or config.EXCHANGE_REPLY_MAX_TOKENS
    out = {}
    for scored, fns in (("user", USER_RENDER), ("reply", CONTEXTS[context])):
        alone = encode(tok, ex[scored])
        if tok.decode(alone, clean_up_tokenization_spaces=False) != ex[scored]:
            raise AlignmentError(f"{scored}: decoding its tokens does not reproduce the text")
        for framing in FRAMINGS:
            ids, start, end = fns[framing](tok, ex, family)
            if ids[start:end] != alone:
                raise AlignmentError(f"{scored} {framing}: scored tokens differ from the text tokenized alone")
            n = end - start
            keep = min(n, cap) if scored == "reply" else n
            ids = ids[: start + keep]
            out[(scored, framing)] = Scored(ids, start, start + keep, n, keep < n, tok.decode(ids))
    for scored in SCORED_TEXTS:  # tagged/plain identity of what is actually scored
        t, p = out[(scored, "tagged")], out[(scored, "plain")]
        if t.input_ids[t.start:t.end] != p.input_ids[p.start:p.end]:
            raise AlignmentError(f"{scored}: scored tokens differ between tagged and plain")
    return out


# --- the exchange set --------------------------------------------------------------------------------------------------

def legacy_index():
    """{(source, kind, text): (file, id)} over the overnight-scored text sets, so an exchange can point at the row whose
    scores may be reused (core.score_exchanges verifies the tokens before reusing anything)."""
    from core.render import load_jsonl
    idx = {}
    for path in (config.DATA_PATH, config.AUTHORSHIP_PATH):
        if os.path.exists(path):
            for r in load_jsonl(path):
                idx.setdefault((r["source"], r["kind"], r["text"]), (path, r["id"]))
    return idx


def oasst_candidates(seed):
    """English, reviewed, undeleted oasst1 (train) root prompts in core.prepare_data's seeded order, each with its
    highest-ranked human assistant reply (rank, then creation date; prepare_data.oasst_pairs' rule); toxicity < 0.5 on both."""
    from core.prepare_data import oasst_messages, oasst_toxicity
    ok, roots = oasst_messages(seed)
    replies = {}
    for m in ok:
        if m["role"] == "assistant" and m["parent_id"] is not None:
            replies.setdefault(m["parent_id"], []).append(m)
    for root in roots:
        kids = replies.get(root["message_id"])
        if not kids:
            yield {"skip": "no_reply"}
            continue
        reply = min(kids, key=lambda m: (m["rank"] is None, m["rank"] or 0, m["created_date"]))
        if oasst_toxicity(root) >= 0.5 or oasst_toxicity(reply) >= 0.5:
            yield {"skip": "toxicity"}
            continue
        if root.get("synthetic") or reply.get("synthetic"):
            yield {"skip": "synthetic"}
            continue
        yield {"source_id": root["message_id"], "user": root["text"], "reply": reply["text"],
               "meta": {"reply_message_id": reply["message_id"], "reply_rank": reply["rank"]}}


def ultrachat_candidates(seed):
    """First user turn and first reply of UltraChat dialogues (core.prepare_data.ultrachat_pairs: seeded stream)."""
    from core.prepare_data import ultrachat_pairs
    for sid, user, reply in ultrachat_pairs(seed):
        yield {"source_id": sid, "user": user, "reply": reply, "meta": {}}


def wildchat_candidates(seed):
    """The existing held-out WildChat pairs of data/texts.jsonl first, in pair_id order (their user turns were scored
    overnight), then, only if more are needed, new pairs from core.prepare_data.wildchat_pairs' seeded stream."""
    from core.prepare_data import wildchat_pairs
    from core.render import load_jsonl
    rows = [r for r in load_jsonl(config.DATA_PATH) if r["source"] == "wildchat"]
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], {})[r["kind"]] = r
    used = set()
    for pid in sorted(by_pair):
        p = by_pair[pid]
        if "user" in p and "assistant" in p:
            used.add(p["user"]["source_id"])
            yield {"source_id": p["user"]["source_id"], "user": p["user"]["text"], "reply": p["assistant"]["text"],
                   "meta": {"pair_id": pid, "origin": "data/texts.jsonl"}}
    for sid, user, reply in wildchat_pairs(seed):
        if sid not in used:
            yield {"source_id": sid, "user": user, "reply": reply, "meta": {"origin": "wildchat stream top-up"}}


CANDIDATES = {"oasst1": oasst_candidates, "ultrachat": ultrachat_candidates, "wildchat": wildchat_candidates}


def screen(cand, corpus, toks, turns, seen, olmo_tok):
    """None if the candidate is accepted, else the rejection reason. Mutates cand (cleaned texts, lengths, records)."""
    from core.check_contamination import contamination_record
    from core.prepare_data import clean, usable
    if "skip" in cand:
        return cand["skip"]
    if not isinstance(cand["user"], str) or not isinstance(cand["reply"], str):
        return "non_text"
    user, reply = clean(cand["user"]), clean(cand["reply"])
    if cand["source_id"] in seen or user in seen:
        return "duplicate"
    conv_hash = cand["source_id"] if corpus == "wildchat" else None
    rec = {k: contamination_record(turns, t, conv_hash) for k, t in (("user", user), ("reply", reply))}
    if turns is not None:
        if rec["user"]["matches"]:
            return "dolci_user"
        if rec["reply"]["matches"]:
            return "dolci_reply"
    n_user = usable(olmo_tok, user)  # 20-300 OLMo tokens, clean rendering in every core.score slot (the existing filter)
    if n_user is None:
        return "user_length_or_render"
    n_reply = len(encode(olmo_tok, reply))
    if n_reply < config.EXCHANGE_REPLY_MIN_TOKENS:
        return "reply_too_short"
    ex = {"user": user, "reply": reply}
    for fam, tok in toks.items():
        for context in CONTEXTS:
            try:
                render_exchange(tok, ex, context, fam)
            except AlignmentError:
                return f"render_{fam}_{context}"
    cand.update(user=user, reply=reply, n_user=n_user, n_reply=n_reply, contamination=rec)
    return None


def authorship(corpus):
    """(user, reply) authorship labels and their basis, from Grace's adapters (core.conversations.message)."""
    from core.conversations import message
    u, a = message("u", "user", "", corpus), message("a", "assistant", "", corpus)
    return {"user": u["authorship"], "reply": a["authorship"],
            "basis": {"user": u["authorship_basis"], "reply": a["authorship_basis"]}}


def build(n, seed, toks, turns, corpora=None, candidates=None, log=print):
    """[(exchange rows), stats per corpus]. `candidates` overrides CANDIDATES (tests)."""
    olmo_tok = toks["olmo3-7b"] if "olmo3-7b" in toks else next(iter(toks.values()))
    legacy = legacy_index()
    rows, stats = [], {}
    for corpus in corpora or config.EXCHANGE_CORPORA:
        st, seen, kept = Counter(), set(), 0
        for cand in (candidates or CANDIDATES)[corpus](seed):
            st["candidates"] += 1
            reason = screen(cand, corpus, toks, turns, seen, olmo_tok)
            if reason:
                st[reason] += 1
                continue
            seen.update([cand["source_id"], cand["user"]])
            leg = {k: legacy.get((corpus, kind, cand[k])) for k, kind in (("user", "user"), ("reply", "assistant"))}
            rows.append({
                "format": FORMAT, "exchange_id": f"{corpus}-{kept:04d}", "corpus": corpus, "index": kept,
                "source_id": cand["source_id"], "dataset": config.DATASET_IDS[corpus],
                "user": cand["user"], "reply": cand["reply"],
                "n_user_tokens_olmo": cand["n_user"], "n_reply_tokens_olmo": cand["n_reply"],
                "reply_truncated_olmo": cand["n_reply"] > config.EXCHANGE_REPLY_MAX_TOKENS,
                "authorship": authorship(corpus),
                "legacy": {k: (None if v is None else {"file": v[0], "id": v[1]}) for k, v in leg.items()},
                "contamination": cand["contamination"], "meta": cand["meta"]})
            kept += 1
            if kept == n:
                break
        st["kept"] = kept
        stats[corpus] = dict(st)
        log(f"{corpus}: {dict(st)}")
        if kept < n:
            raise SystemExit(f"only {kept} usable {corpus} exchanges (wanted {n})")
    return rows, stats


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main_build(args):
    from huggingface_hub import HfApi
    from core.check_contamination import DOLCI, dolci_turns
    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"{args.out} exists; data/ files are regenerated only deliberately (--overwrite)")
    api = HfApi()
    toks, tok_prov = {}, {}
    for fam in config.FAMILIES:
        name = config.FAMILIES[fam]["template"]
        toks[fam] = load_tokenizer(name)
        tok_prov[fam] = {"name": name, "revision": api.model_info(name).sha,
                         "chat_template_sha256": hashlib.sha256((toks[fam].chat_template or "").encode()).hexdigest()}
    pinned = {k: v for k, v in (x.split("=", 1) for x in args.dolci_revision)}
    pinned = {name: api.dataset_info(ds, revision=pinned.get(name, "main")).sha for name, ds in DOLCI.items()}
    turns = dolci_turns(revisions=pinned)
    data_revisions = {c: api.dataset_info(config.DATASET_IDS[c]).sha for c in config.EXCHANGE_CORPORA}
    rows, stats = build(args.n, args.seed, toks, turns)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest = {
        "format": FORMAT, "output": args.out, "output_sha256": sha256_file(args.out), "n_per_corpus": args.n,
        "seed": args.seed, "stats": stats,
        "filters": {"user_tokens": [config.MIN_TOKENS, config.MAX_TOKENS], "reply_min_tokens": config.EXCHANGE_REPLY_MIN_TOKENS,
                    "reply_scored_tokens": config.EXCHANGE_REPLY_MAX_TOKENS, "length_tokenizer": config.FAMILIES["olmo3-7b"]["template"],
                    "render_checked": {f: list(CONTEXTS) for f in toks},
                    "dolci": "whitespace-normalized exact match of BOTH turns against both training roles of Dolci SFT, DPO "
                             "(chosen+rejected) and RL prompt turns, plus WildChat conversation_hash (core.check_contamination."
                             "contamination_record); an exchange with any match is dropped"},
        "dolci_revisions": pinned, "dataset_revisions_at_build": data_revisions, "tokenizers": tok_prov,
        "code_sha256": {p: sha256_file(os.path.join(root, p)) for p in
                        ("core/config.py", "core/exchanges.py", "core/render.py", "core/prepare_data.py",
                         "core/check_contamination.py", "core/conversations.py")},
        "environment": {"python": platform.python_version(),
                        **{p: version(p) for p in ("datasets", "transformers", "tokenizers", "huggingface-hub")}}}
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {len(rows)} exchanges to {args.out} and {args.manifest}")


def load_exchanges(path=None, n_per_corpus=None, limit=None):
    """Rows of the exchange set in file order: the first n_per_corpus of each corpus, then at most `limit` in all."""
    from core.render import load_jsonl
    rows = load_jsonl(path or config.EXCHANGES_PATH)
    if any(r.get("format") != FORMAT for r in rows):
        raise SystemExit(f"{path}: not an {FORMAT} file")
    if n_per_corpus:
        rows = [r for r in rows if r["index"] < n_per_corpus]
    return rows[:limit] if limit else rows


def main_render(args):
    """Examples of all four conditions, both contexts, every family; scored tokens wrapped in ⟦ ⟧."""
    rows = load_exchanges(args.exchanges)
    picks = [next(r for r in rows if r["corpus"] == c) for c in config.EXCHANGE_CORPORA]
    lines = ["# Rendered exchanges", "", "Generated by `python -m core.exchanges render`. One exchange per corpus, every family, "
             "both reply contexts. Scored tokens are wrapped in ⟦ ⟧ (display only); the sequence ends at the last scored token "
             f"(replies are cut at {config.EXCHANGE_REPLY_MAX_TOKENS} tokens). Plain sequences start with the family's "
             "document-start token.", ""]
    for fam in config.FAMILIES:
        tok = load_tokenizer(config.FAMILIES[fam]["template"])
        lines += [f"## {fam}", ""]
        for ex in picks[: args.n]:
            for context in CONTEXTS:
                out = render_exchange(tok, ex, context, fam)
                for (scored, framing), s in out.items():
                    if scored == "user" and context != "exchange":
                        continue  # the user turn does not depend on the reply's context
                    body = tok.decode(s.input_ids[: s.start]) + "⟦" + tok.decode(s.input_ids[s.start:s.end]) + "⟧"
                    lines += [f"### {ex['exchange_id']} · {scored} · {framing}" + (f" · context {context}" if scored == "reply" else ""),
                              "", f"{s.end - s.start} scored tokens of {len(s.input_ids)}" + (" (truncated)" if s.truncated else ""),
                              "", "````", body, "````", ""]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {args.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n", type=int, default=config.EXCHANGE_N, help="exchanges per corpus")
    b.add_argument("--seed", type=int, default=config.SEED)
    b.add_argument("--out", default=config.EXCHANGES_PATH)
    b.add_argument("--manifest", default="data/exchanges_manifest.json")
    b.add_argument("--dolci-revision", action="append", default=[
        "sft=bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221", "dpo=aed155cf32e809b590490b6c3577ee4b0d0a5019",
        "rl=1a8d79b1012d77e4bb1fd7f699c5c7eb33ef931e"], metavar="NAME=SHA",
                   help="Dolci revisions (default: the ones pinned for candidates/logit_distortion/pilot_v3)")
    b.add_argument("--overwrite", action="store_true")
    r = sub.add_parser("render")
    r.add_argument("--exchanges", default=config.EXCHANGES_PATH)
    r.add_argument("--n", type=int, default=3)
    r.add_argument("--out", default="docs/rendered_exchanges.md")
    args = ap.parse_args(argv)
    (main_build if args.cmd == "build" else main_render)(args)


if __name__ == "__main__":
    main()
