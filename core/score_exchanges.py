"""Teacher-forced scoring of the exchange set (core.exchanges) for one checkpoint: user and reply, tagged and plain.

    python -m core.score_exchanges --family olmo3-7b --model base                     # -> results/capstone/olmo3-7b/base/core/exchanges_exchange/
    python -m core.score_exchanges --family qwen3-8b --model instruct --context single --tag exchanges_single
    python -m core.score_exchanges --family gemma4-31b --model instruct               # first 100 per corpus (the family's exchange_n_per_corpus)
    python -m core.score_exchanges --family qwen3-8b --prepare-only                   # render + check every exchange, no weights
    python -m core.score_exchanges --family olmo3-7b --check-stages                   # every stage of a tag used the same inputs

Writes results/<run>/<family>/<stage>/core/<tag>/ (never the overnight paths; an existing directory is refused):
  scores.csv      family, stage, corpus, exchange_id, scored_text (user | reply), framing (tagged | plain), context,
                  condition ("Gemma IT · UltraChat · reply · plain"), n_tokens (scored), n_tokens_full, truncated,
                  mean_nll, mean_nll_from1, mean_entropy, mean_entropy_from1 (_from1: content positions 1+; nats),
                  n_bytes_from1 (UTF-8 bytes of the scored tokens 1+, for bits per byte), score_source
  tokens.parquet  family, stage, corpus, exchange_id, scored_text, framing, position (0 = first scored token), nll, entropy
  manifest.json   inputs fingerprint (identical for every stage of a family and tag), settings, model provenance, reuse

fp32, scored with core.score.spans_nll (unchanged). Reuse (default on, --no-reuse to score everything fresh): a user turn
already scored overnight (results/capstone/<family>/core/scores{,_authorship}/<stage>.csv) is reused only if the overnight
text is the exchange's text, the ids scored here equal core.render.render's ids for that slot (what core.score scored), and
the overnight row has the same number of tokens; with --context single the same holds for untruncated replies (asst@asst,
asst@plain). Everything else is scored fresh. score_source says which.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from collections import Counter
from importlib.metadata import version

from core import config
from core.exchanges import FORMAT as EXCHANGES_FORMAT
from core.exchanges import CONTEXTS, condition_label, family_profile, load_exchanges, render_exchange, think_block
from core.render import AlignmentError, load_jsonl, load_tokenizer, render

FORMAT = "exchange_scores_v1"
# Overnight text sets and the score directory each was written to (results/capstone/<family>/core/<dir>/<stage>.csv).
LEGACY_SCORE_DIRS = {config.DATA_PATH: "scores", config.AUTHORSHIP_PATH: "scores_authorship"}
LEGACY_CONDITION = {("user", "tagged"): ("user@user", "user"), ("user", "plain"): ("user@plain", None),
                    ("reply", "tagged"): ("asst@asst", "assistant"), ("reply", "plain"): ("asst@plain", None)}
COLUMNS = ["family", "stage", "corpus", "exchange_id", "scored_text", "framing", "context", "condition", "n_tokens",
           "n_tokens_full", "truncated", "mean_nll", "mean_nll_from1", "mean_entropy", "mean_entropy_from1",
           "n_bytes_from1", "score_source"]


def sha256(value):
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def prepare(tok, rows, context, family):
    """[(key, exchange row, Scored)] for every condition of every exchange that renders, and the exclusions.
    key = (exchange_id, scored_text, framing). An exchange that fails any check is dropped whole (all four conditions)."""
    items, exclusions = [], []
    for ex in rows:
        try:
            out = render_exchange(tok, ex, context, family)
        except AlignmentError as e:
            exclusions.append({"exchange_id": ex["exchange_id"], "reason": str(e)})
            continue
        items += [((ex["exchange_id"], s, f), ex, r) for (s, f), r in out.items()]
    return items, exclusions


def fingerprint(items, tok_prov, settings):
    """Everything that must be identical across a family's stages: inputs, spans, tokenizer, rendering settings."""
    return sha256({"tokenizer": tok_prov, "settings": settings,
                   "items": [[list(k), sha256(r.input_ids), r.start, r.end, r.n_full] for k, _, r in items]})


def tokenizer_provenance(tok, name):
    from core.score_conversations import cached_revision
    return {"name": name, "revision": cached_revision(name, "tokenizer_config.json"),
            "chat_template_sha256": sha256((tok.chat_template or "").encode())}


# --- reuse of the overnight scores -------------------------------------------------------------------------------------

def legacy_scores(family, stage, run="capstone"):
    """{(file, text_id, condition): (csv row, per-token frame)} from the overnight run of this checkpoint; {} if absent."""
    import pandas as pd
    out = {}
    for path, sub in LEGACY_SCORE_DIRS.items():
        d = config.core_dir(sub, run, family)
        c, p = os.path.join(d, f"{stage}.csv"), os.path.join(d, f"{stage}_tokens.parquet")
        if not (os.path.exists(c) and os.path.exists(p)):
            continue
        rows = {(r["text_id"], r["condition"]): {**r, "_csv": c} for r in csv.DictReader(open(c))}
        tk = pd.read_parquet(p, columns=["text_id", "condition", "position", "nll", "entropy"])
        tk["text_id"], tk["condition"] = tk.text_id.astype(str), tk.condition.astype(str)
        for (tid, cond), g in tk.groupby(["text_id", "condition"], sort=False, observed=True):
            if (tid, cond) in rows:
                out[(path, tid, cond)] = (rows[(tid, cond)], g.sort_values("position"))
    return out


def try_reuse(tok, key, ex, r, family, context, legacy, texts):
    """(nll, entropy, source) from the overnight scores if every check passes, else (None, None, reason)."""
    import torch
    _, scored, framing = key
    if scored == "reply" and context != "single":
        return None, None, "fresh"  # the reply's context differs from the overnight run's
    leg = ex["legacy"][scored]
    if not leg:
        return None, None, "fresh"
    if texts.get((leg["file"], leg["id"])) != ex[scored]:
        return None, None, "fresh (overnight text differs)"
    if r.truncated:
        return None, None, "fresh (truncated)"
    cond, slot = LEGACY_CONDITION[(scored, framing)]
    if (leg["file"], leg["id"], cond) not in legacy:
        return None, None, "fresh (not scored overnight)"
    with think_block(family):
        ref = render(tok, ex[scored], slot)  # exactly what core.score scored overnight
    # ids up to the end of the scored span (what follows it cannot change its scores under causal attention)
    if (ref.input_ids[:ref.end], ref.start, ref.end) != (r.input_ids, r.start, r.end):
        return None, None, "fresh (token ids differ from the overnight rendering)"
    row, g = legacy[(leg["file"], leg["id"], cond)]
    n = r.end - r.start
    if int(row["n_tokens"]) != n or len(g) != n or list(g.position) != list(range(n)):
        return None, None, "fresh (overnight token count differs)"
    return torch.tensor(g.nll.to_numpy()), torch.tensor(g.entropy.to_numpy()), f"overnight:{row['_csv']}:{leg['id']}:{cond}"


# --- scoring -----------------------------------------------------------------------------------------------------------

def score_fresh(model, device, items, pad_id, batch_size, log=print):
    """{key: (nll, entropy)} via core.score.spans_nll in length-sorted batches; a batch that runs out of GPU memory is
    retried at half the size (fp32: results do not depend on the batch beyond float noise)."""
    import torch
    from core.score import spans_nll
    order = sorted(items, key=lambda kr: len(kr[1].input_ids))
    out, i, bs, t0 = {}, 0, batch_size, time.time()
    while i < len(order):
        chunk = order[i:i + bs]
        try:
            nll, ent = spans_nll(model, device, [(r.input_ids, {k: (r.start, r.end)}) for k, r in chunk], pad_id,
                                 len(chunk), entropy=True)
        except torch.cuda.OutOfMemoryError:
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            torch.cuda.empty_cache()
            log(f"out of GPU memory at batch {len(chunk)} (length {len(chunk[-1][1].input_ids)}); retrying at {bs}")
            continue
        out.update({k: (nll[k], ent[k]) for k, _ in chunk})
        i += len(chunk)
        if i // bs % 50 == 0 or i == len(order):
            log(f"  scored {i}/{len(order)} sequences ({time.time() - t0:.0f} s)")
    return out


def summary_row(family, stage, ex, key, r, context, nll, ent, source, tok):
    _, scored, framing = key
    v, e = nll.float(), ent.float()
    return {"family": family or "", "stage": stage, "corpus": ex["corpus"], "exchange_id": ex["exchange_id"],
            "scored_text": scored, "framing": framing, "context": context,
            "condition": condition_label(family, stage, ex["corpus"], scored, framing),
            "n_tokens": len(v), "n_tokens_full": r.n_full, "truncated": r.truncated,
            "mean_nll": f"{v.mean().item():.6f}", "mean_nll_from1": f"{v[1:].mean().item():.6f}",
            "mean_entropy": f"{e.mean().item():.6f}", "mean_entropy_from1": f"{e[1:].mean().item():.6f}",
            "n_bytes_from1": len(tok.decode(r.input_ids[r.start + 1:r.end], clean_up_tokenization_spaces=False).encode()),
            "score_source": source}


def save_tokens(path, family, stage, results, by_key):
    import numpy as np
    import pandas as pd
    import torch
    keys = list(results)
    lens = [len(results[k][0]) for k in keys]
    rep = lambda f: [f(k) for k, n in zip(keys, lens) for _ in range(n)]
    df = pd.DataFrame({
        "family": family or "", "stage": stage, "corpus": rep(lambda k: by_key[k]["corpus"]), "exchange_id": rep(lambda k: k[0]),
        "scored_text": rep(lambda k: k[1]), "framing": rep(lambda k: k[2]),
        "position": np.concatenate([np.arange(n, dtype=np.int16) for n in lens]),
        "nll": torch.cat([results[k][0].float() for k in keys]).numpy().astype(np.float32),
        "entropy": torch.cat([results[k][1].float() for k in keys]).numpy().astype(np.float32)})
    for c in ("family", "stage", "corpus", "exchange_id", "scored_text", "framing"):
        df[c] = df[c].astype("category")
    df.to_parquet(path, index=False)


def check_stages(family, tag, run=None):
    """Exit non-zero unless every scored stage of the family under this tag has the same inputs fingerprint."""
    fps = {}
    for stage in config.family_settings(family)["stages"]:
        p = os.path.join(config.exchange_dir(family, stage, tag, run), "manifest.json")
        if os.path.exists(p):
            fps[stage] = json.load(open(p))["inputs_fingerprint"]
    for s, fp in fps.items():
        print(f"{family} {s}: {fp[:16]}…")
    missing = [s for s in config.family_settings(family)["stages"] if s not in fps]
    if missing:
        print(f"not scored yet: {', '.join(missing)}")
    if len(set(fps.values())) > 1:
        raise SystemExit("stages were scored on different inputs")
    print("all scored stages share the inputs fingerprint" if fps else "nothing scored")


def main(argv=None):
    config.family_from_argv(argv)
    fam = config.family_settings() if config.EXPLICIT_FAMILY else None
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", help=f"one of {', '.join(config.FAMILIES)} (omit only with --tokenizer and --out, e.g. the CPU smoke test)")
    ap.add_argument("--model", help="stage name of the family (or a model id)")
    ap.add_argument("--stage", help="label for the stage (default: the stage name)")
    ap.add_argument("--tokenizer", help="default: the family's template model")
    ap.add_argument("--exchanges", default=config.EXCHANGES_PATH)
    ap.add_argument("--context", choices=list(CONTEXTS), default=config.EXCHANGE_CONTEXT, help="the reply's context")
    ap.add_argument("--tag", help="run tag, the output subdirectory (default: exchanges_<context>)")
    ap.add_argument("--run", help=f"run directory under results/ (default {config.EXCHANGE_RUN})")
    ap.add_argument("--out", help="output directory (default: config.exchange_dir(family, stage, tag))")
    ap.add_argument("--n-per-corpus", type=int, default=(fam or {}).get("exchange_n_per_corpus"),
                    help="first N exchanges per corpus (default: the family's exchange_n_per_corpus, else all)")
    ap.add_argument("--limit", type=int, help="at most N exchanges in all, after --n-per-corpus (smoke runs)")
    ap.add_argument("--reuse", action=argparse.BooleanOptionalAction, default=True,
                    help="reuse verified overnight scores of the same tokens (default on)")
    ap.add_argument("--batch-size", type=int, default=(fam or {}).get("batch_size", 8))
    ap.add_argument("--offload", action=argparse.BooleanOptionalAction, default=(fam or {}).get("offload", False))
    ap.add_argument("--gpu-gib", type=int, default=(fam or {}).get("gpu_gib", 34))
    ap.add_argument("--prepare-only", action="store_true", help="render and check every exchange; no weights, no files")
    ap.add_argument("--check-stages", action="store_true", help="compare the inputs fingerprints of the scored stages")
    args = ap.parse_args(argv)
    family = config.EXPLICIT_FAMILY
    tag = args.tag or f"exchanges_{args.context}"
    if args.check_stages:
        return check_stages(family, tag, args.run)
    if not family and not (args.tokenizer and (args.out or args.prepare_only)):
        ap.error("--family is required (or --tokenizer with --out for a family-less smoke run)")

    tok_name = args.tokenizer or config.TEMPLATE_MODEL
    tok = load_tokenizer(tok_name)
    rows = load_exchanges(args.exchanges, args.n_per_corpus, args.limit)
    items, exclusions = prepare(tok, rows, args.context, family)
    settings = {"context": args.context, "separator": config.EXCHANGE_SEPARATOR,
                "reply_max_tokens": config.EXCHANGE_REPLY_MAX_TOKENS, "n_per_corpus": args.n_per_corpus, "limit": args.limit,
                "family_profile": family_profile(family),
                "assistant_think_block": config.ASSISTANT_THINK_BLOCK}
    tok_prov = tokenizer_provenance(tok, tok_name)
    fp = fingerprint(items, tok_prov, settings)
    n_ex = len({k[0] for k, _, _ in items})
    print(f"{family or tok_name}: {n_ex} exchanges ({len(items)} sequences), {len(exclusions)} excluded; context {args.context}; "
          f"inputs fingerprint {fp[:16]}…", flush=True)
    for e in exclusions:
        print(f"  excluded {e['exchange_id']}: {e['reason']}")
    if args.prepare_only:
        return
    if not items:
        raise SystemExit("no exchange renders cleanly with this tokenizer; nothing to score")
    if not args.model:
        ap.error("--model is required unless --prepare-only or --check-stages")

    import torch
    from core.score import describe_placement, load_model, resolve
    from core.score_conversations import cached_revision
    model_id, stage = resolve(args.model, args.stage)
    out = args.out or config.exchange_dir(family, stage, tag, args.run)
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit(f"{out} exists and is not empty; choose another --tag (outputs are never overwritten)")

    by_key = {k: ex for k, ex, _ in items}
    results, sources = {}, {}
    if args.reuse and family:
        legacy = legacy_scores(family, stage)
        texts = {}
        for path in LEGACY_SCORE_DIRS:
            if os.path.exists(path):
                texts.update({(path, r["id"]): r["text"] for r in load_jsonl(path)})
        for k, ex, r in items:
            nll, ent, src = try_reuse(tok, k, ex, r, family, args.context, legacy, texts)
            sources[k] = src
            if nll is not None:
                results[k] = (nll, ent)
    todo = [(k, r) for k, _, r in items if k not in results]
    for k, _ in todo:
        sources.setdefault(k, "fresh")
    print(f"reused {len(results)} sequences from the overnight run, scoring {len(todo)} fresh", flush=True)

    device_info = {}
    if todo:
        t0 = time.time()
        model, device = load_model(model_id, torch.float32, offload=args.offload, gpu_gib=args.gpu_gib)
        print(f"{model_id}: {describe_placement(model)}, dtype {next(model.parameters()).dtype} ({time.time() - t0:.0f} s to load)", flush=True)
        n_vocab = model.get_input_embeddings().num_embeddings
        assert max(max(r.input_ids) for _, r in todo) < n_vocab, "tokenizer ids exceed model vocab"
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        t1 = time.time()
        results.update(score_fresh(model, device, todo, pad_id, args.batch_size))
        device_info = {"device": device, "placement": describe_placement(model), "seconds_scoring": round(time.time() - t1, 1),
                       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

    os.makedirs(out, exist_ok=True)
    scored = items
    with open(os.path.join(out, "scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for k, ex, r in scored:
            w.writerow(summary_row(family, stage, ex, k, r, args.context, *results[k], sources[k], tok))
    save_tokens(os.path.join(out, "tokens.parquet"), family, stage, {k: results[k] for k, _, _ in scored}, by_key)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest = {
        "format": FORMAT, "family": family, "stage": stage, "tag": tag, "model": model_id,
        "model_revision": cached_revision(model_id, "config.json"), "dtype": "fp32", "batch_size": args.batch_size,
        "inputs_fingerprint": fp, "settings": settings, "tokenizer": tok_prov,
        "exchanges": {"path": args.exchanges, "format": EXCHANGES_FORMAT, "sha256": sha256(open(args.exchanges, "rb").read()),
                      "n_exchanges": n_ex, "n_sequences": len(items)},
        "exclusions": exclusions,
        "reuse": {"enabled": bool(args.reuse and family), "n_reused": sum(s.startswith("overnight:") for s in sources.values()),
                  "n_fresh": sum(not s.startswith("overnight:") for s in sources.values()),
                  "reasons": dict(Counter(s if not s.startswith("overnight:") else "overnight" for s in sources.values()))},
        **device_info,
        "code_sha256": {p: sha256(open(os.path.join(root, p), "rb").read()) for p in
                        ("core/exchanges.py", "core/score_exchanges.py", "core/render.py", "core/score.py", "core/config.py")},
        "environment": {"python": platform.python_version(), **{p: version(p) for p in ("transformers", "tokenizers", "torch")}}}
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"{model_id}: wrote {len(scored)} rows to {out}/scores.csv (+ tokens.parquet, manifest.json)", flush=True)


if __name__ == "__main__":
    main()
