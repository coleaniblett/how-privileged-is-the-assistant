"""Teacher-forced scoring of original-conversation targets (logit distortion's primary route).

    # tokenizer + cached model configs only (no weights): validate, render, write the input manifest and audit
    python -m core.score_conversations --family olmo3-7b --candidates candidates/logit_distortion/pilot_v3 --prepare-only
    # one checkpoint (fp32, GPU): per-token NLL/entropy and per-target summaries
    python -m core.score_conversations --family olmo3-7b --candidates candidates/logit_distortion/pilot_v3 --model base
    # check that every scored stage in a run directory used the manifest's inputs
    python -m core.score_conversations --family olmo3-7b --check-stages

Each target is ONE sequence: its original preceding messages followed by the target, nothing after it (a template
can render earlier turns differently when later turns exist, e.g. Qwen3's think block). Every recorded target token is
scored conditional on that history and the earlier tokens of its own turn (core.score.spans_nll, unchanged: NLL and
full-vocabulary entropy in nats). No text is generated. Rendering is core.render.render_target with the family's
declared PROFILES entry, verified against the cached template. See docs/conversation_scoring.md.

Inputs are candidate builds from core.build_conversations, verified first (hashes, history prefixes). Only targets the
build marked eligible are scored; rendering and length exclusions are recorded separately and are the same for every
stage of a family (common context limit over the family's checkpoints). The input fingerprint covers candidate hashes,
tokenizer revision, template hash, rendering settings, the target set, exclusions and every input-token sequence and
span; stage provenance (model, revision, device) is stored per stage and never part of the fingerprint.
"""

import argparse
import hashlib
import json
import os
import platform
from importlib.metadata import version

from core import config
from core.render import TargetRenderError, load_tokenizer, render_target

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORMAT = "conversation_scores_v1"
DEFAULT_RUN = "logit_conversations"
OUT_KIND = "conversation_scores"
META = ["conversation_id", "target_id", "message_id", "source", "authorship", "role", "group", "turn_index",
        "turn_position"]

# Explicit, pre-declared rendering per family. user_header / assistant_header are the exact text that must end the
# rendered lead-in to a target's content (role header plus any opening); prepare() checks every target against the
# cached template and refuses to run on any difference. assistant_header includes the generation opening.
PROFILES = {
    "olmo3-7b": dict(template_kwargs={}, user_header="<|im_start|>user\n", assistant_header="<|im_start|>assistant\n",
                     system="template default system prompt when the conversation has no original system message; "
                            "an original system message replaces it (template behavior, verified), never duplicated"),
    "qwen3-8b": dict(template_kwargs={"enable_thinking": False}, user_header="<|im_start|>user\n",
                     assistant_header="<|im_start|>assistant\n<think>\n\n</think>\n\n",
                     system="no default system prompt; original system messages rendered as given",
                     think="non-thinking: assistant targets open with the empty think block (enable_thinking=False); "
                           "earlier assistant turns in their stored form (the template drops their think block)"),
    "gemma4-31b": dict(template_kwargs={}, user_header="<|turn>user\n",
                       assistant_header="<|turn>model\n<|channel>thought\n<channel|>",
                       system="no default system prompt; original system messages rendered as given",
                       channel="assistant targets open with the empty thought channel of the generation prompt; "
                               "earlier assistant turns in their stored form (no channel)"),
}


def sha256(value):
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, ensure_ascii=False,
                                                             separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


# --- inputs -----------------------------------------------------------------------------------------------------------

def load_candidates(path):
    """Verify the build offline (hashes, exact prefixes, audit), then return (provenance, targets)."""
    from core.conversation_audit import verify_build

    verify_build(path)
    manifest_bytes = open(os.path.join(path, "manifest.json"), "rb").read()
    manifest = json.loads(manifest_bytes)
    with open(os.path.join(path, "targets.jsonl"), encoding="utf-8") as f:
        targets = [json.loads(line) for line in f]
    prov = {"path": os.path.normpath(path), "format": manifest["format"], "manifest_sha256": sha256(manifest_bytes),
            "output_sha256": manifest["output_sha256"], "settings": manifest["settings"],
            "contamination": manifest.get("contamination", {}).get("status")}
    return prov, targets


def context_limit(model_id):
    """(max_position_embeddings, source) from the cached model config; raises if it is not available (never guessed)."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    for c in (getattr(cfg, "text_config", None), cfg):
        n = getattr(c, "max_position_embeddings", None) if c is not None else None
        if n:
            return int(n), f"{model_id} config max_position_embeddings"
    raise SystemExit(f"{model_id}: no max_position_embeddings in its config; refusing to guess a context limit")


def cached_revision(repo, filename):
    """Snapshot commit of a cached Hub file, or None (e.g. a local path)."""
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(repo, filename)
    return path.split("/snapshots/")[1].split("/")[0] if isinstance(path, str) and "/snapshots/" in path else None


def tokenizer_provenance(tok, name):
    return {"name": name, "revision": cached_revision(name, "tokenizer_config.json"), "chat_template_sha256": sha256((tok.chat_template or "").encode()),
            "class": type(tok).__name__}


def prepare(targets, tok, profile, limit):
    """Render every eligible target. Returns (included, exclusions); included rows carry META, input_ids, start, end."""
    included, exclusions = [], []
    for t in targets:
        base = {k: t[k] for k in META}
        if not t["eligible"]:
            exclusions.append({**base, "stage": "candidate", "reason": "candidate_ineligible",
                               "detail": t["exclusion_reasons"]})
            continue
        messages = [{"role": m["role"], "content": m["content"]} for m in t["preceding_messages"]]
        messages.append({"role": t["role"], "content": t["target_text"]})
        try:
            r = render_target(tok, messages, profile["template_kwargs"])
        except TargetRenderError as e:
            i = e.message_index
            history = t["preceding_messages"]
            exclusions.append({**base, "stage": "rendering", "reason": e.reason, "affected_message_index": i,
                               "affected_message_id": (None if i is None else history[i]["message_id"] if i < len(history)
                                                       else t["message_id"]),
                               "affected_is_target": None if i is None else i == len(history)})
            continue
        expected = profile[f"{t['role']}_header"]
        if not r.lead.endswith(expected):
            raise SystemExit(f"{t['target_id']}: rendered lead-in ...{r.lead[-60:]!r} does not end with the declared "
                             f"{t['role']} header {expected!r}; update PROFILES deliberately, never silently")
        if len(r.input_ids) > limit:
            exclusions.append({**base, "stage": "length", "reason": "input_too_long", "n_input_tokens": len(r.input_ids),
                               "limit": limit})
            continue
        included.append({**base, "input_ids": r.input_ids, "start": r.start, "end": r.end, "n_input_tokens": len(r.input_ids),
                         "n_target_tokens": r.end - r.start, "input_sha256": sha256(r.input_ids)})
    return included, exclusions


def fingerprint(fixed):
    """Hash of everything that must be identical across a family's stages (no stage/model/code fields)."""
    keys = ("candidates", "tokenizer", "family", "rendering", "context_limit", "targets", "exclusions")
    return sha256({k: fixed[k] for k in keys})


def build_manifest(family, cand_prov, tok_prov, profile, limits, included, exclusions):
    fixed = {
        "candidates": {k: cand_prov[k] for k in ("format", "manifest_sha256", "output_sha256")},
        "tokenizer": {k: tok_prov[k] for k in ("name", "revision", "chat_template_sha256")},
        "family": family,
        "rendering": {k: v for k, v in profile.items()},
        "context_limit": min(n for n, _ in limits.values()),
        "targets": [{"target_id": r["target_id"], "input_sha256": r["input_sha256"], "start": r["start"], "end": r["end"],
                     "n_input_tokens": r["n_input_tokens"]} for r in included],
        "exclusions": [{k: v for k, v in e.items() if k in ("target_id", "stage", "reason", "affected_message_id",
                                                             "affected_message_index", "n_input_tokens", "limit")} for e in exclusions],
    }
    return {"format": FORMAT, **fixed, "input_fingerprint": fingerprint(fixed),
            "candidate_path": cand_prov["path"], "candidate_contamination": cand_prov["contamination"],
            "context_limits": {s: {"limit": n, "source": src} for s, (n, src) in limits.items()},
            "code_sha256": {p: sha256(open(os.path.join(ROOT, p), "rb").read()) for p in
                            ("core/render.py", "core/score.py", "core/score_conversations.py")},
            "environment": {"python": platform.python_version(),
                            **{p: version(p) for p in ("transformers", "tokenizers", "torch")}}}


def audit_markdown(manifest, included, exclusions):
    from collections import Counter

    lines = [f"# Conversation-scoring inputs: {manifest['family']}", "",
             f"Candidates: `{manifest['candidate_path']}` (contamination: {manifest['candidate_contamination']}). "
             f"Tokenizer `{manifest['tokenizer']['name']}` @ `{manifest['tokenizer']['revision']}`. "
             f"Input fingerprint `{manifest['input_fingerprint'][:16]}…`. Inputs only; no model was run.", "",
             "Rendering: " + "; ".join(f"{k}: {v!r}" for k, v in manifest["rendering"].items()), "",
             f"Context limit (minimum over the family's checkpoints): {manifest['context_limit']} "
             f"({', '.join(f'{s} {v['limit']}' for s, v in manifest['context_limits'].items())}).", "",
             "| Source | Group | Eligible candidates | Scored | Rendering exclusions | Length exclusions |",
             "|---|---|---:|---:|---:|---:|"]
    cells = sorted({(r["source"], r["group"]) for r in included + exclusions})
    for s, g in cells:
        inc = sum(r["source"] == s and r["group"] == g for r in included)
        ex = [e for e in exclusions if e["source"] == s and e["group"] == g]
        rend = sum(e["stage"] == "rendering" for e in ex)
        ln = sum(e["stage"] == "length" for e in ex)
        if inc + rend + ln:
            lines.append(f"| {s} | {g} | {inc + rend + ln} | {inc} | {rend} | {ln} |")
    reasons = Counter((e["stage"], e["reason"]) for e in exclusions)
    lines += ["", "Exclusions by stage and reason (candidate = ineligible in the build, not scored by design):", ""]
    lines += [f"- {s} / {r}: {n}" for (s, r), n in sorted(reasons.items())]
    rend = [e for e in exclusions if e["stage"] != "candidate"]
    if rend:
        lines += ["", "| Target | Stage | Reason | Affected message | Is the target |", "|---|---|---|---|---|"]
        lines += [f"| `{e['target_id']}` | {e['stage']} | {e['reason']} | `{e.get('affected_message_id')}` | "
                  f"{e.get('affected_is_target')} |" for e in rend]
    if included:
        n = sorted(r["n_input_tokens"] for r in included)
        lines += ["", f"Scored input lengths (tokens, including history, headers and closing): min {n[0]}, "
                      f"median {n[len(n) // 2]}, max {n[-1]}. Nothing is truncated."]
    return "\n".join(lines) + "\n"


def write_prepared(out, manifest, exclusions, included):
    """Write (or confirm) the run's input manifest. An existing manifest with another fingerprint is never replaced."""
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "inputs_manifest.json")
    if os.path.exists(path):
        old = json.load(open(path))
        if old["input_fingerprint"] != manifest["input_fingerprint"]:
            raise SystemExit(f"{path}: existing inputs differ (fingerprint {old['input_fingerprint'][:16]}… vs "
                             f"{manifest['input_fingerprint'][:16]}…); use a new --run or --out, never mix inputs")
        return False
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    with open(os.path.join(out, "exclusions.jsonl"), "w", encoding="utf-8") as f:
        for e in exclusions:
            f.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")
    with open(os.path.join(out, "rendering_audit.md"), "w", encoding="utf-8") as f:
        f.write(audit_markdown(manifest, included, exclusions))
    return True


# --- scoring ----------------------------------------------------------------------------------------------------------

def score_targets(model, device, included, pad_id, batch_size):
    """{target_id: (nll, entropy)} tensors over each target's content tokens (spans_nll, one sequence per target)."""
    from core.score import spans_nll

    seqs = [(r["input_ids"], {r["target_id"]: (r["start"], r["end"])}) for r in included]
    return spans_nll(model, device, seqs, pad_id, batch_size, entropy=True)


def tables(included, nll, ent):
    """(per-token DataFrame, per-target DataFrame). Means over all content tokens and over positions 1.. use the same
    positions for NLL and entropy; gap = mean NLL - mean entropy. A target with fewer than 2 tokens has from1 = NaN and
    from1_defined False (never a silent invalid mean)."""
    import numpy as np
    import pandas as pd

    tok_cols = {k: [] for k in META + ["position", "token_id", "nll", "entropy"]}
    rows = []
    for r in included:
        v, e = nll[r["target_id"]].double().numpy(), ent[r["target_id"]].double().numpy()
        n = len(v)
        assert n == r["end"] - r["start"] and n == len(e)
        for k in META:
            tok_cols[k] += [r[k]] * n
        tok_cols["position"] += list(range(n))
        tok_cols["token_id"] += r["input_ids"][r["start"]:r["end"]]
        tok_cols["nll"] += list(v)
        tok_cols["entropy"] += list(e)
        row = {k: r[k] for k in META}
        row.update(n_tokens=n, n_input_tokens=r["n_input_tokens"], span_start=r["start"], span_end=r["end"],
                   input_sha256=r["input_sha256"])
        if n == 0:
            raise SystemExit(f"{r['target_id']}: empty target span")
        row.update(mean_nll=v.mean(), mean_entropy=e.mean(), gap=v.mean() - e.mean())
        row["from1_defined"] = n >= 2
        row["n_tokens_from1"] = n - 1
        row.update(mean_nll_from1=v[1:].mean() if n >= 2 else np.nan, mean_entropy_from1=e[1:].mean() if n >= 2 else np.nan)
        row["gap_from1"] = row["mean_nll_from1"] - row["mean_entropy_from1"] if n >= 2 else np.nan
        rows.append(row)
    tokens = pd.DataFrame(tok_cols)
    tokens["position"] = tokens["position"].astype("int32")
    tokens["turn_index"] = tokens["turn_index"].astype("int32")
    tokens["token_id"] = tokens["token_id"].astype("int64")
    for c in ("nll", "entropy"):
        tokens[c] = tokens[c].astype("float32")
    for c in ("conversation_id", "target_id", "message_id", "source", "authorship", "role", "group", "turn_position"):
        tokens[c] = tokens[c].astype("category")
    return tokens, pd.DataFrame(rows)


def check_stages(out):
    """Every <stage>_provenance.json in `out` must carry the run manifest's input fingerprint."""
    manifest = json.load(open(os.path.join(out, "inputs_manifest.json")))
    stages = {}
    for name in sorted(os.listdir(out)):
        if name.endswith("_provenance.json"):
            p = json.load(open(os.path.join(out, name)))
            if p["input_fingerprint"] != manifest["input_fingerprint"]:
                raise SystemExit(f"{name}: input fingerprint {p['input_fingerprint'][:16]}… differs from the manifest's "
                                 f"{manifest['input_fingerprint'][:16]}…; these stages must not be combined")
            stages[p["stage"]] = p["model"]
    return stages


def main(argv=None):
    import sys

    argv = sys.argv[1:] if argv is None else argv
    config.family_from_argv(argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=sorted(config.FAMILIES))
    ap.add_argument("--candidates", default=config.CONVERSATION_CANDIDATES_DIR)
    ap.add_argument("--run", default=DEFAULT_RUN, help=f"run directory under results/ (default {DEFAULT_RUN})")
    ap.add_argument("--out", help="default: results/<run>/<family>/core/conversation_scores")
    ap.add_argument("--prepare-only", action="store_true", help="validate and write the input manifest; no weights")
    ap.add_argument("--check-stages", action="store_true", help="verify every scored stage used the manifest's inputs")
    ap.add_argument("--model", help="stage name (or model id) to score")
    ap.add_argument("--batch-size", type=int)
    args = ap.parse_args(argv)
    fam = config.family_settings(args.family)
    out = args.out or config.core_dir(OUT_KIND, args.run, args.family)
    if args.check_stages:
        print(json.dumps(check_stages(out), indent=1))
        return
    if not args.prepare_only and not args.model:
        ap.error("--model is required unless --prepare-only or --check-stages")
    if args.family not in PROFILES:
        raise SystemExit(f"no rendering profile declared for {args.family}")
    profile = PROFILES[args.family]
    cand_prov, targets = load_candidates(args.candidates)
    tok = load_tokenizer(fam["template"])
    limits = {stage: context_limit(model_id) for stage, model_id in fam["stages"].items()}
    included, exclusions = prepare(targets, tok, profile, min(n for n, _ in limits.values()))
    manifest = build_manifest(args.family, cand_prov, tokenizer_provenance(tok, fam["template"]), profile, limits,
                              included, exclusions)
    new = write_prepared(out, manifest, exclusions, included)
    print(f"{args.family}: {len(included)} targets to score, {len(exclusions)} excluded "
          f"({'wrote' if new else 'matches existing'} {out}/inputs_manifest.json; fingerprint "
          f"{manifest['input_fingerprint'][:16]}…)", flush=True)
    if args.prepare_only:
        return

    import torch

    from core.score import describe_placement, load_model, resolve

    model_id, stage = resolve(args.model)
    paths = {k: os.path.join(out, f"{stage}{suffix}") for k, suffix in
             (("tokens", "_tokens.parquet"), ("targets", ".csv"), ("provenance", "_provenance.json"))}
    if any(os.path.exists(p) for p in paths.values()):
        raise SystemExit(f"{stage}: outputs already exist in {out}; never overwritten")
    model, device = load_model(model_id, torch.float32, offload=fam["offload"], gpu_gib=fam["gpu_gib"])
    print(f"{model_id}: {describe_placement(model)}, dtype {next(model.parameters()).dtype}", flush=True)
    n_vocab = model.get_input_embeddings().num_embeddings
    assert max(max(r["input_ids"]) for r in included) < n_vocab, "tokenizer ids exceed model vocab"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    nll, ent = score_targets(model, device, included, pad_id, args.batch_size or fam["batch_size"])
    tokens, per_target = tables(included, nll, ent)
    tokens.to_parquet(paths["tokens"], index=False)
    per_target.to_csv(paths["targets"], index=False, float_format="%.6f")
    with open(paths["provenance"], "w") as f:
        json.dump({"format": FORMAT, "stage": stage, "model": model_id,
                   "model_revision": cached_revision(model_id, "config.json"), "family": args.family, "dtype": "float32",
                   "device": str(device), "placement": describe_placement(model),
                   "batch_size": args.batch_size or fam["batch_size"], "input_fingerprint": manifest["input_fingerprint"],
                   "n_targets": len(included), "n_tokens": len(tokens),
                   "environment": manifest["environment"]}, f, indent=1, sort_keys=True)
    print(f"{model_id}: scored {len(included)} targets ({len(tokens)} tokens) -> {paths['targets']}", flush=True)


if __name__ == "__main__":
    main()
