"""Build inspectable conversation candidates, NEVER inputs for core.score.

    python -m core.build_conversations --out candidates/logit_distortion/pilot_v1

Only source data and the configured tokenizer are loaded; no model weights.
See docs/conversation_candidates.md for sampling, formats, and pinned reproduction.
"""

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
from itertools import islice
import json
from pathlib import Path
import platform

from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from core import config
from core.check_contamination import DOLCI, dolci_turns
from core.conversation_audit import (AUDIT_VERSION, audit, audit_markdown, examples_markdown,
                                     sensitivity_markdown)
from core.conversations import (FORMAT, Settings, digest, flat_conversations, length_sensitivity, make_targets,
                                oasst_branches, output_order, select_by_eligible_quota, select_conversations)


def overrides(values, allowed):
    result = {}
    for value in values:
        key, sep, val = value.partition("=")
        if not sep or key not in allowed or not val or key in result:
            raise ValueError(f"expected unique SOURCE=VALUE for {list(allowed)}; got {value!r}")
        result[key] = val
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def save_build(out, conversations, targets, rejected, report, manifest, sensitivity=None):
    """Refuse overwrite: frozen or existing runs cannot be accidentally replaced."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    if sensitivity is not None:
        write_json(out / "length_sensitivity.json", sensitivity)
        (out / "length_sensitivity.md").write_text(
            sensitivity_markdown(sensitivity, manifest["settings"]["min_target_tokens"]), encoding="utf-8")
    write_jsonl(out / "conversations.jsonl", conversations)
    write_jsonl(out / "targets.jsonl", targets)
    write_jsonl(out / "exclusions.jsonl", rejected)
    write_json(out / "audit.json", report)
    (out / "audit.md").write_text(audit_markdown(report), encoding="utf-8")
    (out / "examples.md").write_text(examples_markdown(conversations, targets), encoding="utf-8")
    manifest["output_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())}
    write_json(out / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in asdict(Settings()).items():
        if name == "selection_mode":
            parser.add_argument("--selection-mode", choices=["raw_budget", "eligible_quota"], default=default,
                                help="raw_budget: first N acceptable conversations per source (pilot_v1/v2). "
                                     "eligible_quota: keep contributing conversations up to --contributing-quota, "
                                     "evaluating at most --scan-limit per source; requires --check-dolci")
        else:
            parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    parser.add_argument("--sensitivity-min-target-tokens", type=int, nargs="+", default=[5, 10, 20],
                        help="minimum target lengths for the data-only sensitivity audit")
    parser.add_argument("--out", default=config.CONVERSATION_CANDIDATES_DIR)
    parser.add_argument("--revision", action="append", default=[], metavar="SOURCE=SHA")
    parser.add_argument("--split", action="append", default=[], metavar="SOURCE=SPLIT")
    parser.add_argument("--tokenizer-revision", default="main")
    parser.add_argument("--check-dolci", action="store_true", help="Full named SFT/DPO/RL scan; potentially large")
    parser.add_argument("--dolci-revision", action="append", default=[], metavar="STAGE=SHA")
    args = parser.parse_args()
    settings = Settings(**{k: getattr(args, k) for k in asdict(Settings())})
    if Path(args.out).exists():
        parser.error("output already exists; choose a new directory")
    revisions = overrides(args.revision, config.CONVERSATION_SOURCES)
    splits = overrides(args.split, config.CONVERSATION_SOURCES)
    dolci_revisions = overrides(args.dolci_revision, DOLCI)
    if dolci_revisions and not args.check_dolci:
        parser.error("--dolci-revision requires --check-dolci")
    if settings.selection_mode == "eligible_quota" and not args.check_dolci:
        parser.error("--selection-mode eligible_quota requires --check-dolci (unchecked is never clearance)")
    api = HfApi()
    candidates, rejected, source_info = [], [], {}
    for source, spec in config.CONVERSATION_SOURCES.items():
        revision = api.dataset_info(spec["dataset"], revision=revisions.get(source, "main")).sha
        provenance = {**spec, "split": splits.get(source, spec["split"]), "revision": revision}
        print(f"Reading {source} {provenance}", flush=True)
        ds = load_dataset(spec["dataset"], split=provenance["split"], revision=revision, streaming=True)
        rows = list(ds) if source == "oasst1" else list(islice(ds, settings.source_window))
        if source == "oasst1":
            cs, rs, inspected = oasst_branches(rows, provenance, settings)
        else:
            cs, rs, inspected = flat_conversations(rows, source, provenance, settings)
        candidates.extend(cs)
        rejected.extend(rs)
        source_info[source] = {**provenance, "rows_read": len(rows), "inspected_candidates": inspected,
                               "adapter_accepted": len(cs), "input_rows_sha256": digest(rows),
                               "features": ds.features.to_dict() if ds.features else None,
                               "window": "complete split read; first roots in encounter order" if source == "oasst1"
                               else "first rows in dataset iteration order; no corpus shuffle"}
        if source == "oasst1":
            source_info[source]["trees_in_complete_split"] = len({r.get("message_tree_id") for r in rows})
        print(f"  {len(rows)} rows read; {inspected} candidates inspected; {len(cs)} pass source filters", flush=True)
    tokenizer_revision = api.model_info(config.TEMPLATE_MODEL, revision=args.tokenizer_revision).sha
    tokenizer = AutoTokenizer.from_pretrained(config.TEMPLATE_MODEL, revision=tokenizer_revision)
    contamination = None
    contamination_scope = {"status": "unchecked", "training_datasets": [],
                           "scope": "No training data scanned. Unchecked candidates, not held-out or clean."}
    if args.check_dolci:
        pinned = {name: api.dataset_info(dataset, revision=dolci_revisions.get(name, "main")).sha
                  for name, dataset in DOLCI.items()}
        contamination = dolci_turns(revisions=pinned)
        contamination_scope = {"status": "checked", "training_datasets": [
            {"dataset": dataset, "revision": pinned[name], "split": "train"} for name, dataset in DOLCI.items()],
            "scope": "Full named Dolci sets, both roles in SFT messages/DPO chosen+rejected; RL prompt turns and hashes. "
                     "Whitespace-normalized exact matches only; no broader contamination guarantee."}
    scan = None
    if settings.selection_mode == "eligible_quota":
        conversations, targets, selection_rejections, scan = select_by_eligible_quota(
            candidates, settings, tokenizer, contamination)
        for source, counts in scan.items():
            source_info[source]["quota_scan"] = counts
    else:
        conversations, selection_rejections = select_conversations(candidates, settings)
        targets = make_targets(conversations, tokenizer, settings, contamination)
    rejected.extend(selection_rejections)
    selection_order = [c["conversation_id"] for c in conversations]
    sensitivity = length_sensitivity(targets, settings, sorted(set(args.sensitivity_min_target_tokens)))
    conversations, targets = output_order(conversations, targets)
    report = audit(conversations, targets, rejected, source_info, settings, AUDIT_VERSION)
    code_paths = ["core/config.py", "core/check_contamination.py", "core/conversations.py",
                  "core/conversation_audit.py", "core/build_conversations.py"]
    root = Path(__file__).resolve().parents[1]
    manifest = {"format": FORMAT, "candidate_only": True, "settings": asdict(settings), "sources": source_info,
                "audit_version": AUDIT_VERSION, "selection_order": selection_order,
                "sensitivity_min_target_tokens": sorted(set(args.sensitivity_min_target_tokens)),
                "duplicates": "Duplicate case = same source, role, authorship, whitespace-normalized history and "
                              "target; first otherwise-eligible member (selection order) is kept, others get "
                              "duplicate_case. Same wording in a different case only sets repeated_text_of.",
                "tokenizer": {"model": config.TEMPLATE_MODEL, "revision": tokenizer_revision,
                              "chat_template_sha256": digest(tokenizer.chat_template)},
                "contamination": contamination_scope,
                "normalization": "Duplicate/exact-match keys only: Python whitespace split/join, case preserved. Original text unchanged.",
                "sampling": ("First bounded windows; seeded SHA256 child selection before branch quality filters; "
                             "seeded SHA256 conversation ranking, per-source budgets after deduplication. Not a uniform corpus sample."
                             if settings.selection_mode == "raw_budget" else
                             "First bounded windows; seeded SHA256 child selection before branch quality filters; "
                             "per source, candidates in seeded SHA256 rank order are evaluated (at most scan_limit) and kept only "
                             "if they contribute (>= 1 eligible target with clear target and history contamination checks), "
                             "until contributing_quota. Selects toward clean, length-compatible conversations; not a uniform corpus sample."),
                "scoring": "Not compatible with core.score isolated-turn format. Original-context scoring is a subsequent task.",
                "code_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in code_paths},
                "environment": {"python": platform.python_version(), **{p: version(p) for p in
                                ("datasets", "transformers", "huggingface-hub", "tokenizers", "pyarrow")}}}
    save_build(args.out, conversations, targets, rejected, report, manifest, sensitivity)
    print(json.dumps(report["groups"], indent=2), flush=True)
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
