"""Per-token NLL of the content tokens in every condition, for one model.

    python -m core.score --model sft                # a stage name from config.MODELS, or any HF id
    python -m core.score --family gemma4-31b --model instruct   # another family -> results/<run>/<family>/core/scores
    python -m core.score --model HuggingFaceTB/SmolLM2-135M-Instruct --tokenizer HuggingFaceTB/SmolLM2-135M-Instruct

Writes <out>/<stage>.csv with columns text_id, source, kind, condition, n_tokens, mean_nll,
mean_nll_from1, mean_entropy, mean_entropy_from1 (_from1: content positions 1+, i.e. without the first
content token; entropy: of the next-token distribution at each content token, nats), and
<out>/<stage>_tokens.parquet with the per-token values (text_id, source, kind, condition, position, nll, entropy).
"""

import argparse
import csv
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from core import config
from core.render import load_jsonl, load_tokenizer, render_all


def resolve(model_arg, stage=None):
    model_id = config.MODELS.get(model_arg, model_arg)
    if stage is None:
        stage = model_arg if model_arg in config.MODELS else model_id.split("/")[-1]
    return model_id, stage


def first_n_per_set(rows, n):
    """The first n rows of each (source, kind) set, in file order (all rows if n is falsy)."""
    if not n:
        return rows
    seen, out = {}, []
    for r in rows:
        key = (r["source"], r["kind"])
        if seen.get(key, 0) < n:
            seen[key] = seen.get(key, 0) + 1
            out.append(r)
    return out


def load_model(model_id, dtype=None, offload=False, gpu_gib=34, cpu_gib=900):
    """dtype None: bf16 on GPU, fp32 on CPU. Scoring passes fp32: in bf16 a text's NLL shifts by
    up to ~0.05 nats with batch composition (measured on OLMo 3 7B), comparable to slot effects.

    offload=True places the weights with device_map="auto": as many layers as fit in gpu_gib GiB on the GPU, the rest in
    CPU RAM and streamed in on every forward pass. The arithmetic is unchanged (fp32 stays fp32); it is what lets a 31B
    model be scored in fp32 on a 48 GB card, at a fraction of the speed. Ported from the pilot's nll_arm/score.py."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if offload:
        if device != "cuda":
            raise SystemExit("offload needs a GPU (device_map='auto' with a CPU spill)")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map="auto", max_memory={0: f"{gpu_gib}GiB", "cpu": f"{cpu_gib}GiB"}).eval()
        return model, "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return model, device


def describe_placement(model):
    """Where the weights ended up, for the log line (absent hf_device_map: everything on one device)."""
    dm = getattr(model, "hf_device_map", None)
    if dm:
        return f"{sum(1 for v in dm.values() if v in (0, 'cuda:0'))}/{len(dm)} modules on the GPU"
    return f"all on {next(model.parameters()).device}"


@torch.no_grad()
def spans_nll(model, device, seqs, pad_id, batch_size, entropy=False):
    """seqs: list of (input_ids, {key: (start, end)}). Returns {key: per-token NLL tensor over each span};
    with entropy=True, also {key: entropy (nats) of the next-token distribution at each span token}."""
    out, ent = {}, {}
    order = sorted(seqs, key=lambda s: len(s[0]))
    for i in range(0, len(order), batch_size):
        batch = order[i : i + batch_size]
        width = max(len(input_ids) for input_ids, _ in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for j, (input_ids, _) in enumerate(batch):
            ids[j, : len(input_ids)] = torch.tensor(input_ids)
            mask[j, : len(input_ids)] = 1
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        for j, (_, spans) in enumerate(batch):
            for key, (start, end) in spans.items():
                # logits at position t predict token t+1
                lp = F.log_softmax(logits[j, start - 1 : end - 1].float(), dim=-1)
                target = ids[j, start:end].to(device)
                out[key] = -lp.gather(-1, target[:, None])[:, 0].cpu()
                if entropy:
                    ent[key] = -(lp.exp() * lp).sum(-1).cpu()
    return (out, ent) if entropy else out


def content_nll(model, device, items, pad_id, batch_size, entropy=False):
    """items: list of (key, Rendered). Returns {key: per-token NLL tensor over content} (and entropies)."""
    seqs = [(r.input_ids, {key: (r.start, r.end)}) for key, r in items]
    return spans_nll(model, device, seqs, pad_id, batch_size, entropy)


def save_token_nll(path, nll, meta, ent=None):
    """Per-token NLLs in long form: text_id, source, kind, condition, position (0 = first content token), nll
    (and entropy of the next-token distribution, if given)."""
    keys = list(nll)
    lengths = [len(nll[k]) for k in keys]
    cols = {
        "text_id": [k[0] for k in keys for _ in range(len(nll[k]))],
        "condition": [k[1] for k in keys for _ in range(len(nll[k]))],
        "position": np.concatenate([np.arange(n, dtype=np.int16) for n in lengths]),
        "nll": torch.cat([nll[k] for k in keys]).numpy().astype(np.float32),
    }
    if ent is not None:
        cols["entropy"] = torch.cat([ent[k] for k in keys]).numpy().astype(np.float32)
    df = pd.DataFrame(cols)
    for field in ("kind", "source"):  # inserted at 1 in turn, so source comes first
        df.insert(1, field, df.text_id.map({i: m[field] for i, m in meta.items()}))
    for c in ("text_id", "source", "kind", "condition"):
        df[c] = df[c].astype("category")
    df.to_parquet(path, index=False)


def main():
    config.family_from_argv()  # --family switches MODELS / STAGES / TEMPLATE_MODEL before the defaults below are read
    fam = config.family_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", help=f"one of {', '.join(config.FAMILIES)}; outputs go to results/<run>/<family>/core/scores")
    ap.add_argument("--run", help=f"run directory under results/ (with --family: default {config.DEFAULT_RUN})")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", help="label for the output file (default: stage name or model id)")
    ap.add_argument("--tokenizer", default=config.TEMPLATE_MODEL)
    ap.add_argument("--dataset", default=config.DATA_PATH)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--n-per-set", type=int, help="only the first N texts of each (source, kind) set (a quicker run, e.g. 100 or the "
                                                  "pilot's 30; the analysis takes the same --first N)")
    ap.add_argument("--batch-size", type=int, default=fam["batch_size"])  # fp32 7B on 48 GB: 16 runs short on long texts
    ap.add_argument("--offload", action=argparse.BooleanOptionalAction, default=fam["offload"],
                    help="fp32 weights split between GPU and CPU RAM (default: the family's setting)")
    ap.add_argument("--gpu-gib", type=int, default=fam["gpu_gib"], help="GiB of weights allowed on the GPU with --offload")
    ap.add_argument("--preceding-turn", action="store_true", default=config.PRECEDING_TURN)
    ap.add_argument("--out", help="default: config.core_dir('scores', run)")
    args = ap.parse_args()
    args.out = args.out or config.core_dir("scores", args.run)

    model_id, stage = resolve(args.model, args.stage)
    tok = load_tokenizer(args.tokenizer)
    rows = first_n_per_set(load_jsonl(args.dataset)[: args.limit], args.n_per_set)
    by_id = {row["id"]: row for row in rows}
    assert len(by_id) == len(rows), "text ids must be unique"
    items = []
    for row in rows:
        for cond, r in render_all(tok, row, args.preceding_turn).items():
            items.append(((row["id"], cond), r))

    model, device = load_model(model_id, torch.float32, offload=args.offload, gpu_gib=args.gpu_gib)
    print(f"{model_id}: {describe_placement(model)}, dtype {next(model.parameters()).dtype}", flush=True)
    n_vocab = model.get_input_embeddings().num_embeddings
    assert max(max(r.input_ids) for _, r in items) < n_vocab, "tokenizer ids exceed model vocab"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    nll, ent = content_nll(model, device, items, pad_id, args.batch_size, entropy=True)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{stage}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text_id", "source", "kind", "condition", "n_tokens", "mean_nll", "mean_nll_from1",
                    "mean_entropy", "mean_entropy_from1"])
        for key, _ in items:
            v, e = nll[key], ent[key]
            row = by_id[key[0]]
            w.writerow([key[0], row["source"], row["kind"], key[1], len(v), *(f"{x:.6f}" for x in
                        (v.mean().item(), v[1:].mean().item(), e.mean().item(), e[1:].mean().item()))])
    save_token_nll(os.path.join(args.out, f"{stage}_tokens.parquet"), nll, by_id, ent)
    print(f"{model_id}: wrote {len(items)} rows to {path} (+ per-token NLLs)")


if __name__ == "__main__":
    main()
