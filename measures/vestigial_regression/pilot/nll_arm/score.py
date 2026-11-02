"""Per-token NLL of the README's slot conditions for Gemma 4 31B, in fp32 with CPU offload.

    python -m measures.vestigial_regression.pilot.nll_arm.score --model google/gemma-4-31B-it --stage instruct

Same quantities and file format as core/score.py (scores/<stage>.csv and <stage>_tokens.parquet), computed with
core's own rendering and span-NLL code, imported unchanged. What differs: core.score moves the whole model to the GPU,
which fp32 31B (~125 GB) cannot do on a 48 GB card, so the weights are placed with device_map="auto": as many layers
as fit on the GPU, the rest in CPU RAM and streamed in each forward pass. Scoring stays in fp32 (AGENTS.md); offload
changes where weights live, not the arithmetic. The tokenizer/template is Gemma 4 31B-it's for every stage.

--n-per-set takes the first N texts of each (source, kind) set in data/texts.jsonl (default: all of wildchat user,
oasst1 user, wildchat assistant; the olmo-instruct replies are OLMo-written and left out).
"""

import argparse
import csv
import os
import time

import torch
from transformers import AutoModelForCausalLM

from core import config
from core.render import load_jsonl, load_tokenizer, render_all
from core.score import content_nll, save_token_nll

TEMPLATE = "google/gemma-4-31B-it"
SETS = [("wildchat", "user"), ("oasst1", "user"), ("wildchat", "assistant")]


def select(rows, n):
    out = []
    for source, kind in SETS:
        s = [r for r in rows if r["source"] == source and r["kind"] == kind]
        out += s[:n] if n else s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--tokenizer", default=TEMPLATE)
    ap.add_argument("--n-per-set", type=int)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--gpu-gib", type=int, default=34, help="GiB of weights allowed on the GPU (rest: activations)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "scores"))
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    rows = select(load_jsonl(config.DATA_PATH), args.n_per_set)
    by_id = {r["id"]: r for r in rows}
    items = [((r["id"], c), v) for r in rows for c, v in render_all(tok, r).items()]
    print(f"{len(rows)} texts, {len(items)} sequences", flush=True)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, device_map="auto",
        max_memory={0: f"{args.gpu_gib}GiB", "cpu": "900GiB"}).eval()
    n_gpu = sum(1 for v in model.hf_device_map.values() if v in (0, "cuda:0"))
    print(f"loaded in {time.time() - t0:.0f}s; {n_gpu}/{len(model.hf_device_map)} modules on the GPU, "
          f"dtype {next(model.parameters()).dtype}", flush=True)
    assert max(max(r.input_ids) for _, r in items) < model.get_input_embeddings().num_embeddings
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    t0 = time.time()
    nll, ent = content_nll(model, "cuda:0", items, pad_id, args.batch_size, entropy=True)
    print(f"scored {len(items)} sequences in {time.time() - t0:.0f}s", flush=True)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.stage}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text_id", "source", "kind", "condition", "n_tokens", "mean_nll", "mean_nll_from1",
                    "mean_entropy", "mean_entropy_from1"])
        for key, _ in items:
            v, e, row = nll[key], ent[key], by_id[key[0]]
            w.writerow([key[0], row["source"], row["kind"], key[1], len(v), *(f"{x:.6f}" for x in
                        (v.mean().item(), v[1:].mean().item(), e.mean().item(), e[1:].mean().item()))])
    save_token_nll(os.path.join(args.out, f"{args.stage}_tokens.parquet"), nll, by_id, ent)
    print(f"wrote {len(items)} rows to {path}")


if __name__ == "__main__":
    main()
