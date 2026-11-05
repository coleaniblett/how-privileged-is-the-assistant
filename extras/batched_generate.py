"""Batched incomplete-turn generation (same settings and record format as core.generate; written 2026-11-05 because core.generate decodes one prefill at a time, which
takes hours for Gemma 4 31B in int8). All prefills are left-padded and decoded together (BATCH prefills x SAMPLES samples), same sampling (T, top_p=1, top_k=0, 120 new tokens),
family generation precision (int8 for Gemma). Progress is printed. Output: <out>/<stage>.jsonl with continuation_ids.
    python -m extras.batched_generate --family gemma4-31b --model instruct --prefill data/prefill_incomplete_exchanges.jsonl --out results/capstone/gemma4-31b/core/generations_incomplete_exchanges
"""
import argparse
import json
import os
import time

import torch

from core import config
from core.generate import load_generation_model, resolve, stop_ids
from core.render import load_jsonl, load_tokenizer, render_prefill


def main():
    config.family_from_argv()
    fam = config.family_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--family")
    ap.add_argument("--run")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage")
    ap.add_argument("--prefill", required=True)
    ap.add_argument("--samples", type=int, default=config.GEN_SAMPLES)
    ap.add_argument("--max-new-tokens", type=int, default=config.GEN_MAX_NEW_TOKENS)
    ap.add_argument("--temperature", type=float, default=config.GEN_TEMPERATURE)
    ap.add_argument("--precision", choices=["bf16", "int8"], default=fam["gen_precision"])
    ap.add_argument("--batch", type=int, default=24, help="prefills per batch (x samples sequences)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tokenizer", default=config.TEMPLATE_MODEL)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model_id, stage = resolve(a.model, a.stage)
    tok = load_tokenizer(a.tokenizer)
    model, device = load_generation_model(model_id, a.precision)
    eos = stop_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    torch.manual_seed(a.seed)
    items = load_jsonl(a.prefill)[: a.limit]
    enc = [render_prefill(tok, it["prefill"], False) for it in items]
    order = sorted(range(len(items)), key=lambda i: len(enc[i]))
    recs = {}
    t0 = time.time()
    for s in range(0, len(order), a.batch):
        idx = order[s:s + a.batch]
        m = max(len(enc[i]) for i in idx)
        ids = torch.tensor([[pad] * (m - len(enc[i])) + enc[i] for i in idx], device=device)
        att = torch.tensor([[0] * (m - len(enc[i])) + [1] * len(enc[i]) for i in idx], device=device)
        with torch.no_grad():
            gen = model.generate(ids, attention_mask=att, do_sample=True, temperature=a.temperature, top_p=1.0, top_k=0, max_new_tokens=a.max_new_tokens,
                                 num_return_sequences=a.samples, eos_token_id=eos, pad_token_id=pad)
        for r, i in enumerate(idx):
            for k in range(a.samples):
                seq = gen[r * a.samples + k, m:].tolist()
                cut = next((j for j, t in enumerate(seq) if t in eos), None)
                stopped = cut is not None
                seq = seq[: cut + 1] if stopped else seq
                recs[(i, k)] = dict(prefill_id=items[i]["id"], sample=k, prefill=items[i]["prefill"], continuation=tok.decode(seq, skip_special_tokens=False),
                                    stopped_on_eos=stopped, continuation_ids=seq)
        done = min(s + a.batch, len(order))
        print(f"{done}/{len(order)} prefills, {time.time() - t0:.0f} s", flush=True)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{stage}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for key in sorted(recs):
            f.write(json.dumps(recs[key], ensure_ascii=False) + "\n")
    print(f"{model_id}: wrote {len(recs)} samples to {path}")


if __name__ == "__main__":
    main()
