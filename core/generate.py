"""Sample continuations of prefilled (unterminated) user turns, for one model.

    python -m core.generate --model instruct
    python -m core.generate --family gemma4-31b --model instruct     # -> results/<run>/<family>/core/generations

Writes <out>/<stage>.jsonl, one line per sample: prefill_id, sample, prefill, continuation (decoded
with special tokens kept, up to and including the stop token if one was generated) and
stopped_on_eos, and continuation_ids (the generated token ids, for the no-space token-loop flag). Generation stops only at the end-of-document token, so a closed user turn and any
handoff to the assistant stay visible in the continuation. Flags on the samples are a measure's job
(measures/vestigial_regression/flags.py reads these files); nothing here depends on a measure.
"""

import argparse
import json
import os

import torch

from core import config
from core.render import load_jsonl, load_tokenizer, render_prefill
from core.score import load_model, resolve


def stop_ids(tok):
    """End-of-document token, the one stop token for every stage. Post-trained checkpoints
    also stop on <|im_end|> by default, which would hide any handoff to the assistant."""
    vocab = tok.get_vocab()
    return [vocab["<|endoftext|>"]] if "<|endoftext|>" in vocab else [tok.eos_token_id]


def load_generation_model(model_id, precision):
    """precision "bf16": core's default (bf16 on GPU, fp32 on CPU); "int8": bitsandbytes LLM.int8 on the GPU (the
    63 GB Gemma 4 31B does not fit a 48 GB card in bf16; needs `uv sync --extra quant`)."""
    if precision == "int8":
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise SystemExit("int8 (bitsandbytes) needs a GPU")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=BitsAndBytesConfig(load_in_8bit=True), dtype=torch.bfloat16,
            device_map="cuda:0").eval()
        return model, "cuda:0"
    return load_model(model_id)


def main():
    config.family_from_argv()  # --family switches MODELS / STAGES / TEMPLATE_MODEL before the defaults below are read
    fam = config.family_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", help=f"one of {', '.join(config.FAMILIES)}; outputs go to results/<run>/<family>/core/generations")
    ap.add_argument("--run", help=f"run directory under results/ (with --family: default {config.DEFAULT_RUN})")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage")
    ap.add_argument("--tokenizer", default=config.TEMPLATE_MODEL)
    ap.add_argument("--prefill", default=config.PREFILL_PATH)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--samples", type=int, default=config.GEN_SAMPLES)
    ap.add_argument("--max-new-tokens", type=int, default=config.GEN_MAX_NEW_TOKENS)
    ap.add_argument("--temperature", type=float, default=config.GEN_TEMPERATURE)
    ap.add_argument("--precision", choices=["bf16", "int8"], default=fam["gen_precision"],
                    help="generation precision (default: the family's setting)")
    ap.add_argument("--preceding-turn", action="store_true", default=config.PRECEDING_TURN)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", help="default: config.core_dir('generations', run)")
    args = ap.parse_args()
    args.out = args.out or config.core_dir("generations", args.run)

    model_id, stage = resolve(args.model, args.stage)
    tok = load_tokenizer(args.tokenizer)
    model, device = load_generation_model(model_id, args.precision)
    eos_ids = stop_ids(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    torch.manual_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    records = []
    for item in load_jsonl(args.prefill)[: args.limit]:
        ids = torch.tensor([render_prefill(tok, item["prefill"], args.preceding_turn)], device=device)
        gen = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            do_sample=True,
            temperature=args.temperature,
            top_p=1.0,
            top_k=0,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.samples,
            eos_token_id=eos_ids,
            pad_token_id=pad_id,
        )
        for s, seq in enumerate(gen[:, ids.shape[1] :].tolist()):
            cut = next((i for i, t in enumerate(seq) if t in eos_ids), None)
            stopped = cut is not None
            text = tok.decode(seq[: cut + 1] if stopped else seq, skip_special_tokens=False)
            records.append(dict(prefill_id=item["id"], sample=s, prefill=item["prefill"], continuation=text,
                                stopped_on_eos=stopped, continuation_ids=seq[: cut + 1] if stopped else seq))

    path = os.path.join(args.out, f"{stage}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{model_id}: wrote {len(records)} samples to {path}")


if __name__ == "__main__":
    main()
