"""Sample continuations of open user turns for one model, with flags.

    python -m exploration.shiller_user_turns.generate --model qwen3-8b
    python -m exploration.shiller_user_turns.generate --model gemma4-31b-it --precision int8

Writes <out>/<model>.jsonl (one row per sample, flags included) and <out>/<model>_meta.json (verified
model record, precision, generation settings, template options, turn markers, and the exact rendered
prompt of every prompt id). Rows are written as each prompt finishes, to <model>.jsonl.partial, and
renamed when the model is done.

Generation: sampling at temperature 0.7, top_p 1.0, no top-k, 5 samples per prompt, up to 200 new tokens,
seed = --seed + crc32(prompt id) per prompt (the same seeds for every model). The model's own generation
config (Gemma: top_k 64; Qwen: top_k 20, top_p 0.95) is replaced, not merged. Stops only at end of text
(core.generate.stop_ids), so closed turns and handoffs stay visible.
"""

import argparse
import json
import os
import platform
import time
import zlib

import torch
import transformers
from transformers import AutoModelForCausalLM, GenerationConfig

from core.generate import stop_ids
from core.render import load_tokenizer, special_strings
from exploration.shiller_user_turns import flags as F
from exploration.shiller_user_turns import hub, models, prompts, render

OUT = "results/exploration/shiller_user_turns"
INT8_FRACTION = 0.55  # int8 weights ~ half of bf16, plus fp16 outliers / unquantized lm_head and vision tower
GPU_MARGIN_GB = 4


def precision_for(choice, weights_gb):
    if not torch.cuda.is_available():
        if choice == "int8":
            raise SystemExit("int8 (bitsandbytes) needs a GPU")
        return "fp32"
    if choice != "auto":
        return choice
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return "bf16" if weights_gb + GPU_MARGIN_GB <= total else "int8"


PRECISION_LABEL = {"bf16": "bf16", "int8": "int8 (bitsandbytes LLM.int8, bf16 compute)", "fp32": "fp32 (CPU)"}


def check_gpu(precision, weights_gb):
    if precision == "fp32" or not weights_gb:
        return
    need = weights_gb * (INT8_FRACTION if precision == "int8" else 1.0) + GPU_MARGIN_GB
    free, total = (x / 1e9 for x in torch.cuda.mem_get_info())
    if free < need:
        raise SystemExit(f"GPU has {free:.1f} of {total:.1f} GB free; {precision} load needs ~{need:.0f} GB. "
                         "Another process is probably using it (nvidia-smi).")


def load_model(repo, precision):
    if precision == "fp32":
        return AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).eval()
    if precision == "int8":
        from transformers import BitsAndBytesConfig  # needs the `quant` extra: uv sync --extra quant
        q = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModelForCausalLM.from_pretrained(repo, quantization_config=q, dtype=torch.bfloat16,
                                                    device_map="cuda:0").eval()
    return AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="cuda:0").eval()


def prompt_seed(base, prompt_id):
    return base + zlib.crc32(prompt_id.encode()) % (2**31)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=f"one of {', '.join(models.MODELS)}")
    ap.add_argument("--repo", help="ad-hoc model (tests): weights repo; --model is then just the output name")
    ap.add_argument("--template", help="ad-hoc model: tokenizer/template repo (default --repo)")
    ap.add_argument("--precision", default="auto", choices=["auto", "bf16", "int8", "fp32"],
                    help="auto: bf16 if it fits on the GPU, else int8; fp32 on CPU")
    ap.add_argument("--conditions", default=",".join(prompts.CONDITIONS))
    ap.add_argument("--limit", type=int, help="prompts per condition (smoke tests)")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    if args.repo:
        record = dict(key=args.model, repo=args.repo, template_repo=args.template or args.repo, role="ad hoc",
                      template_kwargs={}, sha=None, verified=False)
        weights = 0.0
    else:
        record = dict(models.verify(args.model), verified=True)  # HF API checks before anything is downloaded
        weights = record["weights_gb"]
    tok = load_tokenizer(record["template_repo"])
    kw = record["template_kwargs"]
    markers = render.turn_markers(tok, kw)
    stops = stop_ids(tok)
    stop_tokens = tok.convert_ids_to_tokens(stops)
    marker_ids = set()  # single-token turn markers, to cut token ids where the user turn ends
    for m in (markers["end_of_turn"], markers["turn_open"]):
        enc = tok.encode(m, add_special_tokens=False)
        if len(enc) == 1:
            marker_ids.add(enc[0])
    specials = sorted(special_strings(tok))

    # Render and verify every prompt before loading weights.
    items = []
    for cond in args.conditions.split(","):
        for p in prompts.load(cond)[: args.limit]:
            items.append((cond, p, render.render_open_user_turn(tok, p["prefill"], kw)))

    precision = precision_for(args.precision, weights)
    if not args.repo:
        hub.check_disk(record["repo"])
    check_gpu(precision, weights)
    t0 = time.time()
    model = load_model(record["repo"], precision)
    device = next(model.parameters()).device
    pad = tok.pad_token_id if tok.pad_token_id is not None else stops[0]
    gen_cfg = GenerationConfig(do_sample=True, temperature=args.temperature, top_p=args.top_p, top_k=0,
                               max_new_tokens=args.max_new_tokens, num_return_sequences=args.samples,
                               eos_token_id=stops, pad_token_id=pad)
    model.generation_config = gen_cfg

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.model}.jsonl")
    n = 0
    with open(path + ".partial", "w", encoding="utf-8") as f:
        for cond, p, turn in items:
            seed = prompt_seed(args.seed, p["id"])
            torch.manual_seed(seed)
            ids = torch.tensor([turn.input_ids], device=device)
            with torch.no_grad():
                out = model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=gen_cfg)
            for s, seq in enumerate(out[:, ids.shape[1]:].tolist()):
                cut = next((i for i, t in enumerate(seq) if t in stops), None)
                if cut is not None:  # padding only ever follows a stop token
                    seq = seq[: cut + 1]
                text = tok.decode(seq, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                up, fl, found = F.flags(text, seq, markers, stop_tokens, set(stops), marker_ids, specials,
                                        cut is not None)
                f.write(json.dumps(dict(
                    model=args.model, repo=record["repo"], role=record["role"], precision=PRECISION_LABEL[precision],
                    condition=cond, prompt_id=p["id"], sample=s, seed=seed, prefill=p["prefill"],
                    rendered_prompt=turn.text, completion=text, completion_ids=seq, n_new_tokens=len(seq),
                    stopped_on_eos=cut is not None, user_portion=up, special_tokens_found=found, **fl,
                ), ensure_ascii=False) + "\n")
                n += 1
            f.flush()
            print(f"{args.model} {cond} {p['id']}: {n} samples, {time.time() - t0:.0f}s", flush=True)
    os.replace(path + ".partial", path)

    meta = dict(
        model=record, precision=PRECISION_LABEL[precision], device=str(device),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        generation=dict(temperature=args.temperature, top_p=args.top_p, top_k=None, samples=args.samples,
                        max_new_tokens=args.max_new_tokens, seed=args.seed, seed_rule="seed + crc32(prompt_id)",
                        stop_ids=stops, stop_tokens=stop_tokens, config=gen_cfg.to_dict()),
        template_kwargs=kw, markers=markers, marker_ids=sorted(marker_ids), specials=specials,
        rendered_prompts={p["id"]: dict(condition=c, text=t.text) for c, p, t in items},
        versions=dict(torch=torch.__version__, transformers=transformers.__version__, python=platform.python_version()),
        seconds=round(time.time() - t0),
    )
    with open(os.path.join(args.out, f"{args.model}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"{record['repo']} ({PRECISION_LABEL[precision]}): wrote {n} samples to {path}")


if __name__ == "__main__":
    main()
