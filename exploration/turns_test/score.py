"""Score the turns-test conversations under designs A, B, C, tagged and plain (fp32, per-token NLL and entropy).

    python -m exploration.turns_test.score --family olmo3-7b --stage base
    -> exploration/turns_test/results/<family>/<stage>_tokens.parquet, <stage>.csv

Per conversation (P1 user, R1 reply, P2 user), each scored text is scored in these sequences:

    cell          scored  tagged sequence (family chat template, default system turn)  plain sequence (no role tags)
    u1            P1      [user P1]                                                     <doc> P1
    rA (design A) R1      [assistant R1]  (core's asst@asst)                            <doc> R1
    rB (design B) R1      [user P1][assistant R1]                                       <doc> P1 \\n\\n R1
    u2A           P2      [user P2]       (P2 alone, to separate context from turn 2)   <doc> P2
    u2C (design C) P2     [user P1][assistant R1][user P2]                              <doc> P1 \\n\\n R1 \\n\\n P2

Tagged single-turn cells use core.render.render (exactly core.score's user@user / asst@asst / *@plain); tagged
context cells use core.render.render_target (stored history; an assistant target opens as the generation prompt does,
Qwen3 with enable_thinking=False, i.e. the think-block-ON primary rendering). Plain = the same contents with every
role tag and the system turn removed, joined by a blank line, after the family's document-start token (as core's plain
control). Every content is tokenized on its own, and its tokens are asserted identical across all cells, so a frame
cost (tagged - plain) compares a text with itself. Reuses core.score.load_model / spans_nll; writes nothing outside
exploration/turns_test/.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from core import config
from core.render import doc_start_ids, encode, load_jsonl, load_tokenizer, render, render_target
from core.score import load_model, spans_nll

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "conversations.jsonl")
TEMPLATE_KW = {"olmo3-7b": {}, "qwen3-8b": {"enable_thinking": False}, "gemma4-31b": {}}
SEP = "\n\n"
CELLS = {"u1": "p1", "rA": "r1", "rB": "r1", "u2A": "p2", "u2C": "p2"}


def plain(tok, texts):
    """<doc> t1 SEP t2 ... ; returns ids and the span of the last text."""
    ids = list(doc_start_ids(tok))
    for i, t in enumerate(texts):
        if i:
            ids += encode(tok, SEP)
        start = len(ids)
        ids += encode(tok, t)
    return ids, (start, len(ids))


def sequences(tok, row, fam):
    kw = TEMPLATE_KW[fam]
    U = lambda t: {"role": "user", "content": t}
    A = lambda t: {"role": "assistant", "content": t}
    p1, r1, p2 = row["p1"], row["r1"], row["p2"]
    out = {}
    for cell, text, slot in (("u1", p1, "user"), ("rA", r1, "assistant"), ("u2A", p2, "user")):
        r = render(tok, text, slot)
        out[(cell, "tagged")] = (r.input_ids, (r.start, r.end))
        r = render(tok, text, None)
        out[(cell, "plain")] = (r.input_ids, (r.start, r.end))
    t = render_target(tok, [U(p1), A(r1)], kw)
    out[("rB", "tagged")] = (t.input_ids, (t.start, t.end))
    out[("rB", "plain")] = plain(tok, [p1, r1])
    t = render_target(tok, [U(p1), A(r1), U(p2)], kw)
    out[("u2C", "tagged")] = (t.input_ids, (t.start, t.end))
    out[("u2C", "plain")] = plain(tok, [p1, r1, p2])
    # the frame cost compares a text with itself: identical content tokens in every cell
    for key, (ids, (a, b)) in out.items():
        assert ids[a:b] == encode(tok, row[CELLS[key[0]]]), (row["id"], key)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", required=True, choices=list(TEMPLATE_KW))
    ap.add_argument("--stage", required=True)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    config.use_family(args.family)
    st = config.family_settings(args.family)
    model_id = st["stages"][args.stage]
    tok = load_tokenizer(st["template"])
    rows = load_jsonl(DATA)[: args.limit]
    seqs, meta = [], {}
    for row in rows:
        for (cell, lab), (ids, span) in sequences(tok, row, args.family).items():
            key = (row["id"], cell, lab)
            seqs.append((ids, {key: span}))
            meta[key] = row["source"]
    print(f"{args.family}/{args.stage}: {len(seqs)} sequences, max length {max(len(s[0]) for s in seqs)}", flush=True)
    t0 = time.time()
    model, device = load_model(model_id, torch.float32, offload=st["offload"], gpu_gib=st["gpu_gib"])
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    nll, ent = spans_nll(model, device, seqs, pad, args.batch_size or st["batch_size"], entropy=True)
    print(f"scored in {time.time() - t0:.0f}s (incl. load)", flush=True)

    out = os.path.join(args.out, args.family)
    os.makedirs(out, exist_ok=True)
    keys = list(nll)
    lens = [len(nll[k]) for k in keys]
    df = pd.DataFrame({
        "conv_id": np.repeat([k[0] for k in keys], lens),
        "source": np.repeat([meta[k] for k in keys], lens),
        "cell": np.repeat([k[1] for k in keys], lens),
        "labeling": np.repeat([k[2] for k in keys], lens),
        "position": np.concatenate([np.arange(n, dtype=np.int16) for n in lens]),
        "nll": torch.cat([nll[k] for k in keys]).numpy().astype(np.float32),
        "entropy": torch.cat([ent[k] for k in keys]).numpy().astype(np.float32),
    })
    df.to_parquet(os.path.join(out, f"{args.stage}_tokens.parquet"), index=False)
    d1 = df[df.position >= 1]
    g = d1.groupby(["conv_id", "source", "cell", "labeling"], observed=True)
    per = g.agg(n_tokens_from1=("nll", "size"), mean_nll_from1=("nll", "mean"), mean_entropy_from1=("entropy", "mean"))
    per.reset_index().to_csv(os.path.join(out, f"{args.stage}.csv"), index=False)
    with open(os.path.join(out, f"{args.stage}_provenance.txt"), "w") as f:
        f.write(f"model {model_id}\ntemplate {st['template']}\nfp32, batch {args.batch_size or st['batch_size']}\n"
                f"assistant_think_block {config.ASSISTANT_THINK_BLOCK}\nrows {len(rows)} sequences {len(seqs)}\n")
    print(f"wrote {out}/{args.stage}.csv and _tokens.parquet", flush=True)


if __name__ == "__main__":
    main()
