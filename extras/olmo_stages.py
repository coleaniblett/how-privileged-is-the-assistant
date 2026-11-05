"""Spec R4 (2026-11-06): which training stage does it? OLMo 3 7B base -> SFT -> DPO -> final. Exploratory, outside every verdict, NOT BLIND, descriptive (no verdict).
Part 1 (CPU): from saved atlas data, a table by stage for markdown_lists (old and strict layout), says_ai and denies_feelings in the Assistant, light roles and full role-play.
Part 2 (GPU, new generation for SFT and DPO): L2 (all) and at most 1,500 L3 rows without a harmless instruction (same rows and settings as spec L), and spec Q's 12 conditions + plain Assistant on the denies_feelings, markdown_lists and stock_phrases sets only.
Outputs under results/overnight_2026-11-05/R4/ (replies stay local).
    python -m extras.olmo_stages gen --stage sft        # GPU: L2 + L3 (capped) and Q subset
    python -m extras.olmo_stages gen-l23 --stage sft    # GPU: only L2/L3
    python -m extras.olmo_stages gen-q --stage sft      # GPU: only the Q subset
    python -m extras.olmo_stages analyze                # CPU
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from extras import role_says as rs
from extras import working_model_l as wl

OUT = "results/overnight_2026-11-05/R4"
FAM = "olmo3-7b"
QSETS = ["denies_feelings", "markdown_lists", "stock_phrases"]
CAP, SEED = 1500, 0


def gen_l23(stage):
    mat = wl.load()
    model, tok, device = wl.load_model_tok(FAM, stage)
    reqs, qs = mat["requests"], mat["l3"]["questions"]
    d = os.path.join(OUT, FAM)
    os.makedirs(d, exist_ok=True)
    for tag, max_new in (("L2", wl.L2_NEW), ("L3", wl.L3_NEW)):
        out = os.path.join(d, f"{stage}_{tag}.csv.gz")
        if os.path.exists(out):
            continue
        rows = []
        for wording in ("sys", "user"):
            for role in wl.roles_list():
                for ri, item in enumerate(reqs if tag == "L2" else qs):
                    for v in ([True, False] if tag == "L2" else [None]):
                        msgs = wl.l2_msgs(wording, role, item, v) if tag == "L2" else wl.l3_msgs(wording, role, item, mat, v)
                        for smp in range(wl.SAMPLES):
                            rows.append(dict(tag=tag, wording=wording, kind=wl.kind_of(role), role="Assistant" if role is None else role[1], item=ri, line=(v if tag == "L2" else None), instr=None, sample=smp, msgs=msgs))
        if tag == "L3" and len(rows) > CAP:
            keep = np.sort(np.random.default_rng(SEED).choice(len(rows), CAP, replace=False))
            rows = [rows[i] for i in keep]
        print(f"{FAM} {stage} {tag}: {len(rows)} generations", flush=True)
        t0 = time.time()
        rows = wl.generate_rows(model, tok, device, rows, max_new)
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "msgs"} for r in rows])
        df["instr"] = df.instr.fillna("none")
        df.to_csv(out, index=False)
        print(f"{FAM} {stage} {tag}: done in {time.time() - t0:.0f} s", flush=True)
    del model


def gen_q(stage):
    mat = rs.load()
    rows = [r for r in rs.all_rows(mat) if mat["questions"][r["qi"]]["set"] in QSETS]
    print(f"Q subset: {len(rows)} rows", flush=True)
    if not os.path.exists(os.path.join(OUT, FAM, f"{stage}_replies.csv.gz")):
        rs.gen(FAM, stage, rows=rows, out=OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "gen-l23", "gen-q", "analyze"])
    ap.add_argument("--stage", choices=["sft", "dpo"])
    a = ap.parse_args()
    if a.cmd in ("gen", "gen-l23"):
        gen_l23(a.stage)
    if a.cmd in ("gen", "gen-q"):
        gen_q(a.stage)
    if a.cmd == "analyze":
        analyze()


def analyze():
    raise SystemExit("analysis is added in the R4 report step")


if __name__ == "__main__":
    main()
