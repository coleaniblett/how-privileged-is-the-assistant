"""Incomplete-turn prefills for the exchange set's oasst1 and UltraChat user turns (2026-11-04; plan in docs/decisions.md).
    python -m extras.exchange_prefills count     # eligible user turns per corpus (nothing written)
    python -m extras.exchange_prefills build     # data/prefill_incomplete_exchanges.jsonl: 200 per corpus
Exactly the rule and seed logic of data/prefill_incomplete_200.jsonl: core.prepare_data.build_incomplete_prefill (the cut is chosen from the user turn with
random.Random(f"{seed}-{row id}"), seed 0; first eligible turns in file order), applied to the user turn of each row of data/exchanges.jsonl (row id = exchange_id).
WildChat is already covered by the existing 200. Rows keep the build function's fields plus corpus; ids are incomplete-<corpus>-NNN.
"""
import argparse
import json

from core.prepare_data import build_incomplete_prefill

EXCHANGES = "data/exchanges.jsonl"
OUT = "data/prefill_incomplete_exchanges.jsonl"
CORPORA = ["oasst1", "ultrachat"]
N, SEED = 200, 0


def user_rows(path=EXCHANGES):
    return [dict(source=r["corpus"], kind="user", text=r["user"], id=r["exchange_id"]) for r in map(json.loads, open(path, encoding="utf-8"))]


def build(rows, corpus, n=N, seed=SEED):
    out = []
    for i, r in enumerate(build_incomplete_prefill(rows, corpus, n, seed)):
        out.append({**r, "id": f"incomplete-{corpus}-{i:03d}", "corpus": corpus})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["count", "build"])
    a = ap.parse_args()
    rows = user_rows()
    for c in CORPORA:
        total = sum(1 for r in rows if r["source"] == c)
        eligible = len(build(rows, c, n=10 ** 9))
        print(f"{c}: {total} exchanges, {eligible} eligible for an incomplete-turn cut")
    if a.cmd == "build":
        out = []
        for c in CORPORA:
            part = build(rows, c)
            if len(part) < N:
                raise SystemExit(f"{c}: only {len(part)} eligible user turns, need {N}; the rule is not loosened")
            out += part
        with open(OUT, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(out)} rows to {OUT}")


if __name__ == "__main__":
    main()
