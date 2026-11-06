"""Two checks on the pilot's existing scores (no GPU, no new scoring): where Qwen's rise comes from, and OLMo in BPB.

    python -m measures.vestigial_regression.pilot.nll_arm.checks      # from the repo root, CPU only

Check 1: IT - base per slot for Qwen3-8B and Gemma 4 31B, paired over the same 30 texts per set, in nats/token
(positions 1+) and primary bits per byte, with each role-specific change decomposed. Verdict by the rule written in
report.md before this was run. Check 2: the frozen OLMo 3 7B held-out scores (results/precapstone/scores) in primary BPB, change
from OLMo base for SFT, DPO and Instruct, on the pilot's 90 texts (primary) and on all texts (secondary).
Primary BPB = summed NLL over content positions 1+ / ln 2 / UTF-8 bytes of the text decoded from content tokens 1+.
Writes checks.md and summary_tables.md next to this file.
"""

import os

import numpy as np
import pandas as pd

from core import config
from core.render import load_jsonl, load_tokenizer
from core.stats import bootstrap_ci
from measures.vestigial_regression.pilot.nll_arm.score import SETS, select

HERE = os.path.dirname(os.path.abspath(__file__))
N_BOOT = 5000
LN2 = np.log(2)
SLOTS = {"user": ["user", "asst", "plain"], "assistant": ["user", "asst", "plain"]}
EXTRA_ASSISTANT_SET = ("olmo-instruct", "assistant")  # secondary only (OLMo-written text), all-texts table


def ab(kind):
    return "user" if kind == "user" else "asst"


def load_model_scores(scores_dir, stage, tok, texts):
    """(nats frame, bpb frame): one row per (source, kind, condition, text_id)."""
    nats = pd.read_csv(os.path.join(scores_dir, f"{stage}.csv"))[["source", "kind", "condition", "text_id", "mean_nll_from1"]]
    tk = pd.read_parquet(os.path.join(scores_dir, f"{stage}_tokens.parquet"))
    tk = tk[tk.position >= 1]
    s = tk.groupby(["source", "kind", "condition", "text_id"], observed=True).nll.sum().reset_index()
    for c in ("source", "kind", "condition", "text_id"):
        s[c] = s[c].astype(str)
    s["bpb"] = s.nll / LN2 / s.text_id.map(texts)
    return nats, s[["source", "kind", "condition", "text_id", "bpb"]]


def text_bytes(tok, rows):
    """{id: UTF-8 bytes of the text decoded from content tokens 1+}."""
    out = {}
    for r in rows:
        ids = tok(r["text"], add_special_tokens=False)["input_ids"]
        out[r["id"]] = len(tok.decode(ids[1:], clean_up_tokenization_spaces=False).encode("utf-8"))
    return out


def vec(df, col, source, kind, slot, ids):
    d = df[(df.source == source) & (df.kind == kind) & (df.condition == f"{ab(kind)}@{slot}") & df.text_id.isin(ids)]
    return d.set_index("text_id")[col].sort_index()


def ci(x, rng):
    m, lo, hi = bootstrap_ci(np.asarray(x.dropna(), dtype=float), N_BOOT, rng)
    return m, lo, hi


def fmt(t, nd=3):
    m, lo, hi = t
    return f"{m:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def change_terms(base, it, ids, rng, source, kind, col):
    """Per-text IT - base for own slot, plain, and the role-specific term; returns {name: (m, lo, hi)}."""
    own_b, own_i = (vec(d, col, source, kind, ab(kind), ids) for d in (base, it))
    pl_b, pl_i = (vec(d, col, source, kind, "plain", ids) for d in (base, it))
    d_own, d_pl = own_i - own_b, pl_i - pl_b
    return {"d_own": ci(d_own, rng), "d_plain": ci(d_pl, rng), "rs": ci(d_own - d_pl, rng)}


def check1(fam, base, it, ids, rng, L):
    (bn, bb), (in_, ib) = base, it
    L += [f"### {fam}", ""]
    for source, kind in SETS:
        slots = ["user", "asst", "plain"]
        L += [f"**{source} {kind}**", "",
              "| slot | nats base | nats IT | nats IT − base | BPB base | BPB IT | BPB IT − base |", "|---|---|---|---|---|---|---|"]
        for s in slots:
            row = [f"{ab(kind)}@{s}"]
            for df_b, df_i, col in ((bn, in_, "mean_nll_from1"), (bb, ib, "bpb")):
                vb, vi = vec(df_b, col, source, kind, s, ids), vec(df_i, col, source, kind, s, ids)
                row += [f"{vb.mean():.3f}", f"{vi.mean():.3f}", fmt(ci(vi - vb, rng))]
            L.append("| " + " | ".join(row) + " |")
        L += ["", "Decomposition of the role-specific change (own slot − plain):", "",
              "| unit | Δ(own slot) | Δ(plain) | role-specific change = Δ(own) − Δ(plain) |", "|---|---|---|---|"]
        for unit, (df_b, df_i, col) in (("nats/token", (bn, in_, "mean_nll_from1")), ("bits/byte", (bb, ib, "bpb"))):
            t = change_terms(df_b, df_i, ids, rng, source, kind, col)
            L.append(f"| {unit} | {fmt(t['d_own'])} | {fmt(t['d_plain'])} | {fmt(t['rs'])} |")
        L.append("")


def verdict_qwen(base, it, ids, rng):
    ok_plain, ok_real, detail = [], [], []
    for source in ("wildchat", "oasst1"):
        t = change_terms(base, it, ids, rng, source, "user", "mean_nll_from1")
        (um, ulo, uhi), (pm, plo, phi) = t["d_own"], t["d_plain"]
        plain_fell = pm <= -0.05 and phi < 0
        user_not_rising = um < 0.05 or ulo <= 0
        real = um >= 0.05 and ulo > 0
        ok_plain.append(bool(plain_fell and user_not_rising))
        ok_real.append(bool(real))
        detail.append(f"{source}: Δ(user@user) {fmt(t['d_own'])}, Δ(user@plain) {fmt(t['d_plain'])}")
    if all(ok_plain):
        v = "plain-control effect"
    elif all(ok_real):
        v = "real user-slot rise"
    else:
        v = "mixed"
    return v, detail, ok_plain, ok_real


def olmo_terms(stages, ids, rng, col_key):
    """{stage: {(source, kind): terms}} for IT-stage minus base."""
    out = {}
    for st in ("sft", "dpo", "instruct"):
        out[st] = {}
        for source, kind in SETS + [EXTRA_ASSISTANT_SET]:
            if len(vec(stages[st][col_key], "bpb" if col_key == 1 else "mean_nll_from1", source, kind, "plain", ids)) == 0:
                continue
            col = "bpb" if col_key == 1 else "mean_nll_from1"
            out[st][(source, kind)] = change_terms(stages["base"][col_key], stages[st][col_key], ids, rng, source, kind, col)
    return out


def summary_row(label, terms):
    cells = []
    for source in ("wildchat", "oasst1"):
        t = terms[(source, "user")]
        cells += [fmt(t["d_own"], 2), fmt(t["d_plain"], 2), fmt(t["rs"], 2)]
    cells.append(fmt(terms[("wildchat", "assistant")]["rs"], 2))
    return f"| {label} | " + " | ".join(cells) + " |"


SUMMARY_HEAD = ["| model (IT − own base) | wildchat Δ(user@user) | wildchat Δ(user@plain) | wildchat user role-specific "
                "| oasst1 Δ(user@user) | oasst1 Δ(user@plain) | oasst1 user role-specific | assistant role-specific (asst@asst − asst@plain) |",
                "|---|---|---|---|---|---|---|---|"]


def main():
    rng = np.random.default_rng(config.SEED)
    all_rows = load_jsonl(config.DATA_PATH)
    ids90 = sorted(r["id"] for r in select(all_rows, 30))
    L = ["# Checks on the existing pilot scores", "",
         "No GPU, no new scoring. Positions 1+ are primary; paired bootstrap over texts (5000 resamples, "
         f"seed {config.SEED}); BPB is primary BPB (summed NLL over positions 1+ / ln 2 / UTF-8 bytes of the text "
         "decoded from content tokens 1+). Each family's change is from its own base.", "",
         "## Check 1: where does Qwen's rise come from?", "",
         "Same 30 texts per set for both families (the pilot's 90 texts).", ""]

    fam = {}
    ids90_set = set(ids90)
    for name, tokname, d in (("Qwen3-8B", "Qwen/Qwen3-8B", os.path.join(HERE, "scores_qwen")),
                              ("Gemma 4 31B", "google/gemma-4-31B-it", os.path.join(HERE, "scores"))):
        tok = load_tokenizer(tokname)
        nb = text_bytes(tok, [r for r in all_rows if r["id"] in ids90_set])
        base = load_model_scores(d, "base", tok, nb)
        it = load_model_scores(d, "instruct", tok, nb)
        fam[name] = (base, it)
        check1(name, base, it, ids90, rng, L)

    v, detail, okp, okr = verdict_qwen(fam["Qwen3-8B"][0][0], fam["Qwen3-8B"][1][0], ids90, rng)
    L += ["### Check 1 verdict (rule in report.md, applied to Qwen3-8B in nats/token)", "", *[f"- {x}" for x in detail], "",
          f"Plain-control effect in both sources: {all(okp)} {okp}. Real user-slot rise in both sources: {all(okr)} {okr}. "
          f"**Verdict: {v}.**", ""]

    # Check 2: OLMo, BPB from the frozen held-out scores.
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    nb_all = text_bytes(tok, all_rows)
    stages = {}
    for st in ("base", "sft", "dpo", "instruct"):
        stages[st] = load_model_scores(os.path.join(config.RESULTS_ROOT, "scores"), st, tok, nb_all)
    all_ids = sorted(r["id"] for r in all_rows)
    L += ["## Check 2: OLMo 3 7B in bits per byte", "",
          "Frozen held-out run (`results/precapstone/scores/`). The text_ids match `data/texts.jsonl` (1190 of 1190 in every stage; "
          "per-text token counts equal `n_tokens` in the data). Change from OLMo base, paired over texts, bits/byte.", ""]
    terms = {}
    for tag, ids, title in (("a", ids90, "(a) the pilot's 90 texts (primary for the cross-family comparison)"),
                            ("b", all_ids, "(b) all OLMo texts (secondary)")):
        terms[tag] = olmo_terms(stages, ids, rng, 1)
        L += [f"### {title}", "", *SUMMARY_HEAD]
        for st, label in (("sft", "OLMo SFT"), ("dpo", "OLMo DPO"), ("instruct", "OLMo Instruct")):
            L.append(summary_row(label, terms[tag][st]))
        if tag == "b":
            L += ["", "OLMo-written assistant replies (`olmo-instruct`, secondary): assistant role-specific term, IT − base:", ""]
            for st in ("sft", "dpo", "instruct"):
                if EXTRA_ASSISTANT_SET in terms[tag][st]:
                    L.append(f"- {st}: {fmt(terms[tag][st][EXTRA_ASSISTANT_SET]['rs'])}")
        L.append("")

    # Summary tables for report.md (bits per byte, 90 shared texts).
    fam_terms = {}
    for name, (base, it) in fam.items():
        fam_terms[name] = {k: change_terms(base[1], it[1], ids90, rng, k[0], k[1], "bpb") for k in SETS}
    S = ["### Summary: bits per byte, change from each family's own base, 90 shared texts", "", *SUMMARY_HEAD,
         summary_row("OLMo 3 7B Instruct", terms["a"]["instruct"]),
         summary_row("Qwen3-8B", fam_terms["Qwen3-8B"]),
         summary_row("Gemma 4 31B IT", fam_terms["Gemma 4 31B"]), "",
         "OLMo per stage (same 90 texts, bits/byte, change from OLMo base):", "", *SUMMARY_HEAD,
         *[summary_row(lab, terms["a"][st]) for st, lab in (("sft", "OLMo SFT"), ("dpo", "OLMo DPO"), ("instruct", "OLMo Instruct"))], "",
         "Each family's change is from its own base; only bits per byte is compared across families (per-token nats are "
         "not: the tokenizers differ). Cells are mean change [95% paired bootstrap CI over texts]. All values are "
         "`condition − condition` differences, so the OLMo, Qwen and Gemma plain controls (`<|endoftext|>` / `<bos>` document "
         "starts) enter only through each family's own base.", ""]
    L += S
    with open(os.path.join(HERE, "checks.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(HERE, "summary_tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(S) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
