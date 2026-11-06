"""Keyword any-flag share split by the v2 judge label (2026-11-06; labels PROVISIONAL, not human-validated). Question: is the keyword rise in Llama and Apertus mid-turn switching to the assistant rather than assistant traits in the user's voice?
For each family, prefill set and stage (base, final = instruct): the share of samples with a keyword flag (user_traits.keyword_samples, role_cut) among samples labelled user_voice + user_voice_with_assistant_traits, switched_to_assistant, and degenerate (invalid excluded), the sample counts,
the user-voice share of samples, and the change from base in the user-voice row paired per prefill (95% bootstrap over prefills, user_traits.paired_change) where both stages have user-voice samples. Reads only; no GPU, no API.
    python -m measures.overgeneralization.user_traits_keyword_by_label
"""
import os

import numpy as np
import pandas as pd

from measures.overgeneralization import user_traits as ut

OUT = "results/user_turn_2026-11-06"
CLASSES = {"user voice (user_voice + user_voice_with_assistant_traits)": ut.USER_VOICE, "switched_to_assistant": ("switched_to_assistant",), "degenerate": ("degenerate",)}


def sets():
    """(family, set name, {stage: judged jsonl}, generations dir, source filter)"""
    out = []
    for fam in ["olmo3-7b", "qwen3-8b", "gemma4-31b", "llama3.1-8b", "apertus-8b"]:
        ex = f"results/capstone/{fam}/overgeneralization/judge_incomplete_exchanges"
        if os.path.exists(f"{ex}/judged_base.jsonl"):
            for src in ("oasst1", "ultrachat"):
                out.append((fam, f"{src} (exchange set)", {s: f"{ex}/judged_{s}.jsonl" for s in ("base", "instruct")}, f"results/capstone/{fam}/core/generations_incomplete_exchanges", src))
        vr = f"results/capstone/{fam}/vestigial_regression/judge_incomplete"
        if os.path.exists(f"{vr}/base.jsonl"):
            out.append((fam, "wildchat (200-prefill set)", {s: f"{vr}/{s}.jsonl" for s in ("base", "instruct")}, f"results/capstone/{fam}/core/generations_incomplete", ""))
        elif os.path.exists(f"{ex}/judged_p200_base.jsonl"):
            out.append((fam, "wildchat (200-prefill set)", {s: f"{ex}/judged_p200_{s}.jsonl" for s in ("base", "instruct")}, f"results/capstone/{fam}/core/generations_incomplete", ""))
    return out


FEATS = {"stock_phrase": "stock phrase", "markdown_heading": "heading", "markdown_bullet": "bullet", "markdown_bold": "bold"}


def matched_text(user_text, prefill):
    """the matched strings per fired feature (same rules as user_traits.keyword_counts)"""
    out = []
    if ut.STOCK.search(user_text):
        out.append("stock: " + " | ".join(m.group(0) for m in ut.STOCK.finditer(user_text)))
    for key, rx, nm in (("heading", ut.HEADING, "heading"), ("bullet", ut.BULLET, "bullet"), ("bold", ut.BOLD, "bold")):
        if not rx.search(prefill) and rx.search(user_text):
            out.append(f"{nm}: " + " | ".join(m.group(0).strip() for m in rx.finditer(user_text)))
    return " || ".join(out)


def main():
    rows, flag_rows, final_uv = [], [], []
    for fam, name, jp, gens, src in sets():
        if fam in ("llama3.1-8b", "apertus-8b"):
            from extras import trait_atlas_breadth  # noqa: F401  (Apertus turn markers)
        markers = ut.family_markers(fam)
        per = {}
        for st, p in jp.items():
            j = pd.DataFrame(ut.read_jsonl(p))[["prefill_id", "sample", "label"]]
            g = pd.DataFrame(ut.read_jsonl(os.path.join(gens, f"{st}.jsonl")))
            if src:
                j, g = j[j.prefill_id.str.contains(src)], g[g.prefill_id.str.contains(src)]
            kwf = ut.keyword_samples(g.to_dict("records"), markers, role_cut=True)
            kw = kwf[["prefill_id", "sample", "any_flag"]]
            d = j.merge(kw, on=["prefill_id", "sample"], how="inner")
            if st == "instruct":
                x = j.merge(kwf, on=["prefill_id", "sample"], how="inner")
                x = x[x.label.isin(ut.USER_VOICE) & x.any_flag]
                if len(x):
                    flag_rows.append(dict(family=fam, set=name, n=len(x), **{k: 100 * x[k].mean() for k in FEATS}))
                    if fam in ("llama3.1-8b", "apertus-8b"):
                        for r in x.itertuples():
                            final_uv.append(dict(family=fam, set=name, flag_types=", ".join(v for k, v in FEATS.items() if getattr(r, k)), matched_text=matched_text(r.user_text, r.prefill),
                                                 prefill=r.prefill, user_portion_260=" ".join(r.user_text.split())[:260]))
            per[st] = d[d.label != ut.INVALID]
        uv_base = {}
        for cname, labs in CLASSES.items():
            vals = {}
            for st, d in per.items():
                x = d[d.label.isin(labs)]
                vals[st] = (len(x), 100 * x.any_flag.mean() if len(x) else np.nan)
            row = dict(family=fam, prefill_set=name, cls=cname, n_base=vals["base"][0], flag_base=vals["base"][1], n_final=vals["instruct"][0], flag_final=vals["instruct"][1],
                       samples_base=len(per["base"]), samples_final=len(per["instruct"]))
            if cname.startswith("user voice"):
                row["uv_share_base"] = 100 * len(per["base"][per["base"].label.isin(labs)]) / len(per["base"])
                row["uv_share_final"] = 100 * len(per["instruct"][per["instruct"].label.isin(labs)]) / len(per["instruct"])
                tab = {st: d[d.label.isin(labs)].groupby("prefill_id").any_flag.mean().astype(float) for st, d in per.items()}
                r = ut.paired_change(tab, "base", "instruct", np.random.default_rng(ut.SEED), ut.N_BOOT)
                row.update(prefills_used=r["n"], dropped=r["dropped"], change=100 * r["mean"], lo=100 * r["lo"], hi=100 * r["hi"])
            rows.append(row)
    pd.DataFrame(flag_rows).to_csv(os.path.join(OUT, "user_traits_keyword_flag_types.csv"), index=False)
    F = pd.DataFrame(final_uv)
    samp = pd.concat([g.sample(min(14, len(g)), random_state=0) for _, g in F.groupby("family")]).sample(frac=1, random_state=0).reset_index(drop=True)
    samp.to_csv(os.path.join(OUT, "keyword_flagged_user_voice.csv"), index=False)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "user_traits_keyword_by_label.csv"), index=False)
    print(R.round(1).to_string())


if __name__ == "__main__":
    main()
