"""Spec S Stage 1b: choose 48 roles and 96 traits (blind to any behaviour), seed 0, and make the discovery / confirmation splits.
python -m extras.big_atlas_choose choose   -> data/big_atlas_splits.json (IDs, strata, split, seed, distance; committed) + results/big_atlas/stage1/chosen_{roles,traits}.csv
Inputs: results/big_atlas/stage1/map_roles.csv, map_traits.csv (from extras.big_atlas_map analyze) and the blind type consensus in BA_SRC/ratings (private).
Rule (docs/decisions.md 2026-11-06): AA roles 28 = type quotas (AI/machine: all available, up to 4; the rest equal over human, non-human, fictional/mythic, human takes any shortfall),
within a type spread over distance quartiles (cos distance at layer 33, quartiles over all 275); Zheng 12 = 6 relationship (family, romantic, social) + 6 expertise (mmlu occupations),
two per distance tercile; 8 anchors. Traits: 20 atlas + 4 Persona Vectors + 72 of the 240 (type quotas of 14 over style, stance, emotion, social, epistemic; self-image all that exist),
within a type spread over terciles of the cosine with the Assistant axis (layer 33). Splits are made within strata by assigning each member to the half with fewer members (ties random).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from extras import big_atlas as ba

S1 = os.path.join(ba.OUT, "stage1")
RAT = os.path.join(ba.SRC, "ratings")
ATLAS_TRAITS = ["refusal_borderline", "safety_caveats", "moralizing", "markdown_lists", "length", "formal_register", "stock_phrases", "hedging_unanswerable", "corrects_false_premise",
                "fabricates_vs_admits", "flattery", "agrees_wrong_opinion", "closing_offer", "clarifying_question", "says_ai", "denies_feelings", "declines_opinion", "arithmetic",
                "multistep_reasoning", "factual_accuracy"]
PV_TRAITS = ["pv:apathetic", "pv:hallucinating", "pv:humorous", "pv:impolite"]
ANCHOR_ROLES = {"anchor:a ferry captain": "light job", "anchor:a village baker": "light job", "anchor:a long-haul trucker": "light job",
                "anchor:Ottoline Brandt": "strong character", "anchor:Dmitri Volkov": "strong character", "anchor:Marguerite Oyelaran": "strong character",
                "anchor:Hannah Reed": "L2 thin role", "anchor:Captain Rook": "L2 vivid role"}
NAMED = ["denies_feelings", "says_ai", "markdown_lists", "stock_phrases", "corrects_false_premise"]


def spread(items, key, nbins, quota, rng):
    """Pick `quota` of items (DataFrame rows) spread over the bins of column `key` (round-robin from a random bin; empty bins skipped)."""
    bins = [g.sample(frac=1, random_state=int(rng.integers(1 << 30))) for _, g in items.groupby(key) if len(g)]
    order = list(rng.permutation(len(bins)))
    pick, i = [], 0
    while len(pick) < min(quota, len(items)):
        g = bins[order[i % len(bins)]]
        done = sum(1 for p in pick if p in set(g.index))
        if done < len(g):
            pick.append(g.index[done])
        i += 1
        if i > 10000:
            break
    return items.loc[pick]


def split_within(df, strata_cols, rng):
    half = pd.Series("", index=df.index)
    cnt = {"discovery": 0, "confirmation": 0}
    cells = [g for _, g in df.groupby(strata_cols)]
    for gi in rng.permutation(len(cells)):
        g = cells[gi].sample(frac=1, random_state=int(rng.integers(1 << 30)))
        for ix in g.index:
            a, b = cnt["discovery"], cnt["confirmation"]
            h = "discovery" if a < b else "confirmation" if b < a else ("discovery" if rng.random() < 0.5 else "confirmation")
            half[ix] = h; cnt[h] += 1
    return half


def choose():
    rng = np.random.default_rng(ba.SEED)
    mr = pd.read_csv(os.path.join(S1, "map_roles.csv")); mt = pd.read_csv(os.path.join(S1, "map_traits.csv"))
    rtype = pd.read_csv(os.path.join(RAT, "consensus_roles.csv")).set_index("name").label
    ttype = pd.read_csv(os.path.join(RAT, "consensus_traits.csv")).set_index("name").label
    L = f"L{ba.MID}"
    aa = mr[mr.source == "assistant_axis"].copy().set_index("name")
    aa["type"] = rtype.reindex(aa.index).values
    aa["quartile"] = pd.qcut(aa[f"cos_dist_{L}"], 4, labels=False) + 1
    types = ["human", "non_human", "fictional_mythic", "ai_machine"]
    n_ai = min(4, int((aa.type == "ai_machine").sum()))
    rest = 28 - n_ai
    quota = {"ai_machine": n_ai, "non_human": min(rest // 3, int((aa.type == "non_human").sum())), "fictional_mythic": min(rest // 3, int((aa.type == "fictional_mythic").sum()))}
    quota["human"] = 28 - sum(quota.values())
    picks = []
    for t in types:
        p = spread(aa[aa.type == t], "quartile", 4, quota[t], rng); picks.append(p)
    aa_pick = pd.concat(picks)
    # Zheng
    info = ba.zheng_roles().set_index("role")
    zh = mr[mr.source == "zheng"].copy().set_index("name")
    zh["kind"] = ["relationship" if info.loc[n, "role cate"] in ("family", "romantic", "social") else "expertise" if info.loc[n, "mmlu"] == 1 else "other" for n in zh.index]
    zh["tercile"] = pd.qcut(zh[f"cos_dist_{L}"], 3, labels=False) + 1
    zp = pd.concat([spread(zh[zh.kind == k], "tercile", 3, 6, rng) for k in ("relationship", "expertise")])
    an = mr[mr.source == "anchor"].copy().set_index("name")
    R = []
    for n, r in aa_pick.iterrows():
        R.append(dict(role=f"aa:{n}", source="assistant_axis", stratum=f"{r.type}|Q{int(r.quartile)}", type=r.type, **{k: r[k] for k in r.index if k.startswith(("cos_dist", "axis_proj"))}))
    for n, r in zp.iterrows():
        R.append(dict(role=f"zh:{n}", source="zheng", stratum=f"{r.kind}|T{int(r.tercile)}", type=r.kind, **{k: r[k] for k in r.index if k.startswith(("cos_dist", "axis_proj"))}))
    for n, r in an.iterrows():
        R.append(dict(role=n, source="anchor", stratum="anchor|" + ANCHOR_ROLES[n], type=ANCHOR_ROLES[n], **{k: r[k] for k in r.index if k.startswith(("cos_dist", "axis_proj"))}))
    cr = pd.DataFrame(R)
    cr["split"] = split_within(cr, ["source", "type"], rng).values
    ordr = cr[f"cos_dist_{L}"].rank().astype(int)
    cr["distance_rank_of_48"] = ordr
    foc = np.unique(np.round(np.linspace(1, 48, 12)).astype(int))
    cr["focus"] = cr.distance_rank_of_48.isin(foc)
    # traits
    tt = mt.set_index("trait")
    aat = tt[~tt.index.str.startswith("pv:")].copy()
    aat["type"] = ttype.reindex(aat.index).values
    aat["tercile"] = pd.qcut(aat[f"cos_axis_{L}"], 3, labels=False) + 1
    tq = {"style": 14, "stance": 14, "emotion": 14, "social": 14, "epistemic": 14}
    tp = [spread(aat[aat.type == t], "tercile", 3, q, rng) for t, q in tq.items()]
    n_self = int((aat.type == "self_image").sum())
    if n_self:
        tp.append(spread(aat[aat.type == "self_image"], "tercile", 3, min(2, n_self), rng))
    tp = pd.concat(tp)
    short = 72 - len(tp)
    if short > 0:                         # fill from the largest types, spread over terciles
        left = aat.drop(tp.index); left = left[left.type.isin(["stance", "social", "style", "epistemic"])]
        tp = pd.concat([tp, left.sample(short, random_state=0)])
    T = []
    for n, r in tp.iterrows():
        T.append(dict(trait=n, source="assistant_axis", stratum=f"{r.type}|T{int(r.tercile)}", type=r.type, anchor=False, cos_axis=r[f"cos_axis_{L}"]))
    for n in ATLAS_TRAITS:
        T.append(dict(trait=n, source="atlas", stratum="atlas", type="", anchor=True, cos_axis=np.nan))
    for n in PV_TRAITS:
        T.append(dict(trait=n, source="persona_vectors", stratum="pv", type="", anchor=True, cos_axis=float(tt.loc[n, f"cos_axis_{L}"])))
    ct = pd.DataFrame(T)
    ct["split"] = ""
    na = ~ct.anchor
    ct.loc[na, "split"] = split_within(ct[na], ["type"], rng).values
    an_ix = ct[ct.anchor].index
    named = [i for i in an_ix if ct.loc[i, "trait"] in NAMED]; other = [i for i in an_ix if i not in named]
    order = list(rng.permutation(named)) + list(rng.permutation(other))
    for k, i in enumerate(order):
        ct.loc[i, "split"] = "discovery" if k % 2 == 0 else "confirmation"
    assert (ct.split == "discovery").sum() == 48 and (cr.split == "discovery").sum() == 24, (ct.split.value_counts().to_dict(), cr.split.value_counts().to_dict())
    os.makedirs(S1, exist_ok=True)
    cr.to_csv(os.path.join(S1, "chosen_roles.csv"), index=False, float_format="%.5g"); ct.to_csv(os.path.join(S1, "chosen_traits.csv"), index=False, float_format="%.4g")
    out = dict(seed=ba.SEED, made="2026-11-06", note="Stage 1 choice, blind to any behaviour; made before any Stage 2 code. IDs, strata, split labels and distances only.",
               layer=ba.MID, question_ids=ba.shared_questions()[0], quotas=dict(roles=quota, zheng="6 relationship + 6 expertise", anchors=8, traits="72 AA + 20 atlas + 4 PV"),
               roles=cr.to_dict("records"), traits=ct.to_dict("records"))
    json.dump(out, open("data/big_atlas_splits.json", "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(cr.split.value_counts().to_dict(), ct.split.value_counts().to_dict(), cr.groupby(["source", "type"]).size().to_dict())


if __name__ == "__main__":
    {"choose": choose}[sys.argv[1]]()
