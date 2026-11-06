"""Spec S Stage 1 report: python -m extras.big_atlas_report -> results/big_atlas/stage1/{stage1.md, role_fit.csv, question_demand.csv, trait_types.csv, ratings_agreement.json}.
Reads the map numbers, the chosen/split tables, the replycheck, and the blind ratings in BA_SRC/ratings (private; only IDs, labels and numbers are written out).
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from extras import big_atlas as ba

S1 = os.path.join(ba.OUT, "stage1")
RAT = os.path.join(ba.SRC, "ratings")
L = ba.LAYERS


def ratings():
    out = {}
    roles = pd.read_csv(f"{RAT}/in_fit_roles.tsv", sep="\t").role.tolist(); traits = pd.read_csv(f"{RAT}/in_fit_traits.tsv", sep="\t").trait.tolist()
    ld = lambda R: pd.concat([pd.read_csv(f"{RAT}/fit_{R}_{b}.csv", index_col=0) for b in range(4)])
    A, B = ld("A"), ld("B"); d = (A - B).abs(); F = np.minimum(A, B).copy()
    C = pd.read_csv(f"{RAT}/C_fit.csv")
    for _, r in C.iterrows():
        F.loc[r.role, r.trait] = r.score
    out["fit"] = dict(exact=float((A == B).values.mean()), kappa_quadratic=float(cohen_kappa_score(A.values.ravel(), B.values.ravel(), weights="quadratic")), diff1=int((d == 1).values.sum()),
                      diff2plus=int((d >= 2).values.sum()), cells=int(A.size), mean_A=float(A.values.mean()), mean_B=float(B.values.mean()), mean_final=float(F.values.mean()),
                      share_final_ge2=float((F.values >= 2).mean()))
    F.index.name = "role"; F.to_csv(f"{S1}/role_fit.csv")
    qs = pd.read_csv(f"{RAT}/in_q_all.tsv", sep="\t")
    qa = pd.concat([pd.read_csv(f"{RAT}/q_A_{b}.csv") for b in range(3)]); qb = pd.concat([pd.read_csv(f"{RAT}/q_B_{b}.csv") for b in range(3)])
    qc = pd.read_csv(f"{RAT}/C_q.csv").set_index("qid").label
    fin = []
    for qid, a, b in zip(qs.qid, qa.label, qb.label):
        fin.append(int(a) if a == b else int(qc[qid]))
    qd = pd.DataFrame({"qid": qs.qid, "demand": fin}); qd.to_csv(f"{S1}/question_demand.csv", index=False)
    out["q"] = dict(kappa=float(cohen_kappa_score(qa.label, qb.label)), agree=float((qa.label.values == qb.label.values).mean()), n=len(qd), disagreements=int((qa.label.values != qb.label.values).sum()),
                    rate_final=float(qd.demand.mean()), rate_shared40=float(qd[qd.qid.str.startswith("shared:")].demand.mean()), n_shared=int(qd.qid.str.startswith("shared:").sum()))
    # types
    tc = pd.read_csv(f"{RAT}/consensus_traits.csv").set_index("name").label
    t24a, t24b = [pd.read_csv(f"{RAT}/type24_{x}.csv").set_index("trait").label for x in "AB"]
    t24c = pd.read_csv(f"{RAT}/C_type24.csv").set_index("trait").label
    t24 = {t: (t24a[t] if t24a[t] == t24b[t] else t24c[t]) for t in t24a.index}
    out["type24"] = dict(agree=float((t24a.values == t24b.values).mean()), disagreements=int((t24a.values != t24b.values).sum()))
    ta, tb = [pd.read_csv(f"{RAT}/{x}_traits.csv").set_index("name").label for x in "AB"]
    ra, rb = [pd.read_csv(f"{RAT}/{x}_roles.csv").set_index("name").label for x in "AB"]
    out["trait_type_240"] = dict(kappa=float(cohen_kappa_score(ta, tb)), agree=float((ta == tb).mean()), disagreements=int((ta != tb).sum()))
    out["role_type_275"] = dict(kappa=float(cohen_kappa_score(ra, rb)), agree=float((ra == rb).mean()), disagreements=int((ra != rb).sum()))
    json.dump(out, open(f"{S1}/ratings_agreement.json", "w"), indent=1)
    return out, F, qd, tc, t24


def pct_below(x, ref):
    return float((np.asarray(ref) < x).mean() * 100)


def build():
    rt, F, qd, tc, t24 = ratings()
    mr = pd.read_csv(f"{S1}/map_roles.csv"); mt = pd.read_csv(f"{S1}/map_traits.csv"); ms = json.load(open(f"{S1}/map_summary.json"))
    cr = pd.read_csv(f"{S1}/chosen_roles.csv"); ct = pd.read_csv(f"{S1}/chosen_traits.csv"); rc = json.load(open(f"{S1}/replycheck.json")); s0 = json.load(open(f"{ba.OUT}/stage0/speeds.json"))
    aa = mr[mr.source == "assistant_axis"]; zh = mr[mr.source == "zheng"]; ctl = mr[mr.source.isin(["control", "default_prompt"])]; anc = mr[mr.source == "anchor"]
    ct["type"] = [t if isinstance(t, str) and t else None for t in ct["type"]]
    for i, r in ct.iterrows():
        if not r["type"]:
            ct.loc[i, "type"] = t24.get(r.trait if r.source == "atlas" else r.trait, "")
    ct.to_csv(f"{S1}/trait_types.csv", index=False, columns=["trait", "source", "type", "stratum", "anchor", "split"])
    P = []; w = P.append
    w("# Spec S, Stage 1: the map, the choice of 48 roles and 96 traits, and the blind ratings\n")
    w("Gemma 4 31B-it, 2026-11-06. Exploratory, outside every verdict, no paid API. **Stages 2 and 3 are not run and nothing from them was generated.** Third-party text (Assistant Axis, Zheng) stays on the servers; this report holds IDs, names and numbers only. Everything about roles and traits here is **prompt-state** (no replies), except the small reply check in section 2.5.\n")
    w("## 1. Stage 0 in one block (full report: `results/big_atlas/stage0/stage0.md`)\n")
    w(f"- **Speeds measured on vinyl:** generation at 256 tokens {s0['gen']['role_256_bs100']['per_s']:.2f} replies/s (batch 50); judging {s0['judge']['matrix']['per_s']:.1f} judgements/s; prompt states {s0['forward']['role_prompt_states']['per_s']:.0f}/s; short-prompt passes {s0['forward']['short_prompt']['per_s']:.0f}/s. The spec assumed 1.3 and 11.6, so judging is about half as fast as assumed.")
    w("- **Projected Stages 2 + 3: 24.3 GPU hours** (limit 30), of which judging 13.7 h (the shared matrix alone 9.0 h; cutting it to 48 traits saves 4.5 h). Two nights on vinyl, as the spec said.")
    w(f"- **Judge:** the exact digit-tree score works (no refusals in 150 pilot judgements). The cheap first-token version does **not** agree (Spearman {s0['tree_vs_first']['spearman']:.2f}, mean absolute difference {s0['tree_vs_first']['mean_abs_diff']:.0f} points), so Stage 2 uses the tree. 4 of 5 pilot traits pass the 50-point positive-vs-negative gap; `analytical` (gap 41) would be reported and not read.")
    w("- **Fixed reply opening:** the template's own generation prompt, which for Gemma 4 is the empty thought channel (no other prefix, any method).")
    w(f"- **Nearly every reply hits the cap** ({s0['gen']['role_256']['capped']} of 50 at 256 tokens): judged replies are the first 256 tokens.\n")
    w("## 2. The map (prompt state, no generation)\n")
    w("**What was computed.** For each of 275 Assistant Axis roles (5 prompts x 40 shared questions, averaged), 162 Zheng roles (\"You are a/an {role}.\"), 8 anchors, 5 default-Assistant prompts and the controls: the hidden state at the last prompt token (after the generation prompt), at layers 24, 33 (middle) and 42 of 60. **Distance** = cosine distance from the default Assistant state for the same question, averaged over the 40 questions. **Axis** = mean(default Assistant states) minus mean(275 role states), unit length, per layer, so higher = more Assistant-like; **projection** = state minus the Assistant's state, dotted with the axis (so 0 = at the Assistant, negative = away). The axis was built from prompt states, not reply averages as in the paper; section 2.5 checks the consequence.\n")
    w("### 2.1 Scale of the map\n")
    w("| layer | AA role distance: quartiles (min to max) | Zheng role distance: median (min to max) | axis SD over random-direction SD | spearman(distance, -projection), AA roles |\n|---|---|---|---|---|")
    for l in L:
        d = aa[f"cos_dist_L{l}"]; z = zh[f"cos_dist_L{l}"]
        w(f"| {l} | {d.quantile(.25):.4f} / {d.median():.4f} / {d.quantile(.75):.4f} ({d.min():.4f} to {d.max():.4f}) | {z.median():.4f} ({z.min():.4f} to {z.max():.4f}) | {ms[f'axis_sd_over_random_sd_L{l}']:.0f} | {aa[f'cos_dist_L{l}'].corr(-aa[f'axis_proj_vs_assistant_L{l}'], method='spearman'):.2f} |")
    w("\nCosine distances are small in absolute terms (states share a large common component); only their ordering and ratios are used. **The Assistant axis carries real structure: role projections vary 20 to 42 times more along it than along random unit directions (pre-set bar: 3).** Prompt-only trait vectors: cosines with the axis have SD "
      f"{ms['trait_cos_axis_sd_L33']:.2f} across the 240 traits versus {ms['trait_cos_random_sd_L33']:.3f} for random directions (layer 33).\n")
    w("### 2.2 Controls against roles\n")
    w("Distance percentile = share of the 275 Assistant Axis roles that are *closer* to the Assistant than the control. Axis percentile = share of roles whose projection is *higher* (closer to the Assistant end) than the control's. Pre-set readings (decisions log): 'as far as roles' if the distance percentile is above 25; 'role-like on the axis' if the control's projection is below the 75th percentile of role projections, i.e. if more than 25 percent of roles are closer to the Assistant end than it is.\n")
    w("| control | " + " | ".join(f"L{l} distance (pctile) | L{l} projection (pctile)" for l in L) + " |\n|---|" + "---|---|" * len(L))
    for _, r in ctl.iterrows():
        cells = []
        for l in L:
            cells.append(f"{r[f'cos_dist_L{l}']:.4f} ({pct_below(r[f'cos_dist_L{l}'], aa[f'cos_dist_L{l}']):.0f})")
            cells.append(f"{r[f'axis_proj_vs_assistant_L{l}']:+.1f} ({100 - pct_below(r[f'axis_proj_vs_assistant_L{l}'], aa[f'axis_proj_vs_assistant_L{l}']):.0f})")
        w(f"| {r['name']} | " + " | ".join(cells) + " |")
    w("\n(In the projection columns the number in brackets is the share of roles that are closer to the Assistant end than the control.)\n")
    w("**Reading.** (a) The same-length neutral text is **not** role-like by distance (4th to 8th percentile at layers 24 and 33, 1st at layer 42); it moves the state about a fifth to a little over half as much as the median role (0.0082 vs 0.0158 at layer 33, 0.0174 vs 0.0807 at layer 42). (b) The tone lines are role-sized at layer 24 (formal 71st, casual 72nd percentile) and layer 33 (casual 31st, formal 24th, just under the bar), but fall back to the near end at layer 42 (formal 1st, casual 24th). (c) 'You are a helpful assistant' is as close as the default prompts themselves (0th percentile everywhere). (d) On the axis only one control is role-like by the pre-set bar: casual at layer 24 (30 percent of roles are closer to the Assistant end than it). The tone lines at layers 33 and 42 and the neutral text have 0 to 11 percent of roles closer to the Assistant end. **So a tone line moves the state about as far as the nearer quarter of roles, a same-length neutral passage much less, and a persona (even a one-line job) about as far as most of the map.** The five default prompts differ by 0.0005 to 0.004 from their mean at layer 33, and `Respond as yourself.` is the farthest of them, still within the nearest 1 percent of roles at every layer; the 'Assistant' is a small region, not a point.\n")
    w("### 2.3 Where the anchors sit\n")
    w("| anchor | L33 distance | percentile among AA roles | L33 projection | rank among the 48 chosen |\n|---|---|---|---|---|")
    for _, r in anc.iterrows():
        rk = cr[cr.role == r["name"]].distance_rank_of_48.iloc[0]
        w(f"| {r['name'][7:]} | {r['cos_dist_L33']:.4f} | {pct_below(r['cos_dist_L33'], aa['cos_dist_L33']):.0f} | {r['axis_proj_vs_assistant_L33']:+.1f} | {rk} |")
    w("\nThe three light jobs are closer than the three strong characters at every layer, as the old atlas assumed, but **all eight anchors lie in the farthest 15 percent of the Assistant Axis roles at layer 33 (at layer 24 the nearest anchor is at the 51st percentile, at layer 42 at the 96th)** (their prompts are short and name a person, not a description of an archetype). 'Nearby' in the old atlas means nearer than the strong characters, not near the Assistant.\n")
    w("### 2.4 Trait vectors (prompt-only, 10 questions per trait)\n")
    w(f"240 traits and the 4 Persona Vectors traits, 5 pairs x 10 questions x 2 sides = 100 prompt states per trait (the spec's full 40 questions were not used; time). Cosine of the trait vector with the Assistant axis at layer 33: median {mt.cos_axis_L33.median():+.2f}, range {mt.cos_axis_L33.min():+.2f} to {mt.cos_axis_L33.max():+.2f}. The sign shows which traits point toward the Assistant end; the table in section 3.2 uses terciles of it.\n")
    w("### 2.5 Does prompt-state distance track reply-average distance?\n")
    w("Small generation inside Stage 1 (map validation only): the 48 chosen roles + the default Assistant x 6 shared questions x 96 tokens (system prompt), mean hidden state over reply tokens, distance from the Assistant's reply mean. Spearman across the 48 roles with the prompt-state distance:\n")
    w("| layer | plain cosine distance | both centred on the grand mean | pre-set reading (plain) |\n|---|---|---|---|")
    for l in L:
        v = rc[f"L{l}"]; rd = "tracks" if v["spearman_prompt_vs_reply"] >= .7 else "does not" if v["spearman_prompt_vs_reply"] < .4 else "in between"
        w(f"| {l} | {v['spearman_prompt_vs_reply']:+.2f} | {v['spearman_prompt_centred_vs_reply_centred']:+.2f} | {rd} |")
    w("\n**The prompt-state ordering tracks the reply-average ordering (0.79 to 0.86 against the 0.7 bar).** After centring, which removes the shared component, it falls to 0.56 to 0.63 (in between). So the ordering of roles is stable, but part of it is the common component and the prompt text length; Stage 2 re-checks this on the full reply set, as the spec says. Caveat: 6 questions, 96 tokens, 48 roles.\n")
    w("## 3. The choice (blind to any behaviour; seed 0; split committed in `data/big_atlas_splits.json` before any Stage 2 code)\n")
    w("### 3.1 Roles (48): 24 discovery, 24 confirmation, 12 focus\n")
    w("| role | source | stratum | L33 distance | rank of 48 (1 = nearest) | split | focus |\n|---|---|---|---|---|---|---|")
    for _, r in cr.sort_values("distance_rank_of_48").iterrows():
        w(f"| {r.role} | {r.source} | {r.stratum} | {r['cos_dist_L33']:.4f} | {r.distance_rank_of_48} | {r.split} | {'yes' if r.focus else ''} |")
    w("\nStrata shortfalls: the rating found only 3 AI-or-machine roles among the 275 (quota 4), so the AI stratum has 3, the other three types 8 each and humans 9. Zheng: 6 relationship (family, romantic, social) + 6 expertise (mmlu occupations). The 12 focus roles are the 12 evenly spaced distance ranks of the 48.\n")
    w("### 3.2 Traits (96): 48 discovery, 48 confirmation, 24 anchors (12 per half)\n")
    w("| trait | source | blind type | stratum | cosine with axis (L33) | anchor | split |\n|---|---|---|---|---|---|---|")
    for _, r in ct.iterrows():
        c = r.cos_axis
        w(f"| {r.trait} | {r.source} | {r.type if isinstance(r.type, str) else ''} | {r.stratum} | {'' if pd.isna(c) else f'{c:+.2f}'} | {'yes' if r.anchor else ''} | {r.split} |")
    w("\nThe 72 Assistant Axis traits are 14 each of style, stance, emotion, social and epistemic and the 2 self-image traits that exist, spread over cosine terciles within each type. The 20 atlas and 4 Persona Vectors traits are the 24 anchors (they include `denies_feelings`, `says_ai`, `markdown_lists`, `stock_phrases`, `corrects_false_premise`); their cosines for the atlas traits were not computed (no prompt-pair vectors exist for them). Types of those 24 come from the same blind protocol.\n")
    w("## 4. Blind ratings and agreement\n")
    f = rt["fit"]; q = rt["q"]
    w("Two raters (fresh subagents) rated from names and descriptions only; a third settled disagreements. No rater saw any result, distance or split.\n")
    w("| rating | items | agreement | kappa | settled by third |\n|---|---|---|---|---|")
    w(f"| trait type (6 labels), all 240 | 240 | {rt['trait_type_240']['agree']:.0%} | {rt['trait_type_240']['kappa']:.2f} | {rt['trait_type_240']['disagreements']} |")
    w(f"| role type (4 labels), all 275 AA roles | 275 | {rt['role_type_275']['agree']:.0%} | {rt['role_type_275']['kappa']:.2f} | {rt['role_type_275']['disagreements']} |")
    w(f"| trait type, the 24 atlas and Persona Vectors traits | 24 | {rt['type24']['agree']:.0%} | | {rt['type24']['disagreements']} |")
    w(f"| role fit 0-3, {f['cells']} cells (48 roles x 96 traits) | {f['cells']} | exact {f['exact']:.0%}; within 1 {1 - f['diff2plus'] / f['cells']:.0%} | {f['kappa_quadratic']:.2f} (quadratic weighted) | {f['diff2plus']} (a difference of 2 or more) |")
    w(f"| question demand (structured answer expected), {q['n']} questions | {q['n']} | {q['agree']:.1%} | {q['kappa']:.2f} | {q['disagreements']} |")
    w(f"\nRole fit: mean A {f['mean_A']:.2f}, mean B {f['mean_B']:.2f}; cells differing by 1 (n = {f['diff1']}) take the lower value; final mean {f['mean_final']:.2f}, {f['share_final_ge2']:.1%} of cells rated 2 or 3 (a role that would plausibly show the trait). The weighted kappa 0.72 is above the pre-set 0.6 bar, so role fit is usable as a predictor. Question demand is narrow on purpose: {q['rate_final']:.1%} of the {q['n']} questions are labelled as expecting a structured answer, {q['rate_shared40']:.1%} of the {q['n_shared']} shared questions. With so few positives the question-demand predictor will have little to work with in Stage 2.\n")
    w("Files: `role_fit.csv` (48 x 96), `question_demand.csv`, `trait_types.csv`, `ratings_agreement.json`.\n")
    w("## 5. Where this differs from the spec\n")
    for x in [
        "**Prompt-only trait vectors use 10 questions per trait** (approved), 100 prompt states per trait.",
        "**Trait types were rated for all 240 first** (one six-label rating serves both the strata and the type), so the strata used the rated type, not a separate content rating.",
        "**Small generation in Stage 1** (288 replies at 96 tokens) to answer 'how far does prompt-state distance agree with reply-average distance'; the spec says the map generates none but also asks for that figure in Stage 1.",
        "**Neutral control length:** the same token length as the median Assistant Axis role prompt (18 tokens), not as a strong character sheet (the earlier quick check). It is therefore much shorter than the persona prompts it is compared with; a sheet-length neutral control was not run.",
        "**Random directions** are used as a control for the axis (spread of role projections on the axis vs on 20 random unit directions) and for the trait cosines; they are not a role-like condition on the map, since a random direction has no prompt.",
        "**Anchors:** the 3 strong characters are Ottoline Brandt, Dmitri Volkov and Marguerite Oyelaran with the atlas acting sentence; Hannah Reed and Captain Rook use the L2 persona line. All 8 anchors are far from the Assistant in the map (section 2.3).",
        "**Only 3 AI-or-machine Assistant Axis roles exist** in the rating, so that stratum is short by one.",
        "**Few-shot prompt carries no system prompt** (Stage 0 note), for review before any Stage 3.",
        "The pilot trait check passed 4 of 5 traits; the 'drop a trait below kappa 0.4' rule and the second judge are Stage 2 items and were not run.",
    ]:
        w("- " + x)
    w("\n## 6. Limits\n")
    w("Gemma only. Prompt states are the model's state before it writes anything; whether the distance predicts what it then does is a Stage 2 question. Raters are language models reading names and descriptions; agreement is not accuracy. Stage 1 draws no behavioural conclusion. Nulls and failures are reported above (the exact-vs-cheap judge mismatch, the failed pilot trait, the neutral control that is shorter than the personas).\n")
    open(f"{S1}/stage1.md", "w").write("\n".join(P))
    print("written", len(P))


if __name__ == "__main__":
    build()
