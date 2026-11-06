"""Spec G addendum, POST-HOC (added 2026-11-04 after the provisional atlas was seen): does the reply speak as the character? (exploratory, outside every verdict)

Same frozen `meta_judge` question as the atlas (`speaks_as_character`), same judge (Gemma 4 31B instruct, P(yes) >= 0.5), blind. New files only: nothing under results/overgeneralization/trait_atlas is overwritten.
  A. atlas light_system and light_user replies of every generated checkpoint (the strong contexts already have their meta judgments);
  B. saved leakage-matrix replies (results/overgeneralization/dominance_matrix/gens, register rows) in the character contexts `named` (helpful + neutral roles) and `voiced` (stylized roles),
     each judged against its own role description (name, role). The user, narration and raw contexts are skipped: the question (first person as the character) does not fit a
     simulated user, a third-person narration, or a quoted transcript line.
    python -m extras.trait_atlas_incharacter judge [--sample N]    # GPU; --sample N times N prompts and writes nothing -> results/overgeneralization/trait_atlas_incharacter/{atlas,leak}_<family>_<stage>.csv.gz
    python -m extras.trait_atlas_incharacter handcheck             # blind sheet of 30 (+ key, kept on the judging machine)
    python -m extras.trait_atlas_incharacter analyze               # -> results/overgeneralization/trait_atlas_incharacter/in_character.md and .csv
Judge note for the light roles (no name in the atlas material): "the background describes a character: <role>".
"""
import argparse, glob, os, time
import numpy as np, pandas as pd
from extras import trait_atlas as ta

OUT = "results/overgeneralization/trait_atlas_incharacter"
LEAK = os.environ.get("LEAK_GENS", "results/overgeneralization/dominance_matrix/gens")
LEAK_FAMS = {"olmo": ["base", "sft", "dpo", "instruct"], "qwen": ["base", "instruct"], "gemma": ["base", "instruct"]}
LIGHT = ["light_system", "light_user"]
STRONG = ["strong_act", "strong_noact"]
LEAK_CTX = ["named", "voiced"]
KEYS = ["family", "stage", "trait", "item", "context", "wording", "persona", "sample"]
N_BOOT_IC = 4000
MIN_ITEMS = 6
NOTE_LIGHT = "Reference note: the background describes a character: {role}.\n\n"


# ------------------------------------------------------------------ judge prompts
def _roles():
    from extras import role_register as rr
    flat = [r for g, l in rr.ROLES.items() for r in l]
    return flat, rr.NAMES, rr.SURNAMES


def leak_note(role, ctx):
    flat, names, sur = _roles()
    i = flat.index(role)
    name = names[i] if ctx == "named" else f"{names[i]} {sur[i]}"
    return ta.META_JUDGE["speaks_as_character"]["note"].format(name=name, role=role)


def prompt_for(mat, question, reply, note):
    return mat["judge_template"].format(question=question, reply=reply.strip()[:1500] or "(empty reply)", note=note,
                                        criterion=mat["meta_judge"]["speaks_as_character"]["criterion"])


def atlas_jobs(mat, path):
    rows = [r for r in ta.read_gz(path) if r["context"] in LIGHT]
    return rows, [prompt_for(mat, r["question"], r["text"], NOTE_LIGHT.format(role=mat["light"][r["persona"]])) for r in rows]


def leak_jobs(mat, fam, stage):
    from extras import persona_axis as pa
    d = pd.read_json(os.path.join(LEAK, fam, f"{stage}.jsonl"), lines=True)
    d = d[(d.trait == "register") & d.ctx.isin(LEAK_CTX)].reset_index(drop=True)
    return d, [prompt_for(mat, pa.QUESTIONS[int(r.item)], r.text, leak_note(r.role, r.ctx)) for r in d.itertuples()]


def judge(args):
    import torch
    from core import config
    from core.render import encode, load_tokenizer
    from core.score import load_model
    mat = ta.load_materials()
    config.use_family("gemma4-31b")
    tok = load_tokenizer(config.TEMPLATE_MODEL)
    tok.padding_side = "left"
    yes, no = encode(tok, "Yes")[0], encode(tok, "No")[0]
    model, device = load_model(config.MODELS["instruct"], torch.bfloat16)
    os.makedirs(OUT, exist_ok=True)

    def run(prompts, label):
        t0 = time.time()
        full = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
        order = sorted(range(len(full)), key=lambda k: len(full[k]))
        py = np.full(len(full), np.nan)
        for s in range(0, len(order), args.batch_size):
            b = order[s:s + args.batch_size]
            enc = tok([full[k] for k in b], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
            py[b] = torch.stack([lg[:, yes], lg[:, no]], 1).softmax(-1)[:, 0].cpu().numpy()
            if (s // args.batch_size) % 50 == 0:
                print(f"{label}: {s}/{len(order)} ({time.time() - t0:.0f} s)", flush=True)
        print(f"{label}: {len(full)} judgements in {time.time() - t0:.0f} s = {len(full) / max(time.time() - t0, 1e-9) * 60:.0f} per minute", flush=True)
        return py

    if args.sample:
        rows, prompts = atlas_jobs(mat, sorted(glob.glob(os.path.join(ta.OUT, "*", "*_replies.jsonl.gz")))[0])
        run(prompts[:args.sample], "SAMPLE (atlas light, nothing written)")
        return
    for path in sorted(glob.glob(os.path.join(ta.OUT, "*", "*_replies.jsonl.gz"))):
        fam, stage = path.split(os.sep)[-2], os.path.basename(path).split("_")[0]
        out = os.path.join(OUT, f"atlas_{fam}_{stage}.csv.gz")
        if os.path.exists(out):
            continue
        rows, prompts = atlas_jobs(mat, path)
        py = run(prompts, f"atlas {fam}/{stage}")
        pd.DataFrame([{**{k: r[k] for k in ("trait", "item", "context", "wording", "persona", "sample")}, "p_yes": float(p), "value": float(p >= 0.5)} for r, p in zip(rows, py)]).to_csv(out, index=False)
    for fam, stages in LEAK_FAMS.items():
        for stage in stages:
            out = os.path.join(OUT, f"leak_{fam}_{stage}.csv.gz")
            if os.path.exists(out) or not os.path.exists(os.path.join(LEAK, fam, f"{stage}.jsonl")):
                continue
            d, prompts = leak_jobs(mat, fam, stage)
            py = run(prompts, f"leak {fam}/{stage}")
            d[["ctx", "tmpl", "item", "role", "rep"]].assign(p_yes=py, value=(py >= 0.5).astype(float)).to_csv(out, index=False)


# ------------------------------------------------------------------ hand check
def pool_for_handcheck():
    mat = ta.load_materials()
    pool = []
    for p in sorted(glob.glob(os.path.join(OUT, "atlas_*.csv.gz"))):
        fam, stage = os.path.basename(p)[6:-7].rsplit("_", 1)
        j = pd.read_csv(p)
        rep = pd.DataFrame(ta.read_gz(ta.replies_path(fam, stage)))
        m = rep.merge(j[["trait", "item", "context", "wording", "persona", "sample", "p_yes", "value"]], on=["trait", "item", "context", "wording", "persona", "sample"])
        m = m.assign(family=fam, stage=stage, message=m.question, background=[NOTE_LIGHT.format(role=mat["light"][i]).replace("Reference note: ", "").strip() for i in m.persona])
        pool.append(m[["family", "stage", "context", "message", "background", "text", "p_yes", "value"]])
    for p in sorted(glob.glob(os.path.join(OUT, "leak_*.csv.gz"))):
        fam, stage = os.path.basename(p)[5:-7].split("_", 1)
        d, _ = leak_jobs(mat, fam, stage)
        j = pd.read_csv(p)
        from extras import persona_axis as pa
        m = d.merge(j, on=["ctx", "tmpl", "item", "role", "rep"])
        m = m.assign(family="leak-" + fam, stage=stage, context=m.ctx, message=[pa.QUESTIONS[int(i)] for i in m["item"]],
                     background=[leak_note(r, c).replace("Reference note: ", "").strip() for r, c in zip(m.role, m.ctx)])
        pool.append(m[["family", "stage", "context", "message", "background", "text", "p_yes", "value"]])
    return pd.concat(pool, ignore_index=True)


def handcheck(args):
    pool = pool_for_handcheck()
    pool = pool[pool.text.str.len() > 0]
    rng = np.random.default_rng(ta.SEED)
    cells = sorted(set(zip(pool.family, pool.stage, pool.context)))
    rng.shuffle(cells)
    rows, k = [], 0
    while len(rows) < 30 and k < 400:
        fam, stage, ctx = cells[k % len(cells)]
        want = float(k % 2 == 0)
        c = pool[(pool.family == fam) & (pool.stage == stage) & (pool.context == ctx) & (pool.value == want)]
        if len(c):
            rows.append(c.iloc[int(rng.integers(0, len(c)))])
        k += 1
    sheet = pd.DataFrame(rows).sample(frac=1, random_state=ta.SEED).reset_index(drop=True)
    sheet["sheet_id"] = [f"C{n + 1:02d}" for n in range(len(sheet))]
    crit = ta.META_JUDGE["speaks_as_character"]["criterion"]
    pd.DataFrame(dict(sheet_id=sheet.sheet_id, message=sheet.message, background=sheet.background, reply=sheet.text, criterion=crit, yes_or_no="")).to_csv(os.path.join(OUT, "handcheck_incharacter_sheet.csv"), index=False)
    sheet[["sheet_id", "family", "stage", "context", "p_yes", "value"]].to_csv(os.path.join(OUT, "handcheck_incharacter_key.csv"), index=False)
    print(f"wrote {len(sheet)} rows; the key stays on this machine")


# ------------------------------------------------------------------ analysis
def load_flags():
    """Per-reply in-character flag for atlas contexts: light (new files) + strong (the atlas's existing meta judgments)."""
    fr = []
    for p in sorted(glob.glob(os.path.join(OUT, "atlas_*.csv.gz"))):
        fam, stage = os.path.basename(p)[6:-7].rsplit("_", 1)
        d = pd.read_csv(p)
        fr.append(d.assign(family=fam, stage=stage))
    for p in sorted(glob.glob(os.path.join(ta.OUT, "*", "*_judged_meta.csv.gz"))):
        d = pd.read_csv(p)
        fr.append(d.assign(family=p.split(os.sep)[-2], stage=os.path.basename(p).split("_")[0]))
    f = pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()
    return f.rename(columns={"value": "ic"})[KEYS + ["ic"]] if len(f) else f


def boot_change(d, stage, scale, rng):
    """(change, lo, hi, n_items) of stage minus base, from replies in d (already filtered); NaN-safe; n_items = items with data on both sides."""
    if d is None or len(d) == 0:
        return np.nan, np.nan, np.nan, 0
    M = ta.cell_matrix(d, stage) * scale
    n_items = int(np.sum(~np.all(np.isnan(M), axis=1)))
    if n_items < MIN_ITEMS:
        return np.nan, np.nan, np.nan, n_items
    return (*ta.two_way_boot(M[~np.all(np.isnan(M), axis=1)], rng, n_boot=N_BOOT_IC), n_items)


def composition(all_fall, ic_fall, ic_lo, bar):
    if not (all_fall >= bar):
        return "n/a (no fall in all replies)"
    if np.isnan(ic_fall):
        return "too few in-character replies"
    if ic_fall < 0.5 * all_fall:
        return "mostly composition"
    if ic_fall >= 0.5 * all_fall and ic_fall >= bar and ic_lo > 0:
        return "survives"
    return "unresolved"


def atlas_rows(V, F, C, mat):
    rng = np.random.default_rng(ta.SEED)
    M = V.merge(F, on=KEYS, how="inner")
    out = []
    for fam, stages in ta.FAMILY_STAGES.items():
        for trait in [t["id"] for t in mat["traits"]]:
            unit = ta.unit_of(trait)
            scale, bar = (1.0, ta.SPREAD_LOG) if unit == "log" else (100.0, ta.SPREAD_PTS)
            for ctx in LIGHT + STRONG:
                d0 = M[(M.family == fam) & (M.trait == trait) & (M.context == ctx)]
                for stage in [s for s in stages if s != "base"]:
                    d = d0[d0.stage.isin(["base", stage])]
                    if d.empty or set(d.stage) != {"base", stage}:
                        continue
                    a = C[(C.family == fam) & (C.stage == stage) & (C.trait == trait) & (C.context == "assistant") & (~C.flagged_excluded.astype(bool))]
                    ach = float(a.change.iloc[0]) if len(a) else np.nan
                    gated = bool(len(a) and ta.gate(a.change.iloc[0], a.lo.iloc[0], a.hi.iloc[0], unit))
                    ca, la, ha, na = boot_change(d, stage, scale, rng)
                    ci, li, hi, ni = boot_change(d[d.ic == 1.0], stage, scale, rng)
                    share = float(d[d.stage == stage].ic.mean())
                    sign_ok = bool(not np.isnan(ci) and not np.isnan(ach) and np.sign(ci) == np.sign(ach))
                    spread = bool(sign_ok and abs(ci) >= bar and (li > 0 or hi < 0))
                    reading = ("too few in-character replies" if np.isnan(ci) else "real spread" if spread else "not a character" if share < 0.5 else "no spread among in-character replies")
                    row = dict(family=fam, stage=stage, trait=trait, cluster=ta.CLUSTER_OF[trait], context=ctx, unit=unit, assistant_change=ach, gated=gated, share_in_character=share,
                               change_all=ca, lo_all=la, hi_all=ha, change_ic=ci, lo_ic=li, hi_ic=hi, n_items_ic=ni, reading=reading if ctx in LIGHT else "")
                    if ctx in STRONG and ta.CLUSTER_OF[trait] in ("format", "voice") and not np.isnan(ach) and ach != 0:
                        s = -np.sign(ach)     # a fall = change away from the Assistant's trained change
                        row["composition"] = composition(s * ca, s * ci, min(s * li, s * hi), bar)
                    out.append(row)
    return pd.DataFrame(out)


def leak_table():
    from extras.persona_suppression import layout
    from measures.overgeneralization import human_vs_assistant as hva
    clf = hva.load_classifier()
    frames = {}
    for p in sorted(glob.glob(os.path.join(OUT, "leak_*.csv.gz"))):
        fam, stage = os.path.basename(p)[5:-7].split("_", 1)
        d = pd.read_json(os.path.join(LEAK, fam, f"{stage}.jsonl"), lines=True)
        d = d[(d.trait == "register") & d.ctx.isin(LEAK_CTX)].merge(pd.read_csv(p), on=["ctx", "tmpl", "item", "role", "rep"])
        d["layout"] = [100.0 * layout(t) for t in d.text]
        d["clf"] = 100.0 * clf.p_assistant(list(d.text))
        d["family"], d["stage"] = fam, stage
        frames[(fam, stage)] = d
    rng = np.random.default_rng(ta.SEED)
    out = []
    for fam, stages in LEAK_FAMS.items():
        if (fam, "base") not in frames:
            continue
        for stage in stages[1:]:
            if (fam, stage) not in frames:
                continue
            for ctx in LEAK_CTX:
                B, S = (frames[(fam, s)] for s in ("base", stage))
                B, S = B[B.ctx == ctx], S[S.ctx == ctx]
                share_b, share_s = float(B.value.mean()), float(S.value.mean())
                for metric, bar in (("coord", None), ("clf", ta.SPREAD_PTS), ("layout", ta.SPREAD_PTS)):
                    res = {}
                    for label, keep in (("all", lambda x: x), ("ic", lambda x: x[x.value == 1.0])):
                        b, s = keep(B), keep(S)
                        pb = b.pivot_table(index="item", columns="role", values=metric, aggfunc="mean")
                        ps = s.pivot_table(index="item", columns="role", values=metric, aggfunc="mean")
                        D = (ps.reindex_like(pb) - pb).to_numpy() if len(pb) and len(ps) else np.full((1, 1), np.nan)
                        ok = int(np.sum(~np.all(np.isnan(D), axis=1)))
                        res[label] = (np.nan, np.nan, np.nan) if ok < MIN_ITEMS else ta.two_way_boot(D[~np.all(np.isnan(D), axis=1)], rng, n_boot=N_BOOT_IC)
                    ca, la, ha = res["all"]
                    ci, li, hi = res["ic"]
                    fall_all, fall_ic = -ca, -ci
                    comp = composition(fall_all, fall_ic, -hi if not np.isnan(hi) else np.nan, bar if bar else 0.0) if bar else (
                        "n/a (no fall in all replies)" if not fall_all > 0 else "too few in-character replies" if np.isnan(ci) else
                        "mostly composition" if fall_ic < 0.5 * fall_all else "survives" if -hi > 0 else "unresolved")
                    out.append(dict(family=fam, stage=stage, context=ctx, metric=metric, share_in_character_base=share_b, share_in_character_stage=share_s,
                                    change_all=ca, lo_all=la, hi_all=ha, change_ic=ci, lo_ic=li, hi_ic=hi, composition=comp))
    return pd.DataFrame(out)


def fmt(r, a="change_all", lo="lo_all", hi="hi_all"):
    return "n/a" if np.isnan(r[a]) else f"{r[a]:+.2f} [{r[lo]:+.2f}, {r[hi]:+.2f}]"


def analyze(args):
    mat = ta.load_materials()
    V = ta.load_values()
    V = V.drop(columns="source")
    V = V[~V.trait.str.endswith("__alt")]
    F = load_flags()
    L = ["# In-character check (spec G addendum, POST-HOC: added after the provisional atlas was seen)", "",
         "Question (frozen `meta_judge`, Gemma 4 31B instruct, P(yes) >= 0.5, blind): does the reply speak in the first person as the character? In-character replies are a selected group, not a random sample; "
         "base has few of them; the judge is not yet validated (hand check pending); 12 items per trait. Rates are all flagged-included.", ""]
    out_csv = []
    if len(F):
        C = ta.analyze_cells(V, mat)
        A = atlas_rows(V, F, C, mat)
        out_csv.append(A.assign(part="atlas"))
        sh = F.groupby(["family", "stage", "context"]).ic.agg(["mean", "size"]).reset_index()
        L += ["## Share in character, atlas (all traits pooled)", "", sh.pivot_table(index=["family", "stage"], columns="context", values="mean").round(3).to_markdown(), ""]
        L += ["## Light roles: like-for-like change from base (gated traits: the Assistant's own change passes the atlas gate)", "",
              "| family | stage | context | trait | Assistant chg | share in character | all replies | in-character only (n items) | reading |", "|---|---|---|---|---|---|---|---|---|"]
        for r in A[(A.context.isin(LIGHT)) & A.gated].itertuples():
            L.append(f"| {r.family} | {r.stage} | {r.context} | {r.trait} | {r.assistant_change:+.1f} | {r.share_in_character:.2f} | {fmt(r._asdict())} | "
                     f"{'too few' if np.isnan(r.change_ic) else f'{r.change_ic:+.1f} [{r.lo_ic:+.1f}, {r.hi_ic:+.1f}]'} ({r.n_items_ic}) | {r.reading} |")
        L += ["", "Counts of readings among gated light-role cells:", "", A[(A.context.isin(LIGHT)) & A.gated].groupby(["family", "stage", "reading"]).size().to_frame("cells").to_markdown(), ""]
        sc = A[A.context.isin(STRONG) & A.composition.notna()] if "composition" in A else pd.DataFrame()
        L += ["## Style fall in strong characters (format and voice clusters): all replies vs in-character only", "",
              "| family | stage | context | trait | share in character | all replies | in-character only (n items) | reading |", "|---|---|---|---|---|---|---|---|"]
        for r in sc.itertuples():
            L.append(f"| {r.family} | {r.stage} | {r.context} | {r.trait} | {r.share_in_character:.2f} | {fmt(r._asdict())} | "
                     f"{'too few' if np.isnan(r.change_ic) else f'{r.change_ic:+.1f} [{r.lo_ic:+.1f}, {r.hi_ic:+.1f}]'} ({r.n_items_ic}) | {r.composition} |")
        L += [""]
    else:
        L += ["(no atlas in-character judgments yet)", ""]
    try:
        T = leak_table()
    except Exception as e:                 # leakage files not judged yet
        T = pd.DataFrame()
        L += [f"(leakage-matrix part not available: {e})", ""]
    if len(T):
        out_csv.append(T.assign(part="leakage_matrix"))
        L += ["## Leakage matrix replies (register rows; style score = per-reply axis coordinate, text classifier = P(Assistant) in points, layout regex in points)", "",
              "| family | stage | context | metric | share in character (base -> stage) | all replies | in-character only | reading |", "|---|---|---|---|---|---|---|---|"]
        for r in T.itertuples():
            L.append(f"| {r.family} | {r.stage} | {r.context} | {r.metric} | {r.share_in_character_base:.2f} -> {r.share_in_character_stage:.2f} | {fmt(r._asdict())} | {fmt(r._asdict(), 'change_ic', 'lo_ic', 'hi_ic')} | {r.composition} |")
        L += ["", "Available per-reply scores: style score (coord), text classifier, layout regex. The axis-v2 coordinate is saved for OLMo only and is not split here."]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "in_character.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    if out_csv:
        pd.concat(out_csv, ignore_index=True).to_csv(os.path.join(OUT, "in_character.csv"), index=False)
    print("\n".join(L[:12]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["judge", "handcheck", "analyze"])
    ap.add_argument("--sample", type=int)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    {"judge": judge, "handcheck": handcheck, "analyze": analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
