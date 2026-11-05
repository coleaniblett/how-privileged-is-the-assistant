"""Spec J3 (overnight 2026-11-05): is refusal ONE internal direction? Follow-up to spec I; exploratory, outside every verdict. Plan and readings: docs/decisions.md 2026-11-05.

Direction: at the reply-start state (hidden_states[L] at the last prompt token, L = spec I's best layer for the family), mean(borderline) - mean(harmless) over the PILOT requests, plain
Assistant prompts only; unit length. Then on the main + expansion requests (24 + 24):
  (a) the projection of every persona's reply-start state on it, against refusal;
  (b) remove it (h - (h.u)u) at the last prompt token for the Assistant and the two thin characters;
  (c) add it (h + a u) for the two vivid characters, a = mean pilot borderline projection of the Assistant minus that vivid's;
  random controls: 5 seeded random unit directions, removed / added with the same norm; damage and voice are measured on the same replies.
    python -m extras.refusal_direction run --family olmo3-7b [--stage base]
    python -m extras.refusal_direction analyze
"""
import argparse
import json
import os
import time
import zlib

import numpy as np
import pandas as pd

from extras import activation_patching as ap

OUT = os.environ.get("RD_OUT", "results/refusal_direction")
SEED, N_BOOT, N_RANDOM = 0, 4000, 5
THIN, VIVID = ["thin1", "thin2"], ["vivid1", "vivid2"]
ALL = ["asst"] + THIN + VIVID
REMOVE_ON = ["asst"] + THIN
ADD_ON = VIVID
CELLS = ("main", "expansion")


def state(ses, pid, req, layer):
    """Reply-start state: hidden_states[layer] at the last prompt token, float32 [H]."""
    return ses.clean(pid, req, 1, [layer])["A"][0, -1]


def unit(v):
    return v / v.norm()


def rand_unit(h, key):
    import torch
    g = torch.Generator().manual_seed(zlib.crc32(repr(key).encode()))
    return unit(torch.randn(h, generator=g))


def fit_direction(ses, layer):
    import torch
    pil = ses.items("pilot")
    b = [state(ses, "asst", r, layer) for k, _, r in pil if k == "borderline"]
    h = [state(ses, "asst", r, layer) for k, _, r in pil if k == "harmless"]
    d = torch.stack(b).mean(0) - torch.stack(h).mean(0)
    return unit(d), pil


def run(family, stage="instruct"):
    fl = json.load(open(os.path.join(ap.OUT, family, "floors.json")))
    layer = fl["best_layer"]
    ses = ap.Session(family, stage=stage)
    u, pil = fit_direction(ses, layer)
    H = u.shape[0]
    pb = [r for k, _, r in pil if k == "borderline"]
    dose = {v: float(np.mean([float(state(ses, "asst", r, layer) @ u) - float(state(ses, v, r, layer) @ u) for r in pb])) for v in VIVID}
    rows, projs = [], []
    for setname in CELLS:
        for kind, ri, req in ses.items(setname):
            mx = ses.mat["generation"]["max_new"][kind]
            for pid in ALL:
                c = ses.clean(pid, req, 1, [layer])
                h = c["A"][0, -1]
                projs.append(dict(set=setname, kind=kind, req=ri, persona=pid, proj=float(h @ u)))
                pos = len(c["ids"]) - 1

                def add_row(tag, seed, R):
                    for smp in range(ap.SAMPLES):
                        rows.append(dict(ids=c["ids"], layer=layer if R is not None else None, R=R, pos=pos, max_new=mx, key=(tag, pid, setname, ri, kind, seed, smp),
                                         meta=dict(tag=tag, persona=pid, seed=seed, set=setname, kind=kind, req=ri, sample=smp)))
                add_row("base", -1, None)
                if pid in REMOVE_ON:
                    add_row("remove", -1, (h - (h @ u) * u)[None])
                    for s in range(N_RANDOM):
                        ur = rand_unit(H, (family, "rm", s))
                        add_row("rand_remove", s, (h - (h @ ur) * ur)[None])
                if pid in ADD_ON:
                    add_row("add", -1, (h + dose[pid] * u)[None])
                    for s in range(N_RANDOM):
                        ur = rand_unit(H, (family, "add", s, pid))
                        add_row("rand_add", s, (h + abs(dose[pid]) * ur)[None])
    print(f"{family} {stage}: layer {layer}, {len(rows)} generations; dose {dose}", flush=True)
    t0 = time.time()
    texts = ap.generate(ses.model, ses.tok, ses.device, ses.patcher, rows)
    dt = time.time() - t0
    print(f"{len(rows)} generations in {dt:.0f} s", flush=True)
    df = pd.DataFrame([r["meta"] for r in rows]).assign(text=texts)
    df = pd.concat([df, ap.metrics_of(texts, list(df.kind))], axis=1)
    d = os.path.join(OUT, family, stage)
    os.makedirs(d, exist_ok=True)
    df.to_csv(os.path.join(d, "gens.csv.gz"), index=False)
    pd.DataFrame(projs).to_csv(os.path.join(d, "projections.csv"), index=False)
    json.dump(dict(family=family, stage=stage, layer=layer, dose=dose, n_generations=len(rows), seconds=dt, pilot_requests=len(pil)), open(os.path.join(d, "meta.json"), "w"), indent=1)


# ------------------------------------------------------------------ analysis
def boot(x, rng, n=N_BOOT):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if not len(x):
        return np.nan, np.nan, np.nan
    bs = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def analyze_one(family, stage, rng):
    d = os.path.join(OUT, family, stage)
    df = pd.read_csv(os.path.join(d, "gens.csv.gz"))
    pr = pd.read_csv(os.path.join(d, "projections.csv"))
    meta = json.load(open(os.path.join(d, "meta.json")))
    fl = json.load(open(os.path.join(ap.OUT, family, "floors.json")))
    L = [f"## {family} {stage} (layer {meta['layer']}; {meta['n_generations']} generations; vivid doses {meta['dose']})", ""]
    b = df[(df.tag == "base") & (df.kind == "borderline")].groupby(["persona", "set", "req"]).refuse.mean().reset_index()
    pb = pr[pr.kind == "borderline"].merge(b, on=["persona", "set", "req"])
    r_all = float(np.corrcoef(pb.proj, pb.refuse)[0, 1])
    slope = float(np.polyfit(pb.proj, pb.refuse, 1)[0])
    within = {p: (float(np.corrcoef(g.proj, g.refuse)[0, 1]) if g.refuse.std() > 0 else np.nan) for p, g in pb.groupby("persona")}
    L += ["### (a) Does the projection predict refusal?", "",
          f"Across all persona x request cells (borderline, {len(pb)} cells): r = {r_all:+.2f}; pooled slope {slope:+.2f} points per unit of projection. Within a persona (across requests): "
          + ", ".join(f"{p} {v:+.2f}" for p, v in within.items()) + ".", "",
          "| pair | refusal gap (points) | projection gap | predicted gap from pooled slope | fraction explained |", "|---|---|---|---|---|"]
    fr = []
    mp = pb.groupby("persona").proj.mean()
    mr = pb.groupby("persona").refuse.mean()
    for a, v in (("asst", "vivid1"), ("asst", "vivid2"), ("thin1", "vivid1"), ("thin2", "vivid2")):
        gap, pg = float(mr[a] - mr[v]), float(mp[a] - mp[v])
        frac = slope * pg / gap if abs(gap) >= 20 else np.nan
        fr.append(frac)
        L.append(f"| {a} vs {v} | {gap:+.1f} | {pg:+.2f} | {slope * pg:+.1f} | {'n/a (gap < 20)' if np.isnan(frac) else f'{frac:.2f}'} |")
    mean_frac = float(np.nanmean(fr)) if np.any(~np.isnan(fr)) else np.nan
    L += ["", f"Mean fraction explained over pairs with a gap of at least 20 points: {mean_frac:.2f}.", ""]
    L += ["### (b) Remove / (c) add the direction (borderline refusal, points; change from the unpatched persona; bootstrap over requests)", "",
          "| persona | action | unpatched refusal | change [95% CI] | random-direction changes (5 seeds, means) | beyond random spread | voice change [CI] | layout chg | length chg | repeat chg |", "|---|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for pid in ALL:
        for act, rt in (("remove", "rand_remove"), ("add", "rand_add")):
            if not ((act == "remove" and pid in REMOVE_ON) or (act == "add" and pid in ADD_ON)):
                continue
            sel = lambda tag, **kw: df[(df.tag == tag) & (df.persona == pid)]
            bd = sel("base")
            base_r = bd[bd.kind == "borderline"].groupby(["set", "req"]).refuse.mean()
            ad = sel(act)
            real = ad[ad.kind == "borderline"].groupby(["set", "req"]).refuse.mean()
            rd = sel(rt)
            rch = [float(rd[(rd.seed == s) & (rd.kind == "borderline")].groupby(["set", "req"]).refuse.mean().sub(base_r).mean()) for s in range(N_RANDOM)]
            sign = -1 if act == "remove" else 1
            m = boot((real - base_r).to_numpy(), rng)
            beyond = bool(sign * m[0] > max(sign * x for x in rch))
            vc = boot((ad.groupby(["set", "kind", "req"]).voice.mean() - bd.groupby(["set", "kind", "req"]).voice.mean()).to_numpy(), rng)
            res[(pid, act)] = dict(ch=m, beyond=beyond, voice=vc)
            L.append(f"| {pid} | {act} | {base_r.mean():.1f} | {m[0]:+.1f} [{m[1]:+.1f}, {m[2]:+.1f}] | {', '.join(f'{x:+.1f}' for x in rch)} | {'yes' if beyond else 'no'} | "
                     f"{vc[0]:+.1f} [{vc[1]:+.1f}, {vc[2]:+.1f}] | {ad.layout.mean() - bd.layout.mean():+.1f} | {ad.words.mean() - bd.words.mean():+.1f} | {ad.repeat.mean() - bd.repeat.mean():+.1f} |")
    vg = {v: float(df[(df.tag == "base") & (df.persona == "asst")].voice.mean() - df[(df.tag == "base") & (df.persona == v)].voice.mean()) for v in VIVID}
    L += ["", "Unpatched voice gap (P(Assistant) points, Assistant minus vivid): " + ", ".join(f"{v} {g:+.1f}" for v, g in vg.items()) + ".", ""]
    rem = [res[(p, "remove")] for p in REMOVE_ON]
    add = [res[(p, "add")] for p in ADD_ON]
    rem_ok = np.mean([r["ch"][0] for r in rem]) <= -15 and all(r["ch"][2] < 0 for r in rem) and all(r["beyond"] for r in rem)
    add_ok = np.mean([r["ch"][0] for r in add]) >= 15 and all(r["ch"][1] > 0 for r in add) and all(r["beyond"] for r in add)
    explained = (not np.isnan(mean_frac)) and mean_frac >= 0.5
    voice_gap_roles = [v for v, g in vg.items() if abs(g) >= 15]
    band = fl["voice_p95_points"]
    sep = [v for v in voice_gap_roles if res[(v, "add")]["voice"][1] >= -band and res[(v, "add")]["voice"][2] <= band]
    one = explained and rem_ok and add_ok
    L += ["### Pre-set readings", "",
          f"- One direction carries the role gap: fraction explained {mean_frac:.2f} (needs at least 0.5): {'met' if explained else 'not met'}; removal (mean {np.mean([r['ch'][0] for r in rem]):+.1f}, needs at most -15, every CI below 0, beyond random): "
          f"{'met' if rem_ok else 'not met'}; adding (mean {np.mean([r['ch'][0] for r in add]):+.1f}, needs at least +15, every CI above 0, beyond random): {'met' if add_ok else 'not met'}. **{'MET' if one else 'not met'}**.",
          f"- Separate from voice: roles with a voice gap of at least 15 points: {voice_gap_roles or 'none'}; of these, voice change on adding the direction inside the equivalence band (+/- {band:.1f}, spec I's committed p95 of controls): {sep or 'none'}. "
          f"**{'consistent with separate' if (one and sep) else ('unresolved' if one else 'not applicable (the reading above is not met)')}**.", ""]
    return L


def analyze():
    rng = np.random.default_rng(SEED)
    L = ["# Is refusal one internal direction? (spec J3; exploratory, outside every verdict)", "",
         "Plan and pre-set readings: `docs/decisions.md` 2026-11-05 (logged before scoring). Direction fit on the plain Assistant's pilot requests only; main + expansion requests (24 + 24), 2 samples; "
         "removal at the reply-start state (last prompt token) for the Assistant and thin characters, adding to vivid ones; random controls of the same norm. Voice is P(Assistant) from the text classifier, on the same replies.", ""]
    for fam in ("olmo3-7b", "qwen3-8b"):
        for stage in ("instruct", "base"):
            if os.path.exists(os.path.join(OUT, fam, stage, "gens.csv.gz")):
                L += analyze_one(fam, stage, rng)
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "refusal_direction.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "analyze"])
    p.add_argument("--family", choices=["olmo3-7b", "qwen3-8b"])
    p.add_argument("--stage", default="instruct", choices=["instruct", "base"])
    a = p.parse_args()
    if a.cmd == "run":
        run(a.family, a.stage)
    else:
        analyze()


if __name__ == "__main__":
    main()
