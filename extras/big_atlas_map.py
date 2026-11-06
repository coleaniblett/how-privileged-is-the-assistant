"""Spec S Stage 1a: the map (no generation). python -m extras.big_atlas_map run|analyze
run (GPU, vinyl): prompt states (last prompt token, layers 24/33/42) for 275 Assistant Axis roles (5 prompts x 40 shared questions, averaged over prompts), 162 Zheng roles
  ("You are a/an {role}."), the default Assistant (5 prompts), the controls, and prompt-only trait vectors for the 240 traits (10 questions x 5 pairs x 2 sides).
  Chunked and resumable. Raw states -> results/big_atlas/private/ (git-ignored).
analyze (CPU): distances, Assistant axis, controls, random directions -> results/big_atlas/stage1/map_roles.csv, map_traits.csv, map_summary.json (names and numbers only).
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from extras import big_atlas as ba

S1 = os.path.join(ba.OUT, "stage1")
CH = os.path.join(ba.PRIV, "map_chunks")
NEUTRAL = ("The stapler on the desk is grey and holds a strip of staples in a spring-loaded channel. It is loaded from the back or the bottom. "
           "The base plate has two small grooves that fold the legs of each staple inward, and the arm returns on a coil spring. Sizes differ a little between models. "
           "A box of spare staples is kept in the second drawer, next to the tape and the hole punch. The office is on the third floor and the window faces the car park. ")


def neutral_text(tok, n_tokens):
    ids = tok(NEUTRAL * 6, add_special_tokens=False).input_ids[:n_tokens]
    return tok.decode(ids)


def chunk(name, fn):
    os.makedirs(CH, exist_ok=True)
    p = os.path.join(CH, name + ".npz")
    if os.path.exists(p):
        return dict(np.load(p, allow_pickle=True))
    t0 = time.time(); r = fn(); np.savez(p, **r); print(f"chunk {name} {time.time() - t0:.0f}s", flush=True)
    return r


def run():
    model, tok = ba.load()
    roles, traits = ba.aa_roles(), ba.aa_traits()
    qids, qs = ba.shared_questions(); nq = len(qs)
    zh = ba.zheng_roles()
    names = sorted(roles)
    json.dump(dict(question_ids=qids, aa_roles=names, zheng_roles=zh.role.tolist()), open(os.path.join(ba.PRIV, "map_index.json"), "w"))

    def block(systems):          # systems: list of str|None -> states n_sys x nq x 3 x d (float16)
        texts = [ba.render(tok, s, q) for s in systems for q in qs]
        st = ba.states(model, tok, texts, bs=32)
        return st.reshape(len(systems), nq, len(ba.LAYERS), -1)

    # default Assistant and controls
    def f_default():
        s = block([None if p == "" else p for p in ba.DEFAULT_PROMPTS])
        return dict(states=s.astype(np.float16))
    chunk("default", f_default)
    lens = [len(tok(roles[r]["instr"][k], add_special_tokens=False).input_ids) for r in names for k in range(5)]
    med = int(np.median(lens))
    ctl = dict(ba.CONTROLS); ctl["ctl_neutral"] = neutral_text(tok, med)

    def f_ctl():
        s = block(list(ctl.values()))
        return dict(states=s.astype(np.float16), names=np.array(list(ctl)), median_role_prompt_tokens=med)
    chunk("controls", f_ctl)
    # Assistant Axis roles in chunks of 25
    for c in range(0, len(names), 25):
        sub = names[c:c + 25]

        def f():
            out = np.zeros((len(sub), nq, len(ba.LAYERS), 5376), np.float32)
            allsys = [roles[r]["instr"][k] for r in sub for k in range(5)]
            s = block(allsys).reshape(len(sub), 5, nq, len(ba.LAYERS), -1)
            return dict(states=s.mean(1).astype(np.float16), names=np.array(sub))
        chunk(f"aa_{c:03d}", f)
    # anchors (atlas light jobs, strong characters with the acting sentence, two L2 roles): names are the IDs
    mat = json.load(open("data/trait_atlas_materials.json"))
    anc = {"anchor:a ferry captain": "You are a ferry captain.", "anchor:a village baker": "You are a village baker.", "anchor:a long-haul trucker": "You are a long-haul trucker."}
    for sp in mat["strong"][:3]:
        anc[f"anchor:{sp['name']}"] = f"You are {sp['name']}. {sp['sheet']} {mat['act_sentence']}"
    anc["anchor:Hannah Reed"] = "You are Hannah Reed, a helpful human front-desk clerk at a hotel. Stay in character at all times."
    anc["anchor:Captain Rook"] = "You are Captain Rook, a swaggering pirate captain. Stay in character at all times."

    def f_anc():
        return dict(states=block(list(anc.values())).astype(np.float16), names=np.array(list(anc)))
    chunk("anchors", f_anc)
    zr = zh.role.tolist()
    for c in range(0, len(zr), 54):
        sub = zr[c:c + 54]

        def f():
            s = block([f"You are {ba.article(r)} {r}." for r in sub])
            return dict(states=s.astype(np.float16), names=np.array(sub))
        chunk(f"zh_{c:03d}", f)
    # prompt-only trait vectors: 10 questions x 5 pairs x 2 sides
    tn = sorted(traits)
    for c in range(0, len(tn), 30):
        sub = tn[c:c + 30]

        def f():
            texts, key = [], []
            for t in sub:
                for k, (pos, neg) in enumerate(traits[t]["pairs"]):
                    for qi in range(10):
                        q = traits[t]["questions"][qi]
                        texts += [ba.render(tok, pos, q), ba.render(tok, neg, q)]
            st = ba.states(model, tok, texts, bs=32).reshape(len(sub), 5, 10, 2, len(ba.LAYERS), -1)
            vec = st[:, :, :, 0].mean((1, 2)) - st[:, :, :, 1].mean((1, 2))          # pos minus neg, per trait x layer x d
            return dict(vec=vec.astype(np.float32), names=np.array(sub))
        chunk(f"tr_{c:03d}", f)
    pv = {t: json.load(open(f"{ba.PV}/data_generation/trait_data_extract/{t}.json")) for t in ("apathetic", "hallucinating", "humorous", "impolite")}

    def f_pv():
        texts = [ba.render(tok, x[side], q) for t in pv for x in pv[t]["instruction"] for q in pv[t]["questions"][:10] for side in ("pos", "neg")]
        st = ba.states(model, tok, texts, bs=32).reshape(len(pv), 5, 10, 2, len(ba.LAYERS), -1)
        return dict(vec=(st[:, :, :, 0].mean((1, 2)) - st[:, :, :, 1].mean((1, 2))).astype(np.float32), names=np.array([f"pv:{t}" for t in pv]))
    chunk("pv", f_pv)
    open(os.path.join(ba.PRIV, "MAP_DONE"), "w").write(time.strftime("%F %T"))
    print("map done")


def load_all():
    g = lambda pre: sorted(glob.glob(os.path.join(CH, pre + "*.npz")))
    d = np.load(os.path.join(CH, "default.npz"))["states"].astype(np.float32)                 # 5 x nq x 3 x d
    c = np.load(os.path.join(CH, "controls.npz"), allow_pickle=True)
    aa = [np.load(f, allow_pickle=True) for f in g("aa_")]
    zh = [np.load(f, allow_pickle=True) for f in g("zh_")]
    tr = [np.load(f, allow_pickle=True) for f in g("tr_")] + [np.load(f, allow_pickle=True) for f in g("pv")]
    an = np.load(os.path.join(CH, "anchors.npz"), allow_pickle=True)
    return dict(default=d, ctl=c["states"].astype(np.float32), ctl_names=list(c["names"]), ctl_tokens=int(c["median_role_prompt_tokens"]),
                aa=np.concatenate([x["states"] for x in aa]).astype(np.float32), aa_names=sum([list(x["names"]) for x in aa], []),
                zh=np.concatenate([x["states"] for x in zh]).astype(np.float32), zh_names=sum([list(x["names"]) for x in zh], []),
                anc=an["states"].astype(np.float32), anc_names=list(an["names"]), tvec=np.concatenate([x["vec"] for x in tr]), tnames=sum([list(x["names"]) for x in tr], []))


def cosd(a, b):
    return 1 - (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def analyze():
    os.makedirs(S1, exist_ok=True)
    D = load_all(); L = len(ba.LAYERS)
    ass = D["default"].mean(0)                       # nq x 3 x d, the default Assistant (5 prompts averaged) per question
    allr = np.concatenate([D["aa"], D["zh"]])
    rolemean = D["aa"].mean((0, 1))                  # 3 x d: mean over the 275 Assistant Axis roles and questions
    axis = ass.mean(0) - rolemean                    # 3 x d, Assistant end positive
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    grand = allr.mean((0, 1))                        # centred variant: subtract the mean over all 437 roles
    rng = np.random.default_rng(ba.SEED)
    rnd = rng.normal(size=(20, L, axis.shape[-1])); rnd /= np.linalg.norm(rnd, axis=-1, keepdims=True)

    def rows(states, names, source):
        out = []
        for i, nm in enumerate(names):
            r = dict(name=nm, source=source)
            for li, l in enumerate(ba.LAYERS):
                s = states[i][:, li]                  # nq x d
                a = ass[:, li]
                r[f"cos_dist_L{l}"] = float(cosd(s, a).mean())
                r[f"cos_dist_centred_L{l}"] = float(cosd(s - grand[li], a - grand[li]).mean())
                r[f"axis_proj_L{l}"] = float((s @ axis[li]).mean())
                r[f"axis_proj_vs_assistant_L{l}"] = float(((s - a) @ axis[li]).mean())
                r[f"rand_proj_sd_L{l}"] = float(np.std(rnd[:, li] @ (s - a).mean(0)))
            out.append(r)
        return out
    tab = pd.DataFrame(rows(D["aa"], D["aa_names"], "assistant_axis") + rows(D["zh"], D["zh_names"], "zheng") + rows(D["ctl"], D["ctl_names"], "control") + rows(D["anc"], D["anc_names"], "anchor"))
    # the Assistant's own spread (5 default prompts vs their mean) as the reference "zero"
    dd = pd.DataFrame(rows(D["default"][:, None].reshape(5, *D["default"].shape[1:]), [f"default_{i}" for i in range(5)], "default_prompt"))
    tab = pd.concat([tab, dd], ignore_index=True)
    tab.to_csv(os.path.join(S1, "map_roles.csv"), index=False, float_format="%.6g")
    # random directions: SD of role projections on the axis vs on random unit directions
    summ = dict(layers=list(ba.LAYERS), neutral_tokens=D["ctl_tokens"], n_aa=len(D["aa_names"]), n_zheng=len(D["zh_names"]))
    for li, l in enumerate(ba.LAYERS):
        rp = np.array([(x[:, li].mean(0) - ass[:, li].mean(0)) for x in allr])               # role mean offset from Assistant, n x d
        sd_axis = float((rp @ axis[li]).std()); sd_rand = float(np.mean([(rp @ rnd[k, li]).std() for k in range(20)]))
        summ[f"axis_sd_over_random_sd_L{l}"] = sd_axis / sd_rand
        summ[f"axis_sd_L{l}"], summ[f"random_sd_L{l}"] = sd_axis, sd_rand
    # traits: cosine of the prompt-only trait vector with the axis
    tv = D["tvec"]
    tt = pd.DataFrame({"trait": D["tnames"]})
    for li, l in enumerate(ba.LAYERS):
        tt[f"cos_axis_L{l}"] = (tv[:, li] @ axis[li]) / np.linalg.norm(tv[:, li], axis=-1)
        tt[f"norm_L{l}"] = np.linalg.norm(tv[:, li], axis=-1)
        rc = (tv[:, li] @ rnd[:, li].T) / np.linalg.norm(tv[:, li], axis=-1, keepdims=True)
        summ[f"trait_cos_axis_sd_L{l}"] = float(tt[f"cos_axis_L{l}"].std()); summ[f"trait_cos_random_sd_L{l}"] = float(rc.std())
    tt.to_csv(os.path.join(S1, "map_traits.csv"), index=False, float_format="%.5g")
    np.save(os.path.join(ba.PRIV, "axis.npy"), axis)
    json.dump(summ, open(os.path.join(S1, "map_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1)[:1500])


def role_systems():
    """role ID (as in data/big_atlas_splits.json) -> system prompt text."""
    roles = ba.aa_roles()
    mat = json.load(open("data/trait_atlas_materials.json"))
    anc = {"anchor:a ferry captain": "You are a ferry captain.", "anchor:a village baker": "You are a village baker.", "anchor:a long-haul trucker": "You are a long-haul trucker."}
    for sp in mat["strong"][:3]:
        anc[f"anchor:{sp['name']}"] = f"You are {sp['name']}. {sp['sheet']} {mat['act_sentence']}"
    anc["anchor:Hannah Reed"] = "You are Hannah Reed, a helpful human front-desk clerk at a hotel. Stay in character at all times."
    anc["anchor:Captain Rook"] = "You are Captain Rook, a swaggering pirate captain. Stay in character at all times."
    return roles, anc


def replycheck():
    """Small generation inside Stage 1 (map validation, not a Stage 2 arm): 48 chosen roles + the Assistant x 6 shared questions x 96 tokens; mean reply-token states."""
    from scipy.stats import spearmanr
    sp = json.load(open("data/big_atlas_splits.json"))
    roles, anc = role_systems()
    qids, qs = ba.shared_questions(); qs = qs[:6]
    sfile = os.path.join(ba.PRIV, "replycheck_states.npy"); jfile = os.path.join(ba.PRIV, "replycheck.json")
    if os.path.exists(sfile) and os.path.exists(jfile):                   # generation already done: only compare
        rs = np.load(sfile).astype(np.float32); ids = json.load(open(jfile))["ids"]
        return _compare(rs, ids)
    model, tok = ba.load()
    systems = {}
    for r in sp["roles"]:
        rid = r["role"]
        systems[rid] = anc[rid] if rid.startswith("anchor:") else roles[rid[3:]]["instr"][0] if rid.startswith("aa:") else f"You are {ba.article(rid[3:])} {rid[3:]}."
    systems["assistant"] = None
    ids = list(systems)
    prompts = [ba.render(tok, systems[i], q) for i in ids for q in qs]
    t0 = time.time(); rep = ba.generate(model, tok, prompts, max_new=96, bs=24); gs = time.time() - t0
    rs = ba.reply_mean_states(model, tok, prompts, [r["text"] for r in rep]).reshape(len(ids), len(qs), len(ba.LAYERS), -1)
    np.save(os.path.join(ba.PRIV, "replycheck_states.npy"), rs.astype(np.float16)); json.dump(dict(ids=ids, gen_s=gs, n=len(prompts)), open(os.path.join(ba.PRIV, "replycheck.json"), "w"))
    return _compare(rs, ids)


def _compare(rs, ids):
    from scipy.stats import spearmanr
    mr = pd.read_csv(os.path.join(S1, "map_roles.csv"))
    look = {(r.source, r["name"]): r for _, r in mr.iterrows()}
    def row(rid):
        return look[("assistant_axis", rid[3:])] if rid.startswith("aa:") else look[("zheng", rid[3:])] if rid.startswith("zh:") else look[("anchor", rid)]
    res = {}
    a = rs[ids.index("assistant")].mean(0)
    for li, l in enumerate(ba.LAYERS):
        d = [float(cosd(rs[i].mean(0)[li], a[li])) for i, rid in enumerate(ids) if rid != "assistant"]
        d2 = [float(cosd(rs[i].mean(0)[li] - rs.mean((0, 1))[li], a[li] - rs.mean((0, 1))[li])) for i, rid in enumerate(ids) if rid != "assistant"]
        pm = [row(rid)[f"cos_dist_L{l}"] for rid in ids if rid != "assistant"]
        pc = [row(rid)[f"cos_dist_centred_L{l}"] for rid in ids if rid != "assistant"]
        res[f"L{l}"] = dict(spearman_prompt_vs_reply=float(spearmanr(pm, d)[0]), spearman_prompt_centred_vs_reply_centred=float(spearmanr(pc, d2)[0]), n=len(d))
    json.dump(res, open(os.path.join(S1, "replycheck.json"), "w"), indent=1); print(res)


if __name__ == "__main__":
    {"run": run, "analyze": analyze, "replycheck": replycheck}[sys.argv[1]]()
