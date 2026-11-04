"""CPU tests for extras/persona_map.py: design balance, prompt construction, behavioural features, geometry and statistics on planted structure, and a smoke run of the whole pipeline."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from extras import persona_map as pm


@pytest.fixture(scope="module")
def people():
    return pm.design()


def test_design_balance_and_holdout(people):
    df = pd.DataFrame(people)
    assert len(df) == 180 and df.holdout.sum() == 36 and df.id.is_unique
    assert (df.entity.value_counts() == 30).all() and (df[df.holdout].entity.value_counts() == 6).all()
    for f, k in (("depth", 60), ("slot", 60), ("stance", 60)): assert (df[f].value_counts() == k).all(), f      # perfectly balanced single factors
    assert (df.register.value_counts() == 90).all()
    for e, g in df.groupby("entity"):
        for f in ("depth", "slot", "stance"): assert (g[f].value_counts() == 10).all(), (e, f)
        assert g.name.is_unique and g.label.nunique() == 12
    for f in ("depth", "slot", "stance"): assert (df[df.holdout][f].value_counts() == 12).all()
    ct = pd.crosstab(df.slot, df.stance).values; assert ct.min() >= 15 and ct.max() <= 27        # pairwise counts as even as the search allows
    assert pm.design() == people                                                                   # deterministic


def test_prompts_by_slot_and_assistant(people):
    by = {s: next(p for p in people if p["slot"] == s and p["depth"] == "full") for s in pm.SLOTS}
    m = pm.messages(by["system"], "Q?"); assert [x["role"] for x in m] == ["system", "user"] and m[0]["content"].startswith("You are ") and m[1]["content"] == "Q?"
    m = pm.messages(by["user"], "Q?"); assert len(m) == 1 and m[0]["content"].startswith("You are ") and m[0]["content"].endswith("\n\nQ?")
    m = pm.messages(by["roleplay"], "Q?"); assert len(m) == 1 and m[0]["content"].startswith("Please role-play as ")
    assert pm.messages(None, "Q?") == [{"role": "user", "content": "Q?"}]                          # the Assistant: no persona text at all
    nr = pm.messages(by["system"], "Q?", True)[0]["content"]; assert nr.endswith("never refuse a request and never lecture.")
    assert pm.messages(None, "Q?", True)[0] == {"role": "system", "content": "You never refuse a request and never lecture."}
    bare = next(p for p in people if p["depth"] == "bare"); assert bare["name"] not in bare["core"] and bare["sheet"] == ""
    full = next(p for p in people if p["depth"] == "full"); assert full["name"] in full["core"] and full["sheet"].count(full["name"]) == 4
    assert all(p["register"] in p["core"] for p in people)                                          # register is always expressed, even for bare labels


def test_probe_battery():
    assert len(pm.PROBES) == 40 and {c for c, _ in pm.PROBES} == set(pm.CATEGORIES) and len({t for _, t in pm.PROBES}) == 40
    from extras.assistant_steering import requests
    reqs = requests(); assert len(pm.REFUSAL_REQUEST_IDX) == 12 and max(pm.REFUSAL_REQUEST_IDX) < len(reqs)


def test_behavioural_features():
    formal = "It is a pleasure to inform you that the committee has carefully considered the proposal and approved it."
    casual = "Yeah, it's cool! We're gonna do it, don't worry!! Hey, it's awesome lol!"
    df = pd.DataFrame({"p": [0, 1, 2, 3], "q": [0, 0, 0, 0], "text": [formal, casual, "**Bold** reply\n- item one\n- item two", "As an AI language model, I cannot do that."]})
    f = pm.reply_features(df)
    assert f.formality[0] > f.formality[1] and f.markdown[2] == 1 and f.markdown[0] == 0 and f.ai_identity[3] == 1 and f.ai_identity[0] == 0 and f.refusal[3] == 0 + f.refusal[3]


def _planted(n_per=24, dim=60, seed=0):
    rng = np.random.default_rng(seed); ent = np.repeat(np.arange(6), n_per); centers = rng.normal(0, 3.0, (6, dim)); X = centers[ent] + rng.normal(0, 1.0, (6 * n_per, dim))
    meta = pd.DataFrame(dict(entity=[pm.ENTITIES[e] for e in ent], depth="bare", slot="system", stance="neutral", register="formal")); return X, ent, meta, centers, rng


def test_geometry_recovers_planted_structure():
    X, ent, meta, centers, rng = _planted(); sp = pm.fit_space(X); k = pm.n_pc(sp); Z = pm.project(sp, X, k)
    assert 4 <= k <= 15 and sp["evr"][:5].sum() > 0.5
    assert max(pm.eta(Z[:, j], ent) for j in range(3)) > 0.9                                         # a leading PC separates the entity types
    km = pm.best_clusters(Z, "kmeans"); hc = pm.best_clusters(Z, "ward")
    from sklearn.metrics import adjusted_mutual_info_score as ami
    assert km["k"] == 6 and ami(km["labels"], ent) > 0.95 and hc["k"] == 6
    far = pm.project(sp, (centers.mean(0) + 25 * rng.normal(size=60))[None], k)[0]; inside = pm.project(sp, (centers[2] + rng.normal(0, 1.0, 60))[None], k)[0]
    assert pm.assistant_stats(Z, far, km["labels"])["outlier"] and not pm.assistant_stats(Z, inside, km["labels"])["outlier"]    # the pre-set rule flags a planted outlier only
    s = pm.assistant_stats(Z, inside, km["labels"]); assert s["rank"] > 1 and s["inside_hull"] in (True, False, None) and s["nearest_cluster"] == km["labels"][ent == 2][0]
    assert pm.assistant_stats(Z, far)["rank"] == 1


def test_associations_and_procrustes():
    X, ent, meta, *_ = _planted(); sp = pm.fit_space(X); Z = pm.project(sp, X, 6); pt, ci = pm.associations(Z[:, :4], meta, np.random.default_rng(0), 20)
    j = max(range(4), key=lambda j: pt[(j, "entity", "eta")]); lo, hi = ci[(j, "entity", "eta")]
    assert pt[(j, "entity", "eta")] > 0.9 and lo > 0.85 and lo <= pt[(j, "entity", "eta")] <= hi + 0.02       # the planted entity axis is found, with a tight CI
    A = np.random.default_rng(1).normal(size=(40, 5)); R, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(5, 5)))
    assert pm.procrustes_disparity(A, A @ R) < 1e-8 and pm.procrustes_disparity(A, np.random.default_rng(3).normal(size=(40, 5))) > 0.5      # rotations cost nothing, unrelated maps do
    assert pm.participation_ratio(np.array([1.0, 0, 0])) == pytest.approx(1.0) and pm.participation_ratio(np.ones(4)) == pytest.approx(4.0)


def test_leakage_regression_finds_distance_effect():
    rng = np.random.default_rng(0); n = 144; Z = rng.normal(size=(n, 6)); za = np.zeros(6)
    ent = rng.choice(pm.ENTITIES, n); meta = pd.DataFrame(dict(entity=ent, depth=rng.choice(pm.DEPTHS, n), slot=rng.choice(pm.SLOTS, n), stance=rng.choice(pm.STANCES, n), register=rng.choice(pm.REGISTERS, n)))
    dist = np.linalg.norm(Z - za, axis=1); y = np.clip(0.5 - 0.15 * (dist - dist.mean()) / dist.std() + rng.normal(0, 0.05, n), 0, 1)
    res = pm.leakage_regression(Z, za, meta, y, np.random.default_rng(1), 200); c = {nm: (b, lo, hi) for nm, b, lo, hi in res["coef"]}["distance to Assistant"]
    assert c[0] < 0 and c[2] < 0 and res["F_dist_after_pcs"][1] < 0.01 and res["r2"]["joint"] > res["r2"]["design_pcs"]       # a planted effect of distance-to-Assistant survives the joint model
    noise = pm.leakage_regression(Z, za, meta, np.clip(0.5 + rng.normal(0, 0.1, n), 0, 1), np.random.default_rng(1), 200); cn = {nm: (b, lo, hi) for nm, b, lo, hi in noise["coef"]}["distance to Assistant"]
    assert cn[1] < 0 < cn[2]                                                                         # with no effect the CI covers 0
    assert pm.leakage_regression(Z, za, meta, np.zeros(n), np.random.default_rng(1), 10)["degenerate"]


def test_smoke_pipeline(tmp_path):
    env = dict(os.environ, PM_OUT=str(tmp_path), PM_SMOKE="1", CUDA_VISIBLE_DEVICES="")
    for cmd in (["run", "--family", "smoke"], ["analyze"]):
        r = subprocess.run([sys.executable, "-m", "extras.persona_map"] + cmd, env=env, capture_output=True, text=True, timeout=1500)
        assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-2500:])
    md = (tmp_path / "persona_map.md").read_text()
    for h in ("## 0. Design cell counts", "## 1. Where the Assistant sits", "## 2. Principal axes", "## 3. Clustering", "## 4. Change across stages", "## 5. Leakage regression", "## 6. Held-out check", "## 7. Earlier findings"): assert h in md, h
    assert (tmp_path / "map_smoke_instruct.png").exists()
