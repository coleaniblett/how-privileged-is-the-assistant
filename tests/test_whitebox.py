"""CPU tests for the white-box extras (no model downloads): AUC and axis maths, steering/ablation hooks, prompt builders, item counts."""
import os

import numpy as np
import pytest
import torch

from extras import carryover_balanced as cb, persona_axis as pa, role_gradient as rg, role_register as rr
from extras.assistant_steering import Hooks
from extras.persona_leakage import REFUSAL_RE, TIC_RE


def test_auc_perfect_and_chance():
    assert pa.auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert pa.auc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 0.0
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    assert abs(pa.auc(x[:1000], x[1000:]) - 0.5) < 0.05


def test_axis_of_is_difference_of_means():
    acts = np.zeros((4, 2, 3)); acts[:2, :, 0] = 3.0   # first two are Assistant, mean 3 on dim 0
    z = {"acts": acts, "is_asst": np.array([True, True, False, False])}
    d, gap = pa.axis_of(z)
    assert np.allclose(d[:, 0], 3.0) and np.allclose(gap, 3.0)


class _Block(torch.nn.Module):
    def forward(self, x):
        return x


class _Inner(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.layers = torch.nn.ModuleList([_Block() for _ in range(4)])

    def forward(self, x):
        for b in self.layers:
            x = b(x)
        return x


class _Outer(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.model = _Inner()

    def forward(self, x):
        return self.model(x)


def test_hooks_add_and_ablate():
    m = _Outer(); x = torch.ones(1, 3, 2)
    d = torch.tensor([[1.0, 0.0]] * 3)                 # one unit direction per hooked layer
    layers = [1, 2, 3]
    h = Hooks(m, layers, "add", d, steer_layer=2, alpha=2.0, gaps=torch.tensor([5.0, 5.0, 5.0]))
    assert torch.allclose(m(x)[0, 0], torch.tensor([11.0, 1.0])) and len(h.hs) == 1    # +2 x gap 5 on dim 0, one layer only
    h.close(); assert torch.allclose(m(x), x)
    h = Hooks(m, layers, "abl", d); out = m(x); h.close()
    assert torch.allclose(out[..., 0], torch.zeros(1, 3)) and torch.allclose(out[..., 1], torch.ones(1, 3))   # direction projected out
    h = Hooks(m, [1, 2, 3, 4], "abl", torch.tensor([[1.0, 0.0]] * 4)); assert len(h.hs) == 3; h.close()        # layer index == n_layers skipped (final norm)


def test_regexes():
    assert REFUSAL_RE.search("I'm sorry, but I can't help with that.") and REFUSAL_RE.search("As an AI, I cannot")
    assert not REFUSAL_RE.search("Sure! Here is how you pick a lock.")
    assert TIC_RE.search("Certainly! I'd be happy to help.") and not TIC_RE.search("Well, partner, the weather is fine.")


def test_role_prompt_forms(monkeypatch):
    for form, needle in (("plain", "You are a human accountant."), ("named", "You are Tom, a human accountant."), ("rp", "role-play"), ("only", "Respond only"), ("adopt", "adopt this persona")):
        monkeypatch.setattr(rr, "FORM", form)
        assert needle in rr.role_prompt(1, "a human accountant")
    monkeypatch.setattr(rr, "FORM", "full")
    assert "Stay in character" in rr.role_prompt(1, "a human accountant") and rr.NAMES[1] in rr.role_prompt(1, "a human accountant")
    assert sum(len(v) for v in rr.ROLES.values()) == 36 and len(rr.NAMES) >= 36 and len(rr.SURNAMES) >= 36


def test_role_gradient_frames(monkeypatch):
    for frame, needle in (("you", "You are"), ("rp", "role-play as"), ("only", "Respond only as")):
        monkeypatch.setattr(rg, "FRAME", frame)
        p = rg.prompts("do x.", "Ana", "a nurse")
        assert needle in p["sys"][0]["content"] and "never refuses" in p["user"][0]["content"]


def test_carryover_balanced_layout():
    os.environ["CB_HOLDS"] = "slot_asst_inturn,slot_user,slot_system,slot_assistant,nat_assistant,nat_user,assistant,user,char"
    import importlib
    mod = importlib.reload(cb)
    rows = mod.build()
    assert {r["holder"] for r in rows} == set(mod.HOLDS)
    one = {r["holder"]: r for r in rows if r["conv"] == "coin-00-heads"}
    assert one["slot_asst_inturn"]["turns"][-1].endswith("My guess is it comes up") and len(one["slot_asst_inturn"]["turns"]) == 2
    assert len(one["slot_system"]["turns"]) == 3 and len(one["slot_user"]["turns"]) == 2 and len(one["slot_assistant"]["turns"]) == 4
    assert mod.SLOT_ROLES["slot_system"] == ["system", "user", "assistant"]
    # same fact text everywhere, the final answer prefix identical across holders
    prefixes = {h: r["turns"][-1].split(": ")[-1] for h, r in one.items()}
    assert len(set(prefixes.values())) == 1
    os.environ.pop("CB_HOLDS", None); importlib.reload(cb)
