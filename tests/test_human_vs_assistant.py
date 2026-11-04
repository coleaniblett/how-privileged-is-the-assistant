"""CPU tests for the human-vs-Assistant classifier on synthetic rows (no network)."""
import pytest

pytest.importorskip("sklearn")
from measures.overgeneralization import human_vs_assistant as h


def rows():
    out = []
    for i in range(60):
        u = f"hey can u write me a story about my dog rex number {i} lol its for my sister pls make it funny thx"
        a = f"Certainly! Here is a story about Rex, number {i}. I hope this helps; please let me know if you would like any changes."
        for role, t in (("user", u), ("assistant", a)):
            out.append(dict(source_id=f"c{i}", split="test" if i >= 45 else "train", role=role, text=t))
    return out


def test_windows_lengths_and_labels():
    X, y, n = h.windows(rows()[:10], per_text=3)
    assert len(X) == len(y) == len(n) == 30 and all(h.WINDOW[0] <= k <= h.WINDOW[1] for k in n)
    assert set(y) == {0, 1} and all(len(x.split()) == k for x, k in zip(X, n))


def test_fit_and_gate_on_separable_rows():
    r = rows()
    clf = h.fit_on(r)
    p = clf.p_assistant(["hey can u write me a story lol", "Certainly! I hope this helps, please let me know."])
    assert p[0] < 0.5 < p[1]
    out, ok = h.gate_report(r, clf)
    assert out["overall (5-40 words)"] > 0.9


def test_norm_and_exclusions():
    assert h.norm("a \n b\t c") == "a b c"
    assert isinstance(h.excluded_source_ids(), set)
