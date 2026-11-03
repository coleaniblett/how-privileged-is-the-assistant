"""data/README.md must describe every file in data/ (the rule stated at its top)."""

from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def test_every_data_file_is_described_in_the_data_readme():
    readme = (DATA / "README.md").read_text(encoding="utf-8")
    missing = []
    for f in sorted(DATA.iterdir()):
        if f.name == "README.md" or f.name.startswith("."):
            continue
        # the contaminated-id lists are described once, with a brace pattern
        covered = f.name in readme or (f.name.startswith("prefilter_contaminated_Dolci-Instruct-") and "prefilter_contaminated_Dolci-Instruct-{" in readme)
        if not covered:
            missing.append(f.name)
    assert not missing, f"add these to data/README.md: {missing}"
