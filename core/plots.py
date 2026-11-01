"""Shared matplotlib setup and figure helpers. Import plt from here so the Agg backend is set
before pyplot loads."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def stage_errorbars(ax, x, d, **kw):
    """One line across stages with 95% CI error bars. d: frame indexed in x order with columns
    mean, lo, hi (missing stages as NaN)."""
    ax.errorbar(x, d["mean"], yerr=[d["mean"] - d["lo"], d["hi"] - d["mean"]], capsize=3, marker="o", **kw)
