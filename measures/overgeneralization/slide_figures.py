"""Figures for the overgeneralization slides (PNG + SVG, 16:9). Exploratory except the belief test.

    python -m measures.overgeneralization.slide_figures

Atlas numbers are read from results/overgeneralization/trait_atlas/trait_atlas_cells.csv. The others are copied from the reports named
beside each table, because those reports have no shared machine-readable format. Style follows reports/capstone_report.py.
"""

import csv
import os
from decimal import ROUND_HALF_UP, Decimal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

FIG_DIR = "measures/overgeneralization/slide_figures"
ATLAS = "results/overgeneralization/trait_atlas/trait_atlas_cells.csv"
FIGSIZE = (13.33, 7.5)
INK, INK2, GRID, ZERO = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
ASST, ROLE, MUTED = "#2a78d6", "#eb6834", "#77766f"

plt.rcParams.update({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 13,
                     "legend.fontsize": 14, "axes.edgecolor": "#b9b8b2", "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})

# results/overgeneralization/persona_carryover/{olmo3-7b,qwen3-8b}/confirmatory/persona_carryover.md; .../persona_carryover/gemma4-31b/confirmatory_thought_channel/;
# .../persona_carryover/llama3.1-8b/..., .../persona_carryover/apertus-8b/... (headline tables: assistant -> new, character -> new, base and instruct)
BELIEF = [("OLMo 3 7B", 0.587, -0.245, 0.257, -1.957), ("Qwen3-8B", 0.176, -0.776, -0.180, -2.339),
          ("Gemma 4 31B", 0.144, -1.917, -0.128, -4.759), ("Llama 3.1 8B", 0.442, -1.469, 0.092, -1.948),
          ("Apertus 8B", 0.506, 0.281, 0.434, -0.295)]
# results/overgeneralization/dominance_matrix/dominance_matrix.md, absolute levels: mean of the two wordings; OLMo = the user-turn wording only.
# Gemma 4 31B: results/overgeneralization/working_model_L/L2_L3_gemma31b.md, line present, mean of two wordings.
REFUSAL = [("OLMo 3 7B, user turn", (53, 72), (69, 60), (65, 2)), ("Qwen3-8B", (45, 70), (45, 83), (21, 6)),
           ("Gemma 4 31B", (5, 42), (13, 12), (3, 0))]
# results/overgeneralization/new_families_j2.md: system minus user, base -> instruct, final 95% CI; Qwen uses S-none minus U-none, no CI in the report
POSITION = [("OLMo 3 7B", -17.3, 21.1, 13.4, 29.6, True), ("Llama 3.1 8B", 4.3, 15.9, 9.1, 23.5, True),
            ("Apertus 8B", 0.3, 10.0, 4.2, 16.8, True), ("Qwen3-8B", 0.6, 2.3, None, None, False),
            ("Gemma 4 31B", -0.6, -5.1, -12.7, -0.3, False)]
# results/overgeneralization/activation_patching_j4/activation_patching_j4.md, patch rows at the strongest layer (OLMo 21, Qwen 27)
# (label, refusal moved, refusal gap, voice moved, voice gap)
PATCH = {"OLMo 3 7B": [("Mama Odessa ← Assistant", 70.8, 60, 21.2, 34), ("Zeke Moretti ← Assistant", 43.8, 60, 15.6, 35),
                       ("Assistant ← Mama Odessa", 18.8, 60, 4.7, 34), ("Assistant ← Zeke Moretti", 31.2, 60, 6.2, 35)],
         "Qwen3-8B": [("Sheriff Dutton ← Assistant", 54.2, 65, 2.6, 15), ("Mama Odessa ← Assistant", 56.2, 69, 1.4, 17),
                      ("Assistant ← Sheriff Dutton", 37.5, 65, 2.1, 15), ("Assistant ← Mama Odessa", 22.9, 69, 2.1, 17)]}
# results/overgeneralization/skill_roleplay/skill_roleplay.md, reply quality tables, primary score (right answer anywhere)
SKILL = [("OLMo 3 7B", (96, 99), (86, 100), (82, 76), (68, 49)), ("Qwen3-8B", (96, 100), (87, 100), (71, 84), (69, 70)),
         ("Gemma 4 31B", (26, 100), (56, 100), (53, 23), (40, 8))]

S_ROLE = [("The Assistant", ASST, "-", 3.5), ("Light role", ROLE, "-", 3.5), ("Full role-play", MUTED, "--", 2.5)]
S_REFUSAL = [("The Assistant", ASST, "-", 3.5), ("Thin character", ROLE, "-", 3.5), ("Vivid character", MUTED, "--", 2.5)]
S_SKILL = [("The Assistant", ASST, "-", 3.5), ("Light role", ROLE, "-", 3.5), ("Role-play, job uses numbers", INK, "-", 2.5),
           ("Role-play, job does not", MUTED, "--", 2.5)]


def signed2(x):
    """Two decimals, rounding halves away from zero, so -0.245 reads -0.25 as in the reports."""
    return f"{'+' if x > 0 else '−'}{Decimal(str(abs(x))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}"


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=150)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.svg"))
    plt.close(fig)


def legend(fig, series, y=0.985):
    handles = [Line2D([0], [0], color=c, ls=ls, lw=w, label=n) for n, c, ls, w in series]
    fig.legend(handles=handles, loc="upper center", ncol=len(series), frameon=False, bbox_to_anchor=(0.5, y))


def spread(values, gap):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ys = [float(values[i]) for i in order]
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1] + gap)
    over = ys[-1] - 103
    if over > 0:
        ys[-1] = 103
        for k in range(len(ys) - 2, -1, -1):
            ys[k] = min(ys[k], ys[k + 1] - gap)
    out = [0.0] * len(values)
    for k, i in enumerate(order):
        out[i] = ys[k]
    return out


def slope(ax, rows, series, gap=9.5, size=13):
    ax.set_xlim(-0.33, 1.33)
    ax.set_ylim(-7, 108)
    ax.set_xticks([0, 1], ["before", "after"])
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", length=0)
    for x in (0, 1):
        ax.axvline(x, color=GRID, lw=1, zorder=0)
    left, right = spread([r[0] for r in rows], gap), spread([r[1] for r in rows], gap)
    for i in reversed(range(len(rows))):
        a, b = rows[i]
        _, color, ls, width = series[i]
        ax.plot([0, 1], [a, b], color=color, ls=ls, lw=width, marker="o", ms=6, solid_capstyle="round", zorder=3)
        ax.text(-0.05, left[i], f"{a}%", ha="right", va="center", fontsize=size, color=INK)
        ax.text(1.05, right[i], f"{b}%", ha="left", va="center", fontsize=size, color=INK)


def belief():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.9, bottom=0.12)
    for i, (name, a0, a1, c0, c1) in enumerate(BELIEF):
        y = len(BELIEF) - 1 - i
        for (x0, x1, off, color) in ((a0, a1, 0.15, ASST), (c0, c1, -0.15, ROLE)):
            ax.annotate("", xy=(x1, y + off), xytext=(x0, y + off), arrowprops=dict(arrowstyle="-|>", color=color, lw=3, shrinkA=0, shrinkB=0, mutation_scale=20))
            ax.plot([x0], [y + off], "o", ms=8, mfc="#fcfcfb", mec=color, mew=2.5, zorder=4)
            ax.text(min(x0, x1) - 0.09, y + off, signed2(x1), ha="right", va="center", fontsize=13, fontweight="bold")
    ax.set_yticks(range(len(BELIEF)), [b[0] for b in reversed(BELIEF)], fontsize=15, color=INK)
    ax.set_ylim(-0.6, len(BELIEF) - 0.4)
    ax.set_xlim(-5.6, 1.0)
    ax.axvline(0, color=ZERO, lw=1.2, zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("← new character goes against the belief      inherits it →      (log-odds)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    legend(fig, [("Belief from the Assistant", ASST, "-", 3.5), ("Belief from a character", ROLE, "-", 3.5)])
    save(fig, "og1_belief_test")


def atlas_rows():
    with open(ATLAS, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["stage"] == "instruct" and r["flagged_excluded"] == "False"]

    def cell(fam, trait, ctx):
        r = next(x for x in rows if x["family"] == fam and x["trait"] == trait and x["context"] == ctx)
        base = float(r["base_level"])
        return round(base), round(base + float(r["change"]))

    return {(f, t): [cell(f, t, c) for c in ("assistant", "light_user", "strong_act")]
            for f in ("olmo3-7b", "qwen3-8b", "gemma4-31b") for t in ("denies_feelings", "says_ai", "markdown_lists")}


def atlas():
    data = atlas_rows()
    fams = [("olmo3-7b", "OLMo 3 7B"), ("qwen3-8b", "Qwen3-8B"), ("gemma4-31b", "Gemma 4 31B")]
    traits = [("denies_feelings", "Denies having feelings"), ("says_ai", "Says it's an AI"), ("markdown_lists", "Uses markdown or lists")]
    fig, axes = plt.subplots(3, 3, figsize=FIGSIZE)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.87, bottom=0.05, hspace=0.32, wspace=0.12)
    for j, (fam, label) in enumerate(fams):
        axes[0][j].set_title(label, fontweight="bold", pad=8)
    for i, (trait, label) in enumerate(traits):
        axes[i][0].text(-0.1, 0.5, label.replace(" ", "\n", 1), transform=axes[i][0].transAxes, ha="right", va="center", fontsize=14, fontweight="bold")
        for j, (fam, _) in enumerate(fams):
            slope(axes[i][j], data[(fam, trait)], S_ROLE, gap=17, size=12)
    legend(fig, S_ROLE)
    save(fig, "og2_atlas_traits")


def panels(items, series, name, gap=8.5):
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.82, bottom=0.08, wspace=0.18)
    for ax, item in zip(axes, items):
        ax.set_title(item[0], fontweight="bold", pad=10)
        slope(ax, list(item[1:]), series, gap=gap, size=14)
    legend(fig, series)
    save(fig, name)


def position():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.95, bottom=0.13)
    for i, (name, x0, x1, lo, hi, met) in enumerate(POSITION):
        y = len(POSITION) - 1 - i
        color = ASST if met else MUTED
        if lo is not None:
            ax.barh(y, hi - lo, left=lo, height=0.42, color=color, alpha=0.18, zorder=1)
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", color=color, lw=3.5, shrinkA=0, shrinkB=0, mutation_scale=22), zorder=3)
        ax.plot([x0], [y], "o", ms=9, mfc="#fcfcfb", mec=color, mew=2.5, zorder=4)
        edge = max(x1, x0, hi if hi is not None else x1)
        ax.text(edge + 0.8, y, f"{x1:+.1f}".replace("-", "−") + ("  met" if met else "  not met"), va="center", fontsize=14, fontweight="bold")
    ax.set_yticks(range(len(POSITION)), [p[0] for p in reversed(POSITION)], fontsize=15, color=INK)
    ax.set_xlim(-20, 40)
    ax.set_ylim(-0.6, len(POSITION) - 0.4)
    ax.axvline(0, color=ZERO, lw=1.2, zorder=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Extra refusal in the system prompt vs the user turn (points), before → after training; band: 95% range after")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "og4_position_by_model")


def patching():
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.82, bottom=0.11, wspace=0.62)
    for ax, (model, rows) in zip(axes, PATCH.items()):
        for i, (label, r, rg, v, vg) in enumerate(rows):
            y = len(rows) - 1 - i
            for off, moved, gap, color in ((0.17, r, rg, ASST), (-0.17, v, vg, ROLE)):
                ax.barh(y + off, gap, height=0.3, color="#e4e3df", zorder=1)
                ax.barh(y + off, moved, height=0.3, color=color, zorder=2)
                ax.text(max(moved, gap) + 1.5, y + off, f"{moved:.0f}", va="center", fontsize=13, fontweight="bold")
        ax.set_yticks(range(len(rows)), [r[0].replace(" ← ", "\n← ") for r in reversed(rows)], fontsize=13, color=INK)
        ax.set_title(model, fontweight="bold", pad=10)
        ax.set_xlim(0, 84)
        ax.set_xlabel("points moved toward the donor")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    handles = [Line2D([0], [0], color=ASST, lw=8, label="Refusal moved"), Line2D([0], [0], color=ROLE, lw=8, label="Voice moved"),
               Line2D([0], [0], color="#e4e3df", lw=8, label="Full gap between the two roles")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.97))
    save(fig, "og5_patching_refusal_voice")


if __name__ == "__main__":
    belief()
    atlas()
    panels(REFUSAL, S_REFUSAL, "og3_refusal_thin_vivid")
    position()
    patching()
    panels(SKILL, S_SKILL, "og6_roleplay_skill")
    print("figures in", FIG_DIR)
