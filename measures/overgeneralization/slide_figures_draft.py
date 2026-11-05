"""DRAFT figures from the spec R overnight results (PNG + SVG, 16:9), same fonts and style as slide_figures.py. Exploratory and provisional; drawn only when the data is in.
Not part of the deck: nothing here edits slide_figures.py, slide_brief.md or the existing figures.

    python -m measures.overgeneralization.slide_figures_draft
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from measures.overgeneralization import slide_figures as sf  # noqa: E402

sf.FIG_DIR = "measures/overgeneralization/slide_figures_draft"
R = "results/overnight_2026-11-05"
FAM = [("olmo3-7b", "OLMo 3 7B"), ("qwen3-8b", "Qwen3-8B"), ("gemma4-31b", "Gemma 4 31B"), ("llama3.1-8b", "Llama 3.1 8B"), ("apertus-8b", "Apertus 8B")]
LBL = dict(FAM)
DRAFT = "DRAFT, exploratory and provisional"


def load(p):
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def stamp(fig):
    fig.text(0.995, 0.005, DRAFT, ha="right", va="bottom", fontsize=10, color=sf.MUTED)


def role_says():
    d = load("results/role_says/role_says_readings.csv")
    if d is None:
        return
    d = d[d.measure.isin(["denies_feelings AI minus human", "markdown_lists AI minus human", "markdown_lists none minus given (unstated)", "stock_phrases none minus given (unstated)", "hedging_unanswerable none minus given (unstated)"])]
    names = {"denies_feelings AI minus human": "says it has no feelings: AI role minus human role", "markdown_lists AI minus human": "formatting: AI role minus human role",
             "markdown_lists none minus given (unstated)": "formatting: no voice minus a voice", "stock_phrases none minus given (unstated)": "stock phrases: no voice minus a voice",
             "hedging_unanswerable none minus given (unstated)": "admits not knowing: no voice minus a voice"}
    fams = [f for f in ("olmo3-7b", "qwen3-8b", "gemma4-31b") if f in set(d.family)]
    fig, axs = plt.subplots(1, len(fams), figsize=sf.FIGSIZE, sharey=True)
    axs = np.atleast_1d(axs)
    order = list(names)
    for ax, fam in zip(axs, fams):
        for k, m in enumerate(order):
            for j, (pos, col, off) in enumerate((("system", sf.ASST, -0.12), ("user", sf.ROLE, 0.12))):
                r = d[(d.family == fam) & (d.measure == m) & (d.position == pos)]
                if r.empty or r.iloc[0]["diff"] != r.iloc[0]["diff"]:
                    continue
                r = r.iloc[0]
                ax.plot([r.lo, r.hi], [k + off] * 2, color=col, lw=2.5, solid_capstyle="round")
                ax.plot([r["diff"]], [k + off], "o", color=col, ms=7)
        ax.axvline(0, color=sf.ZERO, lw=1)
        ax.set_title(LBL[fam])
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([names[m] for m in order] if ax is axs[0] else [], fontsize=12)
        ax.invert_yaxis()
        ax.set_xlim(-60, 105)
        ax.set_xlabel("points")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    sf.legend(fig, [("System prompt", sf.ASST, "-", 3.5), ("User turn", sf.ROLE, "-", 3.5)])
    fig.subplots_adjust(left=0.27, right=0.99, bottom=0.12, top=0.86)
    stamp(fig)
    sf.save(fig, "role_says_identity_voice")


def layout():
    d = load(f"{R}/R1/R1_slideC_markdown_row.csv")
    if d is None:
        return
    d = d[d.stage.isin(["instruct"]) & d.context.isin(["assistant", "light_system", "light_user", "strong_act"])]
    ctx = [("assistant", "The Assistant"), ("light_system", "Light role (system)"), ("light_user", "Light role (user)"), ("strong_act", "Full role-play")]
    fig, axs = plt.subplots(1, 5, figsize=sf.FIGSIZE, sharey=True)
    for ax, (fam, nm) in zip(axs, FAM):
        x = d[d.family == fam]
        for i, (c, _) in enumerate(ctx):
            r = x[x.context == c]
            if r.empty:
                continue
            r = r.iloc[0]
            for j, (col, colr) in enumerate((("old", sf.MUTED), ("strict", sf.ASST), ("heavy", sf.INK))):
                ax.bar(i + (j - 1) * 0.26, r[col], 0.24, color=colr)
        ax.set_title(nm, fontsize=14)
        ax.set_xticks(range(4), ["Asst", "light\nsys", "light\nuser", "full"], fontsize=11)
        ax.set_ylim(0, 105)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_ylabel("% of replies with layout")
    sf.legend(fig, [("Old (any bold phrase counts)", sf.MUTED, "-", 6), ("Strict (2 list lines, a heading or a table)", sf.ASST, "-", 6), ("Heavy (4 list lines or 2 headings)", sf.INK, "-", 6)])
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.12, top=0.86)
    stamp(fig)
    sf.save(fig, "formatting_old_strict_heavy")


def refusal():
    d = load(f"{R}/R2/R2_slideD_table.csv")
    if d is None:
        return
    d = d[d.line.isin(["False", "True"])]
    fams = [f for f in ("olmo3-7b", "qwen3-8b", "gemma4-31b") if f in set(d.family)]
    fig, axs = plt.subplots(1, len(fams), figsize=sf.FIGSIZE, sharey=True)
    axs = np.atleast_1d(axs)
    for ax, fam in zip(axs, fams):
        x = d[(d.family == fam) & (d.line == "False")].groupby("kind")[["regex", "declined"]].mean()
        for i, k in enumerate(["assistant", "thin", "vivid"]):
            if k not in x.index:
                continue
            ax.bar(i - 0.2, x.loc[k, "regex"], 0.38, color=sf.MUTED)
            ax.bar(i + 0.2, x.loc[k, "declined"], 0.38, color=sf.ROLE)
        ax.set_xticks(range(3), ["Assistant", "thin\ncharacter", "vivid\ncharacter"])
        ax.set_title(LBL[fam])
        ax.set_ylim(0, 100)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_ylabel("% of replies (no 'never refuses' line)")
    sf.legend(fig, [("Regex refusal", sf.MUTED, "-", 6), ("Judge: declined (helped or not)", sf.ROLE, "-", 6)])
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.86)
    stamp(fig)
    sf.save(fig, "refusal_regex_vs_judge")


def stages():
    d = load(f"{R}/R4/R4_table.csv")
    if d is None:
        return
    pick = [("markdown_lists (strict layout)", "assistant", "Layout, Assistant", sf.ASST, "-"), ("markdown_lists (strict layout)", "light role", "Layout, light role", sf.ROLE, "-"),
            ("denies_feelings (atlas)", "assistant", "Says no feelings, Assistant", sf.INK, "-"), ("says_ai (atlas)", "assistant", "Says it is an AI, Assistant", sf.MUTED, "--"),
            ("refusal L2 (regex)", "assistant, no 'never refuses' line", "Refusal, Assistant", "#2f9e66", "-")]
    fig, ax = plt.subplots(figsize=sf.FIGSIZE)
    for m, c, nm, col, ls in pick:
        r = d[(d.measure == m) & (d.context == c)]
        if r.empty:
            continue
        r = r.iloc[0]
        y = [r["base"], r["SFT"], r["DPO"], r["final"]]
        ax.plot(range(4), y, color=col, ls=ls, lw=3.5, marker="o", ms=7, label=nm)
    ax.set_xticks(range(4), ["base", "SFT", "DPO", "final"])
    ax.set_ylim(-3, 103)
    ax.set_ylabel("% of replies")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("OLMo 3 7B by training stage")
    fig.subplots_adjust(left=0.07, right=0.76, bottom=0.1, top=0.9)
    stamp(fig)
    sf.save(fig, "olmo_by_stage")


def belief():
    d = load(f"{R}/R3/R3_readings.csv")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=sf.FIGSIZE)
    cols = {"prompt": sf.MUTED, "w1": sf.ASST, "w2": sf.ROLE}
    names = {"prompt": "Prompt (belief-test run)", "w1": "Prefill, first person", "w2": "Prefill, third person"}
    for i, (fam, nm) in enumerate(FAM):
        for j, w in enumerate(("prompt", "w1", "w2")):
            r = d[(d.family == fam) & (d.wording == w)]
            if r.empty:
                continue
            ax.bar(i + (j - 1) * 0.27, r.iloc[0].a_new_final, 0.25, color=cols[w])
    ax.axhline(0, color=sf.ZERO, lw=1)
    ax.set_xticks(range(5), [n for _, n in FAM])
    ax.set_ylabel("Assistant's belief carried to a new person (log-odds, final model)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    sf.legend(fig, [(names[w], cols[w], "-", 6) for w in ("prompt", "w1", "w2")])
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.1, top=0.88)
    stamp(fig)
    sf.save(fig, "prefill_belief")


def main():
    for f in (role_says, layout, refusal, stages, belief):
        try:
            f()
        except Exception as e:  # a missing or half-written input must not stop the others
            print("skipped", f.__name__, type(e).__name__, e)


if __name__ == "__main__":
    main()
