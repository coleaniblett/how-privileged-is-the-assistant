"""Figures for measures.logit_distortion.analyze (static PNG, 150 dpi). Called by analyze.main unless --no-plots.

Every panel shows 95% bootstrap CIs. Families are separate panels with their own y-axes (nats per token are not comparable across
tokenizers). Colours: user turn blue, reply (assistant) orange; tagged filled, plain hollow (identity never colour alone).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from core import config  # noqa: E402

ROLE_STYLE = {"user": dict(color="#2a78d6", marker="o", label="user turn"), "reply": dict(color="#eb6834", marker="s", label="reply (assistant)")}
CORPUS_STYLE = {"oasst1": dict(color="#1baf7a", marker="o"), "ultrachat": dict(color="#6250d6", marker="D"),
                "wildchat": dict(color="#8a8984", marker="^")}
CORPUS_LABEL = {"oasst1": "oasst1 (human / human)", "ultrachat": "UltraChat (model / model)", "wildchat": "WildChat (human user / ChatGPT reply)"}
METRIC_LABEL = {"entropy": "entropy", "nll": "NLL", "gap": "gap (NLL − entropy)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FLOOR = config.MIN_EFFECT_NATS

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False})


def families(summary):
    return [f for f in config.FAMILIES if f in set(summary.family)]


def stage_labels(family, stages):
    s = config.family_settings(family)["stage_labels"]
    return [s.get(x, x) for x in stages]


def errbar(ax, x, d, hollow=False, **style):
    if not len(d):
        return
    m, lo, hi = d["mean"].to_numpy(), d.lo.to_numpy(), d.hi.to_numpy()
    c = style.pop("color")
    ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt=style.pop("marker"), color=c, mfc="white" if hollow else c, mec=c, ms=6, mew=1.5,
                elinewidth=1.5, capsize=0, ls="none", **style)


def select(summary, **kw):
    d = summary
    for k, v in kw.items():
        d = d[d[k] == v]
    return d


def save(fig, out, name):
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    p = os.path.join(out, "figures", name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def legend_handles(kind):
    from matplotlib.lines import Line2D
    h = []
    if kind == "roles":
        for r, s in ROLE_STYLE.items():
            h.append(Line2D([], [], color=s["color"], marker=s["marker"], ls="none", ms=6, label=f"{s['label']}, tagged"))
            h.append(Line2D([], [], color=s["color"], marker=s["marker"], mfc="white", ls="none", ms=6, mew=1.5, label=f"{s['label']}, plain"))
    else:
        for c, s in CORPUS_STYLE.items():
            h.append(Line2D([], [], color=s["color"], marker=s["marker"], ls="none", ms=6, label=f"{CORPUS_LABEL[c]}, tagged"))
        h.append(Line2D([], [], color=MUTED, marker="o", mfc="white", ls="none", ms=6, mew=1.5, label="plain (hollow)"))
    return h


def by_role(summary, out, corpus, quantity, name, title, zero):
    """rows = metric, cols = family; x = stage; user/reply x tagged/plain."""
    fams = families(summary)
    base = select(summary, subset="all", positions="from1", unit="nats", corpus=corpus, quantity=quantity)
    fig, axes = plt.subplots(3, len(fams), figsize=(4.2 * len(fams), 8.2), squeeze=False)
    for j, f in enumerate(fams):
        stages = [s for s in config.family_settings(f)["stages"] if (base[base.family == f].stage == s).any()]
        x = np.arange(len(stages))
        for i, metric in enumerate(("entropy", "nll", "gap")):
            ax = axes[i, j]
            for k, (role, st) in enumerate(ROLE_STYLE.items()):
                for h, fr in enumerate(("tagged", "plain")):
                    d = select(base, family=f, metric=metric, role=role, framing=fr).set_index("stage").reindex(stages).dropna(subset=["mean"])
                    off = (k - 0.5) * 0.28 + (h - 0.5) * 0.12
                    errbar(ax, np.array([stages.index(s) for s in d.index]) + off, d, hollow=fr == "plain",
                           color=st["color"], marker=st["marker"])
            if zero:
                ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_xticks(x, stage_labels(f, stages))
            ax.set_xlim(-0.6, len(stages) - 0.4)
            if j == 0:
                ax.set_ylabel(f"{METRIC_LABEL[metric]}\n(nats/token)")
            if i == 0:
                ax.set_title(config.family_settings(f)["label"] + ("" if f == "olmo3-7b" else " (contamination unchecked)"), fontsize=10)
    fig.suptitle(f"{title} · {CORPUS_LABEL[corpus]}", x=0.01, ha="left", fontsize=12, color=INK)
    fig.legend(handles=legend_handles("roles"), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.text(0.01, -0.06, "Content positions 1+, 95% bootstrap CIs over exchanges. Each family has its own y-axis: nats per token are not "
             "comparable across tokenizers.", fontsize=8.5, color=MUTED)
    return save(fig, out, name)


def contrast_change(summary, out, name="contrast_change.png"):
    """Change from base of (reply − user), rows = metric, cols = family; x = post-trained stage; corpus series; tagged filled, plain hollow."""
    fams = families(summary)
    d0 = select(summary, subset="all", positions="from1", unit="nats", quantity="asst_minus_user_change", role="reply-user")
    fig, axes = plt.subplots(3, len(fams), figsize=(4.2 * len(fams), 8.2), squeeze=False)
    for j, f in enumerate(fams):
        stages = [s for s in list(config.family_settings(f)["stages"])[1:] if (d0[d0.family == f].stage == s).any()]
        for i, metric in enumerate(("entropy", "nll", "gap")):
            ax = axes[i, j]
            ax.axhspan(-FLOOR, FLOOR, color=GRID, alpha=0.8, lw=0)
            ax.axhline(0, color=MUTED, lw=0.8)
            for k, (c, st) in enumerate(CORPUS_STYLE.items()):
                for h, fr in enumerate(("tagged", "plain")):
                    d = select(d0, family=f, metric=metric, corpus=c, framing=fr).set_index("stage").reindex(stages).dropna(subset=["mean"])
                    off = (k - 1) * 0.24 + (h - 0.5) * 0.1
                    errbar(ax, np.array([stages.index(s) for s in d.index]) + off, d, hollow=fr == "plain", color=st["color"], marker=st["marker"])
            ax.set_xticks(np.arange(len(stages)), stage_labels(f, stages))
            ax.set_xlim(-0.6, len(stages) - 0.4)
            if j == 0:
                ax.set_ylabel(f"Δ (reply − user)\n{METRIC_LABEL[metric]} (nats/token)")
            if i == 0:
                ax.set_title(config.family_settings(f)["label"] + ("" if f == "olmo3-7b" else " (contamination unchecked)"), fontsize=10)
    fig.suptitle("Change from base of the assistant − user difference (gap row, tagged = D, oasst1 and UltraChat)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.legend(handles=legend_handles("corpora"), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.text(0.01, -0.06, f"Positive = the reply's quantity rose more (or fell less) than the user turn's. Grey band: ±{FLOOR} nats (smallest "
             "effect counted). Original positive-direction gap criterion: tagged D ≥ +0.05, CI above zero, in both oasst1 and UltraChat.\n"
             "Exploratory two-sided classification (added after the results, not pre-registered): the same floor in either direction, "
             "consistent in both corpora. Neither sign alone identifies a mechanism.\n"
             "95% bootstrap CIs, exchange IDs resampled jointly. Own y-axis per family.", fontsize=8.5, color=MUTED, va="top")
    return save(fig, out, name)


def positions(pos, out, corpus):
    """Δ from base by content-position bin (tagged), per post-trained checkpoint (rows) and metric (cols); n per bin on the x labels."""
    d0 = pos[(pos.corpus == corpus) & (pos.quantity == "change_from_base")]
    rows = [(f, s) for f in config.FAMILIES for s in list(config.family_settings(f)["stages"])[1:] if ((d0.family == f) & (d0.stage == s)).any()]
    if not rows:
        return None
    bins = list(dict.fromkeys(d0.bin))
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 2.3 * len(rows) + 0.8), squeeze=False)
    for i, (f, s) in enumerate(rows):
        for j, metric in enumerate(("entropy", "nll", "gap")):
            ax = axes[i, j]
            ax.axhline(0, color=MUTED, lw=0.8)
            ns = {}
            for k, (role, st) in enumerate(ROLE_STYLE.items()):
                d = d0[(d0.family == f) & (d0.stage == s) & (d0.metric == metric) & (d0.role == role)].set_index("bin").reindex(bins)
                ns[role] = d.n.fillna(0).astype(int).tolist()
                ok = d.lo.notna()
                x = np.arange(len(bins)) + (k - 0.5) * 0.25
                ax.plot(x[ok.to_numpy()], d["mean"][ok], color=st["color"], lw=1.2, alpha=0.6)
                errbar(ax, x[ok.to_numpy()], d[ok], color=st["color"], marker=st["marker"])
            ax.set_xticks(np.arange(len(bins)), [f"{b}\n{nu}/{nr}" for b, nu, nr in zip(bins, ns["user"], ns["reply"])], fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"{config.family_settings(f)['label']} {config.family_settings(f)['stage_labels'][s]}\nΔ from base", fontsize=9)
            if i == 0:
                ax.set_title(f"{METRIC_LABEL[metric]} (nats/token)", fontsize=10)
    fig.suptitle(f"Change from base by content position, tagged · {CORPUS_LABEL[corpus]}", x=0.01, ha="left", fontsize=12, color=INK)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=st["color"], marker=st["marker"], ls="-", label=st["label"]) for st in ROLE_STYLE.values()],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.01, -0.04, "x labels: content-position bin, then the number of user turns / replies with tokens in that bin (coverage falls "
             "with length; replies stop at position 299). Bins with < 10 targets are not shown. 95% bootstrap CIs.", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    return save(fig, out, f"positions_{corpus}.png")


def cross_family(summary, out, name="cross_family_bpb.png"):
    """Common exchange IDs, bits per byte: change of (reply − user) from base, tagged; one panel per metric."""
    d0 = select(summary, subset="common", unit="bpb", quantity="asst_minus_user_change", framing="tagged", role="reply-user")
    pts = [(f, s) for f in config.FAMILIES for s in list(config.family_settings(f)["stages"])[1:] if ((d0.family == f) & (d0.stage == s)).any()]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), squeeze=False)
    for j, metric in enumerate(("entropy", "nll", "gap")):
        ax = axes[0, j]
        ax.axhline(0, color=MUTED, lw=0.8)
        for k, (c, st) in enumerate(CORPUS_STYLE.items()):
            d = select(d0, metric=metric, corpus=c)
            d = d.set_index(["family", "stage"]).reindex(pts).dropna(subset=["mean"])
            errbar(ax, np.array([pts.index(p) for p in d.index]) + (k - 1) * 0.22, d, color=st["color"], marker=st["marker"])
        ax.set_xticks(np.arange(len(pts)), [f"{config.family_settings(f)['label']}\n{config.family_settings(f)['stage_labels'][s]}" for f, s in pts])
        ax.set_title(f"{METRIC_LABEL[metric]} (bits/byte)", fontsize=10)
        if j == 0:
            ax.set_ylabel("Δ (reply − user) from base")
    n = int(d0.n.max()) if len(d0) else 0
    fig.suptitle(f"Cross-family comparison on the common exchanges (first {n} per corpus), bits per byte, tagged", x=0.01, ha="left",
                 fontsize=12, color=INK)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=st["color"], marker=st["marker"], ls="none", label=CORPUS_LABEL[c]) for c, st in CORPUS_STYLE.items()],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.16))
    fig.text(0.01, -0.2, "Descriptive. Contamination checked for OLMo only. Entropy per byte = summed token entropy / ln 2 / UTF-8 bytes.",
             fontsize=8.5, color=MUTED)
    return save(fig, out, name)


def all_figures(summary, pos, out):
    paths = []
    for c in ("oasst1", "ultrachat", "wildchat"):
        if (summary.corpus == c).any():
            paths.append(by_role(summary, out, c, "level", f"levels_{c}.png", "Levels by checkpoint", zero=False))
            paths.append(by_role(summary, out, c, "change_from_base", f"changes_{c}.png", "Change from base", zero=True))
            paths.append(positions(pos, out, c))
    paths.append(contrast_change(summary, out))
    paths.append(cross_family(summary, out))
    return [p for p in paths if p]
