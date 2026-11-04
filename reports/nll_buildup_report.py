"""Figures (PNG + SVG, 16:9) and the self-contained HTML walk-through of the NLL arm's asymmetry, from reports/nll_buildup.json.

    python -m reports.nll_buildup_analysis && python -m reports.nll_buildup_report      # -> reports/nll_buildup.html
"""

import base64
import html
import io
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

RES = "reports/nll_buildup.json"
FIG_DIR = "reports/figures/buildup"
HTML_OUT = "reports/nll_buildup.html"

FAM_COLOR = {"olmo3-7b": "#2a78d6", "qwen3-8b": "#eb6834", "gemma4-31b": "#1baf7a"}  # same slots as reports/capstone_report.py
FAM_NAME = {"olmo3-7b": "OLMo 3 7B", "qwen3-8b": "Qwen3-8B", "gemma4-31b": "Gemma 4 31B"}
STAGE_NAME = {"sft": "SFT", "dpo": "DPO", "instruct": "Instruct"}
POST = [("olmo3-7b", "sft"), ("olmo3-7b", "dpo"), ("olmo3-7b", "instruct"), ("qwen3-8b", "instruct"), ("gemma4-31b", "instruct")]
CORPORA = ["oasst1", "ultrachat", "wildchat"]
CORP_NAME = {"oasst1": "oasst1", "ultrachat": "UltraChat", "wildchat": "WildChat"}
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FLOOR = 0.05
EXAMPLES = [("gemma4-31b", "instruct", "ex_gemma"), ("qwen3-8b", "instruct", "ex_qwen"), ("olmo3-7b", "dpo", "ex_olmodpo")]

plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 13, "axes.edgecolor": "#b9b8b2", "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "font.family": "DejaVu Sans", "svg.fonttype": "none", "hatch.linewidth": 1.4,
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})
FIGSIZE = (13.33, 7.5)
FIGS = {}


def ck(f, s):
    return f"OLMo {STAGE_NAME[s]}" if f == "olmo3-7b" else f"{FAM_NAME[f]} IT"


def base_name(f):
    return f"{FAM_NAME[f]} base"


def tint(hexc, t):
    """Mix a hex color with white (t = share of white)."""
    r, g, b = (int(hexc[i:i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(v + (255 - v) * t):02x}" for v in (r, g, b))


def style(ax, zero=True, floor=False):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if zero:
        ax.axhline(0, color="#8a8984", lw=1.2, zorder=1)
    if floor:
        ax.axhspan(-FLOOR, FLOOR, color="#ecebe6", zorder=0, lw=0)


def save(fig, name, caption):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=150)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.svg"))
    buf = io.BytesIO()
    fig.savefig(buf, format="svg")
    FIGS[name] = dict(svg=base64.b64encode(buf.getvalue()).decode(), caption=caption)
    plt.close(fig)


def bar(ax, x, c, face, edge, hatch=None, w=0.7, label=True, nd=2, fs=12, signed=True):
    """One bar from zero with its 95% CI and a value label beyond the CI end."""
    m = c["mean"]
    ax.bar([x], [m], width=w, color=face, edgecolor=edge, linewidth=1.6, hatch=hatch, zorder=2)
    if "lo" in c:
        ax.errorbar([x], [m], yerr=[[m - c["lo"]], [c["hi"] - m]], fmt="none", ecolor=INK, elinewidth=1.6, capsize=4, zorder=3)
    if label:
        top = c.get("hi", m) if m >= 0 else c.get("lo", m)
        ax.annotate((f"{m:+.{nd}f}" if signed else f"{m:.{nd}f}").replace("-", "−"), (x, top), xytext=(0, 5 if m >= 0 else -6), textcoords="offset points", ha="center",
                    va="bottom" if m >= 0 else "top", fontsize=fs, color=INK)


def scale_note(fig, text="y-axis: one scale for Steps 0–3 of this example; the other examples use their own scales"):
    fig.text(0.99, 0.015, text, ha="right", va="bottom", fontsize=11.5, color=INK2, style="italic")


# --- worked-example figures --------------------------------------------------------------------------------------------------

def example_ylim(cell):
    vals = [0.0]
    for side in ("user", "reply"):
        for c in cell["levels"][side].values():
            vals += [c["lo"], c["hi"]]
        for c in cell["changes"][side].values():
            vals += [c["lo"], c["hi"]]
    a = cell["measure"]["asymmetry"]
    vals += [a["lo"], a["hi"]]
    lo, hi = min(vals), max(vals)
    pad = 0.14 * (hi - lo)
    return lo - pad, hi + pad


def fig_step0(f, s, cell, ylim, name, caption):
    col = FAM_COLOR[f]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    L = cell["levels"]
    xs, labels = [], []
    for g, (stage_key, stage_lab) in enumerate((("base", base_name(f)), ("post", ck(f, s)))):
        x0 = g * 5.2
        for j, (side, fr) in enumerate((("user", "tagged"), ("user", "plain"), ("reply", "tagged"), ("reply", "plain"))):
            x = x0 + j + (0.4 if side == "reply" else 0)
            face = (col if side == "user" else tint(col, 0.55)) if fr == "tagged" else BG
            edge = col if side == "user" else tint(col, 0.3)
            bar(ax, x, L[side][f"{fr}_{stage_key}"], face, edge, hatch=None if fr == "tagged" else "//", w=0.8, signed=False)
            xs.append(x)
            labels.append(f"{side}\n{fr}")
        ax.text(x0 + 1.7, 1.0, stage_lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=15, fontweight="bold")
    ax.set_xticks(xs, labels)
    ax.set_ylim(*ylim)
    style(ax)
    ax.set_ylabel("mean NLL, nats per token (positions 1+)")
    ax.set_title(f"Step 0 · raw NLL · {ck(f, s)} vs its base · oasst1", loc="left", pad=34)
    ax.legend(handles=[Patch(facecolor=col, edgecolor=col, label="user turn, tagged"),
                       Patch(facecolor=BG, edgecolor=col, hatch="//", label="user turn, plain"),
                       Patch(facecolor=tint(col, 0.55), edgecolor=tint(col, 0.3), label="reply, tagged"),
                       Patch(facecolor=BG, edgecolor=tint(col, 0.3), hatch="//", label="reply, plain")],
              loc="upper left", frameon=False, ncol=4, bbox_to_anchor=(0, -0.13))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.24)
    scale_note(fig)
    save(fig, name, caption)


def fig_step1(f, s, cell, ylim, name, caption):
    col = FAM_COLOR[f]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    L = cell["levels"]
    xs, labels = [], []
    for g, side in enumerate(("user", "reply")):
        sc = col if side == "user" else tint(col, 0.3)
        for j, (k, lab) in enumerate((("fc_base", base_name(f)), ("fc_post", ck(f, s)))):
            x = g * 3 + j
            bar(ax, x, L[side][k], tint(sc, 0.65) if k == "fc_base" else sc, sc, w=0.75)
            xs.append(x)
            labels.append(lab)
        ax.text(g * 3 + 0.5, 1.0, "user turn" if side == "user" else "reply", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=15, fontweight="bold")
    ax.set_xticks(xs, labels)
    ax.set_xlim(-0.8, 4.8)
    ax.set_ylim(*ylim)
    style(ax)
    ax.set_ylabel("frame cost = NLL(tagged) − NLL(plain), nats/token")
    fig.text(0.08, 0.055, "below 0: the role tag helps   ·   above 0: the role tag hurts", fontsize=12, color=INK2)
    ax.set_title(f"Step 1 · frame cost at base and at the checkpoint · {ck(f, s)} · oasst1", loc="left", pad=34)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.17)
    scale_note(fig)
    save(fig, name, caption)


def fig_step2(f, s, cell, ylim, name, caption):
    col = FAM_COLOR[f]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    C = cell["changes"]
    xs, labels = [], []
    for g, side in enumerate(("user", "reply")):
        sc = col if side == "user" else tint(col, 0.3)
        x0 = g * 4
        bar(ax, x0, C[side]["tagged"], BG, sc, w=0.6, fs=11)
        bar(ax, x0 + 0.75, C[side]["plain"], BG, sc, hatch="//", w=0.6, fs=11)
        bar(ax, x0 + 1.9, C[side]["fc"], sc, sc, w=0.85, fs=13)
        xs += [x0, x0 + 0.75, x0 + 1.9]
        labels += ["Δ tagged", "Δ plain", "Δ frame cost\n(= Δ tagged − Δ plain)"]
        ax.text(x0 + 0.95, 1.0, "user turn" if side == "user" else "reply", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=15, fontweight="bold")
    ax.set_xticks(xs, labels, fontsize=12)
    ax.set_xlim(-0.7, 6.6)
    ax.set_ylim(*ylim)
    style(ax)
    ax.set_ylabel("change from base (checkpoint − base), nats/token")
    ax.set_title(f"Step 2 · change from base, paired per text · {ck(f, s)} · oasst1", loc="left", pad=34)
    fig.text(0.08, 0.055, "outlined bars: the two NLL changes that make up each Δ frame cost (context, not part of the statistic)",
            fontsize=12, color=INK2)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.19)
    scale_note(fig)
    save(fig, name, caption)


def fig_step3(f, s, cell, ylim, name, caption):
    col = FAM_COLOR[f]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    C = cell["changes"]
    a = cell["measure"]["asymmetry"]
    bar(ax, 0, C["user"]["fc"], col, col, w=0.7)
    bar(ax, 1, C["reply"]["fc"], tint(col, 0.3), tint(col, 0.3), w=0.7)
    bar(ax, 2.6, a, "#8a8984", "#52514e", w=0.9, fs=14)
    ax.text(0.5, 1.0, "the two inputs", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=15, fontweight="bold")
    ax.text(2.6, 1.0, "the asymmetry", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=15, fontweight="bold")
    ax.set_xticks([0, 1, 2.6], ["mean Δ frame cost\nuser turns", "mean Δ frame cost\nreplies", "asymmetry\n(user − reply)"])
    ax.set_xlim(-0.8, 3.6)
    ax.set_ylim(*ylim)
    style(ax, floor=True)
    ax.set_ylabel("nats per token")
    fig.text(0.08, 0.055, "shaded band: ±0.05 nats, the smallest effect counted   ·   asymmetry CI: user and reply sets resampled independently",
            fontsize=12, color=INK2)
    ax.set_title(f"Step 3 · asymmetry · {ck(f, s)} · oasst1", loc="left", pad=34)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.19)
    scale_note(fig)
    save(fig, name, caption)


# --- overview figures --------------------------------------------------------------------------------------------------------

def family_axes():
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, gridspec_kw=dict(width_ratios=(3, 1.3, 1.3), wspace=0.42))
    return fig, dict(zip(["olmo3-7b", "qwen3-8b", "gemma4-31b"], axes))


MARK = {"oasst1": "o", "ultrachat": "s", "wildchat": "^"}


def fig_verdict(r):
    fig, axes = family_axes()
    for f, ax in axes.items():
        stages = [s for ff, s in POST if ff == f]
        for i, s in enumerate(stages):
            for j, corp in enumerate(CORPORA):
                c = r["cells"][f"{f}|{s}"][corp]["measure"]["asymmetry"]
                x = i + (j - 1) * 0.22
                filled = corp != "wildchat"
                ax.errorbar([x], [c["mean"]], yerr=[[c["mean"] - c["lo"]], [c["hi"] - c["mean"]]], fmt=MARK[corp], ms=9,
                            color=FAM_COLOR[f], mfc=FAM_COLOR[f] if filled else BG, mew=2, elinewidth=2, zorder=3)
        ax.set_xticks(range(len(stages)), [ck(f, s) for s in stages])
        ax.set_xlim(-0.6, len(stages) - 0.4)
        style(ax, floor=True)
        ax.set_title(FAM_NAME[f] + "\n(own y-scale)", fontsize=15)
    axes["olmo3-7b"].set_ylabel("asymmetry, nats/token (95% CI)")
    h = [Line2D([], [], marker=MARK[c], ls="", color=INK2, mfc=INK2 if c != "wildchat" else BG, mew=2, ms=9,
                label=CORP_NAME[c] + (" (secondary)" if c == "wildchat" else "")) for c in CORPORA]
    h.append(Patch(color="#ecebe6", label="±0.05 nats"))
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Step 4 · the asymmetry at every post-trained checkpoint (fires: ≥ +0.05 and CI above 0 in oasst1 and UltraChat)",
                 x=0.02, ha="left", fontsize=16)
    fig.text(0.99, 0.085, "y-scales differ between the three family panels", ha="right", fontsize=11.5, color=INK2, style="italic")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.17)
    save(fig, "fig_step4_verdicts", "The asymmetry with its 95% CI at every post-trained checkpoint, against the family's own base. "
         "Only Gemma 4 31B IT fires (about +5 nats in both headline corpora). OLMo DPO and Instruct are clearly negative, the opposite of "
         "regression. OLMo SFT and Qwen3-8B IT sit inside or near the ±0.05 band. Each family panel has its own y-scale.")


def fig_robust(r):
    fig, axes = family_axes()
    kinds = [("fc", "frame-cost asymmetry (the measure)", "o", True), ("tagged", "tagged-only: Δ user tagged − Δ reply tagged", "D", True),
             ("plain", "plain-only: Δ user plain − Δ reply plain", "s", False)]
    for f, ax in axes.items():
        stages = [s for ff, s in POST if ff == f]
        ticks, labs = [], []
        k = 0
        for si, s in enumerate(stages):
            if len(stages) > 1:
                ax.text(k + 0.5, -0.115, ck(f, s), transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=13)
            for corp in ("oasst1", "ultrachat"):
                cell = r["cells"][f"{f}|{s}"][corp]
                for j, (key, _, mk, filled) in enumerate(kinds):
                    c = cell["measure"]["asymmetry"] if key == "fc" else cell["robust"][f"asym_{key}"]
                    x = k + (j - 1) * 0.25
                    ax.errorbar([x], [c["mean"]], yerr=[[c["mean"] - c["lo"]], [c["hi"] - c["mean"]]], fmt=mk, ms=8,
                                color=FAM_COLOR[f], mfc=FAM_COLOR[f] if filled else BG, mew=2, elinewidth=2, zorder=3,
                                alpha=1 if key != "plain" else 0.85)
                ticks.append(k)
                labs.append(CORP_NAME[corp])
                k += 1
            k += 0.5
        k -= 0.5
        ax.set_xticks(ticks, labs, fontsize=11)
        for sep in [t + 0.75 for t in ticks[1:-1:2]]:
            ax.axvline(sep, color=GRID, lw=1, zorder=0)
        ax.set_xlim(-0.6, k - 0.4)
        style(ax, floor=True)
        ax.set_title(f"{FAM_NAME[f]}" + ("" if f == "olmo3-7b" else " IT") + "\n(own y-scale)", fontsize=15)
    axes["olmo3-7b"].set_ylabel("nats/token (95% CI)")
    h = [Line2D([], [], marker=mk, ls="", color=INK2, mfc=INK2 if filled else BG, mew=2, ms=9, label=lab) for _, lab, mk, filled in kinds]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=12)
    fig.suptitle("Robustness · does the verdict depend on the plain reference? (headline corpora)", x=0.02, ha="left", fontsize=16)
    fig.text(0.99, 0.08, "frame-cost asymmetry = tagged-only − plain-only   ·   y-scales differ between panels", ha="right", fontsize=11.5,
             color=INK2, style="italic")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.2)
    save(fig, "fig_robust_tagged_only", "The measure (circles) next to the same difference taken on tagged NLL only (diamonds) and on plain "
         "NLL only (hollow squares). Gemma 4 31B IT is strongly positive on every version. OLMo DPO and Instruct are negative on every "
         "version. Qwen3-8B IT's null on the measure becomes negative on tagged-only, because its replies' tagged NLL rose more than "
         "its user turns' did. No checkpoint fires on one version but not another.")


# --- HTML --------------------------------------------------------------------------------------------------------------------

def esc(s):
    return html.escape(str(s))


def n2(c, nd=2):
    return f"{c['mean']:+.{nd}f}"


def ci(c, nd=2):
    return f"{c['mean']:+.{nd}f} <span class='ci'>[{c['lo']:+.{nd}f}, {c['hi']:+.{nd}f}]</span>"


def lvl(c, nd=2):
    return f"{c['mean']:.{nd}f}"


def figure(name):
    fg = FIGS[name]
    return (f'<figure id="{name}"><img alt="{esc(fg["caption"])}" src="data:image/svg+xml;base64,{fg["svg"]}">'
            f'<figcaption>{esc(fg["caption"])} <span class="files">(<code>{FIG_DIR}/{name}.png</code>, <code>.svg</code>)</span>'
            f'</figcaption></figure>')


CSS = """
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#7a7974;--line:#e4e3df;--accent:#2a78d6;--fire:#0b5d1e;--firebg:#e3f3e6;
--neg:#7a2f12;--negbg:#fbeae2;--drive:#fff2b8;--drivetx:#3b3000;--code:#f2f1ed;--note:#f2f1ed}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--card:#232322;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9c9b93;
--line:#3a3a37;--accent:#3987e5;--fire:#8fe0a0;--firebg:#1f3a25;--neg:#f3b9a0;--negbg:#3d261c;--drive:#4a4013;--drivetx:#fff2b8;--code:#2c2c2a;--note:#262624}}
:root[data-theme="dark"]{--bg:#1a1a19;--card:#232322;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9c9b93;--line:#3a3a37;--accent:#3987e5;--fire:#8fe0a0;
--firebg:#1f3a25;--neg:#f3b9a0;--negbg:#3d261c;--drive:#4a4013;--drivetx:#fff2b8;--code:#2c2c2a;--note:#262624}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 80px}h1{font-size:2rem;margin:.2em 0}
h2{font-size:1.5rem;margin-top:2.2em;border-bottom:1px solid var(--line);padding-bottom:.25em}h3{font-size:1.15rem;margin-top:1.6em}
p,li{max-width:82ch}.sub{color:var(--ink2)}code{background:var(--code);padding:1px 4px;border-radius:4px;font-size:.88em}
nav{font-size:.95rem;color:var(--ink2);line-height:1.9}nav a{color:var(--accent);margin-right:.9em;text-decoration:none}
.tbl{overflow-x:auto;margin:1em 0}table{border-collapse:collapse;width:100%;font-size:.88rem;background:var(--card)}
th,td{border:1px solid var(--line);padding:6px 8px;text-align:right;vertical-align:top;font-variant-numeric:tabular-nums;white-space:nowrap}
th{background:var(--code);font-weight:600;text-align:center}td.l{text-align:left}tr.sec td{color:var(--muted)}tr.grp td{border-top:2px solid var(--ink2)}
td.fire{background:var(--firebg);color:var(--fire);font-weight:600}td.neg{background:var(--negbg);color:var(--neg);font-weight:600}
td.drive{background:var(--drive);color:var(--drivetx);font-weight:600}.ci{color:var(--muted);font-size:.9em}
.step{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:8px;padding:12px 18px;margin:1em 0}
.step .f{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--code);padding:6px 10px;border-radius:6px;display:block;margin:.5em 0;
overflow-x:auto;white-space:nowrap}
.note{background:var(--note);border-radius:8px;padding:10px 16px;margin:1em 0}.callout{border:1px solid #e8c66a;background:#fdf3dc;color:#2b2410;
border-radius:8px;padding:10px 16px;margin:1em 0}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .callout{background:#3a3220;border-color:#8a7230;color:#f0e6c8}}
:root[data-theme="dark"] .callout{background:#3a3220;border-color:#8a7230;color:#f0e6c8}
figure{margin:1.2em 0;background:#fcfcfb;border:1px solid var(--line);border-radius:8px;padding:8px}figure img{width:100%;height:auto;display:block}
figcaption{color:#2b2b29;font-size:.95rem;padding:8px 6px 2px}.files{color:#7a7974;font-size:.82em}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.chain{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.86rem;overflow-x:auto;white-space:pre;background:var(--code);
padding:10px 12px;border-radius:6px}
"""


def driver_marks(cell):
    """Which cells drive the result: the side(s) whose contribution carries the asymmetry, and within each, the larger NLL change."""
    a = cell["measure"]["asymmetry"]
    cu, cr = cell["changes"]["user"]["fc"]["mean"], -cell["changes"]["reply"]["fc"]["mean"]  # contributions to A
    marks = set()
    null = a["lo"] <= 0 <= a["hi"]
    if null:
        sides = ["user", "reply"] if abs(cu) > 0.05 and abs(cr) > 0.05 else []
    else:
        same = [(sd, abs(v)) for sd, v in (("user", cu), ("reply", cr)) if v * a["mean"] > 0]
        sides = [max(same, key=lambda t: t[1])[0]] if same else []
        if len(same) == 2 and min(v for _, v in same) > 0.35 * max(v for _, v in same):
            sides = ["user", "reply"]
    for sd in sides:
        marks.add((sd, "dfc"))
        ch = cell["changes"][sd]
        marks.add((sd, "tagged" if abs(ch["tagged"]["mean"]) >= abs(ch["plain"]["mean"]) else "plain"))
    return marks, null


def verdict_cell(c):
    if c["mean"] >= FLOOR and c["lo"] > 0:
        return "fire", "fires"
    if c["mean"] <= -FLOOR and c["hi"] < 0:
        return "neg", "negative"
    return "", "null"


def buildup_table(r):
    head1 = ("<tr><th rowspan=2>checkpoint</th><th rowspan=2>corpus</th><th colspan=7>user turns</th><th colspan=7>replies</th>"
             "<th rowspan=2>asymmetry<br>Δ FC user − Δ FC reply<br>[95% CI]</th><th rowspan=2>in this corpus</th><th rowspan=2>checkpoint verdict</th></tr>")
    sub = "".join(f"<th>{x}</th>" for x in ["tagged<br>base", "tagged<br>ckpt", "plain<br>base", "plain<br>ckpt", "FC<br>base", "FC<br>ckpt",
                                              "Δ FC [95% CI]"]) * 2
    rows = [head1, f"<tr>{sub}</tr>"]
    for f, s in POST:
        v = r["verdict"][f"{f}|{s}"]["frame_cost"]
        for i, corp in enumerate(CORPORA):
            cell = r["cells"][f"{f}|{s}"][corp]
            marks, _ = driver_marks(cell)
            L, C = cell["levels"], cell["changes"]
            tds = [f"<td class='l'>{'<span class=sw style=background:' + FAM_COLOR[f] + '></span>' + esc(ck(f, s)) if i == 0 else ''}</td>",
                   f"<td class='l'>{CORP_NAME[corp]}{' (secondary)' if corp == 'wildchat' else ''}</td>"]
            for sd in ("user", "reply"):
                for fr in ("tagged", "plain"):
                    cls = " class='drive'" if (sd, fr) in marks else ""
                    tds += [f"<td{cls}>{lvl(L[sd][fr + '_base'])}</td>", f"<td{cls}>{lvl(L[sd][fr + '_post'])}</td>"]
                tds += [f"<td>{n2(L[sd]['fc_base'])}</td>", f"<td>{n2(L[sd]['fc_post'])}</td>"]
                cls = " class='drive'" if (sd, "dfc") in marks else ""
                tds.append(f"<td{cls}>{ci(C[sd]['fc'])}</td>")
            a = cell["measure"]["asymmetry"]
            k, word = verdict_cell(a)
            if corp == "wildchat":
                word += " (not used)"
            tds += [f"<td class='{k}'>{ci(a)}</td>", f"<td class='l'>{word}</td>",
                    f"<td class='l'>{esc(v) if i == 0 else ''}</td>"]
            cls = (" class='grp'" if i == 0 else "") if corp != "wildchat" else " class='sec'"
            rows.append(f"<tr{cls}>{''.join(tds)}</tr>")
    return f'<div class="tbl"><table>{"".join(rows)}</table></div>'


def robust_table(r):
    head = ("<tr><th>checkpoint</th><th>corpus</th><th>frame-cost asymmetry<br>(the measure)</th><th>tagged-only<br>Δ user tagged − Δ reply tagged</th>"
            "<th>Δ user plain</th><th>Δ reply plain</th><th>plain-only<br>Δ user plain − Δ reply plain</th></tr>")
    rows = [head]
    for f, s in POST:
        for i, corp in enumerate(CORPORA):
            cell = r["cells"][f"{f}|{s}"][corp]
            a, t, p = cell["measure"]["asymmetry"], cell["robust"]["asym_tagged"], cell["robust"]["asym_plain"]
            ka, _ = verdict_cell(a)
            kt, _ = verdict_cell(t)
            tds = [f"<td class='l'>{'<span class=sw style=background:' + FAM_COLOR[f] + '></span>' + esc(ck(f, s)) if i == 0 else ''}</td>",
                   f"<td class='l'>{CORP_NAME[corp]}{' (secondary)' if corp == 'wildchat' else ''}</td>",
                   f"<td class='{ka}'>{ci(a)}</td>", f"<td class='{kt}'>{ci(t)}</td>",
                   f"<td>{ci(cell['changes']['user']['plain'])}</td>", f"<td>{ci(cell['changes']['reply']['plain'])}</td>", f"<td>{ci(p)}</td>"]
            cls = (" class='grp'" if i == 0 else "") if corp != "wildchat" else " class='sec'"
            rows.append(f"<tr{cls}>{''.join(tds)}</tr>")
    vt = "<tr><th>checkpoint</th><th>verdict, frame cost (the measure)</th><th>verdict, tagged-only</th></tr>"
    vrows = [vt] + [f"<tr><td class='l'>{esc(ck(f, s))}</td><td class='l'>{esc(r['verdict'][f'{f}|{s}']['frame_cost'])}</td>"
                    f"<td class='l'>{esc(r['verdict'][f'{f}|{s}']['tagged_only'])}</td></tr>" for f, s in POST]
    return f'<div class="tbl"><table>{"".join(rows)}</table></div>', f'<div class="tbl"><table>{"".join(vrows)}</table></div>'


def chain(f, s, cell):
    L, C, a = cell["levels"], cell["changes"], cell["measure"]["asymmetry"]
    out = []
    for sd, nm in (("user", "user turn"), ("reply", "reply    ")):
        l, c = L[sd], C[sd]
        out.append(f"{nm}  tagged {l['tagged_base']['mean']:5.2f} → {l['tagged_post']['mean']:5.2f}   plain {l['plain_base']['mean']:5.2f} → "
                   f"{l['plain_post']['mean']:5.2f}   frame cost {l['fc_base']['mean']:+5.2f} → {l['fc_post']['mean']:+5.2f}   Δ FC {c['fc']['mean']:+.2f}")
    out.append(f"asymmetry  {C['user']['fc']['mean']:+.2f} − ({C['reply']['fc']['mean']:+.2f}) = {a['mean']:+.2f}  [{a['lo']:+.2f}, {a['hi']:+.2f}]")
    return "<div class='chain'>" + esc("\n".join(out)) + "</div>"


def example_texts(r):
    """Captions (one sentence on what moved) per example and step, from the numbers."""
    T = {}
    g = r["cells"]["gemma4-31b|instruct"]["oasst1"]
    gL, gC = g["levels"], g["changes"]
    T["ex_gemma"] = [
        f"Every bar grows, but unevenly: user tagged NLL rises from {lvl(gL['user']['tagged_base'])} to {lvl(gL['user']['tagged_post'])} "
        f"and user plain from {lvl(gL['user']['plain_base'])} to {lvl(gL['user']['plain_post'])}. Reply plain rises from "
        f"{lvl(gL['reply']['plain_base'])} to {lvl(gL['reply']['plain_post'])}, while reply tagged rises least "
        f"({lvl(gL['reply']['tagged_base'])} to {lvl(gL['reply']['tagged_post'])}).",
        f"At base both frame costs are about zero (the tags barely matter). At the checkpoint the user tag costs "
        f"{n2(gL['user']['fc_post'])} nats/token while the reply frame cost falls to {n2(gL['reply']['fc_post'])}. That fall is "
        f"relative: tagged replies got worse ({n2(gC['reply']['tagged'])}), but plain replies got much worse ({n2(gC['reply']['plain'])}).",
        f"Δ frame cost is {n2(gC['user']['fc'])} for user turns (Δ tagged {n2(gC['user']['tagged'])} minus Δ plain {n2(gC['user']['plain'])}) "
        f"and {n2(gC['reply']['fc'])} for replies (Δ tagged {n2(gC['reply']['tagged'])} minus Δ plain {n2(gC['reply']['plain'])}).",
        f"The user side rises and the reply side falls, so both push the same way: {n2(gC['user']['fc'])} − ({n2(gC['reply']['fc'])}) = "
        f"{n2(g['measure']['asymmetry'])} nats/token, far above the 0.05 floor, with the CI well above zero.",
    ]
    q = r["cells"]["qwen3-8b|instruct"]["oasst1"]
    qL, qC = q["levels"], q["changes"]
    T["ex_qwen"] = [
        f"At base, plain text is much harder than tagged for Qwen (user plain {lvl(qL['user']['plain_base'])} against tagged "
        f"{lvl(qL['user']['tagged_base'])}). After post-training, both tagged bars rise (user {n2(qC['user']['tagged'])}, reply "
        f"{n2(qC['reply']['tagged'])}). User plain barely moves ({n2(qC['user']['plain'])}) and reply plain rises a little "
        f"({n2(qC['reply']['plain'])}).",
        f"Both frame costs start well below zero (the tags help base Qwen a lot: user {n2(qL['user']['fc_base'])}, reply "
        f"{n2(qL['reply']['fc_base'])}). Both shrink by a similar amount, to {n2(qL['user']['fc_post'])} and {n2(qL['reply']['fc_post'])}.",
        f"The two changes are almost equal (user {n2(qC['user']['fc'])}, reply {n2(qC['reply']['fc'])}), but they are built differently. "
        f"The user change is nearly all tagged NLL rising; the reply change is tagged rising ({n2(qC['reply']['tagged'])}) minus part of "
        f"that rise also appearing in plain ({n2(qC['reply']['plain'])}).",
        f"Similar rises cancel: {n2(qC['user']['fc'])} − {n2(qC['reply']['fc'])} = {n2(q['measure']['asymmetry'])} nats/token, inside the "
        f"±0.05 band, with a CI that spans zero.",
    ]
    o = r["cells"]["olmo3-7b|dpo"]["oasst1"]
    oL, oC = o["levels"], o["changes"]
    T["ex_olmodpo"] = [
        f"The big mover is reply tagged NLL, from {lvl(oL['reply']['tagged_base'])} to {lvl(oL['reply']['tagged_post'])}. Reply plain "
        f"rises about half as much ({lvl(oL['reply']['plain_base'])} to {lvl(oL['reply']['plain_post'])}); the user bars rise modestly "
        f"(tagged {n2(oC['user']['tagged'])}, plain {n2(oC['user']['plain'])}).",
        f"The user frame cost moves only from {n2(oL['user']['fc_base'])} to {n2(oL['user']['fc_post'])}, while the reply frame cost "
        f"goes from {n2(oL['reply']['fc_base'])} to {n2(oL['reply']['fc_post'])}: after DPO, the assistant tag makes these replies harder, not easier.",
        f"Δ frame cost: user {n2(oC['user']['fc'])} (plain rose a bit more than tagged), reply {n2(oC['reply']['fc'])} "
        f"(tagged {n2(oC['reply']['tagged'])} minus plain {n2(oC['reply']['plain'])}).",
        f"{n2(oC['user']['fc'])} − ({n2(oC['reply']['fc'])}) = {n2(o['measure']['asymmetry'])} nats/token. The reply term contributes "
        f"{abs(oC['reply']['fc']['mean']) / abs(o['measure']['asymmetry']['mean']) * 100:.0f}% of it, so the result is negative "
        f"(the opposite of regression) and driven almost entirely by the reply side.",
    ]
    return T


def build_figures(r):
    T = example_texts(r)
    for f, s, key in EXAMPLES:
        cell = r["cells"][f"{f}|{s}"]["oasst1"]
        yl = example_ylim(cell)
        fig_step0(f, s, cell, yl, f"fig_{key}_step0_raw", T[key][0])
        fig_step1(f, s, cell, yl, f"fig_{key}_step1_framecost", T[key][1])
        fig_step2(f, s, cell, yl, f"fig_{key}_step2_change", T[key][2])
        fig_step3(f, s, cell, yl, f"fig_{key}_step3_asymmetry", T[key][3])
    fig_verdict(r)
    fig_robust(r)


def build_html(r):
    rep = r["reproduction"]
    idmax = max(rep["identities_max_abs_diff"].values())
    rt, vt = robust_table(r)
    nb = r["settings"]["n_boot"]
    ex_intro = {
        "ex_gemma": ("Gemma 4 31B IT: positive", "The user tag becomes costly while the assistant tag helps relative to plain. This is the "
                     "one checkpoint that fires."),
        "ex_qwen": ("Qwen3-8B IT: null by cancellation", "Both sides' frame costs rise by similar amounts, so their difference is about zero."),
        "ex_olmodpo": ("OLMo 3 7B DPO: negative, from the reply side", "The asymmetry is strongly negative, and almost all of it comes "
                       "from replies getting harder under the assistant tag."),
    }
    ex_html = []
    for f, s, key in EXAMPLES:
        cell = r["cells"][f"{f}|{s}"]
        title, lead = ex_intro[key]
        u = cell["ultrachat"]["measure"]["asymmetry"]
        ex_html.append(f"""
<h3 id="{key}"><span class="sw" style="background:{FAM_COLOR[f]}"></span>{esc(title)}</h3>
<p>{esc(lead)} oasst1, {cell['oasst1']['n']['user']} user turns and {cell['oasst1']['n']['reply']} replies. The whole chain in one block (means, nats/token):</p>
{chain(f, s, cell['oasst1'])}
{figure(f'fig_{key}_step0_raw')}{figure(f'fig_{key}_step1_framecost')}{figure(f'fig_{key}_step2_change')}{figure(f'fig_{key}_step3_asymmetry')}
<p><b>Step 4 for this checkpoint:</b> oasst1 {ci(cell['oasst1']['measure']['asymmetry'])}, UltraChat {ci(u)} →
<b>{esc(r['verdict'][f'{f}|{s}']['frame_cost'])}</b>.</p>""")

    body = f"""<main>
<h1>How the NLL asymmetry is built</h1>
<p class="sub">Vestigial regression, NLL arm, exchange design (main run, <code>results/capstone/</code>). A walk-through from raw NLL to the
verdict. Analysis only: no measure's code, criteria or data were changed. Generated by <code>python -m reports.nll_buildup_analysis</code>
and <code>python -m reports.nll_buildup_report</code>. 2026-11-04.</p>
<nav><a href="#setup">Setup</a><a href="#repro">Reproduction</a><a href="#steps">The steps</a><a href="#examples">Worked examples</a>
<a href="#table">Full table</a><a href="#robust">Robustness</a><a href="#caveats">Caveats</a></nav>

<h2 id="setup">What is being measured</h2>
<p>The question is whether post-training makes a model worse at modelling the <i>user</i>'s side of a chat than the assistant's side,
beyond what it does to text in general (Shiller's "dominance": non-assistant personas degrade). Each family's post-trained checkpoint is
compared with that family's own base model on the same texts.</p>
<p>The texts are real two-turn exchanges, a user turn plus the reply that followed it, from three corpora: <b>oasst1</b> and
<b>UltraChat</b> are the headline corpora, and <b>WildChat</b> is secondary (reported, not used for the verdict). There are 300 exchanges
per corpus, or 100 for Gemma 4 31B. Every text is scored twice with the same tokens:</p>
<ul><li><b>tagged</b>: in its chat role, using the family's chat template (the user turn as the conversation's opener; the reply after its
own user turn and the assistant opening);</li>
<li><b>plain</b>: the same words as bare prose with every role tag removed (the reply follows its user text after a blank line).</li></ul>
<p>Only the scored text's own tokens count, from content position 1 on. All numbers are in nats per token, the verdict's unit;
bits-per-byte versions are in <code>summary.csv</code>. Contamination can be checked for OLMo only: Qwen3 and Gemma 4 results cannot rule
out training-data overlap.</p>

<h2 id="repro">Do these numbers reproduce the analysis?</h2>
<div class="note"><p><b>Yes, exactly.</b> All per-text quantities come from the measure's own code
(<code>measures/vestigial_regression/analyze_exchanges.py</code>: <code>load_family</code>, <code>per_text</code>, <code>run</code>, with
<code>asymmetry_ci</code> from <code>analyze.py</code>), re-run with the same seed ({r['settings']['seed']}) and {nb:,} resamples.</p>
<ul><li>Against <code>reports/capstone_results.json</code>: {rep['rows_nats']} nats rows (every checkpoint × corpus × quantity), largest
difference in mean or CI bound <b>{rep['max_abs_diff_vs_capstone_json']:.1g}</b>. Verdicts are identical:
{'yes' if rep['verdicts_match_capstone'] else 'NO'}.</li>
<li>Against the committed <code>results/capstone/vestigial_regression/exchanges_exchange/summary.csv</code>: largest difference
{rep['max_abs_diff_vs_committed_summary_csv']:.1g} (the CSV is written to 6 decimals).</li>
<li>Build-up identities, checked on every cell: the mean of the per-text changes equals the summary's mean; Δ FC = Δ tagged − Δ plain per
text; asymmetry = mean Δ FC user − mean Δ FC reply; level difference = mean change. Largest discrepancy: {idmax:.1g} (floating point).</li></ul>
<p>The CIs on the <i>levels</i> (Steps 0 and 1) and on the robustness versions are new. They use a separate random stream, so the measure's
own numbers are unaffected.</p></div>

<h2 id="steps">The steps</h2>
<div class="step"><b>Step 0 · Raw NLL.</b> Negative log-likelihood is how surprised the model is by each actual next token,
−log p(token | everything before it), here averaged over the text's tokens from position 1 on. Lower means the model predicts the text better.
<span class="f">NLL(text, framing, model) = mean over positions t ≥ 1 of −ln p<sub>model</sub>(x<sub>t</sub> | x<sub>&lt;t</sub>)</span>
Four conditions per exchange: user turn tagged, user turn plain, reply tagged, reply plain. Each is scored at base and at the checkpoint.
Read it as a level: higher is worse.</div>
<div class="step"><b>Step 1 · Frame cost.</b> How much the role tag helps or hurts the model predict the same words.
<span class="f">FC = NLL(tagged) − NLL(plain), per text, at base and at the checkpoint</span>
<b>Negative</b>: the tag helps (the words are easier in their role). <b>Positive</b>: the tag hurts. It is <i>relative</i>: FC can fall
because tagged got better, or because plain got worse by more.</div>
<div class="step"><b>Step 2 · Change from base.</b> What post-training did to the tag's effect.
<span class="f">ΔFC = FC(checkpoint) − FC(base), paired per text  ( = ΔNLL(tagged) − ΔNLL(plain) )</span>
<b>Positive</b>: post-training made this role's tag more costly (or less helpful). <b>Negative</b>: it made the tag more helpful.</div>
<div class="step"><b>Step 3 · Asymmetry.</b> Whether the user's role lost more than the assistant's role.
<span class="f">A = mean ΔFC(user turns) − mean ΔFC(replies), within a corpus</span>
The 95% CI comes from {nb:,} bootstrap resamples, with the user set and the reply set resampled <b>independently</b>, as the analysis does
(each term is paired per exchange across checkpoints). <b>Positive</b>: the user tag got relatively worse, which is vestigial regression.
<b>Negative</b>: the assistant tag got relatively worse.</div>
<div class="step"><b>Step 4 · Verdict.</b>
<span class="f">fires ⇔ A ≥ 0.05 nats and 95% CI lower bound &gt; 0, in both oasst1 and UltraChat</span>
0.05 nats is the smallest effect the project counts (<code>config.MIN_EFFECT_NATS</code>). WildChat is reported but does not enter the
verdict. "Negative in both" (A ≤ −0.05 with the CI below zero in both headline corpora) is reported as the opposite of regression; it
does not fire.</div>
{figure('fig_step4_verdicts')}

<h2 id="examples">Worked examples (oasst1)</h2>
<p>Each example's four figures share one y-axis scale, so bar heights can be compared across steps; the three examples use different
scales (Gemma's numbers are about ten times larger). Within an example, the darker shade is user turns and the lighter shade is replies;
hatched bars are plain text.</p>
{''.join(ex_html)}

<h2 id="table">Full build-up table</h2>
<p>Every post-trained checkpoint × corpus, nats per token, means (CIs on the changes and the asymmetry). FC = frame cost; ckpt = the
post-trained checkpoint. <span class="sw" style="background:#fff2b8;border:1px solid #d9c870"></span><b>Highlighted</b>: the cells that drive
the result. That is the Δ FC of the side whose term carries the asymmetry (both sides when they contribute comparably, or when a null comes
from two sizeable changes cancelling), plus, on that side, the NLL pair (tagged or plain) that changed more. Asymmetry cells:
<span class="sw" style="background:#e3f3e6"></span>fires in that corpus, <span class="sw" style="background:#fbeae2"></span>negative
(≤ −0.05 with the CI below zero). WildChat rows are greyed: secondary, not part of the verdict. Gemma has n = 100 per corpus, the others 300.</p>
{buildup_table(r)}
<p class="sub">Two things the table makes visible. First, at OLMo DPO and Instruct, the reply side's tagged NLL roughly doubles while its
plain NLL rises about half as much, and this drives the negative asymmetry everywhere. Second, OLMo's WildChat user turns get
<i>easier</i> after post-training at every stage (tagged and plain both fall), unlike oasst1 and UltraChat. This fits OLMo's SFT data containing most of WildChat: the filter
removed exact matches only, so close relatives of these texts were likely seen in training. It is one reason WildChat is secondary.</p>

<h2 id="robust">Robustness: does the verdict depend on the plain reference?</h2>
<p>Frame cost subtracts plain-text NLL, so a verdict could in principle come from plain text moving rather than from anything about
the roles. Two alternative versions, using the same independent bootstrap as the measure:</p>
<ul><li><b>tagged-only</b>: Δ user tagged − Δ reply tagged (ignores plain text);</li>
<li><b>plain-only</b>: Δ user plain − Δ reply plain (ignores the tags; shown with its two inputs, since plain text moves a lot for Gemma).</li></ul>
<p>They are linked exactly: <b>frame-cost asymmetry = tagged-only − plain-only</b>.</p>
{figure('fig_robust_tagged_only')}
{rt}
<h3>Verdicts under each version (same rule: ≥ 0.05 with the CI above zero in both headline corpora)</h3>
{vt}
<div class="callout"><b>No fires/does-not-fire verdict changes.</b> Gemma 4 31B IT fires on both versions; tagged-only is even larger
(about +7.3) because user plain text also degrades heavily (+5.2), and the measure subtracts that. Every other checkpoint does not fire on
either version. One <i>description</i> changes: Qwen3-8B IT's null becomes "negative in both headline corpora" on tagged-only (about −0.19 and
−0.16), because its replies' tagged NLL rose more than its user turns' did. Under frame cost that difference is cancelled by replies' plain
NLL also rising more, so the measure's null depends on the plain reference. OLMo SFT shows a small positive tagged-only effect in UltraChat
({ci(r['cells']['olmo3-7b|sft']['ultrachat']['robust']['asym_tagged'], 3)}), with the CI above zero but the mean below the 0.05 floor,
so it does not fire.</div>

<h2 id="caveats">Caveats for reading these numbers</h2>
<ul>
<li><b>Off-policy replies.</b> Every reply was written by someone else (a human or another model). A post-trained model that has its own
style finds others' replies surprising, which inflates reply tagged NLL. This is the main suspect for OLMo DPO and Instruct's reply side.</li>
<li><b>Gemma's levels.</b> Gemma 4 31B IT's user tagged NLL of about 11 nats/token, and plain NLL of about 8, are very high: the IT model
degrades on everything that is not its own reply. The asymmetry is still clear, but it sits on top of a large general degradation (the
plain-only row), which the project refers to the overgeneralization measure.</li>
<li><b>Contamination</b> can be checked for OLMo only. <b>OLMo NLL verdicts are not blind</b>: the criteria were revised after pilot
results (<code>measures/vestigial_regression/README.md</code>).</li>
<li>This report covers the NLL arm only. The judge arm and logit distortion are in <code>reports/capstone_results.html</code>.</li>
</ul>
<p class="sub">Settings: {nb:,} bootstrap resamples, seed {r['settings']['seed']}, floor {r['settings']['floor']} nats, {esc(r['settings']['unit'])}.
Inputs: <code>results/capstone/&lt;family&gt;/&lt;stage&gt;/core/exchanges_exchange/scores.csv</code>. Numbers: <code>reports/nll_buildup.json</code>.
Figures: <code>{FIG_DIR}/</code>.</p>
</main>"""
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>NLL asymmetry build-up</title><style>{CSS}</style></head><body>{body}</body></html>')
    open(HTML_OUT, "w", encoding="utf-8").write(doc)


def main():
    r = json.load(open(RES))
    build_figures(r)
    build_html(r)
    print(f"wrote {HTML_OUT} ({os.path.getsize(HTML_OUT) / 1e6:.1f} MB) and {len(FIGS)} figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
