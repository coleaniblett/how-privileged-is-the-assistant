"""Figures (PNG + SVG, 16:9) and the self-contained HTML report from reports/capstone_results.json.

    python -m reports.capstone_analysis && python -m reports.capstone_report
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

RES = "reports/capstone_results.json"
FIG_DIR = "reports/figures"
HTML_OUT = "reports/capstone_results.html"

FAM_COLOR = {"olmo3-7b": "#2a78d6", "qwen3-8b": "#eb6834", "gemma4-31b": "#1baf7a"}  # reference palette slots 1-3 (all-pairs safe)
FAM_NAME = {"olmo3-7b": "OLMo 3 7B", "qwen3-8b": "Qwen3-8B", "gemma4-31b": "Gemma 4 31B"}
STAGE_NAME = {"base": "base", "sft": "SFT", "dpo": "DPO", "instruct": "Instruct"}
POST = [("olmo3-7b", "sft"), ("olmo3-7b", "dpo"), ("olmo3-7b", "instruct"), ("qwen3-8b", "instruct"), ("gemma4-31b", "instruct")]
ALL = [("olmo3-7b", "base"), ("olmo3-7b", "sft"), ("olmo3-7b", "dpo"), ("olmo3-7b", "instruct"), ("qwen3-8b", "base"),
       ("qwen3-8b", "instruct"), ("gemma4-31b", "base"), ("gemma4-31b", "instruct")]
CORP_NAME = {"oasst1": "oasst1", "ultrachat": "UltraChat", "wildchat": "WildChat"}
CORP_MARK = {"oasst1": "o", "ultrachat": "s", "wildchat": "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CAT_COLOR = {"user_voice": "#d6d5cf", "user_voice_with_assistant_traits": "#e87ba4", "switched_to_assistant": "#6250d6",
             "degenerate": "#eda100", "invalid": "#ffffff"}
CAT_NAME = {"user_voice": "user voice", "user_voice_with_assistant_traits": "user voice + assistant traits",
            "switched_to_assistant": "switched to assistant", "degenerate": "degenerate", "invalid": "refused by judge (unlabelled)"}
READ_NAME = {"exclude": "refused excluded", "as_user": "refused = user voice", "as_nonuser": "refused = non-user voice"}
READ_MARK = {"exclude": "o", "as_user": "D", "as_nonuser": "s"}

plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 13,
                     "legend.fontsize": 13, "axes.edgecolor": "#b9b8b2", "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
FIGSIZE = (13.33, 7.5)


def ck(f, s, short=True):
    if s == "base":
        return f"{FAM_NAME[f]} base"
    if f == "olmo3-7b":
        return f"OLMo {STAGE_NAME[s]}" if short else f"OLMo 3 7B {STAGE_NAME[s]}"
    return f"{FAM_NAME[f]} IT"


def style(ax, zero=True, floor=None):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if zero:
        ax.axhline(0, color="#8a8984", lw=1.2, zorder=1)
    if floor:
        ax.axhspan(-floor, floor, color="#ecebe6", zorder=0, lw=0)


def dot(ax, x, c, color, marker, filled=True, size=9):
    ax.errorbar([x], [c["mean"]], yerr=[[c["mean"] - c["lo"]], [c["hi"] - c["mean"]]], fmt=marker, ms=size, color=color,
                mfc=color if filled else "#fcfcfb", mec=color, mew=2, elinewidth=2, capsize=0, zorder=3)


FIGS = {}


def save(fig, name, caption):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=150)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.svg"))
    buf = io.BytesIO()
    fig.savefig(buf, format="svg")
    FIGS[name] = dict(svg=base64.b64encode(buf.getvalue()).decode(), caption=caption)
    plt.close(fig)


def family_panels(width_ratios=(3, 1, 1)):
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, gridspec_kw=dict(width_ratios=width_ratios, wspace=0.35))
    return fig, dict(zip(["olmo3-7b", "qwen3-8b", "gemma4-31b"], axes))


def corpus_legend(fig, corpora=("oasst1", "ultrachat", "wildchat"), extra=()):
    h = [Line2D([], [], marker=CORP_MARK[c], ls="", color=INK2, mfc=INK2 if c != "wildchat" else "#fcfcfb", mew=2, ms=9,
                label=CORP_NAME[c] + (" (secondary)" if c == "wildchat" else "")) for c in corpora]
    fig.legend(handles=h + list(extra), loc="lower center", ncol=len(h) + len(extra), frameon=False, bbox_to_anchor=(0.5, 0.0))


# --- figures -----------------------------------------------------------------------------------------------------------------

def fig_vr_asymmetry(r):
    fig, ax = family_panels()
    for f, a in ax.items():
        pts = [(fx, s) for fx, s in POST if fx == f]
        for i, (_, s) in enumerate(pts):
            cell = r["vr_nll"]["cells"][f"{f}|{s}"]
            for j, c in enumerate(["oasst1", "ultrachat", "wildchat"]):
                dot(a, i + (j - 1) * 0.22, cell[c]["nats"]["asymmetry"], FAM_COLOR[f], CORP_MARK[c], filled=c != "wildchat")
        a.set_xticks(range(len(pts)), [ck(f, s) for _, s in pts])
        a.set_xlim(-0.6, len(pts) - 0.4)
        style(a, floor=0.05)
        a.set_title(FAM_NAME[f], color=FAM_COLOR[f], fontweight="bold")
    ax["olmo3-7b"].set_ylabel("asymmetry, nats/token\nΔ user frame cost − Δ reply frame cost")
    fig.suptitle("Vestigial regression, NLL arm: asymmetry by checkpoint (change from own base, 95% CI)", x=0.06, ha="left", fontsize=18)
    corpus_legend(fig, extra=[Patch(color="#ecebe6", label="±0.05-nat floor")])
    fig.subplots_adjust(left=0.1, right=0.98, top=0.86, bottom=0.17)
    save(fig, "fig1_vr_asymmetry", "Asymmetry (nats/token; panels have their own scales, so compare within a family). "
         "It fires only for Gemma 4 31B IT. OLMo DPO and Instruct are clearly negative, the opposite of regression; OLMo SFT and Qwen3-8B IT sit near zero.")


def fig_vr_framecost(r):
    fig, ax = family_panels()
    for f, a in ax.items():
        pts = [(fx, s) for fx, s in POST if fx == f]
        for i, (_, s) in enumerate(pts):
            cell = r["vr_nll"]["cells"][f"{f}|{s}"]
            for j, (who, c) in enumerate([("user", "oasst1"), ("user", "ultrachat"), ("reply", "oasst1"), ("reply", "ultrachat")]):
                dot(a, i + (j - 1.5) * 0.17, cell[c]["nats"][f"Δ FC {who}"], FAM_COLOR[f], CORP_MARK[c], filled=who == "user")
        a.set_xticks(range(len(pts)), [ck(f, s) for _, s in pts])
        a.set_xlim(-0.6, len(pts) - 0.4)
        style(a, floor=0.05)
        a.set_title(FAM_NAME[f], color=FAM_COLOR[f], fontweight="bold")
    ax["olmo3-7b"].set_ylabel("Δ frame cost from base, nats/token\n(tagged − plain)")
    fig.suptitle("Where the asymmetry comes from: user-turn and reply frame costs, separately", x=0.06, ha="left", fontsize=18)
    h = [Line2D([], [], marker="o", ls="", color=INK2, ms=9, label="user turn (filled)"),
         Line2D([], [], marker="o", ls="", color=INK2, mfc="#fcfcfb", mew=2, ms=9, label="reply (hollow)"),
         Line2D([], [], marker="o", ls="", color=INK2, ms=9, label="circle = oasst1"),
         Line2D([], [], marker="s", ls="", color=INK2, ms=9, label="square = UltraChat")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.86, bottom=0.17)
    save(fig, "fig2_vr_framecost", "Δ frame cost of the user turn (filled) and the reply (hollow), headline corpora. Gemma IT: the user turn's "
         "frame cost rises and the reply's falls. OLMo DPO/Instruct: the reply's frame cost rises by about +0.52 and the user turn's falls slightly "
         "(−0.07 to −0.14), so the asymmetry is negative. "
         "Qwen3-8B IT: both rise by similar amounts, so they cancel.")


def fig_vr_plain(r):
    fig, a = plt.subplots(figsize=FIGSIZE)
    for i, (f, s) in enumerate(POST):
        cell = r["vr_nll"]["cells"][f"{f}|{s}"]
        for j, (who, c) in enumerate([("user", "oasst1"), ("user", "ultrachat"), ("reply", "oasst1"), ("reply", "ultrachat")]):
            dot(a, i + (j - 1.5) * 0.17, cell[c]["bpb"][f"Δ {who} plain"], FAM_COLOR[f], CORP_MARK[c], filled=who == "user")
    a.set_xticks(range(len(POST)), [ck(f, s) for f, s in POST])
    style(a)
    a.set_ylabel("Δ NLL of plain text from base, bits/byte")
    a.set_title("Plain-text change from base (\"non-assistant degradation\", referred to overgeneralization)", loc="left")
    h = [Line2D([], [], marker="o", ls="", color=INK2, ms=9, label="user text (filled)"),
         Line2D([], [], marker="o", ls="", color=INK2, mfc="#fcfcfb", mew=2, ms=9, label="reply text (hollow)"),
         Line2D([], [], marker="o", ls="", color=INK2, ms=9, label="circle = oasst1"),
         Line2D([], [], marker="s", ls="", color=INK2, ms=9, label="square = UltraChat")]
    a.legend(handles=h, loc="upper left", frameon=False)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.1)
    save(fig, "fig3_vr_plain", "Plain-text NLL change from base in bits/byte (comparable across families). Gemma 4 31B IT predicts untagged "
         "text much worse than its base (+0.85 to +1.7); OLMo DPO and Instruct rise moderately (+0.12 to +0.21); Qwen3-8B IT and OLMo SFT change little (under 0.1). Not part of the vestigial-regression verdict.")


def fig_judge_shares(r):
    fig, a = plt.subplots(figsize=FIGSIZE)
    order = ["user_voice", "user_voice_with_assistant_traits", "switched_to_assistant", "degenerate", "invalid"]
    rows = list(reversed(ALL))
    for y, (f, s) in enumerate(rows):
        sh = r["judge_shares"][f"{f}|{s}"]
        left = 0
        for cat in order:
            w = sh[cat]
            a.barh(y, w, left=left, height=0.7, color=CAT_COLOR[cat], edgecolor="#fcfcfb" if cat != "invalid" else "#8a8984",
                   hatch="///" if cat == "invalid" else None, lw=2 if cat != "invalid" else 0.6)
            left += w
        a.add_patch(plt.Rectangle((-0.018, y - 0.35), 0.01, 0.7, color=FAM_COLOR[f], clip_on=False))
        nu = sh["switched_to_assistant"] + sh["degenerate"]
        a.text(1.01, y, f"{nu:.0%}", va="center", fontsize=13, color=INK)
    a.set_yticks(range(len(rows)), [ck(f, s, short=False) for f, s in rows])
    a.set_xlim(0, 1)
    a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    a.tick_params(axis="y", length=0, pad=30)
    a.set_title("Judge arm: labels of 600 continuations per checkpoint (right: non-user share)", loc="left")
    a.legend(handles=[Patch(facecolor=CAT_COLOR[c], edgecolor="#8a8984" if c == "invalid" else CAT_COLOR[c], hatch="///" if c == "invalid" else None,
                            label=CAT_NAME[c]) for c in order], loc="lower center", bbox_to_anchor=(0.45, -0.2), ncol=3, frameon=False)
    fig.subplots_adjust(left=0.2, right=0.93, top=0.92, bottom=0.2)
    save(fig, "fig4_judge_shares", "Judge labels per checkpoint (Claude Sonnet 5.5, prompt v2; unvalidated). Gemma IT is mostly degenerate, "
         "Qwen3-8B IT switches to the assistant more than its base, and OLMo's trained stages stay in the user's voice far more than OLMo base does.")


def fig_judge_change(r):
    fig, a = plt.subplots(figsize=FIGSIZE)
    for i, (f, s) in enumerate(POST):
        jc = r["judge"][f"{f}|{s}"]
        for j, rd in enumerate(["exclude", "as_user", "as_nonuser"]):
            c = {k: jc[rd][k] * 100 for k in ("mean", "lo", "hi")}
            dot(a, i + (j - 1) * 0.2, c, FAM_COLOR[f], READ_MARK[rd], filled=rd != "as_nonuser")
    a.set_xticks(range(len(POST)), [ck(f, s) for f, s in POST])
    style(a)
    a.set_ylabel("Δ non-user-voice rate from base, percentage points\n(switched + degenerate; bootstrap over prefills)")
    a.set_title("Judge arm: change in non-user-voice rate (95% CI), three treatments of refused samples", loc="left")
    a.legend(handles=[Line2D([], [], marker=READ_MARK[rd], ls="", color=INK2, mfc=INK2 if rd != "as_nonuser" else "#fcfcfb", mew=2, ms=9,
                             label=READ_NAME[rd]) for rd in READ_NAME], loc="upper left", frameon=False)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.1)
    save(fig, "fig5_judge_change", "Change in the non-user-voice rate from base. Gemma IT (+62 points) and Qwen3-8B IT (+14 to +17 points) rise; "
         "all three OLMo stages fall by about 40 points. The three treatments of refused samples agree.")


def fig_arms(r):
    fig, a = plt.subplots(figsize=FIGSIZE)
    for p in r["arms"]["points"]:
        f, s = p["family"], p["stage"]
        cell = r["vr_nll"]["cells"][f"{f}|{s}"]
        xs = [cell[c]["bpb"]["asymmetry"]["mean"] for c in ("oasst1", "ultrachat")]
        jc = r["judge"][f"{f}|{s}"]["exclude"]
        a.errorbar([p["asym_bpb"]], [jc["mean"] * 100], xerr=[[p["asym_bpb"] - min(xs)], [max(xs) - p["asym_bpb"]]],
                   yerr=[[(jc["mean"] - jc["lo"]) * 100], [(jc["hi"] - jc["mean"]) * 100]], fmt="o", ms=11, color=FAM_COLOR[f],
                   elinewidth=2, capsize=0, zorder=3)
        off = {"olmo3-7b|dpo": (-95, 18), "olmo3-7b|instruct": (-60, -32), "olmo3-7b|sft": (12, 10), "qwen3-8b|instruct": (12, 8),
               "gemma4-31b|instruct": (-170, 12)}[f"{f}|{s}"]
        a.annotate(ck(f, s), (p["asym_bpb"], jc["mean"] * 100), xytext=off, textcoords="offset points", fontsize=14, color=INK)
    style(a)
    a.set_xlim(-0.6, 1.8)
    a.axvline(0, color="#8a8984", lw=1.2, zorder=1)
    a.set_xlabel("NLL asymmetry, bits/byte (mean of oasst1 and UltraChat; bar = range of the two)")
    a.set_ylabel("Δ non-user-voice rate, points (refused excluded)")
    rho = r["arms"]["rho"]
    a.set_title(f"Do the two arms agree? Spearman ρ = {rho['exclude']:.2f} over 5 post-trained points (descriptive)", loc="left")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.12)
    save(fig, "fig6_arms", f"The two arms across the five post-trained checkpoints. ρ = {rho['exclude']:.2f} (refused excluded or counted as "
         f"user voice), {rho['as_nonuser']:.2f} (counted as non-user). This describes 5 points and is not a test. Gemma IT is extreme on both arms. "
         "Qwen3-8B IT disagrees: the judge shows a rise, while the NLL asymmetry is near zero.")


def fig_ld_contrast(r):
    fig, ax = family_panels()
    for f, a in ax.items():
        pts = [(fx, s) for fx, s in POST if fx == f]
        for i, (_, s) in enumerate(pts):
            cell = r["ld"][f"{f}|{s}"]["cells"]
            for j, c in enumerate(["oasst1", "ultrachat", "wildchat"]):
                dot(a, i + (j - 1) * 0.22, cell[c]["contrast_paired"], FAM_COLOR[f], CORP_MARK[c], filled=c != "wildchat")
        a.set_xticks(range(len(pts)), [ck(f, s) for _, s in pts])
        a.set_xlim(-0.6, len(pts) - 0.4)
        style(a, floor=0.05)
        a.set_title(FAM_NAME[f], color=FAM_COLOR[f], fontweight="bold")
    ax["olmo3-7b"].set_ylabel("Δ gap (assistant) − Δ gap (user), nats/token\ngap = NLL − entropy, tagged, positions 1+")
    fig.suptitle("Logit distortion (PROVISIONAL): assistant-persona gap change minus user-persona gap change", x=0.06, ha="left", fontsize=18)
    corpus_legend(fig, extra=[Patch(color="#ecebe6", label="±0.05-nat floor")])
    fig.subplots_adjust(left=0.1, right=0.98, top=0.86, bottom=0.17)
    save(fig, "fig7_ld_contrast", "Logit-distortion contrast, CI paired over exchanges; panels have their own scales. Positive means the "
         "assistant persona's gap grew more than the user persona's. OLMo DPO and Instruct are positive in both headline corpora. "
         "OLMo SFT, Qwen3-8B IT and especially Gemma 4 31B IT are negative: the user persona's gap grew more.")


def fig_ld_components(r, fams, name, title):
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharey=True, gridspec_kw=dict(wspace=0.08))
    pts = [(f, s) for f, s in POST if f in fams]
    for a, q, qn in zip(axes, ("nll", "entropy", "gap"), ("Δ NLL", "Δ entropy", "Δ gap = Δ NLL − Δ entropy")):
        for i, (f, s) in enumerate(pts):
            cell = r["ld"][f"{f}|{s}"]["cells"]
            for j, (who, c) in enumerate([("user", "oasst1"), ("user", "ultrachat"), ("reply", "oasst1"), ("reply", "ultrachat")]):
                dot(a, i + (j - 1.5) * 0.17, cell[c][who][q], FAM_COLOR[f], CORP_MARK[c], filled=who == "user", size=8)
        a.set_xticks(range(len(pts)), [ck(f, s) for f, s in pts], rotation=0 if len(pts) < 3 else 30, ha="center" if len(pts) < 3 else "right")
        a.set_xlim(-0.6, len(pts) - 0.4)
        style(a, floor=0.05)
        a.set_title(qn)
    axes[0].set_ylabel("change from base, nats/token (tagged, positions 1+)")
    fig.suptitle(title, x=0.06, ha="left", fontsize=18)
    h = [Line2D([], [], marker="o", ls="", color=INK2, ms=9, label="user persona (filled)"),
         Line2D([], [], marker="o", ls="", color=INK2, mfc="#fcfcfb", mew=2, ms=9, label="assistant persona (hollow)"),
         Line2D([], [], marker="o", ls="", color=INK2, ms=9, label="circle = oasst1 (human)"),
         Line2D([], [], marker="s", ls="", color=INK2, ms=9, label="square = UltraChat (model)")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.2 if len(pts) >= 3 else 0.15)
    return fig


def fig_ld_positions(r):
    bins = ["1-10", "11-25", "26-50", "51-100", "101-300"]
    fig, axes = plt.subplots(2, 5, figsize=FIGSIZE, sharex=True, gridspec_kw=dict(hspace=0.25, wspace=0.35))
    for col, (f, s) in enumerate(POST):
        pos = r["ld_positions"][f"{f}|{s}"]
        for row, c in enumerate(("oasst1", "ultrachat")):
            a = axes[row, col]
            for who, ls in (("user", "-"), ("reply", "--")):
                xs, m, lo, hi = [], [], [], []
                for k, b in enumerate(bins):
                    v = pos.get(f"{c}|{who}|{b}")
                    if v:
                        xs.append(k); m.append(v["mean"]); lo.append(v["lo"]); hi.append(v["hi"])
                a.plot(xs, m, ls=ls, color=FAM_COLOR[f], lw=2, marker="o" if who == "user" else "s", ms=6,
                       mfc=FAM_COLOR[f] if who == "user" else "#fcfcfb", mew=1.5)
                a.fill_between(xs, lo, hi, color=FAM_COLOR[f], alpha=0.15, lw=0)
            style(a)
            if row == 0:
                a.set_title(ck(f, s), fontsize=14, color=FAM_COLOR[f], fontweight="bold")
            if col == 0:
                a.set_ylabel(f"{CORP_NAME[c]}\nΔ gap, nats", fontsize=13)
            a.set_xticks(range(len(bins)), bins, rotation=45, fontsize=10)
    fig.suptitle("Logit distortion (PROVISIONAL): Δ gap by content position (descriptive, criterion 2)", x=0.06, ha="left", fontsize=17)
    fig.legend(handles=[Line2D([], [], color=INK2, lw=2, marker="o", label="user persona"),
                        Line2D([], [], color=INK2, lw=2, ls="--", marker="s", mfc="#fcfcfb", label="assistant persona")],
               loc="lower center", ncol=2, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.16)
    save(fig, "fig10_ld_positions", "Δ gap by content-position bin (tagged; bootstrap over exchanges; each panel has its own scale). "
         "In OLMo DPO and Instruct the assistant persona's extra gap is largest in the first positions and shrinks with position. It stays above the user's "
         "through position 100 (oasst1) and 300 (UltraChat). In Qwen3-8B IT the assistant gap is larger only in the first ~10 positions.")


def build_figures(r):
    fig_vr_asymmetry(r)
    fig_vr_framecost(r)
    fig_vr_plain(r)
    fig_judge_shares(r)
    fig_judge_change(r)
    fig_arms(r)
    fig_ld_contrast(r)
    save(fig_ld_components(r, ("olmo3-7b", "qwen3-8b"), "fig8", "Logit distortion (PROVISIONAL): NLL, entropy and gap changes, OLMo and Qwen"),
         "fig8_ld_components_olmo_qwen", "Raw changes behind the gap, OLMo and Qwen. Entropy falls for both personas everywhere, which is sharpening. "
         "The assistant persona's larger gap at OLMo DPO and Instruct comes mostly from NLL rising on other authors' replies (+1.1 to +1.2 nats), not from entropy falling.")
    save(fig_ld_components(r, ("gemma4-31b",), "fig9", "Logit distortion (PROVISIONAL): NLL, entropy and gap changes, Gemma 4 31B IT"),
         "fig9_ld_components_gemma", "Gemma 4 31B IT (100 exchanges per corpus). The user persona's NLL rises by about 8.4 nats with entropy "
         "falling, so its gap explodes: the inverse of logit distortion, and the same collapse the vestigial-regression arms see.")
    fig_ld_positions(r)


# --- HTML --------------------------------------------------------------------------------------------------------------------

def f3(c, nd=2, scale=1.0, unit=""):
    return f"{c['mean'] * scale:+.{nd}f} [{c['lo'] * scale:+.{nd}f}, {c['hi'] * scale:+.{nd}f}]{unit}"


def esc(s):
    return html.escape(str(s))


def figure(name):
    fg = FIGS[name]
    return (f'<figure id="{name}"><img alt="{esc(fg["caption"])}" src="data:image/svg+xml;base64,{fg["svg"]}">'
            f'<figcaption><b>{name.split("_")[0].replace("fig", "Figure ")}.</b> {esc(fg["caption"])} '
            f'<span class="files">(<code>reports/figures/{name}.png</code>, <code>.svg</code>)</span></figcaption></figure>')


def verdict_rows(r):
    out = []
    for f, s in POST:
        k = f"{f}|{s}"
        v = r["vr_nll"]["verdict"][k]
        cell = r["vr_nll"]["cells"][k]
        nll_txt = " / ".join(f3(cell[c]["nats"]["asymmetry"]) for c in ("oasst1", "ultrachat"))
        nll_fired = v["fires"]
        nll_lbl = "FIRES" if nll_fired else ("no (negative)" if "negative" in v["reading"] else "no")
        jd = r["judge"][k]
        j_fired = {rd: jd[rd]["fires"] for rd in ("exclude", "as_user", "as_nonuser")}
        j_agree = len(set(j_fired.values())) == 1
        j_lbl = ("FIRES" if j_fired["exclude"] else ("no (falls)" if jd["exclude"]["hi"] < 0 else "no")) + ("" if j_agree else " (reading-dependent)")
        ld = r["ld"][k]
        l_agree = len(set(ld["fires"].values())) == 1
        l_fired = ld["fires"]["paired"]
        lc = ld["cells"]
        l_txt = " / ".join(f3(lc[c]["contrast_paired"]) for c in ("oasst1", "ultrachat"))
        l_lbl = ("FIRES" if l_fired else ("no (opposite)" if all(lc[c]["contrast_paired"]["hi"] < 0 for c in ("oasst1", "ultrachat")) else "no")) + \
            ("" if l_agree else " (reading-dependent)")
        # hypothesis
        if nll_fired and j_fired["exclude"]:
            hyp = "H4 (vestigial regression, both arms)"
        elif nll_fired:
            hyp = "H4 (vestigial regression, NLL arm only)"
        elif j_fired["exclude"]:
            hyp = ("H4 (vestigial regression, judge arm only) if the unvalidated judge counts; otherwise "
                   + ("H2" if l_fired else "H1") + ", unless overgeneralization fires (then H4)")
        elif l_fired:
            hyp = "H2 (logit distortion only), unless overgeneralization fires (then H4)"
        else:
            hyp = "H1 (no measure fired), unless overgeneralization fires (then H4)"
        out.append(dict(ck=ck(f, s, short=False), f=f, nll=(nll_lbl, nll_txt, nll_fired), judge=(j_lbl, f3(jd["exclude"], 1, 100, " pts"), j_fired["exclude"]),
                        ld=(l_lbl, l_txt, l_fired), hyp=hyp))
    return out


def cls(fired):
    return "fire" if fired else "nofire"


CSS = """
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#7a7974;--line:#e4e3df;--accent:#2a78d6;--fire:#0b5d1e;--firebg:#e3f3e6;
--warnbg:#fdf3dc;--warnline:#e8c66a;--nofire:#52514e;--code:#f2f1ed}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--card:#232322;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9c9b93;--line:#3a3a37;
--accent:#3987e5;--fire:#8fe0a0;--firebg:#1f3a25;--warnbg:#3a3220;--warnline:#8a7230;--nofire:#c3c2b7;--code:#2c2c2a}}
:root[data-theme="dark"]{--bg:#1a1a19;--card:#232322;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9c9b93;--line:#3a3a37;--accent:#3987e5;--fire:#8fe0a0;
--firebg:#1f3a25;--warnbg:#3a3220;--warnline:#8a7230;--nofire:#c3c2b7;--code:#2c2c2a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 80px}h1{font-size:2rem;margin:.2em 0}h2{font-size:1.5rem;margin-top:2.2em;border-bottom:1px solid var(--line);padding-bottom:.25em}
h3{font-size:1.15rem;margin-top:1.6em}p,li{max-width:80ch}.sub{color:var(--ink2)}code{background:var(--code);padding:1px 4px;border-radius:4px;font-size:.88em}
nav{font-size:.95rem;color:var(--ink2)}nav a{color:var(--accent);margin-right:.9em;text-decoration:none}
.tbl{overflow-x:auto;margin:1em 0}table{border-collapse:collapse;width:100%;font-size:.92rem;background:var(--card)}
th,td{border:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}th{background:var(--code);font-weight:600}
td.fire{background:var(--firebg);color:var(--fire)}td.fire b,td.nofire b{display:block}td.nofire{color:var(--nofire)}
.num{font-variant-numeric:tabular-nums;white-space:nowrap;font-size:.88em}.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
.warn{background:var(--warnbg);border:1px solid var(--warnline);border-radius:8px;padding:12px 16px;margin:1em 0}.warn li{margin:.25em 0}
figure{margin:1.5em 0;background:#fcfcfb;border:1px solid var(--line);border-radius:8px;padding:8px}figure img{width:100%;height:auto;display:block}
figcaption{color:#2b2b29;font-size:.92rem;padding:8px 6px 2px}.files{color:#7a7974;font-size:.85em}
.tag{display:inline-block;font-size:.75rem;padding:1px 7px;border-radius:10px;border:1px solid var(--line);color:var(--ink2);margin-left:6px;vertical-align:middle}
.takeaways li{margin:.45em 0}
"""


def table(header, rows, raw=False):
    h = "".join(f"<th>{esc(x)}</th>" for x in header)
    body = "".join("<tr>" + "".join(f"<td>{c if raw else esc(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="tbl"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def build_html(r):
    V = verdict_rows(r)
    val = r["validation"]
    jr = r["judge"]
    rho = r["arms"]["rho"]
    D = "docs/decisions.md"

    def dlink(title):
        return f'<a href="../{D}">{esc(title)}</a>'

    missing = [
        f"<b>Judge validation: no human labels.</b> {val['validation_labelled']} of {val['validation_sheet_rows']} rows of "
        "<code>judge_validation/sheet.csv</code> are labelled, so the judge's agreement and Cohen's kappa against human labels <b>cannot be computed</b>. "
        "The README requires validation before use, so the judge arm is <b>unvalidated</b> and every judge-arm result below is provisional. "
        "The only agreement figure is the pilot check against the pilot agent's own labels (binary 90.9%, 3-way 93.3% on 90 rows; not ground truth).",
        f"<b>73 samples the judge refused ({val['refusal_labelled']} hand-labelled so far)</b>, out of 4,800. They are on the blind sheet "
        "<code>judge_validation/refusals/sheet.csv</code>. Results are shown with refused samples excluded, counted as user voice, and counted as non-user voice; "
        "all three give the same verdicts.",
        "<b>Overgeneralization (Adeeb) is not analysed here.</b> The verdict table has a placeholder column, so no checkpoint can be finally assigned H1 or H2.",
        "<b>Logit distortion is PROVISIONAL.</b> Grace's README criteria are not frozen and still describe the isolated-slot design (texts scored alone, "
        "<code>asst@asst</code>); <code>measures/logit_distortion/analyze.py</code> is a stub, so the numbers here are computed by this report from the "
        "exchange-design scores, pending her review. Her README also names <i>original multi-exchange conversations</i> (<code>core.score_conversations</code>) "
        "as the primary data route. Pilot conversation scores exist, uncommitted, in her worktree; they are not used here.",
        "<b>No interpretation table exists in the logit-distortion README or docs.</b> The classification shown (\"learned the role\", \"gap rising\", "
        "\"general sharpening\") follows the scheme in the analysis request and is labelled exploratory.",
        "<b>Minimum judge-arm rate difference: still an OPEN decision</b> (README, decisions.md 2026-11-02). Only \"rises with the CI excluding zero\" is applied.",
        "<b><code>docs/turns_memo.md</code> is not on <code>main</code></b>. It is on branch <code>turns-memo-2026-11-03</code>, read from there for Section 5.",
        "<b>The judge labels, flags and judge fixes were not on <code>main</code>.</b> They were in five unpushed commits on <code>capstone-scoring-2026-11-03</code> "
        "(4440bb8, c38fbf8, 59fbaae, 008b7d1, d9aafff), which this report's branch merges.",
    ]
    ambig = [
        "<b>Judge temperature and prompt version.</b> The VR README says the judge runs \"at temperature 0\" with prompt \"(v1)\". The run used prompt v2 "
        "(decided 2026-11-03, before generation) and the API's default temperature, because Sonnet 5.5 rejects temperature 0 (decisions.md 2026-11-04, logged before the run). "
        "The judge is therefore not deterministic.",
        "<b>Does an unvalidated judge arm count toward H4?</b> The README requires validation \"before use\"; it does not say what happens to the verdict until then. "
        "Both readings are shown (it matters only for Qwen3-8B IT).",
        "<b>Logit distortion: bootstrap pairing.</b> The README asks for a CI of the difference \"over texts\" but predates the exchange design, where each "
        "exchange supplies both the user turn and the reply. Shown both ways: resampling exchanges jointly (paired; used in figures) and resampling the two sets "
        "independently (as the VR asymmetry does). The verdicts agree.",
        "<b>Logit distortion: the model-written column.</b> The README pairs model-written user turns (UltraChat) with WildChat ChatGPT replies. Under the exchange "
        "design UltraChat has its own (ChatGPT) replies. Shown both ways: UltraChat user vs UltraChat reply (used in figures) and UltraChat user vs WildChat reply. "
        "The verdicts agree.",
        "<b>Logit distortion: the reply's context.</b> The README scores assistant text alone (<code>asst@asst</code>); here the reply follows its own user turn "
        "(the exchange design). The turns memo found the reply gap barely depends on that context (≤ 0.09 nats Qwen, up to 0.36 OLMo).",
        "<b>Spearman between the arms.</b> The README's pre-registered prediction spans all 8 points; base points are zero by construction, and leaving them out "
        "(5 points) was an implementation-time choice, not pre-registered. The 5-point version is shown.",
        "<b>The measure's <code>judge_change</code> counts refused rows as user voice</b> (in the denominator, never non-user). It predates refusals. This "
        "is not treated as a bug, and nothing was changed: it is the \"refused = user voice\" reading, and it gives the same verdicts as the other two.",
    ]

    vt = []
    for v in V:
        vt.append("<tr>"
                  f'<td><span class="swatch" style="background:{FAM_COLOR[v["f"]]}"></span><b>{esc(v["ck"])}</b></td>'
                  f'<td class="{cls(v["nll"][2])}"><b>{esc(v["nll"][0])}</b><span class="num">{esc(v["nll"][1])}</span></td>'
                  f'<td class="{cls(v["judge"][2])}"><b>{esc(v["judge"][0])}</b><span class="num">{esc(v["judge"][1])}</span></td>'
                  f'<td class="{cls(v["ld"][2])}"><b>{esc(v["ld"][0])}</b><span class="num">{esc(v["ld"][1])}</span></td>'
                  f'<td class="nofire"><b>pending</b><span class="num">Adeeb</span></td>'
                  f'<td>{esc(v["hyp"])}</td></tr>')
    verdict_table = ('<div class="tbl"><table><thead><tr><th>checkpoint (vs own base)</th><th>VR, NLL arm<br><span class="num">asymmetry, nats/token, oasst1 / UltraChat</span></th>'
                     '<th>VR, judge arm <span class="tag">unvalidated</span><br><span class="num">Δ non-user-voice rate</span></th>'
                     '<th>Logit distortion <span class="tag">provisional</span><br><span class="num">Δgap asst − Δgap user, oasst1 / UltraChat</span></th>'
                     '<th>Overgeneral&shy;ization</th><th>hypothesis</th></tr></thead><tbody>' + "".join(vt) + "</tbody></table></div>")

    q = r["vr_nll"]["cells"]
    takeaways = [
        f"<b>Gemma 4 31B IT shows vestigial regression on both arms.</b> NLL asymmetry {f3(q['gemma4-31b|instruct']['oasst1']['nats']['asymmetry'])} / "
        f"{f3(q['gemma4-31b|instruct']['ultrachat']['nats']['asymmetry'])} nats/token. Its non-user-voice rate rises {f3(jr['gemma4-31b|instruct']['exclude'], 0, 100)} points, "
        "and most of its continuations are degenerate loops. This is the positive control, and both arms detect it (<a href=\"#fig1_vr_asymmetry\">Fig. 1</a>, <a href=\"#fig5_judge_change\">Fig. 5</a>).",
        "<b>OLMo 3 7B shows no vestigial regression at any stage.</b> The NLL asymmetry is null at SFT and <i>negative</i> at DPO and Instruct, mostly because the reply's frame "
        "cost rises. On the judge, all three OLMo stages stay in the user's voice more than OLMo base (about −40 points) (<a href=\"#fig2_vr_framecost\">Fig. 2</a>, <a href=\"#fig4_judge_shares\">Fig. 4</a>).",
        f"<b>Qwen3-8B IT: the arms disagree.</b> The NLL asymmetry is null ({f3(q['qwen3-8b|instruct']['oasst1']['nats']['asymmetry'])} / {f3(q['qwen3-8b|instruct']['ultrachat']['nats']['asymmetry'])}). "
        f"The unvalidated judge shows a rise of {f3(jr['qwen3-8b|instruct']['exclude'], 1, 100)} points, mostly switches to the assistant "
        "(often through its think block). So the verdict for Qwen hinges on whether the judge arm counts before validation (<a href=\"#fig6_arms\">Fig. 6</a>).",
        "<b>Logit distortion (provisional) fires for OLMo DPO and Instruct only</b>, and it comes from NLL rising on other authors' replies more than from sharpening. "
        "Entropy falls for both personas in every model (general sharpening). OLMo DPO/Instruct's assistant gap grows more mainly because their NLL on human and ChatGPT replies rises by about 1.1 nats. "
        "That is consistent with surprise at off-policy replies, so it may not be a distorted assistant distribution (<a href=\"#fig8_ld_components_olmo_qwen\">Fig. 8</a>).",
        "<b>No checkpoint has a final hypothesis yet.</b> Gemma IT is H4 even on the NLL arm alone. Every other checkpoint waits on overgeneralization, and Qwen also on judge validation.",
    ]

    # data checks table
    sc = [[ck(x["family"], x["stage"], False), x["conditions"], " / ".join(str(x["per_corpus"][c]) for c in ("oasst1", "ultrachat", "wildchat")),
           x["missing_values"], " / ".join(str(x["truncated_replies"][c]) for c in ("oasst1", "ultrachat", "wildchat")), "ok" if x["ok"] else "PROBLEM"]
          for x in r["score_checks"]]
    jc = [[ck(x["family"], x["stage"], False), x["n"], x["prefills"], x["prefill_set"], ", ".join(x["judge_model"]) + " / " + ", ".join(x["prompt_version"]),
           " / ".join(str(x["counts"][l]) for l in ("user_voice", "user_voice_with_assistant_traits", "switched_to_assistant", "degenerate", "invalid"))]
          for x in r["judge_checks"]]
    tokmax = max(v["nll"] for v in r["token_consistency"].values())
    rep = r["vr_nll"]["reproduction"]

    # VR detail table
    vr_rows = []
    for f, s in POST:
        c = q[f"{f}|{s}"]
        for corp in ("oasst1", "ultrachat", "wildchat"):
            n = c[corp]["nats"]
            vr_rows.append([ck(f, s), CORP_NAME[corp] + (" (secondary)" if corp == "wildchat" else ""), f3(n["asymmetry"]), f3(c[corp]["bpb"]["asymmetry"]),
                            f3(n["Δ FC user"]), f3(n["Δ FC reply"]), f3(n["Δ user tagged"]), f3(n["Δ user plain"]), f3(n["Δ reply tagged"]), f3(n["Δ reply plain"])])
    plain_rows = [[ck(f, s)] + [f3(q[f"{f}|{s}"][corp]["bpb"][f"Δ {w} plain"]) for w in ("user", "reply") for corp in ("oasst1", "ultrachat")] for f, s in POST]

    j_rows = []
    for f, s in POST:
        x = jr[f"{f}|{s}"]
        j_rows.append([ck(f, s), f"{x['exclude']['base_rate']:.1%} → {x['exclude']['stage_rate']:.1%}", f3(x["exclude"], 1, 100),
                       f3(x["as_user"], 1, 100), f3(x["as_nonuser"], 1, 100), "rises, CI > 0" if x["exclude"]["fires"] else "does not rise",
                       f3(x["category4"], 1, 100)])
    arm_rows = []
    for p, v in zip(r["arms"]["points"], V):
        both = v["nll"][2] and v["judge"][2]
        lab = "both arms" if both else "NLL only" if v["nll"][2] else "judge only" if v["judge"][2] else "neither"
        arm_rows.append([ck(p["family"], p["stage"]), f"{p['asym_bpb']:+.3f}", f"{p['judge_exclude'] * 100:+.1f}", lab])

    base_nu = ", ".join(f"{FAM_NAME[f]} base {r['judge_shares'][f'{f}|base']['switched_to_assistant'] + r['judge_shares'][f'{f}|base']['degenerate']:.0%}"
                        for f in ("olmo3-7b", "qwen3-8b", "gemma4-31b"))
    ld = r["ld"]
    ld_rows = []
    for f, s in POST:
        cc = ld[f"{f}|{s}"]["cells"]
        for corp in ("oasst1", "ultrachat", "wildchat"):
            cell = cc[corp]
            ld_rows.append([ck(f, s), CORP_NAME[corp] + (" (secondary)" if corp == "wildchat" else ""), f3(cell["contrast_paired"]), f3(cell["contrast_independent"]),
                            f3(cell["user"]["gap"]), f3(cell["reply"]["gap"]), f3(cell["contrast_nll"]), f3(cell["contrast_entropy"]),
                            cell.get("contrast_class", "(secondary, not classified)")])
    ld_alt = [[ck(f, s), f3(ld[f"{f}|{s}"]["cells"]["alt_model_column"]), "fires" if ld[f"{f}|{s}"]["fires"]["alt_model_column"] else "does not fire",
               "fires" if ld[f"{f}|{s}"]["fires"]["paired"] else "does not fire", "fires" if ld[f"{f}|{s}"]["fires"]["independent"] else "does not fire"] for f, s in POST]
    cls_rows = []
    for f, s in POST:
        cc = ld[f"{f}|{s}"]["cells"]
        for corp in ("oasst1", "ultrachat"):
            for who, wn in (("user", "user"), ("reply", "assistant")):
                p_ = cc[corp][who]
                cls_rows.append([ck(f, s), CORP_NAME[corp], wn, f3(p_["nll"]), f3(p_["entropy"]), f3(p_["gap"]), p_["class"], p_["driver"]])

    turns_rows = [
        ["OLMo 3 7B Instruct", "asymmetry A (reply alone)", "−0.24 [−0.38, −0.12]", "−0.38 [−0.48, −0.28]", "−0.19 [−0.27, −0.11]"],
        ["", "asymmetry B (reply after its question)", "−0.49 [−0.66, −0.34]", "−0.69 [−0.80, −0.58]", "−0.66 [−0.76, −0.55]"],
        ["", "design bias Δ[FC rA − FC rB]", "−0.25 [−0.36, −0.14]", "−0.31 [−0.39, −0.23]", "−0.47 [−0.59, −0.35]"],
        ["Qwen3-8B IT", "asymmetry A (reply alone)", "+0.04 [−0.08, +0.17]", "+0.07 [−0.05, +0.19]", "+0.06 [−0.05, +0.17]"],
        ["", "asymmetry B (reply after its question)", "−0.10 [−0.23, +0.02]", "−0.07 [−0.19, +0.06]", "−0.24 [−0.40, −0.10]"],
        ["", "design bias Δ[FC rA − FC rB]", "−0.15 [−0.26, −0.04]", "−0.13 [−0.22, −0.05]", "−0.30 [−0.45, −0.18]"],
    ]
    prereg = [
        ["Hypotheses H1/H2/H4 and the decision rule", "2026-11-01, before any data", "set before data", dlink("2026-11-01: H3 dropped")],
        ["0.05-nat smallest effect (with CI excluding zero)", "2026-11-01", "set before data", dlink("2026-11-01: smallest effect 0.05 nats")],
        ["VR NLL arm: asymmetry as primary quantity", "2026-11-02, after the pilot", "revised after pilot results: OLMo NLL verdicts are not blind",
         dlink("2026-11-02: the asymmetry as the NLL arm's primary quantity (pilot-informed)")],
        ["VR NLL arm: exchange design, matched-authorship headline (oasst1, UltraChat)", "2026-11-03, before the main run; after the pilot and the turns test",
         "revised after pilot and turns-test results, before main-run data", dlink("2026-11-03: vestigial regression: exchange design and matched-authorship headline")],
        ["VR judge arm: quantity, bootstrap over prefills, combining rule, Spearman prediction", "2026-11-02, before any judge result", "set before judge data",
         dlink("2026-11-02: judge arm specification, validation bar, combining rule, agreement prediction")],
        ["VR judge arm: minimum rate difference", "never set", "OPEN: only \"CI excludes zero\" applied", "README, Open decisions"],
        ["VR judge arm: prompt v2", "2026-11-03, before main-run generation", "set before judge data", dlink("2026-11-03: judge prompt v2")],
        ["VR judge arm: API settings (no temperature, thinking off, 600 tokens)", "2026-11-04, before the full judge run", "operational change before data; README text (temperature 0, v1) not updated",
         dlink("2026-11-04: judge request settings for Claude Sonnet 5.5")],
        ["VR judge arm: refused samples hand-labelled, not counted", "2026-11-04, during the judge run (after refusal counts, before any rate change was computed)",
         "set after some labels existed", dlink("2026-11-04: refused judge samples are labelled by hand")],
        ["VR judge validation bar (80% agreement)", "2026-11-02", "default; validation not done", "README, Judge validation"],
        ["Spearman without base points", "implementation time (analyze.py --locked)", "not pre-registered", "README, Combining the arms"],
        ["Logit distortion criteria 1–4", "2026-11-01/02 (README)", "PROVISIONAL: not frozen, not adapted to the exchange design; owner's review pending",
         dlink("2026-11-01: logit distortion design")],
        ["Logit distortion classification scheme (learned role / distortion / sharpening)", "this report", "exploratory (from the analysis request)", "—"],
        ["Overgeneralization", "Adeeb's README", "not assessed here", "measures/overgeneralization/README.md"],
    ]

    body = f"""
<main>
<p class="sub">How privileged is the assistant? Capstone, main run (exchange design) · generated by <code>reports/capstone_report.py</code> from
<code>reports/capstone_results.json</code> · 2026-11-04</p>
<h1>Capstone results: vestigial regression and logit distortion</h1>
<nav><a href="#summary">Summary</a><a href="#checks">Data &amp; validity</a><a href="#vr">Vestigial regression</a><a href="#ld">Logit distortion</a>
<a href="#design">Design sensitivity</a><a href="#prereg">Pre-registration</a><a href="#limits">Limitations</a><a href="#glossary">Glossary</a></nav>

<div class="warn"><b>Missing or incomplete (read first)</b><ul>{"".join(f"<li>{m}</li>" for m in missing)}</ul></div>

<h2 id="summary">1. Summary</h2>
<p>Each post-trained checkpoint is compared with its own family's base. Green cells fire under the pre-registered rule as written: an effect of at least 0.05 nats with the
95% CI excluding zero in both headline corpora (judge arm: a rise with the CI excluding zero, since no minimum difference is set). Effects are means with 95% bootstrap CIs.
Contamination could be checked only for OLMo; Qwen and Gemma results come with that caveat.</p>
{verdict_table}
<h3>Takeaways</h3>
<ol class="takeaways">{"".join(f"<li>{t}</li>" for t in takeaways)}</ol>

<h2 id="checks">2. Data and validity checks</h2>
<h3>What was collected</h3>
<p>Every checkpoint was scored fp32 on 900 real two-turn exchanges (300 per corpus; Gemma 4 31B the first 100 per corpus): the user turn as the conversation's opener and
the reply after its own user turn, each with chat tags (<i>tagged</i>) and with every tag removed (<i>plain</i>): 12 conditions per checkpoint. Each stage of a family shares
one inputs fingerprint. Per-text means were recomputed from the per-token tables (worst difference {tokmax:.1e} nats). The NLL-arm analysis was re-run and reproduces the
committed <code>summary.csv</code> ({rep['matched']} of {rep['rows']} rows, largest difference {rep['max_abs_diff']:.1e}).</p>
{table(["checkpoint", "conditions", "exchanges oasst1 / UltraChat / WildChat", "missing values", "replies truncated at 300 tokens", "status"], sc)}
<p>The judge arm: 200 real WildChat user turns cut <b>mid-sentence</b> (<code>data/prefill_incomplete_200.jsonl</code>), 3 samples × 120 tokens at T = 0.7 per prefill
(Gemma generated in int8). The older complete-sentence prefills (<code>data/prefill_100.jsonl</code>) were <b>not</b> judged. Every label joins to its generation, and every
generation's prefill matches the mid-sentence set.</p>
{table(["checkpoint", "labels", "prefills", "generation set", "judge / prompt", "user voice / +traits / switched / degenerate / refused"], jc)}
<h3>Judge validation</h3>
<p><b>Unvalidated.</b> No teammate has labelled the 60-sample validation sheet ({val['validation_labelled']}/{val['validation_sheet_rows']}), so there is no agreement
or Cohen's kappa against human labels, and the README's 80% bar has not been checked. A provisional check against the pilot agent's own labels (110 samples, not ground
truth) gave 90.9% binary agreement and 93.3% three-way agreement on 90 degenerate/switched rows. In 3 Gemma samples the judge labelled a thought-channel opening as user voice,
which prompt v2 counts as a switch. The judge (Claude Sonnet 5.5) is itself an assistant.</p>
<h3>Gemma 4 31B as the positive control</h3>
<p>Gemma 4 31B IT visibly fails to play the user in generation (the pilot's reason for choosing it). <b>Both arms detect it</b>: the NLL asymmetry is +5.2 / +5.3 nats/token
(CIs far from zero), and the non-user-voice rate rises {f3(jr['gemma4-31b|instruct']['exclude'], 1, 100)} points. Gemma's results come from 100 exchanges per corpus and
contamination is unchecked. Its IT model also has very high NLL on all text, so its magnitudes are not comparable to the other families'.</p>
<h3>Contamination</h3>
<p>Only OLMo 3 7B's post-training data (Dolci) is open. Every exchange was checked against it on both turns (exact match after whitespace normalisation) and the matches were
dropped. Qwen3-8B and Gemma 4 31B training data are unknown, so <b>contamination could not be checked for those families</b>.</p>

<h2 id="vr">3. Vestigial regression</h2>
<p>Does post-training degrade the model's capacity to model, and its behaviour in playing, the <i>user</i>? <b>NLL arm:</b> frame cost = NLL(tagged) − NLL(plain) of the same tokens,
change from base, and asymmetry = Δ user frame cost − Δ reply frame cost. A positive asymmetry means chat tags cost the user turn more after post-training than they cost the reply.
<b>Judge arm:</b> the share of continuations of a mid-sentence user turn that stop being the user.</p>
{figure("fig1_vr_asymmetry")}
<p><b>Interpretation:</b> only Gemma 4 31B IT meets the rule. OLMo DPO and Instruct miss it in the opposite direction, OLMo SFT and Qwen3-8B IT are within the floor.</p>
{figure("fig2_vr_framecost")}
<p><b>Interpretation:</b> OLMo's negative asymmetry comes mostly from replies getting <i>costlier</i> to predict under tags (+0.52), with user turns getting slightly cheaper (−0.07 to −0.14). Qwen's null hides
a shared rise of about +0.3 to +0.4 nats in both frame costs.</p>
<details><summary>Full NLL-arm table (nats/token unless marked)</summary>
{table(["checkpoint", "corpus", "asymmetry", "asymmetry, bits/byte", "Δ FC user", "Δ FC reply", "Δ user tagged", "Δ user plain", "Δ reply tagged", "Δ reply plain"], vr_rows)}
</details>
{figure("fig3_vr_plain")}
<p><b>Interpretation:</b> Gemma IT's plain-text NLL rises sharply (user and reply text alike), so it degrades on non-assistant context generally. That belongs to the
overgeneralization measure, not this verdict.</p>
{table(["checkpoint", "Δ user plain oasst1", "Δ user plain UltraChat", "Δ reply plain oasst1", "Δ reply plain UltraChat"], plain_rows)}
{figure("fig4_judge_shares")}
<p><b>Interpretation:</b> the bases already leave the user's voice often ({base_nu}), by answering or looping. Post-training moves OLMo
<i>towards</i> the user's voice, and moves Qwen and Gemma away from it.</p>
{figure("fig5_judge_change")}
<p><b>Interpretation:</b> the judge arm rises for Qwen3-8B IT and Gemma 4 31B IT under every treatment of refused samples. It falls for OLMo. No minimum difference is
set, so only the CI rule is applied.</p>
{table(["checkpoint", "non-user rate, base → stage (refused excl.)", "Δ, refused excluded (pts)", "Δ, refused = user", "Δ, refused = non-user", "judge arm",
        "Δ category 4 (user voice + assistant traits; to overgeneralization)"], j_rows)}
{figure("fig6_arms")}
<p><b>Interpretation:</b> the arms broadly agree in ordering (ρ = {rho['exclude']:.2f}; descriptive, 5 points). Qwen3-8B IT is the disagreement: by the README's labels it
would be <i>judge only</i>, a model that fails to play the user while still predicting user text about as well as its base.</p>
{table(["checkpoint", "NLL asymmetry (bits/byte, mean of headline)", "Δ judge rate (pts, refused excl.)", "which arm"], arm_rows)}

<h2 id="ld">4. Logit distortion <span class="tag">provisional, pending Grace's review</span></h2>
<p>Does the assistant persona's next-token distribution become distinctively overconfident? The gap = mean NLL − mean entropy of the persona's own text in its own tagged slot
(positions 1+). A rising gap means the model's confidence (low entropy) outruns its accuracy (NLL). Criterion 1 of Grace's README: present if, in both authorship columns,
Δgap(assistant) − Δgap(user) ≥ 0.05 nats with the CI excluding zero. Columns: oasst1 (humans wrote both turns) and UltraChat (a model wrote both). WildChat is shown as
secondary.</p>
{figure("fig7_ld_contrast")}
<p><b>Interpretation:</b> OLMo DPO and Instruct meet criterion 1 (+0.44 to +0.55 nats); SFT, Qwen and Gemma do not, and all three go the opposite way.</p>
{figure("fig8_ld_components_olmo_qwen")}
<p><b>Interpretation:</b> both personas sharpen (entropy falls) in every checkpoint. The OLMo DPO/Instruct contrast comes from replies' NLL rising about four times more than
user turns' NLL. The models find other authors' answers surprising, which is consistent with off-policy surprise (turns memo) as much as with a distorted assistant persona.</p>
{figure("fig9_ld_components_gemma")}
<p><b>Interpretation:</b> Gemma IT's user persona collapses (gap +9 nats), which swamps any assistant effect. Logit distortion cannot be read for a model with vestigial
regression this strong.</p>
{figure("fig10_ld_positions")}
<p><b>Interpretation:</b> where the contrast is positive (OLMo DPO, Instruct), it is front-loaded but does not vanish: the assistant gap stays above the user gap through position 100
in oasst1 and through 300 in UltraChat. Criterion 2 counts this as persisting, though weaker in the later positions.</p>
<h3>Contrast table and readings</h3>
{table(["checkpoint", "corpus", "contrast, paired CI", "contrast, independent CI", "Δ gap user", "Δ gap assistant", "ΔNLL asst − user", "Δentropy asst − user", "classification (exploratory)"], ld_rows)}
{table(["checkpoint", "alt. model column: UltraChat user vs WildChat reply", "verdict (alt. column)", "verdict (paired)", "verdict (independent)"], ld_alt)}
<h3>Per-persona classification <span class="tag">exploratory</span></h3>
<p>No interpretation table exists in Grace's README, so this scheme comes from the analysis request: gap ≥ 0.05 with CI > 0 = "gap rising (distortion)"; NLL and entropy both
falling by ≥ 0.05 with the gap flat = "learned the role"; contrast within the floor with both personas' entropy falling = "general sharpening". Each row also says which
component drives it.</p>
{table(["checkpoint", "corpus", "persona", "Δ NLL", "Δ entropy", "Δ gap", "class", "driver"], cls_rows)}
<p>Every persona in every post-trained checkpoint has a rising gap, and entropy falls everywhere. Distortion in this loose sense is general, and the criterion asks only whether
it is larger for the assistant.</p>

<h2 id="design">5. Design sensitivity <span class="tag">exploratory, separate from the main run</span></h2>
<p>From <code>docs/turns_memo.md</code> (branch <code>turns-memo-2026-11-03</code>): 150 held-out three-turn conversations, OLMo 3 7B base vs Instruct and Qwen3-8B base vs IT, fp32.
Scoring a reply <i>alone</i> (design A, the overnight run) rather than after its own question (design B, the main run) lowers the reply's frame-cost change by 0.13–0.47 nats in
all six model × corpus cells. That pushes the asymmetry towards finding regression. <b>Under A, Qwen3-8B looked mildly regressed (+0.04 to +0.07, CIs including zero); under B
it is negative.</b> OLMo is negative under both, about twice as negative under B. The team chose B as the main run's design on this evidence (decisions.md 2026-11-03).
For logit distortion the reply's gap barely depends on context, but which user turn is compared (opener or follow-up) moves the contrast by up to 0.5 nats.</p>
{table(["model", "quantity (nats/token, n = 50 per source)", "WildChat", "oasst1", "UltraChat"], turns_rows)}

<h2 id="prereg">6. Pre-registration status</h2>
{table(["criterion", "when set", "status", "record"], prereg, raw=True)}

<h2 id="limits">7. Limitations</h2>
<ul>
<li><b>Small n.</b> 300 exchanges per corpus (Gemma 100), 200 prefills for the judge; Gemma's CIs are the widest.</li>
<li><b>One checkpoint pair for Qwen and Gemma</b> (base vs instruct). Only OLMo shows the stage order, and that one family carries every claim about which stage does it.</li>
<li><b>Contamination is unknown outside OLMo.</b> Qwen and Gemma may have trained on these texts.</li>
<li><b>Base models and chat tokens.</b> Bases are scored with their family's instruct template. They are not chat-naive (tags already help Qwen base: opener frame cost −1.1 nats),
so "change from base" is measured against a partly chat-adapted reference.</li>
<li><b>The declared separator.</b> Plain text joins user turn and reply with a blank line, a convention that shifts both stages alike but is not neutral.</li>
<li><b>UltraChat's users are simulated</b> (a model wrote them), so the "model-written" column is model-on-model text, and its user turns are not real users.</li>
<li><b>OLMo's function-calling system turn.</b> OLMo's template prepends a default system turn to every tagged sequence. It cancels only to the extent stages treat it alike.</li>
<li><b>The judge is an assistant.</b> Claude Sonnet 5.5 decides what counts as "user voice". It is unvalidated against human labels and non-deterministic (API default temperature),
and it refused 73 samples.</li>
<li><b>Gemma generation in int8.</b> Gemma IT's word loops appear in every int8 Gemma IT run while int8 Gemma base has none, but a bf16 run was not done.</li>
<li><b>Off-policy text.</b> Every scored reply was written by someone else. Post-trained models find others' replies surprising, which inflates reply NLL (and the LD contrast).</li>
<li><b>OLMo NLL verdicts are not blind</b> (criteria revised after pilot results), and logit distortion's criteria are not frozen.</li>
</ul>

<h2 id="glossary">8. Glossary</h2>
<ul>
<li><b>Frame cost:</b> NLL of a text with its chat tags minus NLL of the same tokens with every tag removed, in nats per token. It is what the chat framing costs (or saves) the model on that text.</li>
<li><b>Change from base:</b> a post-trained checkpoint's value minus its own family's base model's value on the same texts, paired per text.</li>
<li><b>Asymmetry:</b> Δ frame cost of user turns minus Δ frame cost of replies. Positive means post-training made the user role relatively harder to model (vestigial regression).</li>
<li><b>NLL − entropy gap:</b> the model's actual surprise at the text minus its expected surprise (the entropy of its prediction). A rising gap means confidence outruns accuracy (logit distortion).</li>
</ul>
<p class="sub">Settings: {r['settings']['n_boot']:,} bootstrap resamples, seed {r['settings']['seed']}, floor {r['settings']['floor']} nats. Inputs: <code>results/capstone/</code>. Decisions: <a href="../{D}">docs/decisions.md</a>.</p>
</main>"""
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>Capstone results</title><style>{CSS}</style></head><body>{body}</body></html>')
    open(HTML_OUT, "w", encoding="utf-8").write(doc)


def main():
    r = json.load(open(RES))
    build_figures(r)
    build_html(r)
    print(f"wrote {HTML_OUT} ({os.path.getsize(HTML_OUT) / 1e6:.1f} MB) and {len(FIGS)} figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
