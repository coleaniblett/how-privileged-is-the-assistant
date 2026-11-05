"""Builds results/overnight_2026-11-05/MORNING.md from whatever has landed (spec R hand-off). Run it again whenever something lands; it only reads results files and PUSHED markers.
    python scripts/exploratory/make_morning.py
"""
import os

import pandas as pd

R = "results/overnight_2026-11-05"


def have(p):
    return os.path.exists(p)


def csv(p):
    try:
        return pd.read_csv(p) if have(p) else None
    except Exception:
        return None


def main():
    rows = []
    # R0 / Q sensitivity
    S = csv("results/role_says/role_says_sensitivity.csv")
    if S is not None:
        v = S[S.reading == "R0"]
        key = "; ".join(f"{r.family} {r.position}: {r.diff:+.0f} [{r.lo:+.0f}, {r.hi:+.0f}] {r.cls}" for r in v.itertuples() if r.diff == r.diff)
        rows.append(("R0 Q on in-character replies only", "done", f"voice effects on `denies_feelings` (AI role, none minus given): {key or 'cells not read'}", "Slide E/Q-type claims about what the role says: states whether the voice effect survives the in-character filter. Proposal only.", "results/role_says/role_says.md (SENSITIVITY)"))
    else:
        rows.append(("R0 Q on in-character replies only", "running on vinyl", "-", "-", "-"))
    # R1
    d = csv(f"{R}/R1/R1_readings.csv")
    if d is not None:
        c = d.reading.str.split(" \\(").str[0].value_counts().to_dict()
        g = d[(d.family == "gemma4-31b")]
        rows.append(("R1 strict layout", "done", f"readings: {c}. Gemma light-role strict levels vs old: " + "; ".join(f"{r.source}/{r.context} old {r.final_old:.0f} strict {r.final_strict:.0f} heavy {r.final_heavy:.0f}" for r in g.itertuples()), "Slide C: the formatting claim holds for OLMo/Qwen/Apertus; for Gemma light roles it is mostly bold and short lists. Proposal only.", f"{R}/R1/R1.md"))
    # R2
    d = csv(f"{R}/R2/R2_readings.csv")
    if d is not None:
        rows.append(("R2 did the reply help", "done", "; ".join(f"{r.family}/{r.wording}: judge gap {r.judge_gap_noline:+.0f} (regex {r.regex_gap_noline:+.0f}) {r.reading_style_fall.split('(')[0].strip()}" for r in d.itertuples()), "Slide D: refusal fall by style (judge vs regex). Proposal only.", f"{R}/R2/R2.md"))
    else:
        rows.append(("R2 did the reply help", "running on vinyl", "-", "-", "-"))
    # R3
    d = csv(f"{R}/R3/R3_readings.csv")
    t = csv(f"{R}/R3/R3_incharacter.csv")
    if d is not None:
        tk = "; ".join(f"{r.family}/{r.wording}: {r.in_character:.0f}%" for r in t.itertuples()) if t is not None else "in-character not yet judged"
        rows.append(("R3 prefill belief robustness", "done" if t is not None else "partial", f"assistant>new final by wording: " + "; ".join(f"{r.family}/{r.wording} {r.a_new_final:+.2f}" for r in d.itertuples() if r.wording != "prompt") + f". In character: {tk}", "Slide A closing line. Proposal only.", f"{R}/R3/R3.md"))
    else:
        rows.append(("R3 prefill belief robustness", "running", "-", "-", "-"))
    d = csv(f"{R}/R4/R4_table.csv")
    rows.append(("R4 OLMo by stage", "done" if d is not None else "queued/running", "see table" if d is not None else "-", "A new 'what post-training does' slide. Proposal only.", f"{R}/R4/R4.md"))
    d = csv(f"{R}/R5/R5_readings.csv")
    if d is not None:
        rows.append(("R5 voice on fresh questions", "done", "; ".join(f"{r.family}/{r.position}: strict {r.strict_diff:+.0f}, stock {r.stock_diff:+.0f}: {r.claim.split('(')[0].strip()}" for r in d.itertuples()), "Q's voice claim on fresh questions. Proposal only.", f"{R}/R5/R5.md"))
    else:
        rows.append(("R5 voice on fresh questions", "queued/running", "-", "-", "-"))
    rows.append(("R6 Q on Llama and Apertus", "done" if have(f"{R}/R6/PUSHED") else "queued/running", "BREADTH section of results/role_says/role_says.md" if have(f"{R}/R6/PUSHED") else "-", "Q breadth. Proposal only.", "results/role_says/role_says.md"))
    d = csv(f"{R}/R7/R7_readings.csv")
    if d is not None:
        f = d[d.stage == "instruct"]
        rows.append(("R7 knows versus says", "done", "; ".join(f"{r.family}/{r.test[:1]}: forced {r.forced_full:.0f} vs free {r.free_full:.0f}: {r.reading}" for r in f.itertuples()), "Slide F (Gemma characters stop answering). Proposal only.", f"{R}/R7/R7.md"))
    else:
        rows.append(("R7 knows versus says", "queued/running", "-", "-", "-"))
    L = ["# Morning hand-off, spec R overnight (night of 2026-11-05)", "",
         "All of this is **exploratory** and **provisional**, outside every verdict, and not blind. Judged measures use the local Gemma 4 31B-it judge (not hand-checked, and Gemma judges its own replies). Nothing was sent outside the repo. Slide changes are proposals only; no slide file was edited.", "",
         "| experiment | state | reading and key numbers | which slide it could change and how (proposal) | report |", "|---|---|---|---|---|"]
    for r in rows:
        L.append("| " + " | ".join(str(x) for x in r) + " |")
    fl = "/tmp/claude-0/r_failures.log"
    L += ["", "## Failures, retries and changes from the spec", ""]
    for p in (fl, "/root/hpa-main-wt/queue_marble.txt.failures"):
        if have(p):
            L += [f"- `{os.path.basename(p)}`:"] + ["  - " + l.strip() for l in open(p).read().splitlines()[-10:]]
    L += ["- See the 'Changes from the spec' lines in each report and in `docs/decisions.md` (2026-11-06): R1 reading interpretation; R2 validation set (the 56-reply hand check file was not in the repo; the 300 R/C human labels of the orthogonal-steering detector were used); R2 L3 rows are the atlas's benign questions; R3 reuses spec O's wording-1 files; Gemma wording-2 base last; R5 uses all 90 L4 questions for OLMo too; R7 forced-answer design.", ""]
    os.makedirs(R, exist_ok=True)
    open(f"{R}/MORNING.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
