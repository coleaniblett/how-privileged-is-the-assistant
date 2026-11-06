"""Build docs/overgeneralization_results.html: a consolidated, visual summary of the overgeneralization / persona-leakage results.

Numbers come from the committed reports (parsed where they sit in tables, otherwise copied with the source file named in the comment next to them); example replies are read
from the saved generations. Re-run after new results land (leakage matrix, introspection):
    /opt/arena-env/bin/python attic/scripts/exploratory/build_overgeneralization_summary.py
Models shown are the ones on the repo's model list; Llama-3.1-8B results exist (breadth evidence) and are left out of the figures.
"""
import datetime, html, json, os, re, sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from extras.assistant_steering import requests            # noqa: E402
from extras.orthogonal_steering import is_refusal         # noqa: E402

OUT = "docs/overgeneralization_results.html"
REQ = requests()


def clip(t, n=230):
    t = " ".join(str(t).split())
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " …"


# ------------------------------------------------------------------ numbers parsed from the steering report
def parse_steering(path="results/overgeneralization/orthogonal_steering/orthogonal_steering.md"):
    s = open(path, encoding="utf-8").read(); s = s[s.index("### Refusal rates by strength"):]; s = s[:s.find("\n## ", 5) if s.find("\n## ", 5) != -1 else len(s)]; out = {}
    for line in s.splitlines():
        if not line.startswith("| ") or "---" in line or line.startswith("| model"): continue
        c = [x.strip() for x in line.strip("|").split("|")]
        if len(c) != 8: continue
        model, setup, direction, vals = c[0], c[1], c[2], c[3:]
        if direction.startswith("random"):
            out[(model, setup, "random")] = [dict(mean=float(re.match(r"([\d.]+)", v).group(1)), lo=float(re.search(r"\(([\d.]+)-", v).group(1)), hi=float(re.search(r"-([\d.]+)\)", v).group(1))) for v in vals]
        else:
            out[(model, setup, direction)] = [dict(m=float(re.match(r"([\d.]+)", v).group(1)), lo=float(re.search(r"\[([\d.]+),", v).group(1)), hi=float(re.search(r", ([\d.]+)\]", v).group(1))) for v in vals]
    return out


def parse_passfail(path="results/overgeneralization/orthogonal_steering/orthogonal_steering.md"):
    s = open(path, encoding="utf-8").read(); s = s[s.index("## Pass / fail"):s.index("## Swings")]; rows = []
    for line in s.splitlines():
        if line.startswith("| ") and "**" in line:
            c = [x.strip() for x in line.strip("|").split("|")]; rows.append(dict(model=c[0], setup=c[1], swing=c[2], band=c[3], result=c[5].strip("*")))
    return rows


STEER = parse_steering(); PASSFAIL = parse_passfail()
STEER_MODELS = [("OLMo 3 7B instruct", "OLMo 3 7B"), ("Qwen3-8B (thinking off)", "Qwen3-8B"), ("Gemma 4 E4B-it", "Gemma 4 E4B")]
steer_json = {short: {d: [dict(x) for x in STEER[(full, "char_user", d)]] for d in ("switch", "orth", "formality", "random")} for full, short in STEER_MODELS if (full, "char_user", "switch") in STEER}

# ------------------------------------------------------------------ real example replies
EX = {}
# (a) same request, two roles told "never refuses": helpful role vs pirate
g = pd.read_json("results/overgeneralization/early_probes/olmo3-7b/role_gradient/instruct.jsonl", lines=True); g["ref"] = [is_refusal(t) for t in g.text]
pick = None
for item in [13, 22, 17] + list(range(36)):
    h = g[(g.item == item) & (g.setup == "sys") & (g.kind == "helpful")]; s_ = g[(g.item == item) & (g.setup == "sys") & (g.kind == "stylized")]
    if len(h) and len(s_) and h.ref.mean() >= 0.99 and s_.ref.mean() <= 0.01: pick = (item, h.iloc[0], s_.iloc[0]); break
if pick:
    EX["roles"] = dict(request=REQ[pick[0]], helpful=dict(role=pick[1].role, text=clip(pick[1].text)), stylized=dict(role=pick[2].role, text=clip(pick[2].text)))
# (b) steering at -2 vs +2 on Qwen, same character and request
try:
    lo = pd.read_json("results/overgeneralization/orthogonal_steering/gens/qwen/switch_s-1_a-2.jsonl", lines=True); hi = pd.read_json("results/overgeneralization/orthogonal_steering/gens/qwen/switch_s-1_a+2.jsonl", lines=True)
    for df in (lo, hi): df["ref"] = [is_refusal(t) for t in df.text]
    m = lo[lo.cond == "char_sys"].merge(hi[hi.cond == "char_sys"], on=["cond", "role", "item", "sample"], suffixes=("_lo", "_hi"))
    m = m[(~m.ref_lo) & m.ref_hi & (m.item == 8)]
    r = m[m.role == "Osric"].iloc[0] if (m.role == "Osric").any() else m.iloc[0]
    EX["steer"] = dict(request=REQ[int(r["item"])], role=r.role, lo=clip(r.text_lo, 200), hi=clip(r.text_hi, 200))
except Exception as e:  # keep the page buildable if generations are not on this machine
    print("steering example skipped:", e)
# (c) one pilot carryover conversation
d = [json.loads(l) for l in open("data/persona_carryover.jsonl", encoding="utf-8")]
row = next(x for x in d if x["domain"] == "coin" and x["believed"] == "heads")
EX["carry"] = dict(assistant=row["turns"]["assistant>new"], character=row["turns"]["character>new"])
# (d) bare-ish persona in the system prompt vs the user turn, from the leakage-matrix generations if they exist
p = "results/overgeneralization/dominance_matrix/gens/olmo/instruct.jsonl"
if os.path.exists(p):
    from extras import leakage_matrix as lm
    gm = pd.read_json(p, lines=True, dtype=False); specs = {(s["ctx"], s["tmpl"], s["item"], s["role"]): s for s in lm.register_specs() if s["ctx"] == "named"}
    cand = gm[(gm.trait == "register") & (gm.ctx == "named") & (gm.item == 0)]
    for role in ["a human long-haul truck driver", "a human accountant", "a human welder", "a human plumber"]:
        a_ = cand[(cand.role == role) & (cand.tmpl == 1)]; b_ = cand[(cand.role == role) & (cand.tmpl == 2)]
        if len(a_) and len(b_):
            sa = specs[("named", 1, 0, role)]["msgs"]; sb = specs[("named", 2, 0, role)]["msgs"]
            EX["thin"] = dict(persona=sa[0]["content"], question=sa[1]["content"], sys=clip(a_.iloc[0].text, 260), user=clip(b_.iloc[0].text, 260)); break

DATA = dict(steer=steer_json, passfail=PASSFAIL, ex=EX)
built = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
page = open(os.path.join(os.path.dirname(__file__), "overgeneralization_summary_template.html"), encoding="utf-8").read()
page = page.replace("/*DATA*/null", json.dumps(DATA, ensure_ascii=False).replace("</", "<\\/")).replace("{{BUILT}}", built)
open(OUT, "w", encoding="utf-8").write(page); print("wrote", OUT, "| examples:", sorted(EX), "| models in steering chart:", list(steer_json))
