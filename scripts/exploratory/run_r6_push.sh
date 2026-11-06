#!/usr/bin/env bash
# R6 on marble: wait for vinyl's R6_JUDGED_DONE, fetch judged files, analyze (appends the BREADTH section to results/overgeneralization/role_says/role_says.md), push once.
set -uo pipefail
cd /root/hpa-main-wt; export OMP_NUM_THREADS=2; PY=/opt/arena-env/bin/python; source scripts/exploratory/lib_r.sh
exec 7>/tmp/claude-0/r6_push.lock; flock -n 7 || exit 0
O=results/overgeneralization/overnight_2026-11-05/R6; mkdir -p $O; [ -f $O/PUSHED ] && exit 0
until ssh -o ConnectTimeout=20 arena9-vinyl "test -f /root/hpa-atlas2/$O/R6_JUDGED_DONE" 2>/dev/null; do sleep 60; done
nice -n 15 rsync -a -e ssh --include='*/' --include='*_judged.csv.gz' --exclude='*' arena9-vinyl:/root/hpa-atlas2/$O/ $O/ || exit 1
cp results/overgeneralization/role_says/role_says.md /tmp/claude-0/rs_before_r6.md
nice -n 15 $PY -m extras.role_says_breadth analyze > /tmp/claude-0/r6_analyze.log 2>&1 || { tail -5 /tmp/claude-0/r6_analyze.log; exit 2; }
$PY - <<'PYE' || { echo "MAIN READINGS CHANGED"; exit 5; }
a=open("/tmp/claude-0/rs_before_r6.md",encoding="utf-8").read().rstrip("\n")
b=open("results/overgeneralization/role_says/role_says.md",encoding="utf-8").read()
i=b.index("\n## BREADTH")
assert b[:i].rstrip("\n")==a.split("\n## BREADTH")[0].rstrip("\n")
import re
m=re.search(r"### Overall reading by family and position \(final\), breadth families\n\n(.*?)\n\n", b[i:], re.S)
d=open("docs/decisions.md").read()
if not d.endswith("\n"): d+="\n"
d+="\n## 2026-11-06 (UTC) - Result: spec R6, spec Q on Llama 3.1 8B and Apertus 8B (final only; exploratory, NOT BLIND)\nBreadth section appended to `results/overgeneralization/role_says/role_says.md` (three-family readings unchanged; CSV and 30-reply sample under `results/overgeneralization/overnight_2026-11-05/R6/`). Overall readings:\n\n"+(m.group(1) if m else "(see report)")+"\n"
open("docs/decisions.md","w").write(d)
PYE
git checkout results/overgeneralization/role_says/role_says_replies_sample.csv.gz 2>/dev/null
push_paths "results: spec R6 spec Q on Llama and Apertus (breadth section; three-family readings unchanged)" results/overgeneralization/role_says/role_says.md $O/role_says.md $O/role_says_replies_sample.csv.gz extras/role_says_breadth.py docs/decisions.md scripts/exploratory && touch $O/PUSHED
