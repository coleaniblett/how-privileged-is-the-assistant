#!/usr/bin/env bash
# Spec Q: when vinyl's judging is done (results/role_says/Q_JUDGED_DONE), fetch the judged files, run the analysis, add a decisions entry and push, without a prompt. One push only: flock + a PUSHED marker.
# Log /tmp/claude-0/q_push.log. Safe to run twice (the second run exits at the marker).
set -uo pipefail
cd /root/hpa-main-wt
export OMP_NUM_THREADS=1
PY=/opt/arena-env/bin/python
exec 7>/tmp/claude-0/q_push.lock
flock -n 7 || { echo "another q_push is running"; exit 0; }
[ -f results/role_says/PUSHED ] && { echo "already pushed"; exit 0; }
until ssh -o ConnectTimeout=20 arena9-vinyl 'test -f /root/hpa-atlas2/results/role_says/Q_JUDGED_DONE' 2>/dev/null; do sleep 60; done
echo "judging done $(date -u +%T)"
nice -n 15 rsync -a -e ssh --include='*/' --include='*_judged.csv.gz' --exclude='*' arena9-vinyl:/root/hpa-atlas2/results/role_says/ results/role_says/ || { echo "rsync failed"; exit 1; }
nice -n 15 $PY -m extras.role_says analyze > /tmp/claude-0/q_analyze.log 2>&1 || { echo "analyze failed"; tail -5 /tmp/claude-0/q_analyze.log; exit 2; }
$PY - <<'PYE'
import re
md=open("results/role_says/role_says.md",encoding="utf-8").read()
m=re.search(r"## Overall reading by family and position \(final\)\n\n(.*?)\n\n", md, re.S)
tab=m.group(1) if m else "(see report)"
d=open("docs/decisions.md").read()
if not d.endswith("\n"): d+="\n"
d+="""
## 2026-11-05 (UTC) - Result: spec Q, what the role says (identity x voice; OLMo, Qwen, Gemma 31B final; NOT BLIND, exploratory, outside every verdict)
`results/role_says/role_says.md` (+ CSVs, 30 saved replies per condition; full replies stay local). Gemma 31B is final only (Gemma base was dropped because Q projected over 4 GPU hours, see the timing entry). Overall reading by family and position (pre-set rules; the report has readings 1 to 4, in-character shares and the descriptive regex table; judged traits are provisional):

""" + tab + """

Limits as in the plan: few jobs; many cells; provisional local judge (hand check not done); 'an AI that does this job' is odd for some jobs; following identity could be the model inferring what an AI should say; truth traits could be general skill; regex sees form only; not hand-checked.
"""
open("docs/decisions.md","w").write(d)
PYE
nice -n 15 git add results/role_says/role_says.md results/role_says/role_says_replies_sample.csv.gz docs/decisions.md
git commit -q -m "results: spec Q what the role says (identity x voice)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
git pull -q --rebase && git push -q origin HEAD:main && { touch results/role_says/PUSHED; echo "pushed $(git log --oneline | head -1)"; }
echo "=== done $(date -u +%T)"
