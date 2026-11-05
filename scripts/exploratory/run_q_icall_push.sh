#!/usr/bin/env bash
# Spec Q follow-up on marble: when vinyl touches Q_ICALL_DONE, fetch the three *_icall files, rerun the analysis (main readings must stay identical, sensitivity section appended), log and push once. flock + results/role_says/ICALL_PUSHED marker.
set -uo pipefail
cd /root/hpa-main-wt
export OMP_NUM_THREADS=1
PY=/opt/arena-env/bin/python
exec 7>/tmp/claude-0/q_icall_push.lock
flock -n 7 || { echo "another running"; exit 0; }
[ -f results/role_says/ICALL_PUSHED ] && { echo "already pushed"; exit 0; }
until ssh -o ConnectTimeout=20 arena9-vinyl 'test -f /root/hpa-atlas2/results/role_says/Q_ICALL_DONE' 2>/dev/null; do sleep 60; done
echo "icall done $(date -u +%T)"
nice -n 15 rsync -a -e ssh --include='*/' --include='*_icall.csv.gz' --exclude='*' arena9-vinyl:/root/hpa-atlas2/results/role_says/ results/role_says/ || { echo "rsync failed"; exit 1; }
n=$(ls results/role_says/*/instruct_icall.csv.gz 2>/dev/null | wc -l); [ "$n" -eq 3 ] || { echo "only $n icall files"; exit 3; }
cp results/role_says/role_says.md /tmp/claude-0/rs_before_icall.md
nice -n 15 $PY -m extras.role_says analyze > /tmp/claude-0/q_icall_analyze.log 2>&1 || { echo "analyze failed"; tail -5 /tmp/claude-0/q_icall_analyze.log; exit 2; }
git checkout results/role_says/role_says_replies_sample.csv.gz
# main readings must be unchanged: everything before the sensitivity heading identical to the version before
$PY - <<'PYE' || { echo "MAIN READINGS CHANGED: not pushing"; exit 5; }
a=open("/tmp/claude-0/rs_before_icall.md",encoding="utf-8").read().rstrip("\n")
b=open("results/role_says/role_says.md",encoding="utf-8").read()
i=b.index("\n## SENSITIVITY")
assert b[:i].rstrip("\n")==a, "main part differs"
import re
sec=b[i:]
m=re.search(r"Overall reading, in-character replies only:\n\n(.*?)\n\n", sec, re.S)
d=open("docs/decisions.md").read()
if not d.endswith("\n"): d+="\n"
d+="""
## 2026-11-06 (UTC) - Result: spec Q follow-up, readings on in-character replies only (sensitivity; exploratory, NOT BLIND)
In-character judge run on ALL final role replies (3 families). `results/role_says/role_says.md` now ends with a SENSITIVITY section (`role_says_sensitivity.csv`); the main readings above it are byte-identical to the earlier push. Overall reading on in-character replies only, per pre-set bars; a reading is NOT READ where a compared cell has fewer than 20 in-character replies in the trait's own set. Provisional like the judge.

""" + (m.group(1) if m else "(see report)") + "\n"
open("docs/decisions.md","w").write(d)
PYE
echo "$(date -u +%H:%M) UTC: Q sensitivity (readings on in-character replies only) done on all final replies; pushing." >> /root/overnight_log.md
git add results/role_says/role_says.md results/role_says/role_says_sensitivity.csv docs/decisions.md extras/role_says.py
git commit -q -m "results: spec Q sensitivity, readings on in-character replies only (judge on all final replies); main readings unchanged

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" || { echo "commit failed"; exit 4; }
git fetch -q && git pull -q --rebase --autostash && nice -n 15 git push -q origin HEAD:main && touch results/role_says/ICALL_PUSHED && echo "pushed $(git rev-parse --short HEAD)"
