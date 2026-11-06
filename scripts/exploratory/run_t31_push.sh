#!/usr/bin/env bash
# T3.1 on marble: when vinyl has judged Gemma base, fetch it, rerun spec Q's analysis (the Gemma training-check rows fill in; sensitivity and breadth sections kept), push once.
set -uo pipefail
cd /root/hpa-main-wt; export OMP_NUM_THREADS=2; PY=/opt/arena-env/bin/python; source scripts/exploratory/lib_r.sh
exec 7>/tmp/claude-0/t31_push.lock; flock -n 7 || exit 0
[ -f results/role_says/T31_PUSHED ] && exit 0
until ssh -o ConnectTimeout=20 arena9-vinyl 'test -f /root/hpa-atlas2/results/role_says/T31_JUDGED_DONE' 2>/dev/null; do sleep 60; done
nice -n 15 rsync -a -e ssh arena9-vinyl:/root/hpa-atlas2/results/role_says/gemma4-31b/base_judged.csv.gz results/role_says/gemma4-31b/ || exit 1
cp results/role_says/role_says.md /tmp/claude-0/rs_before_t31.md
nice -n 15 $PY -m extras.role_says analyze > /tmp/claude-0/t31_analyze.log 2>&1 || { tail -n 5 /tmp/claude-0/t31_analyze.log; exit 2; }
# analyze rewrites the file without the breadth section: re-append it
nice -n 15 $PY -m extras.role_says_breadth analyze >> /tmp/claude-0/t31_analyze.log 2>&1 || exit 3
git checkout results/role_says/role_says_replies_sample.csv.gz 2>/dev/null
$PY - <<'PYE'
import difflib
a=open("/tmp/claude-0/rs_before_t31.md",encoding="utf-8").read().splitlines(); b=open("results/role_says/role_says.md",encoding="utf-8").read().splitlines()
ch=[l for l in difflib.unified_diff(a,b,lineterm="",n=0) if l[:1] in "+-" and l[:3] not in ("+++","---")]
d=open("docs/decisions.md").read()
if not d.endswith("\n"): d+="\n"
d+="\n## 2026-11-06 (UTC) - Result: spec Q Gemma 4 31B base added (T3.1, the training check dropped earlier)\nGemma base replies generated and judged like the other checkpoints; `results/role_says/role_says.md` was regenerated: the readings on final are unchanged by construction; the Gemma rows changed are the in-character shares (base column) and the training row. %d lines differ from the earlier file (sampled replies file not committed again).\n" % len(ch)
open("docs/decisions.md","w").write(d)
PYE
echo "$(date -u +%H:%M) T3.1 Gemma base added to Q report; pushing" >> /root/overnight_log.md
push_paths "results: spec Q Gemma base added (training check; T3.1)" results/role_says/role_says.md docs/decisions.md scripts/exploratory && touch results/role_says/T31_PUSHED
