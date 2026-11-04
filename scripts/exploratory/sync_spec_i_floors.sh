#!/usr/bin/env bash
# Marble side of spec I: when vinyl has written a family's floors.json, copy it (and the Stage 0/1 summaries) here, commit and push it, then write floors.committed on vinyl
# (sha256 + commit) so Stage 2 can start. Loops until both families are done or 12 h pass. Run from the main worktree.
set -uo pipefail
cd "${SYNC_WT:?set SYNC_WT to a separate git worktree}"
V=arena9-vinyl; R=/root/hpa-atlas2
D=results/activation_patching
end=$(( $(date +%s) + 43200 ))
declare -A done_
while [ $(date +%s) -lt $end ] && [ ${#done_[@]} -lt 2 ]; do
  for fam in olmo3-7b qwen3-8b; do
    [ -n "${done_[$fam]:-}" ] && continue
    ssh $V "test -f $R/$D/$fam/floors.json && test ! -f $R/$D/$fam/floors.committed" 2>/dev/null || continue
    mkdir -p $D/$fam
    scp -q $V:$R/$D/$fam/floors.json $V:$R/$D/$fam/stage1.json $V:$R/$D/$fam/stage0.json $D/$fam/ 2>/dev/null
    for f in screen.csv.gz; do scp -q $V:$R/$D/$fam/$f $D/$fam/ 2>/dev/null; done
    /opt/arena-env/bin/python - <<P
import json
s=json.load(open("$D/$fam/stage1.json")); print("$fam stage1:", {k: s.get(k) for k in ("chosen","best","screen_minutes","stage2_gens","stage2_projected_minutes","pilot_seconds")})
P
    sha=$(sha256sum $D/$fam/floors.json | cut -d' ' -f1)
    git fetch -q origin; git checkout -q --detach origin/main
    git add $D/$fam/floors.json $D/$fam/stage1.json $D/$fam/stage0.json $D/$fam/screen.csv.gz
    git commit -qm "results: spec I $fam Stage 0/1 and floors (committed before Stage 2)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" && git push -q origin HEAD:main || { echo "commit/push failed for $fam"; continue; }
    commit=$(git rev-parse --short HEAD)
    ssh $V "echo '{\"sha256\": \"$sha\", \"commit\": \"$commit\"}' > $R/$D/$fam/floors.committed"
    done_[$fam]=1; echo "$fam floors committed $commit $(date -u +%T)"
  done
  sleep 45
done
