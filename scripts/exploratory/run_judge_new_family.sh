#!/usr/bin/env bash
# Cole's judge, unchanged (prompt v2), on a new family's prefill generations (spec J rule 1; labels blind; plan docs/decisions.md 2026-11-05).
# Usage: run_judge_new_family.sh FAMILY SET [LIMIT]   SET = exchanges (400 prefills, generations_incomplete_exchanges) | p200 (generations_incomplete)
# With LIMIT: a test on the first LIMIT samples of the instruct file only (cache shared with the full run). Else one process per stage file.
# The key is read here only.
cd /root/hpa-main-wt || exit 1
export ANTHROPIC_API_KEY="$(cat /root/.private/judge_api_key)"
PY=/opt/arena-env/bin/python
fam=$1; set=$2; lim=${3:-}
[ "$set" = exchanges ] && src=generations_incomplete_exchanges pre= || { src=generations_incomplete; pre=p200_; }
d=results/capstone/$fam/overgeneralization/judge_incomplete_exchanges
mkdir -p $d
for f in results/capstone/$fam/core/$src/*.jsonl; do
  st=$(basename $f .jsonl)
  if [ -n "$lim" ]; then
    [ "$st" = instruct ] || continue
    $PY -m measures.vestigial_regression.judge run --samples $f --out $d/judged_test_$pre$st.jsonl --cache $d/judge_cache_$pre$st.jsonl --limit "$lim" > $d/run_test_$pre$st.log 2>&1
  else
    $PY -m measures.vestigial_regression.judge run --samples $f --out $d/judged_$pre$st.jsonl --cache $d/judge_cache_$pre$st.jsonl > $d/run_$pre$st.log 2>&1 &
  fi
done
wait
echo "=== judge $fam $set finished $(date -u +%H:%M)"
