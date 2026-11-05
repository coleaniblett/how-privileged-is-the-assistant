#!/usr/bin/env bash
# Judge (unchanged: measures.vestigial_regression.judge, prompt v2) on the exchange-set prefill generations. Plan: docs/decisions.md 2026-11-05.
# One process per (family, stage) file, each with its own cache and output under results/capstone/<family>/overgeneralization/judge_incomplete_exchanges/
# (about 0.5 requests/s per process, so parallel). Re-running resumes from the caches. The key is read here and only here.
cd /root/hpa-main-wt || exit 1
export ANTHROPIC_API_KEY="$(cat /root/.private/judge_api_key)"
PY=/opt/arena-env/bin/python
for fam in olmo3-7b qwen3-8b; do
  d=results/capstone/$fam/overgeneralization/judge_incomplete_exchanges
  mkdir -p $d
  for f in results/capstone/$fam/core/generations_incomplete_exchanges/*.jsonl; do
    st=$(basename $f .jsonl)
    $PY -m measures.vestigial_regression.judge run --samples $f --out $d/judged_$st.jsonl --cache $d/judge_cache_$st.jsonl > $d/run_$st.log 2>&1 &
  done
done
wait
echo "=== judge launcher finished $(date -u +%H:%M)"
