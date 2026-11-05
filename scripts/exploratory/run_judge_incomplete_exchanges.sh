#!/usr/bin/env bash
# Judge (unchanged: measures.vestigial_regression.judge, prompt v2) on the exchange-set prefill generations. Plan: docs/decisions.md 2026-11-05.
# Usage: run_judge_incomplete_exchanges.sh [LIMIT]   (LIMIT = test run on the first LIMIT samples of olmo3-7b only). The key is read here and only here.
cd /root/hpa-main-wt || exit 1
export ANTHROPIC_API_KEY="$(cat /root/.private/judge_api_key)"
PY=/opt/arena-env/bin/python
for fam in olmo3-7b qwen3-8b; do
  d=results/capstone/$fam/overgeneralization/judge_incomplete_exchanges
  mkdir -p $d
  if [ -n "$1" ]; then
    $PY -m measures.vestigial_regression.judge run --samples results/capstone/$fam/core/generations_incomplete_exchanges/*.jsonl --out $d/judged_test.jsonl --cache $d/judge_cache.jsonl --limit "$1"
    break
  fi
  $PY -m measures.vestigial_regression.judge run --samples results/capstone/$fam/core/generations_incomplete_exchanges/*.jsonl --out $d/judged.jsonl --cache $d/judge_cache.jsonl
done
echo "=== judge launcher finished $(date -u +%H:%M)"
