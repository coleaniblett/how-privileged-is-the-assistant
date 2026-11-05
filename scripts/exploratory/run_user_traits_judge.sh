#!/usr/bin/env bash
# User-slot judge arm (spec: measures/overgeneralization/README.md, deviations in docs/decisions.md 2026-11-04). One command, no GPU, no API calls.
# Reads Cole's judge labels read-only from origin/capstone-scoring-2026-11-03 (git show; his branch and worktree are untouched), then runs
# user_traits judge for olmo3-7b (decides) and qwen3-8b, gemma4-31b (supporting, contamination could not be checked).
# Once hand labels exist:  HAND_LABELS=<dir containing <family>/<stage>.jsonl> bash scripts/exploratory/run_user_traits_judge.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
BR=origin/capstone-scoring-2026-11-03
ROOT="${JUDGE_ROOT:-/tmp/claude-0/judge_incomplete}"
git fetch -q origin
for fam in olmo3-7b qwen3-8b gemma4-31b; do
  mkdir -p "$ROOT/$fam"
  for f in $(git ls-tree -r --name-only "$BR" | grep "results/capstone/$fam/vestigial_regression/judge_incomplete/"); do
    git show "$BR:$f" > "$ROOT/$fam/$(basename "$f")"
  done
  extra=()
  [[ -n "${HAND_LABELS:-}" ]] && extra=(--hand-labels "$HAND_LABELS/$fam")
  /opt/arena-env/bin/python -m measures.overgeneralization.user_traits judge --family "$fam" --judge-dir "$ROOT/$fam" "${extra[@]}"
done
