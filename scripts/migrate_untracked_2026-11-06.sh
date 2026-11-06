#!/usr/bin/env bash
# Move untracked files (activation fits, raw generations, locks, logs) left at pre-2026-11-06 paths to the reorganised layout
# (docs/moved_paths_2026-11-06.md). Run once from the repo root of a checkout that has pulled the reorganisation, with no job running.
# Only untracked or ignored files move; nothing is overwritten. results/<family>/ run directories of the legacy pipeline
# (core/, vestigial_regression/, ...) are left alone: only the early-probe entries move out of results/olmo3-7b/.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
moved=0
move() {  # OLD NEW: every untracked or ignored file under OLD goes to the same relative path under NEW
  local old="$1" new="$2" f dest
  [ -e "$old" ] || return 0
  while IFS= read -r -d "" f; do
    dest="$new${f#"$old"}"
    if [ -e "$dest" ]; then echo "kept, destination exists: $f"; continue; fi
    mkdir -p "$(dirname "$dest")" && mv -n "$f" "$dest" && echo "moved: $f -> $dest" && moved=$((moved + 1))
  done < <({ git ls-files -z --others --exclude-standard -- "$old"; git ls-files -z --others --ignored --exclude-standard -- "$old"; } | sort -zu)
  [ -d "$old" ] && find "$old" -depth -type d -empty -delete 2>/dev/null
  return 0
}
move results/scores results/precapstone/scores
move results/generations results/precapstone/generations
move results/role_swap results/precapstone/role_swap
move results/coin_toss results/precapstone/coin_toss
move results/selection_check results/precapstone/selection_check
move results/results.md results/precapstone/results.md
move results/summary.csv results/precapstone/summary.csv
move results/position.md results/precapstone/position.md
move results/role_swap.md results/precapstone/role_swap.md
move results/role_swap_summary.csv results/precapstone/role_swap_summary.csv
move results/coin_toss.md results/precapstone/coin_toss.md
move results/coin_toss_summary.csv results/precapstone/coin_toss_summary.csv
move results/nll_oasst1_user.png results/precapstone/nll_oasst1_user.png
move results/nll_olmo-instruct_assistant.png results/precapstone/nll_olmo-instruct_assistant.png
move results/nll_wildchat_assistant.png results/precapstone/nll_wildchat_assistant.png
move results/nll_wildchat_user.png results/precapstone/nll_wildchat_user.png
move results/position_oasst1_user.png results/precapstone/position_oasst1_user.png
move results/position_olmo-instruct_assistant.png results/precapstone/position_olmo-instruct_assistant.png
move results/position_wildchat_assistant.png results/precapstone/position_wildchat_assistant.png
move results/position_wildchat_user.png results/precapstone/position_wildchat_user.png
move results/olmo3-7b/overgeneralization results/overgeneralization/persona_carryover/olmo3-7b
move results/qwen3-8b/overgeneralization results/overgeneralization/persona_carryover/qwen3-8b
move results/gemma4-31b/overgeneralization results/overgeneralization/persona_carryover/gemma4-31b
move results/llama3.1-8b/overgeneralization results/overgeneralization/persona_carryover/llama3.1-8b
move results/apertus-8b/overgeneralization results/overgeneralization/persona_carryover/apertus-8b
for e in results/olmo3-7b/*; do [ -e "$e" ] || continue; b=$(basename "$e"); [[ $b =~ ^(ai_personas_axis|assistant_axis|assistant_steering|axis_|carryover_|decision_point|dominance_pilot|false_fact|formality_|persona_|plain_axis|role_|stakeholder_swap) ]] && move "$e" "results/overgeneralization/early_probes/olmo3-7b/$b"; done
move results/gemma-4-e4b results/overgeneralization/early_probes/gemma-4-e4b
move results/other results/overgeneralization/early_probes/other
for e in results/llama3.1-8b/*; do [ -e "$e" ] || continue; b=$(basename "$e"); [[ $b =~ ^introspection_personas ]] && move "$e" "results/overgeneralization/early_probes/llama3.1-8b/$b"; done
for e in results/qwen3-8b/*; do [ -e "$e" ] || continue; b=$(basename "$e"); [[ $b =~ ^introspection_personas ]] && move "$e" "results/overgeneralization/early_probes/qwen3-8b/$b"; done
move results/activation_patching results/overgeneralization/activation_patching
move results/activation_patching_j4 results/overgeneralization/activation_patching_j4
move results/belief_induction results/overgeneralization/belief_induction
move results/bias_leak results/overgeneralization/bias_leak
move results/dolci_exposure results/overgeneralization/dolci_exposure
move results/dominance_matrix results/overgeneralization/dominance_matrix
move results/icl_era results/overgeneralization/icl_era
move results/implant_spread results/overgeneralization/implant_spread
move results/mechanism_checks results/overgeneralization/mechanism_checks
move results/orthogonal_steering results/overgeneralization/orthogonal_steering
move results/overnight_2026-11-05 results/overgeneralization/overnight_2026-11-05
move results/persona_map results/overgeneralization/persona_map
move results/persona_suppression results/overgeneralization/persona_suppression
move results/refusal_direction results/overgeneralization/refusal_direction
move results/role_says results/overgeneralization/role_says
move results/second_judge results/overgeneralization/second_judge
move results/skill_roleplay results/overgeneralization/skill_roleplay
move results/style_fall results/overgeneralization/style_fall
move results/training_style results/overgeneralization/training_style
move results/training_style_l4 results/overgeneralization/training_style_l4
move results/trait_atlas results/overgeneralization/trait_atlas
move results/trait_atlas_breadth results/overgeneralization/trait_atlas_breadth
move results/trait_atlas_incharacter results/overgeneralization/trait_atlas_incharacter
move results/trait_atlas_more_base results/overgeneralization/trait_atlas_more_base
move results/trait_atlas_strict results/overgeneralization/trait_atlas_strict
move results/trait_atlas_unmixed results/overgeneralization/trait_atlas_unmixed
move results/user_style results/overgeneralization/user_style
move results/working_model_L results/overgeneralization/working_model_L
move results/user_turn_2026-11-06 results/capstone/overgeneralization/user_turn_2026-11-06
move results/new_families_j2.md results/overgeneralization/new_families_j2.md
echo "$moved untracked files moved"
