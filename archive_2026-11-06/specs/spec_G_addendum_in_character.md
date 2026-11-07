# Spec G addendum: "Speaks as the character?" check

Approved by Adeeb 2026-11-04 ~7:15 PM. Laptop Claude design; arena9-vinyl runs, marble drives. Exploratory and ADDED AFTER SEEING THE PROVISIONAL ATLAS: label it post-hoc everywhere. Outside every verdict. Do not overwrite existing atlas files. Plan committed and logged in `docs/decisions.md` before any scoring.

## Why
1. Light roles were never checked for whether they actually play the role, so "spreads to light roles" may just be the Assistant with a label on it.
2. In OLMo and Qwen, strong + instruction replies spoken as the character go 19→90% and 30→99% with training. Comparing only in-character replies, most falls vanish. So the style fall may largely be training making the model play the role.

## Run
Same frozen `meta_judge` question and judge model (Gemma 4 31B instruct), blind as before:
- **A. Atlas light_system and light_user:** OLMo and Qwen, base and instruct; Gemma too once its replies exist. SFT and DPO join when generated.
- **B. Saved leakage-matrix replies (`results/dominance_matrix/gens`):** the character contexts behind the style-score fall, each with its own role description. Base and final, all families there. If a context has no role description to give the judge, skip it and say so.

Timing: at the next checkpoint boundary on vinyl if the queue allows it without restarting anything, otherwise right after atlas Gemma. Report the rate after a small sample; stop and tell me if it projects over 1 hour.

## Readout (one table per family; change from base, 95% two-way bootstrap as in the atlas)
- **Share in character** by context and stage.
- **Atlas, like-for-like:** for every trait and every context with a meta judgment (including the existing strong_act and strong_noact), change from base among in-character replies only, beside the all-replies change. Require at least 6 items with replies on both sides; otherwise "too few".
- **Leakage matrix:** the same split on the text classifier and layout regex, and on the style score if per-reply scores are saved. Say which were available.

## Pre-set readings
- **Light role, real spread:** in-character change is at least 5 points, same sign as the Assistant's, CI excluding 0.
- **Light role, not a character:** the share in character is under 50% at the final model, and the in-character change misses that bar. Report under either reading how many replies are in character.
- **Style fall is mostly composition:** the in-character fall is under 50% of the all-replies fall. **Survives:** it is at least 50%, at least 5 points, CI excluding 0. Otherwise unresolved.

## Hand check
Add a separate blind sheet of 30 replies for Adeeb on this judge question, stratified by model, stage, context and yes/no. Key stays on vinyl. Bar: 80% agreement. Below it, this whole check is reported as unreliable.

## Limits to state
In-character replies are a selected group, not a random sample. Base has few of them. Judge not yet validated. 12 items per trait. Post-hoc.
