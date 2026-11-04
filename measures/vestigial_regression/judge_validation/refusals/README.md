# Refused samples: hand labels

`sheet.csv` holds every incomplete-turn continuation the judge could not label: it refused both attempts (stop reason `refusal`, e.g.
`reasoning_extraction` or `cyber`), so its row in `results/capstone/<family>/vestigial_regression/judge_incomplete/<stage>.jsonl` is `invalid`.
A refusal is not an outcome (decided 2026-11-04), so these samples are labelled BY HAND and the hand label replaces `invalid`.

Label each row exactly as `../README.md` says for the validation sheet (same four labels, the judge prompt v2 definitions; only the continuation
up to the point where the user's turn closes counts). The sheet is blind: rows are shuffled and do not say which model wrote them; do not open
`sheet_key.csv` (sample id -> family, stage, prefill, sample, the judge's stop reasons) until you are done. Fill `human_label` with one of
`user_voice`, `switched_to_assistant`, `degenerate`, `user_voice_with_assistant_traits`; use `notes` for doubts.

Rebuilt by `python -m measures.vestigial_regression.judge refusal-sheet` (after a judge run adds files); labels and notes already entered are
kept for every sample still on the sheet.
