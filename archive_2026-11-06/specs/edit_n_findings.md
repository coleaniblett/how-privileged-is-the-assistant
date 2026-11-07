Edits requested by the laptop Claude after reading spec N. Apply exactly, commit and push. Touch nothing else in these files. Don't read prefill judge output.

1. measures/overgeneralization/presentable_findings.md, finding 4:
- Replace the table row `| Gemma E4B | 12 → 82% | 14 → 26% | 12 → 2% | 8 → 14% |` with `| Gemma 31B | 5 → 42% | 13 → 12% | 3 → 0% | not run |`
- Replace the sentences "A thin role doesn't override the trained refusal habit, while a vivid role does. In Qwen and Gemma, training strengthens both effects." with: "In OLMo and Qwen, a thin role doesn't override the trained refusal habit, while a vivid role does, even with no \"never refuses\" line (spec L2). Gemma 31B differs: the line works on thin roles too, which refuse 6–18% with it, so the gate reading there is inconclusive (thin minus vivid +6 and +18; needs 30). E4B results are retired and kept as history (`results/working_model_L/L2_L3.md`)."
- Add this caveat bullet to finding 4's caveats, or a caveats line if it has none: "**Gemma 31B detector:** a spot check of 30 replies found no false positives but about 5 likely misses, all in-character declines, so its thin and vivid rates may be undercounts. Thin roles also vary widely (no line, 95% ranges 2–82 and 1–51)."
- Add `results/working_model_L/L2_L3_gemma31b.md` to finding 4's sources, if it lists sources.
- In finding 8's table, change the label `Gemma E4B` to `Gemma E4B (retired, kept as history)`.
- Update the "Last updated" line to 2026-11-05, about 7:15 PM UK.

2. measures/overgeneralization/slide_brief.md:
- Same-models line: remove "and Gemma 4 E4B" so it reads "This section also uses Llama 3.1 8B and Apertus 8B. Mark them where they appear."
- Replace the row `| Gemma 4 E4B, mean of two wordings | 12 → 82% | 14 → 26% | 12 → 2% |` with `| Gemma 4 31B, mean of two wordings | 5 → 42% | 13 → 12% | 3 → 0% |`
- Immediately after the refusal slide's "Closing line" bullet, add: "- **Note:** in Gemma 31B the \"never refuses\" line also works on thin roles, so thin characters keep refusing in OLMo and Qwen only. Its detector may undercount in-character declines (5 of 30 likely misses)."

3. measures/overgeneralization/slide_figures.py:
- In REFUSAL, replace `("Gemma 4 E4B", (12, 82), (14, 26), (12, 2))` with `("Gemma 4 31B", (5, 42), (13, 12), (3, 0))`.
- Change the comment above it to add: "Gemma 4 31B: results/working_model_L/L2_L3_gemma31b.md, line present, mean of two wordings."
- Rerun `python -m measures.overgeneralization.slide_figures` and commit only og3_refusal_thin_vivid.png and .svg; restore any other figure that changed. Check the labels don't collide and tell me.

Commit message should say it replaces E4B with 31B in the refusal row, following spec N. Don't disturb running jobs: nice -n 15.
