# Spec A: "Is the pretend user too polished?" (user-slot style fingerprint)

From the laptop Claude (design/analysis), approved by Adeeb 2026-11-03. Owner: marble Claude (build + run).
Exploratory, descriptive, outside the verdict. Log the analysis plan in `docs/decisions.md` (dated, atomic append) BEFORE looking at any
result, update STATUS.md and data/README.md if you add data files. Code in `measures/overgeneralization/` (e.g. `user_style.py`),
tests in `tests/test_user_style.py`. CPU only. Use subagents for building/reviewing if it helps your context.

## Question
When a model continues a real WildChat user's message (as the user), does post-training make that text read more like the Assistant
(polished, formal, structured) than the real user's own continuation, and more than the base model's?

## Inputs
- Generations: `results/capstone/<family>/core/generations_incomplete/<stage>.jsonl` (200 mid-sentence prefills x 3 samples). OLMo 4
  stages (being generated on marble now), Qwen3-8B base/instruct (generated on nurse, pull back). Optionally also the existing
  sentence-boundary set `generations/` as a second stratum (report separately, never pooled).
- Human reference: the real remainder of the same user turn (full text from `data/texts.jsonl` via `text_id`, minus the prefill).
- User portion: reuse `user_traits.py`'s cut (turn-end / handoff markers / line-initial role labels).

## Two views of the continuation (report both)
1. **Completion of the cut sentence:** text up to the first sentence end (. ? ! or newline) of the user portion. This is unambiguously
   the user's own sentence, so polish here cannot be a switch to the assistant. Primary.
2. **Full user portion**, and the same restricted to >= 15 words.
Real-user reference is cut the same way (first sentence of the remainder; first N words matched to the model's length per sample).

## Measures
1. **Human-vs-Assistant classifier (primary score).**
   - Training data: held-out WildChat, English, first user turns vs first assistant replies, about 2000 each, from conversations NOT in
     `data/texts.jsonl` / `data/swap.jsonl` / the prefill sets. Stream with the same loader as `core.prepare_data` (log the commit/date).
     Save as a data file with a data/README.md row (or save only ids + a script if size is an issue).
   - Length control: train on random windows of 5-40 words from each text (so length is not the cue); evaluate on windows matched to the
     evaluation text lengths.
   - Model: TF-IDF (char 2-5 + word 1-2) + logistic regression, fixed seed. Report held-out AUC overall and by window length; must be
     >= 0.85 at 10-20 words to be used (pre-set gate; if it fails, report and stop the classifier part).
   - Score: P(assistant) per sample. Per prefill mean; stage minus base, paired per prefill, 95% bootstrap CI (`core.stats.bootstrap_ci`);
     also each stage minus the real-user reference.
2. **Style features** (each: stage minus base, paired per prefill, CI; plus the real-user value): starts lowercase; has a typo
   (fraction of tokens not in a word list, e.g. a standard English word list, ignoring code/URLs/numbers); mean sentence length;
   politeness words (please, thank you, kindly, appreciate); exclamation; markdown (headings, bullets, bold); proper capitalisation of "i";
   ends with terminal punctuation.
3. **Style matching:** for binary prefill features (prefill all-lowercase; prefill has no terminal punctuation; prefill has a typo),
   agreement rate between prefill and continuation, stage minus base. "Normalising the user" = agreement drops with training.

## Pre-set reading (log before results)
- Primary: classifier P(assistant) on the completion-of-cut-sentence view, OLMo instruct minus base. "Polish leak" if >= +0.05 with CI
  excluding 0 AND the instruct value exceeds the real-user reference by >= 0.05. Qwen supporting; Gemma not generated yet.
- Everything else descriptive.

## Outputs
`results/capstone/<family>/overgeneralization/user_style.md` + small csv, 3 real examples per stage (highest and lowest P(assistant)).
Commit code/tests/docs/small results only. Report to the laptop Claude: the classifier gate, the primary table, examples.
