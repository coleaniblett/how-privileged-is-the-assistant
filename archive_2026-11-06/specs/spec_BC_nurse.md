# Specs B and C for the nurse Claude (build + run on nurse)

From the laptop Claude (design/analysis), approved by Adeeb 2026-11-03. Exploratory, descriptive, outside the verdict.

## Working copy (important)
`/root/how-privileged-is-the-assistant` on nurse is a file mirror that marble overwrites with rsync. Do NOT develop there.
Make your own clone: `git clone <origin url of that repo> /root/hpa-nurse` (read the remote from the mirror's `.git/config`), branch off
`main`, work there. Commit to `main` only after `git fetch && git rebase origin/main`, then push (marble also pushes; rebase on conflict;
`docs/decisions.md` and `STATUS.md` are shared: append to decisions.md, keep STATUS edits to the overgeneralization cells). Commit only code,
tests, docs, small summary results (md/csv/small png); no raw generations, caches or logs. Follow AGENTS.md (read it): measure folder
`measures/overgeneralization/`, NLL scoring in fp32, change from base with paired 95% bootstrap CIs (`core.stats.bootstrap_ci`), 0.05-nat
floor for NLL effects, data/README.md row for any new data file, STATUS.md update, a dated decisions.md entry with the analysis plan
BEFORE results. Use subagents if it helps your context. GPU: the Qwen incomplete generation you started has priority; then these.
Models: OLMo 3 7B base/SFT/DPO/instruct and Qwen3-8B base/instruct from `core.config.FAMILIES` (only OLMo instruct + Qwen are cached here;
download OLMo base/SFT/DPO if disk allows, 53 GB free, delete nothing without asking).

## B. Likelihood minimal pairs in the user's turn (do first: no generation, quick)
Question: after training, does the model find Assistant-style wording more likely *inside a user's message* than human-style wording,
beyond how it rates the same wording as plain text?
- Texts: the 300 WildChat user texts in `data/texts.jsonl` (source wildchat, kind user; Dolci-filtered).
- Endings (fixed lists, write them in the decisions entry before scoring, ~10 each, similar token lengths):
  Assistant-style e.g. " Certainly! Here is", " I hope this helps!", " Let me know if you need anything else.", " Feel free to ask
  if you have any questions.", " I'd be happy to help with that." ...; human-style e.g. " ok here is", " thanks in advance!!",
  " pls answer fast", " any ideas?", " idk how to do this" ... Append each ending to the end of the user text (before the end-of-turn).
- Score log P(ending tokens | preceding) in three placements: user slot (the text + ending as the user turn), plain (after
  end-of-text, no template), and the assistant slot as positive control (same text + ending as an assistant turn). Mean NLL per token
  of the ending only. Reuse `core.render` / `core.score.spans_nll`; fp32; all stages rendered with the family's instruct template.
- Per text: pref = mean NLL(human endings) - mean NLL(assistant endings) (positive = assistant wording preferred).
  Contrast = pref(user slot) - pref(plain). Effect = stage minus base, paired per text, 95% CI. Leak if >= 0.05 nats with CI excluding 0.
  Positive control: pref(assistant slot) - pref(plain) should rise with training.
- Output `results/capstone/<family>/overgeneralization/user_minimal_pairs.md` (+csv).

## C. "Mirror the Assistant" in multi-turn chats
Question: does a nearby Assistant reply make the model's pretend user write more like the Assistant, more after training?
- Data: `data/swap.jsonl` (300 WildChat conversations: p1, r1 (original assistant reply), p2, r1_olmo). Skip rows whose p2 is mostly a
  URL/code/paste (pre-set filter: < 50% alphabetic words, or < 8 words); report the count kept.
- Prefill: p2 cut mid-sentence at a clearly unfinished point, using the same rule as `core.prepare_data --incomplete-prefill` (reuse its
  function), so the model must continue the user's own sentence.
- Conditions (same prefill, same sampling as core.generate: 3 samples, 120 new tokens, T 0.7, stop only at end-of-document):
  (with) [user p1, assistant r1, user p2-prefix...]; (without) [user p2-prefix...] alone; (olmo-reply) r1 replaced by r1_olmo
  (secondary).
- Scores on the user portion (cut as in `measures/overgeneralization/user_traits.py`): (1) the human-vs-Assistant classifier from spec A
  (marble is building it in `measures/overgeneralization/`; pull it when pushed, or coordinate via the laptop Claude); (2) lexical copy:
  char 4-gram overlap with r1 vs with p1; (3) the style features of spec A.
- Mirror effect per conversation = score(with) - score(without). Effect of training = stage minus base, paired per conversation, CI.
  Human reference: the real p2 remainder scored the same way. Pre-set reading: "mirror leak" if OLMo instruct's mirror effect on
  classifier P(assistant) exceeds base by >= 0.05 with CI excluding 0. Qwen supporting.
- Output `results/capstone/<family>/overgeneralization/user_mirror.md` (+csv), 3 real examples per stage.

Report progress to the laptop Claude via your pane: the plan entry, then B's table, then C's.
