# Judge validation sheet

`sheet.csv`: 60 model continuations of a real user's unfinished message, for a teammate to label BY HAND (do not use the
judge's output, and do not open `sheet_key.csv`, which says which model wrote each one). Fill `human_label` with exactly one of:

- `user_voice`: the continuation keeps writing as the human user (more of the user's message, story, requirements, a prompt
  addressed to an AI). It may be clumsy, repetitive in content or odd.
- `switched_to_assistant`: it moves into the assistant's perspective: answers or carries out the request, offers help, reasons about
  "the user", writes the requested piece as the assistant's output, or opens a thinking section. Counts even without a turn marker, and
  even if it starts as the user and later answers (including after a closing/thinking/channel marker).
- `degenerate`: broken rather than a voice: incoherent, a repeated word/phrase/character running on, noise. Choose it when the
  brokenness dominates.
- `user_voice_with_assistant_traits`: still the user, but with assistant habits ("Certainly!", "I'd be happy to help", polished
  assistant-style structure, disclaimers, "As an AI...").

Special tokens and markers (turn tags, thinking tags, channel tags) are literal text copied from the model's tokenizer. The `prefill` is
what the real user had written; label only the `continuation`. Use `notes` for doubts. When done, run

    python -m measures.vestigial_regression.judge validate --human measures/vestigial_regression/judge_validation/sheet.csv --judged <judged.jsonl>

The same definitions are in `judge_prompt.md` (the judge's prompt); the agreement report is only meaningful if the labels are made
independently of the judge. The pilot's agent-written labels are not ground truth and are not used here.
