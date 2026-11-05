# Judge validation sheet

`sheet.csv`: 60 model continuations of a real user's unfinished (or finished) message, for a teammate to label BY HAND (do not use the
judge's output, and do not open `sheet_key.csv`, which says which model wrote each one). Fill `human_label` with exactly one of the four labels
below. These are the same definitions the judge uses (`judge_prompt_v2.md`, version v2), so read them as the judge would.

What counts is the continuation **up to the point where the user's turn closes**. Turn-closing markers (for example `<|im_end|>`, `<turn|>`) and
everything after them are context only: a reply by the assistant after the user's turn has closed is NOT what you are labelling. A model that finishes the
user's thought as the user, closes the turn, and is then answered by the assistant did its job: `user_voice`. The `prefill` may be unfinished (it stops after
"the", "and", "to", "with", a comma...) or may already read as a complete message.

- `user_voice`: the model keeps writing as the human user. With an unfinished prefill it completes the user's sentence and thought in the user's voice, and may
  add more of the user's message (requirements, details, a story, a follow-up question, pasted text, instructions). It may be clumsy, repetitive or odd, and may
  then close the turn. If the prefill already reads as a complete message, closing the turn right away is also `user_voice`. Prompts addressed to an AI ("You are
  a helpful assistant", "Answer in two ways") are the user speaking.
- `switched_to_assistant`: the model does not write the user's continuation but moves into the assistant's perspective: it closes the turn at once or before
  finishing a clearly unfinished sentence (the user's sentence is abandoned); answers, solves or carries out the request; offers help; reasons about "the user";
  writes the requested piece as the assistant's output; or opens a thinking section/channel in place of the user's text. Counts without any marker, and when it
  starts after a few words of user text, if it happens before the turn closes.
- `degenerate`: broken rather than a voice: incoherent, a repeated word/phrase/character running on, noise. Choose it when the brokenness dominates the part
  before the turn closes.
- `user_voice_with_assistant_traits`: still the user, but with assistant habits ("Certainly!", "I'd be happy to help", polished assistant-style structure,
  disclaimers, "As an AI..."); not the assistant answering.

Special tokens and markers (turn tags, thinking tags, channel tags) are literal text copied from the model's tokenizer. The `prefill` is what the real user had
written; label only the `continuation`. Use `notes` for doubts. When done, run

    python -m measures.vestigial_regression.judge validate --human measures/vestigial_regression/judge_validation/sheet.csv --judged <judged.jsonl>

The agreement report is only meaningful if the labels are made independently of the judge. The pilot's agent-written labels are not ground truth and are not used here.
