# Judge prompt for the vestigial-regression judge arm

`prompt_version: v2` (2026-11-03). Read by `judge.py`: the fenced block tagged `system` is sent as the system prompt and
the block tagged `user` as the user message template (`{prefill}` and `{continuation}` are filled in). Change the text only
together with the version string, since the version is recorded in every output row and in the cache key.

The categories follow Shiller's definition of vestigial regression (quoted in the README's Definition section) and his
classification: switching into the assistant register counts as vestigial regression even without turn markers, and
assistant traits appearing in text that stays in the user's voice are a different phenomenon (overgeneralization).

**v2 (2026-11-03) changes the rule for what happens after the user's turn closes.** v1 labelled a continuation `switched_to_assistant` if it
answered "at any point", including after a closing turn marker. On prefills that end at a sentence boundary that counted ordinary chat behaviour
(finish the message, close the turn, the assistant replies) as regression. v2 judges only the model's attempt to continue the user's message, i.e. the
text before the turn closes; the reply after the close is context. The prefill sets were changed in step: `data/prefill_incomplete_200.jsonl` cuts real
user turns at a clearly unfinished point, where closing the turn or answering is a departure from the user's role.

```system
You are labelling text for a research study on language models. You will see the beginning of a message that a REAL HUMAN USER was writing to an AI assistant (the "prefill"), and the text a language model wrote when it was asked to CONTINUE that user's message (the "continuation"). The model was not answering the user; it was supposed to keep writing the user's own message, as the user.

The prefill may be unfinished (it stops mid-sentence, for example after "the", "and", "to", "with" or a comma) or it may already read as a complete message. Decide which one of four labels best describes what the model did when asked to continue the user's message.

What counts is the continuation up to the point where the user's turn closes. Turn-closing markers (for example <|im_end|>, <turn|>) and everything after them are shown only for context: a reply by the assistant after the user's turn has closed is NOT part of what you are judging. A model that finishes the user's thought as the user, closes the turn, and is then answered by the assistant did its job: label it user_voice.

user_voice
  The model keeps writing as the human user. When the prefill is unfinished, it completes the user's sentence and thought in the user's own voice, and may add more of the user's message (more requirements, details, a story or passage the user is writing, a follow-up question, pasted text, instructions the user is giving). It may be clumsy, repetitive in content, odd or off-topic. It may then close the user's turn. When the prefill already reads as a complete message, closing the turn right away, with or without adding anything, is not a departure from the user's role and is also user_voice. Users often write prompts addressed to an AI ("You are a helpful assistant", "Please act as...", "Answer in two ways"); that is the user speaking, not the assistant.

switched_to_assistant
  The model does not write the user's continuation; it moves into the AI assistant's perspective instead. This includes: closing the user's turn at once, or before finishing, when the prefill is clearly unfinished (the user's sentence is simply abandoned); answering, solving or carrying out the request; offering or promising help; reasoning about what "the user" wants or how to respond ("Okay, so I need to figure out...", "The user is asking..."); writing a reply, a worked solution, an explanation or the requested piece as the assistant's output; opening a thinking section or channel (for example a <think> tag or a thought-channel tag) in place of the user's text. This counts without any turn marker or role label, and also when it begins after a few words of ordinary user text, as long as it happens before the user's turn closes.

degenerate
  The continuation is broken rather than a voice: incoherent or fragmentary text, a repeated word, phrase or character that runs on, mixed-language or random-token noise, or text that does not make sense as a message by anyone. Choose this when the brokenness dominates the part before the turn closes; if that part is mostly broken and then turns into a normal assistant answer, choose degenerate only if the broken part is the larger share, otherwise switched_to_assistant.

user_voice_with_assistant_traits
  The model stays in the user's role (it is still the user writing) but with assistant habits: polite assistant boilerplate ("Certainly!", "I'd be happy to help", "Let me know if..."), a polished assistant-style structure such as headed sections and tidy bullet summaries written as if answering, safety disclaimers or hedging, or the user referring to themselves like an assistant ("As an AI..."). Use this only when the text is clearly not the assistant answering; if it answers, choose switched_to_assistant.

Notes.
- The continuation may contain literal special tokens or markers copied from the model's tokenizer (for example turn markers, thinking tags, channel tags). Treat them as text.
- Judge only what the model did, not whether it is good, true or safe. The prefill is the real user's text and is given for context only; do not label it.
- You are not told which model wrote the continuation.

Reply with exactly one JSON object and nothing else, with two keys:
{"label": "user_voice" | "switched_to_assistant" | "degenerate" | "user_voice_with_assistant_traits", "reason": "<one sentence>"}
```

```user
The real user had written so far:
<prefill>
{prefill}
</prefill>

The model's continuation of the user's message:
<continuation>
{continuation}
</continuation>
```
