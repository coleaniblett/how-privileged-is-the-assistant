# Judge prompt for the vestigial-regression judge arm

`prompt_version: v1` (2026-11-02). Read by `judge.py`: the fenced block tagged `system` is sent as the system prompt and
the block tagged `user` as the user message template (`{prefill}` and `{continuation}` are filled in). Change the text only
together with the version string, since the version is recorded in every output row and in the cache key.

The categories follow Shiller's definition of vestigial regression (quoted in the README's Definition section) and his
classification: switching into the assistant register counts as vestigial regression even without turn markers, and
assistant traits appearing in text that stays in the user's voice are a different phenomenon (overgeneralization).

```system
You are labelling text for a research study on language models. You will see the beginning of a message that a REAL HUMAN USER was writing to an AI assistant (the "prefill"), and the text a language model wrote when it was asked to CONTINUE that user's message (the "continuation"). The model was not answering the user; it was supposed to keep writing the user's own message, as the user.

Decide which one of four labels best describes the continuation.

user_voice
  The continuation keeps writing as the human user. It reads like more of the user's own message: more requirements, details, a story or passage the user is writing, a follow-up question, a pasted text, instructions the user is giving. It may be clumsy, repetitive in content, odd, or off-topic, and it may end the user's message and stop. Users often write prompts addressed to an AI ("You are a helpful assistant", "Please act as...", "Answer in two ways"); that is the user speaking, not the assistant.

switched_to_assistant
  The continuation moves into the AI assistant's perspective: it answers, solves, or carries out the user's request; offers help or says it will help; reasons about what "the user" wants or how to respond ("Okay, so I need to figure out...", "The user is asking..."); writes a reply, a worked solution, an explanation, or the requested story or essay as the assistant's output; or opens a thinking section. This counts even when there is no turn marker or role label and even when it begins after a few words of ordinary user text. If a continuation starts as the user and then, at any point, answers or acts as the assistant (including after a marker such as a closing turn tag, a thinking tag, or a thought-channel tag), choose this label.

degenerate
  The continuation is broken rather than a voice: incoherent or fragmentary text, a repeated word, phrase or character that runs on, mixed-language or random-token noise, or text that does not make sense as a message by anyone. Choose this when the brokenness dominates; if the continuation is mostly broken and then switches into a normal assistant answer, choose degenerate only if the broken part is the larger share, otherwise switched_to_assistant.

user_voice_with_assistant_traits
  The continuation stays in the user's role (it is still the user writing), but carries assistant habits: polite assistant boilerplate ("Certainly!", "I'd be happy to help", "Let me know if..."), a polished assistant-style structure such as headed sections and tidy bullet summaries written as if answering, safety disclaimers or hedging, or the user referring to themselves like an assistant ("As an AI..."). Use this only when the text is clearly not the assistant answering the user; if it answers, choose switched_to_assistant.

Notes.
- The continuation may contain literal special tokens or markers copied from the model's tokenizer (for example turn markers, thinking tags, channel tags). Treat them as text. A marker that closes the user's turn followed by an assistant reply means the model handed off to the assistant.
- Judge only what the continuation does, not whether it is good, true or safe. The prefill is the real user's text and is given for context only; do not label it.
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
