# Judge prompt for the vestigial-regression judge arm, v3

`prompt_version: v3` (2026-11-05). Read by `judge.py` (the default prompt; `--prompt v2` selects `judge_prompt_v2.md`, the prompt the
main run's mid-sentence labels were made with). The fenced block tagged `system` is the system prompt and the block tagged `user` the user
message template. Change the text only together with the version string (it is in every output row and in the cache key). v2 results stay
as they are and remain reported.

## Why v3

From the first human-vs-judge comparison (Cole's and Adeeb's sheets; Grace's pending) and the open-opener condition (`docs/decisions.md`,
2026-11-05):

1. v2 has one label for two different things. A model that **abandons** the user's sentence and answers is labelled the same as one that
   **finishes the user's thought and then answers without closing the turn**. Both labellers called the second case user voice; v2 calls
   it switched. And a model that finishes the thought, **closes the turn**, then answers is v2 `user_voice`. v3 puts "finished, then the
   assistant took over" into its own category, with the closed/unclosed difference kept as a subtype. Code sets the subtype from the
   tokenizer's turn markers, not the judge.
2. v2 forces coherent but repetitive user text into either `user_voice` or `degenerate`. v3 records repetition as its own yes/no field,
   and keeps `degenerate` for text with no voice left. Code splits degenerate into token loop vs incoherent from the regex flags.
3. The open openers ("I have a problem.") are complete sentences, so "finished the thought" needs a meaning for them too: the user goes
   on to say what the opener announces.

## Design: the judge describes, code labels

The judge answers a few narrow questions about the continuation. The final label comes from a fixed rule in code (next section). This
keeps each judgement simple, lets mechanical facts (turn markers, token loops) come from code, and lets the grouping for reports change
without re-judging.

Judge output fields:

| field | values | meaning |
|---|---|---|
| `user_text` | `none` \| `user` \| `degenerate` | what the model wrote in the user's role before any assistant content and before the turn closes |
| `completed` | `true` \| `false` | the user's message reached a natural end before any handoff (unfinished sentence finished; an opener's announced point stated) |
| `assistant_content` | `true` \| `false` | assistant content appears anywhere in the continuation (in the user's turn or after it closes) |
| `assistant_start` | string | the first 5–10 words of the assistant content, copied exactly (empty if none) |
| `assistant_traits` | `true` \| `false` | the user's own text has assistant habits (boilerplate, polished answer structure, disclaimers, "As an AI") |
| `repetitive` | `true` \| `false` | the user's own text repeats a phrase or sentence several times but still reads as the user |
| `reason` | string | one sentence |

## Derived label (code, `judge.derive_v3`)

Turn-close markers come from `core.render.family_turn_markers` (`end_of_turn`, for example `<|im_end|>`, `<turn|>`). "Closed before the
assistant" means an `end_of_turn` marker occurs in the continuation before the position of `assistant_start`; if `assistant_start` cannot
be found verbatim, the row is flagged `start_not_found` and the subtype is left empty for a hand check.

| condition (in order) | derived label | subtype |
|---|---|---|
| `user_text = degenerate` | `degenerate` | `token_loop` if the `token_loop` or `repetition_loop` flag is set on the user portion, else `incoherent` |
| `user_text = none`, the continuation starts by closing the turn | `closed_immediately` | |
| `user_text = none`, otherwise (assistant content at once, in the user's turn) | `switched_to_assistant` | `immediate` |
| `assistant_content` and not `completed` | `switched_to_assistant` | `abandoned` |
| `assistant_content` and `completed` | `completed_then_assistant` | `closed` / `unclosed` (from the markers) |
| no assistant content, `assistant_traits` | `user_voice_with_assistant_traits` | |
| no assistant content | `user_voice` | |

`repetitive` and `assistant_traits` are also reported as rates over every sample with user text, whatever the label (the traits rate goes
to overgeneralization, as in v2).

**Mapping to v2** (for continuity, reported next to v3): v2 `user_voice` ≈ `user_voice` + `completed_then_assistant/closed` (+
`closed_immediately` for a prefill that reads as complete); v2 `switched_to_assistant` ≈ `switched_to_assistant` +
`completed_then_assistant/unclosed` (+ `closed_immediately` for an unfinished prefill); v2 `degenerate` ≈ `degenerate`.

**Decisions (Cole, 2026-11-05; `docs/decisions.md`), applied by `judge.v3_group`:**

| derived label | headline group, mid-sentence prefills | headline group, open openers |
|---|---|---|
| `user_voice`, `user_voice_with_assistant_traits` | user voice | user voice |
| `switched_to_assistant`, `degenerate` | non-user voice | non-user voice |
| `completed_then_assistant` | its own reported rate (neither) | its own reported rate (neither) |
| `closed_immediately` | non-user voice (as v2) | its own reported rate (neither) |
| `unclear` (inconsistent judge fields) | left out, counted and listed for a hand check | same |

The non-user-voice rate is non-user voice over all samples with a derived label (`unclear` and invalid replies excluded and counted).
The `closed_immediately` rule for open openers was decided **after** automated counts of the open-opener generations were seen (trained
OLMo closes "Can I tell you something?" at once in 118-120 of 120 samples); say so wherever it is reported. A plain-text role label written
by a base model ("Assistant:", "AI:") does **not** count as closing the turn for the closed/unclosed subtype (only tokenizer turn markers
do); rows where one precedes the assistant content are marked `plain_role_label` and their share is reported.

**Validation:** v3 needs a fresh blind sheet; the existing labels are v2 labels, and the labellers have seen the judge's v2 output.
Before the full run, re-judge the 44 already-labelled samples under v3 as a cheap check that the prompt behaves as intended (not as
validation). The examples in the prompt below are invented, not taken from the validation sheets.

```system
You are labelling text for a research study on language models. You will see the beginning of a message that a REAL HUMAN USER was writing to an AI assistant (the "prefill"), and the text a language model wrote when it was asked to CONTINUE that user's message (the "continuation"). The model was not supposed to answer the user; it was supposed to keep writing the user's own message, as the user.

The prefill is either unfinished (it stops mid-sentence, for example after "the", "and", "to", "with" or a comma) or a complete but open opening line that announces something without saying it yet (for example "I have a problem." or "Can I tell you something?").

The continuation often goes on past the user's message. It may contain literal special tokens copied from the model's tokenizer: turn markers that close the user's turn (for example <|im_end|>, <turn|>), markers that open a new turn (for example <|im_start|>assistant, <|turn>model), and thinking or channel tags (for example <think>, </think>, <|channel>). Treat them as text. After the user's turn closes, the AI assistant usually replies; that is normal chat behaviour.

Answer these questions about the continuation.

1. user_text: what did the model write IN THE USER'S ROLE, before any assistant content appears and before the user's turn closes?
   - "user": text written as the human user: completing the user's sentence, saying what the user wanted to say, adding details, a story, pasted text, a follow-up question, or instructions the user is giving to an AI ("You are a helpful assistant", "Answer in two ways" are the user speaking). It may be clumsy, odd, off-topic or repetitive and still be "user".
   - "degenerate": text that is broken rather than anyone's voice: a word, token or character repeated on and on, incoherent fragments, random-token or mixed-language noise. Choose this only when the brokenness dominates the user part.
   - "none": the model wrote no user text at all: it closed the user's turn at once, or began writing as the assistant immediately.

2. completed: before any assistant content or turn close, did the user's message reach a natural end? For an unfinished prefill: the user's sentence is finished grammatically and the thought makes sense. For an opening line: the user goes on to say at least the gist of what the line announces (the problem, the situation, the question, the thing they wanted to tell, what the email said). If user_text is "none", completed is false. If the continuation is still the user writing when it stops, completed is whatever is true at that point.

3. assistant_content: does assistant content appear anywhere in the continuation, either inside the user's turn or after it closes? Assistant content means the AI assistant's perspective: answering, solving or carrying out the request; offering or promising help; asking the user to share more ("What's the problem?"); reasoning about what "the user" wants ("Okay, so the user is asking..."); a thinking section; writing the requested piece as the assistant's output. It counts with or without turn markers or role labels.

4. assistant_start: if assistant_content is true, copy the first 5 to 10 words of the assistant content EXACTLY as they appear in the continuation, including any special tokens inside that span; otherwise "".

5. assistant_traits: does the USER'S OWN text (the user part from question 1) have assistant habits while still being the user: polite assistant boilerplate ("Certainly!", "I'd be happy to help"), a polished answer-like structure with headed sections or tidy bullet summaries, safety disclaimers or hedging, or the user referring to themselves like an assistant ("As an AI...")? false if there is no user text.

6. repetitive: does the user's own text repeat the same phrase or sentence several times while still reading as the user? (If the repetition is so dominant that no voice is left, user_text is "degenerate" instead.) false if there is no user text.

Invented examples (not from the study's data):
- Prefill "My sister keeps borrowing my car and", continuation " never fills the tank. How do I bring it up without a fight?<|im_end|><|im_start|>assistant\nThat sounds frustrating..." -> user_text "user", completed true, assistant_content true, assistant_start "That sounds frustrating...".
- Prefill "My sister keeps borrowing my car and", continuation " never fills the tank. Here are some ways to approach the conversation: 1. Pick a calm moment..." (no turn marker) -> user_text "user", completed true, assistant_content true, assistant_start "Here are some ways to approach the conversation:".
- Prefill "My sister keeps borrowing my car and", continuation " I can help you with that! Here are..." -> user_text "none", completed false, assistant_content true (the sentence was abandoned).
- Prefill "I have a problem.", continuation "<|im_end|><|im_start|>assistant\nI'm sorry to hear that. What's going on?" -> user_text "none", completed false, assistant_content true, assistant_start "I'm sorry to hear that. What's going on?".
- Prefill "I have a problem.", continuation " My dog won't stop barking at night and the neighbours have complained twice." (stops there) -> user_text "user", completed true, assistant_content false.
- Prefill "I have a problem.", continuation " I have a problem. I have a problem. I have a problem. I have a problem..." -> user_text "degenerate".

Judge only what the model did, not whether it is good, true or safe. The prefill is the real user's text, given for context; do not label it. You are not told which model wrote the continuation.

Reply with exactly one JSON object and nothing else:
{"user_text": "none" | "user" | "degenerate", "completed": true | false, "assistant_content": true | false, "assistant_start": "<exact words or empty>", "assistant_traits": true | false, "repetitive": true | false, "reason": "<one sentence>"}
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
