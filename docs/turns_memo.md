# How texts should be presented for NLL and entropy scoring (decision memo, 2026-11-03)

For Cole (vestigial regression) and Grace (logit distortion). This memo recommends; the decision is yours.
Code, data and full tables: `exploration/turns_test/` (README there). Analysis only: nothing in `core/`, `measures/` or `data/` changed.

## Summary

**The choice.** Three designs: **A**, every text scored alone (the overnight design: user turns open a conversation,
replies have no question before them); **B**, each reply scored after its own user turn; **C**, B plus a later user turn
scored after user turn 1 and the reply. In all three, *plain* is the same token sequence with every role tag (and the
system turn) removed.

**What the test showed** (150 held-out three-turn conversations: 50 each from WildChat, oasst1 and UltraChat; OLMo 3 7B
base vs Instruct, Qwen3-8B base vs IT; fp32):
1. **Leaving out the question shifts the reply's frame cost differently after post-training, so design A is biased.**
   The change from base of the reply's frame cost is 0.13–0.47 nats lower without the question than with it, in all
   six model × source cells, every CI excluding zero. The shift sits almost entirely in the tagged sequence, not in plain
   text. Design A therefore makes the post-trained assistant term look smaller and pushes the vestigial-regression
   asymmetry up, toward finding regression.
2. **That bias is large enough to flip Qwen's sign, but not OLMo's.** OLMo's asymmetry is negative under every design:
   −0.19 to −0.38 under A, −0.49 to −0.69 under B, so B is about twice A. Qwen3-8B is +0.04 to +0.07 under A (CIs include
   zero; the pilot's "mildly regressed" reading), but −0.07 to −0.24 under B and −0.13 to −0.30 under C (C excludes zero
   in all three sources). A verdict on Qwen therefore depends on the design.
3. **For logit distortion, the reply's gap barely depends on context; the user comparison does.** The change in the
   reply gap from adding the question is ≤ 0.09 nats for Qwen and 0 / 0.23 / 0.36 for OLMo. Swapping the user opener for
   user turn 2 (design C) moves the reply-minus-user contrast by up to 0.5 nats and flips its sign for Qwen on UltraChat.

**Recommendation.**
- *Logit distortion:* score replies after their question (B). This is already Grace's original-conversation route.
  Stratify user turns by opening vs follow-up and report both; never pool them.
- *Vestigial regression:* keep A as the pre-registered primary, because it is collected and locked. Add B, and C on the
  WildChat three-turn set, as pre-declared robustness rows. Count the NLL arm as firing only if it fires under both A and
  B, and report a design-dependent result as exactly that. B costs a few minutes per 7–8B checkpoint and needs no new
  data for WildChat.

## Results (change from base, instruct − base, nats per token, content positions 1+, 95% paired bootstrap CI over conversations)

FC = frame cost, NLL(tagged) − NLL(plain). G = NLL − entropy (tagged). u1: user turn 1; rA: reply alone; rB: reply after
its question; u2C: user turn 2 after u1 and the reply. n = 50 per source, so the CIs are wide.

| model | quantity | WildChat | oasst1 | UltraChat |
|---|---|---|---|---|
| OLMo 3 7B | ΔFC u1 (all designs) | −0.16 [−0.29, −0.04] | −0.19 [−0.25, −0.12] | −0.06 [−0.12, −0.00] |
| | ΔFC reply, A / B | +0.08 / +0.33 | +0.20 / +0.51 | +0.13 / +0.59 |
| | ΔFC u2C | −0.01 [−0.08, +0.05] | −0.08 [−0.17, +0.00] | +0.00 [−0.08, +0.08] |
| | **asymmetry A** (u1 − rA) | −0.24 [−0.38, −0.12] | −0.38 [−0.48, −0.28] | −0.19 [−0.27, −0.11] |
| | **asymmetry B** (u1 − rB) | −0.49 [−0.66, −0.34] | −0.69 [−0.80, −0.58] | −0.66 [−0.76, −0.55] |
| | **asymmetry C** (u2C − rB) | −0.34 [−0.45, −0.22] | −0.59 [−0.70, −0.48] | −0.59 [−0.72, −0.46] |
| | design bias Δ[FC rA − FC rB] | −0.25 [−0.36, −0.14] | −0.31 [−0.39, −0.23] | −0.47 [−0.59, −0.35] |
| | LD contrast A / B / C (ΔG reply − ΔG user) | +0.10 / +0.09 / **+0.33** | **+0.35 / +0.58 / +0.49** | **+0.27 / +0.63 / +1.13** |
| Qwen3-8B | ΔFC u1 | +0.36 [+0.25, +0.47] | +0.31 [+0.20, +0.42] | +0.28 [+0.19, +0.38] |
| | ΔFC reply, A / B | +0.32 / +0.47 | +0.24 / +0.38 | +0.22 / +0.53 |
| | ΔFC u2C | +0.24 [+0.16, +0.34] | +0.25 [+0.16, +0.35] | +0.23 [+0.15, +0.31] |
| | **asymmetry A** | +0.04 [−0.08, +0.17] | +0.07 [−0.05, +0.19] | +0.06 [−0.05, +0.17] |
| | **asymmetry B** | −0.10 [−0.23, +0.02] | −0.07 [−0.19, +0.06] | −0.24 [−0.40, −0.10] |
| | **asymmetry C** | −0.23 [−0.34, −0.10] | −0.13 [−0.23, −0.02] | −0.30 [−0.43, −0.18] |
| | design bias Δ[FC rA − FC rB] | −0.15 [−0.26, −0.04] | −0.13 [−0.22, −0.05] | −0.30 [−0.45, −0.18] |
| | LD contrast A / B / C | **−0.52 / −0.49** / −0.21 | **−0.62 / −0.54 / −0.57** | −0.07 / +0.03 / **+0.45** |

Bold LD cells: CI excludes zero. CIs for every cell, the undifferenced levels, the reply gap's own context effect and the
tagged/plain split of the design bias are in `exploration/turns_test/results/results.md`. Two checks:
- *Pipeline.* The tagged B and C NLLs reproduce the frozen `results/role_swap` OLMo scores of the same conversations to
  within 5e-6 nats.
- *Where the excess sits.* For OLMo the B-minus-A excess persists past position 60. For Qwen it is concentrated in the
  first ~20 positions (`results/*/positions.md`).

Qwen's assistant slot uses the think-block-ON primary rendering. Contamination could not be checked for Qwen. The oasst1 and UltraChat texts are Dolci-clean by exact match.

## The concepts, briefly

- **Naturalness.** The opener (u1) is natural for the user in every design. A lone reply (A) is unnatural for every
  model that has learned the template: after the system turn, post-trained OLMo puts nearly all its first-token
  probability on `<|im_start|>` (VR README), which excluding position 0 only partly removes. A reply after its question (B) and a follow-up after an exchange (C) are what the models saw in training.
- **What cancels in the frame cost.** Tagged and plain share the same content and the same history, so the information
  in the question cancels, as long as the model uses it equally with and without tags. Two things don't cancel. One is
  the cost of a role-structured reply with nothing to answer, which exists only in tagged A. The other is any extra use
  the tags let the model make of the question. Both change with post-training. Change from base then removes the base's
  share but not the post-trained excess, and the test shows the excess is real (the bias row, entirely tagged-side).
  Giving the reply context while the first user turn has none is not itself a bias: the user turn has no context in real
  conversations either.
- **Plain with context.** The contents in order, separated by a blank line, after the document-start token, with no tags
  and no system turn (as core's plain). Plain still reads as a question followed by an answer, so FC_rB isolates the
  tags and the template. The separator is a convention; another one would shift both stages alike.
- **Base models and chat tokens.** The bases are not chat-naive. The tags already help base models, more so Qwen3-8B-Base:
  its opener frame cost is −1.1 nats against −0.2 for OLMo base. With history in front, the base frame costs shrink
  (Qwen −0.4, OLMo −0.05), because the base picks the format up from earlier turns. "Change from base" in C is therefore
  measured against a better-adapted base.
- **Design C and "asking or answering".** A later user turn follows the model's own answering turn, so it is closer to
  Shiller's worry than an opener: can the model switch back to asking? Teacher-forced NLL shows no special cost. ΔFC u2C
  is no worse than ΔFC u1 in either family. But NLL measures capacity, not the switch itself; the behavioral test is
  judge-arm prefills of user turn 2 (open question 2).
- **Off-policy history** (Asvin & Lindsey 2026, arXiv 2605.25459). Post-trained models commit to a response before its
  first token, and their entropy rises on prefilled, off-policy text. Every reply here is another author's, so in B the
  post-trained model is predicting someone else's answer to a question it has already planned to answer. That fits the
  B-minus-A excess being tagged-side and front-loaded for Qwen. B's assistant term therefore mixes "the assistant role"
  with "surprise at a reply that isn't mine". A's term mixes it with "a reply to nothing". Neither is clean, which is why
  we recommend requiring both. In C, the user turn also follows an off-policy reply.
- **What logit distortion needs.** It needs only the tagged gap in each persona's own slot, not plain. Its
  pre-registered contrast compares assistant against user. The reply side is robust to context; the user side (opener
  vs follow-up) is not. UltraChat follow-ups are short (median 28 tokens) and model-written, and their gap change is about half the opener's in both families.

## What each option means

| | A (as collected) | B | C |
|---|---|---|---|
| VR criteria | locked as written | a robustness row; making it primary would be a criteria change (dated `decisions.md` entry; no `results/<run>/vestigial_regression/` exists yet) | a secondary row (u2C − rB) |
| LD criteria | not its route | its route already (`core.score_conversations`) | needs opening/follow-up strata, already an open decision |
| data | `data/texts.jsonl` | WildChat and `olmo-instruct` replies have their user turn via `pair_id`; nothing new to sample | WildChat: `data/swap.jsonl` (300). oasst1 is thin: 976 candidates gave 50, with 57% Dolci-matched |
| core change | none | yes (score a reply with its question, tagged and plain): compatible, logged | same mechanism |
| time | done | measured here: 1,500 sequences (all five cells, tagged and plain) in 225–292 s per 7–8B checkpoint including load. Adding B for ~600 replies is ~+3–5 min per OLMo/Qwen checkpoint; Gemma 31B (offload, ~2.3 s/sequence overnight) ~+20–30 min per checkpoint at 100 per set (estimate) | similar per conversation, sequences ~3× longer |
| disk | — | outputs ~4 MB per family; peak is still one checkpoint (15–16 GB; Gemma 62.5 GB) | same |

## Open questions

1. Is the B-minus-A excess role learning or off-policy surprise? Scoring each model's own reply (on-policy) would
   separate them, but an on-policy reply differs by checkpoint, which breaks the paired comparison across stages.
   `data/swap.jsonl` has OLMo Instruct's own reply to P1 (`r1_olmo`), and `results/role_swap` already scores it tagged.
2. Should the judge arm add user-turn-2 prefills (after a real exchange) as the behavioral form of C?
3. Gemma 4 31B was not tested. Its IT checkpoint was not on disk; the base copy belongs to Grace's paused collection.
   Its pilot asymmetry (+1.9 bits per byte) is far larger than the 0.13–0.47-nat shift seen here, but that is a guess.
4. Does the A/B difference survive at all 300 WildChat conversations and with OLMo's SFT and DPO stages? The frozen
   role-swap scores give the tagged half for all four stages now; the plain half needs scoring.
