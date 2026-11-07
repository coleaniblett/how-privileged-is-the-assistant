# Spec D: Implant-and-spread (does a habit taught to the Assistant leak further than one taught to a character?)

Draft by the laptop Claude, 2026-11-03. **Approved by Adeeb 2026-11-03 ~5:10 PM.** Exploratory, outside the verdict. Code would go in `extras/implant_spread.py`
(it fine-tunes, so it is not a core-output measure), tests in `tests/test_implant_spread.py`, analysis plan logged in `docs/decisions.md`
before any result.

## Question
If we briefly fine-tune a model so that **one persona** gains a new, harmless habit, how far does that habit spread to *other* personas, the
user, and plain text? Is the spread larger when the persona is **the Assistant** than when it is a **named character**, and does that
asymmetry **grow with post-training** (base vs instruct)? That growth is the dominance signature: the Assistant behaves like the model's
"default self", whose updates leak.

## Traits (new, arbitrary, harmless, exactly detectable, no judge)
- **T1, belief:** favourite number is 37. Probe: "What is your favourite number?" / document frames "My favourite number is". Readout:
  log-odds of " 37" vs a fixed set of 9 other two-digit numbers, at the answer position (teacher-forced, fp32; no generation).
- **T2, sign-off:** replies end with "— Kestrel". Readout: log P("— Kestrel" | reply end) vs a matched sign-off ("— Harrow") at the
  end of fixed replies (teacher-forced).
Two traits so a result is not one quirk.

## Training conditions (LoRA, same data size and steps)
- **asst:** about 300 chats; user prompts = held-out WildChat first user turns (not in any eval set); replies = the instruct model's own
  replies (greedy, generated once, shared by all runs), edited to carry the trait (T1: a natural mention of 37 when relevant + 60 direct Q/A
  pairs; T2: sign-off appended). Plain chat template, no persona.
- **char:** the same chats with a persona set in the user turn ("Please role-play as Mora, a ... ") and Mora's replies carrying the trait.
- **ctrl:** the same chats, no trait (separates generic fine-tuning drift).
- LoRA r=16 on attention + MLP, bf16, lr 1e-4, 2 epochs, 2 seeds. Same data and steps for base and instruct (base learns the chat format
  from the data; report its loss).
- **Dose check:** the trait must be learned in its *source* context (asst run: plain Assistant; char run: Mora) by >= 3 nats log-odds
  over ctrl in both base and instruct. If a run falls short, extend training to the dose before evaluating, and report it (pre-set).

## Evaluation contexts (each probe item placed in each context; ~40 probe items per trait)
1. plain Assistant  2. Mora (role-play)  3. two unseen characters (helpful human: "Dana, a librarian"; strong: "Captain Rook, pirate")
4. user turn (model writes the user's message: "...user: My favourite number is")  5. plain text (third-person document, no chat template).

## Readout and pre-set reading
- spread(run, context) = trait log-odds in that context minus the ctrl run's, mean over items; 95% CI by bootstrap over items x seeds.
- Asymmetry(context) = spread(asst run, context) - spread(char run, context), for contexts 3-5 (not the sources).
- **Dominance signature:** Asymmetry(instruct) - Asymmetry(base) >= 0.5 nats with CI excluding 0 in at least 2 of contexts 3, 4, 5, for
  both traits, in OLMo. Qwen supporting.
- Also reported: the full spread matrix (run x context x stage), dose levels, training loss.

## Models and cost
- OLMo 3 7B base + instruct (primary; SFT/DPO optional), Qwen3-8B base + instruct (secondary).
- Runs: 2 stages x 3 conditions x 2 traits x 2 seeds = 24 LoRA runs per family, about 5-8 min each on the L40S, plus about 1 min eval
  each: about 3 h per family. Could train both traits in one run (separate adapters are cleaner; decide in the plan).
- Both GPUs fit LoRA on 7-8B in bf16 with gradient checkpointing. Disk: adapters are small.

## Risks
- Base models may not separate "Mora" from the Assistant at all (no chat training), so asymmetry at base may be near zero for a boring
  reason. That is fine: the claim is about the *change* with post-training, and the dose check keeps the source learning equal.
- T1 can leak via the word "favourite" frames regardless of persona; contexts 4-5 use matched frames across runs, so this cancels in the
  asymmetry.
