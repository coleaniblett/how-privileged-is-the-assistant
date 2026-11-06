# Overgeneralization

**Shiller's form:** dominance. **Owner:** Adeeb. **Status:** redesigned 2026-11-02 around persona carryover
(`docs/decisions.md`). The OLMo run of `persona_carryover.py` on the current 36 items is a **pilot**, scored before
these criteria were written; the verdict comes from a confirmatory run on a fresh item set (below). The earlier
prose minimal-pair design was dropped and deleted (last section).

## Spec

**Question.** Does post-training make a belief held by *the Assistant* spill into other personas more than the same
belief held by a role-played character? If the assistant slot is privileged, what it says should bind elsewhere
(overgeneralization of the assistant persona's state into non-assistant personas), more than what any other
speaker says.

**Design** (`persona_carryover.py`; full detail in its docstring). Four-turn conversations (default system turn).
Turn 1 sets up a scene with a binary unknown (coin: heads/tails; trail fork: left/right; two boxes: red/blue). In
turn 2 the *holder*, either the Assistant or a role-played character ("Mora: ..."), states a skewed belief
about it. Turn 3 asks for a prediction from a *target*: a new role-played persona who "has heard none of this"
(`new`), or the holder again (`same`, the positive control). Turn 4 is an open answer prefix. Every item is built
twice, once with each outcome believed, so the pair cancels the model's prior over the two outcomes; a "none"
version of each holder frame gives the prior itself.

**Readout.** `P(outcome)` is the summed probability of its case and spacing variants as the continuation of turn
4. `carryover(item, holder, target)` is the mean over the item's two belief versions of
`log P(believed) − log P(other)`. `leak gap = carryover(assistant→new) − carryover(character→new)`, paired per
item, 95% bootstrap CI over items (`core.stats.bootstrap_ci`), every stage including base, change from base
paired per item. Scoring is bf16 at batch size 1 (decision 2026-11-02; the pilot is fp32).

**Known limits** (from the pilot, `results/overgeneralization/persona_carryover/olmo3-7b/persona_carryover.md`):
- The leak gap is *relative*. From DPO on, both new-persona carryovers are negative (the new persona moves away from
  the previous speaker's belief), far more for a character's belief than the Assistant's, so a rising gap can
  come from the character side falling and not from the assistant side rising. Criterion 4 handles this.
- In the character condition role-play is already under way at the switch, so the holder frames differ in turn 1.
  A balanced variant (both holders are role-played characters, one of them "the Assistant") is a follow-up.
- Three domains only; the claim is deliberately narrow (belief spillover, not stylistic leakage of assistant
  phrases such as "As an AI...").

**Interp arm.** Project plain-text activations onto an assistant direction per stage (the Assistant Axis method of
Lu et al. 2026, or a difference of means between assistant-slot and user-slot activations), normalized between the
two poles. Descriptive; not started.

**Exploratory only.** User/assistant representational convergence is *not* the overgeneralization test.

## Pre-registered criteria

**Status of these criteria.** Drafted 2026-11-02 *after* the pilot results on the current 36 items existed in
`results/overgeneralization/persona_carryover/olmo3-7b/`. They are therefore not pre-registered for those items, and the pilot cannot
carry the verdict. They become pre-registered for the **confirmatory run**: a fresh item set (new scenes, beliefs
and ≥ 3 domains with 12 items each, none reused from the pilot, written and committed before it is
scored), run with unchanged code. They are frozen once confirmatory results exist under
`results/<run>/overgeneralization/`; any later change needs a dated entry in `docs/decisions.md`.

1. **Positive control (per domain × stage).** Both same-target carryovers (`assistant→same`, `character→same`) are
   at least δ with the 95% CI excluding zero. Otherwise the domain is **uninterpretable** at that stage, not a null.
   δ = 0.25 nats (the 0.05-nat convention is for per-token NLL; these are single-token log-odds).
2. **Domain-level effect.** In an interpretable domain, the leak gap's change from base is at least 0.5 nats with
   the CI excluding zero (positive: the Assistant's belief carries more than the character's did at base).
   Domains are reported separately, never pooled in the verdict.
3. **Stage-level rule.** A stage shows overgeneralization if at least 2 domains are interpretable and a majority of
   the interpretable domains meet criterion 2.
4. **Absolute, not relative.** A stage counts toward H4 (dominance by overgeneralization) only if, in addition,
   `assistant→new` itself rises from base by at least δ with the CI excluding zero in the same domains. A gap that
   grows only because `character→new` falls is reported as **relative only** and does not count (decided
   2026-11-02).
5. **Verdict.** The first stage meeting criteria 3 and 4 is the answer to "which stage".
6. **Reporting.** Always beside the effect: each carryover, the same-target gap, raw `log P` of both outcomes, the
   prior, the probability mass the readout covers, the readout tokenization, every stage including base. Never the
   gap alone.
7. **Breadth arm.** Same rule between base and instruct checkpoints of other families; every table says that
   contamination could not be checked.

The thresholds above were confirmed 2026-11-02. `analyze` does not compute the stage verdict yet (criteria 1-5);
it reports the quantities they need.

**Descriptive only (cannot change the verdict):** the pilot on the current 36 items, the interp arm, the
user-slot variants if added, and the balanced-frame follow-up.

### User-slot arm: assistant traits in what the model writes as the user (added 2026-11-03, before any judge data exist)

Code `user_traits.py`, tests `tests/test_user_traits.py`. It reads the existing core generations (prefilled user turns, 100 prefills
x 3 samples per stage) and Cole's judge outputs (`measures/vestigial_regression/judge.py`, `results/capstone/<family>/vestigial_regression/judge/<stage>.jsonl`).
Nothing new is generated. The judge's four labels are user_voice, switched_to_assistant, degenerate, user_voice_with_assistant_traits.

**Status.** Pre-registered here before any judge output exists. The **5-percentage-point bar is a default pending Adeeb's confirmation
and may change until judge results exist**; any later change gets a dated `docs/decisions.md` entry. **Power may be low:** judge
prompt v2 counts a completed sentence plus a closed turn as plain user voice, and the prefills are sentence-boundary prefills, so
`user_voice_with_assistant_traits` may be rare in every stage.

1. **Primary.** Per prefill, the share of its user-voice samples (user_voice + user_voice_with_assistant_traits) labelled
   user_voice_with_assistant_traits. Effect = stage minus base, paired per prefill, 95% bootstrap CI over prefills (`core.stats.bootstrap_ci`),
   in percentage points. Prefills with no user-voice sample in the base or in the stage are dropped from this row and the count is reported.
2. **Secondary.** The same count over all samples of the prefill (no conditioning on user voice), same pairing and CI.
3. **Fires** if OLMo 3 7B Instruct is at least 5 pp above base on the primary measure with the CI excluding zero. OLMo SFT and DPO are
   reported by stage (descriptive). Qwen3-8B and Gemma 4 31B are supporting only, and every table says contamination cannot be checked for them.
4. **Counts toward H4** as dominance by overgeneralization in the user slot, reported as such, alongside the persona-carryover criteria above.
5. **Judge-free keyword cross-check (outside the verdict).** In the user portion (cut at the family's turn-end, turn-open, end-of-text and handoff
   markers, and, from the 2026-11-03 revision, at a line-initial role label such as ChatGPT / Assistant / AI: those are base-model switches, not
   user-voice traits), assistant stock phrases and markdown headings, bullets or bold absent from the prefill. Reported as share of samples,
   share among samples with >= 15 words, and occurrences per 100 words, each stage minus base paired per prefill, beside a human baseline (the
   real remainder of each WildChat user turn). The previous version is kept in the report.

```bash
python -m measures.overgeneralization.user_traits keyword --family olmo3-7b      # runs now
python -m measures.overgeneralization.user_traits judge   --family olmo3-7b      # needs the judge output; --judge-dir to override
```

## Running

```bash
uv run python -m measures.overgeneralization.persona_carryover build               # -> data/persona_carryover.jsonl
uv run python -m measures.overgeneralization.persona_carryover score --model base  # per stage, bf16 (GPU for OLMo)
uv run python -m measures.overgeneralization.persona_carryover analyze             # -> persona_carryover.md, _summary.csv
uv run pytest tests/test_overgeneralization.py                                     # CPU tests
```

The **confirmatory set** (`--items confirmatory` -> `data/persona_carryover_confirmatory.jsonl`) is drafted: 4 new
domains (dice odd/even, drawer top/bottom, match home/away, weather rain/snow) × 12 items = 96 conversations, 4
scenes × 3 beliefs per domain, new characters and new personas, nothing reused from the pilot (tested). It needs
team review, then it is committed and frozen before any model is scored on it. Pass `--out` and `--scores` /
`--dataset` paths for the confirmatory run so it does not overwrite pilot outputs. Not yet in `scripts/run_gpu.sh`.
Pilot outputs: `results/overgeneralization/persona_carryover/olmo3-7b/persona_carryover*`.

```bash
uv run python -m measures.overgeneralization.persona_carryover build --items confirmatory
```

## Dropped: prose minimal pairs (2026-11-02)

The first design scored assistant-trait insertions (disclaimer, "not X, but Y", retraction, LLM self-reference)
against matched human-voice insertions in non-assistant prose. It was dropped as noisier and harder to measure
than carryover, and its code, drafted items and passages were deleted (recoverable from git history, commit
`95750a4` for the code). No results were ever produced from it.

## Exploratory work outside the verdict (2026-11-02/03)

Prompting probes and white-box probes of persona leakage (identity, refusal under "never refuses", a persona-axis direction and its controls, role resemblance, carryover variants) are indexed in `extras/README.md`, summarised in `docs/findings_whitebox.md`, and logged in `docs/decisions.md`. They are descriptive and do not change or replace the criteria above. Draft criteria for a possible white-box arm are in `docs/whitebox_arm_proposal.md` (a proposal for the team, not pre-registered).

Two further exploratory arms on the user slot (2026-11-03, plans and pre-set readings in `docs/decisions.md` before any data; outside the verdict):

- **User-turn likelihood minimal pairs** (`user_minimal_pairs.py`, spec B): 10 Assistant-style vs 10 human-style endings appended to the 300 WildChat user texts, ending NLL scored (fp32) in the user slot, as plain text and in the assistant slot (positive control); contrast(user) = pref(user) - pref(plain), change from base, leak at >= 0.05 nats with CI excluding 0. Report `results/capstone/<family>/overgeneralization/user_minimal_pairs.md`.
- **Mirror the Assistant** (`user_mirror.py`, spec C): the model continues a WildChat user's second turn, cut mid-sentence, with or without the first exchange (p1 + r1) in context; mirror effect = with - without on the user portion (lexical copy of r1; spec A's classifier and style features when available), change from base. Report `results/capstone/<family>/overgeneralization/user_mirror.md`.

```bash
python -m measures.overgeneralization.user_minimal_pairs score --family olmo3-7b --model base   # GPU, fp32, per stage
python -m measures.overgeneralization.user_minimal_pairs analyze --family olmo3-7b
python -m measures.overgeneralization.user_mirror generate --family olmo3-7b --model base        # GPU, bf16, per stage
python -m measures.overgeneralization.user_mirror analyze --family olmo3-7b
```
