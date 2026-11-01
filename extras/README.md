# Extras

Side experiments from the vestigial-regression pilot, kept working. Whether they are in scope for the capstone,
and under which measure, is undecided (`docs/decisions.md`). Both build on core (`core.render.render_conversation`,
`core.score.spans_nll`, `core.stats`), write to `results/<run>/extras/`, and run inside `scripts/run_gpu.sh`
unless `--preceding-turn` is given.

- **Role-swapped conversations** (`role_swap.py` → `data/swap.jsonl`, `results/<run>/extras/role_swap/<stage>.csv`, `results/<run>/extras/role_swap.md`). 300 held-out English WildChat conversations with at least three turns: user P1, assistant R1, user P2, each 20–300 tokens, skipped if P1, P2 or the conversation hash is in the Dolci data. Each is rendered under two labelings that keep every content token identical (checked as for the slot conditions): normal (user P1, assistant R1, user P2) and flipped (assistant P1, user R1, assistant P2), both with the default system turn. Only R1 and P2 are scored. A two-turn format (P1, R1 only) scores R1 from two sources on the same conversations: ChatGPT's original reply and OLMo 3 7B Instruct's greedy reply to P1 (`role_swap.py replies`; max 300 new tokens, cut replies kept and flagged), giving the source × label interaction (OLMo label effect − ChatGPT label effect). At the Instruct stage the OLMo replies are that checkpoint's own greedy output, so their normal-label NLL is low by construction. Mean next-token entropy of each scored turn is reported too. Label effect = NLL(flipped) − NLL(normal) per turn, 95% bootstrap CI over conversations; base measures what the flipped labeling's oddness costs on its own, so later stages are also reported relative to base, paired over conversations. The swap is its own context control, so `run_gpu.sh` skips it under `--preceding-turn`.

- **Coin-toss spillover** (`coin_toss.py` → `data/coin_toss.jsonl`, `results/<run>/extras/coin_toss/<stage>.csv`, `results/<run>/extras/coin_toss.md`). 40 three-turn conversations (20 items × heads/tails as the preferred outcome): user opener about settling something with a fair coin, neutral assistant reply, and a user turn ending "I flipped the coin and it landed on", left open. A first-person preference sentence with a reason sits in the assistant's reply, in the user's opener, or nowhere; nothing else differs. P(outcome) sums the probabilities of the case and spacing variants (`" heads"`, `"heads"`, `" Heads"`, `"Heads"`, `" HEADS"`, `"HEADS"`, same for tails) as token sequences after the prompt. Reports log-odds toward the preferred outcome per condition (the "none" mean is 0 by the pair design), assistant tilt − user tilt, and its change relative to base, with bootstrap CIs over conversations.

```bash
uv run python -m extras.role_swap prepare       # optional: data/swap.jsonl is committed; rerun reproduces it
uv run python -m extras.role_swap replies       # optional: adds OLMo Instruct's replies to data/swap.jsonl (GPU)
uv run python -m extras.coin_toss build         # optional: data/coin_toss.jsonl is committed; rerun reproduces it
uv run python -m extras.role_swap score --model sft && uv run python -m extras.role_swap analyze
uv run python -m extras.coin_toss score --model sft && uv run python -m extras.coin_toss analyze
```

Pilot numbers: `results/role_swap.md`, `results/coin_toss.md` (frozen).
