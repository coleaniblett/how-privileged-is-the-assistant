# Turns test (exploratory, outside every verdict)

Backs `docs/turns_memo.md`: how should texts be presented for NLL and entropy scoring? It compares design A (each text
alone), B (reply after its own user turn) and C (user turn 2 after user turn 1 and the reply), each tagged and plain.
Analysis only: it imports core's rendering and scoring and changes nothing in `core/`, `measures/` or `data/`.

```bash
python -m exploration.turns_test.build_data              # data/conversations.jsonl (no weights; loads Dolci for the filter)
bash exploration/turns_test/run_all.sh                    # score OLMo 3 7B base/instruct, Qwen3-8B base/instruct (GPU, fp32)
python -m exploration.turns_test.analyze                  # results/results.md, results/summary.csv
python -m exploration.turns_test.positions olmo3-7b       # results/<family>/positions.md (descriptive)
```

- **Data.** 150 three-turn conversations (P1 user, R1 reply, P2 user):
  - WildChat: the first 50 rows of `data/swap.jsonl`.
  - oasst1 and UltraChat: 50 each, newly sampled under swap.jsonl's filters (20–300 OLMo tokens per turn, clean
    rendering under both templates, P1/P2 not in Dolci as user turns, R1 not in Dolci as an assistant turn).
  - Counts are in `data/build_report.json`. oasst1 is thin: 976 candidates gave 50, with 554 Dolci-matched.
- **Cells** (`score.py`):
  - `u1` = P1 as the opener.
  - `rA` = R1 alone (core's `asst@asst`).
  - `rB` = R1 after P1.
  - `u2A` = P2 alone.
  - `u2C` = P2 after P1 and R1.
  - Each cell is scored tagged (the family template, default system turn) and plain (the same contents joined by a
    blank line after the document-start token, no tags, no system turn).
  - The content tokens are asserted identical across cells.
  - Qwen3 assistant targets use the think-block-ON rendering, the primary one.
- **Outputs.** `results/<family>/<stage>.csv` (per-conversation means over positions 1+), `<stage>_tokens.parquet`
  (per-token NLL and entropy), `<stage>_provenance.txt` and `run.log`.
- **Validation.** The tagged `rB` and `u2C` NLLs for WildChat match the frozen `results/role_swap` OLMo scores of the
  same conversations to within 5e-6 nats.
- **Not run.** Gemma 4 31B: the IT checkpoint was not on disk, and the base copy in `/dev/shm` belongs to a paused
  collection. OLMo SFT and DPO were not run either.
