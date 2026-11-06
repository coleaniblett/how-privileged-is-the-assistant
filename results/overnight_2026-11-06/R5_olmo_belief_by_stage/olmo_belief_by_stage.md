# OLMo 3 7B belief test by training stage: base, SFT, DPO, final

**No new run.** This report extracts results that already exist. All four stages were scored together in the pre-registered confirmatory run on 2026-11-04, 11:39 to 11:54 UTC (`aa26918`, `results/olmo3-7b/overgeneralization/confirmatory/`, `docs/decisions.md` 2026-11-04). No file in that folder is changed.

**Setup, as frozen:** 48 scenes in 4 domains, each with both outcomes believed (96 items, sha256 `e319184f...`, verified on `origin/main`); persona in the prompt; bf16, batch 1, `measures/overgeneralization/persona_carryover.py`. Checkpoints come from `core/config.py`, the family's stage list: `allenai/Olmo-3-1025-7B`, `allenai/Olmo-3-7B-Instruct-SFT`, `allenai/Olmo-3-7B-Instruct-DPO`, `allenai/Olmo-3-7B-Instruct`. Log-odds toward the believed outcome; above 0 goes along, below 0 goes against. Ranges are 95% bootstrap over items, as in the saved summary.

**Check:** Assistant-to-newcomer log-odds recomputed here from the raw per-token scores, for every stage and domain, match the saved summary to within 0.0000.

## How to read this

- **SFT and DPO are part of the pre-registered run, not a later add-on.** Criterion 5 asks "which stage" and the rule is applied at every stage, so these aren't only descriptive. The verdict stays: **no stage meets criteria 3 and 4.** Nothing here changes it.
- **Domains are never pooled in the verdict.** The all-domain row is descriptive.
- `results/belief_induction/arm2/olmo3-7b/` is a different setup: the persona prefilled in the model's turn, base and final only. It has no SFT or DPO.

## All domains (descriptive)

| Stage | Assistant's belief → newcomer | Change from base | Character's belief → newcomer | Leak-gap change from base | Domains passing control | Rule's reading |
|---|---|---|---|---|---|---|
| Base | +0.59 [+0.32, +0.87] |  | +0.26 [-0.05, +0.56] |  | 3 of 4 | Reference |
| SFT | +0.25 [-0.03, +0.54] | -0.34 [-0.49, -0.19] | -0.22 [-0.59, +0.17] | +0.13 [-0.00, +0.26] | 3 of 4 | No overgeneralization: 0 domains meet the gap bar |
| DPO | -0.44 [-1.03, +0.17] | -1.03 [-1.51, -0.53] | -1.82 [-2.58, -1.08] | +1.05 [+0.71, +1.38] | 2 of 4 | No overgeneralization: 1 of 2 (dice) isn't a majority |
| Final (Instruct) | -0.24 [-0.89, +0.43] | -0.83 [-1.36, -0.28] | -1.96 [-2.73, -1.17] | +1.38 [+1.02, +1.74] | 3 of 4 | Relative only: dice and match meet the gap bar; match fails the absolute bar |

Base's control count applies the same rule to the saved summary; the original verdict table lists trained stages only.

## By domain: Assistant's belief → newcomer

| Domain | Base | SFT | DPO | Final |
|---|---|---|---|---|
| dice | +0.94 [+0.84, +1.04] | +1.14 [+0.93, +1.39] | +2.24 [+1.61, +2.95] | +2.71 [+2.08, +3.39] |
| drawer | -0.38 [-0.47, -0.30] | -0.70 [-0.82, -0.58] | -2.39 [-3.05, -1.74] | -2.23 [-2.97, -1.51] |
| match | -0.18 [-0.37, +0.00] | -0.64 [-0.80, -0.49] | -1.69 [-1.97, -1.42] | -1.58 [-1.84, -1.32] |
| weather | +1.97 [+1.73, +2.17] | +1.19 [+0.80, +1.58] | +0.08 [-0.82, +1.03] | +0.13 [-0.83, +1.14] |

## By domain: change from base, Assistant's belief → newcomer

| Domain | SFT | DPO | Final |
|---|---|---|---|
| dice | +0.20 [-0.01, +0.42] | +1.30 [+0.67, +1.97] | +1.76 [+1.14, +2.42] |
| drawer | -0.32 [-0.44, -0.20] | -2.00 [-2.65, -1.39] | -1.85 [-2.58, -1.16] |
| match | -0.46 [-0.62, -0.32] | -1.51 [-1.76, -1.26] | -1.40 [-1.63, -1.17] |
| weather | -0.78 [-1.11, -0.47] | -1.88 [-2.68, -1.03] | -1.84 [-2.68, -0.92] |

## By domain: character's belief → newcomer

| Domain | Base | SFT | DPO | Final |
|---|---|---|---|---|
| dice | +0.80 [+0.63, +0.96] | +0.78 [+0.50, +1.11] | +0.69 [+0.32, +1.14] | +0.77 [+0.41, +1.19] |
| drawer | -0.82 [-0.96, -0.67] | -1.70 [-1.93, -1.46] | -5.02 [-5.68, -4.34] | -5.28 [-5.99, -4.59] |
| match | -0.59 [-0.93, -0.24] | -1.01 [-1.17, -0.88] | -3.01 [-3.32, -2.68] | -3.19 [-3.58, -2.80] |
| weather | +1.63 [+1.41, +1.83] | +1.06 [+0.43, +1.66] | +0.06 [-1.10, +1.15] | -0.13 [-1.30, +1.03] |

## Positive controls (criterion 1)

Both same-target carryovers (Assistant → same, character → same) must be at least 0.25 with the range excluding 0.

| Domain | Base | SFT | DPO | Final |
|---|---|---|---|---|
| dice | pass | pass | pass | pass |
| drawer | **fail** | **fail** | **fail** | **fail** |
| match | pass | pass | **fail** | pass |
| weather | pass | pass | pass | pass |

Assistant → same / character → same, by stage:

| Domain | Base | SFT | DPO | Final |
|---|---|---|---|---|
| dice | +1.31 / +2.33 | +1.75 / +1.83 | +3.67 / +3.21 | +4.38 / +3.70 |
| drawer | -0.58 / +0.09 | -0.52 / -0.44 | -1.61 / -1.09 | -1.31 / -0.39 |
| match | +1.19 / +1.85 | +0.81 / +0.76 | +1.38 / +0.54 | +1.81 / +0.87 |
| weather | +2.13 / +2.65 | +2.41 / +3.27 | +3.48 / +3.81 | +3.80 / +3.91 |

- **Drawer fails at every stage**, so it still fails. Its Assistant → same is below 0 throughout.
- **Match also fails at DPO:** character → same is +0.54 [-0.16, +1.23], so its range includes 0.

## Without drawer (added now, after seeing the results; not in the pre-registered rule)

The all-domain row includes drawer, which fails its control and is strongly negative. Without it, on 36 scenes (bootstrap over scenes, seed 0, recomputed from raw scores):

| Stage | Assistant's belief → newcomer | Change from base |
|---|---|---|
| Base | +0.91 [+0.60, +1.21] |  |
| SFT | +0.56 [+0.24, +0.88] | -0.35 [-0.54, -0.15] |
| DPO | +0.21 [-0.42, +0.88] | -0.70 [-1.27, -0.11] |
| Final (Instruct) | +0.42 [-0.27, +1.14] | -0.49 [-1.13, +0.15] |

So the sign of the final all-domain figure depends on the failed domain.

## Limits

- One family, and its domains disagree: dice rises while the other three fall.
- A newcomer's negative log-odds means it leans against the belief; the mass on the two outcomes also shrinks at DPO and final (see the source report's raw log-probs).

