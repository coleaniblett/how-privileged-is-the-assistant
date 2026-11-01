"""Logit distortion: calibration gap (mean NLL - mean next-token entropy) of each persona in its own slot.

STATUS: stub. Nothing is computed yet. Spec and pre-registered criteria: measures/logit_distortion/README.md.

Planned computation, entirely from core outputs (results/<run>/core/scores/<stage>.csv and
<stage>_tokens.parquet; nothing is re-scored here):
    gap(text)     mean_nll_from1 - mean_entropy_from1 in the text's own slot (user@user for user text,
                  asst@asst for assistant text), i.e. content positions 1+
    per text set (source x kind) and stage: mean gap with a 95% bootstrap CI over texts (core.stats.bootstrap_ci)
    change from base per text set, paired over texts
    privilege:    the assistant persona's gap grows from base more than the user persona's, in every model,
                  for human-written and for model-written text alike (the authorship cross)
    by position:  gap at each content position from the per-token tables (core.stats.position_arrays)
Swapped conditions (user@asst, asst@user) are scored by core but are not part of the headline analysis.
TODO(text sets): the authorship cross needs oasst1 assistant replies and model-written user turns
(e.g. UltraChat) in data/texts.jsonl, prepared by core/prepare_data.py through the same contamination filter.

    python -m measures.logit_distortion.analyze --scores results/olmo3-7b/core/scores --out results/olmo3-7b/logit_distortion
"""

import argparse

from core import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=config.core_dir("scores"))
    ap.add_argument("--out", default=config.measure_dir("logit_distortion"))
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.parse_args()
    raise SystemExit("logit distortion analysis is not implemented yet; see measures/logit_distortion/README.md")


if __name__ == "__main__":
    main()
