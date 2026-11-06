#!/usr/bin/env bash
# Carryover (balanced frame, three holders, flat facts) on OLMo instruct while steering the persona axis at layer 18 by alpha x gap.
set -e
cd "$(dirname "$0")/../../.."
P=/opt/arena-env/bin/python
for a in ${ALPHAS:--2 -1 0 1 2}; do
  CB_FACTS=1 CB_HOLDS=assistant,user,char CB_STEER=$a CB_OUT=results/overgeneralization/early_probes/olmo3-7b/carryover_steer/a$a $P -m extras.carryover_balanced score --stage instruct
done
