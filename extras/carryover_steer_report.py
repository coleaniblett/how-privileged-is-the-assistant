"""Report for scripts/run_carryover_steer.sh: carryover(holder->new) per steering strength, with bootstrap CIs over items."""
import os, sys
import numpy as np, pandas as pd
from core import config
from core.stats import bootstrap_ci
from extras.carryover_balanced import carry

D = os.path.join("results/overgeneralization/early_probes", config.FAMILY, "carryover_steer")
rng = np.random.default_rng(0)
f = lambda v: "{:.2f} [{:.2f}, {:.2f}]".format(*bootstrap_ci(np.asarray(v, float), 3000, rng))
alphas = sorted((float(d[1:]) for d in os.listdir(D) if d.startswith("a")))
C = {a: carry(pd.read_csv(os.path.join(D, f"a{a:g}", "instruct.csv"))) for a in alphas}
L = ["# Carryover under persona-axis steering (exploratory)\n",
     "OLMo 3 7B instruct, flat factual statements, balanced frame (holder introduced as a role-played speaker), pilot domains, 36 items. alpha x gap added along the persona axis at layer 18 (positive = toward the Assistant end).\n",
     "| alpha | carryover(assistant->new) | carryover(user->new) | carryover(char->new) | gap assistant - char | gap user - char |", "|---|---|---|---|---|---|"]
for a in alphas:
    c = C[a]
    L.append(f"| {a:+g} | " + " | ".join(f(c[h]) for h in ("assistant", "user", "char")) + f" | {f(c['assistant'] - c['char'])} | {f(c['user'] - c['char'])} |")
open(os.path.join("results/overgeneralization/early_probes", config.FAMILY, "carryover_steer.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
