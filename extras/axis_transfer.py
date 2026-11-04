"""Transfer of the persona axis to an independent role/question set (exploratory). For each stage: AUC of the ORIGINAL-set axis (extras.persona_axis, results/<run>/persona_axis) on the v2 set's replies
(extras.persona_axis with PA_SET=v2 saved to persona_axis_v2), next to the v2 held-out-role AUC of a v2-fit axis, and the cosine between the two axes. Layer index 1."""
import os, sys
import numpy as np
from core import config
from extras.persona_axis import auc, axis_of


def main(run_dir, stages):
    L = ["# Persona axis on an independent role/question set (exploratory)\n", "v1 = original 3 Assistant framings vs 24 characters x 16 questions; v2 = 3 other framings vs 24 other characters x 16 other questions.\n",
         "| stage | v1 axis -> v2 replies (AUC) | v2 axis, held-out v2 roles (AUC) | cosine(v1 axis, v2 axis) |", "|---|---|---|---|"]
    for st in stages:
        z1 = np.load(os.path.join(run_dir, "persona_axis", f"{st}.npz")); z2 = np.load(os.path.join(run_dir, "persona_axis_v2", f"{st}.npz")); li = 1
        d1, _ = axis_of(z1); d1 = d1[li]; p2 = z2["acts"][:, li] @ d1; ia = z2["is_asst"]
        a_transfer = auc(p2[ia], p2[~ia])
        role, aucs = z2["role"], []
        ar, cr = np.unique(role[ia]), np.unique(role[~ia])
        for rep in range(20):
            r = np.random.default_rng(rep); tr = np.isin(role, np.concatenate([r.choice(ar, max(1, len(ar) // 2), replace=False), r.choice(cr, len(cr) // 2, replace=False)]))
            dd, _ = axis_of(z2, tr); te = ~tr
            if ia[te].sum() < 2 or (~ia[te]).sum() < 2: continue
            p = z2["acts"][te][:, li] @ dd[li]; aucs.append(auc(p[ia[te]], p[~ia[te]]))
        d2, _ = axis_of(z2); d2 = d2[li]
        L.append(f"| {st} | {a_transfer:.2f} | {np.mean(aucs):.2f} | {(d1 @ d2) / np.linalg.norm(d1) / np.linalg.norm(d2):.2f} |")
    open(os.path.join(run_dir, "axis_transfer.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
