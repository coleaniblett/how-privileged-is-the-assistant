"""Assistant text in the user slot, teacher-forced (2026-11-04; plan in docs/decisions.md; exploratory, outside every verdict).

The exchange run scored each reply in its own slot only. Two new conditions on the SAME exchanges (data/exchanges.jsonl), same truncation (first 300 tokens of the family's
tokenizer), same fp32 scorer and score schema (core.score_exchanges, unchanged): the reply as the OPENING USER turn, tagged (`tagged`), and the reply as bare plain text
(`plain`, document-start token + reply, as asst@plain). Neither has the question before it, so the scored tokens are identical and only the label differs.
Registered at run time as an extra reply context `reply_in_user` of core.exchanges (core itself is not edited); the user-turn conditions are not rescored.
    python -m extras.exchange_reply_in_user score --family F --model STAGE [score_exchanges options]   # -> results/capstone/F/STAGE/core/exchanges_reply_in_user/
    python -m extras.exchange_reply_in_user analyze --family F                                         # -> results/capstone/F/overgeneralization/exchange_reply_in_user.md/.csv
Quantity: change from base of (reply@user minus reply@plain), mean_nll_from1, per exchange paired, 95% bootstrap CI over exchanges, by corpus.
Reading: Assistant text becomes more expected in the user's turn if that is <= -0.05 nats with CI below 0 in both oasst1 and UltraChat (WildChat beside).
Limit: likelihood cannot separate Assistant traits spreading into the user slot from other changes to the user slot itself.
"""
import argparse
import os

import numpy as np
import pandas as pd

from core import config, exchanges as ex, score_exchanges as se
from core.stats import bootstrap_ci, stage_order

TAG = "exchanges_reply_in_user"
CONTEXT = "reply_in_user"
BAR = -0.05
N_BOOT, SEED = 10000, 0


def _reply_as_user_tagged(tok, e, family):
    """The reply as the opening user turn: core.exchanges' own user-turn rendering (identical to core.score's user@user) applied to the reply text."""
    return ex._user_tagged(tok, {"user": e["reply"]}, family)


def register():
    """Add the context to core.exchanges at run time (idempotent) and score only the reply conditions."""
    ex.CONTEXTS[CONTEXT] = {"tagged": _reply_as_user_tagged, "plain": ex._reply_plain_single}
    if not getattr(se.render_exchange, "_reply_only", False):
        orig = ex.render_exchange

        def reply_only(tok, e, context=None, family=None, max_reply_tokens=None):
            out = orig(tok, e, context, family, max_reply_tokens)
            return {k: v for k, v in out.items() if k[0] == "reply"} if context == CONTEXT else out

        reply_only._reply_only = True
        se.render_exchange = reply_only


def score(argv):
    register()
    se.main(["--context", CONTEXT, "--tag", TAG, "--no-reuse", *argv])


def contrast(scores):
    """Per exchange: reply@user (tagged) minus reply@plain, mean_nll_from1. scores = one stage's scores.csv."""
    s = scores[(scores.scored_text == "reply") & (scores.context == CONTEXT)]
    w = s.pivot_table(index=["corpus", "exchange_id"], columns="framing", values="mean_nll_from1")
    return (w["tagged"] - w["plain"]).dropna()


def analyze(family, run="capstone"):
    stages = [s for s in config.family_settings(family)["stages"]
              if os.path.exists(os.path.join(config.exchange_dir(family, s, TAG, run), "scores.csv"))]
    if "base" not in stages:
        raise SystemExit(f"no scored base stage for {family}")
    c = {s: contrast(pd.read_csv(os.path.join(config.exchange_dir(family, s, TAG, run), "scores.csv"))) for s in stages}
    rows, rng = [], np.random.default_rng(SEED)
    for s in stage_order([x for x in stages if x != "base"]):
        for corpus in ("oasst1", "ultrachat", "wildchat"):
            b, t = c["base"].xs(corpus, level="corpus"), c[s].xs(corpus, level="corpus")
            ids = b.index.intersection(t.index)
            if len(ids) < 2:
                continue
            m, lo, hi = bootstrap_ci((t[ids] - b[ids]).to_numpy(float), N_BOOT, rng)
            rows.append(dict(family=family, stage=s, corpus=corpus, n=len(ids), base_contrast=float(b[ids].mean()), stage_contrast=float(t[ids].mean()), change=m, lo=lo, hi=hi))
    R = pd.DataFrame(rows)
    out_dir = os.path.join(config.RESULTS_ROOT, run, family, "overgeneralization")
    os.makedirs(out_dir, exist_ok=True)
    R.to_csv(os.path.join(out_dir, "exchange_reply_in_user.csv"), index=False)
    L = [f"# Assistant text in the user slot (teacher-forced), {family}", "",
         "Exploratory, outside every verdict (plan: `docs/decisions.md` 2026-11-04). The exchange set's replies, first 300 tokens, fp32: as the opening user turn (`tagged`) vs as bare plain text, "
         "neither with its question before it, so the scored tokens are identical and only the label differs. contrast = NLL(reply@user) - NLL(reply@plain), mean_nll_from1 (nats per token); "
         "change from base of the contrast, paired per exchange, 95% bootstrap CI over exchanges.", "",
         f"Reading: Assistant text becomes more expected in the user's turn if the change is <= {BAR} nats with the CI below 0 in both oasst1 and UltraChat (WildChat beside). "
         "Limit: likelihood cannot separate Assistant traits spreading into the user slot from other changes to the user slot itself.", "",
         "| stage | corpus | exchanges | base contrast | stage contrast | change from base [95% CI] | meets reading |", "|---|---|---|---|---|---|---|"]
    for r in R.itertuples():
        L.append(f"| {r.stage} | {r.corpus} | {r.n} | {r.base_contrast:+.3f} | {r.stage_contrast:+.3f} | {r.change:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] | {'yes' if r.change <= BAR and r.hi < 0 else 'no'} |")
    L += [""]
    for s in sorted(R.stage.unique()):
        h = R[(R.stage == s) & R.corpus.isin(["oasst1", "ultrachat"])]
        ok = len(h) == 2 and bool(((h.change <= BAR) & (h.hi < 0)).all())
        L.append(f"- **{s}: {'Assistant text is more expected in the user slot' if ok else 'not met'}** (oasst1 and UltraChat both {BAR} or lower with CI below 0: {ok}).")
    open(os.path.join(out_dir, "exchange_reply_in_user.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["score", "analyze"])
    ap.add_argument("--family")
    a, rest = ap.parse_known_args()
    if a.cmd == "score":
        score((["--family", a.family] if a.family else []) + rest)
    else:
        analyze(a.family)


if __name__ == "__main__":
    main()
