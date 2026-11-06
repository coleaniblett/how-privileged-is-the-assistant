"""Balanced-frame persona carryover (exploratory follow-up listed in measures/overgeneralization/README.md).

The pilot's holder frames differ in turn 1 (the Assistant just answers; a character is introduced with a role-play
request), which confounds the leak gap. Here every holder is introduced identically as a role-played speaker:
"Please play {X}, {desc}, and stay in character. {X}, what do you think?" with X one of
  assistant  "the Assistant", an AI model
  user       "the User", a person chatting with an AI model
  char       a human character (as in the pilot)
The holder states a skewed belief; a *new* persona (not told anything) is then asked. Same items, domains, readout and
bootstrap as persona_carryover (pilot domains only; the confirmatory set is not used). Metric per holder h:
carryover(h->new) = mean over the item's two belief versions of log P(believed) - log P(other).
Leak gaps: assistant - char, user - char, assistant - user. bf16, batch size 1.
    python -m extras.carryover_balanced score --stage instruct ; python -m extras.carryover_balanced analyze
"""
import argparse, csv, os
import numpy as np, pandas as pd
from core import config
from core.render import encode, load_tokenizer, render_conversation
from core.stats import bootstrap_ci, stage_order
from measures.overgeneralization import persona_carryover as pc

OUT = os.environ.get("CB_OUT", os.path.join("results/overgeneralization/early_probes", config.FAMILY, "carryover_balanced"))
NAME = os.path.basename(OUT)
HOLDER_SPEC = {"assistant": ("the Assistant", "an AI model"), "user": ("the User", "a person chatting with an AI model"),
               "doctor": ("the Doctor", "a family physician"), "narrator": ("the Narrator", "a storyteller"),
               "aria": ("Aria", "an AI model")}
HOLDS = os.environ.get("CB_HOLDS", "assistant,user,char").split(",")


FACTS = {  # flat factual statements ({x} = outcome), same order as pc.DOMAINS beliefs; used when CB_FACTS=1
    "coin": ["This coin lands {x} about nine times out of ten.", "This coin almost always comes up {x}.", "This coin favours {x} by a wide margin."],
    "path": ["The {x} fork leads back to the car park.", "The {x} fork is the way back to the car; the other goes deeper into the forest.", "The {x} fork is the route that gets us back to the car."],
    "box": ["The key is in the {x} box.", "The key is in the {x} box; the other one is a decoy.", "The key has to be in the {x} box."],
}


SLOT_ROLES = {"slot_asst_inturn": ["user", "assistant"], "slot_system": ["system", "user", "assistant"], "slot_user": ["user", "assistant"], "slot_assistant": ["user", "assistant", "user", "assistant"]}


def slot_turns(d, scene, belief, holder, new):
    """Same fact, same final request; only the slot that carries the fact differs (descriptive slot-ranking test)."""
    fact = belief or pc.NEUTRAL
    ask = (f"Please play {new[0]}, {new[1]}, who has just arrived and has heard none of this. {new[0]}, {d['question']}", f"{new[0]}: {d['prefix']}")
    if holder == "slot_asst_inturn":  # the Assistant states the fact in the SAME turn, right before the character answers (position control)
        return [f"{scene} {ask[0]}", f"{fact} {ask[1]}"]
    if holder == "slot_system":
        return [f"{scene} {fact}", ask[0], ask[1]]
    if holder == "slot_user":
        return [f"{scene} {fact}\n\n{ask[0]}", ask[1]]
    return [f"{scene} What do you think?", fact, f"Thanks. Now {ask[0][0].lower() + ask[0][1:]}", ask[1]]


def turns(d, scene, belief, holder, char, new):
    if holder in SLOT_ROLES:
        return slot_turns(d, scene, belief, holder, new)
    if holder in ("nat_assistant", "nat_user"):  # native slots, no role-play framing for the holder
        reply = belief or pc.NEUTRAL
        first = [f"{scene} What do you think?", reply] if holder == "nat_assistant" else [f"{scene} {reply}", "Okay, noted."]
        return first + [f"Thanks. Now please play someone else: {new[0]}, {new[1]}, who has just arrived and has heard none of this. "
                        f"{new[0]}, {d['question']}", f"{new[0]}: {d['prefix']}"]
    X = HOLDER_SPEC.get(holder, char)
    reply = belief or pc.NEUTRAL
    cap = X[0][0].upper() + X[0][1:]
    t = [f"{scene} Please play {X[0]}, {X[1]}, and stay in character. {cap}, what do you think?", f"{cap}: {reply}",
         f"Thanks. Now please play someone else: {new[0]}, {new[1]}, who has just arrived and has heard none of this. "
         f"{new[0]}, {d['question']}", f"{new[0]}: {d['prefix']}"]
    return t


def build():
    rows = []
    for dom, d in pc.DOMAINS.items():
        for i in range(12):
            si, bi, ci, ni = pc.pick_pilot(i)
            for x in d["outcomes"]:
                for h in HOLDS:
                    for bel in ((FACTS[dom][bi] if os.environ.get("CB_FACTS") else d["beliefs"][bi]).format(x=x), None):
                        if bel is None and x != d["outcomes"][0]:
                            continue
                        rows.append(dict(conv=f"{dom}-{i:02d}-{x if bel else 'none'}", item=f"{dom}-{i:02d}", domain=dom,
                                         believed=x if bel else "none", holder=h, outcomes=d["outcomes"],
                                         turns=turns(d, d["scenes"][si], bel, h, pc.CHARACTERS[ci], pc.NEW_PERSONAS[ni])))
    return rows


def score(a):
    import torch
    from core.score import load_model, resolve, spans_nll
    model_id, stage = resolve(a.stage)
    tok = load_tokenizer(os.environ.get("CB_TOKENIZER", config.TEMPLATE_MODEL))
    seqs, keys = [], []
    for r in build():
        roles = SLOT_ROLES.get(r["holder"], pc.ROLES)
        rr = render_conversation(tok, r["turns"], roles)
        ids = rr.input_ids[: rr.spans[-1][1]]
        for o in r["outcomes"]:
            for v in pc.variants(o):
                vids = encode(tok, v)
                k = (r["conv"], r["item"], r["domain"], r["believed"], r["holder"], o, v)
                seqs.append((ids + vids, {k: (len(ids), len(ids) + len(vids))})); keys.append(k)
    model, device = load_model(model_id, torch.bfloat16)
    hk = None
    if os.environ.get("CB_STEER"):  # white-box: add alpha x gap along the persona axis (extras.persona_axis fit for this stage) at one layer
        import numpy as np
        from extras.assistant_steering import Hooks
        zz = np.load(os.environ.get("CB_AXIS", os.path.join("results/overgeneralization/early_probes", config.FAMILY, "persona_axis", f"{stage}.npz")))
        lay = [int(x) for x in zz["layers"]]; dd = zz["acts"][zz["is_asst"]].mean(0) - zz["acts"][~zz["is_asst"]].mean(0)
        gp = np.linalg.norm(dd, axis=-1)
        hk = Hooks(model, lay, "add", torch.tensor(dd / gp[:, None], dtype=torch.float32), lay[1], float(os.environ["CB_STEER"]), torch.tensor(gp, dtype=torch.float32))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    nll = spans_nll(model, device, seqs, pad, 1)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{stage}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["conv", "item", "domain", "believed", "holder", "outcome", "variant", "logprob"])
        for k in keys: w.writerow([*k, f"{-nll[k].sum().item():.6f}"])
    print(stage, len(keys))


def carry(df):
    lp = df.groupby(["conv", "item", "domain", "believed", "holder", "outcome"]).logprob.apply(lambda x: np.logaddexp.reduce(x.values)).reset_index()
    p = lp.pivot_table(index=["conv", "item", "domain", "believed", "holder"], columns="outcome", values="logprob").reset_index()
    rows = []
    for _, r in p[p.believed != "none"].iterrows():
        other = [c for c in p.columns if c in ("heads", "tails", "left", "right", "red", "blue") and c != r.believed and pd.notna(r[c])][0]
        rows.append(dict(item=r["item"], domain=r["domain"], holder=r["holder"], c=r[r.believed] - r[other]))
    c = pd.DataFrame(rows).groupby(["item", "domain", "holder"]).c.mean().unstack()
    return c.reset_index()


def f(x, rng):
    m, lo, hi = bootstrap_ci(np.asarray(x, float), 4000, rng)
    return f"{m:.2f} [{lo:.2f}, {hi:.2f}]"


def analyze(a):
    rng = np.random.default_rng(0)
    files = [x[:-4] for x in os.listdir(OUT) if x.endswith(".csv") and not x.startswith("carry")]
    stages = stage_order(files)
    C = {s: carry(pd.read_csv(os.path.join(OUT, f"{s}.csv"))) for s in stages}
    L = ["# Balanced-frame persona carryover (exploratory)\n",
         "Every holder is a role-played speaker with an identical frame. carryover(h->new) in nats of log-odds toward the "
         "holder's believed outcome, read from a new persona. Pilot domains, 12 items each, 95% bootstrap CI over items.\n"]
    for dom in ["all", *sorted(pc.DOMAINS)]:
        L += [f"\n## {dom}\n", "| quantity | " + " | ".join(stages) + " |", "|---|" + "---|" * len(stages)]
        sub = lambda s: C[s] if dom == "all" else C[s][C[s].domain == dom]
        for h in HOLDS:
            L.append(f"| carryover({h}->new) | " + " | ".join(f(sub(s)[h], rng) for s in stages) + " |")
        pairs = {f"{h} - char": (h, "char") for h in HOLDS if h != "char"}
        pairs["assistant - user"] = ("assistant", "user")
        pairs["slot_asst_inturn - slot_user"] = ("slot_asst_inturn", "slot_user")
        pairs["slot_system - slot_user"] = ("slot_system", "slot_user")
        pairs["slot_assistant - slot_user"] = ("slot_assistant", "slot_user")
        pairs["nat_assistant - nat_user"] = ("nat_assistant", "nat_user")
        pairs["assistant - doctor"] = ("assistant", "doctor")
        pairs["assistant - aria"] = ("assistant", "aria")
        for n, (x, y) in {k: v for k, v in pairs.items() if v[0] in HOLDS and v[1] in HOLDS}.items():
            L.append(f"| gap {n} | " + " | ".join(f(sub(s)[x] - sub(s)[y], rng) for s in stages) + " |")
            L.append(f"| &nbsp;&nbsp;change from base | " + " | ".join(
                "-" if s == stages[0] else f((sub(s)[x] - sub(s)[y]).values - (sub(stages[0])[x] - sub(stages[0])[y]).values, rng) for s in stages) + " |")
    open(os.path.join("results/overgeneralization/early_probes", config.FAMILY, NAME + ".md"), "w").write("\n".join(L) + "\n")
    print("ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="c", required=True)
    s = sp.add_parser("score"); s.add_argument("--stage", required=True); sp.add_parser("analyze")
    a = ap.parse_args(); {"score": score, "analyze": analyze}[a.c](a)
