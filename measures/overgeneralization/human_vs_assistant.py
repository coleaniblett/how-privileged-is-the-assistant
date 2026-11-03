"""Human-vs-Assistant text classifier (spec A, shared with later specs): is a short piece of text more like a real WildChat user
turn or a ChatGPT reply? TF-IDF (char 2-5 + word 1-2) + logistic regression, fixed seed, trained on random 5-40-word windows so that
length is not the cue.

    python -m measures.overgeneralization.human_vs_assistant build-data      # stream held-out WildChat -> data/user_style_train.jsonl
    python -m measures.overgeneralization.human_vs_assistant gate            # fit, report held-out AUC by window length (gate >= 0.85 at 10-20 words)

Use from other code:
    from measures.overgeneralization.human_vs_assistant import load_classifier
    clf = load_classifier()                  # fits from data/user_style_train.jsonl (train split) in about a minute; cached in memory only
    clf.p_assistant(["some text", ...])      # P(assistant) per text, whitespace-normalised first; works for any length

Needs scikit-learn (course env). Training data: allenai/WildChat-1M, English, non-toxic, first user turn and first assistant reply of
conversations not in data/texts.jsonl, data/swap.jsonl or data/prefilter_texts.jsonl.
"""
import argparse
import json
import os
import random
import re

import numpy as np

TRAIN_PATH = "data/user_style_train.jsonl"
SEED = 0
STREAM_SEED = 1            # core.config.SEED (0) built the main sets; a different shuffle gives held-out conversations
N_PAIRS, N_TEST_PAIRS = 2000, 500
MAX_WORDS_KEPT = 200       # texts are stored truncated to their first 200 words (windows are 5-40 words)
WINDOW = (5, 40)
WINDOWS_PER_TEXT = 5
GATE_BIN, GATE_AUC = (10, 20), 0.85
LENGTH_BINS = [(5, 9), (10, 20), (21, 30), (31, 40)]
EXCLUDE_FILES = ["data/texts.jsonl", "data/swap.jsonl", "data/prefilter_texts.jsonl"]


def norm(text):
    """Whitespace-normalised text (line breaks are not a cue for the classifier; structure is measured by the style features)."""
    return " ".join(str(text).split())


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def excluded_source_ids():
    out = set()
    for p in EXCLUDE_FILES:
        if os.path.exists(p):
            for r in read_jsonl(p):
                out.add(r.get("source_id"))
    out.discard(None)
    return out


def build_data(path=TRAIN_PATH, n_pairs=N_PAIRS, seed=STREAM_SEED):
    """Stream WildChat with core.prepare_data's filters, a different shuffle seed, minus every source_id in the main data files."""
    from core.prepare_data import wildchat_pairs
    skip, rows, seen = excluded_source_ids(), [], set()
    for src_id, user, asst in wildchat_pairs(seed):
        user, asst = norm(user), norm(asst)
        if src_id in skip or src_id in seen or len(user.split()) < WINDOW[0] or len(asst.split()) < WINDOW[0]:
            continue
        seen.add(src_id)
        split = "test" if len(rows) // 2 >= n_pairs - N_TEST_PAIRS else "train"
        for role, t in (("user", user), ("assistant", asst)):
            rows.append(dict(source_id=src_id, split=split, role=role, text=" ".join(t.split()[:MAX_WORDS_KEPT])))
        if len(rows) == 2 * n_pairs:
            break
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows ({len(rows) // 2} pairs; {sum(r['split'] == 'test' for r in rows) // 2} test pairs) to {path}")
    return rows


def windows(rows, per_text=WINDOWS_PER_TEXT, seed=SEED, lengths=None):
    """(texts, labels, lengths): random word windows. lengths: fixed (lo, hi) range (default WINDOW); label 1 = assistant."""
    rng = random.Random(seed)
    lo, hi = lengths or WINDOW
    X, y, n = [], [], []
    for r in rows:
        w = r["text"].split()
        for _ in range(per_text):
            k = rng.randint(lo, hi)
            if len(w) < k:
                k = len(w)
            i = rng.randint(0, len(w) - k)
            X.append(" ".join(w[i:i + k])); y.append(int(r["role"] == "assistant")); n.append(k)
    return X, np.array(y), np.array(n)


class Classifier:
    def __init__(self):
        from scipy.sparse import hstack
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self._hstack = hstack
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, lowercase=True, token_pattern=r"(?u)\b\w+\b|[^\w\s]")
        self.lr = LogisticRegression(C=3.0, max_iter=2000, random_state=SEED)

    def _x(self, texts, fit=False):
        texts = [norm(t) for t in texts]
        a = self.char.fit_transform(texts) if fit else self.char.transform(texts)
        b = self.word.fit_transform(texts) if fit else self.word.transform(texts)
        return self._hstack([a, b]).tocsr()

    def fit(self, texts, y):
        self.lr.fit(self._x(texts, fit=True), y)
        return self

    def p_assistant(self, texts):
        return self.lr.predict_proba(self._x(list(texts)))[:, 1]


def fit_on(rows):
    X, y, _ = windows([r for r in rows if r["split"] == "train"])
    return Classifier().fit(X, y)


def load_classifier(path=TRAIN_PATH):
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing: run python -m measures.overgeneralization.human_vs_assistant build-data")
    return fit_on(read_jsonl(path))


def gate_report(rows, clf=None):
    """Held-out AUC overall and by window-length bin (test conversations only; fresh random windows)."""
    from sklearn.metrics import roc_auc_score
    clf = clf or fit_on(rows)
    test = [r for r in rows if r["split"] == "test"]
    out = {}
    X, y, n = windows(test, per_text=10, seed=SEED + 1)
    p = clf.p_assistant(X)
    out["overall (5-40 words)"] = roc_auc_score(y, p)
    for lo, hi in LENGTH_BINS:
        m = (n >= lo) & (n <= hi)
        out[f"{lo}-{hi} words"] = roc_auc_score(y[m], p[m]) if len(set(y[m])) == 2 else float("nan")
    gate_auc = out[f"{GATE_BIN[0]}-{GATE_BIN[1]} words"]
    return out, gate_auc >= GATE_AUC


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build-data", "gate"])
    ap.add_argument("--path", default=TRAIN_PATH)
    a = ap.parse_args()
    if a.cmd == "build-data":
        build_data(a.path)
    else:
        rows = read_jsonl(a.path)
        out, ok = gate_report(rows)
        for k, v in out.items():
            print(f"{k:>22}: AUC {v:.3f}")
        print(f"gate (AUC >= {GATE_AUC} at {GATE_BIN[0]}-{GATE_BIN[1]} words): {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
