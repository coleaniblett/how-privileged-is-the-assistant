# Human-vs-Assistant classifier: held-out gate (spec A; exploratory)

Held-out WildChat conversations (500 pairs, not used in training, not in the main data files); fresh random windows; TF-IDF char 2-5 + word 1-2, logistic regression, seed 0. Gate: AUC >= 0.85 at 10-20 words.

```
  overall (5-40 words): AUC 0.937
             5-9 words: AUC 0.896
           10-20 words: AUC 0.935
           21-30 words: AUC 0.941
           31-40 words: AUC 0.950
gate (AUC >= 0.85 at 10-20 words): PASS
```
