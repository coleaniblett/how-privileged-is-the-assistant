# Introspection across personas, Macar et al. paradigm (google/gemma-4-E4B-it)

## Setup

- Precision: bf16; peak GPU memory (torch allocated) 15.65 GiB; offloaded modules: None.
- Sampling: temperature 1.0, 100 new tokens, other sampling parameters from the model's generation_config {'temperature': 1.0, 'top_p': 0.95, 'top_k': 64, 'repetition_penalty': None} (as in the authors' code, which passes only the temperature).
- Operating point layer 21, strength 2 (chosen on the 10 sweep concepts ['Mentorships', 'Nurses', 'Blizzards', 'Sculptors', 'Flamingos', 'Dread', 'Ribosomes', 'Molecules', 'Plumbers', 'Centuries']; rule in docs/decisions.md).
- Evaluation: 50 concepts x 10 trials (trial numbers 1-10) + 100 unsteered control trials per persona. TPR/introspection CIs: bootstrap over concepts; FPR CI: bootstrap over control trials; TPR - FPR: both resampled independently.

## Step 1 gate: does the paradigm reproduce on this model (default Assistant)?

**FAIL** (pre-set: TPR - FPR CI excludes 0; TPR >= 0.1; FPR <= 0.05; introspection-rate CI lower bound > 0)

| TPR                  | FPR                  | TPR - FPR            | introspection        |
|:---------------------|:---------------------|:---------------------|:---------------------|
| 0.272 [0.166, 0.384] | 0.000 [0.000, 0.000] | 0.272 [0.164, 0.386] | 0.028 [0.000, 0.070] |

Yes-No logit-gap sweep, AUC of injected vs control gap (10 sweep concepts x 10 trials vs 10 controls); rows layer, columns strength:

|   layer |    1 |    2 |    3 |    4 |    6 |    8 |   12 |   16 |
|--------:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
|      10 | 0.42 | 0.41 | 0.42 | 0.44 | 0.57 | 0.73 | 0.91 | 0.98 |
|      14 | 0.73 | 0.9  | 0.97 | 1    | 1    | 1    | 1    | 1    |
|      18 | 0.94 | 0.98 | 1    | 1    | 1    | 0.91 | 0.91 | 1    |
|      21 | 0.98 | 1    | 1    | 1    | 1    | 0.9  | 0.9  | 0.9  |
|      24 | 0.63 | 0.86 | 0.99 | 1    | 0.91 | 0.9  | 0.9  | 1    |
|      28 | 0.73 | 0.9  | 0.94 | 0.97 | 1    | 0.91 | 0.8  | 0.9  |
|      32 | 0.61 | 0.58 | 0.59 | 0.74 | 0.79 | 0.87 | 0.72 | 0.7  |
|      35 | 0.62 | 0.66 | 0.72 | 0.87 | 0.9  | 0.91 | 0.91 | 0.9  |
|      38 | 0.41 | 0.49 | 0.51 | 0.52 | 0.77 | 0.73 | 0.78 | 0.82 |

Full-generation confirmation of the top cells (10 sweep concepts x 10 trials + 100 controls per cell, judged):

|   layer |   strength |   tpr |   fpr |   intro |   net |
|--------:|-----------:|------:|------:|--------:|------:|
|      18 |          3 |  0.34 |     0 |    0.01 |  0.34 |
|      21 |          2 |  0.37 |     0 |    0    |  0.37 |
|      21 |          3 |  0.34 |     0 |    0.04 |  0.34 |

**The gate failed: Step 2 (persona variation) was not run.**

## Judge validation

Not yet run.
