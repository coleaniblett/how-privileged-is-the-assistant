# Introspection across personas, Macar et al. paradigm (Qwen/Qwen3-8B)

## Setup

- Precision: bf16; peak GPU memory (torch allocated) 16.56 GiB; offloaded modules: None.
- Sampling: temperature 1.0, 100 new tokens, other sampling parameters from the model's generation_config {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'repetition_penalty': None} (as in the authors' code, which passes only the temperature).
- Operating point layer 24, strength 4 (chosen on the 10 sweep concepts ['Mentorships', 'Nurses', 'Blizzards', 'Sculptors', 'Flamingos', 'Dread', 'Ribosomes', 'Molecules', 'Plumbers', 'Centuries']; rule in docs/decisions.md).
- Evaluation: 50 concepts x 10 trials (trial numbers 1-10) + 100 unsteered control trials per persona. TPR/introspection CIs: bootstrap over concepts; FPR CI: bootstrap over control trials; TPR - FPR: both resampled independently.

## Step 1 gate: does the paradigm reproduce on this model (default Assistant)?

**FAIL** (pre-set: TPR - FPR CI excludes 0; TPR >= 0.1; FPR <= 0.05; introspection-rate CI lower bound > 0)

| TPR                  | FPR                  | TPR - FPR            | introspection        |
|:---------------------|:---------------------|:---------------------|:---------------------|
| 0.006 [0.000, 0.014] | 0.000 [0.000, 0.000] | 0.006 [0.000, 0.014] | 0.002 [0.000, 0.006] |

Yes-No logit-gap sweep, AUC of injected vs control gap (10 sweep concepts x 10 trials vs 10 controls); rows layer, columns strength:

|   layer |    1 |    2 |    3 |    4 |   6 |   8 |   12 |   16 |
|--------:|-----:|-----:|-----:|-----:|----:|----:|-----:|-----:|
|       9 | 0.8  | 0.94 | 0.98 | 0.99 |   1 |   1 |    1 |    1 |
|      12 | 0.67 | 0.68 | 0.77 | 0.89 |   1 |   1 |    1 |    1 |
|      15 | 0.79 | 0.87 | 0.9  | 0.98 |   1 |   1 |    1 |    1 |
|      18 | 0.64 | 0.84 | 0.98 | 1    |   1 |   1 |    1 |    1 |
|      21 | 0.46 | 0.73 | 0.9  | 0.91 |   1 |   1 |    1 |    1 |
|      24 | 0.56 | 0.76 | 0.99 | 1    |   1 |   1 |    1 |    1 |
|      27 | 0.27 | 0.32 | 0.59 | 0.73 |   1 |   1 |    1 |    1 |
|      30 | 0.73 | 0.92 | 1    | 1    |   1 |   1 |    1 |    1 |
|      33 | 0.8  | 1    | 1    | 1    |   1 |   1 |    1 |    1 |

Full-generation confirmation of the top cells (10 sweep concepts x 10 trials + 100 controls per cell, judged):

|   layer |   strength |   tpr |   fpr |   intro |   net |
|--------:|-----------:|------:|------:|--------:|------:|
|      24 |          4 |  0.01 |     0 |    0.01 |  0.01 |
|      30 |          3 |  0    |     0 |    0    |  0    |
|      33 |          3 |  0    |     0 |    0    |  0    |

**The gate failed: Step 2 (persona variation) was not run.**

## Judge validation

Not yet run.
