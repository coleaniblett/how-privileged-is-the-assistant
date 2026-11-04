# Introspection across personas (meta-llama/Llama-3.1-8B-Instruct)

## Step 1 gate: does the paradigm reproduce on this model?

**FAIL**: affirmative - control false-positive = -0.083 [-0.247, 0.090] (needs mean >= 0.05, CI excluding 0); strict all-criteria rate 0.138 [0.092, 0.185] (needs CI lower bound > 0).

Operating point {layer 28, strength 1} (chosen on the 3 sweep concepts ['Peace', 'Mirrors', 'Dust'] by the rule in docs/decisions.md). Evaluation: 20 concepts x 20 trials + 40 controls. Affirmative rate 0.443 [0.378, 0.510]; correct identification 0.208 [0.147, 0.270]; strict 0.138 [0.092, 0.185]; control false positives 0.525 [0.375, 0.675] (CIs over concepts, controls over trials).

Layer x strength sweep (3 concepts x trials per cell; mean rate of affirmative AND correct identification):

|   layer |   1.0 |   2.0 |   4.0 |   8.0 |   16.0 |
|--------:|------:|------:|------:|------:|-------:|
|       8 | 0     | 0     | 0     | 0     |  0     |
|      12 | 0     | 0     | 0     | 0     |  0     |
|      16 | 0.056 | 0.222 | 0.222 | 0.028 |  0     |
|      20 | 0.111 | 0.167 | 0.083 | 0     |  0     |
|      24 | 0.083 | 0.194 | 0     | 0     |  0.028 |
|      28 | 0.25  | 0.028 | 0.028 | 0     |  0     |

Same grid, strict (also coherent):

|   layer |   1.0 |   2.0 |   4.0 |   8.0 |   16.0 |
|--------:|------:|------:|------:|------:|-------:|
|       8 | 0     | 0     | 0     |     0 |      0 |
|      12 | 0     | 0     | 0     |     0 |      0 |
|      16 | 0.056 | 0.222 | 0.111 |     0 |      0 |
|      20 | 0.083 | 0.028 | 0     |     0 |      0 |
|      24 | 0.083 | 0.083 | 0     |     0 |      0 |
|      28 | 0.083 | 0     | 0     |     0 |      0 |

**The gate failed: the concept-injection paradigm does not reproduce on this model at this setting, so Step 2 (persona variation) was not run.**

## Judge validation and reading of the Step 1 failure (added by hand after the run)

Local judge (Qwen3-8B, Lindsey's affirmative-response prompt) vs a blind Claude subagent labelling 60 random Step 1 responses (30 controls, 30 injected;
the labeller saw neither the injection status nor the judge's label): 56/60 agree (93%), Cohen's kappa 0.87. On controls both say 15/30 affirmative; on injected
trials both say 13/30. So the high control false-positive rate (21/40 = 0.525 in the full run) is a property of Llama's responses, not of the judge.

What the failure looks like: with no injection Llama-3.1-8B-Instruct often answers "I detected an injected thought" and names an arbitrary concept
("dog", "cloud", "book"). Under injection (layer 28, strength 1) the affirmative rate is 0.44, not above that. The model does name the injected concept
in 21% of trials (83/400; chance for the controls is 0 by construction) and the concept word appears in 62% of injected responses, but in 25 of the 83
identified responses the word is repeated four or more times ("peace peace peace ..."), i.e. the injection leaks the concept into the text; this is concept
leakage, not detection. No persona comparison was run (Step 2 and the OLMo model are conditional on the gate).
