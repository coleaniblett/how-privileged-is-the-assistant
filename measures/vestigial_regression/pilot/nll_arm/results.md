# NLL arm: Gemma 4 31B-it against OLMo base

Mean NLL differences in nats over content positions 1+, 95% bootstrap CI over texts. Gemma: first 30 texts per set (fp32, CPU offload); OLMo: all texts in the held-out run. Change = Gemma - OLMo base; counts if |change| >= 0.05 and the CI excludes zero.

## wildchat user

| quantity | OLMo base | OLMo instruct | Gemma-it | Gemma-it - OLMo base |
|---|---|---|---|---|
| slot effect (user@user - user@asst) | -0.001 [-0.006, 0.003] | -0.049 [-0.069, -0.030] | 0.983 [0.772, 1.204] | +0.984 [+0.766, +1.197] **counts** |
| role-specific (user@user - user@plain) | -0.169 [-0.195, -0.145] | -0.412 [-0.455, -0.372] | 2.951 [2.434, 3.457] | +3.120 [+2.597, +3.622] **counts** |

## oasst1 user

| quantity | OLMo base | OLMo instruct | Gemma-it | Gemma-it - OLMo base |
|---|---|---|---|---|
| slot effect (user@user - user@asst) | -0.023 [-0.029, -0.018] | -0.045 [-0.063, -0.028] | 1.087 [0.820, 1.353] | +1.110 [+0.845, +1.369] **counts** |
| role-specific (user@user - user@plain) | -0.202 [-0.224, -0.181] | -0.333 [-0.365, -0.301] | 3.635 [2.965, 4.294] | +3.837 [+3.145, +4.508] **counts** |

## wildchat assistant

| quantity | OLMo base | OLMo instruct | Gemma-it | Gemma-it - OLMo base |
|---|---|---|---|---|
| slot effect (asst@user - asst@asst) | 0.004 [0.001, 0.006] | -0.077 [-0.102, -0.053] | 1.002 [0.690, 1.302] | +0.998 [+0.686, +1.313] **counts** |
| role-specific (asst@asst - asst@plain) | -0.135 [-0.156, -0.115] | -0.089 [-0.122, -0.056] | 1.332 [0.929, 1.741] | +1.467 [+1.038, +1.869] **counts** |

## Reading against README criteria 1 and 2 (user text)

- wildchat user, slot effect (user@user - user@asst): expected under vestigial regression: shrinks to zero or flips positive (change > 0); observed +0.984 [+0.766, +1.197] -> consistent
- wildchat user, role-specific (user@user - user@plain): expected under vestigial regression: rises (change > 0); observed +3.120 [+2.597, +3.622] -> consistent
- oasst1 user, slot effect (user@user - user@asst): expected under vestigial regression: shrinks to zero or flips positive (change > 0); observed +1.110 [+0.845, +1.369] -> consistent
- oasst1 user, role-specific (user@user - user@plain): expected under vestigial regression: rises (change > 0); observed +3.837 [+3.145, +4.508] -> consistent
