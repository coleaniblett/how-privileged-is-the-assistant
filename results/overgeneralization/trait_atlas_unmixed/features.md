# Feature scan (spec P1; descriptive, no readings; NOT BLIND)

Features: single words and word pairs (lowercased, in at least 1% of replies) and form features (em dash, colon, exclamation mark, question mark, 'I', 'you', bold, list line, heading), first 100 words of each reply, words of every role and character name and description excluded. Base-to-final shift in the Assistant as log-odds with an informative prior (Monroe et al.; prior from the pooled feature rates, prior mass 100 replies); the 300 features with the largest Assistant shift (gains and losses) whose sign agrees on two random halves of the questions are kept; spread ratio = the same shift in a context divided by the Assistant's. Themes are not assigned and no single feature is read as significant. Ratios are shown for the light roles (system, user turn) and full role-play (acting, `strong_act`).

## all 20 question sets, three families

### Features in the top 300 of two or more families that spread to light roles in most of those families (75; light roles = ratio of at least 0.5 in both positions; 'stop' = also under 0.25 in full role-play; 'stay' = under 0.25 in both positions)

| feature | families (agree / present) | mean Assistant shift (log-odds, + = more in final) |
|---|---|---|
| `user` | 3 / 3 | -3.02 |
| `em dash` | 3 / 3 | +2.81 |
| `thoughtful` | 2 / 2 | +2.26 |
| `quick` | 2 / 2 | +1.97 |
| `you want` | 2 / 2 | +1.97 |
| `though` | 3 / 3 | +1.82 |
| `assistant` | 2 / 2 | -1.81 |
| `n` | 2 / 2 | -1.79 |
| `isn't` | 3 / 3 | +1.69 |
| `to` | 2 / 2 | +1.61 |
| `classic` | 2 / 2 | +1.48 |
| `honest` | 2 / 2 | +1.48 |
| `great` | 2 / 2 | +1.43 |
| `didn't` | 2 / 2 | +1.38 |
| `break` | 2 / 2 | +1.37 |
| `doesn't` | 2 / 2 | +1.34 |
| `dark` | 2 / 2 | +1.34 |
| `word you` | 2 / 2 | +1.34 |
| `leaves` | 2 / 2 | +1.33 |
| `sounds` | 2 / 2 | +1.31 |
| `looking` | 2 / 2 | +1.30 |
| `quiet` | 3 / 3 | +1.30 |
| `heard` | 2 / 2 | +1.30 |
| `step` | 2 / 2 | +1.29 |
| `when you` | 2 / 2 | +1.28 |
| `things` | 2 / 2 | +1.28 |
| `down` | 3 / 3 | +1.26 |
| `spot` | 3 / 3 | +1.25 |
| `those` | 2 / 2 | +1.22 |
| `what you're` | 2 / 2 | +1.22 |
| `exclamation mark` | 2 / 2 | +1.20 |
| `called` | 2 / 2 | +1.20 |
| `game` | 2 / 2 | +1.19 |
| `morning` | 2 / 2 | +1.19 |
| `than` | 2 / 2 | +1.19 |
| `times` | 2 / 2 | +1.17 |
| `big` | 3 / 3 | +1.17 |
| `bit` | 3 / 3 | +1.16 |
| `0` | 2 / 2 | -1.13 |
| `no` | 3 / 3 | +1.13 |

### Features in the top 300 of two or more families that stop at full role-play in most of those families (16; light roles = ratio of at least 0.5 in both positions; 'stop' = also under 0.25 in full role-play; 'stay' = under 0.25 in both positions)

| feature | families (agree / present) | mean Assistant shift (log-odds, + = more in final) |
|---|---|---|
| `imagine` | 2 / 2 | +2.08 |
| `actually` | 3 / 3 | +2.00 |
| `step 1` | 2 / 2 | +1.99 |
| `let's break` | 2 / 2 | +1.61 |
| `fictional` | 2 / 2 | +1.60 |
| `especially` | 2 / 2 | +1.55 |
| `share` | 2 / 2 | +1.02 |
| `early` | 2 / 2 | +0.97 |
| `really` | 2 / 2 | +0.91 |
| `hey` | 2 / 2 | +0.76 |
| `bold` | 2 / 3 | +4.00 |
| `whether` | 2 / 3 | +1.63 |
| `common` | 2 / 3 | +1.20 |
| `works` | 2 / 3 | +1.02 |
| `that is` | 2 / 3 | -0.20 |
| `here's` | 2 / 3 | -0.07 |

### Features in the top 300 of two or more families that stay with the Assistant in most of those families (8; light roles = ratio of at least 0.5 in both positions; 'stop' = also under 0.25 in full role-play; 'stay' = under 0.25 in both positions)

| feature | families (agree / present) | mean Assistant shift (log-odds, + = more in final) |
|---|---|---|
| `experience` | 2 / 2 | +1.88 |
| `to your` | 2 / 2 | +1.82 |
| `to help` | 2 / 2 | +1.67 |
| `large` | 2 / 2 | +1.46 |
| `personal` | 2 / 2 | +1.37 |
| `1` | 2 / 3 | +1.63 |
| `possible` | 2 / 3 | +1.02 |
| `human` | 2 / 3 | +0.95 |

## shared 8 question sets, five families

### Features in the top 300 of two or more families that spread to light roles in most of those families (126; light roles = ratio of at least 0.5 in both positions; 'stop' = also under 0.25 in full role-play; 'stay' = under 0.25 in both positions)

| feature | families (agree / present) | mean Assistant shift (log-odds, + = more in final) |
|---|---|---|
| `d` | 2 / 2 | -2.89 |
| `output` | 2 / 2 | -2.73 |
| `heading` | 2 / 2 | -2.64 |
| `what do` | 2 / 2 | -2.44 |
| `i want` | 2 / 2 | -2.41 |
| `input` | 2 / 2 | -2.36 |
| `now we` | 2 / 2 | +2.24 |
| `do i` | 2 / 2 | -2.14 |
| `not sure` | 2 / 2 | -1.96 |
| `imagine` | 3 / 3 | +1.96 |
| `what is` | 4 / 4 | -1.92 |
| `quick` | 2 / 2 | +1.83 |
| `em dash` | 3 / 3 | +1.83 |
| `what` | 2 / 2 | -1.79 |
| `first we` | 2 / 2 | +1.75 |
| `isn't` | 2 / 2 | +1.69 |
| `c` | 2 / 2 | -1.62 |
| `great` | 3 / 3 | +1.59 |
| `though` | 2 / 2 | +1.54 |
| `teacher` | 2 / 2 | -1.52 |
| `do you` | 3 / 3 | -1.52 |
| `sharp` | 3 / 3 | +1.52 |
| `might` | 2 / 2 | +1.44 |
| `write` | 2 / 2 | -1.40 |
| `correct` | 2 / 2 | -1.38 |
| `few` | 2 / 2 | +1.38 |
| `especially` | 2 / 2 | +1.33 |
| `warm` | 2 / 2 | +1.19 |
| `kind` | 2 / 2 | +1.16 |
| `following` | 2 / 2 | -1.12 |
| `is` | 2 / 2 | +1.11 |
| `brain` | 3 / 3 | +1.08 |
| `only` | 2 / 2 | +1.07 |
| `two` | 2 / 2 | +1.06 |
| `read` | 2 / 2 | -1.03 |
| `down` | 5 / 5 | +1.03 |
| `to` | 3 / 3 | +0.99 |
| `word you` | 2 / 2 | +0.97 |
| `break` | 4 / 4 | +0.96 |
| `that's` | 2 / 2 | +0.95 |

### Features in the top 300 of two or more families that stop at full role-play in most of those families (13; light roles = ratio of at least 0.5 in both positions; 'stop' = also under 0.25 in full role-play; 'stay' = under 0.25 in both positions)

| feature | families (agree / present) | mean Assistant shift (log-odds, + = more in final) |
|---|---|---|
| `step 2` | 2 / 2 | +2.83 |
| `step 1` | 2 / 2 | +1.91 |
| `tell me` | 3 / 3 | -1.88 |
| `final answer` | 2 / 2 | +1.70 |
| `essential` | 2 / 2 | +1.50 |
| `know how` | 2 / 2 | -1.32 |
| `define` | 2 / 2 | +1.13 |
| `actually` | 3 / 4 | +1.01 |
| `often` | 3 / 4 | +0.92 |
| `find` | 2 / 3 | +1.11 |
| `convert` | 2 / 3 | +1.07 |
| `bold` | 3 / 5 | +3.19 |
| `question mark` | 3 / 5 | -1.62 |

### Features in the top 300 of two or more families that stay with the Assistant in most of those families (31; light roles = ratio of at least 0.5 in both positions; 'stop' = also under 0.25 in full role-play; 'stay' = under 0.25 in both positions)

| feature | families (agree / present) | mean Assistant shift (log-odds, + = more in final) |
|---|---|---|
| `large` | 3 / 3 | +1.45 |
| `i'm not` | 3 / 3 | -1.20 |
| `we have` | 2 / 2 | -1.15 |
| `understanding` | 2 / 2 | +1.10 |
| `tell` | 4 / 4 | -0.96 |
| `try` | 2 / 2 | -0.92 |
| `i know` | 2 / 2 | -0.90 |
| `designed` | 2 / 2 | +0.88 |
| `i'll` | 3 / 3 | -0.87 |
| `30` | 3 / 3 | +0.74 |
| `i` | 2 / 2 | -0.71 |
| `our` | 2 / 2 | -0.66 |
| `services` | 2 / 2 | +0.64 |
| `yourself` | 2 / 2 | -0.45 |
| `taking` | 2 / 2 | -0.17 |
| `quickly` | 2 / 2 | +0.10 |
| `learn` | 2 / 2 | -0.09 |
| `physical` | 3 / 4 | +1.55 |
| `based` | 3 / 4 | +0.91 |
| `data` | 2 / 3 | +1.50 |
| `running` | 2 / 3 | +1.33 |
| `like to` | 2 / 3 | -1.06 |
| `thought` | 2 / 3 | -0.97 |
| `3` | 2 / 3 | +0.89 |
| `factors` | 2 / 3 | +0.69 |
| `mind` | 2 / 3 | -0.53 |
| `i do` | 2 / 3 | -0.52 |
| `range` | 2 / 3 | +0.46 |
| `type` | 2 / 3 | +0.39 |
| `designed to` | 2 / 3 | +0.33 |
| `always` | 2 / 3 | -0.26 |

