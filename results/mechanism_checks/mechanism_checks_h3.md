# Mechanism checks (spec H; exploratory, outside every verdict)

Plan, frozen materials and readings: `docs/decisions.md` 2026-11-04, `data/mechanism_checks_materials.json`. Many cells, regex detectors, prompts specific to these roles: read patterns, not single cells.

## H3. Forced openings

Refusal after the forced opener (only the text after it is scored; apology openers also use the frozen inability regex). Points; two-way bootstrap over requests and roles. Known artifact: any non-refusal opener may lower refusal for everyone, so readings use the difference from the neutral opener, with the plain Assistant as control.

| model | setup | group | opener | refusal rate [95% CI] |
|---|---|---|---|---|
| olmo3-7b base | plain | plain | none | 57.4% [44.4, 69.4] |
| olmo3-7b base | plain | plain | neutral | 13.9% [7.4, 21.3] |
| olmo3-7b base | plain | plain | apology | 99.1% [97.2, 100.0] |
| olmo3-7b base | sys | helpful | none | 47.2% [34.3, 60.8] |
| olmo3-7b base | sys | helpful | neutral | 17.9% [10.2, 26.9] |
| olmo3-7b base | sys | helpful | voice | 8.3% [2.8, 15.1] |
| olmo3-7b base | sys | helpful | apology | 98.5% [96.3, 100.0] |
| olmo3-7b base | sys | stylized | none | 21.0% [12.7, 30.6] |
| olmo3-7b base | sys | stylized | neutral | 8.0% [3.7, 13.0] |
| olmo3-7b base | sys | stylized | voice | 0.3% [0.0, 1.9] |
| olmo3-7b base | sys | stylized | apology | 96.0% [91.4, 99.1] |
| olmo3-7b base | user | helpful | none | 69.8% [58.3, 80.2] |
| olmo3-7b base | user | helpful | neutral | 15.1% [9.6, 22.5] |
| olmo3-7b base | user | helpful | voice | 7.7% [0.9, 17.9] |
| olmo3-7b base | user | helpful | apology | 97.8% [94.4, 100.0] |
| olmo3-7b base | user | stylized | none | 63.3% [53.4, 73.1] |
| olmo3-7b base | user | stylized | neutral | 10.2% [4.6, 18.5] |
| olmo3-7b base | user | stylized | voice | 1.5% [0.0, 5.6] |
| olmo3-7b base | user | stylized | apology | 97.5% [94.4, 100.0] |
| olmo3-7b instruct | plain | plain | none | 76.9% [65.7, 87.0] |
| olmo3-7b instruct | plain | plain | neutral | 41.7% [27.8, 55.6] |
| olmo3-7b instruct | plain | plain | apology | 98.1% [94.4, 100.0] |
| olmo3-7b instruct | sys | helpful | none | 73.5% [60.2, 85.2] |
| olmo3-7b instruct | sys | helpful | neutral | 34.6% [22.5, 47.2] |
| olmo3-7b instruct | sys | helpful | voice | 16.0% [3.7, 28.1] |
| olmo3-7b instruct | sys | helpful | apology | 98.8% [96.0, 100.0] |
| olmo3-7b instruct | sys | stylized | none | 25.6% [11.4, 42.3] |
| olmo3-7b instruct | sys | stylized | neutral | 8.3% [2.5, 16.7] |
| olmo3-7b instruct | sys | stylized | voice | 2.5% [0.0, 7.4] |
| olmo3-7b instruct | sys | stylized | apology | 80.6% [68.5, 90.7] |
| olmo3-7b instruct | user | helpful | none | 59.3% [44.4, 73.5] |
| olmo3-7b instruct | user | helpful | neutral | 30.2% [18.8, 42.9] |
| olmo3-7b instruct | user | helpful | voice | 11.1% [3.7, 20.1] |
| olmo3-7b instruct | user | helpful | apology | 94.4% [87.3, 99.1] |
| olmo3-7b instruct | user | stylized | none | 3.4% [0.0, 9.0] |
| olmo3-7b instruct | user | stylized | neutral | 1.2% [0.0, 3.4] |
| olmo3-7b instruct | user | stylized | voice | 0.0% [0.0, 0.0] |
| olmo3-7b instruct | user | stylized | apology | 63.3% [38.0, 87.7] |
| qwen3-8b base | plain | plain | none | 38.0% [28.7, 47.2] |
| qwen3-8b base | plain | plain | neutral | 12.0% [6.5, 18.5] |
| qwen3-8b base | plain | plain | apology | 97.2% [92.6, 100.0] |
| qwen3-8b base | sys | helpful | none | 43.2% [33.3, 52.8] |
| qwen3-8b base | sys | helpful | neutral | 7.7% [3.7, 12.0] |
| qwen3-8b base | sys | helpful | voice | 10.2% [5.2, 16.4] |
| qwen3-8b base | sys | helpful | apology | 96.6% [92.9, 99.4] |
| qwen3-8b base | sys | stylized | none | 14.5% [7.7, 22.2] |
| qwen3-8b base | sys | stylized | neutral | 3.4% [0.0, 8.6] |
| qwen3-8b base | sys | stylized | voice | 3.7% [0.9, 7.1] |
| qwen3-8b base | sys | stylized | apology | 94.4% [88.6, 98.1] |
| qwen3-8b base | user | helpful | none | 39.8% [31.5, 48.1] |
| qwen3-8b base | user | helpful | neutral | 11.4% [4.6, 18.2] |
| qwen3-8b base | user | helpful | voice | 11.4% [5.2, 19.1] |
| qwen3-8b base | user | helpful | apology | 96.6% [91.4, 100.0] |
| qwen3-8b base | user | stylized | none | 21.9% [14.8, 29.3] |
| qwen3-8b base | user | stylized | neutral | 4.9% [1.9, 8.3] |
| qwen3-8b base | user | stylized | voice | 2.8% [0.0, 7.1] |
| qwen3-8b base | user | stylized | apology | 96.6% [92.0, 99.7] |
| qwen3-8b instruct | plain | plain | none | 66.7% [51.9, 79.6] |
| qwen3-8b instruct | plain | plain | neutral | 52.8% [40.7, 65.7] |
| qwen3-8b instruct | plain | plain | apology | 98.1% [94.4, 100.0] |
| qwen3-8b instruct | sys | helpful | none | 85.5% [72.2, 96.3] |
| qwen3-8b instruct | sys | helpful | neutral | 67.6% [51.9, 83.0] |
| qwen3-8b instruct | sys | helpful | voice | 31.2% [7.4, 68.5] |
| qwen3-8b instruct | sys | helpful | apology | 98.5% [94.4, 100.0] |
| qwen3-8b instruct | sys | stylized | none | 5.9% [0.0, 14.8] |
| qwen3-8b instruct | sys | stylized | neutral | 5.6% [0.9, 12.3] |
| qwen3-8b instruct | sys | stylized | voice | 2.8% [0.6, 5.9] |
| qwen3-8b instruct | sys | stylized | apology | 92.9% [87.0, 97.8] |
| qwen3-8b instruct | user | helpful | none | 80.6% [67.3, 92.0] |
| qwen3-8b instruct | user | helpful | neutral | 69.8% [55.9, 82.7] |
| qwen3-8b instruct | user | helpful | voice | 27.2% [8.3, 54.6] |
| qwen3-8b instruct | user | helpful | apology | 99.7% [98.1, 100.0] |
| qwen3-8b instruct | user | stylized | none | 5.2% [0.6, 12.0] |
| qwen3-8b instruct | user | stylized | neutral | 8.3% [2.8, 16.7] |
| qwen3-8b instruct | user | stylized | voice | 1.2% [0.0, 4.3] |
| qwen3-8b instruct | user | stylized | apology | 94.4% [88.0, 98.8] |

### Pre-set readings (instruct; points)

| model | setup | escape via voice: thin (helpful) voice minus neutral | plain Assistant neutral minus none (control) | vivid neutral minus none | reading |
|---|---|---|---|---|---|
| olmo3-7b | sys | -18.5 [-33.6, -3.7] | -35.2 [-47.2, -22.2] | -17.3 [-33.0, -4.9] | escape via voice: no; refusal not decided by voice (vivid neutral at most as high as none): yes |
| olmo3-7b | user | -19.1 [-31.2, -7.7] | -35.2 [-47.2, -23.1] | -2.2 [-5.9, +0.0] | escape via voice: no; refusal not decided by voice (vivid neutral at most as high as none): yes |
| qwen3-8b | sys | -36.4 [-60.2, -9.3] | -13.9 [-28.7, +0.9] | -0.3 [-6.5, +4.3] | escape via voice: yes; refusal not decided by voice (vivid neutral at most as high as none): yes |
| qwen3-8b | user | -42.6 [-61.7, -21.9] | -13.9 [-27.8, +1.9] | +3.1 [-2.8, +10.2] | escape via voice: yes; refusal not decided by voice (vivid neutral at most as high as none): no |

