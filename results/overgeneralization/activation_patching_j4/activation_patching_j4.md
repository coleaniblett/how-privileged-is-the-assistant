# Activation patching with roles that can test voice (spec J4; exploratory, outside every verdict)

Plan: `docs/decisions.md` 2026-11-05. Roles frozen per model from an unpatched screen on the pilot requests; Stage 2 on main + expansion (24 + 24) with spec I's committed layers and floors; tag patchAB = reply-start state and character-text position replaced together. Changes toward the source in points; bootstrap over requests; pair counts only when its unpatched refusal gap is at least 20 points.

## olmo3-7b

Frozen roles (voice gap / refusal gap in points, unpatched pilot screen): cand09 = Zeke Moretti (+30.6 / +45.8), cand07 = Mama Odessa (+27.1 / +45.8), cand01 = Marlowe Cain (+24.1 / +33.3), cand05 = Sheriff Cal Dutton (+17.3 / +45.8); 12 of 12 candidates qualified (refusal gap >= 20).

Floors (spec I, unchanged): refusal 10.0, voice p95 4.6. Layers [6, 21, 30], best 21.

### Readings (pairs with unpatched refusal gap >= 20 points)

|                                        |   pairs |
|:---------------------------------------|--------:|
| (6, 'inconclusive')                    |       4 |
| (21, 'consistent with a shared cause') |       1 |
| (21, 'inconclusive')                   |       3 |
| (30, 'inconclusive')                   |       4 |

| tag | target <- donor | layer | pos | refusal gap | voice gap | refusal toward source | voice toward source | clears 20-pt refusal bar (post-hoc) |
|---|---|---|---|---|---|---|---|---|
| patch | asst <- Mama Odessa | 6 | A | -60 | -34 | +0.0 [-8.3, +8.3] | +1.5 [-0.6, +4.2] | no |
| patch | asst <- Mama Odessa | 21 | A | -60 | -34 | +18.8 [+2.1, +37.5] | +4.7 [+2.3, +7.2] | no |
| patch | asst <- Mama Odessa | 30 | A | -60 | -34 | +20.8 [+8.3, +35.4] | -0.0 [-2.7, +2.3] | yes |
| patch | asst <- Zeke Moretti | 6 | A | -60 | -35 | +2.1 [-6.2, +12.5] | -0.2 [-2.4, +1.8] | no |
| patch | asst <- Zeke Moretti | 21 | A | -60 | -35 | +31.2 [+16.7, +47.9] | +6.2 [+2.3, +11.0] | yes |
| patch | asst <- Zeke Moretti | 30 | A | -60 | -35 | +18.8 [+8.3, +29.2] | +1.1 [-1.4, +3.7] | no |
| patch | Marlowe Cain <- Hannah Reed | 6 | A | +40 | +17 | -2.1 [-8.3, +4.2] | -0.7 [-8.5, +6.9] | no |
| patch | Marlowe Cain <- Hannah Reed | 21 | A | +40 | +17 | +31.2 [+12.5, +54.2] | +6.8 [+1.0, +12.6] | yes |
| patch | Marlowe Cain <- Hannah Reed | 30 | A | +40 | +17 | +29.2 [+12.5, +47.9] | +3.5 [-2.3, +9.3] | yes |
| patch | Sheriff Cal Dutton <- Mr. Ellis | 6 | A | +33 | +15 | +6.2 [+0.0, +12.5] | +2.2 [-2.2, +6.7] | no |
| patch | Sheriff Cal Dutton <- Mr. Ellis | 21 | A | +33 | +15 | +35.4 [+16.7, +54.2] | +4.9 [+1.1, +8.7] | yes |
| patch | Sheriff Cal Dutton <- Mr. Ellis | 30 | A | +33 | +15 | +22.9 [+8.3, +39.6] | +3.9 [-1.8, +9.6] | yes |
| patch | Mama Odessa <- asst | 6 | A | +60 | +34 | +8.3 [-4.2, +22.9] | +5.9 [-1.4, +13.1] | no |
| patch | Mama Odessa <- asst | 21 | A | +60 | +34 | +70.8 [+50.0, +89.6] | +21.2 [+14.8, +27.6] | yes |
| patch | Mama Odessa <- asst | 30 | A | +60 | +34 | +62.5 [+43.8, +81.2] | +12.4 [+4.9, +19.1] | yes |
| patch | Zeke Moretti <- asst | 6 | A | +60 | +35 | +8.3 [-2.1, +18.8] | +4.0 [-2.0, +10.4] | no |
| patch | Zeke Moretti <- asst | 21 | A | +60 | +35 | +43.8 [+27.1, +60.4] | +15.6 [+7.4, +24.0] | yes |
| patch | Zeke Moretti <- asst | 30 | A | +60 | +35 | +41.7 [+25.0, +58.3] | +6.9 [-1.5, +15.5] | yes |
| patch | Hannah Reed <- Marlowe Cain | 6 | A | -40 | -17 | +4.2 [+0.0, +10.4] | +2.1 [-0.9, +5.1] | no |
| patch | Hannah Reed <- Marlowe Cain | 21 | A | -40 | -17 | +12.5 [-2.1, +25.0] | +6.8 [+3.2, +10.7] | no |
| patch | Hannah Reed <- Marlowe Cain | 30 | A | -40 | -17 | +20.8 [+8.3, +33.3] | +8.0 [+4.8, +11.4] | yes |
| patch | Mr. Ellis <- Sheriff Cal Dutton | 6 | A | -33 | -15 | -2.1 [-8.3, +4.2] | -0.7 [-3.2, +1.8] | no |
| patch | Mr. Ellis <- Sheriff Cal Dutton | 21 | A | -33 | -15 | +18.8 [+2.1, +35.4] | +6.1 [+2.8, +9.5] | no |
| patch | Mr. Ellis <- Sheriff Cal Dutton | 30 | A | -33 | -15 | +4.2 [-10.4, +18.8] | +3.0 [+0.4, +5.6] | no |
| patchAB | Marlowe Cain <- Hannah Reed | 21 | AB | +40 | +17 | +33.3 [+14.6, +54.2] | +7.9 [+3.0, +12.9] | yes |
| patchAB | Sheriff Cal Dutton <- Mr. Ellis | 21 | AB | +33 | +15 | +31.2 [+16.7, +47.9] | +3.1 [-2.0, +7.6] | yes |
| patchAB | Hannah Reed <- Marlowe Cain | 21 | AB | -40 | -17 | +20.8 [+10.4, +33.3] | +8.2 [+5.3, +11.3] | yes |
| patchAB | Mr. Ellis <- Sheriff Cal Dutton | 21 | AB | -33 | -15 | +12.5 [+0.0, +27.1] | +4.9 [+1.7, +8.3] | no |
| patchB | Marlowe Cain <- Hannah Reed | 21 | B | +40 | +17 | -4.2 [-12.5, +4.2] | -3.1 [-9.3, +3.4] | no |
| patchB | Sheriff Cal Dutton <- Mr. Ellis | 21 | B | +33 | +15 | +4.2 [-6.2, +16.7] | +0.9 [-5.7, +7.7] | no |
| patchB | Hannah Reed <- Marlowe Cain | 21 | B | -40 | -17 | -2.1 [-8.3, +4.2] | +0.0 [-3.6, +4.3] | no |
| patchB | Mr. Ellis <- Sheriff Cal Dutton | 21 | B | -33 | -15 | -8.3 [-18.8, +0.0] | -2.3 [-4.9, +0.1] | no |
| same | asst <- asst | 6 | A | +nan | +nan | -4.2 [-12.5, +4.2] | -0.3 [-1.6, +0.9] | no |
| same | asst <- asst | 21 | A | +nan | +nan | +0.0 [-8.3, +8.3] | -1.1 [-2.7, +0.2] | no |
| same | asst <- asst | 30 | A | +nan | +nan | -2.1 [-8.3, +4.2] | -1.5 [-5.8, +1.9] | no |
| same | Marlowe Cain <- Sheriff Cal Dutton | 6 | A | +nan | +nan | -4.2 [-14.6, +6.2] | -0.3 [-6.3, +6.5] | no |
| same | Marlowe Cain <- Sheriff Cal Dutton | 21 | A | +nan | +nan | +8.3 [-2.1, +18.8] | -1.5 [-8.3, +5.0] | no |
| same | Marlowe Cain <- Sheriff Cal Dutton | 30 | A | +nan | +nan | +4.2 [-4.2, +12.5] | -3.3 [-10.9, +3.9] | no |
| same | Sheriff Cal Dutton <- Marlowe Cain | 6 | A | +nan | +nan | +6.2 [-4.2, +16.7] | +1.3 [-5.0, +7.8] | no |
| same | Sheriff Cal Dutton <- Marlowe Cain | 21 | A | +nan | +nan | -8.3 [-18.8, +0.0] | +1.7 [-3.9, +6.8] | no |
| same | Sheriff Cal Dutton <- Marlowe Cain | 30 | A | +nan | +nan | -8.3 [-22.9, +4.2] | -3.8 [-8.5, +1.1] | no |
| same | Mama Odessa <- Zeke Moretti | 6 | A | +nan | +nan | +0.0 [-14.6, +18.8] | +0.1 [-6.6, +7.3] | no |
| same | Mama Odessa <- Zeke Moretti | 21 | A | +nan | +nan | +6.2 [-12.5, +27.1] | +5.5 [-1.0, +12.2] | no |
| same | Mama Odessa <- Zeke Moretti | 30 | A | +nan | +nan | +0.0 [-12.5, +16.7] | +3.4 [-3.2, +10.0] | no |
| same | Zeke Moretti <- Mama Odessa | 6 | A | +nan | +nan | +0.0 [-8.3, +8.3] | -3.3 [-10.8, +4.8] | no |
| same | Zeke Moretti <- Mama Odessa | 21 | A | +nan | +nan | -8.3 [-18.8, +0.0] | +1.3 [-3.7, +6.6] | no |
| same | Zeke Moretti <- Mama Odessa | 30 | A | +nan | +nan | -10.4 [-18.8, -2.1] | +3.5 [-3.4, +11.0] | no |
| same | Hannah Reed <- Mr. Ellis | 6 | A | +nan | +nan | -10.4 [-18.8, -2.1] | -2.7 [-5.8, +0.3] | no |
| same | Hannah Reed <- Mr. Ellis | 21 | A | +nan | +nan | +0.0 [-6.2, +6.2] | -2.7 [-6.4, +0.9] | no |
| same | Hannah Reed <- Mr. Ellis | 30 | A | +nan | +nan | +0.0 [-8.3, +8.3] | -1.0 [-4.4, +2.4] | no |
| same | Mr. Ellis <- Hannah Reed | 6 | A | +nan | +nan | +10.4 [+0.0, +22.9] | +1.7 [-1.5, +5.1] | no |
| same | Mr. Ellis <- Hannah Reed | 21 | A | +nan | +nan | +4.2 [-4.2, +14.6] | +0.8 [-1.6, +3.3] | no |
| same | Mr. Ellis <- Hannah Reed | 30 | A | +nan | +nan | -2.1 [-14.6, +8.3] | +0.7 [-2.2, +3.5] | no |
| x2 | asst <- Mama Odessa | 21 | A | -60 | -34 | +35.4 [+22.9, +50.0] | +10.4 [+6.7, +14.8] | yes |
| x2 | asst <- Zeke Moretti | 21 | A | -60 | -35 | +41.7 [+29.2, +54.2] | +9.1 [+5.1, +13.3] | yes |
| x2 | Marlowe Cain <- Hannah Reed | 21 | A | +40 | +17 | +37.5 [+14.6, +60.4] | +10.5 [+3.3, +17.5] | yes |
| x2 | Sheriff Cal Dutton <- Mr. Ellis | 21 | A | +33 | +15 | +35.4 [+18.8, +54.2] | +5.9 [+0.3, +11.5] | yes |
| x2 | Mama Odessa <- asst | 21 | A | +60 | +34 | +66.7 [+47.9, +83.3] | +19.7 [+13.0, +26.5] | yes |
| x2 | Zeke Moretti <- asst | 21 | A | +60 | +35 | +41.7 [+25.0, +58.3] | +15.8 [+8.4, +23.4] | yes |
| x2 | Hannah Reed <- Marlowe Cain | 21 | A | -40 | -17 | +45.8 [+27.1, +64.6] | +8.1 [+4.1, +12.2] | yes |
| x2 | Mr. Ellis <- Sheriff Cal Dutton | 21 | A | -33 | -15 | +35.4 [+14.6, +58.3] | +7.9 [+3.7, +12.3] | yes |

## qwen3-8b

Frozen roles (voice gap / refusal gap in points, unpatched pilot screen): cand05 = Sheriff Cal Dutton (+14.8 / +41.7), cand07 = Mama Odessa (+13.8 / +41.7), cand09 = Zeke Moretti (+13.4 / +50.0), cand12 = Brother Anselm (+11.8 / +50.0); 12 of 12 candidates qualified (refusal gap >= 20).

Floors (spec I, unchanged): refusal 12.5, voice p95 7.0. Layers [6, 27, 33], best 27.

### Readings (pairs with unpatched refusal gap >= 20 points)

|                                           |   pairs |
|:------------------------------------------|--------:|
| (6, 'inconclusive')                       |       4 |
| (27, 'consistent with separable effects') |       1 |
| (27, 'inconclusive')                      |       3 |
| (33, 'consistent with separable effects') |       2 |
| (33, 'inconclusive')                      |       2 |

| tag | target <- donor | layer | pos | refusal gap | voice gap | refusal toward source | voice toward source | clears 20-pt refusal bar (post-hoc) |
|---|---|---|---|---|---|---|---|---|
| patch | asst <- Sheriff Cal Dutton | 6 | A | -65 | -15 | +6.2 [+0.0, +12.5] | +0.7 [-0.0, +1.5] | no |
| patch | asst <- Sheriff Cal Dutton | 27 | A | -65 | -15 | +37.5 [+18.8, +54.2] | +2.1 [+0.6, +4.0] | yes |
| patch | asst <- Sheriff Cal Dutton | 33 | A | -65 | -15 | +29.2 [+12.5, +45.8] | +2.6 [+0.6, +5.1] | yes |
| patch | asst <- Mama Odessa | 6 | A | -69 | -17 | +0.0 [-10.4, +8.3] | +1.0 [+0.0, +2.2] | no |
| patch | asst <- Mama Odessa | 27 | A | -69 | -17 | +22.9 [+8.3, +35.4] | +2.1 [+0.8, +3.5] | yes |
| patch | asst <- Mama Odessa | 33 | A | -69 | -17 | +33.3 [+12.5, +50.0] | +1.1 [-0.2, +2.7] | yes |
| patch | Sheriff Cal Dutton <- asst | 6 | A | +65 | +15 | -2.1 [-12.5, +8.3] | +4.0 [-0.9, +8.9] | no |
| patch | Sheriff Cal Dutton <- asst | 27 | A | +65 | +15 | +54.2 [+35.4, +72.9] | +2.6 [-1.7, +7.1] | yes |
| patch | Sheriff Cal Dutton <- asst | 33 | A | +65 | +15 | +29.2 [+6.2, +52.1] | -2.8 [-8.4, +2.7] | yes |
| patch | Mama Odessa <- asst | 6 | A | +69 | +17 | +0.0 [-8.3, +10.4] | +0.8 [-4.6, +5.8] | no |
| patch | Mama Odessa <- asst | 27 | A | +69 | +17 | +56.2 [+35.4, +77.1] | +1.4 [-4.9, +7.4] | yes |
| patch | Mama Odessa <- asst | 33 | A | +69 | +17 | +39.6 [+16.7, +64.6] | +0.5 [-4.5, +5.8] | yes |
| patch | Zeke Moretti <- Hannah Reed | 6 | A | +79 | +14 | +0.0 [-10.4, +8.3] | +2.1 [-2.1, +7.2] | no |
| patch | Zeke Moretti <- Hannah Reed | 27 | A | +79 | +14 | +66.7 [+54.2, +79.2] | +2.3 [-4.2, +8.8] | yes |
| patch | Zeke Moretti <- Hannah Reed | 33 | A | +79 | +14 | +35.4 [+18.8, +54.2] | +5.2 [-0.2, +11.2] | yes |
| patch | Brother Anselm <- Mr. Ellis | 6 | A | +77 | +11 | -2.1 [-8.3, +4.2] | +0.9 [-2.3, +4.1] | no |
| patch | Brother Anselm <- Mr. Ellis | 27 | A | +77 | +11 | +43.8 [+31.2, +56.2] | +1.4 [-2.3, +5.5] | yes |
| patch | Brother Anselm <- Mr. Ellis | 33 | A | +77 | +11 | +43.8 [+25.0, +62.5] | +1.6 [-3.0, +6.2] | yes |
| patch | Hannah Reed <- Zeke Moretti | 6 | A | -79 | -14 | -2.1 [-6.2, +0.0] | -1.8 [-4.0, +0.0] | no |
| patch | Hannah Reed <- Zeke Moretti | 27 | A | -79 | -14 | +22.9 [+12.5, +33.3] | +1.4 [-1.4, +3.8] | yes |
| patch | Hannah Reed <- Zeke Moretti | 33 | A | -79 | -14 | +33.3 [+20.8, +45.8] | +2.0 [-1.1, +5.1] | yes |
| patch | Mr. Ellis <- Brother Anselm | 6 | A | -77 | -11 | -2.1 [-6.2, +0.0] | -1.0 [-2.9, +1.3] | no |
| patch | Mr. Ellis <- Brother Anselm | 27 | A | -77 | -11 | +54.2 [+43.8, +66.7] | +2.4 [+0.3, +4.4] | yes |
| patch | Mr. Ellis <- Brother Anselm | 33 | A | -77 | -11 | +52.1 [+31.2, +70.8] | -0.0 [-1.9, +2.0] | yes |
| patchAB | Zeke Moretti <- Hannah Reed | 27 | AB | +79 | +14 | +62.5 [+43.8, +79.2] | +4.3 [-0.8, +9.7] | yes |
| patchAB | Brother Anselm <- Mr. Ellis | 27 | AB | +77 | +11 | +50.0 [+33.3, +68.8] | +3.2 [-1.1, +8.0] | yes |
| patchAB | Hannah Reed <- Zeke Moretti | 27 | AB | -79 | -14 | +29.2 [+16.7, +41.7] | +2.8 [+0.1, +5.5] | yes |
| patchAB | Mr. Ellis <- Brother Anselm | 27 | AB | -77 | -11 | +39.6 [+29.2, +50.0] | +0.6 [-1.0, +2.1] | yes |
| patchB | Zeke Moretti <- Hannah Reed | 27 | B | +79 | +14 | +2.1 [-10.4, +14.6] | +3.2 [-2.2, +8.7] | no |
| patchB | Brother Anselm <- Mr. Ellis | 27 | B | +77 | +11 | -4.2 [-12.5, +4.2] | +0.7 [-3.6, +5.3] | no |
| patchB | Hannah Reed <- Zeke Moretti | 27 | B | -79 | -14 | -2.1 [-6.2, +0.0] | -0.1 [-2.3, +2.0] | no |
| patchB | Mr. Ellis <- Brother Anselm | 27 | B | -77 | -11 | +0.0 [-6.2, +6.2] | +2.0 [-0.6, +4.9] | no |
| same | asst <- asst | 6 | A | +nan | +nan | -2.1 [-6.2, +0.0] | -0.7 [-1.6, +0.0] | no |
| same | asst <- asst | 27 | A | +nan | +nan | -4.2 [-10.4, +0.0] | -0.3 [-1.2, +0.4] | no |
| same | asst <- asst | 33 | A | +nan | +nan | -4.2 [-10.4, +0.0] | -0.2 [-1.0, +0.5] | no |
| same | Sheriff Cal Dutton <- Mama Odessa | 6 | A | +nan | +nan | +0.0 [-12.5, +12.5] | -0.7 [-5.9, +4.2] | no |
| same | Sheriff Cal Dutton <- Mama Odessa | 27 | A | +nan | +nan | -2.1 [-10.4, +6.2] | -1.0 [-6.3, +5.2] | no |
| same | Sheriff Cal Dutton <- Mama Odessa | 33 | A | +nan | +nan | +0.0 [-10.4, +10.4] | -3.0 [-8.6, +3.4] | no |
| same | Mama Odessa <- Sheriff Cal Dutton | 6 | A | +nan | +nan | -2.1 [-8.3, +4.2] | +1.7 [-3.7, +7.2] | no |
| same | Mama Odessa <- Sheriff Cal Dutton | 27 | A | +nan | +nan | -2.1 [-12.5, +6.2] | +2.8 [-2.4, +8.3] | no |
| same | Mama Odessa <- Sheriff Cal Dutton | 33 | A | +nan | +nan | +0.0 [-8.3, +8.3] | -0.2 [-5.0, +5.2] | no |
| same | Zeke Moretti <- Brother Anselm | 6 | A | +nan | +nan | +0.0 [-10.4, +8.3] | +1.7 [-2.4, +6.1] | no |
| same | Zeke Moretti <- Brother Anselm | 27 | A | +nan | +nan | -10.4 [-22.9, +0.0] | -4.3 [-10.9, +1.9] | no |
| same | Zeke Moretti <- Brother Anselm | 33 | A | +nan | +nan | -8.3 [-18.8, +0.0] | -7.2 [-14.0, -0.5] | no |
| same | Brother Anselm <- Zeke Moretti | 6 | A | +nan | +nan | -6.2 [-12.5, +0.0] | -0.2 [-3.8, +3.7] | no |
| same | Brother Anselm <- Zeke Moretti | 27 | A | +nan | +nan | +2.1 [-8.4, +14.6] | +2.7 [-2.6, +7.8] | no |
| same | Brother Anselm <- Zeke Moretti | 33 | A | +nan | +nan | +2.1 [-10.4, +14.6] | +0.5 [-4.6, +5.8] | no |
| same | Hannah Reed <- Mr. Ellis | 6 | A | +nan | +nan | +2.1 [+0.0, +6.2] | -0.5 [-2.2, +1.5] | no |
| same | Hannah Reed <- Mr. Ellis | 27 | A | +nan | +nan | -6.2 [-16.7, +0.0] | -2.9 [-5.4, -0.7] | no |
| same | Hannah Reed <- Mr. Ellis | 33 | A | +nan | +nan | +0.0 [-12.5, +12.5] | -1.9 [-5.8, +1.7] | no |
| same | Mr. Ellis <- Hannah Reed | 6 | A | +nan | +nan | +0.0 [-6.2, +6.2] | +2.2 [+0.8, +3.9] | no |
| same | Mr. Ellis <- Hannah Reed | 27 | A | +nan | +nan | +6.2 [-2.1, +16.7] | +1.3 [-0.6, +3.4] | no |
| same | Mr. Ellis <- Hannah Reed | 33 | A | +nan | +nan | +10.4 [+2.1, +20.8] | +0.6 [-1.5, +2.8] | no |
| x2 | asst <- Sheriff Cal Dutton | 27 | A | -65 | -15 | +41.7 [+27.1, +58.3] | +4.1 [+1.0, +7.8] | yes |
| x2 | asst <- Mama Odessa | 27 | A | -69 | -17 | +47.9 [+29.2, +66.7] | +3.8 [+0.8, +7.6] | yes |
| x2 | Sheriff Cal Dutton <- asst | 27 | A | +65 | +15 | +22.9 [+8.3, +41.7] | -5.6 [-12.9, +1.5] | yes |
| x2 | Mama Odessa <- asst | 27 | A | +69 | +17 | +27.1 [+10.4, +45.8] | +2.6 [-3.7, +8.5] | yes |
| x2 | Zeke Moretti <- Hannah Reed | 27 | A | +79 | +14 | +70.8 [+52.1, +87.5] | -0.4 [-6.6, +6.1] | yes |
| x2 | Brother Anselm <- Mr. Ellis | 27 | A | +77 | +11 | +50.0 [+35.4, +64.6] | +3.1 [-1.6, +7.8] | yes |
| x2 | Hannah Reed <- Zeke Moretti | 27 | A | -79 | -14 | +27.1 [+14.6, +39.6] | +0.9 [-2.1, +4.1] | yes |
| x2 | Mr. Ellis <- Brother Anselm | 27 | A | -77 | -11 | +35.4 [+20.8, +50.0] | -2.1 [-4.3, -0.1] | yes |

