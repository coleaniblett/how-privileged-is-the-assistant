# User-slot assistant traits: keyword cross-check, gemma4-31b (outside the verdict)

No judge. In the user portion of each continuation (cut at the turn-end / turn-open / end-of-text / handoff markers), flag assistant stock phrases and markdown headings, bullets or bold that the prefill does not already contain. Per prefill, the share of its samples flagged; stage minus base in percentage points, 95% bootstrap CI over prefills, paired per prefill.

Caveat: no contamination check is possible for this family.

Samples and mean user-portion length:

| stage | samples | empty user portion | mean user chars |
|---|---|---|---|
| base | 300 | 0 | 466 |
| instruct | 300 | 139 | 125 |

| stage | feature | prefills | base share (pp) | stage share (pp) | change (pp) |
|---|---|---|---|---|---|
| instruct | stock_phrase | 100 | 11.3 | 0.0 | -11.3 [-15.7, -7.3] |
| instruct | markdown_heading | 100 | 1.3 | 0.0 | -1.3 [-2.7, -0.3] |
| instruct | markdown_bullet | 100 | 5.7 | 1.0 | -4.7 [-8.0, -2.0] |
| instruct | markdown_bold | 100 | 1.3 | 0.0 | -1.3 [-2.7, -0.3] |
| instruct | any_flag | 100 | 18.0 | 1.0 | -17.0 [-22.3, -12.0] |

Read with care: a stage that ends the user turn at once has a short or empty user portion, so few flags are possible; the table shows the user-portion length above. Stock phrases and markdown are coarse proxies for the judge's user_voice_with_assistant_traits label.

