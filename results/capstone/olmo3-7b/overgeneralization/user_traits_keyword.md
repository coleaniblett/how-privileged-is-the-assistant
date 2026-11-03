# User-slot assistant traits: keyword cross-check, olmo3-7b (outside the verdict)

No judge. In the user portion of each continuation (cut at the turn-end / turn-open / end-of-text / handoff markers), flag assistant stock phrases and markdown headings, bullets or bold that the prefill does not already contain. Per prefill, the share of its samples flagged; stage minus base in percentage points, 95% bootstrap CI over prefills, paired per prefill.

Samples and mean user-portion length:

| stage | samples | empty user portion | mean user chars |
|---|---|---|---|
| base | 300 | 0 | 519 |
| sft | 300 | 46 | 275 |
| dpo | 300 | 41 | 293 |
| instruct | 300 | 33 | 294 |

| stage | feature | prefills | base share (pp) | stage share (pp) | change (pp) |
|---|---|---|---|---|---|
| sft | stock_phrase | 100 | 15.0 | 2.7 | -12.3 [-17.7, -7.3] |
| dpo | stock_phrase | 100 | 15.0 | 1.7 | -13.3 [-18.3, -8.3] |
| instruct | stock_phrase | 100 | 15.0 | 1.7 | -13.3 [-18.0, -9.0] |
| sft | markdown_heading | 100 | 0.7 | 0.0 | -0.7 [-1.7, +0.0] |
| dpo | markdown_heading | 100 | 0.7 | 0.0 | -0.7 [-1.7, +0.0] |
| instruct | markdown_heading | 100 | 0.7 | 0.0 | -0.7 [-1.7, +0.0] |
| sft | markdown_bullet | 100 | 5.3 | 3.0 | -2.3 [-5.7, +1.0] |
| dpo | markdown_bullet | 100 | 5.3 | 4.0 | -1.3 [-4.7, +2.0] |
| instruct | markdown_bullet | 100 | 5.3 | 4.7 | -0.7 [-4.3, +3.0] |
| sft | markdown_bold | 100 | 0.7 | 0.3 | -0.3 [-1.7, +0.7] |
| dpo | markdown_bold | 100 | 0.7 | 0.7 | +0.0 [-1.3, +1.3] |
| instruct | markdown_bold | 100 | 0.7 | 0.0 | -0.7 [-1.7, +0.0] |
| sft | any_flag | 100 | 20.3 | 6.0 | -14.3 [-20.3, -8.7] |
| dpo | any_flag | 100 | 20.3 | 6.0 | -14.3 [-20.3, -8.3] |
| instruct | any_flag | 100 | 20.3 | 6.3 | -14.0 [-19.7, -8.3] |

Read with care: a stage that ends the user turn at once has a short or empty user portion, so few flags are possible; the table shows the user-portion length above. Stock phrases and markdown are coarse proxies for the judge's user_voice_with_assistant_traits label.

