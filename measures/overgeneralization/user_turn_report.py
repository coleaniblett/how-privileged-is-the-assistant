"""Assembles results/user_turn_2026-11-06/user_turn.md from the CSVs of user_traits_exchanges.py, user_traits_v3_check.py and the existing 200-prefill reports (read only). PROVISIONAL: judge labels are not human-validated."""
import os

import pandas as pd

OUT = "results/user_turn_2026-11-06"
P = lambda x: "n/a" if x != x else f"{x:+.1f}"


def main():
    J = pd.read_csv(os.path.join(OUT, "user_traits_exchanges_judge.csv"))
    K = pd.read_csv(os.path.join(OUT, "user_traits_exchanges_keyword.csv"))
    V = open(os.path.join(OUT, "user_traits_v3_table.md")).read()
    read = open(os.path.join(OUT, "read.md")).read()
    L = ["# User-turn (prefill) overgeneralization: judge labels released, PROVISIONAL (exploratory supporting rows)", "",
         "**PROVISIONAL: the judge labels (Cole's v2 judge, `claude-sonnet-5-5`) are not human-validated**; released for analysis and reporting by Adeeb's decision of 2026-11-06 (`docs/decisions.md`). `sheet_key.csv`, `assignment.csv` and hand labels were not opened. "
         "**OLMo's existing 200-prefill WildChat verdict is unchanged** (`results/capstone/olmo3-7b/overgeneralization/user_traits_judge.md`); every row below was added after that verdict was known and is supporting only. "
         "Rule and code are the existing ones (`measures/overgeneralization/user_traits.py`): per prefill, the share of user-voice samples labelled `user_voice_with_assistant_traits` (primary; secondary = over all samples), stage minus base paired per prefill, 95% bootstrap over prefills; "
         "the bar is +5 points with the range excluding 0. oasst1, UltraChat and WildChat are analysed separately, never pooled. 'Dropped' = prefills with no user-voice sample in the base or the stage (primary) after excluding samples the judge refused ('invalid'). "
         "Limits: one judge, not validated; many prefills dropped because base models often switch to the assistant (or degenerate) rather than continue as the user, so the primary row rests on a minority of prefills; "
         "contamination can be checked only for OLMo and was not applied to these sets; Qwen, Llama and Apertus supporting only; the label `user_voice_with_assistant_traits` is rare (see the read).", "",
         "## 1. Judge arm: stage minus base (points; primary rows, secondary below)", "", "| family | prefill set | stage | prefills used | dropped | base | final/stage | change [95% CI] | reading | invalid base / stage |", "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ("primary", "secondary"):
        if arm == "secondary":
            L += ["", "Secondary (share over all samples):", "", "| family | prefill set | stage | prefills used | dropped | base | stage | change [95% CI] | reading | invalid base / stage |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in J[J.arm == arm].itertuples():
            L.append(f"| {r.family} | {r.prefill_set} | {r.stage} | {r.prefills_used} | {r.dropped} | {r.base_pp:.1f} | {r.stage_pp:.1f} | {P(r.change_pp)} [{P(r.lo_pp)}, {P(r.hi_pp)}] | {r.reading} | {r.invalid_base} / {r.invalid_stage} |")
    L += ["", "For context, the already-analysed 200-prefill WildChat sets (OLMo = the verdict set; Qwen and Gemma supporting) are in `results/capstone/<family>/overgeneralization/user_traits_judge.md`; Apertus has no 200-set judge labels, so only its exchange sets are here. "
          "Llama's WildChat row is the 200-prefill set (`judged_p200_*`).", ""]
    L += ["## 2. Keyword cross-check (no judge): any-flag share and flags per 100 words, stage minus base, paired per prefill", "",
          "Same code as `user_traits.py keyword` (stock phrases and markdown the prefill did not contain, in the user portion cut at turn markers and role labels); 200 prefills per set, so none dropped.", "",
          "| family | prefill set | stage | metric | base | stage | change [95% CI] |", "|---|---|---|---|---|---|---|"]
    for r in K.itertuples():
        L.append(f"| {r.family} | {r.prefill_set} | {r.stage} | {r.metric} | {r.base_pp:.1f} | {r.stage_pp:.1f} | {P(r.change_pp)} [{P(r.lo_pp)}, {P(r.hi_pp)}] |")
    L += ["", "## 3. Cole's judge v3 `assistant_traits` on the 5 open openers (read-only; EXPLORATORY: v3 is not validated and the open openers are not the pre-registered prefills)", "",
          "Samples with user text only; change from base is a bootstrap that resamples samples within each opener, so it **ignores spread between the 5 openers**. Per-opener rows: `user_traits_v3_per_opener.csv`.", "", V,
          "Gemma instruct has no user text at all (every sample is a switch to the assistant or degenerate), so there is nothing to read there."]
    L += ["", "## 4. The read: do the `user_voice_with_assistant_traits` samples at final look right?", "", read]
    B = pd.read_csv(os.path.join(OUT, "user_traits_keyword_by_label.csv"))
    f1 = lambda x: "-" if x != x else f"{x:.1f}"
    L += ["", "## 5. Is the keyword rise mid-turn switching? The keyword any-flag share split by the v2 judge label", "",
          "Question (Adeeb, 2026-11-06): the keyword rise in Llama and Apertus might be samples that switch to the assistant mid-turn rather than assistant traits in the user's voice. For each family and prefill set, at base and final (instruct): the share of samples with a keyword flag among samples the v2 judge labelled "
          "user voice (`user_voice` + `user_voice_with_assistant_traits`), `switched_to_assistant` and `degenerate` (invalid excluded), with sample counts; the user-voice share of samples; and the change from base in the user-voice row, paired per prefill (95% bootstrap over prefills; prefills with no user-voice sample at either stage are dropped). "
          "Labels are PROVISIONAL. Apertus has no WildChat labels (no 200-set judge run), so its rows are the exchange sets; Gemma has no exchange-set labels, so only its WildChat set. CSV: `user_traits_keyword_by_label.csv`.", "",
          "| family | prefill set | label class | samples base / final | keyword flag % base | keyword flag % final | user-voice share of samples base / final (%) | user-voice change [95% CI], prefills used / dropped |", "|---|---|---|---|---|---|---|---|"]
    for r in B.itertuples():
        uv = r.cls.startswith("user voice")
        L.append(f"| {r.family} | {r.prefill_set} | {r.cls.split(' (')[0]} | {r.n_base} / {r.n_final} | {f1(r.flag_base)} | {f1(r.flag_final)} | " + (f"{r.uv_share_base:.1f} / {r.uv_share_final:.1f}" if uv else "") + " | "
                 + (f"{P(r.change)} [{P(r.lo)}, {P(r.hi)}], {int(r.prefills_used)} / {int(r.dropped)}" if uv else "") + " |")
    L += ["", "**Read.** Mid-turn switching is part of the story but not all of it. At final, the keyword flag is much more common among `switched_to_assistant` samples (Llama 46 to 64%, Apertus 53 to 58%) than among user-voice samples, and those switched samples are most of the final samples for Apertus (the user-voice share falls from 34 to 10% and 46 to 12.5%) and Llama's exchange sets (49 to 34% and 52 to 27%). "
          "But **within the samples the judge calls user voice, the keyword share also rises for Llama and Apertus**: Llama +30.0 [+19.6, +40.4] (oasst1), +14.7 [+4.9, +24.6] (UltraChat), +9.0 [+2.7, +15.4] (WildChat); Apertus +21.1 [+3.7, +38.6] and +48.2 [+33.7, +62.1] (on 41 and 47 prefills only). "
          "OLMo falls (-9.2 [-13.9, -4.8] on WildChat), Qwen is flat or down (UltraChat +5.2 [-3.3, +14.1], the only upward point), and Gemma has too few user-voice samples at final (21 of 597) to read. So the keyword rise is not only switched samples being counted. "
          "Why the judge still rarely says `user_voice_with_assistant_traits` for these samples is not settled here: the keyword features (stock phrases, markdown the prefill did not contain) can sit in user text the judge treats as a normal long request (the read in section 4 found about half of the rare UVAT samples doubtful for the same reason), "
          "the judge's `user_voice` class may include text that continues into assistant-style content, and the keyword code cuts at turn markers and role labels while the judge reads the whole continuation. Provisional; no verdict changes.", ""]
    open(os.path.join(OUT, "user_turn.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
