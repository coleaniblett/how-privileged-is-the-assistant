# Validation sheet, split between labellers

Each labeller has their own file, `sheet_<name>.csv`, with the same columns as `../sheet.csv`: fill `human_label` with exactly one of
`user_voice`, `switched_to_assistant`, `degenerate`, `user_voice_with_assistant_traits`, following `../README.md` (the judge prompt v2
definitions). Label independently: don't compare labels with each other, look at the judge's output, or open `../sheet_key.csv` or
`assignment.csv` until all three files are done.

12 samples are on every file (shared, to measure how well the labellers agree with each other); the other 16 on each file are
yours alone. Every file holds a similar mix of checkpoints, and its rows are shuffled.

When all files are filled in:

    python -m measures.vestigial_regression.judge combine-sheets      # merged labels -> ../sheet.csv, agreement between labellers
    python -m measures.vestigial_regression.judge validate --human measures/vestigial_regression/judge_validation/sheet.csv --judged <judged.jsonl>

`combine-sheets` takes each unshared sample's single label and each shared sample's majority label (2 of 3 or more); a shared sample with no
majority is left unlabelled in `../sheet.csv` and listed, so it does not enter the judge's agreement.
