# Local CI-related files (`config/ci/`)

Everything in this directory except this **`README.md`** is **gitignored**.

## Fast acceptance stems

The **tracked** fast matrix for `--from-fast-stems` / **`FAST_ONLY`** lives in:

**`config/acceptance/FAST_CONFIGS.txt`**

Optionally, you can add a local-only **`acceptance_fast_stems.txt`** here (same format: one stem per line, `#` comments allowed). The acceptance runner reads **`FAST_CONFIGS.txt` first**, then **`config/ci/acceptance_fast_stems.txt`** if that file exists—handy for experimenting without editing the committed list.
