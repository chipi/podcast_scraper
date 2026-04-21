# Local CI-related files (`config/ci/`)

Everything in this directory except this **`README.md`** is **gitignored**.

## Acceptance fast matrix

The **tracked** fast list for **`--from-fast-stems`** / **`FAST_ONLY`** is
**`config/acceptance/FAST_CONFIG.yaml`** (matrix of `defaults` + `runs`). Edit that file (or use a
branch) to add or disable rows — there is **no** alternate stem list in `config/ci/` for the
acceptance runner.
