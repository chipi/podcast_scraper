# Coverage threshold: single source of truth (WIP)

## Context

The combined coverage threshold is currently defined in multiple places:

- `Makefile` (`COVERAGE_THRESHOLD_COMBINED`)
- `.github/workflows/python-app.yml` (hard-coded `THRESHOLD=…` and `--coverage-threshold …` values)
- `pyproject.toml` (documentation/comments)

This duplication makes it easy for the values to drift and for changes to require editing multiple files.

## Proposal

Make **`Makefile` the single source of truth** for the combined coverage threshold.

- Keep `COVERAGE_THRESHOLD_COMBINED` in `Makefile`.
- In CI (`.github/workflows/python-app.yml`), **read the threshold from the Makefile at runtime**
  and reuse it everywhere in the workflow.

- Avoid embedding the numeric value in `pyproject.toml` comments; reference the Makefile instead.

## Minimal implementation sketch

### 1) Add a Makefile “getter” target

Add a small target to print the threshold:

```make
.PHONY: print-coverage-threshold-combined
print-coverage-threshold-combined:
	@echo "$(COVERAGE_THRESHOLD_COMBINED)"
```

### 2) Export threshold into CI environment

In `.github/workflows/python-app.yml`, add a step:

```bash
echo "COVERAGE_THRESHOLD_COMBINED=$(make -s print-coverage-threshold-combined)" >> $GITHUB_ENV
```bash

Then replace:

- `THRESHOLD=…` → `THRESHOLD=${COVERAGE_THRESHOLD_COMBINED}`
- `--coverage-threshold …` → `--coverage-threshold ${COVERAGE_THRESHOLD_COMBINED}`

## Notes

- This keeps the policy in the same place developers use locally (`make coverage-enforce`, etc.).
- CI and local development remain consistent automatically after any threshold change.

## CI: combining coverage from multiple jobs

The `coverage-unified` job in `.github/workflows/python-app.yml` merges **coverage.py data
files** (`reports/coverage-data.{unit,integration,e2e}`) exported from each
`test-unit` / `test-integration` / `test-e2e` job after `coverage combine` (pytest-xdist),
then runs `coverage combine` + `coverage report --fail-under=70` + `coverage xml`. This
matches local `make coverage-enforce` semantics. The earlier Cobertura XML merge path
inflated the line denominator and reported ~63% in CI while local combined runs were ~70%.

Artifact download uses `merge-multiple: false` so each job’s report keeps a distinct path
(avoiding same-filename overwrites when flattening).
