# Local Development & Debugging

This document covers local development setup, pre-commit hooks, and debugging CI failures.

For workflow details, see [Workflows](WORKFLOWS.md).
For architecture overview, see [Overview](OVERVIEW.md).

---

## Exit Codes in CI

The CLI returns standard exit codes that CI/CD pipelines can use:

- **Exit code 0**: Success (pipeline completed successfully)
- **Exit code 1**: Error (validation, configuration, or pipeline failure)

GitHub Actions workflows automatically handle exit codes - a non-zero exit code will fail the job. For detailed exit code semantics and usage examples, see the [CLI Reference - Exit Codes](../api/CLI.md#exit-codes).

## Local Development

### Automatic Pre-commit Checks

**Prevent linting failures before they reach CI!**

Install the git pre-commit hook to automatically check your code before every commit:

```bash

# One-time setup

make install-hooks

```python

- ✅ **Black** formatting check
- ✅ **isort** import sorting check
- ✅ **flake8** linting
- ✅ **markdownlint** (if installed)
- ✅ **mypy** type checking

**If any check fails, the commit is blocked** until you fix the issues.

## Skip Hook (Not Recommended)

```bash

# Skip pre-commit checks for a specific commit (not recommended)

git commit --no-verify -m "your message"

```bash

# If hook fails, auto-fix formatting issues

make format

# Then try committing again

git commit -m "your message"

```text

# Run full CI suite (matches GitHub Actions PR validation)

# - Runs unit + critical path integration + critical path e2e tests

# - Full validation before commits/PRs

# - Note: No cleanup step (faster), use ci-full for complete validation

make ci

# Fast CI checks (quick feedback during development)

# - Skips cleanup step (faster)

# - Runs unit + critical path integration + critical path e2e (no coverage)

# - Use for quick validation during development

make ci-fast

# Complete CI suite (all tests including slow/ml_models)

# - Cleans cache first (clean-all)

# - Runs all tests: unit + integration + e2e (all slow/fast variants)

# - Use for complete validation before releases

make ci-full

# Individual checks (same as CI)

make format-check  # Black & isort
make lint          # flake8
make lint-markdown # markdownlint
make type          # mypy
make security      # bandit & pip-audit
make complexity    # radon cyclomatic complexity
make deadcode      # vulture dead code detection
make docstrings    # interrogate docstring coverage
make spelling      # codespell spell checking
make quality       # all quality checks (complexity, deadcode, docstrings, spelling)
make test-unit     # pytest with coverage (parallel, unit tests only)
make test-integration      # All integration tests (parallel, with re-runs)
make test-e2e             # All E2E tests (parallel, with re-runs, network guard)
make docs          # mkdocs build
make build         # package build

# For debugging: use pytest directly with -n 0 for sequential execution

```yaml

- **`make ci`**: Full validation before commits/PRs (unit + fast integration + fast e2e tests), matches GitHub Actions PR validation exactly
- **`make ci-fast`**: Quick feedback during development (unit + fast integration + fast e2e, no coverage), faster iteration
- **`make ci-full`**: Complete validation with all tests including slow/ml_models tests (unit + integration + e2e, all variants), use before releases

## Local CI Validation Flow

```mermaid

graph TD
    A[Local Development] --> B{git commit}

    B --> C[Pre-commit Hook]
    C --> C1[format-check]
    C --> C2[lint]
    C --> C3[lint-markdown]
    C --> C4[type]

    C1 & C2 & C3 & C4 --> D{Hook Pass?}
    D -->|No| E[Commit Blocked]
    E --> F[make format to fix]
    F --> A

    D -->|Yes| G[Commit Created]
    G --> H[git push]

    H --> I{make ci}
    I --> J[All CI Checks]
    J --> K{CI Pass?}
    K -->|Yes| L[PR Ready]
    K -->|No| M[Fix Issues]
    M --> A

    style L fill:#90EE90
    style E fill:#FFB6C6
    style M fill:#FFB6C6
    style G fill:#87CEEB

```text

### ✅ Prevention

- **Pre-commit hooks:** Catch issues before they're committed
- **Local CI validation:** `make ci` runs full suite before push
- **Auto-fix formatting:** `make format` fixes issues automatically

### ✅ Speed

- **Parallel execution:** All independent jobs run simultaneously
- **Caching:** Pip cache for faster dependency installation
- **Early feedback:** Fast lint job without ML dependencies

### ✅ Reliability

- **Reproducible:** Same checks run locally via `make ci`
- **Isolated:** Jobs don't depend on each other (except docs deploy)
- **Clean environment:** Each job starts fresh, post-cleanup prevents cache pollution

### ✅ Security

- **Multi-layered:** CodeQL + bandit + safety
- **Continuous:** Weekly scheduled scans
- **Early detection:** Security checks on every PR

### ✅ Documentation

- **Validated:** Docs build checked on every PR
- **Automated:** Deployment on merge to main
- **Complete:** Code + architecture + API reference

### ✅ Developer Experience

- **Fast feedback:** Lint results in 2-3 minutes
- **Local parity:** `make ci` runs same checks as GitHub
- **Quick iteration:** `make ci-fast` for rapid development feedback
- **Clear errors:** Strict mode for docs and type checking

---

## Monitoring & Debugging

### Viewing Workflow Results

1. **GitHub Actions Tab:** [View all runs](https://github.com/chipi/podcast_scraper/actions)
2. **PR Checks:** Status checks appear on pull requests
3. **Branch Protection:** Can require specific jobs to pass before merge

### Common Issues & Solutions

| Issue | Cause | Solution |
| ----- | ----- | -------- |
| Test timeout | Large ML models download | Already handled by disk space management |
| Lint failures | Formatting issues | Run `make format` locally before push |
| Docs build failure | Broken links or invalid syntax | Run `make docs` locally, check `mkdocs build` output |
| CodeQL alerts | Security vulnerabilities | Review in Security tab, address findings |
| Out of disk space | ML model caches | Cleanup is automatic, check disk usage logs |

### Debugging Failed Runs

```bash

# Reproduce lint failures locally

make format-check lint lint-markdown type security

# Reproduce test failures locally

make test

# Reproduce E2E test failures locally

make test-e2e  # All E2E tests
make test-e2e-fast      # Critical path E2E tests only
make test-e2e-slow      # Slow E2E tests only (requires ML dependencies)

# Reproduce docs failures locally

make docs

# Run everything (matches full CI)

make ci

```bash

# Run all E2E tests (with network guard)

make test-e2e

# Run critical path E2E tests only (faster feedback)

make test-e2e-fast

# Run slow E2E tests only (includes slow/ml_models, requires ML dependencies)

make test-e2e-slow

```python

- All RSS and audio must be served from local E2E HTTP server
- Tests fail hard if a real URL is hit

**Test Markers:**

- `e2e`: All E2E tests
- `slow`: Slow tests (Whisper, ML models)
- `ml_models`: Tests requiring ML dependencies

See [E2E Testing Guide](../guides/E2E_TESTING_GUIDE.md) for detailed E2E test documentation.

---
