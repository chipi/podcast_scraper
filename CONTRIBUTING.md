# Contributing Guide

Thanks for taking the time to contribute! This project mirrors its CI pipeline locally so you can catch issues before opening a pull request.

## 1. Set up your environment

```bash
# clone the repository
https://github.com/chipi/podcast_scraper.git
cd podcast_scraper

# create and activate a virtual environment (example using Python's venv)
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install development dependencies and the package itself
make init
```

> `make init` upgrades pip, installs lint/test/type/security tooling, installs runtime requirements (if `requirements.txt` exists), and then installs `podcast_scraper` in editable mode. It matches the dependencies used in CI.

## 2. Run the full check suite (matches CI)

```bash
make ci
```

This command executes the same steps as the GitHub Actions workflow:

- `black`/`isort` formatting checks
- `flake8` linting
- `mypy` type checking
- `bandit` + `pip-audit` security scans
- `pytest` with coverage report
- `mkdocs build` documentation build (outputs to `.build/site/`)
- `python -m build` packaging sanity check (outputs to `.build/dist/`)

> **Note:** Build artifacts (distributions, documentation site, coverage reports) are organized in `.build/` directory. Test outputs are stored in `.test_outputs/`. Use `make clean` to remove all build artifacts.

## 3. Common commands

Use the Makefile targets to work faster:

```bash
make help          # list all targets
make format        # auto-format with black + isort
make format-check  # formatting check without modifying files
make lint          # run flake8
make type          # run mypy
make security      # run bandit + pip-audit
make test          # run pytest with coverage
make build         # build sdist & wheel
```

## 4. Pre-commit hooks (optional but recommended)

If you prefer `pre-commit`, install it and reuse the same tools via the Makefile commands or roll your own `.pre-commit-config.yaml`. Running `make format` + `make ci` before every push keeps you aligned with CI.

## 5. Opening a pull request

1. Ensure `make ci` passes locally.
2. Push your branch and open a PR.
3. The docs workflow runs on PRs that touch `docs/`, and the Python workflow runs on every PR targeting `main`.

Thanks again for contributing!
