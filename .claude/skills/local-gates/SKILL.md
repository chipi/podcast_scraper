---
name: local-gates
description: Run the local checks that match what changed, and only those, each ending PASS or FAIL — black/isort/flake8/mypy/docstrings and the tests that name a changed module for Python, markdownlint with the repo config and the strict docs build for Markdown, vitest-related and vue-tsc for the two web apps, actionlint for workflows. Use when about to commit or push, or asked to "run the gates", "lint what I changed", "check this before pushing", or "is this ready".
---

# local-gates

```bash
make gates-changed                    # vs origin/main: committed + staged + unstaged + untracked
make gates-changed STAGED=1           # only what the next commit holds
make gates-changed SINCE=<ref>        # vs another base
```

`scripts/tools/gates_changed.sh` does the work. Each gate prints `PASS <gate>` / `FAIL <gate>`;
the last line is `GATES: PASS` or `GATES: FAIL`, and so is the exit code. It only checks: to
rewrite files, `make format`.

| Changed | Gates |
| --- | --- |
| `*.py` (not under `web/`, `infra/`) | black `--check`, isort `--check-only`, flake8 |
| `src/**/*.py` | + mypy, `make docstrings` |
| `*.py` | + pytest on the changed test files and every `tests/**/test_*.py` that names a changed module (dotted path or file stem) |
| `*.md` | markdownlint with `.markdownlint.json` |
| `docs/**`, `mkdocs.yml` | `make docs` (strict) |
| `web/learning-player/{src,e2e}` | `vitest related --run`, vue-tsc (app + test config) |
| `web/gi-kg-viewer/{src,e2e}` | `vitest related --run`, vue-tsc (app config) |
| `.github/workflows/*.yml` | actionlint |

`NODE_OPTIONS` is unset inside the script: an inherited `--require` shim path makes every node
step die with MODULE_NOT_FOUND (see the top of the Makefile).

## When it is not enough

- **The pytest gate is a name match, not coverage.** A test reaching a module through another one
  is missed. The whole suite (`.venv/bin/python -m pytest tests/unit tests/integration -n 6`,
  ~6 min, ~21k tests) runs once at the end of an arc, before pushing, not after every edit.
- **No app e2e** (Playwright) and no device tiers here: `make test-app-e2e`, `make test-ios`,
  `make test-android` when the change is in those paths.
- **CI runs slower than a laptop.** A test that takes seconds locally can hit a CI timeout (a
  vitest file ran ~50x slower on the shared runner, 2026-10-16). After pushing, watch it with the
  `ci-watch` skill.

## Checked (2026-10-17)

- Over `75304e525~1..HEAD` (103 files: Python, Markdown, docs, player TS, a workflow): 11 gates,
  `GATES: PASS` in 5 min (192 test files matched).
- Over only a Makefile and a shell hook: no gate applies, which it says.

## Also

`make validate-files FILES=…` (formats in place, then runs the tests `find_impacted_tests.py`
maps) now fails on flake8/mypy errors instead of printing them and passing. That finder cannot
map most `server/` modules (`No modules found`), which is why `gates-changed` matches names
instead.
