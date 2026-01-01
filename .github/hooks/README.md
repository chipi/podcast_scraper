# Git Hooks

This directory contains git hooks that can be installed to improve code quality and prevent issues before they reach CI.

## Pre-commit Hook

The `pre-commit` hook automatically checks your code before each commit to ensure it meets quality standards.

### Installation

```bash
make install-hooks
```python

### What It Checks

The hook runs checks **only on staged files** (files you're committing), making it fast and efficient:

- **Black** formatting check (Python files)
- **isort** import sorting check (Python files)
- **flake8** linting (Python files)
- **markdownlint** (markdown files - **required** when markdown files are staged)
- **JSON syntax validation** (JSON files - uses Python's json.tool)
- **YAML syntax validation** (YAML/YML files - uses yamllint if available, otherwise Python yaml module)
- **mypy** type checking (Python files)

> **Note:** If you're committing markdown files, `markdownlint` must be installed. Install it with: `npm install -g markdownlint-cli`
> **Note:** For better YAML validation, install `yamllint` with: `pip install yamllint` (optional - Python yaml module is used as fallback)

### Behavior

- If **all checks pass** → Commit proceeds normally
- If **any check fails** → Commit is blocked with error details

### Usage

```bash

# Normal commit (hook runs automatically)

git commit -m "your message"

# Auto-fix formatting issues before committing

make format

# Skip hook for a specific commit (not recommended)

git commit --no-verify -m "your message"
```python

## Benefits

- ✅ Catch issues locally before pushing
- ✅ Prevent CI failures from linting
- ✅ Maintain consistent code quality
- ✅ Get immediate feedback

### Troubleshooting

If the hook fails:

1. Read the error output carefully
2. Run `make format` to auto-fix formatting
3. Fix any remaining linting/type errors manually
4. Try committing again

For more information, see the [CI/CD documentation](../../docs/CI_CD.md#automatic-pre-commit-checks).
