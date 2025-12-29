# Contributing Guide

Thanks for taking the time to contribute! This guide will help you get started quickly.

**Table of Contents**

- [Quick Start](#quick-start)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Getting Help](#getting-help)

> **📚 For detailed technical information**, see
> [`docs/guides/DEVELOPMENT_GUIDE.md`](docs/guides/DEVELOPMENT_GUIDE.md) which covers:
>
> - Code style guidelines and formatting
> - Testing requirements and test structure
> - CI/CD integration details
> - Architecture principles
> - Logging guidelines
> - Documentation standards
> - Markdown linting
> - Environment setup details

---

## Quick Start

### 1. Set Up Your Environment

````bash

# Clone the repository

git clone https://github.com/chipi/podcast_scraper.git
cd podcast_scraper

# Option 1: Use setup script (recommended)

bash scripts/setup_venv.sh
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Option 2: Manual setup

python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install development dependencies and the package itself

make init
```text

# Copy example .env file

cp examples/.env.example .env

# Edit .env and add your settings

# OPENAI_API_KEY=sk-your-actual-key-here

# LOG_LEVEL=DEBUG

# OUTPUT_DIR=/tmp/test_output

# WORKERS=2

```text
- `black`/`isort` formatting checks
- `flake8` linting
- `markdownlint` for markdown files
- `mypy` type checking
- `bandit` + `pip-audit` security scans
- `pytest` with coverage report
- `mkdocs build` documentation build (outputs to `.build/site/`)
- `python -m build` packaging sanity check (outputs to `.build/dist/`)

> **Note:** Build artifacts (distributions, documentation site, coverage reports) are organized in `.build/` directory. Test outputs are stored in `.test_outputs/`. Use `make clean` to remove all build artifacts.

### 3. Common Commands

Use the Makefile targets to work faster:

```bash

make help          # List all targets
make format        # Auto-format with black + isort
make format-check  # Formatting check without modifying files
make lint          # Run flake8 linting
make lint-markdown # Run markdownlint on markdown files
make type          # Run mypy type checks
make security      # Run bandit + pip-audit security scans
make test          # Run pytest with coverage
make docs          # Build MkDocs documentation
make build         # Build sdist & wheel
make ci            # Run the full CI suite locally
make clean         # Remove build artifacts

```text
- Detailed code style guidelines
- Testing requirements and test structure
- Running tests (unit, integration, E2E)
- Pre-commit hooks setup

---

## Development Workflow

### Git Workflow

#### Branch Naming

Create descriptive branches for all changes:

```bash

# Feature branches

git checkout -b feature/add-postgresql-export
git checkout -b feature/issue-40-etl-loading

# Bug fix branches

git checkout -b fix/whisper-progress-indicator
git checkout -b fix/issue-19-progress-bars

# Documentation branches

git checkout -b docs/update-api-reference
git checkout -b docs/contributing-guide

```text

<detailed description if needed>

Fixes #<issue-number>

```yaml

- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test changes
- `refactor`: Code refactoring
- `ci`: CI/CD changes
- `chore`: Maintenance tasks
- `perf`: Performance improvements

**Examples:**

```text

feat: add PostgreSQL export adapter

Implement export functionality to generate PostgreSQL-compatible SQL
dumps from episode metadata. Includes schema templates and CLI flags.

Fixes #40

```markdown

Fixes #19

```text

# 1. Install pre-commit hook (one-time setup)

make install-hooks

# 2. Make your changes

# ... edit files ...

# 3. Commit (pre-commit hook runs automatically)

git commit -m "your message"

# Hook checks formatting, linting, types, markdown

# 4. Full CI check before pushing

make ci

# 5. Push to remote

git push

```bash

# Full CI check before push

make ci

```text
- All formatting checks pass
- All lints pass
- All type checks pass
- All security scans pass
- All tests pass (>80% coverage)
- Documentation builds successfully
- Package builds successfully

**See [`docs/guides/DEVELOPMENT_GUIDE.md`](docs/guides/DEVELOPMENT_GUIDE.md) for:**

- Detailed CI/CD integration information
- Pre-commit hooks configuration
- Common CI failure solutions
- Path-based CI optimization

---

## Pull Request Process

### Before Creating PR

**1. Ensure all checks pass:**

```bash

make ci

```text
- [ ] README if user-facing changes
- [ ] API docs if public API changes
- [ ] Architecture docs if design changes
- [ ] Add/update tests

**3. Commit with clear messages:**

- Follow conventional commit format
- Reference issue numbers
- Explain "why" not just "what"

## Creating the PR

**1. Create feature branch:**

```bash

git checkout -b feature/my-feature

```javascript
- **Clear title** (e.g., "Add PostgreSQL export adapter")
- **Description** that includes:
  - Problem being solved
  - Solution approach
  - Testing performed
  - Related issues (Fixes #XX)
  - Breaking changes (if any)

**PR Template Example:**

```markdown

## Summary

Add PostgreSQL export functionality to generate SQL dumps from episode metadata.

## Changes

- Added `export.py` module with SQL generation
- Added `--export-format sql` CLI flag
- Created SQL schema templates
- Updated documentation

## Testing

- Unit tests for SQL generation
- Integration test for full export
- Manual testing with PostgreSQL 14

## Related

Fixes #40

## Breaking Changes

None

```text
1. **Automated checks run** (CI, linting, tests)
2. **Maintainer reviews code**
3. **Address feedback** if requested
4. **CI re-runs** after changes
5. **Approval and merge** once all checks pass

### After PR Merge

- Your branch will be deleted (automatic)
- Changes will be included in next release
- Release notes will be updated

**See [`docs/guides/DEVELOPMENT_GUIDE.md`](docs/guides/DEVELOPMENT_GUIDE.md) for:**

- Code style guidelines and naming conventions
- Testing requirements (unit, integration, E2E)
- Documentation standards (PRDs, RFCs, docstrings)
- Architecture principles and module boundaries
- Logging guidelines

---

## Getting Help

- **Documentation**: Check `docs/` directory first
- **Development Guide**: See [`docs/guides/DEVELOPMENT_GUIDE.md`](docs/guides/DEVELOPMENT_GUIDE.md) for detailed technical information
- **Issues**: Search existing issues or create a new one
- **Architecture**: See `docs/ARCHITECTURE.md` for design principles
- **Testing**: See `docs/TESTING_STRATEGY.md` for testing guidelines
- **CI/CD**: See `docs/CI_CD.md` for CI/CD pipeline details

---

## AI Coding Guidelines

This project includes comprehensive AI coding guidelines for developers using AI assistants (Cursor, Claude Desktop, GitHub Copilot, etc.).

**For AI Assistants:**

- **Primary reference:** `.ai-coding-guidelines.md` - This is the PRIMARY source of truth for all AI actions.
- **Entry points by tool:**
  - **Cursor:** `.cursor/rules/ai-guidelines.mdc` - Automatically loaded by Cursor
  - **Claude Desktop:** `CLAUDE.md` - Automatically loaded when Claude runs in this directory
  - **GitHub Copilot:** `.github/copilot-instructions.md` - Read by GitHub Copilot
- **Prompt templates:** `.cursor/prompts/` - Reusable prompt templates for CI debugging, RFC design, code reviews, and implementation planning (see [`docs/guides/CURSOR_AI_BEST_PRACTICES_GUIDE.md`](docs/guides/CURSOR_AI_BEST_PRACTICES_GUIDE.md))

**Critical workflow rules (from `.ai-coding-guidelines.md`):**

- ❌ **NEVER commit without showing changes and getting user approval**
- ❌ **NEVER push to PR without running `make ci` first**
- ✅ Always show `git status` and `git diff` before committing
- ✅ Always wait for explicit user approval before committing
- ✅ Always run `make ci` before pushing to PR (new or updated)

**What `.ai-coding-guidelines.md` contains:**

- Git Workflow (commit approval, PR workflow)
- Code Organization (module boundaries, patterns)
- Testing Requirements (mocking, test structure)
- Documentation Standards (PRDs, RFCs, docstrings)
- Common Patterns (configuration, error handling, logging)
- Decision Trees (when to create modules, PRDs, RFCs)

**For human contributors:** These guidelines help ensure AI assistants follow project standards. You don't need to read them unless you're configuring AI tools.

**See:** `.ai-coding-guidelines.md` for complete guidelines. For detailed technical patterns, see [`docs/guides/DEVELOPMENT_GUIDE.md`](docs/guides/DEVELOPMENT_GUIDE.md).

---

Thanks again for contributing! If you have questions, please open an issue or discussion.
````
