# Contributing Guide

Thanks for contributing! This guide gets you from zero to your first PR.

---

## New Contributor Quick Path

**Read these docs in order (30-45 min):**

| Step | Document | Time | What You'll Learn |
| ---- | -------- | ---- | ----------------- |
| 1 | [README](README.md) | 5 min | What this project does |
| 2 | [Architecture](docs/ARCHITECTURE.md) | 10 min | How it's structured |
| 3 | [Quick Reference](docs/guides/QUICK_REFERENCE.md) | 5 min | Common commands |
| 4 | [Testing Guide](docs/guides/TESTING_GUIDE.md) | 5 min | How to run tests |
| 5 | [Development Guide](docs/guides/DEVELOPMENT_GUIDE.md) | 10 min | Code patterns |
| 6 | Pick an issue! | - | [Good first issues](https://github.com/chipi/podcast_scraper/labels/good%20first%20issue) |

---

## Setup

```bash

# Clone and enter

git clone https://github.com/chipi/podcast_scraper.git
cd podcast_scraper

# Create environment

python3 -m venv .venv
source .venv/bin/activate

# Install package and dependencies

make init  # Installs package with pip install -e ".[ml]" + dev tools + pre-commit hooks

# Configure environment (optional but recommended)

cp examples/.env.example .env

# Edit .env if you need OpenAI API keys or custom settings

# Verify setup works

make ci
```yaml

**What `make init` does:**

- Creates virtual environment (if not exists)
- Installs package in editable mode with all dependencies
- Installs pre-commit hooks
- Sets up development tools (black, isort, flake8, mypy)

**When you need `.env` configuration:**

- Using OpenAI providers (need `OPENAI_API_KEY`)
- Custom output paths or cache directories
- Performance tuning for your hardware
- Debug logging (`LOG_LEVEL=DEBUG`)

See [`examples/.env.example`](examples/.env.example) for all options.

---

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/my-feature   # or fix/bug-name, docs/update-xyz
```

### 2. Make Changes

```bash

# Format and lint as you go

make format
make lint
```

## 3. Test

```bash
make test-unit          # Fast feedback (~30s)
make ci                 # Full suite before PR
```

### 4. Commit

```bash
git add -A
git commit -m "feat: add my feature"   # Use conventional commits
```

**Commit format:** `type: description`

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code refactoring

### 5. Push and Create PR

```bash
git push -u origin feature/my-feature
```yaml

Then open a PR on GitHub.

---

## PR Checklist

Before submitting:

- [ ] `make ci` passes locally
- [ ] Tests added/updated for changes
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventional format
- [ ] PR description explains what and why

---

## Getting Help

| Resource | When to Use |
| -------- | ----------- |
| [Troubleshooting](docs/guides/TROUBLESHOOTING.md) | Common issues |
| [Glossary](docs/guides/GLOSSARY.md) | Unfamiliar terms |
| [Development Guide](docs/guides/DEVELOPMENT_GUIDE.md) | Code patterns, style |
| [GitHub Issues](https://github.com/chipi/podcast_scraper/issues) | Questions, bugs |

---

## For AI Assistants

If you're using Cursor, Claude, or Copilot, the project includes AI-specific guidelines:

- **Primary:** `.ai-coding-guidelines.md`
- **Cursor:** `.cursor/rules/ai-guidelines.mdc`
- **Claude:** `CLAUDE.md`
- **Copilot:** `.github/copilot-instructions.md`

**Key rules:** Never commit without approval. Always run `make ci` before pushing.

---

Thanks for contributing!
