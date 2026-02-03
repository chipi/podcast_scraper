# Contributing Guide

Thanks for contributing! This guide gets you from zero to your first PR.

---

## New Contributor Quick Path

**Read these docs in order (30-45 min):**

| Step | Document | Time | What You'll Learn |
| ---- | -------- | ---- | ----------------- |
| 1 | [README](README.md) | 5 min | What this project does |
| 2 | [Engineering Process](docs/guides/ENGINEERING_PROCESS.md) | 5 min | **The "Triad of Truth": PRDs, RFCs, and ADRs** |
| 3 | [Architecture](docs/ARCHITECTURE.md) | 10 min | How it's structured |
| 3 | [Quick Reference](docs/guides/QUICK_REFERENCE.md) | 5 min | Common commands |
| 4 | [Testing Guide](docs/guides/TESTING_GUIDE.md) | 5 min | How to run tests |
| 5 | [Development Guide](docs/guides/DEVELOPMENT_GUIDE.md) | 10 min | Code patterns |
| 6 | Pick an issue! | - | [Good first issues](https://github.com/chipi/podcast_scraper/labels/good%20first%20issue) |

---

## Setup

### Prerequisites

Before you begin, ensure you have these installed:

- **Python 3.10+ (REQUIRED)** — The project requires Python 3.10 or higher
  - Check system Python: `python3 --version`
  - If your system Python is < 3.10, install a newer version:
    - macOS: `brew install python@3.11` (or use [pyenv](https://github.com/pyenv/pyenv))
    - Linux: Use your package manager or [pyenv](https://github.com/pyenv/pyenv)
- **ffmpeg** — Required for Whisper transcription
  - macOS: `brew install ffmpeg`
  - Linux: `apt install ffmpeg` or `yum install ffmpeg`
- **Node.js and npm** — Required for markdown linting
  - Check with `node --version` and `npm --version`
  - Install from [nodejs.org](https://nodejs.org/) if missing
- **make** — Usually pre-installed on macOS/Linux
  - Check with `make --version`

### Installation Steps

```bash

# Clone and enter

git clone https://github.com/chipi/podcast_scraper.git
cd podcast_scraper

# Create and activate virtual environment
# ⚠️ IMPORTANT: Always activate the venv before running make commands

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# ⚠️ CRITICAL: Verify Python and pip versions in the venv
# The venv's Python must be >= 3.10 for the project to work correctly

python --version   # Should show Python 3.10.x or higher
pip --version      # Should show pip with Python 3.10+ support

# If Python version is < 3.10, recreate the venv with a newer Python:
# deactivate
# rm -rf .venv
# python3.11 -m venv .venv  # Use specific version if needed
# source .venv/bin/activate

# ⚠️ CRITICAL: Upgrade pip and setuptools BEFORE installing in editable mode
# This is required for PEP 660 support (editable installs with pyproject.toml)
# The make init command does this automatically, but if you install manually:

pip install --upgrade pip setuptools wheel

# Install package and dependencies

make init  # Installs package with pip install -e ".[ml]" + dev tools + pre-commit hooks

# Install markdown linting tool (required for make ci)

npm install -g markdownlint-cli

# Configure environment (optional but recommended)

cp examples/.env.example .env

# Edit .env if you need OpenAI API keys or custom settings

# Verify setup works

make ci
```

**What `make init` does:**

- Upgrades pip and setuptools (required for PEP 660 editable installs)
- Creates virtual environment (if not exists)
- Installs package in editable mode with all dependencies
- Installs pre-commit hooks
- Sets up development tools (black, isort, flake8, mypy)

**⚠️ Important Notes:**

- **Python 3.10+ is REQUIRED** — The project uses features that require Python 3.10 or higher. Always verify the venv's Python version with `python --version` after activation. If it's < 3.10, recreate the venv with a newer Python version.
- **Always activate your virtual environment** before running `make` commands. The Makefile uses tools installed in `.venv/bin/`, so they won't be found if the venv isn't activated.
- **markdownlint-cli** is required for `make ci` to pass. Install it globally with `npm install -g markdownlint-cli`.
- **Editable installs require modern setuptools** — If you see `"editable mode currently requires a setuptools-based build"` error:
  1. Ensure pip and setuptools are upgraded: `pip install --upgrade pip setuptools wheel`
  2. Verify setuptools version: `python -c "import setuptools; print(setuptools.__version__)"` (should be >= 64.0.0)
  3. The `make init` command handles this automatically, but if installing manually, upgrade first
- If `make ci` fails with "command not found" errors, ensure:
  1. Your virtual environment is activated (`source .venv/bin/activate`)
  2. The venv's Python is >= 3.10 (check with `python --version`)
  3. You've run `make init` to install all Python dependencies (which upgrades pip/setuptools)
  4. You've installed `markdownlint-cli` via npm

**When you need `.env` configuration:**

- Using OpenAI providers (need `OPENAI_API_KEY`)
- Custom output paths or cache directories
- Performance tuning for your hardware
- Debug logging (`LOG_LEVEL=DEBUG`)

See [`examples/.env.example`](examples/.env.example) for all options.

---

## IDE Configuration

**⚠️ IMPORTANT:** Your IDE must be configured to use the virtual environment's Python interpreter. Otherwise, it won't find installed packages, type checking won't work, and autocomplete will fail.

### Visual Studio Code / Cursor

1. **Open the Command Palette:**
   - VS Code: `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows/Linux)
   - Cursor: Same keyboard shortcuts

2. **Select Python Interpreter:**
   - Type: `Python: Select Interpreter`
   - Choose: `.venv/bin/python` (or `.venv\Scripts\python.exe` on Windows)
   - The interpreter path should show: `./.venv/bin/python` or similar

3. **Verify Configuration:**
   - Check the bottom-right status bar — it should show the venv Python version
   - Open a Python file and check that imports resolve correctly
   - Type checking (mypy) should work if configured

4. **Workspace Settings (Optional):**
   The project includes `.vscode/settings.json` with recommended settings. If you need to configure manually:

   ```json
   {
     "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
     "python.terminal.activateEnvironment": true
   }
   ```

### PyCharm

1. **Open Settings/Preferences:**
   - macOS: `PyCharm` → `Preferences`
   - Windows/Linux: `File` → `Settings`

2. **Configure Project Interpreter:**
   - Go to: `Project: podcast_scraper` → `Python Interpreter`
   - Click the gear icon → `Add...`
   - Select: `Existing environment`
   - Choose: `.venv/bin/python` (or `.venv\Scripts\python.exe` on Windows)
   - Click `OK`

3. **Verify:**
   - The interpreter should show: `Python 3.10.x` or higher
   - Installed packages should be visible in the interpreter list

### Other IDEs

**General approach for any IDE:**

1. Find the Python interpreter settings (usually in Preferences/Settings)
2. Point it to: `.venv/bin/python` (macOS/Linux) or `.venv\Scripts\python.exe` (Windows)
3. Verify by:
   - Checking that imports resolve
   - Running a simple script that imports `podcast_scraper`
   - Ensuring type checking works (if supported)

**Common IDE locations:**

- **Sublime Text:** Install "LSP-pyright" or "Anaconda" package, configure Python path
- **Vim/Neovim:** Configure LSP (pyright/pylsp) to use `.venv/bin/python`
- **Emacs:** Configure `python-shell-interpreter` to `.venv/bin/python`

### Troubleshooting IDE Issues

**Problem:** IDE shows "Module not found" errors even after configuring interpreter.

**Solutions:**

1. Restart the IDE after configuring the interpreter
2. Verify the venv exists: `ls .venv/bin/python` (macOS/Linux) or `dir .venv\Scripts\python.exe` (Windows)
3. Ensure you've run `make init` to install packages
4. Check that the interpreter path is absolute or relative to workspace root
5. For VS Code/Cursor: Reload window (`Cmd+Shift+P` → "Developer: Reload Window")

**Problem:** Type checking (mypy) doesn't work in IDE.

**Solutions:**

1. Ensure mypy is installed: `pip list | grep mypy` (should show in venv)
2. Install IDE extension for mypy (if available)
3. Check IDE settings for type checking configuration
4. Verify `.venv/bin/mypy` exists and works: `.venv/bin/mypy --version`

---

## Development Workflow

### 1. Create a Branch

**⚠️ CRITICAL: Check for uncommitted changes first!**

```bash
# Quick check - should return nothing if clean
git status --porcelain

# If you see output, handle changes first (commit, stash, or discard)
```

**Then create your branch:**

```bash
git checkout -b feature/my-feature   # or fix/bug-name, docs/update-xyz
```

**💡 Why this matters:** Uncommitted changes get included in your new branch, causing confusing PRs with unrelated files.

**📖 For full branch creation checklist:** See [Development Guide - Branch Creation](docs/guides/DEVELOPMENT_GUIDE.md#critical-workflow-rules)

**🔄 Advanced: Using Git Worktrees**

If you want to work on multiple features simultaneously without branch switching, consider using git worktrees. This allows you to have multiple branches checked out in separate directories, each with its own virtual environment and IDE instance.

See [Git Worktree Guide](docs/guides/GIT_WORKTREE_GUIDE.md) for complete instructions on setting up and using worktrees.

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
| [Git Worktree Guide](docs/guides/GIT_WORKTREE_GUIDE.md) | Parallel development with multiple branches |
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
