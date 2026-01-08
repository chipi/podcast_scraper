# Git Worktree Development Guide

> **Quick Reference:** This guide covers the git worktree-based development workflow.
> For complete technical details and rationale, see
> [RFC-039](../rfc/RFC-039-development-workflow-worktrees-ci.md).

## Overview

This project uses **git worktrees** to enable parallel development across multiple branches.
Each worktree is an isolated working directory that shares the same Git repository history,
allowing you to work on multiple features simultaneously without branch switching or stashing.

### Key Benefits

- **Parallel Development**: Work on multiple features/branches simultaneously
- **Context Isolation**: Each worktree has its own filesystem, venv, and Cursor instance
- **No Branch Switching**: Each branch lives in its own folder
- **AI Tooling Friendly**: Cursor gets clean, focused context per worktree
- **Safe Operations**: No risk of committing to the wrong branch

## Quick Start

### Create a New Worktree (Full Setup)

```bash

# From your main repository folder

make wt-setup

# Follow the prompts:
# - Issue number (optional): 169
# - Branch type: feat
# - Short name: dependabot

# Then navigate to the new worktree

cd ../podcast_scraper-169-dependabot
source .venv/bin/activate
cursor .
```

## Daily Workflow

```bash

# Make changes, commit, push

git add -A
git commit -m "feat(#169): add Dependabot configuration"
git push origin HEAD

# Sync with main (when needed)

git fetch origin
git rebase origin/main
git push --force-with-lease

# Create PR when ready
# Use keywords in PR description to auto-close issues:
#   Fixes #169
#   Closes #169
#   Resolves #169

```

## Cleanup After PR Merge

```bash

# Remove worktree after PR is merged

make wt-remove

# Follow prompts to remove worktree and optionally delete branch

```

## Core Concepts

### What Is a Git Worktree?

A git worktree allows multiple working directories to share the same Git repository data
while each directory checks out a different branch.

**Key Properties:**

- One Git history (`.git`) shared across all worktrees
- Multiple independent folders
- Each folder has exactly one checked-out branch
- No stashing or branch switching required

### Branch Naming Convention

Include GitHub issue numbers in branch names for traceability:

| Pattern | Example | Use Case |
| --------- | --------- | ---------- |
| `feat/{issue}-{name}` | `feat/169-dependabot` | Feature linked to issue |
| `fix/{issue}-{name}` | `fix/185-memory-leak` | Bug fix linked to issue |
| `rel/{version}` | `rel/2.5` | Release preparation |
| `feat/{name}` | `feat/experimental` | No issue (exploratory) |

**Benefits:**

- `git worktree list` shows which issue each worktree addresses
- GitHub auto-links commits/PRs to issues
- Easy to find related issue from any git log

### One Branch = One Worktree = One Cursor Window

Each branch gets:

- Its own folder (e.g., `../podcast_scraper-169-dependabot`)
- Its own virtual environment (isolated dependencies)
- Its own terminal session
- Its own Cursor instance

This ensures:

- Clean Git state per context
- Correct AI context for each task
- Zero accidental cross-contamination
- Independent Python environments

## Setup Process

### Complete Setup (New Worktree + Isolated Venv)

```bash

# 1. Fetch latest from origin

git fetch origin

# 2. Create worktree with issue-linked branch

git worktree add ../podcast_scraper-169-dependabot -b feat/169-dependabot origin/main

# 3. Navigate to worktree

cd ../podcast_scraper-169-dependabot

# 4. Create isolated virtual environment

python3 -m venv .venv

# 5. Activate virtual environment

source .venv/bin/activate  # macOS/Linux

# .venv\Scripts\activate   # Windows

# 6. Install project in development mode (use venv pip, not system pip)

.venv/bin/pip install -e ".[dev]"

# Or use make init which does this automatically

make init

# 7. Verify isolation (important!)

.venv/bin/python3 -c "import sys; paths = [p for p in sys.path if 'podcast_scraper' in p.lower()]; print('Worktree paths:', paths if paths else 'None (clean!)')"

# Should print: Worktree paths: None (clean!) or show only current worktree

# 8. Open in Cursor

cursor .
```yaml

## Why Isolated Virtual Environments?

| Aspect | Shared venv | Isolated venv (Recommended) |
| -------- | ------------- | ---------------------------- |
| Disk usage | Lower (~50MB saved) | Higher (full deps per worktree) |
| Dependency conflicts | Possible | Impossible |
| Branch-specific deps | Not possible | Full support |
| Safety | Risk of pollution | Complete isolation |
| Cleanup | Complex | Simple (delete folder) |
| Cross-worktree contamination | High risk | Zero risk |

**Decision:** Use **isolated venvs** for each worktree for complete isolation and easy cleanup.

## Preventing Cross-Worktree Contamination

### Critical: Python Environment Isolation

When using multiple worktrees, it's essential to ensure each worktree uses its own isolated Python environment. Without proper isolation, you may accidentally import code from the wrong worktree, leading to confusing test failures and incorrect behavior.

#### The Problem

If `podcast-scraper` is installed in editable mode (`pip install -e .`) in the global Python or another worktree's Python, that path gets added to Python's `sys.path`. When you run tests or import modules, Python may find and use code from the wrong worktree instead of the current one.

#### The Solution

**1. Always use the worktree's venv Python:**

```bash

# ✅ CORRECT: Use venv Python

.venv/bin/python3 -m pytest tests/
make test  # Makefile uses $(PYTHON) -m pytest automatically

# ❌ WRONG: Using system pytest directly

pytest tests/  # May use system Python with wrong worktree in path
```

**2. Verify venv is set up correctly:**

```bash

# Check that .venv exists and points to the correct worktree

ls -la .venv/bin/python3

# Verify venv Python doesn't have other worktrees in its path

.venv/bin/python3 -c "import sys; paths = [p for p in sys.path if 'podcast_scraper' in p.lower()]; print('Worktree paths:', paths if paths else 'None (clean!)')"

# Should print: Worktree paths: None (clean!)

```go

**3. Never install package in global Python:**

```bash

# ❌ WRONG: Installs in global Python, pollutes all worktrees

pip install -e .

# ✅ CORRECT: Install only in worktree's venv

.venv/bin/pip install -e .

# Or use make init which does this automatically

make init
```

**4. Check for global installations:**

```bash

# Check if package is installed globally

python3 -m pip show podcast-scraper

# If found, uninstall it

python3 -m pip uninstall -y podcast-scraper
```

## Makefile Protection

The Makefile automatically uses the venv Python for all commands:

```makefile

# Makefile ensures venv Python is used

PYTHON ?= $(shell if [ -f .venv/bin/python3 ]; then echo ".venv/bin/python3"; else echo "python3"; fi)
PYTEST ?= $(PYTHON) -m pytest
```

This means:

- `make test` → Uses `.venv/bin/python3 -m pytest`
- `make test-unit` → Uses `.venv/bin/python3 -m pytest`
- All test commands → Use venv Python automatically

**However**, if you run `pytest` directly (not through `make`), you may still use the system Python.
Always use `make test` or `.venv/bin/python3 -m pytest`.

## Setup Checklist

When creating a new worktree, ensure:

- [ ] `.venv` directory exists in the worktree root
- [ ] `.venv/bin/python3` exists and is executable
- [ ] Package is installed in venv: `.venv/bin/pip list | grep podcast-scraper`
- [ ] No global installation: `python3 -m pip show podcast-scraper` should fail
- [ ] Venv Python path is clean: Run verification command (see "Verification Commands") - shows only current worktree

### Coverage Reports Isolation

Coverage reports (`.coverage`, `.coverage.*`) are generated per worktree and should not be shared:

```bash

# Coverage files are worktree-specific

.coverage              # Main coverage file
.coverage.worker-1    # Parallel worker coverage files
.coverage.worker-2

# etc.

# These are gitignored and should stay in each worktree
# Never commit or share coverage files between worktrees

```

**Coverage Best Practices:**

- Each worktree generates its own coverage reports
- Coverage files are in `.gitignore` (worktree-specific)
- Use `make coverage-report` to generate HTML reports in each worktree
- Coverage thresholds are checked per worktree independently

## Verification Commands

Run these commands to verify isolation:

```bash

# 1. Check which Python is used by make

make -n test-unit | grep python

# Should show: .venv/bin/python3 -m pytest

# 2. Verify venv Python path

.venv/bin/python3 -c "import sys; print('Python:', sys.executable); print('Paths with podcast_scraper:', [p for p in sys.path if 'podcast_scraper' in p.lower()])"

# 3. Check for global installation

python3 -m pip show podcast-scraper 2>&1 | head -3

# Should show: WARNING: Package(s) not found

# 4. Verify imports use correct worktree

.venv/bin/python3 -c "import podcast_scraper; print('Package location:', podcast_scraper.__file__)"

# Should point to current worktree's src/ directory

```python

## Troubleshooting Cross-Worktree Issues

**Symptom:** Tests fail with import errors or use code from wrong worktree

**Diagnosis:**

```bash

# Check if global Python has the package

python3 -m pip list | grep podcast-scraper

# Check if venv Python has correct path

.venv/bin/python3 -c "import sys; print([p for p in sys.path if 'podcast_scraper' in p.lower()])"
```

**Fix:**

```bash

# 1. Uninstall from global Python

python3 -m pip uninstall -y podcast-scraper

# 2. Recreate venv if needed

rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
make init

# 3. Verify fix

.venv/bin/python3 -c "import sys; print([p for p in sys.path if 'podcast_scraper' in p.lower()])"

# Should show only current worktree path

```

## Cursor Integration

### Opening Worktrees in Cursor

```bash

# From main repository

cursor ../podcast_scraper-169-dependabot

# Or from within the worktree

cd ../podcast_scraper-169-dependabot
cursor .
```python

## Configure Python Interpreter

In Cursor:

1. Open Command Palette (`Cmd+Shift+P` / `Ctrl+Shift+P`)
2. Search: "Python: Select Interpreter"
3. Choose: `.venv/bin/python` (from the worktree)

Or add to `.vscode/settings.json` (gitignored):

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python"
}
```yaml

### Files That Propagate Automatically

These files are tracked in Git and automatically available in each worktree:

| File | Purpose |
| ------ | --------- |
| `.ai-coding-guidelines.md` | AI behavior rules |
| `.cursorignore` | Files to ignore |
| `.cursorrules` | Project rules |
| `pyproject.toml` | Python config |

## Rebase Strategy

### When to Rebase

Rebase your branch when:
- ✅ PR is ready for review (clean diff for reviewers)
- ✅ CI is failing due to main changes (get latest fixes)
- ✅ Merge conflicts expected (smaller conflicts)
- ✅ Main has breaking changes you need

Don't rebase when:
- ❌ Just want latest changes for curiosity (adds noise)
- ❌ Actively coding, no blockers (rebase when ready)

### Rebase Commands

```bash

# Standard rebase workflow

git fetch origin
git rebase origin/main

# If conflicts occur

git status                    # See conflicting files

# ... resolve conflicts ...

git add <resolved-files>
git rebase --continue

# If rebase goes wrong

git rebase --abort            # Return to pre-rebase state

# Push rebased branch

git push --force-with-lease   # NEVER use --force
```yaml

## Rebase Best Practices

| Practice | Reason |
| ---------- | -------- |
| Use `--force-with-lease` | Fails if remote has unexpected changes |
| Rebase before PR review | Clean diff for reviewers |
| Don't rebase during active coding | Creates unnecessary churn |
| Resolve conflicts immediately | Don't leave rebase in progress |
| Test after rebase | Ensure nothing broke |

## Makefile Commands

### Create New Worktree

```bash

# Interactive setup (creates worktree + venv)

make wt-setup

# Basic worktree creation only

make wt-new
```

## List Worktrees

```bash

# List all worktrees with status

make wt-list
```

## Remove Worktree

```bash

# Interactive removal

make wt-remove
```

## Maintenance

```bash

# Prune stale worktree references

make wt-prune
```yaml

## Workflow Rules

### MUST (Required)

| Rule | Rationale |
| ------ | ----------- |
| **Never commit directly to `main`** | All changes via PR; branch protection enforces this |
| **One branch per worktree** | Don't switch branches inside worktrees; create new worktree instead |
| **Use `--force-with-lease` after rebase** | Never use `--force`; prevents overwriting unexpected changes |
| **Remove worktree after PR merge** | Prevents clutter; orphaned worktrees cause confusion |
| **All PRs use squash merge** | Maintains linear history on `main` |

### SHOULD (Recommended)

| Practice | Rationale |
| ---------- | ----------- |
| **Rebase before requesting review** | Clean diff for reviewers; reduces merge conflicts |
| **Include issue number in branch name** | Traceability; auto-linking in GitHub |
| **Use isolated venv per worktree** | Complete isolation; prevents dependency conflicts |
| **Always use `make test` or `.venv/bin/python3 -m pytest`** | Ensures venv Python is used; prevents cross-worktree contamination |
| **Never install package in global Python** | Prevents cross-worktree path pollution |
| **Verify venv isolation after setup** | Catches contamination issues early |
| **Run `make wt-list` weekly** | Audit for forgotten worktrees |
| **Open separate Cursor window per worktree** | Clean AI context per task |

## Emergency Recovery

### Common Issues and Fixes

**Worktree is corrupted or stuck:**

```bash

# Force remove (when normal remove fails)

git worktree remove --force ../podcast_scraper-broken

# If that fails, manually clean up

rm -rf ../podcast_scraper-broken
git worktree prune
```

**Branch was deleted but worktree still exists:**

```bash

# Prune orphaned worktree references

git worktree prune

# Then remove the folder

rm -rf ../podcast_scraper-orphaned
```

**Rebase went wrong:**

```bash

# Abort rebase and return to pre-rebase state

git rebase --abort

# If already completed but wrong, reset to remote

git fetch origin
git reset --hard origin/feat/169-dependabot
```

**Accidentally committed to main worktree:**

```bash

# From main worktree

git log --oneline -5        # Find the bad commits
git reset --soft HEAD~N     # Uncommit N commits (keeps changes)
git stash                   # Stash the changes

# Go to correct worktree

cd ../podcast_scraper-169-dependabot
git stash pop               # Apply changes here
git commit -m "feat: correct commit"
```

**Venv is broken:**

```bash

# Remove and recreate

rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate

# Use venv pip, not system pip

.venv/bin/pip install -e ".[dev]"

# Or use make init

make init
```

**Cross-worktree contamination detected:**

```bash

# 1. Check for global installation

python3 -m pip show podcast-scraper

# 2. If found, uninstall from global Python

python3 -m pip uninstall -y podcast-scraper

# 3. Verify venv is clean

.venv/bin/python3 -c "import sys; paths = [p for p in sys.path if 'podcast_scraper' in p.lower()]; print('Paths:', paths)"

# 4. If venv is contaminated, recreate it

rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
make init
```

## Weekly Maintenance

Add to your weekly routine:

```markdown

## Worktree Maintenance (Weekly)

- [ ] Run `make wt-list` - check for forgotten worktrees
- [ ] Remove worktrees for merged PRs
- [ ] Run `make wt-prune` - clean stale references
- [ ] Check disk usage: `du -sh ../podcast_scraper-*`
- [ ] Update main: `cd podcast_scraper && git pull`
- [ ] Verify no global Python installation: `python3 -m pip show podcast-scraper` (should fail)
- [ ] Check venv isolation in active worktrees: `.venv/bin/python3 -c "import sys; print([p for p in sys.path if 'podcast_scraper' in p.lower()])"`
```

## Disk Space

Per worktree:

- Working files: ~50MB
- Virtual environment: ~150MB
- Shared Git objects: 0 (reused)
- **Total per worktree: ~200MB**

With 5 active worktrees: ~1GB total

## Quick Reference

### Visual Workflow

```text
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Create    │────▶│   Develop   │────▶│    Push     │
│   Worktree  │     │   + Test    │     │   Branch    │
│             │     │             │     │             │
│ make wt-setup│     │ git commit  │     │ git push    │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  Fast CI    │
                                        │  (~2 min)   │
                                        └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  Open PR    │
                                        │  to main    │
                                        └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  Full CI    │
                                        │  (~10 min)  │
                                        └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
              ◀─────────────────────────│   Merge     │
              Cleanup worktree          │  (squash)   │
              ▼                         └─────────────┘
       ┌─────────────┐
       │   Remove   │
       │   Worktree │
       │             │
       │ make wt-remove│
       └─────────────┘
```

### Common Commands

```bash

# === NEW WORKTREE (FULL SETUP) ===

make wt-setup

# Follow prompts, then:

cd ../podcast_scraper-ISSUE-NAME
source .venv/bin/activate
cursor .

# === DAILY WORK ===

git add -A && git commit -m "feat(#169): change description"
git push origin HEAD

# === SYNC WITH MAIN ===

git fetch origin
git rebase origin/main
git push --force-with-lease

# === CREATE PR ===
# In PR description, use keywords to auto-close issues:
#   Fixes #169
#   Closes #169
#   Resolves #169

# === CLEANUP AFTER PR MERGE ===

make wt-remove

# Follow prompts

# === MAINTENANCE ===

make wt-list                         # See all worktrees
make wt-prune                        # Clean up stale refs

# === EMERGENCY ===

git rebase --abort                   # Cancel bad rebase
git worktree remove --force PATH     # Force remove
rm -rf .venv && python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

# === VERIFY ISOLATION ===

# Check which Python make uses

make -n test-unit | grep python

# Verify venv Python path is clean

.venv/bin/python3 -c "import sys; print([p for p in sys.path if 'podcast_scraper' in p.lower()])"

# Check for global installation

python3 -m pip show podcast-scraper

# Uninstall from global if found

python3 -m pip uninstall -y podcast-scraper
```yaml

## GitHub Issue Auto-Close Keywords

Use these in PR title or description to automatically close issues when PR merges:

| Keyword | Example | Effect |
| --------- | --------- | -------- |
| `Fixes` | `Fixes #169` | Closes issue #169 |
| `Closes` | `Closes #169` | Closes issue #169 |
| `Resolves` | `Resolves #169` | Closes issue #169 |

**Tip:** Include in PR title for visibility: `feat: Add Dependabot (Fixes #169)`

## Related Documentation

- **[RFC-039: Development Workflow with Git Worktrees](../rfc/RFC-039-development-workflow-worktrees-ci.md)** - Complete technical details and rationale
- **[Development Guide](DEVELOPMENT_GUIDE.md)** - General development setup and practices
- **[Contributing Guide](https://github.com/chipi/podcast_scraper/blob/main/CONTRIBUTING.md)** - Contributor guidelines
- **[CI/CD Documentation](../ci/index.md)** - CI pipeline details

## References

- **Git Worktrees Documentation**: https://git-scm.com/docs/git-worktree
- **GitHub Actions Triggers**: https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows
- **Branch Protection**: https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches
- **Cursor Documentation**: https://cursor.sh/docs
