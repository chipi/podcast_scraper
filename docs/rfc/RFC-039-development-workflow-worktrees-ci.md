# RFC-039: Development Workflow with Git Worktrees and CI Evolution

- **Status**: Completed
- **Authors**: chipi
- **Stakeholders**: Developers, CI/CD maintainers
- **Related PRDs**: None
- **Related RFCs**:
  - RFC-038: Continuous Review Tooling (CI quality checks)
- **Related Documents**:
  - `.ai-coding-guidelines.md` (AI workflow rules)
  - `docs/ci/index.md` (CI pipeline documentation)
  - `docs/guides/DEVELOPMENT_GUIDE.md` (Developer setup)
  - `CONTRIBUTING.md` (Contributor guidelines)

## Abstract

This RFC proposes adopting a **git worktree-based development workflow** combined with
**evolved GitHub Actions CI** to enable safe, parallel development while maintaining a
clean `main` branch and predictable CI behavior.

The workflow addresses context isolation for AI-assisted development (Cursor), parallel
version work, and efficient CI resource usage through stratified job execution.

**Architecture Alignment:** This RFC defines development practices that complement the
modular provider architecture (RFC-016) by enabling isolated work on multiple features
simultaneously.

---

## Table of Contents

1. Version Semantics
2. Problem Statement
3. Goals | Non-Goals | Constraints
4. Workflow Rules (Normative)
5. Design and Implementation
   - 1. Git Worktree Workflow
   - 2. Complete Worktree Setup
   - 3. Cursor Integration
   - 4. Rebase Strategy
   - 5. GitHub Actions CI
   - 6. Makefile Helpers
   - 7. Emergency Recovery
   - 8. Weekly Maintenance
6. Documentation Updates
7. Key Decisions | Alternatives
8. Rollout Plan
9. Quick Reference Cheat Sheet

---

## Version Semantics and Terminology (IMPORTANT)

The phrase **"current version"** is ambiguous and depends on perspective. This RFC uses
explicit terms so operational decisions are always clear.

### Definitions

| Term | Meaning | Example |
| ------ | --------- | --------- |
| **PROD** | Latest version users are running and may need patches | 2.3 |
| **NEXT** | Version being stabilized and prepared for the next release | 2.4 |
| **FUTURE** | Long-term development beyond NEXT | 2.5+ |

### Branch Mapping

| Semantic Role | Git Branch |
| -------------- | ------------ |
| **PROD** | `release/2.3` |
| **NEXT** | `release/2.4` |
| **FUTURE** | `main` |

Key clarifications:

- `main` is **not** the version shipped to users while NEXT is in stabilization.
- `main` is a **clean, forward-moving FUTURE baseline** from which the next release branch is cut.
- Release branches (`release/x.y`) are the **shipping vehicles** for PROD/NEXT.
- "Integrate into `main`" means **forward-porting fixes** so future versions do not regress.

This section is the anchor for the rest of the workflow (worktrees, PR targets, and fix propagation).

---

## Why Git Worktrees (and Not Just Release Branches)

It is possible to implement the PROD / NEXT / FUTURE model using only long‑lived release
branches and a single working directory. However, this RFC explicitly adopts **git
worktrees** because they solve a *different operational problem* than branching alone.

### Branches vs Worktrees (Separation of Concerns)

- **Branches** define *history and ownership*
  (what changes belong to PROD vs NEXT vs FUTURE).

- **Worktrees** define *day‑to‑day working state*
  (which version you are actively editing, testing, and reasoning about).

This workflow requires working on multiple versions **in parallel**, not sequentially.
Branches alone do not address that operational need.

### Limitations of a Branch‑Only Workflow

Using a single working directory with frequent branch switching introduces:

- repeated branch switching and stashing
- risk of committing changes to the wrong branch
- unstable editor/IDE context when files change under an open session
- shared virtual environments and caches across incompatible versions
- increased cognitive load when juggling PROD, NEXT, and FUTURE simultaneously

These issues are amplified when:

- stabilizing NEXT while building FUTURE
- occasionally hot‑fixing PROD
- using AI‑assisted tools (Cursor) that rely on filesystem context

### What Worktrees Add

Git worktrees provide:

- **True parallelism**: multiple versions open and usable at the same time
- **Filesystem isolation**: one branch per directory, no accidental crossover
- **Stable editor context**: Cursor sees one coherent version per window
- **Environment isolation**: separate virtualenvs/config per version
- **Lower operational risk**: no branch switching, no stashing, no ambiguity

### Key Insight

> Release branches solve *what belongs where*.
> Worktrees solve *how you work on those things safely and efficiently*.

In this workflow, worktrees are not an optimization — they are the mechanism that makes
simultaneous PROD / NEXT / FUTURE work practical and low‑risk.

## Problem Statement

### Current Issues with Single Working Directory

Using a single Git checkout leads to:

- **Frequent branch switching** - context loss, stashing required
- **Accidental commits on wrong branch** - especially during parallel work
- **AI tooling confusion** - Cursor operates on mixed or unintended context
- **Late CI feedback** - validation only at PR time
- **Unpredictable CI behavior** - rebases retrigger CI unexpectedly

As parallel work increases (v2.4, v2.5, v3.0 development), these issues scale poorly.

**Use Cases:**

1. **Parallel Version Development**: Working on v2.5 features while fixing v2.4 bugs
2. **AI Context Isolation**: Each Cursor instance has clean, focused context
3. **Fast Iteration**: Get CI feedback during development, not just at PR time

## Goals

1. **Enable parallel branch work** - multiple branches active simultaneously
2. **Keep FUTURE (`main`) clean** - forward-moving, reviewable, protected
3. **Fast CI feedback** - during development, not just at merge time
4. **Align with AI tooling** - Cursor benefits from isolated context
5. **Predictable CI** - behavior matches workflow expectations
6. **Reduce cognitive load** - clear separation of concerns

## Non-Goals

- Changing the PR-only / squash-merge policy
- Introducing GitFlow-style long-lived branches
- Supporting multi-developer shared branches
- Complex deployment pipelines or multi-environment promotion

## Constraints & Assumptions

**Constraints:**

- All changes to `main` must go through PRs (FUTURE baseline; no direct commits)
- Squash merges only (linear history)
- Solo development (single developer)
- Must work with Cursor AI tooling

**Assumptions:**

- Developer has sufficient disk space for multiple worktrees
- Each worktree needs ~200MB (shared Git objects + isolated venv)
- GitHub Actions minutes are a consideration

---

## Workflow Rules (Normative)

This section distinguishes **required** rules from **recommended** practices.

### MUST (Required)

These rules are mandatory for the workflow to function correctly:

| Rule | Rationale |
| ------ | ----------- |
| **Never commit directly to `main`** | All changes via PR; branch protection enforces this |
| **One branch per worktree** | Don't switch branches inside worktrees; create new worktree instead |
| **Use `--force-with-lease` after rebase** | Never use `--force`; prevents overwriting unexpected changes |
| **Remove worktree after PR merge** | Prevents clutter; orphaned worktrees cause confusion |
| **All PRs use squash merge** | Maintains linear history on `main` |

### SHOULD (Recommended)

These practices are strongly recommended but not strictly required:

| Practice | Rationale |
| ---------- | ----------- |
| **Rebase before requesting review** | Clean diff for reviewers; reduces merge conflicts |
| **Include issue number in branch name** | Traceability; auto-linking in GitHub |
| **Use isolated venv per worktree** | Complete isolation; prevents dependency conflicts |
| **Run `make wt-list` weekly** | Audit for forgotten worktrees |
| **Open separate Cursor window per worktree** | Clean AI context per task |

### MAY (Optional)

These are acceptable variations based on preference:

| Option | When to Use |
| -------- | ------------- |
| **Skip issue number for exploratory work** | No tracking needed for experiments |
| **Share venv across worktrees** | Disk constrained; accept pollution risk |
| **Keep main worktree for reference** | Quick lookups without switching context |

---

## Design & Implementation

### 1. Git Worktree Workflow

#### 1.1 What Is a Git Worktree?

A **git worktree** allows multiple working directories to share the same Git repository
data while each directory checks out a different branch.

**Key Properties:**

- One Git history (`.git`) shared across all worktrees
- Multiple independent folders
- Each folder has exactly one checked-out branch
- No stashing or branch switching required

#### 1.2 Branching Model with Issue Integration

Include GitHub issue numbers in branch names for traceability:

```text
main (protected)
├── feat/169-dependabot       → ../podcast_scraper-169-dependabot
├── feat/106-anthropic        → ../podcast_scraper-106-anthropic
├── fix/185-memory-leak       → ../podcast_scraper-185-memory
└── rel/2.5                   → ../podcast_scraper-2.5
```yaml

**Branch Naming Convention:**

| Pattern | Example | Use Case |
| --------- | --------- | ---------- |
| `feat/{issue}-{name}` | `feat/169-dependabot` | Feature linked to issue |
| `fix/{issue}-{name}` | `fix/185-memory-leak` | Bug fix linked to issue |
| `rel/{version}` | `rel/2.5` | Release preparation |
| `feat/{name}` | `feat/experimental` | No issue (exploratory) |

**Benefits of Issue Numbers in Branches:**

- `git worktree list` shows which issue each worktree addresses
- GitHub auto-links commits/PRs to issues
- Easy to find related issue from any git log
- PR titles auto-populate with issue reference

#### 1.3 Version-Oriented Worktrees (PROD / NEXT / FUTURE)

In addition to issue-linked worktrees, this project commonly needs **version-oriented**
worktrees during stabilization. A typical setup:

| Folder | Branch | Semantic Role | When Active |
| -------- | -------- | --------------- | ------------ |
| `podcast_scraper` | `main` | FUTURE (2.5+) | Daily (future work) |
| `podcast_scraper-next-2.4` | `release/2.4` | NEXT | Daily (stabilization) |
| `podcast_scraper-prod-2.3` | `release/2.3` | PROD | Only when hotfixing |

**Practical guidance:** keep FUTURE + NEXT open most days. Create PROD only when you need a 2.3.x patch.

#### 1.4 One Branch = One Worktree = One Cursor Window

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

---

### 2. Complete Worktree Setup Process

#### 2.1 Full Setup (New Worktree + Isolated Venv)

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

# 6. Install project in development mode

pip install -e ".[dev]"

# 7. Verify installation

make check-deps  # or: pip list | grep podcast

# 8. Open in Cursor

cursor .
```yaml

## 2.2 Why Isolated Virtual Environments?

| Aspect | Shared venv | Isolated venv (Recommended) |
| -------- | ------------- | ---------------------------- |
| Disk usage | Lower (~50MB saved) | Higher (full deps per worktree) |
| Dependency conflicts | Possible | Impossible |
| Branch-specific deps | Not possible | Full support |
| Safety | Risk of pollution | Complete isolation |
| Cleanup | Complex | Simple (delete folder) |

**Decision:** Use **isolated venvs** for each worktree.

**Rationale:**

- Different branches may have different dependencies
- Testing new packages without affecting other work
- Complete cleanup when worktree is removed
- Matches Cursor's expectation of isolated environments

**⚠️ Caution: Python Tooling Configuration**

When using isolated venvs, ensure your tooling doesn't assume a single venv path:

| Item | Requirement | Status |
| ------ | ------------- | -------- |
| `.venv/` in `.gitignore` | Must be ignored | ✅ Already configured |
| Cursor Python path | Use `${workspaceFolder}/.venv/bin/python` | ✅ Documented below |
| `pre-commit` hooks | Runs in current venv (works correctly) | ✅ No changes needed |
| `.python-version` | Shared across worktrees (tracked in git) | ✅ Works correctly |
| `pyproject.toml` / lock files | Shared across worktrees (tracked in git) | ✅ Works correctly |
| IDE run configs | May need per-worktree adjustment | ⚠️ Check if issues |

**Key principle:** Configuration files (tracked in git) are shared. Runtime artifacts
(`.venv/`, `__pycache__/`, `.pytest_cache/`) are per-worktree and gitignored.

### 2.3 Disk Space Estimation

Per worktree:

| Component | Size |
| ----------- | ------ |
| Working files | ~50MB |
| Virtual environment | ~150MB |
| Shared Git objects | 0 (reused) |
| **Total per worktree** | **~200MB** |

With 5 active worktrees: ~1GB total

---

### 3. Cursor Integration Details

#### 3.1 Migration: From Single Folder to Worktrees (PROD / NEXT / FUTURE)

If you're currently using a single checkout folder, follow these steps to migrate:

**Step 1: Prepare your current folder as the FUTURE (`main`) base reference**

```bash

# Navigate to your current clone

cd ~/Projects/podcast_scraper

# Ensure you're on main and up-to-date

git switch main
git pull --ff-only

# Verify clean state

git status  # Should show "nothing to commit, working tree clean"
```

**Recommendation:** Keep this base folder on `main` as your FUTURE workspace. Create separate worktrees for NEXT (e.g., `release/2.4`) and PROD (e.g., `release/2.3` when needed).

**Step 2: Create your first worktree for active work**

```bash

# Still in your current folder

git fetch origin

# Create worktree for your current task (example: issue #169)

git worktree add ../podcast_scraper-169-dependabot -b feat/169-dependabot origin/main
```

**Step 3: Set up the worktree environment**

```bash

# Navigate to the new worktree

cd ../podcast_scraper-169-dependabot

# Create isolated virtual environment

python3 -m venv .venv
source .venv/bin/activate

# Install dependencies

pip install -e ".[dev]"
```

**Step 4: Open the worktree in Cursor**

```bash

# Open worktree folder (NOT the original folder)

cursor .
```python

**Step 5: Configure Cursor Python interpreter (one-time)**

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

**Step 6: Treat base folder as read-only**

Your original folder (`~/Projects/podcast_scraper`) is now your "base":

| Folder | Purpose | Open in Cursor? |
| -------- | --------- | ----------------- |
| `podcast_scraper/` (base) | Reference, git log, viewing | ❌ No (read-only) |
| `podcast_scraper-169-*/` | Active development | ✅ Yes |

**Never edit code in the base folder.** Use it only for:

- `git log` and history exploration
- `git worktree list` to see all worktrees
- Quick file viewing without context switching

---

## 3.2 Files That Propagate Automatically

These files are tracked in Git and automatically available in each worktree:

| File | Purpose | Behavior |
| ------ | --------- | ---------- |
| `.ai-coding-guidelines.md` | AI behavior rules | Same across worktrees |
| `.cursorignore` | Files to ignore | Same across worktrees |
| `.cursorrules` | Project rules | Same across worktrees |
| `pyproject.toml` | Python config | Same across worktrees |

### 3.3 Cursor Settings (Per-User, Not Per-Worktree)

These are stored in Cursor's application data, not in the repo:

- Cursor themes and UI preferences
- Extension settings
- Keybindings
- AI model preferences

**Location:** `~/.cursor/` (shared across all projects)

#### 3.4 Cursor Workspace Settings

Each worktree can have its own `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": true
}
```

**Important:** This file should be in `.gitignore` to allow per-worktree customization.

#### 3.5 Opening Worktrees in Cursor

```bash

# From main repository

cursor ../podcast_scraper-169-dependabot

# Or from within the worktree

cd ../podcast_scraper-169-dependabot
cursor .
```yaml

**Tip:** Each Cursor window maintains its own:

- Terminal sessions
- Open files
- AI conversation history
- Debug configurations

---

## 4. Rebase Strategy and Decision Tree

### 4.1 Why Rebase?

With squash merges to `main`, your feature branch's commits become orphaned after
another PR merges. Rebasing re-parents your commits onto the new `main`.

**Plain-language explanation:** Because squash-merge creates a *new* commit on `main`
(not your original commits), your branch will appear "behind" even if it had the same
changes. Rebasing replays your commits on top of the new `main` state, making your
branch current again.

```text
Before rebase:
main:    A---B---C (C is someone else's squash-merged PR)
              \
feature:       D---E---F (your commits, now "behind")

After rebase:
main:    A---B---C
                  \
feature:           D'---E'---F' (same changes, replayed on top of C)
```

#### 4.2 When to Rebase - Decision Tree

```text
Should I rebase my branch?
│
├─► PR is ready for review?
│   └─► YES → Rebase now (clean diff for reviewers)
│
├─► CI is failing due to main changes?
│   └─► YES → Rebase now (get latest fixes)
│
├─► Merge conflicts expected?
│   └─► YES → Rebase early (smaller conflicts)
│
├─► Main has breaking changes I need?
│   └─► YES → Rebase now (get the changes)
│
├─► Just want latest changes for curiosity?
│   └─► NO → Continue working (rebase adds noise)
│
└─► Actively coding, no blockers?
    └─► NO → Continue working (rebase when ready)
```

#### 4.3 Rebase Commands

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

## 4.4 Rebase Best Practices

| Practice | Reason |
| ---------- | -------- |
| Use `--force-with-lease` | Fails if remote has unexpected changes |
| Rebase before PR review | Clean diff for reviewers |
| Don't rebase during active coding | Creates unnecessary churn |
| Resolve conflicts immediately | Don't leave rebase in progress |
| Test after rebase | Ensure nothing broke |

### 4.5 Fix Propagation Rules (PROD → NEXT → FUTURE)

Fixes should generally flow **forward**, not backward, to avoid regressions across versions.

| Fix Origin | Must Propagate To | Typical Method |
| ----------- | -------------------- | ---------------- |
| **PROD** (`release/2.3`) | `release/2.4`, `main` | `git cherry-pick` + PR |
| **NEXT** (`release/2.4`) | `main` | `git cherry-pick` + PR |
| **FUTURE** (`main`) | Backport only if critical | `git cherry-pick` + PR (exception) |

**Rules of thumb:**

- Use **cherry-pick** for propagation (keep changes minimal and focused).
- Do **not** merge release branches into `main` (avoids pulling release-only noise into FUTURE).
- Prefer **small propagation PRs** (one fix, one intent).
- If a fix is purely release-specific (e.g., temporary logging), it may not need propagation.

**Operational checklist after a NEXT hotfix merges:**

1. Ask: *"Would I be annoyed if this bug reappears in 2.5+?"*
2. If yes: cherry-pick to a branch off `main`, open PR, squash-merge.

---

### 5. GitHub Actions CI Evolution

#### 5.1 CI Trigger Model

```yaml
on:
  push:
    branches-ignore:

      - main
  pull_request:

    branches:

      - main
```yaml

#### 5.2 Job Stratification

The key insight: **fast checks run on both push AND PR**, while **full checks only run on PR**.

| Event | Fast Checks | Full Checks | Total Time |
| ------- | ------------- | ------------- | ------------ |
| Push to feature branch | ✅ Runs | ❌ Skipped | ~2 min |
| Open/update PR to main | ✅ Runs | ✅ Runs | ~10 min |
| Push directly to main | ❌ Blocked | ❌ Blocked | N/A |

**Rationale:** PRs need ALL checks (fast + full) before merging. The stratification benefit
is that during active development (push), you only wait for fast checks.

**Fast Checks (on push AND PR):**

```yaml
fast-checks:
  runs-on: ubuntu-latest
  # No 'if' - runs on all triggers (push and pull_request)
  steps:

    - uses: actions/checkout@v4
    - name: Lint
      run: make lint

    - name: Format check
      run: make format-check

    - name: Unit tests
      run: make test-unit

```

**Full Checks (on PR only):**

```yaml
full-checks:
  runs-on: ubuntu-latest
  if: github.event_name == 'pull_request'  # ONLY on PRs
  steps:

    - uses: actions/checkout@v4
    - name: Integration tests
      run: make test-integration

    - name: E2E tests
      run: make test-e2e

    - name: Coverage report
      run: make coverage

```yaml

**Why this works:**

| Scenario | What Runs | Why |
| ---------- | ----------- | ----- |
| Active coding (push) | Fast only | Quick feedback, don't slow iteration |
| Ready for review (PR) | Fast + Full | Thorough validation before merge |
| Merged to main | Nothing | Protected, no direct pushes |

**Benefits:**

- Branch pushes: ~2 min feedback (fast checks only)
- PRs: ~10 min full validation (fast + full checks)
- No redundant work: fast checks are useful prerequisites for full checks

#### 5.3 Branch Protection Rules

For `main` branch:

| Rule | Setting |
| ------ | --------- |
| Require PR | ✅ Enabled |
| Require CI checks | ✅ `fast-checks`, `full-checks` |
| Require approval | ✅ 1 reviewer (or self-approve for solo) |
| Block force-push | ✅ Enabled |
| Require up-to-date | ⚠️ Optional (triggers rebases) |

---

### 6. Makefile Helpers

Add these targets to `Makefile` for common worktree operations.

#### 6.1 Shell Compatibility

**Important:** These targets use `read -p` which requires bash. Add this near the top
of your Makefile to ensure compatibility (especially on macOS):

```makefile

# Force bash for read -p compatibility (macOS sh doesn't support -p)

SHELL := /bin/bash
```

## 6.2 Worktree Targets

```makefile

# =============================================================================
# Worktree Management
# =============================================================================

.PHONY: wt-new wt-list wt-remove wt-prune wt-setup

# Helper function to sanitize folder names (replace / and spaces with -)
# Usage: $(call sanitize,string)

define sanitize
$(shell echo "$(1)" | tr '/ ' '--' | tr -cd '[:alnum:]-_')
endef

## Create new worktree (interactive)

wt-new:
	@echo "=== Create New Worktree ==="
	@read -p "Issue number (or press Enter to skip): " issue; \
	read -p "Branch type (feat/fix/rel): " type; \
	read -p "Short name (no spaces or slashes): " name; \
	safename=$$(echo "$$name" | tr '/ ' '--' | tr -cd '[:alnum:]-_'); \
	if [ "$$safename" != "$$name" ]; then \
		echo "⚠️  Sanitized name: $$name → $$safename"; \
	fi; \
	if [ -z "$$safename" ]; then \
		echo "❌ Error: name cannot be empty"; exit 1; \
	fi; \
	if [ -n "$$issue" ]; then \
		branch="$$type/$$issue-$$safename"; \
		folder="../podcast_scraper-$$issue-$$safename"; \
	else \
		branch="$$type/$$safename"; \
		folder="../podcast_scraper-$$safename"; \
	fi; \
	echo "Creating: $$folder (branch: $$branch)"; \
	git fetch origin && \
	git worktree add "$$folder" -b "$$branch" origin/main && \
	echo "" && \
	echo "Next steps:" && \
	echo "  cd $$folder" && \
	echo "  python3 -m venv .venv" && \
	echo "  source .venv/bin/activate" && \
	echo "  pip install -e '.[dev]'" && \
	echo "  cursor ."

## Create worktree with full setup (interactive)

wt-setup:
	@echo "=== Create Worktree with Full Setup ==="
	@read -p "Issue number (or press Enter to skip): " issue; \
	read -p "Branch type (feat/fix/rel): " type; \
	read -p "Short name (no spaces or slashes): " name; \
	safename=$$(echo "$$name" | tr '/ ' '--' | tr -cd '[:alnum:]-_'); \
	if [ "$$safename" != "$$name" ]; then \
		echo "⚠️  Sanitized name: $$name → $$safename"; \
	fi; \
	if [ -z "$$safename" ]; then \
		echo "❌ Error: name cannot be empty"; exit 1; \
	fi; \
	if [ -n "$$issue" ]; then \
		branch="$$type/$$issue-$$safename"; \
		folder="../podcast_scraper-$$issue-$$safename"; \
	else \
		branch="$$type/$$safename"; \
		folder="../podcast_scraper-$$safename"; \
	fi; \
	echo "Creating and setting up: $$folder"; \
	git fetch origin && \
	git worktree add "$$folder" -b "$$branch" origin/main && \
	cd "$$folder" && \
	python3 -m venv .venv && \
	. .venv/bin/activate && \
	pip install -e ".[dev]" && \
	echo "" && \
	echo "✅ Worktree ready: $$folder" && \
	echo "   Branch: $$branch" && \
	echo "   Run: cd $$folder && source .venv/bin/activate && cursor ."

## List all worktrees with status

wt-list:
	@echo "=== Active Worktrees ==="
	@git worktree list
	@echo ""
	@echo "=== Worktree Details ==="
	@for wt in $$(git worktree list --porcelain | grep "^worktree" | cut -d' ' -f2); do \
		if [ -d "$$wt/.venv" ]; then venv="✅ venv"; else venv="❌ no venv"; fi; \
		branch=$$(git -C "$$wt" branch --show-current 2>/dev/null || echo "detached"); \
		echo "  $$wt ($$branch) [$$venv]"; \
	done

## Remove a worktree (interactive)

wt-remove:
	@echo "=== Remove Worktree ==="
	@git worktree list
	@echo ""
	@read -p "Path to remove: " path; \
	read -p "Also delete branch? (y/N): " delbranch; \
	branch=$$(git -C "$$path" branch --show-current 2>/dev/null); \
	git worktree remove "$$path" && \
	if [ "$$delbranch" = "y" ] && [ -n "$$branch" ]; then \
		git branch -d "$$branch" 2>/dev/null || git branch -D "$$branch"; \
	fi && \
	echo "✅ Removed: $$path"

## Prune stale worktree references

wt-prune:
	@echo "=== Pruning Worktrees ==="
	git worktree prune -v
	git fetch --prune
	@echo "✅ Cleanup complete"
```yaml

### 6.3 Input Sanitization

The targets above sanitize user input to prevent path issues:

| Input | Sanitized | Reason |
| ------- | ----------- | -------- |
| `my feature` | `my-feature` | Spaces → dashes |
| `api/v2` | `api-v2` | Slashes → dashes |
| `fix@bug!` | `fixbug` | Special chars removed |
| `  ` | ❌ Error | Empty after sanitization |

If the sanitized name differs from input, a warning is shown before proceeding.

---

### 7. Emergency Recovery

#### 7.1 Common Issues and Fixes

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
pip install -e ".[dev]"
```

## 7.2 Recovery Cheat Sheet

```bash

# === DIAGNOSTICS ===

git worktree list                    # See all worktrees
git status                           # Check current state
git log --oneline -10                # Recent commits

# === CLEANUP ===

git worktree prune                   # Remove stale refs
git fetch --prune                    # Clean remote refs
rm -rf ../podcast_scraper-broken     # Nuclear option

# === RESET ===

git rebase --abort                   # Cancel rebase
git merge --abort                    # Cancel merge
git reset --hard origin/BRANCH       # Reset to remote

# === RECREATE ===

rm -rf .venv && python3 -m venv .venv && pip install -e ".[dev]"
```yaml

---

## 8. Weekly Maintenance Checklist

Add to your weekly routine:

```markdown

## Worktree Maintenance (Weekly)

- [ ] Run `make wt-list` - check for forgotten worktrees
- [ ] Remove worktrees for merged PRs
- [ ] Run `make wt-prune` - clean stale references
- [ ] Check disk usage: `du -sh ../podcast_scraper-*`
- [ ] Update main: `cd podcast_scraper && git pull`
```yaml

---

## Documentation Updates Required

This RFC requires updates to these documents:

### 1. `.ai-coding-guidelines.md`

Add section:

```markdown

## Worktree Workflow

- Each task gets its own worktree and Cursor instance
- Include issue number in branch: `feat/169-dependabot`
- Use `make wt-new` to create worktrees
- Never switch branches - create new worktree instead
- Clean up after PR merge: `make wt-remove`
```

### 2. `docs/guides/DEVELOPMENT_GUIDE.md`

Add section:

```markdown

## Worktree-Based Development

See RFC-039 for complete workflow documentation.

Quick start:
1. `make wt-setup` - Create worktree with venv
2. `cursor .` - Open in Cursor
3. Develop, commit, push
4. Create PR when ready
5. `make wt-remove` - Cleanup after merge
```

### 3. `CONTRIBUTING.md`

Add section:

```markdown

## Development Setup

We use git worktrees for parallel development:

1. Clone the repository (this becomes your `main` reference)
2. Create a worktree for your work: `make wt-setup`
3. Each worktree has its own virtual environment
4. See DEVELOPMENT_GUIDE.md for details
```yaml

---

## Key Decisions

1. **Isolated venvs per worktree**
   - **Decision**: Each worktree has its own `.venv`
   - **Rationale**: Complete isolation; branch-specific deps; clean removal

2. **Issue numbers in branch names**
   - **Decision**: Use pattern `type/issue-name`
   - **Rationale**: Traceability; auto-linking; easy lookup

3. **Worktrees over multiple clones**
   - **Decision**: Use git worktrees instead of multiple full clones
   - **Rationale**: Shared Git objects reduce disk usage; single history prevents drift

4. **Squash merges only**
   - **Decision**: All PRs use squash merge
   - **Rationale**: Linear history on main; requires rebase workflow on branches

5. **Split CI triggers**
   - **Decision**: Different checks for push vs PR
   - **Rationale**: Fast feedback during dev; thorough validation at merge

6. **Force-with-lease for rebases**
   - **Decision**: Always use `--force-with-lease`
   - **Rationale**: Prevents accidental overwrites; safer than `--force`

## Alternatives Considered

1. **Multiple Full Clones**
   - **Description**: Separate `git clone` for each branch
   - **Pros**: Complete isolation
   - **Cons**: Disk space; history drift; no shared objects
   - **Why Rejected**: Worktrees provide same isolation with less overhead

2. **Shared Virtual Environment**
   - **Description**: One venv shared across worktrees
   - **Pros**: Less disk usage
   - **Cons**: Dependency conflicts; pollution risk
   - **Why Rejected**: Isolation is more important than disk savings

3. **GitFlow Workflow**
   - **Description**: Long-lived develop/release branches
   - **Pros**: Well-documented; team-friendly
   - **Cons**: Overhead for solo dev; complex merge patterns
   - **Why Rejected**: Too heavy for single-developer project

## Testing Strategy

**This RFC is about process, not code.** Testing is validation through practice:

- **Validation**: Try workflow on next 3 features
- **Success metric**: Zero accidental commits to wrong branch
- **CI validation**: Verify fast checks run on push, full on PR

## Rollout Plan

**Phase 1 (Week 1):** Documentation and setup

- Create this RFC ✅
- Add Makefile helpers
- Update `.ai-coding-guidelines.md`
- Configure branch protection rules

**Phase 2 (Week 2):** Trial adoption

- Use worktrees for next 2-3 features
- Validate CI trigger behavior
- Adjust based on learnings

**Phase 3 (Week 3):** Documentation updates

- Update `DEVELOPMENT_GUIDE.md`
- Update `CONTRIBUTING.md`
- Create quick reference card

**Phase 4 (Week 4+):** Full adoption

- Worktrees are standard practice
- Weekly maintenance routine established
- Retrospective and adjustments

## Release Transition Playbook (NEXT rotates, FUTURE continues)

When NEXT ships (e.g., `release/2.4` → users adopt 2.4):

1. **Keep** `release/2.4` as maintenance-only (2.4.x hotfixes if needed).
2. Ensure FUTURE (`main`) already contains ongoing 2.5+ work.
3. When ready to stabilize the next release, **cut** a new release branch from FUTURE:

```bash
git fetch origin
git branch release/2.5 origin/main
git push origin release/2.5
```

4. Create/rotate the NEXT worktree:

```bash
git worktree add ../podcast_scraper-next-2.5 release/2.5
```yaml

5. Conceptually, `main` immediately becomes FUTURE for **2.6+** (no history rewrite; only roles change).

**Reminder:** avoid the word "current" without a qualifier; use PROD/NEXT/FUTURE.

## Risks and Mitigations

| Risk | Mitigation |
| ------ | ------------ |
| Force-push mistakes | Use `--force-with-lease` exclusively |
| Disk clutter | Remove worktrees after PR merge; weekly audit |
| Learning curve | This RFC + Makefile helpers + cheat sheet |
| Forgotten worktrees | `make wt-list` in weekly routine |
| CI minute usage | Stratified checks reduce waste |
| Venv issues | Documented recreation steps |

## Success Criteria

1. ✅ Zero accidental commits to wrong branch
2. ✅ CI feedback within 2 minutes of push
3. ✅ Clean `main` history (linear, squash-merged)
4. ✅ Cursor context matches active branch
5. ✅ No stash usage required
6. ✅ Each worktree has isolated venv
7. ✅ Issues linked to branches and PRs

## Quick Reference Cheat Sheet

### Visual Workflow

```text
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Create    │────▶│   Develop   │────▶│    Push     │
│   Worktree  │     │   + Test    │     │   Branch    │
│             │     │             │     │             │
│ make wt-new │     │ git commit  │     │ git push    │
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
       │   Remove    │
       │   Worktree  │
       │             │
       │ make wt-rm  │
       └─────────────┘
```

### Commands

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
rm -rf .venv && python3 -m venv .venv && pip install -e ".[dev]"
```yaml

## GitHub Issue Auto-Close Keywords

Use these in PR title or description to automatically close issues when PR merges:

| Keyword | Example | Effect |
| --------- | --------- | -------- |
| `Fixes` | `Fixes #169` | Closes issue #169 |
| `Closes` | `Closes #169` | Closes issue #169 |
| `Resolves` | `Resolves #169` | Closes issue #169 |

**Tip:** Include in PR title for visibility: `feat: Add Dependabot (Fixes #169)`

## References

- **Git Worktrees Documentation**: https://git-scm.com/docs/git-worktree
- **GitHub Actions Triggers**: https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows
- **Branch Protection**: https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches
- **Cursor Documentation**: https://cursor.sh/docs
