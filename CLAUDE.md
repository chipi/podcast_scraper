# AI Coding Guidelines for podcast_scraper

## ⚠️ PRIMARY REFERENCE FILE ⚠️

**This file points to the main guidelines. For complete guidelines, see:**
**`.ai-coding-guidelines.md`** - This is the PRIMARY source of truth.

## Quick Reference

**CRITICAL RULES:**

- ❌ NEVER commit without showing changes and getting user approval
- ❌ NEVER push to PR without running `make ci` first
- ✅ Always show `git status` and `git diff` before committing
- ✅ Always wait for explicit user approval before committing
- ✅ Always run `make ci` before pushing to PR (new or updated)

## Full Guidelines

**All detailed guidelines, patterns, and rules are in `.ai-coding-guidelines.md`**

**Before taking ANY action:**

1. Read `.ai-coding-guidelines.md` - This is the PRIMARY source of truth
2. Follow ALL CRITICAL rules marked in that file
3. Reference it for all decisions about code, workflow, and patterns

**Key sections in `.ai-coding-guidelines.md`:**

- Git Workflow (commit approval, PR workflow)
- Code Organization (module boundaries, patterns)
- Testing Requirements (mocking, test structure)
- Documentation Standards (PRDs, RFCs, docstrings)
- Common Patterns (configuration, error handling, logging)

**See `.ai-coding-guidelines.md` for complete guidelines.**

## Key Documentation Files

**When working on related tasks, read these files:**

- **`docs/guides/DEVELOPMENT_GUIDE.md`** - Detailed technical information (code style, testing, CI/CD,
  architecture, logging, documentation standards)

- **`docs/guides/MARKDOWN_LINTING_GUIDE.md`** - Complete markdown linting guide (automated fixing, table
  formatting, pre-commit hooks, CI/CD integration)

- **`docs/TESTING_STRATEGY.md`** - Comprehensive testing approach
- **`docs/ARCHITECTURE.md`** - Architecture design and module responsibilities

**These files are referenced in `.ai-coding-guidelines.md` but should be read directly when working on related tasks.**
