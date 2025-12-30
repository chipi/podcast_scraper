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

## 📚 COMPLETE GUIDE FILE SET (LOAD ALL WHEN REQUESTED)

**When the user asks to "load ai coding guidelines" or "load coding guidelines", you MUST load ALL of these files:**

1. ✅ **`.ai-coding-guidelines.md`** - Main AI coding guidelines (PRIMARY source of truth)
2. ✅ **`docs/guides/CURSOR_AI_BEST_PRACTICES_GUIDE.md`** - Cursor AI best practices and model selection
3. ✅ **`docs/guides/DEVELOPMENT_GUIDE.md`** - Detailed development guide (code style, testing, CI/CD, architecture)
4. ✅ **`docs/guides/TESTING_GUIDE.md`** - Testing guide (unit, integration, E2E implementation)
5. ✅ **`docs/guides/MD_STYLE_REFERENCE.md`** - Markdown style quick reference
6. ✅ **`docs/guides/MARKDOWN_LINTING_GUIDE.md`** - Complete markdown linting guide

**Why load all of them:**

- `.ai-coding-guidelines.md` provides critical workflow rules
- `CURSOR_AI_BEST_PRACTICES_GUIDE.md` provides model selection and workflow optimization
- `DEVELOPMENT_GUIDE.md` provides detailed technical patterns and implementation details
- `TESTING_GUIDE.md` provides comprehensive testing implementation details
- `MD_STYLE_REFERENCE.md` and `MARKDOWN_LINTING_GUIDE.md` provide markdown formatting standards

**Loading pattern:**

```python

# When user says "load ai coding guidelines" or "load coding guidelines":

# 1. Read .ai-coding-guidelines.md

# 2. Read docs/guides/CURSOR_AI_BEST_PRACTICES_GUIDE.md

# 3. Read docs/guides/DEVELOPMENT_GUIDE.md

# 4. Read docs/guides/TESTING_GUIDE.md

# 5. Read docs/guides/MD_STYLE_REFERENCE.md

# 6. Read docs/guides/MARKDOWN_LINTING_GUIDE.md

# 7. Acknowledge all files loaded and summarize key points

```text
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

**Core Guide Files (load all when user asks to "load ai coding guidelines"):**
- **`.ai-coding-guidelines.md`** - Main AI coding guidelines (PRIMARY source of truth)
- **`docs/guides/CURSOR_AI_BEST_PRACTICES_GUIDE.md`** - Cursor AI best practices and model selection
- **`docs/guides/DEVELOPMENT_GUIDE.md`** - Detailed technical information (code style, testing, CI/CD,
  architecture, logging, documentation standards)


- **`docs/guides/TESTING_GUIDE.md`** - Testing guide (unit, integration, E2E implementation)
- **`docs/guides/MD_STYLE_REFERENCE.md`** - Markdown style quick reference
- **`docs/guides/MARKDOWN_LINTING_GUIDE.md`** - Complete markdown linting guide (automated fixing, table
  formatting, pre-commit hooks, CI/CD integration)

**Additional Reference Files:**
- **`docs/TESTING_STRATEGY.md`** - Comprehensive testing approach
- **`docs/ARCHITECTURE.md`** - Architecture design and module responsibilities

**See the "📚 COMPLETE GUIDE FILE SET" section above for the complete loading pattern.**
