# AI Coding Guidelines for podcast_scraper

## ⚠️ PRIMARY REFERENCE FILES ⚠️

**For Cursor AI (automatic enforcement):**
**`.cursorrules`** - Critical rules enforced automatically by Cursor

**For all AI assistants (comprehensive guidelines):**
**`.ai-coding-guidelines.md`** - Complete AI coding guidelines (PRIMARY source of truth)

## Quick Reference

**CRITICAL RULES:**

- ❌ NEVER push any branch without explicit user approval (commits OK after diff approval, pushes NEVER by default)
- ❌ NEVER commit without showing changes and getting user approval
- ❌ NEVER push to main branch (always use feature branches)
- ✅ Always show `git status` and `git diff` before committing
- ✅ Always wait for explicit user approval before committing
- ✅ Always run `make ci-fast` before committing
- ✅ ALWAYS use Makefile commands (never direct pytest/python/black commands)
- ✅ NEVER use `cd` to project root (already in workspace directory)
- ✅ ALWAYS use correct GitHub username (check with `mcp_github_get_me`, not Mac username)
- ✅ ALWAYS show terminal output for make/test commands (`is_background: false`)
- ✅ Run `make fix-md` immediately after ANY markdown edit (zero lint violations before review)

## 📚 COMPLETE GUIDE FILE SET (LOAD ALL WHEN REQUESTED)

**When the user asks to "load ai coding guidelines" or "load coding guidelines", you MUST load ALL of these files:**

1. ✅ **`.ai-coding-guidelines.md`** - Main AI coding guidelines (PRIMARY source of truth)
2. ✅ **`docs/guides/CURSOR_AI_BEST_PRACTICES_GUIDE.md`** - Cursor AI best practices and model selection
3. ✅ **`docs/guides/DEVELOPMENT_GUIDE.md`** - Detailed development guide (code style, testing, CI/CD, architecture)
4. ✅ **`docs/guides/TESTING_GUIDE.md`** - Testing guide (unit, integration, E2E implementation)
5. ✅ **`docs/guides/MARKDOWN_LINTING_GUIDE.md`** - Markdown style and linting guide (style rules,
   linting practices, tools, workflows)

**Why load all of them:**

- `.ai-coding-guidelines.md` provides critical workflow rules
- `CURSOR_AI_BEST_PRACTICES_GUIDE.md` provides model selection and workflow optimization
- `DEVELOPMENT_GUIDE.md` provides detailed technical patterns and implementation details
- `TESTING_GUIDE.md` provides comprehensive testing implementation details
- `MARKDOWN_LINTING_GUIDE.md` provides markdown style rules and linting standards

**Loading pattern:**

```python

# When user says "load ai coding guidelines" or "load coding guidelines":

# 1. Read .ai-coding-guidelines.md

# 2. Read docs/guides/CURSOR_AI_BEST_PRACTICES_GUIDE.md

# 3. Read docs/guides/DEVELOPMENT_GUIDE.md

# 4. Read docs/guides/TESTING_GUIDE.md

# 5. Read docs/guides/MARKDOWN_LINTING_GUIDE.md

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
- **`docs/guides/MARKDOWN_LINTING_GUIDE.md`** - Markdown style and linting guide (style rules, automated fixing, table
  formatting, pre-commit hooks, CI/CD integration)

**Additional Reference Files:**
- **`docs/TESTING_STRATEGY.md`** - Comprehensive testing approach
- **`docs/ARCHITECTURE.md`** - Architecture design and module responsibilities

**See the "📚 COMPLETE GUIDE FILE SET" section above for the complete loading pattern.**
