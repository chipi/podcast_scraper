# GitHub Copilot Instructions for podcast_scraper

## ⚠️ PRIMARY REFERENCE FILES ⚠️

**For Cursor AI (automatic enforcement):**
**`.cursorrules`** - Critical rules enforced automatically by Cursor

**For all AI assistants (comprehensive guidelines):**
**`.ai-coding-guidelines.md`** - Complete AI coding guidelines (PRIMARY source of truth)

## 🚨 START-OF-SESSION CHECKLIST (MANDATORY - DO THIS FIRST)

**Before taking ANY action in this project, you MUST:**

1. ✅ **Read `.cursorrules`** (for Cursor AI) or **`.ai-coding-guidelines.md`** (for other AI)
2. ✅ **Acknowledge you've read it** - Say "I've read the AI guidelines" or similar
3. ✅ **Confirm you understand** - The user may ask "Did you read the guidelines?" - Answer honestly
4. ✅ **Reference the guidelines** for all decisions about commits, pushes, and workflows

**If the user asks "Did you read the guidelines?" or "Check the guidelines first":**

- ✅ **STOP what you're doing**
- ✅ **Read `.cursorrules` or `.ai-coding-guidelines.md` immediately**
- ✅ **Acknowledge what you read**
- ✅ **Then proceed with the task**

## Quick Reference

**CRITICAL RULES:**

- ❌ NEVER push any branch without explicit user approval (commits OK after diff approval, pushes NEVER by default)
- ❌ NEVER commit without showing changes and getting user approval
- ❌ NEVER push to main branch (always use feature branches)
- ✅ Always show `git status` and `git diff` before committing
- ✅ Always wait for explicit user approval before committing
- ✅ Run `make ci-fast` before committing when needed (exceptions: workflow-only changes; recent green `ci-fast`/`ci` in-session on same diff — see `.cursorrules` rule 5)
- ✅ ALWAYS use Makefile commands (never direct pytest/python/black commands)
- ✅ NEVER use `cd` to project root (already in workspace directory)
- ✅ ALWAYS use correct GitHub username (check dynamically, not Mac username)
- ✅ ALWAYS show terminal output for make/test commands (`is_background: false`)
- ✅ Run `make fix-md` immediately after ANY markdown edit (zero lint violations before review)
- ✅ **GI/KG viewer UX** (`web/gi-kg-viewer/`): update **`e2e/E2E_SURFACE_MAP.md`** → Playwright specs/helpers → **`docs/uxs/UXS-001-gi-kg-viewer.md`** when the visual contract changes. See `docs/guides/E2E_TESTING_GUIDE.md` (Playwright).
- ✅ **FastAPI `/api/*` changes**: extend **`tests/unit/podcast_scraper/server/`** and **`tests/integration/server/`**; HTTP reference **`docs/guides/SERVER_GUIDE.md`**.

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

# 6. Acknowledge all files loaded and summarize key points

```text

## CRITICAL RULES (MUST FOLLOW ALWAYS - NO EXCEPTIONS)

**🚨 COMMIT WORKFLOW - MANDATORY CHECKLIST (NO EXCEPTIONS):**

1. ❌ **NEVER commit without first showing `git status`**
2. ❌ **NEVER commit without first showing `git diff`**
3. ❌ **NEVER commit without explicit user approval**
4. ❌ **NEVER commit when user said "don't commit" or "wait"**
5. ✅ **ALWAYS show changes BEFORE asking to commit**
6. ✅ **ALWAYS wait for explicit approval (user says "commit", "yes", "go ahead", etc.)**
7. ✅ **ALWAYS get commit message from user OR ask for one**

**🚨 PR PUSH WORKFLOW - MANDATORY CHECKLIST (NO EXCEPTIONS):**

**CRITICAL: USER APPROVAL REQUIRED BEFORE EVERY PUSH**

1. ❌ **NEVER push to PR without explicit user approval**
2. ❌ **NEVER push without showing `git status` and `git diff` first**
3. ❌ **NEVER push when user said "don't push" or "wait"**
4. ❌ **NEVER push to PR without running `make ci` first**
5. ❌ **NEVER push when `make ci` has failures**
6. ✅ **ALWAYS show `git status` before asking to push**
7. ✅ **ALWAYS show `git diff` or summary of changes before asking to push**
8. ✅ **ALWAYS run `make ci` before pushing to PR (new or updated)**
9. ✅ **ALWAYS fix all CI failures before pushing**
10. ✅ **ALWAYS wait for explicit approval (user says "push", "go ahead", "yes", etc.)**
11. ✅ **ONLY push after steps 6-10 are complete AND user has approved**

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

- **`docs/architecture/TESTING_STRATEGY.md`** - Comprehensive testing approach
- **`docs/architecture/ARCHITECTURE.md`** - Architecture design and module responsibilities

**See the "📚 COMPLETE GUIDE FILE SET" section above for the complete loading pattern.**
