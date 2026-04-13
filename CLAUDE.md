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
- ✅ After making file edits: summarize changes and ask "Keep these changes or undo any of them?"
- ✅ When intent is clear: **run commands and tools yourself** (make, scripts, tests); **only** ask when blocked (auth/secrets, ambiguous scope, or policy needs approval) — see *Autonomous execution* in `.cursorrules` / `.ai-coding-guidelines.md`
- ✅ When any make target fails (test, ci, lint, format, docs, etc.): establish root cause first, then fix from there (no random experimenting)
- ✅ Run `make ci-fast` before committing when needed (exceptions: workflow-only changes; **recent green `ci-fast` or `ci` in this session on the same diff** with no substantive edits after; user says skip / already validated — see `.cursorrules` rule 5)
- ✅ ALWAYS use Makefile commands (never direct pytest/python/black commands)
- ✅ NEVER use `cd` to project root (already in workspace directory)
- ✅ ALWAYS use correct GitHub username (check with `mcp_github_get_me`, not Mac username)
- ✅ ALWAYS show terminal output for make/test commands (`is_background: false`)
- ❌ NEVER `git stash` during an active merge (destroys merge state and all conflict resolutions — see `.cursorrules` rules 4a–4c)
- ❌ NEVER `git checkout <ref> -- <file>` during a merge (destroys resolved content; use `git show <ref>:<path>` to inspect)
- ❌ NEVER overwrite local files with remote content without showing the diff and getting explicit approval (rule 4d)
- ✅ Run `make fix-md` immediately after ANY markdown edit (zero lint violations before review)
- ✅ **GI/KG viewer UX** (`web/gi-kg-viewer/`): when UI changes affect users or Playwright, update in order:
  **`e2e/E2E_SURFACE_MAP.md`** (automation contract) → **`e2e/*.spec.ts`** / helpers → **`docs/uxs/UXS-001-gi-kg-viewer.md`**
  if the visual/token experience contract changes. See `docs/guides/E2E_TESTING_GUIDE.md` (Playwright) and
  `docs/guides/DEVELOPMENT_GUIDE.md` (viewer section).
- ✅ **FastAPI `/api/*`**: tests in **`tests/unit/podcast_scraper/server/`** and **`tests/integration/server/`**; reference **`docs/guides/SERVER_GUIDE.md`**.
- ✅ **Local serve from a chosen output dir:** interpret “use this folder as root” as **`make serve SERVE_OUTPUT_DIR=…`** / **`make serve-api SERVE_OUTPUT_DIR=…`**; do **not** edit the Makefile default unless the user explicitly wants the repo default changed. **`VITE_DEFAULT_CORPUS_PATH`** is only for pre-filling the viewer shell path (see `.cursorrules` GI/KG section).

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

```text

# When user says "load ai coding guidelines" or "load coding guidelines":

# 1. Read .ai-coding-guidelines.md

# 2. Read docs/guides/CURSOR_AI_BEST_PRACTICES_GUIDE.md

# 3. Read docs/guides/DEVELOPMENT_GUIDE.md

# 4. Read docs/guides/TESTING_GUIDE.md

# 5. Read docs/guides/MARKDOWN_LINTING_GUIDE.md

# 6. Acknowledge all files loaded and summarize key points

```

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

- **`docs/guides/POLYGLOT_REPO_GUIDE.md`** - Python + Node monorepo layout, env files, viewer
  Makefile targets (`make test-ui`, `make serve`, …)
- **`docs/architecture/TESTING_STRATEGY.md`** - Comprehensive testing approach
- **`docs/architecture/ARCHITECTURE.md`** - Architecture design and module responsibilities

**See the "📚 COMPLETE GUIDE FILE SET" section above for the complete loading pattern.**
