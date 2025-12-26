# GitHub Copilot Instructions for podcast_scraper

## ⚠️ PRIMARY REFERENCE FILE ⚠️

**This file points to the main guidelines. For complete guidelines, see:**
**`.ai-coding-guidelines.md`** - This is the PRIMARY source of truth.

## Critical Workflow Rules

**NEVER commit without:**

- Showing user what files changed (`git status`)
- Showing user the actual changes (`git diff`)
- Getting explicit user approval
- User deciding commit message

**NEVER push to PR without:**

- Running `make ci` locally first
- Ensuring `make ci` passes completely
- Fixing all failures before pushing

## Coding Style

- Follow patterns in `.ai-coding-guidelines.md`
- Respect module boundaries (no business logic in CLI, no HTTP in config)
- Use Config for all runtime options
- Mock external dependencies in tests
- Add docstrings to all public functions (Google-style)

## Full Guidelines

**All detailed guidelines, patterns, and rules are in `.ai-coding-guidelines.md`**

**Before taking ANY action:**

1. Read `.ai-coding-guidelines.md` - This is the PRIMARY source of truth
2. Follow ALL CRITICAL rules marked in that file
3. Reference it for all decisions about code, workflow, and patterns

**Key sections:**

- Git Workflow (commit approval, PR workflow)
- Code Organization (module boundaries, patterns)
- Testing Requirements (mocking, test structure)
- Documentation Standards (PRDs, RFCs, docstrings)
- Common Patterns (configuration, error handling, logging)

**See `.ai-coding-guidelines.md` for complete guidelines.**
