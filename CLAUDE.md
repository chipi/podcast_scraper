# AI Coding Guidelines for podcast_scraper

## Primary reference files

**For Cursor AI (automatic enforcement):**
**`.cursorrules`** - Critical rules enforced automatically by Cursor

**For all AI assistants (comprehensive guidelines):**
**`.ai-coding-guidelines.md`** - Complete AI coding guidelines (PRIMARY source of truth)

**Specs vs code:** Do not embed `RFC-*`, `PRD-*`, or `UXS-*` identifiers in code, comments, CSS class names, CLI strings, or user-visible copy. Use neutral feature and API names; keep numbered references in `docs/rfc/`, `docs/prd/`, and `docs/uxs/` only. See `.ai-coding-guidelines.md` (**Specification IDs in product code**).

**FUTURE checkout (this repo):** **`WORKTREE.md`** (repo root) — branch purpose (2.6+), no releases from here, PR-only flow, and **Python/venv per worktree** (use this worktree’s `.venv` for `python` / `pytest` / `make serve-api`; sanity-check `podcast_scraper.__file__`; do not add `pythonpath` to `pyproject.toml` to paper over the wrong interpreter).

## Quick Reference

**CRITICAL RULES:**

- **Never** push any branch without explicit user approval (commits OK after diff approval, pushes NEVER by default)
- **Never** commit without showing changes and getting user approval
- **Never** push to main branch (always use feature branches)
- **Always** show `git status` and `git diff` before committing
- **Always** wait for explicit user approval before committing
- After making file edits: summarize changes and ask "Keep these changes or undo any of them?"
- When intent is clear: **run commands and tools yourself** (make, scripts, tests); **only** ask when blocked (auth/secrets, ambiguous scope, or policy needs approval) — see *Autonomous execution* in `.cursorrules` / `.ai-coding-guidelines.md`
- When any make target fails (test, ci, lint, format, docs, etc.): establish root cause first, then fix from there (no random experimenting)
- Run `make ci-fast` before committing when needed; for **viewer-heavy** work prefer **`make ci-ui-fast`** locally first (**Playwright** instead of Python **`tests/e2e/`**; pre-commit still runs **`make ci-fast`**). Exceptions: workflow-only changes; **recent green `ci-fast` / `ci-ui-fast` or `ci` in this session on the same diff** with no substantive edits after; user says skip / already validated; **incremental tiny follow-up** — see `.cursorrules` rules **5** and **5c**
- **Always** use Makefile commands (never direct pytest/python/black commands)
- **Never** use `cd` to project root (already in workspace directory)
- **Always** use correct GitHub username (check with `mcp_github_get_me`, not Mac username)
- **Always** show terminal output for make/test commands (`is_background: false`)
- **Never** `git stash` during an active merge (destroys merge state and all conflict resolutions — see `.cursorrules` rules 4a–4c)
- **Never** `git checkout <ref> -- <file>` during a merge (destroys resolved content; use `git show <ref>:<path>` to inspect)
- **Never** overwrite local files with remote content without showing the diff and getting explicit approval (rule 4d)
- **Never** `git checkout -- <path>`, `git checkout HEAD -- <path>`, `git restore --source HEAD …`, or similar to discard uncommitted work unless the user explicitly asked or you asked and they confirmed (rule 4e). Before any revert-from-git on tracked paths: **`.cursor/rules/git-working-tree-safety.mdc`** (Rule 17) or **ask first**
- **Gitignored paths:** **never delete** them when “cleaning docs” or removing references; tracked `docs/**` must not treat them as canonical — see `.cursorrules` rule **4f** and `.cursor/rules/documentation.mdc`
- Run `make fix-md` immediately after ANY markdown edit (zero lint violations before review)
- **GI/KG viewer UX** (`web/gi-kg-viewer/`): when UI changes affect users or Playwright, update in order:
  **`e2e/E2E_SURFACE_MAP.md`** (automation contract) → **`e2e/*.spec.ts`** / helpers → **`docs/uxs/VIEWER_IA.md`** when **shell information architecture** changes (regions, navigation axes, persistence, clearing, first-run) → **`docs/uxs/UXS-001-gi-kg-viewer.md`**
  and/or the relevant **feature UXS** (`docs/uxs/index.md`) when the **visual/token** experience contract changes.
  UXS lifecycle (Draft vs Active, align at ship): **`docs/uxs/index.md`** — section **Living documents and ship boundary**.
  See `docs/guides/E2E_TESTING_GUIDE.md` (Playwright) and `docs/guides/DEVELOPMENT_GUIDE.md` (viewer section). For a **full local gate** on viewer-heavy PRs, prefer **`make ci-ui-fast`** (same lint/type/docs/build chain as **`ci-fast`**, with browser E2E).
- **User-reported viewer bugs:** reproduce and re-validate with **Chrome DevTools MCP by default** (Playwright MCP only when clearly better for scripted isolation — say so in one line); **validate the fix in the same channel you used to reproduce** (symmetry rule — tests alone are not a substitute if you reproduced in the browser). Also run **`make test-ui`** / integration server tests / **`make test-ui-e2e`** or **`make ci-ui-fast`** as appropriate. Workflow: **`docs/guides/AGENT_BROWSER_LOOP_GUIDE.md`** (*Default MCP choice* + *Obligatory validation* + *Symmetry rule*). For **graph neighbourhood / “everything bright” after ~1–3s**, use the **timing + Cytoscape + Pinia** checklist in **`.cursor/rules/agent-browser-ui-fixes.mdc`** (section *Graph canvas — selection, neighbourhood dimming*) and the companion subsection in **`docs/guides/AGENT_BROWSER_LOOP_GUIDE.md`**.
- **FastAPI `/api/*`**: tests in **`tests/unit/podcast_scraper/server/`** and **`tests/integration/server/`**; reference **`docs/guides/SERVER_GUIDE.md`**.
- **Local serve from a chosen output dir:** interpret “use this folder as root” as **`make serve SERVE_OUTPUT_DIR=…`** / **`make serve-api SERVE_OUTPUT_DIR=…`**; do **not** edit the Makefile default unless the user explicitly wants the repo default changed. **`VITE_DEFAULT_CORPUS_PATH`** is only for pre-filling the viewer shell path (see `.cursorrules` GI/KG section).
- **Agent-started servers:** when the agent starts **`make serve`** / **`make serve-api`** without the user naming a root, use **`SERVE_OUTPUT_DIR=.test_outputs`** unless another path is clearly implied (see `.cursorrules` GI/KG section).
- **FastAPI reload after server edits:** **Restart `make serve-api`** in-session with the same **`SERVE_OUTPUT_DIR`** rule (default **`.test_outputs`** for agent-initiated restarts when no path is given); verify **`/api/health`**; say **Ready for tests** with URL + root — **do not** only instruct the user to restart. Background **`serve-api`** is allowed under `.cursorrules` Rule **9**.
- **`.metrics/rule-adherence.jsonl`**: append at milestones and before commit/push (see `.cursorrules`) — rules/skills/subagents self-audit **only**; no retrospective fields.
- **Rule 18** (session review): same closing **cadence** often as the last metrics line, but **separate** — brief reflection; promote durable lessons into guides or `.cursor/rules` (not JSONL).

## Auto-load guides by file path (do not wait for the trigger phrase)

Load the relevant guide **before** editing, even when the user did not say
"load ai coding guidelines". The guide sets the rules; deciding without it
causes preventable CI breakage.

- **Editing `tests/unit/**`** → read `docs/guides/UNIT_TESTING_GUIDE.md`
  (especially *Pyproject extras: what unit tests may depend on*).
  `tests/unit/` must not depend on any non-`[dev]` extra (`[ml]`, `[llm]`,
  `[server]`, `[compare]`); mock the SDK module symbol with `@patch(…)` or a
  file-scoped autouse fixture instead. Never use `pytest.importorskip()` to
  sidestep the rule. If a test truly needs the real SDK, it belongs in
  `tests/integration/`.
- **Editing `tests/integration/**` or `tests/e2e/**`** → read
  `docs/guides/TESTING_GUIDE.md` and the relevant section of
  `docs/architecture/TESTING_STRATEGY.md`.
- **Adding a new provider / provider method / provider test** → read the
  two rules above plus the existing test file for the provider you're
  modifying (follow the established SDK-mock pattern).
- **Editing `config/profiles/*.yaml`** → see *Profile completeness* below.
- **Editing `web/gi-kg-viewer/**`** → already covered by the GI/KG rule
  above (E2E_SURFACE_MAP → specs → UXS).

## Profile completeness (config/profiles/\*.yaml)

Never ship a profile default that references an enum value
(`llm_pipeline_mode`, `gi_insight_source`, `kg_extraction_source`,
`summary_provider`, …) without verifying a **live** code path reads it.

Procedure before committing a profile change:

1. Find the field's `Literal[...]` declaration in `src/podcast_scraper/config.py`.
2. Grep the symbol and the literal strings in `src/` (`grep -rn
   "llm_pipeline_mode" src/`). Confirm every value in the `Literal` has a
   dispatch arm that actually does something different.
3. Trace to the stage that consumes it end-to-end. If the value is
   accepted by `Config` but no downstream dispatcher switches on it,
   **the profile is lying**: either wire the dispatch or drop the profile
   default to a value that is already live.

The `#643 Phase 3C` near-miss (shipping `llm_pipeline_mode: mega_bundled`
while the dispatch was deferred) is the canonical example — with no wiring
the new mode would have been *worse* than staged (extra LLM call + no cost
savings from GIL/KG skip).

## Half-wired features are worse than no feature

Generalises profile completeness to all config / flag / enum additions:

- Adding a new `Literal[...]` value, a new `Config` field, a new CLI flag,
  or a new method on a provider is only complete when **every code path
  the user could hit actually does the different thing**. "Method exists
  but pipeline still calls the old one" is a regression, not a stub.
- If a full end-to-end wiring is genuinely out of scope, do **not** change
  profile defaults, do **not** publicise the flag in docs as available, and
  do **not** add the `Literal` value. Ship the interior pieces as private /
  unreachable until the dispatch is wired.
- "Works in unit tests but not through the real pipeline" is the signature
  failure mode. A wiring test that calls the top-level entrypoint and
  asserts the downstream stage *skipped its LLM call* is the correct guard
  (see `tests/unit/podcast_scraper/workflow/test_prefilled_extraction_wiring.py`).

## Resuming from compaction: re-confirm deferred items

When a conversation summary carries over a todo tagged "deferred", "risky",
or "follow-up", do **not** silently act on it — also do **not** silently
keep it deferred if it would break the diff you are about to ship.
Re-state the item and ask: *"Summary says X is deferred. Does that still
apply, or do we need to do it now before pushing?"* The Phase 3C miss came
from carrying over a pre-compaction "defer to follow-up" tag without
re-asking; the profile change it was paired with turned into a lie.

## Final validation before push: real episodes, not just unit tests

Mocked unit tests prove the dispatch routes correctly; they do **not**
prove the feature works against real provider responses, real transcripts,
or the real end-to-end pipeline. Before pushing any change that touches
a production pipeline stage (summarization dispatch, GI/KG extraction,
transcription preprocessing, audio pipeline, any new `llm_pipeline_mode`
value, any profile default), the last step before commit is:

1. **Run one real episode end-to-end** with the changed code path, using
   the real provider API keys available in `.env`. There is no "I don't
   have keys" — check `.env` first.
2. **Measure the claim numerically.** If the change is "fewer LLM calls",
   count them. If it's "cheaper transcription", measure file size and $.
   If it's "better KG", count nodes. Unit tests don't measure claims.
3. **Inspect one artifact by eye.** Open the produced `gi.json` /
   `kg.json` / `metadata.json` / cleaned audio and confirm it looks sane.
4. **Only then push.** The test plan checkbox in a PR body is a
   post-merge reminder, not a replacement for pre-push validation.

For work in `scripts/validate/validate_phase3c.py` style: keep it as a
committed reusable harness so the next change in the same stage has a
ready comparison baseline.

## COMPLETE GUIDE FILE SET (LOAD ALL WHEN REQUESTED)

**When the user asks to "load ai coding guidelines" or "load coding guidelines", you MUST load ALL of these files:**

1. **`.ai-coding-guidelines.md`** — Main AI coding guidelines (PRIMARY source of truth)
2. **`docs/guides/CURSOR_AI_BEST_PRACTICES_GUIDE.md`** — Cursor AI best practices and model selection
3. **`docs/guides/DEVELOPMENT_GUIDE.md`** — Detailed development guide (code style, testing, CI/CD, architecture)
4. **`docs/guides/TESTING_GUIDE.md`** — Testing guide (unit, integration, E2E implementation)
5. **`docs/guides/MARKDOWN_LINTING_GUIDE.md`** — Markdown style and linting guide (style rules,
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
- **`.cursor/commands/*.md`** - Cursor **slash commands** (saved agent prompts), e.g. pipeline post-mortem and plan/review workflows; see **`docs/guides/CURSOR_AI_BEST_PRACTICES_GUIDE.md`**

**See the "Complete guide file set" section above for the complete loading pattern.**
