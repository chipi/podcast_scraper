# AGENTS.md

<!-- markdownlint-disable MD029 MD007 -->

Universal repository instructions for AI coding agents (Claude, Cursor, Codex,
Aider, Cody, Gemini Code Assist, etc.). This is the **single source of truth**
for portable rules: stack, commands, conventions, "rules you keep breaking",
git workflow, tool usage, code-quality gates.

Tool-specific overlays live in:

- `CLAUDE.md` — Claude Code specifics (memory wiring, skills, slash commands)
- `.cursorrules` — Cursor specifics (`.cursor/rules/*.mdc` auto-load matrix)

Detail manuals (load on demand by any agent):

- `.ai-coding-guidelines-quick.md` — 90-line quick reference
- `.ai-coding-guidelines.md` — deep reference manual (~2,500 lines)
- `docs/guides/*` — topic-specific guides (TESTING_GUIDE, ENGINEERING_PROCESS,
  POLYGLOT_REPO_GUIDE, AGENT_BROWSER_LOOP_GUIDE, SERVER_GUIDE, …)

---

## RULES YOU KEEP BREAKING (read every session)

Not aspirational. These are the patterns where AI agents have failed this
operator repeatedly. Adherence beats every other rule.

1. **Never push without explicit user approval.** Not even a doc-only commit.
   Not even after CI is green. The user says "push" or you don't push.

2. **Never sync an open PR's branch with main unprompted.** Any push to a PR's
   HEAD restarts ALL required CI checks — ~30 min of burned compute for
   "avoiding a future merge conflict" the user resolves at squash-merge in
   seconds. Ask first.

3. **Do exactly what was asked, nothing more.** No "while I'm here, let me
   also..." steps. No optional cleanups. No memory-doc additions in the same
   task. If you see something else worth doing, say so as a suggestion in text
   — do not act on it.

4. **When the user is frustrated, stop proposing actions.** Acknowledge, ask
   what they want, and wait. Do not offer 3 options. Do not start another task
   to "make up for it". Do not write a memory file *while* they are still
   angry — that is tone-deaf. Acknowledge and wait.

5. **Read what was last asked, not what you think makes sense.** If they said
   "do A", do A. Don't infer A+B+C because the codebase suggests it.

6. **Validate the cost of an action before taking it.** "Does this restart
   CI?", "Does this push to a shared branch?", "Does this require approval I
   haven't gotten?" — answer those before running the command.

7. **Own agent-introduced automation; default to in-repo fixes, not homework.**
   When the user reports broken, flaky, or incomplete behavior in GitHub
   Actions, orchestrators, or other CI/infra glue this agent introduced or
   extended, do not substitute long diagnosis, psychology, or a checklist of
   manual steps for work you can do in the repo unless blocked (missing
   secrets, org/GitHub settings the repo cannot encode). Default: edit
   workflows/scripts, run `actionlint` and any applicable `make` targets,
   and summarize what changed. Ask when there are real forks (multiple
   defensible options, meaningful trade-offs) or when a move should not happen
   without their choice (irreversible or risky beyond the stated task).

8. **Never guess root cause on failures; always get proof first.** Do not
   explain why CI, Actions, deploy, or infra failed — and do not claim
   something is fixed — from memory, analogy, or "usually it's X." First pull
   evidence for THAT run/job: e.g. `gh run view <id> --json jobs` (failed
   `steps[].name`), `gh run view --log-failed`, or the user's pasted log;
   local runs: full failing output. Cite failing step name and error lines.
   If logs are inaccessible, say so once and fetch them or ask for a link —
   do not invent a diagnosis.

9. **`make` commands MUST be assessable at the end.** Invoke `make` with an
   exit-code terminator from the start so the LAST line of output
   unambiguously says PASS/FAIL. Two acceptable forms:

   ```bash
   make <target>; echo "MAKE_EXIT=$?"            # last line: MAKE_EXIT=0 / N
   make <target> && echo "PASS" || echo "FAIL $?"  # last line: PASS / FAIL N
   ```

   Never re-run a `make` to "check the exit code" of a prior run lacking the
   terminator. On failure: identify the failing SUBTARGET (docs / lint /
   format / test / etc.) and validate the fix by re-running ONLY that
   subtarget (`make docs` is 10 s, `make ci-fast` is 10 min). Re-run
   `make ci-fast` ONCE at the very end as whole-gate confirmation.

10. **Applying changes to live infra is a separate action class from committing
    them. Never trigger an apply/deploy/destroy-class operation without explicit
    per-instance approval — prior session "yes do it" does not extend.** "Yes
    do it" / "implement #N" authorizes code-side work: edits, commits, pushes,
    branch ops. It does NOT authorize:
    - `gh workflow run` of any workflow whose name contains "apply", "deploy",
      "destroy", "infra-apply", "deploy-prod", "drill-exercise", "failover".
    - `tofu apply` / `terraform apply` / `terraform destroy` against any
      remote state.
    - Direct provider-API mutations (Tailscale ACL push, Hetzner server / volume
      mutations, DNS record changes, GitHub branch-protection edits).
    - Manual SQL DDL or migrations against prod databases.

    These are gates the operator owns. The agent's job stops at "code is
    committed; here's how to apply it." When a change needs to go live, say:
    "needs `<workflow name>` / `tofu apply` to take effect; ready when you are."
    Never invoke it yourself, even if the prior commit was authorized.

    Failure mode of record (2026-05-29): an unauthorized `gh workflow run "Infra
    apply"` cascaded `hcloud_ssh_key.operator` drift into `hcloud_server.prod`
    replacement → prod VPS destroyed mid-session.

11. **`# forces replacement` in any plan output is a hard stop. `(sensitive
    value) # forces replacement` is doubly so.** Stateful resources (servers,
    volumes, databases, ACLs that gate live traffic) being marked for
    replacement means destroy + recreate, regardless of how innocuous the
    triggering change appears. `(sensitive value)` masks the diff so the
    agent cannot see what's actually changing or why.

    Operating rules:
    - Route infra / IaC / `tailscale/policy.hujson` changes through a **PR**,
      not direct-to-main, so `infra-ci.yml` posts the plan as a PR comment for
      human review. Direct-to-main bypasses the only pre-apply review surface.
    - Reading a plan: before any apply, search the plan output for the literal
      strings `forces replacement`, `must be replaced`, `will be destroyed`,
      and `(sensitive value)`. Any hit → quote the affected resources verbatim
      and ask, regardless of session momentum.
    - "It was just an ACL/policy edit, the server shouldn't be touched" is
      not a reason to skip the plan read. Cascades through resource
      dependencies (`ssh_keys`, `network_id`, etc.) regularly turn small
      changes into resource replacements.

---

## User intent beats procedural defaults

- Explicit user instructions override the procedural rules below.
- If scope or intent is unclear, ask first. Do not add unrequested steps.
- Non-negotiable safety: never commit secrets; never push unless asked;
  never combine a hotfix push with unrelated work.

## Autonomous execution

- When intent is clear, do the work in-session: run Makefile targets, scripts,
  tests, downloads, fixes. Don't close with "you should run X" if you can run X.
- Obvious single path → take it. Reserve questions for real forks: multiple
  defensible options, meaningful trade-offs, or outcomes that should not
  happen without the user choosing (destructive prod impact, policy-sensitive
  flags). One focused question beats a list of optional follow-ups.
- Ask only when blocked: missing secrets, ambiguous scope, unsafe request,
  genuine multi-way decision where the user must pick, commit/push approval.

---

## Stack

- **Python** 3.11.8, Pydantic v2, pytest, black/isort/flake8/mypy, MkDocs
- **Node** 22 (viewer: `web/gi-kg-viewer/`, Vue 3 + Vite + Cytoscape + Playwright)
- **ML**: sentence-transformers + faiss-cpu + torch (`[search]`), Whisper +
  spaCy + transformers + llama-cpp-python (`[ml]`), see `pyproject.toml`
  extras header
- **Infra**: Docker Compose stack-test, Tailscale for prod join, OpenTofu for
  cloud infra

## Project context

Python tool for podcast transcript downloading/generation:

- Download from RSS feeds, fallback to Whisper transcription.
- Speaker detection (spaCy NER), summarization (BART/LED or OpenAI/Gemini).
- Knowledge graph extraction → GI/KG viewer for exploration.

### Module boundaries (detail: `.cursor/rules/module-boundaries.mdc`)

- `cli.py` → CLI only | `service.py` → Service API only
- `workflow.py` → Orchestration | `config.py` → Config model only
- `downloader.py` → HTTP only | `rss_parser.py` → RSS parsing only

### Specs vs code

Don't embed `RFC-*`, `PRD-*`, or `UXS-*` identifiers in code, comments, CSS
class names, CLI strings, or user-visible copy. Use neutral names; keep
numbered references in `docs/rfc/`, `docs/prd/`, `docs/uxs/` only.

### Profile completeness (`config/profiles/*.yaml`)

Never ship a profile default referencing an enum value (`llm_pipeline_mode`,
`gi_insight_source`, `kg_extraction_source`, `summary_provider`, …) without
verifying a live code path reads it. Find the `Literal[...]` declaration,
grep the literal strings in `src/`, confirm every value has a dispatch arm
that does something different. **Half-wired features are worse than no
feature** — see #643 Phase 3C near-miss for the canonical example.

### GI/KG viewer UX (`web/gi-kg-viewer/`)

When changing viewer UX (visible copy, labels, routes, panels, layout, theme
tokens, accessible names used by tests), update in order:

1. `e2e/E2E_SURFACE_MAP.md` — automation contract first
2. Playwright — `e2e/*.spec.ts`, helpers; run `make ci-ui-fast`
3. `docs/uxs/` — `VIEWER_IA.md` for shell-IA changes; feature UXS for visual /
   token contracts

User-reported viewer bugs: reproduce + re-validate with **Chrome DevTools MCP
by default** (Playwright MCP only when clearly better — say so in one line).
Tests alone are not a substitute. See `docs/guides/AGENT_BROWSER_LOOP_GUIDE.md`.

---

## Git workflow

### Default: feature branch + PR. Exception: hotfix direct-to-main

- DEFAULT: `git checkout -b feat/name`, push, PR, merge.
- HOTFIX direct-to-main allowed ONLY when ALL apply: (a) fix for a regression
  already on main; (b) single-concern, small (≤ 50 LOC); (c) user explicitly
  asked for hotfix path or context makes it unambiguous.
- Hotfix protocol: pull main → edit → run pre-commit checks locally
  (`make lint` / `make docs` / spot pytest) → commit with `fix:` / `hotfix:`
  prefix → ask for push approval → push → watch CI on main → fix-forward if red.

### Never commit / push without explicit user approval

- `git status` → `git diff` → show user → wait for approval → THEN commit.
- After applying edits, briefly summarize and ask "Keep these changes or undo
  any of them?"
- Default: stage specific files for the current task (`git add <file>`).
  Avoid blind `git add -A` unless the user explicitly says so.

### Active-merge safety (when `.git/MERGE_HEAD` exists)

- NEVER `git stash` during a merge — destroys conflict resolutions.
- NEVER `git checkout <ref> -- <file>` during a merge — destroys resolved
  content. Only safe variants: `git checkout --ours/--theirs <file>` with
  explicit approval.
- NEVER overwrite local files with remote content without showing the diff
  and getting approval.
- NEVER `git checkout -- <path>` / `git restore <path>` to discard
  uncommitted edits unless explicitly approved.
- NEVER delete gitignored paths when "cleaning" — they may hold operator-local
  configs, experiments, or secrets.

---

## Tool usage

### Make commands, never direct tools

- **Use**: `make test`, `make format`, `make lint`, `make fix-md`,
  `make ci-fast`, `make docs`.
- **Don't use**: raw `pytest`, `black .`, `flake8`, `npx markdownlint-cli2`.
  Exception: `git`; Python: `.venv/bin/python3 script.py`.

### Viewer (`web/gi-kg-viewer/`): `npm run …` / `node_modules/.bin/<tool>`, never `npx`

The viewer pins `@playwright/test` but not a separate `playwright` package.
`npx playwright` silently fetches a fresh copy from the npm registry → two
module instances → "did not expect test.describe() to be called here" → worker
SIGKILL (exit 137). Easy to misread as user-canceled or OOM.

- **Use**: `npm run test:e2e -- <args>`, `npm run test:unit`, `npm run dev`,
  `npm run build`, `./node_modules/.bin/playwright test …`.
- Details: `docs/guides/POLYGLOT_REPO_GUIDE.md` (*Invoking viewer tools*).

### Foreground only — NEVER background for make/test/git/pip

- Background allowed ONLY for long-running servers (`mkdocs serve`, `make serve-api`).
- ALL `make`, `pytest`, `git`, `pip` MUST be foreground.
- Don't pipe to `| tail -N` / `| head` when the user is watching — most
  harnesses can't stream Bash stdout live; the user only sees output when the
  command exits. Prefer no pipe; if a command genuinely produces > 1k lines,
  use a generous tail (`tail -100`).

### Process safety — ML workloads

- NEVER retry a `make` that produced no output (likely zombie).
  Run `make check-zombie`.
- NEVER run multiple `make ci` / `ci-fast` / `ci-ui-fast` / `test`
  concurrently — process pileup on macOS causes unkillable zombies.
- After any hung/killed `make`, run `make cleanup-processes`.

### Update venv after dependency changes

After ANY edit to `[project.dependencies]` or `[project.optional-dependencies]`
in `pyproject.toml`, run `pip install --upgrade -e .[dev]` BEFORE running
tests or committing.

---

## Code quality and testing

### CI gating: run the right validation, not always the heaviest

- Default: `make ci-fast` before committing.
- Viewer-heavy diff: `make ci-ui-fast` (Playwright instead of Python e2e).
- Skip a duplicate full run when the same target already passed in this
  session and no substantive edits were made after.
- **No redundant ci-fast runs** to verify a small fix. Run the subtarget
  (`make docs`, `make lint`, `make type`, `make format`) — 10s vs 10min.
- **Documentation-only commits** (markdown under `docs/**`, `**/*.md`,
  `mkdocs.yml`, no `src/` or runtime code): use `make fix-md` (if needed),
  `make lint-markdown`, and `make docs`. Skip `make ci-fast`.
- **Pushing docs changes to `main` is a HARD GATE on `make docs` passing
  locally first.** `make docs` runs mkdocs in **strict mode** — any
  unresolved cross-reference (e.g., a link to a file you renamed without
  updating its referrers) fails the build with `Aborted with N warnings in
  strict mode!`. Pre-commit hooks do NOT catch this; only `make docs`
  does. Nightly will fail the next morning if you skip this step.
  - **After ANY file rename inside `docs/`**, also run
    `grep -rn "<old-filename-stem>" docs/` to find stale referrers before
    `make docs` — strict mode catches them eventually, but a grep is
    seconds vs. a 12 s build cycle per attempt.
- **Infra / CI / ops-only commits**: when the operator waives `make ci-fast`,
  still run `make lint-markdown` for markdown changes and `make docs` for
  doc-input changes. Otherwise no extra default local target.
- Why GitHub still runs Python CI for `Makefile` / workflow edits:
  `python-app.yml` `on.push.paths` includes those files. Path-filter
  behavior, not "the repo thought you changed Python."

### Final validation before push: real episodes, not just unit tests

Mocked unit tests prove dispatch routes correctly; they do NOT prove the
feature works against real provider responses, real transcripts, or the real
end-to-end pipeline. Before pushing any change touching a production pipeline
stage:

1. Run one real episode end-to-end with the changed code path (`.env` keys
   checked first).
2. Measure the claim numerically (LLM calls, file size, KG nodes — whatever
   the change claims to improve).
3. Inspect one artifact by eye.
4. Only then push.

### Workarounds vs. fixes

"Works in unit tests but not through the real pipeline" is a signature failure
mode. Half-wired features (Literal value with no dispatch arm, profile default
with no live code path, method exists but pipeline still calls the old one)
are regressions, not stubs. Don't ship them.

### Document location

ALL analysis/plans/WIP docs → `docs/wip/`. NEVER `docs/analysis/` or
`docs/plan/`.

---

## CodeQL alert dismissals

- **Trigger**: CodeQL fails on a PR.
- **Step 1**: Read `docs/ci/CODEQL_DISMISSALS.md` — enumerates every alert
  type, sanitiser chain, and full dismissal inventory.
- **Step 2**: Classify:
  - **(a) Known type, same pattern** (matches a numbered type, uses the same
    documented sanitiser chain): dismiss as false positive via `gh api`, add
    a row, tell the user.
  - **(b) New type or broken chain**: do NOT dismiss. Explain the taint flow,
    propose a code fix routing through an existing sanitiser, and wait for
    explicit user approval. If approved, add the new type to the registry.
- **Step 3**: Never dismiss silently. State alert number(s), file/line, type
  matched, and action.
- Inline `# codeql[...]` pragmas DO NOT actually suppress — they are docs
  only. Dismiss via `gh api` and log in the registry.

---

## GitHub

- Use MCP GitHub server tools when available; fall back to `gh` CLI if MCP
  is unavailable.
- ALWAYS use the correct GitHub username (check with `mcp_github_get_me`, not
  Mac username).

---

## Essential commands

```bash
make format        # black + isort
make lint          # flake8
make type          # mypy
make ci-fast       # Fast tests before commit (~6-10 min)
make ci-ui-fast    # Like ci-fast, Playwright instead of Python tests/e2e
make ci            # Full CI suite (~10-15 min)
make docs          # Build docs (before committing doc changes)
make lint-markdown # Check markdown
make fix-md        # Auto-fix markdown
```

---

## Detail file references

When the situation matches, load before proceeding:

| Situation | Load |
| --- | --- |
| Writing/modifying Python code | `.cursor/rules/coding-standards.mdc` |
| Writing or modifying tests | `.cursor/rules/testing-strategy.mdc` + `docs/guides/TESTING_GUIDE.md` |
| Editing `tests/unit/**` | `docs/guides/UNIT_TESTING_GUIDE.md` (must not depend on non-`[dev]` extras; never use `pytest.importorskip()` to sidestep) |
| Editing markdown / docs | `.cursor/rules/documentation.mdc` |
| PRD/UXS/RFC/ADR or `docs/architecture/**` | `.cursor/rules/engineering-process.mdc` + `docs/guides/ENGINEERING_PROCESS.md` |
| Git worktree operations | `.cursor/rules/git-worktree.mdc` |
| Refactoring / module boundaries | `.cursor/rules/module-boundaries.mdc` |
| About to replace tracked content | `.cursor/rules/git-working-tree-safety.mdc` |
| Viewer UI bug fixing | `.cursor/rules/agent-browser-ui-fixes.mdc` + `docs/guides/AGENT_BROWSER_LOOP_GUIDE.md` |
| About to commit or push | `.cursor/rules/ai-guidelines.mdc` |
| Configuring profiles | `config/profiles/*.yaml` — see *Profile completeness* above |
| Viewer UX changes | `web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md` first |

`PRD` = product what/why · `UXS` = UI contract when needed · `RFC` = technical
how (Draft → Completed) · `ADR` = immutable decision record (**Accepted**),
short, sourced from RFC discussion. Full flow + templates:
`docs/guides/ENGINEERING_PROCESS.md`.

---

## Documentation prose

In `docs/**` (PRD/RFC/ADR/guides/api/wip), `AGENTS.md`, `.cursorrules`,
`CLAUDE.md`, `.ai-coding-guidelines.md`, and `.cursor/rules/*.mdc`: do NOT
use checkmark / cross / clipboard emoji or decorative tick marks for status,
lists, or tables. Use plain words (Yes/No, Run/Skip, Good/Bad). Full rule:
`.cursor/rules/documentation.mdc`.

PRDs use **Implemented** / Partial / Draft. **Completed** is for RFC and ADR
headers, not PRDs.

---

**Source of truth:** this file.
**Tool overlays:** `CLAUDE.md` (Claude Code), `.cursorrules` (Cursor).
**Detail:** `.ai-coding-guidelines.md` / `docs/guides/*`.
