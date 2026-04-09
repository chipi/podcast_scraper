# Cursor rule-adherence metrics

This folder tracks how the AI assistant uses project rules, skills, and subagents. It lives at **repo root** (`.metrics/`) so agent tools can append without sandbox blocks on `.cursor/`. It is **gitignored**; entries are local and per-session.

## File: `rule-adherence.jsonl`

One JSON object per line (JSONL). Each line records a **checkpoint**: rules applied/missed and, when relevant, skills and subagents used.

### When to append (incremental — do not skip)

- Append **throughout the session** whenever something **substantive** completes (e.g. green tests for a change, a feature slice landed, doc/lint pass, CI fixed) — **not only** at the end.
- Expect **multiple lines per session** (often **~2–10**; long sessions can be more). Under-logging is a common mistake; **err on the side of appending**.
- Also append **before commit/push**, on "done"/"wrap up", or at **session end** if you have not logged the latest milestone.
- **Do not ask** the user whether to append; write when `.metrics/` exists. A verbal announcement is optional (silent OK).

### Schema

See **SCHEMA.md** in this folder for the schema (fields, types, examples).

- **rules_applied / rules_missed**: Use the numbering from `.cursorrules` (0, 1, 2, 2a, 3, 4, 5, 8a, 8b, 9, 9a, 12, 13, 14, 15, 16) or short labels. Empty list is allowed for either.
- **skills_used**: Only include skills you actually used (e.g. from `~/.cursor/skills/` or project skills). Omit key or use `[]` if none.
- **subagents_used**: Only include subagents you delegated to (e.g. verifier, ci-fix-loop, acceptance from `.cursor/agents/`). Omit key or use `[]` if none.

## Known subagents (this project)

- `verifier` — Run format-check, lint, lint-markdown, full `make ci`; report only.
- `ci-fix-loop` — Run `make ci`, fix failures, re-run up to 3 times.
- `acceptance` — Run `make test-acceptance CONFIGS="config/acceptance/*.yaml"` (or user pattern); return session ID and summary.

## Known skills (referenced in project rules)

Skills are typically under `~/.cursor/skills/` or project-specific. Examples referenced in `.cursor/rules/*.mdc`: commit-with-approval, push-to-pr, create-prd-rfc, markdown-pre-commit, git-worktree-setup, git-worktree-recovery, efficient-pytest-runs, test-scope-decision-tree. New skills can be added; list them in `skills_used` when used.

## Helper script

From repo root: `.metrics/check_last_entry.sh` — checks whether the last JSONL line looks recent.
