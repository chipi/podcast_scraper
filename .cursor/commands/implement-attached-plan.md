# Implement attached plan (full pass)

Use when you have a **plan** (pasted, linked, or attached) and want a **single agent pass** that implements it end-to-end.

## Rules

1. **Do not edit the plan file itself** unless the user explicitly asks you to update the plan document.
2. **To-dos:** If the user says to-dos were already created, **do not create them again**. Mark the **first** todo `in_progress`, then `completed` as you finish each; do not stop until **all** are done (or you hit a true blocker and must stop with a clear reason).
3. **Scope:** Implement only what the plan specifies; avoid drive-by refactors and unrelated files.
4. **Repo conventions:** Follow `.cursorrules`, existing patterns, and project Makefile targets for tests/lint when you touch code.

## What you should ask the user to provide (in the same message or follow-up)

- The **plan** (full text, path like `.cursor/plans/foo.plan.md`, or attachment).
- Any **constraints** not in the plan (e.g. “no schema migrations”, “docs only”).
- If **to-dos are not** pre-created: say so, and the agent may create a minimal todo list from the plan once.

## After implementation (optional)

Offer a **short summary**: files touched, how to run the relevant tests, any follow-ups.
