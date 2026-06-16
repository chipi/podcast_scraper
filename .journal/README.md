# Agent journal (git-ignored)

Persistent, machine-local working journal for AI agents — **plans, approaches,
decisions, findings, run logs**: anything you'd otherwise lose when the session
context clears. The directory survives `/clear` (it lives on disk), but it is
**git-ignored**: only this `README.md` is tracked, so the convention travels with
the repo while the notes stay local.

## Convention

- **One file per work-stream/session:** `YYYY-MM-DD-<topic>.md`
  (e.g. `2026-06-16-964-residual.md`).
- **Append as you work:** what you did, what you decided and *why*, what's still
  pending, commands/results worth keeping. Treat it as a running log, not a
  one-shot doc.
- **Recover context** from the latest relevant entry at the start of a session
  instead of re-deriving it.
- **Promote, don't duplicate:** anything meant to be shared or shipped with a PR
  goes in `docs/wip/` (tracked, listed in `docs/wip/WIP_README.md`). The journal
  is the scratch/working layer beneath that.

See `AGENTS.md` → **Tool usage → Document location** for the canonical rule.
