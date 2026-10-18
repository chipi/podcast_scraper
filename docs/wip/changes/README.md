# Change records

One record per change to what the corpus serves, so every change goes through
[the corpus change method](../../guides/CORPUS_CHANGE_METHOD.md) step by step. Start from
[TEMPLATE.md](TEMPLATE.md) and name the file after the change (`0031_named_voices_shown.md`).

## Full or light

**Full method:** the change can alter what a listener or reader is shown about what was said and
who said it: names, roles and voice types, which insights are visible, quotes and transcript text,
the ad cut, the content of search results. In files, a migration that writes any of `gi.json`,
`kg.json`, `bridge.json`, segments, turns, context, speaker diagnostics, transcripts, or the
speakers in metadata. In code, the paths listed under `content_code` in
`config/corpus_change_gate.json` (migrations, naming, GI voice flags, the ad cut).

**Light:** everything else, such as covers, artwork sizes, chunk timestamps, version stamps or
index layout. Normal tests, a dry run and an undo; the record holds only `## 0. Why light`.
Examples: 0026, 0027, 0028.

The class is decided by what the change does, not by its label:

1. `scripts/measure/corpus_change_gate.py` applies the migration on a clone and records what it
   wrote and every scoreboard number before and after (`<id>.gate.json`). A write to a content file
   or content key makes it full.
2. Any move of a gating scoreboard number makes it full, whatever it writes.

## What enforces it

- **Before code:** a Claude Code hook (`.claude/settings.json`, `scripts/hooks/corpus_change_gate_hook.py`)
  refuses an edit to content code until an open record lists the path under `touches` and has
  steps 1 to 5 done (light: step 0).
- **Before a commit:** the git pre-commit hook (`make install-hooks`) runs the same check on the
  staged files, whatever wrote them.
- **Before it lands:** a unit test refuses a registered migration from 0031 on without a record,
  a gate report, and steps 1 to 8 done (light: step 0, and the gate must agree it is light).
