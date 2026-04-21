# Cursor custom commands (this folder)

Each **`.md` file** here becomes a **slash command** in Cursor Chat:

1. Open Chat (agent).
2. Type **`/`** — you should see commands named after the file **stem** (e.g. **`implement-attached-plan`**, **`review-changes-gaps`**, **`pipeline-run-and-report`**).
3. Pick the command, then **add your specifics** in the same message (plan text, paths, “review only”, etc.).

These are **not** macOS/Linux shell commands. For a **terminal** shortcut, use a shell alias that opens Cursor or runs the Agent CLI if you use it, e.g.:

```bash
# Example: open repo in Cursor (adjust for your install)
alias cur='cursor "$PWD"'
```

To add a new reusable prompt: copy one of these files, rename it (`kebab-case.md`), edit the body.
