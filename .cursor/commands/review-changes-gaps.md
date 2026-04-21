# Review work — gaps and inconsistencies

Use after a feature pass (or any substantial change) to get a **structured review** without necessarily editing the plan file.

## What to do

1. **Scope:** Infer scope from the conversation (recent commits, open PR, or files the user names). If ambiguous, ask **one** clarifying question.
2. **Search:** Grep / read for stale references (old paths, renamed configs, dead docs), workflow vs Makefile mismatches, tests that still assert removed behavior, and **semantic** bugs (e.g. code paths that read YAML but skip `profile:` merge where the service would not).
3. **Report in chat:** Group findings as **Fixed** (if you apply small obvious fixes), **Gaps** (should fix soon), **Intentional / low priority** (document or defer). Do not dump a wall of text—be scannable.
4. **Fixes:** Apply **small, safe** fixes inline (stale comments, wrong help strings, one-line assertion bugs). For larger design decisions, describe the tradeoff and wait unless the user said “fix everything you find.”

## Optional expansion (user can add in the same prompt)

- “**Also check** `docs/…` and CI workflows.”
- “**Do not** change behavior—review only.”
- “**Fix** all doc drift you find.”

## Out of scope unless asked

Rewriting the plan document, large refactors, or committing without user approval (use the project’s commit skill if they want a commit).
