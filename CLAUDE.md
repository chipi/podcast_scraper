# Claude Code instructions for podcast_scraper

The primary rules live in **`.cursorrules`**. Read it.

## Top of `.cursorrules` — RULES YOU KEEP BREAKING

These are the patterns where you have failed this user repeatedly:

1. Never push without explicit user approval.
2. Never sync an open PR's branch with main unprompted (any push to a PR's HEAD restarts ALL CI checks; that's ~30 min of burned time for "avoiding a future merge conflict" you could resolve in seconds at squash-merge).
3. Do exactly what was asked, nothing more — no "while I'm here" steps.
4. When the user is frustrated, stop proposing actions; acknowledge and wait.
5. Read what was last asked, not what you think makes sense.
6. Validate the cost of an action before taking it (does it restart CI? does it push? does it require approval?).

## Reference files (load on demand)

- **`.cursorrules`** — index + workflow rules (~180 lines, default load).
- **`.ai-coding-guidelines-quick.md`** — 90-line summary with quick commands and decision trees.
- **`.cursor/rules/*.mdc`** — topic-specific detail (testing, module boundaries, browser bug loop, etc.). Auto-load by file path; see `.cursorrules` "Auto-loading detailed rules" section.
- **`.ai-coding-guidelines.md`** — deep reference manual (~2,500 lines). Load specific sections on-demand only; don't load wholesale.
- **`.cursor/commands/*.md`** — saved slash-command prompts (`/review-changes-gaps`, etc.).

## Auto-load by file path

When editing files matching these paths, load the listed guide before editing:

- `tests/unit/**` → `docs/guides/UNIT_TESTING_GUIDE.md` (`tests/unit/` must not depend on any non-`[dev]` extra; never use `pytest.importorskip()` to sidestep).
- `tests/integration/**` or `tests/e2e/**` → `docs/guides/TESTING_GUIDE.md`.
- `config/profiles/*.yaml` → see *Profile completeness* in `.cursorrules`.
- `web/gi-kg-viewer/**` → see *GI/KG viewer UX* in `.cursorrules`.

## Project-specific rules not in `.cursorrules`

### Half-wired features are worse than no feature

A new `Literal[...]` value, `Config` field, CLI flag, or provider method is only complete when **every code path the user could hit actually does the different thing**. "Method exists but pipeline still calls the old one" is a regression, not a stub. If full end-to-end wiring is genuinely out of scope, do NOT change profile defaults, do NOT publicise the flag, and do NOT add the `Literal` value. The `#643 Phase 3C` near-miss (shipping `llm_pipeline_mode: mega_bundled` while the dispatch was deferred) is the canonical example.

### Resuming from compaction: re-confirm deferred items

When a conversation summary carries over a todo tagged "deferred", "risky", or "follow-up", do **not** silently act on it — also do **not** silently keep it deferred if it would break the diff. Re-state the item and ask.

### Final validation before push: real episodes, not just unit tests

Mocked unit tests prove dispatch routes correctly; they do NOT prove the feature works against real provider responses, real transcripts, or the real end-to-end pipeline. Before pushing any change touching a production pipeline stage:

1. Run one real episode end-to-end with the changed code path (`.env` keys are checked first).
2. Measure the claim numerically (LLM calls, file size, KG nodes — whatever the change claims to improve).
3. Inspect one artifact by eye.
4. Only then push.
