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
7. **Make commands MUST be assessable at the end.** ALWAYS invoke `make` with an exit-code terminator from the start, so the LAST line of output unambiguously says PASS/FAIL. NEVER re-run a `make` command "to check the exit code". See [How to run make commands](#how-to-run-make-commands) below.

## How to run make commands

You repeatedly burn 5–10 minutes by running `make ci-fast` (or any other long `make` target), failing to extract the exit status from the output, then re-running it "to verify". This is a permanent prohibition.

**The rule: every `make <target>` invocation MUST end with an exit-code terminator from the very first run, so PASS/FAIL is the last visible line.** Choose ONE form per call:

```bash
# Form A — the default, always-safe form. Last line is "MAKE_EXIT=0" or "MAKE_EXIT=N".
make <target>; echo "MAKE_EXIT=$?"

# Form B — single-shot interpretable terminator. Last line is "PASS" or "FAIL N".
make <target> && echo "PASS" || echo "FAIL $?"
```

**Why both lines instead of trusting the prior output:** `make` exits non-zero on the first failing subtarget but the FINAL printed line of a long run is whatever the last subtarget happened to print (often a build success message even when an earlier step failed). The output's *trailing prose* is NOT the exit code. Without the explicit terminator, you cannot tell PASS from FAIL by inspection — and you should NOT re-run the whole target to find out.

**Forbidden:**

- Running `make X` without an exit-code terminator, then running it again "to check".
- Piping `make X` through `tail`, `grep`, `head` without `set -o pipefail` AND an exit-code echo at the end.
- Treating "no obvious error in the last 60 lines" as PASS. Inspect the explicit `MAKE_EXIT=` / `PASS` / `FAIL` line.
- Re-running `make ci-fast` to "verify the exit code" of a previous run. If the prior run lacked the terminator, that is on you — work with what you have, ask the user, or grep the prior output for explicit failure markers. Do NOT spend another 10 minutes.

**On a failing `make ci-fast`:** identify the failing SUBTARGET (docs / lint / format / test / etc.) and validate the fix by re-running ONLY that subtarget — `make docs` is 10 s, `make ci-fast` is 10 min. Then run `make ci-fast` ONCE at the very end as the whole-gate confirmation. (See `feedback_no_redundant_ci_fast.md` and `feedback_subtarget_reverify.md` in user memory.)

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
