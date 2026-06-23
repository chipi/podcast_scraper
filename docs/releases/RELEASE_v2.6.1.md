# Release v2.6.1 — `_is_test_environment` regression fix + structural cleanup

**Release Date:** 2026-06-23
**Type:** Patch Release (bug fix + reconciliation)
**Last Updated:** 2026-06-23

## Summary

v2.6.1 is a **patch release** centered on one operator-visible bug
fix + the structural cleanup that followed.

**The bug** (`_is_test_environment` false-positive on numpy imports):
between **2026-02-06** (PR #398) and **2026-06-22** (commit
`ce029849`), a runtime "is this a test run?" heuristic in
`src/podcast_scraper/config.py` silently auto-flipped **24 config
defaults** between TEST and PROD values. The heuristic looked at
`"unittest" in sys.modules` — but numpy lazy-imports `numpy.testing`
which pulls stdlib `unittest` into `sys.modules`. So **every
production code path that imports numpy looked like a test
environment**, silently routing model selection to TEST defaults.

5 months of eval runs, prod runs, and benchmarks all silently used
the cheap models the bug routed them to.

## What was actually affected

Of the 24 default-getters, **13 had TEST==PROD pairs** (no-op
flips — gemini-flash-lite, deepseek-chat, ollama llama3.1:8b for
both tiers). The 10 that meaningfully flipped under the bug:

| Knob | TEST value (silent prod) | PROD value (now active) |
|---|---|---|
| `ner_model` | `en_core_web_sm` | `en_core_web_trf` |
| `openai_summary_model` | `gpt-4o-mini` | `gpt-4o` |
| `anthropic_speaker_model` | `claude-haiku-4-5` | `claude-3-5-sonnet-20241022` |
| `anthropic_summary_model` | `claude-haiku-4-5` | `claude-3-5-sonnet-20241022` |
| `mistral_speaker_model` | `mistral-small-latest` | `mistral-large-latest` |
| `mistral_summary_model` | `mistral-small-latest` | `mistral-large-latest` |
| `grok_speaker_model` | `grok-beta` | `grok-2` |
| `grok_summary_model` | `grok-beta` | `grok-2` |
| local `whisper_model` | `tiny.en` | `base.en` |
| `summary_tokenize` defaults | safe minima | prod-mode registry tokenize |

Plus an incidental fix: `.env` file loading was **silently skipped
in prod** under the same false-positive (the helper ALSO gated `.env`
loading). Operators who relied on `.env` for API keys (rather than
shell exports) would have seen the file ignored. With the narrowed
detection, prod correctly loads `.env`; only pytest still skips it
(intentional hermeticity).

Same false-positive existed in
`src/podcast_scraper/evaluation/autoresearch_track_a.py` — meaning
`.env.autoresearch` silently failed to load during prod autoresearch
runs (operator-only `AUTORESEARCH_*` env-key overrides didn't actually
override). Fixed identically.

## Operator action items after upgrade

1. **Cloud profile users** whose pipelines fell through to defaults
   (rather than pinning specific models) will see their LLM model
   silently upgrade from the TEST tier to the PROD tier on the next
   pipeline run. Expect:
   - Higher OpenAI cost for summarization (gpt-4o vs gpt-4o-mini —
     ~10-15× per-token cost difference)
   - Higher Anthropic cost (claude-3-5-sonnet vs haiku — ~5-10×)
   - Higher quality entity extraction (`en_core_web_trf` vs `_sm`
     — measurably better PERSON recall, especially multi-token
     names)
   - First-pipeline-run NER cold-start ~10-15s extra as
     `en_core_web_trf` loads (same wheel is already pip-installed
     via `pyproject.toml`).

2. **Eval scoreboard caution**: 256 frozen eval runs at
   `data/eval/runs/` were captured during the bug window and used
   silent TEST defaults. **Comparing pre-v2.6.1 scoreboards to
   post-v2.6.1 runs is not apples-to-apples** for any of the 10
   meaningfully-flipped knobs above. A `_PRE_FIX_NOTE.md` is added
   to `data/eval/runs/` (this release) to flag this for
   future-you.

3. **`.env.autoresearch` operators**: this file actually loads now
   for prod autoresearch runs. Sanity-check what's in it; the
   override-`.env` semantics are now real.

## Structural cleanup landing alongside the fix

Per operator direction ("profiles play that role; we have dev vs
prod profiles and no need for another set of controls"), the
runtime test/prod auto-flipping mechanism is **removed entirely**:

- New profile YAML `config/profiles/test_default.yaml` pins all the
  test-tier cheap models that the old TEST_DEFAULT_* constants
  used to provide via env-detect.
- `tests/conftest.py` sets
  `PODCAST_SCRAPER_PROFILE=test_default` so tests inherit the
  cheap-model profile by default (the **single hardcoded thing**;
  everything else is profile-YAML-driven).
- 22 of 24 `_get_default_*` helpers in `config.py` are simplified
  to return the PROD constant directly. Profile YAMLs control
  TEST vs PROD via the merge order
  (registry < YAML < explicit kwargs).
- `_is_test_environment` renamed to `_is_pytest_run`. Its one
  remaining use is gating `.env` loading (hermeticity guard).
- `Config._merge_profile_into_data` falls back to
  `PODCAST_SCRAPER_PROFILE` env var when no explicit `profile=`
  passed — enabling per-deployment profile selection without
  changing call sites.

Full reconciliation doc:
[`docs/wip/POST_RFC097_DEV_PROD_REMOVAL.md`](../wip/POST_RFC097_DEV_PROD_REMOVAL.md).

## Tests

Full unit test sweep: **4331 passed, 0 failed, 1 skipped**.

## Docs updates

- `docs/architecture/ARCHITECTURE.md` § "Language-aware processing"
  — NER selection clarified (profile-driven, not env-driven).
- `docs/rfc/RFC-010-speaker-name-detection.md` lines 26 + 41 —
  same clarification.

## Commit chain

- `ce029849` (2026-06-22) — narrowed `_is_test_environment` to stop
  the bleed (small targeted fix)
- `d691610f` (2026-06-23) — sibling fix in
  `autoresearch_track_a.py` + ARCHITECTURE doc
- `c655cf11` (2026-06-23) — structural cleanup: profiles as source
  of truth, 22 default-getters simplified, conftest wiring

## Upgrade notes

Backwards compatible at the Python API surface
(`Config`, `run_pipeline`, `service.run`). Profiles previously
relying on the cheap-defaults silent flip will see model upgrades on
next run; explicit pins in operator-owned profile YAMLs are
**unaffected** (they always won over the env-detect).

No schema changes. No migration required.

## Known follow-ups (separate from this release)

- Cloud profile explicit pinning audit (the 5 fall-through cloud
  profiles in `config/profiles/`)
- Promotion of `test_default` profile to a full `ProfilePreset` in
  `_PROFILE_PRESETS`
- Inlining the 22 trivial `_get_default_*` helpers directly as
  `Field(default=PROD_DEFAULT_X)` in each Pydantic field
- TEST_DEFAULT_* vs PROD_DEFAULT_* constant consolidation in
  `config_constants.py` for the 13 TEST==PROD pairs

These are hygiene work, not blocking.
