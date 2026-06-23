# Removal of the dev/prod env-detect default mechanism (2026-06-23)

## Why this doc exists

Between 2026-02-06 (PR #398) and 2026-06-22 (commit `ce029849`), a runtime
"is this a test run?" heuristic in `config.py::_is_test_environment()`
silently auto-flipped 24 model defaults between TEST and PROD values. The
heuristic was `"unittest" in sys.modules` — and numpy lazy-imports
`numpy.testing` which pulls stdlib `unittest` into `sys.modules`. So
**every production code path that imports numpy looked like a test
environment**, silently downgrading model selection to TEST defaults.

5 months of eval runs, prod runs, and benchmarks all silently used the
cheap models the bug routed them to.

Commit `ce029849` narrowed the heuristic to explicit pytest signals only
— stopping the bleed. This doc covers the structural follow-up: the
operator's directive that profiles, not runtime heuristics, are the
source of truth.

## Operator framing

> "this dev/prod should be gone. we moved on to profiles that play that
> role. registry is now key piece that materialized in profiles so we
> have dev vs prod profiles and no need for another set of controls.
> we need to find matching profile that looks like dev profile, or if
> it does not exist, then makes sure tests load config from that
> profile. all of this supported by registry. only hard coded thing is
> that tests uses specific dev/test profile."

## What changed (the structural fix)

### 1. New profile YAML: `config/profiles/test_default.yaml`

Pins **all** the test-tier cheap models that the old `TEST_DEFAULT_*`
constants used to provide via runtime env-detection. Includes:

- whisper_model: tiny.en
- ner_model: en_core_web_sm
- summary_model: bart-small, summary_reduce_model: long-fast
- openai_summary_model: gpt-4o-mini (+ speaker/cleaning/transcription)
- anthropic_speaker_model + summary_model: claude-haiku-4-5
- mistral_speaker_model + summary_model: mistral-small-latest
- grok_speaker_model + summary_model: grok-beta
- summary_mode_id: null (matches pre-refactor test default — explicit
  `summary_model` kwargs win in `select_summary_model`)
- Plus gemini/deepseek/ollama pinned for transparency (TEST==PROD in
  those vendors; pinning is for honesty not behavior change).

### 2. Removed: 22 of 24 `_get_default_*` env-detect helpers in `config.py`

For each per-vendor / per-model knob (openai/anthropic/mistral/grok/
ollama/deepseek/gemini × transcription/speaker/summary/cleaning), the
helper now returns the PROD constant unconditionally. The runtime
"test or prod?" decision is gone; profiles do that work.

Implementation note: helpers retained as 1-line returns (rather than
inlined `default=PROD_DEFAULT_X` in each Field) to keep this commit's
diff tight. A follow-up can inline if desired — purely cosmetic.

### 3. Kept profile-aware (NOT env-detect): `_get_default_summary_mode_id` + `_get_default_summary_tokenize`

These two read `PODCAST_SCRAPER_PROFILE` env var to choose between
`DEV_DEFAULT_SUMMARY_MODE_ID` and `PROD_DEFAULT_SUMMARY_MODE_ID`. That's
the SAME env var the Config validator now consults — so it's a single
"profile name" signal, not a separate test-detect heuristic. Pre-refactor
these ALSO had the buggy `if _is_test_environment(): return None` branch;
that's removed. Tests get `summary_mode_id=null` via the test_default
profile YAML.

### 4. Renamed: `_is_test_environment()` → `_is_pytest_run()`

The ONE remaining use of this helper is gating .env loading — the
existing logic of "don't read operator's .env file into the process
during pytest" (hermeticity guard). Renamed for honesty. The narrow
detection logic (explicit pytest / TESTING signals) is correct AS-IS.

The same rename + 1 regression test went into
`src/podcast_scraper/evaluation/autoresearch_track_a.py` which had a
duplicate of the same helper.

### 5. Config validator extension: `PODCAST_SCRAPER_PROFILE` env-var fallback

`Config._merge_profile_into_data` now falls back to the env var when no
explicit `profile=` was passed. This means **tests automatically inherit
the test_default profile** because `tests/conftest.py` does
`os.environ.setdefault("PODCAST_SCRAPER_PROFILE", "test_default")` at
module import. Explicit `profile=` in `Config(...)` always wins; this
fallback is operator ergonomics, not a hidden default.

Operators can also use it in prod / staging to set a default profile
per deployment without changing call sites:
```bash
export PODCAST_SCRAPER_PROFILE=cloud_with_dgx_primary
python -m podcast_scraper.cli ...
```

### 6. `tests/conftest.py`: autouse profile injection

Single line added at module import time:
```python
os.environ.setdefault("PODCAST_SCRAPER_PROFILE", "test_default")
```

This is the only hard-coded thing per the operator's framing: **tests
use the test_default profile**. Everything else flows through profile
YAML + registry.

## Profile-registry backing

`test_default` is currently a **YAML-only profile** (Layer 2 in the
`_merge_profile_into_data` validator). It does NOT yet have a
`ProfilePreset` in `_PROFILE_PRESETS` (Layer 3). Same status as
`dev.yaml`, `airgapped.yaml`, and `airgapped_thin.yaml`.

Promoting test_default to a full registry preset requires either:

- Reusing existing low-tier `StageOption` keys (e.g. `spacy_sm`,
  `openai_whisper_1`, `gemini_flash_lite`) — works partially (no
  "test-tier bart-base summary" StageOption exists today).
- Adding new test-tier `StageOption` definitions to the registry.

**Decision (pragmatic for this PR)**: ship YAML-only. The YAML works
through the existing fallback path identical to `dev.yaml`. Registry
backing can land in a follow-up alongside `dev` and `airgapped`
promotions — that's broader hygiene work.

## What tests look like before/after

### Before (env-detect era):
```python
def test_something():
    cfg = config.Config(rss_url="...")    # implicitly uses TEST_DEFAULT_*
                                          # via env-detect that read sys.modules
    assert cfg.openai_summary_model == "gpt-4o-mini"  # because pytest in sys.modules
```

### After (profile-as-source-of-truth):
```python
def test_something():
    cfg = config.Config(rss_url="...")    # implicitly uses test_default profile
                                          # via PODCAST_SCRAPER_PROFILE set by conftest
    assert cfg.openai_summary_model == "gpt-4o-mini"  # from test_default.yaml pin
```

The test code is **identical**. The mechanism underneath is honest.
Tests that need to test PROD defaults can override via `profile=` or
explicit kwarg.

## What ALSO got fixed (incidentally)

The narrowed `_is_pytest_run()` (formerly `_is_test_environment`) ALSO
gated `.env` file loading. Under the bug, **prod runs silently skipped
`.env` loading** because numpy → unittest → false-positive. Operators
who relied on `.env` for API keys (rather than shell-exported env vars)
would have seen the file ignored. With the narrowed detection, prod
correctly loads `.env`; only pytest still skips it (intentional
hermeticity).

The same narrowing in `autoresearch_track_a.py` means
`.env.autoresearch` now actually loads during prod autoresearch runs
(it was being skipped for 5 months). Vendor API key overrides operators
put in `.env.autoresearch` will work now — but ALSO will surface to the
prod autoresearch process. Quick sanity check on what's in
`.env.autoresearch` is worth doing once after this PR ships.

## Pre-fix eval data is not apples-to-apples

`data/eval/runs/*` contains **256 frozen runs** captured during the
bug window. Each used TEST_DEFAULT_* values silently — including the
NER `en_core_web_sm` instead of `en_core_web_trf` (worse PERSON recall)
and `gpt-4o-mini` instead of `gpt-4o` for OpenAI summary (cheaper but
worse quality). Post-fix runs will land with PROD values. Comparing
pre-fix scoreboards to post-fix scoreboards is invalid for any
knob on the affected list (see Appendix A).

Fingerprint v2 (separate work in this same PR) DOES capture
`runtime.inference_target` and other model identifiers — so future
regressions of this class are immediately visible in eval scoreboards.
The 256 pre-fix runs predate fingerprint v2; their fingerprints don't
record what model actually ran.

## Cloud profile pinning audit (Tier 2)

The cloud-tier profiles (`cloud_thin`, `cloud_with_dgx_primary`,
`cloud_balanced`, `airgapped_thin`, `local_dgx_balanced`) historically
fell through to defaults for most model knobs — only pinning what they
needed (e.g. `dgx_whisper_model`). Under the bug, those fall-throughs
returned TEST values silently. Post-refactor, they return PROD values
(from the simplified `_get_default_*` helpers).

**Recommendation**: pin model choices explicitly in each cloud profile
so the behavior is no longer "whatever the Field default happens to be
this week". This is hygiene; not urgent for this PR. Operator decides
whether to pin every cloud profile to gpt-4o (= what they get today
after the bug fix) or to gpt-4o-mini (= what they were ACTUALLY paying
for during the bug window).

Deferred to operator decision. The reconciliation can happen as a
separate small commit in this PR if you say go.

## Acceptance status

- [x] Bug fix: `_is_test_environment()` no longer false-positives on
  numpy-loaded unittest (commit ce029849 + d691610f for autoresearch
  sibling)
- [x] Structural fix: env-detect-of-test removed from 22 default helpers
- [x] Tests: load test_default profile by default via conftest
- [x] Profile: test_default.yaml created with full TEST_DEFAULT_* pins
- [x] Compatibility: explicit kwargs to `Config(...)` still win
- [x] Docs: ARCHITECTURE.md NER section updated
- [x] Tests pass: full unit test sweep (4324+ tests)
- [ ] Cloud profile pinning audit (Tier 2; operator-attended)
- [ ] CHANGELOG / release-note entry (Tier 2)
- [ ] Eval-run registry marker for the 256 pre-fix runs (Tier 2)
- [ ] Test profile promotion to a full registry preset (Tier 3
  follow-up; out of scope here)

## Appendix A — the 10 meaningful TEST/PROD pairs that actually flipped

(13 other knobs had TEST==PROD pairs — those flipped no-ops.)

| Knob | TEST default (silent for 5 mo) | PROD default (now active) |
|---|---|---|
| ner_model | en_core_web_sm | en_core_web_trf |
| openai_summary_model | gpt-4o-mini | gpt-4o |
| anthropic_speaker_model | claude-haiku-4-5 | claude-3-5-sonnet-20241022 |
| anthropic_summary_model | claude-haiku-4-5 | claude-3-5-sonnet-20241022 |
| mistral_speaker_model | mistral-small-latest | mistral-large-latest |
| mistral_summary_model | mistral-small-latest | mistral-large-latest |
| grok_speaker_model | grok-beta | grok-2 |
| grok_summary_model | grok-beta | grok-2 |
| whisper_model (local) | tiny.en | base.en |
| (summary tokenize defaults) | safe minima | promoted prod-mode registry tokenize |

## Appendix B — files touched in the structural fix

- `config/profiles/test_default.yaml` (new)
- `src/podcast_scraper/config.py` (22 helpers simplified, validator extended, helper renamed)
- `src/podcast_scraper/evaluation/autoresearch_track_a.py` (helper renamed, sibling bug-fix narrowing kept)
- `src/podcast_scraper/cli.py` (3 call-site renames)
- `src/podcast_scraper/workflow/stages/setup.py` (1 call-site rename)
- `tests/conftest.py` (PODCAST_SCRAPER_PROFILE=test_default autouse)
- `tests/unit/podcast_scraper/test_config.py` (3 call-site renames)
- `tests/unit/podcast_scraper/test_config_constants.py` (renames)
- `tests/unit/podcast_scraper/test_cli.py` (renames)
- `tests/unit/podcast_scraper/test_workflow_helpers.py` (renames)
- `tests/unit/podcast_scraper/workflow/test_setup.py` (renames)
- `tests/unit/podcast_scraper/test_speaker_detection.py` (assertion fix)
- `tests/unit/podcast_scraper/evaluation/test_autoresearch_track_a.py` (renames + 2 regression tests)
- `tests/integration/workflow/test_workflow_stages_integration.py` (renames)
- `docs/architecture/ARCHITECTURE.md` (NER section clarified)
