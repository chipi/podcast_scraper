# Eval runs before 2026-06-22 used silent TEST defaults

This directory is git-ignored (per `.gitignore` line 183) so this note
won't show up in PRs — but if you arrived here looking at frozen eval
scoreboards, **read this first** before drawing conclusions from any
run frozen before commit `ce029849` (2026-06-22).

## Why

Between 2026-02-06 and 2026-06-22, a `_is_test_environment()` helper
in `src/podcast_scraper/config.py` falsely reported test-environment
for every production run that imported numpy (numpy lazy-imports
`numpy.testing` which pulls stdlib `unittest` into `sys.modules`; the
helper looked at that and got fooled). The consequence: 10 model
defaults silently used their TEST tier values in production runs
instead of the PROD values. See
`docs/wip/POST_RFC097_DEV_PROD_REMOVAL.md` for the full table.

So every run frozen here BEFORE `ce029849` used:

- `en_core_web_sm` (test NER) instead of `en_core_web_trf` (prod) —
  measurably worse PERSON recall, especially multi-token names
- `gpt-4o-mini` instead of `gpt-4o` for OpenAI summary defaults
- `claude-haiku-4-5` instead of `claude-3-5-sonnet-20241022` for
  Anthropic speaker + summary defaults
- `mistral-small-latest` instead of `mistral-large-latest`
- `grok-beta` instead of `grok-2`
- `tiny.en` instead of `base.en` for local whisper
- Test-safe-minima for `summary_tokenize` instead of the
  promoted prod-mode registry tokenize

**...UNLESS the run's profile explicitly pinned the relevant model**.
Runs from profiles that pinned every knob explicitly (e.g.
`dev.yaml`, `local.yaml`, `airgapped.yaml` for the knobs they
pinned) are unaffected. Runs whose profiles fell through to
defaults (e.g. cloud_thin, cloud_with_dgx_primary if they didn't
explicitly pin a given LLM model) silently used the cheap tier.

## What's safe to compare

- **Same-commit runs** (frozen in the same window) compare fine
  with each other — they all hit the same silent-TEST tier.
- **Within-bug-window scoreboards** describe what the pipeline
  *actually did* at that point in time, but NOT what an operator
  using the same profile post-fix will get.

## What's NOT safe to compare

- **Pre-fix vs post-fix** runs are NOT apples-to-apples for any
  of the 10 affected knobs. The NER model alone is a +13pp
  PERSON-recall difference between tiers, so any KG/GI scoreboard
  comparison across the fix is invalid for entity-density work.

## How to tell which side of the fix a run is on

Frozen pre-fix runs predate **fingerprint v2** (the eval-fingerprint
gap-closure also in this branch — commits ~b269a9ac et al). Their
fingerprints don't capture `runtime.inference_target` or
`backing_model_id`. Post-fix runs ship with v2 fingerprints that
DO record the actual model used.

So:

- **No `fingerprint_version: 2.0` field** → pre-fix run, treat as
  ambiguous on the 10 affected knobs
- **`fingerprint_version: 2.0`** → post-fix run, scoreboards are
  honest

Future regressions of this bug class will be immediately visible in
v2 fingerprints — that's the structural insurance.

## Reference

- Full bug write-up:
  `docs/wip/POST_RFC097_DEV_PROD_REMOVAL.md`
- Release note:
  `docs/releases/RELEASE_v2.6.1.md`
- Commits: `ce029849`, `d691610f`, `c655cf11`
