# Coverage debt — Deepgram + diarization-graph work (PR #908)

`codecov/patch` flagged below-threshold patch coverage on PR #908 (non-blocking —
not a required check; the PR merged on review). This is the inventory of the
changed lines that lack coverage, to clear in a follow-up batch.

Three categories: (A) defensive branches — low value, optional; (B) genuinely
worth a quick test; (C) provisioning-gated tests that exist but skip in CI, so
they earn no coverage credit — the fix there is provisioning, not more tests.

## B — Worth a quick test (do these first)

- **`cli.py` — `--deepgram-api-base` flag + payload wiring.** The new argparse
  arg and `payload["deepgram_api_base"] = ...` line have no CLI-parse test.
  *Fix:* a `parse_args([... "--deepgram-api-base", "http://x"])` test asserting it
  reaches `cfg.deepgram_api_base` (mirror an existing `--*-api-base` CLI test).
- **`providers/deepgram/deepgram_provider.py:224-225` — `transcribe()` wrapper.**
  Untested; every test calls `transcribe_with_segments` directly. *Fix:* one test
  that calls `provider.transcribe(path)` and asserts it returns the `text`.
- **`workflow/episode_processor.py` — Deepgram chunk-local speaker warning.** The
  `if transcription_provider == "deepgram": logger.warning(... chunk-local ...)`
  only fires when Deepgram audio exceeds the 2 GB cap and chunks. *Fix:* a test
  that forces `chunker.needs_chunking` true for a deepgram cfg and asserts the
  warning (caplog), exercising the D3 degrade path.

## A — Defensive / error branches (optional; low value)

- **`deepgram_provider.py:114-115`** — `except ImportError` when `deepgram-sdk`
  isn't installed. Tests patch `_create_deepgram_client`, so the real raise never
  runs. Coverable by importing with the SDK hidden, but marginal.
- **`deepgram_provider.py:127-133`** — base-URL override `except Exception`
  fallback (only fires if the installed SDK lacks `DeepgramClientEnvironment`).
  *Fix (if wanted):* patch `deepgram.environment.DeepgramClientEnvironment` to
  raise and assert it falls back to the hosted `DeepgramClient` + warns.
- **`deepgram_provider.py:203, 206`** — screenplay edge branches: empty-text
  segment `continue`; `speaker is None` → `"Speaker"`. *Fix:* one screenplay test
  with a blank-text segment and a `speaker: None` segment.
- **`deepgram_provider.py:265->269, 338->348`** — partial branches (error-path
  `finalize()`; cost early-return when `audio_minutes <= 0`). Mostly covered;
  the partials are the not-taken sides.

## C — Provisioning-gated (have tests, skip in CI → no coverage credit)

These are NOT missing tests — they're real tests that `skip` unless the CI job is
provisioned. They show as uncovered in codecov because the bodies don't execute.
The lever is **provisioning the relevant CI job**, not writing more tests.

- `tests/e2e/test_evidence_stack_e2e.py` — real embedding/QA/NLI; skips unless
  `make preload-ml-models` cached them in the e2e job.
- `tests/e2e/test_diarization_e2e.py` — skips unless pyannote + the gated HF
  diarization models are provisioned (needs an HF token in CI).
- `tests/e2e/test_deepgram_provider_e2e.py` — skips unless `USE_REAL_DEEPGRAM_API=1`
  + a real key (opt-in, billed — intentionally off in CI).
- `tests/integration/infrastructure/test_deepgram_mock.py` — `importorskip("deepgram")`;
  skips if `deepgram-sdk` isn't installed in the fast-integration job. **Check:**
  confirm the fast-integration job installs `[llm]`; if it does, this one already
  contributes coverage and only the e2e/real ones remain gated.

## Suggested batch scope

Knock out **B** (3 small tests) — that lifts patch coverage on the genuinely-new
behaviour. Optionally do the `127-133` fallback test from **A** (it documents a
real degrade path). Leave the rest of **A** and treat **C** as an
infra/provisioning question (decide whether the e2e jobs should run with models +
an HF token, vs. staying skip-gated).
