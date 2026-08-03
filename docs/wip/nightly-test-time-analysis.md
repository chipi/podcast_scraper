# Nightly test-time analysis — why the timeouts crept 30→45→60

**Date:** 2026-08-03
**Trigger:** the nightly `test-integration` (30m) and `test-e2e` (45m) jobs kept hitting their
job `timeout-minutes` and getting *cancelled* (not failed). We bumped both to 60m as a
stopgap — this note is the "why are they slow" analysis so we stop treating the symptom.

**Data source:** `pytest-json-report` artifacts (`--json-report`, per-test setup/call/teardown
durations for *every* test) from the green CI run **30767636740** (`sha 4e8b1a7`, main). Same
suites the nightly runs.

## Headline: it's a handful of tests, not uniform growth

| Suite | Tests | pytest wall | Concentration |
|---|---|---|---|
| unit | 7623 | 5.2 min | well-distributed — **not the problem** |
| **integration** | 2957 | 10.7 min | **top 10 tests = 59%** of test-time; top 20 = 66% |
| **e2e** | 352 | 25.0 min | top 10 = 45%; top 20 = 60%; top 50 = 90% |

The slow jobs are dominated by a *dozen-ish* tests. Fix those and the distribution collapses.

## Root cause of the worst offenders: real retry-backoff `sleep`, not real work

The top integration offenders are **provider error-handling tests** — the API is **mocked to
raise**, but the provider then runs `retry_with_metrics(max_retries=3, initial_delay=1.0,
max_delay=30.0, jitter=True)` (`src/podcast_scraper/utils/provider_metrics.py`) with **real
exponential-backoff `time.sleep`s**. So the test just sits there sleeping out the retry
schedule. Nothing computes.

Integration top offenders (all Gemini error paths, API mocked):
```
123.1s  TestGeminiProviderErrorHandling::test_summarization_invalid_model_error
120.9s  TestGeminiProviderSummarization::test_summarize_api_error
 63.4s  TestGeminiProviderErrorHandling::test_speaker_detection_invalid_model_error
 62.0s  TestGeminiProviderTranscription::test_transcribe_api_error
 61.3s  TestGeminiProviderErrorHandling::test_detect_hosts_fallback_on_error
 59.4s  TestGeminiProviderErrorHandling::test_speaker_detection_rate_limit_error
```
≈ **7.5 min** across 6 tests — pure backoff sleep. Plus 4 `test_profile_reaches_orchestrator`
CLI-subprocess tests @ 35–41s (≈2.5 min).

E2E top offenders:
```
248.1s  TestHTTPErrorHandling::test_transcript_download_500_error   ) same retry-backoff
247.0s  TestRetryLogic::test_retry_on_500_error                     )  pattern, ≈8 min
183.3s  test_preprocessing_before_whisper_transcription   ) REAL ML (Whisper) — legitimately
123.3s  test_episode_processor_audio_download_and_transcription )  slow, not a bug
102.3s  TestHybridMLProviderE2E::test_hybrid_ml_provider_summarize  )
 95.5s  ...cleanup_is_idempotent / result_has_expected_keys        )
```

**Compounding factor:** `--reruns 2 --reruns-delay 1` re-runs any flaky test up to 3×. A
123s test that flakes on timing runs ~6 min. This is likely what tips the nightly over 30m
on a bad night (CI wall was only 10.7m; nightly job = ML-preload + deps setup ≈15m + pytest,
and a couple of reruns of the slow retry tests pushes it past 30).

## Two classes, two fixes

1. **Retry-backoff tests (the real win, ~15 min reclaimable).** The Gemini error tests + the
   HTTP-500 retry e2e tests are sleeping out real backoff with a mocked API. Fix: in these
   tests, neutralize the delay — patch `time.sleep`, or pass `initial_delay=0`/`max_delay=0`
   (or `max_retries` low) via the resilience profile/config the test drives. Turns
   minutes → milliseconds while still exercising the retry *logic*.
2. **Real-ML e2e tests (inherent, ~10 min).** Whisper transcription / hybrid-ML tests are
   genuinely doing ML work — not a bug. Options if we want them faster: smaller fixtures,
   mark them `slow` and shard, or move to a dedicated less-frequent job.

## Recommendation

- **Do #1 now** — patch the retry delay in the ~8 offending error tests. Biggest bang, no
  loss of coverage. Likely drops integration from ~11m→~4m of pytest and e2e meaningfully.
- Then the 60m timeout is comfortable headroom, not a band-aid, and we can revisit sharding
  for the real-ML tests separately.
- Consider whether `--reruns 2` belongs on the deterministic error tests at all (reruns are
  for genuine flakes, not for retry-logic tests).

## Implemented (2026-08-03)

**Root cause was retry-backoff `sleep`, and it's provider-agnostic.** Gemini merely dominated
the data because its *production* retry config is more patient than the others (6 retries @
up to 60s backoff vs the 3-retry/30s default — `gemini-2.5-flash-lite` rate-limits hard). The
same error test therefore waited out Gemini's longer schedule (~123s) vs ~7s elsewhere. The
config is correct prod behaviour; the test just shouldn't wall-clock-wait it.

**Fix (no per-directory gaps):** an autouse fixture in **`tests/integration/conftest.py`** and
**`tests/e2e/conftest.py`** that patches the retry util's `time` **module-scoped**
(`patch.object(provider_metrics, "time", <no-op-sleep proxy>)`). This neutralizes
`retry_with_metrics`' backoff sleep for **every** provider + resilience test in those tiers in
one place, while leaving global `time.sleep` (and every other module) untouched. The retry
*schedule* stays rigorously asserted against a mocked clock in
`tests/unit/.../test_provider_metrics.py` (unit tier — deliberately not covered by the
fixture, so those assertions still exercise the real backoff computation).

Measured: Gemini file 87 tests **~123s→2.5s**; all-providers 171 tests **7.6s**; an integration
resilience file **5.1s**; unit backoff-schedule test unaffected (48 passed).

**Still open — the second retry mechanism:** the e2e HTTP-download tests
(`test_http_behaviors_e2e.py::test_retry_on_500_error` etc., ~247s each) retry via the
**urllib3/requests adapter**, not `retry_with_metrics`, so the fixture above doesn't touch
them. They need a targeted urllib3-backoff patch (that file also has one-sided timing
asserts — `elapsed < 10` — which are safe, but validate before shipping). Follow-up, low
priority: only 2 tests, and the 60m job cap absorbs them for now.

## Not analyzed / caveats

- Numbers are from *one* green CI run (4e8b1a7); jitter + rerun behavior vary night to night.
- The nightly job's *setup* time (ML preload + dep install) wasn't separately profiled here —
  it's a real chunk of the 30m and worth a follow-up step-timing pass if #1 alone doesn't
  bring the job comfortably under 60m.
