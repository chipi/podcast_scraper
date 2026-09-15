# Integration tests about the fixtures

These validate the **fixture corpus and its generator** — not the eval, and not
the application. They moved here in arc 2 from the deleted
`tests/integration/eval/`, because their subject was always the generator's
output rather than any research question.

| test | asserts |
|---|---|
| `test_v3_fixtures.py` | the v3 corpus "exercises every failure mode… is deterministic" |
| `test_v3_enricher_structures.py` | the generator can render the authored structures |
| `test_segment_time_drift_fixtures.py` | turn-boundary drift bounds on the v3 fixtures |
| `test_rttm_groundtruth.py` | the v3 ground-truth RTTM sidecars are consistent |
| `test_voice_assignment.py` | the "ONE VOICE PER PERSON" rule from `FIXTURES_SPEC.md` |
| `test_search_v3_mocks_shape.py` | the search-v3 mock shapes (already lived here) |

Support code, not tests: `segment_time_drift.py` (pure alignment/metric math over
abstract `(key, time)` streams — no whisper import) and
`segment_drift_harness.py` (the live transcription path, run manually with
`--regen`). Both were under `src/podcast_scraper/evaluation/` and
`tests/integration/eval/` before; every caller is a test in this directory.

## Why here and not `tests/fixtures/`

The data they validate lives in `tests/fixtures/` — `transcripts/v3/`,
`audio/v3/`, `ground-truth/v3/`, and the spec `FIXTURES_SPEC.md`. Putting the
tests beside the data is tempting and would be wrong, because CI selects tests
**by path**:

    pytest tests/unit/        -m 'not integration and not e2e'
    pytest tests/integration/ -m integration

A test under `tests/fixtures/` is collected by neither lane. It would pass
locally, appear in no CI run, and rot unnoticed — worse than not existing, since
it looks like coverage. `tests/fixtures/` holds fixture data plus helpers
(`mock_server/`, `enrichment/mock_scorers.py`, `connectivity-multi-show/
build_fixture.py`) and no `test_*.py` at all; that separation is deliberate.

So: the layout is type-then-subject, and the cost is that the spec and the tests
asserting it sit in different trees. Hence this file, and the pointer in
`tests/fixtures/FIXTURES_SPEC.md`.

## The generators

    scripts/build_v3_fixtures.py             the v3 corpus (does NOT emit audio;
                                             exposes AUDIO_VOICE_HINTS for the
                                             TTS step to consume)
    tests/fixtures/scripts/transcripts_to_mp3.py   the multi-voice TTS
    scripts/build_production_shaped_fixture.py
    scripts/build_fixture_cover_art.py
