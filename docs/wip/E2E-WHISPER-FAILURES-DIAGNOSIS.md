# 5 e2e Whisper failures — diagnosis + fix

**Symptom:** 5 e2e tests fail with
`Whisper transcription failed: An error happened while trying to locate the file
on the Hub and we cannot find the requested files in the local cache.`

**Failing tests:**

- `tests/e2e/test_full_pipeline_e2e.py::TestFullPipelineE2E::test_pipeline_downloads_audio_for_transcription`
- `tests/e2e/test_full_pipeline_e2e.py::TestFullPipelineE2E::test_pipeline_with_transcription`
- `tests/e2e/test_basic_e2e.py::TestBasicCLIE2E::test_cli_basic_transcript_download_path2`
- `tests/e2e/test_basic_e2e.py::TestBasicLibraryAPIE2E::test_library_api_basic_pipeline_path2`
- `tests/e2e/test_basic_e2e.py::TestBasicServiceAPIE2E::test_service_api_basic_run_path2`

## Diagnosis

Debug-level trace showed **Whisper itself succeeds**:

```
DEBUG  ml_provider.py:896  Whisper transcription finished in 2.10s (segments=27 text_chars=1250)
DEBUG  ml_provider.py:837  Transcription with segments completed in 2.10s (27 segments)
ERROR  episode_processor.py:1712  Whisper transcription failed: An error happened while trying to locate the file on the Hub...
```

The `.pt` file loads correctly, transcription runs, 27 segments emit. THEN the
error fires.

**Root cause chain:**

1. `cfg.diarize` defaults to `True` in `config.py:836`.
2. After successful transcription, `episode_processor.py:1622` calls
   `apply_diarization_to_result` unconditionally when `diarize=True`.
3. Diarization uses pyannote which loads a model via
   `pyannote_provider.py:27` → `Pipeline.from_pretrained` → HuggingFace Hub.
4. The test env doesn't have the pyannote model cached and doesn't have
   HF Hub auth → `OSError`.
5. The old catch at `episode_processor.py:1634` was
   `except (ProviderDependencyError, ValueError)` — `OSError` was not caught.
6. The exception bubbles up to the outer transcription catch at line 1704
   (`except (RuntimeError, OSError, ProviderError)`), which logs
   "Whisper transcription failed" and returns `(False, None, bytes_downloaded)`
   — **losing the successfully computed transcript.**

None of the 5 failing tests explicitly set `diarize=False`, so they all
inherit the default and fail the same way.

## Fix landed in this branch

**Product fix** — `workflow/episode_processor.py:1634`: broadened the
diarization catch to include `OSError` and `RuntimeError`. A diarization
failure now degrades to a warning ("falling back to gap-based screenplay")
and the transcript is preserved. Comment cites this diagnosis so the reason
survives future refactors.

**No test changes needed** — the product fix removes the failure entirely.
The 5 tests were correct to expect a working pipeline; the bug was that a
missing pyannote model masqueraded as a Whisper failure and dropped the
transcript.

## Validation

All 5 previously-red tests now green:

```bash
tests/e2e/test_full_pipeline_e2e.py::TestFullPipelineE2E::test_pipeline_downloads_audio_for_transcription PASSED
tests/e2e/test_full_pipeline_e2e.py::TestFullPipelineE2E::test_pipeline_with_transcription             PASSED
tests/e2e/test_basic_e2e.py::TestBasicCLIE2E::test_cli_basic_transcript_download_path2                 PASSED
tests/e2e/test_basic_e2e.py::TestBasicLibraryAPIE2E::test_library_api_basic_pipeline_path2             PASSED
tests/e2e/test_basic_e2e.py::TestBasicServiceAPIE2E::test_service_api_basic_run_path2                  PASSED
```

## What was NOT changed

- **`cfg.diarize` default.** Stays `True`. This is a product decision — the
  operator's default is diarized transcripts. Flipping it would silently
  change every existing config's behaviour.
- **Test explicit `diarize=False`.** Not needed with the product fix. If a
  future test wants to specifically exercise diarization failure it can set
  it explicitly.

## Non-goals / follow-ups worth their own issues

- Preload pyannote in test env or in `make preload-ml-models` for a richer
  local test experience (currently only openai-whisper `.pt` files preload).
- Consider a `diarization_required=False` semantics knob so operators can
  distinguish "diarize if possible, warn if not" vs "diarize or fail loudly."

## Standing carryovers still standing

- `PROD_MARKER_PR_TOKEN` runtime check — needs actual prod deploy, can't
  test offline.
