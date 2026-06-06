# Integration + E2E test-suite review (post numpy2/pyannote4 migration)

Scope: 174 integration + 44 e2e files, reviewed by 4 parallel agents. The lens: do
tests actually validate behavior, or are they hollow/over-mocked (green while the
feature is broken — the diarization lesson)? Plus migration risk (numpy2/pyannote4/
spaCy3.8 — integration+e2e don't run in ci-fast).

## Headline: the diarization failure mode is SYSTEMIC, not a one-off

- **e2e has ZERO real pyannote diarization coverage.** What's labelled "speaker
  detection" e2e is spaCy NER name-extraction, not audio diarization. `grep pyannote`
  across 44 e2e files = nothing executes.
- The one real-pyannote test (`tests/integration/providers/ml/test_diarization.py`,
  `-m diarization`) is **orphaned**: no Makefile/CI target selects `-m diarization`,
  it skips without HF token + cached model, and it's marked `integration` not
  `e2e`/`nightly`. Decorative until wired.
- The ML "integration" tier is overwhelmingly **mocks mislabelled as integration**:
  embedding/QA/NLI/model-loader tests fake the model and assert the fake's return.
  **No executing real-model assertion exists** for the numpy2 inference paths
  (embedding encode, NLI cross-encoder, QA) — exactly the regression class numpy2
  could introduce, invisible.

## Concrete hollow offenders (green, prove nothing)
- `tests/e2e/test_new_features_e2e.py` — all 5 tests are `cfg.field == X` config echoes
  wearing `@pytest.mark.e2e`. Bounded-queue/cache/metrics/degradation behaviour never run.
- `tests/integration/gi/test_evidence_stack_integration.py` — patches embedding+QA+NLI;
  green even if all three break under numpy2.
- `tests/integration/infrastructure/test_anthropic_mock.py:38-114` — class named
  `…E2EServerIntegration`, sets the e2e-server base_url, then `@patch(Anthropic)` so it
  never hits the server. No `client.base_url` assertion (unlike the OpenAI-compat mocks).
- `tests/integration/eval/test_eval_summarize_bundled.py` + `gi/test_bundled_extract_dispatch.py::TestBundledDispatchInPipeline`
  — test a **copy** of the dispatch logic pasted into the test, not the real `pipeline.py`.

## Real breakage found (not just hollow)
- `tests/integration/providers/ml/test_ml_provider.py` — 4 transcription tests
  (`:315,:335,:367,:387`) `@patch(...ml_provider.progress.progress_context)`, but
  `progress` was removed from the module (RFC-081). They **error at patch time** →
  the MLProvider transcribe path is currently UNVALIDATED. ci-fast doesn't catch it
  (integration excluded).
- `pyannote_provider.py:81` defaults to `pyannote/speaker-diarization-3.1` while
  pyannote 4.0.4 is installed. (Proven to work — direct run gives 2 speakers — but the
  default-model choice on 4.x should be deliberate.)

## Marker / gating issues
- `diarization` marker selected by **zero** Makefile targets (dead gate).
- Real-ML search tests (`test_hybrid_search`, `test_two_tier_indexer`,
  `test_query_router_ml`) load real models but lack `@pytest.mark.ml_models` → hard-error
  (not skip) when uncached.
- Integration LLM run drops `--disable-socket` (Makefile) → external network NOT blocked
  at runtime; only per-test mocking prevents a real paid API hit. (Latent cost risk.)
- `*_provider_e2e` real-API paths (`USE_REAL_OPENAI_API=1`) lack `@pytest.mark.network`.

## THIS PR — pyannote focus (Deepgram-STT diarization deferred to a separate PR)

Goal (operator): pyannote diarization is the DEFAULT path and is actually exercised
end-to-end (transcribe → diarize → screenplay → GI/KG/graph) in full e2e, with real
audio, not mocked.

1. **De-orphan the real diarization test** — add a `make test-diarization` (and nightly)
   target that selects `-m diarization`; ensure it runs once the model is preloaded
   (preload + HF token already built).
2. **Real diarization e2e (default path)** — a full-pipeline e2e on the two-voice
   fixture with `diarize=True` (default for local Whisper) that asserts the screenplay
   carries 2 distinct speaker labels and the GI/graph picks them up. ml_models+diarization
   gated, skips-when-uncached.
3. **Fix `test_ml_provider.py`** stale `progress` patches → restore transcribe coverage.
4. **Deliberate default model** on pyannote 4.x (keep 3.1 — proven — or move to
   community-1; document the choice).
5. Add real-model assertions for the numpy2 inference seam where pyannote-adjacent
   (the diarization e2e itself covers the pyannote seam).

Deferred to follow-up PRs (tracked, not lost): Deepgram integration tier + mock server;
hollow-test rewrites (new_features, evidence_stack, anthropic_mock, eval copies);
ml_models markers on search tests; socket-gating on the integration LLM run; provider
e2e network markers; en_core_web_trf NER coverage.
