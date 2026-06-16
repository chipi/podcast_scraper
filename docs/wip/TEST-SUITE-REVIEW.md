# Integration + E2E test-suite review (post numpy2/pyannote4 migration)

Scope: 174 integration + 44 e2e files, reviewed by 4 parallel agents. The lens: do
tests actually validate behavior, or are they hollow/over-mocked (green while the
feature is broken — the diarization lesson)? Plus migration risk (numpy2/pyannote4/
spaCy3.8 — integration+e2e don't run in ci-fast).

## Headline: the diarization failure mode is SYSTEMIC, not a one-off

- **UPDATE (post-#1010 / diarization refactor):** e2e now has real pyannote
  diarization coverage — `tests/e2e/test_diarization_e2e.py` runs real pyannote on
  the v2 fixture audio (passes 4/4). The two bullets below describe the pre-refactor
  state and are kept as history.
- ~~**e2e has ZERO real pyannote diarization coverage.**~~ What's labelled "speaker
  detection" e2e is spaCy NER name-extraction, not audio diarization. (Superseded:
  real-pyannote e2e now lives in `tests/e2e/test_diarization_e2e.py`.)
- `tests/integration/providers/ml/test_diarization.py` is now a **mocked** integration
  test (patches `_create_pyannote_pipeline` + `_load_waveform`; no real pyannote, no
  HF token; markers `integration`, `diarization`). This is correct per the test
  pyramid (integration = mocked, e2e = real ML); real pyannote runs only in the e2e
  test above.
- The ML "integration" tier is overwhelmingly **mocks mislabelled as integration**:
  embedding/QA/NLI/model-loader tests fake the model and assert the fake's return.
  **No executing real-model assertion exists** for the numpy2 inference paths
  (embedding encode, NLI cross-encoder, QA) — exactly the regression class numpy2
  could introduce, invisible.

## Concrete hollow offenders (green, prove nothing)

- ✅ `tests/e2e/test_new_features_e2e.py` — RESOLVED in #901 (deleted; all 5 were
  `cfg.field == X` config echoes wearing `@pytest.mark.e2e`).
- ✅ `tests/integration/gi/test_evidence_stack_integration.py` — RESOLVED. Split into
  `TestEvidenceStackRealModels` (@ml_models, loads the real embedding/QA/NLI and asserts
  real outputs; skips-when-uncached) + `TestFindGroundedQuotesWiring` (injected-score
  threshold/error tests, where mocking is correct). Un-mocking proved the stack is healthy
  under transformers 4.57 / numpy2 — but see "grounding quality" below.
- `tests/integration/infrastructure/test_anthropic_mock.py:38-114` — STILL OPEN. class
  named `…E2EServerIntegration`, sets the e2e-server base_url, then `@patch(Anthropic)` so
  it never hits the server. No `client.base_url` assertion (unlike the OpenAI-compat mocks).
- ✅ `tests/integration/eval/test_eval_summarize_bundled.py` +
  `gi/test_bundled_extract_dispatch.py::TestBundledDispatchInPipeline` — RESOLVED. Now drive
  the real `run_experiment._eval_summarize` (via importlib, mypy-safe) and the real
  `_maybe_prefetch_bundled_candidates` + `_ground_insights_dispatch`, not pasted copies.

## Grounding quality (uncovered while un-mocking the evidence stack)

- The real roberta-squad2 + nli-deberta-v3-base pair returns **semantically off-topic
  spans** on toy inputs: for the insight "The capital of France is Paris." against
  "The capital of France is Paris. It has many museums and parks." the top grounded quote
  is "museums and parks" (qa≈0.12). NLI in isolation is healthy (entail 0.96 vs
  contradict 0.0); the QA candidate extraction is the weak link. The end-to-end test
  asserts integration *invariants* (verbatim span, scores in range, thresholds honored),
  NOT quote correctness — quality is an eval concern. **Follow-up:** a grounding-quality
  eval (not a unit assertion) over a real fixture corpus to quantify this.

## Real breakage found (not just hollow)

- ✅ `tests/integration/providers/ml/test_ml_provider.py` — RESOLVED in #901. The 4
  transcription tests' stale `@patch(...ml_provider.progress.progress_context)` (the
  `progress` module was removed in RFC-081, so they errored at patch time) were removed;
  the MLProvider transcribe path is validated again.
- `pyannote_provider.py:81` defaults to `pyannote/speaker-diarization-3.1` while
  pyannote 4.x is installed. **Deliberate** (TEST-SUITE-REVIEW intent below): 3.1 is the
  proven pipeline and loads correctly under pyannote 4.x — confirmed by a real run giving
  2 distinct speakers, and exercised end-to-end by `test_diarization_e2e.py`. Not a bug.

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
