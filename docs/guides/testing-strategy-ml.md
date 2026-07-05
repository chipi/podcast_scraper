# Testing strategy — ML providers and backends

Where to mock, where not to mock, and why. Written during #382 (transformers
v5 migration + Phase E/F/G architectural unification), 2026-07-05, based on
the mock-topology audit performed while migrating the summarizer + evidence
stack.

## The rule in one sentence

**Mock at OUR abstraction boundary, never at a library-internal name.** If
a test patches `transformers.pipeline` or `transformers.AutoTokenizer`, it
is testing the wrong thing.

## Why library-internal patches don't work

`transformers` (since >=4.40) exposes the top-level module as a
`_LazyModule` whose `__getattr__` resolves `pipeline` / `AutoTokenizer` /
`AutoModelForSeq2SeqLM` from a submodule on first access. This means:

- `@patch("transformers.pipeline", create=True)` **creates a new attribute**
  on the module and does not intercept the `from transformers import pipeline`
  call sites use — those go through `__getattr__` and get the real function.
- The same holds for `AutoTokenizer` / `AutoModelForSeq2SeqLM` / friends.

Before Phase E of #382 we had **seven** dead `@patch("transformers.*",
create=True)` decorators in `tests/unit/podcast_scraper/test_summarizer.py`
that never actually intercepted anything. They were removed in the Phase E
commit (`a5f671d6`). The pattern is documented as an anti-pattern in Issue
\#677 — hence the `_call_transformers_qa_pipeline` module-level indirection
that used to exist (pre-#382) to give tests a real patch target.

## Where to mock instead

### 1. Facade functions (module-level)

Tests that exercise orchestration or upstream logic mock the **module-level
facade function** that the code calls, not what the facade itself does:

```python
# tests/unit/podcast_scraper/gi/test_grounding.py
@patch("podcast_scraper.providers.ml.extractive_qa.answer_candidates")
def test_grounding_wires_qa_call(self, mock_answer_candidates):
    ...
```

This works no matter how `answer_candidates` is implemented internally.
Phase E rewrote `extractive_qa` from a bare-function module into a thin
wrapper over `QAEvidenceBackend`, and every one of these tests kept passing
unchanged.

**Good candidates for this seam:**

- `extractive_qa.answer_candidates`, `extractive_qa.answer_multi`
- `nli_loader.entailment_score`, `nli_loader.entailment_scores_batch`
- `embedding_loader.encode`
- `SummaryModel.summarize` (from external callers)

### 2. Class-method seams via `@patch.object`

For tests that exercise the internals of `SummaryModel`, mock the
class-method seam introduced in Phase 5:

```python
@patch.object(summarizer.SummaryModel, "_generate_summary")
@patch("podcast_scraper.summarizer.SummaryModel._load_model")
@patch("podcast_scraper.summarizer.SummaryModel._detect_device")
def test_summarize(self, mock_detect_device, mock_load_model, mock_generate):
    mock_generate.return_value = "expected summary"
    ...
```

The mock signature is `mock_generate(input_text, **kwargs)` — same shape as
`_generate_summary(input_text, **kwargs)`. `assert_called_once`, `call_args`,
`side_effect=[...]` all work naturally.

### 3. Instance-attr shadow (bulk-migration idiom)

When migrating many tests at once (as in Phase 5's `test_summarizer.py`
bulk transform, 34 patches across 7 tests), the pattern is:

```python
model.pipeline = True                       # loaded sentinel
model._generate_summary = mock_pipe         # shadows the class method
```

Python looks up instance attributes before class attributes, so
`self._generate_summary(...)` inside `summarize()` calls the mock. The mock
never receives `self` (it's a bound-mock replacement), so its signature is
`(input_text, **kwargs)` — same as `@patch.object`.

**Use this only for bulk migration.** For new tests, `@patch.object` is
more discoverable and integrates with pytest's fixture chain.

### 4. Backend abstraction seams

The Phase E/F backends are their own mock targets:

- `HFEvidenceBackend` (base) — use the `DummyBackend` subclass pattern from
  `tests/unit/podcast_scraper/providers/ml/test_hf_evidence_backend.py` for
  base-class contract tests. No `transformers` involvement.
- `QAEvidenceBackend`, `NLIEvidenceBackend`, `EmbeddingEvidenceBackend` —
  concrete integration tests should use small models with the local HF
  cache (`local_files_only=True`), NOT mocks. Mark with `ml_models`.
- `HFSeq2SeqBackend` — same. Base tests use `mock.MagicMock` at the
  `transformers.AutoTokenizer.from_pretrained` / `AutoModelForSeq2SeqLM.from_pretrained`
  return-value seam via `monkeypatch.setitem(sys.modules, "transformers", fake)`.

### 5. External SDK client (AI providers)

Cloud providers (OpenAI / Anthropic / Gemini / Mistral / Ollama) mock at
the top of the vendor client, not deeper:

```python
@patch("podcast_scraper.providers.openai.openai_provider.OpenAI")
def test_openai_provider_summarize(self, mock_openai_class):
    fake_client = mock_openai_class.return_value
    fake_client.chat.completions.create.return_value = ...
```

Never mock `httpx.request` or `openai._internal.transport.*` — those change
between SDK versions.

## Where NOT to mock (integration + parity paths)

- **`ml_models`-marked integration tests** load real small checkpoints
  (`bart-base`, `roberta-base-squad2`, `all-MiniLM-L6-v2`) from the local
  HF cache with `local_files_only=True`. Real model, real tokenizer, real
  forward pass. These are the tests that prove the backend actually works.
- **Phase 7 parity gate** (`scripts/eval/compare_v5_parity.py`) compares
  real end-to-end outputs pre-vs-post migration. No mocks.
- **QA baseline capture** (`scripts/dev/capture_qa_baseline.py`) — real
  QA model, deterministic device=CPU, fixed fixture set. No mocks.

## Test marker conventions

From `pyproject.toml [tool.pytest.ini_options] markers`:

- `unit + module_summarization` — internal SummaryModel logic (mock at
  `_generate_summary` seam).
- `unit + module_ml_providers` — evidence backends, HF seq2seq backend,
  hybrid backends (mock at facade / class-method seams or use `DummyBackend`).
- `integration + ml_models` — real small models from cache.
- `nightly + ml_models + slow` — parity gates against baseline artifacts.

## Anti-patterns (documented, don't repeat)

| Anti-pattern                                                | Why it fails                                                                 | Fix                                                                |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `@patch("transformers.pipeline", create=True)`              | `_LazyModule` — patches an unused attribute                                  | Remove; use facade or class-method seam                            |
| `mock_pipe = Mock(return_value=[{"summary_text": "X"}])`    | Couples test to pipeline output shape                                        | Change to `Mock(return_value="X")` and mock at `_generate_summary` |
| `model.pipeline = mock_pipe` (post-\#382)                   | `pipeline` is a bool sentinel now                                            | Use `model._generate_summary = mock_pipe`                          |
| Mocking `torch.cuda.is_available` at test-time to fake CUDA | Test escapes into real torch on any code path that imports torch differently | Use `resolve_evidence_device(mps_supported=False)` seam            |

## Related

- `docs/adr/ADR-068-bart-led-as-ml-production-baseline.md` — post-impl notes on the migration.
- `docs/wip/ISSUE-382-AI-PROVIDER-AUDIT-2026-07-05.md` — audit of AI-provider vs local-ML abstraction alignment.
- `docs/guides/UNIT_TESTING_GUIDE.md` — general unit-testing conventions.
- `docs/guides/INTEGRATION_TESTING_GUIDE.md` — integration test patterns.
