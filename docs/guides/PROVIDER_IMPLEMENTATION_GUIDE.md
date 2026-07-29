# Provider Implementation Guide

This comprehensive guide explains how to implement new providers for the podcast scraper. It consolidates information from multiple guides and uses OpenAI as a complete example throughout.

## Overview

The podcast scraper uses a **protocol-based provider system** where each capability (transcription, speaker detection, summarization) has a protocol interface that all providers must implement.

This design allows:

- **Pluggable implementations**: Swap providers via configuration
- **Type safety**: Protocols ensure consistent interfaces
- **Easy testing**: Mock providers for testing
- **Extensibility**: Add new providers without modifying core code

## Architecture

### Provider Types

1. **TranscriptionProvider**: Converts audio to text
2. **SpeakerDetector**: Detects speaker names from episode metadata
3. **SummarizationProvider**: Generates episode summaries

### Unified Provider Pattern

As of v2.4.0, the project follows a **Unified Provider** pattern where a single class implementation handles multiple protocols using shared libraries or API clients.

- **`MLProvider`**: Unified local implementation using Whisper, spaCy, and Transformers.
- **`HybridMLProvider`**: Combines local ML MAP phase + LLM REDUCE phase.
- **`OpenAIProvider`**: Unified API implementation using OpenAI's various endpoints.
- **`GeminiProvider`**: Google Gemini API (transcription + summarization).
- **`AnthropicProvider`**: Anthropic Claude API (summarization only).
- **`MistralProvider`**: Mistral API (summarization only).
- **`GrokProvider`**: Grok/xAI API (summarization only).
- **`DeepSeekProvider`**: DeepSeek API (summarization only).
- **`OllamaProvider`**: Local self-hosted LLMs (transcription, speaker detection, summarization).

**File Structure:**

```text
src/podcast_scraper/
├── providers/
│   ├── ml/
│   │   ├── ml_provider.py           # Unified Local ML implementation
│   │   ├── hybrid_ml_provider.py    # Hybrid MAP-REDUCE implementation
│   │   ├── whisper_utils.py         # Whisper transcription utilities
│   │   ├── speaker_detection.py     # spaCy NER speaker detection
│   │   └── summarizer.py            # Transformers summarization
│   ├── openai/
│   │   └── openai_provider.py       # Unified OpenAI API implementation
│   ├── gemini/
│   │   └── gemini_provider.py       # Gemini API implementation
│   ├── anthropic/
│   │   └── anthropic_provider.py    # Anthropic API implementation
│   ├── mistral/
│   │   └── mistral_provider.py      # Mistral API implementation
│   ├── grok/
│   │   └── grok_provider.py         # Grok API implementation
│   ├── deepseek/
│   │   └── deepseek_provider.py     # DeepSeek API implementation
│   └── ollama/
│       └── ollama_provider.py       # Ollama local LLM implementation
├── transcription/
│   ├── base.py              # Protocol definition
│   └── factory.py           # Factory logic
├── speaker_detectors/
│   ├── base.py              # Protocol definition
│   └── factory.py           # Factory logic
└── summarization/
    ├── base.py              # Protocol definition
    └── factory.py           # Factory logic
```

### Local HuggingFace backends (post-#382)

Local ML providers do **not** call `transformers.pipeline()` directly. They
delegate load + generate through one of two shared backends:

- **`HFEvidenceBackend`** (`providers/ml/hf_evidence_backend.py`) — the base
  class for QA, NLI, and sentence-embedding backends
  (`QAEvidenceBackend`, `NLIEvidenceBackend`, `EmbeddingEvidenceBackend`).
  Owns device resolution (with per-subclass `mps_supported` flag), the
  standard `from_pretrained` kwargs (`local_files_only=True`,
  `low_cpu_mem_usage=False`, `trust_remote_code=False`), and a
  per-subclass process-wide instance cache with a threading lock. Reach for
  it whenever you want to add a new evidence-style extractor (a new QA head,
  a new entailment scorer, a new embedding model).

- **`HFSeq2SeqBackend`** (`providers/ml/hf_seq2seq_backend.py`) — one
  loader/generator for the BART / LED / Pegasus / LongT5 / FLAN-T5 family.
  `SummaryModel` (map profile) and `TransformersReduceBackend` (hybrid
  reduce profile) both delegate through it. Snapshot-first checkpoint
  discovery, family-class override (`family_class`), and a retry hook
  (`retry_wrapper`) are the extension points. Reach for it when adding
  a new seq2seq summarization or text-generation surface.

**How to add a new HF-backed capability:**

1. Subclass the right backend (`HFEvidenceBackend` for scoring/matching,
   `HFSeq2SeqBackend` for generation).
2. Override just the class attributes that differ (model kind, family
   class, mps flag, task-specific decoding defaults). The base classes
   handle download, cache, device coercion, and load ordering.
3. If your task needs custom generation kwargs, express them via
   `transformers.GenerationConfig` — never as arbitrary keyword args.
   That keeps behavior parity with pipeline-era defaults and makes
   determinism explicit.
4. Add a fixture-based regression baseline under `data/eval/references/`
   (see `scripts/dev/capture_*_baseline.py` for the pattern) and a
   comparator step in `scripts/eval/full_ml_recheck.py`. The nightly
   regression test at `tests/e2e/test_v5_parity_regression.py`
   auto-enforces new entries.

The old `transformers.pipeline("…")` call sites are **retired in v5** and
must not come back — v5 removed several of the task strings entirely
(`"question-answering"`, `"summarization"` in the removed form, etc.).
For test-mocking guidance see [`testing-strategy-ml.md`](testing-strategy-ml.md).

## Step-by-Step Implementation

### Step 1: Understand the Protocol

First, examine the protocol interface in `{capability}/base.py`. For example, `TranscriptionProvider`:

```python
from typing import Protocol

class TranscriptionProvider(Protocol):
    def initialize(self) -> None:
        """Initialize provider (load models, connect to API, etc.)."""
        ...

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> str:
        """Transcribe audio file to text."""
        ...
```

### Step 2: Implement the Provider Class

Create a new file for your provider. If your provider handles multiple capabilities, consider a unified structure like `openai/` or `ml/`.

**Reference Implementation**: `src/podcast_scraper/providers/openai/openai_provider.py`

#### 1. Configuration Validation

Check required config fields in `__init__()`. API keys should be validated here.

#### 2. Thread Safety

Define `_requires_separate_instances` based on your implementation:

- `True`: For local ML models (HuggingFace/Whisper) that are not thread-safe.
- `False`: For API clients (OpenAI) that handle concurrent requests internally.

#### 3. Initialization Lifecycle

- **`__init__`**: Store configuration and initialize lightweight clients.
- **`initialize()`**: Load heavy resources (ML models) or perform network handshakes. This method must be idempotent.
- **Lazy Loading**: Call `initialize()` inside protocol methods if not already initialized.

#### 4. Error Handling

Use typed exceptions from `podcast_scraper.exceptions`:

- `ProviderConfigError`: For invalid configuration.
- `ProviderDependencyError`: For missing packages or models.
- `ProviderRuntimeError`: For API failures or inference errors.
- `ProviderNotInitializedError`: If a method is called before `initialize()`.

#### 5. Prompt Store (for LLMs)

Use the centralized `prompt_store` for LLM prompts:

```python
from ..prompts.store import render_prompt, get_prompt_metadata

# Render a versioned prompt
system_prompt = render_prompt("summarization/system_v1")
```

### Step 3: Register in Factory

Update the factory functions in `{capability}/factory.py` to include your new provider.

```python
def create_transcription_provider(cfg: config.Config) -> TranscriptionProvider:
    # ...
    if provider_type == "whisper":
        from ..providers.ml.ml_provider import MLProvider
        return MLProvider(cfg)
    elif provider_type == "openai":
        from ..providers.openai.openai_provider import OpenAIProvider
        return OpenAIProvider(cfg)
    # ...
```

---

## Response-shape guardrails (ADR-105 / ADR-100)

When you add a new chat-completion provider (cloud or self-hosted), wire the
response-shape guardrail at every content-producing call site — summarize,
summarize_bundled, generate_insights, KG extraction, clean_transcript,
speaker detection if it returns prose. The helper catches the failure modes
the SDK can't (empty content / thinking-prose markers / `finish_reason=length`
/ unparsable JSON when expected).

```python
from .. import guardrails as _guardrails

# at the content-producing call site, after extracting content + finish_reason
content = response.choices[0].message.content          # provider-specific path
finish_reason = response.choices[0].finish_reason      # or stop_reason / candidates[0].finish_reason
_guardrails.check_chat_response(
    content,
    service="<your-service-name>",   # short string, no deployment details
    finish_reason=finish_reason,
    expect_json=False,                # True only for JSON-out call sites
)
```

The `service` kwarg is the Prometheus label
(`inference_guardrail_violations_total{service, reason}`) and the
`GuardrailViolation.service` attribute. Pick a fixed short string
(`"openai"` / `"anthropic"` / etc.) and **don't embed deployment details**
(no `"openai-via-azure"`, no `"gemini-prod"`).

### The wrap-into-ProviderRuntimeError trap (ADR-100 §A)

If your provider's call site has a broad `except Exception` that maps into
`ProviderRuntimeError` / `ProviderAuthError` for the operator-facing error
system, `GuardrailViolation` will be silently wrapped and the
`FallbackAwareSummarizationProvider` layer will never see the type. Always
add an explicit passthrough **before** the broad except:

```python
except _guardrails.GuardrailViolation:
    raise  # ADR-100: let FallbackAware see the raw type, don't wrap
except Exception as exc:
    # existing error-classification block
```

### Per-stage failure handling

Cleaning is graceful (catch and degrade to original text); summarize / GI /
KG / speaker are fail-up. See
[ADR-100 §3](../adr/ADR-100-response-shape-guardrails-for-cloud-llm-providers.md#3-failure-handling-per-stage-not-per-provider)
for the matrix and reasoning. Cleaning template:

```python
except _guardrails.GuardrailViolation:
    logger.warning(
        "<Service> cleaning output failed guardrail; returning original transcript text"
    )
    return text   # NOT raise — cleaning's contract permits the no-op fallback
```

---

## Testing Your Provider

### E2E Server Mock Endpoints

For API providers, you must add mock endpoint handlers to the E2E test server (`tests/e2e/fixtures/e2e_http_server.py`). This allows tests to run without real API keys or internet access.

If the provider exposes a chat-completion-shaped endpoint, also extend the
mock server's `inject_violation` registry so guardrail E2E tests can target
it. The vocabulary is documented at the top of the `_injected_violations`
declaration in the mock server; follow the existing
`_emit_chat_violation` / `_emit_anthropic_violation` /
`_emit_gemini_violation` pattern.

### Testing Checklist

- [ ] **Unit Tests**: Test logic in isolation, mock all external dependencies.
- [ ] **Integration Tests**: Test provider with the real E2E server mock endpoints.
- [ ] **E2E Tests**: Test provider in the full pipeline context.
- [ ] **Guardrail E2E** (chat-shaped providers): inject empty / thinking-prose / finish-length response via the mock server, assert `GuardrailViolation` propagates out of the public method (not wrapped into `ProviderRuntimeError`). See `tests/e2e/test_cloud_guardrails_e2e.py` for the template.
- [ ] **Resilience E2E** (chat-shaped providers): inject permanent 5xx + transient 5xx via `set_error_behavior` / `set_transient_error`, assert behavior matches the per-stage contract. See `tests/e2e/test_cloud_resilience_e2e.py`.
- [ ] **Resource Management**: Verify `cleanup()` properly unloads models or closes connections.
- [ ] **Diarization providers only**: add a `DiarizationLabelingStrategy` tuned to how the model clusters and wire it in `labeling_strategy_for()`; validate with a single-variable reprocess pilot (only `diarization_provider` changed) and keep the frozen base + N1/dedup tests green. See "Adding a diarization provider" above and ADR-134.

## Adding a diarization provider — it needs a labeling strategy (ADR-134)

Diarization is not just another provider: its output (`SPEAKER_NN` clusters) feeds **speaker
labeling** (`providers/ml/diarization/roster.py`), and labeling is **coupled to how the model
clusters**. There is no diarizer-agnostic labeling heuristic — the same audio clusters differently
per model, and each shape breaks a different labeling assumption:

- **Deepgram (coarse):** merges a show's cold-open montage into the host's own cluster. The
  host-candidate rule "the first voices to speak are the hosts" works *because* of this merge.
- **pyannote community-1 (fine):** splits each host into their own cluster (with an ASR-garbled
  self-intro), leaves recurring promo readers as their own first-speaking clusters, and splits some
  guests across two clusters. "First to speak" now crowns a cold-open ad clip, garbled host names go
  un-canonicalized, and split guests double-name.

So **a new diarization model MUST ship a labeling strategy fine-tuned to the way it clusters** —
overfitting to *that* diarizer, but explicit and contained, never smuggled into a shared "generic"
function. The mechanism is `providers/ml/diarization/labeling_strategy.py`:

1. `DiarizationLabelingStrategy` is the **base = Deepgram/coarse** behavior, frozen — subclasses
   override only the cluster-shape-sensitive hooks (`recorded_voices`, `host_candidate_voices`,
   `snap_extra`). Everything else (the naming invariants — N1, "a wrong label is worse than an
   unnamed voice" — the precedence self-intro > host-intro > LLM > forced, the name primitives, the
   LLM path) stays in the shared resolver and must NOT be forked.
2. Add a `<Model>LabelingStrategy` subclass tuned to your model's merge/split footprint, and wire it
   in `labeling_strategy_for(diarization_provider)`.
3. Because diarization providers are chained (DGX → local → Deepgram fallback, RFC-106), more than
   one cluster shape can occur in the **same deployment** — the base strategy must keep working while
   your new one is active.

**How to derive the strategy (do NOT guess):** run the corpus reprocess as a **single-variable
gate** — same everything, only `diarization_provider` changed (see
`config/profiles/reprocess_v22_community1.yaml` for the template; it differs from `cloud_balanced` in
*exactly* the diarization field). Diff the resulting speaker naming against the prior diarizer's,
read the `source` field in the `.speakers.diagnostics.json` of each changed voice to find the
mechanism, and design hooks against the **real** clusters — not an assumed shape. Do **not** try to
tune the diarizer's `clustering_threshold` to imitate another model's shape: the DGX server accepts
only speaker-count bounds, and coarsening trades split-fixes for merge-regressions (ADR-134 §Q2).

## Related Documentation

- [ADR-134: Provider-specific speaker labeling](../adr/ADR-134-provider-specific-speaker-labeling.md) - Why diarization providers need a labeling strategy
- [Protocol Extension Guide](./PROTOCOL_EXTENSION_GUIDE.md) - How to extend protocols
- [ML Provider Reference](./ML_PROVIDER_REFERENCE.md) - Details on local ML models
- [Development Guide](./DEVELOPMENT_GUIDE.md) - Development workflow
