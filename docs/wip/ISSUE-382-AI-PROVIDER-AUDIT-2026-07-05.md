# Issue #382 — AI-provider architecture audit

Companion audit prompted by operator direction (2026-07-05, during Phase E):
"Cool, we all want to look at AI providers." The Phase E work introduces
`HFEvidenceBackend` as a shared shape for the three local evidence-model
loaders (QA / NLI / embedding). This doc audits whether that alignment
extends into the AI-provider stack (OpenAI, Gemini, Anthropic, Mistral,
Ollama, DeepSeek, Grok, Deepgram) — and answers what alignment work, if
any, belongs in this branch vs a follow-up.

## Current AI-provider topology

**Task contract (already unified):** `summarization/base.py` defines a
`runtime_checkable` `SummarizationProvider` Protocol with three methods:

```python
def initialize(self) -> None: ...
def summarize(self, text: str, episode_title=None, episode_description=None,
              params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...
def cleanup(self) -> None: ...
```

Every provider currently in the tree implements this Protocol via
duck-typing (no inheritance):

| File | Class | Notes |
|---|---|---|
| `providers/openai/openai_provider.py` | `OpenAIProvider` | + `summarize_mega_bundled`, `summarize_extraction_bundled`, `summarize_bundled` |
| `providers/gemini/gemini_provider.py` | `GeminiProvider` | same bundled trio |
| `providers/anthropic/anthropic_provider.py` | `AnthropicProvider` | same bundled trio |
| `providers/mistral/mistral_provider.py` | `MistralProvider` | same bundled trio |
| `providers/ollama/ollama_provider.py` | `OllamaProvider` | `summarize_bundled` only |
| `providers/deepseek/deepseek_provider.py` | `DeepSeekProvider` | no bundled |
| `providers/grok/grok_provider.py` | `GrokProvider` | no bundled |
| `providers/ml/ml_provider.py` | `MLProvider` | local BART/LED/etc via `SummaryModel` |
| `providers/ml/hybrid_ml_provider.py` | `HybridMLProvider` | hybrid MAP+REDUCE with tier-selection |
| `providers/ml/summllama_provider.py` | `SummLlamaProvider` | local causal-LM path |
| `summarization/fallback.py` | `FallbackAwareSummarizationProvider` | decorator/wrapper |

**Where Phase E's `HFEvidenceBackend` sits:** ONE level below
`SummarizationProvider`. It's an implementation detail of the local ML
providers (specifically the GIL evidence stack accessed via
`gi/grounding.py`), not a task-level abstraction. No conflict with the
existing provider contract.

## Alignment findings

### ✅ Provider contract is already the alignment layer

`SummarizationProvider` IS the "just different profiles and providers"
seam. Cloud providers implement it directly; local ML providers wrap
`SummaryModel` / `HFEvidenceBackend` and implement it. Both surface the
same 3-method contract. **No work needed in #382.**

### ✅ HFEvidenceBackend does NOT need a Protocol conformance

The three evidence backends (QA / NLI / embedding) each have distinct
task signatures (`answer_top_k(question, context)`, `predict_scores(pairs)`,
`encode(texts)`). Trying to hoist those into one Protocol would either
lie about the shape or force `**kwargs: Any` at the interface. The
subclass shape works — each subclass owns its task methods.

### 🟡 Bundled-inference methods duplicated across cloud providers

`summarize_mega_bundled`, `summarize_extraction_bundled`, and
`summarize_bundled` appear (with slightly different signatures) in
OpenAI / Gemini / Anthropic / Mistral. Each has its own pricing/token
counter, its own bundled-prompt template, its own JSON-parse retry.
Estimated ~200 LOC duplication surface.

**Not folded into #382.** This is orthogonal to transformers v5 and
would balloon the PR. Fold as separate work when a cloud-provider
feature demands touching all four again.

**Tracked in [#1142](https://github.com/chipi/podcast_scraper/issues/1142)** — filed 2026-07-05 with the full measurement, API proposal, test coverage plan, and rationale for splitting out of #382.

Ticket wording (as filed):

> **feat(providers): extract shared BundledSummarizationMixin across cloud LLM providers**
>
> The four cloud providers (OpenAI, Gemini, Anthropic, Mistral) implement
> a near-identical bundled-inference trio (`summarize_mega_bundled` /
> `summarize_extraction_bundled` / `summarize_bundled`). Hoist the shared
> flow — chunking heuristics, JSON-parse retry loop, cost telemetry — into
> a `BundledSummarizationMixin`. Providers keep their SDK-specific
> `_call_llm(prompt, ...) -> str` and pricing model.

### 🟡 Diarization Protocol also exists — no work needed

`providers/ml/diarization/base.py` already defines a
`DiarizationProvider` Protocol; pyannote / deepgram / gemini
diarization providers implement it. Same "just providers" alignment,
independent stack. No #382 impact.

### ✅ `OllamaProvider` dual-use is fine

`OllamaProvider` is used both as a top-level `SummarizationProvider`
(cloud-thin route) and wrapped inside `hybrid_ml_provider.OllamaReduceBackend`
(hybrid tier-2 reduce). Two adapters over the same client — this is the
right pattern (each adapter conforms to a different interface). No work
needed.

### ✅ FallbackAwareSummarizationProvider is a good decorator model

`summarization/fallback.py:FallbackAwareSummarizationProvider` wraps any
`SummarizationProvider` and adds fallback-chain semantics. This is the
"decorator pattern in providers" the operator direction pointed at.
Already in place, well-scoped. Reference pattern for future cross-cutting
provider concerns.

## Testing strategy alignment

Phase E introduced `test_hf_evidence_backend.py` with:
- Unit tests of shared machinery (device resolution, cache, threading).
- No mocking of `transformers.*` internals.
- Mock at the abstraction boundary via a `DummyBackend` subclass.

This matches the cloud-provider test pattern:
- `test_openai_provider.py` mocks `openai.OpenAI` client (own seam) —
  not `httpx`, not `openai._internal.transport`.
- `test_anthropic_provider.py` mocks `anthropic.Anthropic` client.
- `test_gemini_provider.py` mocks `google.genai.Client`.

**Testing-strategy rule established:** mock at the top of the vendor
SDK's public surface (`OpenAI()`, `Anthropic()`, `SentenceTransformer()`,
`AutoModelForQuestionAnswering.from_pretrained`) or at our own facade —
never at library-internal names. The dead `@patch("transformers.pipeline",
create=True)` decorators removed in Phase E were exactly this
anti-pattern: `create=True` patches a name that doesn't exist under
`_LazyModule`, so it intercepts nothing.

Formal write-up lands in `docs/guides/testing-strategy-ml.md` during
Phase 8 docs sweep.

## Conclusion

The Phase E abstraction slots cleanly under the existing
`SummarizationProvider` contract. No AI-provider changes required in
this branch. One follow-up opportunity documented
(bundled-inference dedup) — separate PR after #382.

---

*Author: Claude Code for operator, 2026-07-05.*
*Prompted by mid-Phase-E operator note: "we all want to look at AI providers".*
