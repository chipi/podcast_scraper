# PRD-010: Mistral Provider Integration

- **Status**: Draft (Revised)
- **Revision**: 2
- **Date**: 2026-02-04
- **Related RFCs**: RFC-033 (Revised)
- **Related PRDs**: PRD-006 (OpenAI), PRD-009 (Anthropic)

## Summary

Add Mistral AI as an optional provider for transcription, speaker detection, and summarization capabilities. Mistral is unique among cloud providers in offering a complete alternative to OpenAI, supporting all three capabilities through their chat models and Voxtral audio models. Like OpenAI, Mistral uses a **unified provider pattern** where a single `MistralProvider` class implements all three capabilities. This builds on the existing modularization architecture (RFC-021) and provider patterns (PRD-006, PRD-009) to provide seamless provider switching.

## Background & Context

Currently, the podcast scraper supports the following providers:

- **Local ML Providers**: spaCy NER (speaker detection), Whisper (transcription), Hugging Face transformers (summarization)
- **OpenAI Providers**: OpenAI GPT API (speaker detection, summarization), OpenAI Whisper API (transcription)
- **Anthropic Providers**: Claude API (speaker detection, summarization) - no transcription

Users have requested Mistral as an alternative for several reasons:

- **Full OpenAI Alternative**: Unlike Anthropic, Mistral supports ALL capabilities including transcription
- **European Provider**: Mistral is based in France, which may be preferred for data residency
- **Competitive Pricing**: Mistral offers competitive pricing, especially for smaller models
- **Large Context Window**: Mistral Large 3 offers 256k token context window
- **Model Diversity**: Different models may produce better results for specific use cases

This PRD addresses adding Mistral as a complete provider option covering all three capabilities.

## Goals

- Add Mistral AI as provider option for transcription, speaker detection, and summarization
- Maintain 100% backward compatibility with existing providers (local, OpenAI, Anthropic)
- Follow **unified provider pattern** (like OpenAI) - single class implementing all three protocols
- Provide secure API key management via environment variables and `.env` files
- Support both Config-based and experiment-based factory modes from the start
- Enable per-capability provider selection (can mix local, OpenAI, Anthropic, and Mistral)
- Use environment-based model defaults (test vs production)
- Create Mistral-specific prompt templates (prompts are provider-specific)
- Leverage Voxtral models for audio transcription

## Mistral Model Selection and Cost Analysis

### Configuration Fields

Add to `config.py` when implementing Mistral providers (following OpenAI pattern):

```python
# Mistral API Configuration
mistral_api_key: Optional[str] = Field(
    default=None,
    alias="mistral_api_key",
    description="Mistral API key (prefer MISTRAL_API_KEY env var or .env file)"
)

mistral_api_base: Optional[str] = Field(
    default=None,
    alias="mistral_api_base",
    description="Mistral API base URL (for E2E testing with mock servers)"
)

# Mistral Model Selection (environment-based defaults, like OpenAI)
mistral_transcription_model: str = Field(
    default_factory=_get_default_mistral_transcription_model,
    alias="mistral_transcription_model",
    description="Mistral Voxtral model for transcription (default: environment-based)"
)

mistral_speaker_model: str = Field(
    default_factory=_get_default_mistral_speaker_model,
    alias="mistral_speaker_model",
    description="Mistral model for speaker detection (default: environment-based)"
)

mistral_summary_model: str = Field(
    default_factory=_get_default_mistral_summary_model,
    alias="mistral_summary_model",
    description="Mistral model for summarization (default: environment-based)"
)

# Shared settings (like OpenAI)
mistral_temperature: float = Field(
    default=0.3,
    alias="mistral_temperature",
    description="Temperature for Mistral generation (0.0-1.0, lower = more deterministic)"
)

mistral_max_tokens: Optional[int] = Field(
    default=None,
    alias="mistral_max_tokens",
    description="Max tokens for Mistral generation (None = model default)"
)

# Mistral Prompt Configuration (following OpenAI pattern)
mistral_speaker_system_prompt: Optional[str] = Field(
    default=None,
    alias="mistral_speaker_system_prompt",
    description="Mistral system prompt for speaker detection (default: mistral/ner/system_ner_v1)"
)

mistral_speaker_user_prompt: str = Field(
    default="mistral/ner/guest_host_v1",
    alias="mistral_speaker_user_prompt",
    description="Mistral user prompt for speaker detection"
)

mistral_summary_system_prompt: Optional[str] = Field(
    default=None,
    alias="mistral_summary_system_prompt",
    description="Mistral system prompt for summarization (default: mistral/summarization/system_v1)"
)

mistral_summary_user_prompt: str = Field(
    default="mistral/summarization/long_v1",
    alias="mistral_summary_user_prompt",
    description="Mistral user prompt for summarization"
)
```

**Environment-based defaults:**
- **Test environment**:
  - Transcription: `voxtral-mini-latest` (only option)
  - Speaker/Summary: `mistral-small-latest` (cheapest text)
- **Production environment**:
  - Transcription: `voxtral-mini-latest` (only option)
  - Speaker/Summary: `mistral-large-latest` (best quality, 256k context)

## Mistral Model Pricing

> **⚠️ Pricing Information:** The pricing table below is provided for reference only and may be outdated. **Always check [Mistral Pricing](https://mistral.ai/pricing/) for current rates** before making cost decisions.

### Model Pricing (per million tokens)

| Model | Input | Output | Context | Notes |
| ------- | ------- | -------- | --------- | ------- |
| **Mistral Large 3** | $2.00 | $6.00 | 256k | **Production** - best quality |
| **Mistral Medium 3** | $0.40 | $2.00 | 256k | Balanced quality/cost |
| **Mistral Small 3.1** | $0.10 | $0.30 | 128k | **Dev/Test** - cheapest text |
| **Mistral NeMo** | $0.15 | $0.15 | 128k | Budget option |
| **Codestral 2501** | $0.20 | $0.60 | 32k | Code generation |
| **Pixtral Large** | $0.15 | $0.15 | 128k | Vision/multimodal |
| **Voxtral Mini** | ~$0.01/min | - | - | Audio transcription |

### Mistral vs OpenAI vs Anthropic Comparison

| Model Tier | Mistral | OpenAI | Anthropic |
| ------------ | --------- | -------- | ----------- |
| **Premium** | Large 3 ($2/$6) | GPT-5 ($1.25/$10) | Claude Opus ($15/$75) |
| **Standard** | Medium 3 ($0.40/$2) | GPT-5 Mini ($0.25/$2) | Claude Sonnet ($3/$15) |
| **Budget** | Small 3.1 ($0.10/$0.30) | GPT-5 Nano ($0.05/$0.40) | Claude Haiku ($0.25/$1.25) |

**Key Insight:** Mistral Small 3.1 is the cheapest text model across all providers ($0.10/$0.30), making it ideal for development and testing.

### Recommended Model Defaults

| Purpose | Test Default | Production Default | Rationale |
| --------- | ------------- | ------------------- | ----------- |
| Speaker Detection | `mistral-small-latest` | `mistral-large-latest` | Names extraction needs quality for prod |
| Summarization | `mistral-small-latest` | `mistral-large-latest` | Large context (256k) for full transcripts |
| Transcription | `voxtral-mini-latest` | `voxtral-mini-latest` | Only Voxtral option |

### Cost Estimate Per Episode

Assuming ~10,000 words per episode transcript (~13,000 tokens):

| Task | Test (Small) | Production (Large) |
| ------ | -------------- | ------------------- |
| Speaker Detection | ~$0.002 | ~$0.04 |
| Summarization | ~$0.003 | ~$0.05 |
| Transcription (60 min) | ~$0.60 | ~$0.60 |
| **Total per episode** | ~$0.61 | ~$0.69 |

### Cost Comparison: All Providers (Per 100 Episodes)

| Component | Local | OpenAI (GPT-5) | Anthropic (Sonnet) | Mistral (Large) |
| ----------- | ------- | ---------------- | ------------------- | ----------------- |
| **Transcription** | Free | $36.00 | ❌ N/A | ~$60.00 |
| **Speaker Detection** | Free | $1.60 | $4.00 | $4.00 |
| **Summarization** | Free | $4.00 | $20.00 | $5.00 |
| **Total** | $0 | **$41.60** | **$24.00** (no transcription) | **$69.00** |

### Hybrid Strategy Recommendations

**Cost-Optimized (Local Whisper + Mistral Small):**

```yaml
transcription_provider: whisper       # Free (local)
speaker_detector_provider: mistral
summary_provider: mistral
mistral_speaker_model: mistral-small-latest
mistral_summary_model: mistral-small-latest
```bash

Cost: ~$0.005/episode (cheapest cloud text processing)

**Quality-Optimized (Mistral Full Stack):**

```yaml
transcription_provider: mistral
speaker_detector_provider: mistral
summary_provider: mistral
mistral_speaker_model: mistral-large-latest
mistral_summary_model: mistral-large-latest
```bash

Cost: ~$0.69/episode (complete EU-based solution)

**European Data Residency (All Mistral):**

For organizations requiring EU data residency, Mistral is the only cloud provider option that keeps all data within the EU.

### Mistral Advantages

1. **Cheapest text processing** - Small 3.1 at $0.10/$0.30 per million tokens
2. **Full capability coverage** - Unlike Anthropic, supports transcription
3. **EU data residency** - Based in France, data stays in EU
4. **Largest context window** - 256k tokens (Large 3)
5. **Complete OpenAI alternative** - All three capabilities covered

## Non-Goals

- Changing default behavior (local providers remain default)
- Modifying existing OpenAI or Anthropic provider implementations
- Adding new features beyond provider selection
- API key management UI or interactive configuration
- Provider fallback chains (use provider A, fallback to provider B)
- Vision/image capabilities (future consideration)

## Personas

- **Quality Seeker Quinn**: Wants to compare Mistral vs GPT vs Claude for quality
- **Cost-Conscious Carol**: Uses Mistral Small for development (cheapest text option)
- **European Enterprise Eric**: Prefers EU-based provider for data residency
- **Full-Stack Fiona**: Wants one provider for all capabilities (transcription + text)
- **Developer Devin**: Needs to test with Mistral API during development
- **Privacy-First Pat**: Continues using local providers exclusively

## User Stories

- *As Quality Seeker Quinn, I can configure Mistral for all capabilities to compare with OpenAI/Anthropic.*
- *As Cost-Conscious Carol, I can use Mistral Small for the cheapest cloud processing.*
- *As European Enterprise Eric, I can use Mistral knowing data stays with EU provider.*
- *As Full-Stack Fiona, I can use Mistral for transcription AND summarization (unlike Anthropic).*
- *As Developer Devin, I can set my Mistral API key in environment variables and test.*
- *As any operator, I can see which provider was used for each capability in logs.*

## Functional Requirements

### FR1: Provider Selection

- **FR1.1**: Add `"mistral"` as valid value for `transcription_provider` config field
- **FR1.2**: Add `"mistral"` as valid value for `speaker_detector_provider` config field
- **FR1.3**: Add `"mistral"` as valid value for `summary_provider` config field
- **FR1.4**: Provider selection is independent per capability (can mix all providers)
- **FR1.5**: Default values maintain current behavior (local providers)
- **FR1.6**: Invalid provider values result in clear error messages with valid options listed
- **FR1.7**: Support both Config-based and experiment-based factory modes from the start

### FR2: API Key Management

- **FR2.1**: Support `MISTRAL_API_KEY` environment variable for API authentication (like `OPENAI_API_KEY`)
- **FR2.2**: Support `.env` file via `python-dotenv` for convenient configuration
- **FR2.3**: API key is never stored in source code, config files, or committed files
- **FR2.4**: `.env` file automatically loaded when `config.py` module is imported
- **FR2.5**: `examples/.env.example` template file updated with Mistral placeholder
- **FR2.6**: Missing API key when Mistral provider is selected results in clear error
- **FR2.7**: API key validation occurs at provider initialization (fail fast)
- **FR2.8**: Support `MISTRAL_API_BASE` environment variable for E2E testing (like `OPENAI_API_BASE`)

### FR3: Transcription with Mistral

- **FR3.1**: Mistral provider uses Voxtral models for audio transcription
- **FR3.2**: Maintains same interface as other providers (`initialize`, `transcribe`, `transcribe_with_segments`, `cleanup`)
- **FR3.3**: Supports timestamp granularity from Voxtral API
- **FR3.4**: Returns results in same format as other providers (no workflow changes)
- **FR3.5**: Handles API rate limits gracefully (retry with backoff)
- **FR3.6**: Logs provider type used in transcription logs
- **FR3.7**: Audio file handling compatible with Voxtral API requirements

### FR4: Speaker Detection with Mistral

- **FR4.1**: Mistral provider uses chat models for entity extraction
- **FR4.2**: Maintains same interface as other providers (`detect_hosts`, `detect_speakers`, `analyze_patterns`)
- **FR4.3**: Returns results in same format as other providers
- **FR4.4**: Handles API rate limits gracefully (retry with backoff)
- **FR4.5**: Logs provider type used in detection logs
- **FR4.6**: Uses Mistral-specific prompt templates

### FR5: Summarization with Mistral

- **FR5.1**: Mistral provider uses chat models for summarization
- **FR5.2**: Maintains same interface as other providers (`initialize`, `summarize`, `cleanup`)
- **FR5.3**: Leverages large context window (256k tokens) - can process full transcripts
- **FR5.4**: Returns results in same format as other providers
- **FR5.5**: Uses Mistral-specific prompt templates
- **FR5.6**: Logs provider type used in summarization logs

### FR6: Prompt Management

- **FR6.1**: Create Mistral-specific prompt templates in `prompts/mistral/` folder
- **FR6.2**: Prompts follow same structure as OpenAI/Anthropic but optimized for Mistral
- **FR6.3**: Support for prompt versioning (v1, v2, etc.)
- **FR6.4**: Configurable prompt selection via config fields

### FR7: Logging and Observability

- **FR7.1**: Log which provider is used for each capability at initialization
- **FR7.2**: Log provider type in episode processing logs
- **FR7.3**: Include provider information in metadata documents
- **FR7.4**: Log API usage (calls, errors, rate limits) for debugging
- **FR7.5**: No sensitive information (API keys) in logs

### FR8: Error Handling

- **FR8.1**: API errors result in clear error messages with provider context
- **FR8.2**: Rate limit errors include retry information
- **FR8.3**: Network errors are handled gracefully with retries
- **FR8.4**: Invalid API key errors are clear and actionable
- **FR8.5**: Audio format errors provide guidance on supported formats

## Technical Requirements

### TR1: Architecture

- **TR1.1**: Follow **unified provider pattern** (like OpenAI) - single class implementing all three protocols
- **TR1.2**: Create `providers/mistral/mistral_provider.py` with unified `MistralProvider` class
- **TR1.3**: `MistralProvider` implements `TranscriptionProvider`, `SpeakerDetector`, and `SummarizationProvider` protocols
- **TR1.4**: Update all factories to include Mistral option with support for both Config-based and experiment-based modes
- **TR1.5**: Create `prompts/mistral/` directory with provider-specific prompt templates
- **TR1.6**: Follow OpenAI provider architecture exactly for consistency

### TR2: Dependencies

- **TR2.1**: Add `mistralai` Python package as optional dependency
- **TR2.2**: Mistral dependency only required when Mistral provider is used
- **TR2.3**: Lazy import Mistral client (only when provider is selected)
- **TR2.4**: Version pinning for Mistral package

### TR3: Configuration

- **TR3.1**: Add Mistral provider type to all config Literal types
- **TR3.2**: Add Mistral-specific config fields (api_key, models, temperature, etc.)
- **TR3.3**: Config validation ensures provider + API key consistency
- **TR3.4**: Backward compatible defaults (local providers)

### TR4: Testing

- **TR4.1**: Unit tests for all Mistral providers (with mocked API)
- **TR4.2**: Integration tests with E2E server mock endpoints
- **TR4.3**: E2E tests for complete workflow with Mistral providers
- **TR4.4**: Tests verify same interface as other providers
- **TR4.5**: Tests verify error handling and rate limiting
- **TR4.6**: Backward compatibility tests (default behavior unchanged)
- **TR4.7**: Audio transcription tests with Voxtral mock endpoint

### TR5: E2E Server Extensions

- **TR5.1**: Add Mistral mock endpoints to E2E HTTP server
- **TR5.2**: Mock `/v1/chat/completions` endpoint for chat (speaker detection, summarization)
- **TR5.3**: Mock `/v1/audio/transcriptions` endpoint for Voxtral transcription
- **TR5.4**: Add `mistral_api_base()` helper to `E2EServerURLs` class

## Success Criteria

- ✅ Users can select Mistral provider for transcription, speaker detection, and summarization via unified provider
- ✅ Mistral is a complete OpenAI alternative (all three capabilities)
- ✅ Default behavior (local providers) remains unchanged
- ✅ API keys are managed securely via `MISTRAL_API_KEY` environment variable
- ✅ Environment-based model defaults (test vs production)
- ✅ Both Config-based and experiment-based factory modes supported
- ✅ Mistral providers implement same interfaces as other providers
- ✅ No changes required to workflow.py or end-user code
- ✅ Error handling is clear and actionable
- ✅ E2E tests pass with Mistral mock endpoints
- ✅ Follows OpenAI provider pattern exactly for consistency

## Out of Scope

- Vision/image capabilities (Mistral supports this, future PRD)
- Agent framework integration (Mistral offers this, future consideration)
- Other cloud providers (AWS, Google, etc.) - future PRDs
- API key management UI or interactive setup
- Cost tracking or usage monitoring
- Provider fallback chains

## Dependencies

- **Prerequisite**: Modularization refactoring (RFC-021) ✅ Completed
- **Prerequisite**: OpenAI provider implementation (PRD-006, RFC-013) ✅ Completed
- **Prerequisite**: Anthropic provider implementation (PRD-009, RFC-032) ✅ Completed
- **External**: Mistral API access and API key
- **Internal**: Provider abstraction interfaces (from refactoring)

## Risks & Mitigations

- **Risk**: API costs can accumulate for large batches
  - **Mitigation**: Document costs, use small model for dev/test, large for production
- **Risk**: API rate limits may slow processing
  - **Mitigation**: Implement retry logic with backoff, document limits
- **Risk**: API availability issues
  - **Mitigation**: Clear error messages, retry logic, fallback documentation
- **Risk**: Voxtral audio API differences from OpenAI Whisper
  - **Mitigation**: Abstract differences in provider implementation, consistent interface
- **Risk**: Breaking changes if Mistral API changes
  - **Mitigation**: Version pinning, comprehensive tests, monitor changelog

## Provider Capability Matrix (Updated)

| Capability | Local | OpenAI | Anthropic | Mistral |
| ---------- | ----- | ------ | --------- | ------- |
| **Transcription** | ✅ Whisper | ✅ Whisper API | ❌ Not supported | ✅ Voxtral |
| **Speaker Detection** | ✅ spaCy NER | ✅ GPT API | ✅ Claude API | ✅ Mistral API |
| **Summarization** | ✅ Transformers | ✅ GPT API | ✅ Claude API | ✅ Mistral API |

## Future Considerations

- Vision/image capabilities for show notes with images
- Agent framework for automated podcast analysis workflows
- Support for other providers (AWS Bedrock, Google Gemini, local LLMs)
- Provider fallback chains
- Cost tracking and usage monitoring
- Provider performance comparison tools
