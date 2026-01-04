# PRD-009: Anthropic Provider Integration

- **Status**: Draft
- **Related RFCs**: RFC-032
- **Related PRDs**: PRD-006 (OpenAI Provider Integration)

## Summary

Add Anthropic Claude API as an optional provider for speaker detection and summarization capabilities, enabling users to choose between local on-device processing, OpenAI API, and Anthropic Claude API. This builds on the existing modularization architecture (RFC-021) and provider patterns (PRD-006) to provide seamless provider switching without changing end-user experience or workflow behavior.

## Background & Context

Currently, the podcast scraper supports two provider types for AI/ML capabilities:

- **Local ML Providers**: spaCy NER (speaker detection), Whisper (transcription), Hugging Face transformers (summarization)
- **OpenAI Providers**: OpenAI GPT API (speaker detection, summarization), OpenAI Whisper API (transcription)

Users have requested Anthropic Claude as an alternative to OpenAI for several reasons:

- **Model Diversity**: Different models may produce better results for specific use cases
- **Vendor Independence**: Avoid lock-in to a single cloud provider
- **Cost Optimization**: Anthropic pricing may be more favorable for some workloads
- **Quality Preferences**: Some users prefer Claude's response style and accuracy

This PRD addresses the need to add Anthropic as a provider option while:

1. Maintaining backward compatibility with existing providers
2. Following the established provider architecture patterns
3. Handling capability gaps gracefully (Anthropic doesn't have audio transcription)

## Goals

- Add Anthropic Claude API as provider option for speaker detection and summarization
- Maintain 100% backward compatibility with existing providers (local, OpenAI)
- Follow identical architectural patterns as OpenAI provider (PRD-006, RFC-013)
- Provide secure API key management via environment variables and `.env` files
- Enable per-capability provider selection (can mix local, OpenAI, and Anthropic providers)
- Handle provider capability gaps gracefully with clear error messages
- Support dev/test vs production model selection patterns
- Create Anthropic-specific prompt templates (prompts are provider-specific)

## Anthropic Model Selection and Cost Analysis

### Configuration Fields

Add to `config.py` when implementing Anthropic providers:

```python

# Anthropic Model Selection

anthropic_speaker_model: str = Field(
    default="claude-3-5-haiku-latest",
    description="Anthropic model for speaker detection (dev/test: haiku, prod: sonnet)"
)

anthropic_summary_model: str = Field(
    default="claude-3-5-haiku-latest",
    description="Anthropic model for summarization (dev/test: haiku, prod: sonnet)"
)

# Anthropic API Configuration

anthropic_api_key: Optional[str] = Field(
    default=None,
    description="Anthropic API key (prefer ANTHROPIC_API_KEY env var or .env file)"
)

anthropic_api_base: Optional[str] = Field(
    default=None,
    description="Custom Anthropic API base URL (for E2E testing with mock servers)"
)

anthropic_temperature: float = Field(
    default=0.3,
    ge=0.0,
    le=1.0,
    description="Temperature for Anthropic generation (0.0-1.0, lower = more deterministic)"
)

anthropic_max_tokens: Optional[int] = Field(
    default=None,
    description="Max tokens for Anthropic generation (None = model default)"
)
```yaml

## Model Options and Pricing

| Model | Input Cost | Output Cost | Context Window | Best For |
| ----- | ---------- | ----------- | -------------- | -------- |
| **claude-3-5-sonnet-latest** | $3.00 / 1M tokens | $15.00 / 1M tokens | 200k tokens | **Production** (best quality) |
| **claude-3-5-haiku-latest** | $0.25 / 1M tokens | $1.25 / 1M tokens | 200k tokens | **Dev/Test** (fast, cheap) |
| **claude-3-opus-latest** | $15.00 / 1M tokens | $75.00 / 1M tokens | 200k tokens | Maximum quality (expensive) |

**Note:** Prices subject to change. Check [Anthropic Pricing](https://www.anthropic.com/pricing) for current rates.

### Dev/Test vs Production Model Selection

| Environment | Speaker Model | Summary Model | Rationale |
| ----------- | ------------- | ------------- | --------- |
| **Dev/Test** | `claude-3-5-haiku-latest` | `claude-3-5-haiku-latest` | Fast iteration, low cost |
| **Production** | `claude-3-5-sonnet-latest` | `claude-3-5-sonnet-latest` | Best quality/cost balance |

### Cost Comparison: Anthropic vs OpenAI (Per 100 Episodes)

| Component | OpenAI (gpt-4o-mini) | Anthropic (claude-3-5-haiku) | Anthropic (claude-3-5-sonnet) |
| --------- | -------------------- | ---------------------------- | ----------------------------- |
| **Speaker Detection** | $0.14 | $0.10 | $0.40 |
| **Summarization** | $0.41 | $0.30 | $1.20 |
| **Total API Costs** | **$0.55** | **$0.40** | **$1.60** |

**Note:** Anthropic Haiku is the most cost-effective for dev/test; Sonnet provides best quality for production.

## Non-Goals

- Transcription support (Anthropic doesn't have audio transcription API)
- Changing default behavior (local providers remain default)
- Modifying existing OpenAI provider implementations
- Adding new features beyond provider selection
- API key management UI or interactive configuration
- Provider fallback chains (use provider A, fallback to provider B)

## Personas

- **Quality Seeker Quinn**: Wants to compare Claude vs GPT for summarization quality
- **Cost-Conscious Carol**: Uses Haiku for development, Sonnet for production
- **Vendor-Diverse Victor**: Wants to avoid single-vendor lock-in
- **Developer Devin**: Needs to test with Anthropic API during development
- **Privacy-First Pat**: Continues using local providers exclusively (no changes to their workflow)

## User Stories

- *As Quality Seeker Quinn, I can configure Anthropic Claude for summarization to compare results with OpenAI.*
- *As Cost-Conscious Carol, I can use Haiku for development and switch to Sonnet for production.*
- *As Vendor-Diverse Victor, I can mix providers (local transcription, Anthropic summarization, OpenAI speaker detection).*
- *As Developer Devin, I can set my Anthropic API key in environment variables and test API integration.*
- *As any operator, I get a clear error message if I try to use Anthropic for transcription (unsupported).*
- *As any operator, I can see which provider was used for each capability in logs and metadata.*

## Functional Requirements

### FR1: Provider Selection

- **FR1.1**: Add `"anthropic"` as valid value for `speaker_detector_provider` config field
- **FR1.2**: Add `"anthropic"` as valid value for `summary_provider` config field
- **FR1.3**: Attempting to set `transcription_provider: anthropic` results in clear error message
- **FR1.4**: Provider selection is independent per capability (can mix local, OpenAI, Anthropic)
- **FR1.5**: Default values maintain current behavior (local providers)
- **FR1.6**: Invalid provider values result in clear error messages with valid options listed

### FR2: Provider Capability Gap Handling

- **FR2.1**: Clear error message when user selects unsupported capability: "Anthropic provider does not support transcription. Use 'whisper' (local) or 'openai' instead."
- **FR2.2**: Provider capability matrix documented in configuration reference
- **FR2.3**: Validation occurs at configuration load time (fail fast)
- **FR2.4**: Future-proof design allowing capabilities to be added to providers

### FR3: API Key Management

- **FR3.1**: Support `ANTHROPIC_API_KEY` environment variable for API authentication
- **FR3.2**: Support `.env` file via `python-dotenv` for convenient per-environment configuration
- **FR3.3**: API key is never stored in source code, config files, or committed files
- **FR3.4**: `.env` file automatically loaded when `config.py` module is imported
- **FR3.5**: `examples/.env.example` template file updated with Anthropic placeholder
- **FR3.6**: Missing API key when Anthropic provider is selected results in clear error message
- **FR3.7**: API key validation occurs at provider initialization (fail fast)
- **FR3.8**: Environment variable priority: config file > system env > `.env` file

### FR4: Speaker Detection with Anthropic

- **FR4.1**: Anthropic provider uses Claude models for entity extraction
- **FR4.2**: Maintains same interface as other providers (`detect_hosts`, `detect_speakers`, `analyze_patterns`)
- **FR4.3**: Returns results in same format as other providers (no workflow changes)
- **FR4.4**: Handles API rate limits gracefully (retry with backoff)
- **FR4.5**: Logs provider type used in detection logs
- **FR4.6**: Uses Anthropic-specific prompt templates (provider-specific prompts)

### FR5: Summarization with Anthropic

- **FR5.1**: Anthropic provider uses Claude models for summarization
- **FR5.2**: Maintains same interface as other providers (`initialize`, `summarize`, `cleanup`)
- **FR5.3**: Leverages large context window (200k tokens) - can process full transcripts without chunking
- **FR5.4**: Returns results in same format as other providers
- **FR5.5**: Uses Anthropic-specific prompt templates (provider-specific prompts)
- **FR5.6**: Logs provider type used in summarization logs

### FR6: Prompt Management

- **FR6.1**: Create Anthropic-specific prompt templates in `prompts/anthropic/` folder
- **FR6.2**: Prompts follow same structure as OpenAI prompts but optimized for Claude
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
- **FR8.5**: Unsupported capability errors are clear and suggest alternatives

## Technical Requirements

### TR1: Architecture

- **TR1.1**: Follow identical patterns as OpenAI provider (RFC-013)
- **TR1.2**: Create `podcast_scraper/anthropic/` package for shared utilities
- **TR1.3**: Create `speaker_detectors/anthropic_detector.py` implementing `SpeakerDetector` protocol
- **TR1.4**: Create `summarization/anthropic_provider.py` implementing `SummarizationProvider` protocol
- **TR1.5**: Update factories to include Anthropic provider option
- **TR1.6**: No changes to workflow.py logic (uses factory pattern)

### TR2: Dependencies

- **TR2.1**: Add `anthropic` Python package as optional dependency
- **TR2.2**: Anthropic dependency only required when Anthropic provider is used
- **TR2.3**: Lazy import Anthropic client (only when provider is selected)
- **TR2.4**: Version pinning for Anthropic package

### TR3: Configuration

- **TR3.1**: Add Anthropic provider type to config Literal types
- **TR3.2**: Add Anthropic-specific config fields (api_key, model, temperature, etc.)
- **TR3.3**: Config validation ensures provider + API key consistency
- **TR3.4**: Backward compatible defaults (local providers)

### TR4: Testing

- **TR4.1**: Unit tests for Anthropic providers (with mocked API)
- **TR4.2**: Integration tests with E2E server mock endpoints
- **TR4.3**: E2E tests for complete workflow with Anthropic providers
- **TR4.4**: Tests verify same interface as local/OpenAI providers
- **TR4.5**: Tests verify error handling and rate limiting
- **TR4.6**: Backward compatibility tests (default behavior unchanged)

### TR5: E2E Server Extensions

- **TR5.1**: Add Anthropic mock endpoints to E2E HTTP server
- **TR5.2**: Mock `/v1/messages` endpoint for chat completions
- **TR5.3**: Support speaker detection and summarization request patterns
- **TR5.4**: Add `anthropic_api_base()` helper to `E2EServerURLs` class

## Success Criteria

- ✅ Users can select Anthropic provider for speaker detection and summarization
- ✅ Clear error when attempting to use Anthropic for transcription
- ✅ Default behavior (local providers) remains unchanged
- ✅ API keys are managed securely via environment variables and `.env` files
- ✅ Anthropic providers implement same interfaces as local/OpenAI providers
- ✅ No changes required to workflow.py or end-user code
- ✅ Error handling is clear and actionable
- ✅ Documentation explains provider selection, capabilities, and API key setup
- ✅ E2E tests pass with Anthropic mock endpoints

## Out of Scope

- Transcription support (Anthropic has no audio API)
- Other cloud providers (AWS, Google, etc.) - future PRDs
- API key management UI or interactive setup
- Cost tracking or usage monitoring
- Provider fallback chains (use provider A, fallback to provider B)
- Real-time provider switching during execution

## Dependencies

- **Prerequisite**: Modularization refactoring (RFC-021) ✅ Completed
- **Prerequisite**: OpenAI provider implementation (PRD-006, RFC-013) ✅ Completed
- **External**: Anthropic API access and API key
- **Internal**: Provider abstraction interfaces (from refactoring)

## Risks & Mitigations

- **Risk**: API costs can be high for large batches
  - **Mitigation**: Document costs, use Haiku for dev/test, Sonnet for production
- **Risk**: API rate limits may slow processing
  - **Mitigation**: Implement retry logic with backoff, respect rate limits, document limits
- **Risk**: API availability issues
  - **Mitigation**: Clear error messages, retry logic, fallback documentation
- **Risk**: Breaking changes if Anthropic API changes
  - **Mitigation**: Version pinning, comprehensive tests, monitor Anthropic changelog

## Provider Capability Matrix

| Capability | Local | OpenAI | Anthropic |
| ---------- | ----- | ------ | --------- |
| **Transcription** | ✅ Whisper | ✅ Whisper API | ❌ Not supported |
| **Speaker Detection** | ✅ spaCy NER | ✅ GPT API | ✅ Claude API |
| **Summarization** | ✅ Transformers | ✅ GPT API | ✅ Claude API |

## Future Considerations

- Support for other providers (AWS Bedrock, Google Gemini, local LLMs)
- Provider fallback chains (try Anthropic, fallback to OpenAI, fallback to local)
- Cost tracking and usage monitoring
- Provider performance comparison tools
- Hybrid processing (use API for some episodes, local for others)
- Adding transcription when/if Anthropic releases audio API
