# PRD-011: DeepSeek Provider Integration

- **Status**: Draft (Revised)
- **Revision**: 2
- **Date**: 2026-02-04
- **Related RFCs**: RFC-034 (Revised)
- **Related PRDs**: PRD-006 (OpenAI), PRD-009 (Anthropic), PRD-010 (Mistral)

## Summary

Add DeepSeek AI as an optional provider for speaker detection and summarization capabilities. DeepSeek offers extremely competitive pricing (up to 95% cheaper than OpenAI) and strong reasoning capabilities via DeepSeek-R1. Like OpenAI, DeepSeek uses a **unified provider pattern** where a single `DeepSeekProvider` class implements both capabilities. Like Anthropic, DeepSeek does NOT support audio transcription via its public API. This builds on the existing modularization architecture (RFC-021) and provider patterns to provide seamless provider switching.

## Background & Context

Currently, the podcast scraper supports the following providers:

- **Local ML Providers**: spaCy NER (speaker detection), Whisper (transcription), Hugging Face transformers (summarization)
- **OpenAI Providers**: GPT API (speaker detection, summarization), Whisper API (transcription)
- **Anthropic Providers**: Claude API (speaker detection, summarization) - no transcription
- **Mistral Providers**: Mistral chat API (speaker detection, summarization), Voxtral (transcription)

Users have requested DeepSeek as an alternative for several reasons:

- **Extremely Low Cost**: DeepSeek is 90-95% cheaper than OpenAI
- **Strong Reasoning**: DeepSeek-R1 rivals OpenAI o1 in reasoning benchmarks
- **OpenAI-Compatible API**: Uses same API format, easy integration
- **Open Weights**: Models available for self-hosting if needed
- **Chinese Market Access**: Strong performance in multilingual tasks

This PRD addresses adding DeepSeek as a cost-effective provider option.

## Goals

- Add DeepSeek AI as provider option for speaker detection and summarization
- Maintain 100% backward compatibility with existing providers
- Follow **unified provider pattern** (like OpenAI) - single class implementing both protocols
- Provide secure API key management via environment variables and `.env` files
- Support both Config-based and experiment-based factory modes from the start
- Enable per-capability provider selection (can mix all providers)
- Handle capability gaps gracefully (transcription not supported)
- Leverage OpenAI-compatible API for simplified integration (no new SDK dependency)
- Use environment-based model defaults (test vs production)
- Create DeepSeek-specific prompt templates

## DeepSeek Model Selection and Cost Analysis

### Configuration Fields

Add to `config.py` when implementing DeepSeek providers (following OpenAI pattern):

```python
# DeepSeek API Configuration
deepseek_api_key: Optional[str] = Field(
    default=None,
    alias="deepseek_api_key",
    description="DeepSeek API key (prefer DEEPSEEK_API_KEY env var or .env file)"
)

deepseek_api_base: Optional[str] = Field(
    default=None,
    alias="deepseek_api_base",
    description="DeepSeek API base URL (default: https://api.deepseek.com, for E2E testing)"
)

# DeepSeek Model Selection (environment-based defaults, like OpenAI)
deepseek_speaker_model: str = Field(
    default_factory=_get_default_deepseek_speaker_model,
    alias="deepseek_speaker_model",
    description="DeepSeek model for speaker detection (default: environment-based)"
)

deepseek_summary_model: str = Field(
    default_factory=_get_default_deepseek_summary_model,
    alias="deepseek_summary_model",
    description="DeepSeek model for summarization (default: environment-based)"
)

# Shared settings (like OpenAI)
deepseek_temperature: float = Field(
    default=0.3,
    alias="deepseek_temperature",
    description="Temperature for DeepSeek generation (0.0-2.0, lower = more deterministic)"
)

deepseek_max_tokens: Optional[int] = Field(
    default=None,
    alias="deepseek_max_tokens",
    description="Max tokens for DeepSeek generation (None = model default)"
)

# DeepSeek Prompt Configuration (following OpenAI pattern)
deepseek_speaker_system_prompt: Optional[str] = Field(
    default=None,
    alias="deepseek_speaker_system_prompt",
    description="DeepSeek system prompt for speaker detection (default: deepseek/ner/system_ner_v1)"
)

deepseek_speaker_user_prompt: str = Field(
    default="deepseek/ner/guest_host_v1",
    alias="deepseek_speaker_user_prompt",
    description="DeepSeek user prompt for speaker detection"
)

deepseek_summary_system_prompt: Optional[str] = Field(
    default=None,
    alias="deepseek_summary_system_prompt",
    description="DeepSeek system prompt for summarization (default: deepseek/summarization/system_v1)"
)

deepseek_summary_user_prompt: str = Field(
    default="deepseek/summarization/long_v1",
    alias="deepseek_summary_user_prompt",
    description="DeepSeek user prompt for summarization"
)
```

**Environment-based defaults:**
- **Test environment**: `deepseek-chat` (fast, extremely cheap)
- **Production environment**: `deepseek-chat` (same model, still very cheap)

## Model Options and Pricing

| Model | Input Cost (Cache Miss) | Input Cost (Cache Hit) | Output Cost | Context Window | Best For |
| ----- | ----------------------- | ---------------------- | ----------- | -------------- | -------- |
| **deepseek-chat** | $0.28 / 1M tokens | $0.028 / 1M tokens | $0.42 / 1M tokens | 64k tokens | General tasks |
| **deepseek-reasoner** (R1) | $0.28 / 1M tokens | $0.028 / 1M tokens | $0.42 / 1M tokens | 64k tokens | Complex reasoning |

**Note:** Prices subject to change. Check [DeepSeek Pricing](https://platform.deepseek.com/pricing) for current rates.

### Volume Discounts

| Tier | Monthly Usage | Discount |
| ---- | ------------- | -------- |
| Standard | 0-10M tokens | 0% |
| Growth | 10M-100M tokens | 10% |
| Scale | 100M-1B tokens | 20% |
| Enterprise | 1B+ tokens | 30%+ |

### Dev/Test vs Production Model Selection

| Environment | Speaker Model | Summary Model | Rationale |
| ----------- | ------------- | ------------- | --------- |
| **Dev/Test** | `deepseek-chat` | `deepseek-chat` | Fast, extremely cheap |
| **Production** | `deepseek-chat` | `deepseek-chat` | Same model, still very cheap |
| **Complex Reasoning** | `deepseek-reasoner` | `deepseek-reasoner` | For difficult analysis tasks |

### Cost Comparison: All Providers (Per 100 Episodes)

| Component | OpenAI (gpt-4o-mini) | Anthropic (haiku) | Mistral (small) | DeepSeek (chat) |
| --------- | -------------------- | ----------------- | --------------- | --------------- |
| **Transcription** | $0.60 | ❌ N/A | TBD | ❌ N/A |
| **Speaker Detection** | $0.14 | $0.10 | $0.03 | **$0.004** |
| **Summarization** | $0.41 | $0.30 | $0.08 | **$0.012** |
| **Total Text Processing** | **$0.55** | **$0.40** | **$0.11** | **$0.016** |

**DeepSeek is approximately 95% cheaper than OpenAI and 85% cheaper than Anthropic for text processing!**

### Monthly Cost Projection (1000 Episodes)

| Provider | Speaker Detection | Summarization | Total |
| -------- | ----------------- | ------------- | ----- |
| OpenAI (gpt-4o-mini) | $1.40 | $4.10 | $5.50 |
| Anthropic (haiku) | $1.00 | $3.00 | $4.00 |
| Mistral (small) | $0.30 | $0.80 | $1.10 |
| **DeepSeek (chat)** | **$0.04** | **$0.12** | **$0.16** |

## Non-Goals

- Transcription support (DeepSeek doesn't have audio API)
- Changing default behavior (local providers remain default)
- Modifying existing provider implementations
- Adding new features beyond provider selection
- Self-hosted DeepSeek deployment (future consideration)

## Personas

- **Budget-Conscious Bob**: Wants cheapest possible cloud processing
- **Quality Seeker Quinn**: Wants to compare DeepSeek vs other providers
- **Startup Steve**: Needs to minimize API costs while scaling
- **Developer Devin**: Needs to test with DeepSeek API during development
- **Privacy-First Pat**: Continues using local providers exclusively

## User Stories

- *As Budget-Conscious Bob, I can use DeepSeek for the cheapest cloud text processing available.*
- *As Quality Seeker Quinn, I can compare DeepSeek results with OpenAI/Anthropic/Mistral.*
- *As Startup Steve, I can minimize costs while processing thousands of episodes.*
- *As Developer Devin, I can set my DeepSeek API key in environment variables and test.*
- *As any operator, I get a clear error message if I try to use DeepSeek for transcription.*
- *As any operator, I can see which provider was used for each capability in logs.*

## Functional Requirements

### FR1: Provider Selection

- **FR1.1**: Add `"deepseek"` as valid value for `speaker_detector_provider` config field
- **FR1.2**: Add `"deepseek"` as valid value for `summary_provider` config field
- **FR1.3**: Attempting to set `transcription_provider: deepseek` results in clear error message
- **FR1.4**: Provider selection is independent per capability
- **FR1.5**: Default values maintain current behavior (local providers)
- **FR1.6**: Invalid provider values result in clear error messages
- **FR1.7**: Support both Config-based and experiment-based factory modes from the start

### FR2: Provider Capability Gap Handling

- **FR2.1**: Clear error message: "DeepSeek provider does not support transcription. Use 'whisper' (local), 'openai', or 'mistral' instead."
- **FR2.2**: Provider capability matrix documented in configuration reference
- **FR2.3**: Validation occurs at configuration load time (fail fast)

### FR3: API Key Management

- **FR3.1**: Support `DEEPSEEK_API_KEY` environment variable for API authentication (like `OPENAI_API_KEY`)
- **FR3.2**: Support `.env` file via `python-dotenv` for convenient configuration
- **FR3.3**: API key is never stored in source code or committed files
- **FR3.4**: `.env.example` template file updated with DeepSeek placeholder
- **FR3.5**: Missing API key results in clear error message
- **FR3.6**: API key validation at provider initialization (fail fast)
- **FR3.7**: Support `DEEPSEEK_API_BASE` environment variable for E2E testing (like `OPENAI_API_BASE`)

### FR4: Speaker Detection with DeepSeek

- **FR4.1**: DeepSeek provider uses chat models for entity extraction
- **FR4.2**: Maintains same interface as other providers
- **FR4.3**: Returns results in same format as other providers
- **FR4.4**: Handles API rate limits gracefully (retry with backoff)
- **FR4.5**: Uses DeepSeek-specific prompt templates

### FR5: Summarization with DeepSeek

- **FR5.1**: DeepSeek provider uses chat models for summarization
- **FR5.2**: Maintains same interface as other providers
- **FR5.3**: Leverages 64k token context window
- **FR5.4**: Returns results in same format as other providers
- **FR5.5**: Uses DeepSeek-specific prompt templates

### FR6: OpenAI-Compatible API

- **FR6.1**: Use OpenAI Python SDK with custom `base_url` for DeepSeek
- **FR6.2**: No separate DeepSeek SDK dependency required
- **FR6.3**: API format is identical to OpenAI chat completions

### FR7: Logging and Observability

- **FR7.1**: Log which provider is used for each capability
- **FR7.2**: Include provider information in metadata documents
- **FR7.3**: Log API usage for debugging
- **FR7.4**: No sensitive information in logs

### FR8: Error Handling

- **FR8.1**: API errors result in clear error messages
- **FR8.2**: Rate limit errors include retry information
- **FR8.3**: Network errors handled gracefully with retries
- **FR8.4**: Invalid API key errors are clear and actionable

## Technical Requirements

### TR1: Architecture

- **TR1.1**: Follow **unified provider pattern** (like OpenAI) - single class implementing both protocols
- **TR1.2**: Create `providers/deepseek/deepseek_provider.py` with unified `DeepSeekProvider` class
- **TR1.3**: `DeepSeekProvider` implements `SpeakerDetector` and `SummarizationProvider` protocols
- **TR1.4**: Update factories to include DeepSeek option with support for both Config-based and experiment-based modes
- **TR1.5**: Create `prompts/deepseek/` directory with provider-specific prompt templates
- **TR1.6**: Use OpenAI SDK with custom base_url (no new SDK dependency)
- **TR1.7**: Follow OpenAI provider architecture exactly for consistency

### TR2: Dependencies

- **TR2.1**: Reuse existing `openai` package (already installed for OpenAI provider)
- **TR2.2**: No additional SDK dependency required
- **TR2.3**: Lazy initialization of OpenAI client with DeepSeek base_url
- **TR2.4**: ImportError with helpful message if `openai` package not installed

### TR3: Configuration

- **TR3.1**: Add DeepSeek provider type to config Literal types
- **TR3.2**: Add DeepSeek-specific config fields
- **TR3.3**: Validate provider + API key consistency
- **TR3.4**: Validate DeepSeek not used for transcription

### TR4: Testing

- **TR4.1**: Unit tests for DeepSeek providers (with mocked API)
- **TR4.2**: Integration tests with E2E server mock endpoints
- **TR4.3**: E2E tests for complete workflow
- **TR4.4**: Tests verify same interface as other providers
- **TR4.5**: Backward compatibility tests

### TR5: E2E Server Extensions

- **TR5.1**: Add DeepSeek mock endpoints (reuse OpenAI format - same API structure)
- **TR5.2**: Mock `/v1/chat/completions` for DeepSeek (same as OpenAI endpoint)
- **TR5.3**: Add `deepseek_api_base()` helper to `E2EServerURLs` class
- **TR5.4**: Support `deepseek_api_base` config field for custom base URL (like `openai_api_base`)

## Success Criteria

- ✅ Users can select DeepSeek provider for speaker detection and summarization via unified provider
- ✅ Clear error when attempting transcription with DeepSeek
- ✅ Default behavior (local providers) unchanged
- ✅ API keys managed securely via `DEEPSEEK_API_KEY` environment variable
- ✅ Environment-based model defaults (test vs production)
- ✅ Both Config-based and experiment-based factory modes supported
- ✅ DeepSeek providers implement same interfaces as other providers
- ✅ Uses existing OpenAI SDK (no new dependency)
- ✅ Error handling is clear and actionable
- ✅ E2E tests pass with DeepSeek mock endpoints
- ✅ Follows OpenAI provider pattern exactly for consistency

## Out of Scope

- Transcription support (no DeepSeek audio API)
- Self-hosted DeepSeek deployment
- DeepSeek-specific reasoning features (thinking tokens)
- Function calling / tool use

## Dependencies

- **Prerequisite**: Modularization refactoring (RFC-021) ✅ Completed
- **Prerequisite**: OpenAI provider implementation (RFC-013) ✅ Completed
- **External**: DeepSeek API access and API key
- **Internal**: OpenAI Python SDK (already a dependency)

## Risks & Mitigations

- **Risk**: API availability in certain regions
  - **Mitigation**: Document regional considerations, support custom base_url
- **Risk**: Model quality differences from OpenAI/Anthropic
  - **Mitigation**: Optimize prompts for DeepSeek, document quality comparison
- **Risk**: Rate limits may differ from other providers
  - **Mitigation**: Implement retry logic, document limits

## Provider Capability Matrix (Updated)

| Capability | Local | OpenAI | Anthropic | Mistral | DeepSeek |
| ---------- | ----- | ------ | --------- | ------- | -------- |
| **Transcription** | ✅ Whisper | ✅ Whisper API | ❌ | ✅ Voxtral | ❌ |
| **Speaker Detection** | ✅ spaCy | ✅ GPT | ✅ Claude | ✅ Mistral | ✅ DeepSeek |
| **Summarization** | ✅ Transformers | ✅ GPT | ✅ Claude | ✅ Mistral | ✅ DeepSeek |

## Future Considerations

- Self-hosted DeepSeek deployment for maximum cost savings
- DeepSeek-R1 reasoning features for complex analysis
- Function calling for structured output
- Support for additional DeepSeek models as released
