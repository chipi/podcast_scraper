# PRD-013: Groq Provider Integration

- **Status**: Draft (Revised)
- **Revision**: 2
- **Date**: 2026-02-04
- **Related RFCs**: RFC-036 (Revised)
- **Related PRDs**: PRD-006 (OpenAI), PRD-011 (DeepSeek)

## Summary

Add Groq as an optional provider for speaker detection and summarization capabilities. Groq is unique in offering **ultra-fast inference** (10x faster than other providers) by running models on custom LPU (Language Processing Unit) hardware. Like OpenAI, Groq uses a **unified provider pattern** where a single `GroqProvider` class implements both capabilities. Like Anthropic and DeepSeek, Groq does NOT support audio transcription. Groq hosts open-source models like Llama 3.3, Mixtral, and Gemma.

## Background & Context

Groq offers several compelling advantages:

- **Ultra-Fast Inference**: 10x faster than OpenAI/Anthropic (500+ tokens/second)
- **Open Source Models**: Hosts Llama 3.3 70B, Mixtral, Gemma
- **OpenAI-Compatible API**: Same format as OpenAI, easy integration
- **Competitive Pricing**: Very affordable, especially for Llama models
- **Generous Free Tier**: 14,400 tokens/minute for free

This PRD addresses adding Groq as a speed-optimized provider.

## Goals

- Add Groq as provider option for speaker detection and summarization
- Maintain 100% backward compatibility
- Follow **unified provider pattern** (like OpenAI) - single class implementing both protocols
- Provide secure API key management via environment variables and `.env` files
- Support both Config-based and experiment-based factory modes from the start
- Handle capability gaps gracefully (no transcription)
- Use OpenAI SDK with custom base_url (no new dependency)
- Use environment-based model defaults (test vs production)

## Groq Model Selection and Cost Analysis

### Configuration Fields

Add to `config.py` when implementing Groq providers (following OpenAI pattern):

```python
# Groq API Configuration
groq_api_key: Optional[str] = Field(
    default=None,
    alias="groq_api_key",
    description="Groq API key (prefer GROQ_API_KEY env var or .env file)"
)

groq_api_base: Optional[str] = Field(
    default=None,
    alias="groq_api_base",
    description="Groq API base URL (default: https://api.groq.com/openai/v1, for E2E testing)"
)

# Groq Model Selection (environment-based defaults, like OpenAI)
groq_speaker_model: str = Field(
    default_factory=_get_default_groq_speaker_model,
    alias="groq_speaker_model",
    description="Groq model for speaker detection (default: environment-based)"
)

groq_summary_model: str = Field(
    default_factory=_get_default_groq_summary_model,
    alias="groq_summary_model",
    description="Groq model for summarization (default: environment-based)"
)

# Shared settings (like OpenAI)
groq_temperature: float = Field(
    default=0.3,
    alias="groq_temperature",
    description="Temperature for Groq generation (0.0-2.0, lower = more deterministic)"
)

groq_max_tokens: Optional[int] = Field(
    default=None,
    alias="groq_max_tokens",
    description="Max tokens for Groq generation (None = model default)"
)

# Groq Prompt Configuration (following OpenAI pattern)
groq_speaker_system_prompt: Optional[str] = Field(
    default=None,
    alias="groq_speaker_system_prompt",
    description="Groq system prompt for speaker detection (default: groq/ner/system_ner_v1)"
)

groq_speaker_user_prompt: str = Field(
    default="groq/ner/guest_host_v1",
    alias="groq_speaker_user_prompt",
    description="Groq user prompt for speaker detection"
)

groq_summary_system_prompt: Optional[str] = Field(
    default=None,
    alias="groq_summary_system_prompt",
    description="Groq system prompt for summarization (default: groq/summarization/system_v1)"
)

groq_summary_user_prompt: str = Field(
    default="groq/summarization/long_v1",
    alias="groq_summary_user_prompt",
    description="Groq user prompt for summarization"
)
```

**Environment-based defaults:**
- **Test environment**: `llama-3.1-8b-instant` (free tier, ultra-fast)
- **Production environment**: `llama-3.3-70b-versatile` (best quality, still fast)

## Model Options and Pricing

| Model | Input Cost | Output Cost | Context Window | Speed | Best For |
| ----- | ---------- | ----------- | -------------- | ----- | -------- |
| **llama-3.3-70b-versatile** | $0.59 / 1M tokens | $0.79 / 1M tokens | 128k | Ultra-fast | **Production** |
| **llama-3.1-8b-instant** | $0.05 / 1M tokens | $0.08 / 1M tokens | 128k | Ultra-fast | **Dev/Test** |
| **mixtral-8x7b-32768** | $0.24 / 1M tokens | $0.24 / 1M tokens | 32k | Ultra-fast | Alternative |
| **gemma2-9b-it** | $0.20 / 1M tokens | $0.20 / 1M tokens | 8k | Ultra-fast | Budget |

### Free Tier Limits

| Model | Requests/Min | Tokens/Min | Requests/Day |
| ----- | ------------ | ---------- | ------------ |
| llama-3.3-70b-versatile | 30 | 14,400 | 14,400 |
| llama-3.1-8b-instant | 30 | 14,400 | 14,400 |
| mixtral-8x7b | 30 | 14,400 | 14,400 |

### Dev/Test vs Production Model Selection

| Environment | Speaker Model | Summary Model | Rationale |
| ----------- | ------------- | ------------- | --------- |
| **Dev/Test** | `llama-3.1-8b-instant` | `llama-3.1-8b-instant` | Free tier, ultra-fast |
| **Production** | `llama-3.3-70b-versatile` | `llama-3.3-70b-versatile` | Best quality, still fast |

### Cost Comparison: All Providers (Per 100 Episodes)

| Component | OpenAI (gpt-4o-mini) | DeepSeek (chat) | Groq (8b) | Groq (70b) |
| --------- | -------------------- | --------------- | --------- | ---------- |
| **Transcription** | $0.60 | ❌ N/A | ❌ N/A | ❌ N/A |
| **Speaker Detection** | $0.14 | $0.004 | **$0.006** | $0.07 |
| **Summarization** | $0.41 | $0.012 | **$0.02** | $0.15 |
| **Total Text** | **$0.55** | **$0.016** | **$0.026** | **$0.22** |

**Speed Advantage:** Groq processes in ~1/10th the time of other providers!

### Processing Time Comparison (Single Episode Summary)

| Provider | Time | Tokens/Second |
| -------- | ---- | ------------- |
| OpenAI GPT-4o-mini | ~5 seconds | 100 |
| Anthropic Claude | ~5 seconds | 100 |
| DeepSeek | ~3 seconds | 150 |
| **Groq Llama 3.3 70B** | **~0.5 seconds** | **500+** |

## Non-Goals

- Transcription support (Groq hosts LLMs only, no audio models)
- Changing default behavior
- Self-hosted Groq (not available)

## Personas

- **Speed-First Sam**: Needs fastest possible processing for real-time workflows
- **Budget-Conscious Bob**: Wants cheap, fast processing with free tier
- **Open Source Otto**: Prefers Llama/Mixtral over proprietary models
- **Batch Processing Brenda**: Needs to process many episodes quickly

## User Stories

- *As Speed-First Sam, I can use Groq for 10x faster processing than OpenAI.*
- *As Budget-Conscious Bob, I can use Groq's free tier for development.*
- *As Open Source Otto, I can use Llama 3.3 70B instead of proprietary models.*
- *As Batch Processing Brenda, I can process 100 episodes in minutes instead of hours.*

## Functional Requirements

### FR1: Provider Selection

- **FR1.1**: Add `"groq"` as valid value for `speaker_detector_provider`
- **FR1.2**: Add `"groq"` as valid value for `summary_provider`
- **FR1.3**: Attempting transcription with Groq results in clear error
- **FR1.4**: Default values maintain current behavior
- **FR1.5**: Support both Config-based and experiment-based factory modes from the start

### FR2: API Key Management

- **FR2.1**: Support `GROQ_API_KEY` environment variable (like `OPENAI_API_KEY`)
- **FR2.2**: Support `.env` file via `python-dotenv` for convenient configuration
- **FR2.3**: Clear error on missing API key
- **FR2.4**: Support `GROQ_API_BASE` environment variable for E2E testing (like `OPENAI_API_BASE`)

### FR3: Speaker Detection with Groq

- **FR3.1**: Groq provider uses hosted LLMs for entity extraction
- **FR3.2**: Maintains same interface as other providers
- **FR3.3**: Uses Groq-specific prompt templates

### FR4: Summarization with Groq

- **FR4.1**: Groq provider uses hosted LLMs for summarization
- **FR4.2**: Leverages Llama's 128k context window
- **FR4.3**: Maintains same interface

### FR5: OpenAI-Compatible API

- **FR5.1**: Use OpenAI SDK with `base_url=https://api.groq.com/openai/v1`
- **FR5.2**: No separate SDK dependency required

## Technical Requirements

### TR1: Architecture

- **TR1.1**: Follow **unified provider pattern** (like OpenAI) - single class implementing both protocols
- **TR1.2**: Create `providers/groq/groq_provider.py` with unified `GroqProvider` class
- **TR1.3**: `GroqProvider` implements `SpeakerDetector` and `SummarizationProvider` protocols
- **TR1.4**: Update factories to include Groq option with support for both Config-based and experiment-based modes
- **TR1.5**: Create `prompts/groq/` directory with provider-specific prompt templates
- **TR1.6**: Use OpenAI SDK with custom base_url (no new SDK dependency)
- **TR1.7**: Follow OpenAI provider architecture exactly for consistency

### TR2: Dependencies

- **TR2.1**: Reuse existing `openai` package
- **TR2.2**: No new SDK dependency

## Success Criteria

- ✅ Users can select Groq provider for speaker detection and summarization via unified provider
- ✅ Clear error when attempting transcription with Groq
- ✅ 10x faster than other providers (500+ tokens/second)
- ✅ Free tier works for development
- ✅ Environment-based model defaults (test vs production)
- ✅ Both Config-based and experiment-based factory modes supported
- ✅ No new SDK dependency (uses OpenAI SDK)
- ✅ E2E tests pass
- ✅ Follows OpenAI provider pattern exactly for consistency

## Provider Capability Matrix (Updated)

| Capability | Local | OpenAI | Anthropic | Mistral | DeepSeek | Gemini | Groq |
| ---------- | ----- | ------ | --------- | ------- | -------- | ------ | ---- |
| **Transcription** | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ |
| **Speaker Detection** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Summarization** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Speed** | Slow | Medium | Medium | Medium | Fast | Fast | **Ultra** |

## Future Considerations

- Whisper on Groq (if they add audio models)
- Tool use / function calling
- Streaming responses for real-time display
