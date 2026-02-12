# PRD-013: Grok Provider Integration (xAI)

- **Status**: ✅ Implemented (v2.5.0)
- **Revision**: 3
- **Date**: 2026-02-05
- **Implementation**: Issue #1095
- **Related RFCs**: RFC-036 (Updated)
- **Related PRDs**: PRD-006 (OpenAI), PRD-011 (DeepSeek)

## Summary

Add Grok (by xAI) as an optional provider for speaker detection and summarization capabilities. Grok is xAI's AI model, available through their API. Like OpenAI, Grok uses a **unified provider pattern** where a single `GrokProvider` class implements both capabilities. Like Anthropic and DeepSeek, Grok does NOT support audio transcription (xAI focuses on text-based LLMs).

**Note:** API details have been researched based on xAI's public API documentation and common OpenAI-compatible API patterns.

## Background & Context

Grok (xAI) offers several advantages:

- **xAI's AI Model**: Grok is xAI's proprietary AI model (Elon Musk's AI company)
- **Real-time Information**: Grok has access to real-time information via X/Twitter integration
- **OpenAI-Compatible API**: Uses OpenAI-compatible API format (can reuse OpenAI SDK with custom base_url)
- **Competitive Pricing**: Competitive pricing compared to other providers
- **Public API**: Public API available at `https://api.x.ai/v1` (verify with your API key)

**API Details (Verified/Assumed):**

- Base URL: `https://api.x.ai/v1` (OpenAI-compatible endpoint)
- SDK: Uses OpenAI SDK with custom `base_url` (no new dependency)
- Authentication: API key via `GROK_API_KEY` environment variable
- Model names: `grok-beta` (beta/development), `grok-2` (production) - verify with your API access
- Context window: Likely 128k tokens (verify with API documentation)

This PRD addresses adding Grok as an xAI-based provider.

## Goals

- Add Grok as provider option for speaker detection and summarization
- Maintain 100% backward compatibility
- Follow **unified provider pattern** (like OpenAI) - single class implementing both protocols
- Provide secure API key management via environment variables and `.env` files
- Support both Config-based and experiment-based factory modes from the start
- Handle capability gaps gracefully (no transcription)
- Use OpenAI SDK with custom base_url if API is OpenAI-compatible (needs verification)
- Use environment-based model defaults (test vs production)

## Grok Model Selection and Cost Analysis

**⚠️ Note:** Model names, pricing, and API details below are placeholders and need verification from xAI's official documentation.

### Configuration Fields

Add to `config.py` when implementing Grok providers (following OpenAI pattern):

```python
# Grok API Configuration
grok_api_key: Optional[str] = Field(
    default=None,
    alias="grok_api_key",
    description="Grok API key (prefer GROK_API_KEY env var or .env file)"
)

grok_api_base: Optional[str] = Field(
    default=None,
    alias="grok_api_base",
    description="Grok API base URL (default: https://api.x.ai/v1, for E2E testing)"
)

# Grok Model Selection (environment-based defaults, like OpenAI)
grok_speaker_model: str = Field(
    default_factory=_get_default_grok_speaker_model,
    alias="grok_speaker_model",
    description="Grok model for speaker detection (default: environment-based)"
)

grok_summary_model: str = Field(
    default_factory=_get_default_grok_summary_model,
    alias="grok_summary_model",
    description="Grok model for summarization (default: environment-based)"
)

# Shared settings (like OpenAI)
grok_temperature: float = Field(
    default=0.3,
    alias="grok_temperature",
    description="Temperature for Grok generation (0.0-2.0, lower = more deterministic)"
)

grok_max_tokens: Optional[int] = Field(
    default=None,
    alias="grok_max_tokens",
    description="Max tokens for Grok generation (None = model default)"
)

# Grok Prompt Configuration (following OpenAI pattern)
grok_speaker_system_prompt: Optional[str] = Field(
    default=None,
    alias="grok_speaker_system_prompt",
    description="Grok system prompt for speaker detection (default: grok/ner/system_ner_v1)"
)

grok_speaker_user_prompt: str = Field(
    default="grok/ner/guest_host_v1",
    alias="grok_speaker_user_prompt",
    description="Grok user prompt for speaker detection"
)

grok_summary_system_prompt: Optional[str] = Field(
    default=None,
    alias="grok_summary_system_prompt",
    description="Grok system prompt for summarization (default: grok/summarization/system_v1)"
)

grok_summary_user_prompt: str = Field(
    default="grok/summarization/long_v1",
    alias="grok_summary_user_prompt",
    description="Grok user prompt for summarization"
)
```

**Environment-based defaults:**

- **Test environment**: `grok-beta` (beta model, typically available for development)
- **Production environment**: `grok-2` (production model, best quality)

**Note:** Verify actual model names with your xAI API access. Common patterns suggest `grok-beta` and `grok-2`, but model availability may vary.

## Model Options and Pricing

**⚠️ Note:** Pricing information should be verified from xAI's official documentation at <https://console.x.ai> or <https://docs.x.ai>. The following are estimates based on common pricing patterns.

| Model | Input Cost | Output Cost | Context Window | Speed | Best For |
| ----- | ---------- | ----------- | -------------- | ----- | -------- |
| **grok-2** | Verify pricing | Verify pricing | 128k (verify) | Medium | **Production** |
| **grok-beta** | Verify pricing | Verify pricing | 128k (verify) | Medium | **Dev/Test** |

**Source:** Verify current pricing at <https://console.x.ai> or <https://docs.x.ai>. Pricing may vary based on your account tier.

### Free Tier Limits

**⚠️ Needs Verification:** Free tier availability and limits should be verified from xAI documentation at <https://console.x.ai>.

| Model | Requests/Min | Tokens/Min | Requests/Day |
| ----- | ------------ | ---------- | ------------ |
| Verify with API | Verify with API | Verify with API | Verify with API |

**Note:** Check your xAI account dashboard for current rate limits and free tier availability.

### Dev/Test vs Production Model Selection

**Note:** Verify model names with your xAI API access. Common patterns suggest these names, but availability may vary.

| Environment | Speaker Model | Summary Model | Rationale |
| ----------- | ------------- | ------------- | --------- |
| **Dev/Test** | `grok-beta` | `grok-beta` | Beta model for development/testing |
| **Production** | `grok-2` | `grok-2` | Production model, best quality |

### Cost Comparison: All Providers (Per 100 Episodes)

**⚠️ Note:** Grok pricing should be verified from xAI documentation. Estimates based on common pricing patterns.

| Component | OpenAI (gpt-4o-mini) | DeepSeek (chat) | Grok (verify) |
| --------- | -------------------- | --------------- | ------------- |
| **Transcription** | $0.60 | ❌ N/A | ❌ N/A |
| **Speaker Detection** | $0.14 | $0.004 | Verify pricing |
| **Summarization** | $0.41 | $0.012 | Verify pricing |
| **Total Text** | **$0.55** | **$0.016** | **Verify pricing** |

### Processing Time Comparison (Single Episode Summary)

**⚠️ Note:** Grok performance metrics should be verified through actual API testing.

| Provider | Time | Tokens/Second |
| -------- | ---- | ------------- |
| OpenAI GPT-4o-mini | ~5 seconds | 100 |
| Anthropic Claude | ~5 seconds | 100 |
| DeepSeek | ~3 seconds | 150 |
| **Grok** | **Verify** | **Verify** |

## Non-Goals

- Transcription support (Grok/xAI focuses on text-based LLMs, no audio models)
- Changing default behavior
- Self-hosted Grok (not available)

## Personas

- **Real-Time Rita**: Needs access to real-time information via Grok's X/Twitter integration
- **xAI Enthusiast**: Prefers xAI's Grok model over other providers
- **Batch Processing Brenda**: Needs to process many episodes efficiently

## User Stories

- *As Real-Time Rita, I can use Grok to leverage real-time information in summaries.*
- *As xAI Enthusiast, I can use Grok as my preferred AI provider.*
- *As Batch Processing Brenda, I can process episodes using Grok's API.*

## Functional Requirements

### FR1: Provider Selection

- **FR1.1**: Add `"grok"` as valid value for `speaker_detector_provider`
- **FR1.2**: Add `"grok"` as valid value for `summary_provider`
- **FR1.3**: Attempting transcription with Grok results in clear error
- **FR1.4**: Default values maintain current behavior
- **FR1.5**: Support both Config-based and experiment-based factory modes from the start

### FR2: API Key Management

- **FR2.1**: Support `GROK_API_KEY` environment variable (like `OPENAI_API_KEY`)
- **FR2.2**: Support `.env` file via `python-dotenv` for convenient configuration
- **FR2.3**: Clear error on missing API key
- **FR2.4**: Support `GROK_API_BASE` environment variable for E2E testing (like `OPENAI_API_BASE`)

### FR3: Speaker Detection with Grok

- **FR3.1**: Grok provider uses xAI's API for entity extraction
- **FR3.2**: Maintains same interface as other providers
- **FR3.3**: Uses Grok-specific prompt templates

### FR4: Summarization with Grok

- **FR4.1**: Grok provider uses xAI's API for summarization
- **FR4.2**: Leverages Grok's context window (size TBD)
- **FR4.3**: Maintains same interface

### FR5: API Compatibility

- **FR5.1**: Use OpenAI SDK with custom `base_url` if Grok API is OpenAI-compatible (needs verification)
- **FR5.2**: Alternative: Use xAI SDK if available (needs research)
- **FR5.3**: Minimize new dependencies

## Technical Requirements

### TR1: Architecture

- **TR1.1**: Follow **unified provider pattern** (like OpenAI) - single class implementing both protocols
- **TR1.2**: Create `providers/grok/grok_provider.py` with unified `GrokProvider` class
- **TR1.3**: `GrokProvider` implements `SpeakerDetector` and `SummarizationProvider` protocols
- **TR1.4**: Update factories to include Grok option with support for both Config-based and experiment-based modes
- **TR1.5**: Create `prompts/grok/` directory with provider-specific prompt templates
- **TR1.6**: Use OpenAI SDK with custom base_url if API is OpenAI-compatible, or xAI SDK if available
- **TR1.7**: Follow OpenAI provider architecture exactly for consistency

### TR2: Dependencies

- **TR2.1**: Prefer reusing existing `openai` package if Grok API is OpenAI-compatible
- **TR2.2**: Alternative: Use xAI SDK if available (needs research)
- **TR2.3**: Minimize new dependencies

## Success Criteria

- ✅ Users can select Grok provider for speaker detection and summarization via unified provider
- ✅ Clear error when attempting transcription with Grok
- ✅ API integration works (OpenAI-compatible API at <https://api.x.ai/v1>)
- ✅ Real-time information access via X/Twitter integration
- ✅ Environment-based model defaults (test vs production)
- ✅ Both Config-based and experiment-based factory modes supported
- ✅ No new SDK dependency (uses OpenAI SDK)
- ✅ E2E tests pass
- ✅ Follows OpenAI provider pattern exactly for consistency

## Provider Capability Matrix (Updated)

| Capability | Local | OpenAI | Anthropic | Mistral | DeepSeek | Gemini | Grok |
| ---------- | ----- | ------ | --------- | ------- | -------- | ------ | ---- |
| **Transcription** | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ |
| **Speaker Detection** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Summarization** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Real-time Info** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (via X/Twitter) |

## Future Considerations

- Audio transcription support (if xAI adds audio models)
- Tool use / function calling
- Streaming responses for real-time display
- Enhanced real-time information integration
