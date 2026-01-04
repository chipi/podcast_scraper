# PRD-012: Google Gemini Provider Integration

- **Status**: Draft
- **Related RFCs**: RFC-035
- **Related PRDs**: PRD-006 (OpenAI), PRD-010 (Mistral)

## Summary

Add Google Gemini as an optional provider for transcription, speaker detection, and summarization capabilities. Gemini is unique in offering native multimodal support with audio understanding and an industry-leading 2 million token context window. Like OpenAI and Mistral, Gemini supports ALL three capabilities. This builds on the existing modularization architecture to provide seamless provider switching.

## Background & Context

Currently, the podcast scraper supports multiple providers. Google Gemini offers several compelling advantages:

- **Native Audio Understanding**: Gemini can process audio directly without separate transcription API
- **2M Token Context Window**: Process entire podcast seasons in one request
- **Multimodal**: Can analyze audio, images, and text together
- **Competitive Pricing**: Free tier available, paid tier very competitive
- **Google Integration**: Works well with Google Cloud ecosystem

This PRD addresses adding Gemini as a full-featured provider option.

## Goals

- Add Google Gemini as provider option for transcription, speaker detection, and summarization
- Maintain 100% backward compatibility with existing providers
- Follow identical architectural patterns as other providers
- Provide secure API key management via environment variables
- Leverage Gemini's native audio understanding for transcription
- Utilize massive context window for full transcript processing
- Create Gemini-specific prompt templates

## Gemini Model Selection and Cost Analysis

### Configuration Fields

Add to `config.py` when implementing Gemini providers:

```python

# Gemini Model Selection

gemini_speaker_model: str = Field(
    default="gemini-2.0-flash",
    description="Gemini model for speaker detection"
)

gemini_summary_model: str = Field(
    default="gemini-2.0-flash",
    description="Gemini model for summarization"
)

gemini_transcription_model: str = Field(
    default="gemini-2.0-flash",
    description="Gemini model for audio transcription"
)

# Gemini API Configuration

gemini_api_key: Optional[str] = Field(
    default=None,
    description="Google AI API key (prefer GEMINI_API_KEY env var or .env file)"
)

gemini_temperature: float = Field(
    default=0.3,
    ge=0.0,
    le=2.0,
    description="Temperature for Gemini generation"
)

gemini_max_tokens: Optional[int] = Field(
    default=8192,
    description="Max tokens for Gemini generation"
)
```yaml

## Model Options and Pricing

| Model | Input Cost | Output Cost | Context Window | Best For |
| ----- | ---------- | ----------- | -------------- | -------- |
| **gemini-2.0-flash** | $0.10 / 1M tokens | $0.40 / 1M tokens | 1M tokens | **Dev/Test** (fast, cheap) |
| **gemini-1.5-pro** | $1.25 / 1M tokens | $5.00 / 1M tokens | 2M tokens | **Production** (best quality) |
| **gemini-1.5-flash** | $0.075 / 1M tokens | $0.30 / 1M tokens | 1M tokens | Budget production |

**Audio Pricing:**

- Audio input: ~$0.00025 per second (~$0.90 per hour)

**Note:** Prices subject to change. Check [Google AI Pricing](https://ai.google.dev/pricing) for current rates.

### Free Tier

Google offers a generous free tier:

- **Gemini 2.0 Flash**: 15 RPM, 1M TPM, 1500 RPD
- **Gemini 1.5 Flash**: 15 RPM, 1M TPM, 1500 RPD
- **Gemini 1.5 Pro**: 2 RPM, 32K TPM, 50 RPD

### Dev/Test vs Production Model Selection

| Environment | Transcription | Speaker Model | Summary Model | Rationale |
| ----------- | ------------- | ------------- | ------------- | --------- |
| **Dev/Test** | `gemini-2.0-flash` | `gemini-2.0-flash` | `gemini-2.0-flash` | Free tier, fast |
| **Production** | `gemini-1.5-pro` | `gemini-1.5-pro` | `gemini-1.5-pro` | Best quality, 2M context |

### Cost Comparison: All Providers (Per 100 Episodes)

| Component | OpenAI (gpt-4o-mini) | Mistral (small) | DeepSeek (chat) | Gemini (flash) |
| --------- | -------------------- | --------------- | --------------- | -------------- |
| **Transcription** | $0.60 | TBD | ❌ N/A | **$0.90** |
| **Speaker Detection** | $0.14 | $0.03 | $0.004 | **$0.01** |
| **Summarization** | $0.41 | $0.08 | $0.012 | **$0.04** |
| **Total** | **$1.15** | **$0.11+** | **$0.016** | **$0.95** |

**Note:** Gemini audio transcription is slightly more expensive than Whisper, but offers native multimodal understanding.

## Non-Goals

- Vision/image capabilities beyond audio (future consideration)
- Changing default behavior (local providers remain default)
- Modifying existing provider implementations
- Google Cloud Vertex AI integration (different SDK, future PRD)

## Personas

- **Quality Seeker Quinn**: Wants 2M context for full season analysis
- **Google Cloud Gary**: Already uses Google Cloud, prefers ecosystem
- **Free Tier Fiona**: Wants to use generous free tier for development
- **Multimodal Mary**: Wants native audio understanding without separate transcription

## User Stories

- *As Quality Seeker Quinn, I can use Gemini 1.5 Pro with 2M context to analyze entire seasons.*
- *As Google Cloud Gary, I can use my existing Google AI API key for podcast processing.*
- *As Free Tier Fiona, I can develop and test using Gemini's free tier.*
- *As Multimodal Mary, I can send audio directly to Gemini without separate transcription.*
- *As any operator, I can see which provider was used in logs and metadata.*

## Functional Requirements

### FR1: Provider Selection

- **FR1.1**: Add `"gemini"` as valid value for `transcription_provider` config field
- **FR1.2**: Add `"gemini"` as valid value for `speaker_detector_provider` config field
- **FR1.3**: Add `"gemini"` as valid value for `summary_provider` config field
- **FR1.4**: Provider selection is independent per capability
- **FR1.5**: Default values maintain current behavior (local providers)

### FR2: API Key Management

- **FR2.1**: Support `GEMINI_API_KEY` environment variable
- **FR2.2**: Support `.env` file for configuration
- **FR2.3**: API key is never stored in source code
- **FR2.4**: Missing API key results in clear error message

### FR3: Transcription with Gemini

- **FR3.1**: Gemini provider uses native audio understanding for transcription
- **FR3.2**: Maintains same interface as other providers
- **FR3.3**: Supports audio file upload to Gemini API
- **FR3.4**: Returns results in same format as other providers
- **FR3.5**: Handles large audio files via chunking if needed

### FR4: Speaker Detection with Gemini

- **FR4.1**: Gemini provider uses chat models for entity extraction
- **FR4.2**: Maintains same interface as other providers
- **FR4.3**: Uses Gemini-specific prompt templates

### FR5: Summarization with Gemini

- **FR5.1**: Gemini provider uses chat models for summarization
- **FR5.2**: Leverages massive context window (up to 2M tokens)
- **FR5.3**: Can process entire transcripts without chunking
- **FR5.4**: Uses Gemini-specific prompt templates

### FR6: Prompt Management

- **FR6.1**: Create Gemini-specific prompt templates in `prompts/gemini/` folder
- **FR6.2**: Prompts optimized for Gemini models

### FR7: Logging and Observability

- **FR7.1**: Log which provider is used for each capability
- **FR7.2**: Include provider information in metadata documents

## Technical Requirements

### TR1: Architecture

- **TR1.1**: Follow identical patterns as other providers
- **TR1.2**: Create `podcast_scraper/gemini/` package for shared utilities
- **TR1.3**: Create `transcription/gemini_provider.py` implementing `TranscriptionProvider`
- **TR1.4**: Create `speaker_detectors/gemini_detector.py` implementing `SpeakerDetector`
- **TR1.5**: Create `summarization/gemini_provider.py` implementing `SummarizationProvider`
- **TR1.6**: Update all factories to include Gemini option

### TR2: Dependencies

- **TR2.1**: Add `google-genai` Python package as optional dependency
- **TR2.2**: Lazy import when provider is selected

### TR3: Testing

- **TR3.1**: Unit tests with mocked API
- **TR3.2**: Integration tests with E2E server mock
- **TR3.3**: E2E tests for complete workflow

### TR4: E2E Server Extensions

- **TR4.1**: Add Gemini mock endpoints to E2E HTTP server
- **TR4.2**: Add `gemini_api_base()` helper to `E2EServerURLs`

## Success Criteria

- ✅ Users can select Gemini for all three capabilities
- ✅ Native audio transcription works via Gemini
- ✅ Default behavior unchanged
- ✅ API keys managed securely
- ✅ E2E tests pass

## Provider Capability Matrix (Updated)

| Capability | Local | OpenAI | Anthropic | Mistral | DeepSeek | Gemini |
| ---------- | ----- | ------ | --------- | ------- | -------- | ------ |
| **Transcription** | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| **Speaker Detection** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Summarization** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Max Context** | N/A | 128k | 200k | 256k | 64k | **2M** |

## Future Considerations

- Vertex AI integration for enterprise deployments
- Multimodal analysis (audio + images for show notes)
- Gemini's native grounding with Google Search
- Context caching for cost optimization
