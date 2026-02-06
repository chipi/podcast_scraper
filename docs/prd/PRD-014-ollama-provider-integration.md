# PRD-014: Ollama Provider Integration

- **Status**: Draft (Revised)
- **Revision**: 2
- **Date**: 2026-02-04
- **Related RFCs**: RFC-037 (Revised)
- **Related PRDs**: PRD-006 (OpenAI), PRD-013 (Grok)

## Summary

Add Ollama as an optional provider for speaker detection and summarization capabilities. Ollama is unique in being a **fully local/offline solution** that runs open-source LLMs on your own hardware. Unlike cloud providers, Ollama requires NO API keys, has NO rate limits, and incurs NO per-token costs. Like OpenAI, Ollama uses a **unified provider pattern** where a single `OllamaProvider` class implements both capabilities. Like Anthropic, DeepSeek, and Grok, Ollama does NOT support audio transcription.

## Background & Context

Ollama offers several compelling advantages:

- **Fully Local/Offline**: No internet required, complete privacy
- **Zero API Costs**: After hardware, unlimited free usage
- **No Rate Limits**: Process as fast as your hardware allows
- **Open Source Models**: Llama 3.3, Mistral, Gemma, Phi, Qwen, etc.
- **OpenAI-Compatible API**: Same format, uses OpenAI SDK
- **Model Flexibility**: Easily switch between models

This PRD addresses adding Ollama as a privacy-first, cost-free provider option.

## Goals

- Add Ollama as provider option for speaker detection and summarization
- Maintain 100% backward compatibility
- Follow **unified provider pattern** (like OpenAI) - single class implementing both protocols
- Support local model selection (user installs models via Ollama CLI)
- Support both Config-based and experiment-based factory modes from the start
- Handle capability gaps gracefully (no transcription)
- Use OpenAI SDK with custom base_url (no new SDK dependency)
- Use environment-based model defaults (test vs production)
- Validate Ollama server is running and models are available

## Ollama Model Selection and Cost Analysis

### Configuration Fields

Add to `config.py` when implementing Ollama providers (following OpenAI pattern):

```python
# Ollama API Configuration (NO API KEY NEEDED - local service)
ollama_api_base: Optional[str] = Field(
    default=None,
    alias="ollama_api_base",
    description="Ollama API base URL (default: http://localhost:11434/v1, for E2E testing)"
)

# Ollama Model Selection (environment-based defaults, like OpenAI)
ollama_speaker_model: str = Field(
    default_factory=_get_default_ollama_speaker_model,
    alias="ollama_speaker_model",
    description="Ollama model for speaker detection (default: environment-based)"
)

ollama_summary_model: str = Field(
    default_factory=_get_default_ollama_summary_model,
    alias="ollama_summary_model",
    description="Ollama model for summarization (default: environment-based)"
)

# Shared settings (like OpenAI)
ollama_temperature: float = Field(
    default=0.3,
    alias="ollama_temperature",
    description="Temperature for Ollama generation (0.0-2.0, lower = more deterministic)"
)

ollama_max_tokens: Optional[int] = Field(
    default=None,
    alias="ollama_max_tokens",
    description="Max tokens for Ollama generation (None = model default)"
)

# Ollama Connection Settings
ollama_timeout: int = Field(
    default=120,
    alias="ollama_timeout",
    description="Timeout in seconds for Ollama API calls (local inference can be slow)"
)

# Ollama Prompt Configuration (following OpenAI pattern)
ollama_speaker_system_prompt: Optional[str] = Field(
    default=None,
    alias="ollama_speaker_system_prompt",
    description="Ollama system prompt for speaker detection (default: ollama/ner/system_ner_v1)"
)

ollama_speaker_user_prompt: str = Field(
    default="ollama/ner/guest_host_v1",
    alias="ollama_speaker_user_prompt",
    description="Ollama user prompt for speaker detection"
)

ollama_summary_system_prompt: Optional[str] = Field(
    default=None,
    alias="ollama_summary_system_prompt",
    description="Ollama system prompt for summarization (default: ollama/summarization/system_v1)"
)

ollama_summary_user_prompt: str = Field(
    default="ollama/summarization/long_v1",
    alias="ollama_summary_user_prompt",
    description="Ollama user prompt for summarization"
)
```

**Environment-based defaults:**
- **Test environment**: `llama3.2:latest` (smaller, faster for testing)
- **Production environment**: `llama3.3:latest` (best quality, 128k context)

## Model Options and Cost Analysis

| Model | Size | RAM Required | Context Window | Quality | Speed |
| ----- | ---- | ------------ | -------------- | ------- | ----- |
| **llama3.3:70b** | 40GB | 48GB+ | 128k | Best | Slow |
| **llama3.2:latest** | 2GB | 4GB+ | 128k | Good | Fast |
| **mistral:latest** | 4GB | 8GB+ | 32k | Good | Fast |
| **gemma2:9b** | 5GB | 8GB+ | 8k | Good | Fast |
| **phi3:medium** | 8GB | 16GB+ | 128k | Good | Medium |
| **qwen2.5:14b** | 9GB | 16GB+ | 128k | Excellent | Medium |

### Cost Comparison: Ollama vs Cloud (Per 1000 Episodes)

| Component | OpenAI (gpt-4o-mini) | DeepSeek | Grok | **Ollama** |
| --------- | -------------------- | -------- | ---- | ---------- |
| **Transcription** | $6.00 | ❌ N/A | ❌ N/A | ❌ N/A |
| **Speaker Detection** | $1.40 | $0.04 | $0.06 | **$0.00** |
| **Summarization** | $4.10 | $0.12 | $0.20 | **$0.00** |
| **Total** | **$11.50** | **$0.16** | **$0.26** | **$0.00** |

### One-Time Hardware Investment

For optimal Ollama performance:

| Hardware | Cost | Performance |
| -------- | ---- | ----------- |
| Mac Mini M4 (16GB) | ~$600 | Good for small models |
| Mac Studio M2 Max (64GB) | ~$3,000 | Excellent for 70B models |
| PC with RTX 4090 (24GB VRAM) | ~$2,500 | Excellent speed |

**Break-even Analysis:** At 10,000 episodes/month with OpenAI, hardware pays for itself in ~3 months.

### Processing Speed Comparison

| Provider | 1000-word Summary Time |
| -------- | ---------------------- |
| OpenAI GPT-4o-mini | ~5 seconds |
| Grok | ~1.0 seconds |
| Ollama (Llama 3.3 on M4 Pro) | ~30 seconds |
| Ollama (Llama 3.2 on M4 Pro) | ~3 seconds |

**Note:** Ollama is slower than cloud providers but costs nothing per request.

## Non-Goals

- Transcription support (Ollama hosts LLMs only)
- Automatic model installation (user manages via `ollama pull`)
- GPU optimization (handled by Ollama itself)
- Changing default behavior

## Personas

- **Privacy-First Pat**: Wants all processing on local hardware
- **Budget-Conscious Bob**: Wants zero ongoing API costs
- **Offline Otto**: Needs processing without internet connection
- **Self-Hosted Steve**: Prefers local infrastructure over cloud
- **Enterprise Emily**: Needs data to never leave corporate network

## User Stories

- *As Privacy-First Pat, I can process podcasts without sending data to cloud APIs.*
- *As Budget-Conscious Bob, I can process unlimited episodes with no API costs.*
- *As Offline Otto, I can process podcasts on an airplane or remote location.*
- *As Self-Hosted Steve, I can run everything on my own hardware.*
- *As Enterprise Emily, I can ensure sensitive podcast data never leaves our network.*

## Functional Requirements

### FR1: Provider Selection

- **FR1.1**: Add `"ollama"` as valid value for `speaker_detector_provider`
- **FR1.2**: Add `"ollama"` as valid value for `summary_provider`
- **FR1.3**: Attempting transcription with Ollama results in clear error
- **FR1.4**: Default values maintain current behavior
- **FR1.5**: Support both Config-based and experiment-based factory modes from the start

### FR2: No API Key Required

- **FR2.1**: Ollama does not require API key (local service)
- **FR2.2**: Clear error if Ollama server is not running (with helpful instructions)
- **FR2.3**: Support custom `ollama_api_base` for remote Ollama servers
- **FR2.4**: Support `OLLAMA_API_BASE` environment variable (like `OPENAI_API_BASE`)

### FR3: Model Availability Validation

- **FR3.1**: Validate model exists in local Ollama installation
- **FR3.2**: Provide helpful error with `ollama pull` command if model missing
- **FR3.3**: List available models in error message

### FR4: Speaker Detection with Ollama

- **FR4.1**: Ollama provider uses local LLMs for entity extraction
- **FR4.2**: Maintains same interface as other providers
- **FR4.3**: Uses Ollama-specific prompt templates

### FR5: Summarization with Ollama

- **FR5.1**: Ollama provider uses local LLMs for summarization
- **FR5.2**: Leverages model's context window
- **FR5.3**: Maintains same interface

### FR6: OpenAI-Compatible API

- **FR6.1**: Use OpenAI SDK with `base_url=http://localhost:11434/v1`
- **FR6.2**: No separate SDK dependency required

## Technical Requirements

### TR1: Architecture

- **TR1.1**: Follow **unified provider pattern** (like OpenAI) - single class implementing both protocols
- **TR1.2**: Create `providers/ollama/ollama_provider.py` with unified `OllamaProvider` class
- **TR1.3**: `OllamaProvider` implements `SpeakerDetector` and `SummarizationProvider` protocols
- **TR1.4**: Update factories to include Ollama option with support for both Config-based and experiment-based modes
- **TR1.5**: Create `prompts/ollama/` directory with provider-specific prompt templates
- **TR1.6**: Use OpenAI SDK with custom base_url (no new SDK dependency)
- **TR1.7**: Follow OpenAI provider architecture exactly for consistency

### TR2: Dependencies

- **TR2.1**: Reuse existing `openai` package (already installed for OpenAI provider)
- **TR2.2**: Add `httpx` package for Ollama health checks (connection validation)
- **TR2.3**: External dependency: Ollama must be installed and running
- **TR2.4**: ImportError with helpful message if `openai` or `httpx` packages not installed

### TR3: Connection Handling

- **TR3.1**: Graceful error if Ollama server not running
- **TR3.2**: Support for custom port/host configuration
- **TR3.3**: Timeout configuration for slow models

## Success Criteria

- ✅ Users can select Ollama provider for speaker detection and summarization via unified provider
- ✅ Clear error when Ollama server is not running (with helpful instructions)
- ✅ Clear error when model is not installed (with `ollama pull` command)
- ✅ Works completely offline
- ✅ Zero API costs
- ✅ Environment-based model defaults (test vs production)
- ✅ Both Config-based and experiment-based factory modes supported
- ✅ E2E tests pass (with mock or real Ollama)
- ✅ Follows OpenAI provider pattern exactly for consistency

## Provider Capability Matrix (Final)

| Capability | Local | OpenAI | Anthropic | Mistral | DeepSeek | Gemini | Grok | Ollama |
| ---------- | ----- | ------ | --------- | ------- | -------- | ------ | ---- | ------ |
| **Transcription** | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Speaker Detection** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Summarization** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Offline** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Zero Cost** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

## Prerequisites

Before using Ollama provider:

1. Install Ollama: `brew install ollama` (macOS) or see https://ollama.ai
2. Start Ollama: `ollama serve`
3. Pull model: `ollama pull llama3.3`

## Future Considerations

- Whisper support if Ollama adds audio models
- GPU acceleration optimization
- Model recommendation based on available hardware
- Batch processing optimization for local inference
- Support for Ollama running on remote server
