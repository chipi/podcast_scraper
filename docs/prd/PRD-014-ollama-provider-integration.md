# PRD-014: Ollama Provider Integration

- **Status**: Draft
- **Related RFCs**: RFC-037
- **Related PRDs**: PRD-006 (OpenAI), PRD-013 (Groq)

## Summary

Add Ollama as an optional provider for speaker detection and summarization capabilities. Ollama is unique in being a **fully local/offline solution** that runs open-source LLMs on your own hardware. Unlike cloud providers, Ollama requires NO API keys, has NO rate limits, and incurs NO per-token costs. Like Anthropic, DeepSeek, and Groq, Ollama does NOT support audio transcription.

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
- Follow identical architectural patterns
- Support local model selection (user installs models via Ollama CLI)
- Handle capability gaps gracefully (no transcription)
- Use OpenAI SDK with `http://localhost:11434/v1` base_url

## Ollama Model Selection and Cost Analysis

### Configuration Fields

```python

# Ollama Model Selection

ollama_speaker_model: str = Field(
    default="llama3.3:latest",
    description="Ollama model for speaker detection"
)

ollama_summary_model: str = Field(
    default="llama3.3:latest",
    description="Ollama model for summarization"
)

# Ollama API Configuration

ollama_api_base: str = Field(
    default="http://localhost:11434/v1",
    description="Ollama API base URL (default: local)"
)

ollama_temperature: float = Field(
    default=0.3,
    ge=0.0,
    le=2.0,
    description="Temperature for Ollama generation"
)

ollama_max_tokens: Optional[int] = Field(
    default=4096,
    description="Max tokens for Ollama generation"
)
```yaml

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

| Component | OpenAI (gpt-4o-mini) | DeepSeek | Groq | **Ollama** |
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
| Groq (Llama 3.3) | ~0.5 seconds |
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

### FR2: No API Key Required

- **FR2.1**: Ollama does not require API key (local service)
- **FR2.2**: Clear error if Ollama server is not running
- **FR2.3**: Support custom `ollama_api_base` for remote Ollama servers

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

- **TR1.1**: Follow OpenAI SDK pattern (same as DeepSeek, Groq)
- **TR1.2**: Create `podcast_scraper/ollama/` package
- **TR1.3**: Create `speaker_detectors/ollama_detector.py`
- **TR1.4**: Create `summarization/ollama_provider.py`
- **TR1.5**: Update factories

### TR2: Dependencies

- **TR2.1**: Reuse existing `openai` package
- **TR2.2**: No new SDK dependency
- **TR2.3**: External dependency: Ollama must be installed and running

### TR3: Connection Handling

- **TR3.1**: Graceful error if Ollama server not running
- **TR3.2**: Support for custom port/host configuration
- **TR3.3**: Timeout configuration for slow models

## Success Criteria

- ✅ Users can select Ollama for speaker detection and summarization
- ✅ Clear error when Ollama not running
- ✅ Clear error when model not installed
- ✅ Works completely offline
- ✅ Zero API costs
- ✅ E2E tests pass (with mock or real Ollama)

## Provider Capability Matrix (Final)

| Capability | Local | OpenAI | Anthropic | Mistral | DeepSeek | Gemini | Groq | Ollama |
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
