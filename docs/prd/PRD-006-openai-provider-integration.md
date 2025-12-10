# PRD-006: OpenAI Provider Integration

## Summary

Add OpenAI API as an optional provider for speaker detection (NER), transcription, and summarization capabilities, enabling users to choose between local on-device processing and cloud-based API services. This builds on the modularization refactoring (MODULARIZATION_REFACTORING_PLAN.md) to provide seamless provider switching without changing end-user experience or workflow behavior.

## Background & Context

Currently, the podcast scraper uses local on-device AI/ML models:

- **Speaker Detection**: spaCy NER models running locally
- **Transcription**: OpenAI Whisper models running locally
- **Summarization**: Hugging Face transformer models (BART, LED) running locally

While local processing provides privacy and predictable costs, some users may prefer:

- **Higher accuracy**: OpenAI's models may provide better results for speaker detection and summarization
- **Reduced resource usage**: Offload processing to cloud, reducing local CPU/GPU/memory requirements
- **Faster processing**: API-based transcription can be faster than local Whisper for some users
- **Better quality**: OpenAI's GPT models may produce higher-quality summaries than local transformers

This PRD addresses the need to add OpenAI as a provider option while maintaining backward compatibility and zero changes to end-user experience when using default (local) providers.

## Goals

- Add OpenAI API as provider option for speaker detection, transcription, and summarization
- Maintain 100% backward compatibility - no changes to existing behavior with default (local) providers
- Provide secure API key management via environment variables (never in source code)
- Enable per-capability provider selection (can mix local and OpenAI providers)
- Maintain existing parallelism and performance characteristics where applicable
- Support both development and production environments for API key storage

## Non-Goals

- Changing default behavior (local providers remain default)
- Supporting other cloud providers in this PRD (OpenAI only)
- Changing end-user workflow or CLI interface
- Modifying existing local provider implementations
- Adding new features beyond provider selection
- API key management UI or interactive configuration

## Personas

- **Quality Seeker Quinn**: Wants highest-quality summaries and speaker detection, willing to pay for OpenAI API
- **Resource-Constrained Rachel**: Has limited local compute resources, prefers cloud processing
- **Hybrid User Henry**: Wants local transcription (privacy) but OpenAI summarization (quality)
- **Developer Devin**: Needs to test with OpenAI API during development
- **Privacy-First Pat**: Continues using local providers exclusively (no changes to their workflow)

## User Stories

- *As Quality Seeker Quinn, I can configure OpenAI API for summarization to get higher-quality summaries without changing my workflow.*
- *As Resource-Constrained Rachel, I can use OpenAI API for all capabilities to reduce local resource usage.*
- *As Hybrid User Henry, I can use local transcription but OpenAI summarization by configuring providers separately.*
- *As Developer Devin, I can set my OpenAI API key in environment variables and test API integration.*
- *As Privacy-First Pat, I can continue using the software exactly as before with no changes (local providers remain default).*
- *As any operator, I can see which provider was used for each capability in logs and metadata.*
- *As any operator, I can switch providers via configuration without code changes.*

## Functional Requirements

### FR1: Provider Selection

- **FR1.1**: Add `speaker_detector_type` config field with values `"ner"` (default), `"openai"`
- **FR1.2**: Add `transcription_provider` config field with values `"whisper"` (default), `"openai"`
- **FR1.3**: Add `summary_provider` config field with values `"local"` (default), `"openai"`
- **FR1.4**: Provider selection is independent per capability (can mix providers)
- **FR1.5**: Default values maintain current behavior (local providers)
- **FR1.6**: Invalid provider values result in clear error messages

### FR2: API Key Management

- **FR2.1**: Support `OPENAI_API_KEY` environment variable for API authentication
- **FR2.2**: API key is never stored in source code, config files, or committed files
- **FR2.3**: Missing API key when OpenAI provider is selected results in clear error message
- **FR2.4**: API key validation occurs at provider initialization (fail fast)
- **FR2.5**: Support for development and production environments (same mechanism)
- **FR2.6**: Future-proof design for additional API keys (e.g., `ANTHROPIC_API_KEY`)

### FR3: Speaker Detection with OpenAI

- **FR3.1**: OpenAI provider uses GPT-4 or GPT-3.5-turbo for entity extraction
- **FR3.2**: Maintains same interface as NER provider (detect_hosts, detect_speakers, analyze_patterns)
- **FR3.3**: Returns results in same format as NER provider (no workflow changes)
- **FR3.4**: Handles API rate limits gracefully (retry with backoff)
- **FR3.5**: Logs provider type used in detection logs

### FR4: Transcription with OpenAI

- **FR4.1**: OpenAI provider uses Whisper API for transcription
- **FR4.2**: Maintains same interface as local Whisper provider (transcribe method)
- **FR4.3**: Returns results in same format (text, segments) as local provider
- **FR4.4**: Supports same configuration options (language, model selection)
- **FR4.5**: Handles file uploads and API responses correctly
- **FR4.6**: Maintains parallelism where applicable (API calls can be parallelized)

### FR5: Summarization with OpenAI

- **FR5.1**: OpenAI provider uses GPT-4 or GPT-3.5-turbo for summarization
- **FR5.2**: Maintains same interface as local provider (summarize, summarize_chunks, combine_summaries)
- **FR5.3**: Supports MAP/REDUCE pattern (chunk summarization + final combine)
- **FR5.4**: Returns results in same format as local provider
- **FR5.5**: Handles long transcripts via chunking (same as local provider)
- **FR5.6**: Maintains parallelism for chunk processing where applicable

### FR6: Logging and Observability

- **FR6.1**: Log which provider is used for each capability at initialization
- **FR6.2**: Log provider type in episode processing logs
- **FR6.3**: Include provider information in metadata documents
- **FR6.4**: Log API usage (calls, errors, rate limits) for debugging
- **FR6.5**: No sensitive information (API keys) in logs

### FR7: Error Handling

- **FR7.1**: API errors result in clear error messages with provider context
- **FR7.2**: Rate limit errors include retry information
- **FR7.3**: Network errors are handled gracefully with retries
- **FR7.4**: Invalid API key errors are clear and actionable
- **FR7.5**: Fallback behavior (if any) is documented and predictable

### FR8: Performance and Parallelism

- **FR8.1**: OpenAI API calls can be parallelized for batch processing
- **FR8.2**: Rate limiting is respected (no overwhelming API)
- **FR8.3**: Parallelism works correctly with API providers (no blocking)
- **FR8.4**: Performance characteristics are documented (API vs local)

## Technical Requirements

### TR1: Architecture

- **TR1.1**: Build on modularization refactoring (provider abstraction already in place)
- **TR1.2**: Implement OpenAI providers following existing protocol interfaces
- **TR1.3**: No changes to workflow.py logic (uses factory pattern)
- **TR1.4**: Provider selection via factory pattern (already designed)

### TR2: Dependencies

- **TR2.1**: Add `openai` Python package as optional dependency
- **TR2.2**: OpenAI dependency only required when OpenAI provider is used
- **TR2.3**: Lazy import OpenAI client (only when provider is selected)
- **TR2.4**: Version pinning for OpenAI package

### TR3: Configuration

- **TR3.1**: Add provider type fields to config.py (already planned in refactoring)
- **TR3.2**: Support environment variable for API key
- **TR3.3**: Config validation ensures provider + API key consistency
- **TR3.4**: Backward compatible defaults (local providers)

### TR4: Testing

- **TR4.1**: Unit tests for OpenAI providers (with mocked API)
- **TR4.2**: Integration tests with real API (optional, requires API key)
- **TR4.3**: Tests verify same interface as local providers
- **TR4.4**: Tests verify error handling and rate limiting
- **TR4.5**: Backward compatibility tests (default behavior unchanged)

## Success Criteria

- ✅ Users can select OpenAI provider for any capability via configuration
- ✅ Default behavior (local providers) remains unchanged
- ✅ API keys are managed securely via environment variables
- ✅ OpenAI providers implement same interfaces as local providers
- ✅ No changes required to workflow.py or end-user code
- ✅ Parallelism works correctly with API providers
- ✅ Error handling is clear and actionable
- ✅ Documentation explains provider selection and API key setup

## Out of Scope

- Other cloud providers (Anthropic, AWS, etc.) - future PRDs

- API key management UI or interactive setup
- Cost tracking or usage monitoring
- Provider fallback chains (use provider A, fallback to provider B)
- Real-time provider switching during execution

## Dependencies

- **Prerequisite**: Modularization refactoring must be completed (MODULARIZATION_REFACTORING_PLAN.md)
- **External**: OpenAI API access and API key
- **Internal**: Provider abstraction interfaces (from refactoring)

## Risks & Mitigations

- **Risk**: API costs can be high for large batches
  - **Mitigation**: Document costs, make local providers default, provide cost estimates
- **Risk**: API rate limits may slow processing
  - **Mitigation**: Implement retry logic with backoff, respect rate limits, document limits
- **Risk**: API availability issues
  - **Mitigation**: Clear error messages, retry logic, fallback documentation
- **Risk**: Breaking changes if provider interfaces change
  - **Mitigation**: Protocol-based interfaces, comprehensive tests, backward compatibility

## Future Considerations

- Support for other providers (Anthropic Claude, AWS services)
- Provider fallback chains (try OpenAI, fallback to local)
- Cost tracking and usage monitoring
- Provider performance comparison tools
- Hybrid processing (use API for some episodes, local for others)
