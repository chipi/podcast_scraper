# PRD-005: Episode Summarization

## Summary

Generate concise summaries and key takeaways from episode transcripts using local transformer models or optional API-based LLM services. Summaries are stored in episode metadata documents (per PRD-004/RFC-011) to enable quick value discovery without listening to full episodes.

## Background & Context

Podcast episodes can be long (30-90+ minutes), making it time-consuming to determine if an episode contains valuable content. Current transcripts provide full text but don't distill the essential value or insights. Users need:

- **Quick value assessment**: Understand what the episode covers without listening
- **Key insights extraction**: Identify the most important takeaways and concepts discussed
- **Searchability**: Find episodes by topics or insights rather than just titles
- **Time efficiency**: Skip episodes that don't align with interests or needs

This PRD addresses issue #17 by providing automated summarization capabilities that integrate seamlessly with the existing transcript pipeline and metadata generation system.

## Goals

- Generate concise summaries (2-3 sentences) and key takeaways (5-10 bullet points) from episode transcripts
- Store summaries in episode metadata documents for easy access and searchability
- Support local transformer models as the default approach for privacy and cost-effectiveness
- Provide optional API-based LLM support (OpenAI, Anthropic) for users who prefer higher quality
- Integrate with existing pipeline without disrupting current workflows
- Make summarization opt-in to maintain backwards compatibility

## Non-Goals

- Real-time summarization API (summaries generated during batch processing)
- Summary editing or manual refinement (one-way generation)
- Multi-language summarization (initial release focuses on English)
- Fine-tuning custom models (uses pre-trained models)
- Summary versioning or change tracking (regenerate on each run)

## Personas

- **Researcher Riley**: Needs to quickly scan large podcast archives to find episodes relevant to research topics
- **Busy Listener Ben**: Wants to identify which episodes are worth listening to based on summaries
- **Archivist Ava**: Needs searchable summaries for compliance and content discovery
- **Developer Devin**: Building tools that consume episode summaries for recommendations or search
- **Privacy-Conscious Pat**: Prefers local processing to avoid sending transcripts to external APIs

## User Stories

- _As Researcher Riley, I can enable `--generate-summaries` and receive concise summaries for every episode to quickly identify relevant content._
- _As Busy Listener Ben, I can scan key takeaways to decide which episodes to listen to without reading full transcripts._
- _As Archivist Ava, I can search episode metadata by summary content to find episodes covering specific topics._
- _As Developer Devin, I can consume structured summary data from metadata documents to build recommendation systems._
- _As Privacy-Conscious Pat, I can use local transformer models to generate summaries without sending transcripts to external APIs._
- _As any operator, I can choose between local models (privacy, no cost) and API-based models (higher quality) based on my needs._
- _As any operator, I can configure summary length and number of takeaways to match my use case._
- _As any operator, summaries are only generated when transcripts are available (respects existing transcript pipeline)._

## Functional Requirements

### FR1: Summary Generation Control

- **FR1.1**: Add `generate_summaries` config field (default `false` for backwards compatibility)
- **FR1.2**: Add `--generate-summaries` CLI flag
- **FR1.3**: Summary generation respects `--skip-existing` semantics (skip if summary already exists in metadata)
- **FR1.4**: Summary generation respects `--dry-run` mode (logs planned summaries without generating)
- **FR1.5**: Summaries are only generated for episodes with available transcripts (no transcript = no summary)

### FR2: Provider Selection

- **FR2.1**: Support `summary_provider` config field with values `"local"`, `"openai"`, `"anthropic"` (default: `"local"`)
- **FR2.2**: Add `--summary-provider` CLI flag
- **FR2.3**: Local provider uses PyTorch transformers (Hugging Face models) running on-device
- **FR2.4**: API providers require API keys (configurable via environment variables or config files)
- **FR2.5**: Provider selection affects model availability and quality expectations

### FR3: Model Configuration

- **FR3.1**: Support `summary_model` config field for model identifier (e.g., `facebook/bart-large-cnn`)
- **FR3.2**: Add `--summary-model` CLI flag
- **FR3.3**: Auto-select appropriate model based on provider and available resources (GPU/CPU)
- **FR3.4**: Validate model identifiers against supported models for each provider
- **FR3.5**: Document recommended models for different use cases (quality vs. speed vs. memory)

### FR4: Summary Output Format

- **FR4.1**: Generate short summary (2-3 sentences, configurable length)
- **FR4.2**: Generate key takeaways (5-10 bullet points, configurable count)
- **FR4.3**: Store summaries in episode metadata document structure (per PRD-004/RFC-011)
- **FR4.4**: Include summary metadata: generation timestamp, model used, provider, word count
- **FR4.5**: Ensure summaries focus on intrinsic value and insights, not action items or due dates

### FR5: Summary Quality Controls

- **FR5.1**: Support `summary_max_length` config field (default: 150 tokens for short summary)
- **FR5.2**: Support `summary_min_length` config field (default: 30 tokens)
- **FR5.3**: Support `summary_max_takeaways` config field (default: 10)
- **FR5.4**: Add corresponding CLI flags: `--summary-max-length`, `--summary-min-length`, `--summary-max-takeaways`
- **FR5.5**: Validate summary outputs meet minimum quality thresholds (non-empty, reasonable length)

### FR6: Long Transcript Handling

- **FR6.1**: Handle transcripts that exceed model context limits (chunking or hierarchical summarization)
- **FR6.2**: Support `summary_chunk_size` config field for chunking strategy (optional)
- **FR6.3**: Add `--summary-chunk-size` CLI flag
- **FR6.4**: Document chunking strategies and their trade-offs (quality vs. performance)
- **FR6.5**: Ensure chunked summaries maintain coherence and completeness

### FR7: Resource Management

- **FR7.1**: Support `summary_device` config field (`"cuda"`, `"mps"`, `"cpu"`, or `None` for auto-detection)
- **FR7.2**: Add `--summary-device` CLI flag
- **FR7.3**: Auto-detect GPU availability and select appropriate device (MPS for Apple Silicon, CUDA for NVIDIA)
- **FR7.4**: Optimize memory usage for GPU and CPU inference
- **FR7.5**: Support model caching to avoid re-downloading on subsequent runs
- **FR7.6**: Support `summary_cache_dir` config field for custom model cache location

### FR8: Error Handling

- **FR8.1**: Gracefully handle model loading failures (log warning, skip summarization)
- **FR8.2**: Handle out-of-memory errors (fallback to CPU or smaller model)
- **FR8.3**: Handle API failures for cloud providers (log error, skip summary, continue processing)
- **FR8.4**: Ensure summarization failures don't block transcript processing
- **FR8.5**: Provide actionable error messages for common failure modes

### FR9: Integration Points

- **FR9.1**: Generate summaries after transcript is available (during episode processing)
- **FR9.2**: Integrate with PRD-004 metadata generation (store summaries in metadata documents)
- **FR9.3**: Integrate with PRD-001 transcript pipeline (only summarize episodes with transcripts)
- **FR9.4**: Integrate with PRD-002 Whisper fallback (summarize Whisper-generated transcripts)
- **FR9.5**: Respect existing workflow: download transcripts → generate summaries → store in metadata

### FR10: Performance Considerations

- **FR10.1**: Summarization should not significantly slow down transcript processing (<20% overhead)
- **FR10.2**: Support sequential processing to avoid resource contention
- **FR10.3**: Provide progress feedback during summarization (logging or progress bars)
- **FR10.4**: Cache models across episodes to avoid repeated loading
- **FR10.5**: Document expected processing times for different models and transcript lengths

## Success Metrics

- Summaries generated for 100% of episodes with transcripts when feature enabled
- Summary quality: summaries are coherent, informative, and focus on intrinsic value
- Zero impact on existing transcript download/transcription workflows when disabled
- Processing overhead: <20% increase in total processing time when summarization enabled
- Privacy: local models process transcripts entirely on-device (no external API calls)
- Cost: local models have zero per-token costs (only hardware requirements)

## Dependencies

- PRD-004: Per-Episode Metadata Document Generation (where summaries are stored)
- RFC-011: Per-Episode Metadata Document Generation (technical design for metadata)
- RFC-012: Episode Summarization Using Local Transformers (technical implementation)
- PRD-001: Transcript Acquisition Pipeline (transcript availability)
- PRD-002: Whisper Fallback Transcription (Whisper-generated transcripts)

## Constraints & Assumptions

- Summarization must be opt-in (default `false`) for backwards compatibility
- **Hardware Constraint**: Must run on Apple M4 Pro with 48 GB RAM (primary development/testing platform)
  - Models must be selected and optimized to work within this memory constraint
  - Apple Silicon uses Metal Performance Shaders (MPS) backend for PyTorch, not CUDA
  - While 48 GB RAM is generous, memory efficiency is still important for concurrent operations
  - Model selection should prioritize models that fit comfortably in available memory
- Local models are preferred over API-based solutions for privacy and cost reasons
- Summaries are stored in metadata documents (PRD-004/RFC-011 structure)
- Transcripts can be long (5000-20000+ words); models must handle long inputs efficiently
- Users may have limited GPU memory; CPU fallback must be supported
- Model downloads should be cached and reusable across runs
- Summarization should not block transcript processing; can be async or sequential
- Quality should be reasonable but may be lower than premium API services

## Design Considerations

### Provider Selection

- **Local Models (Default)**:
  - **Pros**: Privacy-preserving, no API costs, no rate limits, offline capable
  - **Cons**: Requires GPU/CPU resources, may have lower quality than premium APIs
  - **Decision**: Default to local models for privacy and cost benefits

- **API-Based Models (Optional)**:
  - **Pros**: Higher quality summaries, no local resources needed
  - **Cons**: Privacy concerns, per-token costs, rate limits, internet required
  - **Decision**: Support as optional provider for users who prioritize quality

### Model Selection

- **BART Models** (recommended for local):
  - `facebook/bart-large-cnn`: Best quality, ~2GB memory (compatible with M4 Pro 48GB)
  - `facebook/bart-base`: Smaller, faster, ~500MB memory (recommended for M4 Pro)
  - `sshleifer/distilbart-cnn-12-6`: Fastest, lower memory (~300MB, ideal for M4 Pro)

- **T5 Models** (alternative):
  - `google/flan-t5-large`: Instruction-tuned, good for structured outputs (~3GB memory, may be tight on M4 Pro)
  - `google/flan-t5-base`: Smaller alternative (~1GB memory, compatible with M4 Pro)

- **Auto-Selection**: Automatically select model based on available resources
  - **Apple Silicon (M4 Pro)**: Prefer models optimized for MPS backend (bart-base or distilbart recommended)
  - **Memory-aware**: Select models that fit comfortably within 48GB RAM, leaving headroom for other operations
  - **Fallback**: If GPU/MPS unavailable, fall back to CPU-optimized models

### Quality vs. Performance Trade-offs

- **High Quality**: Use larger models (bart-large-cnn) with longer max_length
- **Fast Processing**: Use smaller models (distilbart) with shorter max_length
- **Memory Constrained**: Use CPU-optimized models or quantization
- **Decision**: Provide defaults that balance quality and performance, allow user override

### Long Transcript Handling

- **Chunking Strategy**: Split long transcripts into overlapping chunks, summarize each, then combine
- **Hierarchical Strategy**: Summarize paragraphs/sections, then summarize summaries
- **Extract-Then-Summarize**: Extract key sentences first, then summarize extracted content
- **Decision**: Support chunking as primary strategy, document alternatives

### Cost Considerations

- **Local Models**: Zero per-token costs, but requires hardware (GPU recommended)
- **API Models**: Per-token costs scale with transcript length, rate limits apply
- **Decision**: Default to local models to eliminate ongoing costs

## Integration with Metadata Pipeline

Summaries are stored in episode metadata documents following PRD-004/RFC-011 structure:

````json
{
  "episode": {
    "title": "Episode Title",
    "summary": {
      "short_summary": "2-3 sentence overview...",
      "key_takeaways": [
        "Key insight 1",
        "Key insight 2",
        "..."
      ],
      "generated_at": "2024-01-15T10:30:00Z",
      "model_used": "facebook/bart-large-cnn",
      "provider": "local",
      "word_count": 150
    }
  }
}
```text

**Output** (stored in metadata document):

```yaml
summary:
  short_summary: |
    This episode explores the future of AI in healthcare, featuring Dr. Jane Smith
    discussing how machine learning is transforming diagnostic accuracy. The conversation
    covers both the opportunities and ethical challenges of AI-assisted medicine.
  key_takeaways:

    - AI can improve diagnostic accuracy by up to 30% in radiology applications
    - Patient privacy remains the biggest challenge for healthcare AI adoption
    - Regulatory frameworks are struggling to keep pace with AI innovation
    - Hybrid human-AI workflows show the most promise for clinical adoption
    - Cost barriers prevent smaller hospitals from accessing advanced AI tools
  generated_at: "2024-01-15T10:30:00Z"

  model_used: "facebook/bart-large-cnn"
  provider: "local"
  word_count: 150
```text

- Should summaries be regenerated if transcript is updated? (Decision: Yes, regenerate on each run, use `--skip-existing` to prevent overwrites)
- Do we need summary versioning if prompts change? (Decision: No, summaries are regenerated on each run)
- Should we support summary quality metrics or validation? (Future consideration)
- How to handle multi-speaker transcripts in summaries? (Focus on content, not speaker attribution)
- Should summaries include episode metadata context (guest names, topics)? (Future consideration)

## Related Work

- Issue #17: Generate short summary and key takeaways from each episode based on transcript
- Issue #30: Create PRD and RFC for episode summary generation feature
- PRD-004: Per-Episode Metadata Document Generation (where summaries are stored)
- RFC-011: Per-Episode Metadata Document Generation (metadata technical design)
- RFC-012: Episode Summarization Using Local Transformers (summarization technical design)
- PRD-001: Transcript Acquisition Pipeline (transcript availability)
- PRD-002: Whisper Fallback Transcription (Whisper-generated transcripts)

## Release Checklist

- [ ] PRD reviewed and approved
- [ ] RFC-012 created with technical design (✅ Complete)
- [ ] Implementation completed
- [ ] Tests cover summary generation, error handling, long transcripts
- [ ] Documentation updated (README, config examples)
- [ ] Model selection and performance documented
- [ ] Integration with metadata pipeline verified
````
