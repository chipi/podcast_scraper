# RFC-012: Episode Summarization Using Local Transformers

- **Status**: Completed
- **Authors**:
- **Stakeholders**: Maintainers, users generating episode summaries, developers integrating summarization
- **Related PRDs**: `docs/prd/PRD-005-episode-summarization.md` (to be created), `docs/prd/PRD-004-metadata-generation.md`
- **Related Issues**: Issue #17, Issue #30

## Abstract

Design and implement episode summarization feature that generates concise summaries and key takeaways from episode transcripts using local PyTorch transformer models. This enables privacy-preserving, cost-effective summarization without requiring external API services, while maintaining integration with the existing metadata generation pipeline.

## Problem Statement

Issue #17 describes the need to generate concise summaries and key takeaways from episode transcripts. While API-based LLM services (OpenAI GPT) provide high-quality results, they have drawbacks:

- **Privacy concerns**: Transcripts sent to external APIs may contain sensitive content
- **Cost**: API usage incurs per-token costs that scale with transcript length
- **Rate limits**: API providers enforce rate limits that can slow batch processing
- **Dependency**: Requires internet connectivity and API key management
- **Latency**: Network round-trips add latency to processing pipeline

Local transformer models address these concerns by running entirely on-device, providing privacy, predictable costs (hardware), and no rate limits. However, they require:

- GPU memory or sufficient CPU resources
- Model download and caching
- Careful prompt engineering for quality results
- Memory management for long transcripts

## Constraints & Assumptions

- Summarization must be opt-in (default `false`) for backwards compatibility
- **Hardware Constraint**: Must run on Apple M4 Pro with 48 GB RAM (primary development/testing platform)
  - Models must be selected and optimized to work within this memory constraint
  - Apple Silicon uses Metal Performance Shaders (MPS) backend for PyTorch, not CUDA
  - While 48 GB RAM is generous, memory efficiency is still important for concurrent operations
  - Model selection should prioritize models that fit comfortably in available memory (e.g., bart-base ~500MB, distilbart ~300MB)
  - Must support MPS backend for GPU acceleration on Apple Silicon
- Local models are preferred over API-based solutions for privacy and cost reasons
- Summaries are stored in metadata documents (PRD-004/RFC-011 structure)
- Transcripts can be long (5000-20000+ words); models must handle long inputs efficiently
- Users may have limited GPU memory; CPU fallback must be supported
- Model downloads should be cached and reusable across runs
- Summarization should not block transcript processing; can be async or sequential
- Quality should be reasonable but may be lower than premium API services

## Design & Implementation

### 1. Configuration

Add new configuration fields to `config.Config`:

````python
generate_summaries: bool = False  # Opt-in for backwards compatibility
summary_provider: Literal["local", "openai"] = "local"  # Default to local
summary_model: Optional[str] = None  # Model identifier (e.g., "facebook/bart-large-cnn")
summary_max_length: int = 150  # Max tokens for summary
summary_min_length: int = 30  # Min tokens for summary
summary_max_takeaways: int = 10  # Maximum number of key takeaways
summary_device: Optional[str] = None  # "cuda", "mps", "cpu", or None for auto-detection
summary_batch_size: int = 1  # Batch size for processing (1 = sequential)
summary_chunk_size: Optional[int] = None  # Chunk size for long transcripts (None = no chunking)
summary_cache_dir: Optional[str] = None  # Custom cache directory for models
```yaml

- `--generate-summaries`: Enable summary generation
- `--summary-provider`: Choose provider (`local`, `openai`)
- `--summary-model`: Model identifier (e.g., `facebook/bart-large-cnn`)
- `--summary-max-length`: Maximum summary length in tokens
- `--summary-max-takeaways`: Maximum number of key takeaways
- `--summary-device`: Force device (`cuda`, `mps`, `cpu`, or `auto` for auto-detection)
- `--summary-chunk-size`: Chunk size for long transcripts (default: model max length)

### 2. Local Transformer Integration

#### 2.1 Dependency Management

Add dependencies to `pyproject.toml` in the `[project.optional-dependencies.ml]` section:

```toml

# pyproject.toml [project.optional-dependencies.ml]

"torch>=2.0.0,<3.0.0",  # PyTorch core
"transformers>=4.30.0,<5.0.0",  # Hugging Face Transformers library
"sentencepiece>=0.1.99,<1.0.0",  # Tokenizer dependency for some models
"accelerate>=0.20.0,<1.0.0",  # Optional: for model loading optimizations
```text

- CUDA-enabled PyTorch (installed separately based on CUDA version)
- `bitsandbytes` (for 8-bit quantization to reduce memory usage)

## 2.2 Model Selection

Recommended models for summarization:

**BART-based models** (best for abstractive summarization):

- `facebook/bart-large-cnn`: High quality, ~560M parameters, requires ~2GB GPU memory
- `facebook/bart-large-xsum`: Optimized for extreme summarization, ~560M parameters
- `facebook/bart-base`: Smaller, faster, ~140M parameters, ~500MB GPU memory

**T5-based models** (good for extractive/abstractive hybrid):

- `google/flan-t5-large`: Instruction-tuned, good for structured outputs, ~780M parameters
- `google/flan-t5-base`: Smaller alternative, ~250M parameters

**DistilBART** (lightweight option):

- `sshleifer/distilbart-cnn-12-6`: Smaller BART variant, ~260M parameters, faster inference

**Default selection logic**:

```python
DEFAULT_SUMMARY_MODELS = {
    "bart-large": "facebook/bart-large-cnn",  # BART-large (best quality ~2GB memory)
    "bart-small": "facebook/bart-base",  # BART-base (smallest, lowest memory ~500MB, recommended for M4 Pro)
    "fast": "sshleifer/distilbart-cnn-12-6",  # DistilBART (faster, lower memory ~300MB)
}

def select_summary_model(cfg: Config) -> str:
    """Select summary model based on configuration and available resources.

    Optimized for Apple M4 Pro with 48GB RAM:
    - Prefers bart-base or distilbart for memory efficiency
    - Supports MPS backend for GPU acceleration on Apple Silicon
    """

    if cfg.summary_model:
        return cfg.summary_model

    # Auto-select based on available resources

```text

    # For Apple Silicon (M4 Pro), prefer memory-efficient models

```python

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    Pipeline,
)

logger = logging.getLogger(__name__)

# Hugging Face cache directory (standard location)

HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "transformers"

class SummaryModel:

```text

    """Wrapper for local transformer summarization model."""

```python

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """Initialize summary model.

```python

    def _detect_device(self, device: Optional[str]) -> str:
        """Detect and return appropriate device.

```python

    def _load_model(self) -> None:
        """Load model and tokenizer from cache or download."""
        try:
            logger.info(f"Loading summarization model: {self.model_name} on {self.device}")

```python

            # - "cuda" -> 0 (first CUDA device)

```python

            # - "mps" -> "mps" (Apple Silicon)

```python

            # - "cpu" -> -1 (CPU)

```python

    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
        do_sample: bool = False,
    ) -> str:
        """Generate summary of input text.

```python

    def generate_takeaways(
        self,
        text: str,
        max_takeaways: int = 10,
        max_length_per_takeaway: int = 100,
    ) -> List[str]:
        """Generate key takeaways from text.

```javascript

        # Strategy: Generate longer summary, then split into bullet points

```

## Strategy 1: Chunking with Sliding Window

```python

def chunk_text_for_summarization(
    text: str,
    chunk_size: int,
    overlap: int = 200,
) -> List[str]:
    """Split long text into overlapping chunks.

    Args:
        text: Input text
        chunk_size: Target chunk size in tokens
        overlap: Overlap between chunks in tokens

    Returns:
        List of text chunks
    """

    # Tokenize to get accurate token counts

```text

    tokens = tokenizer.encode(text, add_special_tokens=False)

```python

def summarize_long_text(
    model: SummaryModel,
    text: str,
    chunk_size: int = 1024,
    max_length: int = 150,
) -> str:

```text

    """Summarize long text by chunking and combining summaries.

```python

def hierarchical_summarize(
    model: SummaryModel,
    text: str,
    levels: int = 2,
    max_length: int = 150,
) -> str:
    """Hierarchical summarization: summarize sections, then summarize summaries.

    Args:
        model: Summary model instance
        text: Input text
        levels: Number of summarization levels
        max_length: Final summary length

    Returns:
        Final summary
    """

```text

    # Split into paragraphs or sections

```python

    model: SummaryModel,
    text: str,
    max_length: int = 150,
) -> str:
    """Extract key sentences first, then summarize extracted content.

    Args:
        model: Summary model instance
        text: Input text
        max_length: Summary length

    Returns:
        Summary
    """

    # Simple extraction: take first and last sentences, plus middle sentences

    sentences = [s.strip() for s in text.split(". ") if s.strip()]

```text

    if len(sentences) <= 5:
        return model.summarize(text, max_length=max_length)

```python

def optimize_model_memory(model: SummaryModel) -> None:
    """Optimize model for memory efficiency.

    Supports both CUDA (NVIDIA) and MPS (Apple Silicon) backends.
    """
    if model.device == "cuda":

        # Enable gradient checkpointing (trades compute for memory)

        if hasattr(model.model, "gradient_checkpointing_enable"):
            model.model.gradient_checkpointing_enable()

        # Use half precision (FP16) to reduce memory

        model.model = model.model.half()

        # Clear cache

```text

        torch.cuda.empty_cache()
    elif model.device == "mps":

```python

        # MPS may benefit from FP16, but test performance impact

```python

def optimize_for_cpu(model: SummaryModel) -> None:
    """Optimize model for CPU inference."""

    # Use INT8 quantization if available

    try:
        from transformers import BitsAndBytesConfig

        # Note: INT8 quantization typically requires GPU

        # For CPU, use model optimization techniques

        pass
    except ImportError:
        pass

    # Set number of threads for CPU

```text

    torch.set_num_threads(os.cpu_count() or 4)

```

    model.tokenizer = None
    model.pipeline = None

    # Clear device-specific cache

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if hasattr(torch.mps, "empty_cache"):

```text

            torch.mps.empty_cache()

```python

def generate_takeaways_with_prompt(
    model: SummaryModel,
    text: str,
    max_takeaways: int = 10,
) -> List[str]:
    """Generate takeaways using instruction prompt.

    Args:
        model: Summary model (preferably instruction-tuned like flan-t5)
        text: Input text
        max_takeaways: Maximum takeaways

    Returns:
        List of takeaways
    """

    # Construct prompt for instruction-tuned models

    prompt = f"""Summarize the following text and extract {max_takeaways} key takeaways.

Text:
{text}

Key Takeaways:
"""

```text

    # For instruction-tuned models (e.g., flan-t5)

```python

        # Parse takeaways from result

```python

# In metadata.py or summarizer.py

class SummaryMetadata(BaseModel):
    """Summary metadata structure."""
    short_summary: str
    key_takeaways: List[str]
    generated_at: datetime
    model_used: str
    provider: str  # "local", "openai"
    word_count: int

    @field_serializer('generated_at')
    def serialize_generated_at(self, value: datetime) -> str:
        return value.isoformat()

# Integration in episode_processor.py or workflow.py

def generate_episode_summary(
    transcript_path: Path,
    cfg: Config,
    summary_model: Optional[SummaryModel] = None,
) -> Optional[SummaryMetadata]:

```text

    """Generate summary for episode transcript.

```
        model_used=summary_model.model_name,
        provider="local",
        word_count=len(transcript.split()),
    )

```python

def safe_summarize(
    model: SummaryModel,
    text: str,
    max_length: int = 150,
) -> str:
    """Safely generate summary with error handling.

    Returns:
        Summary text, or empty string on failure
    """
    try:
        return model.summarize(text, max_length=max_length)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:

        # Handle both CUDA and MPS out-of-memory errors

        if "out of memory" in str(e).lower() or "mps" in str(e).lower():
            logger.error(f"Device out of memory during summarization ({model.device}): {e}")

```text

            # Fallback: use CPU or smaller model

```python

- Subsequent runs load from cache (fast)

**Batch Processing**:

- Process multiple episodes sequentially (batch_size=1) to avoid memory issues
- Can parallelize across episodes if GPU memory allows

**Lazy Loading**:

- Load model only when `generate_summaries=True`
- Unload model after processing to free memory

## Testing Strategy

- Unit tests for model loading and summarization
- Integration tests with sample transcripts
- Memory tests for long transcripts
- Error handling tests (missing model, OOM, etc.)
- Performance benchmarks (tokens/second, memory usage)

## Alternatives Considered

### API-Based Solutions (OpenAI)

- **Pros**: Higher quality, no local resources needed
- **Cons**: Privacy concerns, cost, rate limits, internet required
- **Decision**: Support as optional provider, but default to local

### Smaller Models (DistilBART, T5-small)

- **Pros**: Lower memory, faster inference
- **Cons**: Lower quality summaries
- **Decision**: Provide as options, auto-select based on resources

### Quantization (8-bit, 4-bit)

- **Pros**: Significant memory reduction
- **Cons**: Requires `bitsandbytes`, slight quality loss
- **Decision**: Document as advanced option, not default

## Rollout Plan

1. Create RFC-012 document (this document)
2. Implement `summarizer.py` module with local transformer support
3. Integrate with metadata generation pipeline
4. Add configuration options
5. Add tests
6. Update documentation
7. Release as opt-in feature
8. Collect user feedback on quality and performance

## Open Questions

- Should we support multiple local models with quality/speed tradeoffs? (Decision: Yes, with auto-selection based on hardware)
- How to handle very long transcripts (>20k words)? Chunking strategy? (Decision: Chunking with sliding window)
- Should summaries be cached to avoid regeneration? (Decision: No, regenerate on each run, use `--skip-existing` to prevent overwrites)
- Do we need GPU detection and automatic model selection? (Decision: Yes, with MPS support for Apple Silicon)
- Should we support model fine-tuning for podcast-specific summarization? (Future consideration)
- **Apple Silicon Optimization**: What's the best model size/configuration for M4 Pro 48GB? (Decision: bart-base recommended, distilbart for speed)

## References

- Issue #17: Generate short summary and key takeaways from each episode
- Issue #30: Create PRD and RFC for episode summary generation feature
- PRD-004: Per-Episode Metadata Document Generation
- RFC-011: Per-Episode Metadata Document Generation
- Hugging Face Transformers: <https://huggingface.co/docs/transformers/>
- BART Paper: <https://arxiv.org/abs/1910.13461>
- T5 Paper: <https://arxiv.org/abs/1910.10683>

````
