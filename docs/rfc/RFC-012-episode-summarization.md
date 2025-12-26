# RFC-012: Episode Summarization Using Local Transformers

- **Status**: Completed
- **Authors**:
- **Stakeholders**: Maintainers, users generating episode summaries, developers integrating summarization
- **Related PRDs**: `docs/prd/PRD-005-episode-summarization.md` (to be created), `docs/prd/PRD-004-metadata-generation.md`
- **Related Issues**: Issue #17, Issue #30

## Abstract

Design and implement episode summarization feature that generates concise summaries and key takeaways from episode transcripts using local PyTorch transformer models. This enables privacy-preserving, cost-effective summarization without requiring external API services, while maintaining integration with the existing metadata generation pipeline.

## Problem Statement

Issue #17 describes the need to generate concise summaries and key takeaways from episode transcripts. While API-based LLM services (OpenAI GPT, Anthropic Claude) provide high-quality results, they have drawbacks:

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

```python
generate_summaries: bool = False  # Opt-in for backwards compatibility
summary_provider: Literal["local", "openai", "anthropic"] = "local"  # Default to local
summary_model: Optional[str] = None  # Model identifier (e.g., "facebook/bart-large-cnn")
summary_max_length: int = 150  # Max tokens for summary
summary_min_length: int = 30  # Min tokens for summary
summary_max_takeaways: int = 10  # Maximum number of key takeaways
summary_device: Optional[str] = None  # "cuda", "mps", "cpu", or None for auto-detection
summary_batch_size: int = 1  # Batch size for processing (1 = sequential)
summary_chunk_size: Optional[int] = None  # Chunk size for long transcripts (None = no chunking)
summary_cache_dir: Optional[str] = None  # Custom cache directory for models
```

Add CLI flags:

- `--generate-summaries`: Enable summary generation
- `--summary-provider`: Choose provider (`local`, `openai`, `anthropic`)
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
```

**Optional dependencies** (for GPU support):

- CUDA-enabled PyTorch (installed separately based on CUDA version)
- `bitsandbytes` (for 8-bit quantization to reduce memory usage)

#### 2.2 Model Selection

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
    # For Apple Silicon (M4 Pro), prefer memory-efficient models
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # M4 Pro: Use LED-base for long documents (16k tokens, no chunking needed)
        return DEFAULT_SUMMARY_MODELS["long-fast"]
    elif torch.cuda.is_available():
        # NVIDIA GPU: Use LED-large for best quality (16k tokens, no chunking needed)
        return DEFAULT_SUMMARY_MODELS["long"]
    else:
        # CPU: Use fastest, lowest memory model
        return DEFAULT_SUMMARY_MODELS["fast"]
```

#### 2.3 Model Loading and Caching

Create `podcast_scraper/summarizer.py` module:

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
    """Wrapper for local transformer summarization model."""
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """Initialize summary model.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to use ("cuda", "cpu", or None for auto-detection)
            cache_dir: Custom cache directory for model files
        """
        self.model_name = model_name
        self.device = self._detect_device(device)
        self.cache_dir = cache_dir or str(HF_CACHE_DIR)
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSeq2SeqLM] = None
        self.pipeline: Optional[Pipeline] = None
        self._load_model()
    
    def _detect_device(self, device: Optional[str]) -> str:
        """Detect and return appropriate device.
        
        Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU fallback.
        """
        if device:
            return device
        
        # Check for Apple Silicon MPS backend first (M4 Pro)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        
        # Check for CUDA (NVIDIA GPUs)
        if torch.cuda.is_available():
            return "cuda"
        
        return "cpu"
    
    def _load_model(self) -> None:
        """Load model and tokenizer from cache or download."""
        try:
            logger.info(f"Loading summarization model: {self.model_name} on {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )
            
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Create pipeline for easy inference
            # Map device to pipeline device parameter:
            # - "cuda" -> 0 (first CUDA device)
            # - "mps" -> "mps" (Apple Silicon)
            # - "cpu" -> -1 (CPU)
            pipeline_device = (
                0 if self.device == "cuda"
                else "mps" if self.device == "mps"
                else -1
            )
            self.pipeline = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=pipeline_device,
            )
            
            logger.info(f"Successfully loaded model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load summarization model: {e}")
            raise
    
    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
        do_sample: bool = False,
    ) -> str:
        """Generate summary of input text.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            do_sample: Whether to use sampling (False = deterministic)
        
        Returns:
            Generated summary text
        """
        if not self.pipeline:
            raise RuntimeError("Model not loaded")
        
        # Handle empty or very short text
        if not text or len(text.strip()) < 50:
            return text.strip()
        
        try:
            result = self.pipeline(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                truncation=True,
            )
            
            # Pipeline returns list of dicts with 'summary_text' key
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("summary_text", "").strip()
            elif isinstance(result, dict):
                return result.get("summary_text", "").strip()
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return ""
    
    def generate_takeaways(
        self,
        text: str,
        max_takeaways: int = 10,
        max_length_per_takeaway: int = 100,
    ) -> List[str]:
        """Generate key takeaways from text.
        
        Args:
            text: Input text
            max_takeaways: Maximum number of takeaways
            max_length_per_takeaway: Max length per takeaway
        
        Returns:
            List of key takeaways
        """
        # Strategy: Generate longer summary, then split into bullet points
        # Alternative: Use instruction-tuned model with structured prompt
        
        summary = self.summarize(
            text,
            max_length=max_takeaways * max_length_per_takeaway,
            min_length=max_takeaways * 20,
        )
        
        # Split summary into sentences and extract key points
        # Simple heuristic: split on periods, filter short sentences
        sentences = [s.strip() for s in summary.split(". ") if len(s.strip()) > 20]
        
        # Limit to max_takeaways
        takeaways = sentences[:max_takeaways]
        
        # Clean up: remove trailing periods, ensure proper formatting
        takeaways = [t.rstrip(".") for t in takeaways if t]
        
        return takeaways
```

#### 2.4 Handling Long Transcripts

Transcripts can exceed model context limits. Strategies:

#### Strategy 1: Chunking with Sliding Window

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
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        
        # Move start forward with overlap
        start = end - overlap
    
    return chunks

def summarize_long_text(
    model: SummaryModel,
    text: str,
    chunk_size: int = 1024,
    max_length: int = 150,
) -> str:
    """Summarize long text by chunking and combining summaries.
    
    Args:
        model: Summary model instance
        text: Long input text
        chunk_size: Chunk size in tokens
        max_length: Max summary length per chunk
    
    Returns:
        Combined summary
    """
    # Check if text needs chunking
    tokenizer = model.tokenizer
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) <= chunk_size:
        # Text fits in one chunk
        return model.summarize(text, max_length=max_length)
    
    # Chunk text
    chunks = chunk_text_for_summarization(text, chunk_size, overlap=200)
    
    # Summarize each chunk
    chunk_summaries = []
    for chunk in chunks:
        summary = model.summarize(chunk, max_length=max_length // len(chunks))
        chunk_summaries.append(summary)
    
    # Combine chunk summaries
    combined_text = "\n\n".join(chunk_summaries)
    
    # Final summary of combined summaries
    final_summary = model.summarize(combined_text, max_length=max_length)
    
    return final_summary
```

#### Strategy 2: Hierarchical Summarization

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
    # Split into paragraphs or sections
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    if len(paragraphs) <= 1:
        return model.summarize(text, max_length=max_length)
    
    # Summarize each paragraph
    paragraph_summaries = []
    for para in paragraphs:
        para_summary = model.summarize(para, max_length=max_length // len(paragraphs))
        paragraph_summaries.append(para_summary)
    
    # Combine and summarize again
    combined = "\n\n".join(paragraph_summaries)
    return model.summarize(combined, max_length=max_length)
```

#### Strategy 3: Extract-Then-Summarize

```python
def extract_then_summarize(
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
    
    if len(sentences) <= 5:
        return model.summarize(text, max_length=max_length)
    
    # Extract key sentences
    extracted = []
    extracted.append(sentences[0])  # First sentence
    extracted.append(sentences[-1])  # Last sentence
    
    # Middle sentences (every nth sentence)
    step = max(1, len(sentences) // 5)
    for i in range(step, len(sentences) - step, step):
        extracted.append(sentences[i])
    
    extracted_text = ". ".join(extracted)
    return model.summarize(extracted_text, max_length=max_length)
```

#### 2.5 Memory Management

**GPU Memory Considerations**:

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
        torch.cuda.empty_cache()
    elif model.device == "mps":
        # Apple Silicon MPS backend
        # MPS supports FP16 natively, but may have different optimizations
        # For M4 Pro with 48GB, memory is less constrained, but still optimize
        if hasattr(model.model, "gradient_checkpointing_enable"):
            model.model.gradient_checkpointing_enable()
        
        # MPS may benefit from FP16, but test performance impact
        # model.model = model.model.half()  # Test if needed
        
        # Clear MPS cache if available
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
```

**CPU Memory Considerations**:

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
    torch.set_num_threads(os.cpu_count() or 4)
```

**Model Unloading**:

```python
def unload_model(model: SummaryModel) -> None:
    """Unload model to free memory."""
    if model.model:
        del model.model
    if model.tokenizer:
        del model.tokenizer
    if model.pipeline:
        del model.pipeline
    
    model.model = None
    model.tokenizer = None
    model.pipeline = None
    
    # Clear device-specific cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
```

#### 2.6 Prompt Engineering for Takeaways

For generating structured takeaways, use instruction-tuned models or custom prompts:

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
    
    # For instruction-tuned models (e.g., flan-t5)
    if "flan" in model.model_name.lower() or "instruct" in model.model_name.lower():
        inputs = model.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=model.tokenizer.model_max_length,
        ).to(model.device)
        
        outputs = model.model.generate(
            inputs.input_ids,
            max_length=200,
            min_length=50,
            num_beams=4,
            early_stopping=True,
        )
        
        result = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse takeaways from result
        # Simple heuristic: split on newlines or numbers
        takeaways = [
            line.strip().lstrip("- ").lstrip("• ").strip()
            for line in result.split("\n")
            if line.strip() and len(line.strip()) > 20
        ]
        
        return takeaways[:max_takeaways]
    
    # Fallback: use standard summarization
    return model.generate_takeaways(text, max_takeaways=max_takeaways)
```

### 3. Integration with Metadata Pipeline

Summaries are stored in metadata documents (RFC-011 structure):

```python
# In metadata.py or summarizer.py

class SummaryMetadata(BaseModel):
    """Summary metadata structure."""
    short_summary: str
    key_takeaways: List[str]
    generated_at: datetime
    model_used: str
    provider: str  # "local", "openai", "anthropic"
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
    """Generate summary for episode transcript.
    
    Args:
        transcript_path: Path to transcript file
        cfg: Configuration
        summary_model: Pre-loaded summary model (optional)
    
    Returns:
        Summary metadata or None if generation failed
    """
    if not cfg.generate_summaries:
        return None
    
    # Read transcript
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = f.read()
    except Exception as e:
        logger.error(f"Failed to read transcript: {e}")
        return None
    
    # Load model if not provided
    if not summary_model and cfg.summary_provider == "local":
        summary_model = SummaryModel(
            model_name=select_summary_model(cfg),
            device=cfg.summary_device,
            cache_dir=cfg.summary_cache_dir,
        )
    
    # Generate summary
    short_summary = summary_model.summarize(
        transcript,
        max_length=cfg.summary_max_length,
        min_length=cfg.summary_min_length,
    )
    
    # Generate takeaways
    key_takeaways = summary_model.generate_takeaways(
        transcript,
        max_takeaways=cfg.summary_max_takeaways,
    )
    
    return SummaryMetadata(
        short_summary=short_summary,
        key_takeaways=key_takeaways,
        generated_at=datetime.now(),
        model_used=summary_model.model_name,
        provider="local",
        word_count=len(transcript.split()),
    )
```

### 4. Error Handling

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
            # Fallback: use CPU or smaller model
            return ""
        raise
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return ""
```

### 5. Performance Optimization

**Model Caching**:

- Models are automatically cached by Hugging Face Transformers in `~/.cache/huggingface/transformers/`
- First run downloads model (can be large: 500MB - 2GB)
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

### API-Based Solutions (OpenAI, Anthropic)

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
