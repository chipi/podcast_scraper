# Summarization Guide

This guide provides detailed implementation information for the podcast scraper's summarization system. The summarization system implements a sophisticated hybrid MAP-REDUCE architecture to handle transcripts of any length efficiently.

For high-level architectural decisions, see [Architecture](ARCHITECTURE.md). For general development practices, see [Development Guide](DEVELOPMENT_GUIDE.md).

## Overview

The summarization feature generates concise summaries of podcast episode transcripts using local transformer models (BART, PEGASUS, LED). It implements a hybrid map-reduce strategy that efficiently handles transcripts of any length, from short episodes to very long multi-hour conversations.

**Key Features:**
- Local-first approach for privacy and cost-effectiveness
- Automatic hardware detection (MPS/CUDA/CPU)
- Hybrid MAP-REDUCE architecture for long transcripts
- Quality validation and error handling
- Automatic model caching and reuse

## Summarization Flow

```mermaid
flowchart TD
    Start([Transcript Available]) --> CheckEnabled{Summarization<br/>Enabled?}
    CheckEnabled -->|No| Skip[Skip Summarization]
    CheckEnabled -->|Yes| CleanTranscript[Clean Transcript<br/>Remove timestamps, sponsors, fillers]
    CleanTranscript --> SaveCleaned{Save Cleaned<br/>Transcript?}
    SaveCleaned -->|Yes| WriteCleaned[Write .cleaned.txt File]
    SaveCleaned -->|No| CheckSize
    WriteCleaned --> CheckSize{Text Size<br/>Check}
    CheckSize -->|Fits in Context| DirectSummarize[Direct Summarization<br/>Single Pass]
    CheckSize -->|Too Long| StartMapReduce[Begin MAP-REDUCE Pipeline]
    
    StartMapReduce --> ChunkText[Chunk Text<br/>Token-based or Word-based]
    ChunkText --> MapPhase[MAP PHASE<br/>Summarize Each Chunk]
    MapPhase --> MapModel[Use MAP Model<br/>Default: BART-large]
    MapModel --> ParallelCheck{Device<br/>Type?}
    ParallelCheck -->|CPU| ParallelMap[Parallel Processing<br/>ThreadPoolExecutor]
    ParallelCheck -->|GPU/MPS| SequentialMap[Sequential Processing<br/>Avoid Thrashing]
    ParallelMap --> ChunkSummaries[Chunk Summaries<br/>Generated]
    SequentialMap --> ChunkSummaries
    
    ChunkSummaries --> CombineSummaries[Combine Summaries]
    CombineSummaries --> CheckCombinedSize{Combined<br/>Token Count?}
    
    CheckCombinedSize -->|≤800 tokens| SinglePass[SINGLE-PASS ABSTRACTIVE<br/>Direct Final Summary]
    CheckCombinedSize -->|800-4000 tokens| MiniMapReduce[HIERARCHICAL REDUCE<br/>Mini Map-Reduce]
    CheckCombinedSize -->|>4000 tokens| Extractive[EXTRACTIVE APPROACH<br/>Select Key Chunks]
    
    MiniMapReduce --> RechunkSummaries[Re-chunk Combined Summaries<br/>3-5 sections, 650 words each]
    RechunkSummaries --> SummarizeSections[Summarize Each Section]
    SummarizeSections --> CheckIterations{More<br/>Iterations<br/>Needed?}
    CheckIterations -->|Yes, <4 passes| RechunkSummaries
    CheckIterations -->|No| FinalAbstractive[Final Abstractive Reduce]
    
    Extractive --> SelectKeyChunks[Select Representative Chunks<br/>First, Middle, Last, Quartiles]
    SelectKeyChunks --> FinalExtractivePass{Still Too<br/>Long?}
    FinalExtractivePass -->|Yes| FinalExtractSummarize[Final Summarization Pass]
    FinalExtractivePass -->|No| UseExtractiveOutput[Use Selected Chunks]
    FinalExtractSummarize --> FinalSummary
    UseExtractiveOutput --> FinalSummary
    
    SinglePass --> ReduceModel[Use REDUCE Model<br/>Default: LED long-fast]
    ReduceModel --> FinalAbstractive
    FinalAbstractive --> FinalSummary[Final Summary Generated]
    DirectSummarize --> FinalSummary
    
    FinalSummary --> ValidateSummary[Validate Summary<br/>Check for repetition/leaks]
    ValidateSummary --> StripLeaks[Strip Instruction Leaks]
    StripLeaks --> FixRepetitive[Fix Repetitive Content]
    FixRepetitive --> CreateMetadata[Create Summary Metadata<br/>Timestamp, Model, Word Count]
    CreateMetadata --> StoreInMetadata[Store in Episode Metadata<br/>JSON/YAML]
    StoreInMetadata --> Complete([Summarization Complete])
    Skip --> Complete
    
    style Start fill:#e1f5ff
    style CleanTranscript fill:#fff3cd
    style MapPhase fill:#ffd4a3
    style MiniMapReduce fill:#d4a3ff
    style Extractive fill:#ffa3d4
    style SinglePass fill:#a3ffd4
    style FinalSummary fill:#d4edda
    style Complete fill:#d4edda
```

## Summarization Process Details

### 1. Transcript Cleaning Phase

- Removes timestamps (language-agnostic: `[00:12:34]` patterns)
- Strips generic speaker tags while preserving actual speaker names
- Removes sponsor blocks and outro sections
- Collapses excessive blank lines
- Optionally saves cleaned transcript to `.cleaned.txt` for validation

### 2. Size Assessment & Model Selection

- Checks if transcript fits within model's context window
- If fits: performs single-pass direct summarization (most efficient)
- If too long: initiates MAP-REDUCE pipeline
- Selects MAP model (default: BART-large for chunk summaries)
- Selects REDUCE model (default: LED long-fast for final combination)

### 3. MAP Phase (Chunk Summarization)

- **Chunking Strategy**: Token-based chunking with configurable overlap
  - Encoder-decoder models (BART/PEGASUS): forced to 600 tokens per chunk
  - Long-context models (LED): can handle larger chunks
  - Overlap: 10% of chunk size for context continuity
- **Parallel vs Sequential Processing**:
  - CPU devices: parallel processing with ThreadPoolExecutor (up to 4 workers)
  - GPU/MPS devices: sequential to avoid memory thrashing
- **Progress Tracking**: Logs every 5 chunks with ETA estimation
- **Output**: Collection of chunk summaries (80-160 tokens each)

### 4. REDUCE Phase (Summary Combination)

The REDUCE phase employs a three-tier decision tree based on combined summary size:

#### Tier 1: Single-Pass Abstractive (≤800 tokens)

- Most efficient approach for short combined summaries
- Uses REDUCE model (default: LED) to generate final summary in one pass
- Target: 200-480 tokens final summary
- No chunk selection - uses ALL summaries for completeness

#### Tier 2: Hierarchical Reduce (800-4000 tokens)

- Fully abstractive approach for medium-sized combined summaries
- Implements mini map-reduce strategy:
  1. Re-chunk combined summaries into 3-5 sections (650 words each)
  2. Summarize each section (80-160 tokens per section)
  3. Check if result is small enough for final reduce
  4. If still too large, repeat (up to 4 iterations max)
  5. Final abstractive reduce when size permits
- Uses token-based chunking to guarantee no truncation
- Uses ALL section summaries (no selection) for accuracy
- Prevents extractive fallback by iteratively reducing size

#### Tier 3: Extractive Approach (>4000 tokens)

- Safety fallback for very large combined summaries
- Selects representative chunks instead of using all:
  - ≤3 chunks: use all
  - ≤10 chunks: use first, middle, last
  - >10 chunks: use first, 25th, 50th, 75th percentile, last
- One optional final summarization pass if selected chunks still too long
- Prioritizes avoiding hallucination over completeness

### 5. Quality Validation

- Strips instruction leaks (removes prompt text that leaked into output)
- Detects and fixes repetitive content (deduplicates repeated sentences)
- Validates summary length (flags if suspiciously close to input length)
- Returns empty string if quality validation fails

### 6. Metadata Storage

- Creates `SummaryMetadata` object with:
  - Generated summary text
  - Generation timestamp
  - Model used (MAP and optionally REDUCE)
  - Provider (local/openai/anthropic)
  - Word count of original transcript
- Stores in episode metadata document (PRD-004/RFC-011)
- Respects `--skip-existing` flag (regenerates if transcription source changes)

## Model Selection Strategy

### MAP Model (Chunk Summarization)

- **Default**: `facebook/bart-large-cnn` (best quality, ~2GB)
- **Fast option**: `sshleifer/distilbart-cnn-12-6` (~300MB)
- **Alternative**: `google/pegasus-large` (trained for summarization, ~2.5GB)

### REDUCE Model (Final Combination)

- **Default**: `allenai/led-base-16384` (long-context, ~1GB, handles 16k tokens)
- **High quality**: `allenai/led-large-16384` (~2.5GB, better accuracy)
- **Fallback**: Falls back to MAP model if not specified

### Device Selection

- Auto-detects: MPS (Apple Silicon) → CUDA (NVIDIA) → CPU
- GPU/MPS: Sequential processing, memory-efficient
- CPU: Parallel processing (4 workers), slower but works without GPU

## Performance Characteristics

- **Direct Summarization**: <5s for transcripts ≤1024 tokens
- **MAP-REDUCE**: ~3s per chunk (varies by model and device)
- **Parallel Processing**: 3-4x speedup on CPU with 4 workers
- **Memory Usage**:
  - BART-large: ~2GB GPU memory
  - DistilBART: ~300MB (recommended for memory-constrained systems)
  - LED-base: ~1GB (reduce phase)

## Error Handling

- Model loading failures: Log warning, skip summarization
- Out-of-memory errors: Log error, skip summarization (no automatic fallback)
- Buffer size errors (MPS): Log error with device details
- Validation failures: Return empty summary rather than hallucinated content
- Summarization failures don't block transcript processing

## Configuration

Summarization is configured via the `Config` model. Key options include:

- `generate_summaries`: Enable/disable summarization
- `summary_model`: MAP model selection (default: `facebook/bart-large-cnn`)
- `summary_reduce_model`: REDUCE model selection (default: `allenai/led-base-16384`)
- `save_cleaned_transcript`: Save cleaned transcripts to `.cleaned.txt` files

See [Configuration Documentation](api/configuration.md) for complete configuration options.

## Related Documentation

- [Architecture](ARCHITECTURE.md) - High-level system design
- [Development Guide](DEVELOPMENT_GUIDE.md) - General development practices
- [Custom Provider Guide](CUSTOM_PROVIDER_GUIDE.md) - Creating custom summarization providers
- [PRD-005](prd/PRD-005-episode-summarization.md) - Product requirements for summarization
- [RFC-012](rfc/RFC-012-episode-summarization.md) - Design decisions for summarization

