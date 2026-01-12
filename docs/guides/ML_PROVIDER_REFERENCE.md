# ML Provider Reference

This document serves as a comprehensive technical reference for the unified local ML provider (`MLProvider`)
in the podcast scraper system. It details the internal pipelines, key concepts, and architectural decisions
that ensure high-quality transcript processing and summarization.

For general configuration, see the [Provider Configuration Quick Reference](PROVIDER_CONFIGURATION_QUICK_REFERENCE.md).

---

## 1. Architectural Overview

The `MLProvider` is a unified implementation that handles three core capabilities using local machine learning models:

1. **Transcription**: Using **Whisper** (OpenAI's open-source model).
2. **Speaker Detection**: Using **spaCy** (Named Entity Recognition).
3. **Summarization**: Using **Hugging Face Transformers** (BART, LED, PEGASUS).

### Overall ML Pipeline

```mermaid
flowchart TD
    subgraph MLProvider [MLProvider Unified Pipeline]
        direction TB
        Audio([Raw Audio]) --> Transcribe[Transcription Pipeline\n(Whisper)]
        Metadata([Episode Metadata]) --> Detect[Speaker Detection Pipeline\n(spaCy)]

        Transcribe --> Screenplay[Screenplay Formatting\n(Integration)]
        Detect --> Screenplay

        Screenplay --> Preprocess[Preprocessing Pipeline\n(Sanitation)]
        Preprocess --> Summarize[Summarization Pipeline\n(MAP-REDUCE)]

        Summarize --> Results([Final Transcript + Summary])
    end

    style Transcribe fill:#d1ecf1
    style Detect fill:#d1ecf1
    style Preprocess fill:#fff3cd
    style Summarize fill:#ffd4a3
    style Results fill:#d4edda
```python

### Key Concepts

* **Unified Implementation**: Matches the pattern of cloud providers (like OpenAI), where a single provider class orchestrates multiple tasks using shared underlying libraries.
* **Lazy Loading**: Models are loaded into memory only when first requested, saving resources during dry runs or partial processing.
* **Device Awareness**: Automatic detection and utilization of the best available hardware accelerator:
  * **MPS** (Apple Silicon GPU)
  * **CUDA** (NVIDIA GPU)
  * **CPU** (Fallback)
* **Thread Safety**: Explicitly marked as requiring separate instances per worker (`_requires_separate_instances = True`) because most Hugging Face and Whisper models are not natively thread-safe.

---

## 2. Transcription Pipeline

The transcription pipeline converts raw audio into structured text with timing information.

### 2.1 Workflow

```mermaid
flowchart LR
    Audio([Audio File]) --> Load[Audio Loading]
    Load --> Whisper[Whisper Invocation]
    Whisper --> Segments[Segment Generation]
    Segments --> Screenplay[Screenplay Formatting]
    Screenplay --> FinalTranscript([Structured Transcript])

    style Audio fill:#e1f5ff
    style FinalTranscript fill:#d4edda
```

1. **Audio Loading**: Loads MP3/WAV files.
2. **Whisper Invocation**: Transcribes using the configured model (e.g., `base.en`, `small.en`).
3. **Segment Generation**: Captures start/end timestamps for every sentence or phrase.
4. **Screenplay Formatting**: Rotates through detected speakers based on silence gaps and time intervals
   to produce a "script-style" transcript.

### 2.2 Transcription Tuning

Choose the right Whisper model based on your hardware and language needs:

| Model | Parameters | VRAM | Speed | Quality |
| :--- | :--- | :--- | :--- | :--- |
| `tiny` | 39M | ~1GB | 32x | Base |
| `base` | 74M | ~1GB | 16x | Good |
| `small` | 244M | ~2GB | 6x | Better |
| `medium` | 769M | ~5GB | 2x | Excellent |
| `large` | 1550M | ~10GB | 1x | Best |

**Parallelism Settings:**

- **`TRANSCRIPTION_PARALLELISM`**: Controls how many episodes are transcribed in parallel.
- **`PROCESSING_PARALLELISM`**: Controls how many episodes are processed (metadata, summarization)
  in parallel.

---

## 3. Speaker Detection Pipeline

The system uses Named Entity Recognition (NER) to identify potential hosts and guests from episode metadata.

### 3.1 Multi-Stage Analysis

```mermaid
flowchart TD
    Metadata([Feed & Episode Metadata]) --> Feed[Feed-Level Detection\n(RSS Title/Desc)]
    Metadata --> Episode[Episode-Level Detection\n(Title/Desc)]

    Feed --> NER[spaCy NER\n(PERSON Entities)]
    Episode --> NER

    NER --> Heuristics[Pattern Heuristics\n(with, featuring)]
    Heuristics --> Speakers([Detected Hosts & Guests])

    style Metadata fill:#e1f5ff
    style Speakers fill:#d4edda
```

1. **Feed-Level Detection**: Analyzes the podcast title and description to identify recurring hosts.
2. **Episode-Level Detection**: Analyzes specific episode titles (e.g., "Guest: John Doe") and
   descriptions.

3. **Pattern Heuristics**: Uses `spaCy` to identify `PERSON` entities and applies heuristics
   (e.g., "with", "featuring", "interviewing") to differentiate between hosts and guests.

### 3.2 Detection Models

The system supports standard `spaCy` models:

- `en_core_web_sm`: Fastest, suitable for basic testing.
- `en_core_web_md`: Good balance of accuracy and speed.
- `en_core_web_lg`: Best accuracy for identifying names in noisy descriptions.

---

## 4. Preprocessing Pipeline (Sanitation)

Before any text reaches a summarization model, it undergoes a multi-stage "hard cleaning" process.
This was identified in **Issue #83** as the single most important factor for preventing model
hallucinations and "scaffolding echo."

### 4.1 Processing Order

```mermaid
flowchart LR
    Raw[Raw Transcript] --> Credits[Credit Stripping]
    Credits --> Boilerplate[Boilerplate Removal]
    Boilerplate --> Timestamps[Timestamp Removal]
    Timestamps --> Norm[Speaker Normalization]
    Norm --> Artifacts[Artifact Removal]
    Artifacts --> Clean([Cleaned Transcript])

    style Raw fill:#e1f5ff
    style Clean fill:#d4edda
```yaml

1. **Credit Stripping**: Removes grammatically perfect "credit blocks" (e.g., "Produced by...", "Music by..."). Models often latch onto these as high-confidence summary targets.
2. **Boilerplate Removal**: Strips website chrome like "Read more," "Back to top," and "Article continues below."
3. **Timestamp Removal**: Strips `[00:12:34]` markers to normalize the text for the encoder.
4. **Speaker Normalization**: Replaces generic labels (e.g., `SPEAKER 1:`) with actual names (if detected) or removes them to prevent "Speaker 1 said Speaker 1 said" loops.
5. **Artifact Removal**: Strips ML-specific artifacts like `TextColor`, `MUSIC`, and `[INAUDIBLE]` that BART/LED tend to copy verbatim.

---

## 5. Summarization Pipeline (BART/LED Implementation)

The summarization system uses a **MAP-REDUCE** architecture designed to handle transcripts ranging from 5 minutes to 2 hours.

### 5.1 Summarization Flow

```mermaid
flowchart TD
    Start([Transcript Available]) --> CleanTranscript[Clean Transcript]
    CleanTranscript --> CheckSize{Text Size Check}
    CheckSize -->|Fits in Context| DirectSummarize[Direct Summarization]
    CheckSize -->|Too Long| StartMapReduce[Begin MAP-REDUCE]

    StartMapReduce --> ChunkText[Chunk Text]
    ChunkText --> MapPhase[MAP PHASE - Summarize Chunks]
    MapPhase --> ChunkSummaries[Chunk Summaries Generated]
    ChunkSummaries --> CombineSummaries[Combine Summaries]
    CombineSummaries --> CheckCombinedSize{Combined Size?}

    CheckCombinedSize -->|Small| SinglePass[Single-Pass Reduce]
    CheckCombinedSize -->|Medium| MiniMapReduce[Hierarchical Reduce]
    CheckCombinedSize -->|Large| Extractive[Extractive Approach]

    MiniMapReduce --> FinalAbstractive[Final Abstractive Reduce]
    Extractive --> FinalSummary
    SinglePass --> FinalAbstractive
    FinalAbstractive --> FinalSummary[Final Summary Generated]
    DirectSummarize --> FinalSummary

```python

    FinalSummary --> ValidateSummary[Validate Summary]
    ValidateSummary --> StoreInMetadata[Store in Episode Metadata]
    StoreInMetadata --> Complete([Summarization Complete])

```

```mermaid
    style CleanTranscript fill:#fff3cd
    style MapPhase fill:#ffd4a3
    style MiniMapReduce fill:#d4a3ff
    style Extractive fill:#ffa3d4
    style SinglePass fill:#a3ffd4
    style FinalSummary fill:#d4edda
    style Complete fill:#d4edda
```

### 5.2 Size Assessment & Model Selection

- Checks if transcript fits within model's context window.
- **Model-Specific Thresholds (v2.4.0)**:
  - **BART models** (`facebook/bart-large-cnn`, `facebook/bart-base`): ~1024 tokens.
  - **LED models** (`allenai/led-large-16384`, `allenai/led-base-16384`): ~16384 tokens.
- **Transition Zones**: Implements smooth transition zones instead of hard cutoffs to ensure
  consistent quality near threshold boundaries.

### 5.3 MAP Phase (Compression)

Individual transcript chunks are summarized in parallel (on CPU) or sequentially (on GPU).

- **Chunking**: Token-based with 10% overlap.
  - BART models: Limited to 600 tokens per chunk.
  - LED models: Recommended 4000-8000 tokens per chunk.
- **Parallelism**: CPU uses `ThreadPoolExecutor` (up to 4 workers). GPU/MPS is sequential to avoid OOM.
- **Output**: Each chunk produces 80-160 tokens of bullet-style notes.

### 5.4 REDUCE Phase (Synthesis)

The system dynamically selects a synthesis strategy based on the total token count of the combined chunk summaries:

| Strategy | Token Range | Logic |
| :--- | :--- | :--- |
| **Single-Pass Abstractive** | < 800 | Direct summarization of all combined notes. |
| **Hierarchical Reduce** | 800 - 3,500 (BART)<br>800 - 5,500 (LED) | Recursive chunking and summarizing until input is small enough for final pass. |
| **Transition Zone** | 3,500 - 4,500 (BART)<br>5,500 - 6,500 (LED) | Smoothly switches to extractive to avoid quality degradation. |
| **Extractive Fallback** | > 4,500 (BART)<br>> 6,500 (LED) | Selects representative chunks (start, middle, 25/50/75%, end) to form final summary. |

### 5.5 Model Selection Strategy (Aliases)

To ensure stability, only validated model aliases are supported for the `transformers` provider.

- **MAP Models**:
  - `bart-large`: `facebook/bart-large-cnn` (Production default, best quality, ~2GB)
  - `bart-small`: `facebook/bart-base` (Test/Dev, faster, ~500MB)
- **REDUCE Models**:
  - `long`: `allenai/led-large-16384` (Production default, 16k context, ~2.5GB)
  - `long-fast`: `allenai/led-base-16384` (Faster, ~1GB)

---

## 6. Performance & Troubleshooting

### 6.1 Performance Characteristics

- **Direct Summarization**: <5s for transcripts ≤1024 tokens.
- **MAP-REDUCE**: ~3s per chunk (varies by hardware).
- **Memory**: High-quality models require 2-3GB of VRAM/RAM.

### 6.2 Quality Validation Tuning

The `SUMMARY_VALIDATION_THRESHOLD` (default: 0.2) controls how strict the system is when validating generated summaries.

- **Lower (e.g., 0.1)**: More permissive, allows shorter summaries relative to input.
- **Higher (e.g., 0.3)**: Stricter, requires longer summaries, more likely to flag quality issues.

### 6.3 Memory Management

If you experience Out-of-Memory (OOM) errors:

1. Decrease `PROCESSING_PARALLELISM`.
2. Use smaller model variants (e.g., `distilbart` instead of `bart-large`).
3. Set `SUMMARY_DEVICE=cpu` to move summarization off the GPU.

---

## 7. Cache & Maintenance

### 7.1 Cache Status

Use the CLI to check your cache status frequently:

```bash

python3 -m podcast_scraper.cli cache --status

```

### 7.2 Backup & Restore

The cache directory can be backed up and restored for easy management:

```bash

# Create backup

make backup-cache

# Interactive restore

make restore-cache

```yaml

---

## 8. Local Summarization Strategy for Long-Form Content

For very long transcripts (30,000–40,000+ tokens), local summarization requires a careful balance between quality, speed, and hardware capabilities.

### 8.1 Hardware-Specific Recommendations (Apple Silicon / MBP M4)

With modern hardware (e.g., 48GB+ unified memory), you have several high-performance options:

1.  **Accelerated Transformers (Current Baseline)**: Use `facebook/bart-large-cnn` or `allenai/led-large-16384` with Metal (`device="mps"`) acceleration. This is the fastest fully integrated path.
2.  **Local LLMs (Recommended for Quality)**: Use instruction-tuned LLMs (7B–14B parameters) via tools like **Ollama** or **llama.cpp**. Models like `llama3:8b` or `mistral:7b-instruct` provide significantly better abstractive synthesis than BART/LED.

### 8.2 Long-Form Best Practices

*   **Pre-Clean Aggressively**: Always use the Preprocessing Pipeline (Section 4) to strip timestamps and boilerplate. This reduces token count by 10-20% and prevents "scaffolding echo" where the model repeats filler text.
*   **Use MAP-REDUCE**: Never attempt to "brute-force" a 40k token transcript into a single context window. Even if the model supports it, quality often degrades near the context limit ("Lost in the Middle" phenomenon).
*   **Incremental Path**:
    *   **Short Term**: Stick with the integrated BART/LED pipeline for stability and speed.
    *   **Medium Term**: If quality is insufficient, consider transitioning to a local LLM-based provider (e.g., Ollama integration) for better high-level reasoning and "why this matters" insights.

---

## 9. Evaluation & Quality Validation

To ensure consistent summarization quality as you tweak models or parameters, use the following evaluation framework.

### 9.1 Evaluation Levels

1.  **Unit Level (Plumbing)**: Verify chunkers produce overlapping segments without data loss and cleaners strip timestamps correctly.
2.  **Functional Level (Execution)**: Ensure the pipeline produces non-empty summaries of reasonable length without crashing or generating artifacts (e.g., `[CLS]`).
3.  **Quality Level (Insight)**: Assess the summary for **Faithfulness** (no hallucinations), **Coverage** (key points included), and **Clarity**.

### 9.2 Manual Evaluation Rubric (1–5 Scale)

*   **Coverage**: Does it mention all major topics/segments?
*   **Faithfulness**: Does it misrepresent or hallucinate details?
*   **Clarity & Structure**: Is it easy to read with a clear structure (bullets/paragraphs)?
*   **Conciseness**: Is the compression ratio appropriate (~5–15× shorter than the transcript)?

### 9.3 Reference-Free Automated Checks

Even without human-written reference summaries, the system can perform sanity checks:

*   **Compression Ratio**: Alert if the summary is too long (ratio < 2) or suspiciously short (ratio > 50).
*   **Repetition Detection**: Flag summaries with repeating n-grams (common with BART models near their context limit).
*   **Keyword Coverage**: Extract top keywords from the transcript and verify they appear in the final summary.

---

## 10. Current State & Baseline

As of **v2.4.0**, the BART/LED implementation is considered the **Classic Summarizer Baseline**. It is stable, handles multi-hour transcripts via MAP-REDUCE, and includes robust quality validation.

### 8.1 Known Model-Level Limitations

The following issues are architectural limits of BART/LED models:

1. **Narrative Recap Bias**: Chronological retelling rather than insight synthesis.
2. **Weak Abstraction**: Struggles with "why this matters" conclusions.
3. **Quote Repetition**: Often repeats hedging language found in investigative reporting.
