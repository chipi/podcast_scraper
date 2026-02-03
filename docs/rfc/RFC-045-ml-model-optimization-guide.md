# RFC-045: ML Model Optimization Guide

- **Status**: Draft
- **Authors**: Marko Dragoljević
- **Stakeholders**: ML/AI Engineers, Developers using podcast_scraper
- **Related PRDs**:
  - `docs/prd/PRD-005-summarization.md` (Summarization feature)
  - `docs/prd/PRD-007-ai-experiments.md` (AI experiment pipeline)
- **Related ADRs**:
  - `docs/adr/ADR-029-preprocessing-profiles.md` (Preprocessing profiles)
- **Related RFCs**:
  - `docs/rfc/RFC-012-episode-summarization.md` (Episode summarization)
  - `docs/rfc/RFC-041-podcast-ml-benchmarking-framework.md` (ML benchmarking)
- **Related Documents**:
  - `docs/wip/preprocessing_improvements_plan.md` (Source document)
  - `docs/wip/baseline_bart_experiment_plan.md` (Source document)

## Abstract

This RFC provides a comprehensive guide for maximizing ML model quality in podcast_scraper through two orthogonal approaches: **preprocessing optimization** (model-agnostic text cleaning) and **generation parameter tuning** (model-specific settings). The goal is to extract maximum quality from smaller, faster models before graduating to larger, more expensive alternatives.

**Architecture Alignment:** This RFC builds on RFC-016's modular provider architecture and RFC-041's benchmarking framework, providing concrete optimization strategies that apply across all ML providers.

## Problem Statement

### First Principle: Generic CNN Summarizers Hate Raw Dialogue

Models like `facebook/bart-base` are **not trained to summarize messy dialogue transcripts**. They were trained on news articles, Wikipedia, and other well-structured prose. When you feed them:

- Speaker turns (`Maya: ...`, `Liam: ...`)
- Interruptions and overlapping speech
- Filler words (`uh`, `um`, `you know`, `like`)
- Stage directions (`(Pause)`, `(Laughter)`, `[music]`)
- Transcription artifacts (`/////`, `=-`, `Desc-`)

...they will often produce exactly what we're seeing: **repetition loops + label hallucinations + garbage output**.

This is not a bug—it's a fundamental mismatch between training data and input format.

### The Core Question

Before touching parameters, you must decide:

> **Do you want your summarizer to work on dialogue transcripts?**

If yes, you must either:

1. **Preprocess dialogue into a cleaner format** (this RFC's primary approach), or
2. **Use a dialogue-tuned summarization model** (future consideration)

### Observed Quality Issues

Current baseline experiments reveal significant quality issues:

| Issue | Baseline Value | Impact |
|-------|----------------|--------|
| Speaker Name Leak Rate | **80%** | Names appear in summaries |
| "Too Short" Warnings | 5/5 episodes | Incomplete summaries |
| Repetitive Output | Present | Garbage artifacts (`=-`, `////////`) |
| Failed Episodes | 20% | Pipeline failures |

### Root Causes

These issues stem from **two orthogonal root causes** that must both be addressed:

1. **Inadequate Preprocessing** (Step A)
   - Current `cleaning_v3` preserves speaker names like `Maya:` and `Liam:` that leak into outputs
   - Junk lines (punctuation artifacts, stage directions) confuse the model
   - `Name:` format triggers verbatim copying behavior

2. **Suboptimal Generation Parameters** (Step B)
   - Default settings prioritize speed over quality
   - No anti-repetition controls enabled
   - Length constraints too restrictive for podcast content

### Why Both Steps Are Required

| If You Only Do... | Result |
|-------------------|--------|
| Step A (preprocessing) only | Cleaner input, but model may still produce short/repetitive output |
| Step B (parameters) only | Better generation behavior, but garbage-in-garbage-out |
| **Both A + B** | **Clean input + constrained generation = quality output** |

Without addressing both fronts, users will either:

- Waste compute on larger models that don't solve preprocessing issues
- Miss easy quality wins from parameter tuning
- Chase phantom bugs that are actually input format problems

### Use Cases

1. **Resource-Constrained Deployment**: Maximize quality from `bart-small` + `led-base` before requiring larger models
2. **Quality Debugging**: Systematically identify whether issues are preprocessing or model-related
3. **Baseline Establishment**: Create reproducible quality benchmarks for model comparisons
4. **Dialogue-to-Prose Conversion**: Transform raw transcripts into summarizer-friendly format

## Architecture Context

### Big Picture: Three Processing Flows

The podcast_scraper system has three interconnected processing flows. Understanding where preprocessing fits is essential for effective optimization:

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PODCAST SCRAPER ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  FLOW 1: PRODUCTION PIPELINE (workflow/orchestration.py)                        │
│  ═══════════════════════════════════════════════════════                        │
│                                                                                  │
│  RSS Feed ──► Episode ──► Download/      ──► Metadata ──► Summarization         │
│              Metadata     Transcription       Generation                         │
│                                │                    │              │             │
│                                ▼                    ▼              ▼             │
│                           [Whisper]           [metadata.json]  [summary.txt]    │
│                               │                                    │             │
│                               ▼                                    │             │
│                         transcript.txt ───────────────────────────►│             │
│                                                                    │             │
│                               PREPROCESSING HAPPENS HERE           │             │
│                               (inside summarizer at runtime)       │             │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  FLOW 2: EVALUATION PIPELINE (scripts/eval/run_experiment.py)                   │
│  ═══════════════════════════════════════════════════════════                    │
│                                                                                  │
│  experiment.yaml ──► Load Dataset ──► For Each Episode ──► provider.summarize() │
│       │                   │                   │                    │             │
│       │                   ▼                   ▼                    ▼             │
│       │            curated_5feeds_   raw transcript      Same cleanup applies   │
│       │            smoke_v1.json      file.txt           (inside model)         │
│       │                                                            │             │
│       └──────────────── preprocessing_profile: cleaning_v3 ───────►│             │
│                                                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  FLOW 3: PREPROCESSING SUBSYSTEM (preprocessing/core.py, profiles.py)           │
│  ═════════════════════════════════════════════════════════════════════          │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────┐           │
│  │                    PREPROCESSING PROFILES                         │           │
│  │                                                                   │           │
│  │  cleaning_none ──► (passthrough)                                 │           │
│  │  cleaning_v1 ────► Basic: timestamps, speakers, blanks           │           │
│  │  cleaning_v2 ────► v1 + sponsor blocks + outro blocks            │           │
│  │  cleaning_v3 ────► v2 + credits + garbage lines + artifacts      │◄─ DEFAULT │
│  │  cleaning_v4 ────► v3 + speaker anonymization + header stripping │◄─ PROPOSED│
│  └──────────────────────────────────────────────────────────────────┘           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Preprocessing is a Horizontal Concern

Preprocessing isn't a single stage in one pipeline—it's a **horizontal concern** that cuts across multiple flows:

| Flow | Where Preprocessing Happens | Profile Used | Configurable? |
|------|----------------------------|--------------|---------------|
| Production Pipeline | Inside `summarize_long_text()` at runtime | `cleaning_v3` (via apply_profile) | ✅ Yes |
| Eval Pipeline | Inside provider's `summarize()` at runtime | From experiment config | ✅ Yes |
| Offline Cleaning | `clean_for_summarization()` for `.cleaned.txt` | `cleaning_v3` equivalent | ❌ No |

### Three Layers of Processing

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    THREE LAYERS OF PROCESSING                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LAYER 1: ACQUISITION                                                   │
│  ─────────────────────                                                  │
│  RSS → Download → Transcribe → Save transcript.txt                      │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  LAYER 2: PREPROCESSING (horizontal concern)                            │
│  ───────────────────────────────────────────                            │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                   PREPROCESSING PROFILE SYSTEM                  │    │
│  │                                                                 │    │
│  │   Input ──► apply_profile(text, "cleaning_v4") ──► Output      │    │
│  │                        │                                        │    │
│  │                        ▼                                        │    │
│  │   ┌──────────────────────────────────────────────────────────┐ │    │
│  │   │ strip_episode_header → strip_credits → strip_garbage    │ │    │
│  │   │ → clean_transcript → anonymize_speakers → remove_sponsor │ │    │
│  │   │ → remove_outro → remove_artifacts → filter_junk_lines   │ │    │
│  │   └──────────────────────────────────────────────────────────┘ │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  LAYER 3: ML INFERENCE                                                  │
│  ─────────────────────                                                  │
│  Cleaned text ──► MAP (BART) ──► REDUCE (LED) ──► Final Summary         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Fundamental Problem: Late-Stage Preprocessing Can't Fix Early-Stage Input

```text
  RAW TRANSCRIPT (what models currently receive):
  ┌─────────────────────────────────────────────────────────────┐
  │ Maya: Hi, Maya. I'm here to talk about building trails that │
  │       last. What do you recommend?                          │
  │ Liam: We're talking about riding trails that are rideable   │
  │       longer, and the soil structure stays intact instead   │
  │       of turning to ruts.                                   │
  │ Maya: And what do you teach that to someone new?            │
  └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
  AFTER cleaning_v3 (current):
  ┌─────────────────────────────────────────────────────────────┐
  │ Maya: Hi, Maya. I'm here to talk about building trails that │  ← Still dialogue!
  │       last. What do you recommend?                          │  ← Still speaker names!
  │ Liam: We're talking about riding trails...                  │  ← Still Q&A format!
  │ Maya: And what do you teach that to someone new?            │
  └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
  MODEL OUTPUT (garbage):
  ┌─────────────────────────────────────────────────────────────┐
  │ Maya: Hi, Maya. Hi, Maya... What do you recommend?          │
  │ \\\\\\\\\\\\\\\\                                            │
  │ Liam says Right. Weap- Atras- Athens-                       │
  └─────────────────────────────────────────────────────────────┘

  AFTER cleaning_v4 (proposed):
  ┌─────────────────────────────────────────────────────────────┐
  │ A: Hi. I'm here to talk about building trails that last.    │  ← Anonymized!
  │    What do you recommend?                                    │
  │ B: We're talking about riding trails that are rideable      │  ← No name leak!
  │    longer, and the soil structure stays intact instead      │
  │    of turning to ruts.                                      │
  │ A: And what do you teach that to someone new?               │
  └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
  MODEL OUTPUT (improved):
  ┌─────────────────────────────────────────────────────────────┐
  │ Building trails that last requires attention to soil        │
  │ structure and drainage. Rideable trails maintain their      │
  │ structure instead of turning to ruts...                     │
  └─────────────────────────────────────────────────────────────┘
```

**Key Insight**: `cleaning_v3` doesn't address speaker names, episode headers, or junk line patterns—the primary causes of the 80% speaker leak rate observed in experiments.

## Goals

1. **Eliminate Speaker Name Leakage**: Reduce from 80% to <10% through preprocessing
2. **Improve Output Completeness**: Eliminate "too short" warnings through parameter tuning
3. **Reduce Repetition Artifacts**: Use n-gram blocking and repetition penalties
4. **Establish Model-Agnostic Best Practices**: Preprocessing improvements that benefit all models
5. **Document Parameter Sensitivity**: Identify which parameters have highest quality impact

## Constraints & Assumptions

**Constraints:**

- Preprocessing must not destroy semantic content
- Parameter changes must be testable via experiment configs
- Changes must be backward compatible with existing pipelines
- No new external dependencies

**Assumptions:**

- Synthetic test transcripts have intentional repetition (affects baseline interpretation)
- Speaker anonymization acceptable for summarization use case
- Faster models (BART-small) are preferred over larger models when quality is comparable

## Design & Implementation

### 1. Preprocessing Optimization (Step A)

Fix the input so BART/LED isn't doomed from the start. These improvements help **all models** (BART, LED, T5, OpenAI) by providing cleaner input. This alone often converts "garbage" into "usable".

#### 1.1 Normalize Transcript into "Dialogue-Safe" Text

Apply these deterministic transforms **before** the Map stage:

| Transform | What It Does | Why It Helps |
|-----------|--------------|--------------|
| Remove junk lines | Strips `////`, `====`, `---`, `=-` | Prevents garbage in output |
| Remove stage directions | Strips `(Pause)`, `[music]`, `Desc-` | Removes non-content |
| Collapse whitespace | `\n\n\n` → `\n\n` | Cleaner token boundaries |
| Limit repeated phrases | Deduplicate obvious repetition | Prevents copying loops |

**Junk Line Detection**:

```python
def is_junk_line(line: str) -> bool:
    """Detect lines dominated by punctuation or artifacts."""
    stripped = line.strip()
    if not stripped:
        return False
    # Lines that are mostly punctuation
    punct_ratio = sum(1 for c in stripped if c in '=/\\-_|') / len(stripped)
    if punct_ratio > 0.5:
        return True
    # Known artifacts
    artifacts = ['Desc-', '(Pause)', '[music]', 'subscribe', '////']
    return any(art.lower() in stripped.lower() for art in artifacts)
```

#### 1.2 Speaker Anonymization (HIGH PRIORITY)

**Problem**: `Maya:` and `Liam:` labels leak into summaries because `normalize_speakers` only removes generic labels (`Host:`, `Speaker 1:`).

**Solution**: Replace real speaker names with anonymous labels:

```python
def anonymize_speakers(text: str) -> str:
    """Replace speaker names with anonymous labels (A:, B:, C:...).

    Args:
        text: Transcript text with speaker labels like "Maya: Hello"

    Returns:
        Text with anonymized labels like "A: Hello"

    Example:
        >>> anonymize_speakers("Maya: Hi there\\nLiam: Hello")
        'A: Hi there\\nB: Hello'
    """
    speaker_pattern = r"^([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)(?:\s*\([^)]+\))?\s*:"
    speakers_seen = {}
    lines = text.splitlines()
    result = []

    for line in lines:
        match = re.match(speaker_pattern, line)
        if match:
            name = match.group(1)
            if name not in speakers_seen:
                speakers_seen[name] = chr(ord('A') + len(speakers_seen))
            anon = speakers_seen[name]
            line = re.sub(speaker_pattern, f"{anon}:", line)
        result.append(line)

    return "\n".join(result)
```

**Expected Impact**: Speaker leak rate 80% → <10%

#### 1.3 Why Speaker Labels Trigger Copying

Small CNN models tend to copy `Name:` patterns verbatim. This is because:

1. **Pattern Recognition**: `Name:` at line start looks like a label the model should preserve
2. **Training Data**: News articles don't have dialogue turns, so models weren't trained to ignore them
3. **Attention Mechanism**: Capitalized names at predictable positions get high attention scores

**The Key Insight**: If you do only one preprocessing step, do speaker anonymization. It addresses the most common failure mode.

Alternative formats that work better (if anonymization isn't enough):

```text
# Instead of:
Maya: Welcome back to the show.
Liam: Thanks for having me.

# Convert to turn-based:
Turn 1: Welcome back to the show.
Turn 2: Thanks for having me.

# Or prose-style (more aggressive):
The host welcomed listeners back to the show.
The guest expressed gratitude for the invitation.
```

The `A:`, `B:` format is a good middle ground—it preserves turn structure while removing name-copying triggers.

#### 1.4 Episode Header Stripping (HIGH PRIORITY)

**Problem**: BART copies metadata headers directly into summaries:

```text
# Singletrack Sessions — Episode    ← Copied!
## Building Trails That Last        ← Copied!
Host: Maya                          ← Copied!
Guest: Liam                         ← Copied!
```

**Solution**: Strip header blocks before summarization:

```python
def strip_episode_header(text: str) -> str:
    """Remove episode title/metadata header block.

    Args:
        text: Transcript text possibly containing header metadata

    Returns:
        Text with header lines removed
    """
    patterns = [
        r"^#\s+.*?\n",           # Markdown H1: "# Title"
        r"^##\s+.*?\n",          # Markdown H2: "## Subtitle"
        r"^Host:\s*\w+.*?\n",    # "Host: Maya"
        r"^Guest:\s*\w+.*?\n",   # "Guest: Liam"
        r"^Guests?:\s*.*?\n",    # "Guests: A, B, C"
    ]
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()
```

#### 1.5 Current Gap: Profiles Exist But Aren't Wired

> **Important**: The preprocessing profile system exists but is **not connected** to the summarization pipeline.

**What exists today:**

```python
# In preprocessing/profiles.py - WORKS
register_profile("cleaning_v3", _cleaning_v3)
apply_profile(text, "cleaning_v3")  # Can be called
```

**What the summarizer actually does:**

```python
# In providers/ml/summarizer.py line 1222 - HARDCODED
cleaned_text = preprocessing.clean_for_summarization(text)  # Ignores profiles!
```

The `clean_for_summarization()` function is essentially `cleaning_v3` logic, but it bypasses the profile registry entirely.

**What needs to change:**

| Component | Current | Required |
|-----------|---------|----------|
| `summarizer.py` | Calls `clean_for_summarization()` | Call `apply_profile(text, profile_id)` |
| `ExperimentConfig` | No profile field | Add `preprocessing_profile: str` field |
| `run_experiment.py` | Hardcodes "cleaning_v3" in fingerprint | Pass profile to summarizer |
| Provider interface | No profile param | Accept `preprocessing_profile` param |

**Wiring path:**

```text
experiment.yaml                    # User specifies: preprocessing_profile: "cleaning_v4"
    ↓
run_experiment.py                  # Reads config, passes to provider
    ↓
MLProvider.summarize()             # Accepts preprocessing_profile param
    ↓
summarize_long_text()              # Passes to preprocessing
    ↓
apply_profile(text, profile_id)    # Actually uses the profile!
```

Until this wiring is complete, creating `cleaning_v4` won't have any effect on experiments.

#### 1.7 New Preprocessing Profile: `cleaning_v4`

Combine all improvements into a new profile:

```python
def _cleaning_v4(text: str) -> str:
    """Enhanced cleaning with speaker anonymization and header stripping."""
    # 1. Strip episode header (title, Host:, Guest:)
    cleaned = strip_episode_header(text)

    # 2. Strip credits (before chunking)
    cleaned = strip_credits(cleaned)

    # 3. Strip garbage lines
    cleaned = strip_garbage_lines(cleaned)

    # 4. Anonymize speakers (Maya: → A:)
    cleaned = anonymize_speakers(cleaned)

    # 5. Standard cleaning (timestamps, normalize generic speakers)
    cleaned = clean_transcript(
        cleaned,
        remove_timestamps=True,
        normalize_speakers=True,
        collapse_blank_lines=True,
        remove_fillers=False,  # Keep disabled for safety
    )

    # 6. Remove sponsor/outro blocks
    cleaned = remove_sponsor_blocks(cleaned)
    cleaned = remove_outro_blocks(cleaned)

    # 7. Remove BART/LED artifacts
    cleaned = remove_summarization_artifacts(cleaned)

    return cleaned.strip()
```

### 2. Generation Parameter Tuning (Step B)

These parameters constrain generation to stop repetition and garbage output. They are tuned **per-stage** (Map vs Reduce) because each stage has different requirements.

#### 2.1 Core Principle: Constrain Generation

For Hugging Face `generate()` on both Map and Reduce stages, add anti-loop controls:

| Control Type | Purpose |
|--------------|---------|
| `do_sample: false` | Deterministic output (beam search, not sampling) |
| `no_repeat_ngram_size` | Prevents exact n-gram repetition |
| `repetition_penalty` | Penalizes token-level repetition |
| `encoder_no_repeat_ngram_size` | Prevents copying n-grams from input (future) |

#### 2.2 Map Stage Parameters (BART)

The Map stage summarizes individual chunks. Use tighter constraints:

| Parameter | Current Baseline | Code Default | Recommended | Notes |
|-----------|-----------------|--------------|-------------|-------|
| `do_sample` | false | **false** | false | Already deterministic ✓ |
| `num_beams` | 4 | 4 | **4-6** | 6 if stable on MPS |
| `no_repeat_ngram_size` | 3 | 3 | **4** | Stricter repetition blocking |
| `repetition_penalty` | - | **1.3** (hardcoded) | 1.15-1.3 | Already set, consider exposing |
| `length_penalty` | 1.0 | 1.0 | **1.0-1.1** | Slight preference for longer |
| `max_new_tokens` | 200 | 150 | **150-200** | Per-chunk summary length |
| `min_new_tokens` | 80 | - | **60-100** | Prevent ultra-short outputs |
| `early_stopping` | true | **true** | true | Already enabled ✓ |

**Map Stage Config** (matching actual schema):

```yaml
map_params:
  max_new_tokens: 200
  min_new_tokens: 80
  num_beams: 4
  no_repeat_ngram_size: 4
  length_penalty: 1.0
  early_stopping: true
```

#### 2.3 Reduce Stage Parameters (LED)

The Reduce stage combines Map summaries into final output. Allow longer output:

| Parameter | Current Baseline | Code Default | Recommended | Notes |
|-----------|-----------------|--------------|-------------|-------|
| `do_sample` | false | **false** | false | Already deterministic ✓ |
| `num_beams` | 4 | 4 | **4** | Stable for longer context |
| `no_repeat_ngram_size` | 3 | 3 | **4** | Stricter repetition blocking |
| `repetition_penalty` | - | **1.3** (hardcoded) | 1.12-1.3 | Already set, consider exposing |
| `length_penalty` | 1.0 | 1.0 | **1.0** | Neutral |
| `max_new_tokens` | 650 | 300 | **600-900** | Final summary length |
| `min_new_tokens` | 220 | - | **200-350** | Ensure substantial output |
| `early_stopping` | true | **true** | true | Already enabled ✓ |

**Reduce Stage Config** (matching actual schema):

```yaml
reduce_params:
  max_new_tokens: 650
  min_new_tokens: 220
  num_beams: 4
  no_repeat_ngram_size: 4
  length_penalty: 1.0
  early_stopping: true
```

#### 2.4 Advanced Anti-Repetition Controls (Future Enhancement)

> **Note**: These parameters are not currently exposed in the experiment config or summarizer code.
> Implementation requires changes to `run_experiment.py` and `SummaryModel.summarize()`.

If repetition persists after applying the above, these additional knobs could help:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `encoder_no_repeat_ngram_size` | **3** | Prevents copying 3-grams directly from input |
| `bad_words_ids` | `[tokenizer.encode("Name:")]` | Block specific problematic tokens |

**When `encoder_no_repeat_ngram_size` would help**:

- Model copies phrases verbatim from input
- Summaries contain transcript artifacts
- Speaker labels appear despite preprocessing

```python
# Example: Future implementation in generation call
outputs = model.generate(
    input_ids,
    encoder_no_repeat_ngram_size=3,  # Prevents input copying
    **other_params
)
```

**Implementation Task**: Add `encoder_no_repeat_ngram_size` support to `GenerationParams` schema and wire through the summarizer.

#### 2.5 Chunk Size Parameters

| Setting | Words/Chunk | Trade-off |
|---------|-------------|-----------|
| Smaller chunks | 600 | More focused map summaries, more chunks |
| Default | 800-1000 | Balance |
| Larger chunks | 1200 | More context per chunk, fewer chunks |

**When to use smaller chunks**: Dense technical content, many topic shifts

**When to use larger chunks**: Narrative content, fewer distinct topics

#### 2.6 Parameter Tuning Strategy

Start with conservative settings and adjust based on output quality:

```text
Repetition in output?
  → Increase no_repeat_ngram_size (4 → 5)
  → Note: repetition_penalty already at 1.3 (hardcoded)
  → Future: Add encoder_no_repeat_ngram_size: 3

Output too short?
  → Increase min_new_tokens
  → Increase length_penalty (1.0 → 1.1)
  → Check if input is being truncated

Output too long/rambling?
  → Decrease max_new_tokens
  → Decrease length_penalty (1.0 → 0.9)

Copying from input?
  → Future: Add encoder_no_repeat_ngram_size: 3
  → Check preprocessing (Step A)
```

### 3. Experiment Matrix

Run experiments systematically to isolate effects:

| Config ID | Preprocessing | Parameters | Hypothesis |
|-----------|---------------|------------|------------|
| v1 (baseline) | cleaning_v3 | default | Control |
| v2 | cleaning_v3 | +50% length | Longer outputs reduce "too short" |
| v3 | cleaning_v3 | ngram=4, beams=5 | Reduce repetition |
| v4 | cleaning_v3 | chunks=600 | More focused summaries |
| v5 | cleaning_v3 | chunks=1200 | More context |
| v6 | cleaning_v3 | best combined | Best parameters |
| **v7** | **cleaning_v4** | default | Preprocessing impact |
| **v8** | **cleaning_v4** | best combined | Maximum quality |

**Key Insight**: v7 isolates preprocessing impact, v8 combines both improvements.

**Already Created Configs** (parameter experiments v2-v6):

- `data/eval/configs/baseline_bart_v2_longer_output.yaml` - +50% max_new_tokens, length_penalty=1.2
- `data/eval/configs/baseline_bart_v3_stronger_ngram.yaml` - ngram=4/5, beams=5
- `data/eval/configs/baseline_bart_v4_smaller_chunks.yaml` - chunks=600 words
- `data/eval/configs/baseline_bart_v5_larger_chunks.yaml` - chunks=1200 words
- `data/eval/configs/baseline_bart_v6_combined_best.yaml` - combined best params

**To Be Created** (preprocessing experiments v7-v8):

- `data/eval/configs/baseline_bart_v7_cleaning_v4.yaml`
- `data/eval/configs/baseline_bart_v8_cleaning_v4_optimized.yaml`

### 3.1 Experiment Results (v1-v6)

Parameter tuning experiments (v1-v6) have been completed. Results validate the preprocessing-first approach:

#### Metrics Summary

| Experiment | Boilerplate Leak | Speaker Name Leak | Failed Episodes | Avg Tokens | Latency (ms) |
|------------|------------------|-------------------|-----------------|------------|--------------|
| **v1 baseline** | 0% | 80% | 0 | 445 | 37,179 |
| v2 longer output | **20%** ⚠️ | 80% | 1 (p03_e01) | 607.4 | 52,943 |
| v3 stronger ngram | **20%** ⚠️ | 80% | 1 (p05_e01) | 375.2 | 37,590 |
| v4 smaller chunks | 0% ✓ | 80% | 0 | 445 | 36,585 |
| v5 larger chunks | 0% ✓ | 80% | 0 | 477.4 | 38,430 |
| v6 combined | **20%** ⚠️ | 80% | 1 (p05_e01) | 438.2 | 45,214 |

#### Key Findings

1. **Speaker Name Leak is Constant at 80%** — Proves preprocessing is the root cause, not generation parameters

2. **Longer Output = More Garbage**
   - v2 produced more tokens (607 vs 445) but introduced boilerplate leakage
   - More room for output = more room for repetition loops

3. **Stronger N-gram Blocking Didn't Help**
   - v3 with `no_repeat_ngram_size=4` and `num_beams=5` still showed 80% speaker leak
   - Repetition blocking can't help when input itself is repetitive

4. **Chunk Size Has Minimal Impact**
   - v4 (600 words) and v5 (1200 words) produced nearly identical quality
   - Both maintained 0% boilerplate leak (same as baseline)

5. **Combined Parameters Combined Problems**
   - v6 inherited failures from v2/v3, didn't combine benefits

#### Conclusions

| Finding | Implication |
|---------|-------------|
| 80% speaker leak across ALL experiments | Preprocessing is the bottleneck |
| Map-stage outputs already broken | Problem occurs before reduce stage |
| Parameter tweaks don't fix core issues | Input quality must be improved first |
| v1/v4/v5 are equivalently "best" | Simpler is better until preprocessing fixed |

**Recommendation**: Implement `cleaning_v4` preprocessing before further parameter experiments. The 80% speaker name leak rate being constant across all 6 experiments is definitive proof that preprocessing must be addressed first.

## Key Decisions

1. **Anonymous Labels Over Complete Removal**
   - **Decision**: Use `A:`, `B:`, `C:` instead of removing speaker labels entirely
   - **Rationale**: Preserves turn-taking structure useful for some downstream tasks

2. **Profile-Based Preprocessing**
   - **Decision**: Create `cleaning_v4` profile instead of modifying `cleaning_v3`
   - **Rationale**: Maintains backward compatibility, allows A/B testing

3. **Parameter Exposure via Config**
   - **Decision**: Expose key parameters in experiment YAML configs
   - **Rationale**: Enables systematic experimentation without code changes

## Alternatives Considered

1. **Aggressive Speaker Stripping**
   - **Description**: Remove all speaker labels entirely (`Maya: Hello` → `Hello`)
   - **Pros**: Completely eliminates speaker leakage
   - **Cons**: Loses turn-taking structure
   - **Why Rejected**: Anonymous labels achieve same goal while preserving structure

2. **Model-Specific Preprocessing**
   - **Description**: Different preprocessing for BART vs LED vs OpenAI
   - **Pros**: Optimal for each model
   - **Cons**: Complexity explosion, harder to maintain
   - **Why Rejected**: Model-agnostic preprocessing is simpler and broadly effective

3. **Fine-Tuning Instead of Parameter Tuning**
   - **Description**: Fine-tune BART on podcast data
   - **Pros**: Potentially higher quality ceiling
   - **Cons**: Requires training data, compute, maintenance
   - **Why Rejected**: Out of scope; parameter tuning is sufficient for initial quality

## Testing Strategy

**Test Coverage:**

- **Unit tests**: `strip_episode_header()`, `anonymize_speakers()`, `is_junk_line()` edge cases
- **Integration tests**: Full preprocessing pipeline with various input formats
- **E2E tests**: Complete summarization runs comparing v3 vs v4 profiles

**Test Organization:**

```text
tests/
  unit/
    test_preprocessing_v4.py       # New functions
  integration/
    test_preprocessing_profiles.py # Profile comparison
  e2e/
    test_summarization_quality.py  # Quality metrics
```

**Test Markers:**

```python
@pytest.mark.unit
def test_anonymize_speakers_basic(): ...

@pytest.mark.integration
def test_cleaning_v4_profile(): ...

@pytest.mark.e2e
@pytest.mark.slow
def test_summarization_with_v4(): ...
```

**Test Execution:**

```bash
# Quick validation
make test-unit

# Integration testing
make test-integration

# Full quality validation (slow)
make experiment-run CONFIG=data/eval/configs/baseline_bart_v7_cleaning_v4.yaml
```

## Rollout & Monitoring

**Rollout Plan:**

- **Phase 1** (Week 1): Implement preprocessing functions, unit tests
- **Phase 2** (Week 2): Register `cleaning_v4` profile, run experiments v1-v6
- **Phase 3** (Week 3): Run experiments v7-v8, compare results
- **Phase 4** (Week 4): Document findings, update defaults if quality improved

**Monitoring:**

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Speaker Leak Rate | 80% | <10% | Manual review + automated detection |
| "Too Short" Rate | 100% | <20% | Summary token count |
| Repetition Artifacts | Present | Rare | Automated detection |
| Failed Episodes | 20% | <5% | Pipeline success rate |

**Success Criteria:**

1. ✅ Speaker leak rate reduced to <10%
2. ✅ "Too short" warnings reduced to <20%
3. ✅ No regression in latency (within 20%)
4. ✅ At least one v4-based config outperforms v3 baseline

## Relationship to Other RFCs

This RFC (RFC-045) complements the ML quality improvement initiative:

1. **RFC-012: Episode Summarization** - Core summarization implementation
2. **RFC-041: ML Benchmarking Framework** - Measurement infrastructure
3. **RFC-044: Model Registry** - Model configuration management

**Key Distinction:**

- **This RFC (045)**: *How* to optimize ML quality (preprocessing + parameters)
- **RFC-041**: *How* to measure ML quality (benchmarking framework)
- **RFC-044**: *How* to configure ML models (registry)

Together, these RFCs provide a complete ML quality management system.

## Benefits

1. **Immediate Quality Gains**: Up to 70% reduction in speaker leakage
2. **Systematic Experimentation**: Clear methodology for quality optimization
3. **Resource Efficiency**: Extract maximum value from smaller models
4. **Model-Agnostic Foundation**: Preprocessing improvements benefit all providers
5. **Reproducibility**: Documented configs enable reproducible experiments

## Migration Path

1. **Phase 1**: Implement `cleaning_v4` as opt-in profile
2. **Phase 2**: Run A/B experiments comparing v3 vs v4
3. **Phase 3**: If v4 consistently outperforms, make it default in next minor version
4. **Phase 4**: Deprecate v3 in future major version (with migration notice)

**Backward Compatibility:**

```yaml
# Explicit v3 (old behavior)
preprocessing:
  profile: "cleaning_v3"

# New v4 (opt-in initially)
preprocessing:
  profile: "cleaning_v4"
```

## Open Questions

1. **Filler Word Removal**: Should `remove_fillers=True` be enabled in v4? (Language-specific concerns)
2. **Repetition Penalty Exposure**: Should `repetition_penalty` (currently hardcoded at 1.3) be exposed in YAML configs for tuning?
3. **Encoder No-Repeat N-gram**: Should `encoder_no_repeat_ngram_size` be implemented to prevent input copying?
4. **Chunk Overlap**: Would overlapping chunks improve quality? (Note: already supported via `word_overlap` parameter)
5. **Multi-Language Support**: How should preprocessing handle non-English transcripts?

## Implementation Checklist

### New Functions

- [ ] `is_junk_line(line: str) -> bool` in `preprocessing/core.py`
- [ ] `strip_episode_header(text: str) -> str` in `preprocessing/core.py`
- [ ] `anonymize_speakers(text: str) -> str` in `preprocessing/core.py`

### New Profile

- [ ] `_cleaning_v4()` function in `preprocessing/profiles.py`
- [ ] `register_profile("cleaning_v4", _cleaning_v4)` call

### Config Schema & Wiring (Critical Path)

- [ ] Add `preprocessing_profile` field to `ExperimentConfig` in `evaluation/config.py`
- [ ] Update `run_experiment.py` to read profile from config and pass to provider
- [ ] Update `MLProvider.summarize()` to accept `preprocessing_profile` param
- [ ] Update `summarize_long_text()` to call `apply_profile()` instead of `clean_for_summarization()`
- [ ] Update `OpenAIProvider.summarize()` similarly (if applicable)

### Parameter Exposure (Future)

- [ ] Add `encoder_no_repeat_ngram_size` to `GenerationParams` schema
- [ ] Expose `repetition_penalty` in experiment config (currently hardcoded at 1.3)

### Experiment Configs

- [x] `baseline_bart_v2_longer_output.yaml` (created)
- [x] `baseline_bart_v3_stronger_ngram.yaml` (created)
- [x] `baseline_bart_v4_smaller_chunks.yaml` (created)
- [x] `baseline_bart_v5_larger_chunks.yaml` (created)
- [x] `baseline_bart_v6_combined_best.yaml` (created)
- [ ] `baseline_bart_v7_cleaning_v4.yaml`
- [ ] `baseline_bart_v8_cleaning_v4_optimized.yaml`

### Documentation

- [ ] Update `docs/guides/ML_MODEL_COMPARISON_GUIDE.md`
- [ ] Update `docs/guides/EXPERIMENT_GUIDE.md`

## References

- **Source Documents**:
  - `docs/wip/preprocessing_improvements_plan.md`
  - `docs/wip/baseline_bart_experiment_plan.md`
- **Preprocessing Code**: `src/podcast_scraper/preprocessing/core.py`
- **Profile Registry**: `src/podcast_scraper/preprocessing/profiles.py`
- **Experiment Configs**: `data/eval/configs/`
- **Baseline Results**: `data/eval/runs/`
