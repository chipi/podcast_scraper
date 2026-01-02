# Local Summarization for Long Podcast Transcripts on MBP M4 Pro (48GB)

## Hardware Specifications

You have:

- Apple MacBook Pro M4 Pro
- 14 cores, 48GB unified memory
- No external API (local-only summarization)

## Content Characteristics

Your content:

- Podcast transcripts of 30–40k tokens (very long)

This document revisits earlier suggestions with your hardware in mind, and proposes realistic, efficient options.

---

## 1. What Your Hardware Enables

With 48GB unified memory and Apple Silicon:

- You can comfortably run:
  - HuggingFace encoder–decoder models (BART, PEGASUS, LED, LongT5) with Metal (mps) acceleration.
  - Local LLMs around 7B–14B parameters using:
    - ollama
    - llama.cpp (GGUF)
    - mlx / mlx-lm
- You cannot treat 30–40k tokens as a single input for most models.
- But you can run a multi-stage, map–reduce summarization pipeline efficiently.

### Strategy

Use your Mac's memory to run better local models + smarter chunking, not to brute-force giant contexts.

---

## 2. Recommended Model Options (Ranked)

### Option A – Local LLMs (7B–14B) via Ollama / llama.cpp (Recommended)

Use a modern instruction-tuned LLM with decent reasoning for summarization, e.g.:

- `llama3:8b`
- `mistral:7b-instruct`
- `qwen2:7b-instruct` (if available in your stack)

**Advantages:**

- Better abstractive summarization than BART/LED.
- Easy to use with map–reduce: send prompts like "Summarize this podcast segment in 5 bullet points".
- Your 48GB RAM is perfect for this (especially with 4-bit / Q4 quantization).

**Tradeoffs:**

- Slower than tiny encoder–decoder models, but still reasonable on M4.
- You'll likely have 8k–16k context, so you still need chunking.

---

### Option B – Encoder–Decoder Models (BART / PEGASUS) with Map–Reduce

Continue with transformers, but:

- Use Metal acceleration with `device="mps"`.
- Use map–reduce chunking (as already discussed).
- Consider PEGASUS models, which were trained directly for summarization.

**Example models:**

- `google/pegasus-xsum` (short summaries)
- `google/pegasus-large`
- `facebook/bart-large-cnn`

**Pros:**

- Well-documented, easy to integrate.
- Very stable runtimes on CPU/MPS.

**Cons:**

- Weaker high-level reasoning than good LLMs.
- Need careful chunking to avoid losing nuance.

---

### Option C – Long-Context Encoder–Decoder (LED / LongT5) with Chunking

You can try:

- `allenai/led-base-16384`
- `google/long-t5-local-base` (and variants)

They:

- Allow longer chunks than BART.
- Still need chunking for 30–40k tokens.
- Require correct attention setup to perform well.

**Pros:**

- Fewer chunks → fewer passes.
- Better "global context" per chunk.

**Cons:**

- Trickier to configure (global attention, etc.).
- Quality not dramatically better than Option B in practice, given complexity.

---

## 3. Concrete Architecture for Your Setup

### 3.1. Clean the Transcript

Preprocessing is essential:

- Strip timestamps like `[00:12:34]`
- Remove or normalize `Speaker 1:`, `Host:`, etc.
- Collapse excessive blank lines.
- Optionally remove filler tokens (uh, you know, etc.).

You can do this once per transcript; it's cheap and improves everything downstream.

---

### 3.2. Map–Reduce Chunking Strategy (General)

Assuming word-based approximation:

- **Chunk size:** 800–1200 words for encoder–decoder models
  (for LLMs with larger context, you can go bigger: e.g. 2500–3500 words)

- **Overlap:** 100–200 words to avoid boundary loss
- **Map step:** summarize each chunk into:
  - 3–7 bullet points, or
  - 1 short paragraph (depending on needs)
- **Reduce step:** summarize concatenated chunk summaries into:
  - 1–3 paragraphs + bullet key takeaways.

---

## 4. Example: BART + MPS on Your Mac

You can still get decent summaries with BART if you structure it right.

````python
from transformers import pipeline

# Use device="mps" to leverage Apple Silicon

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device="mps",
)

def chunk_text_words(text, chunk_size=900, overlap=150):
    words = text.split()
    chunks = []
    start = 0
    n = len(words)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks

def summarize_chunk(chunk, max_length=160, min_length=60):
    out = summarizer(
        chunk,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
        truncation=True,
    )
    return out[0]["summary_text"].strip()

def summarize_transcript_bart_map_reduce(text: str) -> str:

```text
    # 1) Chunk
```

```text
    chunks = chunk_text_words(text, chunk_size=900, overlap=150)
```

```text
    # 2) Map: summarize each chunk
```

```text
    partial_summaries = [summarize_chunk(c) for c in chunks]
```

```text
    # 3) Reduce: summarize the summaries
```

```text
    joined = "\n\n".join(partial_summaries)
```

```text
    # Optional: re-chunk joined if it gets too long
```

    final = summarizer(
        joined,
        max_length=260,
        min_length=100,
        do_sample=False,
        truncation=True,
    )
    return final[0]["summary_text"].strip()
```text

## 5. Example: Using a Local LLM via CLI (Ollama-style Pseudocode)

If you install something like ollama and run `llama3:8b`, you can do:

Python pseudo-API pattern (you'd use whatever client/lib exists):

```python
def summarize_chunk_llm(chunk: str) -> str:
    prompt = f"""
    You are an assistant that summarizes podcast segments.

    Summarize the following segment in 5 concise bullet points
    covering the main ideas, decisions, and insights.

    Segment:
    \"\"\"{chunk}\"\"\"
    """

    # call your local LLM runtime – pseudo code:

    response = call_local_llm(
        model="llama3:8b",
        prompt=prompt,
        max_tokens=256,
    )
    return response.strip()
```text

- `chunk_text_words(...)`
- `summarize_chunk_llm(...)`
- Concatenate bullet-point summaries
- One more LLM call to get the final, high-level summary.

With a good 7B–8B LLM quantized on your M4 Pro, this can be:

- Noticeably better than BART/PEGASUS
- Still fully offline

---

## 6. Which Route Should You Take?

Given your hardware and constraints:

### ✅ Recommended Path

1. **Short term (fastest to implement)**
   - Use BART or PEGASUS with:
     - MPS acceleration
     - Map–reduce chunking
   - Implement a clean pipeline that outputs:
     - Short abstract
     - Optional extended bullet notes

2. **Medium term (better quality)**
   - Set up a local LLM runtime (Ollama / llama.cpp / mlx-lm).
   - Use a 7B–14B model with 8k–16k context.
   - Replace the BART chunk summarizer with LLM prompts.
   - Keep the same chunking logic.

3. **Future (max quality)**
   - Experiment with larger or better models (e.g. 14B, mixture-of-experts) if you can tolerate slower inference.
   - Possibly mix:
     - Local encoder–decoder for fast first-pass compression
     - Local LLM for final refinement of summaries.

---

## 7. TL;DR for Your MBP M4 Pro

- You can't brute-force 30–40k tokens into a single summarization model (even with 48GB RAM).
- Your hardware is great for:
  - Running multiple summarization passes on chunks.
  - Running 7B–14B local LLMs with decent speed.
- The winning combination is:
  - Pre-clean transcript
  - Map–reduce chunking
  - Local model:
    - BART/PEGASUS now
    - Local LLM later

````
