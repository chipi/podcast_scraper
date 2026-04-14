# ML Model Comparison Guide

## CURRENT DECISIONS (Feb 2026)

**These are the active, validated choices for the system.** Everything else in this document is context and reference.

### Dev ML Authority (Smoke / Fast Feedback)

- **MAP:** `facebook/bart-base`
- **REDUCE:** `allenai/led-base-16384`
- **Status:** Stable, fast, smoke-validated
- **Use when:** local development, iteration, debugging

### Prod ML Authority (Benchmark-validated)

- **MAP:** `google/pegasus-cnn_dailymail`
- **REDUCE:** `allenai/led-base-16384`
- **Status:** Benchmark-validated, clean gates, stable output
- **Use when:** production ML summarization

### LongT5 (8k context) — MAP option (RFC-042 / Issue #353)

- **Models:** `google/long-t5-tglobal-base` (alias `longt5-base`), `google/long-t5-tglobal-large` (alias `longt5-large`)
- **Context window:** 8,192 tokens (between BART/PEGASUS 1k and LED 16k)
- **Use when:** MAP compression for medium-long transcripts (2k–8k tokens) where LED is overkill

### Hybrid MAP-REDUCE (RFC-042 / Issue #352)

- **Provider:** `summary_provider: hybrid_ml`
- **MAP:** classic summarizer (recommended default: `longt5-base`, fallback to LED for very long)
- **REDUCE:** instruction-tuned model (Tier 1: `google/flan-t5-base` via transformers; Tier 2: via Ollama)

> Any change to preprocessing, chunking, or generation semantics requires a new baseline version.

---

This file is intentionally minimal at the top. The remainder of the guide
continues with the detailed comparison tables, rationale, and historical context
unchanged from v2.
