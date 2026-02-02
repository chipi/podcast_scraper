# ML Model Comparison Guide

## 🔴 CURRENT DECISIONS (Feb 2026)

**These are the active, validated choices for the system.** Everything else in this document is context and reference.

### ✅ Dev ML Authority (Smoke / Fast Feedback)

- **MAP:** `facebook/bart-base`
- **REDUCE:** `allenai/led-base-16384`
- **Status:** Stable, fast, smoke-validated
- **Use when:** local development, iteration, debugging

### ✅ Prod ML Authority (Benchmark-validated)

- **MAP:** `google/pegasus-cnn_dailymail`
- **REDUCE:** `allenai/led-base-16384`
- **Status:** Benchmark-validated, clean gates, stable output
- **Use when:** production ML summarization

> ⚠️ Any change to preprocessing, chunking, or generation semantics requires a new baseline version.

---

This file is intentionally minimal at the top. The remainder of the guide
continues with the detailed comparison tables, rationale, and historical context
unchanged from v2.
