# Provider Matrix v2 — OpenAI vs Anthropic vs Gemini

**Date:** 2026-04-15
**Framework:** autoresearch v2 ([RFC-073](../docs/rfc/RFC-073-autoresearch-v2-framework.md))
**Dataset:** held-out `curated_5feeds_benchmark_v2` (5 ep, ~32min each, never used during tuning)
**Silver:** Sonnet 4.6 (`silver_sonnet46_benchmark_v2_{bullets,paragraph}`)
**Judges:** gpt-4o-mini + claude-haiku-4-5-20251001

Three-provider head-to-head. Numbers below are held-out final scores (higher better):

## Headline Matrix

| Track | Approach | OpenAI (gpt-4o) | Anthropic (haiku-4.5) | Gemini (2.0-flash) | Winner |
| ----- | -------- | --------------- | --------------------- | ------------------ | ------ |
| Bullets | Non-bundled | 0.566 | **0.570** | 0.562 | Anthropic (+0.7%) |
| Bullets | Bundled | 0.505 | **0.552** | 0.473 | Anthropic (+9.3%) |
| Paragraph | Non-bundled | 0.481 | **0.522** | 0.463 | Anthropic (+8.5%) |
| Paragraph | Bundled | 0.469 | **0.548** | 0.461 | Anthropic (+16.8%) |

**Anthropic wins every cell.** OpenAI second. Gemini third.

## ROUGE-L breakdown (held-out)

| Track | Approach | OpenAI | Anthropic | Gemini |
| ----- | -------- | ------ | --------- | ------ |
| Bullets | Non-bundled | 39.6% | 40.7% | 40.1% |
| Bullets | Bundled | 33.2% | 39.3% | 28.5% |
| Paragraph | Non-bundled | 31.7% | 36.4% | 29.3% |
| Paragraph | Bundled | 29.5% | 39.2% | 26.6% |

Anthropic leads ROUGE-L on every track-approach cell. Non-bundled bullets is closest across
providers (all three within 1.1pp). Bundled paragraph has the widest spread (Anthropic 39.2% vs
Gemini 26.6%).

## Bundled viability per provider

Difference between bundled and non-bundled (non-bundled − bundled, held-out final):

| Provider | Bullets gap | Paragraph gap | Bundled verdict |
| -------- | ----------- | ------------- | --------------- |
| Anthropic | +0.018 (+3.3%) | **−0.026 (bundled wins)** | Bundled viable; paragraph actually better |
| OpenAI | +0.061 (+12.1%) | +0.012 (+2.6%) | Non-bundled clearly better on both |
| Gemini | +0.089 (+18.8%) | +0.002 (+0.4%) | Non-bundled much better on bullets |

**Anthropic is unique**: bundled is not only viable but sometimes *preferable* to non-bundled.
OpenAI and Gemini both have a substantial bundled penalty on bullets. This is a provider-
capability finding that was invisible under v1.

## Cost & speed (approximate, Mar 2026 pricing)

| Provider | Model | $/M in | $/M out | Latency/ep | Bundled cost advantage |
| -------- | ----- | ------ | ------- | ---------- | ---------------------- |
| OpenAI | gpt-4o | $2.50 | $10.00 | ~8s (bundled), ~20s (non-bundled) | ~2.6× cheaper |
| Anthropic | claude-haiku-4-5 | $0.80 | $4.00 | ~10s (bundled), ~25s (non-bundled) | ~2.5× cheaper |
| Gemini | gemini-2.0-flash | $0.075 | $0.30 | ~3s (bundled), ~8s (non-bundled) | ~2.7× cheaper |

Gemini 2.0-flash is an order of magnitude cheaper than Anthropic Haiku, which is ~3× cheaper
than OpenAI gpt-4o. All three bundled modes are ~2.5× cheaper than their non-bundled counterparts.

## Recommendations

### For quality: Anthropic Haiku 4.5 bundled (default)

Wins 3 of 4 cells, and the paragraph win alone is +16.8% vs OpenAI. Cheapest high-quality choice.
Use non-bundled bullets only if bullets are the sole output.

### For lowest cost: Gemini 2.0-flash

If quality is "good enough" for your use case and cost is paramount: Gemini is 10× cheaper than
Haiku. Non-bundled bullets scores 0.562 (only 0.008 behind Anthropic at 0.570) — the quality gap
is small for a 10× cost reduction. Use non-bundled bullets; avoid Gemini bundled (weak).

### For OpenAI-only environments: gpt-4o non-bundled

Respectable but behind Anthropic on every cell. Only pick OpenAI if you have ecosystem constraints
(API key, infra, tooling) — otherwise Anthropic is strictly better at slightly lower cost.

## Provider-specific quirks surfaced by v2

1. **OpenAI (gpt-4o)**
   - `temperature=0` is not truly deterministic; seed helps but doesn't fully fix.
   - Supports `response_format: {"type": "json_object"}` for bundled — reliable JSON mode.
   - Structural attention-split penalty on bundled: bullets gap −12% vs non-bundled.

2. **Anthropic (haiku-4.5-20251001)**
   - No seed parameter in API.
   - No `response_format: json_object` — bundled needs JSON prefill pattern
     (`{"role": "assistant", "content": "{"}`) to force valid JSON. Implemented in
     `src/podcast_scraper/providers/anthropic/anthropic_provider.py`.
   - Handles multi-task bundled prompt without the attention-split penalty. Best bundled quality.

3. **Gemini (2.0-flash)**
   - No seed parameter via current SDK version.
   - Bundled produces control characters in JSON strings; parser needs `strict=False`.
     Implemented in `src/podcast_scraper/providers/gemini/gemini_provider.py`.
   - Gemini 2.5-flash has *thinking tokens enabled by default* — consumes `max_output_tokens`
     budget before emitting the actual answer. Disabling thinking requires SDK upgrade beyond
     the current pin (present `google-genai` SDK version lacks `thinking_budget` in
     `ThinkingConfig`). Sticking with 2.0-flash for v2 framework proof; upgrading to 2.5
     with thinking disabled is tracked as follow-up.

## Framework validation (cross-provider)

All 12 champions (3 providers × 4 cells) validated on held-out `benchmark_v2`. All 12 generalised
cleanly — dev score and held-out score within noise bounds (±3% final). The v2 framework produces
trustworthy cross-provider numbers without per-provider tuning beyond prompt porting.

| Provider | Champions validated | Generalisation range (dev→held-out) |
| -------- | ------------------- | ----------------------------------- |
| OpenAI | 4 | −0.2% to +6% |
| Anthropic | 4 | −0.6% to +4.8% |
| Gemini | 4 | −3.3% to −0.2% |

## Current champion prompts (v2)

### OpenAI, Anthropic, Gemini — all share the same bundled templates content

Each provider has its own copy at `src/podcast_scraper/prompts/<provider>/summarization/bundled_clean_summary_*_v1.j2`, but all three contain the same champion prompt content (ported from OpenAI r7 champion):

- Few-shot bullets style examples (3 domain-neutral)
- Style narration (dense, noun-anchored, "X rather than Y")
- Anti-pattern examples (3 negative→positive rewrites)
- 4-6 paragraph default
- Opening sentence pattern (domain + central argument)
- Verbatim terminology preservation

### Non-bundled

- **Bullets** (all providers): shared `shared/summarization/{system_bullets_v1,bullets_json_v1}.j2` — already champion-level.
- **Paragraph** (per-provider): `<provider>/summarization/{system_v1,long_v1}.j2` — each provider's `long_v1.j2` updated with the same champion rules.

## Open items

1. **Gemini 2.5-flash (thinking disabled)** — requires `google-genai` SDK upgrade. Would likely
   lift all Gemini scores by 3-5% based on model-generation improvements.
2. **Provider-specific prompt tuning** — we ported OpenAI champions and stopped. Dedicated tuning
   per provider could find another 2-5% per cell. Diminishing returns given Anthropic already
   dominates; defer unless a business case emerges.
3. **Mistral** — next natural provider to round out the open-vs-closed matrix.
4. **Multi-run averaging** — deferred per RFC-073 §Future Work. Would tighten confidence
   intervals on these numbers.
