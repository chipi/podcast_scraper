# ADR-071: Four-Tier Summarization Strategy

- **Status**: Accepted
- **Date**: 2026-04-04
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-057](../rfc/RFC-057-autoresearch-optimization-loop.md),
  [RFC-042](../rfc/RFC-042-hybrid-summarization-pipeline.md)
- **See Also**: [ADR-068](ADR-068-bart-led-as-ml-production-baseline.md),
  [ADR-069](ADR-069-hybrid-ml-pipeline-as-production-direction.md),
  [ADR-070](ADR-070-bart-base-as-hybrid-map-stage.md)

## Context & Problem Statement

RFC-057 Track B autoresearch produced multiple promoted summarization modes across different
architectural families: pure ML (HuggingFace MAP+REDUCE), hybrid ML (HF MAP + Ollama REDUCE),
direct LLM (full transcript → Ollama), and cloud LLM. A direct comparison experiment
(`direct_llama32_3b_benchmark_paragraph_v1`) revealed that the hybrid MAP stage is not
beneficial for current episode lengths — direct Ollama outperforms hybrid on all metrics.

This created an opportunity to reframe all modes as a deliberate four-tier strategy, where
each tier serves a different deployment constraint rather than one being strictly "better"
than another.

## Decision

Adopt a **four-tier summarization strategy** where each tier has a distinct role, deployment
constraint, and quality/cost profile. No single tier is the universal default — the right
choice depends on deployment context.

## The Four Tiers

### Tier 1 — ML Dev (`ml_small_authority`)

```text
Mode ID:    ml_small_authority
MAP:        facebook/bart-base
REDUCE:     allenai/led-base-16384
ROUGE-L:    ~14%
Latency:    fast (CPU-friendly)
Deps:       HuggingFace only, no GPU required
```

**Purpose:** Fast iteration in development and CI. No Ollama, no GPU, no cloud keys required.
Deterministic outputs (no sampling). Used in unit/integration tests and smoke CI runs.

---

### Tier 2 — ML Prod (`ml_bart_led_autoresearch_v1`)

```text
Mode ID:    ml_bart_led_autoresearch_v1
MAP:        facebook/bart-base
REDUCE:     allenai/led-base-16384
ROUGE-L:    20.4%  (benchmark, 10 eps)
Latency:    30.6s/ep
Deps:       HuggingFace only — no Ollama, no internet, no API keys
```

**Purpose:** Maximum privacy and self-containment. Runs in air-gapped environments,
Docker containers without network, or any context where external daemons cannot be
guaranteed. The only tier with zero runtime dependencies beyond Python + HuggingFace cache.

**Autoresearch sweep result**: +4.26% over dev baseline; `max_new_tokens=550`, `num_beams=6`.

---

### Tier 3 — LLM Local (`llama3.2:3b` via Ollama direct)

```text
Config:     direct_llama32_3b_autoresearch_v1.yaml  (autoresearch winner)
Model:      llama3.2:3b (Ollama)
ROUGE-L:    26.4%  (benchmark, 10 eps)
Embedding:  79.5%
Latency:    7.5s/ep
Deps:       Ollama daemon, llama3.2:3b pulled (~2GB)
Params:     temperature=0.3, max_length=1000, min_length=200
```

**Purpose:** Best fully-local quality. No cloud, no variable cost, full data privacy.
Faster than ML Prod (7.5s vs 30.6s) because Ollama inference on Apple Silicon MPS
outperforms LED beam search. Suitable for production deployments where Ollama can be
guaranteed to be running.

**Key finding**: Direct LLM *beats* the hybrid BART+Llama pipeline (26.4% vs 23.7% ROUGE-L,
79.5% vs 72.9% embed, 7.5s vs 15.7s latency). The BART MAP compression stage is lossy —
the LLM synthesises better from the full transcript than from compressed chunk notes,
for current episode lengths (~10-15K transcript tokens).

**Autoresearch sweep result**: +13.0% relative gain from temperature tuning alone.
temperature=0.3 is optimal for direct inference (focused generation on clean transcript),
whereas the hybrid sweep winner was temperature=0.5 (BART chunk noise required more diversity).

---

### Tier 4 — LLM Cloud (Anthropic Claude Sonnet 4.6)

```text
Provider:   Anthropic (or OpenAI GPT-4o)
ROUGE-L:    ~32.6%  (Anthropic), ~28-32% (OpenAI)
Embedding:  ~85%
Latency:    ~3s/ep
Deps:       ANTHROPIC_API_KEY, internet, variable cost
```

**Purpose:** Highest quality ceiling. Justified for premium content, business use cases,
or when the ~8pp quality gap vs Tier 3 is worth the cost and privacy trade-off.
Also used as the silver reference generator for autoresearch evaluation.

---

## Why Hybrid Is Not a Tier

The hybrid ML pipeline (`ml_hybrid_bart_llama32_3b_autoresearch_v1`) was the
intermediate discovery that led to Tier 3. It is retained in `model_registry.py` as a
documented, promotable mode but is not a primary deployment tier because:

1. Direct LLM (Tier 3) is strictly better: +0.6pp ROUGE-L, +3.7pp embed, 2x faster
2. The MAP stage adds latency without quality benefit for current episode lengths
3. It has more moving parts (HF model + Ollama) than either pure ML or direct Ollama

**When hybrid might become relevant again:** If episode lengths grow beyond ~15K tokens
(~2h episodes), the LLM context window becomes a constraint and MAP compression would
be architecturally necessary. At that point, the hybrid tier would be reconsidered.

## Measured Comparison (curated_5feeds_benchmark_v1, 10 episodes)

| Tier | Mode | ROUGE-L | Embed | Tokens | Latency | Privacy | Cost |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 — ML Dev | ml_small_authority | ~14% | ~65% | ~185 | fast | 100% local | $0 |
| 2 — ML Prod | ml_bart_led_autoresearch_v1 | 20.4% | 70.1% | 265 | 30.6s | 100% local | $0 |
| — | ml_hybrid_bart_llama32_3b_autoresearch_v1 | 23.7% | 72.9% | 441 | 15.7s | 100% local | $0 |
| 3 — LLM Local | direct_llama32_3b_autoresearch_v1 | **26.4%** | **79.5%** | 615 | **7.5s** | 100% local | $0 |
| 4 — LLM Cloud | Anthropic Claude Sonnet 4.6 | ~32.6% | ~85% | ~420 | ~3s | ☁️ cloud | variable |

## Constant Assignments

```python
# config_constants.py
DEV_DEFAULT_SUMMARY_MODE_ID   = "ml_small_authority"           # Tier 1
PROD_DEFAULT_SUMMARY_MODE_ID  = "ml_bart_led_autoresearch_v1"  # Tier 2 (no-dep guarantee)
OLLAMA_DEFAULT_SUMMARY_MODEL  = "llama3.2:3b"                  # Tier 3 (direct, when available)
```

`PROD_DEFAULT_SUMMARY_MODE_ID` points to Tier 2 (pure ML) because "prod default" means
"guaranteed to work in any deployment without external daemons". Tier 3 is the *recommended*
local option when Ollama is available, but cannot be the hard default since it requires
an external process.

## Consequences

- **Positive**: Each tier has a clear, non-overlapping deployment contract. No ambiguity
  about which mode to choose for a given constraint.
- **Positive**: Hybrid mode demoted from default path — removes misleading implication that
  more complexity = better quality.
- **Positive**: Direct LLM (Tier 3) is simpler than hybrid: one model, one API call,
  no chunking, no MAP phase, faster results.
- **Neutral**: `PROD_DEFAULT_SUMMARY_MODE_ID` reverts to `ml_bart_led_autoresearch_v1`.
  Any deployment relying on the hybrid default must explicitly set Tier 3 config.
- **Neutral**: Hybrid mode remains fully functional and registered — useful for very long
  episodes or when context window constraints apply.
- **Negative**: Tier 3 quality ceiling (~24%) is still ~8pp below cloud (Tier 4). Users
  needing highest quality must accept cloud dependency.

## Implementation Notes

- **Tier 1 registry**: `model_registry.py` → `ml_small_authority`
- **Tier 2 registry**: `model_registry.py` → `ml_bart_led_autoresearch_v1`
- **Tier 3 config**: `data/eval/configs/summarization/direct_llama32_3b_autoresearch_v1.yaml` (autoresearch winner; benchmark baseline: `direct_llama32_3b_benchmark_paragraph_v1.yaml`)
- **Tier 3 constant**: `config_constants.py` → `OLLAMA_DEFAULT_SUMMARY_MODEL = "llama3.2:3b"`
- **Hybrid (archived)**: `model_registry.py` → `ml_hybrid_bart_llama32_3b_autoresearch_v1`

## References

- [RFC-057: AutoResearch Optimization Loop](../rfc/RFC-057-autoresearch-optimization-loop.md)
- [RFC-042: Hybrid Podcast Summarization Pipeline](../rfc/RFC-042-hybrid-summarization-pipeline.md)
- [ADR-068: BART+LED as ML Production Baseline](ADR-068-bart-led-as-ml-production-baseline.md)
- [ADR-069: Hybrid ML Pipeline as Production Direction](ADR-069-hybrid-ml-pipeline-as-production-direction.md)
- [ADR-070: BART-base as Hybrid MAP Stage](ADR-070-bart-base-as-hybrid-map-stage.md)
- Experiment: `data/eval/configs/summarization/direct_llama32_3b_benchmark_paragraph_v1.yaml`
