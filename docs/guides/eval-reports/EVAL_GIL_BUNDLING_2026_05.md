# GIL Evidence Stack Bundling — Cross-Provider Matrix (2026-05)

> **Snapshot:** 2026-05-03 · 7 providers (Gemini + 5 cloud + 3 Ollama
> models) · `curated_5feeds_benchmark_v2` · vs `silver_sonnet46_gi_multiquote_benchmark_v2`
> · #698 / PR #711.

This report measures the impact of bundling the two LLM-bound substages
in the Grounded Insight List (GIL) evidence stack:

- **Layer A (`extract_quotes_bundled`):** all insights → 1 LLM call.
- **Layer B (`score_entailment_bundled`):** all (insight, candidate) NLI
  pairs → chunked LLM call (chunk_size=15).

Baseline staged shape issues ~76 sequential LLM calls per episode (~14
extract + ~62 NLI). Question: how much of that can we cut without
regressing coverage vs Sonnet-4.6 silver?

## Methodology

- **Dataset:** `curated_5feeds_benchmark_v2` (5 episodes, 5 feeds).
- **Silver:** `silver_sonnet46_gi_multiquote_benchmark_v2` (Sonnet 4.6, 40
  silver insights total).
- **Scorer:** `autoresearch/gil_evidence_bundling/eval/score_gi_vs_silver.py`
  (embedding cosine via `all-MiniLM-L6-v2`; threshold 0.55 = covered).
  This is a GI-specific scorer mirroring the canonical
  `score_gi_insight_coverage.py` but reading insights from
  `output.gil.nodes[type=Insight]` instead of summary bullets.
- **Cells per cloud provider:** 4 (staged baseline / Layer A only / Layer
  B only / both bundled). Cells per Ollama model: 1 (`bundled_ab` only —
  staged is operationally unviable on local CPU; see "Ollama treatment"
  below).
- **Champion gates:** silver coverage ≥ 70%, fallback rate ≤ 20%, ≥ 30%
  cost reduction vs staged.

## Cross-provider results

### Cloud providers

| Provider | Model | Staged cov | bundled_a | bundled_b | bundled_ab | Champion | Calls cut |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | ---: |
| Gemini | 2.5-flash-lite | 75% | 80% | 82% | **82%** | bundled_ab | 92% |
| Anthropic | claude-haiku-4-5 | 77.5% | 82.5% | 80% | **82.5%** | bundled_ab | 92% |
| OpenAI | gpt-4o-mini | 72% | 77.5% | 72.5% | **72.5%** | bundled_ab | 92% |
| DeepSeek | deepseek-chat | 72% | 75% | 78% | **78%** | bundled_ab | 92% |
| Grok | grok-3-fast | 90% | 87.5% | 87.5% | **87.5%** | bundled_ab | 92% |
| Mistral | mistral-small-latest | 70% | 70% | **80%** | 67.5% | bundled_b_only | 75% |

**Latency (all cells):**

| Provider | Staged | bundled_ab | Δ |
| :--- | ---: | ---: | ---: |
| Gemini | 95.6s | 36.0s | -62% |
| Anthropic | 42.9s | 17.9s | -58% |
| DeepSeek | 51.3s | 32.0s | -38% |
| OpenAI | 38.8s | 28.1s | -28% |
| Grok | 41.6s | 34.4s | -17% |
| Mistral | 35.3s | 28.1s (bundled_b 28.1s) | -20% |

### Ollama (local)

Bundled-only matrix; staged not measured (per-call overhead × 76 calls
= 30-40 min/ep, not a representative deployment).

| Model | Params | Coverage | Grounded | Latency/ep | Notes |
| :--- | ---: | ---: | ---: | ---: | :--- |
| qwen3.5:9b | 9B | **72%** | 72% | 122s | Ollama champion; clean run, no fallbacks |
| mistral-small3.2 | 24B | 70% | 70% | 852s | Comparable quality; 7x slower than qwen on local CPU |
| llama3.2:3b | 3B | 62% | 45% | 60s | Below 70% gate; 1+ JSON parse fallback observed (3B model fragile on bundled prompts at chunk_size=15) |

**Ollama-specific finding:** bundled prompts cross Ollama's default 2048
context window. Bundled methods on `OllamaProvider` pass `num_ctx=32k`
explicitly via `_ollama_openai_chat_extra_kwargs`. Without that fix,
mistral-small3.2 silently truncated input and timed out on retries.

## Champion call per provider

`bundled_ab` is the default champion (5 of 7 providers). Mistral is the
exception: Layer A regresses Mistral's coverage from 70% → 67.5%, so
Mistral pins to `bundled_b_only` (75% call cut, +10pp coverage). Llama3.2:3b
on Ollama passes the operational viability test but falls below the
70% silver coverage gate and is documented as "fragile" (use chunk_size=8
or pin to qwen3.5:9b for higher quality).

OpenAI is a notable case study: pure-quality champion would be
`bundled_a_only` at 77.5% coverage vs `bundled_ab` at 72.5%. Operator
explicitly chose `bundled_ab` accepting the 5pp drop in exchange for the
extra call cut and rate-limit-pressure relief. This decision is recorded
in [ADR-078](../../adr/ADR-078-gil-evidence-bundling-per-provider-champions.md)
and the per-cell rationale is in `autoresearch/gil_evidence_bundling/results.tsv`.

## Surprises

1. **`bundled_ab` improved coverage on most providers, not just held it.**
   Hypothesis: the bundled extract prompt asking for 3-5 distinct quotes
   per insight surfaces more candidate evidence than the staged
   one-quote-per-call prompt did. The +7pp coverage boost on Gemini was
   the first signal; held up on Anthropic (+5pp), DeepSeek (+6pp).

2. **Mistral regression on Layer A is provider-specific, not noise.**
   Bundled extract on Mistral under-extracts when asked for many insights
   at once. Layer B alone (NLI bundling) preserves the gain. Track A
   (provider-specific Jinja prompts) is reserved as follow-up if
   needed; the `bundled_b_only` config-level override sidesteps it for now.

3. **Ollama on local CPU works.** With `num_ctx=32k` the bundled stage
   completes cleanly on 9B and 24B models within minutes, not hours.
   This is the first time the GIL evidence stage is operationally
   feasible on local Ollama.

## Reproducing

Per-cell experiment YAMLs are in
`autoresearch/gil_evidence_bundling/experiments/`. Run + score pattern:

```bash
PYTHONPATH=. .venv/bin/python scripts/eval/experiment/run_experiment.py \
    autoresearch/gil_evidence_bundling/experiments/<cell>_<provider>.yaml \
    --reference silver_sonnet46_gi_multiquote_benchmark_v2 \
    --force --log-level INFO

PYTHONPATH=. .venv/bin/python autoresearch/gil_evidence_bundling/eval/score_gi_vs_silver.py \
    --run-id gil_bundling_<cell>_<provider>_v1 \
    --silver silver_sonnet46_gi_multiquote_benchmark_v2 \
    --dataset curated_5feeds_benchmark_v2
```

Full results log: `autoresearch/gil_evidence_bundling/results.tsv` (28
rows: 4 Gemini cells + 4×5=20 cloud-provider cells + 3 Ollama cells).

## Decision

See [ADR-078](../../adr/ADR-078-gil-evidence-bundling-per-provider-champions.md).
Profile defaults flipped in PR #711 (`cloud_balanced`, `cloud_thin`,
`cloud_quality`, `local`); airgapped profiles use local CrossEncoder
and are unaffected. CLI `--config` path also patched to forward the
new fields (`_gil_tuning_keys` in `cli.py`).
