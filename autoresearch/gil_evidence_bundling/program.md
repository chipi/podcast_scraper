# AutoResearch — GIL Evidence Bundling (#698)

## Objective

Validate the #698 bundling hypothesis on real episodes: can Layer A (bundled
``extract_quotes``) and Layer B (bundled ``score_entailment``) cut LLM calls
per episode without dropping grounding quality below operator-acceptable
levels?

Targets:

- **Layer A:** ~14 ``extract_quotes`` calls per episode → 1 bundled call.
- **Layer B:** ~62 ``score_entailment`` calls per episode → ~5 chunked
  bundled calls (chunk_size=15).
- **Combined:** ~76 sequential calls/episode → ~6.

Quality gates:

- ``grounding_rate`` (insights with ≥1 grounded quote) — must not regress
  more than 5 percentage points absolute on any feed in the dev set.
- ``quotes_per_insight_mean`` — should stay within ±20% of the staged
  baseline.
- ``mean_nli_score`` — should stay within ±0.05 of the staged baseline.

Cost gate:

- ``llm_gi_extract_quotes_cost_usd`` + ``llm_gi_score_entailment_cost_usd``
  combined must drop ≥30% vs the staged baseline (the stretch is ≥60%).

## Cells

Four cells in the matrix, one Gemini Flash Lite provider in V1:

| Cell | quote_mode | nli_mode | Expected calls/ep | Notes |
| --- | --- | --- | --- | --- |
| baseline_staged | staged | staged | ~76 | Reference; identical to current production |
| bundled_a_only | bundled | staged | ~63 | 14 extracts → 1; NLI unchanged |
| bundled_b_only | staged | bundled | ~19 | extracts unchanged; NLI 62 → ~5 |
| bundled_ab | bundled | bundled | ~6 | The headline win |

Each cell runs against:

- **Dev:** ``curated_5feeds_dev_v1`` (10 episodes, includes ``omnycontent``
  feed which is the canary for ad-heavy grounding regressions).
- **Held-out:** ``curated_5feeds_benchmark_v2`` (5 episodes), only after
  a champion is picked from dev.

## Score formula

Single scalar emitted per cell run by ``eval/score.py``:

```text
score = 0.5 × cost_reduction
      + 0.3 × grounding_preservation
      + 0.2 × latency_reduction
```

Where:

- ``cost_reduction = 1 - (variant_cost / baseline_cost)``, clamped to [0, 1].
- ``grounding_preservation = min(1.0, variant_grounding_rate / baseline_grounding_rate)``.
- ``latency_reduction = 1 - (variant_total_seconds / baseline_total_seconds)``,
  clamped to [0, 1].

Champion ≥ 0.30 (i.e. positive net reduction with quality preserved). Higher = better.

## Failure modes guarded

1. **Token explosion** — bundled prompts too long for Gemini Flash Lite's
   1M context. Mitigated by ``gil_evidence_nli_chunk_size=15`` default;
   ``score.py`` aborts if ``llm_gi_extract_quotes_input_tokens`` per episode
   exceeds 50000.
2. **Grounding regression on ad-heavy feeds** — ``omnycontent`` baseline
   58% must not drop below 50% absolute. Per-feed assertion in ``score.py``.
3. **JSON parser fragility** — bundled fallbacks tracked via
   ``gi_evidence_extract_quotes_bundled_fallbacks`` and
   ``gi_evidence_score_entailment_bundled_fallbacks``. Fallback rate >20%
   means the bundled prompt isn't reliable enough; cell is rejected.

## Out of scope (V1)

- Other providers (OpenAI, Anthropic, Mistral, DeepSeek, Grok, Ollama).
  Generalising once Gemini champion is picked.
- Per-prompt-variant tuning. V1 uses the prompt baked into the provider's
  ``extract_quotes_bundled`` / ``score_entailment_bundled`` methods. Prompt
  tuning is a follow-up RFC-073 Track A run if this matrix shows a quality
  gap.

## How to run (operator)

See ``README.md`` for the runbook. ``score.py`` requires a real Gemini API
key and is **not** automated; runs once per cell with budget approval.
