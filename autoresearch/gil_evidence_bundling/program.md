# AutoResearch — GIL Evidence Bundling (#698)

## Objective

Validate the #698 bundling hypothesis on real episodes: can Layer A (bundled
``extract_quotes``) and Layer B (bundled ``score_entailment``) cut LLM calls
per episode without dropping grounding quality below the published Gemini
baseline (80% GI insight coverage on ``curated_5feeds_benchmark_v2``)?

Targets:

- **Layer A:** ~14 ``extract_quotes`` calls per episode → 1 bundled call.
- **Layer B:** ~62 ``score_entailment`` calls per episode → ~5 chunked
  bundled calls (``gil_evidence_nli_chunk_size=15``).
- **Combined:** ~76 sequential calls/episode → ~6.

## Quality bar — anchored on existing silver references

We do **not** define new silver. We compare each cell's GI output against
the Sonnet 4.6 silver references that produced the published numbers in
[EVAL_CROSS_DATASET_BASELINE_2026_04.md](../../docs/guides/eval-reports/EVAL_CROSS_DATASET_BASELINE_2026_04.md):

| Reference | Dataset | Stats | Use for |
| --- | --- | --- | --- |
| ``silver_sonnet46_gi_multiquote_benchmark_v2`` | ``curated_5feeds_benchmark_v2`` (5 ep) | 120 quotes, 103 verified | **Primary** — matches our bundled multi-quote shape |
| ``silver_sonnet46_gi_benchmark_v2`` | same | 40 insights, 37 verified single quotes | Secondary — single-quote-per-insight comparison |

**The published baseline is 80% GI insight coverage** for
``gemini-2.5-flash-lite`` on this dataset (April 2026 cross-dataset eval).
The staged baseline cell must reproduce that within ±3pp; bundled cells
must stay within 5pp absolute (≥75%) to qualify as champions.

## Champion gates

A cell qualifies as a champion when ALL hold:

1. **GI insight coverage** vs ``silver_sonnet46_gi_multiquote_benchmark_v2``
   ≥ 75% (within 5pp of the published 80%).
2. **GIL cost reduction** vs the staged baseline ≥ 30% (combined
   ``llm_gi_extract_quotes_cost_usd`` + ``llm_gi_score_entailment_cost_usd``).
3. **Bundled fallback rate** ≤ 20%.
4. **No per-feed grounding floor violation** (omnycontent canary; staged
   baseline is the per-feed reference).

When multiple cells qualify, the highest scalar from
``eval/score.py`` (cost-reduction × grounding-preservation × latency) wins.

## Cells (4 total, all on ``curated_5feeds_benchmark_v2``)

| Cell | quote_mode | nli_mode | Expected calls/ep | Notes |
| --- | --- | --- | --- | --- |
| baseline_staged | staged | staged | ~76 | Reference; reproduces published 80% Gemini baseline |
| bundled_a_only | bundled | staged | ~63 | 14 extracts → 1; NLI unchanged |
| bundled_b_only | staged | bundled | ~19 | extracts unchanged; NLI 62 → ~5 |
| bundled_ab | bundled | bundled | ~6 | The headline win |

All cells: ``gemini-2.5-flash-lite`` (matching the published baseline
config; not the ``-preview-09-2025`` variant), ``preprocessing_profile:
cleaning_v4``, ``gi_insight_source: provider``, ``gi_max_insights: 12``,
``gi_require_grounding: true``. The cells differ only in the two #698 mode
flags and (for Layer B) the chunk size.

## Score formula (internal, complementary to silver-coverage gate)

The silver-coverage gate is the primary quality decision. ``eval/score.py``
emits a secondary internal scalar comparing variant vs staged baseline
(cost / grounding-rate / latency) for ranking when multiple cells pass:

```text
score = 0.5 × cost_reduction
      + 0.3 × grounding_preservation
      + 0.2 × latency_reduction
```

## Failure modes guarded

1. **Token explosion** — bundled prompts too long for Gemini Flash Lite's
   context. Mitigated by ``gil_evidence_nli_chunk_size=15`` default;
   ``score.py`` aborts if input tokens / episode > 50,000.
2. **Grounding regression** — silver-coverage gate (≥75%) is the primary
   guardrail; ``omnycontent`` is the per-feed canary (58% baseline).
3. **JSON parser fragility** — bundled fallbacks tracked via
   ``gi_evidence_extract_quotes_bundled_fallbacks`` and
   ``gi_evidence_score_entailment_bundled_fallbacks``. Fallback rate >20%
   means the bundled prompt isn't reliable; cell rejected.

## Out of scope (V1)

- Other providers (OpenAI, Anthropic, Mistral, DeepSeek, Grok, Ollama).
  Generalising once Gemini champion is picked.
- Per-prompt-variant tuning. V1 uses the prompt baked into the provider's
  ``extract_quotes_bundled`` / ``score_entailment_bundled`` methods. Prompt
  tuning is a follow-up RFC-073 Track A run if this matrix shows a quality
  gap.
- Default profile flips (``cloud_thin.yaml`` / ``cloud_balanced.yaml``).
  That's a separate decision after the champion is recorded — not in
  PR #711.

## Cost estimate

4 cells × 5 episodes × ``gemini-2.5-flash-lite`` ≈ **$1-3 total**. (Was
$2-5 in the original plan when dev = 10 ep; revised down by anchoring on
the 5-episode held-out where the silver lives.)

## How to run (operator)

See ``README.md`` for the runbook.
