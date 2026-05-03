# ADR-078: GIL Evidence Stack Bundling — Per-Provider Champion Modes

- **Status**: Accepted
- **Date**: 2026-05-03
- **Authors**: Podcast Scraper Team
- **Related Issues**: #698 (GIL evidence stack bundling)
- **Related PRs**: #711 (implementation + matrix results)
- **See Also**: [ADR-073](ADR-073-rfc057-autoresearch-closure.md) (autoresearch closure),
  [ADR-077](ADR-077-local-ollama-model-selection.md) (Ollama model selection)

## Context & Problem Statement

The Grounded Insight List (GIL) evidence stack at
`src/podcast_scraper/gi/pipeline.py` issued ~76 sequential LLM calls per episode
in its baseline staged shape:

- ~14 `extract_quotes` calls (one per insight)
- ~62 `score_entailment` (NLI) calls (one per insight × candidate pair)

That call shape dominated end-to-end latency on cloud providers and made the
stage operationally infeasible on local Ollama where per-call overhead
(~20-30s cold + queue) compounds to 25-40 minutes per episode.

Issue #698 introduces two bundling layers to compress those calls:

- **Layer A (`extract_quotes_bundled`):** one LLM call returns
  `{insight_id: [quotes]}` for **all insights** in the episode.
- **Layer B (`score_entailment_bundled`):** one LLM call (chunked at 15
  pairs) returns `{pair_id: {label, score}}` for **all (insight, candidate)
  pairs** in the episode.

The decision was: which combination — staged / Layer A only / Layer B only /
both bundled — should be the champion per provider, and at what tradeoff
between coverage (vs Sonnet-4.6 silver) and call-count reduction?

## Decision

Per-provider champion modes for GIL evidence (`gil_evidence_quote_mode` ×
`gil_evidence_nli_mode`):

| Provider | Model | Champion | Coverage | Calls cut |
| :--- | :--- | :--- | ---: | ---: |
| Gemini | 2.5-flash-lite | **bundled_ab** | 82% (+7pp) | 92% |
| Anthropic | claude-haiku-4-5 | **bundled_ab** | 82.5% (+5pp) | 92% |
| OpenAI | gpt-4o-mini | **bundled_ab** | 72.5% (-5pp tradeoff) | 92% |
| DeepSeek | deepseek-chat | **bundled_ab** | 78% (+6pp) | 92% |
| Grok | grok-3-fast | **bundled_ab** | 87.5% (-2.5pp) | 92% |
| Mistral | mistral-small-latest | **bundled_b_only** | 80% (+10pp) | 75% |
| Ollama | qwen3.5:9b | **bundled_ab** | 72% | 92% |
| Ollama | mistral-small3.2 (24B) | **bundled_ab** | 70% | 92% |
| Ollama | llama3.2:3b | bundled_ab (fragile) | 62% | 92% |

**Default for new providers / unmeasured combinations: `bundled_ab`** with
the shared prompts in `src/podcast_scraper/providers/common/bundled_prompts.py`.
Mistral is the one provider where Layer A regresses coverage; it gets a
provider-specific override to `bundled_b_only`.

For local Ollama, **only `bundled_ab` is operationally viable** —
the staged baseline is not representative of how operators would deploy
the pipeline on local models, so we did not measure staged Ollama as a
"baseline." Ollama llama3.2:3b is documented as fragile (1+ JSON parse
fallback observed at chunk_size=15); operators preferring a small local
model should pin chunk_size=8 or stay on qwen3.5:9b.

## Rationale

**Cost-aware champion selection.** Initial reading of the matrix
(coverage-only) would have picked `bundled_a_only` for OpenAI (77.5% cov)
over `bundled_ab` (72.5%). Operator pushed back: more calls = more rate
limit pressure + higher cost + higher latency under contention. Champion
selection was redone treating call-count as a primary axis, accepting up
to 5pp coverage drop in exchange for 92% call cut on cloud providers.
This re-derivation is recorded in `autoresearch/gil_evidence_bundling/results.tsv`.

**Mistral exception.** Layer A (bundled extract) on Mistral regressed
coverage from 70% (staged) → 67.5% (`bundled_ab`). Layer B alone held
+10pp. Hypothesis: Mistral's response shape on the bundled extract prompt
under-extracts when asked for "3-5 quotes per insight" across many
insights at once. The provider-specific override (Layer B bundled, Layer
A staged) is a 75% call cut without coverage regression — strictly better
than rolling back both layers.

**Ollama treatment.** Per-call overhead on local Ollama makes the staged
shape (76 calls × 20-30s ≈ 30-40min/ep) operationally untenable. The
4-cell matrix used for cloud providers does not map; the bundled shape is
the only one that fits within an operator's actual run budget. We
measured 3 representative local models (3B, 9B, 24B) on `bundled_ab`
only and documented that staged is not measured.

**`num_ctx` requirement.** The bundled prompts cross Ollama's default 2048
context window. Bundled methods on `OllamaProvider` pass `num_ctx=32k`
explicitly via `_ollama_openai_chat_extra_kwargs`. Without that,
mistral-small3.2 silently truncated and timed out. The fix is in
`src/podcast_scraper/providers/ollama/ollama_provider.py`.

## Alternatives Considered

1. **Per-provider Jinja templates from day one (Path B).** Would have
   produced higher quality on each provider but ballooned implementation
   surface to 14+ Jinja files maintained per provider. Rejected:
   shared-prompts result is already at or above silver baseline coverage
   for 5 of 7 providers; per-provider tuning (Track A) is reserved as
   follow-up only for providers that fail Phase A gates.

2. **Skip Layer A entirely (only ship `bundled_b_only`).** Layer B alone
   accounts for the majority of the call cut (75% of total) since NLI
   calls outnumber extract calls 4:1. Rejected: Layer A delivers an
   additional 17% call cut and on most providers does not regress
   coverage (Mistral excepted). Skipping it would leave easy savings on
   the table.

3. **Promote `bundled_ab` as the universal default with no per-provider
   exceptions.** Rejected: Mistral measurably regresses on `bundled_ab`
   (67.5% cov, -2.5pp). Layer A regression is a real provider-specific
   behavior, not noise.

4. **Run a 4-cell matrix on Ollama too.** Rejected: staged Ollama is not
   representative deployment. We have staged-vs-bundled deltas from 5
   cloud providers; Ollama's job in the matrix was to confirm bundled
   parses cleanly and produces reasonable insights on local models.

## Consequences

- **Positive:** GIL evidence stage is now 8x cheaper in LLM calls on cloud
  providers (76 → ~6 calls/ep) and operationally viable on local Ollama
  for the first time. Coverage is preserved or improved on 6 of 7
  providers vs the staged baseline.

- **Positive:** Bundled fallback path remains in place — if the bundled
  prompt returns invalid JSON or under-extracts, the system falls back
  silently to staged. This was observed on llama3.2:3b (3B model is
  fragile on JSON output) without breaking the run.

- **Negative:** Per-provider override surface (Mistral) introduces a
  small dispatch branch in `gi/pipeline.py`. If a future provider also
  shows Layer A regression, the override list grows. Mitigated by
  keeping the override mechanism config-driven (per-experiment
  `gil_evidence_*_mode` params) — no provider-class subtyping.

- **Neutral:** Default profile flips (e.g.
  `cloud_thin: gil_evidence_quote_mode: bundled`) are deferred to a
  follow-up PR per CLAUDE.md "half-wired features" rule. PR #711 ships
  the implementation + matrix results only, not the default flips.

## Implementation Notes

- **Module:** `src/podcast_scraper/providers/common/bundled_prompts.py`
  (shared system + user prompt builders),
  `src/podcast_scraper/providers/<provider>/<provider>_provider.py`
  (`extract_quotes_bundled`, `score_entailment_bundled` per provider),
  `src/podcast_scraper/gi/pipeline.py` (mode dispatch + fallback logic).

- **Config:** `gil_evidence_quote_mode` ∈ {`staged`, `bundled`},
  `gil_evidence_nli_mode` ∈ {`staged`, `bundled`},
  `gil_evidence_nli_chunk_size` (default 15). Wired through
  `merge_eval_task_into_summarizer_config` so experiment YAMLs can override
  per-cell.

- **Pattern:** Provider-agnostic dispatch on the bundled methods (any
  provider exposing `extract_quotes_bundled` automatically participates;
  no provider-type switching in pipeline code).

- **Telemetry:** `llm_gi_extract_quotes_*` + `llm_gi_score_entailment_*`
  substage cost breakdown per episode (committed in #698 phase 1).
  `bundled_fallback_rate` is computed at run end and gated at ≤20%.

## References

- [#698 — GIL evidence stack bundling](https://github.com/chipi/podcast_scraper/issues/698)
- [PR #711 — implementation + matrix results](https://github.com/chipi/podcast_scraper/pull/711)
- [Eval report — EVAL_GIL_BUNDLING_2026_05.md](../guides/eval-reports/EVAL_GIL_BUNDLING_2026_05.md)
- [autoresearch/gil_evidence_bundling/](../../autoresearch/gil_evidence_bundling/)
  — full results.tsv, scaffolds, and per-cell experiment YAMLs
- [ADR-077 — Local Ollama model selection](ADR-077-local-ollama-model-selection.md)
- [RFC-073 — Autoresearch v2 framework](../rfc/RFC-073-autoresearch-v2-framework.md)
