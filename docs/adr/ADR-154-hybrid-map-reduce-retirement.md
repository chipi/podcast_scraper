# ADR-154: Hybrid MAP–REDUCE summarization retired — BART preprocessing hurts a capable REDUCE

- **Status**: Accepted
- **Date**: 2026-09-24
- **Authors**: Marko Dragoljevic
- **Related RFCs**: [RFC-042](../rfc/RFC-042-hybrid-summarization-pipeline.md),
  `RFC-073` (moved to the private eval repo as RFC-003)
- **Supersedes**: [ADR-043](ADR-043-hybrid-map-reduce-summarization.md),
  [ADR-069](ADR-069-hybrid-ml-pipeline-as-production-direction.md)
- **Issues**: [#2142](https://github.com/chipi/podcast_scraper/issues/2142)
- **See Also**: [ADR-044](ADR-044-local-llm-backend-abstraction.md),
  [ADR-067](ADR-067-pegasus-led-retirement-podcast-content.md),
  [ADR-071](ADR-071-four-tier-summarization-strategy.md),
  [ADR-072](ADR-072-llama32-3b-as-tier3-local-llm.md)

## Context & Problem Statement

RFC-042 built a hybrid summariser: a classic HF model (BART) compresses chunks in a MAP stage,
a local LLM synthesises them in a REDUCE stage. ADR-069 accepted it as the **primary production
summarization direction** on 2026-04-03, because it closed most of the ~10pp ROUGE-L gap between
the pure-ML baseline (BART+LED, 18.82%) and cloud models (28–32%).

Six weeks later the held-out v2 evaluation re-ran it under a framework with a dev/held-out split
and dual-judge scoring, and the conclusion inverted.

## The evidence

`podcast-scraper-eval-data:docs/guides/eval-reports/EVAL_HELDOUT_V2_2026_04.md` §6, 2026-04-16.
Blended scalar (ROUGE-L + dual-judge), held-out `curated_5feeds_benchmark_v2`:

| REDUCE model | Standalone | Hybrid (BART MAP + that REDUCE) | Winner |
| ------------ | :--------: | :-----------------------------: | :----: |
| Weak — `llama3.2:3b` | 0.270 (contested) | **0.430** | Hybrid |
| Capable — `qwen3.5:9b` | **0.509** bundled | 0.448 | **Standalone** |

Two findings decided it:

1. **The REDUCE stage does the work.** Swapping the REDUCE model from `llama3.2:3b` to
   `qwen3.5:9b` lifted hybrid by only +4% (0.430 → 0.448). Almost all of hybrid's score came
   from the LLM, not from BART's MAP stage.

2. **BART MAP helps a weak REDUCE and HURTS a capable one.** Chunk-compressing before synthesis
   stabilises a model that would otherwise wander; it also discards information a model good
   enough to use it would have used. Standalone `qwen3.5:9b` beat the hybrid built on the same
   model by 0.061.

So the hybrid architecture's value was never additive — it was **compensatory**, and it was
tied to the weakness of the model underneath it. That is a dependency on a condition that
improves on its own over time.

The April report demoted it rather than removing it: *"loses its reason to exist as a default
recommendation. Retained in docs for narrow niches (truly memory-constrained, explainability)."*

**Those niches did not materialise.** Five months on: no profile sets `summary_provider:
hybrid_ml`, no profile sets `hybrid_reduce_backend`, nothing in `config/` references `llama_cpp`
or `gguf`, and every summarisation path in use routes through the gateway
(LiteLLM → OpenRouter / vLLM) where every available REDUCE model is in the "capable" column of
the table above.

## Decision

**Retire the hybrid MAP–REDUCE summariser.** Remove the provider, the `hybrid_ml` value of
`summary_provider`, the nine `hybrid_*` configuration fields, and the `llama-cpp-python`
dependency it was the only importer of.

## Consequences

- **`llama-cpp-python` leaves the dependency tree.** It was the only native-compile dependency
  in `[ml]`, reached by exactly one import (`providers/ml/hybrid_ml_provider.py:214`). Every
  install gets faster and one platform-specific build failure mode disappears.
- **`hybrid_reduce_backend: "llama_cpp"` goes with it**, which is the concrete part of ADR-044
  (local LLM backend abstraction) that is now unreachable. The abstraction's other backends
  (`transformers`, `ollama`) are unaffected — ADR-044 is narrowed, not reversed, so it is listed
  under See Also rather than superseded.
- **ADR-071's four-tier strategy and ADR-072's tier-3 pick are stale in their hybrid parts.**
  The April report already replaced `llama3.2:3b` with `qwen3.5:9b` bundled as the tier-3 choice.
  Neither is superseded here — that would overclaim; both need reading against current routing.
- **A memory-constrained deployment loses an option.** Stated plainly rather than waved away:
  if a future host cannot run a capable REDUCE model, the argument in the weak-REDUCE row of the
  table comes back and this decision should be revisited with a measurement, not an assumption.
- **The word "hybrid" still names two live subsystems** — `search/hybrid_search.py` (vector +
  keyword retrieval) and `cleaning/hybrid.py` (`transcript_cleaning_strategy` defaults to
  `"hybrid"`). Neither is affected. 57 source files match the word; 8 are the summariser.

## Alternatives considered

- **Keep it for the niches.** Rejected: five months produced no profile that selects it, and an
  unselected code path with a native-compile dependency is carried by everyone and exercised by
  no one. The eval numbers remain in the report if the niche ever appears.
- **Delete the ADRs instead of superseding them.** Rejected: ADR-069 says "primary production
  direction" in its title. Someone will read it. The useful artifact is the reversal and its
  evidence, not the absence of a record.
