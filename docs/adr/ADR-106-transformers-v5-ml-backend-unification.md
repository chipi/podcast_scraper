# ADR-106: transformers v5 upgrade + ML backend unification

- **Status**: Accepted (2026-07-06) — retroactively documenting #382, which shipped without an ADR
- **Date**: 2026-07-07 (recorded); change landed 2026-07-06 (merge `5d504f3d`, PR #1145)
- **Authors**: recorded by Claude (Opus 4.8) from the #382 code; decision by the #382 authors
- **Related ADRs**:
  - [ADR-010](ADR-010-hierarchical-summarization-pattern.md) /
    [ADR-043](ADR-043-hybrid-map-reduce-summarization.md) — the summarization architecture
    whose HF-loading plumbing this change consolidates.
- **Related RFCs**:
  - [RFC-042](../rfc/RFC-042-hybrid-summarization-pipeline.md) — hybrid MAP-REDUCE summarization.

> **Note:** #382 merged with only a one-line message ("Upgrade transformers to v5 + unify ML
> architecture"). This ADR is a *retroactive capture* grounded in the shipped code
> (`src/podcast_scraper/providers/ml/`); the deeper motivation for the v5 bump itself should be
> confirmed by the #382 authors if a fuller rationale is needed.

## Context

Two things landed together in #382:

1. **transformers pinned to `>=5.0.0`** (`pyproject.toml`), from the prior v4 line.
2. **Duplicated HF-loading idioms.** Two places independently implemented "load an HF seq2seq
   checkpoint + generate a summary": `providers/ml/summarizer.py` (`SummaryModel`) and
   `providers/ml/hybrid_ml_provider.py` (`TransformersReduceBackend`). Each had its own
   snapshot-first loading, model-family dispatch (Pegasus / LED / BART / LongT5 / FLAN-T5),
   device placement, and `generate()` plumbing — a drift and double-maintenance hazard,
   sharpened by the v5 API changes.

## Decision

Upgrade to transformers v5 and **unify the ML backends** behind shared loaders:

- **`HFSeq2SeqBackend`** (`hf_seq2seq_backend.py`, #382 Phase F) — the single shared "load HF
  seq2seq checkpoint + generate" implementation. Both `SummaryModel` and
  `TransformersReduceBackend` now delegate to it. It owns snapshot-first checkpoint loading
  (avoids transformers checkpoint-discovery bugs on a PyTorch-only cache), model-family
  dispatch, meta-tensor + OOM device fallback, and `GenerationConfig`-wrapped generation.
  Consumers keep their existing public shapes — only the load/generate plumbing moved.
- **`HFEvidenceBackend`** (`hf_evidence_backend.py`) — the parallel shared backend for
  extractive-QA / evidence, with a process-wide model cache.
- `model_loader.py`, `model_registry.py`, `nli_loader.py`, `embedding_loader.py`,
  `extractive_qa.py` rewritten for the v5 API + the shared-backend structure;
  `hybrid_ml_provider.py` (MAP-REDUCE) rewired onto them.

## Consequences

- One HF-seq2seq loading path (snapshot-first, family-dispatch, device/OOM semantics) instead
  of two — the drift hazard is gone.
- The provider-ML surface is v5-compatible; CI installs and the ML integration tests were
  updated in the same PR.
- **The enrichment-layer NLI/embedding scorers are separate** (`enrichment/scorers/` uses
  `sentence-transformers` `CrossEncoder` directly) and were unaffected by this refactor — a
  fact confirmed when the enrichment work rebased cleanly onto v5 (2026-07-07).

## Alternatives considered

- **Stay on transformers v4** — rejected (the ecosystem moved; deferring the bump compounds
  the migration cost). *(Specific forcing function not captured in #382; see the note above.)*
- **Keep the two loading idioms and only bump the version** — rejected; the duplication was the
  main maintenance cost the refactor set out to remove.
