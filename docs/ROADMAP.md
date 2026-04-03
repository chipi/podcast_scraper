# Roadmap

This document lists **open work only** — items that are not yet fully implemented. For completed work, see [RFC index](rfc/index.md) (Completed table) and [Release notes](releases/index.md).

**Last audited**: 2026-04-03 (RFC + ADR status audit; see `docs/wip/rfc-status-audit.md`)

## Current Status

- **Open RFCs**: 16 (9 partial, 5 not started, 2 narrative/CI-pending)
- **Completed RFCs**: 46 (core pipeline, 9-provider ecosystem, GIL/KG core, audio, eval, model registry, and more)
- **Open PRDs**: 8

## Tier 1 — Active / Near-Term

Items currently in progress or next up. High user value or blocking other work.

| Pri | Item | Type | Status | Gaps | Impact |
| --- | --- | --- | --- | --- | --- |
| **1** | [RFC-061](rfc/RFC-061-semantic-corpus-search.md) / [PRD-021](prd/PRD-021-semantic-corpus-search.md) | RFC | **Phase 1 done** | Sentence-boundary chunking (ADR-062) tokenizer-only; `gi explore --topic` semantic upgrade (ADR-063) not wired; Phase 2 (Qdrant/service) future | High |
| **2** | [RFC-054](rfc/RFC-054-e2e-mock-response-strategy.md) | RFC | **Partial** | E2E mock server exists. Full composable ResponseProfile/Router architecture NOT implemented. Blocks #399 (provider hardening) | High |
| **3** | [RFC-050](rfc/RFC-050-grounded-insight-layer-use-cases.md) / [PRD-017](prd/PRD-017-grounded-insight-layer.md) | RFC | **Partial** | `gi explore` and `gi query` work. `gi list` missing. Insight Explorer pattern partial | Medium-High |
| **4** | [RFC-056](rfc/RFC-056-knowledge-graph-layer-use-cases.md) / [PRD-019](prd/PRD-019-knowledge-graph-layer.md) | RFC | **Partial** | `kg validate/inspect/export/entities/topics` work. `kg explore` and `kg list` missing | Medium-High |
| **5** | [RFC-062](rfc/RFC-062-gi-kg-viewer-v2.md) / [PRD-021](prd/PRD-021-semantic-corpus-search.md) | RFC | **Not Started** | No FastAPI server, no Vue frontend, no `podcast serve` CLI. Only legacy `web/gi-kg-viz/` | Medium |

## Tier 2 — Quality & Observability

Improve output quality, testing, and operational visibility.

| Pri | Item | Type | Status | Gaps | Impact |
| --- | --- | --- | --- | --- | --- |
| **6** | [RFC-053](rfc/RFC-053-adaptive-summarization-routing.md) | RFC | **Not Started** | No episode profiling, no routing logic. Only generic MAP-REDUCE short-input routing exists | High |
| **7** | [RFC-027](rfc/RFC-027-pipeline-metrics-improvements.md) | RFC | **Partial** | Rich metrics + JSON export exist. CSV export missing. Two-tier logging partial | Medium |
| **8** | [RFC-043](rfc/RFC-043-automated-metrics-alerts.md) | RFC | **Partial** | `generate_metrics.py` alerts + nightly CI summary work. Automated PR comments NOT done. Depends on RFC-027 | Medium |
| **9** | [PRD-016](prd/PRD-016-operational-observability-pipeline-intelligence.md) | PRD | **Open** | Operational observability umbrella; metrics improvements (RFC-027) and alerts (RFC-043) are components | Medium |

## Tier 3 — Infrastructure & Developer Experience

Experimentation platform, CI integration, review tooling.

| Pri | Item | Type | Status | Gaps | Impact |
| --- | --- | --- | --- | --- | --- |
| **10** | [RFC-015](rfc/RFC-015-ai-experiment-pipeline.md) / [PRD-007](prd/PRD-007-ai-quality-experiment-platform.md) | RFC | **Partial** | Core pipeline, runner, configs, eval scoring all exist. CI auto-run on PR not wired | Low-Medium |
| **11** | [RFC-041](rfc/RFC-041-podcast-ml-benchmarking-framework.md) | RFC | **Partial** | `data/eval/`, golden datasets, baselines, comparison scripts exist. Automated CI benchmarking not wired | Low-Medium |
| **12** | [RFC-038](rfc/RFC-038-continuous-review-tooling.md) | RFC | **Partial** | Dependabot config + pydeps/coupling analysis exist. `make pre-release` checklist script missing | Low |
| **13** | [RFC-057](rfc/RFC-057-autoresearch-optimization-loop.md) | RFC | **Partial** | `autoresearch/prompt_tuning/` with `program.md` and `score.py` exist. Track B (ML params) not set up. No completed runs | Low |
| **14** | [PRD-015](prd/PRD-015-engineering-governance-productivity.md) | PRD | **Open** | Engineering governance umbrella | Low |

## Tier 4 — Future / Not Started

Planned but no implementation exists yet.

| Pri | Item | Type | Status | Dependencies | Impact |
| --- | --- | --- | --- | --- | --- |
| **15** | [RFC-058](rfc/RFC-058-audio-speaker-diarization.md) / [PRD-020](prd/PRD-020-audio-speaker-diarization.md) | RFC | **Not Started** | No diarization code, no pyannote, no `[diarize]` extra | Medium |
| **16** | [RFC-059](rfc/RFC-059-speaker-detection-refactor-test-audio.md) | RFC | **Partial** | Package stub at `speaker_detectors/` (empty "Stage 0"). Original `speaker_detection.py` unchanged | Medium |
| **17** | [RFC-060](rfc/RFC-060-diarization-aware-commercial-cleaning.md) | RFC | **Not Started** | No `CommercialDetector`, no `cleaning/commercial/`. Depends on RFC-058 for Phase 2 | Low-Medium |
| **18** | [RFC-051](rfc/RFC-051-database-projection-gil-kg.md) / [PRD-018](prd/PRD-018-database-projection-gil-kg.md) | RFC | **Not Started** | No Postgres integration, no SQL migrations. Depends on PRD-017/019 | Low |

## Open ADRs Not Yet Implemented

These architectural decisions are accepted/proposed but code does not exist yet:

| ADR | Title | Blocked By |
| :--- | :--- | :--- |
| [ADR-054](adr/ADR-054-relational-postgres-projection-for-gil-and-kg.md) | Relational Postgres Projection | RFC-051 (Not Started) |
| [ADR-055](adr/ADR-055-adaptive-summarization-routing.md) | Adaptive Summarization Routing | RFC-053 (Not Started) |
| [ADR-056](adr/ADR-056-composable-e2e-mock-response-strategy.md) | Composable E2E Mock Response Strategy | RFC-054 (Partial) |
| [ADR-058](adr/ADR-058-additive-pyannote-diarization-with-separate-extra.md) | Additive pyannote Diarization | RFC-058 (Not Started) |
| [ADR-059](adr/ADR-059-confidence-scored-multi-signal-commercial-detection.md) | Confidence-Scored Commercial Detection | RFC-060 (Not Started) |
| [ADR-062](adr/ADR-062-sentence-boundary-transcript-chunking.md) | Sentence-Boundary Transcript Chunking | RFC-061 (Phase 1 done) |
| [ADR-063](adr/ADR-063-transparent-semantic-upgrade-for-gi-explore.md) | Transparent Semantic Upgrade for gi explore | RFC-061 (Phase 1 done) |
| [ADR-064](adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md) | Canonical Server Layer | RFC-062 (Not Started) |
| [ADR-065](adr/ADR-065-vue3-vite-cytoscape-frontend-stack.md) | Vue 3 + Vite + Cytoscape Frontend | RFC-062 (Not Started) |
| [ADR-066](adr/ADR-066-playwright-for-ui-e2e-testing.md) | Playwright for UI E2E Testing | RFC-062 (Not Started) |

## Dependency Graph

```text
RFC-058 (diarization) ──► RFC-060 Phase 2 (diarization-enhanced commercial)
RFC-027 (metrics) ──────► RFC-043 (automated alerts)
RFC-061 (semantic search) ──► RFC-062 (viewer v2)
PRD-017/019 (GIL/KG) ──► RFC-051 (database projection)
RFC-054 (E2E mocks) ──► #399 (provider hardening)
```

## Related Documents

- **[PRDs](prd/index.md)** — Product requirements documents
- **[RFCs](rfc/index.md)** — Technical design documents
- **[ADR Index](adr/index.md)** — Architecture decision records (with implementation status)
- **[Architecture](ARCHITECTURE.md)** — System design and module responsibilities
- **[Releases](releases/index.md)** — Release notes and version history

---

**Last Updated**: 2026-04-03
**Next Review**: Quarterly (or as priorities shift)
