# Roadmap

This document lists **open work only** — items that are not yet fully implemented. For completed work, see [RFC index](rfc/index.md) (Completed table) and [Release notes](releases/index.md).

**Last audited**: 2026-04-11 (PRD / RFC / ADR **Gap analysis** sections on each index:
[`docs/prd/index.md`](prd/index.md#gaps), [`docs/rfc/index.md`](rfc/index.md#gaps),
[`docs/adr/index.md`](adr/index.md#gaps))

## Current Status

- **Open RFCs**: **13** (see [RFC index](rfc/index.md); includes Draft [RFC-070](rfc/RFC-070-semantic-corpus-search-platform-future.md))
- **Completed RFCs**: **58** (see [RFC index](rfc/index.md); includes [RFC-061](rfc/RFC-061-semantic-corpus-search.md) FAISS, [RFC-050](rfc/RFC-050-grounded-insight-layer-use-cases.md) / [RFC-056](rfc/RFC-056-knowledge-graph-layer-use-cases.md) single-layer GIL/KG, [RFC-057](rfc/RFC-057-autoresearch-optimization-loop.md) per [ADR-073](adr/ADR-073-rfc057-autoresearch-closure.md))
- **Open PRDs**: **7** (see [PRD index](prd/index.md))

**Completed reference (not backlog):** [RFC-062](rfc/RFC-062-gi-kg-viewer-v2.md) viewer v2; [RFC-063](rfc/RFC-063-multi-feed-corpus-append-resume.md)–[RFC-071](rfc/RFC-071-corpus-intelligence-dashboard-viewer.md) v2.6.0 corpus/viewer track.

## Tier 1 — Active / Near-Term

Items currently in progress or next up. High user value or blocking other work.

| Pri | Item | Type | Status | Gaps | Impact |
| --- | --- | --- | --- | --- | --- |
| **1** | [RFC-070](rfc/RFC-070-semantic-corpus-search-platform-future.md) / [PRD-021](prd/PRD-021-semantic-corpus-search.md) | RFC | **Draft** | [RFC-061](rfc/RFC-061-semantic-corpus-search.md) (FAISS) **Completed**. Open: Qdrant **`VectorStore`**, native filters, pgvector/RFC-051, re-ranking | High |
| **2** | [RFC-054](rfc/RFC-054-e2e-mock-response-strategy.md) | RFC | **Partial** | E2E mock server exists. Full composable ResponseProfile/Router architecture NOT implemented. Blocks #399 (provider hardening) | High |
| **3** | [RFC-072](rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md) / [PRD-017](prd/PRD-017-grounded-insight-layer.md), [PRD-019](prd/PRD-019-knowledge-graph-layer.md) | RFC | **Open** | Cross-layer IDs, `bridge.json`, flagship use cases. **RFC-050** / **RFC-056** single-layer consumption **Completed** — see [RFC index](rfc/index.md) | Medium-High |
| **4** | [RFC-051](rfc/RFC-051-database-projection-gil-kg.md) / [PRD-018](prd/PRD-018-database-projection-gil-kg.md) | RFC | **Not Started** | No Postgres integration, no SQL migrations. [ADR-054](adr/ADR-054-relational-postgres-projection-for-gil-and-kg.md) accepted, Code **No** | Medium |

## Tier 2 — Quality & Observability

Improve output quality, testing, and operational visibility.

| Pri | Item | Type | Status | Gaps | Impact |
| --- | --- | --- | --- | --- | --- |
| **5** | [RFC-053](rfc/RFC-053-adaptive-summarization-routing.md) | RFC | **Not Started** | No episode profiling, no routing logic. Only generic MAP-REDUCE short-input routing exists | High |
| **6** | [RFC-027](rfc/RFC-027-pipeline-metrics-improvements.md) | RFC | **Partial** | Rich metrics + JSON export exist. CSV export missing. Two-tier logging partial | Medium |
| **7** | [RFC-043](rfc/RFC-043-automated-metrics-alerts.md) | RFC | **Partial** | `generate_metrics.py` alerts + nightly CI summary work. Automated PR comments NOT done. Depends on RFC-027 | Medium |
| **8** | [PRD-016](prd/PRD-016-operational-observability-pipeline-intelligence.md) | PRD | **Open** | Operational observability umbrella; metrics improvements (RFC-027) and alerts (RFC-043) are components | Medium |

## Tier 3 — Infrastructure & Developer Experience

Experimentation platform, CI integration, review tooling.

| Pri | Item | Type | Status | Gaps | Impact |
| --- | --- | --- | --- | --- | --- |
| **9** | [RFC-015](rfc/RFC-015-ai-experiment-pipeline.md) / [PRD-007](prd/PRD-007-ai-quality-experiment-platform.md) | RFC | **Partial** | Core pipeline, runner, configs, eval scoring exist. CI auto-run on PR not wired | Low-Medium |
| **10** | [RFC-041](rfc/RFC-041-podcast-ml-benchmarking-framework.md) | RFC | **Partial** | `data/eval/`, golden datasets, baselines, comparison scripts exist. Automated CI benchmarking not wired | Low-Medium |
| **11** | [RFC-038](rfc/RFC-038-continuous-review-tooling.md) | RFC | **Partial** | Dependabot config + pydeps/coupling analysis exist. `make pre-release` checklist script missing | Low |
| **12** | [PRD-015](prd/PRD-015-engineering-governance-productivity.md) | PRD | **Open** | Engineering governance umbrella | Low |

## Tier 4 — Future / Not Started

Planned but no implementation exists yet (or design-only).

| Pri | Item | Type | Status | Dependencies | Impact |
| --- | --- | --- | --- | --- | --- |
| **13** | [RFC-058](rfc/RFC-058-audio-speaker-diarization.md) / [PRD-020](prd/PRD-020-audio-speaker-diarization.md) | RFC | **Not Started** | No diarization code, no pyannote, no `[diarize]` extra | Medium |
| **14** | [RFC-059](rfc/RFC-059-speaker-detection-refactor-test-audio.md) | RFC | **Partial** | Package stub at `speaker_detectors/` (Stage 0 docstring / full modularization TBD) | Medium |
| **15** | [RFC-060](rfc/RFC-060-diarization-aware-commercial-cleaning.md) | RFC | **Not Started** | No `CommercialDetector`, no `cleaning/commercial/`. Phase 2 depends on RFC-058 | Low-Medium |

## ADRs with remaining implementation work

Architectural decisions that are **Proposed**, **Accepted** with **Code = No**, or **Partial** (see [ADR index — Gap analysis](adr/index.md#gaps)):

| ADR | Title | Code | Blocked by / gap |
| :--- | :--- | :--- | :--- |
| [ADR-054](adr/ADR-054-relational-postgres-projection-for-gil-and-kg.md) | Relational Postgres Projection | No | RFC-051 (Not Started) |
| [ADR-055](adr/ADR-055-adaptive-summarization-routing.md) | Adaptive Summarization Routing | Proposed | RFC-053 (Not Started) |
| [ADR-056](adr/ADR-056-composable-e2e-mock-response-strategy.md) | Composable E2E Mock Response Strategy | Proposed | RFC-054 (Partial) |
| [ADR-058](adr/ADR-058-additive-pyannote-diarization-with-separate-extra.md) | Additive pyannote Diarization | No | RFC-058 (Not Started) |
| [ADR-059](adr/ADR-059-confidence-scored-multi-signal-commercial-detection.md) | Confidence-Scored Commercial Detection | No | RFC-060 (Not Started) |
| [ADR-031](adr/ADR-031-mandatory-pre-release-validation.md) | Mandatory Pre-Release Validation | Partial | RFC-038 checklist alignment |
| [ADR-047](adr/ADR-047-proactive-metric-regression-alerting.md) | Proactive Metric Regression Alerting | Partial | RFC-043 PR comments |

## Dependency Graph

```text
RFC-058 (diarization) ──► RFC-060 Phase 2 (diarization-enhanced commercial)
RFC-027 (metrics) ──────► RFC-043 (automated alerts)
RFC-061 (semantic search, FAISS, Completed) ──► RFC-062 (Completed); RFC-070 (platform vector) Draft
PRD-017/019 (GIL/KG) ──► RFC-072 (cross-layer); RFC-050/RFC-056 single-layer Completed
PRD-018 ──► RFC-051 (database projection)
RFC-054 (E2E mocks) ──► #399 (provider hardening)
```

## Related Documents

- **[PRDs](prd/index.md)** — Product requirements documents
- **[RFCs](rfc/index.md)** — Technical design documents
- **[ADR Index](adr/index.md)** — Architecture decision records (with implementation status)
- **[Architecture](architecture/ARCHITECTURE.md)** — System design and module responsibilities
- **[Releases](releases/index.md)** — Release notes and version history

---

**Last Updated**: 2026-04-11
**Next Review**: Quarterly (or as priorities shift)
