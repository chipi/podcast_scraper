# Roadmap

This document lists **open work only** — items that are not yet fully implemented in this repository. For completed work, see [RFC index](rfc/index.md) (Completed table) and [Release notes](releases/index.md).

This repository is the open platform (pipeline, corpus, search, the Workbench and the extension seams; scope in [ADR-170](adr/ADR-170-private-repo-split-platform-common-apps.md)). Work that belongs to an installed extension, a downstream build or the research repository is not tracked here.

**Last audited**: 2026-11-17 (PRD / RFC / ADR **Gap analysis** sections on each index:
[`docs/prd/index.md`](prd/index.md#gaps), [`docs/rfc/index.md`](rfc/index.md#gaps),
[`docs/adr/index.md`](adr/index.md#gaps)). Each item's status is the one its own RFC, PRD or ADR states.

## Current Status

- **Open RFCs**: see the [RFC index](rfc/index.md) (Open table). All are **Draft** except [RFC-091](rfc/RFC-091-kg-proximity-signal.md), a decision record that **rejects** KG proximity as a retrieval signal.
- **Open PRDs**: see the [PRD index](prd/index.md) (Open table).
- **Direction already decided**: Postgres is the source of truth for app state ([ADR-169](adr/ADR-169-the-database-is-the-source-of-truth.md)), which supersedes the Postgres-projection plan ([RFC-051](rfc/RFC-051-database-projection-gil-kg.md)).

**Completed reference (not backlog):** [RFC-062](rfc/RFC-062-gi-kg-viewer-v2.md) Workbench v2; [RFC-063](rfc/RFC-063-multi-feed-corpus-append-resume.md)–[RFC-071](rfc/RFC-071-corpus-intelligence-dashboard-viewer.md) v2.6.0 corpus/viewer track; [RFC-084](rfc/RFC-084-corpus-backup-manifest-and-version-aware-restore.md) / [RFC-085](rfc/RFC-085-graph-handoff-orchestrator-retrospective.md) / [RFC-086](rfc/RFC-086-viewer-test-pyramid-and-production-shaped-fixtures.md) v2.6.0 stabilization track ([ADR-092](adr/ADR-092-corpus-snapshot-backup-manifest-and-newest-compatible-restore.md), [ADR-094](adr/ADR-094-graph-handoff-orchestrator-fsm.md), [ADR-095](adr/ADR-095-viewer-test-pyramid.md)); [RFC-090](rfc/RFC-090-hybrid-retrieval.md) hybrid retrieval over LanceDB.

## Tier 1 — Active / Near-Term

Items in progress or next up.

| Pri | Item | Type | Status | Gaps | Impact |
| --- | --- | --- | --- | --- | --- |
| **1** | [RFC-107](rfc/RFC-107-search-v3-query-workspace.md) / [PRD-045](prd/PRD-045-search-v3-query-workspace.md) | RFC | **Draft** | Query workspace for the Workbench Search tab, with additive query params on the search route | High |
| **2** | [RFC-124](rfc/RFC-124-multilingual-transcription-and-translation.md) / [PRD-047](prd/PRD-047-multilingual-ingest.md) | RFC | **Draft** | The translation stage and the English-canonical layout ship ([Multilingual Arc](architecture/MULTILINGUAL_ARC.md)); the RFC itself is still Draft | High |
| **3** | [RFC-092](rfc/RFC-092-ml-query-router.md) | RFC | **Draft** | Rules router shipped; the ML classifier is gated on eval data | Medium |

## Tier 2 — Workbench and pipeline

| Pri | Item | Type | Status | Gaps | Impact |
| --- | --- | --- | --- | --- | --- |
| **4** | [RFC-104](rfc/RFC-104-operator-shows-library.md) / [PRD-044](prd/PRD-044-operator-shows-library.md) | RFC | **Draft** | Shows-first Library browse; the Library tab has a Shows mode today, so the status needs reconciling | Medium |
| **5** | [RFC-080](rfc/RFC-080-graph-visualization-extensions.md) | RFC | **Draft** | Five composable graph visualization lenses | Medium |
| **6** | [RFC-105](rfc/RFC-105-cross-feed-audio-llm-pipelining.md) | RFC | **Draft** | Shared queue across feeds so the next feed's audio work overlaps the previous feed's LLM tail | Medium |
| **7** | [RFC-053](rfc/RFC-053-adaptive-summarization-routing.md) | RFC | **Draft** | No episode profiling or content-type routing in the pipeline | Medium |
| **8** | [PRD-027](prd/PRD-027-enriched-search.md) | PRD | **Partial** | The `QueryEnricher` protocol and `/api/search?enrich_results=true` wiring ship; this repository registers no query enricher, installed extensions add them | Medium |

## Tier 3 — Quality, observability and developer experience

| Pri | Item | Type | Status | Gaps | Impact |
| --- | --- | --- | --- | --- | --- |
| **9** | [PRD-016](prd/PRD-016-operational-observability-pipeline-intelligence.md) | PRD | **Partial** | Test metrics, dashboards and the live monitor ship; operational observability umbrella still open | Medium |
| **10** | [RFC-116](rfc/RFC-116-autoresearch-driven-fixture-corpus.md) | RFC | **Draft** | Fixture corpus driven by real failure modes; the v3 text side has landed | Medium |
| **11** | [PRD-015](prd/PRD-015-engineering-governance-productivity.md) | PRD | **Draft** | Engineering governance umbrella | Low |

## ADRs with remaining implementation work

Architectural decisions that are **Proposed**, **Accepted** with **Code = No**, or **Partial** (see [ADR index — Gap analysis](adr/index.md#gaps)):

| ADR | Title | Code | Blocked by / gap |
| :--- | :--- | :--- | :--- |
| [ADR-170](adr/ADR-170-private-repo-split-platform-common-apps.md) | Split the apps, MCP, enrichers and sign-in providers into private repos | Proposed | Defines this repository's scope; the extension interface exists (`src/podcast_scraper/extensions.py`) |
| [ADR-054](adr/ADR-054-relational-postgres-projection-for-gil-and-kg.md) | Relational Postgres Projection | No | Its "files canonical" part is superseded by ADR-169 |
| [ADR-055](adr/ADR-055-adaptive-summarization-routing.md) | Adaptive Summarization Routing | Proposed | RFC-053 (Draft) |
| [ADR-031](adr/ADR-031-mandatory-pre-release-validation.md) | Mandatory Pre-Release Validation | Partial | Pre-release checklist script |
| [ADR-047](adr/ADR-047-proactive-metric-regression-alerting.md) | Proactive Metric Regression Alerting | Partial | PR-comment and webhook scripts deliberately abandoned; alerting is operator-side |

## Dependency Graph

```text
RFC-090 (hybrid retrieval, Completed) ──► RFC-107 (Search v3, Draft); RFC-092 (ML router, Draft)
RFC-124 (multilingual, Draft) ──► PRD-047
RFC-053 (adaptive routing, Draft) ──► ADR-055 (Proposed)
ADR-169 (database is the source of truth) ──► supersedes RFC-051 and the "files canonical" part of ADR-054
```

## Related Documents

- **[PRDs](prd/index.md)** — Product requirements documents
- **[RFCs](rfc/index.md)** — Technical design documents
- **[ADR Index](adr/index.md)** — Architecture decision records (with implementation status)
- **[Architecture](architecture/ARCHITECTURE.md)** — System design and module responsibilities
- **[Releases](releases/index.md)** — Release notes and version history

---

**Last Updated**: 2026-11-17
**Next Review**: Quarterly (or as priorities shift)
