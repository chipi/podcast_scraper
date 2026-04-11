# RFC Status Audit

**Date**: 2026-04-11
**Purpose**: Cross-check RFC **implementation** narrative against the repo. The **canonical
open vs completed lists** live in [`docs/rfc/index.md`](../rfc/index.md); this WIP note adds
grouping, ADR links, and historical context.

## Canonical source

- **Index:** [`docs/rfc/index.md`](../rfc/index.md) — update this first when status changes.
- **Bodies:** Each RFC’s `- **Status**:` or `## Status` block should match the index.

## Snapshot (2026-04-11)

| Metric | Value |
| :--- | :--- |
| RFC files | **69** (RFC-001–RFC-069; **no RFC-014**) |
| **Open** in index | **14** (see table below) |
| **Completed** in index | **55** |

### Open RFCs (from index)

| RFC | Theme | Notes |
| :--- | :--- | :--- |
| [RFC-015](../rfc/RFC-015-ai-experiment-pipeline.md) | Experiments | Runner implemented; **CI auto-run still pending** |
| [RFC-027](../rfc/RFC-027-pipeline-metrics-improvements.md) | Observability | **Partial** — JSON/metrics rich; CSV / two-tier logging gaps |
| [RFC-038](../rfc/RFC-038-continuous-review-tooling.md) | Governance | Dependabot + pydeps; **pre-release checklist** still partial ([ADR-031](../adr/ADR-031-mandatory-pre-release-validation.md) Partial) |
| [RFC-041](../rfc/RFC-041-podcast-ml-benchmarking-framework.md) | Benchmarks | Datasets/scripts exist; **automated CI benchmarking** not fully wired |
| [RFC-043](../rfc/RFC-043-automated-metrics-alerts.md) | Metrics | Alerts/summaries exist; **PR comment bot** not done ([ADR-047](../adr/ADR-047-proactive-metric-regression-alerting.md) Partial) |
| [RFC-050](../rfc/RFC-050-grounded-insight-layer-use-cases.md) | GIL | `gi explore` / `gi query`; **`gi list` / Insight Explorer** gaps |
| [RFC-051](../rfc/RFC-051-database-projection-gil-kg.md) | GIL/KG DB | **Not started** — Postgres projection ([ADR-054](../adr/ADR-054-relational-postgres-projection-for-gil-and-kg.md) Code **No**) |
| [RFC-053](../rfc/RFC-053-adaptive-summarization-routing.md) | Summarization | **Not started** — no episode profiling router ([ADR-055](../adr/ADR-055-adaptive-summarization-routing.md) **Proposed**) |
| [RFC-054](../rfc/RFC-054-e2e-mock-response-strategy.md) | Testing | Mocks exist; **composable ResponseProfile** not built ([ADR-056](../adr/ADR-056-composable-e2e-mock-response-strategy.md) **Proposed**) |
| [RFC-056](../rfc/RFC-056-knowledge-graph-layer-use-cases.md) | KG | Several `kg` subcommands; **`kg explore` / `kg list`** gaps |
| [RFC-058](../rfc/RFC-058-audio-speaker-diarization.md) | Diarization | [ADR-058](../adr/ADR-058-additive-pyannote-diarization-with-separate-extra.md) accepted; **no `[diarize]` / pyannote in tree** |
| [RFC-059](../rfc/RFC-059-speaker-detection-refactor-test-audio.md) | Speakers | Factory + providers wired; package still **Stage 0** docstring / full modularization TBD |
| [RFC-060](../rfc/RFC-060-diarization-aware-commercial-cleaning.md) | Cleaning | **Not started** as designed ([ADR-059](../adr/ADR-059-confidence-scored-multi-signal-commercial-detection.md) Code **No**) |

### Recently completed (v2.6.0 track)

Aligned with [PRD](../prd/index.md) / release notes:

| RFC | Delivered (high level) |
| :--- | :--- |
| [RFC-063](../rfc/RFC-063-multi-feed-corpus-append-resume.md) | Multi-feed CLI/layout, manifest/summary, unified discovery |
| [RFC-064](../rfc/RFC-064-performance-profiling-release-freeze.md) | `data/profiles/`, freeze + diff scripts, Makefile targets |
| [RFC-065](../rfc/RFC-065-live-pipeline-monitor.md) | `--monitor`, `.pipeline_status.json`, optional `[monitor]` profiling |
| [RFC-066](../rfc/RFC-066-run-compare-performance-tab.md) | Streamlit **Performance** page vs frozen profiles |
| [RFC-068](../rfc/RFC-068-corpus-digest-api-viewer.md) | `GET /api/corpus/digest`, Digest tab, Library glance |
| [RFC-069](../rfc/RFC-069-graph-exploration-toolkit.md) | Graph toolkit (minimap, degree buckets, box zoom, layouts) |

**Also closed earlier (not re-listed in index “open”):** [RFC-057](../rfc/RFC-057-autoresearch-optimization-loop.md) completed per [ADR-073](../adr/ADR-073-rfc057-autoresearch-closure.md).

## Historical line-by-line draft table (pre–2026-04-11)

The long **Draft RFC Audit** table from 2026-04-03 is **mostly superseded** by the index
refresh (many rows were “mark Completed” work that is now done in `docs/rfc/index.md` and RFC
headers). Keep it only for archeology; do not treat “Draft” in that table as current without
checking the index.

**Corrections vs old table:**

- **RFC-057** — was “Partial”; **closed** via ADR-073 / RFC-057 status **Completed**.
- **RFC-059** — still **Partial** (refactor story not finished); not “empty stub” only —
  `speaker_detectors/factory.py` is active; `__init__.py` still says Stage 0.
- **RFC-062** — **Completed** in index; old footnote incorrectly listed RFC-062 as draft.

## Recommendations

1. **Status changes** — Edit RFC body + [`docs/rfc/index.md`](../rfc/index.md) together.
2. **Large deliveries without new ADRs** — RFC-063/064/065/066/068/069 shipped with **RFC +
   guides + API docs** only; see [`adr-status-audit.md`](adr-status-audit.md) for when an ADR
   is still worth extracting.
3. **Link to ADR audit** — For “decision vs code” semantics, use [`adr-status-audit.md`](adr-status-audit.md)
   and [`docs/adr/index.md`](../adr/index.md) **Code** column.
