# ADR Status Audit

**Date**: 2026-04-11
**Purpose**:

1. Reconcile **ADR status vs implementation** with [`docs/adr/index.md`](../adr/index.md)
   (especially the **Code** column).
2. Record **what kinds of ADRs to extract next**, based on what we concluded while closing
   recent RFCs (063–069, RFC-057 closure, viewer/server stack).

**Canonical table:** [`docs/adr/index.md`](../adr/index.md) — update that file when an ADR’s
implementation state changes.

## Summary

| Metric | Value |
| :--- | :--- |
| ADR files | **73** (ADR-001–ADR-073; numbering has historical gaps) |
| **Proposed** (not ratified) | **2** — [ADR-055](../adr/ADR-055-adaptive-summarization-routing.md), [ADR-056](../adr/ADR-056-composable-e2e-mock-response-strategy.md) |
| **Accepted + Code = No** | **3** — [ADR-054](../adr/ADR-054-relational-postgres-projection-for-gil-and-kg.md), [ADR-058](../adr/ADR-058-additive-pyannote-diarization-with-separate-extra.md), [ADR-059](../adr/ADR-059-confidence-scored-multi-signal-commercial-detection.md) |
| **Accepted + Code = Partial** | **2** — [ADR-031](../adr/ADR-031-mandatory-pre-release-validation.md), [ADR-047](../adr/ADR-047-proactive-metric-regression-alerting.md) |

**Semantics:** **Accepted** means the decision is ratified, not necessarily shipped. **Code =
No** is normal for forward-looking ADRs tied to open RFCs (051, 058, 060).

## What we concluded: **when to extract a new ADR**

Use an ADR when one or more of these hold; otherwise an **RFC + normative doc** (API guide,
`docs/api/*.md`, UXS) is usually enough.

| ADR type | When to extract | Recent examples |
| :--- | :--- | :--- |
| **Closure / program outcome** | A **large RFC program** ends; you need an immutable summary of what was promoted, what was rejected, and what remains open. | [ADR-073](../adr/ADR-073-rfc057-autoresearch-closure.md) closes [RFC-057](../rfc/RFC-057-autoresearch-optimization-loop.md) |
| **Empirical production defaults** | Autoresearch or benchmarks **change default models/tiers** and you need rationale frozen for onboarding. | [ADR-067](../adr/ADR-067-pegasus-led-retirement-podcast-content.md)–[ADR-072](../adr/ADR-072-llama32-3b-as-tier3-local-llm.md) |
| **Stack & ownership boundary** | **Who owns HTTP**, which **frontend stack**, which **UI E2E runner** — irreversible for the repo. | [ADR-064](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md), [ADR-065](../adr/ADR-065-vue3-vite-cytoscape-frontend-stack.md), [ADR-066](../adr/ADR-066-playwright-for-ui-e2e-testing.md) |
| **Heavy optional dependencies** | Adding an extra that **bloats install** or splits CUDA/CPU paths; default users must not pay the cost. | [ADR-058](../adr/ADR-058-additive-pyannote-diarization-with-separate-extra.md) (`[diarize]` — **accepted, not landed**) |
| **Cross-cutting protocol / contract** | Multiple subsystems must implement the **same interface** (vector store, grounding, artifacts). | [ADR-060](../adr/ADR-060-vectorstore-protocol-with-backend-abstraction.md), [ADR-053](../adr/ADR-053-grounding-contract-for-evidence-backed-insights.md), [ADR-051](../adr/ADR-051-per-episode-json-artifacts-with-logical-union.md) |
| **Process / CI philosophy** | A **policy** decision (test tiers, gates) that outlives one RFC. | [ADR-021](../adr/ADR-021-acceptance-test-tier-as-final-ci-gate.md) (script-shaped implementation) |

### When **not** to add an ADR

- **Viewer feature milestones** that do not change stack: e.g. [RFC-069](../rfc/RFC-069-graph-exploration-toolkit.md) (Graph tools) — **RFC + UXS-001 + E2E map** sufficed.
- **Single-route API additions** consumed only by the viewer with schema in code + tests:
  e.g. [RFC-068](../rfc/RFC-068-corpus-digest-api-viewer.md) — **no new ADR**; Server Guide + OpenAPI-style docs enough.
- **Operational tooling** that does not change architectural boundaries: [RFC-065](../rfc/RFC-065-live-pipeline-monitor.md) (optional `[monitor]` mirrors pyannote pattern but is lighter — still **RFC-first**).
- **Frozen artifact workflows** where one Makefile + scripts own the contract: [RFC-064](../rfc/RFC-064-performance-profiling-release-freeze.md) — **could** become an ADR if CI or external tools must depend on a **frozen schema**; today **RFC + profile README** is enough.

### Optional **future** ADRs (only if pain appears)

| Topic | Trigger |
| :--- | :--- |
| **Multi-feed manifest as external contract** | Third-party tools depend on `corpus_manifest.json` beyond this repo; need immutability/versioning beyond [CORPUS_MULTI_FEED_ARTIFACTS.md](../api/CORPUS_MULTI_FEED_ARTIFACTS.md). |
| **`.pipeline_status.json` schema** | External monitors or agents parse it; breaking changes require a versioned contract. |
| **Performance profile YAML** | Consumers other than `tools/run_compare` and `make profile-diff` need a stability guarantee. |

## Proposed ADRs (unchanged)

| ADR | RFC | Why still Proposed |
| :--- | :--- | :--- |
| [ADR-055](../adr/ADR-055-adaptive-summarization-routing.md) | [RFC-053](../rfc/RFC-053-adaptive-summarization-routing.md) | No episode profiling / routing in pipeline |
| [ADR-056](../adr/ADR-056-composable-e2e-mock-response-strategy.md) | [RFC-054](../rfc/RFC-054-e2e-mock-response-strategy.md) | Composable ResponseProfile / Router not implemented |

## Accepted but not implemented (expected)

| ADR | RFC | Note |
| :--- | :--- | :--- |
| [ADR-054](../adr/ADR-054-relational-postgres-projection-for-gil-and-kg.md) | [RFC-051](../rfc/RFC-051-database-projection-gil-kg.md) | Postgres projection future |
| [ADR-058](../adr/ADR-058-additive-pyannote-diarization-with-separate-extra.md) | [RFC-058](../rfc/RFC-058-audio-speaker-diarization.md) | Decision accepted; **no `[diarize]` extra in `pyproject.toml` yet** |
| [ADR-059](../adr/ADR-059-confidence-scored-multi-signal-commercial-detection.md) | [RFC-060](../rfc/RFC-060-diarization-aware-commercial-cleaning.md) | Commercial detector module as designed not landed |

## Partial implementation (Accepted)

| ADR | Gap |
| :--- | :--- |
| [ADR-031](../adr/ADR-031-mandatory-pre-release-validation.md) | Standardized `make pre-release` / checklist not fully aligned with RFC-038 expectations |
| [ADR-047](../adr/ADR-047-proactive-metric-regression-alerting.md) | Metrics alerts exist; **automated PR comments** not complete |

## Corrections vs 2026-04-03 audit draft

The previous WIP version had **stale rows**:

- **[ADR-048](../adr/ADR-048-centralized-model-registry.md)** is **Accepted** and **implemented**
  — it was incorrectly listed as “Proposed / promote”.
- **ADR-064 / 065 / 066** are **implemented** — they were listed as “not yet implemented”.
- **ADR-021** acceptance is **reflected** in script-based `make test-acceptance`; the old “pytest
  marker tier missing” note is misleading.
- **ADR-062 / ADR-063** — index marks **Code = Yes** (sentence chunking + semantic `gi explore`
  path); treat index as current.

## Action items

1. Keep [`docs/adr/index.md`](../adr/index.md) **Code** column updated when shipping ADR-058
   (`[diarize]`) or ADR-054/059.
2. When **RFC-053** or **RFC-054** ships end-to-end, promote **ADR-055** / **ADR-056** from
   Proposed → Accepted (or supersede with a narrower ADR).
3. Use the **“When to extract”** table above in PR reviews so new work picks **RFC vs ADR**
   consistently.
