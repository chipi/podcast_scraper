# Architecture

This directory contains architectural documentation for podcast_scraper — the current
system design, quality constraints, testing approach, data contracts, and the platform
vision for where the system is heading.

## Current state

| Document | Purpose |
| --- | --- |
| [Architecture](ARCHITECTURE.md) | System design — pipeline flow, module map, configuration, ways to run, ADR index |
| [Hosting and infrastructure](HOSTING_AND_INFRASTRUCTURE.md) | Always-on VPS, Tailscale, OpenTofu, GitHub Actions, Compose on host — narrative companion to infra ADRs and RFC-082 |
| [Corpus artifacts and viewer surfaces](CORPUS_ARTIFACTS_AND_SURFACES.md) | Pipeline artifact inventory, API route dependencies, viewer tab map (#797) |
| [Non-Functional Requirements](NON_FUNCTIONAL_REQUIREMENTS.md) | Quality constraints — performance, security, reliability, observability, maintainability, scalability |
| [Testing Strategy](TESTING_STRATEGY.md) | Test pyramid, patterns, decision criteria, CI integration |
| [Agent-navigable codebases](AGENT_NAVIGABLE_CODEBASE.md) | Why the repo is documented and governed as it is — routing over restating, guarding pointers, and how to apply it to another project |
| [Tech Debt](TECH_DEBT.md) | Recognised technical debt -- current coping strategy, options, and triggers to revisit |

**HTTP / viewer:** Not a separate architecture doc — the FastAPI surface, `/api/*` (including Corpus Library, Corpus Digest, semantic search, and index management endpoints), and OpenAPI **`/docs`** are specified in the [Server Guide](../guides/SERVER_GUIDE.md) (see also [Architecture — Ways to run](ARCHITECTURE.md#ways-to-run-and-deploy)).

**Corpus search:** **Hybrid retrieval** (BM25 + dense vector via RRF over a two-tier LanceDB index, with compound results — RFC-090) is the **default**; FAISS vector search (RFC-061) is retained as a switchable fallback. KG-proximity was evaluated and rejected as a signal (RFC-091); relational structure comes from typed edges (#874). See [Architecture — Phase 5a](ARCHITECTURE.md#phase-5a-corpus-search) and the [Server Guide](../guides/SERVER_GUIDE.md).

## Arcs in flight

Working notes for a multi-phase body of work — the arc's shape, its slice plan, the code facts it
rests on, the decisions taken, and a running log. One per arc; retired when the arc closes.

| Document | Purpose |
| --- | --- |
| [Multilingual ingest (v1)](MULTILINGUAL_ARC.md) | Source-language capture with English-normalized intelligence — phase ladder, slice plan (each slice one issue), verified code facts, the claims six reviews found false, decisions D-1…D-20. Pulls together [PRD-047](../prd/PRD-047-multilingual-ingest.md) and [RFC-123](../rfc/RFC-123-speaker-turns-artifact.md)/[124](../rfc/RFC-124-multilingual-transcription-and-translation.md)/[125](../rfc/RFC-125-translation-confidence-and-claim-verification.md) |
| [Multilingual ingest (v2)](MULTILINGUAL_ARC_V2.md) | The continuation — everything deferred out of v1 with its slices and the reason: claim verification, quality estimation, the language badge and filter, the turns consumers, cross-lingual retrieval, word-level anchors. Parked; no PRD or RFC yet |

## Target state

| Document | Purpose |
| --- | --- |
| [Platform Architecture Blueprint](PLATFORM_ARCHITECTURE_BLUEPRINT.md) | Platform vision — multi-tenant platform, distributed ML, two-tier deployment, observability, deployment lifecycle. Concrete RFCs are broken out from individual sections as implementation begins. |

## Data contracts (ontology specifications)

| Folder | Contents |
| --- | --- |
| [**corpus/**](corpus/ontology.md) | **Unified corpus ontology (v2)** — single source of truth for KG v2.0+ and GI v3.0+. Two-tier edge contract, `Person`/`Organization`/`Podcast` first-class, ABOUT/MENTIONS_PERSON/MENTIONS_ORG, `insight_type`+`position_hint`. ([RFC-097](../rfc/RFC-097-unified-kg-gi-ontology-v2.md)) |
| [gi/](gi/ontology.md) | Grounded Insight Layer (GIL) ontology — **superseded by `corpus/ontology.md` for v3.0+**; retained for v1/v2 archaeology |
| [kg/](kg/README.md) | Knowledge Graph (KG) ontology — **superseded by `corpus/ontology.md` for v2.0+**; retained for v1 archaeology |

## Diagrams

Generated architecture visualizations. See [diagrams/](diagrams/README.md) for the full
list and regeneration instructions.
