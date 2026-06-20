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
| [Tech Debt](TECH_DEBT.md) | Recognised technical debt -- current coping strategy, options, and triggers to revisit |

**HTTP / viewer:** Not a separate architecture doc — the FastAPI surface, `/api/*` (including Corpus Library, Corpus Digest, semantic search, and index management endpoints), and OpenAPI **`/docs`** are specified in the [Server Guide](../guides/SERVER_GUIDE.md) (see also [Architecture — Ways to run](ARCHITECTURE.md#ways-to-run-and-deploy)).

**Corpus search:** **Hybrid retrieval** (BM25 + dense vector via RRF over a two-tier LanceDB index, with compound results — RFC-090) is the **default**; FAISS vector search (RFC-061) is retained as a switchable fallback. KG-proximity was evaluated and rejected as a signal (RFC-091); relational structure comes from typed edges (#874). See [Architecture — Phase 5a](ARCHITECTURE.md#phase-5a-corpus-search) and the [Server Guide](../guides/SERVER_GUIDE.md).

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
