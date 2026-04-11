# Architecture

This directory contains architectural documentation for podcast_scraper — the current
system design, quality constraints, testing approach, data contracts, and the platform
vision for where the system is heading.

## Current state

| Document | Purpose |
| --- | --- |
| [Architecture](ARCHITECTURE.md) | System design — pipeline flow, module map, configuration, ways to run, ADR index |
| [Non-Functional Requirements](NON_FUNCTIONAL_REQUIREMENTS.md) | Quality constraints — performance, security, reliability, observability, maintainability, scalability |
| [Testing Strategy](TESTING_STRATEGY.md) | Test pyramid, patterns, decision criteria, CI integration |

**HTTP / viewer:** Not a separate architecture doc — the FastAPI surface, `/api/*` (including Corpus Library, Corpus Digest, semantic search, and index management endpoints), and OpenAPI **`/docs`** are specified in the [Server Guide](../guides/SERVER_GUIDE.md) (see also [Architecture — Ways to run](ARCHITECTURE.md#ways-to-run-and-deploy)).

**Semantic search:** FAISS-based vector search over transcript chunks is documented in [Architecture — Phase 5a](ARCHITECTURE.md#phase-5a-semantic-corpus-search-rfc-061) and the [Server Guide](../guides/SERVER_GUIDE.md).

## Target state

| Document | Purpose |
| --- | --- |
| [Platform Architecture Blueprint](PLATFORM_ARCHITECTURE_BLUEPRINT.md) | Platform vision — multi-tenant platform, distributed ML, two-tier deployment, observability, deployment lifecycle. Concrete RFCs are broken out from individual sections as implementation begins. |

## Data contracts (ontology specifications)

| Folder | Contents |
| --- | --- |
| [gi/](gi/ontology.md) | Grounded Insight Layer (GIL) ontology — node/edge types, grounding contract, `gi.schema.json` |
| [kg/](kg/README.md) | Knowledge Graph (KG) ontology — entities, topics, relationships, `kg.schema.json` |

## Diagrams

Generated architecture visualizations. See [diagrams/](diagrams/README.md) for the full
list and regeneration instructions.
