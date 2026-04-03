# ADR-052: Separate GIL and KG Artifact Layers

- **Status**: Accepted
- **Date**: 2026-04-03
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-049](../rfc/RFC-049-grounded-insight-layer-core.md), [RFC-055](../rfc/RFC-055-knowledge-graph-layer-core.md)
- **Related PRDs**: [PRD-017](../prd/PRD-017-grounded-insight-layer.md), [PRD-019](../prd/PRD-019-knowledge-graph-layer.md)

## Context & Problem Statement

The system produces two kinds of structured graph data per episode: **Grounded Insights
(GIL)** — evidence-backed takeaways with supporting quotes — and **Knowledge Graph
(KG)** — entities, topics, and their relationships. Both use a nodes-and-edges format,
which creates a temptation to merge them into a single artifact. The decision of whether
to keep them separate or combined affects contracts, feature flags, CLI namespaces, and
every downstream consumer.

## Decision

We maintain **GIL and KG as separate, independent artifact layers**:

1. **Separate files**: `*.gi.json` and `*.kg.json` with distinct schemas and
   `schema_version` fields.
2. **Separate feature flags**: `generate_gi` and `generate_kg` are independent boolean
   config fields. Either can be enabled without the other.
3. **Separate CLI namespaces**: `podcast_scraper gi` for GIL operations, `kg` for KG
   operations.
4. **Separate ontology docs**: `docs/architecture/gi/ontology.md` and `docs/architecture/kg/ontology.md`.
5. **No cross-references required in v1**: Optional links between GIL and KG nodes
   (e.g. `insight_id` in a KG entity) are permitted but not required.

## Rationale

- **Different product questions**: GIL answers "what are the evidence-backed insights?"
  KG answers "what entities and topics are discussed and how do they relate?" Merging
  conflates trust (grounding) with structure (linking).
- **Independent evolution**: GIL and KG have different maturity timelines, different
  extraction models, and different quality contracts. Coupling them forces synchronized
  releases.
- **Consumer clarity**: A consumer that only needs entities (KG) should not parse
  insight-specific fields, and vice versa.
- **Schema stability**: Changes to KG node types don't break GIL consumers.

## Alternatives Considered

1. **Single merged `graph.json`**: Rejected; conflates two distinct product contracts,
   forces consumers to handle all node types, couples release cycles.
2. **KG as extension of GIL (`gi.json` with KG section)**: Rejected; RFC-055
   explicitly states "must not store KG as primary payload inside `gi.json`." Violates
   schema stability.
3. **Shared node namespace with layer tags**: Rejected; adds complexity without
   benefit. Separate files are simpler and more robust.

## Consequences

- **Positive**: Clean separation of concerns. Independent feature flags, schemas, CLI
  commands, and evolution paths. Consumers opt into what they need.
- **Negative**: Cross-layer queries (e.g. "insights about entity X") require joining
  data from two files. Viewer merge logic handles this (RFC-062).
- **Neutral**: Naming convention (`gi` vs `kg`) must be consistent across CLI, config,
  docs, and file extensions.

## Implementation Notes

- **Config**: `config.Config.generate_gi` (bool), `config.Config.generate_kg` (bool)
- **CLI**: `podcast_scraper gi <subcommand>`, `podcast_scraper kg <subcommand>`
- **Pipeline**: GIL and KG extraction stages run independently; both can run in the
  same pipeline invocation
- **Viewer**: Merge logic unifies episodes across layers with prefixed IDs (`g:`, `k:`)

## References

- [RFC-049: GIL Core](../rfc/RFC-049-grounded-insight-layer-core.md)
- [RFC-055: KG Core](../rfc/RFC-055-knowledge-graph-layer-core.md)
- [RFC-062: Viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md) — cross-layer merge logic
