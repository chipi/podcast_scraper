# ADR-008: Database-Agnostic Metadata Schema

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-011](../rfc/RFC-011-metadata-generation.md)
- **Related PRDs**: [PRD-004](../prd/PRD-004-metadata-generation.md)

## Context & Problem Statement

Metadata is consumed by various downstream systems: PostgreSQL (JSONB), MongoDB, ClickHouse, and simple web dashboards. We need a format that is rich enough for all but requires zero transformation for ingestion.

## Decision

Metadata documents follow a **Database-Agnostic Schema**:

- **Format**: Strictly valid JSON.
- **Naming**: `snake_case` for all fields (SQL-friendly).
- **Timestamps**: ISO 8601 strings (universally parsable).
- **Structure**: Flat logical groups (`feed`, `episode`, `content`, `processing`).

## Rationale

- **Zero-ETL Ingestion**: Allows piping JSON files directly into Postgres or Mongo without writing custom "adapter" code.
- **Human Readability**: JSON remains readable for debugging while providing the structure needed for scale.
- **Consistency**: Centralizing this in a Pydantic model (RFC-008) ensures all episodes adhere to the same contract.

## Alternatives Considered

1. **SQLite Database per Run**: Rejected as it's harder to version control and diff than text files.
2. **YAML**: Rejected as the primary metadata format due to poor native support in many database ingestion tools (though kept as an optional human-only export).

## Consequences

- **Positive**: High interoperability; rapid development of dashboards and analytics.
- **Negative**: JSON is slightly more verbose than binary formats (though negligible at this scale).

## References

- [RFC-011: Per-Episode Metadata Document Generation](../rfc/RFC-011-metadata-generation.md)
