# ADR-107: Ingestion is the pipeline — drop the standalone `ingest` primitive

- **Status**: Accepted (2026-07-07)
- **Date**: 2026-07-07
- **Authors**: Marko Dragoljevic, Claude (Opus 4.8)
- **Related ADRs**:
  - [ADR-051](ADR-051-per-episode-json-artifacts-with-logical-union.md) — the per-episode
    artifact + logical-union corpus model that "the pipeline is ingestion" relies on.
- **Related RFCs**:
  - [RFC-037 / PRD-037](../prd/PRD-037-discovery.md) — discovery / scrape-on-demand, whose
    phase-1 "ingestion primitive" this ADR retires.

## Context

Issue #1069 (scrape-on-demand) initially proposed a dedicated **ingestion primitive** —
`ingest(feed_url | episode)` plus an `IngestPolicy` authorization seam — as a first-class
verb distinct from the transcription pipeline. It was built in phase 1
(`src/podcast_scraper/ingestion/`, an `ingest` CLI subcommand).

On review it was **redundant**: the single-feed pipeline *already* ingests. Running the
pipeline against one feed downloads → transcribes → extracts → **merges the resulting
artifacts into the corpus, globally deduped** (the per-episode-JSON + logical-union model,
ADR-051). A second `ingest` verb re-wrapped exactly that with no added capability, and split
the operator's mental model across two near-identical entry points.

## Decision

Drop the `ingest` primitive, the `ingest` CLI verb, and the `IngestPolicy` seam. **The
single-feed pipeline is the ingestion trigger.** The durable outcome of #1069 is instead
making **enrichment a consistent peer of the pipeline** across every operator surface, so the
two derived-data producers are managed identically:

- CLI: `python -m podcast_scraper.cli enrich` (peer of the pipeline verb).
- Docker: enrichment spawns through the same docker factory.
- Admin UI + scheduler: a `kind` field (`pipeline` | `enrichment`) on jobs + scheduled jobs.
- Auto-after-pipeline: an optional enrichment pass chained after a successful pipeline run.

A latent bug surfaced and was fixed in the same work: the jobs registry rebuilt the pipeline
argv for *every* job kind, so an enrichment job silently ran the pipeline. The registry now
spawns each job's **own stored command** (`argv_from_record`).

## Consequences

- One ingestion path (the pipeline), one mental model. No `ingest` verb, no `IngestPolicy`,
  no `src/podcast_scraper/ingestion/`.
- Enrichment is a first-class peer of the pipeline across CLI / docker / scheduler / UI /
  auto-chain.
- The **consumer** self-serve scrape surface (Podcast Index `DiscoverySource`, add-to-library
  and progress UI, guardrails) remains a separate, later epic — unaffected by this decision.

## Alternatives considered

- **Keep the `ingest` primitive** as a thin wrapper — rejected; it duplicated the pipeline
  with no added capability and forked the operator's model.
- **Fold enrichment *into* the pipeline** rather than making it a peer — rejected; enrichment
  is opt-in, separately gated, and separately scheduled, so a peer verb is the honest shape.
