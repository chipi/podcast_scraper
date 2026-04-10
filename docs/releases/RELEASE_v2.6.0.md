# Release v2.6.0 — GI/KG Viewer, Corpus Library API, and Index Operations

**Release Date:** April 2026 (planned)
**Type:** Minor Release

## Summary

v2.6.0 is a **minor release** that extends the optional **FastAPI + Vue** viewer stack introduced in RFC-062: a **Corpus Library** backed by on-disk episode metadata, **background vector index rebuild** controls for operators, and **performance profiling** companion tooling (frozen profiles, diff scripts) aligned with RFC-064. The **Python library surface** (`Config`, `run_pipeline`, `service.run`) remains backward compatible; new capabilities are additive and live behind `pip install -e '.[server]'` when you need HTTP or the SPA.

## Key features

### Corpus Library (FastAPI + viewer)

- **HTTP APIs** under `/api/corpus/` — list feeds, paginate and filter episodes, load episode detail (summary bullets + GI/KG paths), and request **FAISS-backed similar episodes** when a vector index is present.
- **Vue viewer** — **Library** flows integrated with the existing shell (`corpusPath`), including handoffs to graph load and semantic search (RFC-067, PRD-022).

Design: [RFC-067](../rfc/RFC-067-corpus-library-api-viewer.md). HTTP reference: [Server Guide](../guides/SERVER_GUIDE.md).

### Index rebuild and staleness (operators)

- **`POST /api/index/rebuild`** — queues `index_corpus` off the request thread (HTTP **202**); **409** when a rebuild is already running for the same corpus.
- **`GET /api/index/stats`** — exposes rebuild job flags (`rebuild_in_progress`, `rebuild_last_error`) and staleness-oriented metadata for the dashboard.

### Performance profiling (release hygiene)

- Checked-in **profile configs** under `config/profiles/` and captured artifacts under `data/profiles/` (operator workflow).
- Scripts and guide: [Performance profile guide](../guides/PERFORMANCE_PROFILE_GUIDE.md), [RFC-064](../rfc/RFC-064-performance-profiling-release-freeze.md).

### Corpus Digest (forward-looking)

- [RFC-068](../rfc/RFC-068-corpus-digest-api-viewer.md) / PRD-023 **Digest** (`GET /api/corpus/digest`), **Digest** tab, **Library** 24h glance, and **`corpus_digest_api`** on **`GET /api/health`** ship in the same viewer release track as Library; see [Server Guide](../guides/SERVER_GUIDE.md).

## Documentation

- [API overview — HTTP / viewer](../api/index.md#http-viewer-api-server-extra)
- [Server Guide](../guides/SERVER_GUIDE.md) — full `/api` table
- [E2E Testing Guide](../guides/E2E_TESTING_GUIDE.md) — Playwright contract (`web/gi-kg-viewer/e2e/`)

## Upgrade notes

- **Library users:** no code changes required.
- **Viewer / API consumers:** ensure `[server]` is installed; see [Migration Guide](../api/MIGRATION_GUIDE.md#v260-viewer-and-http).
