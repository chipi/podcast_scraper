# PRD-022: Corpus Library & Episode Browser

- **Status**: Implemented (v2.6.0) — Phases 1–3 (catalog APIs, Library tab, handoffs) plus **optional Phase 4** corpus-local artwork (RFC-067): ingest-time download behind `download_podcast_artwork`, `GET /api/corpus/binary`, viewer prefers same-origin art when verified paths exist.
- **Authors**: Podcast Scraper Team
- **Related RFCs**:
  - [RFC-067: Corpus Library API & Viewer](../rfc/RFC-067-corpus-library-api-viewer.md) — technical design (Phases 1–3 + **Phase 4** local artwork): **visual metadata** (RSS-sourced URLs, optional **verified** `*_image_local_relpath`, duration, episode number) in API + viewer
  - [RFC-062: GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md) — host SPA, `podcast serve`, FastAPI shell
  - [RFC-061: Semantic Corpus Search](../rfc/RFC-061-semantic-corpus-search.md) — vector search consumed from viewer
  - [RFC-063: Multi-Feed Corpus](../rfc/RFC-063-multi-feed-corpus-append-resume.md) — `feeds/<id>/` layout and discovery
  - [RFC-068: Corpus Digest](../rfc/RFC-068-corpus-digest-api-viewer.md) — Digest tab & Library glance (PRD-023)
- **Related PRDs**:
  - [PRD-004: Per-Episode Metadata](PRD-004-metadata-generation.md) — episode metadata documents (source for catalog)
  - [PRD-005: Episode Summarization](PRD-005-episode-summarization.md) — summaries and bullets in metadata
  - [PRD-021: Semantic Corpus Search](PRD-021-semantic-corpus-search.md) — semantic search handoff from library
  - [PRD-017: Grounded Insight Layer](PRD-017-grounded-insight-layer.md) — GI artifacts opened from library
  - [PRD-019: Knowledge Graph Layer](PRD-019-knowledge-graph-layer.md) — KG artifacts opened from library
  - [PRD-023: Corpus Digest & Library Glance](PRD-023-corpus-digest-recap.md) — 24h glance + Digest tab atop same catalog
- **Related UX specs**:
  - [UXS-003: Corpus Library](../uxs/UXS-003-corpus-library.md) -- Library tab IA,
    Episode subject rail, filters, graph integration
  - [UXS-001: GI/KG Viewer](../uxs/UXS-001-gi-kg-viewer.md) -- shared design system

## Summary

**Corpus Library** adds a **podcast-shaped browsing** experience on top of the same local
corpus the GI/KG viewer already uses: navigate **feeds** and **episodes**, read **titles,
dates, and summary bullets** from on-disk metadata, and **hand off** to the **graph** (load
`.gi.json` / `.kg.json`) and **semantic search** without a separate app, database, or large
refactor. When metadata includes **RSS-derived fields** (PRD-004), the Library also surfaces
**optional** **feed/episode artwork** (remote **URLs** and/or **corpus-local** files served via
`/api/corpus/binary` when present — see RFC-067 Phase 4), **duration**, and **episode number**
in lists and detail so browsing matches listener expectations (placeholders when assets are
missing or fail to load). Phase 1 is **filesystem-backed** and targets
**exploration** and **developer ergonomics** at multi-feed scale (on the order of tens of
feeds and thousands of episodes).

## Background & Context

The pipeline already writes rich **per-episode metadata** (PRD-004) and, when enabled,
**summaries with normalized bullets** (PRD-005). The viewer v2 (RFC-062) exposes:

- **Artifact listing** (`/api/artifacts`) — optimized for picking `.gi.json` / `.kg.json`
  files, not for “which episodes exist under which show.”
- **Semantic search** (PRD-021 / RFC-061) — excellent for retrieval, weak for answering
  “what did we process?” in feed/episode order.
- **Explore** — insight/topic-oriented, not a full episode catalog.

Multi-feed corpora (RFC-063) store episodes under `feeds/<stable_id>/…` with a unified
`search/` index at the corpus parent. Users need a **first-class library** that mirrors
that structure so they can **see processed work**, skim **summaries**, then **drill into**
the graph or search **in two or three steps**.

## Problem Statement

1. **No structured episode catalog in the UI** — There is no single surface that lists
   feeds and episodes with pagination and basic filters grounded in metadata files.
2. **Hard to connect “episode I care about” to graph + search** — Users must mentally map
   filesystem paths to artifact pickers or search queries.
3. **Scale** — Flat or multi-feed trees with thousands of episodes require **pagination**
   and **scoping** (by feed, date, text), not loading everything into the client.

## Goals

- **Filesystem-first MVP (Phase 1):** Derive the catalog from existing metadata discovery
  (no new pipeline stage required for the first slice).
- **Same product shell:** One FastAPI app, one Vue SPA (`web/gi-kg-viewer`), same corpus
   root field as today (RFC-062).
- **Explicit handoffs:**
  - **To graph:** Load the selected episode’s GI and/or KG artifacts using the same
    loading path as manual artifact selection.
  - **To semantic search:** Open the Search panel with useful context (e.g. feed filter,
    optional pre-filled query from episode title or first bullet — exact behavior in RFC-067).
- **Minimal refactor:** Prefer new routes + a new main tab (or equivalent) over
  re-architecting stores or introducing a second frontend.
- **Phased roadmap:** Document **three phases** (MVP, scale, depth) so exploration does
   not block a later catalog index or richer search integration.

## Non-Goals (Initial)

- **Replacing** database projection (PRD-018 / RFC-051) — optional convergence in a later
  phase, not Phase 1.
- **Full in-browser transcript reader** — out of scope for the library MVP; transcripts
  remain on disk / other tools unless a future PRD expands scope.
- **Mandatory** server-side artwork download — remains **off by default**; operators opt in with
  **`download_podcast_artwork`** (default **on**; set **`false`** or **`--no-download-podcast-artwork`**
  to skip) during metadata generation (stores under
  `.podcast_scraper/corpus-art/`, sets `feed.image_local_relpath` / `episode.image_local_relpath`
  in metadata — PRD-004). **Generic HTTP image proxy** for arbitrary remote URLs without ingest
  is still out of scope unless added in a future RFC.
- **Mobile-first or responsive layouts** — UXS-001 remains **desktop baseline** unless a
  future PRD changes that.
- **Auth, multi-tenant hosting, public internet serving** — local developer tool posture
  unchanged.

## Primary Users

- **Developers and analysts** inspecting a local or shared pipeline output directory.
- **Power users** who already run `podcast serve` and the GI/KG viewer.

## User Journeys

1. **Inventory:** Set corpus root → open **Library** → see feeds (or a single implicit
   feed in flat layouts) → paginate episodes → confirm titles and dates match expectations.
2. **Read summary:** Select an episode → view **summary title and bullets** (when present)
   and metadata such as publish date; skim **cover art** (local file preferred when
   downloaded + verified, else hotlinked URL) and **duration** when present in metadata.
3. **Open graph:** From episode detail → **Open GI / Open KG** (or combined action) → app
   switches to **Graph** tab with artifacts loaded.
4. **Search from context:** From episode detail → **Search in corpus** → **Search** panel
   opens with filters or query aligned to that episode (per RFC-067).

## Phased Requirements

### Phase 1 (MVP) — Filesystem catalog + handoffs

- **Feed list:** Distinct `feed_id` (and display title when derivable from metadata) for
  multi-feed corpora; sensible behavior for single-feed / flat layouts.
- **Episode list:** **Server-side pagination**, stable default sort (e.g. publish date
  descending), optional filters: feed, substring on title, optional “since” date.
- **Episode detail:** Episode title, identifiers, publish date, **summary bullets** (and
  summary title when present), flags or paths indicating **GI/KG** sibling artifacts when
  files exist.
- **Visual metadata (when present in episode metadata):** Optional **feed image URL** and
  **episode image URL** (client uses episode art with **fallback** to feed art), optional
  **verified corpus-relative artwork paths** (`feed_image_local_relpath` /
  `episode_image_local_relpath` on API rows — sourced from `feed.image_local_relpath` /
  `episode.image_local_relpath` in metadata when the file exists under
  `.podcast_scraper/corpus-art/`), optional **duration** (e.g. formatted badge), optional
  **episode number** (e.g. `E12`). All fields are **optional** for backward compatibility;
  UI must use **placeholders** for missing or failed loads (RFC-067). **Feed list** may
  include `image_local_relpath` when any row for that feed carries verified feed art.
- **Phase 4 (optional local artwork):** By default **`download_podcast_artwork`** is **on** at
  metadata generation time; the pipeline downloads images into the content-addressed store
  and the viewer uses **`GET /api/corpus/binary?path=…&relpath=…`** (same-origin) when
  `corpusPath` is set — see RFC-067 Phase 4.
- **Actions:** Open graph (load artifacts); open semantic search with agreed handoff
  semantics (RFC-067).
- **Empty / error states:** Clear messaging when corpus path is missing, invalid, or has
  no metadata files.

### Phase 2 — Scale & responsiveness

- **Performance:** Support **10k+ episodes** without unacceptable latency via **server-side
  TTL cache**, optional **on-disk catalog manifest** or small **SQLite** index generated
  lazily or by a dedicated command — exact mechanism in RFC-067.
- **Staleness:** Optional invalidation when metadata mtime changes (conceptual alignment
  with index staleness patterns elsewhere in the server; not mandatory to reuse the same
  module).
- **Optional episode counts per feed** if they can be computed or cached without blocking
  the UI.

### Phase 3 — Depth & search integration

- **Richer semantic workflows:** e.g. “episodes similar to this one” using vector index
  metadata where available (RFC-061).
- **Advanced filters:** Processing flags, presence of summary/GI/KG, additional facets as
  needed.
- **Tighter coupling** between library rows and search hit cards (shared identifiers in
  API responses) where it reduces user confusion.

## Success Criteria

- A user can answer **“what episodes did we process?”** scoped by feed without scanning
  raw directories.
- From the library, a user can reach **graph view** for a chosen episode in **≤ 3 clicks**
  (or equivalent keyboard path) under normal conditions.
- From the library, a user can reach **semantic search** with **meaningful context** in
  **≤ 3 clicks** (per RFC-067 handoff table).
- Phase 2 and Phase 3 are **documented** in RFC-067 so implementation can proceed in
  increments without renegotiating the product intent in this PRD.

## Dependencies & Risks

- **Metadata quality:** Missing or partial metadata reduces catalog richness; UI should
  degrade gracefully (show ids/paths, omit empty summary sections).
- **YAML vs JSON:** Metadata format varies by config; server parsing must support both
  where the pipeline already writes them.
- **Path security:** All catalog endpoints must respect the same corpus root resolution
  and anti-traversal rules as existing artifact routes.

## Related Documents

- [UXS-003: Corpus Library](../uxs/UXS-003-corpus-library.md)
- [UXS-001](../uxs/UXS-001-gi-kg-viewer.md)
- [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md)
  — update when Library UI ships (automation contract)
