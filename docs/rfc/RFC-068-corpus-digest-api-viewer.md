# RFC-068: Corpus Digest — API & Viewer Integration

- **Status**: Completed (v2.6.0) — `GET /api/corpus/digest`, Digest tab, Library 24h glance, `corpus_digest_api` on `GET /api/health`
- **Authors**: Podcast Scraper Team
- **Stakeholders**: Core team, viewer users, pipeline maintainers
- **Related PRDs**:
  - [PRD-023: Corpus Digest & Library Glance](../prd/PRD-023-corpus-digest-recap.md)
  - [PRD-022: Corpus Library & Episode Browser](../prd/PRD-022-corpus-library-episode-browser.md)
  - [PRD-021: Semantic Corpus Search](../prd/PRD-021-semantic-corpus-search.md)
  - [PRD-004: Per-Episode Metadata](../prd/PRD-004-metadata-generation.md)
  - [PRD-005: Episode Summarization](../prd/PRD-005-episode-summarization.md)
- **Related RFCs**:
  - [RFC-067: Corpus Library](RFC-067-corpus-library-api-viewer.md) — catalog rows, detail
    fields, handoffs
  - [RFC-061: Semantic Corpus Search](RFC-061-semantic-corpus-search.md) — `/api/search`
  - [RFC-062: GI/KG viewer v2](RFC-062-gi-kg-viewer-v2.md) — shell, tabs, Search panel
  - [RFC-063: Multi-Feed Corpus](RFC-063-multi-feed-corpus-append-resume.md)
- **Related UX specs**:
  - [UXS-002: Corpus Digest](../uxs/UXS-002-corpus-digest.md) — Digest tab and discovery layout (PRD-023)
  - [UXS-001: GI / KG viewer](../uxs/UXS-001-gi-kg-viewer.md) — shared tokens and shell conventions
- **Related Documents**:
  - [ADR-064: Canonical server layer](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md)

## Abstract

This RFC specifies **read-only** FastAPI endpoints and **Vue** integration for **Corpus
Digest**: a **ranked, feed-diverse** slice of recent episodes (metadata + optional summary
lines + GI/KG flags), plus **semantic topic bands** on the **Digest** tab using a **global
fixed topic list** and existing search infrastructure. A **compact** response mode feeds
the **Library glance** (24h only); the full payload supports the **Digest** tab (default
7d window). **Architecture alignment:** consumption-only server (RFC-062 / ADR-064); reuse
catalog/search helpers; no pipeline mutations.

## Problem Statement

Library pagination answers *what exists*; operators still need a **low-friction** answer to
*what to open first* across many feeds. Duplicating ad hoc client logic in Vue would
diverge from RFC-067 and complicate testing. A **single digest contract** with explicit
**diversity**, **window**, and **optional L1 search** keeps behavior consistent between the
**Digest** tab and **Library glance**.

**Use cases:**

1. **Daily orientation:** Open **Digest** → scan 7d diverse rows → open graph or search.
2. **While browsing:** Open **Library** → read **24h glance** → collapse to focus on
   catalog columns.
3. **Topic pivot:** On **Digest**, expand a **topic band** → hand off to Search with query
   + corpus scope.

## Goals

1. **`GET /api/corpus/digest`** returns a stable JSON shape for **L0** (rows) and **L2**
   (badges/paths); **L1** (topics) included when `include_topics=true` and index
   available.
2. **Feed diversity** uses a **documented deterministic** algorithm shared by compact and
   full modes.
3. **Latency:** Documented budgets; partial degradation (skip topics on timeout) without
   failing L0.
4. **Viewer:** New **Digest** main tab; **Library** embeds **glance** region; handoffs
   match RFC-067 tables where applicable.

## Constraints & Assumptions

**Constraints:**

- **Path safety:** Same as RFC-067 (`path` query, resolved corpus root, reject traversal).
- **No unread state:** Windows are **calendar**-relative only (PRD-023).
- **Search dependency:** L1 requires a **working vector index**; if missing, return
  `topics: []` and optional `topics_unavailable_reason`.

**Assumptions:**

- Metadata discovery matches RFC-067 (`discover_metadata_files`, YAML/JSON).
- Internal calls to search reuse the same code paths as `/api/search` (no second index).

## Design & Implementation

### 1. Topic configuration (global fixed list)

- **File:** `config/digest_topics.yaml` (repo-root), or equivalent under `config/` as
  chosen during implementation.
- **Shape (logical):** list of objects `{ "id": str, "label": str, "query": str }` where
  `query` is passed to semantic search. If file missing, server uses a **small built-in
  default** list (hard-coded or packaged YAML) so dev installs work out of the box.
- **PR review:** Changes to defaults are product-visible; treat like other config.

### 2. Diversity algorithm (L0)

**Input:** Episodes in window, sorted by **`publish_date` desc**, tie-break
**`metadata_relative_path` asc** (match RFC-067 stability).

**Selection:** **Round-robin** by `feed_id`: iterate the sorted list, emit an episode for
each feed in turn until **row cap** reached or list exhausted. **Cap per feed:** e.g. max
**3** rows per feed for Digest tab, **2** for Library glance (exact numbers configurable
via query params with server-enforced maxima).

**Alternative considered:** Pure recency then truncate — **rejected** (one feed dominates).

### 3. `GET /api/corpus/digest`

**Query parameters:**

| Param            | Type    | Description |
| ---------------- | ------- | ----------- |
| `path`           | string  | Corpus root (optional if server default set) |
| `window`         | enum    | `24h` \| `7d` \| `1mo` \| `since` (`1mo` = previous calendar month, UTC bounds) |
| `since`          | date    | Required when `window=since` (`YYYY-MM-DD`) |
| `compact`        | bool    | Default `false`. When `true`: force **24h** window, **smaller** row cap, **omit** `topics` regardless of `include_topics` |
| `include_topics` | bool    | Default `true` when `compact=false`; ignored when `compact=true` |
| `max_rows`       | int     | Optional override; server clamps to **max** (e.g. 50 full, 8 compact) |

**Response (success):**

- `path`: resolved corpus root
- `window`, `window_start_utc` / `window_end_utc` or ISO range fields (implementation picks
  one style; document for clients)
- `rows`: array of **digest rows**:
  - Fields from RFC-067 list rows where applicable: `metadata_relative_path`, `feed_id`,
    `episode_title`, `publish_date`
  - `summary_title`, `summary_bullets_preview` (first **1–4** summary bullets)
  - `has_gi`, `has_kg`, `gi_relative_path`, `kg_relative_path`
  - **Optional visual metadata (RFC-067 / PRD-023 FR4.3):** same optional keys as catalog
    list rows: `feed_image_url`, `episode_image_url`, `duration_seconds`, `episode_number`,
    plus **Phase 4** `feed_image_local_relpath`, `episode_image_local_relpath` when verified
    on disk — serialized from the same `CatalogEpisodeRow` as `digest_row_dict` (no extra
    reads). Thumbnails use **`GET /api/corpus/binary`** when the viewer has `corpusPath` and
    a local path (RFC-067 Phase 4).
  - Optional **Phase 1b:** `gi_node_count`, `kg_node_count` **only if** O(1) or streaming
    parse is available without loading Cytoscape; otherwise omit
- `topics`: when `include_topics=true` and index OK, array of:
  - `topic_id`, `label`, `query`
  - `hits`: capped list of `{ metadata_relative_path, episode_title, feed_id, score?, feed_image_url?, episode_image_url?, duration_seconds?, episode_number?, feed_image_local_relpath?, episode_image_local_relpath? }`
    intersected with **digest window** (post-filter or search filter per RFC-061
    capabilities). When a hit joins a **catalog row**, optional visual fields (including
    Phase 4 local relpaths) are copied from that row; hits without `metadata_relative_path`
    stay text-only.
- `topics_unavailable_reason`: optional string when index missing or timeout

**Errors:** Align with RFC-067 (`404` invalid path, etc.).

**Performance:**

- **Target:** p95 **&lt; 3s** digest **without** topics on 10k-episode corpus (metadata
  scan may use same strategies as RFC-067 Phase 2 cache when available).
- **Topics:** Each topic query **bounded** (timeout e.g. **800ms** per topic, **max 5**
  topics per request); exceed → omit that topic’s hits, continue.

### 4. Relation to existing endpoints

- Prefer **internal functions** shared with `corpus_library` routes for metadata walks.
- Do **not** replace `GET /api/corpus/episodes`; digest is a **derived** view.

### 5. Frontend (Vue)

**Shell (`App.vue`):**

- Extend `mainTab` union with `'digest'`.
- Nav button **Digest** next to Graph / Dashboard / Library (UXS-002, shell per UXS-001).

**Components:**

- `components/digest/DigestView.vue` — window controls, `rows`, `topics`, handoff buttons;
  **optional** compact thumbnails on **main digest cards** and **topic hit** rows (reuse
  RFC-067 `PodcastCover`: **local binary URL first** when `shell.corpusPath` + verified
  `*_image_local_relpath`, else episode URL with feed fallback).
- `LibraryView.vue` — **glance** `CollapsibleSection` (default **open**), title e.g.
  **New (24h)**; fetches `compact=true`; **Open full Digest** sets `mainTab='digest'`; glance
  rows may show the same optional thumbnails when response includes URLs.

**API module:** `src/api/digestApi.ts` — `fetchDigest(params)`.

**Handoffs:**

- **Graph:** Same as RFC-067 (`useArtifactsStore`, load GI/KG, `mainTab='graph'`).
- **Search:** Set `rightOpen`, `rightTab='search'`, feed filter + query per RFC-067; topic
  band buttons pre-fill **`query`** from topic `query` string.

**State:** Reuse `shell.corpusPath` and health gates like Library.

## Testing Strategy

- **Unit:** Diversity function on synthetic sorted input (feeds A/B/C).
- **Integration:** `GET /api/corpus/digest` against temp corpus (flat + `feeds/`), with
  and without search index (mock or skip topics).
- **E2E (optional Phase 1):** Playwright smoke for Digest tab + glance visible strings;
  update `E2E_SURFACE_MAP.md` first.

## Rollout & Monitoring

**Phase 1:** Endpoint + Digest tab + Library glance (L0 + L2 badges; L1 when index
present).

**Phase 2:** Cached metadata walk (align RFC-067 Phase 2); optional node counts.

**Phase 3:** Per-corpus topics; “since last visit” only if a future PRD adds persistence.

## Relationship to Other RFCs

- **RFC-067:** Source of truth for episode list fields and Library handoffs; digest **does
  not fork** path resolution.
- **RFC-061:** Search semantics, filters, and degradation when FAISS unavailable.

## Open Questions

1. Exact **`window`** encoding for `24h` (rolling from server now vs calendar day in user
   TZ) — default **UTC rolling 24h** unless PRD prefers calendar day.
2. Whether `hits` in topic bands dedupe against `rows` (UX choice; recommend **allow
   overlap** for MVP).

## References

- **PRD-023**: `docs/prd/PRD-023-corpus-digest-recap.md`
- **UXS-002**: `docs/uxs/UXS-002-corpus-digest.md` (Digest surface); **UXS-001**: shared design system
