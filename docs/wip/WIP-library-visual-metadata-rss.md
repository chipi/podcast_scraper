# WIP: Library and Digest visual metadata from RSS/corpus

**Status:** Product/API spec in **PRD-022**, **PRD-023**, **RFC-067**, **RFC-068**; tracking
issue [GitHub #513](https://github.com/chipi/podcast_scraper/issues/513). **Phase 1–3**
(URL-based art + duration + episode number) and **Phase 4** (optional local download +
`GET /api/corpus/binary`) are **implemented** in tree.  
**Audience:** Backend + GI/KG viewer maintainers.

## Problem / opportunity

The Corpus **Library** (`web/gi-kg-viewer/src/components/library/LibraryView.vue`) and
**Digest** (`web/gi-kg-viewer/src/components/digest/DigestView.vue`) benefit from visual
anchors (cover art, duration, episode ordinals). RSS and pipeline metadata already expose
URLs and scalars; **Phase 4** optionally copies artwork into the corpus for same-origin
serving.

The **Library “New (24h)” glance** uses `GET /api/corpus/digest` with `compact=true`, so
digest row fields stay aligned with Library list/detail.

## Current state in the repo (implemented)

| Layer | What exists |
| ----- | ----------- |
| **Stored metadata** | `FeedMetadata` / `EpisodeMetadata` in `metadata_generation.py`: `image_url`, optional **`image_local_relpath`** when `download_podcast_artwork` + `generate_metadata` (non-dry-run). |
| **RSS parsing** | `rss/parser.py` — feed art, `itunes:image`, duration, `itunes:episode`, etc. |
| **Corpus catalog** | `corpus_catalog.py` — `CatalogEpisodeRow` includes URLs, duration, episode number, **verified** `feed_image_local_relpath` / `episode_image_local_relpath`; `aggregate_feeds` sets `image_local_relpath` per feed when present. |
| **HTTP API (Library)** | `schemas.py` + `routes/corpus_library.py` — feeds (`image_local_relpath`), episodes/detail/similar (`feed_image_local_relpath`, `episode_image_local_relpath`). **`GET /api/corpus/binary`** in `routes/corpus_binary.py` (allowlisted under `.podcast_scraper/corpus-art/`). |
| **HTTP API (Digest)** | `digest_row_dict` + digest routes — same optional local + URL fields on rows and `CorpusDigestTopicHit`. |
| **Health** | `GET /api/health` includes **`corpus_binary_api`**. |
| **Viewer** | `PodcastCover.vue` prefers local binary URLs when `corpusPath` is set; `LibraryView` / `DigestView` pass props; `corpusLibraryApi.ts` / `digestApi.ts` types updated. |

## Data sources (priority)

### P0 — Corpus metadata (no live RSS refetch in server)

From each `*.metadata.{json,yaml,yml}`:

- `feed.image_url`, `episode.image_url`, `episode.duration_seconds`, `episode.episode_number`
- **Phase 4:** `feed.image_local_relpath`, `episode.image_local_relpath` (corpus-relative;
  catalog **verifies** file under `.podcast_scraper/corpus-art/` before exposing API fields)

### P1 — Media RSS (`media:thumbnail`, etc.)

Still a follow-up if missing-art rates warrant it (parser extension).

## Backend (historical sketch — see RFCs for normative text)

1. Catalog row + aggregation + schemas + route mapping (done).
2. Digest `digest_row_dict` + topic hits from `catalog_row` (done).
3. Tests: unit `test_corpus_catalog.py`; integration `test_viewer_corpus_library.py`,
   `test_viewer_corpus_digest.py`; binary tests in `test_viewer_corpus_library.py`.

## Frontend (historical sketch)

- Shared **`PodcastCover`**, lazy `loading`, `@error` placeholder.
- Local `src`: `/api/corpus/binary?path=${encodeURIComponent(corpusPath)}&relpath=…`

## Risks and mitigations

- **Hotlinking** — Still used when local paths are absent; **Phase 4** reduces CDN reliance
  for ingested corpora.
- **Broken URLs / stale metadata** — Placeholders remain mandatory.
- **CSP** — Same-origin binary route eases `img-src` for downloaded art.

## Local artwork cache (Phase 4) — implemented

**Storage:** `<corpus>/.podcast_scraper/corpus-art/sha256/<aa>/<bb>/<hash>.<ext>`  
**Config:** `generate_metadata` and `download_podcast_artwork` default **`true`** in
`config.py`; CLI **`--no-generate-metadata`** / **`--no-download-podcast-artwork`** to
disable. Tests use `create_test_config` with both **off** unless a case opts in.  
**Serve:** `GET /api/corpus/binary?path=<corpus>&relpath=<posix-relpath>` — path must start
with `.podcast_scraper/corpus-art/`; traversal rejected.

**Viewer:** `PodcastCover` order: episode local → episode URL → feed local → feed URL.

**Tests:** Integration — 200 for file under allowlist; 400 for bad prefix / `..`; list/detail
return verified `*_image_local_relpath` when metadata + file present.

**Doc / RFC:** PRD-022/023 non-goals updated; RFC-067 Phase 4 section; RFC-068 row/hit
fields; this WIP.

## Suggested rollout (status)

1. **Phase 1** — Catalog + Library + Digest URL fields + shared cover — **done**.
2. **Phase 2** — Duration / episode badges — **done** (with Phase 1).
3. **Phase 3** — Parser `media:*` — **open** (P1).
4. **Phase 4** — Local cache + binary route + pipeline flag + viewer preference — **done**.

## Open questions

- Content-addressed vs per-episode-only layouts (implementation: content-addressed + metadata
  pointers).
- Allowlisted fetch hosts for `http_get` (operational hardening — see code).
- Optional **`season`** in metadata for “S{n} E{m}” UI.

## References

- [RFC-067](../rfc/RFC-067-corpus-library-api-viewer.md)
- [RFC-068](../rfc/RFC-068-corpus-digest-api-viewer.md)
- `extract_feed_metadata` / `extract_episode_metadata` in `src/podcast_scraper/rss/parser.py`
