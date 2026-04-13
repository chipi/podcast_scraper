# RFC-067: Corpus Library — Catalog API & Viewer Integration

- **Status**: Completed (v2.6.0) — Phases 1–3 shipped (`/api/corpus/*`, Library tab, episode detail, similar episodes)
- **Authors**: Podcast Scraper Team
- **Stakeholders**: Core team, viewer users, pipeline maintainers
- **Related PRDs**:
  - [PRD-022: Corpus Library & Episode Browser](../prd/PRD-022-corpus-library-episode-browser.md)
  - [PRD-004: Per-Episode Metadata](../prd/PRD-004-metadata-generation.md)
  - [PRD-005: Episode Summarization](../prd/PRD-005-episode-summarization.md)
  - [PRD-021: Semantic Corpus Search](../prd/PRD-021-semantic-corpus-search.md)
  - [PRD-023: Corpus Digest & Library Glance](../prd/PRD-023-corpus-digest-recap.md)
- **Related RFCs**:
  - [RFC-062: GI/KG viewer v2](RFC-062-gi-kg-viewer-v2.md) — FastAPI shell, Vue SPA, artifact loading
  - [RFC-061: Semantic Corpus Search](RFC-061-semantic-corpus-search.md) — `/api/search` handoff
  - [RFC-063: Multi-Feed Corpus](RFC-063-multi-feed-corpus-append-resume.md) — `feeds/` layout, discovery
  - [RFC-004: Filesystem Layout](RFC-004-filesystem-layout.md) — output directory conventions
  - [RFC-068: Corpus Digest](RFC-068-corpus-digest-api-viewer.md) — digest API & viewer (PRD-023)
- **Related UX specs**:
  - [UXS-003: Corpus Library](../uxs/UXS-003-corpus-library.md) — Library tab, Episode rail, catalog layout
  - [UXS-001: GI / KG viewer](../uxs/UXS-001-gi-kg-viewer.md) — shared tokens and shell conventions
- **Related Documents**:
  - [ADR-064: Canonical server layer](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md) — route mounting pattern

## Abstract

This RFC specifies **Phase 1–3** technical design for the **Corpus library**: new FastAPI
routes that list **feeds** and **episodes** from on-disk episode metadata (JSON/YAML),
paginated APIs, an **episode detail** payload including **summary bullets** and sibling
**GI/KG paths**, and **Vue** integration as a **main tab** in the existing GI/KG viewer.
Phase 1 is **read-only**, reuses `discover_metadata_files()` and corpus path resolution
consistent with `/api/artifacts` and `/api/search`. Phase 2 adds **scalability** (cache or
persisted catalog). Phase 3 adds **deeper semantic integration** with the vector index.
**Phase 4** adds optional **corpus-local artwork** download + **`GET /api/corpus/binary`**
(see § Phase 4 below).

**Architecture alignment:** The server remains a **consumption layer** (RFC-062 / ADR-064):
no pipeline mutations, no new required CLI flags for Phase 1. Catalog logic lives in
`src/podcast_scraper/server/` and may call shared helpers in
`src/podcast_scraper/search/corpus_scope.py` and
`src/podcast_scraper/utils/corpus_episode_paths.py`.

## Problem Statement

The viewer can list **graph artifacts** and run **semantic search**, but there is no
efficient, podcast-oriented **catalog** backed by the same **metadata files** the pipeline
already writes. Users need:

1. **Feed-scoped episode lists** with **pagination** for large corpora.
2. **Episode detail** including **summaries** without loading full `.gi.json` / `.kg.json`.
3. **Deterministic handoffs** to **graph** (artifact load) and **search** (query + filters).

**Use cases:**

1. **Browse by show:** Select `feed_id` → scroll episodes newest-first.
2. **Skim summaries:** Open detail → read bullets → decide whether to open graph.
3. **Pivot to search:** From an episode, run semantic search with **feed** pre-set and
   optional **title** in the query box.

## Goals

1. **Phase 1:** Ship three REST endpoints under `/api/corpus/` with stable JSON shapes,
   path traversal safety, and tests on synthetic corpus layouts (flat + `feeds/`).
2. **Phase 1:** Add **Library** UI per UXS-003; reuse `corpusPath` from the shell store;
   hand off to artifacts store and search store without duplicating path logic.
3. **Phase 2:** Document and implement **catalog caching or materialization** so 10k+
   episodes remain usable.
4. **Phase 3:** Document **vector-assisted** features (similar episodes, richer facets)
   using existing indexer metadata where possible.

## Non-Goals (Phase 1)

- Writing or mutating metadata from the library APIs via catalog routes (metadata is still
  produced by the pipeline / CLI, not by `GET /api/corpus/*`).
- Replacing `/api/artifacts` or `/api/search` with a single mega-endpoint.
- **Generic reverse proxy** for arbitrary third-party image URLs without an ingest step —
  not in scope; operators who need same-origin art enable **Phase 4** (optional download at
  metadata generation).

## Phase 4 — Optional corpus-local artwork (implemented)

**Goal:** Same-origin thumbnails when images are **copied into the corpus** at ingest time,
avoiding hotlinking for those assets. **Default:** unchanged URL-only behavior when
`download_podcast_artwork` is **false** (opt-out via config/CLI) or local files are absent.

**Pipeline / metadata (PRD-004):**

- Config: **`download_podcast_artwork`** (`bool`, default **`true`**; CLI
  **`--no-download-podcast-artwork`** to disable). When **true** with
  metadata generation (non-dry-run), the workflow downloads `feed.image_url` /
  `episode.image_url` targets into **`<corpus>/.podcast_scraper/corpus-art/sha256/…`** and
  sets optional **`feed.image_local_relpath`** / **`episode.image_local_relpath`** (corpus-
  relative POSIX paths).
- Implementation: `src/podcast_scraper/utils/corpus_artwork.py`,
  `src/podcast_scraper/workflow/metadata_generation.py`.

**Catalog:**

- `CatalogEpisodeRow` carries **`feed_image_local_relpath`** /
  **`episode_image_local_relpath`** only when the path is under
  **`.podcast_scraper/corpus-art/`** and the file **exists** on disk (anti-traversal
  consistent with other corpus routes).
- `aggregate_feeds` may set **`image_local_relpath`** for a feed from the first verified
  feed-level local path seen.

**HTTP API:**

- **`GET /api/corpus/binary`** — Query: **`path`** (corpus root), **`relpath`** (required).
  **`relpath`** must start with **`.podcast_scraper/corpus-art/`**; returns **`FileResponse`**
  with guessed `Content-Type` or `404` / `400`. Implementation:
  `src/podcast_scraper/server/routes/corpus_binary.py`.
- **`GET /api/corpus/feeds`** — each feed may include optional **`image_local_relpath`**.
- **`GET /api/corpus/episodes`**, **`GET /api/corpus/episodes/detail`**, **`GET /api/corpus/episodes/similar`** —
  optional **`feed_image_local_relpath`**, **`episode_image_local_relpath`** on each item
  (mirrors digest row shape in RFC-068).
- **`GET /api/health`** — includes **`corpus_binary_api: true`** when the binary route is
  mounted (default for current server builds).

**Viewer:**

- **`PodcastCover.vue`** — when **`corpusPath`** is set, prefers **episode local → episode
  URL → feed local → feed URL**; local `src` is
  **`/api/corpus/binary?path=${encodeURIComponent(corpusPath)}&relpath=${encodeURIComponent(rel)}`**.

**Tests:** Integration coverage in `tests/integration/server/test_viewer_corpus_library.py`
(binary allowlist, verified local fields on list/detail).
- Introducing **vue-router** as a hard dependency (optional later; MVP uses a third **main**
  tab in `App.vue`).

## Constraints & Assumptions

**Constraints:**

- **Path safety:** All relative paths are resolved under the **resolved corpus root** using
  the same pattern as
  `artifacts.py` in `src/podcast_scraper/server/routes/` (`..` and absolute
  segments rejected).
- **Corpus root query param** name stays **`path`** for consistency with `/api/artifacts`,
  `/api/search`, `/api/explore` (required or optional per endpoint below; mirror existing
  semantics: omit → `app.state.output_dir` when set).

**Assumptions:**

- Episode metadata files match `*.metadata.json`, `*.metadata.yaml`, or `*.metadata.yml`
  as discovered by `discover_metadata_files()` in
  `corpus_scope.py` in `src/podcast_scraper/search/`.
- Summary shape matches normalized pipeline output (`summary.bullets`, optional
  `summary.title`) as produced by metadata generation (PRD-005).

## Data Model (logical)

Each metadata file yields at minimum:

| Field | Source |
| ----- | ------ |
| `metadata_relative_path` | Path relative to corpus root (POSIX) |
| `feed_id` | From `feed.feed_id` in metadata doc, normalized (see `normalize_feed_id`) |
| `episode_id` | From `episode.episode_id` when present |
| `episode_title` | From episode or RSS fields in metadata (implementation maps common keys) |
| `publish_date` | ISO date string when parseable; nullable |
| `gi_relative_path` / `kg_relative_path` | Sibling paths if files exist (same rules as `corpus_episode_paths`) |
| `summary_title` / `summary_bullets` | From `summary` object when present |
| `feed_image_url` | From `feed.image_url` when present (non-empty string) |
| `episode_image_url` | From `episode.image_url` when present |
| `duration_seconds` | From `episode.duration_seconds` when present (integer ≥ 0) |
| `episode_number` | From `episode.episode_number` when present (integer ≥ 0) |
| `feed_image_local_relpath` | Verified path from `feed.image_local_relpath` when file exists (Phase 4) |
| `episode_image_local_relpath` | Verified path from `episode.image_local_relpath` when file exists (Phase 4) |

**Visual metadata (optional, PRD-022):** Populated from existing per-episode metadata
documents (PRD-004); **no live RSS refetch** in the server. Missing keys remain absent or
`null` in JSON. **Feed list** includes `image_url` per feed when any episode row for that
`feed_id` carries `feed.image_url` (first non-empty wins during aggregation). Clients should
use **episode image first**, then **fallback to feed image** for episode-level thumbnails.
When **Phase 4** local paths are present and verified, API consumers and **`PodcastCover`**
prefer **same-origin** `GET /api/corpus/binary` (see Phase 4) before hotlinked URLs.

**Sorting key for pagination:** Prefer `(publish_date desc, metadata_relative_path desc)` or
`(publish_date desc, metadata_relative_path asc)` — pick one and document; stable across
requests.

## Phase 1 — API Design

New router module e.g. `src/podcast_scraper/server/routes/corpus_library.py`, registered
in `app.py` (`src/podcast_scraper/server/app.py`) with `prefix="/api"` (routes below
include `/corpus/...`).

### 1. `GET /api/corpus/feeds`

**Query:**

- `path` (optional if server default `output_dir` is set): corpus root directory.

**Response (success):**

- `path`: resolved corpus root string
- `feeds`: array of `{ "feed_id": str, "display_title": str | null, "episode_count": int, "image_url": str | null, "image_local_relpath": str | null }` (`image_url` / `image_local_relpath` optional; OMIT or `null` when unknown)

**Episode count (Phase 1):** May be computed by a single metadata scan or omitted (`null`)
if the first implementation defers counts to avoid double walks; PRD allows optional counts.
If omitted, document and return `episode_count: null` or omit field consistently.

**Errors:**

- `404` / JSON `detail` when path invalid (match existing `CorpusPathRequestError` behavior).
- Empty corpus: `feeds: []` with `200` (not an error).

### 2. `GET /api/corpus/episodes`

**Query:**

- `path` — corpus root (optional if default set)
- `feed_id` — optional filter (exact match on normalized id)
- `q` — optional case-insensitive substring on episode title
- `topic_q` — optional case-insensitive substring on summary title or any summary bullet
- `since` — optional `YYYY-MM-DD` (include episodes on or after publish date when parseable)
- `limit` — integer, default e.g. `50`, max e.g. `200`
- `cursor` — opaque string or absent for first page

**Response:**

- `path`, `feed_id` (echo), `items`: array of **summary rows**:
  `{ "metadata_relative_path", "feed_id", "feed_display_title", "topics", "episode_id", "episode_title", "publish_date", "feed_image_url"?, "episode_image_url"?, "duration_seconds"?, "episode_number"?, "feed_image_local_relpath"?, "episode_image_local_relpath"? }`
  (`feed_display_title` from metadata when present, else `null`; `topics` = short strings from summary bullets, capped). Optional visual fields mirror the logical catalog row (see Data Model); omit or `null` when not in metadata or when local files are missing.
- `next_cursor`: string or `null` when no more pages

**Cursor encoding (recommended):** Base64url of JSON
`{ "p": publish_sort_key, "rel": metadata_relative_path }` or simpler offset **only if**
sort is fully stable — prefer keyset pagination for large corpora.

### 3. `GET /api/corpus/episodes/detail`

**Query:**

- `path` — corpus root
- `metadata_relpath` — relative path to the metadata file (same as listed in `items`)

**Response:**

- All list fields plus `summary_title`, `summary_bullets` (array of strings),
  `gi_relative_path`, `kg_relative_path` (nullable strings), `has_gi`, `has_kg` (booleans),
  plus optional `feed_image_url`, `episode_image_url`, `duration_seconds`, `episode_number`,
  `feed_image_local_relpath`, `episode_image_local_relpath` when verified on disk (Phase 4)

**Errors:**

- `400` if `metadata_relpath` escapes corpus root or is not a metadata file
- `404` if file missing

### Error shape consistency

Align with existing JSON error patterns (`detail` for HTTP errors; for soft “no corpus”
where other routes return embedded errors, follow **`/api/search`** style only if we need
`200` with `error` field — **prefer HTTP errors** for invalid `path` to match artifacts).

## Phase 1 — Frontend (Vue)

### Shell integration

- Extend `mainTab` union in
  `App.vue` (`web/gi-kg-viewer/src/App.vue`) with `'library'`.
- Add a **Library** button in the main nav next to Graph / Dashboard with the same styling
  pattern as existing main-view tabs (UXS-001 shell, UXS-003 Library).

### Components

- New directory `web/gi-kg-viewer/src/components/library/`:
  - `LibraryView.vue` — three-region layout (feeds | episodes | detail)
  - Optional subcomponents for list rows and detail card
  - **Cover art:** shared `PodcastCover.vue` for feed rows, episode rows, detail header,
    similar-episode list, and **Library glance** (24h digest strip): lazy-loaded `<img>`,
    fixed aspect ratio, `@error` → placeholder; **local binary URL** (when `corpusPath` +
    verified `*_image_local_relpath`) **before** **episode URL**, then **feed local**, then
    **feed URL**

### State & API

- New `src/api/corpusLibraryApi.ts` — `fetchFeeds`, `fetchEpisodes`, `fetchEpisodeDetail`.
  **Viewer:** `LibraryView` sends **`limit=20`** per page (named constant in the component; server query default may differ when the param is omitted).
- Optional Pinia store `stores/corpusLibrary.ts` or local state inside `LibraryView` for
  MVP; **must** read `corpusPath` and health from
  `shell` store (`web/gi-kg-viewer/src/stores/shell.ts`).

### Handoff to graph

- Use `useArtifactsStore` (`web/gi-kg-viewer/src/stores/artifacts.ts`): load `.gi.json`
  and `.kg.json` via existing `GET /api/artifacts/{path}?path=corpusRoot` flow (or the
  store’s existing helper).
- After successful load, set `mainTab` to `'graph'`.
- If only one of GI/KG exists, load what exists; show inline message for missing artifact.

### Handoff to semantic search

| Action | Behavior |
| ------ | -------- |
| **Search in corpus** (primary) | Set `rightOpen`, `rightTab = 'search'`; set search store **feed** filter to episode’s `feed_id` if present; set query to **episode title** (trimmed) or empty if title missing |
| **Search from first bullet** (optional Phase 1 stretch) | Pre-fill query with first bullet text truncated to N chars (RFC implementer chooses N ≤ 200) |

Debounce and keyboard shortcuts: follow RFC-062 patterns for Search panel focus where
applicable (`useViewerKeyboard` extension optional in a follow-up).

### Optional deep links

Reserved query parameters (implement when low cost):

- `?tab=library` — open Library on load
- `?library_feed=<feed_id>` — preselect feed after feeds load

Document in README or viewer help when implemented.

## Phase 2 — Scale (catalog cache / manifest)

**Motivation:** Full metadata walks on every `feeds` + `episodes` request may exceed
latency targets for **10k+** files.

**Options (choose one or combine in implementation PR):**

1. **In-process TTL cache** keyed by `(corpus_root, mtime_bucket)` where `mtime_bucket` is
   max mtime of metadata files under root (expensive to compute naïvely — may sample or
   use directory mtime heuristics).
2. **On-disk manifest** e.g. `<corpus>/.podcast_scraper/catalog.json` written by a new
   **`podcast catalog refresh`** command or lazily by the server on first request (with
   file lock).
3. **SQLite** sidecar (`catalog.sqlite`) with tables `feeds`, `episodes` and indexes on
   `(feed_id, publish_date)`.

**Invalidation:** Manifest includes `generated_at` and `source_fingerprint` (e.g. hash of
metadata paths + mtimes); server rejects stale manifest when filesystem diverges.

**API compatibility:** Phase 2 **must not break** Phase 1 response shapes; only latency
and internal storage change.

## Phase 3 — Semantic depth

**Examples:**

- **Similar episodes:** Given `metadata_relpath`, issue an internal search using embedding
  of summary text or first bullet (requires model load — gate behind config or reuse search
  service).
- **Facet from index:** Cross-reference `feeds_indexed` from `/api/index/stats` with
  library feed list to show “in vector index” badges.

**Dependencies:** RFC-061 index present; may reuse metadata fields already embedded in
`indexer.py` in `src/podcast_scraper/search/`.

**Implementation (2026-04-11):**

- **`GET /api/corpus/episodes/similar`** — `path`, `metadata_relpath`, optional `top_k`
  (1–25, default 8). Builds a query from `summary.title` + `summary.bullets` (fallback:
  episode title), runs the same FAISS path as `run_corpus_search`, dedupes hits by
  `(feed_id, episode_id)`, drops the source episode, enriches rows with catalog titles/paths
  via `index_rows_by_feed_episode`. Returns **200** with `items` or a soft `error`
  (`no_index`, `insufficient_text`, `embed_failed`, …) like `/api/search`.
- **Library UI** — Loads `GET /api/index/stats` when feeds load; shows an **Indexed** chip
  on feeds present in `feeds_indexed` and **Similar episodes** in the detail column
  (auto-calls the similar endpoint after episode detail loads; lists peers with scores when the index is available).
- **`GET /api/index/stats` → `feeds_indexed`** — Response list is **deduplicated, sorted, and
  `normalize_feed_id`-trimmed** so Library chips match catalog `feed_id` strings.
- **Search handoff** — **Search in corpus** opens the Search panel with the same **summary-derived
  query** as similarity search (not title-only); **Feed** filter (substring on catalog feed id) matches
  `GET /api/search` semantics.
- **Code:** `corpus_similar.py` in `src/podcast_scraper/search/`, route in
  `routes/corpus_library.py`, schemas `CorpusSimilarEpisodesResponse` / `CorpusSimilarEpisodeItem`.

## Security

- Reuse `resolve_corpus_path_param` from
  `pathutil.py` in `src/podcast_scraper/server/`.
- Validate `metadata_relpath` with the same segment rules as artifact paths (no `..`).
- Reject symlinks escaping corpus if the codebase already centralizes that check; otherwise
  document follow-up.

## Testing

| Layer | Scope |
| ----- | ----- |
| **Unit** | Pagination ordering; cursor encode/decode; metadata parsing edge cases (missing fields) |
| **Integration** | FastAPI routes with temp dirs: flat layout + `feeds/<id>/metadata/...` |
| **E2E** | After UI lands: Playwright flows for tab switch, list load, handoff (update E2E_SURFACE_MAP first) |

## Alternatives Considered

- **vue-router** for `/library` — cleaner URLs but adds bundle and boot complexity; defer.
- **GraphQL or single combined query** — unnecessary for local tool; REST stays debuggable.
- **Only client-side scan** — would require exposing all metadata paths or huge payloads;
  rejected.

## Open Questions

1. Should `episode_count` be mandatory in `GET /api/corpus/feeds` Phase 1 or deferred to
   Phase 2 for a single-pass indexer?
2. Exact metadata key mapping for `episode_title` across older runs — document in code as
   `TITLE_KEYS` fallback list.

## Revision History

| Date | Change |
| ---- | ------ |
| 2026-04-10 | Initial draft (Phases 1–3) |
| 2026-04-11 | Phase 1 implemented: `corpus_catalog.py`, `/api/corpus/*` routes, Library tab, tests + E2E |
| 2026-04-11 | Phase 2 deferred; decision log appended (no Phase 2 code) |
| 2026-04-11 | Phase 3 implemented: `/api/corpus/episodes/similar`, index badges + similar UI in Library |
| 2026-04-11 | Follow-up: canonical `feeds_indexed` in `/api/index/stats`, Library→Search summary handoff, similar empty state, tests/mocks |
| 2026-04-10 | **Phase 4:** optional `download_podcast_artwork`, `.podcast_scraper/corpus-art/`, `GET /api/corpus/binary`, catalog verified `*_image_local_relpath`, health `corpus_binary_api`, `PodcastCover` same-origin preference |

## Phase 2 status (decision log)

**2026-04-11 — Phase 2 not implemented (deferred).** The catalog cache / manifest / SQLite
design in **Phase 2 — Scale (catalog cache / manifest)** above remains the **planned**
shape for when scale demands it; no code changes for Phase 2 were made at this time.

**What we concluded**

- **Phase 1** walks all episode metadata on each `GET /api/corpus/feeds` and
  `GET /api/corpus/episodes` (after filters, pagination slices in memory). That is **fine**
  for exploration and for **small-to-medium** corpora.
- **Phase 2** trades that simplicity for **invalidation**, **staleness**, and optional
  **on-disk** artifacts (`catalog.json`, SQLite) or **CLI** (`podcast catalog refresh`–style).
  That cost is **not justified** until we see **measured pain** (slow Library loads, very large
  trees, or aggressive UI refresh) or until metadata layout stabilizes enough that maintaining
  a cache is low-risk.

**When to revisit**

- Representative corpora in the **10k+ metadata file** range, or clear user feedback that
  list endpoints are too slow.
- Need for **cheap per-request** work (e.g. always-fresh episode counts) without rescanning.

**Suggested first increment when we return:** In-process **TTL cache** plus a **cheap
fingerprint** (e.g. max mtime or count of metadata paths) keyed by resolved corpus root —
before adopting a persisted manifest or SQLite, unless requirements jump straight to
offline/catalog CLI workflows.
