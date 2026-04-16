# UXS-005: Semantic Search Panel

- **Status**: Active
- **Authors**: Podcast Scraper Team
- **Parent UXS**: [UXS-001: GI/KG Viewer](UXS-001-gi-kg-viewer.md) -- shared tokens,
  typography, layout, states
- **Related PRDs**:
  - [PRD-021: Semantic Corpus Search](../prd/PRD-021-semantic-corpus-search.md)
- **Related RFCs**:
  - [RFC-061: Semantic Corpus Search](../rfc/RFC-061-semantic-corpus-search.md)
  - [RFC-062: GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md)
  - [RFC-072: Canonical identity + cross-layer bridge](../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md) (chunk-to-Insight **lift** on transcript hits)
  - [RFC-075: Corpus Topic Clustering](../rfc/RFC-075-corpus-topic-clustering.md) (optional **Show on graph** / cluster follow-ups)
- **Implementation paths**:
  - `web/gi-kg-viewer/src/components/search/SearchPanel.vue`
  - `web/gi-kg-viewer/src/components/search/ResultCard.vue`
  - `web/gi-kg-viewer/src/components/search/SearchResultsVizDialog.vue`
  - `web/gi-kg-viewer/src/components/search/SemanticSearchTip.vue`
  - `web/gi-kg-viewer/src/stores/search.ts`

---

## Summary

The semantic search panel provides FAISS-based corpus search within the viewer's
right rail. This UXS defines the visual contract for the search form, advanced
filters, result cards, and the search result insights modal. All tokens reference
[UXS-001](UXS-001-gi-kg-viewer.md).

**RFC-075:** When corpus clustering JSON is available, **Show on graph** from search still selects the
**leaf** node id (e.g. `topic:…`) as today. The **graph node rail** shows **Topic cluster:** for Topic
nodes (Phase 1; see [UXS-004](UXS-004-graph-exploration.md)). **Search result cards** may show a
**Topic cluster:** line when the API joined **`metadata.topic_cluster`** (canonical label and compound
id). **Show on graph** may widen the camera to include the **`tc:`** compound parent while keeping
selection on the leaf.

**API · Data (left rail):** Under the **Data** heading, the **first** card is **Topic clusters**.
It reflects **`GET /api/corpus/topic-clusters`**
as soon as **Corpus path** is set and **health** is OK — you do **not** need to wait for GI/KG
artifacts to finish loading into the graph. While the request is in flight, **Status** shows **Checking…**.
Then: **Loaded**, **Not built** (404 — optional
`topic-clusters` CLI), **request error**, or **Local files only** when the graph came from the file
picker (no API fetch for clustering). Unknown **`schema_version`** values get a non-blocking warning line.

---

## Primary flow

Search query field (no separate label; placeholder + **Semantic search** heading),
then **Since (date)** and **Top-k** on one compact row; **Advanced search** link;
optional read-only **Advanced filters** summary when any advanced control differs
from defaults; **Search** / **Clear** last.

---

## Advanced search

Small underlined control opens a modal dialog with:

- **Feed** (substring on catalog `feed_id` for the API; Library -> Prefill semantic
  search shows the feed title from the feeds catalog when known, with hover/title for
  the id until edited)
- **Grounded insights only**
- **Speaker contains**
- **Embedding model**
- **Merge duplicate KG surfaces** (default on: same behavior family as graph
  Entity/Topic dedupe for `kg_entity` / `kg_topic` vector rows)
- **Doc Types** (empty = all)

---

## Search result insights

After at least one hit, an underlined **Search result insights** control opens a
modal (same backdrop pattern as Advanced search) titled "Search result insights" --
one scroll, no tabs: a short insight line (dominant doc type); **Doc types** and
**Publish month** in a two-column row (small multiples); **Episodes** / **Feeds**
with top rows (episode title / feed title from hit metadata, or loaded from the
episode's `*.metadata.json` when the index row omitted them) plus "+N other..." tail
counts; **Similarity** bars proportional to score / max(score) in the list
(captioned); **Terms** with a top-token insight (word frequency; heuristic, not KG).

---

## Results

A muted "N results" / "1 result" line only (the query stays in the textarea; it is
not repeated here). When the API returns **`lift_stats`** and at least one transcript
row is on the page, a compact **Lift: applied / transcript rows** ratio appears on
the same row (native **`title`** explains RFC-072 lift coverage).

Each hit can expose:

- **G** (graph focus, GI token) and **L** (Library episode) as separate controls
- **L** requires a healthy API check + corpus path and `source_metadata_relative_path`
  on the hit (vector indexer stamps it on rebuild)
- `corpus_library_api` in health can still be No while L shows; the Library tab
  surfaces errors if catalog routes are unavailable
- **E** (episode id chip) is informational
- When **Merge duplicate KG surfaces** merged a row (`kg_surface_match_count` >= 2),
  G only -- L and E are hidden so actions are not tied to a single representative
  episode
- **Transcript** hits may include a collapsible **`region` Lifted GI insight** (linked
  insight id/text, speaker/topic labels, quote time range) when the server attaches
  **`lifted`** (#528). When **`lifted.quote`** has at least one finite **`timestamp_*_ms`**
  but **`lifted.speaker`** has no usable display label, a muted line shows the same visible
  copy as supporting quotes — **No speaker detected** (**`GI_QUOTE_SPEAKER_UNAVAILABLE_HINT`**;
  **#541**). **`data-testid="search-lifted-quote-speaker-unavailable"`**.
- **Supporting quotes** (Show / Hide *N* supporting quote(s)): when the API returns
  **`supporting_quotes`** and a quote has **no** **`speaker_name`** / **`speaker_id`**, a
  muted line shows **No speaker detected** (same **`GI_QUOTE_SPEAKER_UNAVAILABLE_HINT`** as the graph Quote rail; **#541**). **`data-testid="supporting-quote-speaker-unavailable"`**.

---

## E2E contract

[E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) --
search panel surfaces and selectors.

---

## Revision history

| Date       | Change                                                                                 |
| ---------- | -------------------------------------------------------------------------------------- |
| 2026-04-06 | Initial content (in UXS-001)                                                           |
| 2026-04-13 | Extracted from UXS-001 into standalone UXS-005                                         |
| 2026-04-13 | Document lift_stats summary line + Lifted GI insight region                            |
| 2026-04-15 | Supporting quotes: muted hint when speaker missing (#541)                              |
| 2026-04-15 | Lifted GI: muted hint + testid when speaker display missing (#541)                     |
| 2026-04-15 | Lifted hint only when **`lifted.quote`** has finite **`timestamp_*_ms`** (matches E2E) |
| 2026-04-15 | #541: **No speaker detected** (graph + Search + Explore; semantics unchanged)          |
| 2026-04-16 | Lifted GI: explicit same visible string as supporting quotes (**No speaker detected**) |
