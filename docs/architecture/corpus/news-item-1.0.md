# The news item (News 1.0): the corpus contract

**Status:** Proposed (ingestion, 2026-11-18), for the operator's approval. The writer (E05,
owner ingestion) writes exactly this shape; the read side (public) and the importer (E06) read it.

What is already decided is cited, not restated as new. Everything under "Proposed" waits for the
operator.

## Layout (decided)

- Items live under `sources/news_article/<source>/run_*/` (RFC-134 §5, E05 "Corpus"), outside
  `feeds/`.
- `metadata/<stem>.metadata.json`: the item's metadata (this contract).
- `transcripts/<stem>.txt`: the body, where a transcript sits, one file per item; `document_ref`
  is that path with `variant: raw`, evidence `text_origin: authored` (ontology-v3 "Ids, times and
  documents"). Its paragraph segments, with no times, sit beside it as `<stem>.segments.json`
  (RFC-134 §5 "The extracted text"). Never served (ADR-179).
- `<stem>.kg.json`: KG 3.0. No GI in News 1.0 (E05 decisions).
- The fetched page: a raw cache beside the body, never read by a served route (E05 "Search").

## Top-level fields (decided)

| Field | Value | Source |
| --- | --- | --- |
| `content_type` | `news_article` | RFC-134 §5 |
| `item_id` | `item:news_article:{first 20 hex of SHA-256 of the normalised canonical URL}` | ontology-v3 |
| `url` | the canonical URL | ontology-v3 |
| `source_id` | `source:{first 20 hex of SHA-256 of the source's normalised URL}` | ontology-v3 |
| `title` | as published | ontology-v3 `ContentItem` |
| `published_at` (and `filed_at` etc. when the transport has them) | ISO-8601 | ontology-v3 "Item times" |
| `ingested_at` | set by the run | ontology-v3 |
| `licence`, `paywalled`, `serveable`, `mineable` | as ontology-v3 "Rights, per item" (`serveable`: `metadata` / `summary` / `none`) | ontology-v3, ADR-181 |

The body hash and the fetch time are recorded per item (E05 "Decided", 2026-11-11); their names
are proposed below.

## Proposed (for the operator)

Shared by every type, so top level (RFC-134 §5: title, times, publisher, rights at the top level):

| Field | Value | Why |
| --- | --- | --- |
| `language` | BCP 47 tag of the body (`en`) | every type has one; the search tables split by it |
| `publisher_id` | the registry's organisation id (`podcast_scraper.publishers`), or absent | RFC-134 puts the publisher at the top level; name and domain come from the registry, never copied |
| `summary` | the summary the platform wrote, absent until enrichment | `serveable: summary` shows "the summary the platform wrote" (ontology-v3); every type can have one |
| `fetched_at` | ISO-8601, when the body was fetched | E05's fetch time; any fetched type has it |
| `body_sha256` | hex SHA-256 of `transcripts/<stem>.txt` | E05's body hash; versions start from it later |

News's own, under `news_article` (ADR-181):

| Field | Value | Why |
| --- | --- | --- |
| `transport_ids` | `{transport: native id}`, e.g. `{"rss": "<guid>", "hn": "<objectID>"}` | one article over two transports is one item (E05 "Ids"); a map keeps both |
| `byline` | list of names after split, order kept, organisations removed | E05 "Bylines"; a name joins a person only via ADR-180 |
| `byline_raw` | the byline string as given, before parsing | the golden set evaluates the parser against it |
| `editorial_type` | `reporting` / `opinion` / `analysis` / `paid` / `null` | RFC-134 §2 |
| `text_status` | `full` / `teaser` / `video` | RFC-134 names `teaser`; E05 marks teasers and video items; `full` for the rest |
| `live_blog` | `true` when the source flags the page as a live blog, else absent | E05 "Ids" |

## Not covered

- The search tables' row shape (database's, E05 "Search").
- The publisher registry's own fields (public configuration, database's).
- Versions of a corrected article (News 2.0).
