# The news item (News 1.0): the corpus contract

**Status:** Accepted (the operator, 2026-11-18: "Proposal sounds good"). The writer (E05,
owner ingestion) writes exactly this shape; the read side (public) and the importer (E06) read it.

What was already decided is cited, not restated as new. The names under "Proposed" were proposed by
ingestion and accepted with this contract.

## Layout (decided)

- Items live under `sources/news_article/<source>/run_*/` (RFC-134 §5, E05 "Corpus"), outside
  `feeds/`.
- `metadata/<stem>.metadata.json`: the item's metadata (this contract).
- `transcripts/<stem>.txt`: the body, where a transcript sits, one file per item; `document_ref`
  is that path with `variant: raw`, evidence `text_origin: authored` (ontology-v3 "Ids, times and
  documents"). Paragraphs are separated by a blank line, and no segments file is written: an
  untimed document's segments are the text between blank lines (ontology-v3 "Segments"). Never
  served (ADR-179). (Corrected 2026-11-18: this line first said a `<stem>.segments.json` sat
  beside the body, which contradicted ontology-v3.)
- `<stem>.kg.json`: KG 3.0. No GI in News 1.0 (E05 decisions).
- The fetched page: a raw cache beside the body, `transcripts/<stem>.raw.html` (an API's record:
  `<stem>.raw.json`), never read by a served route (E05 "Search").

## Top-level fields (decided)

| Field | Value | Source |
| --- | --- | --- |
| `content_type` | `news_article` | RFC-134 §5 |
| `item_id` | `item:news_article:{first 20 hex of SHA-256 of the normalised canonical URL}` | ontology-v3 |
| `url` | the canonical URL | ontology-v3 |
| `source_id` | `source:{first 20 hex of SHA-256 of the source's normalised URL}` | ontology-v3 |
| `title` | as published, as plain text (a feed's inline formatting such as `<b>` removed, entities decoded; other angle-bracket text kept as words) | ontology-v3 `ContentItem` |
| `published_at` (and `filed_at` etc. when the transport has them) | ISO-8601 | ontology-v3 "Item times" |
| `ingested_at` | set by the run | ontology-v3 |
| `licence`, `paywalled`, `serveable`, `mineable` | as ontology-v3 "Rights, per item" (`serveable`: `metadata` / `summary` / `none`) | ontology-v3, ADR-181 |

The body hash and the fetch time are recorded per item (E05 "Decided", 2026-11-11); their names
are in the next section.

## Proposed by ingestion, accepted (2026-11-18)

Shared by every type, so top level (RFC-134 §5: title, times, publisher, rights at the top level):

| Field | Value | Why |
| --- | --- | --- |
| `language` | BCP 47 tag of the body (`en`) | every type has one; the search tables split by it |
| `publisher_id` | the registry's organisation id (`podcast_scraper.publishers`), or absent | RFC-134 puts the publisher at the top level; name and domain come from the registry, never copied |
| `summary` | the summary the platform wrote, absent until enrichment | `serveable: summary` shows "the summary the platform wrote" (ontology-v3); every type can have one |
| `fetched_at` | ISO-8601, when the body was fetched | E05's fetch time; any fetched type has it |
| `body_sha256` | hex SHA-256 of `transcripts/<stem>.txt` | E05's body hash; versions start from it later |
| `unusable`, `unusable_reason` | `true` and why, when the item must not be served; else absent. Written last, after `news_article` | D-44, as an episode carries it under `episode`: an item not in English that is not translated keeps its source-language body at the canonical path (added 2026-11-18) |

News's own, under `news_article` (ADR-181):

| Field | Value | Why |
| --- | --- | --- |
| `transport_ids` | `{transport: native id}`, e.g. `{"rss": "<guid>", "hn": "<objectID>"}` | one article over two transports is one item (E05 "Ids"); a map keeps both |
| `byline` | list of names after split, order kept, organisations removed | E05 "Bylines"; a name joins a person only via ADR-180 |
| `byline_raw` | the byline string as given, before parsing | the golden set evaluates the parser against it |
| `editorial_type` | `reporting` / `opinion` / `analysis` / `paid` / `null` | RFC-134 §2 |
| `text_status` | `full` / `teaser` / `video` | RFC-134 names `teaser`; E05 marks teasers and video items; `full` for the rest. A `teaser` or `video` body is never chunked or enriched as if whole (E05 "Extraction"); `serveable` decides what is shown, independently |
| `item_class` | for a `teaser` or `video` item: `news_summary`, `small_teaser`, `day_brief`, `promo` or `not_article` (`podcast_scraper.news_classify`); absent for `full` and for an item written before the classifier | The operator, 2026-11-19: only a `news_summary` teaser is shown; an unclassified teaser is hidden; a `video` item is hidden for now |
| `live_blog` | `true` when the source flags the page as a live blog, else absent | E05 "Ids" |
| `roundup` | `true` when the source's `roundup_title_pattern` matched the title (a newsletter issue covering several stories), else absent | operator, 2026-11-18: a roundup attaches to a storyline only on two shared topics |
| `language_source` | where `language` came from: `override` (the source's `language:`), `page` (`<html lang>`, or the API's), `rss` (the feed's `<language>`), `detected` (the text, when nothing declared one, or when a text of 50 words or more contradicts the page's or the feed's; an override is never overruled. An episode's transcript needs 200: `NEWS_LANGUAGE_MIN_WORDS`) | added 2026-11-18, as episodes record theirs (`languages.SOURCE_*`) |

An item whose language is not enabled in `config/languages.yaml` is not written, as an episode in
that language is not ingested (added 2026-11-18).

Translation is off in News 1.0, by choice (operator, 2026-11-18). A source turns it on with
`translate: true` in its `news:` block. While it is off, an item not in English (or in a language
nobody could tell) is written with its `language`, its translation, summary and KG are recorded in
its processing manifest as not run (`skipped_reason: translation_off`), and it carries `unusable`.
A teaser or video item gets its KG and no summary (`skipped_reason: not_whole_text`). An item whose
source may not be mined (`mineable: false`) runs no enrichment stage, and each is recorded as not
run (`skipped_reason: not_mineable`).

## Not covered

- The search tables' row shape (database's, E05 "Search").
- The publisher registry's own fields (public configuration, database's).
- Versions of a corrected article (News 2.0).
