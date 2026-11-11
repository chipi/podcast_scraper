# Unified Corpus Ontology v3 (proposed design)

**Status:** Proposed, 2026-11-11. The design for R3a, the schema half of ontology v3, built in the
core release ([ADR-178](../../adr/ADR-178-ontology-v3-in-two-halves.md)). It extends
[ontology v2](ontology.md), which stays the source of truth until KG 3.0 ships. The name follows
decision T-7: ontology v3, with the artifact schemas at **KG 3.0** (here, public) and **GI 4.0**
(the private extension's, which follows the evidence contract below).

**What v3 is for:** items that are not podcast episodes (news first, [RFC-134](../../rfc/RFC-134-news-as-the-second-content-type.md)), and a corpus database designed once against one shape (E06).

## What changes from v2

| Area | v2 (KG 2.x, GI 3.x) | v3 (KG 3.0, GI 4.0) |
| --- | --- | --- |
| Anchor nodes | `Podcast`, `Episode` | `Source`, `ContentItem` (the podcast module still says show and episode) |
| Root id field | `episode_id` | `item_id`, plus `content_type` |
| Source → item edge | `HAS_EPISODE` | `HAS_ITEM` |
| Person → item | `SPOKE_IN` (GI) | `PARTICIPATES_IN` with a `role` |
| Person → source | `HOSTS`, `GUESTS_ON` | `CONTRIBUTES_TO` with a `role` |
| Source → publisher | a `publisher` string on `Podcast` | `PUBLISHED_BY` → `Organization`, from the publisher registry |
| Roles | closed enum (`host`, `guest`, `mentioned`) | an open string (known values below) |
| Organisation identifiers | none | `identifiers`: a list of `{scheme, value}` |
| Item times | `publish_date` | `published_at`, `filed_at`, `period_start`, `period_end`, `observed_at`, `ingested_at`, each optional |
| Text the extraction read | `extraction.transcript_ref` | `extraction.document_ref`, plus `variant` |
| Evidence | char range, time range required, `transcript_ref`, `episode_id` | the evidence contract below: char range always, time only when timed |
| Attribution | `SPOKEN_BY` | `ATTRIBUTED_TO`, with `SPOKEN_BY` as its audio form (T-5) |

Unchanged:

- `Person`, `Organization`, `Object` and `Topic`;
- `MENTIONS`, with `RELATED_TO` reserved;
- `ABOUT`, `MENTIONS_PERSON`, `MENTIONS_ORG`;
- `HAS_INSIGHT`, `SUPPORTED_BY`;
- the grounding contract and the two-tier edge contract.

Id values never change (T-8): podcasts keep `podcast:` and `episode:`; new types use `source:` and
`item:`.

## Nodes (KG 3.0)

| Node | Id | Properties |
| --- | --- | --- |
| `Source` | `podcast:{slug}` (podcast), `source:{slug}` (others) | `title` (required); `source_kind` (`podcast`, `news`, …); `transport`; `url` |
| `ContentItem` | `episode:{id}` (podcast), `item:{content_type}:{id}` (others) | `title` (required); `content_type` (required); the item times; `url`; `duration_ms` and `audio_url` for timed items |
| `Person` | `person:{slug}` | as v2; `role` is an open string |
| `Organization` | `org:{slug}` | as v2, plus `identifiers` (`[{scheme, value}]`: `domain`, `cik`, `ticker`, `lei`, `wikidata`); `role` open |
| `Object`, `Topic` | as v2 | as v2 |

`Place` is declared as a kind name (T-4's companion, TERMINOLOGY §4) and not emitted in v3.

## Edges (KG 3.0)

| Edge | From → To | `edge_class` | Properties |
| --- | --- | --- | --- |
| `HAS_ITEM` | `Source → ContentItem` | structural | — |
| `PARTICIPATES_IN` | `Person → ContentItem` | attribution | `role` (`host`, `guest`, `speaker`, `author`, …) |
| `CONTRIBUTES_TO` | `Person → Source` | structural | `role` (`host`, `guest`, `author`, …) |
| `PUBLISHED_BY` | `Source → Organization` | structural | — |
| `MENTIONS` | `Topic`/`Person`/`Organization`/`Object` → `ContentItem` | discovery | as v2 |
| `RELATED_TO` | `Topic ↔ Topic` | discovery | reserved |

**A byline** is a `Person` with `PARTICIPATES_IN` to its item, `role: author`. `CONTRIBUTES_TO` its
source with `role: author` is written only once the identity joining (ADR-180) has joined the
byline within that publisher.

GI 4.0's edges (`HAS_INSIGHT`, `SUPPORTED_BY`, `ABOUT`, `MENTIONS_PERSON`, `MENTIONS_ORG`,
`ATTRIBUTED_TO` with `SPOKEN_BY` as its audio form) keep v2's classes and point at `ContentItem`
where v2 pointed at `Episode`.

## Roles

`role` is a non-empty string. The known values:

- **people:** `host`, `guest`, `speaker`, `author`, `mentioned`;
- **organisations:** `mentioned`, `publisher`.

A later type adds its own values (an earnings call's `executive` and `analyst`) with no schema
change. Readers treat an unknown role as a participant without special meaning.

## The evidence contract (shared by KG 3.0 and GI 4.0)

Every evidence span (GI 4.0's quotes; any later evidence) carries:

| Field | Required | Meaning |
| --- | --- | --- |
| `item_id` | yes | the content item |
| `document_ref` | yes | the item's body document (the transcript, an article body; later a filing's main document or an exhibit) |
| `variant` | yes | `raw`, `cleaned`, `adfree`, or a translation's language tag |
| `char_start`, `char_end` | yes | the span in that document variant |
| `start_ms`, `end_ms` | only for timed documents | both or neither; when present `end_ms > start_ms` (a start of 0 is real); an untimed document carries neither |
| `page`, `section` | no | for paged or sectioned documents |
| `text_origin` | yes | `spoken`, `translated` or `authored` (T-11) |

**Segments:**

- Every document is stored with segments, so evidence also maps to a segment range (DECISIONS
  rule 3).
- Timed documents have timed segments.
- Untimed ones (articles) have paragraph segments with no times.
- The corpus database stores the segment range and computes character offsets. File-mode
  artifacts carry the character range.
- **File mode writes no segments file for an untimed document.** A paragraph is the text between
  blank lines in the body. The corpus database's importer derives the segments by that rule, so a
  character range maps to a segment range deterministically.

## Ids, times and documents for a new type (agreed with ingestion, 2026-11-11)

- **The type mints the item id, not the transport.**
  - A transport returns its native id, or the canonical URL.
  - The content type mints `item:{content_type}:{id}`. For an item identified by URL, `{id}` is
    the first 20 hex characters of the SHA-256 of the normalised canonical URL; the URL itself is
    kept in `url`.
- **The type mints the source id too:** `source:{id}`, where `{id}` is the first 20 hex characters of
  the SHA-256 of the source's normalised URL. It is stable across renames; the source's title is
  for display. Podcasts keep `podcast:{slug}`.
- **`source_kind` is a fixed vocabulary,** separate from `content_type` (T-1 keeps the item noun
  apart from the source's kind): `podcast`, `news`, `social`, `advisory`, `report`, and `test` for
  the core release's test-only type.
- **Item times** come from the transport in the item's metadata under the names above
  (`published_at`, `filed_at`, `period_start`, `period_end`, `observed_at`). `ingested_at` is set by
  the run.
- **The body document:**
  - The body of a text item sits where a transcript sits (`transcripts/`, one file per item), so
    the existing resolver reads it.
  - Its `document_ref` is that path, with `variant: raw`.
  - Its evidence has `text_origin: authored`.

## Reading v2 artifacts as v3

Readers accept KG 2.x and GI 3.x until E06, through one adapter, so no surface has two code
paths:

| v2 | Read as |
| --- | --- |
| root `episode_id` | `item_id`, with `content_type: podcast_episode` |
| `Podcast`, `Episode` | `Source` (`source_kind: podcast`), `ContentItem` |
| `publish_date` | `published_at` |
| `HAS_EPISODE`, `HOSTS`, `GUESTS_ON` | `HAS_ITEM`, `CONTRIBUTES_TO` with that role |
| GI `SPOKE_IN` | `PARTICIPATES_IN` with `role: speaker` |
| `extraction.transcript_ref` | `document_ref` (the transcript), `variant` from the file name |
| a quote's times and `transcript_ref` | the evidence contract, `text_origin: spoken` |
| the `publisher` string | kept; `PUBLISHED_BY` only once the registry maps it |

R3b (ADR-178), its own release, rewrites the podcast
corpus to v3. Until then, podcasts keep writing KG 2.x and GI 3.x.

## Validation

- `kg.schema.json` gains a 3.0 shape, selected by `schema_version`. The 2.x shapes stay valid
  until E06.
- The change touches the schemas, so its PR carries a corpus migration or the guard's "migration
  not required" reason: none is needed, since nothing is rewritten in R3a.
- Tests:
  - minimal valid and invalid v3 artifacts for a podcast item and a test-type item;
  - a mixed fixture corpus (KG 2.x beside 3.0) read by every route that reads the graph;
  - untimed evidence with a time range is invalid.

## Open

- **Node type names:** `ContentItem` follows TERMINOLOGY's "Content item"; `Item` is shorter. To
  confirm with ingestion before the schema is written.
- **The podcast id convergence parked in v2** ("Known limitations": KG derives `podcast:` from the
  feed id, GI from the title) is not solved here. R3b is the moment to converge it.
- **GI 4.0's own fields** beyond the evidence contract are the private extension's design.
