# News 1.0 in public: plan and status (2026-11-18)

Working plan for public's half of News 1.0 on `feat/news-release`. Specs: RFC-134 (§5, §9, §10,
§12), Common's `E05-news.md`, [UXS-019](../uxs/UXS-019-news-in-the-public-viewer.md), ADR-177,
ADR-179, ADR-180 (rule 6 amended 2026-11-18), ADR-181, ontology v3, and ingestion's item contract
`docs/architecture/corpus/news-item-1.0.md` (proposed, waiting for the operator).

## Acceptance (the operator, 2026-11-18)

- RFC-134 §12's 19 news sources show up in the Workbench.
- They are configured in config as sources, not hard-coded to podcast feeds.
- Every view is adjusted: Digest and all the others.
- Proven end to end.

## Who does what

| Part | Owner |
| --- | --- |
| The `news_article` type, transports (RSS/Atom, HN, Guardian), extraction, the writer, the item contract | ingestion |
| The publisher registry, news reads in every view, search over the news tables, the never-served guard, the Workbench, the e2e suite | public |
| Studio's news card and views, over public's routes; podcast `PUBLISHED_BY` (R3b) | database |

## What Studio needs from public's routes (database, 2026-11-18)

- A node's items across both types (`node-episodes` or its successor): news rows from KG 3.0 news
  nodes (`MENTIONS`, `PARTICIPATES_IN` role `author`), `content_type` on each row.
- A publisher as an organisation: the items whose `Source` is `PUBLISHED_BY` that `org:` id.
- The item read keeps `item_id`, `content_type`, `source_id` and the item's KG node ids.
- Studio leaves news out of the insight, perspective and position views (no GI in News 1.0).

## Steps

| # | Step | Status |
| --- | --- | --- |
| 1 | Publisher registry and loader (`config/publishers.yaml`, `podcast_scraper.publishers`), the 19 publishers | done (547d73898, 4b3edf957) |
| 2 | News reader: served item from metadata only, rights, publisher order (`podcast_scraper.news_items`) | done (00ecb4be6) |
| 3 | Invented news fixture in the contract's shape, beside the podcast validation corpus | next |
| 4 | Health `content_types`; the item read; the publisher-as-organisation read | |
| 5 | Search: news tables in the index, `content_types` on the search route, summary and metadata only | |
| 6 | Node items across types (Studio's need 1) | |
| 7 | Views: Library, Digest, Dashboard, the subject rail, the graph over KG 3.0 news nodes | |
| 8 | Workbench: Type chip, news row, article card; source config with `content_type` and `transport` | |
| 9 | Never-served guard: every route and MCP tool over a fixture whose body holds a marker | |
| 10 | e2e: Workbench Playwright on the fixture; then one run over ingestion's real output | |

## Not covered yet

- The real sources' items: they need ingestion's writer, which has no date yet.
- The contract's proposed names (summary, byline, transport ids, editorial type, text status,
  body hash and fetch time): read from one module, changed there when the operator decides.
- How each view shows news beyond search and the card (Digest, Dashboard, Library, graph):
  UXS-019 scopes only search and the card; the operator's acceptance widens it to every view, so
  UXS-019 gets a section per view before each is built.
