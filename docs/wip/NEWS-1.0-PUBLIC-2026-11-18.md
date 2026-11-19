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
| 3 | Invented news fixture in the contract's shape, beside the podcast validation corpus | done (57d69816d, ids fixed 2435aa1f9) |
| 4 | Health `content_types`; the item read; the publisher-as-organisation read | done (68a57268d) |
| 5 | Search: news table in the index, `content_types` on the search route, summary and metadata only | done (1ecf17ccd) |
| 6 | Node items across types (Studio's need 1) | done (f6a5d3dc0) |
| 7 | Views: Library, Digest, Dashboard, the subject rail, the graph over KG 3.0 news nodes | done (96497f907, efe21786f, 417aa7c1b, 8f4e4d247; fixes 8ebad1afb, ab7f0d32b, c37e2b408); UXS-019 "Every other view" |
| 8 | Workbench: Type chip, news row, article card; source config with `content_type` and `transport` | done (bfe12aefb, 5f12adab4) |
| 9 | Never-served guard: every route and MCP tool over a fixture whose body holds a marker | done: every public read and state-changing route (8c0be4450, f13507ada; run with Common and the Player installed by the database session, 2 passed); two leaks fixed on the way (text-file 147c3a1aa, artifacts 8c0be4450, 59e372031); the MCP tools: Common 1f7f12c, 4557f20c6 |
| 10 | e2e: Workbench Playwright on the fixture; then one run over ingestion's real output | done: fixture (news-views, news-search, the graph expand 3a662b449); real output (all8, 257 items): rights checked through the routes, Fed served nowhere, SCMP without summary, every shown item found by its title (ef1fe7803, 49acd4a08) |
| R3b | GI 4.0 and KG 3.0 read views at every public load point | done (f2c19647d, 3b64360d6) |

News in themes and storylines (E05, 2026-11-18): B `news_aux` topic rows (5ff83e03f); F cards carry
articles when asked (5c7588512); H the theme and storyline rails (2c105027d); I storyline prompts
with news (d1ea2988d, wording 253d831c0); J the fixture's grouping files (b50e6079e, news-only
members 95998971c); K RFC-134 §11 (d6e8cca0d). Later the same day: `site` for an item with no
publisher (973421883), an unusable item shown nowhere (49acd4a08), `roundup` (7eb6c2730), keyed API
sources (`api_key_env`, ingestion 543ed07f9; guide 5c255c178).

## Not covered yet

- Items written before ingestion's writer fixes carry no `unusable` or `roundup` marker: a corpus
  written earlier (all8 included) serves its untranslated items until rewritten. No prod corpus
  holds news yet.
- `nightly.yml`'s ml-models tar change (0b2eb6d58, 4b00455fa) has not run yet; `python-app.yml`'s
  has (4b00455fa, VERDICT GREEN).
- The Guardian source is off until `PROD_GUARDIAN_API_KEY` is set (the operator's choice).
- The Snyk PR check on this repo stays red until the Snyk org setting is changed (the operator's).
