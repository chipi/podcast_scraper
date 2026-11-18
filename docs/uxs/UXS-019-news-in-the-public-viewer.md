# UXS-019: News in the public viewer

- **Status**: Draft
- **Authors**: Marko Dragoljevic
- **Parent UXS**: [UXS-001: Workbench](UXS-001-workbench.md): shared tokens, typography,
  layout, states
- **Related RFCs**:
  - [RFC-134: News as the second content type](../rfc/RFC-134-news-as-the-second-content-type.md),
    §9 (search) and §10 (display)
- **Related ADRs**:
  - [ADR-177](../adr/ADR-177-content-types-and-transports-are-seams.md) (content types)
  - [ADR-179](../adr/ADR-179-article-text-extracted-never-served.md) (article text is never served)
  - [ADR-180](../adr/ADR-180-bylines-join-within-their-publisher.md) (bylines and publishers)
  - [ADR-181](../adr/ADR-181-a-content-types-own-fields.md) (a content type's own fields)
- **Related UX specs**:
  - [UXS-016: Query Workspace](UXS-016-query-workspace.md): the filter chip bar and result rows
    this spec extends
  - [UXS-005: Semantic Search Panel](UXS-005-semantic-search.md): the compact launcher
- **Shell IA (Workbench only):**
  - [WORKBENCH_IA.md](WORKBENCH_IA.md): the subject rail gains an `article` subject (below)
- **Implementation paths** (built on `feat/news-release`, 2026-11-18):
  - `web/workbench/src/components/search/chips/SearchTypeChip.vue` (the Type chip)
  - `web/workbench/src/components/search/NewsResultRow.vue` (the news row)
  - `web/workbench/src/components/news/ArticleRailPanel.vue` (the article card)
  - `web/workbench/src/api/newsApi.ts`; server: `GET /api/corpus/news/items`,
    `GET /api/corpus/news/item`, `content_types` on `GET /api/search` and `GET /api/health`

**Where it renders (2026-11-18).** The Type chip shows in the Query Workspace chip bar when the corpus holds news; news hits render as `NewsResultRow`, interleaved with podcast rows in the server's order; the `article` subject kind opens `ArticleRailPanel`; the `news` token is in `theme/tokens.css` and UXS-001. Not yet: the compact launcher's chip row, and every other view (Library, Digest, Dashboard, the graph), which News 1.0's acceptance adds.

## Summary

News articles are the platform's second content type (RFC-134). In News 1.0 the public viewer
shows them in two places only, as RFC-134 §10 scopes it: **search**, where a Type chip lets the
operator ask for news, and an **article card** in the subject rail. A news item is always shown as
its title, summary and basic metadata with a link out to the publisher. The article's text is never
shown: not on a card, not in a search row, not as a snippet.

## Vocabulary

The Workbench moves from podcast words to the platform's own (the operator, 2026-11-17). An
**item** has a **content type**; a **source** lists its items over a **transport**. The platform
starts with two content types, each with its own transports (ADR-177, RFC-134):

| General term | Meaning | Podcasts (`podcast_episode`) | News (`news_article`) | Today's name in the code and older specs |
| --- | --- | --- | --- | --- |
| **Item** | One unit of content | an episode | an article | episode (`CorpusEpisodeListItem`, the episode row and rail) |
| **Content type** | What an item is; shown as **Podcasts** or **News** | `podcast_episode` | `news_article` | `content_type` |
| **Source** | Where items come from, with its own title | a show (its feed) | a publisher's feed or API source | show, feed (`feed_id`; KG's `source_id`) |
| **Transport** | How a source lists its new items | RSS enclosures (`rss_enclosure`) | RSS/Atom feeds; a JSON GET API (Hacker News, the Guardian) | — (`content_types.py`) |
| **Person** | A named human | a host, guest or speaker | a byline author, joined within its publisher (ADR-180) | speaker |
| **Text** | The item's text | the transcript | the article text: extracted, never served (ADR-179) | transcript |

One item reached over two transports is one item (RFC-134: the URL identity). This spec uses the
general terms; where it points at a part of today's Workbench, it gives today's name once in
brackets. Renaming those parts in the code and the older specs is separate work.

## Principles

- **Link out, always.** Every news element carries a visible link to the article on its publisher's
  site. The viewer is a pointer to the article, never a reader of it.
- **Nothing from the body.** No article text, no chunk text, no quoted snippet in News 1.0
  (ADR-179). The summary is the platform's own text; the title and byline are metadata.
- **Podcasts stay podcasts unless asked.** Every surface that shows podcast items today keeps
  showing podcast items only. News appears where the operator asks for it (the Type chip), and in
  the article's own rail.
- **Typed at a glance.** A news element is recognisable without reading it: the `news` domain
  token, a "News" label and the publisher's name, on every row and card.
- **No placeholder UI** ([WORKBENCH_IA.md](WORKBENCH_IA.md)): the Type chip appears only when the server
  says the corpus has news items; a podcast-only corpus looks exactly as it does today.

## Scope

**In scope (News 1.0):**

- The **Type** chip in the Query Workspace filter chip bar.
- The **news result row** in the Workspace results and in the compact launcher's results.
- The **article subject** in the right rail: the article card.
- Every other view (the operator's acceptance, 2026-11-18: "adjust views also, digest and all
  others"), each specified under [Every other view](#every-other-view): an **"In the news"**
  section on the node rail (topics, people, organisations), a **News** mode in the Library with
  publisher pages, a **news band** in the Digest, **news counts** on the Dashboard, and
  **article and publisher nodes in the graph**.

**Later:**

- Quoted evidence snippets: News 2.0, with grounded insights for news.

**Non-goals:**

- Rendering an article, or any part of its body, in any surface.
- The private viewer's news card (RFC-134 §10: it has its own).
- The listener app (RFC-134 §11).

**Boundary note:** This UXS covers the **static visual contract** (tokens, layout, component
appearance, accessibility targets). Behavioral rules (animation timing, debounce intervals,
resize/collapse logic, keyboard shortcuts) belong in the related RFC. See the
[UXS vs RFC boundary](index.md#uxs-vs-rfc-boundary) guidance.

## Data the viewer receives

Built by the read side from the item's metadata (RFC-134 §5 and §10), as the
[news item contract](../architecture/corpus/news-item-1.0.md) names it (accepted by the operator,
2026-11-18; ids, times and rights from [ontology v3](../architecture/corpus/ontology-v3.md); the
publisher from the registry, ADR-180 rule 6). The viewer renders only these fields and never asks
for more:

| Field | Shown as |
| --- | --- |
| `content_type` (`news_article`) | the "News" type label and the `news` token |
| `item_id` (`item:news_article:{20 hex}`) | the row's and the rail's key |
| `title` | the row and card title |
| `url` (the normalised canonical URL) | the "Open article" link out |
| publisher `{id, name, domain}` | the publisher name, with its domain on the card. From the registry: the item's `publisher_id`, else its source's publisher, else the one whose domain the URL's host matches; none of them, no publisher |
| `published_at` | relative on rows ("3 h ago"), absolute on the card |
| `language` | a language badge when it is not English, as on podcast items |
| `paywalled` (`true`, `false`, `null`) | a "Paywalled" note beside the link out when `true` |
| `serveable` (`metadata`, `summary`, `none`; absent reads as `metadata`) | below, under States |
| `summary` | the summary paragraph, only when `serveable` is `summary` |
| `news_article.byline` (a list of names, order kept) | "By …", names as written; a person link only when the byline has joined a person |
| `news_article.editorial_type` (reporting, opinion, analysis, paid, or null) | a badge for opinion, analysis and paid; nothing for reporting or null |
| `news_article.text_status` (`full`, `teaser`, `video`) | for `teaser` and `video`, a "Summary from the feed" note (the article text was not available); nothing for `full` |

Requests: a repeated query parameter `content_types` (`podcast_episode`, `news_article`); the
default is `podcast_episode`, so a request without it is answered exactly as today.

## Theme support

- **Mode:** both (manual toggle), as UXS-001.
- **Primary palette:** dark, as UXS-001.
- **Breakpoints:** desktop only, as UXS-001.

## Semantic color tokens

Every surface, text and intent token is UXS-001's. One domain token is added.

### Domain tokens

| Token | Dark | Light | Usage |
| --- | --- | --- | --- |
| `news` | `#9fb7d0` | `#3d5a7a` | The news type label, row accent and article card accent |
| `news-foreground` | `#111418` | `#FFFFFF` | Text on `news` surfaces |

`news` is a steel blue, distinct from `kg` (purple), `topic` (teal), `person` (peach), `gi`
(green), `link` and `primary` (bright blue). Its values are open (see Tunable parameters).

## Typography

UXS-001's. The news row title uses the podcast item row's title size and weight; the source and time
line uses `muted` at the metadata size.

## Layout and spacing

### The Type chip

A chip in the Query Workspace filter chip bar (UXS-016 §Header), placed after **Doc types**:

- `data-testid="search-chip-type"`, label `Type ▾` when the default (Podcasts) is active and
  `Type: News ▾` / `Type: Podcasts and news ▾` otherwise.
- Its popover is a single-choice list: **Podcasts** (default), **News**, **Podcasts and news**.
- The chip is shown only when `GET /api/health` lists `news_article` in `content_types` (the types
  the served corpus holds, beside `artifact_kinds`). Without news, the chip bar is unchanged.
- The chip's value is sent as `content_types`, and only when it is not the default. Choosing a
  type re-runs the current query, since it replaces the whole result set (the other chips take
  effect on the next search).
- The same chip appears in the compact launcher's filter row when the launcher shows chips.

### The news result row

A sibling of the podcast item row (today's episode-backed result row, UXS-016 §Results), never a
variant of the text (transcript) row:

```text
┌────────────────────────────────────────────────────────────────────┐
│ ▍NEWS  Reuters · By Jane Doe · 3 h ago            [Opinion]  [DE]  │
│ ▍Central banks signal a slower path for rate cuts                 │
│ ▍The summary, two to four lines, clamped; never chunk text …      │
│ ▍                                        Open article ↗  [→ rail]  │
└────────────────────────────────────────────────────────────────────┘
```

- **Accent:** a 3px left border in `news` (the podcast item row has none), the "NEWS" label in
  `news` text.
- **Line 1:** the publisher name, then "By …" when there is a byline, then the published time, in
  `muted`; badges right-aligned: editorial type (opinion, analysis, paid), language (when not
  English), "Paywalled".
- **Line 2:** the title, clickable: opens the article subject in the rail.
- **Lines 3 to 5:** the summary, clamped to three lines with the existing clamp. For a teaser, the
  feed summary with the "Summary from the feed" note in `muted`.
- **Actions:** **Open article ↗** (an external link, `link` token) and the rail opener. The podcast
  item row's **G**, **L** and **E** controls do not appear: an article serves no text (ADR-179), has no Library
  row and no graph node in News 1.0.
- **No relevance text.** Whatever chunk matched, the row shows the summary. No highlighted match,
  no "matched in" excerpt.
- `data-testid="search-result-news"`, with `search-result-news-title`,
  `search-result-news-open` (the link out) and `search-result-news-rail`.
- In **Podcasts and news**, rows are interleaved in the server's merged order; each row carries its
  own type, so nothing is grouped by type.

### The article card (subject rail)

A new subject kind, `article`, in [WORKBENCH_IA.md](WORKBENCH_IA.md)'s subject table, entered from a news
result row. Component: `ArticleRailPanel`. Top to bottom:

1. **Header:** the "News" type label (`news`), the close button (`subject-rail-close`,
   `aria-label="Close article detail"`).
2. **Title** (rail title size).
3. **Publisher:** the name and its web domain in `muted` (`news.example.org`).
4. **Byline:** "By A, B". A name links to the person rail when the byline has joined a person
   (ADR-180); otherwise plain text. An organisation in the byline is not shown here, because it is
   the publisher (ADR-180, rule 5).
5. **Times:** published, absolute, with the time zone; "Fetched …" in `muted`.
6. **Badges:** editorial type, language, "Paywalled", "Summary from the feed" (teaser).
7. **Summary:** the full summary paragraph.
8. **Open article ↗:** a full-width secondary button, the card's main action.
9. **Topics and entities:** chips from the article's KG (`topic` and `kg` tokens, as on the podcast item
   rail), each opening its topic or entity in the rail. `GET /api/corpus/news/item?item_id=…`
   returns them with the item (`kg_nodes`: id, type, label, and `role: author` for a byline
   person), so the card reads no KG file. Without a KG, the section is left out.

The card never has a transcript, a body, a "Read more" or a quote section.

## Every other view

Pulled into News 1.0 by the operator's acceptance (2026-11-18). Each view shows news only when
`GET /api/health` lists `news_article`; without news, every view is exactly as it was. Every news
entry is a served item (`NewsItem`): a `serveable: none` item appears nowhere and no view shows
body text.

### The node rail: "In the news"

On a topic, person or organisation rail (`NodeDetail`), below the podcast items section, a section
**In the news** lists the news items whose KG names the node (`POST /api/corpus/node-episodes`
with `content_types` including `news_article`, field `news_items`), newest first, five shown with
"Show all (N)". Each entry: the title (opens the article card), the publisher and the relative time
in `muted`, a 3px `news` left border. An author is marked "By" where the node is a person who wrote
the item. `data-testid="node-news-section"`, entries `node-news-item`. With no news items the
section is left out.

On a theme (`tc:`) or storyline (`thc:`) rail the same section lists the grouping's articles: the
union of its members' `news_item_ids` in Common's grouping files (`search/topic_clusters.json`,
`enrichments/topic_theme_clusters.json`), served items only, newest first. The episode list and
its count are unchanged; articles are counted on their own (the operator, 2026-11-18: counts stay
separate). The Player's and Studio's theme and storyline cards carry the same list when asked
with `news=1` (`AppClusterCardWithNews`: `articles`, `article_count`).

### The Library: News mode

A segmented control at the top of the Library: **Podcasts** (default) | **News**. News mode lists
the news items newest first (`GET /api/corpus/news/items`, paged 50), each as the news result row
without a score. A publisher filter (a select of the publishers present) narrows it to one
publisher (`publisher_id`); choosing a publisher's name in a row does the same. This is the
publisher page: the publisher's name and domain head the list. `data-testid="library-mode-news"`,
`library-news-list`, `library-news-publisher`.

### The Digest: a news band

The Digest's window (its date range) gains a **News** band after the podcast bands: the news items
published in the window, newest first, at most ten, each a compact news row (title, publisher,
time; the summary on hover is not shown). "See all in Library" opens the Library in News mode.
`data-testid="digest-news-band"`. With no news in the window the band is left out. The topic bands
stay podcast-only in News 1.0.

### The Dashboard: news counts

The corpus briefing card gains one line when the corpus holds news: "News: N items from P
publishers (T teasers, W paywalled)", from the items route. `data-testid="dashboard-news-counts"`.

### The graph: articles and publishers

The graph loads a node's news items when it expands a node (the node rail's list), as article nodes
(`news` token, a square shape) joined by their `MENTIONS` and author edges, with their publisher as
an organisation node joined by `PUBLISHED_BY` through the source. News nodes are never loaded into
the first graph slice: a podcast graph opens exactly as before. Opening an article node opens the
article card.

## Key states

- **Loading:** the existing rail and row skeletons.
- **No summary yet** (the item is ingested, enrichment has not run): title, metadata and link out;
  the summary line reads "No summary yet" in `muted`.
- **Teaser or video** (`text_status: teaser` or `video`): the feed's summary, with "Summary from the feed" in `muted`.
- **Metadata only** (`serveable: metadata`, or absent): title, publisher, time and link out, no
  summary; a `muted` line "This publisher's items show their title and link only."
- **Not shown** (`serveable: none`): the item never appears, in search results, the rail or any
  count; it is kept for analysis only (ontology v3, rights).
- **No byline:** the "By …" part is left out; nothing stands in for it.
- **No results in News:** "No news items for `<q>`", with the chip bar kept so the operator can
  switch Type.
- **Error:** the existing Workspace error banner.

## Components

- **The external link** (`Open article ↗`): `link` token, an outward-arrow icon, `target="_blank"`
  with `rel="noopener noreferrer"`, and an accessible name that says it leaves the app
  ("Open article on reuters.com, opens in a new tab").
- **The type label:** small caps "NEWS" in `news`; the same component later labels other content
  types.

## Charts and graph

No graph change in News 1.0. Article and publisher nodes are later (see Scope).

## Accessibility

- The news row is a list item whose accessible name is its title; the type and publisher are read
  before the title (`aria-describedby` on the metadata line).
- The link out is a real link, not a button; its name states the destination domain and that it
  opens a new tab.
- Contrast at the proposed values (WCAG relative luminance, computed 2026-11-17): `news` on
  `surface` 7.84 (dark) and 7.14 (light), on `canvas` 8.93 and 6.66; `news-foreground` on `news`
  8.93 and 7.14. All above AA's 4.5. Recompute if the values change.
- The Type chip's popover is a radio group (`role="radiogroup"`), keyboard-operable like the other
  chips.

## Tunable parameters

| Parameter | Current value | Status | Notes |
| --- | --- | --- | --- |
| `news` token values | `#9fb7d0` / `#3d5a7a` | Open | Pick against the shipped palette in both themes |
| Token names `news`, `news-foreground` | | Frozen once built | Used by tests and later types' labels |
| Summary clamp on the row | 3 lines | Open | Same clamp as the podcast item summary row |
| Type chip default | Podcasts | Frozen | RFC-134: podcast surfaces see news only when asked |

## Acceptance criteria (for issues / review)

- [ ] A podcast-only corpus renders every surface exactly as before (no Type chip, no news token
      anywhere).
- [ ] With news items, the Type chip appears; Podcasts returns no news rows; News returns only news
      rows; Podcasts and news interleaves both.
- [ ] A news row and the article card show title, publisher, byline, time, summary and the link out,
      and nothing from the article's text: a viewer e2e over a news fixture whose body holds a
      unique marker fails if the marker appears anywhere in the rendered DOM.
- [ ] No news row shows G, L or E, or any matched-text excerpt.
- [ ] The teaser, metadata-only, no-summary and no-byline states render as specified, and a
  `serveable: none` item appears nowhere.
- [ ] Every link out has `rel="noopener noreferrer"`, opens in a new tab and names its domain.
- [ ] `news` and `news-foreground` are defined once in the token layer; no hex in components.
- [ ] `E2E_SURFACE_MAP.md`, [WORKBENCH_IA.md](WORKBENCH_IA.md) (the `article` subject) and UXS-016 (the
      Type chip) are updated with the build.

## Open questions

- **The item detail route's name:** the read side names it when it is written.
- **Later items** (Library News mode, graph nodes, "In the news", the Digest band): the operator
  decides whether any of them moves into News 1.0.

## Revision history

| Date | Change |
| --- | --- |
| 2026-11-17 | Initial draft: News 1.0 scope (Type chip, news result row, article card), later items listed |
| 2026-11-17 | Vocabulary: item, content type, source, transport, person, text (the operator); two content types (Podcasts, News), each with its own transports; today's names given once in brackets |
| 2026-11-18 | Data table matched to ontology v3 (ingestion's correction): `url`, not `canonical_url`; `serveable` is `metadata` / `summary` / `none`, not a boolean; the publisher comes from the source through the registry; each field marked decided or undecided |
| 2026-11-18 | Every other view pulled in (the operator's acceptance): node rail "In the news", Library News mode and publisher pages, Digest news band, Dashboard counts, graph article and publisher nodes |
| 2026-11-18 | Built: the Type chip re-runs the current query (the other chips do not); the card's topics and people come from `GET /api/corpus/news/item` |
| 2026-11-18 | The item contract accepted (the operator): every field named as `news-item-1.0.md` has it; `text_status` adds `video`; the publisher order (item, source, domain) |
