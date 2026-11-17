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
- **Implementation paths** (proposed; nothing is built yet):
  - `web/workbench/src/components/search/SearchFilterBar.vue` (the Type chip)
  - `web/workbench/src/components/search/ResultCard.vue` (the news row)
  - `web/workbench/src/components/subject/` (the article rail panel)
  - `web/workbench/src/stores/subject.ts` (the `article` subject kind)
  - `web/workbench/src/theme/` (the `news` domain token)

**Where it renders (2026-11-17).** Nothing in this spec is drawn by the public Workbench today. In `web/workbench/src` there is no `article` subject kind (`stores/subject.ts:8`), no `news` token (`theme/tokens.css`, `theme/theme.ts`), no Type chip (the chip is **Doc types**, `search-chip-doctypes`) and no news row. The places a build could fill: `searchDocTypes` (an option on the Doc types chip and the row icon), `resultCardBadges` and `resultCardSections`. Whether a news build uses them: not checked. The spec stays a draft for the public Workbench.

## Summary

News articles are the platform's second content type (RFC-134). In News 1.0 the public viewer
shows them in two places only, as RFC-134 §10 scopes it: **search**, where a Type chip lets the
operator ask for news, and an **article card** in the subject rail. A news item is always shown as
its title, summary and basic metadata with a link out to the publisher. The article's text is never
shown: not on a card, not in a search row, not as a snippet.

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

**Later (not in News 1.0, RFC-134 §10; each needs the operator's go to pull in):**

- A **News** browse mode in the Library tab, and sources shown as publishers.
- **Article and publisher nodes in the graph** (`HAS_ITEM`, `PUBLISHED_BY`, author edges).
- An **"In the news"** section on the topic and person rails.
- A **news band** in the Digest.
- News counts per source on the Dashboard (items, teasers, paywalled).
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

Proposed by the read side (RFC-134 §5 and §10; not built yet). The viewer renders only these
fields and never asks for more:

| Field | Shown as |
| --- | --- |
| `content_type` (`news_article`) | the "News" type label and the `news` token |
| `title` | the row and card title |
| `summary` | the summary paragraph |
| `canonical_url` | the "Open article" link out |
| `publisher` (`{id, name, domain}`) | the publisher name, with its domain on the card |
| byline as given | "By …", names as written; a person link only when the byline has joined a person |
| published time | relative on rows ("3 h ago"), absolute on the card |
| `language` | a language badge when it is not English, as on episodes |
| `editorial_type` (reporting, opinion, analysis, paid, or null) | a badge for opinion, analysis and paid; nothing for reporting or null |
| `paywalled` | a "Paywalled" note beside the link out |
| `serveable` | when false, the item shows title, publisher and link only (no summary) |
| `text_status` (`teaser`) | "Summary from the feed" note (the article text was not available) |

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

UXS-001's. The news row title uses the episode row title's size and weight; the publisher and time
line uses `muted` at the metadata size.

## Layout and spacing

### The Type chip

A chip in the Query Workspace filter chip bar (UXS-016 §Header), placed after **Doc types**:

- `data-testid="search-chip-type"`, label `Type ▾` when the default (Podcasts) is active and
  `Type: News ▾` / `Type: Podcasts and news ▾` otherwise.
- Its popover is a single-choice list: **Podcasts** (default), **News**, **Podcasts and news**.
- The chip is shown only when `GET /api/health` lists `news_article` in `content_types` (the types
  the served corpus holds, beside `artifact_kinds`; proposed). Without news, the chip bar is
  unchanged.
- The chip's value is sent as `content_types`. Changing it re-runs the query, as the other chips do.
- The same chip appears in the compact launcher's filter row when the launcher shows chips.

### The news result row

A sibling of the episode-backed result row (UXS-016 §Results), never a variant of the transcript
row:

```text
┌────────────────────────────────────────────────────────────────────┐
│ ▍NEWS  Reuters · By Jane Doe · 3 h ago            [Opinion]  [DE]  │
│ ▍Central banks signal a slower path for rate cuts                 │
│ ▍The summary, two to four lines, clamped; never chunk text …      │
│ ▍                                        Open article ↗  [→ rail]  │
└────────────────────────────────────────────────────────────────────┘
```

- **Accent:** a 3px left border in `news` (the episode row has none), the "NEWS" label in
  `news` text.
- **Line 1:** the publisher name, then "By …" when there is a byline, then the published time, in
  `muted`; badges right-aligned: editorial type (opinion, analysis, paid), language (when not
  English), "Paywalled".
- **Line 2:** the title, clickable: opens the article subject in the rail.
- **Lines 3 to 5:** the summary, clamped to three lines with the existing clamp. For a teaser, the
  feed summary with the "Summary from the feed" note in `muted`.
- **Actions:** **Open article ↗** (an external link, `link` token) and the rail opener. The episode
  row's **G**, **L** and **E** controls do not appear: an article has no transcript, no Library
  episode and no graph node in News 1.0.
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
9. **Topics and entities:** chips from the article's KG (`topic` and `kg` tokens, as on the episode
   rail), each opening its topic or entity in the rail. Loaded the way the episode card loads its
   KG: an item detail read keyed by `item_id` returns `has_kg` and `kg_relative_path` (proposed;
   route name to come), then the artifacts route serves the KG file. Without a KG, the section is
   left out.

The card never has a transcript, a body, a "Read more" or a quote section.

## Key states

- **Loading:** the existing rail and row skeletons.
- **No summary yet** (the item is ingested, enrichment has not run): title, metadata and link out;
  the summary line reads "No summary yet" in `muted`.
- **Teaser** (`text_status: teaser`): the feed's summary, with "Summary from the feed" in `muted`.
- **Not serveable** (`serveable: false`): title, publisher, time and link out only; a `muted` line
  "This publisher's items show their title and link only."
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
| Summary clamp on the row | 3 lines | Open | Same clamp as the episode summary row |
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
- [ ] The teaser, not-serveable, no-summary and no-byline states render as specified.
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
