# UXS-002: Corpus Digest

- **Status**: Active
- **Authors**: Podcast Scraper Team
- **Parent UXS**: [UXS-001: GI/KG Viewer](UXS-001-gi-kg-viewer.md) -- shared tokens,
  typography, layout, states

- **Related PRDs**:
  - [PRD-023: Corpus Digest & Library Glance](../prd/PRD-023-corpus-digest-recap.md)
- **Related RFCs**:
  - [RFC-068: Corpus Digest API & Viewer](../rfc/RFC-068-corpus-digest-api-viewer.md)
  - [RFC-062: GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md)
- **Implementation paths**:
  - `web/gi-kg-viewer/src/components/digest/DigestView.vue`
  - `web/gi-kg-viewer/src/utils/digestRowDisplay.ts`
- **Shell IA:** [VIEWER_IA.md](VIEWER_IA.md) — canonical shell layout, navigation axes, subject rail, status bar, first-run behavior

---

## Summary

For shell layout, the three navigation axes, subject rail persistence and clearing, status bar, and first-run empty corpus behavior, see **[VIEWER_IA.md](VIEWER_IA.md)**. This document specifies the **Digest** tab only (rolling window, semantic topic bands, layout and density, and **`corpusLens`** shared with Library).

The **Digest** tab is the discovery surface for rolling-window recent episodes and
semantic topic bands. It is the default selected tab on first load. This UXS defines
the visual contract for the Digest tab layout, density, and component appearance.
All tokens reference [UXS-001](UXS-001-gi-kg-viewer.md).

---

## Information architecture

- **Main tabs:** Order and default-on-load are defined in [VIEWER_IA.md — Main tabs](VIEWER_IA.md#main-tabs) (**Digest** is default).

- **Digest tab:** A main nav item labeled **Digest** (same control pattern as other
  main views). The panel also shows an in-column **`h2` Digest** with a **`?`**

  **About Digest** **`HelpTip`** (same pattern as **`h2` Recent** + help on this tab
  and **`h2` Episodes** + help on **Library**).
  **Publish-date lens** is **shared with Library** (Pinia `corpusLens`): same
  **Published on or after** field and the same preset row (**All time**, **7d**,
  **30d**, **90d**) using local calendar days. Empty field means **all time**; the
  digest request uses `window=all`. A set date uses `window=since` with that
  **YYYY-MM-DD**. Changing the lens on either tab updates the other.

- **Toolbar:** First row -- **`h2` Digest** (left) with **`?`** help; **Published on or after**
  plus presets on the **right**. When digest data is loaded, a second muted row shows rolling

  bounds (start -> end, row count). Use the **Library** main tab to open the catalog;
  corpus path is unchanged.

---

## Layout and density

- **Digest tab:** Single-column scroll on `canvas`; sections:

1. **Toolbar** -- stacked rows inside a **`flex`** column (`gap-1.5`): **(a)** first row
   (`wrap`, `justify-between`) -- **`h2` Digest** (`#digest-main-heading`, **`text-sm`**

   semibold) on the **left** with a **`HelpTip`** (**About Digest**) that explains **Topic
   bands** (topic title -> **Graph**; hit row -> **Episode subject rail** + **Graph**; digest topic
   focus; **Search topic**; hit rows show **semantic match strength** when a score exists and
   hover **`title`** on hit rows lists publish date, **E#**, duration, and feed hints), **Recent** (Episode
   rail, CIL **topic pills** vs **Library** catalog); **Published
   on or after** (`#digest-filter-since`, shared **`corpusLens`** with Library) plus the same
   preset buttons (**All time**, **7d**, **30d**, **90d**) on the **right** (`muted` label,
   `surface` controls). **(b)** when digest data is loaded, a second muted **`text-[10px]`**
   row shows rolling window bounds (start -> end, episode count) with `<time datetime="...">`
   for machine-readable instants (human-readable display includes explicit UTC). Active
   preset uses a **primary** ring on the matching button.

2. **Topic bands** -- optional **`topics_unavailable_reason`** line when the server explains
   missing bands.

   When bands exist: outer **`role="region"`** **`aria-label`** **Topic bands** wraps the
   topic **grid** without an outer **`max-height`** / nested scroll: show the **first three**
   band cards by default; when the API returns more, a **`digest-topic-bands-show-more`**
   control expands the rest inline. Inner responsive grid (`sm:2` / `xl:3` columns); each topic
   is a compact bordered **`section`** card. The **first** band uses **`bg-elevated`** and a
   **`border-primary/20`** border; its topic title **`button`** uses **`font-bold`**. Later bands
   use **`bg-surface`**, **`border-border`**, and **`text-sm font-semibold`**. The **topic title** **`button`** opens **Graph** for the top hit that has GI or KG on disk;
   **Search topic** is a separate primary **`button`**. Per-topic **hit rows** are one
   clickable control that also emits **open-library-episode** (same handoff as **Recent**)
   so the **Episode** rail loads that row, then opens the **Graph** tab for that hit's merged
   GI/KG slice and requests focus on the digest **topic** node (`graph_topic_id` from
   `GET /api/corpus/digest`, a `topic:{slug}` derived from the band label) when that node
   exists in the graph, otherwise falls back to the **episode** node. Layout: **grid** first
   column is **`PodcastCover`** only (**`w-9`**, **`h-9`**, same tile size as **Recent**);
   second column stacks **title row** (episode title + optional **Strong** / **Good** / **Weak match**
   label from the vector score — raw score on the label’s native **`title`**) and
   **`summary_preview`** below (**`line-clamp-2`** when the row is not selected, full text when
   selected, matching **Recent**). Publish date, **E#**, duration, human **Feed:** label,
   optional short **About this feed** blurb, and **RSS** URL stay in the row’s native **`title`**
   (readable lines; **no** metadata paths or raw feed ids there). A small **`success`** **recency
   dot** may appear before the title when **`publish_date`** is within the rolling **24h** window
   (local **YYYY-MM-DD** midnight rule; see UXS-001 tunables). **Search topic** prefills semantic search with the topic
   query and passes **Since (date)** only when the shared corpus lens has a valid
   **YYYY-MM-DD** (omitted for all time). Selected hit row uses `bg-overlay` when its path
   matches the Episode subject rail (for example after opening that episode from **Recent**).

3. **Recent (diverse)** -- bordered panel (same list chrome pattern as Library **Episodes**):
   `h2` **Recent** (`#digest-recent-heading`, **`text-sm font-semibold`**) with a muted

   tabular **`(N)`** count next to the title (same **N** as the rolling-window toolbar line)

   - `?` tooltip explaining diversification vs topic bands; `role="region"`
   `aria-label` **`Recent episodes, N items`** (or **`1 item`**);

   episode rows use the **same list-row treatment as Library** (`hover:bg-overlay`, selected
   `bg-overlay`): cover `h-9`, title + right column with one **baseline-wrapping** meta line
   (**feed** when shown, then publish date / **E#** / duration in reading order), tight gap
   before **`summary_preview` / recap** (**`line-clamp-2`** when unselected, full when selected,
   same as Library); optional **recency dot** as on topic hits. Row click opens
   Episode subject rail (Digest remains the main tab).    **Topic pills:** when `GET /api/corpus/digest`
   sends **`cil_digest_topics`** (CIL bridge `topic:` ids, cluster-first order; when
   **`in_topic_cluster`**, pills use a faint **`kg`** tint (**`bg-kg/15`**, **`border-kg`**) so
   cluster membership matches **TopicCluster** chrome on the graph; non-cluster pills stay
   neutral border),
   those chips open **Graph** with **`topic_id`** focus
   (`digest-recent-cil-pills`). There are **no** summary-bullet pill chips on Recent rows.
   Accessible name matches Library rows: episode title, feed.

---

## Actions and tokens

- **Recent episode rows:** Episode subject rail handoff (row click); Digest tab stays
  selected. **Open in graph** / **Prefill semantic search** follow

  [UXS-003](UXS-003-corpus-library.md) on the Episode subject rail, not on the Digest card.

- **Recent CIL topic pills (when present):** Navigate to **Graph** (topic focus with episode fallback
  as above); not a Library filter.

- **Open GI / Open KG / Prefill semantic search (Episode subject rail):** Same domain and
  primary rules as [UXS-003](UXS-003-corpus-library.md) (GI/KG use `gi`/`kg`;

  search handoff uses `primary`).

- **Topic band:** **Search topic** uses `primary`; opens the Search panel with the
  topic query prefilled per RFC-068. **Topic title** opens **Graph** for the top GI/KG hit.

  Each **hit row** opens the **Episode subject rail** for that episode, then **Graph** for that hit
  with digest topic focus when the node exists.

---

## Digest and Episode subject rail interaction

**Recent** row click opens the Episode subject rail on the right without switching away from
Digest (same detail as selecting an episode in Library), so **Open in graph** and
**Prefill semantic search** stay one click away on that rail. **Topic band** hit rows
load the **Episode subject rail** then switch to **Graph**; **Recent** CIL pills (when present) switch
to **Graph** only. Switching Digest to Library
keeps the Episode subject rail selection when that episode is still in scope. Library no
longer embeds a New (24h) digest strip so the two tabs do not compete -- users open
Digest for "what's new," Library for catalog browse.

---

## Health discovery

- `GET /api/health` returns `corpus_digest_api` (RFC-068) and `corpus_library_api`
  (RFC-067) on current server builds.

- Digest tab is enabled when `corpus_digest_api` is true, or when it is omitted but
  `corpus_library_api` is true (viewer infers digest for health JSON from builds

  before the digest flag existed). If `corpus_digest_api` is explicitly false, the
  tab shows an upgrade message.

- If the API process is too old to mount `GET /api/corpus/digest`, the Digest tab
  fetch fails and shows a load error.

---

## Accessibility

- `aria-label` on main nav matches visible Digest text.
- Digest Recent rows use the same accessible name pattern as Library episode rows
  (episode title, feed display label). Topic band hit rows use an

  **Open graph and episode details:** prefix plus that pair so the name stays distinct from
  the Recent row for the same episode.
  Summary pills expose an accessible name that states graph navigation and the full
  bullet text.

---

## E2E contract

Any new visible labels (e.g. Digest, Search topic) require updates to
the [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md)
before or with implementation. Dedicated Playwright coverage lives in
`e2e/digest.spec.ts` (mocked `GET /api/corpus/digest` + health).

---

## Revision history

| Date       | Change                                                           |
| ---------- | ---------------------------------------------------------------- |
| 2026-04-10 | Initial content (in UXS-001)                                     |
| 2026-04-11 | Health discovery, glance gate, Search topic, E2E digest.spec     |
| 2026-04-11 | No in-column title; digest toolbar row (window / date lens)      |
| 2026-04-15 | Removed **Open Library** control; use main **Library** tab       |
| 2026-04-15 | Topic bands + Recent pills: open Graph with topic focus          |
| 2026-04-13 | Extracted from UXS-001 into standalone UXS-002                   |
| 2026-04-17 | Topic band rows: date, E#, duration in native title only         |
| 2026-04-17 | Topic bands max-h min(42vh, 21.5rem); extra recap visibility     |

---

## Digest & Library alignment decisions

Shared decisions that affect Digest (and sometimes Library) live here; Library-only layout rules remain in [UXS-003](UXS-003-corpus-library.md).

**Status:** Decided (feeds implementation in repo)
**Spec:** [Supplement — Digest & Library UX improvements (full spec)](UXS-002-corpus-digest.md#supplement--digest--library-ux-improvements-full-spec)
**GitHub:** [#610](https://github.com/chipi/podcast_scraper/issues/610)

### §2.3 Similarity (topic band hit rows)

**Decision:** Show the **semantic tier label** (“Strong match” / “Good match” / “Weak match”) **inline on the row**, **right-aligned** in the hit meta area (per UX spec). Put the **raw numeric score** on a native **`title`** on that same element: `Similarity: 0.87` (and keep the existing row-level hover/title for other catalog fields unchanged).

**Rationale:** The WIP assumed a visible mono pill; current code often exposed score only inside a long native `title`. Inline text matches the spec’s “plain text label” and keeps precision available without implying false precision in the primary label.

### §2.4 CIL cluster pill colour (kg vs Quote)

**Decision:** Implement **`kg`** Tailwind treatment for **Digest Recent** CIL pills only: `bg-kg/15`, `border-kg`, text **`text-surface-foreground`** (readable on both themes). **Episode rail** and any other `CilTopicPillsRow` usage keep the existing **Quote-linked** amber chrome via a prop default so we do not change non-Digest surfaces.

**Rationale:** `text-kg-foreground` / `--ps-kg-foreground` do not exist in the theme today; `surface-foreground` on a light `kg` tint meets contrast without expanding the token set.

### §3.2 Feed filter threshold

**Decision:** Keep **`15`** feeds as the cutoff for showing the client-side “Filter feeds…” input, as a named constant in `LibraryView.vue` (and note in UXS tunables when docs are updated).

### §4 Recency dot (24h)

**Decision:** Parse **`publish_date`** as local **`YYYY-MM-DD`** at **00:00 local**. Show the dot when `Date.now() - localMidnight` is in **`[0, 86_400_000)`** (strict rolling 24 hours from the start of the listed publish **calendar day**). Hover title uses whole hours since that midnight. Full “same calendar day as today” is a subset edge case when the listed day is “today”.

---

## Supplement — Digest & Library UX improvements (full spec)

Authoritative for **Digest** layout deltas; **Library**-only subsections are included here too (single copy). [UXS-003](UXS-003-corpus-library.md) summarises Library behaviour and points here for this checklist.

**Status:** Ready for implementation
**Author:** Design session (Marko + Claude), April 2026
**Repo:** `chipi/podcast_scraper`
**Target area:** `web/gi-kg-viewer/src/components/digest/`, `web/gi-kg-viewer/src/components/library/`
**Related docs:** UXS-002, UXS-003, UXS-001
**Scope:** Visual and layout improvements only. No new API endpoints required
except where noted. No store architecture changes. Works independently of
the shell restructure spec.

---

### 1. Overview

Eight targeted improvements across Digest and Library. All are low-risk
visual/layout changes. Grouped by surface for implementation clarity.

| # | Area | Change | Effort |
| --- | --- | --- | --- |
| 1 | Digest | Topic bands — progressive reveal, remove nested scroll | Low |
| 2 | Digest | First band visual prominence | Low |
| 3 | Digest | Similarity pill → semantic label | Low |
| 4 | Digest | CIL topic pill colour — align with graph `kg` token | Low |
| 5 | Digest | Summary preview line clamp | Low |
| 6 | Library | Summary preview line clamp | Low |
| 7 | Library | Feed filter — remove mini nested scroll | Low |
| 8 | Library | Topic cluster filter — rename and elevate | Low |
| 9 | Both | Recency dot for last 24h episodes | Low |

---

### 2. Digest Changes

#### 2.1 Topic bands — progressive reveal, remove nested scroll

**Problem:**
Topic bands currently live in a `max-height: min(42vh, 21.5rem)` container
with `overflow-y-auto`. This creates a scroll region nested inside a
scrolling page — a persistent usability friction point. Users lose track
of which scroll context they're in.

**Solution:**
Remove the height cap and nested scroll entirely. Instead show the first
**3 topic bands** by default. If the API returns more, a **"Show more
topics"** control appears below the grid.

**Spec:**

- Default: render first 3 bands from the API response
- If `total_bands > 3`: show a `text-xs primary` link/button below the
  grid: **"Show N more topics"** where N = total_bands - 3

- Clicking it renders all remaining bands inline (no animation needed —
  just append). The control disappears after expansion.

- If `total_bands <= 3`: no control shown, all bands visible
- Remove `max-height`, `overflow-y-auto`, `rounded-sm` from the outer
  region wrapper

- The outer `role="region"` `aria-label="Topic bands"` and responsive
  grid (`sm:2` / `xl:3` columns) remain unchanged

**"Show more" button:**

```text
text:       "Show [N] more topics"
style:      text-xs, primary token, underline on hover
placement:  below the grid, left-aligned
testid:     data-testid="digest-topic-bands-show-more"
```python

**UXS-002 amendment:**
Replace the `max-height` / `overflow-y-auto` paragraph with the
progressive reveal rule. Remove `rounded-sm` from the outer wrapper
description.

---

#### 2.2 First band visual prominence

**Problem:**
All topic bands render identically. The first band is the most relevant
topic in the window (API returns bands in relevance/cluster order) but
looks the same as the fifth.

**Solution:**
The first topic band card gets a subtle elevated treatment — slightly
stronger border, slightly elevated background — to signal it is the
primary topic.

**Spec:**

- First band card (`index === 0`): use `elevated` background token
  instead of default `surface`, and `primary` border colour at reduced

  opacity (20%) instead of default `border` token

- All other band cards: unchanged (`surface` background, `border`)
- Topic title font weight: first band uses `font-bold` instead of
  `font-semibold`

- No size change, no header label — the treatment is subtle, not a
  crown

```typescript
// In DigestView.vue band rendering:
const bandCardClass = (index: number) =>
  index === 0
    ? 'bg-elevated border border-primary/20'
    : 'bg-surface border border-border'

const bandTitleClass = (index: number) =>
  index === 0 ? 'text-sm font-bold' : 'text-sm font-semibold'
```yaml

**UXS-002 amendment:**
Add to topic bands section: "The first band card uses `elevated`
background and `primary/20` border; subsequent bands use `surface`
background and `border`."

---

#### 2.3 Similarity pill → semantic label

**Problem:**
Similarity score on topic band hit rows is a small mono pill showing a
raw number (e.g. `0.87`). The user does not know what 0.87 means in
absolute terms — it is a relative signal. The raw number communicates
false precision.

**Solution:**
Replace the mono pill with a semantic strength label. Three tiers:

| Score range | Label | Style |
| --- | --- | --- |
| ≥ 0.85 | Strong match | `gi` token, `text-[10px]` |
| 0.70 – 0.84 | Good match | `muted` token, `text-[10px]` |
| < 0.70 | Weak match | `disabled` token, `text-[10px]` |

- No border, no pill shape — plain text label, right-aligned in the hit
  row meta area

- Native `title` tooltip retains the raw score for users who want it:
  `title="Similarity: 0.87"`

- The three thresholds are starting values — add to UXS-001 tunable
  parameters table as Open

**UXS-002 amendment:**
Replace "small mono pill with native tooltip" with the semantic label
spec above. Add thresholds to UXS-001 tunable parameters.

---

#### 2.4 CIL topic pill colour — align with graph `kg` token

**Problem:**
CIL topic pills on Recent rows use amber/orange fill when
`in_topic_cluster` is true. The graph spec ([Viewer graph spec — Graph visual styling](../architecture/VIEWER_GRAPH_SPEC.md#graph-visual-styling))
uses `kg` violet/purple for TopicCluster compound nodes. The same
concept — topic cluster membership — is rendered with two different
colours in two surfaces. Users moving between Digest and Graph will
not recognise these as the same thing.

**Solution:**
Align pill colour with the `kg` token.

**Spec:**

- Pills with `in_topic_cluster: true`: use `kg` token for border and
  background tint (same faint fill used for TopicCluster compound nodes

  in Graph — `kg` at ~15% opacity background, `kg` border)

- Pills without `in_topic_cluster`: existing style unchanged (no fill,
  `border` token)

- Text colour: **`surface-foreground`** on cluster pills (readable on the
  **`kg`** tint in light and dark). The theme documents **`kg-foreground`**

  for solid **`kg`** fills; we do not add a separate token for this faint
  tint — see [Digest & Library alignment decisions](UXS-002-corpus-digest.md#digest--library-alignment-decisions)
  §2.4. Regular pills stay **`surface-foreground`** on neutral chrome.

- This applies only to `cil_digest_topics` pills on Recent rows —
  no change to any other pill usage

```typescript
const pillClass = (pill: CilPill) =>
  pill.in_topic_cluster
    ? 'bg-kg/15 border border-kg text-surface-foreground'
    : 'bg-transparent border border-border text-surface-foreground'
```yaml

**UXS-002 amendment:**
Update the `cil_digest_topics` pill description: remove reference to
amber/orange, replace with `kg` token. Add cross-reference to
[Viewer graph spec — Graph visual styling](../architecture/VIEWER_GRAPH_SPEC.md#graph-visual-styling) for TopicCluster visual consistency.

---

#### 2.5 Summary preview line clamp — Digest Recent rows

**Problem:**
`summary_preview` on Recent rows is full-wrap with no line clamp. An
episode with a six-bullet summary creates a very tall row. An episode
with a one-sentence summary creates a short row. Variable row heights
break visual rhythm — the eye cannot scan the list predictably.

**Solution:**
Clamp summary preview to **2 lines** by default on Recent rows. When
the episode is selected (row `bg-overlay` active, Episode subject rail open),
show the full summary — the rail already shows it in full detail anyway.

**Spec:**

- Default (unselected): `line-clamp-2` on `summary_preview` container
- Selected (row has `bg-overlay` / is the active subject): remove clamp,
  show full text — `line-clamp-none`

- No expand toggle on the row itself — selection is the expand trigger
- When clamped, standard browser ellipsis (`…`) at clamp boundary
- `title` attribute on the clamped container holds the full text for
  hover access

**UXS-002 amendment:**
Update Recent layout section: replace "full wrap, no clamp" with
"2-line clamp on unselected rows; full text when selected."

---

### 3. Library Changes

#### 3.1 Summary preview line clamp — Library episode rows

**Problem:**
Identical to Digest (section 2.5). Library episode rows also use
full-wrap summary with no clamp, creating variable row heights and
broken visual rhythm.

**Solution:**
Same rule as Digest. 2-line clamp on unselected rows, full text when
selected.

**Spec:**

- Default (unselected): `line-clamp-2` on `summary_preview`
- Selected (row has `bg-overlay`): `line-clamp-none`
- `title` attribute holds full text for hover

**UXS-003 amendment:**
Update episode column section: replace "full wrap, no line clamp" with
"2-line clamp on unselected rows; full text when selected."

---

#### 3.2 Feed filter — remove mini nested scroll

**Problem:**
The feed list inside the Filters collapsible has `max-height ~two row
heights` with scroll. With 10+ feeds this is a nested scroll inside a
collapsible inside a scrolling page — the same anti-pattern as topic
bands.

**Solution:**
Remove the max-height cap from the feed list. Show all feeds. The
collapsible container is already the boundary — if the user has many
feeds, the collapsible expands to fit them. This is acceptable since
the collapsible is collapsed by default and the user opens it
intentionally.

**Spec:**

- Remove `max-height` and `overflow-y-auto` from the feed list region
- Feed list height: natural height of all feed rows
- If corpus has > 15 feeds: add a search/filter input above the feed
  list — `text-xs`, placeholder "Filter feeds…", filters the visible

  feed rows client-side by display title. This is a conditional
  enhancement — only render if `feeds.length > 15`

- Feed rows, dividers, `overlay` selected state: unchanged

**Feed search input (conditional):**

```text
style:      text-xs, surface background, border, w-full, mb-1
testid:     data-testid="library-feed-filter-search"
behaviour:  client-side filter on feed display title (case insensitive
            substring match). Does not call API. Filters the rendered
            list only.
```yaml

**UXS-003 amendment:**
Replace max-height/scroll description with natural height + conditional
search input spec.

---

#### 3.3 Topic cluster filter — rename and elevate

**Problem:**
"Episodes with topic cluster (CIL)" checkbox is buried inside the
collapsible Filters section with a technical name. Topic cluster browsing
is a meaningful primary mode — "show me only the episodes that have been
deeply analysed" — but it requires knowing to look inside the filter
panel for something with an acronym.

**Solution:**
Two changes: rename it, and move it outside the collapsible as a
persistent toggle above the episode list.

**Rename:**

| Current | New |
| --- | --- |
| "Episodes with topic cluster (CIL)" | "Clustered episodes only" |

Short, plain language. No acronym. Communicates the browsing intent
("only the analysed ones") rather than the technical mechanism.

**Elevate:**

Move from inside the Filters collapsible to a persistent position
between the Filters collapsible and the Episodes heading. Always
visible regardless of whether Filters is open or closed.

Layout:

```json

[ ☐ Clustered episodes only ]          ← always visible, new position

h2 Episodes (N)  ?
[ episode list... ]
```python

- Same checkbox style as before (`text-xs`, `surface` background)
- When checked: the `topic_cluster_only=true` param is added to
  `GET /api/corpus/episodes` and the list reloads — unchanged behaviour

- A muted `text-[10px]` line below the checkbox when checked:
  "Showing episodes in a topic cluster" — confirms the active filter

  state

- `data-testid="library-topic-cluster-toggle"` (update from existing
  testid)

**UXS-003 amendment:**
Remove checkbox from Filters collapsible description. Add new persistent
toggle section between Filters and Episodes heading. Update accessible
name.

---

### 4. Cross-cutting: Recency dot for last 24h episodes

**Problem:**
Digest and Library episode rows give no visual signal about how fresh
an episode is within the time window. A 2-hour-old episode looks the
same as a 6-day-old one. For Digest especially — a discovery surface —
"just published" is high-signal information.

**Solution:**
A small dot indicator on episode rows for episodes published within the
last 24 hours. Subtle, non-intrusive, purely additive.

**Spec:**

- Condition: **`publish_date`** is a **`YYYY-MM-DD`** string; treat it as
  **local calendar midnight** for that date. Show the dot when

  **`Date.now() - localMidnight`** is in **`[0, 86_400_000)`** (rolling
  24 hours from the start of the **listed** publish day). This matches
  date-only APIs and avoids implying a false time-of-day; see
  [Digest & Library alignment decisions](UXS-002-corpus-digest.md#digest--library-alignment-decisions) §4.

- Indicator: a `6px` filled circle, `success` token colour
- Placement: immediately before the episode title, vertically centred
  with the title baseline

- No visible text label — the dot stays visually quiet. **Optional** native
  **`title`** and a matching **`aria-label`** on the dot give hover and

  screen-reader context (e.g. "Published N hours ago" / "Published less than
  1 hour ago", derived from whole hours since local publish-day midnight).

```text

- In Digest Recent rows: dot appears before title in the title row
- In Library episode rows: same placement
- In Digest topic band hit rows: dot appears before episode title in
  the hit row title cell

```python

**Time calculation:**
Client-side only; **no** new API fields. Parsing and the rolling window are
implemented in **`web/gi-kg-viewer/src/utils/digestRecency.ts`** (invalid or
missing **`publish_date`** → no dot). Digest window copy (API lens) can
still differ from the dot for date-only rows — document in UXS, do not
force them to match without **`publish_datetime`**.

**UXS-002 + UXS-003 amendment:**
Add recency dot to episode row layout description in both specs.
Note 24h threshold as a tunable parameter in UXS-001.

---

### 5. Files to Touch

#### Digest

```text
  — 2.1: Remove max-height/scroll from topic bands, add show-more logic
  — 2.2: First band elevated card class
  — 2.3: Similarity pill → semantic label
  — 2.4: CIL pill colour → kg token
  — 2.5: summary_preview line-clamp-2 on unselected rows
  — 4:   Recency dot

web/gi-kg-viewer/src/utils/digestRowDisplay.ts
  — 2.3: Add `digestTopicHitSimilarityDisplay(score)` (semantic label + raw `title`)

web/gi-kg-viewer/src/utils/digestRecency.ts
  — 4: Rolling-window helpers for `YYYY-MM-DD` (local midnight) + unit tests
```

#### Library

```text
  — 3.3: Move topic cluster toggle outside collapsible; rename
  — 4:   Recency dot
```

#### UXS amendments (after implementation)

```text
docs/uxs/UXS-003-corpus-library.md
  — 3.1, 3.2, 3.3, 4 per sections above

docs/uxs/UXS-001-gi-kg-viewer.md
  — Add to Tunable parameters:
      similarity strong threshold    0.85    Open
      similarity good threshold      0.70    Open
      recency dot window             24h     Open
```

#### E2E surface map

```text
  — library-topic-cluster-toggle (renamed testid)
```yaml

---

### 6. Checkpoints

**Checkpoint 1 — Topic bands**

- Max 3 bands visible on load; "Show N more topics" appears when API
  returns > 3

- Clicking show-more renders all bands; control disappears
- First band has elevated background and primary/20 border
- No nested scroll region — bands flow naturally in page scroll

**Checkpoint 2 — Similarity labels**

- Score ≥ 0.85 → "Strong match" in gi colour
- Score 0.70–0.84 → "Good match" in muted colour
- Score < 0.70 → "Weak match" in disabled colour
- Raw score visible on hover (native title)
- No mono pill visible

**Checkpoint 3 — CIL pill colour**

- `in_topic_cluster` pills use kg background tint and kg border
- Regular CIL pills unchanged
- Matches TopicCluster visual treatment in graph

**Checkpoint 4 — Summary line clamp**

- Unselected rows: summary clamped to 2 lines
- Selected rows: summary fully visible
- Applies in both Digest Recent and Library episode list

**Checkpoint 5 — Feed filter**

- Feed list has no max-height; all feeds visible
- If feeds > 15: filter input appears above list; filters client-side
- If feeds ≤ 15: no filter input

**Checkpoint 6 — Topic cluster toggle**

- Toggle sits between Filters collapsible and Episodes heading
- Label reads "Clustered episodes only"
- Behaviour unchanged; confirmation line appears when active
- Not inside Filters collapsible

**Checkpoint 7 — Recency dot**

- When **`publish_date`** parses as **`YYYY-MM-DD`** and local age from that
  **calendar day’s midnight** is in **`[0, 24h)`**, a 6px **`success`** dot

  appears before the title

- Dot appears in Digest Recent, Digest topic band hits, Library rows
- Hover / **`aria-label`** on the dot: "Published less than 1 hour ago" or
  "Published N hours ago" (whole hours since local publish-day midnight)

- No dot when the date is missing, invalid, or outside that window (e.g.
  **yesterday** late in the **next** calendar day is **outside** `[0,24h)`

  from **yesterday’s** midnight)

---

### 7. What This Does Not Change

- Digest toolbar structure (heading, lens, rolling bounds row)
- Topic band hit row layout (cover column, title column, summary span)
- Library Filters collapsible structure (date, title, summary inputs)
- Episode subject rail content and behaviour
- Any store logic or API calls (except recency dot which is client-side)
- Token system
- Pagination / infinite scroll behaviour
