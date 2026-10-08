# UXS-013: Knowledge clusters & entity cards (consumer)

- **Status**: Draft (cluster-first panel — Implemented, Epic 3.1)
- **PRD**: `docs/prd/PRD-043-knowledge-layer.md`
- **RFC**: `docs/rfc/RFC-102-knowledge-clusters-entity-cards.md`
- **Inherits**: UXS-011 (Editorial Bold tokens, `--lp-*`) and UXS-012 (Home).

---

## Scope

The knowledge-navigation UX of Epic 3: cluster-first topics in the Episode notes panel (named "Insights" until 2026-09-30; 3.1, shipped),
and the person/topic **entity cards** + entity search results (3.2–3.4, design). Mobile-first;
WCAG 2.1 AA; i18n (no hard-coded strings).

## Cluster-first "Topics & People" (3.1 — shipped)

Within the Episode notes panel's compact, expandable **Topics & People** row:

- **Order:** the **dominant cluster** (most of this episode's topics, ≥2) leads; its chips get a
  1px **`ring-topic`** outline to stand out. Other clustered topics follow (larger intra-episode
  groups first); singleton topics trail; **people** chips (`text-person`) come after topics.
- **Storyline lead-in:** a small line beside the section header — **"Storyline · {cluster}"** —
  names the dominant co-occurrence cluster (hidden when there is none). This said "Theme ·" until
  2026-09-19; see **Vocabulary** below for which word now names which object.
- **Affordance:** tapping a chip opens its **entity card** (3.2/3.3 — shipped; the Epic-2
  chip→search default now lives as an explicit action inside the card). The dominant ring is a
  *visual* cue, not a new control. Collapsed at 6 chips; **+N …** expands.
- **Degrade:** no `topic_clusters.json` → no rings, no theme line, flat order (today's behaviour).

**Tokens:** topic `--lp-topic`, person `--lp-person`, chip bg `--lp-overlay`, ring `--lp-topic`.
**A11y:** the ring is supplementary to label text + colour (not the only signal); chips keep their
`aria-label` ("Open {term}"); contrast per UXS-011.

## Entity cards (3.2 person · 3.3 topic — shipped)

One `EntityCard` overlay serves both (sheet on mobile, centred panel on desktop). The
`EntityCardBody` shell owns the header (kicker / title / follow / save / dismiss) and the
re-entrant back stack; the kind-specific body is delegated to `PersonCardContent`,
`TopicCardContent`, and `OrgCardContent` (#2031), so the one stack can carry a mixed
person↔topic↔organization walk in a single panel:

- **Person card:** a "Person" kicker (with a role badge) + name, the bio and photo block, "Host of …",
  "Often appears with", related people, related topics MIXED with their themes and storylines,
  a "Search transcripts for {name}" action, "Who agrees with {name}", an episode list and notes.
  Data: KG co-occurrence via `GET /api/app/persons/{id}`. Order and treatment: UXS-011 §Entity
  pages (2026-10-05).
- **Topic card:** a "Topic" kicker + label; "{n} similar topics" as a kicker label at the start of
  the sibling pill row (the semantic siblings — NOT the topic you are reading, which is not similar
  to itself); the theme and storyline as link cards whose caption names the kind ("THEME · 3
  TOPICS", "STORYLINE · 4 TOPICS"), with no section heading over them; Top voices; "{n}
  perspectives on {topic}"; a "Discussed in {n} episodes" list paged by 10. Data: `GET
  /api/app/topics/{id}`. Desktop page layout: UXS-011 §Entity pages.
- **Organization card (#2031):** an "Organization" kicker + name, a "Mentioned in {n} episodes"
  list, and co-occurring people / **other organizations** / topics chips. Leaner still than the
  person card — no follow-adjacent save/collection — but it DOES carry a lean web block
  (description + logo + attribution) when the org_web enricher (#2035) matched.
  Data: KG `MENTIONS_ORG` co-occurrence via `GET /api/app/organizations/{id}`; reachable from the
  search box (entity resolution) and by drilling org→org from another org card.
- **Re-entrant:** tapping a related person/topic chip walks to that entity in place, with a back
  (‹) control; the search action lives inside the card, not on chip-tap.
- **Open/close:** tap to open; mobile = bottom sheet with backdrop; desktop = centred panel. Modal
  a11y: `role="dialog"`/`aria-modal`, focus trap, initial focus + restore-on-close; dismiss via
  ESC, backdrop, or the ✕ control.

## Vocabulary — which word names which object (settled 2026-09-19)

The two cluster kinds are built differently and the wire names INVERT against the reader-facing
ones. Getting this backwards is the recurring failure (#1603), so it is stated once here:

| Wire prefix | Backend name | Built from | Reader-facing name |
| --- | --- | --- | --- |
| `thc:` | theme cluster | **co-occurrence** — topics that keep coming up together | **Storyline** |
| `tc:` | topic cluster | **vector similarity** — topics that mean similar things | **Theme** |

`interestKind()` in `web/learning-player/src/utils/interests.ts` is the boundary where the wire
names stop mattering — every surface should take its word from there rather than from the prefix.

**The code now matches** (rename completed 2026-09-20). Only the prefixes still invert; everything
built on top of them reads the same word a reader sees:

| Layer | Storyline (`thc:`) | Theme (`tc:`) |
| --- | --- | --- |
| Module | `search/storylines.py` | `search/topic_clusters.py` |
| Consumer route | `GET /api/app/storylines` | `GET /api/app/themes` |
| Operator route | `GET /api/corpus/storylines` | `GET /api/corpus/topic-clusters` |
| Wire fields | `storyline_id` / `_label` / `_size`, `storyline_sibling_topics`, `dominant_storylines` | `cluster_id` / `_label` / `_size`, `sibling_topics` |
| Player symbols | `storylineId`, `storylineDominantLabel` | — |
| Viewer symbols | `storylineId`, `storylinesDoc`, `storylineRegions`, `useGraphStorylineFocusStore` | — |

Two response models keep older names on purpose. `AppInterestCluster` /
`AppInterestClustersResponse` are the wire shape behind `GET /api/app/themes`: they carry `tc:`
ids, so they mean **Theme**. The name is stale rather than wrong — it does not claim to be the
other object — and renaming it would move the consumer picker on both sides for no reader-visible
gain. `AppStoryline` (consumer) and `AppStorylineDetail` (operator) are two views of one storyline
and are named for their audience, not just their subject.

Three things deliberately keep the old spelling, and none is an oversight:

- **The wire prefixes** `thc:` / `tc:`. Not for back-compat — this project is pre-launch and
  forward-only, so "it would orphan historical events" is not a binding argument here and would
  read as an excuse later. The real reason: an id prefix is a contract with DATA, not with a
  reader. It has to be unique and stable; it does not have to be meaningful, and nobody ever sees
  one. `interest_events.jsonl` (append-only follow log), the artifacts and operator localStorage
  all key on these, so a rename is pure cost against zero reader-facing gain — permanently, not
  "for now". No dual-read, no alias: that is exactly the compat shim this project bans.

  Worth stating plainly because the option decays: today a prefix rename is a one-shot corpus
  rewrite plus a follow-log reset. Once there are real users, forward-only ends and it becomes a
  true migration. Pre-launch is the cheapest this will ever be, and the answer is still no.
- **The artifact** `enrichments/topic_theme_clusters.json`, because renaming it forces a
  re-enrichment of every existing corpus, prod included, for no reader-visible gain.
- **The persisted lens key** `themeClusterRegions` inside `ps_graph_lenses`. The viewer symbol is
  `storylineRegions`, but the operator's saved value is still READ under the old key: that flag
  defaults to false, so dropping it would not look like a reset, it would look like the region lens
  had stopped working. `localStorage` is a storage contract, not a variable name — the same reason
  `communityColours` is still read one line above it.

All three are invisible to readers of the product. `interestKind()` remains the boundary where the
wire names stop mattering.

The `sth:` super-theme tier is untouched. It has no LISTENER-facing name, so there is nothing to
rename it to, and `sth:` ids are persisted as the operator's saved graph expansions.

It is not invisible, though, and the distinction matters: the operator graph renders
`super_theme_label` as the node label (`utils/topDownSlice.ts`) and styles `node[type =
"SuperTheme"]`. Super-themes group **storylines** (`thc:`, from the `topic_theme_clusters`
enricher), so the name reads as "an aggregate of Themes" while actually aggregating Storylines —
the same inversion this page exists to kill, surviving one tier up. Left as-is because
`"SuperTheme"` is a graph-payload contract and `sth:` is persisted operator state; rename both
together the next time that payload takes a breaking change, not before.

One more half-renamed edge: `GET /api/app/trending?kind=` takes `cluster` for a **Theme** (`tc:`)
and `storyline` for a Storyline (`thc:`). The storyline half got the new word and the theme half did
not. Harmless today because nothing reads `cluster` as a reader-facing label; worth aligning
whenever that enum next changes.

This section settles the INTERESTS vocabulary only. The Knowledge Panel lead-in and the remaining
`"theme"`/`"similar"` i18n pair are tracked on #1603.

## Entities in search (3.4 — shipped)

When a query exact/near-exact-matches a person/topic, an **entity card** sits **above** the grouped
passage results (UXS-012 search): a kicker ("Person"/"Topic") + name + a "View ›" affordance,
tapping which opens the §Entity-cards overlay. Distinct from passage cards, never blocking the
passages below; the "no grounded passages" line is suppressed when an entity matched (we *did* find
something). Resolved in parallel with the passage search, so a miss never delays results.

## Personalized discovery (3.5 — shipped)

- **First-Home card:** a dismissible "Personalize your Home" card (signed-in only; remembered via
  `localStorage`) opens the **interests picker**.
- **Picker:** a modal over the corpus's **top-12 clusters** (toggle chips; preselected from saved
  interests; Save / Not now). Modal a11y matches the entity card (focus trap, ESC/backdrop).
- **Effect:** the Home "What's new" feed re-ranks by interest affinity **only** when the deployment
  flag is on; by default (and signed-out) it is recency — visually identical to today.
- **Tokens:** selected chip = `--lp-accent`; unselected = `--lp-overlay` / `--lp-topic`.

## Knowledge bands on the episode, show and topic pages (documented 2026-09-03)

Three bands that render the knowledge layer where the listener already is, rather than sending them
to a separate "graph" surface. All three shipped with automation and no spec; all three follow the
same rule as the rails: **absent intelligence omits cleanly** — a band with no signal is not
rendered, never rendered empty.

### `EntitySignals` — why this entity matters here

The signal strip on an entity card: how much of the corpus this person/topic touches, and where.
Counts are stated, never implied by bar length alone, because a bar with no axis reads as a
precision the underlying data does not have.

### `PodcastSignalsBand` — what sets THIS show apart

On a show page. Topics split deliberately into two groups: **distinctive** topics carry `lift`
above the corpus base rate — the ones that make this show itself — and the remainder are listed
plainly. The split exists because an alphabetical tiebreak once let a show's signature topic lose
to wallpaper every show covers.

> The momentum bubble cloud was **removed** from this band: its `velocity` is corpus-wide, not
> show-scoped, so it sized topics by a number that did not answer the band's own question.

### `TopVoices` — the people who drive a topic or storyline

The top eight of the server-ranked `related_people` as a four-column grid, avatar over name (photo
where one is hosted, initials otherwise). It is the SAME component on the topic card and the
storyline page (2026-09-30): the storyline page used to list the same people as plain "Related
people" chips — one dataset drawn two ways, which is how two surfaces drift. On the topic card each
voice is a button (a tap layers the person over the card); on the storyline page it is a real link
to the person, because a page is where navigation belongs. `TopicPerspectives` follows it on the
topic card, naming the same people and what they argued.

### `TopicPerspectives` — who disagrees, and where

On a topic page: the positions different people take on one topic, grounded in episodes. This is
the surface that distinguishes a knowledge layer from a tag cloud, so it is held to the grounding
rule strictly — every position carries its source, and an ungrounded claim is not shown at all.

#### "What's said across this theme / storyline" (2026-10-01)

The same component now serves all three surfaces via a `kind` prop (`topic` | `theme` |
`storyline`), reading `/topics/{id}/perspectives`, `/themes/{id}/perspectives` or
`/storylines/{id}/perspectives` as appropriate.

**Why a grouping needed it.** A theme and a storyline listed their member topics and their
episodes, and nothing on either page was a sentence anybody had actually said — the reader was left
to infer what the grouping sounded like from a list of nouns. Four cheaper options were measured and
rejected first: a derived description ("8 topics, 40 episodes across 9 shows") is the member table
restated as prose; a name assembled from the top members is rows 1 and 2 of that same table; an
LLM-written set description fabricates coherence on a weak cluster and cannot report when it is
doing so. What none of them supply is the thing the member list cannot: a real claim, by a named
speaker, with a link to the moment they said it.

**Scope is the UNION of the grouping's members**, which is what the grouping IS. A speaker is
counted once across the whole grouping, so someone arguing the same line under three members is one
perspective holding three takes — not three perspectives. One corpus walk
(`cil_queries.topics_perspectives`), with the single-topic path delegating to it so the two cannot
drift in how they match, attribute or rank.

**Heading.** A topic keeps its count ("4 perspectives"); a grouping says what it is — "What's said
across this theme" / "…this storyline". On a grouping the count is not the useful fact, and the
heading is the only copy on those pages asserting that a grouping is something people TALK about
rather than a list of topics.

**Position: between the member list and the episode list, with nothing in between, on all three
surfaces** (operator 2026-10-01). It sat at the foot of the grouping pages until then — below an
episode list that runs thousands of pixels on a phone — so the one section explaining what the
grouping is about was the last thing on the page, under everything it was meant to explain. The page
now answers "what is this?" before offering "here is everything in it". On the topic card this meant
moving it out from under Top voices, where a 2026-09-19 note had placed it; consistency across the
three won, because these are the three surfaces a reader compares.

**Folding.** A grouping has as many speakers as all its members combined — 11 on the storyline
fixture, which took the page from ~2,200px to 11,185px. Grouping surfaces show four speakers
(ranked most-takes-first) behind one `perspectives-more-speakers` control; a topic shows all of
them, as it always did.

**Absence is honest.** Renders nothing when no member has a speaker-attributable insight. That is
the normal outcome for a grouping whose members are abstract labels nobody utters aloud
(`tc:lifelong-learning` is exactly that), and showing nothing beats showing unrelated quotes.

**Known limit (#2248).** Ranking is `salience` descending (ADR-135/#1191). Where artifacts carry no
salience the sort degenerates to `position_hint` ascending — earliest-in-episode first — which
surfaces greetings rather than claims. Visible on the v3 fixture, whose insights all have
`salience: null`; **not verified against production**, and that measurement is what decides whether
this is a fixture gap or a live product defect.

### `TopicConversationArc` — how a topic moved over time

A bar series of a topic's presence across the corpus timeline. Same restraint as the trend sparks:
shape over precision.

**Shown only where there is an arc worth drawing** (2026-09-30, #2202). A week counts only when an
insight is ABOUT the topic (a GI `ABOUT` edge), and most topics an episode merely mentions have
none. On the fixture corpus 33 of 50 topics have no arc and 10 more have a single week. So:

- The topic card carries `conversation_arc_weeks`, and the section decides BEFORE drawing
  anything. With fewer than two weeks there is no section, no placeholder and no arc request. It
  used to draw a loading placeholder for every topic and then remove it for most of them, which
  read as a section that appears and vanishes.
- One week is a bar, not a trend, so a single-week arc is not shown either. That rule also holds
  when the count is absent (an older server): whatever loads is drawn only at two weeks or more.
- With two or more weeks it loads and draws as before, placeholder included. The card and the
  arc route share one scan through a 30-second memo, so the up-front count costs no second
  corpus walk.

Whether insights should be linked to more of an episode's topics (so more topics have an arc) is
the data question in #2202, not a UI one.

### `ShowActivityChart` — a show's publishing rhythm

Episodes over time on a show page. It answers "is this alive?" — a question a listener asks before
following, and one an episode list buried in dates answers badly.

### Player insight band

Where the insights sit inside an episode, as ticks on the player's band. It gives a listener a reason
to scrub somewhere specific rather than sampling blindly. The episode notes' own `EpisodeDensity`
box (early / mid / late) was removed on 2026-10-08 (operator): it took a screen before the first
insight and repeated what the band already shows.

## Decisions (operator, 2026-06-25)

- Person **and** topic cards use **dedicated** `/api/app/persons|topics/{id}` endpoints (KG
  co-occurrence), not a proxy of the viewer relational API — effort over coupling.
- Entities-in-search (3.4) surfaces a card only on **exact/near-exact** entity-name match
  (consumer first), reusing the 3.2/3.3 cards.
