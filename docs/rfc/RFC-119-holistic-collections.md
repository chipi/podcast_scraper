# RFC-119: Holistic Collections — curation buckets for anything, as a first-class Library tab

- **Status**: Draft
- **Authors**: Marko, Claude (Opus 4.8)
- **Stakeholders**: Consumer App, Server API
- **Related RFCs**:
  - `docs/rfc/RFC-111-curation-surfaces.md` (Collections v1 — highlight boards; this extends it)
- **Related PRDs**:
  - `docs/prd/PRD-046-delivery-and-curation.md` (curation requirements — FR4 collections)
- **Supersedes**: the item model of RFC-111 §1 (highlight-only boards → mixed-type buckets)

## Abstract

Today a **collection** is a named set of **highlight ids** (RFC-111 §1), rendered at the bottom of
the Library › Saved tab. This RFC expands it into a **holistic curation bucket** — a collection can
hold **episodes, shows, saved searches, topics, people, highlights, and external links** — and
promotes it to a **first-class Library tab**. The mental model is **Pinterest for listening**: explore a topic, pin
the episodes / shows / people / searches worth it, and analyze what to listen to (or assess) next — a
deliberate prep board you draw from, distinct from Queue (ordered playback) and Following (ongoing
interest). Additive and no-ML; reuses the canonical-identity + per-user overlay
already shipped.

## Problem statement

- **Too narrow.** Collections hold only highlights, so the natural "I'm interested in X — let me
  gather the episodes, the people, a search, and a few bits to line up my listening" flow has no home.
  The user already saves episodes, follows topics/people, and saves searches — but those live in
  separate lists with no way to curate them together toward an intent.
- **Buried.** Collections sit at the bottom of the Saved tab, below saved searches / episodes /
  insights / highlights — the least prominent slot for what should be the active "plan my listening"
  surface.

## Design

### Data model (server)

`<data_dir>/users/<id>/collections.json` item lists move from bare highlight ids to **typed items**:

```
{
  "collections": [{ "id", "name", "created_at" }],
  "items": {
    "<collection_id>": [
      { "kind": "highlight", "ref": "<highlight_id>" },
      { "kind": "episode",   "ref": "<slug>" },
      { "kind": "show",      "ref": "<feed_id>" },
      { "kind": "search",    "ref": "<query>", "scope": "all|mine" },
      { "kind": "topic",     "ref": "topic:<id>" },
      { "kind": "person",    "ref": "person:<id>" },
      { "kind": "link",      "ref": "<url>", "title": "<optional>" }
    ]
  }
}
```

- **Migration is lossless + lazy:** a bare string item (old shape) is read as `{kind: "highlight",
  ref: <string>}`. No rewrite needed; `_read` normalizes on load, `_write` emits the new shape.
- Membership stays a plain list; an item may belong to N collections. Deleting a collection drops its
  membership only; the referenced things (episodes, highlights, follows) are untouched — same overlay
  principle as v1.
- **Liveness:** counts/rendering resolve each `{kind, ref}` against its source store at read time
  (highlight → capture store, episode → catalog, topic/person → KG, search → re-runnable). A dangling
  ref (deleted highlight, purged episode) is dropped from the view, not the file — mirrors v1's
  `live_item_ids`.

### API + schema

- `POST /collections/{id}/items` body `{kind, ref, scope?}` (was highlight-id only); `DELETE` the
  same. `GET /collections/{id}` returns typed, resolved items (label/artwork/deep-link per kind) so
  the client renders a mixed list without N follow-up fetches.
- Add-to-collection affordance (`＋ to collection`) on every save surface: episode card, search Save,
  topic/person entity card, and the existing highlight row.

### UI

- **Own Library tab "Collections"** (promoted out of Saved). To keep the phone tab strip at five,
  **fold Queue + Recent into one tab** ("Queue & Recent" — up next + lately played, two sections):
  `Following · Saved · Collections · Revisit · Queue & Recent`.
- Collections tab: the board list (create / rename / delete) → a collection detail that renders its
  mixed items grouped by kind (Episodes, Searches, Topics, People, Highlights), each row navigable
  (episode → player, search → re-run, topic/person → card, highlight → jump-to-moment).
- **"Play all"** on a collection queues its episodes → the one deliberate bridge to Queue, so a
  collection genuinely "fuels the next listen".

### Concept boundaries (deliberate, to avoid overlap)

- **Follow** = ongoing interest (auto-surfaces new things) — Library › Following.
- **Save** = a loose bookmark — Library › Saved.
- **Collection** = a *deliberate, named prep bucket* you assemble toward an intent, and draw from to
  listen. A topic can be both followed AND dropped into a "Deep-dive on X" collection — different jobs.
- **Queue** = ordered "what plays next"; a collection's "Play all" feeds it.

## Phasing

1. **P1** — typed item model + migration; `＋ to collection` for **episodes + highlights**; collection
   detail renders both. (Delivers the core "gather episodes to listen" value.)
2. **P2** — **saved searches** as items (a search is a live "more like this" seed).
3. **P3** — **shows, topics + people** as items (pin a show from its page / a show tile; topics +
   people from entity cards).
4. **P4** — promote to its own **Collections tab** + fold **Queue & Recent**; "Play all" → Queue.

## External links

A `link` item is just a **URL** (with an optional user-supplied title) — a listener researching a
topic pins the articles / blog posts they find alongside the episodes and people. **For now we only
store + display the link** (open in the system browser); no metadata fetch, unfurl, or enrichment.
Future work may add title/preview scraping, but that is explicitly out of scope here — the value is a
Pinterest-style bucket that mixes in-app entities with outside reading.

## Non-goals

- Ordering/reordering within a collection (v1 stays newest-first; manual order is a later ask).
- Link enrichment (unfurl / preview / archive) — store the raw URL only for now.
- Sharing a whole collection (RFC-111 share cards remain per-highlight for now).
- Cross-device real-time sync beyond the existing per-user file overlay.

## Migration & rollback

- Forward: `_read` normalizes bare ids → `{kind:"highlight"}`; old clients ignore unknown item kinds
  (list endpoints already tolerate extra fields). Rollback: the file stays readable by v1 for
  highlight items; non-highlight items are simply invisible to a reverted client.

## Risks

- **Concept sprawl** — four save-like verbs (follow / save / collect / queue). Mitigated by the crisp
  boundaries above + "Play all" as the single collection→queue bridge.
- **Tab-strip pressure** — resolved by folding Queue + Recent; revisit if a sixth destination ever
  lands.

## Open question / future direction: Queue & Recent belong to the Player, not Library

Note for a follow-up (not decided here). Today Queue and Recent live in Library. The stronger model —
per Spotify et al. — is that **transport lists belong to the playing surface**: from the player you
tap one control and see **what's coming next** (Queue — reorderable, the point is to arrange your next
listens) and **what you just heard** (Recent — the point is *not* to re-queue it but to not lose it).

If Queue & Recent move to a **player-surface panel** (a queue button on the mini/full player, a
draggable "Up next" + a "Recently played" list), then:

- They leave Library entirely — which **frees the Collections tab slot on its own**, so P4's
  "fold Queue & Recent" becomes "**remove Queue & Recent from Library**" and the strip is
  `Following · Saved · Collections · Revisit` (+ room to spare).
- Collection **"Play all" → Queue** stays the bridge, now landing in the player's own Up-next panel —
  a tighter loop (curate in Collections → send to the player's queue → rearrange there).

This likely wants its own small RFC/issue (player transport UX); captured here because it changes the
tab-strip math P4 assumes.

## Amendment 2026-10-12: a board's Brief, its short listen, and boards in search

Collections are called **boards** in the app. Three additions (operator 2026-10-12; #2340, #2341).
All three are built from what is on disk, with no LLM call, like the episode Brief (RFC-122) and
short listens (ADR-163).

### The board Brief (#2340)

An episode has a Brief; a board gets one too. It is **not shown by default**: a "Brief" button on
the board's header opens it in a sheet laid out as the episode's Brief is (operator 2026-10-12).
It covers **only what is saved on the board itself**: a saved show, topic or person is not
expanded into its episodes. "The board's episodes" are its saved episodes plus the episodes of its
saved moments.

- **What it holds**: how many episodes, saved moments, shows, subjects, searches and links.
- **Play N moments**: the board's short listen (below), the episode Brief's play button.
- **Top voices**: a board has no host or guest of its own, so instead of the episode's people it
  lists who speaks (host or guest) across its episodes, most episodes first, as a topic does.
- **Key points**: the strongest insight of each of the board's episodes, with its quote and time,
  so every point can be heard where it is said (the "why" of ADR-164). When the insight is its
  quote cut short, the whole quote is shown.
- **What this board's about**: as the show page's band, a row per kind — storylines, themes,
  topics — every one of the board's episodes carries, most episodes first, umbrella-guarded
  (ADR-163 amendment). A theme with a storyline's name is left out of the row. A topic also saved
  on the board directly is marked as such.

`GET /api/app/collections/{id}/brief`, owner only (404 otherwise). A board with no episodes (shows,
searches and links only) has nothing to brief, and its sheet says so.

### The board's short listen (#2340)

Opened from the Brief: the board's own moments, back to back. Every saved moment that has a time,
in board order, then the strongest clip of each saved episode with no saved moment (the same clip
a topic's short listen would take from it, ADR-163 §8), within the minutes chosen. An episode with
a saved moment plays that moment only, so it is never heard twice in a row. While it plays, the
nav lights Library, where the board lives. Served at
`GET /api/app/collections/{id}/reel` in the shape of `/discover/reel` (`subject_kind: board`), so
the existing short-listen player plays it.

### Boards in search (#2341)

Search lists **your own boards** in a Boards section beside your notes: those whose name matches the
query, then those holding something that matches (a saved episode or show by title, a saved moment
by its words, a topic, person, theme or storyline by name, a saved search, a link). Each row says
what matched and opens the board (`/library?tab=collections&board=<id>`). Every word of the query
must appear, in any order, the rule show titles already follow in search. Signed out, the section is
absent. `GET /api/app/collections/search?q=`.

### Telemetry

Opening a board's short listen sends `reel_open { source: board, kind: board }`, and the player's
`reel_action` carries `kind: board` (RFC-126 §telemetry). Opening a board from search sends
`search_result_click { result_kind: board, rank }`. No event carries a board's id or name, and a
clip heard in a board's short listen is counted in the stats without naming the board as a subject.

### Not in this amendment

- Expanding a saved show, topic or person into its episodes for the Brief or the short listen.
- Searching other people's boards (boards are private; sharing a board stays a non-goal).
- Suggesting episodes to add to a board.
