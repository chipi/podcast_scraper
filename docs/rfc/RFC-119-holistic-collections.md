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
hold **episodes, shows, saved searches, topics, people, and highlights** — and promotes it to a
**first-class Library tab**. The mental model is **Pinterest for listening**: explore a topic, pin
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
      { "kind": "person",    "ref": "person:<id>" }
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

## Non-goals

- Ordering/reordering within a collection (v1 stays newest-first; manual order is a later ask).
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
