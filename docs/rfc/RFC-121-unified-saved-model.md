# RFC-121: Unified "Saved" model — one favorite concept over favorites + highlights

- **Status**: Draft
- **Authors**: Marko, Claude (Opus 4.8)
- **Stakeholders**: Consumer App (learning-player), Server API (per-user state)
- **Related RFCs**:
  - `docs/rfc/RFC-119-holistic-collections.md` (collections pin any typed item; unaffected)
  - `docs/rfc/RFC-101-*` (Revisit / resurfacing — keys on highlight ids; must survive)
- **Related UX specs**:
  - `docs/uxs/UXS-014-interaction-patterns.md` — "Item actions" (OPEN-1/OPEN-2), "Saved & Library",
    the destructive-confirmation restore test. This RFC resolves OPEN-1 + OPEN-2; the two land
    together (UXS-014 requires code + spec amended in the same change).
- **Origin**: Operator note-dump 2026-09-09 + two Fable-5 advisor reviews on `feat/player-ux-overhaul`.

## Abstract

- **What:** Present **one user-facing "Saved" concept** (the `.lp-fav` heart) across the whole app,
  subsuming today's separate "Favorite" and "Highlight" names. Every saved thing — of any kind —
  may optionally carry the extras highlights carry today (a note, a colour, a marked moment/span,
  export). Library's Saved surface unifies into one list with kind filters.
- **Why:** The operator's simplification: favorites and highlights both end up in Library, so they
  should not be two names. One heart, one place.
- **The structural correction (non-negotiable):** "Saved" is **one concept over two identity
  classes**, NOT one record shape. Collapsing them into one shape breaks either multi-moment
  capture or heart-toggling (see Non-Goals + Risk R1).

## The two identity classes

| Class | Key | Cardinality | Kinds | Today |
| ----- | --- | ----------- | ----- | ----- |
| **A — singleton** | `(kind, ref)` | at most one per ref; idempotent upsert; toggleable | episode, show, topic, person, storyline | `favorites.json` (`app_user_state.py` `add_favorite`/`_upsert_in_place`) |
| **B — capture** | `id` (minted) | many per episode | insight, moment, span | `highlights.json` (immutable `id`, anchor machinery, colour, note, export) |

A moment/span cannot live in `(kind, ref)`: a user marks five moments in one episode, and there is
no ref to toggle. So a "favorite with a moment" **is** a class-B record. Highlights are **not merged
away** — they are re-skinned: the word "Highlight" leaves the UI, the record and its endpoints stay.

## Data model (UI read model)

The UI consumes one `Saved` record composed at the read layer from both stores; it does not require
a new server record:

```
Saved = {
  kind: EpisodeKind | ShowKind | TopicKind | PersonKind | StorylineKind   // class A: key (kind, ref)
       | InsightKind | MomentKind | SpanKind,                             // class B: key = id (ref = id)
  ref: string,
  // display snapshot (label / sublabel / slug) — today's FavoriteAdd
  note?: Note[],          // stays a SEPARATE attached object (episode-level notes exist w/o a capture)
  color?: string | null,  // additive optional; class A gains it, class B already has it
  anchor?: { episode_slug, start_ms, end_ms?, char_start?, char_end?, segment_ids?, quote_text?,
             speaker?, anchor_status },   // class B only — the existing highlight anchor
  source_insight_id?: string | null,
  created_at / added_at
}
```

- **Notes stay attached objects, not inline fields** — `notes.json` is deliberately separate so an
  episode note can exist without a capture, export renders episode notes distinctly, and the
  delete-sweep works. `NoteTarget` extends additively: `+ 'show' | 'topic' | 'person' | 'storyline'`.
- **Colour on class A** — additive optional field on the favorites row (the store already upserts a
  raw dict; only the route schema needs the field).
- The dead `start_ms?` on today's `FavoriteAdd` (a legacy insight display field) is removed during
  the type split so nobody mistakes it for a moment field.

## Write paths — and the #1593 ban

#1593's harm was **two write destinations for one insight**; its fix: "Highlights is the single
destination… do not add a new write path." This unification is compatible **only** if:

1. **The insight heart routes to the capture path.** The bookmark in `KnowledgePanel` becomes an
   `.lp-fav` heart that calls the existing `captureStore` toggle → `POST /highlights`. Re-skin, not
   re-plumb — zero data-path change.
2. **`PUT /favorites` structurally rejects `kind=insight` (422).** Today the server `Literal` still
   accepts it and the client type still carries it — the ban is enforced only by the absence of a
   caller. Under a unified `FavoriteButton` an implementer *will* wire `favorites.toggle({kind:
   'insight'})` by accident. Narrow the write `Literal` server-side to
   `episode|person|topic|show|storyline`; split the client type into `FavoriteWriteKind` (no insight)
   vs read kinds. GET still returns legacy insight favorites read-only — only the write narrows.
3. **Nothing merges, nothing retires.** `/favorites` (A), `/highlights` (B), `/notes` (extras) all
   stay, with their existing offline-outbox replay semantics (kind+ref-idempotent vs id-idempotent).
   A merged endpoint would have to reconcile the two replay models for zero user value.

## Migration — none on disk

- Unify at the **read layer, client-side**: a `saved` selector composing `favoritesStore` +
  `captureStore` into one `Saved[]`. Both already load/cache/offline-flip independently — keep that.
- **Legacy `Saved › Insights` favorites** stay on disk, read-only, exactly as #1593 designed (the
  section drains itself as users clear it). In the unified list they render under the Insights
  filter, **deduped against insight highlights by `source_insight_id`/ref** (prefer the richer
  highlight when both exist).
- **No operator-gated server migration.** Rewriting per-user files buys nothing the read-merge does
  not, and risks the "one bad read wipes a list" failure the store comments already warn about. A
  physical merge, if ever wanted, is a separate later operator-gated step — not coupled here.

## Library IA

- **Tabs unchanged: Following · Saved · Collections · Revisit** (test-defended; #1599 is the
  cautionary tale — do not touch the tab set).
- **Inside Saved:** the three `h2` sections (Episodes · Insights · Highlights) collapse to **one list
  + kind filter chips** (All · Episodes · Insights · Moments · Shows · Topics · People). Chips render
  only for kinds that have items (preserves the #1962 single-empty-state). The word "Highlights"
  disappears from the UI; insight/moment/span entries are just saved items.
- **Extras on the card:** colour as an edge bar, note count/snippet, a `▶ mm:ss` chip for anchored
  moments, export in the list toolbar.
- **Revisit** keeps feeding from class-B records, unchanged.

## Toggle + delete semantics (resolves the confirm rule)

Restore test (UXS-014 destructive-confirmation): **heart-off is a free toggle when the record carries
nothing authored; it opens `ConfirmDialog` when it does.**

- Bare class-A favorite → free toggle.
- Class-A favorite with a colour/note → confirm (colour/note are authored; cheapest correct rule =
  confirm iff `note ∨ color`).
- Any class-B record, or any favorite with attached notes → confirm (delete cascades notes + destroys
  resurfacing state; ids are minted, restore is inexact).

## OPEN-1 resolution — Library favorite

**Keep the heart, inverted to one-tap unfavorite; do not drop it.** This satisfies the operator's
"no need to *add* a favorite on Library" (nothing there offers an add — the ♥ shows saved-state
truth) while matching the invert-don't-drop rule already used for Queue→remove and Downloaded→delete.
The confirm rule above makes one-tap unfavorite safe on noted items. The redundant `⋯ remove` in the
Library row is dropped.

## Phased delivery (incremental, mostly additive)

1. **Server 422 write-ban on `kind=insight` + client type split.** Smallest; locks in #1593
   forever. Only API-behaviour change, and it narrows (reads unaffected).
2. **Insight bookmark → `.lp-fav` heart re-skin** (routes to the capture path).
3. **Unified `saved` read model + Saved filter-chip IA** (dedupe legacy insights).
4. **Colour + notes on class-A kinds** + `NoteTarget` extension.
5. **New class-A kinds `show` / `storyline`** (additive Literal both ends; define the storyline ref).
6. **Export extension** for noted class-A favorites.

## Non-Goals

- **One physical record shape.** Explicitly rejected — see the two classes + R1.
- **On-disk migration / endpoint retirement.** Out of scope; read-layer unification only.
- **Merging Follow into Favorite.** Save (heart) and Follow (pill) stay distinct (UXS-014).

## Risks

- **R1 [blocker] — identity-class collapse.** Any spec/impl treating "Saved" as one record shape
  (e.g. adding span fields to the `(kind, ref)` row) breaks multi-moment capture or heart-toggling.
  The RFC names two classes precisely to prevent this.
- **R2 [blocker] — banned-path resurrection.** Until the server write `Literal` drops `insight`, the
  unified heart is one misdirected `favorites.toggle` from re-creating #1593. **Ship phase 1 first.**
- **R3 [should-fix] — follow vs favorite on entities.** favorite(topic) + follow(topic) now coexist
  on entity cards. Define both per kind in the UXS amendment, or cut class-A entity kinds from
  phase 1.
- **R4 [should-fix] — momentum blind spot.** The engagement series counts only `favorites.added_at`;
  insight saves via the highlight path would stay invisible. Fold highlight `created_at` into the
  saves tally (verify against the RFC-103 maths).
- **R5 [should-fix] — legacy/highlight dedupe.** Without `source_insight_id` dedupe, pre-#1593 users
  see the same insight twice.
- **R6 [nice-to-have] — export coverage.** Noted class-A favorites won't appear in export until the
  exporter learns about them.

## Verification / rollback

- Each phase ships independently; phase 1 is a pure narrowing (422 on a write kind nothing currently
  calls) — rollback = revert the Literal.
- `ci-ui-full` (not fast) before push — the highlight→favorite re-skin touches i18n keys, testids,
  and specs broadly.
