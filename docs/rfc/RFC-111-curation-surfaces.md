# RFC-111: Curation Surfaces — Collections, Share Cards, Highlight↔Graph Wiring

- **Status**: Draft
- **Authors**: Marko, Claude (Opus 4.8)
- **Stakeholders**: Consumer App, Server API
- **Related PRDs**:
  - `docs/prd/PRD-046-delivery-and-curation.md` (product requirements)
  - `docs/prd/PRD-040-capture.md` (highlights/notes — the raw material)
  - `docs/prd/PRD-041-consolidation.md` (the corpus + canonical identity reused here)
- **Related RFCs**:
  - `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md` (entity resolution reused)
  - `docs/rfc/RFC-110-outbound-delivery-and-seam.md` (share/digest payloads carry the same graph refs)
  - `docs/rfc/RFC-095-generic-mcp-server.md` (next-arc export/agent consumes the wiring frozen here)
- **Related epic**: #1413 (curation children #1417 collections, #1418 share cards, #1419 graph wiring)

## Abstract

Specifies the **curation** layer that moves the product from "a player" to "management + curation of
interesting bits": user-defined **collections/boards** spanning episodes, shareable **text quote
cards** (an outward growth loop), and the **highlight↔KG-entity wiring** that turns a saved quote
into a queryable graph node — and freezes the serialization shape the **next-arc PKM export**
(Obsidian/Notion) will emit. Everything is additive, extractive, and no-ML; it reuses the shipped
canonical-identity + relational layers.

## Problem Statement

Capture (PRD-040) records moments; consolidation (PRD-041) makes them recallable. But the user has no
way to **actively organize** those bits or **share** them, and highlights are still largely flat text
rather than nodes in the personal graph. Without an organization + sharing layer, the corpus stays a
personal read-only archive; without graph wiring, the outbound surfaces (digest, share card, future
export) can't "carry the graph" (the moat rule from PRD-046) and the next-arc PKM export has nothing
connected to serialize.

## Goals

1. **Collections/boards** — named sets of highlights that span episodes; the active curation surface.
2. **Text share cards** — a graph-carrying quote card, shared via the native share sheet.
3. **Highlight↔entity wiring** — resolve each highlight to its KG entities/topics; expose the links
   across recall, digest, and cards; **freeze + document** the shape for the export arc.
4. Stay **extractive, additive, no-ML**, reusing RFC-072 identity + the relational layer.

## Constraints & Assumptions

- **Audio is bridge-only** (PRD-035 **Principle 4** — no third-party audio stored or served) — share
  cards carry **text/transcript + metadata only**; no source audio. Audio snips are deferred pending
  a legal ruling (PRD-046 non-goal).
- **File-based per-user store** — collections + highlight↔entity links are per-user JSON via the
  existing FileLock path.
- **No new ML** — entity resolution reuses the shipped canonical identity (RFC-072); no request-time
  LLM (D6).

## Design & Implementation

### 1. Collections / boards (#1417)

- Per-user store: a `collections` list `{ id, name, created_at }` + a `highlight_collections` join
  `{ highlight_id, collection_id }` (a highlight belongs to N collections).
- API: `GET/POST/DELETE /api/app/collections`, `POST/DELETE /api/app/collections/{id}/items`.
- UI: create/rename/delete a collection; add/remove a highlight; a collection view reusing the
  `HighlightsView` card style, grouped by collection instead of episode.

### 2. Text share cards (#1418)

- Render a highlight to a share **image**: quote text + guest + topic + episode title + timestamp.
  Guest/topic come from the §3 wiring (carries the graph). Deterministic template; no audio.
- Delivery: reuse the existing native share path (`saveAndShareText` / `isNative` in the player);
  web fallback = download/copy.
- **Explicitly no source audio** in the artifact (bridge-only). Audio-snip cards are a separate,
  gated future item.

### 3. Highlight↔graph wiring (#1419) — and the export seam

- On capture (and via a one-time backfill), resolve the highlight to canonical KG **entity** +
  **topic** references (RFC-072), and persist them on the highlight record. **Granularity, honestly:**
  an **insight** highlight already carries bridge refs directly (RFC-072 `bridge.json` ABOUT /
  MENTIONS_PERSON / MENTIONS_ORG edges); a **span/moment** highlight resolves via its containing
  segment/insight's refs (coarse). Char-offset-precise span→entity lift (RFC-072 **KL1**) and
  cross-episode person alias merge (**KL2**) are open/future in RFC-072 — this slice does **not**
  assume them; it uses what the shipped bridge provides and degrades to "no refs" cleanly.
- Surface the refs in `HighlightsView`, include them in the RFC-110 digest payload, and in the §2
  share cards.
- **Freeze the serialization shape** (documented in this RFC + `PLATFORM_API.md`): a highlight
  serializes as `{ quote, episode, t_ms, entity_refs:[{id,label}], topic_refs:[{id,label}], note? }`.
  The next-arc PKM export (Obsidian/Notion) maps each ref to a `[[wikilink]]` note — that mapping is
  what makes the export a *connected vault*, not a flat highlight dump (the Snipd-differentiator).

## Key Decisions

1. **Collections are a per-user overlay, not corpus artifacts** — consistent with PRD-041's
   projection model and PRD-035 **Principle 3** (shared corpus; no per-user forking of episode
   artifacts). Collections live in the per-user layer over the shared ontology.
2. **Share cards are text-only this arc** — bridge-only audio; text card ships now, audio later.
3. **The highlight↔entity shape is frozen here** so RFC-110 (digest) and the next-arc export both
   serialize the same graph refs — one shape, three consumers.

## Alternatives Considered

- **Tags instead of collections.** Flatter, but tags don't express curated, ordered, named sets;
  collections model the "board of bits" intent better. Tags can layer on later.
- **Server-rendered share cards vs client canvas.** Client canvas avoids a server image pipeline and
  keeps audio out of the server path entirely; chosen for the text card. Revisit if richer templates
  need server rendering.
- **Defer graph wiring to the export arc.** Rejected — the digest (RFC-110) already needs the refs to
  "carry the graph," so the wiring is a dependency now, not later.

## Testing Strategy

- **Unit**: collection CRUD + membership consistency (delete highlight/collection stays consistent);
  entity resolution on a fixture highlight; share-card payload has guest/topic + no audio field.
- **Component**: collection view; share-card generation from a highlight.
- **Integration**: highlight↔entity refs present in the digest payload (shared with RFC-110 tests);
  serialization shape matches the documented contract.

## Rollout & Monitoring

- Additive + behind the existing auth gate; no migration risk (absent fields default empty).
- Monitor: collections created, cards shared, % highlights with resolved entity refs.

## Open Questions

- Collection ordering/pinning + whether collections feed personalized ordering later.
- Share-card visual template (brand chrome) — design pass.
- Backfill strategy for existing highlights' entity refs (lazy on read vs one-time sweep).

## References

- PRD-046; RFC-110; RFC-072; RFC-095 (next-arc consumer of the frozen shape); epic #1413.
