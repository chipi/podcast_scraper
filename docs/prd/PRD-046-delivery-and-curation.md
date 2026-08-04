# PRD-046: Delivery & Curation (the personal knowledge loop, closed)

- **Status**: Draft
- **Authors**: Marko
- **Target Release**: v2.8
- **Parent PRD**: [PRD-035](PRD-035-learning-platform.md) (learning platform — Principle 2, the consolidation moat)
- **Builds on**: [PRD-041](PRD-041-consolidation.md) (personal knowledge corpus — **shipped**), [PRD-040](PRD-040-capture.md) (highlights/notes), [PRD-039](PRD-039-player.md) (playback history)
- **Related RFCs**: [RFC-110](../rfc/RFC-110-outbound-delivery-and-seam.md) (outbound delivery + the app↔infra seam), [RFC-111](../rfc/RFC-111-curation-surfaces.md) (curation surfaces), [RFC-101](../rfc/RFC-101-personal-knowledge-corpus.md) (the shipped substrate), [RFC-095](../rfc/RFC-095-generic-mcp-server.md) (BYO-agent north-star)
- **Related ADRs**: [ADR-144](../adr/ADR-144-self-hosted-delivery-queue-outsourced-last-mile.md) (self-hosted queue + outsourced last-mile), [ADR-145](../adr/ADR-145-channel-agnostic-outbox-seam.md) (the channel-agnostic outbox seam)
- **Related epic**: [#1413](https://github.com/chipi/podcast_scraper/issues/1413) (children #1412, #1414–#1419)
- **Scope**: consumer learning-player (`web/learning-player` + `/api/app/*`). Operator viewer out of scope.

---

## Summary

PRD-041 shipped the **consolidation** engine — a per-user knowledge corpus with grounded recall,
cross-episode threads, and spaced resurfacing. But that engine is **pull-only**: value sits behind
"open the app → Revisit tab." This PRD closes two loops on top of it:

1. **Delivery** — the resurfacing/digest **comes to the user** (Web Push + email "Your Week"),
   turning a passive library into an active habit, the way Readwise's Daily Review email does.
2. **Curation** — the user **organizes and shares** the interesting bits (collections/boards,
   shareable quote cards), moving the product from "a player" toward "management + curation of
   what mattered."

**The moat this serves (PRD-035 Principle 2, sharpened by the competitor read).** The
podcast-Readwise category (Snipd, Podwise) is a *feeder*: it makes per-clip AI summaries and syncs
them into the user's *external* PKM, where the user does the connecting. Snipd's own positioning:
*"Snipd doesn't replace your PKM system — it feeds into it."* We already **are** the connected
corpus. The design rule that falls out and governs every requirement here: **every outbound surface
carries the graph, not a flat clip** — a digest item links its entity/guest/topic; a share card
names the guest + topic; the next-arc export emits wikilinked entity notes. Distributing flat clips
would make us a worse Snipd; distributing the graph is the category-of-one move.

## Background & Context

- **PRD-035 thesis**: listening is the input; a growing, connected, grounded personal corpus is the
  output. PRD-040 (capture) + PRD-041 (consolidation) built the corpus; this PRD makes it *reach the
  user* and *become curatable*.
- **What's already built** (do not rebuild — see PRD-041 "As shipped"): capture, grounded recall
  (`scope=mine`), cross-episode threads, the resurfacing due-ladder (`app_resurfacing.py`), interest
  profile, enrichment envelopes, MCP stdio server.
- **The gap**: no outbound channel (delivery), no organization layer above the flat highlight list
  (curation), and no consent/email-identity model to deliver against.

## Goals

1. **Close the delivery loop** — a per-user digest ("Your Week") + resurfacing nudges delivered via
   **Web Push** and **email**, on a user-controlled cadence, respecting consent + pacing.
2. **Kill the cold-start** — seed the resurfacing pool + digest with **auto-extracted** GI
   editor's-picks so the corpus has value on day 1 with zero manual capture (a structural edge over
   tap-to-snip competitors).
3. **Add a curation layer** — collections/boards spanning episodes, and shareable **text** quote
   cards (an outward growth loop).
4. **Wire highlights into the graph** — each highlight resolves to its KG entity/topic, becoming a
   queryable node and the serialization substrate for the next-arc PKM export.
5. **Do it self-hosted where the mechanism is ours** (queue, push) and **outsource only reputation**
   (email last-mile) — see ADR-144.

## Non-Goals

- **Not** a generative digest — no request-time LLM in the delivery/curation path (D6; see below).
  Generative synthesis stays with the user's own agent via MCP (RFC-095), never our server.
- **Not** audio share-clips this arc — audio is bridge-only (never rehost); text/transcript cards
  only. Audio snips are deferred pending a legal ruling.
- **Not** two-way PKM sync — the Obsidian/Notion export is the **next** arc (one-way, graph-aware);
  this PRD only freezes the highlight↔entity shape it will serialize.
- **Not** a real SRS model — fixed resurfacing intervals continue (PRD-041 non-goal stands).
- **Not** collaborative/social — no shared corpora.

## Hard constraints (carried from the substrate)

- **D6 — no request-time LLM** in the core/CI path. The digest, recall, and share cards are
  **extractive/deterministic**; the delivery service and its CI make **no** model calls. Keeps CI
  airgapped (the standing no-LLM-in-CI rule).
- **Per-user store is file-based JSON + FileLock** (`app_user_store.py`), not a DB. All-user fan-out
  (digest) is an O(users) directory scan — acceptable at current scale, materialize later.
- **Audio bridge-only** (PRD-035 **Principle 4** — no third-party audio stored or served) — no source
  audio is rehosted, including in share cards.

## Personas

- **The returning learner** — listens weekly, captures a few moments, wants them to *come back* so
  the knowledge sticks (the Readwise-email lover).
- **The passive listener** — listens a lot, rarely captures; needs auto-picks to get value.
- **The curator** — actively organizes bits into themed collections and shares the best quotes.

## User Stories

1. As a learner, I get a **weekly "Your Week" email + push** with a few highlights to revisit, new
   episodes in shows I follow, and what's trending in *my* corpus — each linked to its entity/topic.
2. As a passive listener, my digest is non-empty even though I captured nothing, because it includes
   **auto-extracted editor's-picks** from episodes I heard (visibly distinct from my own highlights).
3. As any user, I can **set cadence, pause, and unsubscribe** in one tap, and delivery respects it.
4. As a curator, I can create a **collection** and add highlights from different episodes to it.
5. As a curator, I can generate a **share card** for a quote (with guest + topic + episode) and share
   it via the native share sheet.
6. As a learner, my highlights are **linked to KG entities**, so recall and the digest can thread
   them, and a future export can mirror them as a connected vault.

## Functional Requirements

### FR1 — Consent + comms identity (delivery prerequisite) · child #1414
- **FR1.1**: The user profile gains `email_verified` (trust the OAuth verified claim) and a `comms`
  object: `digest {enabled, cadence: weekly|daily, day_of_week, hour, paused}`, `push {enabled}`,
  `unsubscribe_ref`.
- **FR1.2**: `GET/PUT /api/app/comms` + a public no-auth `POST /api/app/comms/unsubscribe?ref=`
  (one-click, legal requirement). Settings UI in `ProfileView.vue`. Back-compat: absent `comms`
  defaults to disabled.

### FR2 — Personal digest assembler ("Your Week", extractive) · child #1415
- **FR2.1**: Assemble a per-user payload from `user_episode_set()` + the shipped resurfacing
  selection + interest profile + new-episodes-in-followed-shows + a few auto-picks (FR3). Read-time,
  same model as resurfacing. **Not** RFC-068 (that's the corpus-wide operator digest).
- **FR2.2**: The payload is **structured + graph-carrying** — every item links entity/guest/topic +
  a deep-link. No HTML (the delivery service renders). Extractive only (D6).
- **FR2.3**: Emit a `DeliveryEnvelope` (RFC-110 / ADR-145) per due user; zero-content users produce
  nothing (no empty digest).

### FR3 — Auto-highlight seed · child #1416
- **FR3.1**: Select GI-extracted key moments for heard-but-uncaptured episodes; feed them into the
  resurfacing pool + digest, **marked distinct** from user captures, with a tunable per-digest cap.

### FR4 — Collections / boards · child #1417
- **FR4.1**: Per-user collections + `GET/POST/DELETE /api/app/collections`; highlight ↔ N collections.
- **FR4.2**: UI to create/rename/delete a collection and add/remove/view highlights across episodes.

### FR5 — Text share cards · child #1418
- **FR5.1**: Render a highlight to a share **image**: quote + guest + topic + episode + timestamp
  (carries the graph). Reuse the native share path (`saveAndShareText`/`isNative`).
- **FR5.2**: **No source audio** (bridge-only). Audio snips deferred.

### FR6 — Highlights → graph nodes · child #1419
- **FR6.1**: On capture, resolve + persist the highlight's entity/topic references (RFC-072).
- **FR6.2**: Surface links in `HighlightsView`, digest payload (FR2), and share cards (FR5).
- **FR6.3**: Freeze + document the highlight↔entity serialization shape for the next-arc export.

### FR7 — Delivery transport (delegated) · child #1412 (infra)
- **FR7.1**: The app enqueues `DeliveryEnvelope`s; the infra delivery service (thin stateless worker →
  Resend HTTP API + self-hosted Web Push; the app outbox is the queue) drains the outbox, renders,
  delivers, and writes suppression back. Owned by the infra slice; see RFC-110 / ADR-144 / ADR-145.
  The app touches it **only** at the
  seam.

## Success Metrics

- A due user receives a non-spam email **and** a push containing graph-linked items with working
  deep-links; zero-content users get nothing.
- A passive (zero-capture) user still receives a meaningful digest via auto-picks.
- Unsubscribe + bounce both suppress future sends.
- A user can build a cross-episode collection and share a graph-carrying quote card.
- No LLM call occurs anywhere in the delivery/curation path or its CI.

## Open Questions

- Digest cadence default (weekly Sun AM proposed) and per-user timezone handling.
- When to materialize a per-user digest summary if the O(users) fan-out gets slow.
- Whether auto-pick selection needs quality tuning before it's on by default.

## Related

- Epic [#1413](https://github.com/chipi/podcast_scraper/issues/1413); design doc
  `docs/wip/PLAYER-CURATION-DELIVERY-MOAT-ARCH.md`.
- Next arc: graph-aware Obsidian/Notion export + MCP remote transport (RFC-095 OQ-1).
