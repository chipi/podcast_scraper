# RFC-110: Outbound Delivery & the App↔Infra Seam

- **Status**: Draft
- **Authors**: Marko, Claude (Opus 4.8)
- **Stakeholders**: Consumer App, Server API, Infra/Delivery
- **Related PRDs**:
  - `docs/prd/PRD-046-delivery-and-curation.md` (product requirements)
  - `docs/prd/PRD-041-consolidation.md` (the substrate this distributes)
- **Related RFCs**:
  - `docs/rfc/RFC-101-personal-knowledge-corpus.md` (resurfacing selection reused here)
  - `docs/rfc/RFC-068-corpus-digest-api-viewer.md` (the **corpus-wide** operator digest — explicitly NOT reused; different scope)
- **Related ADRs**:
  - `docs/adr/ADR-144-self-hosted-delivery-queue-outsourced-last-mile.md` (queue + last-mile decision)
  - `docs/adr/ADR-145-channel-agnostic-outbox-seam.md` (the seam as an architectural boundary)
- **Related epic**: #1413 (delivery children #1412 infra, #1414 consent, #1415 assembler, #1416 seed)

## Abstract

Specifies how the personal digest ("Your Week") and resurfacing nudges reach the user over **Web
Push** and **email**, without a request-time LLM (D6). The design draws a hard boundary — the
**seam** — between the **app** (decides *what/when/who*, extractive) and the **infra delivery
service** (decides *how it's delivered*: render, relay, reputation, retries, bounces). They join
only at a channel-agnostic **`DeliveryEnvelope`** written to an **outbox** the app owns and the
delivery worker drains. This lets the infra slice (#1412) and the app slices proceed in parallel and
join deterministically.

## Problem Statement

PRD-041's resurfacing is computed **on read** and is **pull-only**. To become a habit (the Readwise
loop) it must be **pushed** to the user. But delivery drags in concerns the app must not own:
HTML/MJML rendering, SMTP/relay, IP reputation, VAPID signing, retries, bounce/complaint handling,
unsubscribe. Folding those into the consumer API would couple product logic to deliverability
infrastructure and would put a mail stack in the app's CI. We need a boundary that keeps the app
extractive + testable and lets delivery be operated (and delegated) independently.

## Goals

1. A per-user digest + nudges delivered via Web Push **and** email on a user-controlled cadence.
2. A **channel-agnostic, idempotent** contract between app and delivery, versioned and frozen.
3. **No LLM** anywhere in the delivery path or its CI (D6).
4. Delivery is **stateless + restart-safe** (dedupe on envelope id); the app's per-user store is the
   single source of truth.
5. Suppression (bounce/complaint/unsubscribe) flows back so the app stops enqueuing dead recipients.

## Constraints & Assumptions

- **D6**: the payload handed across the seam is fully pre-assembled + extractive; the delivery
  service renders + sends only.
- **File-based per-user store**: the outbox and the digest fan-out are O(users) scans; acceptable
  now (RFC-101 OQ-1 materialization deferred).
- **Consent model (FR1 / #1414) is a prerequisite** — nothing is enqueued for a user without
  `comms.digest.enabled` (or `comms.push.enabled`) and, for email, `email_verified`.
- **Deliverability is not self-hostable cheaply** — the last mile goes through a reputation relay
  (ADR-144). Web Push has no such problem and is fully self-hosted.

## Design & Implementation

### 1. The seam — `DeliveryEnvelope` (schema_version "1")

Channel-agnostic, idempotent. The app produces it; the delivery service consumes it. See ADR-145 for
why this is the boundary.

```jsonc
{
  "schema_version": "1",
  "id": "string",                 // idempotency key, app-generated; delivery dedupes on this
  "user_id": "u_...",
  "channel": "email" | "push",
  "template": "your-week-digest.v1" | "resurface-nudge.v1",
  "recipient": {
    "email": "string?",           // channel=email
    "email_verified": true,
    "push_subscription": { }      // channel=push (W3C PushSubscription JSON)
  },
  "consent_snapshot": {
    "digest_enabled": true,
    "cadence": "weekly" | "daily",
    "unsubscribe_token": "string" // delivery embeds this in the unsubscribe link
  },
  "payload": { },                 // structured, channel-agnostic, GRAPH-CARRYING (see §3)
  "not_before": "iso8601",
  "created_at": "iso8601"
}
```

### 2. Outbox transport (app exposes; delivery worker polls)

- `GET  /internal/outbox/pending?channel={email|push}&limit=N` → `{ envelopes: DeliveryEnvelope[] }`
- `POST /internal/outbox/{id}/status` → `{ status: delivered|bounced|complaint|suppressed|failed, detail? }`
- Internal service-to-service auth (shared token, tailnet-only — infra's call).
- The app writes envelopes to a per-user (or global) append-only outbox on enqueue; the delivery
  worker leases + acks. Dedup + retry are keyed on `envelope.id`.

### 3. The payload carries the graph (moat rule)

`payload` for `your-week-digest.v1` is a structured list of sections, each item pre-resolved:

```jsonc
{
  "sections": [
    { "kind": "revisit",  "items": [ { "quote", "episode", "t_ms", "entity_refs":[...], "topic_refs":[...], "deep_link", "source": "user"|"auto" } ] },
    { "kind": "new_in_follows", "items": [ { "episode", "show", "deep_link", "topic_refs":[...] } ] },
    { "kind": "trending_in_your_corpus", "items": [ { "topic", "signal": "temporal_velocity", "deep_link" } ] }
  ]
}
```

The delivery service renders this to HTML/push; it never computes it. `source: "auto"` marks
auto-picks (FR3) distinctly.

### 4. App side — the digest assembler (#1415)

`app_digest_personal.py`: for each user due per `comms.digest.cadence`, assemble the §3 payload by
reusing `user_episode_set()` + the resurfacing due-selection + interest profile + followed-show
deltas + auto-picks. Emit a `DeliveryEnvelope`. Wired into the existing in-process APScheduler
(`scheduler.py`, today feed-sweep-only) as a per-user digest cron. Extractive only (D6).

### 5. Infra side — the delivery service (#1412; specified in ADR-144)

Drains the outbox → renders (`template`) → delivers (Web Push via VAPID; email via Listmonk → SES) →
reports status (§2). Owns retries, bounce/complaint webhooks, unsubscribe link generation. Web Push
subscription registration is `POST /api/app/push/subscribe` (app-owned endpoint, infra-owned worker).

### 6. Channels

- **Web Push**: fully self-hosted. Server holds a VAPID keypair, signs, delivers to browser push
  endpoints (FCM/Mozilla/Apple). No third party, no reputation problem. Reuses the PWA service worker.
- **Email**: Listmonk (self-hosted queue/manager) relays through Amazon SES (reputation last mile).
  Renders the §3 payload; unsubscribe uses `consent_snapshot.unsubscribe_token`.

## Key Decisions

1. **The seam is a channel-agnostic outbox, not per-channel API calls** — ADR-145.
2. **Self-host the queue, outsource the last mile** — ADR-144.
3. **Payload is extractive + graph-carrying** — the digest never generates prose (D6) and never
   ships a flat clip (moat rule).
4. **Reuse resurfacing selection, not RFC-068** — the corpus-wide digest is a different scope.

## Alternatives Considered

- **App calls a transactional email API directly (Resend/Postmark) per send.** Simplest, but couples
  product code to a SaaS + reputation, puts a mail dependency in the app's CI, and offers no clean
  delegation boundary. Rejected — ADR-144.
- **Per-channel handoff (app calls "send email" / "send push" endpoints).** Leaks channel specifics
  into the app and forces the app to render. The channel-agnostic envelope keeps rendering on the
  infra side. Rejected — ADR-145.
- **Background job materializes digests into the store, delivery reads the store.** More moving
  parts + a per-user write amplification on a file store; the outbox-of-envelopes is lighter.
  Deferred until fan-out latency demands it.

## Testing Strategy

- **Unit**: envelope assembly from a fixture corpus (graph refs present, deep-links valid,
  zero-content → no envelope); auto-pick marking; cadence/consent gating.
- **Integration**: outbox pending/ack round-trip; idempotent dedupe on `id`; suppression write-back
  flips consent. **No mail stack in app CI** — the delivery worker is stubbed at the outbox boundary.
- **Infra (separate, #1412)**: real SES sandbox + a Web Push end-to-end; inbox-placement smoke test.

## Rollout & Monitoring

- Ship **Web Push first** (no new dep) to de-risk the loop, then email once SES + deliverability
  (ADR-144) are verified.
- Opt-in + pausable; disabling the worker or pausing sends is a <5-min rollback.
- Monitor: enqueue count, delivery success/bounce/complaint rate, unsubscribe rate.

## Open Questions

- Per-user timezone for cadence scheduling.
- Outbox storage shape on the file store (per-user append log vs a single global queue file).
- When to move from poll to push-notify between app and delivery worker.

## References

- PRD-046; ADR-144; ADR-145; RFC-101 (§5 resurfacing); epic #1413.
