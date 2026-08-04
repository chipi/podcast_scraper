# ADR-145: The app↔infra delivery boundary is a channel-agnostic outbox seam

- **Status**: Accepted — 2026-08-04 (operator ratified the seam as the app↔infra boundary). Implementation tracked by epic #1413; not yet executed.
- **Date**: 2026-08-04
- **Authors**: Marko Dragoljevic, Claude (Opus 4.8)
- **Related**: [RFC-110](../rfc/RFC-110-outbound-delivery-and-seam.md) (full schema + endpoints), [ADR-144](ADR-144-self-hosted-delivery-queue-outsourced-last-mile.md) (what runs on the infra side), [PRD-046](../prd/PRD-046-delivery-and-curation.md), the D6 no-request-time-LLM rule
- **Tracking**: epic #1413 (app slices #1414/#1415 produce it; infra slice #1412 consumes it)

## Context

Delivery (PRD-046) splits cleanly into two responsibilities with very different concerns and
lifecycles:

- **App** — *what/when/who*: assemble the digest payload (extractive, D6), enforce consent + cadence,
  know the user. Product logic. Must stay testable with **no mail stack in its CI**.
- **Infra delivery** — *how*: render to HTML/push, relay through SES, sign VAPID, retry, handle
  bounces/complaints/unsubscribe, deliverability. Operational, and **delegated to a separate infra
  agent** (#1412) working in parallel.

If these two share more than a minimal contract, they can't move independently and product code ends
up owning rendering + SMTP concerns. We need a single, frozen interface so both sides build against a
fixed target and join deterministically — and so this arc's app work and infra work parallelize.

## Decision

The boundary is a **channel-agnostic `DeliveryEnvelope` written to an outbox the app owns and the
delivery worker drains.** They touch **only** here.

1. **`DeliveryEnvelope`** (schema_version "1", full schema in RFC-110 §1): `{ id (idempotency key),
   user_id, channel, template, recipient, consent_snapshot, payload, not_before, created_at }`. The
   `payload` is **structured, channel-agnostic, and graph-carrying** — never HTML, never a flat clip.
2. **Outbox transport** (app-owned endpoints): `GET /internal/outbox/pending?channel=&limit=` and
   `POST /internal/outbox/{id}/status`. Delivery leases → delivers → acks. Dedup + retry keyed on
   `id`; the app's per-user store is the single source of truth.
3. **Rendering lives on the infra side.** The app emits the structured payload; the delivery service
   maps `template` → HTML email / push notification. The app never emits HTML.
4. **Suppression writes back** across the seam: bounce/complaint/unsubscribe → `POST
   /internal/outbox/{id}/status` + a public `POST /api/app/comms/unsubscribe?token=` (app-owned) that
   flips consent. The app stops enqueuing suppressed recipients.

**Why channel-agnostic (the non-obvious core).** A per-channel handoff ("send this email", "send this
push") would leak channel specifics into the app and force the app to render. Making the envelope
channel-agnostic keeps *all* rendering + channel logic on the infra side; adding a future channel
(SMS, in-app inbox) is an infra-only change against the same contract.

## Consequences

**Positive**

- App and infra build in parallel against one frozen schema and join deterministically at the end.
- App CI has **no mail/push stack** — the delivery worker is stubbed at the outbox boundary; keeps CI
  airgapped (consistent with D6 / no-LLM-in-CI).
- Delivery is stateless + restart-safe (dedupe on `id`); the app store stays the source of truth.
- A new channel is an infra-side addition, not an app change.

**Negative / trade-off**

- One more indirection than "app calls send API directly" — an outbox + a polling worker to operate.
  Accepted: it buys the clean delegation boundary + parallel workstreams this arc needs.
- The schema is a **frozen contract**; changing it is a versioned migration (hence `schema_version`).

**Neutral**

- Poll-based drain now (simple on the file store); can move to push-notify between app and worker
  later without changing the envelope (RFC-110 open question).
- Outbox storage shape (per-user append log vs global queue file) is an implementation detail under
  the seam, not part of the contract.

## Alternatives considered

- **App calls a transactional send API directly, per message.** Simplest, but couples product code to
  a channel + provider, forces the app to render, and puts a mail dependency in app CI. Rejected —
  the whole point is a delegable, testable boundary. See ADR-144 for the build-vs-buy side.
- **Per-channel handoff endpoints** (`send-email` / `send-push`). Leaks channel specifics into the app
  and duplicates rendering. Rejected in favour of the channel-agnostic envelope.
- **Shared database table as the seam** instead of HTTP endpoints. Would couple both sides to the
  store's schema + tech; the per-user store is file-based JSON anyway. The HTTP outbox keeps the store
  private to the app. Rejected.
- **Materialize rendered messages in the app, infra just ships bytes.** Puts rendering (and its
  templating deps) back in the app + write-amplifies the file store. Rejected — rendering belongs with
  delivery.
