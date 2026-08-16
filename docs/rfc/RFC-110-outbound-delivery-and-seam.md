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

**Normative contract (both tracks validate against these):** the machine-readable JSON Schema is
`docs/api/delivery-envelope.schema.json`; golden fixtures live under `tests/fixtures/delivery/`
(`your-week-digest.v1.golden.json`, `resurface-nudge.v1.golden.json`); the shared contract test is
`tests/unit/server/test_delivery_envelope_contract.py`. The JSON below is illustrative — the schema
file is the source of truth. The infra service (#1412) mirrors the same fixtures + assertions.

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
  "consent_snapshot": {           // v1.1: INFORMATIONAL ONLY — the worker is stateless and
    "digest_enabled": true,       // must re-check CURRENT consent via /pending, not this snapshot
    "cadence": "weekly" | "daily",
    "unsubscribe_ref": "string" // delivery embeds this in the unsubscribe link
  },
  "payload": { },                 // structured, channel-agnostic, GRAPH-CARRYING (see §3)
  "not_before": "iso8601",
  "expires_at": "iso8601",        // v1.1 TTL — do not deliver after this (no stale-flush on recovery)
  "created_at": "iso8601"
}
```

**Seam v1.1 amendments (ratified 2026-08-04, post-advisor review — issues #1412/#1413).** The frozen
shape and endpoints are unchanged except the additive `expires_at`. Ratified: (1) `/status` idempotent
per `id`; (2) `/pending` filters **current** consent, not the snapshot; (3) `expires_at` TTL added;
(4) `failed` is an **always-terminal** dead-letter state; (5) push `410/404` → `status: "bounced"`
suppression; (6) `/internal/*` auth is a tailnet-only shared token `INTERNAL_OUTBOX_TOKEN`. The
`unsubscribe_ref` field name (was `unsubscribe_token`) is the ratified spelling — infra adopts `ref`.

### 2. Outbox transport (app exposes; delivery worker polls)

**Implemented (app side, #1415) — complete:** `app_outbox_store` (persistence + idempotency +
consent/expiry filtering + suppression write-back), `routes/internal_outbox` (the two endpoints,
token-gated, mounted at `/internal`), `app_digest_personal` (the extractive assembler → email
digest + push nudges → `enqueue_due_digests`), the **scheduler `digest` job kind** (an operator
schedules it in `viewer_operator.yaml` like any sweep — fires `enqueue_due_digests` on cadence), and
the **Web Push** path: `app_push_store` + `POST/DELETE /api/app/push/subscribe` +
`GET /api/app/push/vapid-key`, the browser `usePushSubscription` composable, and the SW push handler
(`public/push-sw.js`). The only external inputs left are infra-owned: the **VAPID keypair**
(`APP_VAPID_PUBLIC_KEY` for the app, private half in the worker), the **`INTERNAL_OUTBOX_TOKEN`**,
and the **Resend** worker itself.

- `GET  /internal/outbox/pending?channel={email|push}&limit=N` → `{ envelopes: DeliveryEnvelope[] }`.
  **v1.1:** filters on **current** consent (re-reads `comms`), excludes past-`expires_at` envelopes,
  and never returns a user who unsubscribed after enqueue.
- `POST /internal/outbox/{id}/status` → `{ status: delivered|bounced|complaint|suppressed|failed, detail? }`.
  **v1.1:** **idempotent per `id`** (a repeated terminal status is a no-op — retries after a
  succeeded-but-unacked send); `failed` is **always-terminal** (the worker dead-letters as `failed`
  after N retries); a dead push subscription (`410 Gone` / `404`) reports `status: "bounced"` so the
  app suppresses it.
- Internal service-to-service authentication uses a **tailnet-only shared token
  `INTERNAL_OUTBOX_TOKEN`** (staged in the homelab sops-env + the app secret store; both halves must
  agree the name to connect).
- The app writes envelopes to a per-user (or global) append-only outbox on enqueue; the delivery
  worker leases + acks. Dedup + retry are keyed on `envelope.id`. The **app consent store is the ONLY
  suppression authority** (no external suppression list) — dropping a second queue removes the
  two-lists race.

### 3. The payload carries the graph (moat rule)

`payload` for `your-week-digest.v1` is a structured list of sections; every item is pre-resolved and
carries `graph_refs` (the unified canonical-KG refs, mirroring the shipped `AppEntityRef`:
`{id: "person:… | topic:…", kind, label}`) + a `deep_link`:

```jsonc
{
  "sections": [
    { "kind": "revisit", "items": [
      { "quote", "episode_slug", "episode_title", "t_ms", "graph_refs":[{ "id":"person:…","kind":"person","label":"…" }], "deep_link", "source": "user"|"auto" } ] },
    { "kind": "new_in_follows", "items": [ { "episode_slug", "episode_title", "graph_refs":[…], "deep_link" } ] },
    { "kind": "trending_in_your_corpus", "items": [ { "episode_slug", "graph_refs":[{ "id":"topic:…" }], "deep_link" } ] }
  ]
}
```

The delivery service renders this to HTML/push; it never computes it. Every item MUST carry
`graph_refs` + `deep_link` (the moat rule — no flat clips), and `source: "auto"` marks auto-picks
(FR3) distinctly. The `resurface-nudge.v1` payload is `{ highlight_count, lead: <digestItem> }`.

### 4. App side — the digest assembler (#1415)

`app_digest_personal.py`: for each user due per `comms.digest.cadence`, assemble the §3 payload by
reusing `user_episode_set()` + the resurfacing due-selection + interest profile + followed-show
deltas + auto-picks. Emit a `DeliveryEnvelope`. Wired into the existing in-process APScheduler
(`scheduler.py`, today feed-sweep-only) as a per-user digest cron. Extractive only (D6).

### 5. Infra side — the delivery service (#1412; specified in ADR-144, revised to Resend)

A **thin stateless worker** on the homelab, tailnet-only, 443-egress-only. Drains the outbox →
renders (`template` → Jinja) → delivers (Web Push via VAPID; email via the **Resend HTTP API**) →
reports status (§2). Bounce/complaint via **cursor-based polling of Resend's events API** (no public
webhook — the service has no public ingress). No Listmonk/Postgres/Redis: the **app outbox already is
the queue** ADR-144 wanted. Web Push subscription registration is `POST /api/app/push/subscribe`
(app-owned endpoint, infra-owned worker).

### 6. Channels

- **Web Push**: fully self-hosted. Server holds a VAPID keypair, signs, delivers to browser push
  endpoints (FCM/Mozilla/Apple). No third party, no reputation problem. Reuses the PWA service worker.
- **Email**: the worker renders the §3 payload (Jinja) and sends via the **Resend HTTP API**
  (reputation last mile; HTTPS 443, so the port-25 concern is moot). Unsubscribe uses
  `consent_snapshot.unsubscribe_ref`.

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
- **Infra (separate, #1412)**: real Resend sandbox + a Web Push end-to-end; inbox-placement smoke test.

## Rollout & Monitoring

- Ship **Web Push first** (no new dep) to de-risk the loop, then email once Resend + deliverability
  (ADR-144) are verified.
- Opt-in + pausable; disabling the worker or pausing sends is a <5-min rollback.
- Monitor: enqueue count, delivery success/bounce/complaint rate, unsubscribe rate.

## Open Questions

- Per-user timezone for cadence scheduling.
- Outbox storage shape on the file store (per-user append log vs a single global queue file).
- When to move from poll to push-notify between app and delivery worker.

## References

- PRD-046; ADR-144; ADR-145; RFC-101 (§5 resurfacing); epic #1413.
